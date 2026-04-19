"""SimPy discrete-event simulation engine.

This module contains the SimPy generator process for a single vehicle trip.
Each agent is a SimPy process that:

  1. Waits until its departure time.
  2. Decides initial SOC (endogenous pre-trip charging decision).
  3. Plans route with charging stops.
  4. Drives segment by segment, yielding SimPy timeouts.
  5. At each planned charging stop, decides whether to charge (en-route logic).
  6. If charging: requests a SimPy Resource (connector), waits, charges.
  7. At non-planned nodes with stations: considers opportunistic top-up.
  8. Completes (or strands if energy runs out).

Queue congestion feedback is automatic: SimPy Resources queue agents that
arrive when all connectors are busy.  Later agents can see
``station.resource.count`` and ``station.resource.queue`` before deciding
whether to use a station.

Key design choices
------------------
- Time unit: minutes throughout.
- Energy unit: kWh throughout.
- ``yield from`` is used to delegate sub-generators (charging process).
- Each agent modifies its own VehicleAgent object; no shared mutable state
  except station.resource (SimPy-managed) and station statistics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional

import simpy

from models.agent import VehicleAgent
from models.network import RoadNetwork
from models.results import ResultsCollector
from models.station import ChargingStation
from behavior.energy import compute_segment_energy, compute_charge_duration
from behavior.pre_trip import decide_initial_soc
from behavior.routing import plan_route_with_stops, build_trip_waypoints, Waypoint
from behavior.station_choice import decide_charge_target
from behavior.en_route import (
    decide_to_charge_here,
    find_emergency_station,
    should_add_unplanned_stop,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main trip process
# ---------------------------------------------------------------------------

def vehicle_trip_process(
    env: simpy.Environment,
    agent: VehicleAgent,
    network: RoadNetwork,
    stations_by_node: Dict[str, List[ChargingStation]],
    config: Dict,
    collector: ResultsCollector,
    rng: Any,
) -> Generator:
    """
    SimPy generator for one BEV trip.

    Parameters
    ----------
    env:              SimPy environment.
    agent:            VehicleAgent (mutable; state is updated here).
    network:          Road network.
    stations_by_node: Dict mapping node_id → list of ChargingStation.
    config:           Flat dict loaded from base_config.yaml.
    collector:        ResultsCollector to write events to.
    rng:              NumPy random generator for stochastic elements.
    """
    # --- Wait until departure time ---
    if env.now < agent.departure_time_min:
        yield env.timeout(agent.departure_time_min - env.now)

    agent.start_time_min = env.now
    agent.current_node = agent.origin

    # --- Pre-trip: decide initial SOC ---
    initial_soc = decide_initial_soc(agent, network, stations_by_node, config, rng)
    agent.current_soc_kwh = initial_soc
    collector.record_departure(agent.agent_id, initial_soc)
    logger.debug(
        "[t=%.0f] Agent %s departs %s→%s | SOC=%.1f/%.0f kWh",
        env.now, agent.agent_id, agent.origin, agent.destination,
        agent.current_soc_kwh, agent.usable_capacity_kwh,
    )

    # --- Plan route with charging stops ---
    route, charging_stops = plan_route_with_stops(
        agent, network, stations_by_node, config
    )
    if not route:
        agent.status = "failed_no_route"
        agent.failure_reason = "no_path_found"
        collector.record_completion(
            agent_id=agent.agent_id,
            origin=agent.origin,
            destination=agent.destination,
            departure_time_min=agent.departure_time_min,
            arrival_time_min=env.now,
            status="failed_no_route",
            num_charge_stops=0,
            total_wait_min=0.0,
            total_charge_min=0.0,
            total_energy_kwh=0.0,
            total_distance_km=0.0,
            route_node_count=0,
            final_soc_kwh=agent.current_soc_kwh,
            failure_reason="no_path_found",
        )
        return

    agent.route = route
    agent.charging_plan_station_ids = [s.station_id for s in charging_stops]
    agent.status = "driving"

    # --- Build waypoints (driving segments between stops) ---
    waypoints = build_trip_waypoints(route, charging_stops)
    total_dist = network.subpath_distance_km(route)
    agent.total_distance_km = total_dist

    # --- Drive the route waypoint by waypoint ---
    for waypoint in waypoints:
        # Check energy feasibility before driving this segment
        segment_energy = compute_segment_energy(
            waypoint.nodes, network, agent.consumption_kwh_per_km
        )
        reserve_kwh = agent.usable_capacity_kwh * config.get("min_reserve_soc_fraction", 0.10)

        if agent.current_soc_kwh - segment_energy < reserve_kwh:
            # Need emergency charge before this segment
            handled = yield from _handle_emergency_charge(
                env, agent, waypoint.nodes, network,
                stations_by_node, config, collector
            )
            if not handled:
                # Emergency station search failed: no reachable station within
                # remaining SOC. Agent runs out before the next stop.
                _record_strand(
                    agent, env.now, collector,
                    reason="no_reachable_station",
                )
                return

        # Drive the segment
        drive_time = network.subpath_travel_time_min(waypoint.nodes)
        yield env.timeout(drive_time)
        agent.current_soc_kwh -= segment_energy
        agent.current_soc_kwh = max(0.0, agent.current_soc_kwh)  # clamp
        agent.current_node = waypoint.end_node

        logger.debug(
            "[t=%.0f] Agent %s at %s | SOC=%.1f kWh",
            env.now, agent.agent_id, agent.current_node, agent.current_soc_kwh,
        )

        # --- Charging decision at this waypoint ---
        if waypoint.is_charging_stop and waypoint.station is not None:
            remaining_route = network.subpath_from_node(route, waypoint.end_node)
            should_charge = decide_to_charge_here(
                agent=agent,
                station=waypoint.station,
                remaining_route=remaining_route,
                stations_by_node=stations_by_node,
                network=network,
                config=config,
            )
            if should_charge:
                charged = yield from _execute_charging(
                    env, agent, waypoint.station,
                    route, network, stations_by_node, config, collector
                )
                # On abandonment, the loop continues and the next segment
                # check at line ~138 will trigger an emergency charge at a
                # different station if SOC runs low. No extra retry needed
                # here because the abandonment guard already verified an
                # alternative is reachable within current SOC.
                if charged is False:
                    logger.debug(
                        "Agent %s abandoned %s; relying on emergency path.",
                        agent.agent_id, waypoint.station.station_id,
                    )

        elif not waypoint.is_charging_stop and waypoint.end_node in stations_by_node:
            # Opportunistic top-up at an unplanned node
            station = max(
                stations_by_node[waypoint.end_node],
                key=lambda s: s.max_power_kw,
            )
            remaining_route = network.subpath_from_node(route, waypoint.end_node)
            remaining_dist = network.subpath_distance_km(remaining_route)
            stations_ahead = sum(
                1 for n in remaining_route[1:] if n in stations_by_node
            )
            if should_add_unplanned_stop(
                agent, station, remaining_dist, stations_ahead, config
            ):
                yield from _execute_charging(
                    env, agent, station,
                    route, network, stations_by_node, config, collector
                )

    # --- Trip complete ---
    agent.status = "completed"
    agent.end_time_min = env.now
    logger.debug(
        "[t=%.0f] Agent %s completed %s→%s | SOC=%.1f kWh | "
        "stops=%d | wait=%.1f min",
        env.now, agent.agent_id, agent.origin, agent.destination,
        agent.current_soc_kwh, len(agent.charge_events),
        agent.total_wait_time_min,
    )
    collector.record_completion(
        agent_id=agent.agent_id,
        origin=agent.origin,
        destination=agent.destination,
        departure_time_min=agent.departure_time_min,
        arrival_time_min=env.now,
        status="completed",
        num_charge_stops=len(agent.charge_events),
        total_wait_min=agent.total_wait_time_min,
        total_charge_min=agent.total_charge_time_min,
        total_energy_kwh=agent.total_energy_charged_kwh,
        total_distance_km=agent.total_distance_km,
        route_node_count=len(agent.route),
        final_soc_kwh=agent.current_soc_kwh,
    )


# ---------------------------------------------------------------------------
# Charging sub-process
# ---------------------------------------------------------------------------

def _execute_charging(
    env: simpy.Environment,
    agent: VehicleAgent,
    station: ChargingStation,
    full_route: List[str],
    network: RoadNetwork,
    stations_by_node: Dict[str, List[ChargingStation]],
    config: Dict,
    collector: ResultsCollector,
) -> Generator:
    """
    SimPy generator fragment: queue at station, wait for connector, charge.

    Returns True when charging completes successfully, False when the agent
    abandons the queue because the wait became intolerable AND another
    station is still reachable within current SOC minus reserve (Nicolas's
    safety constraint). Agents that cannot safely reach an alternative stay
    in the queue indefinitely to avoid being stranded at the roadside.
    """
    arrival_time = env.now
    agent.status = "waiting"

    remaining_route = network.subpath_from_node(full_route, station.node_id)
    remaining_dist = network.subpath_distance_km(remaining_route)
    stations_ahead = sum(
        1 for n in remaining_route[1:] if n in stations_by_node
    )
    target_soc = decide_charge_target(
        agent, station, remaining_dist, stations_ahead, config
    )
    soc_before = agent.current_soc_kwh

    logger.debug(
        "[t=%.0f] Agent %s queueing at %s | queue=%d | target_soc=%.1f kWh",
        env.now, agent.agent_id, station.station_id,
        station.current_queue_length(), target_soc,
    )

    # --- Queue-abandonment wait cap (scaled by agent queue_aversion) ---
    base_wait_cap = float(config.get("max_queue_wait_min", 60.0))
    wait_cap = base_wait_cap / max(0.1, getattr(agent, "queue_aversion", 1.0))

    req = station.resource.request()
    try:
        wait_result = yield req | env.timeout(wait_cap)
    except Exception:
        req.cancel()
        raise

    if req not in wait_result:
        # Wait cap exceeded. Check Nicolas's constraint: can we safely
        # reach another station with current SOC? If not, commit to waiting.
        if _can_abandon_safely(
            agent, station, remaining_route, stations_by_node, network, config
        ):
            req.cancel()
            wait_time = env.now - arrival_time
            agent.total_wait_time_min += wait_time
            agent.status = "driving"
            logger.debug(
                "[t=%.0f] Agent %s abandoned queue at %s after %.1f min "
                "(alternative within SOC)",
                env.now, agent.agent_id, station.station_id, wait_time,
            )
            return False
        # No safe alternative: must keep waiting for this connector.
        logger.debug(
            "[t=%.0f] Agent %s hit wait cap at %s but no safe alternative; "
            "staying in queue.",
            env.now, agent.agent_id, station.station_id,
        )
        yield req

    try:
        wait_time = env.now - arrival_time
        agent.status = "charging"

        # Compute charging duration
        charge_time = compute_charge_duration(
            soc_before_kwh=agent.current_soc_kwh,
            soc_target_kwh=target_soc,
            battery_capacity_kwh=agent.battery_capacity_kwh,
            charger_power_kw=station.max_power_kw,
            vehicle_max_acceptance_kw=agent.max_acceptance_kw,
        )

        if charge_time < 0.5:
            # Nothing meaningful to charge — release connector immediately
            agent.status = "driving"
            return True

        yield env.timeout(charge_time)

        energy_added = target_soc - agent.current_soc_kwh
        agent.current_soc_kwh = target_soc

        # Update station stats
        station.record_session(wait_time, charge_time, energy_added)
        peak_q = station.current_queue_length()
        if peak_q > station.peak_queue_length:
            station.peak_queue_length = peak_q

        # Update agent stats
        agent.record_charge_event(
            station_id=station.station_id,
            sim_time_min=env.now,
            wait_time_min=wait_time,
            charge_time_min=charge_time,
            energy_kwh=energy_added,
            soc_before_kwh=soc_before,
            soc_after_kwh=target_soc,
        )

        # Write to collector
        collector.record_charge_event(
            agent_id=agent.agent_id,
            station_id=station.station_id,
            node_id=station.node_id,
            sim_time_min=env.now,
            wait_time_min=wait_time,
            charge_time_min=charge_time,
            energy_kwh=energy_added,
            soc_before_kwh=soc_before,
            soc_after_kwh=target_soc,
            price_per_kwh=station.price_per_kwh,
        )

        agent.status = "driving"
        logger.debug(
            "[t=%.0f] Agent %s finished at %s | +%.1f kWh | wait=%.1f | charge=%.1f min",
            env.now, agent.agent_id, station.station_id,
            energy_added, wait_time, charge_time,
        )
        return True
    finally:
        station.resource.release(req)


# ---------------------------------------------------------------------------
# Queue abandonment helper
# ---------------------------------------------------------------------------

def _can_abandon_safely(
    agent: VehicleAgent,
    current_station: ChargingStation,
    remaining_route: List[str],
    stations_by_node: Dict[str, List[ChargingStation]],
    network: RoadNetwork,
    config: Dict,
) -> bool:
    """
    Return True iff the agent could reach a different station with its
    current SOC above the hard reserve. Prevents agents from abandoning
    into a strand (Nicolas's constraint).
    """
    reserve_frac = float(config.get("min_reserve_soc_fraction", 0.10))
    reserve_kwh = agent.usable_capacity_kwh * reserve_frac

    # Option 1: a different station at the same node.
    same_node_alts = [
        s for s in stations_by_node.get(current_station.node_id, [])
        if s.station_id != current_station.station_id
    ]
    if same_node_alts:
        return True

    # Option 2: walk forward along the remaining route until we find the
    # next node with a station we can reach within the reserve.
    cumulative_energy = 0.0
    for i in range(len(remaining_route) - 1):
        edge_energy = compute_segment_energy(
            [remaining_route[i], remaining_route[i + 1]],
            network,
            agent.consumption_kwh_per_km,
        )
        cumulative_energy += edge_energy
        if agent.current_soc_kwh - cumulative_energy < reserve_kwh:
            return False  # Cannot reach this far, no safe alternative
        next_node = remaining_route[i + 1]
        if next_node in stations_by_node and next_node != current_station.node_id:
            return True
    return False


# ---------------------------------------------------------------------------
# Emergency charging
# ---------------------------------------------------------------------------

def _handle_emergency_charge(
    env: simpy.Environment,
    agent: VehicleAgent,
    segment_nodes: List[str],
    network: RoadNetwork,
    stations_by_node: Dict[str, List[ChargingStation]],
    config: Dict,
    collector: ResultsCollector,
) -> Generator:
    """
    Handle low-SOC emergency: find and use the nearest station on the segment.

    Yields True if emergency charging succeeded, False if agent strands.
    """
    # Find the nearest station reachable from current position within the segment
    station = find_emergency_station(
        agent, segment_nodes, stations_by_node, network, config
    )

    if station is None:
        logger.warning(
            "[t=%.0f] Agent %s stranded at %s: SOC=%.1f, no reachable station",
            env.now, agent.agent_id, agent.current_node, agent.current_soc_kwh,
        )
        return

    # Drive to emergency station
    path_to_station = network.subpath_up_to_node(segment_nodes, station.node_id)
    em_drive_time = network.subpath_travel_time_min(path_to_station)
    em_energy = compute_segment_energy(
        path_to_station, network, agent.consumption_kwh_per_km
    )

    yield env.timeout(em_drive_time)
    agent.current_soc_kwh = max(0.0, agent.current_soc_kwh - em_energy)
    agent.current_node = station.node_id

    # Charge
    yield from _execute_charging(
        env, agent, station,
        segment_nodes, network, stations_by_node, config, collector
    )
    return True


def _record_strand(
    agent: VehicleAgent,
    sim_time: float,
    collector: ResultsCollector,
    reason: str = "soc_depleted",
) -> None:
    agent.status = "stranded"
    agent.end_time_min = sim_time
    agent.failure_reason = reason
    collector.record_completion(
        agent_id=agent.agent_id,
        origin=agent.origin,
        destination=agent.destination,
        departure_time_min=agent.departure_time_min,
        arrival_time_min=sim_time,
        status="stranded",
        num_charge_stops=len(agent.charge_events),
        total_wait_min=agent.total_wait_time_min,
        total_charge_min=agent.total_charge_time_min,
        total_energy_kwh=agent.total_energy_charged_kwh,
        total_distance_km=agent.total_distance_km,
        route_node_count=len(agent.route),
        final_soc_kwh=agent.current_soc_kwh,
        failure_reason=reason,
    )
