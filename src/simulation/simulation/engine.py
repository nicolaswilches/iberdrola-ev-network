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
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import simpy

from models.agent import VehicleAgent
from models.network import RoadNetwork
from models.results import ResultsCollector
from models.station import ChargingStation
from behavior.energy import compute_segment_energy, compute_charge_duration
from behavior.pre_trip import decide_initial_soc
from behavior.routing import (
    build_trip_waypoints,
    is_geo_terminal_route,
    plan_route_with_stops,
    Waypoint,
)
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
    od_flow_observer: Optional[Callable] = None,
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
            **_trip_diagnostic_kwargs(agent, network),
            failure_reason="no_path_found",
        )
        _record_failure_diagnostic(
            agent, env.now, "failed_no_route", "no_path_found",
            network, stations_by_node, config, collector,
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
    # While loop (not for) so that detour + replan can replace the waypoint
    # list mid-trip without restarting the entire process.
    wi = 0
    while wi < len(waypoints):
        waypoint = waypoints[wi]

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
                    network=network,
                    stations_by_node=stations_by_node,
                    config=config,
                    reason="no_reachable_station",
                )
                return

        # Drive the segment
        drive_time = network.subpath_travel_time_min(waypoint.nodes)
        _record_edge_traversals(
            agent, env.now, waypoint.nodes, network, collector, od_flow_observer
        )
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
            chosen_station = decide_to_charge_here(
                agent=agent,
                station=waypoint.station,
                remaining_route=remaining_route,
                stations_by_node=stations_by_node,
                network=network,
                config=config,
            )

            if chosen_station is None:
                # Skip charging — destination reachable and top-up not worthwhile.
                pass

            elif chosen_station.node_id == waypoint.end_node:
                # Charge at this node (may be current or a different station
                # at the same node — no driving required either way).
                charged = yield from _execute_charging(
                    env, agent, chosen_station,
                    route, network, stations_by_node, config, collector
                )
                # On queue abandonment the emergency path takes over if SOC
                # drops low on the next segment (checked at loop top).
                if charged is False:
                    logger.debug(
                        "Agent %s abandoned %s; relying on emergency path.",
                        agent.agent_id, chosen_station.station_id,
                    )

            else:
                # Detour to a different node (forward, backward, or off-route).
                result = yield from _execute_detour_and_replan(
                    env, agent, chosen_station,
                    network, stations_by_node, config, collector,
                )
                if result is not None:
                    route, waypoints = result
                    agent.route = route
                    wi = 0
                    continue
                # Detour infeasible (SOC race condition) — fall back to planned stop.
                yield from _execute_charging(
                    env, agent, waypoint.station,
                    route, network, stations_by_node, config, collector
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

        wi += 1

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
        **_trip_diagnostic_kwargs(agent, network),
    )


# ---------------------------------------------------------------------------
# Detour sub-process
# ---------------------------------------------------------------------------

def _execute_detour_and_replan(
    env: simpy.Environment,
    agent: VehicleAgent,
    chosen_station: ChargingStation,
    network: RoadNetwork,
    stations_by_node: Dict[str, List[ChargingStation]],
    config: Dict,
    collector: ResultsCollector,
) -> Generator:
    """
    Drive to chosen_station (which may be backward or off the current route),
    charge there, then replan the remaining trip to destination.

    Returns (new_route, new_waypoints) on success, or None if the detour is
    no longer feasible (SOC race condition between decision and execution).
    """
    # Path to the chosen station from current position.
    detour_path = network.shortest_path_gc(
        agent.current_node,
        chosen_station.node_id,
        agent.value_of_time_eur_per_hour,
        getattr(agent, "max_comfortable_speed_kmh", None),
    )
    if not detour_path or len(detour_path) < 2 or not is_geo_terminal_route(detour_path):
        return None

    detour_energy = compute_segment_energy(
        detour_path, network, agent.consumption_kwh_per_km
    )
    reserve_kwh = agent.usable_capacity_kwh * float(
        config.get("min_reserve_soc_fraction", 0.10)
    )
    if agent.current_soc_kwh - detour_energy < reserve_kwh:
        # SOC depleted more than expected since the decision was made.
        return None

    detour_time = network.subpath_travel_time_min(detour_path)
    logger.debug(
        "[t=%.0f] Agent %s detouring to %s (%.0f min, %.1f kWh)",
        env.now, agent.agent_id, chosen_station.station_id,
        detour_time, detour_energy,
    )

    _record_edge_traversals(agent, env.now, detour_path, network, collector)
    yield env.timeout(detour_time)
    agent.current_soc_kwh = max(0.0, agent.current_soc_kwh - detour_energy)
    agent.current_node = chosen_station.node_id

    # Build a context route (station → destination) for _execute_charging.
    context_route = network.shortest_path_gc(
        chosen_station.node_id,
        agent.destination,
        agent.value_of_time_eur_per_hour,
        getattr(agent, "max_comfortable_speed_kmh", None),
    ) or [chosen_station.node_id, agent.destination]
    if not is_geo_terminal_route(context_route):
        context_route = [chosen_station.node_id, agent.destination]

    yield from _execute_charging(
        env, agent, chosen_station, context_route,
        network, stations_by_node, config, collector,
    )

    # Replan remaining route from new position to destination.
    saved_origin = agent.origin
    agent.origin = agent.current_node
    new_route, new_stops = plan_route_with_stops(
        agent, network, stations_by_node, config
    )
    agent.origin = saved_origin

    if not new_route:
        new_route = network.shortest_path_gc(
            agent.current_node,
            agent.destination,
            agent.value_of_time_eur_per_hour,
            getattr(agent, "max_comfortable_speed_kmh", None),
        )
        if not is_geo_terminal_route(new_route):
            new_route = []
        new_stops = []

    if not new_route:
        return None

    new_waypoints = build_trip_waypoints(new_route, new_stops)
    return new_route, new_waypoints


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

    _record_edge_traversals(agent, env.now, path_to_station, network, collector)
    yield env.timeout(em_drive_time)
    agent.current_soc_kwh = max(0.0, agent.current_soc_kwh - em_energy)
    agent.current_node = station.node_id

    # Charge
    yield from _execute_charging(
        env, agent, station,
        segment_nodes, network, stations_by_node, config, collector
    )
    return True


def _record_edge_traversals(
    agent: VehicleAgent,
    sim_time_min: float,
    nodes: List[str],
    network: RoadNetwork,
    collector: Optional[ResultsCollector],
    od_flow_observer: Optional[Callable] = None,
) -> None:
    """Record each graph edge traversed by an agent for flow calibration."""
    for from_node, to_node, attrs in network.iter_edges_on_path(nodes):
        road_name = str(attrs.get("road_name", ""))
        distance_km = float(attrs.get("distance_km", 0.0) or 0.0)
        segment_ids = attrs.get("source_segment_ids", ())
        if isinstance(segment_ids, (list, tuple)):
            segment_ids_str = "|".join(str(x) for x in segment_ids)
        else:
            segment_ids_str = str(segment_ids)
        target_flow = float(attrs.get("target_daily_bev_traffic_2027", 0.0) or 0.0)
        if collector is not None:
            collector.record_edge_traversal(
                agent_id=agent.agent_id,
                sim_time_min=sim_time_min,
                from_node=from_node,
                to_node=to_node,
                road_name=road_name,
                distance_km=distance_km,
                source_segment_ids=segment_ids_str,
                target_daily_bev_traffic_2027=target_flow,
                demand_weight=agent.demand_weight,
            )
        agent.traversed_edges.append((from_node, to_node))
        if od_flow_observer is not None:
            od_flow_observer(road_name, agent.agent_id, sim_time_min)


def _record_strand(
    agent: VehicleAgent,
    sim_time: float,
    collector: ResultsCollector,
    network: RoadNetwork,
    stations_by_node: Dict[str, List[ChargingStation]],
    config: Dict,
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
        **_trip_diagnostic_kwargs(agent, network),
        failure_reason=reason,
    )
    _record_failure_diagnostic(
        agent, sim_time, "stranded", reason,
        network, stations_by_node, config, collector,
    )


def _trip_diagnostic_kwargs(
    agent: VehicleAgent,
    network: Optional[RoadNetwork],
) -> Dict[str, Any]:
    preferred_distance, actual_distance, adherence, exact = _path_adherence(
        agent, network
    )
    return {
        "od_pair_id": agent.od_pair_id,
        "demand_path_id": agent.demand_path_id,
        "vehicle_type": agent.vehicle_type,
        "usable_capacity_kwh": agent.usable_capacity_kwh,
        "preferred_path_distance_km": preferred_distance,
        "actual_route_distance_km": actual_distance,
        "path_adherence_ratio": adherence,
        "exact_preferred_path_match": exact,
        "route_infeasible_events": agent.route_infeasible_events,
        "demand_weight": agent.demand_weight,
    }


def _path_adherence(
    agent: VehicleAgent,
    network: Optional[RoadNetwork],
) -> Tuple[float, float, float, bool]:
    preferred = list(getattr(agent, "preferred_route", []) or [])
    traversed = list(getattr(agent, "traversed_edges", []) or [])
    preferred_edges = list(zip(preferred, preferred[1:])) if len(preferred) >= 2 else []

    preferred_distance = (
        network.subpath_distance_km(preferred)
        if network is not None and len(preferred) >= 2
        else 0.0
    )
    actual_route = list(getattr(agent, "route", []) or [])
    actual_distance = (
        network.subpath_distance_km(actual_route)
        if network is not None and len(actual_route) >= 2
        else agent.total_distance_km
    )

    if not preferred_edges:
        return preferred_distance, actual_distance, 0.0, False

    preferred_set = set(preferred_edges)
    traversed_set = set(traversed)
    adherence = len(preferred_set & traversed_set) / max(1, len(preferred_set))
    exact = traversed == preferred_edges
    return preferred_distance, actual_distance, float(adherence), bool(exact)


def _record_failure_diagnostic(
    agent: VehicleAgent,
    sim_time: float,
    status: str,
    reason: str,
    network: Optional[RoadNetwork],
    stations_by_node: Dict[str, List[ChargingStation]],
    config: Dict,
    collector: ResultsCollector,
) -> None:
    preferred_distance, actual_distance, adherence, exact = _path_adherence(
        agent, network
    )
    station_node, station_dist, station_energy, reachable_count = (
        _nearest_reachable_station(agent, network, stations_by_node, config)
    )
    initial_soc = collector._initial_soc_by_agent.get(agent.agent_id, 0.0)
    usable = max(agent.usable_capacity_kwh, 1e-9)
    collector.record_failure_diagnostic(
        agent_id=agent.agent_id,
        origin=agent.origin,
        destination=agent.destination,
        od_pair_id=agent.od_pair_id,
        demand_path_id=agent.demand_path_id,
        current_node=agent.current_node,
        status=status,
        failure_reason=reason,
        vehicle_type=agent.vehicle_type,
        battery_capacity_kwh=agent.battery_capacity_kwh,
        usable_capacity_kwh=agent.usable_capacity_kwh,
        initial_soc_kwh=initial_soc,
        current_soc_kwh=agent.current_soc_kwh,
        initial_soc_fraction=initial_soc / usable,
        current_soc_fraction=agent.current_soc_kwh / usable,
        consumption_kwh_per_km=agent.consumption_kwh_per_km,
        home_charging_access=agent.home_charging_access,
        destination_charging_access=agent.destination_charging_access,
        preferred_path_distance_km=preferred_distance,
        actual_route_distance_km=actual_distance,
        first_reachable_station_node=station_node,
        first_reachable_station_distance_km=station_dist,
        first_reachable_station_energy_kwh=station_energy,
        reachable_station_count=reachable_count,
        route_infeasible_events=agent.route_infeasible_events,
        path_adherence_ratio=adherence,
        exact_preferred_path_match=exact,
    )


def _nearest_reachable_station(
    agent: VehicleAgent,
    network: Optional[RoadNetwork],
    stations_by_node: Dict[str, List[ChargingStation]],
    config: Dict,
) -> Tuple[str, float, float, int]:
    if network is None or not agent.current_node:
        return "", 0.0, 0.0, 0

    reserve = agent.usable_capacity_kwh * float(
        config.get("min_reserve_soc_fraction", 0.10)
    )
    best_node = ""
    best_dist = float("inf")
    best_energy = 0.0
    reachable = 0

    for node_id in stations_by_node:
        path = network.shortest_path_gc(
            agent.current_node,
            node_id,
            agent.value_of_time_eur_per_hour,
            getattr(agent, "max_comfortable_speed_kmh", None),
            warn_on_missing=False,
        )
        if not path or len(path) < 2:
            continue
        energy = compute_segment_energy(path, network, agent.consumption_kwh_per_km)
        if agent.current_soc_kwh - energy < reserve:
            continue
        reachable += 1
        dist = network.subpath_distance_km(path)
        if dist < best_dist:
            best_node = node_id
            best_dist = dist
            best_energy = energy

    if not best_node:
        return "", 0.0, 0.0, reachable
    return best_node, float(best_dist), float(best_energy), reachable
