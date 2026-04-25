"""Route planning with charging stops.

Core algorithm
--------------
1. Find the time-shortest path from origin to destination.
2. Walk the path, tracking SOC segment by segment.
3. Whenever projected SOC would fall below the reserve threshold before
   the next available station (or destination), insert the best reachable
   station as a mandatory charging stop.
4. Return the full route (node list) and the ordered list of station stops.

This produces a *just-in-time* charging plan — the minimal set of stops
needed to complete the trip.  The en-route module (en_route.py) may later
decide to skip a planned stop (if queue is too long and SOC allows) or to
add an unplanned stop (emergency).

Design principles
-----------------
- The plan is computed once at departure using *expected* station queue
  states.  It is revised en-route as the agent observes actual conditions.
- Stations on the exact shortest-path route are preferred over detours.
  A detour penalty (via the generalized cost in station_choice.py) is
  applied when the nearest station requires leaving the route.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.agent import VehicleAgent
from models.network import RoadNetwork
from models.station import ChargingStation
from behavior.energy import compute_segment_energy, compute_charge_duration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Waypoint: one driving segment ending at a stop (charge or destination)
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    """
    A contiguous driving segment of the trip.

    nodes:             ordered node list for this segment (first = current, last = end_node)
    end_node:          the node the agent arrives at after driving this segment
    is_charging_stop:  True if a charge is planned at end_node
    station:           the planned station at end_node (None if not a stop)
    """

    nodes: List[str]
    end_node: str
    is_charging_stop: bool = False
    station: Optional[ChargingStation] = None

    @property
    def start_node(self) -> str:
        return self.nodes[0] if self.nodes else ""


# ---------------------------------------------------------------------------
# Main routing function
# ---------------------------------------------------------------------------

def plan_route_with_stops(
    agent: VehicleAgent,
    network: RoadNetwork,
    stations_by_node: Dict[str, List[ChargingStation]],
    config: Dict,
) -> Tuple[List[str], List[ChargingStation]]:
    """
    Compute an energy-feasible route + ordered list of charging stops.

    Route choice uses generalized cost (travel time + toll / VoT) so that
    high-VoT agents prefer fast AP- toll motorways while budget-conscious
    agents take the slower parallel free A-/N- roads.

    A driver's app does not just pick the fastest road; it picks the fastest
    road where charging is feasible. To mirror that, this function enumerates
    the top-K routes by generalized cost and returns the first one whose
    charging plan is feasible. If none are feasible, it falls back to the
    primary (GC-shortest) route with its partial plan and lets the simulation
    surface the strand.

    Returns
    -------
    route:          full node list from origin to destination
    charging_stops: ordered list of ChargingStation objects the agent
                    plans to use (may be empty for short trips)
    """
    preferred = _validated_preferred_route(agent, network)
    if preferred:
        stops = _plan_charging_stops(
            agent, preferred, network, stations_by_node, config
        )
        if _is_stop_plan_feasible(agent, preferred, stops, network, config):
            agent.planned_route_source = "preferred_calibrated"
            return preferred, stops
        agent.route_infeasible_events += 1
        logger.debug(
            "Agent %s: calibrated path %s infeasible, falling back to route choice.",
            agent.agent_id,
            agent.demand_path_id or "<unknown>",
        )

    k = int(config.get("route_candidates_k", 5))
    candidates = network.k_shortest_paths_gc(
        agent.origin,
        agent.destination,
        agent.value_of_time_eur_per_hour,
        agent.max_comfortable_speed_kmh,
        k=k,
    )
    candidates = [route for route in candidates if is_geo_terminal_route(route)]
    if not candidates:
        logger.warning(
            "Agent %s: no path found %s -> %s",
            agent.agent_id, agent.origin, agent.destination,
        )
        return [], []

    primary_route = candidates[0]
    primary_stops: List[ChargingStation] = []

    for i, route in enumerate(candidates):
        stops = _plan_charging_stops(
            agent, route, network, stations_by_node, config
        )
        if _is_stop_plan_feasible(agent, route, stops, network, config):
            if i > 0:
                logger.debug(
                    "Agent %s: primary route infeasible, using candidate %d of %d.",
                    agent.agent_id, i + 1, len(candidates),
                )
                agent.route_infeasible_events += i
                agent.planned_route_source = "fallback_candidate"
            else:
                agent.planned_route_source = "generalized_cost_primary"
            return route, stops
        if i == 0:
            primary_stops = stops

    # All candidates infeasible. Return the primary plan and let the engine
    # surface the strand with a clear failure_reason.
    agent.route_infeasible_events += len(candidates)
    agent.planned_route_source = "infeasible_primary"
    logger.debug(
        "Agent %s: no feasible route among %d candidates, returning primary.",
        agent.agent_id, len(candidates),
    )
    return primary_route, primary_stops


def _validated_preferred_route(
    agent: VehicleAgent,
    network: RoadNetwork,
) -> List[str]:
    """Return the calibrated route if it is a valid graph path for this agent."""
    route = list(getattr(agent, "preferred_route", []) or [])
    if len(route) < 2:
        return []
    if route[0] != agent.origin or route[-1] != agent.destination:
        logger.debug(
            "Agent %s: ignoring calibrated path with endpoint mismatch.",
            agent.agent_id,
        )
        return []
    if not is_geo_terminal_route(route):
        logger.debug(
            "Agent %s: ignoring calibrated path with intermediate GEO boundary node.",
            agent.agent_id,
        )
        return []
    for u, v in zip(route, route[1:]):
        attrs = network.get_edge_attrs(u, v)
        if not attrs or not attrs.get("geometry_backed", False):
            logger.debug(
                "Agent %s: ignoring calibrated path with invalid edge %s -> %s.",
                agent.agent_id, u, v,
            )
            return []
    return route


def is_geo_terminal_route(route: List[str]) -> bool:
    """GEO boundary nodes may be route endpoints, never intermediate nodes."""
    if len(route) <= 2:
        return True
    return not any(_is_geo_node(node) for node in route[1:-1])


def _is_geo_node(node_id: str) -> bool:
    return str(node_id).startswith("GEO_")


def _destination_reserve_kwh(
    agent: VehicleAgent,
    reserve_kwh: float,
    no_dest_charger_frac: float,
) -> float:
    if agent.destination_charging_access or _is_geo_node(agent.destination):
        return reserve_kwh
    return agent.usable_capacity_kwh * no_dest_charger_frac


def _is_stop_plan_feasible(
    agent: VehicleAgent,
    route: List[str],
    stops: List[ChargingStation],
    network: RoadNetwork,
    config: Dict,
) -> bool:
    """Walk the route against the stop plan and verify SOC stays viable."""
    if len(route) < 2:
        return True

    reserve_frac = config.get("min_reserve_soc_fraction", 0.10)
    reserve_kwh = agent.usable_capacity_kwh * reserve_frac
    no_dest_frac = config.get("no_dest_charger_arrival_soc_fraction", 0.50)
    dest_reserve_kwh = _destination_reserve_kwh(agent, reserve_kwh, no_dest_frac)

    # Index of every planned stop in the route (by first matching node_id).
    stop_node_ids = {s.node_id for s in stops}
    stop_idx_list: List[int] = [
        i for i, n in enumerate(route) if n in stop_node_ids
    ]

    soc = agent.current_soc_kwh
    prev = 0
    checkpoints = stop_idx_list + [len(route) - 1]

    for cp in checkpoints:
        if cp <= prev:
            continue
        segment = route[prev: cp + 1]
        energy = compute_segment_energy(
            segment, network, agent.consumption_kwh_per_km
        )
        arriving = soc - energy
        threshold = dest_reserve_kwh if cp == len(route) - 1 else reserve_kwh
        if arriving < threshold:
            return False
        # Simulate charging at an intermediate stop to 80%.
        if cp != len(route) - 1:
            target_soc = agent.usable_capacity_kwh * 0.80
            soc = max(arriving, min(target_soc, agent.usable_capacity_kwh))
        else:
            soc = arriving
        prev = cp
    return True


def _plan_charging_stops(
    agent: VehicleAgent,
    route: List[str],
    network: RoadNetwork,
    stations_by_node: Dict[str, List[ChargingStation]],
    config: Dict,
) -> List[ChargingStation]:
    """
    Walk the route and insert the minimum necessary charging stops.

    Strategy (queue-aware greedy):
    - At each iteration, collect ALL waypoints reachable with comfortable SOC.
    - Among them, pick the one with the best queue-penalised progress score:
        score = distance_progress_km - wait_km_equivalent
      where wait_km_equivalent converts expected queue wait to equivalent
      driving distance at interurban speed, scaled by agent.queue_aversion.
      This causes agents to prefer a closer, emptier station over a farther,
      massively congested one when the queue penalty exceeds the distance gain.
    - If no waypoint is reachable with comfort SOC, fall back to the farthest
      reachable with hard reserve (emergency, queue-awareness suppressed).

    Station-within-node selection also prefers lower queue ratio over raw power,
    so agents at a node with multiple stations use the less-loaded one.
    """
    min_reserve = config.get("min_reserve_soc_fraction", 0.10)
    comfort_frac = config.get("arrival_comfort_soc_fraction", 0.20)
    no_dest_charger_frac = config.get("no_dest_charger_arrival_soc_fraction", 0.50)
    reserve_kwh = agent.usable_capacity_kwh * min_reserve
    dest_reserve_kwh = _destination_reserve_kwh(
        agent, reserve_kwh, no_dest_charger_frac
    )
    comfort_kwh = agent.usable_capacity_kwh * (min_reserve + comfort_frac)

    # Find all stations along the route, picking the least-loaded station at
    # each node (tie-break: highest power).
    station_indices: Dict[int, ChargingStation] = {}
    for idx, node_id in enumerate(route):
        if node_id in stations_by_node:
            best = max(
                stations_by_node[node_id],
                key=lambda s: (
                    -(s.current_queue_length() / max(1, s.num_connectors)),
                    s.max_power_kw,
                ),
            )
            station_indices[idx] = best

    destination_idx = len(route) - 1
    candidates = sorted(set(list(station_indices.keys()) + [destination_idx]))

    stops: List[ChargingStation] = []
    soc = agent.current_soc_kwh
    current_idx = 0

    while current_idx < destination_idx:
        # --- Pass 1: collect all waypoints reachable with comfortable SOC ---
        reachable: List[Tuple[int, float]] = []  # (waypoint_idx, arriving_soc)
        for waypoint_idx in candidates:
            if waypoint_idx <= current_idx:
                continue
            segment = route[current_idx : waypoint_idx + 1]
            energy_needed = compute_segment_energy(
                segment, network, agent.consumption_kwh_per_km
            )
            arriving_soc = soc - energy_needed
            threshold = dest_reserve_kwh if waypoint_idx == destination_idx else comfort_kwh
            if arriving_soc >= threshold:
                reachable.append((waypoint_idx, arriving_soc))

        if reachable:
            best_idx = _pick_queue_aware_stop(
                reachable, station_indices, destination_idx,
                soc, agent.consumption_kwh_per_km, agent.queue_aversion,
                agent.max_comfortable_speed_kmh,
            )
        else:
            best_idx = None

        # --- Pass 2 fallback: farthest reachable with only hard reserve ---
        if best_idx is None:
            for waypoint_idx in reversed(candidates):
                if waypoint_idx <= current_idx:
                    continue
                segment = route[current_idx : waypoint_idx + 1]
                energy_needed = compute_segment_energy(
                    segment, network, agent.consumption_kwh_per_km
                )
                threshold = dest_reserve_kwh if waypoint_idx == destination_idx else reserve_kwh
                if soc - energy_needed >= threshold:
                    best_idx = waypoint_idx
                    break

        if best_idx is None:
            agent.route_infeasible_events += 1
            logger.debug(
                "Agent %s: route infeasible, no reachable station from node %s",
                agent.agent_id, route[current_idx],
            )
            break

        # Compute arriving SOC at chosen stop
        segment = route[current_idx : best_idx + 1]
        energy_needed = compute_segment_energy(
            segment, network, agent.consumption_kwh_per_km
        )
        arriving_soc = soc - energy_needed

        if best_idx == destination_idx:
            soc = arriving_soc
            current_idx = best_idx
        else:
            station = station_indices[best_idx]
            stops.append(station)
            target_soc = agent.usable_capacity_kwh * 0.80
            soc = max(arriving_soc, min(target_soc, agent.usable_capacity_kwh))
            current_idx = best_idx

    return stops


def _pick_queue_aware_stop(
    reachable: List[Tuple[int, float]],
    station_indices: Dict[int, "ChargingStation"],
    destination_idx: int,
    current_soc: float,
    consumption_kwh_per_km: float,
    queue_aversion: float,
    speed_kmh: float,
) -> int:
    """
    Among reachable (waypoint_idx, arriving_soc) pairs, return the index that
    maximises queue-penalised progress:

        score = progress_km - wait_km_equivalent

    where:
        progress_km        = energy consumed to reach stop / consumption rate
        wait_km_equivalent = expected_wait_min × speed_kmh / 60 × queue_aversion

    Intuitively: the agent trades off "how far does this stop advance me" against
    "how much time (in equivalent km) will I lose waiting in the queue there."
    If the farthest station has a smaller score than a closer one, the agent
    prefers the closer one — it makes more real progress per minute spent.

    Destination is always preferred when reachable (no queue cost).
    """
    # Destination reachable → take it, no charging stop needed.
    for idx, _ in reachable:
        if idx == destination_idx:
            return destination_idx

    best_idx = reachable[-1][0]  # default: farthest reachable
    best_score = float("-inf")

    for waypoint_idx, arriving_soc in reachable:
        station = station_indices.get(waypoint_idx)
        if station is None:
            continue

        energy_consumed = current_soc - arriving_soc
        progress_km = energy_consumed / max(consumption_kwh_per_km, 1e-9)

        wait_min = station.expected_wait_time_min()
        wait_km_equiv = wait_min * speed_kmh / 60.0 * queue_aversion

        score = progress_km - wait_km_equiv
        if score > best_score:
            best_score = score
            best_idx = waypoint_idx

    return best_idx


def _find_farthest_reachable_station(
    current_idx: int,
    route: List[str],
    station_indices: Dict[int, ChargingStation],
    current_soc: float,
    reserve_kwh: float,
    agent: VehicleAgent,
    network: RoadNetwork,
) -> Optional[int]:
    """Return the index of the farthest reachable station from current position."""
    best_idx = None
    for idx in sorted(station_indices.keys()):
        if idx <= current_idx:
            continue
        segment = route[current_idx : idx + 1]
        energy = compute_segment_energy(segment, network, agent.consumption_kwh_per_km)
        if current_soc - energy >= reserve_kwh:
            best_idx = idx
    return best_idx


# ---------------------------------------------------------------------------
# Waypoint builder
# ---------------------------------------------------------------------------

def build_trip_waypoints(
    route: List[str],
    charging_stops: List[ChargingStation],
) -> List[Waypoint]:
    """
    Break the full route into driving segments separated by charging stops.

    Example
    -------
    route = [A, B, C, D, E]
    stops at nodes C and D

    Result:
        Waypoint([A, B, C], end_node=C, is_charging_stop=True, station=...)
        Waypoint([C, D],    end_node=D, is_charging_stop=True, station=...)
        Waypoint([D, E],    end_node=E, is_charging_stop=False)
    """
    if len(route) < 2:
        return []

    stop_node_map: Dict[str, ChargingStation] = {
        s.node_id: s for s in charging_stops
    }

    waypoints: List[Waypoint] = []
    segment_start = 0

    for idx in range(1, len(route)):
        node = route[idx]
        is_last = idx == len(route) - 1
        is_stop = node in stop_node_map

        if is_stop or is_last:
            segment_nodes = route[segment_start : idx + 1]
            waypoints.append(
                Waypoint(
                    nodes=segment_nodes,
                    end_node=node,
                    is_charging_stop=is_stop and not is_last,
                    station=stop_node_map.get(node),
                )
            )
            segment_start = idx

    return waypoints
