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

    Returns
    -------
    route:          full node list from origin to destination
    charging_stops: ordered list of ChargingStation objects the agent
                    plans to use (may be empty for short trips)
    """
    route = network.shortest_path(agent.origin, agent.destination)
    if not route:
        logger.warning(
            "Agent %s: no path found %s → %s",
            agent.agent_id, agent.origin, agent.destination,
        )
        return [], []

    charging_stops = _plan_charging_stops(agent, route, network, stations_by_node, config)
    return route, charging_stops


def _plan_charging_stops(
    agent: VehicleAgent,
    route: List[str],
    network: RoadNetwork,
    stations_by_node: Dict[str, List[ChargingStation]],
    config: Dict,
) -> List[ChargingStation]:
    """
    Walk the route and insert the minimum necessary charging stops.

    Strategy (farthest-reachable greedy):
    - In each iteration, find the FARTHEST waypoint (station or destination)
      the agent can reach from the current position with reserve SOC intact.
    - If that farthest waypoint is the destination, the agent drives straight
      through without stopping.
    - If it is an intermediate station, the agent stops there and charges to
      80% before the next iteration.
    - If no waypoint is reachable at all, the route is infeasible.

    This avoids the earlier "skip if SOC > 50%" heuristic, which allowed the
    planner to pass through a needed station and then get stuck with nothing
    reachable ahead.
    """
    min_reserve = config.get("min_reserve_soc_fraction", 0.10)
    reserve_kwh = agent.usable_capacity_kwh * min_reserve

    # Find all stations along the route (by node index)
    station_indices: Dict[int, ChargingStation] = {}
    for idx, node_id in enumerate(route):
        if node_id in stations_by_node:
            # Pick the highest-power station at this node
            best = max(stations_by_node[node_id], key=lambda s: s.max_power_kw)
            station_indices[idx] = best

    destination_idx = len(route) - 1
    candidates = sorted(set(sorted(station_indices.keys()) + [destination_idx]))

    stops: List[ChargingStation] = []
    soc = agent.current_soc_kwh
    current_idx = 0

    while current_idx < destination_idx:
        # Find the FARTHEST reachable candidate from current position
        best_idx = None
        for waypoint_idx in reversed(candidates):
            if waypoint_idx <= current_idx:
                continue
            segment = route[current_idx : waypoint_idx + 1]
            energy_needed = compute_segment_energy(
                segment, network, agent.consumption_kwh_per_km
            )
            if soc - energy_needed >= reserve_kwh:
                best_idx = waypoint_idx
                break

        if best_idx is None:
            logger.warning(
                "Agent %s: route infeasible, no reachable station from node %s",
                agent.agent_id, route[current_idx],
            )
            break

        # Compute arriving SOC
        segment = route[current_idx : best_idx + 1]
        energy_needed = compute_segment_energy(
            segment, network, agent.consumption_kwh_per_km
        )
        arriving_soc = soc - energy_needed

        if best_idx == destination_idx:
            # Reached destination without needing another stop
            soc = arriving_soc
            current_idx = best_idx
        else:
            # Intermediate station — always stop and charge to 80%
            station = station_indices[best_idx]
            stops.append(station)
            target_soc = agent.usable_capacity_kwh * 0.80
            soc = max(arriving_soc, min(target_soc, agent.usable_capacity_kwh))
            current_idx = best_idx

    return stops


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
