"""En-route adaptive charging decisions.

This module answers three questions the agent faces during the trip:

1. decide_to_charge_here(agent, station, ...)
   At a planned charging stop: which station should the agent use?
   It searches ALL stations reachable within current SOC — including
   backward/detour paths — and returns the one minimising total trip
   time (drive + wait + drive to destination).  The engine drives to
   whichever station is returned.

2. find_emergency_station(agent, nodes_ahead, ...)
   The agent's SOC is critically low.  Find the nearest reachable station
   even if it means stopping earlier than planned.

3. should_add_unplanned_stop(agent, node_id, ...)
   At an unplanned node that has a station: is it worth stopping now
   to top up, given current conditions?

Design
------
These functions are called from simulation/engine.py during the SimPy
process execution.  They must be fast (no heavy computation) since they
run inside the simulation loop.

The decision rules use the same generalized cost as station_choice.py
so that scenario changes (price, capacity) propagate correctly into
adaptive behavior.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import networkx as nx

from models.agent import VehicleAgent
from models.network import RoadNetwork
from models.station import ChargingStation
from behavior.energy import compute_segment_energy
from behavior.station_choice import (
    generalized_cost_for_station,
    decide_charge_target,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision 1: stop at planned station, detour to a better one, or skip?
# ---------------------------------------------------------------------------

def decide_to_charge_here(
    agent: VehicleAgent,
    station: ChargingStation,
    remaining_route: List[str],
    stations_by_node: Dict[str, List[ChargingStation]],
    network: RoadNetwork,
    config: Dict,
) -> Optional[ChargingStation]:
    """
    Decide whether and where to charge given the current planned stop.

    Returns the ChargingStation the agent should charge at next:
      - The *current* station  → charge here (no detour needed).
      - A *different* station  → skip this stop and drive to the returned
                                  station instead.  The engine handles the
                                  drive; the station may be forward, backward,
                                  or on a completely different road — the
                                  decision is purely time-optimal.
      - None                   → skip charging entirely (destination is
                                  reachable without any more stopping, and
                                  an opportunistic top-up is not worthwhile).

    The decision minimises estimated total remaining trip time across ALL
    stations reachable within current SOC from the agent's current node:

        total_time = drive_to_station + expected_wait + drive_station_to_dest
    """
    min_reserve_frac = config.get("min_reserve_soc_fraction", 0.10)
    reserve_kwh = agent.usable_capacity_kwh * min_reserve_frac
    no_dest_frac = config.get("no_dest_charger_arrival_soc_fraction", 0.50)
    dest_reserve_kwh = (
        agent.usable_capacity_kwh * no_dest_frac
        if not agent.destination_charging_access
        else reserve_kwh
    )

    destination = remaining_route[-1] if remaining_route else agent.destination

    chosen = _find_best_charging_option(
        agent=agent,
        current_station=station,
        destination=destination,
        reserve_kwh=reserve_kwh,
        dest_reserve_kwh=dest_reserve_kwh,
        stations_by_node=stations_by_node,
        network=network,
    )

    if chosen is None:
        # Destination reachable without any more charging.
        if _is_topping_up_worthwhile(agent, config):
            logger.debug(
                "Agent %s: dest reachable but topping up at %s",
                agent.agent_id, station.station_id,
            )
            return station
        return None

    if chosen.station_id != station.station_id:
        logger.debug(
            "Agent %s: rerouting from %s to %s (lower estimated trip time)",
            agent.agent_id, station.station_id, chosen.station_id,
        )

    return chosen


def _find_best_charging_option(
    agent: VehicleAgent,
    current_station: ChargingStation,
    destination: str,
    reserve_kwh: float,
    dest_reserve_kwh: float,
    stations_by_node: Dict[str, List[ChargingStation]],
    network: RoadNetwork,
) -> Optional[ChargingStation]:
    """
    Search ALL stations reachable from agent.current_node within SOC budget
    (including backward paths and detours) and return the one that minimises:

        total_time = drive_time_to_station + expected_wait + drive_time_to_dest

    Returns None if the destination is reachable without any charging stop.
    Returns current_station if staying is optimal.
    """
    current_node = agent.current_node
    max_range_km = (agent.current_soc_kwh - reserve_kwh) / max(
        agent.consumption_kwh_per_km, 1e-9
    )

    if max_range_km <= 0:
        return current_station  # forced to stay, no range left

    # Dijkstra from current position — explores forward AND backward edges.
    try:
        dist_to: Dict[str, float] = nx.single_source_dijkstra_path_length(
            network.graph, current_node, cutoff=max_range_km, weight="distance_km"
        )
    except nx.NodeNotFound:
        return current_station

    # Check if destination is reachable without any more charging.
    dist_to_dest_km = dist_to.get(destination, float("inf"))
    energy_to_dest = dist_to_dest_km * agent.consumption_kwh_per_km
    if agent.current_soc_kwh - energy_to_dest >= dest_reserve_kwh:
        return None

    # Memoised travel-time from a node to the destination.
    _time_to_dest_cache: Dict[str, float] = {}

    def _time_to_dest(node: str) -> float:
        if node not in _time_to_dest_cache:
            _time_to_dest_cache[node] = network.shortest_path_length(
                node, destination, "travel_time_min"
            )
        return _time_to_dest_cache[node]

    best_station: Optional[ChargingStation] = None
    best_cost = float("inf")

    # --- Current station: already here, zero detour drive time ---
    cost_here = current_station.expected_wait_time_min() + _time_to_dest(current_node)
    if cost_here < best_cost:
        best_cost = cost_here
        best_station = current_station

    # --- Other stations at current node (if multiple stations share a node) ---
    for s in stations_by_node.get(current_node, []):
        if s.station_id == current_station.station_id:
            continue
        cost = s.expected_wait_time_min() + _time_to_dest(current_node)
        if cost < best_cost:
            best_cost = cost
            best_station = s

    # --- All other reachable station nodes ---
    for node, dist_km in dist_to.items():
        if node == current_node or node not in stations_by_node:
            continue

        energy_needed = dist_km * agent.consumption_kwh_per_km
        if agent.current_soc_kwh - energy_needed < reserve_kwh:
            continue  # would arrive below hard reserve

        drive_time_to = network.shortest_path_length(
            current_node, node, "travel_time_min"
        )
        if drive_time_to == float("inf"):
            continue

        time_from = _time_to_dest(node)
        if time_from == float("inf"):
            continue  # station not connected onward to destination

        # Pick least-loaded station at this node (ties broken by highest power).
        alt_station = max(
            stations_by_node[node],
            key=lambda s: (
                -(s.current_queue_length() / max(1, s.num_connectors)),
                s.max_power_kw,
            ),
        )
        cost = drive_time_to + alt_station.expected_wait_time_min() + time_from

        if cost < best_cost:
            best_cost = cost
            best_station = alt_station

    return best_station


def _is_topping_up_worthwhile(
    agent: VehicleAgent,
    config: Dict,
) -> bool:
    """Check if a top-up stop is worth it even when not strictly needed."""
    min_charge_frac = config.get("min_charge_gain_fraction", 0.10)
    min_charge_kwh = agent.usable_capacity_kwh * min_charge_frac
    headroom = agent.usable_capacity_kwh - agent.current_soc_kwh
    return headroom >= min_charge_kwh


# ---------------------------------------------------------------------------
# Decision 2: emergency station search
# ---------------------------------------------------------------------------

def find_emergency_station(
    agent: VehicleAgent,
    remaining_route: List[str],
    stations_by_node: Dict[str, List[ChargingStation]],
    network: RoadNetwork,
    config: Dict,
) -> Optional[ChargingStation]:
    """
    Find the nearest reachable station when SOC is critically low.

    Checks the agent's current node first (remaining_route[0]), then
    searches forward along remaining_route for the first station
    reachable with at least the minimum reserve SOC.
    Returns None if no reachable station exists (agent will strand).
    """
    min_reserve_frac = config.get("min_reserve_soc_fraction", 0.10)
    reserve_kwh = agent.usable_capacity_kwh * min_reserve_frac

    # Check current node first — the agent is already here, zero driving needed
    current_node = remaining_route[0] if remaining_route else None
    if current_node and current_node in stations_by_node:
        return max(stations_by_node[current_node], key=lambda s: s.max_power_kw)

    cumulative_energy = 0.0
    for i in range(len(remaining_route) - 1):
        node = remaining_route[i]
        next_node = remaining_route[i + 1]
        edge_energy = compute_segment_energy(
            [node, next_node], network, agent.consumption_kwh_per_km
        )
        cumulative_energy += edge_energy

        if agent.current_soc_kwh - cumulative_energy < reserve_kwh:
            break  # Can't reach nodes beyond here

        if next_node in stations_by_node:
            # Return the highest-power (fastest) station at this node
            return max(stations_by_node[next_node], key=lambda s: s.max_power_kw)

    return None


# ---------------------------------------------------------------------------
# Decision 3: unplanned top-up at an opportunistic station
# ---------------------------------------------------------------------------

def should_add_unplanned_stop(
    agent: VehicleAgent,
    station: ChargingStation,
    _remaining_distance_km: float,
    stations_ahead_count: int,
    config: Dict,
) -> bool:
    """
    Decide whether to stop at an unplanned station for an opportunistic top-up.

    Called when the agent passes through a node that has a station but
    that was NOT in the original charging plan.

    Returns True if the agent decides to top up here.
    """
    min_charge_frac = config.get("min_charge_gain_fraction", 0.10)
    headroom_kwh = agent.usable_capacity_kwh - agent.current_soc_kwh
    min_charge_kwh = agent.usable_capacity_kwh * min_charge_frac

    # Only consider stopping if there's meaningful headroom
    if headroom_kwh < min_charge_kwh:
        return False

    # Don't stop if queue is unreasonably long
    expected_wait = station.expected_wait_time_min()
    max_tolerance = config.get("max_queue_wait_tolerance_min", 30.0) * 0.5
    if expected_wait > max_tolerance / agent.queue_aversion:
        return False

    # Stop if SOC is below 40% and there are no stations ahead
    soc_frac = agent.current_soc_kwh / agent.usable_capacity_kwh
    if soc_frac < 0.40 and stations_ahead_count == 0:
        return True

    # Otherwise: risk-averse agents stop proactively; risk-tolerant don't
    soc_threshold = 0.60 - agent.risk_tolerance * 0.30
    # risk=0 → stop if below 60%; risk=1 → stop if below 30%
    return soc_frac < soc_threshold
