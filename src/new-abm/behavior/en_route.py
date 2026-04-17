"""En-route adaptive charging decisions.

This module answers three questions the agent faces during the trip:

1. decide_to_charge_here(agent, station, ...)
   At a planned charging stop: should the agent actually stop here,
   given the current queue?  Or should it skip to the next station?

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
from typing import Dict, List, Optional, Tuple

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
# Decision 1: stop at planned station or skip?
# ---------------------------------------------------------------------------

def decide_to_charge_here(
    agent: VehicleAgent,
    station: ChargingStation,
    remaining_route: List[str],
    stations_by_node: Dict[str, List[ChargingStation]],
    network: RoadNetwork,
    config: Dict,
) -> bool:
    """
    Decide whether to charge at a planned stop.

    The agent skips this station only if:
      - The queue is long enough to make waiting unacceptable, AND
      - There is a reachable alternative station further along the route, AND
      - The agent has enough SOC to reach that alternative.

    Returns True (charge here) or False (skip).
    """
    min_reserve_frac = config.get("min_reserve_soc_fraction", 0.10)
    reserve_kwh = agent.usable_capacity_kwh * min_reserve_frac
    no_dest_frac = config.get("no_dest_charger_arrival_soc_fraction", 0.50)
    dest_reserve_kwh = (agent.usable_capacity_kwh * no_dest_frac
                        if not agent.destination_charging_access
                        else reserve_kwh)

    # --- Is the agent forced to charge here? ---
    next_station_node, energy_to_next = _next_station_ahead(
        station.node_id, remaining_route, stations_by_node, network, agent
    )

    if next_station_node is None:
        remaining_distance = network.subpath_distance_km(remaining_route)
        energy_to_dest = remaining_distance * agent.consumption_kwh_per_km
        if agent.current_soc_kwh - energy_to_dest < dest_reserve_kwh:
            logger.debug(
                "Agent %s forced to charge at %s (no stations ahead)",
                agent.agent_id, station.station_id,
            )
            return True
        # Has enough to reach destination without charging
        # Check if topping up is still worthwhile
        return _is_topping_up_worthwhile(agent, station, config)

    # --- Has enough SOC to skip to next station? ---
    can_skip = (agent.current_soc_kwh - energy_to_next) >= reserve_kwh

    if not can_skip:
        logger.debug(
            "Agent %s cannot skip %s (insufficient SOC for next station)",
            agent.agent_id, station.station_id,
        )
        return True

    # --- Is the wait time acceptable? ---
    expected_wait = station.expected_wait_time_min()
    max_tolerance = config.get("max_queue_wait_tolerance_min", 30.0)
    # Scale by agent queue_aversion: a more averse agent has effectively lower tolerance
    tolerance = max_tolerance / agent.queue_aversion

    if expected_wait <= tolerance:
        return True  # Queue is fine, charge here

    # Queue is too long AND we can skip → skip
    logger.debug(
        "Agent %s skips %s (wait=%.1f min > tolerance=%.1f min)",
        agent.agent_id, station.station_id, expected_wait, tolerance,
    )
    return False


def _is_topping_up_worthwhile(
    agent: VehicleAgent,
    station: ChargingStation,
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
    remaining_distance_km: float,
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


# ---------------------------------------------------------------------------
# Helper: find next station ahead on the remaining route
# ---------------------------------------------------------------------------

def _next_station_ahead(
    current_node: str,
    remaining_route: List[str],
    stations_by_node: Dict[str, List[ChargingStation]],
    network: RoadNetwork,
    agent: VehicleAgent,
) -> Tuple[Optional[str], float]:
    """
    Return (node_id, cumulative_energy_kwh) for the first station ahead
    on the route after *current_node*.  Returns (None, inf) if none found.
    """
    cumulative_energy = 0.0
    in_scope = False

    for i in range(len(remaining_route) - 1):
        node = remaining_route[i]
        next_node = remaining_route[i + 1]

        if node == current_node:
            in_scope = True

        if in_scope:
            edge_energy = compute_segment_energy(
                [node, next_node], network, agent.consumption_kwh_per_km
            )
            cumulative_energy += edge_energy

            if next_node in stations_by_node and next_node != current_node:
                return next_node, cumulative_energy

    return None, float("inf")
