"""Station choice model.

Agents evaluate candidate stations using a generalized cost function
that combines time costs, monetary costs, and risk.

This is the key behavioral mechanism for endogenous demand redistribution:
  - Lower price at a station → lower GC → more agents choose it
  - Longer queue at a station → higher GC → agents avoid it
  - These forces reach a dynamic equilibrium during the simulation

Generalized cost formula
------------------------
    GC(agent, station) =
        VoT_eur_per_min * (
            detour_time_min
          + wait_time_min  * queue_aversion
          + charge_time_min
        )
      + price_sensitivity * charge_amount_kwh * price_per_kwh
      + reliability_risk_eur

All terms are in EUR.  Agents choose the station with the lowest GC
(subject to energy feasibility).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

from models.agent import VehicleAgent
from models.network import RoadNetwork
from models.station import ChargingStation
from behavior.energy import compute_charge_duration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core generalized cost
# ---------------------------------------------------------------------------

def generalized_cost_for_station(
    agent: VehicleAgent,
    station: ChargingStation,
    detour_time_min: float,
    target_soc_kwh: float,
    avg_session_min: float = 22.0,
) -> float:
    """
    Compute the generalized cost (EUR) of using *station* for *agent*.

    Parameters
    ----------
    agent:
        The evaluating agent (carries VoT, price_sensitivity, etc.).
    station:
        The station being evaluated.
    detour_time_min:
        Extra driving time to reach this station vs. staying on route.
        0 if the station is directly on the planned route.
    target_soc_kwh:
        The SOC the agent would charge to at this station (kWh).
    avg_session_min:
        Used to estimate wait time from current queue length.

    Returns
    -------
    Generalized cost in EUR (lower is better).
    """
    vot = agent.value_of_time_eur_per_min  # EUR / min

    # --- Expected wait time ---
    # Use the station's live queue if the resource is initialised,
    # otherwise assume zero wait (pre-trip planning approximation).
    expected_wait_min = station.expected_wait_time_min(avg_session_min)

    # --- Expected charge time ---
    charge_time_min = compute_charge_duration(
        soc_before_kwh=agent.current_soc_kwh,
        soc_target_kwh=target_soc_kwh,
        battery_capacity_kwh=agent.battery_capacity_kwh,
        charger_power_kw=station.max_power_kw,
        vehicle_max_acceptance_kw=agent.max_acceptance_kw,
    )

    # --- Time component ---
    time_cost_eur = vot * (
        detour_time_min
        + expected_wait_min * agent.queue_aversion
        + charge_time_min
    )

    # --- Monetary component ---
    charge_amount_kwh = max(0.0, target_soc_kwh - agent.current_soc_kwh)
    monetary_cost_eur = (
        agent.price_sensitivity * charge_amount_kwh * station.price_per_kwh
    )

    # --- Reliability risk penalty ---
    # If the station fails (P = 1 - reliability), the agent wastes ~30 min
    # finding an alternative.  Expected penalty = P(fail) × VoT × 30 min.
    failure_penalty_eur = (1.0 - station.reliability) * vot * 30.0

    return time_cost_eur + monetary_cost_eur + failure_penalty_eur


def rank_candidate_stations(
    agent: VehicleAgent,
    candidate_stations: List[Tuple[ChargingStation, float]],
    target_soc_kwh: float,
    avg_session_min: float = 22.0,
) -> List[Tuple[ChargingStation, float]]:
    """
    Rank a list of (station, detour_time_min) pairs by generalized cost.

    Returns
    -------
    List of (station, gc_eur) sorted ascending (cheapest first).
    """
    scored = []
    for station, detour_time in candidate_stations:
        gc = generalized_cost_for_station(
            agent, station, detour_time, target_soc_kwh, avg_session_min
        )
        scored.append((station, gc))
    scored.sort(key=lambda x: x[1])
    return scored


# ---------------------------------------------------------------------------
# Target SOC decision
# ---------------------------------------------------------------------------

def decide_charge_target(
    agent: VehicleAgent,
    station: ChargingStation,
    remaining_distance_km: float,
    stations_ahead: int,
    config: Dict,
) -> float:
    """
    Decide how much to charge at a given station.

    Logic:
    - Minimum: enough to reach the next station (or destination) with reserve.
    - Preferred: 80% SOC (avoids taper zone and keeps time reasonable).
    - Maximum: 100% if no more stations ahead and trip is long.
    - Risk-averse agents charge more; risk-tolerant agents stop earlier.

    Returns
    -------
    target_soc_kwh: float
    """
    min_reserve_frac = config.get("min_reserve_soc_fraction", 0.10)
    reserve_kwh = agent.usable_capacity_kwh * min_reserve_frac
    usable = agent.usable_capacity_kwh

    # Energy needed to reach destination (rough)
    energy_to_dest = remaining_distance_km * agent.consumption_kwh_per_km
    # Ensure enough to reach dest with reserve
    min_needed_kwh = min(energy_to_dest + reserve_kwh, usable)

    # Base target: 80% of usable
    base_target_frac = 0.80
    # Risk-averse: charge more. Risk-tolerant: accept 80%.
    risk_adj = (0.5 - agent.risk_tolerance) * 0.15  # risk=0 → +7.5%, risk=1 → -7.5%
    adjusted_frac = base_target_frac + risk_adj

    # If no more stations ahead, must be more conservative (charge higher)
    if stations_ahead == 0:
        adjusted_frac = max(adjusted_frac, min(1.0, energy_to_dest / usable + 0.15))

    target_kwh = max(
        min_needed_kwh,
        adjusted_frac * usable,
    )
    return min(target_kwh, usable)


# ---------------------------------------------------------------------------
# Station feasibility check
# ---------------------------------------------------------------------------

def is_station_reachable(
    agent: VehicleAgent,
    detour_energy_kwh: float,
    min_reserve_kwh: float,
) -> bool:
    """True if the agent has enough SOC to reach the station with reserve."""
    return (agent.current_soc_kwh - detour_energy_kwh) >= min_reserve_kwh
