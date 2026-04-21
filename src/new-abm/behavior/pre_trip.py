"""Pre-trip charging decision.

Before departure, an agent with home charging access decides how much
to charge their battery.  The outcome (initial SOC) is endogenous:
it depends on the trip length, number of en-route charging opportunities,
expected prices, and the agent's risk tolerance.

Key insight
-----------
If the agent has home charging, they can arrive at departure with any
SOC they choose (up to full).  Rational agents balance:

  - the cost of "over-charging" (extra time at home charger, though
    marginal for slow overnight AC charging)
  - the cost of under-charging (higher probability of needing to stop
    at a DC fast-charger en route, which is slower and more expensive)

In the ABM, we implement this with a simple look-ahead rule:
    target_initial_soc = f(trip_distance, charging_opportunities, risk)

Agents WITHOUT home charging (public AC or no charging overnight) start
with a SOC drawn from an empirical distribution representing their prior
state.

Validation
----------
Calibrate this function so that:
  - average initial SOC in simulation ≈ survey data (e.g. 70-80%)
  - fraction departing with <20% SOC ≈ empirical (rare)
  - fraction departing fully charged ≈ empirical
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from models.agent import VehicleAgent
from models.network import RoadNetwork
from models.station import ChargingStation
from behavior.energy import compute_segment_energy

logger = logging.getLogger(__name__)


def decide_initial_soc(
    agent: VehicleAgent,
    network: RoadNetwork,
    stations_by_node: Dict[str, List[ChargingStation]],
    config: Dict,
    rng: np.random.Generator = None,
) -> float:
    """
    Decide the agent's starting SOC (kWh) before departure.

    This is the *endogenous* pre-trip charging decision.

    Parameters
    ----------
    agent:
        The VehicleAgent.  ``agent.current_soc_kwh`` is not yet set;
        this function computes the value that will be assigned to it.
    network:
        Road network for computing trip distance and en-route options.
    stations_by_node:
        Dict of node_id → list of stations, for counting opportunities.
    config:
        Dict with behavioral parameters from base_config.yaml.
    rng:
        Random generator (for stochastic SOC draws).

    Returns
    -------
    initial_soc_kwh: float
        The SOC the agent starts the trip with, in kWh.
    """
    if rng is None:
        rng = np.random.default_rng()

    route = network.shortest_path_gc(agent.origin, agent.destination,
                                     agent.value_of_time_eur_per_hour,
                                     agent.max_comfortable_speed_kmh)
    trip_distance_km = network.subpath_distance_km(route) if route else 200.0

    # Count charging stations reachable on the route
    stations_on_route = sum(
        1 for node in route if node in stations_by_node
    )

    if agent.home_charging_access:
        soc = _home_charger_target_soc(
            agent, trip_distance_km, stations_on_route, config, rng
        )
    else:
        soc = _no_home_charger_soc(agent, rng)

    # Floor: regardless of home-charging access, the agent must have enough SOC
    # to at least reach the first charging station (or the destination if no
    # stations exist).  This represents topping up at a public AC charger near
    # home / origin before departing — a trivially observable real-world behaviour.
    soc = max(soc, _minimum_viable_soc(agent, route, stations_by_node, network, config))
    return min(soc, agent.usable_capacity_kwh)


def _home_charger_target_soc(
    agent: VehicleAgent,
    trip_distance_km: float,
    stations_on_route: int,
    config: Dict,
    rng: np.random.Generator,
) -> float:
    """
    Compute target departure SOC for an agent with home charging.

    Logic:
    1. Compute the minimum SOC needed to reach the first station (or dest).
    2. Add a buffer that depends on risk_tolerance.
    3. If many stations are available, risk-neutral agents depart at ~70%.
    4. Risk-averse agents charge more; risk-seeking agents charge less.
    5. Add small Gaussian noise to represent imperfect execution.
    """
    min_reserve_frac = config.get("min_reserve_soc_fraction", 0.10)
    usable = agent.usable_capacity_kwh

    # Estimate energy needed for the whole trip at our average consumption
    trip_energy_kwh = trip_distance_km * agent.consumption_kwh_per_km
    reserve_kwh = usable * min_reserve_frac

    # Base target: enough to complete trip with reserve, capped at usable
    min_needed = min(trip_energy_kwh + reserve_kwh, usable)

    if stations_on_route == 0:
        # No stations en route — must depart full (or as full as needed)
        target_frac = min(1.0, min_needed / usable + 0.05)
    else:
        # Stations available — can depart with less
        # Risk-averse: depart at ~85%.  Risk-seeking: depart at ~55%.
        # Base = 0.70, shifted by risk_tolerance
        base_frac = 0.70
        risk_adjustment = (agent.risk_tolerance - 0.5) * (-0.30)
        # risk=0 → +0.15 (charge more), risk=1 → -0.15 (charge less)
        target_frac = base_frac + risk_adjustment

        # Also ensure we can at least reach the first station
        first_hop_frac = min(min_needed / usable, 1.0)
        target_frac = max(target_frac, first_hop_frac)

    target_frac = float(np.clip(target_frac, 0.20, 1.0))

    # Stochastic noise: ±5% (agents don't charge to exact fractions)
    noise = rng.normal(0.0, 0.03)
    target_frac = float(np.clip(target_frac + noise, 0.15, 1.0))

    return target_frac * usable


def _minimum_viable_soc(
    agent: VehicleAgent,
    route: List[str],
    stations_by_node: Dict[str, List[ChargingStation]],
    network: RoadNetwork,
    config: Dict,
) -> float:
    """
    Compute the absolute minimum SOC the agent must have at departure.

    This is the energy needed to reach the first charging station on the
    route (or the destination if no stations exist), plus the reserve buffer.
    An agent who has less would be unable to move at all; in reality they
    would charge to this level before departing.
    """
    from behavior.energy import compute_segment_energy

    min_reserve_frac = config.get("min_reserve_soc_fraction", 0.10)
    reserve_kwh = agent.usable_capacity_kwh * min_reserve_frac

    cumulative_energy = 0.0
    for i in range(len(route) - 1):
        seg_energy = compute_segment_energy(
            [route[i], route[i + 1]], network, agent.consumption_kwh_per_km
        )
        cumulative_energy += seg_energy

        next_node = route[i + 1]
        if next_node in stations_by_node or next_node == route[-1]:
            # First station (or destination) reachable from origin
            return cumulative_energy + reserve_kwh

    # Fallback: full usable capacity (no stations at all)
    return min(agent.usable_capacity_kwh, cumulative_energy + reserve_kwh)


def _no_home_charger_soc(
    agent: VehicleAgent,
    rng: np.random.Generator,
) -> float:
    """
    Sample departure SOC for agents without home charging.

    These agents charge at public AC stations (workplace, destination
    of previous trip, etc.).  Their departure SOC is drawn from a
    beta distribution calibrated to survey data:
    - Most agents depart at 40–70% SOC (AC-charged overnight somewhere)
    - Some depart at lower SOC (forgot to plug in or charger busy)
    - Very few depart at ≥90% (lucky fast charger the previous evening)
    """
    usable = agent.usable_capacity_kwh

    # Beta(3, 2) gives mean ~0.60 with realistic spread
    # Scaled to range [0.20, 0.90]
    raw = rng.beta(3.0, 2.0)
    frac = 0.20 + raw * 0.70  # map [0,1] → [0.20, 0.90]
    frac = float(np.clip(frac, 0.15, 0.92))
    return frac * usable
