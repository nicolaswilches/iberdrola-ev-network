"""Scenario configuration and application.

A scenario is a set of modifications applied on top of the base network
and configuration before running the simulation.

Supported modification types:
  - Station price multiplier (per station or global)
  - Station connector delta (add/remove connectors)
  - Fleet home charging penetration override
  - Fleet battery distribution shift
  - Demand volume multiplier
  - Station reliability override

Design: modifications are applied to COPIES of the original objects
so the baseline can always be re-run without re-building the network.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.station import ChargingStation

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """
    Declarative specification of a scenario.

    All fields are optional; unset fields leave the baseline unchanged.
    This makes it easy to define minimal, focused experiments.

    Example — lower price on one corridor::

        sc = ScenarioConfig(
            name="price_reduction",
            description="40% price cut on Madrid-Barcelona corridor",
            station_price_multipliers={
                "STA_MAD_N": 0.60,
                "STA_ZAR_E": 0.60,
                "STA_BCN_S": 0.60,
            },
        )
    """

    name: str = "unnamed"
    description: str = ""

    # --- Station modifications ---
    station_price_multipliers: Dict[str, float] = field(default_factory=dict)
    # station_id → multiplier (e.g. 0.60 = 40% price cut)

    station_connector_deltas: Dict[str, int] = field(default_factory=dict)
    # station_id → delta (e.g. +4 connectors)

    station_reliability_overrides: Dict[str, float] = field(default_factory=dict)
    # station_id → new reliability value

    global_price_multiplier: float = 1.0
    # Applied to ALL stations if != 1.0

    # --- Fleet modifications ---
    home_charging_penetration: Optional[float] = None
    # Overrides config value if set

    battery_capacity_multiplier: float = 1.0
    # Scales all battery capacities (e.g. 1.2 = 20% bigger batteries)

    # --- Demand modifications ---
    demand_volume_multiplier: float = 1.0
    # Scales total number of agents (e.g. 1.5 = 50% more trips)

    # --- Extra config overrides (free-form) ---
    config_overrides: Dict[str, Any] = field(default_factory=dict)


def apply_scenario(
    scenario: ScenarioConfig,
    base_stations: List[ChargingStation],
    base_config: Dict,
) -> tuple[List[ChargingStation], Dict]:
    """
    Apply a scenario to produce modified stations and config.

    Returns deep copies; the originals are never mutated.

    Parameters
    ----------
    scenario:       The ScenarioConfig to apply.
    base_stations:  Original station list (will be copied).
    base_config:    Original config dict (will be copied).

    Returns
    -------
    (modified_stations, modified_config)
    """
    # Use dataclasses.replace so we never deepcopy a live SimPy Resource
    # (SimPy generators cannot be pickled/deepcopied).
    stations = [_copy_station(s) for s in base_stations]
    config = copy.deepcopy(base_config)

    # --- Apply station modifications ---
    for station in stations:
        sid = station.station_id

        # Price
        multiplier = scenario.station_price_multipliers.get(sid, 1.0)
        multiplier *= scenario.global_price_multiplier
        station.price_per_kwh *= multiplier

        # Connectors
        delta = scenario.station_connector_deltas.get(sid, 0)
        station.num_connectors = max(1, station.num_connectors + delta)

        # Reliability
        if sid in scenario.station_reliability_overrides:
            station.reliability = scenario.station_reliability_overrides[sid]

    # --- Apply fleet modifications ---
    if scenario.home_charging_penetration is not None:
        # Update nested and flat copies
        if "fleet" in config:
            config["fleet"]["home_charging_penetration"] = (
                scenario.home_charging_penetration
            )
        config["fleet_home_charging_penetration"] = (
            scenario.home_charging_penetration
        )
        logger.debug(
            "Scenario '%s': home_charging_penetration → %.2f",
            scenario.name, scenario.home_charging_penetration,
        )

    if scenario.battery_capacity_multiplier != 1.0:
        batt = config.get("fleet", {}).get("battery_distribution", {})
        for vtype in batt:
            batt[vtype]["capacity_kwh"] = round(
                batt[vtype]["capacity_kwh"] * scenario.battery_capacity_multiplier,
                1,
            )
        config["fleet_battery_distribution"] = batt

    # --- Apply config overrides ---
    for key, value in scenario.config_overrides.items():
        config[key] = value

    logger.info(
        "Scenario '%s' applied: %d price mods, %d connector mods",
        scenario.name,
        len(scenario.station_price_multipliers),
        len(scenario.station_connector_deltas),
    )
    return stations, config


def _copy_station(s: ChargingStation) -> ChargingStation:
    """
    Return a fresh copy of a ChargingStation with no SimPy resource attached.

    ``copy.deepcopy`` fails when the station has an active SimPy resource
    (which contains generator objects that cannot be pickled).  This helper
    creates a new instance via ``dataclasses.replace``, which does a shallow
    field copy, then explicitly resets the resource and all runtime stats.
    """
    new_s = dataclasses.replace(s, resource=None)
    new_s.total_sessions = 0
    new_s.total_energy_kwh = 0.0
    new_s.total_wait_time_min = 0.0
    new_s.total_charge_time_min = 0.0
    new_s.peak_queue_length = 0
    return new_s


def load_scenario_from_yaml(path: str) -> ScenarioConfig:
    """Load a ScenarioConfig from a YAML file."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    name = data.get("name", "unnamed")
    description = data.get("description", "")
    mods = data.get("modifications", {})

    sc = ScenarioConfig(name=name, description=description)

    # Price multiplier block
    if "station_price_multiplier" in mods:
        block = mods["station_price_multiplier"]
        mult = block.get("multiplier", 1.0)
        for sid in block.get("corridor_stations", []):
            sc.station_price_multipliers[sid] = mult

    # Connector delta block
    if "station_connector_delta" in mods:
        for sid, delta in mods["station_connector_delta"].items():
            sc.station_connector_deltas[sid] = int(delta)

    # Fleet penetration
    if "fleet_home_charging_penetration" in mods:
        sc.home_charging_penetration = float(
            mods["fleet_home_charging_penetration"]
        )

    # Global price
    if "global_price_multiplier" in mods:
        sc.global_price_multiplier = float(mods["global_price_multiplier"])

    return sc
