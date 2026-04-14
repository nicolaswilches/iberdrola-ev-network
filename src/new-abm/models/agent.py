"""Vehicle agent data model.

A VehicleAgent holds all mutable state for a single BEV trip.
The simulation engine reads and writes these fields as the agent
progresses through pre-trip, driving, queuing, charging, and
completion stages.

Behavioral parameters (value_of_time, price_sensitivity, etc.) are
heterogeneous across agents; they are set at creation time from
distributions defined in data_generation/synthetic.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VehicleAgent:
    """
    Represents one BEV trip from origin to destination.

    State lifecycle:
        pending → driving ↔ waiting → charging → driving → completed
                                                         └→ stranded
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    agent_id: str
    origin: str
    destination: str
    departure_time_min: float   # minutes from simulation epoch (midnight)

    # ------------------------------------------------------------------
    # Vehicle specifications
    # ------------------------------------------------------------------
    battery_capacity_kwh: float = 75.0
    usable_capacity_kwh: float = 67.5          # = battery_capacity_kwh * 0.90
    consumption_kwh_per_km: float = 0.20       # base highway consumption
    max_acceptance_kw: float = 150.0           # max DC fast-charge acceptance
    vehicle_type: str = "medium"               # small | medium | large | xl

    # ------------------------------------------------------------------
    # Behavioral / preference parameters
    # ------------------------------------------------------------------
    home_charging_access: bool = True
    destination_charging_access: bool = False
    value_of_time_eur_per_hour: float = 28.0
    price_sensitivity: float = 1.0             # multiplier on monetary cost
    queue_aversion: float = 1.0               # multiplier on wait-time penalty
    risk_tolerance: float = 0.4               # 0 = very risk-averse, 1 = risk-seeking
    minimum_reserve_soc_fraction: float = 0.10 # never deplete below this
    network_familiarity: float = 0.5           # 0 = tourist, 1 = commuter

    # ------------------------------------------------------------------
    # Mutable state — updated during simulation
    # ------------------------------------------------------------------
    current_soc_kwh: float = field(default=0.0)
    current_node: str = field(default="")
    status: str = field(default="pending")
    # pending | driving | waiting | charging | completed | stranded | failed

    # ------------------------------------------------------------------
    # Trip tracking — written by simulation engine
    # ------------------------------------------------------------------
    start_time_min: float = field(default=0.0)
    end_time_min: float = field(default=0.0)
    total_wait_time_min: float = field(default=0.0)
    total_charge_time_min: float = field(default=0.0)
    total_energy_charged_kwh: float = field(default=0.0)
    total_distance_km: float = field(default=0.0)

    route: List[str] = field(default_factory=list)
    charging_plan_station_ids: List[str] = field(default_factory=list)
    charge_events: List[Dict[str, Any]] = field(default_factory=list)

    failure_reason: str = field(default="")

    # ------------------------------------------------------------------
    # Derived convenience
    # ------------------------------------------------------------------

    @property
    def minimum_reserve_kwh(self) -> float:
        """Absolute SOC floor in kWh."""
        return self.usable_capacity_kwh * self.minimum_reserve_soc_fraction

    @property
    def emergency_threshold_kwh(self) -> float:
        """SOC below which emergency charging is triggered (15% of usable)."""
        return self.usable_capacity_kwh * 0.15

    @property
    def soc_fraction(self) -> float:
        """Current SOC as fraction of usable capacity."""
        return self.current_soc_kwh / self.usable_capacity_kwh

    @property
    def value_of_time_eur_per_min(self) -> float:
        return self.value_of_time_eur_per_hour / 60.0

    @property
    def trip_duration_min(self) -> float:
        if self.end_time_min > 0:
            return self.end_time_min - self.start_time_min
        return 0.0

    def record_charge_event(
        self,
        station_id: str,
        sim_time_min: float,
        wait_time_min: float,
        charge_time_min: float,
        energy_kwh: float,
        soc_before_kwh: float,
        soc_after_kwh: float,
    ) -> None:
        """Append a charging stop record."""
        self.charge_events.append(
            {
                "station_id": station_id,
                "sim_time_min": sim_time_min,
                "wait_time_min": wait_time_min,
                "charge_time_min": charge_time_min,
                "energy_kwh": energy_kwh,
                "soc_before_kwh": soc_before_kwh,
                "soc_after_kwh": soc_after_kwh,
            }
        )
        self.total_wait_time_min += wait_time_min
        self.total_charge_time_min += charge_time_min
        self.total_energy_charged_kwh += energy_kwh

    def __repr__(self) -> str:
        return (
            f"VehicleAgent({self.agent_id!r}, "
            f"{self.origin}→{self.destination}, "
            f"soc={self.current_soc_kwh:.1f}/{self.usable_capacity_kwh:.0f} kWh, "
            f"status={self.status!r})"
        )
