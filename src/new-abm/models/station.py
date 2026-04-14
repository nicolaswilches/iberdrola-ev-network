"""Charging station data model.

Each ChargingStation is a physical location with a fixed number of
DC fast-charge connectors.  During the simulation the ``resource``
field holds the SimPy Resource that gates connector access.

Real station data (NAP, ChargeMap, operator APIs) can be loaded
into ChargingStation objects and passed to the simulation runner
instead of the synthetic ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChargingStation:
    """
    A physical fast-charging station at a network node.

    Fields that affect agent behaviour:
        max_power_kw       — determines charge speed (and therefore charge time)
        num_connectors     — capacity of the SimPy Resource (queue depth)
        price_per_kwh      — monetary cost per kWh, felt by price-sensitive agents
        reliability        — P(connector available when agent arrives)

    The SimPy Resource ``resource`` is None until ``simulation/engine.py``
    calls ``station.init_resource(env)`` at simulation start.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    station_id: str
    node_id: str
    name: str
    latitude: float
    longitude: float

    # ------------------------------------------------------------------
    # Technical specifications
    # ------------------------------------------------------------------
    max_power_kw: float          # per connector (e.g. 50, 150, 350 kW)
    num_connectors: int          # number of connectors = SimPy Resource capacity
    price_per_kwh: float         # EUR / kWh
    reliability: float = 0.95   # P(operational at any given moment)

    # ------------------------------------------------------------------
    # SimPy resource — injected by simulation engine
    # ------------------------------------------------------------------
    resource: Optional[Any] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Runtime statistics — accumulated during simulation
    # ------------------------------------------------------------------
    total_sessions: int = field(default=0, init=False, repr=False)
    total_energy_kwh: float = field(default=0.0, init=False, repr=False)
    total_wait_time_min: float = field(default=0.0, init=False, repr=False)
    total_charge_time_min: float = field(default=0.0, init=False, repr=False)
    peak_queue_length: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Live state queries (delegates to SimPy resource when active)
    # ------------------------------------------------------------------

    def current_users(self) -> int:
        """Number of connectors currently in use."""
        if self.resource is None:
            return 0
        return self.resource.count

    def current_queue_length(self) -> int:
        """Number of agents currently waiting for a connector."""
        if self.resource is None:
            return 0
        return len(self.resource.queue)

    def is_full(self) -> bool:
        """True if all connectors are occupied."""
        return self.current_users() >= self.num_connectors

    def effective_connectors(self) -> int:
        """Connectors available accounting for reliability."""
        import math
        return max(1, round(self.num_connectors * self.reliability))

    def expected_wait_time_min(self, avg_session_min: float = 22.0) -> float:
        """
        Rough expected wait time (M/D/c approximation) for a new arrival.

        Uses the simple formula: E[W] ≈ queue_length * avg_session / c
        where c = num_connectors.  Good enough for agent decision-making.
        """
        queue = self.current_queue_length()
        if queue == 0 and not self.is_full():
            return 0.0
        # Agents in queue plus partially-served fraction of busy connectors
        effective_load = queue + self.current_users()
        return (effective_load / max(1, self.num_connectors)) * avg_session_min

    def record_session(
        self,
        wait_time_min: float,
        charge_time_min: float,
        energy_kwh: float,
    ) -> None:
        """Accumulate statistics at the end of each charging session."""
        self.total_sessions += 1
        self.total_energy_kwh += energy_kwh
        self.total_wait_time_min += wait_time_min
        self.total_charge_time_min += charge_time_min
        q = self.current_queue_length()
        if q > self.peak_queue_length:
            self.peak_queue_length = q

    def utilization_rate(self, sim_duration_min: float) -> float:
        """
        Fraction of available connector-minutes actually used for charging.
        """
        available_min = self.num_connectors * sim_duration_min
        if available_min <= 0:
            return 0.0
        return min(1.0, self.total_charge_time_min / available_min)

    def init_resource(self, env: Any) -> None:
        """Create the SimPy Resource.  Must be called before simulation starts."""
        import simpy
        self.resource = simpy.Resource(env, capacity=self.effective_connectors())

    def __repr__(self) -> str:
        return (
            f"ChargingStation({self.station_id!r}, "
            f"node={self.node_id!r}, "
            f"{self.max_power_kw:.0f} kW × {self.num_connectors}, "
            f"€{self.price_per_kwh:.2f}/kWh)"
        )
