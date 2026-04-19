"""Simulation results data structures.

ResultsCollector is a lightweight event sink that the simulation engine
writes to as agents complete trips and charging events.

SimulationResults holds the final aggregated state that gets passed to
outputs/aggregator.py for DataFrame conversion and plotting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ChargeEvent:
    """One charging session at one station by one agent."""

    agent_id: str
    station_id: str
    node_id: str
    sim_time_min: float       # simulation clock at charge end
    wait_time_min: float
    charge_time_min: float
    energy_kwh: float
    soc_before_kwh: float
    soc_after_kwh: float
    price_per_kwh: float


@dataclass
class TripRecord:
    """Summary of a completed (or failed) trip."""

    agent_id: str
    origin: str
    destination: str
    departure_time_min: float
    arrival_time_min: float   # 0 if not completed
    status: str               # completed | stranded | failed_no_route
    num_charge_stops: int
    total_wait_min: float
    total_charge_min: float
    total_energy_charged_kwh: float
    total_distance_km: float
    route_node_count: int
    initial_soc_kwh: float
    final_soc_kwh: float
    generalized_cost_eur: float = 0.0
    # Empty for completed trips. For failed trips, one of:
    # no_path_found | no_reachable_station | soc_depleted | sim_window_timeout
    failure_reason: str = ""


class ResultsCollector:
    """
    Receives events from the simulation engine during the run.

    Design:
    -------
    The simulation engine calls ``record_*`` methods as events happen.
    After the run, ``to_results()`` converts accumulated data to a
    SimulationResults object.  Keeping collection separate from storage
    allows the engine to stay focused on simulation logic.
    """

    def __init__(self) -> None:
        self._charge_events: List[ChargeEvent] = []
        self._trip_records: List[TripRecord] = []
        self._initial_soc_by_agent: Dict[str, float] = {}

    def record_departure(self, agent_id: str, initial_soc_kwh: float) -> None:
        """Called once per agent when it departs."""
        self._initial_soc_by_agent[agent_id] = initial_soc_kwh

    def record_charge_event(
        self,
        agent_id: str,
        station_id: str,
        node_id: str,
        sim_time_min: float,
        wait_time_min: float,
        charge_time_min: float,
        energy_kwh: float,
        soc_before_kwh: float,
        soc_after_kwh: float,
        price_per_kwh: float,
    ) -> None:
        """Called at the end of each charging session."""
        self._charge_events.append(
            ChargeEvent(
                agent_id=agent_id,
                station_id=station_id,
                node_id=node_id,
                sim_time_min=sim_time_min,
                wait_time_min=wait_time_min,
                charge_time_min=charge_time_min,
                energy_kwh=energy_kwh,
                soc_before_kwh=soc_before_kwh,
                soc_after_kwh=soc_after_kwh,
                price_per_kwh=price_per_kwh,
            )
        )

    def record_completion(
        self,
        agent_id: str,
        origin: str,
        destination: str,
        departure_time_min: float,
        arrival_time_min: float,
        status: str,
        num_charge_stops: int,
        total_wait_min: float,
        total_charge_min: float,
        total_energy_kwh: float,
        total_distance_km: float,
        route_node_count: int,
        final_soc_kwh: float,
        generalized_cost_eur: float = 0.0,
        failure_reason: str = "",
    ) -> None:
        """Called when an agent completes or fails its trip."""
        initial_soc = self._initial_soc_by_agent.get(agent_id, 0.0)
        self._trip_records.append(
            TripRecord(
                agent_id=agent_id,
                origin=origin,
                destination=destination,
                departure_time_min=departure_time_min,
                arrival_time_min=arrival_time_min,
                status=status,
                num_charge_stops=num_charge_stops,
                total_wait_min=total_wait_min,
                total_charge_min=total_charge_min,
                total_energy_charged_kwh=total_energy_kwh,
                total_distance_km=total_distance_km,
                route_node_count=route_node_count,
                initial_soc_kwh=initial_soc,
                final_soc_kwh=final_soc_kwh,
                generalized_cost_eur=generalized_cost_eur,
                failure_reason=failure_reason,
            )
        )

    def to_results(
        self,
        stations: List[Any],
        scenario_name: str = "unnamed",
        sim_duration_min: float = 1440.0,
    ) -> "SimulationResults":
        """Convert collected events into a SimulationResults object."""
        return SimulationResults(
            scenario_name=scenario_name,
            charge_events=list(self._charge_events),
            trip_records=list(self._trip_records),
            stations=stations,
            sim_duration_min=sim_duration_min,
        )


@dataclass
class SimulationResults:
    """
    Complete output of one simulation run.

    Holds raw event logs plus convenience DataFrame conversion methods
    used by outputs/aggregator.py.
    """

    scenario_name: str
    charge_events: List[ChargeEvent]
    trip_records: List[TripRecord]
    stations: List[Any]      # List[ChargingStation]
    sim_duration_min: float

    # ------------------------------------------------------------------
    # DataFrame exports
    # ------------------------------------------------------------------

    def charge_events_df(self) -> pd.DataFrame:
        """All charging events as a DataFrame."""
        if not self.charge_events:
            return pd.DataFrame()
        return pd.DataFrame(
            [vars(e) for e in self.charge_events]
        )

    def trip_records_df(self) -> pd.DataFrame:
        """All trip records as a DataFrame."""
        if not self.trip_records:
            return pd.DataFrame()
        return pd.DataFrame(
            [vars(r) for r in self.trip_records]
        )

    def station_summary_df(self) -> pd.DataFrame:
        """Per-station operational summary."""
        rows = []
        for s in self.stations:
            rows.append(
                {
                    "station_id": s.station_id,
                    "node_id": s.node_id,
                    "name": s.name,
                    "max_power_kw": s.max_power_kw,
                    "num_connectors": s.num_connectors,
                    "price_per_kwh": s.price_per_kwh,
                    "total_sessions": s.total_sessions,
                    "total_energy_kwh": s.total_energy_kwh,
                    "total_wait_min": s.total_wait_time_min,
                    "total_charge_min": s.total_charge_time_min,
                    "peak_queue": s.peak_queue_length,
                    "utilization_rate": s.utilization_rate(self.sim_duration_min),
                }
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Quick summary metrics
    # ------------------------------------------------------------------

    def completion_rate(self) -> float:
        if not self.trip_records:
            return 0.0
        completed = sum(1 for r in self.trip_records if r.status == "completed")
        return completed / len(self.trip_records)

    def avg_wait_time_min(self) -> float:
        completed = [r for r in self.trip_records if r.status == "completed"]
        if not completed:
            return 0.0
        return sum(r.total_wait_min for r in completed) / len(completed)

    def avg_charge_time_min(self) -> float:
        completed = [r for r in self.trip_records if r.status == "completed"]
        if not completed:
            return 0.0
        return sum(r.total_charge_min for r in completed) / len(completed)

    def total_energy_dispensed_kwh(self) -> float:
        return sum(e.energy_kwh for e in self.charge_events)

    def summary_dict(self) -> Dict[str, Any]:
        """Key metrics as a flat dict, useful for scenario comparison tables."""
        return {
            "scenario": self.scenario_name,
            "num_trips": len(self.trip_records),
            "completion_rate": round(self.completion_rate(), 3),
            "num_stranded": sum(
                1 for r in self.trip_records if r.status == "stranded"
            ),
            "avg_wait_min": round(self.avg_wait_time_min(), 2),
            "avg_charge_min": round(self.avg_charge_time_min(), 2),
            "total_charge_events": len(self.charge_events),
            "total_energy_kwh": round(self.total_energy_dispensed_kwh(), 1),
        }

    def __repr__(self) -> str:
        return (
            f"SimulationResults(scenario={self.scenario_name!r}, "
            f"trips={len(self.trip_records)}, "
            f"charge_events={len(self.charge_events)})"
        )
