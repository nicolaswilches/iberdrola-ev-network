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
class EdgeTraversal:
    """One agent traversal over one ABM graph edge."""

    agent_id: str
    sim_time_min: float
    from_node: str
    to_node: str
    road_name: str
    distance_km: float
    source_segment_ids: str
    target_daily_bev_traffic_2027: float
    demand_weight: float = 1.0


@dataclass
class FailureDiagnostic:
    """Diagnostic context for failed or stranded trips."""

    agent_id: str
    origin: str
    destination: str
    od_pair_id: str
    demand_path_id: str
    current_node: str
    status: str
    failure_reason: str
    vehicle_type: str
    battery_capacity_kwh: float
    usable_capacity_kwh: float
    initial_soc_kwh: float
    current_soc_kwh: float
    initial_soc_fraction: float
    current_soc_fraction: float
    consumption_kwh_per_km: float
    home_charging_access: bool
    destination_charging_access: bool
    preferred_path_distance_km: float
    actual_route_distance_km: float
    first_reachable_station_node: str
    first_reachable_station_distance_km: float
    first_reachable_station_energy_kwh: float
    reachable_station_count: int
    route_infeasible_events: int
    path_adherence_ratio: float
    exact_preferred_path_match: bool


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
    od_pair_id: str = ""
    demand_path_id: str = ""
    vehicle_type: str = ""
    initial_soc_fraction: float = 0.0
    final_soc_fraction: float = 0.0
    preferred_path_distance_km: float = 0.0
    actual_route_distance_km: float = 0.0
    path_adherence_ratio: float = 0.0
    exact_preferred_path_match: bool = False
    route_infeasible_events: int = 0
    demand_weight: float = 1.0
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
        self._edge_traversals: List[EdgeTraversal] = []
        self._failure_diagnostics: List[FailureDiagnostic] = []
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

    def record_edge_traversal(
        self,
        agent_id: str,
        sim_time_min: float,
        from_node: str,
        to_node: str,
        road_name: str,
        distance_km: float,
        source_segment_ids: str,
        target_daily_bev_traffic_2027: float,
        demand_weight: float = 1.0,
    ) -> None:
        """Called whenever an agent traverses a graph edge."""
        self._edge_traversals.append(
            EdgeTraversal(
                agent_id=agent_id,
                sim_time_min=sim_time_min,
                from_node=from_node,
                to_node=to_node,
                road_name=road_name,
                distance_km=distance_km,
                source_segment_ids=source_segment_ids,
                target_daily_bev_traffic_2027=target_daily_bev_traffic_2027,
                demand_weight=demand_weight,
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
        od_pair_id: str = "",
        demand_path_id: str = "",
        vehicle_type: str = "",
        usable_capacity_kwh: float = 0.0,
        preferred_path_distance_km: float = 0.0,
        actual_route_distance_km: float = 0.0,
        path_adherence_ratio: float = 0.0,
        exact_preferred_path_match: bool = False,
        route_infeasible_events: int = 0,
        demand_weight: float = 1.0,
        generalized_cost_eur: float = 0.0,
        failure_reason: str = "",
    ) -> None:
        """Called when an agent completes or fails its trip."""
        initial_soc = self._initial_soc_by_agent.get(agent_id, 0.0)
        usable = max(float(usable_capacity_kwh or 0.0), 1e-9)
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
                od_pair_id=od_pair_id,
                demand_path_id=demand_path_id,
                vehicle_type=vehicle_type,
                initial_soc_fraction=initial_soc / usable,
                final_soc_fraction=final_soc_kwh / usable,
                preferred_path_distance_km=preferred_path_distance_km,
                actual_route_distance_km=actual_route_distance_km,
                path_adherence_ratio=path_adherence_ratio,
                exact_preferred_path_match=exact_preferred_path_match,
                route_infeasible_events=route_infeasible_events,
                demand_weight=demand_weight,
                generalized_cost_eur=generalized_cost_eur,
                failure_reason=failure_reason,
            )
        )

    def record_failure_diagnostic(
        self,
        **kwargs: Any,
    ) -> None:
        """Called when an agent fails or strands with diagnostic context."""
        self._failure_diagnostics.append(FailureDiagnostic(**kwargs))

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
            edge_traversals=list(self._edge_traversals),
            failure_diagnostics=list(self._failure_diagnostics),
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
    edge_traversals: List[EdgeTraversal]
    failure_diagnostics: List[FailureDiagnostic]
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

    def edge_traversals_df(self) -> pd.DataFrame:
        """All graph edge traversals as a DataFrame."""
        if not self.edge_traversals:
            return pd.DataFrame()
        return pd.DataFrame([vars(e) for e in self.edge_traversals])

    def failure_diagnostics_df(self) -> pd.DataFrame:
        """Failed/stranded trip diagnostic context as a DataFrame."""
        if not self.failure_diagnostics:
            return pd.DataFrame()
        return pd.DataFrame([vars(e) for e in self.failure_diagnostics])

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
                    "road_name": getattr(s, "road_name", ""),
                    "road_km": getattr(s, "road_km", 0.0),
                    "cluster_span_km": getattr(s, "cluster_span_km", 0.0),
                    "physical_station_count": getattr(s, "physical_station_count", 1),
                    "cluster_exception": getattr(s, "cluster_exception", ""),
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
