"""Results aggregation helpers.

Functions here convert SimulationResults into analysis-ready DataFrames
suitable for reporting and plotting.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models.results import SimulationResults


def aggregate_results(
    results: SimulationResults,
    demand_df: Optional[pd.DataFrame] = None,
    calibrated_path_flows_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Produce all standard aggregation tables for one simulation run.

    Returns a dict with keys:
        "station_summary"   — per-station operational metrics
        "trip_summary"      — aggregate trip metrics
        "charge_events"     — raw charging event log
        "edge_traversals"   — raw graph edge traversal log
        "trip_records"      — raw trip record log
        "failure_diagnostics" — failed/stranded trip diagnostic context
        "geo_boundary_failures" — GEO boundary failures grouped by node
        "path_adherence"    — calibrated preferred path vs simulated path fit
        "hourly_demand"     — charging demand by hour of day
        "od_completion"     — completion rate by OD pair
        "road_flow_validation" — simulated traversals vs edge traffic targets
        "segment_flow_validation" — simulated traversals vs segment demand targets
        "segment_flow_decomposition" — target vs calibrated/sample/actual flow
    """
    return {
        "station_summary": results.station_summary_df(),
        "trip_summary": trip_summary(results),
        "charge_events": results.charge_events_df(),
        "edge_traversals": results.edge_traversals_df(),
        "trip_records": results.trip_records_df(),
        "failure_diagnostics": results.failure_diagnostics_df(),
        "geo_boundary_failures": geo_boundary_failure_table(results),
        "path_adherence": path_adherence_table(results),
        "hourly_demand": hourly_demand_table(results),
        "od_completion": od_completion_table(results),
        "road_flow_validation": road_flow_validation_table(results),
        "segment_flow_validation": segment_flow_validation_table(results, demand_df),
        "segment_flow_decomposition": segment_flow_decomposition_table(
            results, demand_df, calibrated_path_flows_df
        ),
    }


def station_demand_table(results: SimulationResults) -> pd.DataFrame:
    """
    Per-station demand breakdown: sessions, energy, wait time, utilization.

    Sorted by total sessions descending (busiest first).
    """
    df = results.station_summary_df()
    if df.empty:
        return df
    df = df.sort_values("total_sessions", ascending=False)
    df["avg_wait_min"] = np.where(
        df["total_sessions"] > 0,
        df["total_wait_min"] / df["total_sessions"],
        0.0,
    )
    df["avg_session_energy_kwh"] = np.where(
        df["total_sessions"] > 0,
        df["total_energy_kwh"] / df["total_sessions"],
        0.0,
    )
    return df.round(2)


def trip_summary(results: SimulationResults) -> pd.DataFrame:
    """Aggregate trip-level metrics as a single-row DataFrame."""
    recs = results.trip_records_df()
    if recs.empty:
        return pd.DataFrame()

    completed = recs[recs["status"] == "completed"]
    summary = {
        "scenario": results.scenario_name,
        "total_trips": len(recs),
        "completed": len(completed),
        "stranded": (recs["status"] == "stranded").sum(),
        "failed_no_route": (recs["status"] == "failed_no_route").sum(),
        "completion_rate_pct": round(results.completion_rate() * 100, 1),
        "avg_trip_duration_min": round(completed["arrival_time_min"].sub(
            completed["departure_time_min"]).mean(), 1) if len(completed) else 0,
        "avg_wait_min": round(completed["total_wait_min"].mean(), 2) if len(completed) else 0,
        "avg_charge_min": round(completed["total_charge_min"].mean(), 2) if len(completed) else 0,
        "avg_charge_stops": round(completed["num_charge_stops"].mean(), 2) if len(completed) else 0,
        "pct_zero_stops": round(
            (completed["num_charge_stops"] == 0).mean() * 100, 1
        ) if len(completed) else 0,
        "total_energy_kwh": round(results.total_energy_dispensed_kwh(), 1),
        "total_charge_events": len(results.charge_events),
    }
    return pd.DataFrame([summary])


def hourly_demand_table(results: SimulationResults) -> pd.DataFrame:
    """Charging event counts and energy dispensed by hour of day."""
    ev_df = results.charge_events_df()
    if ev_df.empty:
        return pd.DataFrame(columns=["hour", "num_sessions", "energy_kwh"])

    ev_df = ev_df.copy()
    ev_df["hour"] = (ev_df["sim_time_min"] // 60).astype(int) % 24
    hourly = (
        ev_df.groupby("hour")
        .agg(num_sessions=("energy_kwh", "count"), energy_kwh=("energy_kwh", "sum"))
        .reset_index()
    )
    # Fill missing hours
    all_hours = pd.DataFrame({"hour": range(24)})
    hourly = all_hours.merge(hourly, on="hour", how="left").fillna(0)
    hourly["energy_kwh"] = hourly["energy_kwh"].round(1)
    return hourly


def od_completion_table(results: SimulationResults) -> pd.DataFrame:
    """Completion rate and average wait time by OD pair."""
    recs = results.trip_records_df()
    if recs.empty:
        return pd.DataFrame()

    recs["od_pair"] = recs["origin"] + "→" + recs["destination"]
    grouped = recs.groupby("od_pair").agg(
        total=("status", "count"),
        completed=("status", lambda x: (x == "completed").sum()),
        avg_wait_min=("total_wait_min", "mean"),
        avg_charge_stops=("num_charge_stops", "mean"),
    ).reset_index()
    grouped["completion_rate_pct"] = round(
        grouped["completed"] / grouped["total"] * 100, 1
    )
    return grouped.sort_values("total", ascending=False)


def path_adherence_table(results: SimulationResults) -> pd.DataFrame:
    """Summarize whether calibrated trips followed their preferred paths."""
    recs = results.trip_records_df()
    if recs.empty or "demand_path_id" not in recs.columns:
        return pd.DataFrame()

    calibrated = recs[recs["demand_path_id"].fillna("").astype(str) != ""].copy()
    if calibrated.empty:
        return pd.DataFrame()

    od_ids = calibrated.get("od_pair_id", calibrated["demand_path_id"]).fillna("").astype(str)
    calibrated["is_localod"] = od_ids.str.startswith("LOCALOD_")
    calibrated["is_roadspan"] = od_ids.str.startswith("ROADSPAN_")
    calibrated["is_geo_origin"] = calibrated["origin"].astype(str).str.startswith("GEO_")
    calibrated["is_geo_destination"] = calibrated["destination"].astype(str).str.startswith("GEO_")

    group_cols = ["is_localod", "is_roadspan", "is_geo_origin", "is_geo_destination"]
    grouped = calibrated.groupby(group_cols, dropna=False).agg(
        trips=("agent_id", "count"),
        completed=("status", lambda s: (s == "completed").sum()),
        stranded=("status", lambda s: (s == "stranded").sum()),
        avg_path_adherence=("path_adherence_ratio", "mean"),
        exact_path_match_rate=("exact_preferred_path_match", "mean"),
        avg_route_infeasible_events=("route_infeasible_events", "mean"),
        avg_preferred_path_km=("preferred_path_distance_km", "mean"),
        avg_actual_route_km=("actual_route_distance_km", "mean"),
    ).reset_index()
    grouped["completion_rate_pct"] = (
        grouped["completed"] / grouped["trips"].replace(0, pd.NA) * 100.0
    )
    grouped["exact_path_match_rate_pct"] = grouped["exact_path_match_rate"] * 100.0
    grouped["avg_path_adherence_pct"] = grouped["avg_path_adherence"] * 100.0
    return grouped.sort_values("trips", ascending=False)


def geo_boundary_failure_table(results: SimulationResults) -> pd.DataFrame:
    """Summarize failed/stranded trips involving GEO boundary nodes."""
    fd = results.failure_diagnostics_df()
    if fd.empty:
        return pd.DataFrame()

    for col in ("current_node", "origin", "destination"):
        if col not in fd.columns:
            fd[col] = ""

    df = fd.copy()
    node_cols = ["current_node", "origin", "destination"]
    is_geo = pd.Series(False, index=df.index)
    for col in node_cols:
        is_geo = is_geo | df[col].fillna("").astype(str).str.startswith("GEO_")
    df = df[is_geo]
    if df.empty:
        return pd.DataFrame()

    df["boundary_node"] = np.where(
        df["current_node"].fillna("").astype(str).str.startswith("GEO_"),
        df["current_node"],
        np.where(
            df["origin"].fillna("").astype(str).str.startswith("GEO_"),
            df["origin"],
            df["destination"],
        ),
    )
    df["has_reachable_station"] = (
        pd.to_numeric(df.get("reachable_station_count", 0), errors="coerce")
        .fillna(0)
        .gt(0)
    )
    grouped = df.groupby("boundary_node", dropna=False).agg(
        failures=("agent_id", "count"),
        no_reachable_station=("failure_reason", lambda s: (s == "no_reachable_station").sum()),
        sim_window_timeout=("failure_reason", lambda s: (s == "sim_window_timeout").sum()),
        failed_no_route=("status", lambda s: (s == "failed_no_route").sum()),
        reachable_station_count_min=("reachable_station_count", "min"),
        reachable_station_count_max=("reachable_station_count", "max"),
        avg_current_soc_fraction=("current_soc_fraction", "mean"),
        avg_initial_soc_fraction=("initial_soc_fraction", "mean"),
        avg_route_infeasible_events=("route_infeasible_events", "mean"),
        avg_path_adherence=("path_adherence_ratio", "mean"),
        sample_origin=("origin", "first"),
        sample_destination=("destination", "first"),
        sample_od_pair_id=("od_pair_id", "first"),
        sample_demand_path_id=("demand_path_id", "first"),
    ).reset_index()
    grouped["has_any_reachable_station"] = (
        grouped["reachable_station_count_max"].fillna(0).astype(float) > 0
    )
    grouped["avg_current_soc_pct"] = grouped["avg_current_soc_fraction"] * 100.0
    grouped["avg_initial_soc_pct"] = grouped["avg_initial_soc_fraction"] * 100.0
    grouped["avg_path_adherence_pct"] = grouped["avg_path_adherence"] * 100.0
    return grouped.sort_values(
        ["failures", "no_reachable_station"], ascending=False
    )


def road_flow_validation_table(results: SimulationResults) -> pd.DataFrame:
    """Aggregate simulated edge traversals by road for traffic calibration."""
    edge_df = results.edge_traversals_df()
    if edge_df.empty:
        return pd.DataFrame()

    grouped = edge_df.groupby("road_name").agg(
        simulated_traversals=("agent_id", "count"),
        simulated_agent_count=("agent_id", "nunique"),
        weighted_traversals=("demand_weight", "sum"),
        traversed_km=("distance_km", "sum"),
        target_daily_bev_traffic_2027=("target_daily_bev_traffic_2027", "mean"),
    ).reset_index()
    total_sim = float(grouped["weighted_traversals"].sum())
    total_target = float(grouped["target_daily_bev_traffic_2027"].sum())
    if total_sim > 0 and total_target > 0:
        raw_total = float(grouped["simulated_traversals"].sum())
        uses_unit_weights = abs(total_sim - raw_total) < 1e-6
        scale = total_target / raw_total if uses_unit_weights and raw_total > 0 else 1.0
    else:
        scale = 0.0
    grouped["simulated_daily_equivalent"] = grouped["weighted_traversals"] * scale
    grouped["abs_error_vs_target"] = (
        grouped["simulated_daily_equivalent"] - grouped["target_daily_bev_traffic_2027"]
    ).abs()
    grouped["error_pct_vs_target"] = (
        grouped["abs_error_vs_target"]
        / grouped["target_daily_bev_traffic_2027"].replace(0, pd.NA)
        * 100.0
    )
    return grouped.sort_values("target_daily_bev_traffic_2027", ascending=False)


def segment_flow_validation_table(
    results: SimulationResults,
    demand_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate simulated edge traversals by source segment_id."""
    edge_df = results.edge_traversals_df()
    if edge_df.empty:
        return pd.DataFrame()

    rows: List[Dict] = []
    for row in edge_df.itertuples(index=False):
        raw_ids = str(getattr(row, "source_segment_ids", "") or "")
        segment_ids = [sid for sid in raw_ids.split("|") if sid.strip()]
        for sid in segment_ids:
            try:
                segment_id = int(sid)
            except ValueError:
                continue
            rows.append({
                "segment_id": segment_id,
                "agent_id": row.agent_id,
                "road_name": row.road_name,
                "distance_km": row.distance_km,
                "demand_weight": float(getattr(row, "demand_weight", 1.0) or 1.0),
            })

    if not rows:
        return pd.DataFrame()

    expanded = pd.DataFrame(rows)
    agent_segments = expanded.groupby(["segment_id", "agent_id"]).agg(
        demand_weight=("demand_weight", "first"),
        road_name=("road_name", lambda s: "|".join(sorted(set(str(x) for x in s if str(x))))),
        distance_km=("distance_km", "sum"),
    ).reset_index()

    grouped = agent_segments.groupby("segment_id").agg(
        simulated_traversals=("agent_id", "count"),
        simulated_agent_count=("agent_id", "nunique"),
        weighted_traversals=("demand_weight", "sum"),
        road_name=("road_name", lambda s: "|".join(sorted(set(str(x) for x in s if str(x))))),
        traversed_edge_km=("distance_km", "sum"),
    ).reset_index()

    if demand_df is not None and not demand_df.empty:
        targets = _segment_targets_df(demand_df)
        grouped = targets.merge(grouped, on="segment_id", how="left")
        grouped["simulated_traversals"] = grouped["simulated_traversals"].fillna(0).astype(int)
        grouped["simulated_agent_count"] = grouped["simulated_agent_count"].fillna(0).astype(int)
        grouped["weighted_traversals"] = grouped["weighted_traversals"].fillna(0.0)
        grouped["traversed_edge_km"] = grouped["traversed_edge_km"].fillna(0.0)
        grouped["road_name"] = grouped["road_name"].fillna(grouped.get("route_segment", ""))
        total_sim = float(grouped["weighted_traversals"].sum())
        total_target = float(grouped["target_daily_bev_traffic_2027"].sum())
    else:
        grouped["target_daily_bev_traffic_2027"] = pd.NA
        total_sim = float(grouped["weighted_traversals"].sum())
        total_target = total_sim

    if total_sim > 0 and total_target > 0:
        weight_total = float(grouped["weighted_traversals"].sum())
        raw_total = float(grouped["simulated_traversals"].sum())
        uses_unit_weights = abs(weight_total - raw_total) < 1e-6
        scale = total_target / raw_total if uses_unit_weights and raw_total > 0 else 1.0
    else:
        scale = 0.0
    grouped["simulated_daily_equivalent"] = grouped["weighted_traversals"] * scale
    grouped["abs_error_vs_target"] = (
        grouped["simulated_daily_equivalent"] - grouped["target_daily_bev_traffic_2027"]
    ).abs()
    grouped["error_pct_vs_target"] = (
        grouped["abs_error_vs_target"]
        / grouped["target_daily_bev_traffic_2027"].replace(0, pd.NA)
        * 100.0
    )
    grouped["within_10pct_or_25bev"] = (
        (grouped["target_daily_bev_traffic_2027"] <= 25.0)
        | (grouped["error_pct_vs_target"] <= 10.0)
    )
    return grouped.sort_values("target_daily_bev_traffic_2027", ascending=False)


def segment_flow_decomposition_table(
    results: SimulationResults,
    demand_df: Optional[pd.DataFrame] = None,
    calibrated_path_flows_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Decompose segment error into calibration, sampling, and ABM behavior."""
    if demand_df is None or demand_df.empty:
        return pd.DataFrame()

    targets = _segment_targets_df(demand_df)
    df = targets.rename(columns={
        "target_daily_bev_traffic_2027": "target_flow",
    })

    calibrated = _path_flow_by_segment(calibrated_path_flows_df)
    sampled = _sampled_preferred_flow_by_segment(results, calibrated_path_flows_df)
    actual = segment_flow_validation_table(results, demand_df)

    if not calibrated.empty:
        df = df.merge(calibrated, on="segment_id", how="left")
    else:
        df["calibrated_solution_flow"] = 0.0

    if not sampled.empty:
        df = df.merge(sampled, on="segment_id", how="left")
    else:
        df["sampled_preferred_flow"] = 0.0
        df["sampled_preferred_agents"] = 0

    if not actual.empty:
        actual_cols = [
            "segment_id",
            "simulated_daily_equivalent",
            "weighted_traversals",
            "simulated_traversals",
            "simulated_agent_count",
            "road_name",
        ]
        actual_cols = [c for c in actual_cols if c in actual.columns]
        actual = actual[actual_cols].rename(columns={
            "simulated_daily_equivalent": "actual_simulated_flow",
            "weighted_traversals": "actual_weighted_traversals",
            "simulated_traversals": "actual_unique_agent_segment_crossings",
            "simulated_agent_count": "actual_unique_agents",
            "road_name": "actual_road_name",
        })
        df = df.merge(actual, on="segment_id", how="left")

    fill_zero_cols = [
        "calibrated_solution_flow",
        "sampled_preferred_flow",
        "sampled_preferred_agents",
        "actual_simulated_flow",
        "actual_weighted_traversals",
        "actual_unique_agent_segment_crossings",
        "actual_unique_agents",
    ]
    for col in fill_zero_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["calibration_error"] = df["calibrated_solution_flow"] - df["target_flow"]
    df["sampling_error"] = df["sampled_preferred_flow"] - df["calibrated_solution_flow"]
    df["behavior_error"] = df["actual_simulated_flow"] - df["sampled_preferred_flow"]
    df["actual_error"] = df["actual_simulated_flow"] - df["target_flow"]

    for stage_col, error_col in (
        ("calibrated_solution_flow", "calibration_abs_error"),
        ("sampled_preferred_flow", "sampled_preferred_abs_error"),
        ("actual_simulated_flow", "actual_abs_error"),
    ):
        df[error_col] = (df[stage_col] - df["target_flow"]).abs()

    df["actual_error_pct_vs_target"] = (
        df["actual_abs_error"] / df["target_flow"].replace(0, pd.NA) * 100.0
    )
    df["within_10pct_or_25bev_actual"] = (
        (df["target_flow"] <= 25.0)
        | (df["actual_error_pct_vs_target"] <= 10.0)
    )

    ordered = [
        "segment_id",
        "route_segment",
        "actual_road_name",
        "target_flow",
        "calibrated_solution_flow",
        "sampled_preferred_flow",
        "actual_simulated_flow",
        "calibration_error",
        "sampling_error",
        "behavior_error",
        "actual_error",
        "calibration_abs_error",
        "sampled_preferred_abs_error",
        "actual_abs_error",
        "actual_error_pct_vs_target",
        "within_10pct_or_25bev_actual",
        "sampled_preferred_agents",
        "actual_unique_agent_segment_crossings",
        "actual_unique_agents",
    ]
    ordered = [c for c in ordered if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest].sort_values("target_flow", ascending=False)


def _path_flow_by_segment(
    path_flows_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if path_flows_df is None or path_flows_df.empty:
        return pd.DataFrame(columns=["segment_id", "calibrated_solution_flow"])
    required = {"source_segment_ids", "calibrated_daily_bev_flow"}
    if not required.issubset(path_flows_df.columns):
        return pd.DataFrame(columns=["segment_id", "calibrated_solution_flow"])

    rows: List[Dict] = []
    for row in path_flows_df.itertuples(index=False):
        flow = float(getattr(row, "calibrated_daily_bev_flow", 0.0) or 0.0)
        for segment_id in _parse_segment_ids(getattr(row, "source_segment_ids", "")):
            rows.append({"segment_id": segment_id, "calibrated_solution_flow": flow})
    if not rows:
        return pd.DataFrame(columns=["segment_id", "calibrated_solution_flow"])
    return (
        pd.DataFrame(rows)
        .groupby("segment_id", as_index=False)
        .agg(calibrated_solution_flow=("calibrated_solution_flow", "sum"))
    )


def _sampled_preferred_flow_by_segment(
    results: SimulationResults,
    path_flows_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    recs = results.trip_records_df()
    if recs.empty or path_flows_df is None or path_flows_df.empty:
        return pd.DataFrame(columns=[
            "segment_id", "sampled_preferred_flow", "sampled_preferred_agents"
        ])
    if "demand_path_id" not in recs.columns or "path_id" not in path_flows_df.columns:
        return pd.DataFrame(columns=[
            "segment_id", "sampled_preferred_flow", "sampled_preferred_agents"
        ])

    path_segments = path_flows_df[["path_id", "source_segment_ids"]].drop_duplicates(
        "path_id"
    )
    merged = recs.merge(
        path_segments,
        left_on="demand_path_id",
        right_on="path_id",
        how="left",
    )
    rows: List[Dict] = []
    for row in merged.itertuples(index=False):
        weight = float(getattr(row, "demand_weight", 1.0) or 1.0)
        for segment_id in _parse_segment_ids(getattr(row, "source_segment_ids", "")):
            rows.append({
                "segment_id": segment_id,
                "agent_id": row.agent_id,
                "demand_weight": weight,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "segment_id", "sampled_preferred_flow", "sampled_preferred_agents"
        ])
    expanded = pd.DataFrame(rows).drop_duplicates(["segment_id", "agent_id"])
    return (
        expanded
        .groupby("segment_id", as_index=False)
        .agg(
            sampled_preferred_flow=("demand_weight", "sum"),
            sampled_preferred_agents=("agent_id", "nunique"),
        )
    )


def _parse_segment_ids(raw_ids: object) -> List[int]:
    ids: List[int] = []
    for sid in str(raw_ids or "").split("|"):
        if not sid.strip():
            continue
        try:
            ids.append(int(sid))
        except ValueError:
            continue
    return sorted(set(ids))


def _segment_targets_df(demand_df: pd.DataFrame) -> pd.DataFrame:
    required = {"segment_id", "daily_bev_traffic_2027"}
    missing = required - set(demand_df.columns)
    if missing:
        raise ValueError(f"demand_df missing columns: {sorted(missing)}")

    cols = ["segment_id", "daily_bev_traffic_2027"]
    optional = [c for c in ("route_segment", "length_km") if c in demand_df.columns]
    targets = demand_df[cols + optional].copy()
    targets = targets.rename(columns={"daily_bev_traffic_2027": "target_daily_bev_traffic_2027"})
    targets["segment_id"] = targets["segment_id"].astype(int)
    targets["target_daily_bev_traffic_2027"] = pd.to_numeric(
        targets["target_daily_bev_traffic_2027"], errors="coerce"
    ).fillna(0.0)
    return targets


def compare_scenarios(
    results_dict: Dict[str, SimulationResults],
) -> pd.DataFrame:
    """
    Build a side-by-side metric comparison across scenarios.

    Parameters
    ----------
    results_dict: mapping from scenario_name to SimulationResults.

    Returns
    -------
    DataFrame with one column per scenario.
    """
    rows = {}
    for name, res in results_dict.items():
        ts = trip_summary(res)
        sd = station_demand_table(res)

        row: Dict = {}
        if not ts.empty:
            for col in ts.columns:
                if col != "scenario":
                    row[col] = ts.iloc[0][col]

        if not sd.empty:
            row["max_utilization"] = round(sd["utilization_rate"].max(), 3)
            row["avg_utilization"] = round(sd["utilization_rate"].mean(), 3)
            row["max_peak_queue"] = int(sd["peak_queue"].max())
            row["avg_wait_per_session_min"] = round(
                sd["total_wait_min"].sum() / max(1, sd["total_sessions"].sum()), 2
            )
        rows[name] = row

    df = pd.DataFrame(rows).T
    df.index.name = "scenario"
    return df
