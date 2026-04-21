"""Results aggregation helpers.

Functions here convert SimulationResults into analysis-ready DataFrames
suitable for reporting and plotting.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from models.results import SimulationResults


def aggregate_results(results: SimulationResults) -> Dict[str, pd.DataFrame]:
    """
    Produce all standard aggregation tables for one simulation run.

    Returns a dict with keys:
        "station_summary"   — per-station operational metrics
        "trip_summary"      — aggregate trip metrics
        "charge_events"     — raw charging event log
        "trip_records"      — raw trip record log
        "hourly_demand"     — charging demand by hour of day
        "od_completion"     — completion rate by OD pair
    """
    return {
        "station_summary": results.station_summary_df(),
        "trip_summary": trip_summary(results),
        "charge_events": results.charge_events_df(),
        "trip_records": results.trip_records_df(),
        "hourly_demand": hourly_demand_table(results),
        "od_completion": od_completion_table(results),
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
