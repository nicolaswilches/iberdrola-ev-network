"""Validation metrics for comparing simulated vs observed data."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def geh_statistic(simulated: float, observed: float) -> float:
    """Geoffrey E. Havers statistic for a single count comparison."""
    denom = simulated + observed
    if denom <= 0:
        return 0.0
    return float(np.sqrt(2 * (simulated - observed) ** 2 / denom))


def rmspe(simulated: Sequence[float], observed: Sequence[float]) -> float:
    """Root Mean Squared Percentage Error (%)."""
    s = np.array(simulated, dtype=float)
    o = np.array(observed, dtype=float)
    mask = o > 0
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(((s[mask] - o[mask]) / o[mask]) ** 2)) * 100)


def mae(simulated: Sequence[float], observed: Sequence[float]) -> float:
    """Mean Absolute Error."""
    s = np.array(simulated, dtype=float)
    o = np.array(observed, dtype=float)
    return float(np.mean(np.abs(s - o)))


def link_count_validation_report(
    simulated_counts: Dict[str, float],
    observed_counts: Dict[str, float],
) -> pd.DataFrame:
    """
    Build a per-link validation table.

    Columns: link_id, observed, simulated, geh, within_5pct
    """
    rows = []
    for link_id in observed_counts:
        obs = observed_counts[link_id]
        sim = simulated_counts.get(link_id, 0.0)
        geh = geh_statistic(sim, obs)
        within_20 = abs(sim - obs) / max(1, obs) < 0.20
        rows.append(
            {
                "link_id": link_id,
                "observed": obs,
                "simulated": sim,
                "difference": sim - obs,
                "pct_error": round((sim - obs) / max(1, obs) * 100, 1),
                "geh": round(geh, 2),
                "geh_ok": geh < 5.0,
                "within_20pct": within_20,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("geh", ascending=False)
    return df


class ValidationMetrics:
    """Container for all validation results from one simulation run."""

    def __init__(self) -> None:
        self.link_geh_mean: float = 0.0
        self.link_geh_pct_below5: float = 100.0
        self.station_rmspe: float = 0.0
        self.wait_time_mae_min: float = 0.0
        self.completion_rate: float = 0.0
        self.link_report: pd.DataFrame = pd.DataFrame()

    def summary(self) -> Dict[str, float]:
        return {
            "link_geh_mean": round(self.link_geh_mean, 2),
            "link_geh_pct_below5": round(self.link_geh_pct_below5, 1),
            "station_rmspe_pct": round(self.station_rmspe, 1),
            "wait_time_mae_min": round(self.wait_time_mae_min, 2),
            "completion_rate_pct": round(self.completion_rate * 100, 1),
        }

    def is_acceptable(
        self,
        geh_threshold: float = 5.0,
        geh_pct_target: float = 85.0,
        rmspe_target: float = 20.0,
    ) -> bool:
        """True if the simulation meets standard calibration acceptance criteria."""
        return (
            self.link_geh_pct_below5 >= geh_pct_target
            and self.station_rmspe <= rmspe_target
        )
