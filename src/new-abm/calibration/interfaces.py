"""Calibration interface scaffolding.

This module defines the contracts and placeholder implementations for
calibrating the ABM against real observed data.

In production use, this module connects to:

  1. MiTMA Intensity Máxima Diaria (IMD) data:
     DGT permanent traffic stations measure vehicles/day on each major road.
     Real file: mitma_imd_2024.csv (format: station_id, road, vehicles_day)

  2. NAP or ChargeMap charging station usage data:
     Session counts, energy dispensed, and wait times at real stations.
     Real file: station_observations_2024.csv

  3. INE vehicle registration data:
     BEV share by province for fleet composition calibration.

Standard transport planning calibration procedure
-------------------------------------------------
1. Run the model with an initial OD matrix.
2. Extract simulated link counts (vehicles crossing each road segment).
3. Compare to observed IMD counts using GEH statistic or RMSPE.
4. Adjust OD flows using gradient descent or a matrix estimation method
   (e.g., SPIMAT / ITEROD) to reduce the gap.
5. Repeat until GEH < 5 for 85% of links (HCM standard).

For behavioral parameter calibration:
  - Compare simulated avg_wait_time to observed wait times from station data.
  - Adjust queue_aversion, value_of_time distributions to match.
  - Use Bayesian optimization or simulated annealing for multi-parameter fits.

CURRENTLY: All methods here are stubs.  They define the API so the rest
of the model knows exactly what data slots need filling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CalibrationInterface:
    """
    Interface for connecting real observed data to the ABM.

    Usage (sketch)::

        cal = CalibrationInterface()
        cal.load_imd_counts("data/raw/imd_2024.csv")
        cal.load_station_observations("data/raw/station_obs_2024.csv")

        # After running baseline simulation:
        score = cal.compute_link_count_error(simulated_link_counts)
        # → Score to minimise in calibration loop

        adjusted_od = cal.adjust_od_matrix(
            od_matrix, simulated_link_counts, target_iter=10
        )
    """

    def __init__(self) -> None:
        self.observed_link_counts: Dict[str, float] = {}
        # link_id → observed daily BEV count

        self.observed_station_sessions: Dict[str, float] = {}
        # station_id → observed daily session count

        self.observed_avg_wait_min: Dict[str, float] = {}
        # station_id → observed average wait time (minutes)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_imd_counts(self, filepath: str) -> None:
        """
        Load observed IMD (Intensidad Máxima Diaria) link counts.

        Expected CSV format::

            link_id,road_name,lat,lon,vehicles_day,bev_fraction,source_year
            LNK_001,AP-2_km150,41.5,0.2,12500,0.057,2024
            ...

        The ``bev_fraction`` column converts total vehicle counts to BEV counts.
        If not present, the model default EV penetration rate is used.

        Real data source:
            DGT permanent traffic stations:
            https://www.dgt.es/menuweb/catalogo/catalogopublicaciones_aforos.htm
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning("IMD file not found: %s (using synthetic data)", filepath)
            return

        df = pd.read_csv(filepath)
        required = {"link_id", "vehicles_day"}
        if not required.issubset(df.columns):
            raise ValueError(f"IMD file must contain columns: {required}")

        bev_frac = df.get("bev_fraction", pd.Series([0.057] * len(df)))
        for _, row in df.iterrows():
            self.observed_link_counts[row["link_id"]] = (
                row["vehicles_day"] * float(bev_frac.iloc[_] if hasattr(bev_frac, "iloc") else bev_frac)
            )

        logger.info("Loaded %d IMD link counts from %s", len(self.observed_link_counts), filepath)

    def load_station_observations(self, filepath: str) -> None:
        """
        Load observed station-level usage statistics.

        Expected CSV format::

            station_id,daily_sessions,avg_wait_min,avg_energy_kwh
            STA_001,45,8.3,22.1
            ...

        Real data source:
            Operator APIs (Iberdrola SmartCharge, Endesa X-Way),
            ChargeMap API (aggregated, anonymised),
            or RECHARGE project station reports.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning("Station observations file not found: %s", filepath)
            return

        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            sid = row["station_id"]
            if "daily_sessions" in df.columns:
                self.observed_station_sessions[sid] = float(row["daily_sessions"])
            if "avg_wait_min" in df.columns:
                self.observed_avg_wait_min[sid] = float(row["avg_wait_min"])

        logger.info("Loaded station observations for %d stations", len(self.observed_station_sessions))

    # ------------------------------------------------------------------
    # Error metrics for calibration loop
    # ------------------------------------------------------------------

    def compute_link_geh(
        self, simulated_counts: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Compute GEH statistic for simulated vs observed link counts.

        GEH(i) = sqrt(2 * (simulated_i - observed_i)^2 / (simulated_i + observed_i))

        GEH < 5 for ≥85% of links is the standard acceptance criterion (HCM).

        Returns
        -------
        (mean_geh, pct_below_5)
        """
        if not self.observed_link_counts:
            logger.warning("No observed link counts loaded; cannot compute GEH")
            return 0.0, 100.0

        gehs = []
        for link_id, observed in self.observed_link_counts.items():
            simulated = simulated_counts.get(link_id, 0.0)
            if observed + simulated > 0:
                geh = np.sqrt(
                    2 * (simulated - observed) ** 2 / (simulated + observed)
                )
                gehs.append(geh)

        if not gehs:
            return 0.0, 100.0

        gehs_arr = np.array(gehs)
        return float(np.mean(gehs_arr)), float(np.mean(gehs_arr < 5) * 100)

    def compute_station_rmspe(
        self, simulated_sessions: Dict[str, float]
    ) -> float:
        """
        Root Mean Squared Percentage Error for station session counts.

        Returns RMSPE in percent.  Target: < 20%.
        """
        if not self.observed_station_sessions:
            return 0.0

        errors = []
        for sid, observed in self.observed_station_sessions.items():
            simulated = simulated_sessions.get(sid, 0.0)
            if observed > 0:
                errors.append(((simulated - observed) / observed) ** 2)

        return float(np.sqrt(np.mean(errors)) * 100) if errors else 0.0

    # ------------------------------------------------------------------
    # OD adjustment (stub — replace with SPIMAT/ITEROD in production)
    # ------------------------------------------------------------------

    def adjust_od_matrix(
        self,
        od_pairs: List,
        simulated_link_counts: Dict[str, float],
        observed_link_counts: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.1,
        max_iterations: int = 10,
    ) -> List:
        """
        Placeholder for iterative proportional fitting (IPF) OD adjustment.

        In production this would implement the method from:
        Cascetta & Nguyen (1988) "A unified framework for estimating or
        updating origin/destination matrices from traffic counts"

        Currently returns the input od_pairs unchanged.
        """
        logger.warning(
            "adjust_od_matrix is a stub. "
            "Implement SPIMAT/ITEROD here for production calibration."
        )
        return od_pairs

    # ------------------------------------------------------------------
    # Behavioral parameter calibration (stub)
    # ------------------------------------------------------------------

    def calibrate_behavioral_params(
        self,
        simulation_runner_factory: Callable,
        param_bounds: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Placeholder for behavioral parameter calibration.

        Would use Bayesian optimization or simulated annealing to search
        the parameter space {value_of_time, queue_aversion, price_sensitivity}
        to minimise the gap between simulated and observed wait times.

        param_bounds example::

            {
                "value_of_time_eur_per_hour": (10, 60),
                "queue_aversion": (0.2, 3.0),
                "price_sensitivity": (0.5, 2.0),
            }

        Currently returns default values.
        """
        logger.warning(
            "calibrate_behavioral_params is a stub. "
            "Implement Bayesian optimization here for production use."
        )
        return {
            "value_of_time_eur_per_hour": 28.0,
            "queue_aversion": 1.0,
            "price_sensitivity": 1.0,
        }
