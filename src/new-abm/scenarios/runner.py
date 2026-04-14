"""Scenario comparison runner.

ScenarioRunner runs the baseline and one or more alternative scenarios,
then calls the aggregator and visualiser to produce comparison tables
and charts.

Usage
-----
>>> from scenarios.runner import ScenarioRunner
>>> runner = ScenarioRunner(network, base_stations, base_config, trip_requests)
>>> runner.add_scenario(price_reduction_scenario)
>>> runner.add_scenario(capacity_increase_scenario)
>>> comparison_df = runner.run_all(output_dir="outputs/")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from models.demand import TripRequest
from models.network import RoadNetwork
from models.results import SimulationResults
from models.station import ChargingStation
from scenarios.base_scenario import ScenarioConfig, apply_scenario
from simulation.runner import SimulationRunner

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """
    Orchestrates baseline + multi-scenario simulation runs.

    All scenarios are run with the SAME trip requests (same random seed
    for demand generation) so that differences are due to scenario
    modifications only.
    """

    def __init__(
        self,
        network: RoadNetwork,
        base_stations: List[ChargingStation],
        base_config: Dict,
        trip_requests: List[TripRequest],
    ) -> None:
        self.network = network
        self.base_stations = base_stations
        self.base_config = base_config
        self.trip_requests = trip_requests
        self._scenarios: List[ScenarioConfig] = []
        self._results: Dict[str, SimulationResults] = {}

    def add_scenario(self, scenario: ScenarioConfig) -> None:
        """Register a scenario to run."""
        self._scenarios.append(scenario)

    def run_all(
        self,
        run_baseline: bool = True,
        output_dir: Optional[str] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Run baseline (optional) and all registered scenarios.

        Returns
        -------
        Comparison DataFrame with one row per scenario and key metrics.
        """
        if run_baseline:
            logger.info("Running baseline scenario...")
            baseline_runner = SimulationRunner(
                self.network, self.base_stations, self.base_config
            )
            self._results["baseline"] = baseline_runner.run(
                self.trip_requests, scenario_name="baseline", seed=seed
            )

        for sc in self._scenarios:
            logger.info("Running scenario: %s", sc.name)
            mod_stations, mod_config = apply_scenario(
                sc, self.base_stations, self.base_config
            )
            runner = SimulationRunner(self.network, mod_stations, mod_config)
            self._results[sc.name] = runner.run(
                self.trip_requests, scenario_name=sc.name, seed=seed
            )

        comparison_df = self._build_comparison_table()

        if output_dir:
            self._save_outputs(output_dir)

        return comparison_df

    def get_results(self, scenario_name: str) -> Optional[SimulationResults]:
        """Return simulation results for a named scenario."""
        return self._results.get(scenario_name)

    def all_results(self) -> Dict[str, SimulationResults]:
        return dict(self._results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_comparison_table(self) -> pd.DataFrame:
        """Build a side-by-side comparison DataFrame."""
        rows = []
        for name, results in self._results.items():
            row = results.summary_dict()

            # Per-station peak queue and utilization
            sta_df = results.station_summary_df()
            if not sta_df.empty:
                row["avg_station_utilization"] = round(
                    sta_df["utilization_rate"].mean(), 3
                )
                row["max_peak_queue"] = int(sta_df["peak_queue"].max())
                row["avg_wait_at_stations_min"] = round(
                    sta_df["total_wait_min"].sum()
                    / max(1, sta_df["total_sessions"].sum()),
                    2,
                )
            rows.append(row)

        df = pd.DataFrame(rows)
        if "scenario" in df.columns:
            df = df.set_index("scenario")
        return df

    def _save_outputs(self, output_dir: str) -> None:
        """Save per-scenario CSVs and a comparison table."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for name, results in self._results.items():
            safe_name = name.replace(" ", "_")
            results.charge_events_df().to_csv(
                out / f"{safe_name}_charge_events.csv", index=False
            )
            results.trip_records_df().to_csv(
                out / f"{safe_name}_trip_records.csv", index=False
            )
            results.station_summary_df().to_csv(
                out / f"{safe_name}_station_summary.csv", index=False
            )

        # Comparison table
        comparison = self._build_comparison_table()
        comparison.to_csv(out / "scenario_comparison.csv")
        logger.info("Outputs saved to %s/", output_dir)


# ---------------------------------------------------------------------------
# Convenience: build the three standard demo scenarios
# ---------------------------------------------------------------------------

def make_demo_scenarios() -> List[ScenarioConfig]:
    """Return the three standard scenario experiments."""
    price_reduction = ScenarioConfig(
        name="price_reduction",
        description="40% price cut on Madrid-Barcelona corridor stations.",
        station_price_multipliers={
            "STA_MAD_N": 0.60,
            "STA_ZAR_E": 0.60,
            "STA_ZAR_W": 0.60,
            "STA_LLE_E": 0.60,
            "STA_BCN_S": 0.60,
        },
    )

    capacity_increase = ScenarioConfig(
        name="capacity_increase",
        description="Add 4 connectors at Zaragoza East (most congested station).",
        station_connector_deltas={"STA_ZAR_E": 4},
    )

    high_home_charging = ScenarioConfig(
        name="high_home_charging",
        description="Home charging penetration rises from 70% to 95%.",
        home_charging_penetration=0.95,
    )

    return [price_reduction, capacity_increase, high_home_charging]
