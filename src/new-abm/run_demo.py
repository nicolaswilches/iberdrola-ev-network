#!/usr/bin/env python3
"""
run_demo.py — Baseline simulation demo.

Runs a single baseline simulation on the synthetic Spain-like network,
prints key metrics to stdout, and saves output plots and CSVs.

Usage:
    cd src/new-abm
    python run_demo.py [--agents N] [--seed S] [--output-dir DIR]

    --agents:     Number of trip agents (default 200)
    --seed:       Random seed (default 42)
    --output-dir: Where to save outputs (default outputs/)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# Add parent directory so modules import correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generation.spanish_network import build_spain_real_network
from data_generation.synthetic import build_spain_demo_network, export_network_to_csv
from models.demand import generate_trips_from_calibrated_paths
from outputs.aggregator import aggregate_results, trip_summary, station_demand_table
from simulation.runner import SimulationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_demo")


def main():
    parser = argparse.ArgumentParser(description="BEV Interurban ABM — Baseline Demo")
    parser.add_argument("--agents", type=int, default=10000, help="Number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="src/new-abm/outputs", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--verbose", action="store_true", help="Show DEBUG logs")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Path to data/processed/ directory containing pipeline CSVs. "
            "When supplied (or auto-detected at ../../data/processed), the "
            "real Spanish network is used instead of the synthetic demo network."
        ),
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force use of the synthetic demo network even if real data is available.",
    )
    parser.add_argument(
        "--calibrated-path-flows",
        type=str,
        default=None,
        help=(
            "Path to abm_calibrated_path_flows.csv. When supplied, trips are "
            "sampled from calibrated path flows and each agent follows the "
            "calibrated node_path when feasible."
        ),
    )
    parser.add_argument(
        "--exclude-roadspan-paths",
        action="store_true",
        help="Exclude ROADSPAN calibration-support paths from calibrated trip sampling.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print(" BEV Interurban ABM — Baseline Simulation Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load configuration
    # ------------------------------------------------------------------
    config_path = Path(__file__).parent / "config" / "base_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override number of agents
    if args.agents != 200:
        logger.info("Overriding num_agents to %d", args.agents)

    # ------------------------------------------------------------------
    # 2. Build network — real data preferred, synthetic as fallback
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)

    # Resolve data directory: explicit arg > auto-detect > None
    data_dir: Optional[Path] = None
    if not args.synthetic:
        if args.data_dir:
            data_dir = Path(args.data_dir)
        else:
            # Auto-detect: two levels up from this file → data/processed
            candidate = Path(__file__).parent.parent.parent / "data" / "processed"
            if candidate.exists():
                data_dir = candidate

    use_real = data_dir is not None and not args.synthetic

    if use_real:
        print(f"\n[1/5] Building real Spanish network from {data_dir} ...")
        try:
            network, stations, od_matrix = build_spain_real_network(
                data_dir=data_dir, rng=rng,
                cluster_stations_per_group=10,
            )
            network_label = "real"
        except FileNotFoundError as exc:
            logger.warning("Real data unavailable (%s); falling back to synthetic.", exc)
            network, stations, od_matrix = build_spain_demo_network(rng=rng)
            network_label = "synthetic (fallback)"
    else:
        print(f"\n[1/5] Building synthetic Spain-like network...")
        network, stations, od_matrix = build_spain_demo_network(rng=rng)
        network_label = "synthetic"

    print(f"      Network ({network_label}): {network.node_count} nodes, "
          f"{network.edge_count} directed edges")
    print(f"      Stations: {len(stations)} fast-charging locations")
    print(f"      OD matrix: {len(od_matrix.pairs)} pairs, "
          f"{od_matrix.total_daily_trips():.0f} daily BEV trips")

    demand_targets = None
    if use_real and data_dir is not None and (data_dir / "demand_per_segment.csv").exists():
        demand_targets = pd.read_csv(data_dir / "demand_per_segment.csv")

    # Export network CSVs for inspection
    export_dir = Path(args.output_dir) / "data"
    if network_label == "synthetic":
        export_network_to_csv(network, stations, od_matrix, str(export_dir))
        print(f"      Network CSVs saved to {export_dir}/")

    # ------------------------------------------------------------------
    # 3. Generate trip requests
    # ------------------------------------------------------------------
    print(f"\n[2/5] Generating {args.agents} trip requests...")
    rng2 = np.random.default_rng(args.seed + 1)
    peak_config = config.get("demand", {})
    calibrated_path_flows_df = None
    if args.calibrated_path_flows:
        calibrated_path_flows_df = pd.read_csv(args.calibrated_path_flows)
        trips = generate_trips_from_calibrated_paths(
            args.calibrated_path_flows,
            rng2,
            num_trips=args.agents,
            peak_config=peak_config,
            include_roadspan_paths=not args.exclude_roadspan_paths,
        )
        scenario_name = "calibrated_paths"
        support_count = sum(1 for t in trips if t.is_calibration_support_path)
        print(
            f"      Calibrated path source: {args.calibrated_path_flows}"
        )
        print(
            f"      ROADSPAN support trips: {support_count} "
            f"({'included' if not args.exclude_roadspan_paths else 'excluded'})"
        )
    else:
        trips = od_matrix.generate_trips(rng2, num_trips=args.agents, peak_config=peak_config)
        scenario_name = "baseline"
    if not trips:
        raise RuntimeError("No trip requests generated; check demand inputs and filters.")
    dep_times = [t.departure_time_min for t in trips]
    print(f"      Departure range: {min(dep_times):.0f}–{max(dep_times):.0f} min "
          f"({min(dep_times)/60:.1f}h–{max(dep_times)/60:.1f}h)")

    # ------------------------------------------------------------------
    # 4. Run simulation
    # ------------------------------------------------------------------
    print(f"\n[3/5] Running discrete-event simulation...")
    print(f"      Agents: {args.agents}, seed: {args.seed}")
    t0 = time.time()
    runner = SimulationRunner(network, stations, config)
    results = runner.run(trips, scenario_name=scenario_name, seed=args.seed)
    elapsed = time.time() - t0
    print(f"      Simulation completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 5. Print results summary
    # ------------------------------------------------------------------
    print(f"\n[4/5] Results summary:")
    print("-" * 50)
    summary = results.summary_dict()
    print(f"  Trip completion rate:   {summary['completion_rate']:.1%}")
    print(f"  Stranded agents:        {summary['num_stranded']}")
    print(f"  Total charge events:    {summary['total_charge_events']}")
    print(f"  Total energy dispensed: {summary['total_energy_kwh']:.0f} kWh")
    print(f"  Avg wait time:          {summary['avg_wait_min']:.1f} min")
    print(f"  Avg charge time:        {summary['avg_charge_min']:.1f} min")

    print("\n  Station demand (top 8 by sessions):")
    sta_df = station_demand_table(results).head(8)
    if not sta_df.empty:
        for _, row in sta_df.iterrows():
            print(
                f"    {row['name']:<28} "
                f"sessions={int(row['total_sessions']):>3}  "
                f"energy={row['total_energy_kwh']:>6.0f} kWh  "
                f"util={row['utilization_rate']:.0%}  "
                f"peak_q={int(row['peak_queue'])}"
            )

    ts = trip_summary(results)
    if not ts.empty:
        row = ts.iloc[0]
        print(f"\n  Avg charge stops/trip:  {row['avg_charge_stops']:.2f}")
        print(f"  % trips with 0 stops:   {row['pct_zero_stops']:.0f}%")

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    print(f"\n[5/5] Saving CSVs to {args.output_dir}/...")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    agg = aggregate_results(
        results,
        demand_df=demand_targets,
        calibrated_path_flows_df=calibrated_path_flows_df,
    )
    for table_name, df in agg.items():
        if not df.empty:
            path = out / f"{scenario_name}_{table_name}.csv"
            df.to_csv(path, index=False)

    if not args.no_plots:
        print(f"      Saving plots to {out / 'plots'}...")
        from outputs.visualizer import save_all_plots

        saved_plots = save_all_plots(
            network=network,
            stations=stations,
            results_dict={scenario_name: results},
            output_dir=str(out / "plots"),
        )
        print(f"      Saved {len(saved_plots)} plots")
    else:
        print("      Skipping plots (--no-plots).")

    print("\n" + "=" * 60)
    print(" Demo complete.")
    print(f" Output directory: {Path(args.output_dir).absolute()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
