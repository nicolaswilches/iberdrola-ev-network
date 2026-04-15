#!/usr/bin/env python3
"""
run_scenarios.py — Scenario comparison experiment.

Runs baseline + 3 scenario modifications and prints a side-by-side
comparison of key metrics.  Saves comparison table and plots.

Scenarios:
  1. baseline         — unmodified network
  2. price_reduction  — 40% price cut on Madrid-Barcelona corridor
  3. capacity_increase — +4 connectors at Zaragoza East (busiest station)
  4. high_home_charging — home charging penetration 70% → 95%

Usage:
    cd src/new-abm
    python run_scenarios.py [--agents N] [--seed S] [--output-dir DIR]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generation.spanish_network import build_spain_real_network
from data_generation.synthetic import build_spain_demo_network
from outputs.aggregator import compare_scenarios
from outputs.visualizer import save_all_plots
from scenarios.runner import ScenarioRunner, make_demo_scenarios

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_scenarios")


def main():
    parser = argparse.ArgumentParser(description="BEV ABM — Scenario Comparison")
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/scenarios")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Path to data/processed/ directory.  When supplied (or auto-detected "
            "at ../../data/processed) the real Spanish network is used."
        ),
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force the synthetic demo network regardless of available data.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(" BEV Interurban ABM — Scenario Comparison")
    print("=" * 65)
    print(f" Agents: {args.agents}   Seed: {args.seed}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config_path = Path(__file__).parent / "config" / "base_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Build network — real data preferred, synthetic as fallback
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)

    data_dir: Optional[Path] = None
    if not args.synthetic:
        if args.data_dir:
            data_dir = Path(args.data_dir)
        else:
            candidate = Path(__file__).parent.parent.parent / "data" / "processed"
            if candidate.exists():
                data_dir = candidate

    use_real = data_dir is not None

    if use_real:
        print(f"\n[1/4] Building real Spanish network from {data_dir} ...")
        try:
            network, stations, od_matrix = build_spain_real_network(
                data_dir=data_dir, rng=rng
            )
            network_label = "real"
        except FileNotFoundError as exc:
            logger.warning("Real data unavailable (%s); falling back to synthetic.", exc)
            network, stations, od_matrix = build_spain_demo_network(rng=rng)
            network_label = "synthetic (fallback)"
    else:
        print("\n[1/4] Building synthetic network...")
        network, stations, od_matrix = build_spain_demo_network(rng=rng)
        network_label = "synthetic"

    print(f"      [{network_label}] {network.node_count} nodes · "
          f"{network.edge_count} edges · {len(stations)} stations")

    # ------------------------------------------------------------------
    # Generate trips (same for ALL scenarios)
    # ------------------------------------------------------------------
    print(f"\n[2/4] Generating {args.agents} trips (shared across scenarios)...")
    rng_trips = np.random.default_rng(args.seed + 100)
    trips = od_matrix.generate_trips(
        rng_trips, num_trips=args.agents,
        peak_config=config.get("demand", {}),
    )

    # ------------------------------------------------------------------
    # Run scenarios
    # ------------------------------------------------------------------
    print("\n[3/4] Running 4 scenarios (baseline + 3 modifications)...")
    sc_runner = ScenarioRunner(network, stations, config, trips)
    for sc in make_demo_scenarios():
        sc_runner.add_scenario(sc)

    t0 = time.time()
    comparison_df = sc_runner.run_all(
        run_baseline=True,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"      All scenarios completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print("\n[4/4] Scenario comparison:")
    print("=" * 65)

    all_results = sc_runner.all_results()
    comp = compare_scenarios(all_results)

    display_cols = [
        "completion_rate_pct", "avg_wait_min", "avg_charge_stops",
        "total_energy_kwh", "max_peak_queue", "avg_utilization",
    ]
    avail_cols = [c for c in display_cols if c in comp.columns]
    print(comp[avail_cols].to_string())

    # ------------------------------------------------------------------
    # Narrative interpretation
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" Scenario Interpretation")
    print("=" * 65)

    results_list = all_results
    baseline = results_list.get("baseline")
    price_sc  = results_list.get("price_reduction")
    cap_sc    = results_list.get("capacity_increase")
    home_sc   = results_list.get("high_home_charging")

    if baseline and price_sc:
        base_energy = baseline.total_energy_dispensed_kwh()
        price_energy = price_sc.total_energy_dispensed_kwh()
        delta_pct = (price_energy - base_energy) / max(1, base_energy) * 100
        print(f"\n  Price reduction on Madrid-BCN corridor:")
        print(f"    Energy dispensed: {base_energy:.0f} → {price_energy:.0f} kWh "
              f"({delta_pct:+.1f}%)")
        print(f"    Did demand shift or just queue move? "
              f"{'Demand shifted' if abs(delta_pct) > 5 else 'Queues redistributed'}")

    if baseline and cap_sc:
        base_wait = baseline.avg_wait_time_min()
        cap_wait  = cap_sc.avg_wait_time_min()
        print(f"\n  Capacity increase at Zaragoza East:")
        print(f"    Avg wait: {base_wait:.1f} → {cap_wait:.1f} min "
              f"({'improved' if cap_wait < base_wait else 'no change'})")

    if baseline and home_sc:
        base_e = baseline.total_energy_dispensed_kwh()
        home_e = home_sc.total_energy_dispensed_kwh()
        print(f"\n  High home charging (70% → 95%):")
        print(f"    Total en-route energy: {base_e:.0f} → {home_e:.0f} kWh "
              f"({'reduced' if home_e < base_e else 'not reduced'} as expected)")

    # ------------------------------------------------------------------
    # Save plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print(f"\n  Saving plots to {args.output_dir}/plots/...")
        saved = save_all_plots(
            network=network,
            stations=stations,
            results_dict=all_results,
            output_dir=str(Path(args.output_dir) / "plots"),
        )
        print(f"  Saved {len(saved)} plots")

    print("\n" + "=" * 65)
    print(f" Done. Outputs in: {Path(args.output_dir).absolute()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
