#!/usr/bin/env python3
"""Calibrate path demand on the municipality-attached processed road graph."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_NEW_ABM = _HERE.parent
_REPO_ROOT = _NEW_ABM.parent.parent
sys.path.insert(0, str(_NEW_ABM))

from calibration.path_demand import calibrate_path_demand
from data_generation.municipality_graph import (
    build_municipality_calibration_network,
    build_municipality_od_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "data" / "processed"))
    parser.add_argument("--out-dir", default="/tmp/abm_municipality_path_calibration")
    parser.add_argument("--max-paths-per-od", type=int, default=8)
    parser.add_argument("--max-od-pairs", type=int, default=2000)
    parser.add_argument("--solver-max-iter", type=int, default=500)
    parser.add_argument("--solver-tol", type=float, default=1e-4)
    parser.add_argument(
        "--calibration-objective",
        choices=["absolute", "relative", "sqrt"],
        default="sqrt",
    )
    parser.add_argument("--relative-error-floor", type=float, default=100.0)
    parser.add_argument("--top-contributors-per-segment", type=int, default=5)
    parser.add_argument(
        "--candidate-mode",
        choices=["diagnostic", "boundary", "behavioral"],
        default="behavioral",
    )
    parser.add_argument("--od-conservation-weight", type=float, default=0.5)
    parser.add_argument(
        "--car-mode-share",
        type=float,
        default=None,
        help="Override the default interurban private-car mode share used for people->car-traveler conversion.",
    )
    parser.add_argument(
        "--occupancy-rate",
        type=float,
        default=None,
        help="Override the default interurban private-vehicle occupancy used for people->vehicle conversion.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    network, municipalities, road_segments, network_summary = build_municipality_calibration_network(
        data_dir=data_dir,
        mainland_only=True,
        covered_only=True,
    )

    od_matrix, od_flows, od_summary = build_municipality_od_matrix(
        data_dir=data_dir,
        municipality_codes=set(municipalities["municipality_code"].astype(str)),
        mainland_only=True,
        car_mode_share=args.car_mode_share if args.car_mode_share is not None else None,
        occupancy_rate=args.occupancy_rate if args.occupancy_rate is not None else None,
    )

    demand = pd.read_csv(data_dir / "demand_per_segment.csv")
    result = calibrate_path_demand(
        network=network,
        od_matrix=od_matrix,
        demand_df=demand,
        max_paths_per_od=args.max_paths_per_od,
        max_od_pairs=args.max_od_pairs,
        solver_max_iter=args.solver_max_iter,
        solver_tol=args.solver_tol,
        out_dir=out_dir,
        candidate_mode=args.candidate_mode,
        calibration_objective=args.calibration_objective,
        relative_error_floor=args.relative_error_floor,
        top_contributors_per_segment=args.top_contributors_per_segment,
        od_conservation_weight=args.od_conservation_weight,
    )

    municipalities[[
        "municipality_code",
        "nombre",
        "provincia",
        "node_id",
        "has_endpoint_attachment",
        "endpoint_attachment_count",
        "has_network_anchor",
        "nearest_road",
        "nearest_road_segment_id",
        "nearest_road_distance_km",
    ]].rename(columns={
        "nombre": "municipality_name",
        "provincia": "province_name",
    }).to_csv(out_dir / "abm_municipality_nodes.csv", index=False)
    road_segments[[
        "segment_id",
        "road_name",
        "road_type",
        "length_km",
        "start_municipality_code",
        "start_municipality_name",
        "end_municipality_code",
        "end_municipality_name",
        "start_endpoint_has_covered_municipality",
        "end_endpoint_has_covered_municipality",
        "nearest_covered_municipality_code",
        "nearest_covered_municipality_name",
        "nearest_covered_municipality_distance_km",
        "road_far_from_covered_node",
    ]].to_csv(out_dir / "abm_municipality_road_segments.csv", index=False)
    od_flows[[
        "municipality_origin_code",
        "municipality_destination_code",
        "origin",
        "destination",
        "daily_people",
        "daily_car_travelers",
        "daily_vehicle_trips",
        "daily_bev_trips",
    ]].to_csv(out_dir / "abm_municipality_od_pairs.csv", index=False)

    (out_dir / "abm_municipality_network_summary.json").write_text(
        json.dumps(network_summary, indent=2),
        encoding="utf-8",
    )
    (out_dir / "abm_municipality_od_summary.json").write_text(
        json.dumps(od_summary, indent=2),
        encoding="utf-8",
    )

    combined_summary = {
        "municipality_network": network_summary,
        "municipality_od": od_summary,
        "calibration": result["summary"],
    }
    print(json.dumps(combined_summary, indent=2))
    print(f"\nSaved municipality calibration artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
