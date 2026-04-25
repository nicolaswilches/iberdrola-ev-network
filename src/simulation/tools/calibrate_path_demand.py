#!/usr/bin/env python3
"""Calibrate ABM path demand against segment-level BEV traffic targets.

Example:
    python src/simulation/tools/calibrate_path_demand.py \
      --max-od-pairs 300 --max-paths-per-od 8 --out-dir /tmp/abm_path_calibration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_NEW_ABM = _HERE.parent
_REPO_ROOT = _NEW_ABM.parent.parent
sys.path.insert(0, str(_NEW_ABM))

from calibration.path_demand import calibrate_path_demand
from data_generation.spanish_network import build_spain_real_network


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "data" / "processed"))
    parser.add_argument("--out-dir", default="/tmp/abm_path_calibration")
    parser.add_argument("--max-paths-per-od", type=int, default=8)
    parser.add_argument("--max-od-pairs", type=int, default=None)
    parser.add_argument("--solver-max-iter", type=int, default=500)
    parser.add_argument("--solver-tol", type=float, default=1e-4)
    parser.add_argument(
        "--calibration-objective",
        choices=["absolute", "relative", "sqrt"],
        default="sqrt",
        help=(
            "absolute minimizes raw segment error; relative minimizes error "
            "scaled by max(target, --relative-error-floor); sqrt is an "
            "intermediate weighting."
        ),
    )
    parser.add_argument("--relative-error-floor", type=float, default=100.0)
    parser.add_argument("--top-contributors-per-segment", type=int, default=5)
    parser.add_argument(
        "--candidate-mode",
        choices=["diagnostic", "boundary", "behavioral"],
        default="diagnostic",
        help=(
            "diagnostic includes ROADSPAN and synthetic endpoint OD paths to "
            "measure geometry coverage; boundary includes synthetic endpoint "
            "OD paths as external/local access zones but excludes ROADSPAN; "
            "behavioral excludes both."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    network, _stations, od_matrix = build_spain_real_network(
        data_dir=data_dir,
        rng=np.random.default_rng(args.seed),
        include_proposed_stations=True,
        od_debug_dir=out_dir,
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
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"\nSaved calibration artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
