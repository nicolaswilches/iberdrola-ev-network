#!/usr/bin/env python3
"""Audit ABM station clustering invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NEW_ABM = _HERE.parent
_REPO_ROOT = _NEW_ABM.parent.parent
sys.path.insert(0, str(_NEW_ABM))

from data_generation.spanish_network import build_spain_real_network


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "data" / "processed"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    _network, stations, _od = build_spain_real_network(
        Path(args.data_dir),
        rng=np.random.default_rng(42),
        include_proposed_stations=True,
    )
    rows = []
    violations = []
    for station in stations:
        exception = getattr(station, "cluster_exception", "")
        row = {
            "station_id": station.station_id,
            "road_name": getattr(station, "road_name", ""),
            "road_km": getattr(station, "road_km", 0.0),
            "cluster_span_km": getattr(station, "cluster_span_km", 0.0),
            "num_connectors": station.num_connectors,
            "physical_station_count": getattr(station, "physical_station_count", 1),
            "cluster_exception": exception,
        }
        rows.append(row)
        if row["cluster_span_km"] > 10.0001:
            violations.append({**row, "violation": "cluster_span_gt_10km"})
        if row["num_connectors"] > 24 and exception != "single_station_over_cap":
            violations.append({**row, "violation": "connector_cap_gt_24_without_exception"})

    summary = {
        "n_stations": len(stations),
        "n_violations": len(violations),
        "max_cluster_span_km": max((r["cluster_span_km"] for r in rows), default=0.0),
        "max_regular_connectors": max(
            (r["num_connectors"] for r in rows if r["cluster_exception"] != "single_station_over_cap"),
            default=0,
        ),
        "single_station_over_cap_exceptions": sum(
            1 for r in rows if r["cluster_exception"] == "single_station_over_cap"
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        import pandas as pd

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    if violations:
        print(json.dumps({"violations": violations[:20]}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
