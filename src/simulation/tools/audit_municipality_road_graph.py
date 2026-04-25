#!/usr/bin/env python3
"""Build and export municipality-to-road attachment diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_NEW_ABM = _HERE.parent
_REPO_ROOT = _NEW_ABM.parent.parent
sys.path.insert(0, str(_NEW_ABM))

from data_generation.municipality_graph import export_municipality_road_graph_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "data" / "processed"))
    parser.add_argument("--out-dir", default="/tmp/municipality_road_graph")
    parser.add_argument("--include-non-mainland", action="store_true")
    parser.add_argument("--covered-only-od", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = export_municipality_road_graph_diagnostics(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        mainland_only=not args.include_non_mainland,
        covered_only=args.covered_only_od,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"\nSaved municipality-road diagnostics to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
