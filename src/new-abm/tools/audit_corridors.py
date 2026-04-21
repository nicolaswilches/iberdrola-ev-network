"""
audit_corridors.py — Verify auto-built road corridors against NB06 demand data.

Usage (from repo root):
    cd src/new-abm
    ../../.venv/bin/python tools/audit_corridors.py
    # or with conda:
    /opt/anaconda3/envs/iberdrola_abm/bin/python tools/audit_corridors.py

Prints:
  1. Count of auto-generated corridors.
  2. Top 20 roads by NB06 demand that the auto-builder missed.
  3. 5 randomly sampled corridors with their ordered city lists.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

# Allow running from repo root or from src/new-abm/
_HERE = Path(__file__).resolve().parent
_NEW_ABM = _HERE.parent
_REPO_ROOT = _NEW_ABM.parent.parent
sys.path.insert(0, str(_NEW_ABM))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = _REPO_ROOT / "data" / "processed"
ROADS_PARQUET = DATA_DIR / "interurban_roads.parquet"
DEMAND_CSV = DATA_DIR / "demand_per_segment.csv"


def main() -> None:
    import pandas as pd
    from data_generation.spanish_network import (
        _ROAD_CORRIDORS,
        _build_road_corridors,
    )

    # ── 1. Build corridors ────────────────────────────────────────────────
    print("\n=== Auto-corridor builder ===")
    corridors = _build_road_corridors(ROADS_PARQUET)
    if not corridors:
        print("ERROR: auto-builder returned empty dict — check geopandas / parquet path")
        sys.exit(1)

    print(f"Auto-generated corridors: {len(corridors)}")
    static_roads = {name for name, _ in _ROAD_CORRIDORS}
    new_roads = set(corridors) - static_roads
    print(f"Roads not in hand-curated list: {len(new_roads)}")

    # ── 2. Demand coverage ────────────────────────────────────────────────
    print("\n=== NB06 demand coverage ===")
    demand_df = pd.read_csv(DEMAND_CSV)
    road_demand = (
        demand_df.groupby("route_segment")["daily_bev_traffic_2027"].sum().sort_values(ascending=False)
    )
    total_flow = road_demand.sum()
    mapped_flow = road_demand[road_demand.index.isin(corridors)].sum()
    unmapped_flow = total_flow - mapped_flow
    unmapped_pct = unmapped_flow / total_flow * 100.0

    print(f"Total NB06 daily BEV trips : {total_flow:,.0f}")
    print(f"Mapped to auto-corridors   : {mapped_flow:,.0f} ({100-unmapped_pct:.1f}%)")
    print(f"Unmapped (dropped)         : {unmapped_flow:,.0f} ({unmapped_pct:.1f}%)")

    missed = road_demand[~road_demand.index.isin(corridors)]
    print(f"\nTop 20 roads by NB06 demand that the auto-builder missed ({len(missed)} total):")
    print(missed.head(20).to_string(header=False))

    if unmapped_pct > 10.0:
        print(
            f"\nWARNING: {unmapped_pct:.1f}% of NB06 demand has no corridor entry. "
            "Consider adding more city nodes or increasing buffer_km."
        )

    # ── 3. Sample 5 corridors ─────────────────────────────────────────────
    print("\n=== Sample corridors (5 random) ===")
    sample_keys = random.sample(sorted(corridors), min(5, len(corridors)))
    for road in sample_keys:
        cities = corridors[road]
        print(f"  {road:12s}: {' → '.join(cities)}")

    # ── 4. Cross-check AP-2, A-45, A-44 ──────────────────────────────────
    print("\n=== Spot-check key corridors ===")
    checks = {
        "AP-2": {"ZAR", "LLE", "TAR", "BCN"},
        "A-45": {"COR", "MAL"},
    }
    for road, expected_cities in checks.items():
        if road not in corridors:
            print(f"  {road}: MISSING from auto-corridors")
        else:
            got = set(corridors[road])
            missing = expected_cities - got
            extra = got - expected_cities
            ok = "OK" if not missing else f"MISSING {missing}"
            print(f"  {road}: {ok} | auto={sorted(got)} extra={sorted(extra)}")

    a44 = corridors.get("A-44")
    print(f"  A-44 (was unmapped): {'present → ' + str(a44) if a44 else 'STILL MISSING'}")


if __name__ == "__main__":
    main()
