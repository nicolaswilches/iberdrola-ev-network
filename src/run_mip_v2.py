"""
One-shot driver: load candidates + demand + gaps + baseline, run place_stations_mip,
save v2 outputs. Intended to be called from NB07c or from CLI for testing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import geopandas as gpd

try:
    from src.optimization import compute_coverage_gaps, place_stations_greedy, place_stations_mip
except ModuleNotFoundError:
    from optimization import compute_coverage_gaps, place_stations_greedy, place_stations_mip


def run(
    data_dir: Path = Path("data/processed"),
    out_dir: Path = Path("data/processed"),
    solver_time_limit_s: int = 120,
    ide_share: float = 0.35,
    endesa_share: float = 0.30,
    viesgo_share: float = 0.10,
    unmet_penalty_eur: float = 200_000.0,
    write_submission_files: bool = True,
) -> dict:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(data_dir / "candidates_v2.parquet")
    coverage = pd.read_parquet(data_dir / "candidate_segment_coverage.parquet")
    demand = pd.read_csv(data_dir / "demand_per_segment.csv")
    baseline = pd.read_csv(data_dir / "interurban_chargers_baseline.csv")

    # Gaps: recompute from roads + baseline chargers (same as NB04/NB07 logic)
    roads_gdf = gpd.read_parquet(data_dir / "interurban_roads.parquet")

    # NB07 derives tent_tier from TENT_red_basica (raw column from NB03);
    # compute_coverage_gaps needs the mapped tier to select the right threshold
    def _tent_tier(row):
        val = row.get("TENT_red_basica")
        if isinstance(val, str):
            v = val.strip().lower()
            if v == "core":
                return "core"
            if v == "comprehensive":
                return "comprehensive"
        return "core" if row.get("is_tent", False) else "none"

    roads_gdf["tent_tier"] = roads_gdf.apply(_tent_tier, axis=1)

    gaps = compute_coverage_gaps(
        road_segments_df=roads_gdf,
        existing_stations_df=baseline,
    )
    print(f"Recomputed AFIR gaps: {len(gaps)}")

    result = place_stations_mip(
        candidates_df=candidates,
        coverage_df=coverage,
        demand_df=demand,
        gaps_df=gaps,
        baseline_chargers_df=baseline,
        dso_min_share={
            "i-DE": ide_share,
            "Endesa": endesa_share,
            "Viesgo": viesgo_share,
        },
        unmet_penalty_eur=unmet_penalty_eur,
        solver_time_limit_s=solver_time_limit_s,
        msg=False,
    )

    stations = result["stations"]
    unmet = result["unmet"]
    summary = result["summary"]

    # --- Post-placement AFIR closer ---------------------------------------
    # The MIP's AFIR constraint (≥ceil(L/T)-1 stations per gap) ensures the
    # right count but not perfect distribution. Re-run gap detection against
    # baseline + v2 stations; if any gap remains, close it with the legacy
    # greedy placer.
    if len(stations) > 0:
        v2_as_chargers = gpd.GeoDataFrame(
            {
                "max_power_kw": 150.0,
                "nearest_road": stations["route_segment"].values,
                "road_prefix": stations["route_segment"].astype(str).str.extract(
                    r"^([A-Z]+)", expand=False
                ).values,
                "is_tent": stations["is_tent"].values,
            },
            geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
            crs="EPSG:4326",
        )
        combined = pd.concat([baseline, v2_as_chargers.drop(columns="geometry")], ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(
            combined,
            geometry=gpd.points_from_xy(combined["longitude"].fillna(0), combined["latitude"].fillna(0)),
            crs="EPSG:4326",
        )
        gaps_remaining = compute_coverage_gaps(roads_gdf, combined)
        if len(gaps_remaining) > 0:
            print(f"\nPost-MIP AFIR gaps remaining: {len(gaps_remaining)} — closing with greedy")
            # Use demand and service areas for the greedy closer
            sa = gpd.read_file(data_dir / "service_areas_clean.geojson")
            closer_stations = place_stations_greedy(
                gap_segments_df=gaps_remaining,
                demand_df=demand,
                service_areas_gdf=sa,
            )
            # Re-id and merge
            closer_stations["location_id"] = [
                f"STAV2_FIX_{i:03d}" for i in range(1, len(closer_stations) + 1)
            ]
            closer_stations["grid_status"] = "Congested"
            closer_stations["distributor_network"] = "Endesa"  # default (gap midpoint)
            closer_stations["source"] = "afir_closer"
            closer_stations["estimated_demand_kw"] = (
                closer_stations["n_chargers_proposed"] * 150
            )
            closer_stations["connection_distance_km"] = None
            closer_stations["available_capacity_mw"] = 0.0
            closer_stations["tent_tier"] = closer_stations.get("tent_tier", "none")
            closer_stations["nearest_substation_id"] = "NO_SUBSTATION"
            closer_stations["fixed_cost_eur"] = 1_150_000.0
            closer_stations["candidate_id"] = [
                f"CAND_AFIR_{i:03d}" for i in range(1, len(closer_stations) + 1)
            ]
            cols = [
                "location_id", "latitude", "longitude", "route_segment",
                "n_chargers_proposed", "grid_status", "distributor_network",
                "source", "connection_distance_km", "available_capacity_mw",
                "is_tent", "tent_tier", "nearest_substation_id",
                "estimated_demand_kw", "fixed_cost_eur", "candidate_id",
            ]
            closer_stations = closer_stations[cols]
            stations = pd.concat([stations, closer_stations], ignore_index=True)
            # Refresh summary fields
            summary["n_stations"] = int(len(stations))
            summary["n_chargers"] = int(stations["n_chargers_proposed"].sum())
            summary["total_kw"] = int(summary["n_chargers"] * 150)
            summary["afir_closer_added"] = int(len(closer_stations))
        else:
            summary["afir_closer_added"] = 0

    # Save v2 outputs
    stations_out = out_dir / "proposed_stations_v2.csv"
    unmet_out = out_dir / "unmet_demand_v2.csv"
    summary_out = out_dir / "mip_v2_summary.json"
    stations.to_csv(stations_out, index=False)
    unmet.to_csv(unmet_out, index=False)
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Brief §5.2 submission files — File_2_v2 and File_3_v2
    # Only written when write_submission_files=True so sensitivity sweeps
    # don't clobber the canonical v2 submission files.
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    if not write_submission_files:
        print("(skipping output/File_*_v2.csv writes — sweep mode)")
    file2 = stations[[
        "location_id", "latitude", "longitude", "route_segment",
        "n_chargers_proposed", "grid_status",
    ]].copy()
    friction = stations[stations["grid_status"].isin(["Moderate", "Congested"])].copy()
    friction["bottleneck_id"] = [f"FRICV2_{i:04d}" for i in range(1, len(friction) + 1)]
    file3 = friction[[
        "bottleneck_id", "latitude", "longitude", "route_segment",
        "distributor_network", "estimated_demand_kw", "grid_status",
    ]].copy()
    baseline_fast = baseline[baseline["max_power_kw"] >= 50]
    file1 = pd.DataFrame([{
        "total_proposed_stations": int(summary["n_stations"]),
        "total_existing_stations_baseline": int(len(baseline_fast)),
        "total_friction_points": int(len(file3)),
        "total_ev_projected_2027": 2_498_159,
    }])
    dso_summary = stations.groupby("distributor_network").agg(
        n_stations=("location_id", "count"),
        n_chargers=("n_chargers_proposed", "sum"),
    ).reset_index()
    dso_summary["total_mw"] = dso_summary["n_chargers"] * 150 / 1000.0
    dso_summary["share_pct"] = (dso_summary["total_mw"] / dso_summary["total_mw"].sum() * 100).round(1)

    file1_path = output_dir / "File_1_v2.csv"
    file2_path = output_dir / "File_2_v2.csv"
    file3_path = output_dir / "File_3_v2.csv"
    dso_path = output_dir / "dso_investment_summary_v2.csv"
    if write_submission_files:
        file1.to_csv(file1_path, index=False)
        file2.to_csv(file2_path, index=False)
        file3.to_csv(file3_path, index=False)
        dso_summary.to_csv(dso_path, index=False)

    # Also save stations_with_grid_status_v2 mirror
    stations_grid_out = out_dir / "stations_with_grid_status_v2.csv"
    stations.to_csv(stations_grid_out, index=False)

    print("\n=== v2 MIP result ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nDSO investment (v2):")
    print(dso_summary.to_string(index=False))
    print(f"\nSaved: {stations_out}")
    print(f"Saved: {stations_grid_out}")
    print(f"Saved: {unmet_out}")
    print(f"Saved: {summary_out}")
    if write_submission_files:
        print(f"Saved: {file1_path}")
        print(f"Saved: {file2_path}")
        print(f"Saved: {file3_path}")
        print(f"Saved: {dso_path}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--time-limit", type=int, default=120)
    ap.add_argument("--ide-share", type=float, default=0.35)
    ap.add_argument("--endesa-share", type=float, default=0.30)
    ap.add_argument("--viesgo-share", type=float, default=0.10)
    ap.add_argument("--unmet-penalty", type=float, default=200_000.0)
    args = ap.parse_args()
    run(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        solver_time_limit_s=args.time_limit,
        ide_share=args.ide_share,
        endesa_share=args.endesa_share,
        viesgo_share=args.viesgo_share,
        unmet_penalty_eur=args.unmet_penalty,
    )


if __name__ == "__main__":
    main()
