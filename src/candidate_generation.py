"""
Candidate set generation for the v2 MIP network optimization (NB07c).

Produces an enriched candidate universe (service areas, upgrade sites,
grid-friendly substation-adjacent points, high-IMD spaced points, gap midpoints)
with per-candidate grid metadata and a segment-coverage matrix.

Outputs:
  - data/processed/candidates_v2.parquet
  - data/processed/candidate_segment_coverage.parquet

Brief constraint (§2 Scope of Analysis): every candidate must be on an
interurban road (AP-, A-, N-). Urban sections are excluded by construction
because all source datasets here are already filtered to interurban segments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import linemerge, unary_union
from sklearn.neighbors import BallTree

try:
    from src.constants import (
        GRID_MODERATE_MIN_MW,
        GRID_SUFFICIENT_MIN_MW,
        MAX_SUBSTATION_SEARCH_RADIUS_KM,
        SUBSTATION_DIST_FEASIBLE_KM,
        SUBSTATION_DIST_HIGH_COST_KM,
        SUBSTATION_DIST_OPTIMAL_KM,
    )
except ModuleNotFoundError:  # when src/ is on path directly
    from constants import (
        GRID_MODERATE_MIN_MW,
        GRID_SUFFICIENT_MIN_MW,
        MAX_SUBSTATION_SEARCH_RADIUS_KM,
        SUBSTATION_DIST_FEASIBLE_KM,
        SUBSTATION_DIST_HIGH_COST_KM,
        SUBSTATION_DIST_OPTIMAL_KM,
    )

EARTH_KM = 6371.0088

# Fixed-cost (Capex site-preparation surcharge) by candidate source, EUR
# (assumption F1 €80-130k/charger is the per-charger component; these
# are the per-site fixed overheads that differ across source types)
FIXED_COST_BY_SOURCE = {
    "service_area": 100_000,
    "upgrade": 50_000,
    "grid_friendly": 150_000,
    "high_imd": 200_000,
    "gap_midpoint": 150_000,
}

# Per-charger variable cost (mid of F1 band)
VARIABLE_COST_PER_CHARGER = 100_000

# Source priority for dedupe (lower = preferred)
SOURCE_PRIORITY = {
    "service_area": 0,
    "upgrade": 1,
    "grid_friendly": 2,
    "gap_midpoint": 3,
    "high_imd": 4,
}


# ----------------------------- Helpers ---------------------------------

def _classify_grid_status(mw: float) -> str:
    if pd.isna(mw):
        return "Congested"
    if mw >= GRID_SUFFICIENT_MIN_MW:
        return "Sufficient"
    if mw >= GRID_MODERATE_MIN_MW:
        return "Moderate"
    return "Congested"


def _classify_connection_tier(dist_km: float) -> str:
    if dist_km <= SUBSTATION_DIST_OPTIMAL_KM:
        return "optimal"
    if dist_km <= SUBSTATION_DIST_FEASIBLE_KM:
        return "feasible"
    if dist_km <= SUBSTATION_DIST_HIGH_COST_KM:
        return "high_cost"
    return "remote"


def _haversine_tree(coords_deg: np.ndarray) -> BallTree:
    return BallTree(np.radians(coords_deg), metric="haversine")


# ----------------------------- Source generators -----------------------

def _sa_candidates(sa_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Service areas on interurban roads."""
    if sa_gdf.empty:
        return pd.DataFrame()
    g = sa_gdf.copy()
    # Points are already in EPSG:4326
    if g.crs is None or str(g.crs).upper() not in ("EPSG:4326", "WGS 84"):
        g = g.to_crs("EPSG:4326")
    g["latitude"] = g.geometry.y
    g["longitude"] = g.geometry.x
    # nombre_via -> route_segment. Strip trailing N/S if present (AP-7N -> AP-7)
    road = g["nombre_via"].astype(str).str.replace(r"[NS]$", "", regex=True)
    df = pd.DataFrame({
        "latitude": g["latitude"].values,
        "longitude": g["longitude"].values,
        "route_segment": road.values,
        "source": "service_area",
    })
    return df


def _upgrade_candidates(
    chargers_df: pd.DataFrame,
    min_power_kw: float = 50.0,
    cluster_km: float = 0.5,
) -> pd.DataFrame:
    """Existing ≥50 kW interurban chargers as upgrade candidates.

    Clusters nearby sites within cluster_km so each physical location
    contributes one candidate.
    """
    fast = chargers_df[chargers_df["max_power_kw"] >= min_power_kw].copy()
    fast = fast.dropna(subset=["latitude", "longitude", "road_prefix"])
    # Keep only interurban (AP, A, N) — should already be the case
    fast = fast[fast["road_prefix"].isin(["AP", "A", "N"])]
    if fast.empty:
        return pd.DataFrame()
    # Cluster by proximity
    coords = fast[["latitude", "longitude"]].to_numpy()
    tree = _haversine_tree(coords)
    cluster_rad = cluster_km / EARTH_KM
    visited = np.zeros(len(fast), dtype=bool)
    keep_idx: list[int] = []
    for i in range(len(fast)):
        if visited[i]:
            continue
        keep_idx.append(i)
        close = tree.query_radius(np.radians(coords[i : i + 1]), r=cluster_rad)[0]
        visited[close] = True
    kept = fast.iloc[keep_idx].copy()
    df = pd.DataFrame({
        "latitude": kept["latitude"].values,
        "longitude": kept["longitude"].values,
        "route_segment": kept["nearest_road"].astype(str).values,
        "source": "upgrade",
    })
    return df


def _grid_friendly_candidates(
    grid_df: pd.DataFrame,
    road_segments_df: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    min_mw: float = 0.6,
    max_road_dist_km: float = 5.0,
) -> pd.DataFrame:
    """Substations with usable capacity projected onto the nearest interurban road.

    For each eligible substation, find the closest point on any AP/A/N road
    segment (within max_road_dist_km). That projected point is the candidate.
    """
    elig = grid_df[grid_df["available_capacity_mw"] >= min_mw].copy()
    elig = elig.dropna(subset=["latitude", "longitude"])
    if elig.empty or roads_gdf.empty:
        return pd.DataFrame()

    # Build road segment geometry in UTM for accurate projection
    roads_utm = roads_gdf.to_crs("EPSG:25830")
    sub_gdf = gpd.GeoDataFrame(
        elig,
        geometry=gpd.points_from_xy(elig["longitude"], elig["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:25830")

    # Use spatial join to nearest road — handle each substation
    candidates = []
    road_geoms = roads_utm.geometry.values
    road_carretera = roads_utm["Carretera"].values
    # BallTree over road centroids for coarse nearest; fall back to exact
    road_cents = np.array([[g.centroid.y, g.centroid.x] for g in road_geoms])
    # EPSG:25830 is meters; use euclidean
    from scipy.spatial import cKDTree
    tree = cKDTree(road_cents)
    sub_coords = np.c_[sub_gdf.geometry.y.values, sub_gdf.geometry.x.values]
    # Query 10 nearest roads per substation, exact projection on each
    dists, idxs = tree.query(sub_coords, k=min(10, len(road_geoms)))
    for i, sub in enumerate(sub_gdf.itertuples(index=False)):
        best_dist = np.inf
        best_pt = None
        best_road = None
        for ri in idxs[i]:
            geom = road_geoms[ri]
            proj = geom.interpolate(geom.project(sub.geometry))
            d = sub.geometry.distance(proj)  # meters
            if d < best_dist:
                best_dist = d
                best_pt = proj
                best_road = road_carretera[ri]
        if best_pt is None or best_dist / 1000.0 > max_road_dist_km:
            continue
        # Back to WGS84
        p = gpd.GeoSeries([best_pt], crs="EPSG:25830").to_crs("EPSG:4326").iloc[0]
        candidates.append({
            "latitude": p.y,
            "longitude": p.x,
            "route_segment": str(best_road),
            "source": "grid_friendly",
        })
    return pd.DataFrame(candidates)


def _high_imd_candidates(
    demand_df: pd.DataFrame,
    road_segments_df: pd.DataFrame,
    imd_threshold: float = 15000.0,
    spacing_km: float = 30.0,
) -> pd.DataFrame:
    """Regularly-spaced candidates along segments with IMD > threshold.

    Uses segment midpoint as candidate when length < spacing; otherwise
    evenly spaces candidates along the segment linear extent.
    """
    hot = road_segments_df.merge(
        demand_df[["segment_id", "imd_total"]],
        on="segment_id",
        how="inner",
        suffixes=("", "_d"),
    )
    hot = hot[hot["imd_total"] >= imd_threshold].copy()
    hot = hot.dropna(subset=["PK_inicio", "PK_fin"])
    if hot.empty:
        return pd.DataFrame()
    # For each segment, place candidates at (PK_start + spacing, ...) —
    # we don't have direct road geometry here, so we use segment midpoint
    # as a proxy. True along-road sampling is done downstream in the MIP
    # coverage matrix via segment_id relevance.
    pts = []
    # Use road_segments_with_imd has no lat/lon directly; defer to geometry
    # For now, use segment_id as unique; candidate coordinates come from
    # NB03's interurban_roads.parquet merge below.
    # (Fallback: synthetic points are unusable without lat/lon, so skip
    # geometry-less segments. We'll rely on demand-seeded candidates via
    # the supporting road geometry load in build_candidate_set.)
    return pd.DataFrame()  # placeholder; we build from geometry in caller


def _high_imd_candidates_from_geometry(
    demand_df: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    imd_threshold: float = 15000.0,
    spacing_km: float = 30.0,
) -> pd.DataFrame:
    """Place candidates at regular intervals along high-IMD road geometries."""
    hot = demand_df[demand_df["imd_total"] >= imd_threshold].copy()
    if hot.empty or roads_gdf.empty:
        return pd.DataFrame()
    # Merge geometry by Carretera/route_segment where possible
    g = roads_gdf.copy()
    if g.crs is None or str(g.crs) != "EPSG:25830":
        g = g.to_crs("EPSG:25830")
    # Group segments by Carretera and build one merged line per route
    merged = (
        g.dissolve(by="Carretera")
        .reset_index()[["Carretera", "geometry"]]
    )
    hot_roads = set(hot["route_segment"].astype(str).unique())
    merged = merged[merged["Carretera"].astype(str).isin(hot_roads)]
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for _, m in merged.iterrows():
        geom = m["geometry"]
        try:
            merged_geom = linemerge(unary_union(geom))
        except Exception:
            merged_geom = geom
        if hasattr(merged_geom, "geoms"):
            parts = list(merged_geom.geoms)
        else:
            parts = [merged_geom]
        for part in parts:
            length_m = part.length
            if length_m <= 0:
                continue
            step_m = spacing_km * 1000.0
            if length_m < step_m:
                # Route shorter than spacing — emit one candidate at midpoint
                positions = np.array([length_m / 2])
            else:
                positions = np.arange(step_m / 2, length_m, step_m)
            for pos in positions:
                pt_utm = part.interpolate(pos)
                pt_ll = gpd.GeoSeries([pt_utm], crs="EPSG:25830").to_crs("EPSG:4326").iloc[0]
                rows.append({
                    "latitude": pt_ll.y,
                    "longitude": pt_ll.x,
                    "route_segment": str(m["Carretera"]),
                    "source": "high_imd",
                })
    return pd.DataFrame(rows)


def _gap_midpoint_candidates(proposed_stations_df: pd.DataFrame) -> pd.DataFrame:
    """Reuse the current 8 AFIR gap-midpoint locations as baseline candidates."""
    if proposed_stations_df.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        "latitude": proposed_stations_df["latitude"].values,
        "longitude": proposed_stations_df["longitude"].values,
        "route_segment": proposed_stations_df["route_segment"].astype(str).values,
        "source": "gap_midpoint",
    })
    return df


# ----------------------------- Enrichment ------------------------------

def _attach_substation_info(
    candidates: pd.DataFrame,
    grid_df: pd.DataFrame,
) -> pd.DataFrame:
    """Nearest substation match per candidate (within 25 km), with DSO & tier."""
    if candidates.empty:
        return candidates
    sub = grid_df.dropna(subset=["latitude", "longitude"]).copy()
    sub["_sub_idx"] = np.arange(len(sub))
    sub_tree = _haversine_tree(sub[["latitude", "longitude"]].to_numpy())
    cand_coords = np.radians(candidates[["latitude", "longitude"]].to_numpy())
    max_rad = MAX_SUBSTATION_SEARCH_RADIUS_KM / EARTH_KM
    dists, idxs = sub_tree.query(cand_coords, k=1)
    dist_km = dists[:, 0] * EARTH_KM
    matched_idx = idxs[:, 0]

    out = candidates.copy().reset_index(drop=True)
    out["connection_distance_km"] = dist_km
    within = dist_km <= MAX_SUBSTATION_SEARCH_RADIUS_KM
    out["nearest_substation_id"] = np.where(
        within,
        sub.iloc[matched_idx]["substation_name"].values,
        "NO_SUBSTATION",
    )
    out["available_capacity_mw"] = np.where(
        within,
        sub.iloc[matched_idx]["available_capacity_mw"].values,
        0.0,
    )
    out["distributor_network"] = np.where(
        within,
        sub.iloc[matched_idx]["distributor_network"].values,
        # Fall back to nearest DSO without radius cap (so File_3 has a DSO)
        sub.iloc[matched_idx]["distributor_network"].values,
    )
    out["grid_status"] = [_classify_grid_status(mw) for mw in out["available_capacity_mw"]]
    out["connection_tier"] = [_classify_connection_tier(d) for d in out["connection_distance_km"]]
    out["grid_eligible"] = within & (out["available_capacity_mw"] >= 0.6)
    return out


def _attach_demand_metadata(
    candidates: pd.DataFrame,
    demand_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach is_tent / tent_tier / nearest segment_id using route_segment match."""
    if candidates.empty:
        return candidates
    demand_lookup = (
        demand_df.groupby("route_segment").agg(
            is_tent=("is_tent", "max"),
            tent_tier=("tent_tier", "first"),
        ).reset_index()
    )
    out = candidates.merge(demand_lookup, on="route_segment", how="left")
    out["is_tent"] = out["is_tent"].fillna(False).astype(bool)
    out["tent_tier"] = out["tent_tier"].fillna("none").astype(str)
    return out


def _deduplicate(candidates: pd.DataFrame, radius_km: float = 2.0) -> pd.DataFrame:
    """Keep one candidate per radius_km cluster, preferring lower-priority source."""
    if candidates.empty:
        return candidates
    df = candidates.copy().reset_index(drop=True)
    df["_priority"] = df["source"].map(SOURCE_PRIORITY).fillna(99).astype(int)
    coords = df[["latitude", "longitude"]].to_numpy()
    tree = _haversine_tree(coords)
    rad = radius_km / EARTH_KM
    kept = np.zeros(len(df), dtype=bool)
    visited = np.zeros(len(df), dtype=bool)
    order = df.sort_values("_priority").index.tolist()
    for i in order:
        if visited[i]:
            continue
        kept[i] = True
        close = tree.query_radius(np.radians(coords[i : i + 1]), r=rad)[0]
        visited[close] = True
    out = df[kept].drop(columns=["_priority"]).reset_index(drop=True)
    return out


def _assign_candidate_ids(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy().reset_index(drop=True)
    out["candidate_id"] = [f"CAND_{i:05d}" for i in range(1, len(out) + 1)]
    # Add cost columns
    out["fixed_cost_eur"] = out["source"].map(FIXED_COST_BY_SOURCE).astype(float)
    out["variable_cost_per_charger_eur"] = VARIABLE_COST_PER_CHARGER
    return out


def _attach_along_route_km(
    candidates: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Project each candidate onto its Carretera's merged geometry and store
    the along-route position (km) as `candidate_km`. Required for precise
    AFIR sub-gap coverage in the MIP."""
    if candidates.empty or roads_gdf.empty:
        out = candidates.copy()
        out["candidate_km"] = np.nan
        return out

    g = roads_gdf.to_crs("EPSG:25830") if str(roads_gdf.crs) != "EPSG:25830" else roads_gdf.copy()
    merged = g.dissolve(by="Carretera").reset_index()[["Carretera", "geometry"]]
    merged_map = {str(r["Carretera"]): r["geometry"] for _, r in merged.iterrows()}

    cand = candidates.copy().reset_index(drop=True)
    pts_ll = gpd.GeoSeries(
        gpd.points_from_xy(cand["longitude"], cand["latitude"]),
        crs="EPSG:4326",
    )
    pts_utm = pts_ll.to_crs("EPSG:25830")
    km = np.full(len(cand), np.nan)
    for i, row in cand.iterrows():
        geom = merged_map.get(str(row["route_segment"]))
        if geom is None:
            continue
        try:
            km[i] = geom.project(pts_utm.iloc[i]) / 1000.0
        except Exception:
            pass
    cand["candidate_km"] = km
    return cand


def _build_coverage_matrix(
    candidates: pd.DataFrame,
    demand_df: pd.DataFrame,
    catchment_km: float = 30.0,
) -> pd.DataFrame:
    """Sparse (candidate_id, segment_id, weight=1) pairs: same route + within haversine radius.

    Haversine is used as a coarse proxy for same-corridor coverage. A candidate
    on route X serves every segment of route X whose centroid is within
    catchment_km as-the-crow-flies.
    """
    if candidates.empty or demand_df.empty:
        return pd.DataFrame(columns=["candidate_id", "segment_id", "weight"])

    # We need segment centroids; join demand_df to road_segments_with_imd for PKs
    # For simplicity, use route_segment match only (same road) + an along-road
    # distance proxy via haversine to candidate lat/lon.
    # Since demand_df does not carry lat/lon, compute from interurban_roads.parquet?
    # For robustness, fallback: candidate covers all segments on same route_segment.
    pairs = candidates[["candidate_id", "route_segment"]].merge(
        demand_df[["segment_id", "route_segment"]],
        on="route_segment",
        how="inner",
    )
    pairs["weight"] = 1.0
    return pairs[["candidate_id", "segment_id", "weight"]]


def _build_coverage_with_geometry(
    candidates: pd.DataFrame,
    demand_df: pd.DataFrame,
    roads_gdf: gpd.GeoDataFrame,
    catchment_km: float = 30.0,
) -> pd.DataFrame:
    """Same-route haversine coverage: candidate covers segment if both sit on the
    same route AND the haversine distance from the candidate to the segment
    centroid is within catchment_km.

    Haversine is used here as a coarse proxy for along-route distance. The
    route-match prefilter (route_segment == Carretera) ensures we never count
    a cross-corridor pairing; the catchment radius captures "same corridor
    and reasonably close" without depending on PK milestone alignment (which
    is inconsistent across the road dataset).
    """
    if candidates.empty or demand_df.empty:
        return pd.DataFrame(columns=["candidate_id", "segment_id", "weight"])
    if roads_gdf.empty or "geometry" not in roads_gdf.columns:
        return _build_coverage_matrix(candidates, demand_df, catchment_km)

    # Segment centroids in WGS84 (lat, lon)
    g_ll = roads_gdf.copy()
    if str(g_ll.crs) != "EPSG:4326":
        g_ll = g_ll.to_crs("EPSG:4326")
    seg_cent = gpd.GeoSeries(
        [geom.centroid for geom in g_ll.geometry], crs="EPSG:4326"
    )
    seg_df = pd.DataFrame({
        "segment_id": g_ll["segment_id"].values,
        "Carretera": g_ll["Carretera"].astype(str).values,
        "seg_lat": seg_cent.y.values,
        "seg_lon": seg_cent.x.values,
    })

    cand = candidates[["candidate_id", "route_segment", "latitude", "longitude"]].copy()
    cand["route_segment"] = cand["route_segment"].astype(str)
    merged = cand.merge(
        seg_df, left_on="route_segment", right_on="Carretera", how="inner",
    )
    # Haversine distance in km
    lat1 = np.radians(merged["latitude"].to_numpy())
    lon1 = np.radians(merged["longitude"].to_numpy())
    lat2 = np.radians(merged["seg_lat"].to_numpy())
    lon2 = np.radians(merged["seg_lon"].to_numpy())
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    merged["dist_km"] = 2 * EARTH_KM * np.arcsin(np.sqrt(a))

    within = merged[merged["dist_km"] <= catchment_km]
    result = within[["candidate_id", "segment_id"]].copy()
    result["weight"] = 1.0
    return result.drop_duplicates(["candidate_id", "segment_id"])


# ----------------------------- Orchestration ---------------------------

def build_candidate_set(
    data_dir: Path | str = "data/processed",
    out_dir: Path | str = "data/processed",
    catchment_km: float = 30.0,
    dedupe_km: float = 2.0,
    high_imd_threshold: float = 15000.0,
    high_imd_spacing_km: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    sa = gpd.read_file(data_dir / "service_areas_clean.geojson")
    chargers = pd.read_csv(data_dir / "interurban_chargers_baseline.csv")
    grid = pd.read_csv(data_dir / "grid_consolidated.csv")
    demand = pd.read_csv(data_dir / "demand_per_segment.csv")
    proposed = pd.read_csv(data_dir / "proposed_stations.csv")
    road_segments = pd.read_csv(data_dir / "road_segments_with_imd.csv")
    roads_gdf = gpd.read_parquet(data_dir / "interurban_roads.parquet")

    print(f"Loaded: SA={len(sa)} chargers={len(chargers)} grid={len(grid)} "
          f"demand={len(demand)} roads_seg={len(road_segments)}")

    parts = []
    parts.append(_sa_candidates(sa)); print(f"  service_area: {len(parts[-1])}")
    parts.append(_upgrade_candidates(chargers)); print(f"  upgrade: {len(parts[-1])}")
    parts.append(_grid_friendly_candidates(grid, road_segments, roads_gdf))
    print(f"  grid_friendly: {len(parts[-1])}")
    parts.append(_high_imd_candidates_from_geometry(
        demand, roads_gdf, high_imd_threshold, high_imd_spacing_km,
    ))
    print(f"  high_imd: {len(parts[-1])}")
    parts.append(_gap_midpoint_candidates(proposed))
    print(f"  gap_midpoint: {len(parts[-1])}")

    all_cands = pd.concat([p for p in parts if len(p) > 0], ignore_index=True)
    print(f"Total raw candidates: {len(all_cands)}")

    all_cands = _deduplicate(all_cands, radius_km=dedupe_km)
    print(f"After {dedupe_km} km dedupe: {len(all_cands)}")

    all_cands = _assign_candidate_ids(all_cands)
    all_cands = _attach_substation_info(all_cands, grid)
    all_cands = _attach_demand_metadata(all_cands, demand)
    all_cands = _attach_along_route_km(all_cands, roads_gdf)

    # Coverage matrix
    coverage = _build_coverage_with_geometry(
        all_cands, demand, roads_gdf, catchment_km=catchment_km,
    )
    print(f"Coverage pairs: {len(coverage)}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / "candidates_v2.parquet"
    cov_path = out_dir / "candidate_segment_coverage.parquet"
    all_cands.to_parquet(cand_path, index=False)
    coverage.to_parquet(cov_path, index=False)
    print(f"Saved {cand_path} ({len(all_cands)} rows)")
    print(f"Saved {cov_path} ({len(coverage)} rows)")

    # Summary by source
    print("\nBy source:")
    print(all_cands.groupby("source").agg(
        n=("candidate_id", "count"),
        grid_eligible=("grid_eligible", "sum"),
        avg_conn_km=("connection_distance_km", "mean"),
    ).round(2))
    print("\nGrid status distribution:")
    print(all_cands["grid_status"].value_counts())
    print("\nDSO distribution:")
    print(all_cands["distributor_network"].value_counts())

    return all_cands, coverage


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate candidate set for v2 MIP")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--catchment-km", type=float, default=30.0)
    ap.add_argument("--dedupe-km", type=float, default=2.0)
    ap.add_argument("--high-imd-threshold", type=float, default=15000.0)
    ap.add_argument("--high-imd-spacing-km", type=float, default=30.0)
    args = ap.parse_args()
    build_candidate_set(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        catchment_km=args.catchment_km,
        dedupe_km=args.dedupe_km,
        high_imd_threshold=args.high_imd_threshold,
        high_imd_spacing_km=args.high_imd_spacing_km,
    )


if __name__ == "__main__":
    main()
