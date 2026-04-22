"""
Network optimization for EV charging station placement.

Objective: place the minimum number of new stations such that no gap on any
interurban road segment exceeds the AFIR-mandated spacing threshold:
  - TEN-T Core:          60 km  (legal requirement)
  - TEN-T Comprehensive: 100 km
  - General interurban:  120 km

Algorithm: sequential greedy set-cover — each candidate location is scored by
demand × coverage_km. The highest-scoring uncovered candidate is selected,
covered segments are marked, and residual demand is recalculated until all
AFIR gaps are closed.

Distance methodology: all coverage gap detection and station placement use
road-following (linear referencing) distances, not birds-eye haversine.
AFIR spacing rules are defined along the route, so coverage must be measured
the same way.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import linemerge, unary_union, substring

from src.constants import (
    MAX_STATION_SPACING_KM,
    MAX_STATION_SPACING_TENT_CORE_KM,
    MAX_STATION_SPACING_TENT_COMP_KM,
    CHARGING_PROBABILITY,
    AVG_CHARGE_DURATION_HOURS,
    EFFECTIVE_OPERATING_HOURS,
    MIN_CHARGERS_TENT,
    MIN_CHARGERS_STANDARD,
    MAX_CHARGERS_HIGH_TRAFFIC,
    MAX_CHARGERS_STANDARD,
    HIGH_TRAFFIC_IMD_THRESHOLD,
    MIN_EXISTING_CHARGER_POWER_KW,
)


def calculate_chargers_needed(
    daily_ev_traffic: float,
    charging_probability: float = CHARGING_PROBABILITY,
    avg_charge_hours: float = AVG_CHARGE_DURATION_HOURS,
    operating_hours: float = EFFECTIVE_OPERATING_HOURS,
    is_tent: bool = False,
    imd_total: float = 0.0,
) -> int:
    """
    Calculate number of 150 kW chargers needed for a given daily BEV flow.

    Parameters
    ----------
    daily_ev_traffic : float
        Daily BEV count passing through the station location (already scaled
        from IMD × EV_penetration × BEV_fraction by the caller).
    charging_probability : float
        Fraction of passing BEVs that stop to charge (default 0.12, B1).
    avg_charge_hours : float
        Average session duration in hours (default 0.37 = 22 min, B2).
    operating_hours : float
        Effective daily availability (default 20 hrs, B3).
    is_tent : bool
        True if location is on a TEN-T corridor (enforces 4-charger minimum).
    imd_total : float
        Raw IMD total used to select high-traffic vs standard charger cap.

    Returns
    -------
    int
        Number of chargers, clamped to [min_chargers, max_chargers].
    """
    daily_demand_hours = daily_ev_traffic * charging_probability * avg_charge_hours
    n_chargers = int(np.ceil(daily_demand_hours / operating_hours))

    min_c = MIN_CHARGERS_TENT if is_tent else MIN_CHARGERS_STANDARD
    max_c = (
        MAX_CHARGERS_HIGH_TRAFFIC
        if imd_total > HIGH_TRAFFIC_IMD_THRESHOLD
        else MAX_CHARGERS_STANDARD
    )
    return max(min_c, min(n_chargers, max_c))


def _get_spacing_threshold(row) -> float:
    """Return the applicable AFIR spacing threshold (km) for a road segment row."""
    tent_tier = str(row.get('tent_tier', '')).lower()
    is_tent = bool(row.get('is_tent', False))
    if tent_tier == 'core':
        return MAX_STATION_SPACING_TENT_CORE_KM    # 60 km
    if tent_tier == 'comprehensive':
        return MAX_STATION_SPACING_TENT_COMP_KM    # 100 km
    # If is_tent is True but tent_tier is missing or unrecognised, default to
    # the stricter Core threshold. AFIR requires a tier for every TEN-T
    # segment, so a blank tier is data drift rather than a legitimate case.
    if is_tent:
        return MAX_STATION_SPACING_TENT_CORE_KM    # 60 km
    return MAX_STATION_SPACING_KM                  # 120 km


def _segment_midpoint(row) -> tuple:
    """Return (lat, lon) midpoint of a road segment from geometry or centroid."""
    geom = row.get('geometry')
    if geom is not None:
        pt = geom.interpolate(0.5, normalized=True)
        return pt.y, pt.x
    # Fall back to None — caller must handle
    return None, None


def compute_coverage_gaps(
    road_segments_df,
    existing_stations_df,
    spacing_km: float = None,
) -> 'gpd.GeoDataFrame':
    """
    Identify coverage gaps on interurban routes using road-following linear referencing.

    For each route (Carretera), merges all segments and evaluates every contiguous
    geometry component independently. Fast chargers (≥50 kW) are projected onto each
    component using shapely linear referencing (.project()), consecutive charger
    positions are walked, and any stretch longer than the AFIR spacing threshold is
    flagged as a coverage gap. Gaps are measured *along the route*, not as birds-eye
    distances — this is the methodologically correct interpretation of AFIR.

    Parameters
    ----------
    road_segments_df : gpd.GeoDataFrame
        Interurban road segments with geometry. Must have: 'Carretera', 'is_tent'.
        Optional: 'tent_tier', 'segment_id'.
    existing_stations_df : pd.DataFrame
        NAP charging stations. Must have: 'latitude', 'longitude', 'max_power_kw'.
    spacing_km : float, optional
        Override AFIR threshold for all routes. If None, uses tiered thresholds
        (60 / 100 / 120 km) per contiguous route component from is_tent / tent_tier.

    Returns
    -------
    gpd.GeoDataFrame
        One row per contiguous uncovered stretch with columns:
        Carretera, route_component_id, component_length_km,
        gap_start_km, gap_end_km, gap_mid_km, gap_length_km,
        gap_mid_lat, gap_mid_lon, is_tent, tent_tier,
        gap_spacing_threshold_km, n_chargers_on_route,
        segment_id (representative nearest segment), geometry (EPSG:4326).
    """
    _EMPTY_COLS = [
        'Carretera', 'route_component_id', 'component_length_km',
        'gap_start_km', 'gap_end_km', 'gap_mid_km', 'gap_length_km',
        'gap_mid_lat', 'gap_mid_lon', 'is_tent', 'tent_tier',
        'gap_spacing_threshold_km', 'n_chargers_on_route', 'segment_id', 'geometry',
    ]

    if not hasattr(road_segments_df, 'geometry') or road_segments_df is None:
        return gpd.GeoDataFrame(columns=_EMPTY_COLS, crs='EPSG:4326')

    # Filter to fast chargers only (C4: ≥50 kW count toward AFIR coverage)
    fast = existing_stations_df[
        existing_stations_df['max_power_kw'] >= MIN_EXISTING_CHARGER_POWER_KW
    ].copy()

    # Reproject roads to UTM for accurate metric distance measurements
    roads_utm = road_segments_df.to_crs('EPSG:25830')

    # Build fast charger GeoDataFrame in UTM
    if len(fast) > 0:
        fast_gdf = gpd.GeoDataFrame(
            fast,
            geometry=gpd.points_from_xy(fast['longitude'], fast['latitude']),
            crs='EPSG:4326',
        ).to_crs('EPSG:25830').reset_index(drop=True)
    else:
        fast_gdf = gpd.GeoDataFrame(geometry=gpd.array.GeometryArray([]), crs='EPSG:25830')

    gap_records = []

    for carretera, road_group in roads_utm.groupby('Carretera'):
        # Merge all segments of this route, then evaluate each contiguous piece.
        try:
            merged = linemerge(unary_union(road_group.geometry.values))
        except Exception:
            merged = unary_union(road_group.geometry.values)

        if merged.geom_type == 'MultiLineString':
            components = sorted(
                list(merged.geoms),
                key=lambda g: (-g.length, round(g.centroid.x, 3), round(g.centroid.y, 3)),
            )
        else:
            components = [merged]

        for component_idx, component_geom in enumerate(components, start=1):
            component_length_km = component_geom.length / 1000
            if component_length_km < 5:
                continue  # skip very short components

            # Use only segments that belong to this contiguous component so we do
            # not apply one component's TEN-T tier or coverage state to another.
            component_segments = road_group[
                road_group.geometry.intersects(component_geom.buffer(1.0))
            ]
            if len(component_segments) == 0:
                component_segments = road_group

            is_tent = bool(component_segments['is_tent'].any())
            if 'tent_tier' in component_segments.columns:
                tier_vals = component_segments['tent_tier'].fillna('none').astype(str).str.lower()
                if (tier_vals == 'core').any():
                    tent_tier = 'core'
                elif (tier_vals == 'comprehensive').any():
                    tent_tier = 'comprehensive'
                elif is_tent:
                    tent_tier = 'core'
                else:
                    tent_tier = 'none'
            else:
                tent_tier = 'core' if is_tent else 'none'

            component_segments = component_segments.copy()
            component_segments['_spacing_threshold_km'] = component_segments.apply(
                _get_spacing_threshold, axis=1
            )
            component_segments['_seg_start_m'] = component_segments.geometry.apply(
                lambda seg: min(
                    component_geom.project(Point(seg.coords[0])),
                    component_geom.project(Point(seg.coords[-1])),
                )
            )
            component_segments['_seg_end_m'] = component_segments.geometry.apply(
                lambda seg: max(
                    component_geom.project(Point(seg.coords[0])),
                    component_geom.project(Point(seg.coords[-1])),
                )
            )

            if spacing_km is not None:
                threshold_km = spacing_km
            elif tent_tier == 'core':
                threshold_km = MAX_STATION_SPACING_TENT_CORE_KM
            elif tent_tier == 'comprehensive':
                threshold_km = MAX_STATION_SPACING_TENT_COMP_KM
            else:
                threshold_km = MAX_STATION_SPACING_KM
            threshold_m = threshold_km * 1000

            if len(fast_gdf) > 0:
                route_buffer = component_geom.buffer(2000)
                nearby = fast_gdf[fast_gdf.geometry.within(route_buffer)].copy()
            else:
                nearby = fast_gdf.iloc[:0].copy()

            if len(nearby) == 0:
                positions = [0.0, component_geom.length]
            else:
                nearby = nearby.copy()
                nearby['along_m'] = nearby.geometry.apply(lambda p: component_geom.project(p))
                nearby = nearby.sort_values('along_m').reset_index(drop=True)
                positions = [0.0] + nearby['along_m'].tolist() + [component_geom.length]

            n_chargers = len(nearby)
            route_component_id = f'{carretera}__component_{component_idx:02d}'

            for i in range(len(positions) - 1):
                interval_segments = component_segments[
                    (component_segments['_seg_end_m'] > positions[i])
                    & (component_segments['_seg_start_m'] < positions[i + 1])
                ]
                if len(interval_segments) > 0:
                    interval_tiers = interval_segments['tent_tier'].fillna('none').astype(str).str.lower()
                    gap_is_tent = bool(interval_segments['is_tent'].any())
                    if spacing_km is not None:
                        interval_threshold_km = spacing_km
                        gap_tent_tier = tent_tier
                    elif (interval_tiers == 'core').any():
                        interval_threshold_km = MAX_STATION_SPACING_TENT_CORE_KM
                        gap_tent_tier = 'core'
                    elif (interval_tiers == 'comprehensive').any():
                        interval_threshold_km = MAX_STATION_SPACING_TENT_COMP_KM
                        gap_tent_tier = 'comprehensive'
                    else:
                        interval_threshold_km = MAX_STATION_SPACING_KM
                        gap_tent_tier = 'none'
                else:
                    interval_threshold_km = threshold_km
                    gap_is_tent = is_tent
                    gap_tent_tier = tent_tier

                gap_m = positions[i + 1] - positions[i]
                if gap_m <= interval_threshold_km * 1000:
                    continue

                try:
                    gap_geom = substring(component_geom, positions[i], positions[i + 1])
                except Exception:
                    gap_geom = None

                gap_mid_km = round(((positions[i] + positions[i + 1]) / 2) / 1000, 2)

                if gap_geom is not None and not gap_geom.is_empty:
                    mid_pt_utm = gap_geom.interpolate(0.5, normalized=True)
                    mid_wgs = gpd.GeoDataFrame(
                        geometry=[mid_pt_utm], crs='EPSG:25830'
                    ).to_crs('EPSG:4326')
                    gap_mid_lon = float(mid_wgs.geometry.iloc[0].x)
                    gap_mid_lat = float(mid_wgs.geometry.iloc[0].y)
                else:
                    gap_mid_lat = None
                    gap_mid_lon = None

                gap_records.append({
                    'Carretera': carretera,
                    'route_component_id': route_component_id,
                    'component_length_km': round(component_length_km, 2),
                    'gap_start_km': round(positions[i] / 1000, 2),
                    'gap_end_km': round(positions[i + 1] / 1000, 2),
                    'gap_mid_km': gap_mid_km,
                    'gap_length_km': round(gap_m / 1000, 2),
                    'gap_mid_lat': gap_mid_lat,
                    'gap_mid_lon': gap_mid_lon,
                    'is_tent': gap_is_tent,
                    'tent_tier': gap_tent_tier,
                    'gap_spacing_threshold_km': interval_threshold_km,
                    'n_chargers_on_route': n_chargers,
                    'segment_id': None,  # filled below
                    'geometry': gap_geom,
                })

    if len(gap_records) == 0:
        return gpd.GeoDataFrame(columns=_EMPTY_COLS, crs='EPSG:4326')

    # Build GeoDataFrame in UTM, convert gap geometries to WGS84
    gaps_gdf = gpd.GeoDataFrame(
        gap_records, geometry='geometry', crs='EPSG:25830'
    ).to_crs('EPSG:4326')

    # Attach representative segment_id (nearest original segment to each gap midpoint)
    valid_mid = gaps_gdf['gap_mid_lat'].notna() & gaps_gdf['gap_mid_lon'].notna()
    if valid_mid.any() and 'segment_id' in road_segments_df.columns:
        mid_pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                gaps_gdf.loc[valid_mid, 'gap_mid_lon'],
                gaps_gdf.loc[valid_mid, 'gap_mid_lat'],
            ),
            crs='EPSG:4326',
            index=gaps_gdf.index[valid_mid],
        ).to_crs('EPSG:25830')

        roads_for_join = road_segments_df[['segment_id', 'geometry']].to_crs('EPSG:25830')
        joined = gpd.sjoin_nearest(
            mid_pts[['geometry']],
            roads_for_join[['segment_id', 'geometry']],
            how='left',
        ).drop_duplicates(keep='first')
        gaps_gdf.loc[valid_mid, 'segment_id'] = joined['segment_id'].values

    return gaps_gdf.reset_index(drop=True)


def place_stations_greedy(
    gap_segments_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    service_areas_gdf=None,
) -> pd.DataFrame:
    """
    Sequential greedy station placement.

    Algorithm (from the sequential investment methodology):
      1. For each uncovered gap segment, generate a candidate location at its midpoint
         (or nearest service area if available — preferred for land/utilities).
      2. Score each candidate: V_i = n_chargers_needed × gap_length_km
         (maximises demand served per station placed).
      3. Select highest-score candidate, mark all same-component gaps within its
         spacing threshold as covered, update residual gaps.
      4. Repeat until no gaps remain.

    Parameters
    ----------
    gap_segments_df : pd.DataFrame
        Output of compute_coverage_gaps(). Must have: 'Carretera', 'is_tent',
        'gap_spacing_threshold_km', 'geometry' or midpoint lat/lon,
        and 'length_km'.
    demand_df : pd.DataFrame
        Output of NB06 demand_per_segment.csv. Must have: 'segment_id',
        'n_chargers_needed', 'daily_bev_traffic_2027'.
    service_areas_gdf : gpd.GeoDataFrame, optional
        Motorway service areas from service_areas_clean.geojson. When a gap
        midpoint is near a service area (<5 km), the service area coordinates
        are preferred as the candidate location.

    Returns
    -------
    pd.DataFrame
        Proposed stations. Core submission columns are:
        location_id, latitude, longitude, route_segment, n_chargers_proposed
        Extra traceability columns are also retained for downstream validation.
    """
    from sklearn.neighbors import BallTree

    if len(gap_segments_df) == 0:
        return pd.DataFrame(columns=[
            'location_id', 'latitude', 'longitude',
            'route_segment', 'n_chargers_proposed'
        ])

    # --- Build demand lookup ---
    demand_lookup = {}
    if demand_df is not None and len(demand_df) > 0:
        for _, row in demand_df.iterrows():
            sid = row.get('segment_id')
            if sid is not None:
                demand_lookup[sid] = {
                    'n_chargers': int(row.get('n_chargers_needed', MIN_CHARGERS_STANDARD)),
                    'bev_flow': float(row.get('daily_bev_traffic_2027', 0)),
                    'is_tent': bool(row.get('is_tent', False)),
                }

    # --- Use pre-computed road-following midpoints from compute_coverage_gaps() ---
    gaps = gap_segments_df.copy()
    if 'gap_mid_lat' in gaps.columns and 'gap_mid_lon' in gaps.columns:
        gaps['_mid_lat'] = gaps['gap_mid_lat']
        gaps['_mid_lon'] = gaps['gap_mid_lon']
    elif hasattr(gaps, 'geometry') and gaps.geometry is not None:
        # Fallback for legacy input without pre-computed midpoints
        try:
            gaps_utm = gaps.to_crs('EPSG:25830')
            centroids = gaps_utm.geometry.centroid.to_crs('EPSG:4326')
            gaps['_mid_lat'] = centroids.y
            gaps['_mid_lon'] = centroids.x
        except Exception:
            gaps['_mid_lat'] = gaps.geometry.centroid.y
            gaps['_mid_lon'] = gaps.geometry.centroid.x
    else:
        gaps['_mid_lat'] = gaps.get('latitude', None)
        gaps['_mid_lon'] = gaps.get('longitude', None)

    gaps = gaps[gaps['_mid_lat'].notna() & gaps['_mid_lon'].notna()].copy()

    # --- Build service area BallTree (optional, for preferred siting) ---
    sa_tree = None
    sa_coords_deg = None
    if service_areas_gdf is not None and len(service_areas_gdf) > 0:
        sa_pts = service_areas_gdf.copy()
        if hasattr(sa_pts, 'geometry'):
            try:
                sa_centroids = sa_pts.to_crs('EPSG:25830').geometry.centroid.to_crs('EPSG:4326')
            except Exception:
                sa_centroids = sa_pts.geometry.centroid
            sa_pts['_sa_lat'] = sa_centroids.y
            sa_pts['_sa_lon'] = sa_centroids.x
        sa_pts = sa_pts[sa_pts['_sa_lat'].notna()].reset_index(drop=True)
        if len(sa_pts) > 0:
            sa_coords_deg = sa_pts[['_sa_lat', '_sa_lon']].values
            sa_tree = BallTree(np.radians(sa_coords_deg), metric='haversine')

    # --- Greedy sequential selection ---
    covered_indices = set()
    stations = []
    loc_counter = 1

    # Build a BallTree of gap segment midpoints for coverage radius checks
    gap_mid_coords = np.radians(gaps[['_mid_lat', '_mid_lon']].values)

    while True:
        remaining_mask = ~gaps.index.isin(covered_indices)
        remaining = gaps[remaining_mask]
        if len(remaining) == 0:
            break

        # Score remaining candidates
        scores = []
        for i, (idx, row) in enumerate(remaining.iterrows()):
            seg_id = row.get('segment_id', idx)
            demand_info = demand_lookup.get(seg_id, {})
            n_chargers = demand_info.get('n_chargers', MIN_CHARGERS_STANDARD)
            is_tent = bool(row.get('is_tent', demand_info.get('is_tent', False)))
            if is_tent:
                n_chargers = max(n_chargers, MIN_CHARGERS_TENT)
            length_km = float(
                row.get('gap_length_km', row.get('length_km', row.get('Longitud', 5000))) or 5000
            )
            if length_km > 1000:
                length_km = length_km / 1000  # Convert m to km if needed
            score = n_chargers * length_km
            scores.append((score, idx, row, n_chargers))

        # Select highest-score candidate
        scores.sort(key=lambda x: x[0], reverse=True)
        _, best_idx, best_row, best_n_chargers = scores[0]

        # Determine station coordinates
        cand_lat = best_row['_mid_lat']
        cand_lon = best_row['_mid_lon']
        spacing_thresh = float(best_row.get('gap_spacing_threshold_km', MAX_STATION_SPACING_KM))

        # Prefer nearby service area if within 5 km
        if sa_tree is not None:
            query = np.radians([[cand_lat, cand_lon]])
            dist_rad, sa_idx = sa_tree.query(query, k=1)
            dist_km = dist_rad[0][0] * 6371
            if dist_km <= 5.0:
                sa_row = sa_coords_deg[sa_idx[0][0]]
                cand_lat, cand_lon = sa_row[0], sa_row[1]

        road_name = str(best_row.get('Carretera', best_row.get('route_segment', 'Unknown')))
        station_pos_km = float(
            best_row.get(
                'gap_mid_km',
                (
                    float(best_row.get('gap_start_km', 0))
                    + float(best_row.get('gap_end_km', 0))
                ) / 2,
            )
        )

        stations.append({
            'location_id': f'STA_{loc_counter:04d}',
            'latitude': round(cand_lat, 6),
            'longitude': round(cand_lon, 6),
            'route_segment': road_name,
            'n_chargers_proposed': best_n_chargers,
            'source_segment_id': best_row.get('segment_id'),
            'route_component_id': best_row.get('route_component_id'),
            'placement_km': round(station_pos_km, 2),
            'gap_spacing_threshold_km': spacing_thresh,
            'source_gap_length_km': float(best_row.get('gap_length_km', np.nan)),
            'tent_tier': best_row.get('tent_tier'),
            'is_tent': bool(best_row.get('is_tent', False)),
        })
        loc_counter += 1

        # --- Road-following coverage marking ---
        # Primary: along-route distance for gaps on the same contiguous component.
        # A station at position P covers all gap stretches [start, end] on
        # the same road/component where any part of the gap is within
        # spacing_thresh km.
        if 'gap_start_km' in gaps.columns and 'gap_end_km' in gaps.columns:
            same_route = gaps['Carretera'] == road_name
            if 'route_component_id' in gaps.columns and pd.notna(best_row.get('route_component_id')):
                same_route &= gaps['route_component_id'] == best_row.get('route_component_id')
            within_reach = (
                same_route
                & (gaps['gap_start_km'] < station_pos_km + spacing_thresh)
                & (gaps['gap_end_km'] > station_pos_km - spacing_thresh)
            )
            covered_indices.update(gaps.index[within_reach].tolist())

        # Secondary: 2 km haversine proximity for cross-route coverage at
        # road intersections (a single interchange may serve two routes).
        # NOTE: must use BallTree's returned index array (second return value),
        # not the sorted-distance positions — those are positional in the sorted
        # result, not indices into gaps.index.
        station_coord = np.radians([[cand_lat, cand_lon]])
        dists_rad, bt_idxs = BallTree(gap_mid_coords, metric='haversine').query(
            station_coord, k=len(gaps)
        )
        dists_km = dists_rad[0] * 6371
        close_bt_positions = bt_idxs[0][np.where(dists_km <= 2.0)[0]]
        covered_indices.update(gaps.index[close_bt_positions].tolist())

        # Always mark the selected gap itself
        covered_indices.add(best_idx)

    return pd.DataFrame(stations)


# =======================================================================
# MIP FORMULATION — v2 optimizer (Core 4 constraints)
# =======================================================================
# Produces an AFIR-compliant, demand-satisfying, grid-feasible, DSO-balanced
# station network by solving a Mixed Integer Program (PuLP + CBC) over an
# enriched candidate set. Coexists with the legacy `place_stations_greedy`
# above so the locked 2026-04-13 submission remains reproducible.
#
# Decision variables
#   x_i ∈ {0,1} — place station at candidate i
#   c_i ∈ ℤ, c_i ∈ [min_c_i, max_c_i] — chargers at i (0 when x_i=0)
#   u_j ∈ ℝ≥0 — unmet demand at segment j (slack)
#
# Objective
#   min  Σ (fixed_i * x_i + 150kW_cost * c_i) + W_UNMET * Σ u_j
#
# Core 4 constraints (in this order):
#   1. AFIR spacing  — each baseline gap covered by ≥1 station
#   2. Demand        — Σ c_i · relevance(i,j) + baseline_j + u_j ≥ demand_j
#   3. Grid          — candidates without a substation within 25 km excluded
#   4. DSO equity    — per-DSO kW share bounded below
# =======================================================================


def _dso_label(name: str) -> str:
    """Map distributor strings to canonical File_3 values."""
    if not isinstance(name, str):
        return "Endesa"
    n = name.strip().lower()
    if "ide" in n or "iberdrola" in n or "i-de" in n or "i_de" in n:
        return "i-DE"
    if "endesa" in n or "distribución" in n:
        return "Endesa"
    if "viesgo" in n:
        return "Viesgo"
    return name


def _fixed_cost_with_connection_penalty(
    base_fixed: float,
    connection_distance_km: float,
) -> float:
    """Add grid-extension surcharge as connection distance rises (D4 tiers)."""
    if pd.isna(connection_distance_km):
        return base_fixed + 1_000_000
    if connection_distance_km <= 5:
        return base_fixed
    if connection_distance_km <= 15:
        return base_fixed + 100_000
    if connection_distance_km <= 25:
        return base_fixed + 300_000  # high-cost extension
    if connection_distance_km <= 50:
        return base_fixed + 1_000_000  # remote greenfield grid build
    return base_fixed + 3_000_000  # very remote (e.g., AP-9 at 98 km)


def _infer_candidate_tent(candidate_row, demand_by_route: dict) -> bool:
    """Candidate inherits TEN-T flag from its route_segment."""
    if "is_tent" in candidate_row and not pd.isna(candidate_row["is_tent"]):
        return bool(candidate_row["is_tent"])
    return bool(demand_by_route.get(str(candidate_row.get("route_segment")), False))


def _candidate_charger_bounds(
    candidate_row,
    imd_by_route: dict,
) -> tuple[int, int]:
    """Min / max chargers per AFIR B4/B5, keyed on TEN-T and traffic."""
    is_tent = bool(candidate_row.get("is_tent", False))
    min_c = MIN_CHARGERS_TENT if is_tent else MIN_CHARGERS_STANDARD
    imd = float(imd_by_route.get(str(candidate_row.get("route_segment")), 0.0))
    max_c = MAX_CHARGERS_HIGH_TRAFFIC if imd >= HIGH_TRAFFIC_IMD_THRESHOLD else MAX_CHARGERS_STANDARD
    return min_c, max_c


def _existing_baseline_per_segment(
    baseline_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    min_power_kw: float = MIN_EXISTING_CHARGER_POWER_KW,
) -> pd.Series:
    """Per-segment baseline charger count (≥50 kW within segment_id)."""
    fast = baseline_df[baseline_df["max_power_kw"] >= min_power_kw].copy()
    # baseline_df has n_connectors, prefer that; fallback to 1 per site
    if "n_connectors" in fast.columns:
        fast["_n"] = fast["n_connectors"].fillna(1).clip(lower=1)
    else:
        fast["_n"] = 1
    by_seg = fast.groupby("segment_id")["_n"].sum()
    return demand_df["segment_id"].map(by_seg).fillna(0.0)


def _build_gap_coverage(
    candidates_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
) -> dict:
    """For each gap, the list of candidate indices that can close it.

    A candidate closes a gap iff it sits on the same route (Carretera) and
    its along-route position is within the gap's AFIR threshold of the
    gap's start/end. For candidates that lack an along-route position
    (projection unavailable), we fall back to same-route membership.
    """
    gap_coverage = {}
    # Normalize keys
    cand = candidates_df.reset_index(drop=True).copy()
    cand["_route_key"] = cand["route_segment"].astype(str).str.strip().str.upper()

    for gi, gap in gaps_df.reset_index(drop=True).iterrows():
        gap_route = str(gap.get("Carretera", "")).strip().upper()
        thresh = float(gap.get("gap_spacing_threshold_km", MAX_STATION_SPACING_KM) or MAX_STATION_SPACING_KM)
        start_km = float(gap.get("gap_start_km", 0) or 0)
        end_km = float(gap.get("gap_end_km", 0) or 0)
        # Candidates on the same route
        same_route = cand[cand["_route_key"] == gap_route]
        # Prefer along-route filtering via candidate_km if present
        if "candidate_km" in same_route.columns and same_route["candidate_km"].notna().any():
            within = (
                (same_route["candidate_km"] >= start_km - thresh)
                & (same_route["candidate_km"] <= end_km + thresh)
            )
            idxs = same_route[within].index.tolist()
        else:
            idxs = same_route.index.tolist()
        gap_coverage[gi] = idxs
    return gap_coverage


def place_stations_mip(
    candidates_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    baseline_chargers_df: pd.DataFrame,
    dso_min_share: dict | None = None,
    unmet_penalty_eur: float = 200_000.0,
    solver_time_limit_s: int = 120,
    msg: bool = False,
) -> dict:
    """Solve the Core 4 MIP and return selected stations + diagnostics.

    Returns dict with:
      stations: DataFrame (location_id, latitude, longitude, route_segment,
                           n_chargers_proposed, source, grid_status, ...)
      unmet:    DataFrame (segment_id, unmet_demand)
      summary:  dict of solver / cost / coverage metrics
    """
    import pulp

    dso_min_share = dso_min_share or {"i-DE": 0.35, "Endesa": 0.30, "Viesgo": 0.10}

    # --- Pre-compute helpers ---------------------------------------------
    cand = candidates_df.reset_index(drop=True).copy()
    cand["dso_canonical"] = cand["distributor_network"].map(_dso_label)
    # Grid filter: exclude candidates >25 km from any substation, EXCEPT
    # gap_midpoint candidates — those are AFIR-required and must remain in the
    # feasible set even if grid connection is costly (they inherit the remote
    # penalty via _fixed_cost_with_connection_penalty below).
    keep = (cand["connection_distance_km"] <= 25.0) | (cand["source"] == "gap_midpoint")
    cand = cand[keep].reset_index(drop=True)

    demand_by_route = demand_df.groupby("route_segment")["is_tent"].max().to_dict()
    imd_by_route = demand_df.groupby("route_segment")["imd_total"].max().fillna(0).to_dict()

    # Charger bounds per candidate
    min_c = []
    max_c = []
    for _, row in cand.iterrows():
        lo, hi = _candidate_charger_bounds(row, imd_by_route)
        min_c.append(lo)
        max_c.append(hi)
    cand["_min_c"] = min_c
    cand["_max_c"] = max_c

    # Fixed cost with connection penalty
    cand["_fixed_cost"] = [
        _fixed_cost_with_connection_penalty(
            float(r.get("fixed_cost_eur", 150_000)),
            float(r.get("connection_distance_km", 25.0)),
        )
        for _, r in cand.iterrows()
    ]

    # Per-segment baseline absorption
    baseline_per_seg = _existing_baseline_per_segment(baseline_chargers_df, demand_df)
    demand_j = demand_df["n_chargers_needed"].to_numpy(dtype=float)
    baseline_j = baseline_per_seg.to_numpy(dtype=float)
    net_demand = np.maximum(demand_j - baseline_j, 0.0)

    # Gap coverage (candidate indices that can close each gap)
    gap_coverage = _build_gap_coverage(cand, gaps_df) if len(gaps_df) else {}

    # Coverage matrix (candidate_id -> list of segment_id)
    cov = coverage_df[coverage_df["candidate_id"].isin(cand["candidate_id"])]
    cand_idx_by_id = {cid: i for i, cid in enumerate(cand["candidate_id"].tolist())}
    seg_idx_by_id = {sid: j for j, sid in enumerate(demand_df["segment_id"].tolist())}
    cov_pairs = [
        (cand_idx_by_id[c], seg_idx_by_id[s], float(w))
        for c, s, w in zip(cov["candidate_id"], cov["segment_id"], cov["weight"])
        if c in cand_idx_by_id and s in seg_idx_by_id
    ]

    n_cand = len(cand)
    n_seg = len(demand_df)
    print(f"MIP setup: {n_cand} candidates (after grid filter), {n_seg} segments, "
          f"{len(gap_coverage)} AFIR gaps, {len(cov_pairs)} coverage pairs")

    # --- Build model -----------------------------------------------------
    prob = pulp.LpProblem("ev_network_v2", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(n_cand)]
    c = [
        pulp.LpVariable(f"c_{i}", lowBound=0, upBound=int(cand.iloc[i]["_max_c"]),
                        cat=pulp.LpInteger)
        for i in range(n_cand)
    ]
    u = [pulp.LpVariable(f"u_{j}", lowBound=0, cat=pulp.LpContinuous)
         for j in range(n_seg)]

    # Objective
    var_cost = 100_000.0  # per-charger cost, F1 midpoint
    prob += (
        pulp.lpSum(float(cand.iloc[i]["_fixed_cost"]) * x[i] for i in range(n_cand))
        + pulp.lpSum(var_cost * c[i] for i in range(n_cand))
        + pulp.lpSum(unmet_penalty_eur * u[j] for j in range(n_seg))
    )

    # Charger bounds linked to x
    for i in range(n_cand):
        lo = int(cand.iloc[i]["_min_c"])
        hi = int(cand.iloc[i]["_max_c"])
        prob += c[i] >= lo * x[i], f"min_c_{i}"
        prob += c[i] <= hi * x[i], f"max_c_{i}"

    # Constraint 1 — AFIR spacing (each gap needs enough stations so no sub-gap
    # exceeds the tier threshold). For a gap of length L and threshold T,
    # minimum stations = ceil(L / T) - 1 (even distribution produces L/(n+1) <= T).
    import math as _math
    for gi, cand_idxs in gap_coverage.items():
        if not cand_idxs:
            print(f"  WARN: gap {gi} has no eligible candidates — infeasible AFIR")
            continue
        gap_row = gaps_df.reset_index(drop=True).iloc[gi]
        L = float(gap_row.get("gap_length_km", 0) or 0)
        T = float(gap_row.get("gap_spacing_threshold_km", MAX_STATION_SPACING_KM) or MAX_STATION_SPACING_KM)
        n_min = max(1, _math.ceil(L / T) - 1) if T > 0 else 1
        prob += pulp.lpSum(x[i] for i in cand_idxs) >= n_min, f"afir_gap_{gi}"

    # Constraint 2 — Demand satisfaction (with slack u_j)
    seg_pairs = {}
    for i, j, w in cov_pairs:
        seg_pairs.setdefault(j, []).append((i, w))
    for j in range(n_seg):
        pairs = seg_pairs.get(j, [])
        if not pairs:
            # No candidate serves this segment — any unmet demand absorbed by u_j
            prob += u[j] >= float(net_demand[j]), f"demand_noncov_{j}"
            continue
        prob += (
            pulp.lpSum(w * c[i] for i, w in pairs) + u[j] >= float(net_demand[j])
        ), f"demand_{j}"

    # Constraint 4 — DSO equity (min share of total kW per DSO)
    dso_groups = {d: [] for d in dso_min_share.keys()}
    for i, row in cand.iterrows():
        dso = row["dso_canonical"]
        if dso in dso_groups:
            dso_groups[dso].append(i)
    total_chargers = pulp.lpSum(c[i] for i in range(n_cand))
    for dso, share in dso_min_share.items():
        members = dso_groups.get(dso, [])
        if not members:
            continue
        prob += (
            pulp.lpSum(c[i] for i in members) >= share * total_chargers
        ), f"dso_min_{dso}"

    # --- Solve -----------------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=solver_time_limit_s)
    status = prob.solve(solver)
    status_str = pulp.LpStatus[status]
    print(f"Solver status: {status_str}")

    # --- Extract result --------------------------------------------------
    selected = []
    for i in range(n_cand):
        xv = pulp.value(x[i]) or 0
        if xv >= 0.5:
            cv = int(round(pulp.value(c[i]) or 0))
            row = cand.iloc[i]
            selected.append({
                "candidate_id": row["candidate_id"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "route_segment": str(row["route_segment"]),
                "n_chargers_proposed": cv,
                "source": str(row["source"]),
                "grid_status": str(row["grid_status"]),
                "distributor_network": row["dso_canonical"],
                "connection_distance_km": float(row["connection_distance_km"]),
                "available_capacity_mw": float(row.get("available_capacity_mw", 0.0)),
                "is_tent": bool(row.get("is_tent", False)),
                "tent_tier": str(row.get("tent_tier", "none")),
                "nearest_substation_id": str(row.get("nearest_substation_id", "")),
                "fixed_cost_eur": float(row["_fixed_cost"]),
            })
    stations = pd.DataFrame(selected)
    if len(stations) > 0:
        stations = stations.sort_values(
            ["distributor_network", "route_segment", "latitude"]
        ).reset_index(drop=True)
        stations["location_id"] = [f"STAV2_{i:04d}" for i in range(1, len(stations) + 1)]
        stations["estimated_demand_kw"] = stations["n_chargers_proposed"] * 150
        # Reorder for readability
        cols = [
            "location_id", "latitude", "longitude", "route_segment",
            "n_chargers_proposed", "grid_status", "distributor_network",
            "source", "connection_distance_km", "available_capacity_mw",
            "is_tent", "tent_tier", "nearest_substation_id",
            "estimated_demand_kw", "fixed_cost_eur", "candidate_id",
        ]
        stations = stations[cols]

    # Unmet demand
    unmet = []
    for j in range(n_seg):
        uv = float(pulp.value(u[j]) or 0)
        if uv > 0.5:
            unmet.append({
                "segment_id": int(demand_df.iloc[j]["segment_id"]),
                "route_segment": str(demand_df.iloc[j]["route_segment"]),
                "unmet_chargers": uv,
                "demand_chargers": float(demand_df.iloc[j]["n_chargers_needed"]),
            })
    unmet_df = pd.DataFrame(unmet)

    total_kw = stations["n_chargers_proposed"].sum() * 150 if len(stations) else 0
    dso_kw = (
        stations.groupby("distributor_network")["n_chargers_proposed"].sum() * 150
        if len(stations) else pd.Series(dtype=float)
    )
    summary = {
        "status": status_str,
        "n_stations": int(len(stations)),
        "n_chargers": int(stations["n_chargers_proposed"].sum() if len(stations) else 0),
        "total_kw": int(total_kw),
        "total_capex_eur": float(pulp.value(prob.objective) or 0),
        "unmet_total_chargers": float(unmet_df["unmet_chargers"].sum() if len(unmet_df) else 0),
        "unmet_segments_count": int(len(unmet_df)),
        "dso_kw_shares": {
            d: float(dso_kw.get(d, 0) / total_kw) if total_kw else 0.0
            for d in dso_min_share
        },
        "afir_gaps_unreachable": [
            gi for gi, idxs in gap_coverage.items() if not idxs
        ],
    }
    return {"stations": stations, "unmet": unmet_df, "summary": summary}

