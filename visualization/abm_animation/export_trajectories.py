"""
Export ABM simulation results as deck.gl-compatible trip trajectories.

Reads the existing trip_records.csv and charge_events.csv from an ABM run,
reconstructs each agent's trajectory using REAL road geometries from the
pipeline's roads_clean.parquet, and outputs a JSON file for deck.gl.

Includes:
  - real road geometry polylines (simplified) for corridor highlighting
  - per-waypoint SOC for battery-level color gradient
  - charging pauses (agent stops at station, resumes after charging)
  - real charger locations from the baseline + proposed stations

Usage:
    python export_trajectories.py [--run-dir path/to/abm/outputs] [--out trajectories.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, nearest_points, unary_union

# ---------------------------------------------------------------------------
# Source-of-truth imports from the ABM itself — prevents drift between the
# simulator and the animation exporter. When the ABM's city list or corridor
# chains change, this file picks up the updates automatically on next run.
# ---------------------------------------------------------------------------
_ABM_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "simulation"
if str(_ABM_ROOT) not in sys.path:
    sys.path.insert(0, str(_ABM_ROOT))

from data_generation.spanish_network import (  # noqa: E402
    _CITY_NODES as _ABM_CITY_NODES,
    _ROAD_CORRIDORS as _ABM_ROAD_CORRIDORS,
    _SEGMENT_KM as _ABM_SEGMENT_KM,
    _ROADS_PARQUET_FILENAME as _ABM_ROADS_PARQUET_FILENAME,
    _build_road_corridors as _abm_build_road_corridors,
)

# Derive export-facing lookups from the ABM's source-of-truth tables.
_CITY_COORDS: Dict[str, Tuple[float, float]] = {
    code: (lat, lon) for code, _name, lat, lon, _kind, _pop in _ABM_CITY_NODES
}
# Hand-curated chains (fast lookup, small dict)
_HAND_CORRIDORS: Dict[str, List[str]] = {
    road: list(chain) for road, chain in _ABM_ROAD_CORRIDORS
}
# _ROAD_CORRIDORS is filled in at runtime by `_init_all_corridors()` once the
# data directory is known: hand-curated ∪ auto-detected, matching what the
# ABM actually routes on. Initialised to hand-curated so any import-time use
# still works.
_ROAD_CORRIDORS: Dict[str, List[str]] = dict(_HAND_CORRIDORS)
_SEGMENT_KM: Dict[Tuple[str, str], float] = dict(_ABM_SEGMENT_KM)


def _init_all_corridors(data_dir: Path) -> None:
    """Populate the module-level `_ROAD_CORRIDORS` with both hand-curated and
    auto-detected chains, so trip geometry lookup succeeds for every corridor
    the ABM could route on."""
    global _ROAD_CORRIDORS
    try:
        full = _abm_build_road_corridors(data_dir / _ABM_ROADS_PARQUET_FILENAME)
    except Exception as exc:
        print(f"  Warning: auto-corridor rebuild failed ({exc}); using hand-curated only")
        full = {}
    if full:
        _ROAD_CORRIDORS = full
    else:
        _ROAD_CORRIDORS = dict(_HAND_CORRIDORS)

# Fallback: which road name in the parquet to use when corridor name is missing
_ROAD_FALLBACKS: Dict[str, str] = {
    "AP-7": "A-7",      # AP-7 not in parquet, but A-7 covers same corridor
    "A-7S": "A-7",      # southern section
    "AP-4": "A-4",
    "AP-1": "A-1",
    "AP-6": "A-6",
    "A-68": "AP-68",
    "N-630": "A-66",    # parallel route
}

# Corridor families: parquet road names whose geometry collectively defines the
# full real-road shape of each hand-curated corridor. Parquet often splits a
# single motorway into several named segments (A-4 + A-4A + A-4R1 + ...), and
# parallel roads (AP-4 ‖ A-4) share nearly the same geometry. Merging the
# family gives us a single line that spans the entire corridor's city chain
# using ONLY real road geometry — no straight-line connectors. This is how
# we achieve curvy, real-road-shaped trip paths in the animation.
# Families list every parquet road that physically carries a leg of the
# corresponding chain. Continuity is enforced downstream by
# `_stitch_chain_fragments` and the Phase-6 jump guard — bad merges are
# rejected per-trip rather than pre-filtered here. Keep this list broad so
# multi-road chains (e.g. N-322 ALB-JAE on N-322, JAE-COR on N-432) always
# have geometry to stitch from.
_CORRIDOR_FAMILIES: Dict[str, List[str]] = {
    "AP-2":  ["AP-2", "A-2"],
    "A-2":   ["A-2", "AP-2", "N-2", "N-2A", "N-2R"],
    "AP-7":  ["AP-7", "A-7", "AP-7N", "AP-7R", "AP-7S"],
    "A-7":   ["A-7", "AP-7", "AP-7N", "AP-7R", "AP-7S"],
    "A-7S":  ["A-7S", "AP-7S", "A-7", "N-340"],
    "A-3":   ["A-3"],
    "AP-4":  ["AP-4", "A-4", "AP-4A"],
    "A-4":   ["A-4", "AP-4", "A-4A", "A-4R1", "A-4R2", "N-4", "N-4A"],
    "A-1":   ["A-1", "AP-1", "AP-68", "A-68"],
    "AP-1":  ["AP-1", "A-1", "AP-68", "A-68"],
    "A-6":   ["A-6", "N-6", "N-6A", "N-603", "A-62", "N-601"],
    "AP-68": ["AP-68", "A-68"],
    "A-68":  ["A-68", "AP-68"],
    "A-8":   ["A-8", "N-634", "N-634A", "N-634R", "A-15", "AP-15", "N-121-A"],
    "A-66":  ["A-66", "A-66R", "N-630", "N-630A", "A-62"],
    "AP-46": ["AP-46", "A-45", "A-44", "A-92"],
    "A-23":  ["A-23"],
    "AP-9":  ["AP-9", "AP-9V", "AP-9F"],
    "N-322": ["N-322", "N-322A", "A-4", "N-432"],
    "N-433": ["N-433", "A-49", "N-630", "N-432", "N-430"],
    "N-435": ["N-435", "N-435A", "A-49", "N-630"],
    "N-502": ["N-502", "N-502A", "A-4"],
    "N-621": ["N-621", "N-621A", "A-67"],
    "A-67":  ["A-67", "N-611", "N-611A", "N-621"],
    "A-62":  ["A-62", "A-1", "N-620", "N-620A"],
    "N-630": ["N-630", "A-66", "N-630A", "A-62"],
    "A-45":  ["A-45", "N-331", "N-331R"],
    "A-92":  ["A-92", "A-92M", "A-92N", "A-92R", "A-91", "A-30", "RM-15"],
    "N-340": ["N-340", "N-340A", "N-340R"],
    "AP-8":  ["AP-8", "A-8"],
    "AP-6":  ["AP-6", "A-6", "N-6", "N-603", "A-62", "N-601"],
    "AP-15": ["AP-15", "AP-15-R", "A-15", "AP-68", "A-68"],
    "A-52":  ["A-52", "N-120", "N-525"],
    "N-120": ["N-120", "A-52", "A-231", "N-232"],
    "A-40":  ["A-40", "N-400", "N-320", "A-3"],
    "N-122": ["N-122", "A-11", "N-232", "AP-68"],
}

_BATTERY_KWH = 55.0

# Trip geometry continuity thresholds (km). Trips whose densified path has
# an inter-waypoint leap above HARD are dropped. Between WARN and HARD we
# only warn — the trip is still usable but deserves triage.
_TRIP_JUMP_KM_WARN = 2.0
_TRIP_JUMP_KM_HARD = 5.0
# Upper bound on the haversine distance between consecutive final-path coords.
# DP simplification collapses near-straight stretches; densification restores
# intermediate points so no segment exceeds this distance.
_PATH_MAX_SEGMENT_KM = 3.0
# Same idea for the display-layer corridor polylines (coarser simplification,
# looser density — the map outlines only need to look smooth, not match the
# trip-path QA thresholds).
_DISPLAY_MAX_SEGMENT_KM = 5.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _segment_distance(n1: str, n2: str) -> float:
    if (n1, n2) in _SEGMENT_KM:
        return _SEGMENT_KM[(n1, n2)]
    c1 = _CITY_COORDS.get(n1)
    c2 = _CITY_COORDS.get(n2)
    if c1 and c2:
        return _haversine_km(c1[0], c1[1], c2[0], c2[1]) * 1.15
    return 100.0


# ---------------------------------------------------------------------------
# Real road geometry loading
# ---------------------------------------------------------------------------

def _load_road_geometries(
    data_dir: Path,
) -> Tuple[Dict[str, object], Dict[str, LineString]]:
    """
    Load road geometries from roads_clean.parquet and merge segments per road.

    Returns two dicts:
      - full_geoms: full merged geometry (may be MultiLineString). Used for
        city-proximity / distance checks so auto-detected corridors match
        the ABM's `_build_road_corridors` logic exactly.
      - display_lines: single LineString per road (longest component if the
        merged geom is a MultiLineString). Used for polyline rendering and
        agent routing, which both need a single ordered line.
    """
    parquet_path = data_dir / "roads_clean.parquet"
    if not parquet_path.exists():
        print(f"WARNING: {parquet_path} not found, falling back to straight lines")
        return {}, {}

    df = pd.read_parquet(parquet_path, engine="fastparquet")
    df["geometry"] = df["geometry"].apply(
        lambda x: shapely.from_wkb(x) if x is not None else None
    )
    df = df.dropna(subset=["geometry"])

    full_geoms: Dict[str, object] = {}
    display_lines: Dict[str, LineString] = {}
    for road_name, grp in df.groupby("Carretera"):
        grp_sorted = grp.sort_values("PK_inicio")
        geoms = [g for g in grp_sorted["geometry"].tolist() if g is not None and not g.is_empty]
        if not geoms:
            continue
        try:
            union = unary_union(geoms)
            if union.geom_type == "LineString":
                full_geoms[road_name] = union
                display_lines[road_name] = union
            elif union.geom_type == "MultiLineString":
                try:
                    merged = linemerge(union)
                except Exception:
                    merged = union
                full_geoms[road_name] = merged
                if merged.geom_type == "LineString":
                    display_lines[road_name] = merged
                elif merged.geom_type == "MultiLineString":
                    display_lines[road_name] = max(merged.geoms, key=lambda g: g.length)
            elif union.geom_type == "GeometryCollection":
                lines = [g for g in union.geoms if g.geom_type == "LineString"]
                if lines:
                    full_geoms[road_name] = union
                    display_lines[road_name] = max(lines, key=lambda g: g.length)
        except Exception as e:
            print(f"  Warning: could not merge {road_name}: {e}")

    print(f"  Loaded real geometries for {len(display_lines)} roads")
    return full_geoms, display_lines


def _extend_line_to_cities(
    line: LineString, city_chain: List[str],
) -> LineString:
    """
    Orient a road geometry along city_chain, first → last. No straight-line
    prepend/append: with _CORRIDOR_FAMILIES merging, every corridor's real
    geometry already reaches within ~30 km of both endpoint cities, and
    `_subsection_of_line` projects origin/destination onto the line for
    accurate trip path extraction. Adding straight connectors reintroduces
    the exact visual artifact we're trying to eliminate.
    """
    if len(city_chain) < 2:
        return line

    first = _CITY_COORDS.get(city_chain[0])
    last = _CITY_COORDS.get(city_chain[-1])
    if not first or not last:
        return line

    first_pt = Point(first[1], first[0])
    coords = list(line.coords)
    if first_pt.distance(Point(coords[0])) > first_pt.distance(Point(coords[-1])):
        coords = coords[::-1]
    return LineString(coords)


def _simplify_line(line: LineString, tolerance: float = 0.005) -> List[List[float]]:
    """Simplify a LineString and return [[lon, lat], ...] with 4 decimal precision.

    After DP simplification, densify any segment longer than
    `_DISPLAY_MAX_SEGMENT_KM` with linearly-interpolated intermediates so
    the rendered corridor polyline doesn't show single long straight leaps
    where DP collapsed a near-straight stretch of real road.
    """
    simplified = line.simplify(tolerance, preserve_topology=True)
    coords = [[round(x, 4), round(y, 4)] for x, y in simplified.coords]
    if len(coords) < 2:
        return coords
    out = [coords[0]]
    for i in range(1, len(coords)):
        a, b = coords[i - 1], coords[i]
        gap = _haversine_km(a[1], a[0], b[1], b[0])
        if gap > _DISPLAY_MAX_SEGMENT_KM:
            n_sub = int(math.ceil(gap / _DISPLAY_MAX_SEGMENT_KM))
            for j in range(1, n_sub):
                f = j / n_sub
                out.append([round(a[0] + f * (b[0] - a[0]), 4),
                            round(a[1] + f * (b[1] - a[1]), 4)])
        out.append(b)
    return out


def _family_components(
    full_geoms: Dict[str, object],
    family: List[str],
) -> List[LineString]:
    """
    Return every LineString component across the family's parquet geometries,
    with linemerge applied first to stitch segments that share an endpoint.
    """
    geoms: List[LineString] = []
    for name in family:
        g = full_geoms.get(name)
        if g is None or g.is_empty:
            continue
        if g.geom_type == "LineString":
            geoms.append(g)
        elif g.geom_type == "MultiLineString":
            geoms.extend(g.geoms)
        elif g.geom_type == "GeometryCollection":
            for part in g.geoms:
                if part.geom_type == "LineString":
                    geoms.append(part)
    if not geoms:
        return []

    union = unary_union(geoms)
    try:
        merged = linemerge(union)
    except Exception:
        merged = union

    if merged.geom_type == "LineString":
        return [merged]
    if merged.geom_type == "MultiLineString":
        return list(merged.geoms)
    if merged.geom_type == "GeometryCollection":
        return [g for g in merged.geoms if g.geom_type == "LineString"]
    return []


def _merge_family(
    full_geoms: Dict[str, object],
    family: List[str],
    city_chain: List[str],
) -> Optional[LineString]:
    """
    Pick the single LineString that best represents the corridor for agent
    routing. Chosen by the number of chain cities it passes within ~30 km of
    (tie-break: length).
    """
    components = _family_components(full_geoms, family)
    if not components:
        return None
    if len(components) == 1:
        return components[0]

    chain_pts = [
        Point(_CITY_COORDS[c][1], _CITY_COORDS[c][0])
        for c in city_chain if c in _CITY_COORDS
    ]

    def score(comp: LineString) -> Tuple[int, float]:
        close = sum(1 for pt in chain_pts if comp.distance(pt) < 0.27)
        return (close, comp.length)

    return max(components, key=score)


def _family_display_components(
    full_geoms: Dict[str, object],
    family: List[str],
    city_chain: List[str],
    min_km: float = 20.0,
) -> List[LineString]:
    """
    Fragments to draw as display polylines for a corridor: every component
    whose bbox passes within ~15 km of ≥1 chain city AND whose length is at
    least `min_km`. Tighter than the previous 30 km / 10 km thresholds —
    prevents disconnected stubs of unrelated roads from padding the corridor
    fragment set (which produced N-322's 4-fragment / N-433's 8-fragment
    display artifacts).
    """
    components = _family_components(full_geoms, family)
    if not components:
        return []

    chain_pts = [
        Point(_CITY_COORDS[c][1], _CITY_COORDS[c][0])
        for c in city_chain if c in _CITY_COORDS
    ]
    if not chain_pts:
        return [c for c in components if c.length * 111.0 >= min_km]

    kept: List[LineString] = []
    for comp in components:
        if comp.length * 111.0 < min_km:
            continue
        if any(comp.distance(pt) < 0.135 for pt in chain_pts):
            kept.append(comp)
    return kept


_AUTO_BUFFER_KM = 25.0
_AUTO_UTM_EPSG = 25830  # Spain


def _build_real_corridor_polylines(
    full_geoms: Dict[str, object],
    display_lines: Dict[str, LineString],
) -> Tuple[List[dict], Dict[str, LineString], Dict[str, List[LineString]]]:
    """
    Build corridor polylines for display AND for agent routing.

    Display list mirrors the ABM's `_build_road_corridors` in
    `spanish_network.py`: every hand-curated road in `_ROAD_CORRIDORS` plus
    every parquet road with ≥2 `_CITY_COORDS` within 25 km. Parallel routes
    (e.g. AP-2 and A-2) both appear — same as the ABM.

    Agent routing still resolves via `_ROAD_CORRIDORS` (the 32 city-chain
    corridors), so parallel roads share a single routing line.

    Returns:
      - corridors: list of {"road": name, "path": [[lon,lat],...]} for deck.gl
      - corridor_lines: {road_name: single representative LineString} — fallback
      - corridor_fragments: {road_name: [LineString, ...]} — every near-chain
        parquet fragment. `_get_corridor_line_for_trip` picks the fragment
        closest to the trip's origin/destination so VAL-ALI doesn't route on
        the Girona-Barcelona fragment of AP-7, etc.
    """
    corridors: List[dict] = []
    corridor_lines: Dict[str, LineString] = {}
    corridor_fragments: Dict[str, List[LineString]] = {}
    seen_city_chains: Dict[Tuple[str, ...], str] = {}

    # Every corridor known to the ABM (hand-curated + auto-detected after
    # `_init_all_corridors` is called) becomes a display polyline + routing line.
    for road_name, city_chain in _ROAD_CORRIDORS.items():
        chain_key = tuple(city_chain)
        rev_key = tuple(reversed(city_chain))

        # Routing line: share across parallel corridors so trips get one line
        if chain_key in seen_city_chains:
            prev = seen_city_chains[chain_key]
            corridor_lines[road_name] = corridor_lines[prev]
            corridor_fragments[road_name] = corridor_fragments.get(prev, [])
        elif rev_key in seen_city_chains:
            prev = seen_city_chains[rev_key]
            corridor_lines[road_name] = corridor_lines[prev]
            corridor_fragments[road_name] = corridor_fragments.get(prev, [])
        else:
            # Primary: merge all parquet roads in the corridor family to get
            # maximal real-road coverage (curvy geometry, no straight-line
            # connectors).
            family = _CORRIDOR_FAMILIES.get(road_name, [road_name])
            geom = _merge_family(full_geoms, family, city_chain)

            # Secondary: single parquet road (display layer)
            if geom is None:
                geom = display_lines.get(road_name)
            # Tertiary: explicit fallback mapping
            if geom is None:
                fallback = _ROAD_FALLBACKS.get(road_name)
                if fallback:
                    geom = display_lines.get(fallback)
            # Last resort: straight-line chain of cities
            if geom is None:
                coords = []
                for city in city_chain:
                    if city in _CITY_COORDS:
                        lat, lon = _CITY_COORDS[city]
                        coords.append((lon, lat))
                if len(coords) >= 2:
                    geom = LineString(coords)
            if geom is None or geom.is_empty:
                continue
            extended = _extend_line_to_cities(geom, city_chain)
            corridor_lines[road_name] = extended
            seen_city_chains[chain_key] = road_name

            # Collect every near-chain fragment for trip-aware routing below
            frags = _family_display_components(full_geoms, family, city_chain)
            if not frags:
                frags = [extended]
            corridor_fragments[road_name] = frags

        # Display polylines: emit every family fragment that passes near a
        # chain city. Prevents "corridor cut off mid-country" when the parquet
        # has disconnected LineStrings for a road (e.g. A-66 has separate
        # Sevilla-Zamora and León fragments that linemerge cannot stitch).
        family = _CORRIDOR_FAMILIES.get(road_name, [road_name])
        display_frags = _family_display_components(full_geoms, family, city_chain)
        if not display_frags:
            fallback_geom = corridor_lines.get(road_name)
            if fallback_geom is not None and not fallback_geom.is_empty:
                display_frags = [fallback_geom]
        for frag in display_frags:
            if frag.is_empty:
                continue
            path = _simplify_line(frag, tolerance=0.003)
            if len(path) >= 2:
                corridors.append({"road": road_name, "path": path})

    return corridors, corridor_lines, corridor_fragments


_DEG_PER_KM_LAT = 1.0 / 111.0


def _pt_distance_km(pt_line: LineString, pt: Point) -> float:
    """Approximate km distance from a Point to a LineString at Spanish latitudes.
    deg→km conversion at 40°N: 1° lat ≈ 111 km, 1° lon ≈ 85 km. We use the
    cheaper ~111 km scalar as a conservative (inflates lon gaps); actual value
    is only used for thresholds, not analytics."""
    return pt_line.distance(pt) * 111.0


def _component_covers_od(
    comp: LineString,
    o_pt: Point,
    d_pt: Point,
    tol_km: float = 5.0,
    min_span_frac: float = 0.5,
) -> bool:
    """Return True iff `comp` is a single LineString whose geometry plausibly
    covers the OD trip: both endpoints project within `tol_km`, AND the
    substring length between the projections is at least `min_span_frac` of
    the haversine OD distance (so a tiny component near both endpoints doesn't
    pass)."""
    if comp.is_empty:
        return False
    if _pt_distance_km(comp, o_pt) > tol_km:
        return False
    if _pt_distance_km(comp, d_pt) > tol_km:
        return False
    frac_o = comp.project(o_pt, normalized=True)
    frac_d = comp.project(d_pt, normalized=True)
    span = abs(frac_d - frac_o) * comp.length * 111.0
    od_km = _haversine_km(o_pt.y, o_pt.x, d_pt.y, d_pt.x)
    return span >= min_span_frac * od_km


def _select_trip_fragment(
    fragments: List[LineString],
    origin_pt: Point,
    dest_pt: Point,
    fallback: LineString,
) -> LineString:
    """Pick the fragment that best covers both origin and destination.

    Preference order:
      1. A single fragment that passes `_component_covers_od` (covers the OD
         span within 5 km tolerance). Among these, the longest.
      2. The fragment minimising max(dist(O), dist(D)) — old heuristic, used
         when no fragment fully covers OD (stitcher / multi-hop will handle).
      3. `fallback` when fragments list is empty.
    """
    if not fragments:
        return fallback
    candidates = [f for f in fragments if not f.is_empty]
    if not candidates:
        return fallback

    covering = [f for f in candidates if _component_covers_od(f, origin_pt, dest_pt)]
    if covering:
        return max(covering, key=lambda f: f.length)

    return min(
        candidates,
        key=lambda f: max(f.distance(origin_pt), f.distance(dest_pt)),
    )


def _subsection_between_points(
    line: LineString, a_pt: Point, b_pt: Point,
) -> Optional[LineString]:
    """Extract the subsection of `line` between projections of a_pt and b_pt,
    oriented a→b. Returns None if the subsection is empty or degenerate."""
    frac_a = line.project(a_pt, normalized=True)
    frac_b = line.project(b_pt, normalized=True)
    if abs(frac_a - frac_b) < 0.001:
        return None
    if frac_a > frac_b:
        frac_a, frac_b = frac_b, frac_a
    sub = shapely.ops.substring(line, frac_a * line.length, frac_b * line.length)
    if sub.is_empty or sub.geom_type != "LineString" or len(sub.coords) < 2:
        return None
    sub_start = Point(sub.coords[0])
    if a_pt.distance(sub_start) > b_pt.distance(sub_start):
        sub = LineString(list(sub.coords)[::-1])
    return sub


_STITCH_GAP_KM_OK = 3.0    # tolerance for a silent straight-line hub bridge
_STITCH_GAP_KM_MAX = 10.0  # beyond this, abort stitch; trip falls through


def _pair_haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return _haversine_km(a[1], a[0], b[1], b[0])


def _stitch_chain_fragments(
    origin_city: str, dest_city: str, city_chain: List[str],
    fragments: List[LineString], fallback: LineString,
) -> Optional[LineString]:
    """Walk the city_chain between origin and destination, picking the best
    fragment for each adjacent-city pair and stitching subsections.

    Continuity-checked: for each pair we require the chosen fragment to pass
    `_component_covers_od` for that sub-pair (within 5 km tolerance). The
    stitch boundary between consecutive substrings must be within
    `_STITCH_GAP_KM_MAX`. Gaps ≤ `_STITCH_GAP_KM_OK` are accepted silently
    (short hub transitions). Returns None on violation so callers fall through
    to multi-hop via MAD or drop the trip.
    """
    if origin_city not in city_chain or dest_city not in city_chain:
        return None
    i = city_chain.index(origin_city)
    j = city_chain.index(dest_city)
    if i == j:
        return None
    forward = i < j
    lo, hi = (i, j) if forward else (j, i)
    candidates = [f for f in fragments if not f.is_empty] or [fallback]

    stitched: List[Tuple[float, float]] = []
    for k in range(lo, hi):
        a_city = city_chain[k]
        b_city = city_chain[k + 1]
        if a_city not in _CITY_COORDS or b_city not in _CITY_COORDS:
            continue
        a_pt = Point(_CITY_COORDS[a_city][1], _CITY_COORDS[a_city][0])
        b_pt = Point(_CITY_COORDS[b_city][1], _CITY_COORDS[b_city][0])

        # Prefer a fragment that covers this adjacent pair end-to-end; fall
        # back to min-max-distance if none qualifies.
        covering = [f for f in candidates if _component_covers_od(f, a_pt, b_pt)]
        frag = (max(covering, key=lambda f: f.length) if covering
                else min(candidates,
                         key=lambda f: max(f.distance(a_pt), f.distance(b_pt))))

        sub = _subsection_between_points(frag, a_pt, b_pt)
        if sub is None:
            return None  # one pair uncovered → abort stitch

        coords = list(sub.coords)
        if not coords:
            return None
        if stitched:
            gap_km = _pair_haversine_km(stitched[-1], coords[0])
            if gap_km > _STITCH_GAP_KM_MAX:
                return None  # geometry discontinuity too large to paper over
            if gap_km > 1e-6 and gap_km > _STITCH_GAP_KM_OK:
                # Medium gap: abort rather than produce a visible jump.
                return None
            if stitched[-1] == coords[0]:
                coords = coords[1:]
        stitched.extend(coords)

    if len(stitched) < 2:
        return None
    line = LineString(stitched)
    if not forward:
        line = LineString(list(line.coords)[::-1])
    return line


def _get_corridor_line_for_trip(
    origin: str, destination: str,
    corridor_lines: Dict[str, LineString],
    corridor_fragments: Dict[str, List[LineString]],
    agent_index: int = 0,
) -> Optional[Tuple[str, LineString, bool]]:
    """
    Find the best corridor LineString for a trip from origin to destination.
    Returns (road_name, line, reversed) or None.

    For each candidate corridor, picks the parquet fragment closest to both
    origin and destination — avoids e.g. routing VAL→ALI through AP-7's
    Girona-Barcelona fragment (best-by-chain-cities but wrong for this trip).

    When multiple corridors tie for the shortest chain distance, the winner
    is chosen by `agent_index % len(tied)` so parallel corridors split their
    agents.
    """
    origin_city = origin.split("_")[0] if "_" in origin else origin
    dest_city = destination.split("_")[0] if "_" in destination else destination

    if origin_city not in _CITY_COORDS or dest_city not in _CITY_COORDS:
        return None
    if origin_city == dest_city:
        return None

    o_lat, o_lon = _CITY_COORDS[origin_city]
    d_lat, d_lon = _CITY_COORDS[dest_city]
    origin_pt = Point(o_lon, o_lat)
    dest_pt = Point(d_lon, d_lat)

    best_dist = float("inf")
    candidates: List[Tuple[str, LineString, bool]] = []

    for road_name, city_chain in _ROAD_CORRIDORS.items():
        if origin_city not in city_chain or dest_city not in city_chain:
            continue
        fallback = corridor_lines.get(road_name)
        if fallback is None:
            continue
        i = city_chain.index(origin_city)
        j = city_chain.index(dest_city)
        is_reversed = i > j

        frag_list = corridor_fragments.get(road_name, [])
        # First choice: a single family component that covers the OD pair
        # end-to-end. Avoids stitching (which introduces boundary gaps) when
        # one real road already reaches both cities.
        covering = [f for f in frag_list if _component_covers_od(f, origin_pt, dest_pt)]

        if covering:
            frag = max(covering, key=lambda f: f.length)
            trip_is_reversed = is_reversed
        elif abs(i - j) >= 2:
            # Multi-city chain: stitch adjacent-pair substrings, Phase-4
            # continuity-checked. On failure, fall back to best single
            # fragment — trip will still be validated by Phase-6 jump guard.
            stitched = _stitch_chain_fragments(
                origin_city, dest_city, city_chain, frag_list, fallback,
            )
            if stitched is not None:
                frag = stitched
                trip_is_reversed = False
            else:
                frag = _select_trip_fragment(frag_list, origin_pt, dest_pt, fallback)
                trip_is_reversed = is_reversed
        else:
            frag = _select_trip_fragment(frag_list, origin_pt, dest_pt, fallback)
            trip_is_reversed = is_reversed

        if i < j:
            seg = city_chain[i:j + 1]
        else:
            seg = city_chain[j:i + 1][::-1]
        dist = sum(_segment_distance(seg[k], seg[k + 1]) for k in range(len(seg) - 1))

        if dist < best_dist - 0.1:
            best_dist = dist
            candidates = [(road_name, frag, trip_is_reversed)]
        elif abs(dist - best_dist) <= 0.1:
            candidates.append((road_name, frag, trip_is_reversed))

    if candidates:
        return candidates[agent_index % len(candidates)]

    # Multi-hop via Madrid — anchor the junction at MAD and gap-check.
    if origin_city != "MAD" and dest_city != "MAD" and "MAD" in _CITY_COORDS:
        leg1 = _get_corridor_line_for_trip(
            origin_city, "MAD", corridor_lines, corridor_fragments, agent_index,
        )
        leg2 = _get_corridor_line_for_trip(
            "MAD", dest_city, corridor_lines, corridor_fragments, agent_index,
        )
        if leg1 and leg2:
            line1 = leg1[1]
            line2 = leg2[1]
            if leg1[2]:
                line1 = LineString(list(line1.coords)[::-1])
            if leg2[2]:
                line2 = LineString(list(line2.coords)[::-1])

            mad_lat, mad_lon = _CITY_COORDS["MAD"]
            mad_xy = (mad_lon, mad_lat)
            l1_end = tuple(line1.coords[-1])
            l2_start = tuple(line2.coords[0])

            # Expect both endpoints near MAD. If so, concat directly; if close
            # enough for a pivot, bridge through MAD; else drop the trip.
            gap_km = _pair_haversine_km(l1_end, l2_start)
            l1_to_mad = _pair_haversine_km(l1_end, mad_xy)
            l2_to_mad = _pair_haversine_km(l2_start, mad_xy)

            if gap_km <= _STITCH_GAP_KM_OK:
                c1 = list(line1.coords)
                c2 = list(line2.coords)
                if c1 and c2 and c1[-1] == c2[0]:
                    c2 = c2[1:]
                return ("multi", LineString(c1 + c2), False)

            if (l1_to_mad <= _STITCH_GAP_KM_OK and l2_to_mad <= _STITCH_GAP_KM_OK
                    and max(l1_to_mad, l2_to_mad) + gap_km < _STITCH_GAP_KM_MAX * 2):
                # Both legs end near MAD but from different directions — pivot
                # through the true MAD coordinate for a clean junction.
                c1 = list(line1.coords)
                c2 = list(line2.coords)
                if c1[-1] != mad_xy:
                    c1.append(mad_xy)
                if c2 and c2[0] == mad_xy:
                    c2 = c2[1:]
                return ("multi", LineString(c1 + c2), False)

            # Legs don't converge at MAD — geometry would jump. Drop trip.

    return None


def _subsection_of_line(
    line: LineString, city_chain: List[str],
    origin: str, destination: str,
) -> LineString:
    """
    Extract the subsection of a corridor line between origin and destination.
    Projects both cities onto the line and extracts the segment between them.
    """
    origin_city = origin.split("_")[0] if "_" in origin else origin
    dest_city = destination.split("_")[0] if "_" in destination else destination

    if origin_city not in _CITY_COORDS or dest_city not in _CITY_COORDS:
        return line

    o_lat, o_lon = _CITY_COORDS[origin_city]
    d_lat, d_lon = _CITY_COORDS[dest_city]
    origin_pt = Point(o_lon, o_lat)
    dest_pt = Point(d_lon, d_lat)

    frac_o = line.project(origin_pt, normalized=True)
    frac_d = line.project(dest_pt, normalized=True)

    if abs(frac_o - frac_d) < 0.01:
        # Projections are too close — the geometry doesn't span these cities.
        # This shouldn't happen with extended lines, but guard against it.
        return line

    if frac_o > frac_d:
        frac_o, frac_d = frac_d, frac_o

    frac_o = max(0.0, frac_o - 0.002)
    frac_d = min(1.0, frac_d + 0.002)

    start_dist = frac_o * line.length
    end_dist = frac_d * line.length

    sub = shapely.ops.substring(line, start_dist, end_dist)
    if sub.is_empty or sub.geom_type != "LineString":
        return line

    # Orient from origin to destination
    sub_start = Point(sub.coords[0])
    if origin_pt.distance(sub_start) > dest_pt.distance(sub_start):
        sub = LineString(list(sub.coords)[::-1])

    return sub


# ---------------------------------------------------------------------------
# Charger locations
# ---------------------------------------------------------------------------

def _densify_trip_path(
    path: List[List[float]],
    timestamps: List[float],
    soc_values: List[float],
    max_segment_km: float,
) -> Tuple[List[List[float]], List[float], List[float]]:
    """Insert linearly-interpolated intermediate points, timestamps, and SOC
    values so no two consecutive path coords are more than `max_segment_km`
    apart. Preserves the original coords — only inserts between them.

    Charging pauses (two consecutive identical coords with differing
    timestamps) are kept exactly as-is: no intermediates are inserted when
    the two coords are equal.
    """
    if len(path) < 2:
        return path, timestamps, soc_values

    out_path = [path[0]]
    out_t = [timestamps[0]]
    out_soc = [soc_values[0]]

    for k in range(1, len(path)):
        a = path[k - 1]
        b = path[k]
        if a == b:
            # charging pause — keep the pair, no densification
            out_path.append(b)
            out_t.append(timestamps[k])
            out_soc.append(soc_values[k])
            continue
        gap_km = _haversine_km(a[1], a[0], b[1], b[0])
        if gap_km <= max_segment_km:
            out_path.append(b)
            out_t.append(timestamps[k])
            out_soc.append(soc_values[k])
            continue
        # Insert n_sub - 1 intermediate points strictly between a and b.
        n_sub = int(math.ceil(gap_km / max_segment_km))
        for i in range(1, n_sub):
            t_frac = i / n_sub
            lon = a[0] + t_frac * (b[0] - a[0])
            lat = a[1] + t_frac * (b[1] - a[1])
            ts = timestamps[k - 1] + t_frac * (timestamps[k] - timestamps[k - 1])
            sc = soc_values[k - 1] + t_frac * (soc_values[k] - soc_values[k - 1])
            out_path.append([round(lon, 5), round(lat, 5)])
            out_t.append(round(ts, 2))
            out_soc.append(round(sc, 3))
        out_path.append(b)
        out_t.append(timestamps[k])
        out_soc.append(soc_values[k])

    return out_path, out_t, out_soc


def _load_charger_locations(data_dir: Path) -> Tuple[List[dict], List[dict]]:
    """Load real charger positions. Returns (existing_chargers, proposed_stations)."""
    chargers = []
    proposed = []

    baseline_path = data_dir / "interurban_chargers_baseline.csv"
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        fast = df[df["max_power_kw"] >= 50].copy()
        for _, row in fast.iterrows():
            chargers.append({
                "position": [round(row["longitude"], 5), round(row["latitude"], 5)],
            })

    proposed_path = data_dir / "proposed_stations.csv"
    if proposed_path.exists():
        df = pd.read_csv(proposed_path)
        for _, row in df.iterrows():
            proposed.append({
                "position": [round(row["longitude"], 5), round(row["latitude"], 5)],
                "id": row["location_id"],
                "road": row["route_segment"],
                "chargers": int(row["n_chargers_proposed"]),
            })

    return chargers, proposed


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export_trajectories(
    run_dir: Path,
    output_path: Path,
    data_dir: Path,
    max_agents: int = 5000,
) -> dict:
    """Export trajectories with real road geometry, SOC, charging pauses, and chargers."""
    print("Initialising corridor dictionary (hand-curated + auto-detected)...")
    _init_all_corridors(data_dir)
    print(f"  {len(_ROAD_CORRIDORS)} corridors known")

    print("Loading road geometries...")
    full_geoms, display_lines = _load_road_geometries(data_dir)

    print("Loading charger locations...")
    chargers, proposed_stations = _load_charger_locations(data_dir)
    print(f"  {len(chargers)} existing chargers, {len(proposed_stations)} proposed stations")

    trips_df = pd.read_csv(run_dir / "baseline_trip_records.csv")
    charge_df = pd.read_csv(run_dir / "baseline_charge_events.csv")
    completed = trips_df[trips_df["status"] == "completed"].copy()

    print("Building corridor polylines (hand-curated ABM-routed roads only)...")
    corridors, corridor_lines, corridor_fragments = _build_real_corridor_polylines(
        full_geoms, display_lines,
    )
    print(f"  {len(corridors)} corridors displayed")

    print("Processing trip data...")
    if len(completed) > max_agents:
        completed = completed.sample(n=max_agents, random_state=42)

    charge_by_agent: Dict[str, List[dict]] = {}
    for agent_id, grp in charge_df.groupby("agent_id"):
        charge_by_agent[agent_id] = grp.sort_values("sim_time_min").to_dict("records")

    trips_json = []
    no_geom_count = 0
    jump_rejects: List[dict] = []
    jump_warnings = 0

    for trip_idx, (_, row) in enumerate(completed.iterrows()):
        origin = row["origin"]
        destination = row["destination"]
        dep_time = row["departure_time_min"]
        arr_time = row["arrival_time_min"]
        initial_soc_kwh = row["initial_soc_kwh"]
        final_soc_kwh = row["final_soc_kwh"]

        if arr_time <= dep_time:
            continue

        # Find the real corridor geometry for this trip. trip_idx threads a
        # round-robin tie-breaker so parallel corridors split their agents.
        result = _get_corridor_line_for_trip(
            origin, destination, corridor_lines, corridor_fragments, trip_idx,
        )
        if result is None:
            no_geom_count += 1
            continue

        road_name, full_line, is_reversed = result

        # Extract subsection for this specific OD pair
        city_chain = _ROAD_CORRIDORS.get(road_name, [])
        if road_name != "multi" and city_chain:
            trip_line = _subsection_of_line(full_line, city_chain, origin, destination)
        else:
            trip_line = full_line

        if is_reversed:
            trip_line = LineString(list(trip_line.coords)[::-1])

        # Simplify for per-trip resolution (less aggressive than corridor display)
        trip_line_simple = trip_line.simplify(0.0015, preserve_topology=True)
        if trip_line_simple.is_empty or len(list(trip_line_simple.coords)) < 2:
            trip_line_simple = trip_line

        # Build the path as [[lon, lat], ...]
        base_path = [[round(x, 5), round(y, 5)] for x, y in trip_line_simple.coords]
        if len(base_path) < 2:
            no_geom_count += 1
            continue

        agent_charges = charge_by_agent.get(row["agent_id"], [])
        total_charge_time = sum(
            c.get("charge_time_min", 0) + c.get("wait_time_min", 0)
            for c in agent_charges
        )
        drive_time = max(1.0, (arr_time - dep_time) - total_charge_time)

        # Compute drive fractions for each charge event
        charge_events_with_frac = []
        for ce in agent_charges:
            ce_time = ce["sim_time_min"]
            drive_elapsed = ce_time - dep_time - sum(
                prev.get("charge_time_min", 0) + prev.get("wait_time_min", 0)
                for prev in agent_charges if prev["sim_time_min"] < ce_time
            )
            frac = max(0.0, min(0.999, drive_elapsed / drive_time))
            charge_events_with_frac.append((frac, ce))
        charge_events_with_frac.sort(key=lambda x: x[0])

        # Helper: interpolate position along the trip line at fraction f
        def pos_at_frac(f):
            f = max(0.0, min(1.0, f))
            pt = trip_line_simple.interpolate(f, normalized=True)
            return [round(pt.x, 5), round(pt.y, 5)]

        final_path = []
        final_timestamps = []
        final_soc_values = []

        current_soc = initial_soc_kwh
        prev_frac = 0.0
        prev_time = dep_time

        num_drive_pts = len(base_path)  # use the actual geometry point count

        for charge_frac, ce in charge_events_with_frac:
            soc_before = ce["soc_before_kwh"]
            soc_after = ce["soc_after_kwh"]
            charge_duration = ce.get("wait_time_min", 0) + ce.get("charge_time_min", 0)

            # Drive segment: use real geometry points within this fraction range
            n_pts = max(3, int((charge_frac - prev_frac) * num_drive_pts))
            for i in range(n_pts):
                seg_f = i / max(n_pts - 1, 1)
                frac = prev_frac + (charge_frac - prev_frac) * seg_f
                t = prev_time + (charge_frac - prev_frac) * drive_time * seg_f
                soc = current_soc + seg_f * (soc_before - current_soc)
                final_path.append(pos_at_frac(frac))
                final_timestamps.append(round(t, 1))
                final_soc_values.append(round(max(0.0, min(1.0, soc / _BATTERY_KWH)), 3))

            # Charging pause
            charge_start = prev_time + (charge_frac - prev_frac) * drive_time
            station_pos = pos_at_frac(charge_frac)
            final_path.append(station_pos)
            final_timestamps.append(round(charge_start, 1))
            final_soc_values.append(round(max(0.0, min(1.0, soc_before / _BATTERY_KWH)), 3))
            final_path.append(list(station_pos))
            final_timestamps.append(round(charge_start + charge_duration, 1))
            final_soc_values.append(round(max(0.0, min(1.0, soc_after / _BATTERY_KWH)), 3))

            prev_frac = charge_frac
            prev_time = charge_start + charge_duration
            current_soc = soc_after

        # Final driving segment
        n_pts = max(3, int((1.0 - prev_frac) * num_drive_pts))
        for i in range(n_pts):
            seg_f = i / max(n_pts - 1, 1)
            frac = prev_frac + (1.0 - prev_frac) * seg_f
            t = prev_time + (1.0 - prev_frac) * drive_time * seg_f
            soc = current_soc + seg_f * (final_soc_kwh - current_soc)
            final_path.append(pos_at_frac(frac))
            final_timestamps.append(round(t, 1))
            final_soc_values.append(round(max(0.0, min(1.0, soc / _BATTERY_KWH)), 3))

        if len(final_path) < 2:
            continue

        # Ensure timestamps are strictly non-decreasing
        for i in range(1, len(final_timestamps)):
            if final_timestamps[i] <= final_timestamps[i - 1]:
                final_timestamps[i] = final_timestamps[i - 1] + 0.1

        # Densify the path so no two consecutive coords are more than
        # _PATH_MAX_SEGMENT_KM apart. DP simplification collapses long near-
        # straight stretches of real road to just their endpoints, which the
        # jump guard would then flag. Linear interpolation is safe because the
        # simplified line only collapses sections within ~167 m of straight.
        final_path, final_timestamps, final_soc_values = _densify_trip_path(
            final_path, final_timestamps, final_soc_values, _PATH_MAX_SEGMENT_KM,
        )

        # Phase-6 jump guard: reject any trip whose densified geometry still
        # has a leap > TRIP_JUMP_KM_HARD. After densification, remaining big
        # gaps are genuine discontinuities (routing bugs), not simplification
        # artifacts. Logs rejected trips to qa_jumps.csv for triage.
        trip_max_jump = 0.0
        trip_jump_idx = -1
        for _k in range(1, len(final_path)):
            pa, pb = final_path[_k - 1], final_path[_k]
            if pa == pb:
                continue
            _dk = _haversine_km(pa[1], pa[0], pb[1], pb[0])
            if _dk > trip_max_jump:
                trip_max_jump = _dk
                trip_jump_idx = _k
        if trip_max_jump > _TRIP_JUMP_KM_HARD:
            jump_rejects.append({
                "agent_id": row["agent_id"],
                "origin": origin,
                "destination": destination,
                "road": road_name,
                "max_jump_km": round(trip_max_jump, 2),
                "jump_idx": trip_jump_idx,
                "n_pts": len(final_path),
            })
            no_geom_count += 1
            continue
        if trip_max_jump > _TRIP_JUMP_KM_WARN:
            jump_warnings += 1

        trips_json.append({
            "path": final_path,
            "timestamps": final_timestamps,
            "soc": final_soc_values,
            "numStops": int(row["num_charge_stops"]),
        })

    if no_geom_count:
        print(f"  Skipped {no_geom_count} trips (no geometry or jump-guard rejects)")
    if jump_warnings:
        print(f"  {jump_warnings} trips have inter-waypoint leaps in [{_TRIP_JUMP_KM_WARN}, "
              f"{_TRIP_JUMP_KM_HARD}] km — kept but flagged")
    if jump_rejects:
        log_path = output_path.parent / "qa_jumps.csv"
        pd.DataFrame(jump_rejects).to_csv(log_path, index=False)
        print(f"  {len(jump_rejects)} trips rejected by jump guard → {log_path}")

    # Metadata
    all_deps = [t["timestamps"][0] for t in trips_json]
    all_arrs = [t["timestamps"][-1] for t in trips_json]
    metadata = {
        "total_trips": len(trips_json),
        "total_chargers": len(chargers),
        "total_proposed": len(proposed_stations),
        "time_range_min": [
            float(min(all_deps)) if all_deps else 300,
            float(max(all_arrs)) if all_arrs else 1440,
        ],
        "source_run": str(run_dir),
        "source_agents": int(trips_df["agent_id"].nunique()),
        "corridor_count": len(corridors),
    }

    output = {
        "trips": trips_json,
        "corridors": corridors,
        "chargers": chargers,
        "proposed": proposed_stations,
        "metadata": metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f)

    size_kb = output_path.stat().st_size / 1024
    print(f"\nExported {len(trips_json)} trips, {len(corridors)} corridors, "
          f"{len(chargers)} chargers, {len(proposed_stations)} proposed stations")
    print(f"Time range: {metadata['time_range_min'][0]:.0f} – {metadata['time_range_min'][1]:.0f} min")
    print(f"Output: {output_path} ({size_kb:.0f} KB)")

    return output


def main():
    parser = argparse.ArgumentParser(description="Export ABM trajectories for deck.gl animation")
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path(__file__).parent.parent.parent / "src" / "simulation" / "feedback_loop" / "iter_02",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).parent.parent.parent / "data" / "processed",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "trajectories.json")
    parser.add_argument("--max-agents", type=int, default=5000)
    args = parser.parse_args()

    if not (args.run_dir / "baseline_trip_records.csv").exists():
        print(f"ERROR: No trip records found in {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    export_trajectories(args.run_dir, args.out, args.data_dir, args.max_agents)


if __name__ == "__main__":
    main()
