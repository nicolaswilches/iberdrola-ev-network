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
# City node coordinates (from spanish_network.py)
# ---------------------------------------------------------------------------
_CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "MAD": (40.416, -3.703), "BCN": (41.385, 2.173),
    "VAL": (39.470, -0.376), "SEV": (37.389, -5.984),
    "BIL": (43.263, -2.935), "ZAR": (41.649, -0.887),
    "MAL": (36.720, -4.420), "MUR": (37.983, -1.130),
    "VLD": (41.652, -4.724), "ALI": (38.345, -0.490),
    "GRN": (37.177, -3.599), "COR": (37.888, -4.780),
    "BUR": (42.344, -3.697), "PMP": (42.820, -1.644),
    "SSB": (43.321, -1.980), "VIT": (42.849, -2.672),
    "LLE": (41.617, 0.620),  "TAR": (41.119, 1.245),
    "CAS": (39.987, -0.050), "ALB": (38.995, -1.856),
    "TER": (40.345, -1.107), "ACO": (43.362, -8.412),
    "SCQ": (42.880, -8.545), "VIG": (42.232, -8.712),
    "JAE": (37.779, -3.790), "BAD": (38.876, -6.970),
    "MER": (38.917, -6.342), "HUE": (37.261, -6.949),
    "AVL": (40.657, -4.700), "TAL": (39.762, -4.829),
    "PAL": (42.010, -4.527), "STD": (43.463, -3.800),
    "LOG": (42.466, -2.443),
}

# Road corridors: road_name -> ordered city nodes
_ROAD_CORRIDORS: Dict[str, List[str]] = {
    "AP-2": ["MAD", "ZAR", "LLE", "TAR", "BCN"],
    "A-2": ["MAD", "ZAR", "LLE", "TAR", "BCN"],
    "AP-7": ["BCN", "TAR", "CAS", "VAL", "ALI", "MUR"],
    "A-7": ["BCN", "TAR", "CAS", "VAL", "ALI", "MUR"],
    "A-7S": ["MAL", "ALI", "MUR"],
    "A-3": ["MAD", "ALB", "VAL"],
    "AP-4": ["MAD", "COR", "SEV"],
    "A-4": ["MAD", "COR", "SEV"],
    "A-1": ["MAD", "BUR", "VIT", "BIL"],
    "AP-1": ["MAD", "BUR", "VIT", "BIL"],
    "A-6": ["MAD", "VLD"],
    "AP-68": ["ZAR", "LOG", "VIT", "BIL"],
    "A-68": ["ZAR", "LOG", "VIT", "BIL"],
    "A-8": ["BIL", "SSB", "PMP"],
    "A-66": ["SEV", "MER", "VLD"],
    "AP-46": ["MAL", "GRN"],
    "A-23": ["ZAR", "TER", "VAL"],
    "AP-9": ["ACO", "SCQ", "VIG"],
    "N-322": ["ALB", "JAE", "COR"],
    "N-433": ["SEV", "MER", "BAD"],
    "N-435": ["HUE", "SEV"],
    "N-502": ["AVL", "TAL", "COR"],
    "N-621": ["PAL", "STD"],
    "A-67": ["PAL", "STD"],
    "A-62": ["MAD", "VLD", "BUR"],
    "N-630": ["SEV", "MER", "VLD"],
    "A-45": ["COR", "MAL"],
    "A-92": ["SEV", "GRN", "MUR"],
    "N-340": ["ALI", "MAL"],
    "AP-8": ["BIL", "SSB"],
    "AP-6": ["MAD", "VLD"],
    "AP-15": ["PMP", "ZAR"],
}

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
_CORRIDOR_FAMILIES: Dict[str, List[str]] = {
    "AP-2":  ["AP-2", "A-2"],
    "A-2":   ["A-2", "AP-2", "N-2", "N-2A", "N-2R"],
    "AP-7":  ["AP-7", "A-7", "AP-7N", "AP-7R", "AP-7S"],
    "A-7":   ["A-7", "AP-7", "AP-7N", "AP-7R", "AP-7S"],
    "A-7S":  ["A-7S", "AP-7S", "A-7", "N-340"],
    "A-3":   ["A-3"],
    "AP-4":  ["AP-4", "A-4", "AP-4A"],
    "A-4":   ["A-4", "AP-4", "A-4A", "A-4R1", "A-4R2", "N-4", "N-4A"],
    "A-1":   ["A-1", "AP-1", "A-1A", "N-1", "N-1A", "N-1R", "A-8", "AP-8"],
    "AP-1":  ["AP-1", "A-1", "N-1", "A-8", "AP-8"],
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
}

_SEGMENT_KM: Dict[Tuple[str, str], float] = {
    ("MAD", "ZAR"): 310, ("ZAR", "MAD"): 310,
    ("ZAR", "LLE"): 155, ("LLE", "ZAR"): 155,
    ("LLE", "TAR"): 95, ("TAR", "LLE"): 95,
    ("TAR", "BCN"): 95, ("BCN", "TAR"): 95,
    ("TAR", "CAS"): 150, ("CAS", "TAR"): 150,
    ("CAS", "VAL"): 75, ("VAL", "CAS"): 75,
    ("VAL", "ALI"): 165, ("ALI", "VAL"): 165,
    ("ALI", "MUR"): 85, ("MUR", "ALI"): 85,
    ("MAL", "ALI"): 320, ("ALI", "MAL"): 320,
    ("MAD", "ALB"): 250, ("ALB", "MAD"): 250,
    ("ALB", "VAL"): 190, ("VAL", "ALB"): 190,
    ("MAD", "COR"): 400, ("COR", "MAD"): 400,
    ("COR", "SEV"): 140, ("SEV", "COR"): 140,
    ("MAD", "BUR"): 240, ("BUR", "MAD"): 240,
    ("BUR", "VIT"): 110, ("VIT", "BUR"): 110,
    ("VIT", "BIL"): 60, ("BIL", "VIT"): 60,
    ("MAD", "VLD"): 190, ("VLD", "MAD"): 190,
    ("ZAR", "LOG"): 170, ("LOG", "ZAR"): 170,
    ("LOG", "VIT"): 95, ("VIT", "LOG"): 95,
    ("BIL", "SSB"): 95, ("SSB", "BIL"): 95,
    ("SSB", "PMP"): 80, ("PMP", "SSB"): 80,
    ("SEV", "MER"): 195, ("MER", "SEV"): 195,
    ("MER", "VLD"): 400, ("VLD", "MER"): 400,
    ("MAL", "GRN"): 130, ("GRN", "MAL"): 130,
    ("ZAR", "TER"): 185, ("TER", "ZAR"): 185,
    ("TER", "VAL"): 145, ("VAL", "TER"): 145,
    ("ACO", "SCQ"): 65, ("SCQ", "ACO"): 65,
    ("SCQ", "VIG"): 90, ("VIG", "SCQ"): 90,
    ("ALB", "JAE"): 200, ("JAE", "ALB"): 200,
    ("JAE", "COR"): 115, ("COR", "JAE"): 115,
    ("MER", "BAD"): 65, ("BAD", "MER"): 65,
    ("HUE", "SEV"): 92, ("SEV", "HUE"): 92,
    ("AVL", "TAL"): 135, ("TAL", "AVL"): 135,
    ("TAL", "COR"): 250, ("COR", "TAL"): 250,
    ("PAL", "STD"): 195, ("STD", "PAL"): 195,
    ("VLD", "BUR"): 120, ("BUR", "VLD"): 120,
    ("COR", "MAL"): 185, ("MAL", "COR"): 185,
    ("SEV", "GRN"): 255, ("GRN", "SEV"): 255,
    ("GRN", "MUR"): 225, ("MUR", "GRN"): 225,
    ("ALI", "MAL"): 320, ("MAL", "ALI"): 320,
    ("ZAR", "PMP"): 170, ("PMP", "ZAR"): 170,
    ("PAL", "BUR"): 95, ("BUR", "PAL"): 95,
}

_BATTERY_KWH = 55.0


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
    """Simplify a LineString and return [[lon, lat], ...] with 4 decimal precision."""
    simplified = line.simplify(tolerance, preserve_topology=True)
    return [[round(x, 4), round(y, 4)] for x, y in simplified.coords]


def _merge_family(
    full_geoms: Dict[str, object],
    family: List[str],
    city_chain: List[str],
) -> Optional[LineString]:
    """
    Merge all parquet geometries for a corridor family into a single ordered
    LineString. Handles LineString, MultiLineString, and GeometryCollection.

    When linemerge produces a MultiLineString (disconnected components), picks
    the component that passes closest to the most cities in the chain — this
    is the one that actually represents the corridor route.
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
        return None

    union = unary_union(geoms)
    try:
        merged = linemerge(union)
    except Exception:
        merged = union

    if merged.geom_type == "LineString":
        return merged

    # MultiLineString: pick the component nearest to the most chain cities
    components: List[LineString] = []
    if merged.geom_type == "MultiLineString":
        components = list(merged.geoms)
    elif merged.geom_type == "GeometryCollection":
        components = [g for g in merged.geoms if g.geom_type == "LineString"]
    if not components:
        return None

    chain_pts = [
        Point(_CITY_COORDS[c][1], _CITY_COORDS[c][0])
        for c in city_chain if c in _CITY_COORDS
    ]

    def score(comp: LineString) -> Tuple[int, float]:
        # Count cities within ~30 km (0.27 deg) of this component; tie-break by length
        close = sum(1 for pt in chain_pts if comp.distance(pt) < 0.27)
        return (close, comp.length)

    return max(components, key=score)


_AUTO_BUFFER_KM = 25.0
_AUTO_UTM_EPSG = 25830  # Spain


def _build_real_corridor_polylines(
    full_geoms: Dict[str, object],
    display_lines: Dict[str, LineString],
) -> Tuple[List[dict], Dict[str, LineString]]:
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
      - corridor_lines: {road_name: extended+oriented LineString} for agent routing
    """
    corridors: List[dict] = []
    corridor_lines: Dict[str, LineString] = {}
    seen_city_chains: Dict[Tuple[str, ...], str] = {}

    # Hand-curated corridors (city-chain indexed)
    for road_name, city_chain in _ROAD_CORRIDORS.items():
        chain_key = tuple(city_chain)
        rev_key = tuple(reversed(city_chain))

        # Routing line: share across parallel corridors so trips get one line
        if chain_key in seen_city_chains:
            corridor_lines[road_name] = corridor_lines[seen_city_chains[chain_key]]
        elif rev_key in seen_city_chains:
            corridor_lines[road_name] = corridor_lines[seen_city_chains[rev_key]]
        else:
            # Primary: merge all parquet roads in the corridor family to get
            # maximal real-road coverage (curvy geometry, no straight-line
            # connectors). Covers 32/32 corridors to within ~30km of all
            # chain-endpoint cities.
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

        # Display line: every hand-curated road shown (no parallel dedup)
        display_geom = corridor_lines.get(road_name)
        if display_geom is not None and not display_geom.is_empty:
            path = _simplify_line(display_geom, tolerance=0.003)
            if len(path) >= 2:
                corridors.append({"road": road_name, "path": path})

    # Auto-detected corridors: parquet roads not in _ROAD_CORRIDORS with
    # ≥2 cities within 25 km. Mirrors the ABM's _build_road_corridors.
    cities_gdf = gpd.GeoDataFrame(
        {"city": list(_CITY_COORDS.keys())},
        geometry=[Point(lon, lat) for lat, lon in _CITY_COORDS.values()],
        crs="EPSG:4326",
    ).to_crs(epsg=_AUTO_UTM_EPSG)
    city_pts_utm: Dict[str, Point] = dict(zip(cities_gdf["city"], cities_gdf.geometry))
    buffer_m = _AUTO_BUFFER_KM * 1000

    for road_name in sorted(full_geoms.keys()):
        if road_name in _ROAD_CORRIDORS:
            continue
        full_geom = full_geoms[road_name]
        display_geom = display_lines.get(road_name)
        if display_geom is None:
            continue

        full_utm = gpd.GeoSeries([full_geom], crs="EPSG:4326").to_crs(
            epsg=_AUTO_UTM_EPSG
        ).iloc[0]

        nearby: List[Tuple[float, str]] = []
        for city, pt_utm in city_pts_utm.items():
            if full_utm.distance(pt_utm) <= buffer_m:
                nearby.append((full_utm.project(pt_utm), city))

        if len(nearby) < 2:
            continue

        nearby.sort(key=lambda x: x[0])
        path = _simplify_line(display_geom, tolerance=0.003)
        if len(path) < 2:
            continue

        # Orient start → first nearby city (bbox-diagonal ordering)
        first_city = nearby[0][1]
        lat_f, lon_f = _CITY_COORDS[first_city]
        start = path[0]
        end = path[-1]
        d_start = (start[0] - lon_f) ** 2 + (start[1] - lat_f) ** 2
        d_end = (end[0] - lon_f) ** 2 + (end[1] - lat_f) ** 2
        if d_start > d_end:
            path = path[::-1]

        corridors.append({"road": road_name, "path": path})

    return corridors, corridor_lines


def _get_corridor_line_for_trip(
    origin: str, destination: str, corridor_lines: Dict[str, LineString]
) -> Optional[Tuple[str, LineString, bool]]:
    """
    Find the best corridor LineString for a trip from origin to destination.
    Returns (road_name, line, reversed) or None.
    """
    origin_city = origin.split("_")[0] if "_" in origin else origin
    dest_city = destination.split("_")[0] if "_" in destination else destination

    if origin_city not in _CITY_COORDS or dest_city not in _CITY_COORDS:
        return None
    if origin_city == dest_city:
        return None

    best = None
    best_dist = float("inf")

    for road_name, city_chain in _ROAD_CORRIDORS.items():
        if origin_city in city_chain and dest_city in city_chain:
            i = city_chain.index(origin_city)
            j = city_chain.index(dest_city)
            if i < j:
                seg = city_chain[i:j + 1]
            else:
                seg = city_chain[j:i + 1][::-1]
            dist = sum(_segment_distance(seg[k], seg[k + 1]) for k in range(len(seg) - 1))
            if dist < best_dist and road_name in corridor_lines:
                best_dist = dist
                is_reversed = i > j
                best = (road_name, corridor_lines[road_name], is_reversed)

    # Multi-hop via Madrid
    if best is None and origin_city != "MAD" and dest_city != "MAD":
        leg1 = _get_corridor_line_for_trip(origin_city, "MAD", corridor_lines)
        leg2 = _get_corridor_line_for_trip("MAD", dest_city, corridor_lines)
        if leg1 and leg2:
            line1 = leg1[1]
            line2 = leg2[1]
            if leg1[2]:
                line1 = LineString(list(line1.coords)[::-1])
            if leg2[2]:
                line2 = LineString(list(line2.coords)[::-1])
            combined = LineString(list(line1.coords) + list(line2.coords))
            return ("multi", combined, False)

    return best


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
    max_agents: int = 2000,
) -> dict:
    """Export trajectories with real road geometry, SOC, charging pauses, and chargers."""
    print("Loading road geometries...")
    full_geoms, display_lines = _load_road_geometries(data_dir)

    print("Loading charger locations...")
    chargers, proposed_stations = _load_charger_locations(data_dir)
    print(f"  {len(chargers)} existing chargers, {len(proposed_stations)} proposed stations")

    trips_df = pd.read_csv(run_dir / "baseline_trip_records.csv")
    charge_df = pd.read_csv(run_dir / "baseline_charge_events.csv")
    completed = trips_df[trips_df["status"] == "completed"].copy()

    print("Building corridor polylines (full ABM set: hand-curated + auto-detected)...")
    corridors, corridor_lines = _build_real_corridor_polylines(full_geoms, display_lines)
    print(f"  {len(corridors)} corridors displayed")

    print("Processing trip data...")
    if len(completed) > max_agents:
        completed = completed.sample(n=max_agents, random_state=42)

    charge_by_agent: Dict[str, List[dict]] = {}
    for agent_id, grp in charge_df.groupby("agent_id"):
        charge_by_agent[agent_id] = grp.sort_values("sim_time_min").to_dict("records")

    trips_json = []
    no_geom_count = 0

    for _, row in completed.iterrows():
        origin = row["origin"]
        destination = row["destination"]
        dep_time = row["departure_time_min"]
        arr_time = row["arrival_time_min"]
        initial_soc_kwh = row["initial_soc_kwh"]
        final_soc_kwh = row["final_soc_kwh"]

        if arr_time <= dep_time:
            continue

        # Find the real corridor geometry for this trip
        result = _get_corridor_line_for_trip(origin, destination, corridor_lines)
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

        trips_json.append({
            "path": final_path,
            "timestamps": final_timestamps,
            "soc": final_soc_values,
            "numStops": int(row["num_charge_stops"]),
        })

    if no_geom_count:
        print(f"  Skipped {no_geom_count} trips with no corridor geometry")

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
        default=Path(__file__).parent.parent.parent / "src" / "new-abm" / "feedback_loop" / "iter_02",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).parent.parent.parent / "data" / "processed",
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "trajectories.json")
    parser.add_argument("--max-agents", type=int, default=2000)
    args = parser.parse_args()

    if not (args.run_dir / "baseline_trip_records.csv").exists():
        print(f"ERROR: No trip records found in {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    export_trajectories(args.run_dir, args.out, args.data_dir, args.max_agents)


if __name__ == "__main__":
    main()
