"""Municipality-level road attachment helpers and calibration graph builders."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import networkx as nx
import pandas as pd

try:
    from src.constants import (
        BEV_FRACTION,
        EV_PENETRATION_RATE,
        INTERURBAN_PRIVATE_CAR_MODE_SHARE,
        INTERURBAN_PRIVATE_VEHICLE_OCCUPANCY,
    )
except ModuleNotFoundError:
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from src.constants import (
        BEV_FRACTION,
        EV_PENETRATION_RATE,
        INTERURBAN_PRIVATE_CAR_MODE_SHARE,
        INTERURBAN_PRIVATE_VEHICLE_OCCUPANCY,
    )
from data_generation.graph_vocab import (
    EdgeKind,
    NodeKind,
    validate_network,
)
from models.demand import ODMatrix, ODPair
from models.network import RoadEdge, RoadNetwork, RoadNode
from data_generation.spanish_network import (
    _MUNICIPALITY_REFERENCE_RELATIVE_PATH,
    _NON_MAINLAND_PROVINCES,
    _SPEED_KMH,
    _attach_nearest_road_anchor,
    _haversine_km,
    _load_official_municipality_reference,
)

logger = logging.getLogger(__name__)

_RAW_ROADS_RELATIVE_PATH = Path("raw") / "ministry_roads" / "hermes_roads.parquet"
_PROCESSED_ROADS_RELATIVE_PATH = Path("processed") / "interurban_roads.parquet"
_CNIG_BOUNDARIES_RELATIVE_PATH = (
    Path("raw")
    / "additional"
    / "cnig_boundaries"
    / "LIMITES_UNIDADES_ADMINISTRATIVAS.ZIP"
)
_CNIG_MAINLAND_MUNICIPALITY_SHP = (
    "zip://{zip_path}!SHP_ETRS89/"
    "recintos_municipales_inspire_peninbal_etrs89/"
    "recintos_municipales_inspire_peninbal_etrs89.shp"
)
_BOUNDARY_SNAP_MAX_DISTANCE_M = 100.0
_OD_RAW_RELATIVE_PATH = (
    Path("raw")
    / "trips_people_overnight_pyspainmobility"
    / "Pernoctaciones_municipios_2022-01-01_2022-01-06_v2.parquet"
)
_ROAD_FAR_FROM_COVERED_NODE_THRESHOLD_KM = 25.0
_MUNICIPALITY_ENDPOINT_CONNECTOR_KM = 0.01
_MUNICIPALITY_ENDPOINT_CONNECTOR_MIN = 0.1
_JUNCTION_CLUSTER_TOLERANCE_M = 25.0
_SAME_ROAD_GAP_BRIDGE_MIN_KM = 0.01
_NEAR_ROAD_CONNECTION_TOLERANCE_M = 25.0


def _polygon_municipality_code(raw_code: object) -> str:
    digits = "".join(ch for ch in str(raw_code or "").strip() if ch.isdigit())
    if len(digits) < 5:
        return ""
    return digits[-5:]


def _municipality_node_id(code: str) -> str:
    return f"MUNI_{str(code).zfill(5)}"


def _geo_endpoint_node_id(road_name: str, raw_segment_id: int, suffix: str) -> str:
    road_token = str(road_name or "ROAD").replace(" ", "_")
    return f"GEO_{road_token}_{int(raw_segment_id)}_{suffix}"


def _road_anchor_node_id(code: str) -> str:
    return f"ANCHOR_{str(code).zfill(5)}"


def _processed_endpoint_node_id(segment_id: int, suffix: str) -> str:
    return f"SEG_{int(segment_id)}_{suffix}"


def _road_type(road_name: str) -> str:
    token = str(road_name or "").upper()
    if token.startswith("AP-"):
        return "AP"
    if token.startswith("A-"):
        return "A"
    if token.startswith("N-"):
        return "N"
    return "road"


def _base_municipality_code(raw_code: object) -> str:
    digits = "".join(ch for ch in str(raw_code or "").strip() if ch.isdigit())
    return digits[:5] if len(digits) >= 5 else ""


def _load_od_covered_municipality_codes(raw_dir: Path, mainland_only: bool = True) -> Set[str]:
    path = Path(raw_dir) / _OD_RAW_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing OD parquet at {path}")
    od = pd.read_parquet(path, columns=["residence_area", "overnight_stay_area", "people"])
    od["people"] = pd.to_numeric(od["people"], errors="coerce").fillna(0.0)
    od = od[od["people"] > 0.0].copy()
    for col in ("residence_area", "overnight_stay_area"):
        od[col] = od[col].map(_base_municipality_code)
    covered = set(od["residence_area"].unique()) | set(od["overnight_stay_area"].unique())
    covered = {code for code in covered if len(code) == 5}
    if mainland_only:
        covered = {code for code in covered if code[:2] not in _NON_MAINLAND_PROVINCES}
    return covered


def _load_raw_roads(raw_dir: Path):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to build the municipality road graph") from exc

    path = Path(raw_dir) / _RAW_ROADS_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing raw Hermes roads at {path}")
    roads = gpd.read_parquet(path).copy()
    roads["raw_segment_id"] = np.arange(len(roads), dtype=int)
    roads["length_km"] = pd.to_numeric(roads["Longitud"], errors="coerce") / 1000.0
    roads["road_name"] = roads["Carretera"].fillna("").astype(str)
    roads["road_type"] = roads["road_name"].map(_road_type)
    return roads


def _load_processed_roads(data_dir: Path):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to build the municipality calibration graph") from exc

    path = Path(data_dir) / _PROCESSED_ROADS_RELATIVE_PATH.name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed roads at {path}")
    roads = gpd.read_parquet(path).copy()
    roads["raw_segment_id"] = roads["segment_id"].astype(int)
    roads["length_km"] = pd.to_numeric(roads.get("length_km"), errors="coerce")
    missing_length = roads["length_km"].isna()
    if missing_length.any() and "Longitud" in roads.columns:
        roads.loc[missing_length, "length_km"] = (
            pd.to_numeric(roads.loc[missing_length, "Longitud"], errors="coerce") / 1000.0
        )
    roads["road_name"] = roads["Carretera"].fillna("").astype(str)
    roads["road_type"] = roads["road_name"].map(_road_type)
    return roads


def _load_mainland_municipality_polygons(raw_dir: Path):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to build the municipality road graph") from exc

    zip_path = Path(raw_dir) / _CNIG_BOUNDARIES_RELATIVE_PATH
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing CNIG municipality boundaries at {zip_path}")
    shp_path = _CNIG_MAINLAND_MUNICIPALITY_SHP.format(zip_path=zip_path)
    polygons = gpd.read_file(shp_path)[["NATCODE", "NAMEUNIT", "geometry"]].copy()
    polygons["municipality_code"] = polygons["NATCODE"].map(_polygon_municipality_code)
    polygons["municipality_name"] = polygons["NAMEUNIT"].fillna("").astype(str)
    polygons = polygons[polygons["municipality_code"].astype(str).str.len() == 5].copy()
    polygons["province_code"] = polygons["municipality_code"].astype(str).str[:2]
    return polygons


def _endpoint_points(roads_gdf, which: str):
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise RuntimeError("geopandas and shapely are required to build endpoint diagnostics") from exc

    if which not in {"START", "END"}:
        raise ValueError(f"Unsupported endpoint selector: {which}")
    coords = roads_gdf.geometry.apply(lambda geom: geom.coords[0] if which == "START" else geom.coords[-1])
    return gpd.GeoDataFrame(
        roads_gdf[["raw_segment_id", "road_name", "PK_inicio", "PK_fin", "length_km"]].copy(),
        geometry=[Point(xy) for xy in coords],
        crs=roads_gdf.crs,
    )


def _assign_endpoint_points_to_municipalities(
    point_gdf,
    municipality_polygons_gdf,
    suffix: str,
) -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to assign road endpoints to municipalities") from exc

    point_df = point_gdf.copy()
    polygons = municipality_polygons_gdf.to_crs(point_df.crs)
    direct = gpd.sjoin(
        point_df,
        polygons[["municipality_code", "municipality_name", "geometry"]],
        how="left",
        predicate="within",
    )
    direct = direct.drop_duplicates(subset=["raw_segment_id"], keep="first")
    exact_mask = direct["municipality_code"].notna()

    if (~exact_mask).any():
        unresolved = point_df.loc[~point_df["raw_segment_id"].isin(direct.loc[exact_mask, "raw_segment_id"])].copy()
        unresolved = unresolved.to_crs(epsg=3857)
        nearest = gpd.sjoin_nearest(
            unresolved,
            polygons.to_crs(epsg=3857)[["municipality_code", "municipality_name", "geometry"]],
            how="left",
            distance_col="distance_m",
        )
        nearest = nearest.drop_duplicates(subset=["raw_segment_id"], keep="first")
        snap_ok = pd.to_numeric(nearest["distance_m"], errors="coerce") <= _BOUNDARY_SNAP_MAX_DISTANCE_M
        nearest.loc[~snap_ok, ["municipality_code", "municipality_name"]] = np.nan
        nearest["assignment_method"] = np.where(snap_ok, "boundary_snap", "geo_fallback")
        nearest["assignment_distance_m"] = pd.to_numeric(nearest["distance_m"], errors="coerce")
        nearest = nearest.to_crs(point_gdf.crs)
        nearest["endpoint_latitude"] = nearest.geometry.y.astype(float)
        nearest["endpoint_longitude"] = nearest.geometry.x.astype(float)
        nearest = nearest[[
            "raw_segment_id",
            "municipality_code",
            "municipality_name",
            "assignment_method",
            "assignment_distance_m",
            "endpoint_latitude",
            "endpoint_longitude",
        ]].copy()
    else:
        nearest = pd.DataFrame(columns=[
            "raw_segment_id",
            "municipality_code",
            "municipality_name",
            "assignment_method",
            "assignment_distance_m",
            "endpoint_latitude",
            "endpoint_longitude",
        ])

    exact = direct.copy()
    exact["assignment_method"] = np.where(exact["municipality_code"].notna(), "exact_polygon", "geo_fallback")
    exact["assignment_distance_m"] = np.where(exact["municipality_code"].notna(), 0.0, np.nan)
    exact["endpoint_latitude"] = exact.geometry.y.astype(float)
    exact["endpoint_longitude"] = exact.geometry.x.astype(float)
    exact = exact[[
        "raw_segment_id",
        "municipality_code",
        "municipality_name",
        "assignment_method",
        "assignment_distance_m",
        "endpoint_latitude",
        "endpoint_longitude",
    ]].copy()

    unresolved_ids = set(nearest["raw_segment_id"])
    exact = exact[~exact["raw_segment_id"].isin(unresolved_ids)]
    combined = pd.concat([exact, nearest], ignore_index=True)
    combined["municipality_code"] = combined["municipality_code"].fillna("").astype(str)
    combined["municipality_name"] = combined["municipality_name"].fillna("").astype(str)
    combined = combined.rename(columns={
        "municipality_code": f"{suffix.lower()}_municipality_code",
        "municipality_name": f"{suffix.lower()}_municipality_name",
        "assignment_method": f"{suffix.lower()}_assignment_method",
        "assignment_distance_m": f"{suffix.lower()}_assignment_distance_m",
        "endpoint_latitude": f"{suffix.lower()}_latitude",
        "endpoint_longitude": f"{suffix.lower()}_longitude",
    })
    return combined


def _fill_missing_anchor_points(municipalities: pd.DataFrame, roads_gdf) -> pd.DataFrame:
    try:
        import geopandas as gpd
        from shapely.ops import nearest_points
    except ImportError as exc:
        raise RuntimeError("geopandas and shapely are required to fill municipality road anchors") from exc

    updated = municipalities.copy()
    missing_mask = (
        updated["nearest_road_segment_id"].notna()
        & (
            updated["road_anchor_latitude"].isna()
            | updated["road_anchor_longitude"].isna()
        )
        & updated["latitude"].notna()
        & updated["longitude"].notna()
    )
    if not missing_mask.any():
        return updated

    roads_metric = roads_gdf.to_crs(epsg=3857).set_index("raw_segment_id")
    missing = updated.loc[missing_mask].copy()
    point_gdf = gpd.GeoDataFrame(
        missing[["municipality_code", "nearest_road_segment_id"]].copy(),
        geometry=gpd.points_from_xy(missing["longitude"], missing["latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    snapped = []
    for row in point_gdf.itertuples(index=False):
        segment_id = int(float(row.nearest_road_segment_id))
        road_geom = roads_metric.loc[segment_id, "geometry"]
        snapped.append(nearest_points(row.geometry, road_geom)[1])
    snapped_gs = gpd.GeoSeries(snapped, crs=roads_metric.crs).to_crs(epsg=4326)
    updated.loc[missing_mask, "road_anchor_latitude"] = snapped_gs.y.to_numpy(dtype=float)
    updated.loc[missing_mask, "road_anchor_longitude"] = snapped_gs.x.to_numpy(dtype=float)
    return updated


def _attach_road_distance_to_covered_nodes(
    edge_df: pd.DataFrame,
    covered_municipalities: pd.DataFrame,
    roads_gdf,
) -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to audit road proximity to covered municipalities") from exc

    covered = covered_municipalities[
        covered_municipalities["latitude"].notna() & covered_municipalities["longitude"].notna()
    ][["municipality_code", "nombre", "latitude", "longitude"]].copy()
    if covered.empty:
        audited = edge_df.copy()
        audited["nearest_covered_municipality_code"] = ""
        audited["nearest_covered_municipality_name"] = ""
        audited["nearest_covered_municipality_distance_km"] = np.nan
        audited["road_far_from_covered_node"] = False
        return audited

    covered_metric = gpd.GeoDataFrame(
        covered[["municipality_code", "nombre"]].copy(),
        geometry=gpd.points_from_xy(covered["longitude"], covered["latitude"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    roads_metric = roads_gdf.to_crs(epsg=3857)[["raw_segment_id", "geometry"]].copy()
    nearest = gpd.sjoin_nearest(
        roads_metric,
        covered_metric[["municipality_code", "nombre", "geometry"]],
        how="left",
        distance_col="nearest_covered_municipality_distance_m",
    )
    nearest = nearest.sort_values(
        ["raw_segment_id", "nearest_covered_municipality_distance_m", "municipality_code"],
        ascending=[True, True, True],
    ).drop_duplicates(subset=["raw_segment_id"], keep="first")
    nearest = nearest.rename(columns={
        "municipality_code": "nearest_covered_municipality_code",
        "nombre": "nearest_covered_municipality_name",
    })
    nearest["nearest_covered_municipality_distance_km"] = (
        pd.to_numeric(nearest["nearest_covered_municipality_distance_m"], errors="coerce") / 1000.0
    )
    nearest["road_far_from_covered_node"] = (
        nearest["nearest_covered_municipality_distance_km"] > _ROAD_FAR_FROM_COVERED_NODE_THRESHOLD_KM
    )
    return edge_df.merge(
        nearest[[
            "raw_segment_id",
            "nearest_covered_municipality_code",
            "nearest_covered_municipality_name",
            "nearest_covered_municipality_distance_km",
            "road_far_from_covered_node",
        ]],
        on="raw_segment_id",
        how="left",
    )


def _daily_municipality_od_flows_from_travel_df(
    travel_df: pd.DataFrame,
    municipality_codes: Set[str],
    mainland_only: bool = True,
) -> pd.DataFrame:
    travel = travel_df.copy()
    travel["people"] = pd.to_numeric(travel.get("people"), errors="coerce").fillna(0.0)
    travel = travel[travel["people"] > 0.0].copy()
    for col in ("residence_area", "overnight_stay_area"):
        travel[col] = travel[col].map(_base_municipality_code)
    if mainland_only:
        travel = travel[
            travel["residence_area"].map(lambda code: len(code) == 5 and code[:2] not in _NON_MAINLAND_PROVINCES)
            & travel["overnight_stay_area"].map(lambda code: len(code) == 5 and code[:2] not in _NON_MAINLAND_PROVINCES)
        ].copy()
    travel = travel[
        travel["residence_area"].isin(municipality_codes)
        & travel["overnight_stay_area"].isin(municipality_codes)
        & travel["residence_area"].ne(travel["overnight_stay_area"])
    ].copy()
    if travel.empty:
        return pd.DataFrame(columns=[
            "municipality_origin_code",
            "municipality_destination_code",
            "daily_people",
        ])

    if "date" in travel.columns:
        per_day = travel.groupby(
            ["date", "residence_area", "overnight_stay_area"],
            dropna=False,
            as_index=False,
        )["people"].sum()
        grouped = per_day.groupby(
            ["residence_area", "overnight_stay_area"],
            dropna=False,
            as_index=False,
        )["people"].mean()
    else:
        grouped = travel.groupby(
            ["residence_area", "overnight_stay_area"],
            dropna=False,
            as_index=False,
        )["people"].sum()

    grouped = grouped.rename(columns={
        "residence_area": "municipality_origin_code",
        "overnight_stay_area": "municipality_destination_code",
        "people": "daily_people",
    })
    grouped["daily_people"] = pd.to_numeric(grouped["daily_people"], errors="coerce").fillna(0.0)
    return grouped[grouped["daily_people"] > 0.0].sort_values(
        ["daily_people", "municipality_origin_code", "municipality_destination_code"],
        ascending=[False, True, True],
    )


def _apply_people_to_bev_conversion(
    flows_df: pd.DataFrame,
    car_mode_share: float | None = INTERURBAN_PRIVATE_CAR_MODE_SHARE,
    occupancy_rate: float | None = INTERURBAN_PRIVATE_VEHICLE_OCCUPANCY,
    ev_penetration_rate: float | None = EV_PENETRATION_RATE,
    bev_fraction: float | None = BEV_FRACTION,
) -> tuple[pd.DataFrame, dict[str, float]]:
    flows = flows_df.copy()
    safe_car_mode_share = min(max(float(
        INTERURBAN_PRIVATE_CAR_MODE_SHARE if car_mode_share is None else car_mode_share
    ), 0.0), 1.0)
    safe_occupancy = max(float(
        INTERURBAN_PRIVATE_VEHICLE_OCCUPANCY if occupancy_rate is None else occupancy_rate
    ), 1e-9)
    safe_ev_penetration = max(float(EV_PENETRATION_RATE if ev_penetration_rate is None else ev_penetration_rate), 0.0)
    safe_bev_fraction = max(float(BEV_FRACTION if bev_fraction is None else bev_fraction), 0.0)
    flows["daily_car_travelers"] = pd.to_numeric(flows["daily_people"], errors="coerce").fillna(0.0) * safe_car_mode_share
    flows["daily_vehicle_trips"] = flows["daily_car_travelers"] / safe_occupancy
    flows["daily_bev_trips"] = flows["daily_vehicle_trips"] * safe_ev_penetration * safe_bev_fraction
    return flows, {
        "car_mode_share": float(safe_car_mode_share),
        "occupancy_rate": float(safe_occupancy),
        "ev_penetration_rate": float(safe_ev_penetration),
        "bev_fraction": float(safe_bev_fraction),
    }


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _cluster_metric_points(points_df: pd.DataFrame, tolerance_m: float) -> pd.DataFrame:
    if points_df.empty:
        return points_df.assign(cluster_id=pd.Series(dtype=int))

    coords = points_df[["x_m", "y_m"]].to_numpy(dtype=float)
    uf = _UnionFind(len(coords))
    cell_size = max(float(tolerance_m), 1e-9)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(coords):
        key = (int(np.floor(x / cell_size)), int(np.floor(y / cell_size)))
        buckets.setdefault(key, []).append(idx)

    for idx, (x, y) in enumerate(coords):
        base = (int(np.floor(x / cell_size)), int(np.floor(y / cell_size)))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in buckets.get((base[0] + dx, base[1] + dy), []):
                    if other <= idx:
                        continue
                    ox, oy = coords[other]
                    if float(np.hypot(x - ox, y - oy)) <= tolerance_m:
                        uf.union(idx, other)

    roots = [uf.find(i) for i in range(len(coords))]
    root_order = {root: order for order, root in enumerate(sorted(set(roots)))}
    clustered = points_df.copy()
    clustered["cluster_id"] = [root_order[root] for root in roots]
    return clustered


def _extract_intersection_points(geometry) -> list[object]:
    if geometry is None or geometry.is_empty:
        return []
    geom_type = geometry.geom_type
    if geom_type == "Point":
        return [geometry]
    if geom_type == "MultiPoint":
        return list(geometry.geoms)
    if geom_type == "GeometryCollection":
        points: list[object] = []
        for geom in geometry.geoms:
            points.extend(_extract_intersection_points(geom))
        return points
    if geom_type == "LineString":
        boundary = list(geometry.boundary.geoms) if hasattr(geometry.boundary, "geoms") else [geometry.boundary]
        return [point for point in boundary if point is not None and not point.is_empty]
    if geom_type == "MultiLineString":
        points: list[object] = []
        for geom in geometry.geoms:
            points.extend(_extract_intersection_points(geom))
        return points
    return []


def _build_stitched_segment_topology(roads_gdf):
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import nearest_points
    except ImportError as exc:
        raise RuntimeError("geopandas is required to stitch processed road topology") from exc

    roads_metric = roads_gdf.to_crs(epsg=3857).copy()
    endpoint_rows: list[dict[str, object]] = []
    for row in roads_metric.itertuples(index=False):
        geom = row.geometry
        endpoint_rows.append({
            "segment_id": int(row.segment_id),
            "road_name": str(row.road_name),
            "road_type": str(row.road_type),
            "point_kind": "endpoint",
            "endpoint_role": "START",
            "x_m": float(geom.coords[0][0]),
            "y_m": float(geom.coords[0][1]),
            "proj_m": 0.0,
        })
        endpoint_rows.append({
            "segment_id": int(row.segment_id),
            "road_name": str(row.road_name),
            "road_type": str(row.road_type),
            "point_kind": "endpoint",
            "endpoint_role": "END",
            "x_m": float(geom.coords[-1][0]),
            "y_m": float(geom.coords[-1][1]),
            "proj_m": float(geom.length),
        })
    endpoint_points = pd.DataFrame(endpoint_rows)
    sindex = roads_metric.sindex
    near_road_rows: list[dict[str, object]] = []
    for endpoint in endpoint_points.itertuples(index=False):
        point = Point(float(endpoint.x_m), float(endpoint.y_m))
        candidate_idx = sindex.query(point.buffer(_NEAR_ROAD_CONNECTION_TOLERANCE_M), predicate="intersects")
        for other_idx in candidate_idx:
            other = roads_metric.loc[other_idx]
            if int(other["segment_id"]) == int(endpoint.segment_id):
                continue
            if str(other["road_name"]) == str(endpoint.road_name):
                continue
            distance_m = float(other.geometry.distance(point))
            if distance_m > _NEAR_ROAD_CONNECTION_TOLERANCE_M:
                continue
            snapped = nearest_points(point, other.geometry)[1]
            near_road_rows.append({
                "segment_id": int(other["segment_id"]),
                "road_name": str(other["road_name"]),
                "road_type": str(other["road_type"]),
                "point_kind": "near_road_connection",
                "endpoint_role": "",
                "x_m": float(snapped.x),
                "y_m": float(snapped.y),
                "proj_m": float(other.geometry.project(snapped)),
            })

    intersection_rows: list[dict[str, object]] = []
    for left_idx, left in roads_metric.iterrows():
        candidate_idx = sindex.query(left.geometry, predicate="intersects")
        for right_idx in candidate_idx:
            if right_idx <= left_idx:
                continue
            right = roads_metric.loc[right_idx]
            if int(left["segment_id"]) == int(right["segment_id"]):
                continue
            if str(left["road_name"]) == str(right["road_name"]):
                continue
            intersection = left.geometry.intersection(right.geometry)
            for point in _extract_intersection_points(intersection):
                left_proj = float(left.geometry.project(point))
                right_proj = float(right.geometry.project(point))
                if left_proj <= 1e-6 or left_proj >= float(left.geometry.length) - 1e-6:
                    left_kind = "intersection_endpoint"
                else:
                    left_kind = "intersection"
                if right_proj <= 1e-6 or right_proj >= float(right.geometry.length) - 1e-6:
                    right_kind = "intersection_endpoint"
                else:
                    right_kind = "intersection"
                intersection_rows.append({
                    "segment_id": int(left["segment_id"]),
                    "road_name": str(left["road_name"]),
                    "road_type": str(left["road_type"]),
                    "point_kind": left_kind,
                    "endpoint_role": "",
                    "x_m": float(point.x),
                    "y_m": float(point.y),
                    "proj_m": left_proj,
                })
                intersection_rows.append({
                    "segment_id": int(right["segment_id"]),
                    "road_name": str(right["road_name"]),
                    "road_type": str(right["road_type"]),
                    "point_kind": right_kind,
                    "endpoint_role": "",
                    "x_m": float(point.x),
                    "y_m": float(point.y),
                    "proj_m": right_proj,
                })

    extra_frames = [endpoint_points]
    if near_road_rows:
        extra_frames.append(pd.DataFrame(near_road_rows))
    if intersection_rows:
        extra_frames.append(pd.DataFrame(intersection_rows))
    point_rows = pd.concat(extra_frames, ignore_index=True)
    point_rows = _cluster_metric_points(point_rows, tolerance_m=_JUNCTION_CLUSTER_TOLERANCE_M)

    cluster_summary = point_rows.groupby("cluster_id", dropna=False).agg(
        x_m=("x_m", "mean"),
        y_m=("y_m", "mean"),
        road_name_count=("road_name", "nunique"),
        segment_id_count=("segment_id", "nunique"),
        endpoint_count=("point_kind", lambda s: int((s == "endpoint").sum())),
        intersection_count=("point_kind", lambda s: int((s.str.contains("intersection")).sum())),
    ).reset_index()
    cluster_summary["node_id"] = cluster_summary["cluster_id"].map(lambda cid: f"JUNC_{int(cid):06d}")
    cluster_id_to_node = dict(zip(cluster_summary["cluster_id"], cluster_summary["node_id"]))
    point_rows["node_id"] = point_rows["cluster_id"].map(cluster_id_to_node)

    roads_metric = roads_metric.set_index("segment_id")
    segment_cut_rows: list[dict[str, object]] = []
    segment_subedges: list[dict[str, object]] = []
    endpoint_node_map: dict[tuple[int, str], str] = {}
    for segment_id, group in point_rows.groupby("segment_id", dropna=False):
        segment_id = int(segment_id)
        geom = roads_metric.loc[segment_id, "geometry"]
        total_m = max(float(geom.length), 1e-9)
        pk_start = float(roads_metric.loc[segment_id, "PK_inicio"] or 0.0)
        pk_end = float(roads_metric.loc[segment_id, "PK_fin"] or 0.0)
        road_name = str(roads_metric.loc[segment_id, "road_name"])
        road_type = str(roads_metric.loc[segment_id, "road_type"])
        target_flow = float(roads_metric.loc[segment_id, "daily_bev_traffic_2027"] or 0.0)
        grouped = group.groupby("node_id", dropna=False).agg(
            proj_m=("proj_m", "mean"),
            x_m=("x_m", "mean"),
            y_m=("y_m", "mean"),
            has_endpoint=("point_kind", lambda s: bool((s == "endpoint").any())),
            has_intersection=("point_kind", lambda s: bool(s.str.contains("intersection").any())),
            endpoint_role=("endpoint_role", lambda s: "|".join(sorted({str(v) for v in s if str(v)}))),
        ).reset_index()
        grouped = grouped.sort_values(["proj_m", "node_id"])
        for cut_idx, cut in enumerate(grouped.itertuples(index=False)):
            frac = min(max(float(cut.proj_m) / total_m, 0.0), 1.0)
            road_km = pk_start + frac * (pk_end - pk_start)
            endpoint_role = str(cut.endpoint_role or "")
            if "START" in endpoint_role:
                endpoint_node_map[(segment_id, "START")] = str(cut.node_id)
            if "END" in endpoint_role:
                endpoint_node_map[(segment_id, "END")] = str(cut.node_id)
            segment_cut_rows.append({
                "segment_id": segment_id,
                "road_name": road_name,
                "road_type": road_type,
                "node_id": str(cut.node_id),
                "proj_m": float(cut.proj_m),
                "road_km": float(road_km),
                "cut_index": int(cut_idx),
                "has_endpoint": bool(cut.has_endpoint),
                "has_intersection": bool(cut.has_intersection),
                "endpoint_role": endpoint_role,
                "x_m": float(cut.x_m),
                "y_m": float(cut.y_m),
            })
        cut_rows = sorted(segment_cut_rows[-len(grouped):], key=lambda item: (item["proj_m"], item["node_id"]))
        for part_idx in range(len(cut_rows) - 1):
            left = cut_rows[part_idx]
            right = cut_rows[part_idx + 1]
            if left["node_id"] == right["node_id"]:
                continue
            distance_km = max((float(right["proj_m"]) - float(left["proj_m"])) / 1000.0, _SAME_ROAD_GAP_BRIDGE_MIN_KM)
            segment_subedges.append({
                "edge_id": f"ROADSEG_{segment_id}_{part_idx:03d}",
                "segment_id": segment_id,
                "road_name": road_name,
                "road_type": road_type,
                "from_node_id": str(left["node_id"]),
                "to_node_id": str(right["node_id"]),
                "distance_km": float(distance_km),
                "from_road_km": float(left["road_km"]),
                "to_road_km": float(right["road_km"]),
                "source_segment_ids": (segment_id,),
                "target_daily_bev_traffic_2027": target_flow,
                "is_gap_bridge": False,
            })

    segment_cuts_df = pd.DataFrame(segment_cut_rows)
    segment_subedges_df = pd.DataFrame(segment_subedges)
    gap_bridges: list[dict[str, object]] = []
    for road_name, road_segments in roads_metric.reset_index().groupby("road_name", dropna=False):
        ordered = road_segments.sort_values(["PK_inicio", "PK_fin", "segment_id"])
        rows = list(ordered.itertuples(index=False))
        for idx in range(len(rows) - 1):
            left = rows[idx]
            right = rows[idx + 1]
            left_node = endpoint_node_map.get((int(left.segment_id), "END"))
            right_node = endpoint_node_map.get((int(right.segment_id), "START"))
            if not left_node or not right_node or left_node == right_node:
                continue
            left_point = cluster_summary.loc[cluster_summary["node_id"] == left_node].iloc[0]
            right_point = cluster_summary.loc[cluster_summary["node_id"] == right_node].iloc[0]
            gap_geom_km = float(np.hypot(left_point["x_m"] - right_point["x_m"], left_point["y_m"] - right_point["y_m"])) / 1000.0
            pk_gap_km = max(float(right.PK_inicio) - float(left.PK_fin), 0.0)
            bridge_km = max(gap_geom_km, pk_gap_km, _SAME_ROAD_GAP_BRIDGE_MIN_KM)
            gap_bridges.append({
                "edge_id": f"GAPBRIDGE_{road_name}_{int(left.segment_id)}_{int(right.segment_id)}",
                "segment_id_left": int(left.segment_id),
                "segment_id_right": int(right.segment_id),
                "road_name": str(road_name),
                "road_type": str(left.road_type),
                "from_node_id": str(left_node),
                "to_node_id": str(right_node),
                "distance_km": float(bridge_km),
                "from_road_km": float(left.PK_fin),
                "to_road_km": float(right.PK_inicio),
                "source_segment_ids": tuple(),
                "target_daily_bev_traffic_2027": 0.0,
                "is_gap_bridge": True,
                "pk_gap_km": float(pk_gap_km),
                "geometry_gap_km": float(gap_geom_km),
            })

    gap_bridges_df = pd.DataFrame(gap_bridges)
    all_edges_df = pd.concat(
        [segment_subedges_df, gap_bridges_df[[col for col in gap_bridges_df.columns if col in segment_subedges_df.columns]]],
        ignore_index=True,
    ) if not gap_bridges_df.empty else segment_subedges_df.copy()

    junction_nodes = gpd.GeoDataFrame(
        cluster_summary[["node_id", "x_m", "y_m", "road_name_count", "segment_id_count", "endpoint_count", "intersection_count"]].copy(),
        geometry=gpd.points_from_xy(cluster_summary["x_m"], cluster_summary["y_m"]),
        crs="EPSG:3857",
    ).to_crs(epsg=4326)
    junction_nodes["latitude"] = junction_nodes.geometry.y.astype(float)
    junction_nodes["longitude"] = junction_nodes.geometry.x.astype(float)
    junction_nodes["node_type"] = np.where(
        junction_nodes["road_name_count"] > 1,
        NodeKind.ROAD_EXCHANGE.value,
        NodeKind.ROAD_JUNCTION.value,
    )
    return {
        "junction_nodes": junction_nodes.drop(columns=["geometry"]),
        "segment_cuts": segment_cuts_df,
        "road_edges": all_edges_df,
        "gap_bridges": gap_bridges_df,
        "endpoint_node_map": endpoint_node_map,
    }


def build_municipality_od_matrix(
    data_dir: Path,
    municipality_codes: Set[str],
    mainland_only: bool = True,
    car_mode_share: float | None = INTERURBAN_PRIVATE_CAR_MODE_SHARE,
    occupancy_rate: float | None = INTERURBAN_PRIVATE_VEHICLE_OCCUPANCY,
    ev_penetration_rate: float | None = EV_PENETRATION_RATE,
    bev_fraction: float | None = BEV_FRACTION,
) -> Tuple[ODMatrix, pd.DataFrame, Dict[str, object]]:
    data_dir = Path(data_dir)
    raw_dir = data_dir.parent
    path = raw_dir / _OD_RAW_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing municipality OD parquet at {path}")

    travel = pd.read_parquet(path, columns=["date", "residence_area", "overnight_stay_area", "people"])
    flows = _daily_municipality_od_flows_from_travel_df(
        travel,
        municipality_codes=municipality_codes,
        mainland_only=mainland_only,
    )
    total_daily_people = float(flows["daily_people"].sum()) if not flows.empty else 0.0
    flows, conversion = _apply_people_to_bev_conversion(
        flows,
        car_mode_share=car_mode_share,
        occupancy_rate=occupancy_rate,
        ev_penetration_rate=ev_penetration_rate,
        bev_fraction=bev_fraction,
    )
    flows["origin"] = flows["municipality_origin_code"].map(_municipality_node_id)
    flows["destination"] = flows["municipality_destination_code"].map(_municipality_node_id)

    od_matrix = ODMatrix()
    for row in flows.itertuples(index=False):
        od_matrix.add_pair(
            ODPair(
                origin=str(row.origin),
                destination=str(row.destination),
                daily_bev_trips=float(row.daily_bev_trips),
                purpose="leisure",
            )
        )

    summary = {
        "municipality_nodes_in_od": int(len(municipality_codes)),
        "n_municipality_od_pairs": int(len(flows)),
        "total_daily_people": total_daily_people,
        **conversion,
        "total_daily_car_travelers": float(flows["daily_car_travelers"].sum()) if not flows.empty else 0.0,
        "total_daily_vehicle_trips": float(flows["daily_vehicle_trips"].sum()) if not flows.empty else 0.0,
        "scaled_total_daily_bev_trips": float(flows["daily_bev_trips"].sum()) if not flows.empty else 0.0,
    }
    return od_matrix, flows, summary


def build_municipality_calibration_network(
    data_dir: Path,
    mainland_only: bool = True,
    covered_only: bool = True,
) -> Tuple[RoadNetwork, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    data_dir = Path(data_dir)
    raw_dir = data_dir.parent

    roads = _load_processed_roads(data_dir)
    demand_path = data_dir / "demand_per_segment.csv"
    if demand_path.exists():
        demand = pd.read_csv(demand_path)[["segment_id", "daily_bev_traffic_2027"]].copy()
        demand["segment_id"] = demand["segment_id"].astype(int)
        roads = roads.merge(demand, on="segment_id", how="left")
    else:
        roads["daily_bev_traffic_2027"] = np.nan

    municipality_ref = _load_official_municipality_reference(
        raw_dir=raw_dir,
        municipality_reference_path=(raw_dir / _MUNICIPALITY_REFERENCE_RELATIVE_PATH),
    )
    polygons = _load_mainland_municipality_polygons(raw_dir)
    covered_codes: Set[str] | None = None
    if covered_only:
        covered_codes = _load_od_covered_municipality_codes(raw_dir, mainland_only=mainland_only)

    if mainland_only:
        municipality_ref = municipality_ref[
            ~municipality_ref["province_code"].astype(str).isin(_NON_MAINLAND_PROVINCES)
        ].copy()
        polygons = polygons[
            ~polygons["province_code"].astype(str).isin(_NON_MAINLAND_PROVINCES)
        ].copy()
    if covered_codes is not None:
        municipality_ref = municipality_ref[
            municipality_ref["municipality_code"].astype(str).isin(covered_codes)
        ].copy()
    valid_codes = set(municipality_ref["municipality_code"].astype(str))
    polygons = polygons[polygons["municipality_code"].astype(str).isin(valid_codes)].copy()

    start_df = _assign_endpoint_points_to_municipalities(_endpoint_points(roads, "START"), polygons, "START")
    end_df = _assign_endpoint_points_to_municipalities(_endpoint_points(roads, "END"), polygons, "END")
    edge_df = roads.merge(start_df, on="raw_segment_id", how="left").merge(end_df, on="raw_segment_id", how="left")

    municipalities = municipality_ref.copy()
    municipalities["node_id"] = municipalities["municipality_code"].map(_municipality_node_id)
    municipalities["node_type"] = NodeKind.MUNICIPALITY.value
    municipalities["is_mainland"] = ~municipalities["province_code"].astype(str).isin(_NON_MAINLAND_PROVINCES)
    municipalities = _attach_nearest_road_anchor(municipalities, roads)
    municipalities = municipalities.sort_values(
        ["municipality_code", "nearest_road_distance_km", "nearest_road_segment_id"],
        ascending=[True, True, True],
    ).drop_duplicates(subset=["municipality_code"], keep="first")
    municipalities = _fill_missing_anchor_points(municipalities, roads)
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to build the municipality calibration graph") from exc

    topology = _build_stitched_segment_topology(edge_df)
    junction_nodes = topology["junction_nodes"]
    segment_cuts = topology["segment_cuts"]
    road_edges = topology["road_edges"]
    gap_bridges = topology["gap_bridges"]
    endpoint_node_map = topology["endpoint_node_map"]
    roads_metric = roads.to_crs(epsg=3857).set_index("segment_id")
    cut_rows_by_segment = {
        int(segment_id): grp.sort_values(["proj_m", "cut_index", "node_id"]).copy()
        for segment_id, grp in segment_cuts.groupby("segment_id", dropna=False)
    }

    network = RoadNetwork()
    for row in junction_nodes.itertuples(index=False):
        network.add_node(
            RoadNode(
                node_id=str(row.node_id),
                name=str(row.node_id),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                node_type=str(row.node_type),
                population=0,
            )
        )
    for row in municipalities.itertuples(index=False):
        population = 0 if pd.isna(row.pob_2025) else int(float(row.pob_2025))
        network.add_node(
            RoadNode(
                node_id=str(row.node_id),
                name=str(row.nombre),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                node_type=NodeKind.MUNICIPALITY.value,
                population=population,
            )
        )

    for row in road_edges.itertuples(index=False):
        speed = float(_SPEED_KMH.get(str(row.road_type), 90.0))
        network.add_undirected_road(
            RoadEdge(
                edge_id=str(row.edge_id),
                from_node=str(row.from_node_id),
                to_node=str(row.to_node_id),
                distance_km=float(row.distance_km),
                travel_time_min=max(float(row.distance_km) / speed * 60.0, _MUNICIPALITY_ENDPOINT_CONNECTOR_MIN),
                road_type=str(row.road_type),
                road_name=str(row.road_name),
                from_road_km=float(row.from_road_km),
                to_road_km=float(row.to_road_km),
                source_segment_ids=tuple(row.source_segment_ids) if isinstance(row.source_segment_ids, tuple) else tuple(),
                target_daily_bev_traffic_2027=float(row.target_daily_bev_traffic_2027),
                geometry_backed=True,
            )
        )

    municipality_attachment_counts: Dict[str, int] = {}
    endpoint_connector_keys: set[tuple[str, str]] = set()
    for row in edge_df.itertuples(index=False):
        segment_id = int(row.segment_id)
        start_node = endpoint_node_map.get((segment_id, "START"), "")
        end_node = endpoint_node_map.get((segment_id, "END"), "")
        for municipality_code, endpoint_node in (
            (str(getattr(row, "start_municipality_code", "") or ""), start_node),
            (str(getattr(row, "end_municipality_code", "") or ""), end_node),
        ):
            if municipality_code not in valid_codes or not endpoint_node:
                continue
            municipality_attachment_counts[municipality_code] = municipality_attachment_counts.get(municipality_code, 0) + 1
            municipality_node = _municipality_node_id(municipality_code)
            edge_key = tuple(sorted((municipality_node, endpoint_node)))
            if edge_key in endpoint_connector_keys:
                continue
            endpoint_connector_keys.add(edge_key)
            network.add_undirected_road(
                RoadEdge(
                    edge_id=f"MUNIEND_{municipality_code}_{segment_id}_{endpoint_node}",
                    from_node=municipality_node,
                    to_node=endpoint_node,
                    distance_km=_MUNICIPALITY_ENDPOINT_CONNECTOR_KM,
                    travel_time_min=_MUNICIPALITY_ENDPOINT_CONNECTOR_MIN,
                    road_type="connector",
                    road_name=f"CONNECTOR_{municipality_code}",
                    geometry_backed=True,
                )
            )

    municipalities["endpoint_attachment_count"] = municipalities["municipality_code"].map(municipality_attachment_counts).fillna(0).astype(int)
    municipalities["has_endpoint_attachment"] = municipalities["endpoint_attachment_count"] > 0

    anchor_candidates = municipalities[
        ~municipalities["has_endpoint_attachment"]
        & municipalities["nearest_road_segment_id"].notna()
        & municipalities["road_anchor_latitude"].notna()
        & municipalities["road_anchor_longitude"].notna()
    ][[
        "municipality_code",
        "node_id",
        "nearest_road_segment_id",
        "nearest_road_distance_km",
        "road_anchor_latitude",
        "road_anchor_longitude",
    ]].copy()
    if not anchor_candidates.empty:
        anchor_gdf = gpd.GeoDataFrame(
            anchor_candidates,
            geometry=gpd.points_from_xy(anchor_candidates["road_anchor_longitude"], anchor_candidates["road_anchor_latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)
        anchored_codes: set[str] = set()
        for row in anchor_gdf.itertuples(index=False):
            municipality_code = str(row.municipality_code).zfill(5)
            anchored_codes.add(municipality_code)
            anchor_node = _road_anchor_node_id(municipality_code)
            if anchor_node not in network.nodes:
                network.add_node(
                    RoadNode(
                        node_id=anchor_node,
                        name=anchor_node,
                        latitude=float(row.road_anchor_latitude),
                        longitude=float(row.road_anchor_longitude),
                        node_type=NodeKind.ROAD_ANCHOR.value,
                        population=0,
                    )
                )
            network.add_undirected_road(
                RoadEdge(
                    edge_id=f"ANCHORCONNECT_{municipality_code}",
                    from_node=str(row.node_id),
                    to_node=anchor_node,
                    distance_km=max(float(row.nearest_road_distance_km or 0.0), _MUNICIPALITY_ENDPOINT_CONNECTOR_KM),
                    travel_time_min=max(float(row.nearest_road_distance_km or 0.0) / 40.0 * 60.0, _MUNICIPALITY_ENDPOINT_CONNECTOR_MIN),
                    road_type="connector",
                    road_name=f"CONNECTOR_{municipality_code}",
                    geometry_backed=True,
                )
            )

            segment_id = int(float(row.nearest_road_segment_id))
            segment_chain = cut_rows_by_segment.get(segment_id)
            if segment_chain is None or segment_chain.empty:
                continue
            road_geom_metric = roads_metric.loc[segment_id, "geometry"]
            proj_m = float(road_geom_metric.project(row.geometry))
            left = segment_chain[segment_chain["proj_m"] <= proj_m].tail(1)
            right = segment_chain[segment_chain["proj_m"] >= proj_m].head(1)
            linked_nodes: dict[str, float] = {}
            if not left.empty:
                linked_nodes[str(left.iloc[0]["node_id"])] = max((proj_m - float(left.iloc[0]["proj_m"])) / 1000.0, _MUNICIPALITY_ENDPOINT_CONNECTOR_KM)
            if not right.empty:
                linked_nodes[str(right.iloc[0]["node_id"])] = max((float(right.iloc[0]["proj_m"]) - proj_m) / 1000.0, _MUNICIPALITY_ENDPOINT_CONNECTOR_KM)
            road_type = str(roads_metric.loc[segment_id, "road_type"])
            road_name = str(roads_metric.loc[segment_id, "road_name"])
            target_flow = float(roads_metric.loc[segment_id, "daily_bev_traffic_2027"] or 0.0)
            speed = float(_SPEED_KMH.get(road_type, 90.0))
            for linked_node, split_distance_km in linked_nodes.items():
                if linked_node == anchor_node:
                    continue
                network.add_undirected_road(
                    RoadEdge(
                        edge_id=f"ANCHORLINK_{municipality_code}_{segment_id}_{linked_node}",
                        from_node=anchor_node,
                        to_node=linked_node,
                        distance_km=split_distance_km,
                        travel_time_min=max(split_distance_km / speed * 60.0, _MUNICIPALITY_ENDPOINT_CONNECTOR_MIN),
                        road_type=road_type,
                        road_name=road_name,
                        source_segment_ids=(segment_id,),
                        target_daily_bev_traffic_2027=target_flow,
                        geometry_backed=True,
                    )
                )
        municipalities["has_network_anchor"] = municipalities["municipality_code"].astype(str).isin(anchored_codes)
    else:
        municipalities["has_network_anchor"] = False

    edge_df["start_node_id"] = edge_df["segment_id"].map(lambda sid: endpoint_node_map.get((int(sid), "START"), ""))
    edge_df["end_node_id"] = edge_df["segment_id"].map(lambda sid: endpoint_node_map.get((int(sid), "END"), ""))
    edge_df["start_endpoint_has_covered_municipality"] = edge_df["start_municipality_code"].fillna("").astype(str).isin(valid_codes)
    edge_df["end_endpoint_has_covered_municipality"] = edge_df["end_municipality_code"].fillna("").astype(str).isin(valid_codes)
    edge_df["is_cross_municipality"] = (
        edge_df["start_municipality_code"].fillna("").astype(str).ne("")
        & edge_df["end_municipality_code"].fillna("").astype(str).ne("")
        & edge_df["start_municipality_code"].astype(str).ne(edge_df["end_municipality_code"].astype(str))
    )
    edge_df["is_same_municipality"] = (
        edge_df["start_municipality_code"].fillna("").astype(str).ne("")
        & edge_df["start_municipality_code"].astype(str).eq(edge_df["end_municipality_code"].astype(str))
    )
    edge_df = _attach_road_distance_to_covered_nodes(edge_df, municipalities, roads)

    reachable_municipalities = {
        node_id for node_id, degree in network.graph.degree()
        if str(node_id).startswith("MUNI_") and degree > 0
    }
    summary = {
        "processed_road_segments": int(len(edge_df)),
        "municipality_nodes": int(len(municipalities)),
        "covered_only": bool(covered_only),
        "road_junction_nodes": int(sum(1 for node in network.nodes.values() if node.node_type == NodeKind.ROAD_JUNCTION.value)),
        "road_exchange_nodes": int(sum(1 for node in network.nodes.values() if node.node_type == NodeKind.ROAD_EXCHANGE.value)),
        "road_anchor_nodes": int(sum(1 for node in network.nodes.values() if node.node_type == NodeKind.ROAD_ANCHOR.value)),
        "directed_edges": int(network.edge_count),
        "stitched_road_edges": int((~road_edges["is_gap_bridge"].fillna(False)).sum()) if not road_edges.empty else 0,
        "same_road_gap_bridge_edges": int(len(gap_bridges)),
        "same_road_gap_bridge_total_km": float(gap_bridges["distance_km"].sum()) if not gap_bridges.empty else 0.0,
        "inter_road_intersection_points": int(segment_cuts["has_intersection"].sum()),
        "start_exact_polygon_pct": float(100.0 * edge_df["start_assignment_method"].eq("exact_polygon").mean()),
        "end_exact_polygon_pct": float(100.0 * edge_df["end_assignment_method"].eq("exact_polygon").mean()),
        "municipalities_with_endpoint_attachment": int(municipalities["has_endpoint_attachment"].sum()),
        "municipalities_with_anchor_fallback": int(municipalities["has_network_anchor"].sum()),
        "reachable_municipality_nodes": int(len(reachable_municipalities)),
        "unreachable_municipality_nodes": int(len(municipalities) - len(reachable_municipalities)),
        "weakly_connected_components": int(nx.number_weakly_connected_components(network.graph)),
        "strongly_connected_components": int(nx.number_strongly_connected_components(network.graph)),
        "largest_weak_component_nodes": int(max((len(comp) for comp in nx.weakly_connected_components(network.graph)), default=0)),
        "roads_far_from_covered_node_count": int(edge_df["road_far_from_covered_node"].fillna(False).sum()),
        "roads_far_from_covered_node_share_pct": float(100.0 * edge_df["road_far_from_covered_node"].fillna(False).mean()),
        "nearest_covered_municipality_distance_km_p50": float(edge_df["nearest_covered_municipality_distance_km"].median()),
        "nearest_covered_municipality_distance_km_p90": float(edge_df["nearest_covered_municipality_distance_km"].quantile(0.90)),
        "nearest_covered_municipality_distance_km_p99": float(edge_df["nearest_covered_municipality_distance_km"].quantile(0.99)),
        "nearest_road_distance_km_p50": float(municipalities["nearest_road_distance_km"].median()),
        "nearest_road_distance_km_p90": float(municipalities["nearest_road_distance_km"].quantile(0.90)),
        "nearest_road_distance_km_p99": float(municipalities["nearest_road_distance_km"].quantile(0.99)),
    }
    validate_network(network)
    return network, municipalities, edge_df, summary


def build_municipality_road_graph(
    data_dir: Path,
    mainland_only: bool = True,
    covered_only: bool = False,
) -> Tuple[RoadNetwork, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    data_dir = Path(data_dir)
    raw_dir = data_dir.parent

    roads = _load_raw_roads(raw_dir)
    municipality_ref = _load_official_municipality_reference(
        raw_dir=raw_dir,
        municipality_reference_path=(raw_dir / _MUNICIPALITY_REFERENCE_RELATIVE_PATH),
    )
    covered_codes: Set[str] | None = None
    if covered_only:
        covered_codes = _load_od_covered_municipality_codes(raw_dir, mainland_only=mainland_only)
    polygons = _load_mainland_municipality_polygons(raw_dir)

    if mainland_only:
        municipality_ref = municipality_ref[
            ~municipality_ref["province_code"].astype(str).isin(_NON_MAINLAND_PROVINCES)
        ].copy()
        polygons = polygons[
            ~polygons["province_code"].astype(str).isin(_NON_MAINLAND_PROVINCES)
        ].copy()
    all_reference_codes = set(municipality_ref["municipality_code"].astype(str))
    if covered_codes is not None:
        municipality_ref = municipality_ref[
            municipality_ref["municipality_code"].astype(str).isin(covered_codes)
        ].copy()
    valid_codes = set(municipality_ref["municipality_code"].astype(str))
    polygons = polygons[polygons["municipality_code"].astype(str).isin(all_reference_codes)].copy()

    start_df = _assign_endpoint_points_to_municipalities(_endpoint_points(roads, "START"), polygons, "START")
    end_df = _assign_endpoint_points_to_municipalities(_endpoint_points(roads, "END"), polygons, "END")
    edge_df = roads.merge(start_df, on="raw_segment_id", how="left").merge(end_df, on="raw_segment_id", how="left")

    municipalities = municipality_ref.copy()
    municipalities["node_id"] = municipalities["municipality_code"].map(_municipality_node_id)
    municipalities["node_type"] = "municipality"
    municipalities["is_mainland"] = ~municipalities["province_code"].astype(str).isin(_NON_MAINLAND_PROVINCES)
    anchor_roads = roads.rename(columns={"raw_segment_id": "segment_id"})
    municipalities = _attach_nearest_road_anchor(municipalities, anchor_roads)
    municipalities = municipalities.sort_values(
        ["municipality_code", "nearest_road_distance_km", "nearest_road_segment_id"],
        ascending=[True, True, True],
    ).drop_duplicates(subset=["municipality_code"], keep="first")
    municipalities = _fill_missing_anchor_points(municipalities, roads)
    roads_metric = roads.to_crs(epsg=3857).set_index("raw_segment_id")

    network = RoadNetwork()
    for row in municipalities.itertuples(index=False):
        population = 0 if pd.isna(row.pob_2025) else int(float(row.pob_2025))
        network.add_node(
            RoadNode(
                node_id=str(row.node_id),
                name=str(row.nombre),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                node_type=NodeKind.MUNICIPALITY.value,
                population=population,
            )
        )

    geo_nodes: Dict[str, Tuple[float, float]] = {}
    municipality_attachment_counts: Dict[str, int] = {}
    for row in edge_df.itertuples(index=False):
        start_code = str(getattr(row, "start_municipality_code", "") or "")
        end_code = str(getattr(row, "end_municipality_code", "") or "")
        start_is_covered = start_code in valid_codes
        end_is_covered = end_code in valid_codes
        if start_code and start_is_covered:
            start_node = _municipality_node_id(start_code)
            municipality_attachment_counts[start_code] = municipality_attachment_counts.get(start_code, 0) + 1
        else:
            start_node = _geo_endpoint_node_id(row.road_name, row.raw_segment_id, "START")
            geo_nodes[start_node] = (float(row.start_latitude), float(row.start_longitude))
        if end_code and end_is_covered:
            end_node = _municipality_node_id(end_code)
            municipality_attachment_counts[end_code] = municipality_attachment_counts.get(end_code, 0) + 1
        else:
            end_node = _geo_endpoint_node_id(row.road_name, row.raw_segment_id, "END")
            geo_nodes[end_node] = (float(row.end_latitude), float(row.end_longitude))

        for node_id in (start_node, end_node):
            if node_id in geo_nodes and node_id not in network.nodes:
                lat, lon = geo_nodes[node_id]
                network.add_node(
                    RoadNode(
                        node_id=node_id,
                        name=node_id,
                        latitude=float(lat),
                        longitude=float(lon),
                        node_type=NodeKind.GEO_ENDPOINT.value,
                        population=0,
                    )
                )

        edge = RoadEdge(
            edge_id=f"RAWROAD_{int(row.raw_segment_id)}",
            from_node=start_node,
            to_node=end_node,
            distance_km=float(row.length_km or 0.0),
            travel_time_min=max(float(row.length_km or 0.0) / 100.0 * 60.0, 0.1),
            road_type=str(row.road_type),
            road_name=str(row.road_name),
            from_road_km=float(row.PK_inicio or 0.0),
            to_road_km=float(row.PK_fin or 0.0),
            geometry_backed=True,
        )
        if start_node == end_node:
            network.add_edge(edge)
        else:
            network.add_undirected_road(edge)

    municipalities["endpoint_attachment_count"] = (
        municipalities["municipality_code"].map(municipality_attachment_counts).fillna(0).astype(int)
    )
    municipalities["has_endpoint_attachment"] = municipalities["endpoint_attachment_count"] > 0
    municipalities["node_id"] = municipalities["municipality_code"].map(_municipality_node_id)

    edge_df["start_node_id"] = np.where(
        edge_df["start_municipality_code"].fillna("").astype(str).isin(valid_codes),
        edge_df["start_municipality_code"].map(_municipality_node_id),
        edge_df.apply(
            lambda row: _geo_endpoint_node_id(row["road_name"], int(row["raw_segment_id"]), "START"),
            axis=1,
        ),
    )
    edge_df["end_node_id"] = np.where(
        edge_df["end_municipality_code"].fillna("").astype(str).isin(valid_codes),
        edge_df["end_municipality_code"].map(_municipality_node_id),
        edge_df.apply(
            lambda row: _geo_endpoint_node_id(row["road_name"], int(row["raw_segment_id"]), "END"),
            axis=1,
        ),
    )
    edge_df["start_endpoint_has_covered_municipality"] = edge_df["start_municipality_code"].fillna("").astype(str).isin(valid_codes)
    edge_df["end_endpoint_has_covered_municipality"] = edge_df["end_municipality_code"].fillna("").astype(str).isin(valid_codes)
    edge_df["is_cross_municipality"] = (
        edge_df["start_municipality_code"].fillna("").astype(str).ne("")
        & edge_df["end_municipality_code"].fillna("").astype(str).ne("")
        & edge_df["start_municipality_code"].astype(str).ne(edge_df["end_municipality_code"].astype(str))
    )
    edge_df["is_same_municipality"] = (
        edge_df["start_municipality_code"].fillna("").astype(str).ne("")
        & edge_df["start_municipality_code"].astype(str).eq(edge_df["end_municipality_code"].astype(str))
    )
    segment_start_node_by_id = edge_df.set_index("raw_segment_id")["start_node_id"].to_dict()
    segment_end_node_by_id = edge_df.set_index("raw_segment_id")["end_node_id"].to_dict()
    segment_name_by_id = edge_df.set_index("raw_segment_id")["road_name"].to_dict()
    segment_type_by_id = edge_df.set_index("raw_segment_id")["road_type"].to_dict()
    segment_length_by_id = edge_df.set_index("raw_segment_id")["length_km"].to_dict()

    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise RuntimeError("geopandas and shapely are required to attach municipality road anchors") from exc

    anchor_frame = municipalities[
        municipalities["nearest_road_segment_id"].notna()
        & municipalities["road_anchor_latitude"].notna()
        & municipalities["road_anchor_longitude"].notna()
    ][[
        "municipality_code",
        "node_id",
        "nearest_road_segment_id",
        "nearest_road_distance_km",
        "road_anchor_latitude",
        "road_anchor_longitude",
    ]].copy()
    if not anchor_frame.empty:
        anchor_gdf = gpd.GeoDataFrame(
            anchor_frame,
            geometry=gpd.points_from_xy(anchor_frame["road_anchor_longitude"], anchor_frame["road_anchor_latitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)
        municipalities["has_network_anchor"] = False
        anchored_codes: set[str] = set()
        for row in anchor_gdf.itertuples(index=False):
            municipality_code = str(row.municipality_code).zfill(5)
            anchored_codes.add(municipality_code)
            anchor_node_id = _road_anchor_node_id(municipality_code)
            if anchor_node_id not in network.nodes:
                network.add_node(
                    RoadNode(
                        node_id=anchor_node_id,
                        name=anchor_node_id,
                        latitude=float(row.road_anchor_latitude),
                        longitude=float(row.road_anchor_longitude),
                        node_type=NodeKind.ROAD_ANCHOR.value,
                        population=0,
                    )
                )

            connector_edge = RoadEdge(
                edge_id=f"CONNECTOR_{municipality_code}",
                from_node=str(row.node_id),
                to_node=anchor_node_id,
                distance_km=max(float(row.nearest_road_distance_km or 0.0), 0.001),
                travel_time_min=max(float(row.nearest_road_distance_km or 0.0) / 40.0 * 60.0, 0.1),
                road_type="connector",
                road_name=f"CONNECTOR_{municipality_code}",
                geometry_backed=False,
            )
            network.add_undirected_road(connector_edge)

            raw_segment_id = int(row.nearest_road_segment_id)
            road_geom_metric = roads_metric.loc[raw_segment_id, "geometry"]
            road_length_km = float(segment_length_by_id.get(raw_segment_id, 0.0) or 0.0)
            proj_m = float(road_geom_metric.project(row.geometry))
            total_m = max(float(road_geom_metric.length), 1e-6)
            distance_from_start_km = max(road_length_km * (proj_m / total_m), 0.001)
            distance_to_end_km = max(road_length_km - distance_from_start_km, 0.001)
            start_node_id = str(segment_start_node_by_id[raw_segment_id])
            end_node_id = str(segment_end_node_by_id[raw_segment_id])
            road_name = str(segment_name_by_id[raw_segment_id])
            road_type = str(segment_type_by_id[raw_segment_id])

            if anchor_node_id != start_node_id:
                network.add_undirected_road(
                    RoadEdge(
                        edge_id=f"ANCHORLINK_{municipality_code}_{raw_segment_id}_START",
                        from_node=anchor_node_id,
                        to_node=start_node_id,
                        distance_km=distance_from_start_km,
                        travel_time_min=max(distance_from_start_km / 100.0 * 60.0, 0.1),
                        road_type=road_type,
                        road_name=road_name,
                        geometry_backed=True,
                    )
                )
            if anchor_node_id != end_node_id:
                network.add_undirected_road(
                    RoadEdge(
                        edge_id=f"ANCHORLINK_{municipality_code}_{raw_segment_id}_END",
                        from_node=anchor_node_id,
                        to_node=end_node_id,
                        distance_km=distance_to_end_km,
                        travel_time_min=max(distance_to_end_km / 100.0 * 60.0, 0.1),
                        road_type=road_type,
                        road_name=road_name,
                        geometry_backed=True,
                    )
                )
        municipalities["municipality_code"] = municipalities["municipality_code"].astype(str).str.zfill(5)
        municipalities["has_network_anchor"] = municipalities["municipality_code"].isin(anchored_codes)
    else:
        municipalities["has_network_anchor"] = False

    edge_df = _attach_road_distance_to_covered_nodes(edge_df, municipalities, roads)

    top_municipalities = (
        municipalities.loc[municipalities["endpoint_attachment_count"] > 0, [
            "municipality_code",
            "nombre",
            "provincia",
            "pob_2025",
            "endpoint_attachment_count",
            "nearest_road",
            "nearest_road_distance_km",
        ]]
        .sort_values(["endpoint_attachment_count", "pob_2025"], ascending=[False, False])
        .head(20)
    )

    summary = {
        "raw_road_segments": int(len(edge_df)),
        "municipality_nodes": int(len(municipalities)),
        "covered_only": bool(covered_only),
        "geo_nodes": int(sum(1 for node in network.nodes.values() if node.node_type == NodeKind.GEO_ENDPOINT.value)),
        "directed_edges": int(network.edge_count),
        "start_exact_polygon_pct": float(100.0 * edge_df["start_assignment_method"].eq("exact_polygon").mean()),
        "end_exact_polygon_pct": float(100.0 * edge_df["end_assignment_method"].eq("exact_polygon").mean()),
        "start_geo_fallback_count": int(edge_df["start_assignment_method"].eq("geo_fallback").sum()),
        "end_geo_fallback_count": int(edge_df["end_assignment_method"].eq("geo_fallback").sum()),
        "same_municipality_segment_share_pct": float(100.0 * edge_df["is_same_municipality"].mean()),
        "cross_municipality_segment_share_pct": float(100.0 * edge_df["is_cross_municipality"].mean()),
        "municipalities_with_endpoint_attachment": int(municipalities["has_endpoint_attachment"].sum()),
        "municipalities_without_endpoint_attachment": int((~municipalities["has_endpoint_attachment"]).sum()),
        "municipalities_with_nearest_road_anchor": int(municipalities["nearest_road"].fillna("").astype(str).ne("").sum()),
        "municipalities_connected_via_anchor": int(municipalities["has_network_anchor"].sum()),
        "road_anchor_nodes": int(sum(1 for node in network.nodes.values() if node.node_type == NodeKind.ROAD_ANCHOR.value)),
        "roads_with_covered_endpoint_share_pct": float(
            100.0 * (
                edge_df["start_endpoint_has_covered_municipality"]
                | edge_df["end_endpoint_has_covered_municipality"]
            ).mean()
        ),
        "roads_far_from_covered_node_count": int(edge_df["road_far_from_covered_node"].fillna(False).sum()),
        "roads_far_from_covered_node_share_pct": float(100.0 * edge_df["road_far_from_covered_node"].fillna(False).mean()),
        "nearest_covered_municipality_distance_km_p50": float(edge_df["nearest_covered_municipality_distance_km"].median()),
        "nearest_covered_municipality_distance_km_p90": float(edge_df["nearest_covered_municipality_distance_km"].quantile(0.90)),
        "nearest_covered_municipality_distance_km_p99": float(edge_df["nearest_covered_municipality_distance_km"].quantile(0.99)),
        "nearest_road_distance_km_p50": float(municipalities["nearest_road_distance_km"].median()),
        "nearest_road_distance_km_p90": float(municipalities["nearest_road_distance_km"].quantile(0.90)),
        "nearest_road_distance_km_p99": float(municipalities["nearest_road_distance_km"].quantile(0.99)),
        "top_municipalities_by_endpoint_attachment": top_municipalities.to_dict(orient="records"),
        "top_roads_far_from_covered_nodes": (
            edge_df.sort_values(
                ["nearest_covered_municipality_distance_km", "length_km"],
                ascending=[False, False],
            )
            .head(20)[[
                "road_name",
                "raw_segment_id",
                "length_km",
                "nearest_covered_municipality_code",
                "nearest_covered_municipality_name",
                "nearest_covered_municipality_distance_km",
                "start_municipality_name",
                "end_municipality_name",
            ]]
            .to_dict(orient="records")
        ),
    }
    validate_network(network)
    return network, municipalities, edge_df, summary


def export_municipality_road_graph_diagnostics(
    data_dir: Path,
    out_dir: Path,
    mainland_only: bool = True,
    covered_only: bool = False,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    network, municipalities, edge_df, summary = build_municipality_road_graph(
        data_dir=data_dir,
        mainland_only=mainland_only,
        covered_only=covered_only,
    )

    municipality_cols = [
        "municipality_code",
        "province_code",
        "nombre",
        "provincia",
        "pob_2025",
        "latitude",
        "longitude",
        "node_id",
        "has_endpoint_attachment",
        "endpoint_attachment_count",
        "has_network_anchor",
        "nearest_road",
        "nearest_road_segment_id",
        "nearest_road_distance_km",
        "road_anchor_latitude",
        "road_anchor_longitude",
    ]
    municipalities[municipality_cols].rename(columns={
        "nombre": "municipality_name",
        "provincia": "province_name",
        "pob_2025": "population",
    }).to_csv(out_dir / "municipality_nodes.csv", index=False)

    edge_cols = [
        "raw_segment_id",
        "road_name",
        "road_type",
        "PK_inicio",
        "PK_fin",
        "length_km",
        "start_node_id",
        "end_node_id",
        "start_municipality_code",
        "start_municipality_name",
        "start_assignment_method",
        "start_assignment_distance_m",
        "end_municipality_code",
        "end_municipality_name",
        "end_assignment_method",
        "end_assignment_distance_m",
        "start_endpoint_has_covered_municipality",
        "end_endpoint_has_covered_municipality",
        "is_same_municipality",
        "is_cross_municipality",
        "nearest_covered_municipality_code",
        "nearest_covered_municipality_name",
        "nearest_covered_municipality_distance_km",
        "road_far_from_covered_node",
    ]
    edge_df[edge_cols].to_csv(out_dir / "road_segment_endpoint_municipalities.csv", index=False)

    municipalities[[
        "municipality_code",
        "nombre",
        "provincia",
        "has_network_anchor",
        "nearest_road",
        "nearest_road_segment_id",
        "nearest_road_distance_km",
        "road_anchor_latitude",
        "road_anchor_longitude",
    ]].rename(columns={
        "nombre": "municipality_name",
        "provincia": "province_name",
    }).to_csv(out_dir / "municipality_nearest_road_anchors.csv", index=False)
    edge_df[[
        "raw_segment_id",
        "road_name",
        "length_km",
        "nearest_covered_municipality_code",
        "nearest_covered_municipality_name",
        "nearest_covered_municipality_distance_km",
        "road_far_from_covered_node",
    ]].to_csv(out_dir / "road_segment_covered_municipality_proximity.csv", index=False)

    (out_dir / "municipality_road_graph_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Municipality-road diagnostics exported to %s (%d municipalities, %d raw road segments)",
        out_dir, len(municipalities), len(edge_df),
    )
    return {
        "network": network,
        "municipalities": municipalities,
        "road_segments": edge_df,
        "summary": summary,
    }
