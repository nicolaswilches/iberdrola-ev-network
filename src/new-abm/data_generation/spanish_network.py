"""
Real Spanish interurban network loader for the ABM.

Builds a RoadNetwork, a list of ChargingStation objects, and an ODMatrix
directly from the processed pipeline data produced by NB01–NB07.

Usage
-----
    from pathlib import Path
    from data_generation.spanish_network import build_spain_real_network

    data_dir = Path("../../data/processed")
    network, stations, od_matrix = build_spain_real_network(data_dir)

Data sources consumed (all in data/processed/)
----------------------------------------------
    road_segments_with_imd.csv         — 1 295 segments: road name, km
                                         markers, IMD counts, TEN-T tier
    demand_per_segment.csv             — 2027 BEV demand per segment
    interurban_chargers_baseline.csv   — existing ≥50 kW stations on
                                         interurban roads (NAP dataset)
    proposed_stations.csv              — 8 AFIR gap-fill stations from NB07

Design notes
------------
The network is corridor-level, not segment-level.  Each road is a linear
chain of city / hub nodes with station waypoints inserted between them.
This keeps the graph small enough (~200-350 nodes) for fast Dijkstra
routing while placing stations at geographically correct positions.

Station waypoints are inserted by projecting their lat/lon onto the road
polyline and sorting them by along-road km position.  The direct city-to-
city edge is REPLACED by a chain of sub-edges that passes through every
station on that road, ensuring stations appear on the optimal path and
are found naturally by plan_route_with_stops().

OD demand is derived from demand_per_segment.csv: per-road daily BEV
flows are summed and mapped to (origin_city, destination_city) pairs
defined in the corridor table below.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.demand import ODMatrix, ODPair
from models.network import RoadEdge, RoadNetwork, RoadNode
from models.station import ChargingStation

logger = logging.getLogger(__name__)

# Name of the roads geometry file produced by NB03 (relative to data/processed/)
_ROADS_PARQUET_FILENAME = "interurban_roads.parquet"


# ---------------------------------------------------------------------------
# City / hub node definitions — real WGS-84 coordinates
# ---------------------------------------------------------------------------
# Format: (node_id, display_name, lat, lon, node_type, population)
_CITY_NODES: List[Tuple] = [
    # Core Spanish cities (all present in synthetic network)
    ("MAD", "Madrid",                  40.416, -3.703, "city", 3_300_000),
    ("BCN", "Barcelona",               41.385,  2.173, "city", 1_600_000),
    ("VAL", "Valencia",                39.470, -0.376, "city",   800_000),
    ("SEV", "Sevilla",                 37.389, -5.984, "city",   690_000),
    ("BIL", "Bilbao",                  43.263, -2.935, "city",   350_000),
    ("ZAR", "Zaragoza",                41.649, -0.887, "city",   670_000),
    ("MAL", "Málaga",                  36.720, -4.420, "city",   570_000),
    ("MUR", "Murcia",                  37.983, -1.130, "city",   450_000),
    ("VLD", "Valladolid",              41.652, -4.724, "city",   300_000),
    ("ALI", "Alicante",                38.345, -0.490, "city",   330_000),
    ("GRN", "Granada",                 37.177, -3.599, "city",   230_000),
    ("COR", "Córdoba",                 37.888, -4.780, "city",   325_000),
    ("BUR", "Burgos",                  42.344, -3.697, "city",   180_000),
    ("PMP", "Pamplona",                42.820, -1.644, "city",   200_000),
    ("SSB", "San Sebastián",           43.321, -1.980, "city",   185_000),
    ("VIT", "Vitoria-Gasteiz",         42.849, -2.672, "city",   250_000),
    ("LLE", "Lleida",                  41.617,  0.620, "city",   140_000),
    ("TAR", "Tarragona",               41.119,  1.245, "city",   135_000),
    ("CAS", "Castellón de la Plana",   39.987, -0.050, "city",   170_000),
    ("ALB", "Albacete",                38.995, -1.856, "city",   175_000),
    # Additional cities for corridors with AFIR gaps
    ("TER", "Teruel",                  40.345, -1.107, "city",    35_000),  # A-23
    ("ACO", "A Coruña",                43.362, -8.412, "city",   245_000),  # AP-9
    ("SCQ", "Santiago de Compostela",  42.880, -8.545, "city",    96_000),  # AP-9
    ("VIG", "Vigo",                    42.232, -8.712, "city",   296_000),  # AP-9
    ("JAE", "Jaén",                    37.779, -3.790, "city",   117_000),  # N-322
    ("BAD", "Badajoz",                 38.876, -6.970, "city",   150_000),  # N-433
    ("MER", "Mérida",                  38.917, -6.342, "city",    58_000),  # N-433 / A-66
    ("HUE", "Huelva",                  37.261, -6.949, "city",   142_000),  # N-435
    ("AVL", "Ávila",                   40.657, -4.700, "city",    59_000),  # N-502
    ("TAL", "Talavera de la Reina",    39.762, -4.829, "city",    89_000),  # N-502
    ("PAL", "Palencia",                42.010, -4.527, "city",    78_000),  # N-621
    ("STD", "Santander",               43.463, -3.800, "city",   172_000),  # N-621
    ("LOG", "Logroño",                 42.466, -2.443, "city",   153_000),  # AP-68
    # Northern corridor & Galicia interior
    ("LEO", "León",                    42.599, -5.567, "city",   122_000),  # A-66, A-6, N-120
    ("OVI", "Oviedo",                  43.362, -5.844, "city",   219_000),  # A-66, A-8
    ("GIJ", "Gijón",                   43.545, -5.662, "city",   269_000),  # A-8
    ("PON", "Ponferrada",              42.548, -6.596, "city",    64_000),  # A-6, N-120
    ("LUG", "Lugo",                    43.012, -7.556, "city",    98_000),  # A-6, A-54
    ("OUR", "Ourense",                 42.336, -7.864, "city",   105_000),  # A-52, N-120
    # Mediterranean & south coast
    ("GIR", "Girona",                  41.983,  2.824, "city",   103_000),  # AP-7
    ("ALM", "Almería",                 36.834, -2.463, "city",   200_000),  # A-7, A-92, N-340
    ("CAR", "Cartagena",               37.605, -0.987, "city",   216_000),  # AP-7, N-340
    ("ALG", "Algeciras",               36.131, -5.453, "city",   122_000),  # A-7, N-340
    ("JER", "Jerez",                   36.686, -6.137, "city",   213_000),  # AP-4, A-4
    ("CDZ", "Cádiz",                   36.529, -6.292, "city",   113_000),  # AP-4, A-4
    # Ruta de la Plata / Extremadura / Castilla
    ("SAL", "Salamanca",               40.970, -5.664, "city",   143_000),  # A-66, A-62
    ("CAC", "Cáceres",                 39.476, -6.372, "city",    96_000),  # A-66
    ("ZAM", "Zamora",                  41.503, -5.744, "city",    61_000),  # A-66
    # Central Spain
    ("GUA", "Guadalajara",             40.632, -3.163, "city",    87_000),  # A-2, AP-2
    ("CUE", "Cuenca",                  40.071, -2.137, "city",    54_000),  # A-40
    ("SEG", "Segovia",                 40.949, -4.117, "city",    51_000),  # AP-6
    ("SOR", "Soria",                   41.764, -2.468, "city",    39_000),  # N-122, A-15
    ("CRE", "Ciudad Real",             38.986, -3.927, "city",    75_000),  # A-4, A-41
    # Aragón
    ("HUS", "Huesca",                  42.137, -0.408, "city",    53_000),  # A-23
]


# ---------------------------------------------------------------------------
# Road corridor definitions
# ---------------------------------------------------------------------------
# Format: (road_name, [ordered_node_ids_start → end])
# Station waypoints are inserted between adjacent city pairs during loading.
# All corridors listed here will be looked up in demand_per_segment.csv to
# obtain calibrated OD flows.
_ROAD_CORRIDORS: List[Tuple[str, List[str]]] = [
    # TEN-T Core motorways
    ("AP-2",  ["MAD", "GUA", "ZAR", "LLE", "TAR", "BCN"]),
    ("A-2",   ["MAD", "GUA", "ZAR", "LLE", "TAR", "BCN"]),
    ("AP-7",  ["GIR", "BCN", "TAR", "CAS", "VAL", "ALI", "MUR", "CAR"]),
    ("A-7",   ["GIR", "BCN", "TAR", "CAS", "VAL", "ALI", "MUR", "CAR", "ALM", "MAL", "ALG"]),
    ("A-7S",  ["ALG", "MAL", "ALI", "MUR"]),
    ("A-3",   ["MAD", "ALB", "VAL"]),
    ("AP-4",  ["MAD", "CRE", "COR", "SEV", "JER", "CDZ"]),
    ("A-4",   ["MAD", "CRE", "COR", "SEV", "JER", "CDZ"]),
    ("A-1",   ["MAD", "BUR", "VIT", "BIL"]),
    ("AP-1",  ["MAD", "BUR", "VIT", "BIL"]),
    ("A-6",   ["MAD", "VLD", "LEO", "PON", "LUG", "ACO"]),
    ("AP-68", ["ZAR", "LOG", "VIT", "BIL"]),
    ("A-68",  ["ZAR", "LOG", "VIT", "BIL"]),
    # A-8 Cantabrian autovía (BIL→GIJ). PMP removed — A-8 does not go to Pamplona.
    ("A-8",   ["BIL", "SSB", "STD", "OVI", "GIJ"]),
    # A-66 / N-630 Ruta de la Plata: SEV → GIJ via Puerto de Pajares.
    # VLD removed — VLD is not on A-66.
    ("A-66",  ["SEV", "MER", "CAC", "SAL", "ZAM", "LEO", "OVI", "GIJ"]),
    ("N-630", ["SEV", "MER", "CAC", "SAL", "ZAM", "LEO", "OVI", "GIJ"]),
    ("AP-46", ["MAL", "GRN"]),
    # Corridors with AFIR gaps (proposed stations placed on these)
    ("A-23",  ["VAL", "TER", "ZAR", "HUS"]),
    ("AP-9",  ["ACO", "SCQ", "VIG"]),
    ("N-322", ["ALB", "JAE", "COR"]),
    ("N-433", ["SEV", "MER", "BAD"]),
    ("N-435", ["HUE", "SEV"]),
    ("N-502", ["AVL", "TAL", "COR"]),
    ("N-621", ["PAL", "STD"]),
    # Further common corridors
    ("A-67",  ["PAL", "STD"]),
    # A-62 corrected: Salamanca–Valladolid–Burgos. MAD is not on A-62.
    ("A-62",  ["SAL", "VLD", "BUR"]),
    ("A-45",  ["COR", "MAL"]),
    # A-92 corrected: ends at Almería (not Murcia).
    ("A-92",  ["SEV", "GRN", "ALM"]),
    # N-340 coastal: full Mediterranean chain.
    ("N-340", ["ALI", "MUR", "CAR", "ALM", "MAL", "ALG"]),
    # Parallel autopistas kept distinct from their autovia counterparts
    ("AP-8",  ["BIL", "SSB"]),       # parallels A-8 Bilbao to San Sebastian
    ("AP-6",  ["MAD", "SEG", "VLD"]),# parallels A-6 Madrid to Valladolid via Segovia
    ("AP-15", ["PMP", "ZAR"]),       # Navarra autopista, joins AP-68 at Tudela
    # New corridors
    ("A-52",  ["OUR", "VIG"]),                                       # Rías Baixas autovía
    ("N-120", ["LOG", "BUR", "LEO", "PON", "OUR", "VIG"]),          # northern national road
    ("A-40",  ["MAD", "CUE"]),                                       # Madrid – Cuenca
    ("N-122", ["ZAR", "SOR", "VLD"]),                                # Zaragoza – Soria – Valladolid
]

# ---------------------------------------------------------------------------
# Known road distances (km) between adjacent city pairs
# ---------------------------------------------------------------------------
# Covers all adjacent-city pairs in _ROAD_CORRIDORS.
# The loader falls back to haversine × 1.15 for any pair not listed here.
_SEGMENT_KM: Dict[Tuple[str, str], float] = {
    # AP-2 / A-2: Madrid – Zaragoza – Lleida – Tarragona – Barcelona
    ("MAD", "ZAR"): 310, ("ZAR", "MAD"): 310,
    ("ZAR", "LLE"): 155, ("LLE", "ZAR"): 155,
    ("LLE", "TAR"):  95, ("TAR", "LLE"):  95,
    ("TAR", "BCN"):  95, ("BCN", "TAR"):  95,
    # AP-7 / A-7: Barcelona – Tarragona – Castellón – Valencia – Alicante – Murcia
    ("BCN", "TAR"):  95, ("TAR", "BCN"):  95,   # duplicate ok, dict dedup is fine
    ("TAR", "CAS"): 150, ("CAS", "TAR"): 150,
    ("CAS", "VAL"):  75, ("VAL", "CAS"):  75,
    ("VAL", "ALI"): 165, ("ALI", "VAL"): 165,
    ("ALI", "MUR"):  85, ("MUR", "ALI"):  85,
    # A-7S: Málaga – Alicante
    ("MAL", "ALI"): 320, ("ALI", "MAL"): 320,
    # A-3: Madrid – Albacete – Valencia
    ("MAD", "ALB"): 250, ("ALB", "MAD"): 250,
    ("ALB", "VAL"): 190, ("VAL", "ALB"): 190,
    # AP-4 / A-4: Madrid – Córdoba – Sevilla
    ("MAD", "COR"): 400, ("COR", "MAD"): 400,
    ("COR", "SEV"): 140, ("SEV", "COR"): 140,
    # A-1 / AP-1: Madrid – Burgos – Vitoria – Bilbao
    ("MAD", "BUR"): 240, ("BUR", "MAD"): 240,
    ("BUR", "VIT"): 110, ("VIT", "BUR"): 110,
    ("VIT", "BIL"):  60, ("BIL", "VIT"):  60,
    # A-6: Madrid – Valladolid
    ("MAD", "VLD"): 190, ("VLD", "MAD"): 190,
    # AP-68 / A-68: Zaragoza – Logroño – Vitoria – Bilbao
    ("ZAR", "LOG"): 170, ("LOG", "ZAR"): 170,
    ("LOG", "VIT"):  95, ("VIT", "LOG"):  95,
    # A-8: Bilbao – San Sebastián – Pamplona
    ("BIL", "SSB"):  95, ("SSB", "BIL"):  95,
    ("SSB", "PMP"):  80, ("PMP", "SSB"):  80,
    # A-66 / N-630: Sevilla – Mérida – Valladolid
    ("SEV", "MER"): 195, ("MER", "SEV"): 195,
    ("MER", "VLD"): 400, ("VLD", "MER"): 400,
    # AP-46: Málaga – Granada
    ("MAL", "GRN"): 130, ("GRN", "MAL"): 130,
    # A-23: Zaragoza – Teruel – Valencia
    ("ZAR", "TER"): 185, ("TER", "ZAR"): 185,
    ("TER", "VAL"): 145, ("VAL", "TER"): 145,
    # AP-9: A Coruña – Santiago – Vigo
    ("ACO", "SCQ"):  65, ("SCQ", "ACO"):  65,
    ("SCQ", "VIG"):  90, ("VIG", "SCQ"):  90,
    # N-322: Albacete – Jaén – Córdoba
    ("ALB", "JAE"): 200, ("JAE", "ALB"): 200,
    ("JAE", "COR"): 115, ("COR", "JAE"): 115,
    # N-433: Sevilla – Mérida – Badajoz
    ("MER", "BAD"):  65, ("BAD", "MER"):  65,
    # N-435: Huelva – Sevilla
    ("HUE", "SEV"):  92, ("SEV", "HUE"):  92,
    # N-502: Ávila – Talavera – Córdoba
    ("AVL", "TAL"): 135, ("TAL", "AVL"): 135,
    ("TAL", "COR"): 250, ("COR", "TAL"): 250,
    # N-621 / A-67: Palencia – Santander
    ("PAL", "STD"): 195, ("STD", "PAL"): 195,
    # A-62: Madrid – Valladolid – Burgos (uses VLD already above)
    ("VLD", "BUR"): 120, ("BUR", "VLD"): 120,
    # A-45: Córdoba – Málaga
    ("COR", "MAL"): 185, ("MAL", "COR"): 185,
    # A-92: Sevilla – Granada – Murcia
    ("SEV", "GRN"): 255, ("GRN", "SEV"): 255,
    ("GRN", "MUR"): 225, ("MUR", "GRN"): 225,
    # N-340 coastal: Alicante – Málaga
    ("ALI", "MAL"): 320, ("MAL", "ALI"): 320,
    # Cross-connections
    ("ZAR", "PMP"): 170, ("PMP", "ZAR"): 170,
    ("PAL", "BUR"):  95, ("BUR", "PAL"):  95,
    # --- Extensions for expanded corridor chains ---
    # A-66 / N-630 Ruta de la Plata: Sevilla → León
    ("MER", "CAC"):  75, ("CAC", "MER"):  75,
    ("CAC", "SAL"): 205, ("SAL", "CAC"): 205,
    ("SAL", "ZAM"):  65, ("ZAM", "SAL"):  65,
    ("ZAM", "LEO"): 110, ("LEO", "ZAM"): 110,
    # A-8 Cantabrian autovía: BIL → GIJ (via SSB, STD, OVI)
    ("SSB", "STD"): 195, ("STD", "SSB"): 195,
    ("STD", "OVI"): 195, ("OVI", "STD"): 195,
    ("OVI", "GIJ"):  28, ("GIJ", "OVI"):  28,
    # A-6 Madrid → A Coruña
    ("VLD", "LEO"): 140, ("LEO", "VLD"): 140,
    ("LEO", "PON"): 110, ("PON", "LEO"): 110,
    ("PON", "LUG"):  95, ("LUG", "PON"):  95,
    ("LUG", "ACO"):  95, ("ACO", "LUG"):  95,
    # A-52 / N-120 Galicia
    ("OUR", "VIG"): 105, ("VIG", "OUR"): 105,
    ("PON", "OUR"):  95, ("OUR", "PON"):  95,
    ("BUR", "LEO"): 195, ("LEO", "BUR"): 195,
    ("LOG", "BUR"): 145, ("BUR", "LOG"): 145,
    # A-7 southern extension: MUR → ALG
    ("MUR", "CAR"):  50, ("CAR", "MUR"):  50,
    ("CAR", "ALM"): 225, ("ALM", "CAR"): 225,
    ("ALM", "MAL"): 210, ("MAL", "ALM"): 210,
    ("MAL", "ALG"): 130, ("ALG", "MAL"): 130,
    # AP-7 northern extension via Girona
    ("GIR", "BCN"): 100, ("BCN", "GIR"): 100,
    # AP-4 / A-4 south extension: SEV → CDZ
    ("SEV", "JER"):  90, ("JER", "SEV"):  90,
    ("JER", "CDZ"):  35, ("CDZ", "JER"):  35,
    # A-23 northern extension to Huesca
    ("ZAR", "HUS"):  75, ("HUS", "ZAR"):  75,
    # A-62 corrected: SAL → VLD
    ("SAL", "VLD"): 115, ("VLD", "SAL"): 115,
    # A-2 / AP-2 via Guadalajara
    ("MAD", "GUA"):  56, ("GUA", "MAD"):  56,
    ("GUA", "ZAR"): 260, ("ZAR", "GUA"): 260,
    # AP-6 via Segovia
    ("MAD", "SEG"):  95, ("SEG", "MAD"):  95,
    ("SEG", "VLD"): 110, ("VLD", "SEG"): 110,
    # A-92 extension to Almería
    ("GRN", "ALM"): 165, ("ALM", "GRN"): 165,
    # A-4 Madrid – Ciudad Real – Córdoba (MAD-CRE already covered above as 190)
    ("MAD", "CRE"): 190, ("CRE", "MAD"): 190,
    ("CRE", "COR"): 225, ("COR", "CRE"): 225,
    # A-40 Madrid – Cuenca (MAD-CUE already above as 165)
    ("MAD", "CUE"): 165, ("CUE", "MAD"): 165,
    # N-122 Zaragoza – Soria – Valladolid
    ("ZAR", "SOR"): 160, ("SOR", "ZAR"): 160,
    ("SOR", "VLD"): 240, ("VLD", "SOR"): 240,
}

# Speed limits by road prefix (km/h) — used to derive travel time.
# AP- (autopista) and A- (autovía) are both 120 km/h dual carriageways.
# The difference is tolls (AP-) vs free (A-), not speed.
# N- (carretera nacional, older single/dual carriageway) → 90 km/h.
_SPEED_KMH: Dict[str, float] = {"AP": 140.0, "A": 120.0, "N": 90.0}

# Toll rates (EUR per km, one-way) for AP- roads still under concession in 2027.
# Sources: Ministerio de Transportes tariff tables; AP-1 and AP-4 concessions
# expired before 2027 and are now toll-free.  Default for unlisted AP- roads: 0.
_TOLL_EUR_PER_KM: Dict[str, float] = {
    "AP-2":  0.045,   # Zaragoza–Barcelona (Abertis/Acesa): ~€14 for 310 km
    "AP-7":  0.035,   # Mediterranean coast (Abertis/Aucat): ~€12 for 350 km
    "AP-68": 0.040,   # Bilbao–Zaragoza (Euskal Errepideak / Itinere)
    "AP-9":  0.025,   # Galicia ring roads (Autopistas do Atlántico)
    "AP-46": 0.050,   # Málaga–Granada short section (Aucosta)
    "AP-8":  0.030,   # Vizcaya section still tolled in 2027
    "AP-15": 0.040,   # Navarra autopista (Audenasa)
    # AP-1 (expired 2018), AP-4 (never tolled), AP-6 (expired 2019) are
    # intentionally absent → 0.
}

# Default price per kWh for existing stations (EUR) — NAP data lacks prices
_DEFAULT_PRICE_EUR_KWH = 0.40
# Default power for proposed stations (project constant)
_PROPOSED_STATION_POWER_KW = 150.0
_MAX_CLUSTER_SPAN_KM = 10.0
_MAX_CLUSTER_CONNECTORS = 24
_STATION_PROJECTION_TOLERANCE_KM = 5.0
_CITY_PROJECTION_TOLERANCE_KM = 75.0


@dataclass(frozen=True)
class RoadProjection:
    """Projection of a point onto a real road geometry."""

    road_km: float
    distance_km: float


@dataclass(frozen=True)
class EdgeGeometry:
    """Geometry-backed attributes for one ABM graph edge."""

    distance_km: float
    from_road_km: float
    to_road_km: float
    source_segment_ids: Tuple[int, ...]
    target_daily_bev_traffic_2027: float


class RoadGeometryIndex:
    """Strict geometry lookup for ABM real-road distances and traffic targets."""

    def __init__(self, roads_gdf, demand_df: pd.DataFrame) -> None:
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            from shapely.ops import linemerge, unary_union
        except ImportError as exc:
            raise RuntimeError("geopandas and shapely are required for real ABM geometry") from exc

        self._gpd = gpd
        self._point_cls = Point
        roads = roads_gdf.copy()
        if roads.crs is None:
            roads = roads.set_crs("EPSG:4326")
        roads = roads.to_crs("EPSG:25830")

        demand_cols = ["segment_id", "daily_bev_traffic_2027"]
        demand = demand_df[demand_cols].copy() if set(demand_cols).issubset(demand_df.columns) else pd.DataFrame(columns=demand_cols)
        roads = roads.merge(demand, on="segment_id", how="left")
        roads["daily_bev_traffic_2027"] = roads["daily_bev_traffic_2027"].fillna(0.0)

        self._roads: Dict[str, dict] = {}
        for road_name, grp in roads.groupby("Carretera"):
            if grp.empty:
                continue
            union_geom = unary_union(grp.geometry)
            merged = union_geom if union_geom.geom_type == "LineString" else linemerge(union_geom)
            if merged.is_empty:
                continue
            segs = []
            for _, row in grp.iterrows():
                geom = row.geometry
                start_m = float(merged.project(geom.interpolate(0.0, normalized=True)))
                end_m = float(merged.project(geom.interpolate(1.0, normalized=True)))
                lo_m, hi_m = sorted((start_m, end_m))
                segs.append({
                    "segment_id": int(row["segment_id"]),
                    "start_km": lo_m / 1000.0,
                    "end_km": hi_m / 1000.0,
                    "length_km": max(float(row.get("length_km", 0.0) or 0.0), (hi_m - lo_m) / 1000.0),
                    "daily_bev_traffic_2027": float(row.get("daily_bev_traffic_2027", 0.0) or 0.0),
                })
            self._roads[str(road_name)] = {
                "geometry": merged,
                "segments": segs,
                "length_km": float(merged.length / 1000.0),
            }

    @property
    def roads(self) -> set[str]:
        return set(self._roads)

    def project(self, road_name: str, lat: float, lon: float, tolerance_km: float) -> RoadProjection:
        road = self._roads.get(str(road_name))
        if road is None:
            raise ValueError(f"missing geometry for road {road_name}")

        import geopandas as gpd

        pt = gpd.GeoSeries([self._point_cls(lon, lat)], crs="EPSG:4326").to_crs("EPSG:25830").iloc[0]
        geom = road["geometry"]
        road_m = float(geom.project(pt))
        distance_km = float(geom.distance(pt) / 1000.0)
        if distance_km > tolerance_km:
            raise ValueError(
                f"point projects {distance_km:.1f} km from {road_name}, "
                f"above tolerance {tolerance_km:.1f} km"
            )
        return RoadProjection(road_km=road_m / 1000.0, distance_km=distance_km)

    def edge(self, road_name: str, from_km: float, to_km: float) -> EdgeGeometry:
        road = self._roads.get(str(road_name))
        if road is None:
            raise ValueError(f"missing geometry for road {road_name}")
        lo, hi = sorted((float(from_km), float(to_km)))
        distance_km = hi - lo
        if distance_km <= 0.01:
            raise ValueError(f"zero-length geometry edge on {road_name}: {from_km:.3f}->{to_km:.3f}")

        segment_ids: List[int] = []
        weighted_flow = 0.0
        overlap_total = 0.0
        for seg in road["segments"]:
            seg_lo = min(seg["start_km"], seg["end_km"])
            seg_hi = max(seg["start_km"], seg["end_km"])
            overlap = max(0.0, min(hi, seg_hi) - max(lo, seg_lo))
            if overlap <= 0:
                continue
            segment_ids.append(seg["segment_id"])
            weighted_flow += overlap * seg["daily_bev_traffic_2027"]
            overlap_total += overlap
        target_flow = weighted_flow / overlap_total if overlap_total > 0 else 0.0
        return EdgeGeometry(
            distance_km=distance_km,
            from_road_km=float(from_km),
            to_road_km=float(to_km),
            source_segment_ids=tuple(sorted(set(segment_ids))),
            target_daily_bev_traffic_2027=float(target_flow),
        )

    def length_km(self, road_name: str) -> float:
        road = self._roads.get(str(road_name))
        if road is None:
            raise ValueError(f"missing geometry for road {road_name}")
        return float(road["length_km"])

    def point_at_km(self, road_name: str, road_km: float) -> Tuple[float, float]:
        """Return (lat, lon) at an along-road km position."""
        road = self._roads.get(str(road_name))
        if road is None:
            raise ValueError(f"missing geometry for road {road_name}")
        geom = road["geometry"]
        point = geom.interpolate(max(0.0, min(float(road_km), self.length_km(road_name))) * 1000.0)
        wgs = self._gpd.GeoSeries([point], crs="EPSG:25830").to_crs("EPSG:4326").iloc[0]
        return float(wgs.y), float(wgs.x)


# ---------------------------------------------------------------------------
# Auto-corridor builder from NB03 road geometry
# ---------------------------------------------------------------------------

def _build_road_corridors(
    roads_parquet_path: Path,
    buffer_km: float = 25.0,
) -> Dict[str, List[str]]:
    """
    Build a complete {road_name: [ordered_city_codes]} mapping.

    Strategy (hybrid):
      1. Start with all hand-curated corridors from ``_ROAD_CORRIDORS``
         (manually verified; include terminal city nodes that the interurban
         geometry cannot capture, e.g. AP-2 ends before Barcelona in the parquet).
      2. Auto-detect corridors from geometry for every road in the parquet NOT
         already in the hand-curated list.  For each auto-detected road:
           a. Merge segment geometries into one line in UTM EPSG:25830.
           b. Find _CITY_NODES within ``buffer_km`` of the merged line.
           c. Order retained cities by along-route projection position.
           d. Keep only roads with ≥2 retained cities (OD signal required).
      3. Return the merged mapping (hand-curated ∪ auto-detected).

    Uses a 25 km buffer because regional roads often pass within 10-25 km of
    the nearest city node without entering the urban core.  Terminal cities of
    major motorways are preserved via the hand-curated fallback.

    Returns an empty dict (triggering static-list fallback in the caller) if
    geopandas is unavailable or the parquet file is missing.
    """
    try:
        import geopandas as gpd
        from shapely.ops import linemerge, unary_union
        from shapely.geometry import Point
    except ImportError:
        logger.warning(
            "geopandas not available — falling back to static _ROAD_CORRIDORS"
        )
        return {}

    if not roads_parquet_path.exists():
        logger.warning(
            "Roads parquet not found at %s — falling back to static _ROAD_CORRIDORS",
            roads_parquet_path,
        )
        return {}

    # Seed with hand-curated corridors (always kept verbatim)
    static_dict: Dict[str, List[str]] = {name: list(cities) for name, cities in _ROAD_CORRIDORS}

    # Load geometry in WGS84, reproject to metric UTM for Spain
    roads_gdf = gpd.read_parquet(roads_parquet_path).to_crs("EPSG:25830")

    # Convert city nodes to UTM Points
    city_pts_utm: Dict[str, object] = {}
    try:
        city_gdf = gpd.GeoDataFrame(
            [
                {"city_id": nid, "geometry": Point(lon, lat)}
                for nid, _, lat, lon, _, _ in _CITY_NODES
            ],
            crs="EPSG:4326",
        ).to_crs("EPSG:25830")
        for _, row in city_gdf.iterrows():
            city_pts_utm[row["city_id"]] = row["geometry"]
    except Exception as exc:
        logger.warning("Failed to reproject city nodes: %s — using hand-curated only", exc)
        return static_dict

    buffer_m = buffer_km * 1000.0
    auto_detected: Dict[str, List[str]] = {}
    skipped_curated = 0
    skipped_no_cities = 0

    for road_name, grp in roads_gdf.groupby("Carretera"):
        # Preserve hand-curated corridors exactly
        if road_name in static_dict:
            skipped_curated += 1
            continue

        union_geom = unary_union(grp.geometry)
        # linemerge only accepts collections; a single LineString can be used directly
        merged = union_geom if union_geom.geom_type == "LineString" else linemerge(union_geom)
        if merged.is_empty:
            continue

        nearby: List[Tuple[float, str]] = []  # (along_route_m, city_id)
        for city_id, city_pt in city_pts_utm.items():
            if merged.distance(city_pt) <= buffer_m:
                proj_m = merged.project(city_pt)
                nearby.append((proj_m, city_id))

        if len(nearby) < 2:
            skipped_no_cities += 1
            continue

        nearby.sort(key=lambda x: x[0])
        auto_detected[road_name] = [city_id for _, city_id in nearby]

    merged_corridors = {**static_dict, **auto_detected}

    logger.info(
        "Road corridors: %d hand-curated + %d auto-detected = %d total "
        "(buffer %.1f km; %d parquet roads had <2 city hits)",
        len(static_dict), len(auto_detected), len(merged_corridors),
        buffer_km, skipped_no_cities,
    )

    return merged_corridors


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_spain_real_network(
    data_dir: Path,
    rng: Optional[np.random.Generator] = None,
    include_proposed_stations: bool = True,
    cluster_stations_per_group: int = 10,
) -> Tuple[RoadNetwork, List[ChargingStation], ODMatrix]:
    """
    Build the real Spanish interurban network from processed pipeline data.

    Parameters
    ----------
    data_dir:
        Path to ``data/processed/`` directory containing the pipeline CSVs.
        Raises FileNotFoundError with a helpful message if required files
        are missing.
    rng:
        NumPy random generator.  Defaults to seed 42 if not supplied.
    include_proposed_stations:
        When True, the 8 AFIR gap-fill stations from ``proposed_stations.csv``
        are added as waypoint nodes on their respective corridors.
    cluster_stations_per_group:
        Target number of real stations per cluster group.  Roads with fewer
        than this many stations get full resolution (one node per station).
        Larger roads get ``ceil(n_stations / cluster_stations_per_group)``
        clusters.  Default 10 balances accuracy with graph size.

    Returns
    -------
    (network, stations, od_matrix)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    data_dir = Path(data_dir)
    segments_df, demand_df, chargers_df, proposed_df = _load_data(data_dir)

    roads_gdf = _load_roads_geometry(data_dir)
    geometry_index = RoadGeometryIndex(roads_gdf, demand_df)

    # 1. Road corridors: try geometry-based auto-builder, fall back to static list
    auto_corridors = _build_road_corridors(data_dir / _ROADS_PARQUET_FILENAME)
    if auto_corridors:
        corridors: Dict[str, List[str]] = auto_corridors
    else:
        logger.info("Using static _ROAD_CORRIDORS (%d corridors)", len(_ROAD_CORRIDORS))
        corridors = {name: cities for name, cities in _ROAD_CORRIDORS}
    corridors = _add_geometry_fallback_corridors(corridors, geometry_index)

    # 2. Build city-only graph (nodes, no edges yet)
    network = _build_city_graph()

    # 3. Build corridor edge chains, insert station waypoints, collect stations
    stations = _build_corridors_and_stations(
        network=network,
        chargers_df=chargers_df,
        proposed_df=proposed_df,
        corridors=corridors,
        include_proposed=include_proposed_stations,
        cluster_stations_per_group=cluster_stations_per_group,
        geometry_index=geometry_index,
    )
    _validate_geometry_backed_graph(network)

    # 4. OD matrix calibrated to 2027 BEV demand
    od_matrix = _build_od_matrix(demand_df, network, corridors)

    logger.info(
        "Real network ready: %d nodes, %d directed edges, %d stations, "
        "%d OD pairs (%.0f daily BEV trips)",
        network.node_count, network.edge_count,
        len(stations),
        len(od_matrix.pairs), od_matrix.total_daily_trips(),
    )
    return network, stations, od_matrix


def _validate_geometry_backed_graph(network: RoadNetwork) -> None:
    """Fail real-mode builds if any ABM edge is not backed by road geometry."""
    failures = []
    for u, v, attrs in network.graph.edges(data=True):
        if not attrs.get("geometry_backed", False):
            failures.append((u, v, "missing geometry_backed flag"))
        elif not math.isfinite(float(attrs.get("distance_km", 0.0))) or float(attrs.get("distance_km", 0.0)) <= 0:
            failures.append((u, v, "non-positive distance"))
        elif not attrs.get("road_name"):
            failures.append((u, v, "missing road_name"))
    if failures:
        preview = "; ".join(f"{u}->{v}: {reason}" for u, v, reason in failures[:10])
        raise RuntimeError(
            f"ABM real network has {len(failures)} non-geometry-backed edges. "
            f"First failures: {preview}"
        )


def _geo_node_id(road_name: str, suffix: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", road_name).strip("_")
    return f"GEO_{clean}_{suffix}"


def _add_geometry_fallback_corridors(
    corridors: Dict[str, List[str]],
    geometry_index: RoadGeometryIndex,
) -> Dict[str, List[str]]:
    """Ensure every processed geometry road has synthetic full-road endpoints."""
    out = {road: list(nodes) for road, nodes in corridors.items()}
    for road_name in sorted(geometry_index.roads):
        start_id = _geo_node_id(road_name, "START")
        end_id = _geo_node_id(road_name, "END")
        nodes = out.setdefault(road_name, [])
        if start_id not in nodes:
            nodes.insert(0, start_id)
        if end_id not in nodes:
            nodes.append(end_id)
    return out


def _ensure_geo_endpoint_node(
    network: RoadNetwork,
    geometry_index: RoadGeometryIndex,
    road_name: str,
    suffix: str,
) -> str:
    node_id = _geo_node_id(road_name, suffix)
    if node_id in network.nodes:
        return node_id
    road_km = 0.0 if suffix == "START" else geometry_index.length_km(road_name)
    lat, lon = geometry_index.point_at_km(road_name, road_km)
    network.add_node(
        RoadNode(
            node_id=node_id,
            name=f"{road_name} geometry {suffix.lower()}",
            latitude=lat,
            longitude=lon,
            node_type="junction",
            population=0,
        )
    )
    return node_id


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(
    data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four pipeline CSVs required by this module."""
    required = {
        "road_segments_with_imd.csv":       "road segments with IMD data",
        "demand_per_segment.csv":           "2027 BEV demand per segment",
        "interurban_chargers_baseline.csv": "existing interurban chargers",
        "proposed_stations.csv":            "proposed AFIR gap-fill stations",
    }
    for fname, desc in required.items():
        if not (data_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing {desc} at {data_dir / fname}.\n"
                "Run the NB01–NB07 pipeline first, or use "
                "build_spain_demo_network() for a synthetic fallback."
            )

    segments_df = pd.read_csv(data_dir / "road_segments_with_imd.csv")
    demand_df   = pd.read_csv(data_dir / "demand_per_segment.csv")
    chargers_df = pd.read_csv(data_dir / "interurban_chargers_baseline.csv")
    proposed_df = pd.read_csv(data_dir / "proposed_stations.csv")

    logger.info(
        "Loaded pipeline data: %d road segments, %d demand rows, "
        "%d baseline chargers, %d proposed stations",
        len(segments_df), len(demand_df), len(chargers_df), len(proposed_df),
    )
    return segments_df, demand_df, chargers_df, proposed_df


def _load_roads_geometry(data_dir: Path):
    """Load processed interurban road geometry required by strict real ABM mode."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to build the real ABM network") from exc

    path = data_dir / _ROADS_PARQUET_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing road geometry at {path}")
    return gpd.read_parquet(path)


# ---------------------------------------------------------------------------
# Road-level aggregate lengths
# ---------------------------------------------------------------------------

def _compute_road_lengths(segments_df: pd.DataFrame) -> Dict[str, float]:
    """
    Sum ``length_km`` by road name (``Carretera`` column).

    Used to scale OD demand and as a sanity check against the corridor
    distances hard-coded in ``_SEGMENT_KM``.
    """
    if "Carretera" not in segments_df.columns or "length_km" not in segments_df.columns:
        return {}
    road_lengths = (
        segments_df.groupby("Carretera")["length_km"]
        .sum()
        .to_dict()
    )
    logger.debug("Road lengths computed for %d unique roads", len(road_lengths))
    return road_lengths


# ---------------------------------------------------------------------------
# City graph
# ---------------------------------------------------------------------------

def _build_city_graph() -> RoadNetwork:
    """Create a RoadNetwork populated with city nodes only (no edges)."""
    network = RoadNetwork()
    for nid, name, lat, lon, ntype, pop in _CITY_NODES:
        network.add_node(RoadNode(nid, name, lat, lon, ntype, pop))
    logger.debug("City graph: %d nodes", network.node_count)
    return network


# ---------------------------------------------------------------------------
# Corridor building + station placement
# ---------------------------------------------------------------------------

def _build_corridors_and_stations(
    network: RoadNetwork,
    chargers_df: pd.DataFrame,
    proposed_df: pd.DataFrame,
    corridors: Dict[str, List[str]],
    include_proposed: bool,
    cluster_stations_per_group: int,
    geometry_index: RoadGeometryIndex,
) -> List[ChargingStation]:
    """
    For every corridor in *corridors*:
      1. Cluster baseline chargers on that road into dynamic-sized groups.
      2. Optionally insert proposed stations as waypoints.
      3. Add all waypoint nodes to *network*.
      4. Build a chain of edges through city nodes + station waypoints in
         along-road order (removing the unbroken city-to-city edges and
         replacing them with sub-edges through intermediate stops).

    Returns
    -------
    List of ChargingStation objects for every station waypoint added.
    """
    all_stations: List[ChargingStation] = []
    # Track which undirected edges have already been added to avoid duplicates
    added_edges: set = set()

    for road_name, city_chain in corridors.items():
        if road_name not in geometry_index.roads:
            logger.info("Corridor %s skipped: no processed road geometry", road_name)
            continue
        # --- verify all cities exist; synthetic GEO endpoints are added lazily ---
        materialized_chain: List[str] = []
        for node_id in city_chain:
            if node_id in network.nodes:
                materialized_chain.append(node_id)
            elif node_id == _geo_node_id(road_name, "START"):
                materialized_chain.append(
                    _ensure_geo_endpoint_node(network, geometry_index, road_name, "START")
                )
            elif node_id == _geo_node_id(road_name, "END"):
                materialized_chain.append(
                    _ensure_geo_endpoint_node(network, geometry_index, road_name, "END")
                )
        city_chain = materialized_chain
        if len(city_chain) < 2:
            logger.debug("Corridor %s skipped: fewer than 2 valid cities", road_name)
            continue

        # --- collect station waypoints for this road ---
        waypoints: List[Tuple[float, str, float, float]] = []
        # Each entry: (km_along_road, node_id, lat, lon)

        # A) Existing baseline charger clusters
        charger_wps, charger_stations = _cluster_chargers_on_road(
            road_name=road_name,
            chargers_df=chargers_df,
            geometry_index=geometry_index,
            cluster_stations_per_group=cluster_stations_per_group,
        )
        waypoints.extend(charger_wps)
        all_stations.extend(charger_stations)

        # B) Proposed stations
        if include_proposed:
            prop_wps, prop_stations = _get_proposed_on_road(
                road_name=road_name,
                proposed_df=proposed_df,
                geometry_index=geometry_index,
            )
            waypoints.extend(prop_wps)
            all_stations.extend(prop_stations)

        # --- register waypoint nodes in the network ---
        for _, nid, lat, lon in waypoints:
            if nid not in network.nodes:
                network.add_node(
                    RoadNode(
                        node_id=nid,
                        name=nid,
                        latitude=lat,
                        longitude=lon,
                        node_type="junction",
                        population=0,
                    )
                )

        # --- build full ordered chain: cities + waypoints ---
        road_prefix = road_name.split("-")[0]
        speed = _SPEED_KMH.get(road_prefix, 90.0)

        # Assign km positions to city nodes along the corridor
        city_positions: List[Tuple[float, str]] = []
        for nid in city_chain:
            node = network.nodes[nid]
            try:
                proj = geometry_index.project(
                    road_name, node.latitude, node.longitude, _CITY_PROJECTION_TOLERANCE_KM
                )
            except ValueError as exc:
                logger.info("Skipping city %s on %s: %s", nid, road_name, exc)
                continue
            city_positions.append((proj.road_km, nid))

        if len(city_positions) < 2:
            logger.info("Corridor %s using synthetic geometry endpoints", road_name)
            start_id = _ensure_geo_endpoint_node(network, geometry_index, road_name, "START")
            end_id = _ensure_geo_endpoint_node(network, geometry_index, road_name, "END")
            city_positions = [
                (0.0, start_id),
                (geometry_index.length_km(road_name), end_id),
            ]

        # Merge cities + waypoints, sort by km position
        all_nodes: List[Tuple[float, str]] = city_positions + [
            (km, nid) for km, nid, _, _ in waypoints
        ]
        all_nodes.sort(key=lambda x: x[0])

        # Deduplicate by node_id (a proposed station on a city node would
        # appear twice otherwise)
        seen: set = set()
        deduped: List[Tuple[float, str]] = []
        for km, nid in all_nodes:
            if nid not in seen:
                seen.add(nid)
                deduped.append((km, nid))

        # Toll rate for this road (0 for free roads)
        toll_rate = _TOLL_EUR_PER_KM.get(road_name, 0.0)

        # Build directed edges between consecutive nodes in the chain
        for i in range(len(deduped) - 1):
            km_a, nid_a = deduped[i]
            km_b, nid_b = deduped[i + 1]
            if abs(km_b - km_a) <= 0.01:
                logger.debug("Skipping near-zero edge on %s: %s -> %s", road_name, nid_a, nid_b)
                continue
            geom_edge = geometry_index.edge(road_name, km_a, km_b)
            seg_km = geom_edge.distance_km
            travel_time = seg_km / speed * 60.0

            edge_key = (nid_a, nid_b, road_name)
            if edge_key in added_edges:
                continue
            added_edges.add(edge_key)
            added_edges.add((nid_b, nid_a, road_name))

            edge = RoadEdge(
                edge_id=f"{road_name}_E{i:03d}_{nid_a}_{nid_b}",
                from_node=nid_a,
                to_node=nid_b,
                distance_km=seg_km,
                travel_time_min=travel_time,
                road_type=road_prefix,
                speed_limit_kmh=speed,
                slope_grade=0.0,
                toll_eur=seg_km * toll_rate,
                road_name=road_name,
                from_road_km=geom_edge.from_road_km,
                to_road_km=geom_edge.to_road_km,
                source_segment_ids=geom_edge.source_segment_ids,
                target_daily_bev_traffic_2027=geom_edge.target_daily_bev_traffic_2027,
                geometry_backed=True,
            )
            try:
                network.add_undirected_road(edge)
            except ValueError as exc:
                logger.debug("Edge skipped (%s→%s): %s", nid_a, nid_b, exc)

    logger.info(
        "Corridors built: %d nodes, %d directed edges, %d station waypoints",
        network.node_count, network.edge_count, len(all_stations),
    )
    return all_stations


# ---------------------------------------------------------------------------
# Existing charger clustering
# ---------------------------------------------------------------------------

def _normalise_road(name: str) -> str:
    """
    Normalise road name for fuzzy matching.

    AP-X, A-X, and suffixed variants present in the processed geometry are
    kept distinct so strict geometry projection cannot assign chargers to a
    different physical road. Only AP-9 F/V variants are merged.
    """
    name = name.strip()
    # AP-9 has known F and V sub-routes that belong to the same corridor.
    name = re.sub(r'^(AP-9)[FVfv]$', r'\1', name)
    return name


def _cluster_chargers_on_road(
    road_name: str,
    chargers_df: pd.DataFrame,
    geometry_index: RoadGeometryIndex,
    cluster_stations_per_group: int = 10,
) -> Tuple[List[Tuple[float, str, float, float]], List[ChargingStation]]:
    """
    Cluster baseline chargers on *road_name* by true along-road position.

    Clusters are capped at 10 km road span and 24 connectors. A single
    physical station over the connector cap remains one waypoint and is
    flagged via dynamic metadata on the ChargingStation object.
    """
    if "nearest_road" not in chargers_df.columns:
        return [], []

    norm_target = _normalise_road(road_name)
    norm_col = chargers_df["nearest_road"].fillna("").apply(_normalise_road)
    mask = norm_col == norm_target
    road_ch = chargers_df[mask].copy()

    if road_ch.empty:
        return [], []

    road_ch = road_ch.copy()
    projected_rows = []
    skipped = 0
    for _, row in road_ch.iterrows():
        try:
            proj = geometry_index.project(
                road_name, float(row["latitude"]), float(row["longitude"]),
                _STATION_PROJECTION_TOLERANCE_KM
            )
        except ValueError:
            skipped += 1
            continue
        row = row.copy()
        row["_road_km"] = proj.road_km
        projected_rows.append(row)
    if skipped:
        logger.info("Skipped %d baseline chargers on %s outside projection tolerance", skipped, road_name)
    if not projected_rows:
        return [], []
    road_ch = pd.DataFrame(projected_rows)
    road_ch = road_ch.sort_values("_road_km").reset_index(drop=True)

    waypoints: List[Tuple[float, str, float, float]] = []
    stations: List[ChargingStation] = []

    clusters: List[pd.DataFrame] = []
    current_rows = []
    current_connectors = 0
    current_start_km: Optional[float] = None

    for _, row in road_ch.iterrows():
        connectors = int(row.get("n_connectors", 2) if not pd.isna(row.get("n_connectors", 2)) else 2)
        connectors = max(1, connectors)
        road_km = float(row["_road_km"])
        single_over_cap = connectors > _MAX_CLUSTER_CONNECTORS

        should_flush = False
        if current_rows and current_start_km is not None:
            span_exceeded = road_km - current_start_km > _MAX_CLUSTER_SPAN_KM
            cap_exceeded = current_connectors + connectors > _MAX_CLUSTER_CONNECTORS
            should_flush = span_exceeded or cap_exceeded or single_over_cap
        if should_flush:
            clusters.append(pd.DataFrame(current_rows))
            current_rows = []
            current_connectors = 0
            current_start_km = None

        current_rows.append(row)
        current_connectors += connectors
        current_start_km = road_km if current_start_km is None else current_start_km

        if single_over_cap:
            clusters.append(pd.DataFrame(current_rows))
            current_rows = []
            current_connectors = 0
            current_start_km = None

    if current_rows:
        clusters.append(pd.DataFrame(current_rows))

    for cid, grp in enumerate(clusters):
        lat = float(grp["latitude"].median())
        lon = float(grp["longitude"].median())
        n_connectors = int(grp["n_connectors"].fillna(2).astype(int).sum())
        n_connectors = max(1, n_connectors)
        max_power = float(grp["max_power_kw"].fillna(50).max())

        km_pos = float(grp["_road_km"].median())
        node_id = f"{road_name}_EC{cid:03d}"
        station_id = f"EC_{road_name}_{cid:03d}"
        span_km = float(grp["_road_km"].max() - grp["_road_km"].min())
        exception = (
            "single_station_over_cap"
            if len(grp) == 1 and n_connectors > _MAX_CLUSTER_CONNECTORS
            else ""
        )

        waypoints.append((km_pos, node_id, lat, lon))
        station = ChargingStation(
            station_id=station_id,
            node_id=node_id,
            name=f"{road_name} existing cluster {cid}",
            latitude=lat,
            longitude=lon,
            max_power_kw=max_power,
            num_connectors=n_connectors,
            price_per_kwh=_DEFAULT_PRICE_EUR_KWH,
            reliability=0.95,
        )
        station.road_name = road_name
        station.road_km = km_pos
        station.cluster_span_km = span_km
        station.physical_station_count = int(len(grp))
        station.cluster_exception = exception
        stations.append(station)

    return waypoints, stations


# ---------------------------------------------------------------------------
# Proposed station placement
# ---------------------------------------------------------------------------

def _get_proposed_on_road(
    road_name: str,
    proposed_df: pd.DataFrame,
    geometry_index: RoadGeometryIndex,
) -> Tuple[List[Tuple[float, str, float, float]], List[ChargingStation]]:
    """
    Return waypoint descriptors + ChargingStation objects for any proposed
    stations whose ``route_segment`` matches *road_name*.
    """
    mask = proposed_df["route_segment"].fillna("").str.strip() == road_name.strip()
    road_prop = proposed_df[mask]

    if road_prop.empty:
        return [], []

    waypoints: List[Tuple[float, str, float, float]] = []
    stations: List[ChargingStation] = []

    for _, row in road_prop.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        node_id = str(row["location_id"])   # e.g. "STA_0001"
        n_chargers = int(row.get("n_chargers_proposed", 4))

        km_pos = geometry_index.project(
            road_name, lat, lon, _STATION_PROJECTION_TOLERANCE_KM
        ).road_km

        waypoints.append((km_pos, node_id, lat, lon))
        station = ChargingStation(
            station_id=node_id,
            node_id=node_id,
            name=f"Proposed {road_name} ({node_id})",
            latitude=lat,
            longitude=lon,
            max_power_kw=_PROPOSED_STATION_POWER_KW,
            num_connectors=n_chargers,
            price_per_kwh=_DEFAULT_PRICE_EUR_KWH,
            reliability=0.95,
        )
        station.road_name = road_name
        station.road_km = km_pos
        station.cluster_span_km = 0.0
        station.physical_station_count = 1
        station.cluster_exception = ""
        stations.append(station)

    return waypoints, stations


# ---------------------------------------------------------------------------
# OD matrix from demand data
# ---------------------------------------------------------------------------

def _build_od_matrix(
    demand_df: pd.DataFrame,
    network: RoadNetwork,
    corridors: Dict[str, List[str]],
) -> ODMatrix:
    """
    Build an ODMatrix calibrated to 2027 BEV demand from
    ``demand_per_segment.csv``.

    Emits an ODPair for **every unordered city pair** in every hand-curated
    corridor chain (not only adjacent city pairs), so long-haul trips like
    MAD↔BCN are first-class origins and destinations alongside short-hop
    pairs like MAD↔GUA. Pair flow follows an inverse-distance (1/L) gravity
    weighting so shorter intra-corridor trips are proportionally more common
    than end-to-end trips — consistent with observed interurban travel
    distributions.

    Both hand-curated corridors from ``_ROAD_CORRIDORS`` AND geometry-based
    auto-detected corridors (from ``_build_road_corridors``) emit OD pairs,
    giving full coverage of the ~150-road interurban network. Auto-detected
    corridors with fewer than 2 SOC-viable chargers along their chain are
    still emitted but receive reduced demand (50%) to avoid routing thrash
    from agents that cannot complete SOC-feasible paths.
    """
    if "route_segment" not in demand_df.columns or \
       "daily_bev_traffic_2027" not in demand_df.columns:
        logger.warning(
            "demand_per_segment.csv missing required columns; "
            "returning empty ODMatrix"
        )
        return ODMatrix()

    # Length-weighted mean per road: approximates avg vehicles passing any point
    # on the corridor, which proxies unique end-to-end trips. Plain .sum() would
    # count every through-trip once per segment it traverses (~25x inflation).
    def _lw_mean(grp: pd.DataFrame) -> float:
        total_len = grp["length_km"].sum()
        if total_len <= 0:
            return float(grp["daily_bev_traffic_2027"].mean())
        return float(
            (grp["daily_bev_traffic_2027"] * grp["length_km"]).sum() / total_len
        )

    road_demand = (
        demand_df.groupby("route_segment")
        .apply(_lw_mean)
        .to_dict()
    )

    # All corridors (hand-curated + auto-detected) contribute to OD emission.
    # Auto-detected corridors receive a 0.5× demand dampening so any charger-
    # sparse chain cannot dominate routing traffic and thrash the simulator.
    static_names = {name for name, _ in _ROAD_CORRIDORS}
    AUTO_DEMAND_SCALE = 0.5

    od = ODMatrix()
    unmatched_flow = 0.0
    matched_flow = 0.0
    matched_roads = 0
    auto_matched_roads = 0

    for road_name, total_flow in road_demand.items():
        is_auto = road_name not in static_names
        if is_auto:
            total_flow = total_flow * AUTO_DEMAND_SCALE

        nodes = corridors.get(road_name, [])
        valid = [n for n in nodes if n in network.nodes]
        if len(valid) < 2:
            unmatched_flow += total_flow
            logger.debug(
                "Unmapped road %s: chain has <2 valid cities (%.0f trips dropped)",
                road_name, total_flow,
            )
            continue

        seg_lens = [
            _get_segment_km(valid[i], valid[i + 1], network)
            for i in range(len(valid) - 1)
        ]
        total_len = sum(seg_lens) or 1.0
        purpose = _infer_purpose(road_name)
        n = len(valid)

        # Enumerate all unordered city pairs (i, j) with i < j. Pair length is
        # the along-corridor distance from city i to city j.
        pair_lens: Dict[Tuple[int, int], float] = {}
        for i in range(n):
            cumulative = 0.0
            for j in range(i + 1, n):
                cumulative += seg_lens[j - 1]
                pair_lens[(i, j)] = cumulative

        # Gravity weights: flow ∝ 1 / pair_length. Normalise so the sum of all
        # one-way pair flows on this corridor equals ``total_flow`` (preserves
        # the IMD-calibrated corridor-level trip volume from iter_02 logic).
        weights = {k: 1.0 / max(lg, 1.0) for k, lg in pair_lens.items()}
        total_weight = sum(weights.values()) or 1.0

        for (i, j), w in weights.items():
            pair_flow = float(total_flow * w / total_weight)
            if pair_flow < 0.5:
                continue
            o, d = valid[i], valid[j]
            od.add_pair(ODPair(
                origin=o, destination=d,
                daily_bev_trips=pair_flow, purpose=purpose,
            ))
            od.add_pair(ODPair(
                origin=d, destination=o,
                daily_bev_trips=pair_flow, purpose=purpose,
            ))

        matched_flow += total_flow
        matched_roads += 1
        if is_auto:
            auto_matched_roads += 1

    logger.info(
        "OD demand by corridor type: %d hand-curated + %d auto-detected = %d total",
        matched_roads - auto_matched_roads, auto_matched_roads, matched_roads,
    )

    # ------------------------------------------------------------------
    # Cross-corridor OD pairs (gravity model)
    # ------------------------------------------------------------------
    # Fill in city-pairs that don't share a hand-curated corridor (e.g.
    # BCN↔SEV, SEV↔MAL, MAD↔VIG) but are geographically within BEV range.
    # Flow is weighted by population gravity:  f ∝ pop_i * pop_j / d^2 ,
    # capped at 700 km graph distance, and normalised so the total cross-
    # corridor flow equals CROSS_FRACTION of the corridor-based flow.
    CROSS_FRACTION = 0.15
    city_info: Dict[str, Tuple[float, float, int]] = {
        code: (lat, lon, pop) for code, _, lat, lon, _, pop in _CITY_NODES
    }
    existing_pairs = {(p.origin, p.destination) for p in od.pairs}

    raw_cross: List[Tuple[str, str, float, float]] = []
    ids = [c for c in city_info if c in network.nodes]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ci, cj = ids[i], ids[j]
            if (ci, cj) in existing_pairs or (cj, ci) in existing_pairs:
                continue
            try:
                gdist = network.shortest_path_length(
                    ci, cj, weight="distance_km"
                )
            except Exception:
                continue
            if gdist <= 0.0 or gdist > 700.0:
                continue
            pop_i = city_info[ci][2]
            pop_j = city_info[cj][2]
            weight = (pop_i * pop_j) / (gdist * gdist)
            raw_cross.append((ci, cj, gdist, weight))

    target_cross_flow = matched_flow * CROSS_FRACTION
    total_cross_weight = sum(w for _, _, _, w in raw_cross) or 1.0
    cross_scale = target_cross_flow / total_cross_weight
    cross_added = 0
    cross_flow_total = 0.0
    for ci, cj, _gd, w in raw_cross:
        pair_flow = cross_scale * w
        if pair_flow < 0.5:
            continue
        od.add_pair(ODPair(
            origin=ci, destination=cj,
            daily_bev_trips=pair_flow, purpose="leisure",
        ))
        od.add_pair(ODPair(
            origin=cj, destination=ci,
            daily_bev_trips=pair_flow, purpose="leisure",
        ))
        cross_added += 1
        cross_flow_total += 2 * pair_flow

    logger.info(
        "Cross-corridor OD: %d gravity pairs added "
        "(%.0f daily BEV trips ≈ %.0f%% of corridor flow, %d candidates considered)",
        cross_added, cross_flow_total,
        CROSS_FRACTION * 100.0, len(raw_cross),
    )

    total_csv_flow = matched_flow + unmatched_flow
    unmapped_pct = (unmatched_flow / total_csv_flow * 100.0) if total_csv_flow > 0 else 0.0
    log_fn = logger.warning if unmapped_pct > 10.0 else logger.info
    log_fn(
        "OD demand coverage: %.0f mapped / %.0f total daily BEV trips "
        "(%.1f%% unmapped across %d roads without corridor entry)",
        matched_flow, total_csv_flow, unmapped_pct,
        len(road_demand) - matched_roads,
    )
    logger.info(
        "OD matrix: %d roads matched → %d pairs, %.0f mapped daily BEV trips",
        matched_roads, len(od.pairs), od.total_daily_trips(),
    )

    if not od.pairs:
        logger.warning(
            "No OD pairs matched demand data against corridor table. "
            "Check that route_segment names in demand_per_segment.csv "
            "match the road names in _ROAD_CORRIDORS."
        )

    # Filter out infeasible OD pairs (> 700 km network distance).
    # Small-battery BEVs (e.g. MG4 Standard, 255 km range) cannot
    # complete ultra-long-haul trips if intermediate coverage is thin.
    max_feasible_km = 700.0
    feasible_pairs = []
    dropped = 0
    for pair in od.pairs:
        try:
            dist = network.shortest_path_length(
                pair.origin, pair.destination, weight="distance_km"
            )
        except Exception:
            dist = 0.0
        if dist <= max_feasible_km or dist == 0.0:
            feasible_pairs.append(pair)
        else:
            dropped += 1
            logger.debug(
                "Dropped infeasible OD pair %s->%s (%.0f km)",
                pair.origin, pair.destination, dist,
            )
    if dropped:
        logger.info("Filtered %d infeasible OD pairs (> %.0f km)", dropped, max_feasible_km)
    od.pairs = feasible_pairs

    # Wire observed link counts for calibration interface
    od.set_observed_counts({
        road: float(flow) for road, flow in road_demand.items()
    })

    return od


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two WGS-84 points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def _get_segment_km(n1: str, n2: str, network: RoadNetwork) -> float:
    """
    Road distance (km) between adjacent corridor cities.

    Priority: ``_SEGMENT_KM`` lookup → reverse lookup → haversine × 1.15
    (1.15 is a standard road sinuosity factor for Spain's interurban roads).
    """
    d = _SEGMENT_KM.get((n1, n2)) or _SEGMENT_KM.get((n2, n1))
    if d:
        return float(d)
    if n1 in network.nodes and n2 in network.nodes:
        node1 = network.nodes[n1]
        node2 = network.nodes[n2]
        return _haversine_km(
            node1.latitude, node1.longitude,
            node2.latitude, node2.longitude,
        ) * 1.15
    logger.warning(
        "_get_segment_km: neither node %s nor %s in network; using 100 km default.",
        n1, n2,
    )
    return 100.0  # last-resort fallback


def _project_onto_polyline(
    lat: float,
    lon: float,
    city_chain: List[str],
    network: RoadNetwork,
) -> float:
    """
    Return the along-road km position of point (lat, lon) projected onto
    the polyline defined by *city_chain*.

    Projection is done in metric-equivalent space: longitude differences are
    scaled by ``cos(mean_lat)`` so east-west and north-south are comparable.
    Previously this used raw-degree dot-product projection, which compressed
    east-west distance by ~24% at Spain's latitudes and mis-ordered stations
    on diagonal corridors.
    """
    cumulative_km = 0.0
    best_km = 0.0
    min_dist_metric = float("inf")

    for i in range(len(city_chain) - 1):
        nid_a = city_chain[i]
        nid_b = city_chain[i + 1]
        if nid_a not in network.nodes or nid_b not in network.nodes:
            seg_km = _SEGMENT_KM.get((nid_a, nid_b))
            if seg_km is None:
                logger.warning(
                    "_project_onto_polyline: missing network nodes and no "
                    "_SEGMENT_KM entry for (%s, %s); using 100 km default.",
                    nid_a, nid_b,
                )
                seg_km = 100.0
            cumulative_km += seg_km
            continue

        node_a = network.nodes[nid_a]
        node_b = network.nodes[nid_b]
        seg_km = _get_segment_km(nid_a, nid_b, network)

        ax, ay = node_a.longitude, node_a.latitude
        bx, by = node_b.longitude, node_b.latitude

        # Scale longitude deltas by cos(mean_lat) so that dot-product projection
        # works in metric-equivalent space rather than raw degree space.
        mean_lat_rad = math.radians((ay + by + lat) / 3.0)
        lon_scale = math.cos(mean_lat_rad)

        dx = (bx - ax) * lon_scale
        dy = by - ay
        denom = dx * dx + dy * dy

        if denom < 1e-12:
            t = 0.0
        else:
            t = (((lon - ax) * lon_scale) * dx + (lat - ay) * dy) / denom
            t = max(0.0, min(1.0, t))

        cx_scaled = ax * lon_scale + t * (bx - ax) * lon_scale
        cy = ay + t * dy
        dist_metric = math.sqrt(
            (lon * lon_scale - cx_scaled) ** 2 + (lat - cy) ** 2
        )

        if dist_metric < min_dist_metric:
            min_dist_metric = dist_metric
            best_km = cumulative_km + t * seg_km

        cumulative_km += seg_km

    return best_km


# ---------------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------------

def _infer_purpose(road_name: str) -> str:
    """Infer trip purpose from road name (heuristic)."""
    if road_name.startswith("AP") or road_name.startswith("A-"):
        return "leisure"
    return "interurban"
