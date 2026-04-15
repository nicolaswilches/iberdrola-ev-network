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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.demand import ODMatrix, ODPair
from models.network import RoadEdge, RoadNetwork, RoadNode
from models.station import ChargingStation

logger = logging.getLogger(__name__)


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
    ("AP-2",  ["MAD", "ZAR", "LLE", "TAR", "BCN"]),
    ("A-2",   ["MAD", "ZAR", "LLE", "TAR", "BCN"]),
    ("AP-7",  ["BCN", "TAR", "CAS", "VAL", "ALI", "MUR"]),
    ("A-7",   ["BCN", "TAR", "CAS", "VAL", "ALI", "MUR"]),
    ("A-7S",  ["MAL", "ALI", "MUR"]),
    ("A-3",   ["MAD", "ALB", "VAL"]),
    ("AP-4",  ["MAD", "COR", "SEV"]),
    ("A-4",   ["MAD", "COR", "SEV"]),
    ("A-1",   ["MAD", "BUR", "VIT", "BIL"]),
    ("AP-1",  ["MAD", "BUR", "VIT", "BIL"]),
    ("A-6",   ["MAD", "VLD"]),
    ("AP-68", ["ZAR", "LOG", "VIT", "BIL"]),
    ("A-68",  ["ZAR", "LOG", "VIT", "BIL"]),
    ("A-8",   ["BIL", "SSB", "PMP"]),
    ("A-66",  ["SEV", "MER", "VLD"]),
    ("AP-46", ["MAL", "GRN"]),
    # Corridors with AFIR gaps (proposed stations placed on these)
    ("A-23",  ["ZAR", "TER", "VAL"]),
    ("AP-9",  ["ACO", "SCQ", "VIG"]),
    ("N-322", ["ALB", "JAE", "COR"]),
    ("N-433", ["SEV", "MER", "BAD"]),
    ("N-435", ["HUE", "SEV"]),
    ("N-502", ["AVL", "TAL", "COR"]),
    ("N-621", ["PAL", "STD"]),
    # Further common corridors
    ("A-67",  ["PAL", "STD"]),
    ("A-62",  ["MAD", "VLD", "BUR"]),
    ("N-630", ["SEV", "MER", "VLD"]),
    ("A-45",  ["COR", "MAL"]),
    ("A-92",  ["SEV", "GRN", "MUR"]),
    ("N-340", ["ALI", "MAL"]),
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
}

# Speed limits by road prefix (km/h) — used to derive travel time
_SPEED_KMH: Dict[str, float] = {"AP": 120.0, "A": 100.0, "N": 90.0}

# Default price per kWh for existing stations (EUR) — NAP data lacks prices
_DEFAULT_PRICE_EUR_KWH = 0.40
# Default power for proposed stations (project constant)
_PROPOSED_STATION_POWER_KW = 150.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_spain_real_network(
    data_dir: Path,
    rng: Optional[np.random.Generator] = None,
    include_proposed_stations: bool = True,
    max_existing_clusters_per_road: int = 4,
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
    max_existing_clusters_per_road:
        Maximum number of station cluster waypoints to insert per road from
        the baseline charger data.  Capped at the actual number of stations
        found on that road.  Larger values give finer resolution but a bigger
        graph.

    Returns
    -------
    (network, stations, od_matrix)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    data_dir = Path(data_dir)
    segments_df, demand_df, chargers_df, proposed_df = _load_data(data_dir)

    # 1. Road-level aggregate lengths from real data
    road_lengths = _compute_road_lengths(segments_df)

    # 2. Build city-only graph (nodes, no edges yet)
    network = _build_city_graph()

    # 3. Build corridor edge chains, insert station waypoints, collect stations
    stations = _build_corridors_and_stations(
        network=network,
        chargers_df=chargers_df,
        proposed_df=proposed_df,
        road_lengths=road_lengths,
        include_proposed=include_proposed_stations,
        max_clusters=max_existing_clusters_per_road,
    )

    # 4. OD matrix calibrated to 2027 BEV demand
    od_matrix = _build_od_matrix(demand_df, network)

    logger.info(
        "Real network ready: %d nodes, %d directed edges, %d stations, "
        "%d OD pairs (%.0f daily BEV trips)",
        network.node_count, network.edge_count,
        len(stations),
        len(od_matrix.pairs), od_matrix.total_daily_trips(),
    )
    return network, stations, od_matrix


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
    road_lengths: Dict[str, float],
    include_proposed: bool,
    max_clusters: int,
) -> List[ChargingStation]:
    """
    For every corridor in ``_ROAD_CORRIDORS``:
      1. Cluster baseline chargers on that road into ≤max_clusters waypoints.
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

    for road_name, city_chain in _ROAD_CORRIDORS:
        # --- verify all cities exist ---
        city_chain = [c for c in city_chain if c in network.nodes]
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
            city_chain=city_chain,
            network=network,
            max_clusters=max_clusters,
        )
        waypoints.extend(charger_wps)
        all_stations.extend(charger_stations)

        # B) Proposed stations
        if include_proposed:
            prop_wps, prop_stations = _get_proposed_on_road(
                road_name=road_name,
                proposed_df=proposed_df,
                city_chain=city_chain,
                network=network,
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
        cumulative_km = 0.0
        for i, nid in enumerate(city_chain):
            city_positions.append((cumulative_km, nid))
            if i < len(city_chain) - 1:
                cumulative_km += _get_segment_km(
                    city_chain[i], city_chain[i + 1], network
                )

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

        # Build directed edges between consecutive nodes in the chain
        for i in range(len(deduped) - 1):
            km_a, nid_a = deduped[i]
            km_b, nid_b = deduped[i + 1]
            seg_km = max(km_b - km_a, 0.5)   # guard against zero-length
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

def _cluster_chargers_on_road(
    road_name: str,
    chargers_df: pd.DataFrame,
    city_chain: List[str],
    network: RoadNetwork,
    max_clusters: int,
) -> Tuple[List[Tuple[float, str, float, float]], List[ChargingStation]]:
    """
    Cluster baseline chargers on *road_name* into ≤max_clusters groups and
    return waypoint descriptors + ChargingStation objects.
    """
    # Match rows by nearest_road column (strip whitespace for safety)
    if "nearest_road" not in chargers_df.columns:
        return [], []

    mask = chargers_df["nearest_road"].fillna("").str.strip() == road_name.strip()
    road_ch = chargers_df[mask].copy()

    if road_ch.empty:
        return [], []

    # Sort by segment_id as a proxy for along-road position
    if "segment_id" in road_ch.columns:
        road_ch = road_ch.sort_values("segment_id").reset_index(drop=True)

    n_clusters = min(max_clusters, len(road_ch))

    # Assign cluster label by equal-size quantile split
    road_ch["_cluster"] = (
        road_ch.index * n_clusters // len(road_ch)
    ).astype(int)

    waypoints: List[Tuple[float, str, float, float]] = []
    stations: List[ChargingStation] = []

    for cid, grp in road_ch.groupby("_cluster"):
        lat = float(grp["latitude"].median())
        lon = float(grp["longitude"].median())
        n_connectors = int(grp["n_connectors"].fillna(2).astype(int).sum())
        n_connectors = max(1, n_connectors)
        max_power = float(grp["max_power_kw"].fillna(50).max())

        km_pos = _project_onto_polyline(lat, lon, city_chain, network)
        node_id = f"{road_name}_EC{cid:02d}"
        station_id = f"EC_{road_name}_{cid:02d}"

        waypoints.append((km_pos, node_id, lat, lon))
        stations.append(
            ChargingStation(
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
        )

    return waypoints, stations


# ---------------------------------------------------------------------------
# Proposed station placement
# ---------------------------------------------------------------------------

def _get_proposed_on_road(
    road_name: str,
    proposed_df: pd.DataFrame,
    city_chain: List[str],
    network: RoadNetwork,
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

        km_pos = _project_onto_polyline(lat, lon, city_chain, network)

        waypoints.append((km_pos, node_id, lat, lon))
        stations.append(
            ChargingStation(
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
        )

    return waypoints, stations


# ---------------------------------------------------------------------------
# OD matrix from demand data
# ---------------------------------------------------------------------------

def _build_od_matrix(
    demand_df: pd.DataFrame,
    network: RoadNetwork,
) -> ODMatrix:
    """
    Build an ODMatrix calibrated to 2027 BEV demand from
    ``demand_per_segment.csv``.

    For each road in *demand_df*, daily BEV flow is summed and mapped to
    the (origin_city, destination_city) pair defined in ``_ROAD_CORRIDORS``.
    Both directions are added (roads are bidirectional).

    Roads with no matching corridor entry, or whose endpoint cities are not
    in the network, are skipped and their demand is logged at DEBUG level.
    """
    # corridor lookup: road_name -> (origin, dest)
    corridor_lookup: Dict[str, Tuple[str, str]] = {}
    for road_name, nodes in _ROAD_CORRIDORS:
        valid = [n for n in nodes if n in network.nodes]
        if len(valid) >= 2:
            corridor_lookup[road_name] = (valid[0], valid[-1])

    if "route_segment" not in demand_df.columns or \
       "daily_bev_traffic_2027" not in demand_df.columns:
        logger.warning(
            "demand_per_segment.csv missing required columns; "
            "returning empty ODMatrix"
        )
        return ODMatrix()

    road_demand = (
        demand_df.groupby("route_segment")["daily_bev_traffic_2027"]
        .sum()
        .to_dict()
    )

    od = ODMatrix()
    unmatched_flow = 0.0
    matched_roads = 0

    for road_name, total_flow in road_demand.items():
        if road_name not in corridor_lookup:
            unmatched_flow += total_flow
            continue
        origin, dest = corridor_lookup[road_name]
        purpose = _infer_purpose(road_name)
        od.add_pair(ODPair(
            origin=origin,
            destination=dest,
            daily_bev_trips=float(total_flow),
            purpose=purpose,
        ))
        od.add_pair(ODPair(
            origin=dest,
            destination=origin,
            daily_bev_trips=float(total_flow),
            purpose=purpose,
        ))
        matched_roads += 1

    logger.info(
        "OD matrix: %d roads matched, %d pairs, %.0f total daily BEV trips "
        "(%.0f trips on unmatched roads skipped)",
        matched_roads, len(od.pairs), od.total_daily_trips(), unmatched_flow,
    )

    if not od.pairs:
        logger.warning(
            "No OD pairs matched demand data against corridor table. "
            "Check that route_segment names in demand_per_segment.csv "
            "match the road names in _ROAD_CORRIDORS."
        )

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

    Each segment of the polyline is a straight line in geographic coordinates
    (adequate accuracy at Spain's scale for corridor-level positioning).
    The function returns the cumulative km at the closest projection point.
    """
    cumulative_km = 0.0
    best_km = 0.0
    min_dist_deg = float("inf")

    for i in range(len(city_chain) - 1):
        nid_a = city_chain[i]
        nid_b = city_chain[i + 1]
        if nid_a not in network.nodes or nid_b not in network.nodes:
            seg_km = _SEGMENT_KM.get((nid_a, nid_b), 100.0)
            cumulative_km += seg_km
            continue

        node_a = network.nodes[nid_a]
        node_b = network.nodes[nid_b]
        seg_km = _get_segment_km(nid_a, nid_b, network)

        ax, ay = node_a.longitude, node_a.latitude
        bx, by = node_b.longitude, node_b.latitude
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy

        if denom < 1e-12:
            t = 0.0
        else:
            t = ((lon - ax) * dx + (lat - ay) * dy) / denom
            t = max(0.0, min(1.0, t))

        cx = ax + t * dx
        cy = ay + t * dy
        dist_deg = math.sqrt((lon - cx) ** 2 + (lat - cy) ** 2)

        if dist_deg < min_dist_deg:
            min_dist_deg = dist_deg
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
