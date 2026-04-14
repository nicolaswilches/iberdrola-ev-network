"""Synthetic Spain-like national road network for the demo.

The network represents ~25 Spanish cities connected by the main
interurban corridors (AP-7, AP-2, A-3, AP-4, AP-1, A-6 analogues).
Charging stations are placed at major waypoints on each corridor.

This module is DATA GENERATION ONLY.  To use real data:
  1. Load RoadNode/RoadEdge from MiTMA or IGN shapefiles.
  2. Load ChargingStation from the NAP XML or ChargeMap API.
  3. Load ODMatrix from MiTMA study-routes or custom OD surveys.
  4. Pass these objects to SimulationRunner directly.

The calibration interface (calibration/interfaces.py) documents
exactly which files need to be swapped out for production use.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from models.demand import ODMatrix, ODPair
from models.network import RoadEdge, RoadNetwork, RoadNode
from models.station import ChargingStation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_spain_demo_network(
    rng: np.random.Generator = None,
    num_stations: int = 22,
) -> Tuple[RoadNetwork, List[ChargingStation], ODMatrix]:
    """
    Build a synthetic but plausible Spain-like interurban network.

    Returns
    -------
    network:   RoadNetwork with ~25 nodes and ~60 directed edges.
    stations:  List of ChargingStation objects at key waypoints.
    od_matrix: ODMatrix with realistic daily BEV trip volumes.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    network = _build_network()
    stations = _build_stations(network)
    od_matrix = _build_od_matrix(network)

    logger.info(
        "Demo network: %d nodes, %d directed edges, %d stations, "
        "OD matrix: %d pairs (%.0f daily BEV trips)",
        network.node_count, network.edge_count,
        len(stations),
        len(od_matrix.pairs), od_matrix.total_daily_trips(),
    )
    return network, stations, od_matrix


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def _build_network() -> RoadNetwork:
    network = RoadNetwork()

    # ------------------------------------------------------------------
    # Nodes — major Spanish cities + corridor waypoints
    # ------------------------------------------------------------------
    node_data = [
        # (id, name, lat, lon, type, population)
        ("MAD", "Madrid",        40.42, -3.70, "city", 3300000),
        ("BCN", "Barcelona",     41.38,  2.18, "city", 1600000),
        ("VAL", "Valencia",      39.47, -0.38, "city",  800000),
        ("SEV", "Sevilla",       37.39, -5.99, "city",  690000),
        ("BIL", "Bilbao",        43.26, -2.93, "city",  350000),
        ("ZAR", "Zaragoza",      41.65, -0.88, "city",  670000),
        ("MAL", "Málaga",        36.72, -4.42, "city",  570000),
        ("MUR", "Murcia",        37.98, -1.13, "city",  450000),
        ("VLD", "Valladolid",    41.65, -4.72, "city",  300000),
        ("ALI", "Alicante",      38.35, -0.49, "city",  330000),
        ("GRN", "Granada",       37.18, -3.60, "city",  230000),
        ("COR", "Córdoba",       37.89, -4.78, "city",  325000),
        ("BUR", "Burgos",        42.34, -3.70, "city",  180000),
        ("PMP", "Pamplona",      42.82, -1.64, "city",  200000),
        ("SAN", "San Sebastián", 43.32, -1.98, "city",  185000),
        ("VIT", "Vitoria",       42.85, -2.67, "city",  250000),
        ("LLE", "Lleida",        41.62,  0.62, "city",  140000),
        ("TAR", "Tarragona",     41.12,  1.25, "city",  135000),
        ("CAS", "Castellón",     39.99, -0.05, "city",  170000),
        ("ALB", "Albacete",      38.99, -1.86, "city",  175000),
        # Corridor waypoints (smaller towns used as routing nodes)
        ("WPT_CAS_ZAR", "Cambrils–Zaragoza WP",  41.20,  0.95, "junction", 0),
        ("WPT_MAD_VAL", "Madrid–Valencia WP",     39.85, -1.80, "junction", 0),
        ("WPT_MAD_SEV", "Madrid–Sevilla WP",      38.50, -5.00, "junction", 0),
        ("WPT_SEV_MAL", "Sevilla–Málaga WP",      36.90, -5.00, "junction", 0),
        ("WPT_MAD_BIL", "Madrid–Bilbao WP",       42.00, -3.50, "junction", 0),
        # Intermediate waypoints to break long (>160 km) segments
        # A-2: MAD→ZAR was 300 km; split via Guadalajara area
        ("WPT_A2_CTR",  "A-2 Centro (Guadalajara)", 40.63, -2.40, "junction", 0),
        # A-4: MAD→WPT_MAD_SEV was 320 km; split via Manzanares area
        ("WPT_A4_CTR",  "A-4 Centro (Manzanares)",  39.00, -3.85, "junction", 0),
        # A-3: MAD→WPT_MAD_VAL is 200 km (exceeds small battery); split via Tarancón
        ("WPT_A3_MID",  "A-3 Centro (Tarancón)",    40.01, -3.01, "junction", 0),
        # A-1: MAD→WPT_MAD_BIL is 200 km (exceeds small battery); split via Lerma
        ("WPT_A1_MID",  "A-1 Centro (Lerma)",       41.67, -3.69, "junction", 0),
    ]

    for nid, name, lat, lon, ntype, pop in node_data:
        network.add_node(RoadNode(nid, name, lat, lon, ntype, pop))

    # ------------------------------------------------------------------
    # Edges — main interurban corridors (bidirectional)
    # Format: (from, to, dist_km, time_min, road_type, slope)
    # ------------------------------------------------------------------
    edge_data = [
        # AP-2 / A-2: Madrid ↔ Zaragoza ↔ Lleida ↔ Barcelona
        # Split MAD→ZAR (was 300 km, exceeds medium battery) via A-2 Centro waypoint
        ("MAD", "WPT_A2_CTR",  150,  82, "AP", 0.2),
        ("WPT_A2_CTR", "ZAR",  150,  83, "AP", 0.1),
        ("ZAR", "LLE",  150,  85, "AP", 0.1),
        ("LLE", "TAR",   95,  55, "AP", -0.1),
        ("TAR", "BCN",   95,  55, "AP", -0.2),
        # AP-7: Barcelona ↔ Valencia ↔ Alicante
        ("BCN", "TAR",   95,  55, "AP", 0.0),  # (partial reverse already covered)
        ("TAR", "CAS",  150,  90, "AP", -0.1),
        ("CAS", "VAL",   75,  45, "AP", -0.1),
        ("VAL", "ALI",  165,  95, "AP", -0.1),
        ("ALI", "MUR",   85,  50, "AP", 0.0),
        # A-3: Madrid ↔ Valencia
        # Split MAD→WPT_MAD_VAL (was 200 km) via A-3 Centro (Tarancón)
        ("MAD", "WPT_A3_MID",       100,  60, "A", 0.1),
        ("WPT_A3_MID", "WPT_MAD_VAL", 100,  60, "A", 0.0),
        ("WPT_MAD_VAL", "ALB",        100,  65, "A", 0.0),
        ("ALB", "VAL",                180, 105, "A", -0.1),
        # AP-4: Madrid ↔ Sevilla
        # Split MAD→WPT_MAD_SEV (was 320 km, exceeds medium battery) via A-4 Centro waypoint
        ("MAD", "WPT_A4_CTR",         160,  97, "AP", 0.0),
        ("WPT_A4_CTR", "WPT_MAD_SEV", 160,  98, "AP", 0.0),
        ("WPT_MAD_SEV", "COR",  150,  90, "AP", -0.1),
        ("COR", "SEV",           140,  80, "AP", -0.2),
        # Sevilla ↔ Málaga
        ("SEV", "WPT_SEV_MAL",  100,  65, "AP", 0.2),
        ("WPT_SEV_MAL", "MAL",  100,  65, "AP", 0.3),
        # Granada connections
        ("MAL", "GRN",  130,  85, "A", 0.5),
        ("MUR", "GRN",  225, 140, "A", 0.3),
        # AP-1/AP-68: Madrid ↔ Burgos ↔ Vitoria ↔ Bilbao ↔ San Sebastián
        # Split MAD→WPT_MAD_BIL (was 200 km) via A-1 Centro (Lerma)
        ("MAD", "WPT_A1_MID",          100,  58, "AP", 0.3),
        ("WPT_A1_MID", "WPT_MAD_BIL",  100,  57, "AP", 0.2),
        ("WPT_MAD_BIL", "BUR",  140,  80, "AP", 0.1),
        ("BUR", "VIT",           110,  65, "AP", 0.0),
        ("VIT", "BIL",            60,  40, "AP", -0.1),
        ("BIL", "SAN",            95,  65, "AP",  0.1),
        # A-6: Madrid ↔ Valladolid
        ("MAD", "VLD",           190, 110, "A", 0.1),
        ("VLD", "BUR",           120,  75, "A", 0.1),
        # Pamplona connections
        ("VIT", "PMP",            95,  60, "A", 0.1),
        ("PMP", "SAN",            80,  55, "A", 0.2),
        # Zaragoza ↔ Bilbao / Vitoria
        ("ZAR", "PMP",           170, 100, "A", 0.2),
        ("ZAR", "BUR",           290, 175, "A", 0.1),
        # Corridor waypoint connections
        ("WPT_CAS_ZAR", "ZAR",  110,  65, "AP", 0.0),
        ("TAR", "WPT_CAS_ZAR",   65,  40, "AP", 0.0),
    ]

    edge_counter = 0
    for from_n, to_n, dist, time_min, rtype, slope in edge_data:
        # Skip if nodes missing
        if from_n not in network.nodes or to_n not in network.nodes:
            logger.warning("Skipping edge %s→%s: node not found", from_n, to_n)
            continue
        speed = dist / (time_min / 60.0)
        edge = RoadEdge(
            edge_id=f"E{edge_counter:03d}",
            from_node=from_n,
            to_node=to_n,
            distance_km=float(dist),
            travel_time_min=float(time_min),
            road_type=rtype,
            speed_limit_kmh=min(float(speed), 130.0),
            slope_grade=float(slope),
        )
        network.add_undirected_road(edge)
        edge_counter += 1

    logger.info(
        "Network built: %d nodes, %d directed edges",
        network.node_count, network.edge_count,
    )
    return network


# ---------------------------------------------------------------------------
# Charging station construction
# ---------------------------------------------------------------------------

def _build_stations(network: RoadNetwork) -> List[ChargingStation]:
    """
    Place fast-charging stations at key nodes on the network.

    Station design philosophy:
    - Major cities: large hubs (6–8 connectors, high power)
    - Corridor waypoints: medium stops (2–4 connectors)
    - Some nodes intentionally have no station (creates range pressure)
    - Prices vary by operator type (e.g. 0.35–0.50 EUR/kWh range)
    """
    station_specs = [
        # (station_id, node_id, name, power_kw, connectors, price_eur_kwh, reliability)
        ("STA_MAD_N", "MAD", "Madrid Norte HPC", 150, 8, 0.39, 0.97),
        ("STA_MAD_S", "MAD", "Madrid Sur HPC",   150, 6, 0.39, 0.97),
        ("STA_BCN_S", "BCN", "Barcelona Sud HPC", 150, 8, 0.41, 0.96),
        ("STA_ZAR_E", "ZAR", "Zaragoza Est HPC",  150, 4, 0.38, 0.95),
        ("STA_ZAR_W", "ZAR", "Zaragoza Oste HPC", 150, 4, 0.38, 0.95),
        ("STA_VAL_N", "VAL", "Valencia Nord HPC", 150, 6, 0.40, 0.96),
        ("STA_SEV_N", "SEV", "Sevilla Norte HPC", 150, 6, 0.38, 0.95),
        ("STA_BIL_W", "BIL", "Bilbao Oeste HPC",  150, 4, 0.43, 0.93),
        ("STA_MAL_E", "MAL", "Málaga Este HPC",   150, 4, 0.40, 0.94),
        ("STA_ALI_N", "ALI", "Alicante Nord HPC",  50, 2, 0.42, 0.93),
        ("STA_MUR_N", "MUR", "Murcia Norte HPC",   50, 2, 0.42, 0.93),
        ("STA_COR_E", "COR", "Córdoba Este HPC",  150, 4, 0.37, 0.95),
        ("STA_BUR_S", "BUR", "Burgos Sur HPC",    150, 4, 0.39, 0.94),
        ("STA_VIT_E", "VIT", "Vitoria Este HPC",   50, 2, 0.43, 0.92),
        ("STA_PMP_W", "PMP", "Pamplona Oeste HPC", 50, 2, 0.41, 0.92),
        ("STA_LLE_E", "LLE", "Lleida Est HPC",    150, 4, 0.38, 0.95),
        ("STA_TAR_N", "TAR", "Tarragona Nord HPC", 150, 4, 0.38, 0.95),
        ("STA_CAS_S", "CAS", "Castellón Sud HPC",  50, 2, 0.40, 0.93),
        ("STA_ALB_W", "ALB", "Albacete Oeste HPC", 50, 2, 0.41, 0.93),
        ("STA_VLD_S", "VLD", "Valladolid Sur HPC", 50, 2, 0.40, 0.93),
        ("STA_WP_MS", "WPT_MAD_SEV", "A-4 Waypoint HPC",   150, 2, 0.37, 0.94),
        ("STA_WP_MB", "WPT_MAD_BIL", "A-1 Waypoint HPC",   150, 2, 0.39, 0.93),
        # Stations at intermediate waypoints (split long segments)
        ("STA_A2_CTR", "WPT_A2_CTR",  "A-2 Centro HPC",     150, 2, 0.39, 0.94),
        ("STA_A4_CTR", "WPT_A4_CTR",  "A-4 Centro HPC",     150, 2, 0.38, 0.94),
        ("STA_A3_MID", "WPT_A3_MID",  "A-3 Tarancón HPC",   150, 2, 0.39, 0.93),
        ("STA_A3_VAL", "WPT_MAD_VAL", "A-3 Cuenca HPC",     150, 2, 0.39, 0.93),
        ("STA_A1_MID", "WPT_A1_MID",  "A-1 Lerma HPC",      150, 2, 0.40, 0.93),
        # WPT_SEV_MAL bridges the 100+100 km gap between Sevilla and Málaga
        ("STA_SEV_MAL", "WPT_SEV_MAL", "AP-4 Estepona HPC", 150, 2, 0.38, 0.93),
    ]

    stations = []
    for sid, nid, name, power, connectors, price, reliability in station_specs:
        if nid not in network.nodes:
            logger.warning("Station %s: node %s not found, skipping", sid, nid)
            continue
        node = network.nodes[nid]
        stations.append(
            ChargingStation(
                station_id=sid,
                node_id=nid,
                name=name,
                latitude=node.latitude,
                longitude=node.longitude,
                max_power_kw=float(power),
                num_connectors=connectors,
                price_per_kwh=float(price),
                reliability=float(reliability),
            )
        )

    logger.info("Placed %d charging stations", len(stations))
    return stations


# ---------------------------------------------------------------------------
# OD matrix construction
# ---------------------------------------------------------------------------

def _build_od_matrix(network: RoadNetwork) -> ODMatrix:
    """
    Build a synthetic OD matrix calibrated so that total daily BEV trips
    is roughly 2,000 (to give ~200 agents with 10× scaling in the demo).

    Distances drive the trip demand (longer = fewer trips).
    Cities with higher population generate more trips.
    """
    od = ODMatrix()

    # Major city-to-city pairs with approximate daily BEV trip volumes
    # (These numbers approximate a plausible 2027 scenario for Spain)
    pairs_data = [
        # (origin, destination, daily_bev_trips, purpose)
        # AP-2 corridor: Madrid ↔ Barcelona (busiest corridor)
        ("MAD", "BCN",  400, "leisure"),
        ("BCN", "MAD",  400, "leisure"),
        ("MAD", "ZAR",  250, "business"),
        ("ZAR", "MAD",  250, "business"),
        ("ZAR", "BCN",  200, "leisure"),
        ("BCN", "ZAR",  200, "leisure"),
        # Valencia corridor
        ("MAD", "VAL",  300, "leisure"),
        ("VAL", "MAD",  300, "leisure"),
        ("BCN", "VAL",  150, "leisure"),
        ("VAL", "BCN",  150, "leisure"),
        # South corridor
        ("MAD", "SEV",  200, "leisure"),
        ("SEV", "MAD",  200, "leisure"),
        ("MAD", "MAL",  150, "leisure"),
        ("MAL", "MAD",  150, "leisure"),
        ("SEV", "MAL",   75, "leisure"),
        ("MAL", "SEV",   75, "leisure"),
        # North corridor
        ("MAD", "BIL",  150, "business"),
        ("BIL", "MAD",  150, "business"),
        ("MAD", "VLD",  100, "business"),
        ("VLD", "MAD",  100, "business"),
        # East coast
        ("VAL", "ALI",  120, "leisure"),
        ("ALI", "VAL",  120, "leisure"),
        ("VAL", "MUR",   80, "leisure"),
        ("MUR", "VAL",   80, "leisure"),
        # Other pairs
        ("MAD", "COR",  100, "leisure"),
        ("COR", "MAD",  100, "leisure"),
        ("BCN", "ZAR",  150, "business"),  # slight duplication ok for demo
        ("ZAR", "BIL",   80, "business"),
        ("BIL", "ZAR",   80, "business"),
    ]

    for origin, dest, volume, purpose in pairs_data:
        if origin in network.nodes and dest in network.nodes:
            od.add_pair(ODPair(
                origin=origin,
                destination=dest,
                daily_bev_trips=float(volume),
                purpose=purpose,
            ))

    return od


# ---------------------------------------------------------------------------
# CSV export helpers (for inspection / swap-in)
# ---------------------------------------------------------------------------

def export_network_to_csv(
    network: RoadNetwork,
    stations: List[ChargingStation],
    od_matrix: ODMatrix,
    output_dir: str = "data/synthetic",
) -> None:
    """Export the synthetic network to CSV files for inspection."""
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # Nodes
    node_rows = [
        {
            "node_id": n.node_id, "name": n.name,
            "latitude": n.latitude, "longitude": n.longitude,
            "node_type": n.node_type, "population": n.population,
        }
        for n in network.nodes.values()
    ]
    pd.DataFrame(node_rows).to_csv(f"{output_dir}/nodes.csv", index=False)

    # Edges (directed)
    edge_rows = [
        {
            "edge_id": e.edge_id, "from_node": e.from_node, "to_node": e.to_node,
            "distance_km": e.distance_km, "travel_time_min": e.travel_time_min,
            "road_type": e.road_type, "speed_limit_kmh": e.speed_limit_kmh,
            "slope_grade": e.slope_grade,
        }
        for e in network.edges.values()
    ]
    pd.DataFrame(edge_rows).to_csv(f"{output_dir}/edges.csv", index=False)

    # Stations
    sta_rows = [
        {
            "station_id": s.station_id, "node_id": s.node_id, "name": s.name,
            "latitude": s.latitude, "longitude": s.longitude,
            "max_power_kw": s.max_power_kw, "num_connectors": s.num_connectors,
            "price_per_kwh": s.price_per_kwh, "reliability": s.reliability,
        }
        for s in stations
    ]
    pd.DataFrame(sta_rows).to_csv(f"{output_dir}/stations.csv", index=False)

    # OD pairs
    od_rows = [
        {
            "origin": p.origin, "destination": p.destination,
            "daily_bev_trips": p.daily_bev_trips, "purpose": p.purpose,
        }
        for p in od_matrix.pairs
    ]
    pd.DataFrame(od_rows).to_csv(f"{output_dir}/od_demand.csv", index=False)

    logger.info("Exported synthetic network to %s/", output_dir)
