"""Tests for municipality-to-road attachment helpers."""

import math
import os
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generation.municipality_graph import (
    _apply_people_to_bev_conversion,
    _build_stitched_segment_topology,
    _daily_municipality_od_flows_from_travel_df,
    _assign_endpoint_points_to_municipalities,
    _endpoint_points,
    _polygon_municipality_code,
)


def test_polygon_municipality_code():
    assert _polygon_municipality_code("34172626145") == "26145"
    assert _polygon_municipality_code("34011111004") == "11004"
    assert _polygon_municipality_code("") == ""


def test_assign_endpoint_points_to_municipalities_exact_and_geo_fallback():
    roads = gpd.GeoDataFrame(
        [
            {
                "raw_segment_id": 0,
                "road_name": "A-1",
                "PK_inicio": 0.0,
                "PK_fin": 1.0,
                "length_km": 1.0,
                "geometry": LineString([(0.0, 0.0), (0.9, 0.0)]),
            },
            {
                "raw_segment_id": 1,
                "road_name": "A-1",
                "PK_inicio": 1.0,
                "PK_fin": 2.0,
                "length_km": 1.0,
                "geometry": LineString([(2.0, 2.0), (3.0, 2.0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    polygons = gpd.GeoDataFrame(
        [
            {
                "municipality_code": "00001",
                "municipality_name": "Alpha",
                "geometry": Polygon([(-0.5, -0.5), (1.5, -0.5), (1.5, 1.5), (-0.5, 1.5)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    start_points = _endpoint_points(roads, "START")
    assigned = _assign_endpoint_points_to_municipalities(start_points, polygons, "START").set_index("raw_segment_id")

    assert assigned.loc[0, "start_municipality_code"] == "00001"
    assert assigned.loc[0, "start_assignment_method"] == "exact_polygon"
    assert assigned.loc[1, "start_municipality_code"] == ""
    assert assigned.loc[1, "start_assignment_method"] == "geo_fallback"


def test_daily_municipality_od_flows_average_by_day_and_filter_self_pairs():
    travel = pd.DataFrame([
        {"date": "2022-01-01", "residence_area": "28079", "overnight_stay_area": "08019", "people": 12.0},
        {"date": "2022-01-02", "residence_area": "28079", "overnight_stay_area": "08019", "people": 18.0},
        {"date": "2022-01-01", "residence_area": "28079", "overnight_stay_area": "28079", "people": 50.0},
        {"date": "2022-01-01", "residence_area": "01059_AM", "overnight_stay_area": "28079", "people": 8.0},
    ])

    flows = _daily_municipality_od_flows_from_travel_df(
        travel,
        municipality_codes={"28079", "08019", "01059"},
        mainland_only=True,
    )

    assert len(flows) == 2
    madrid_bcn = flows.set_index(["municipality_origin_code", "municipality_destination_code"]).loc[("28079", "08019")]
    assert madrid_bcn["daily_people"] == 15.0
    vitoria_madrid = flows.set_index(["municipality_origin_code", "municipality_destination_code"]).loc[("01059", "28079")]
    assert vitoria_madrid["daily_people"] == 8.0


def test_apply_people_to_bev_conversion():
    flows = pd.DataFrame([
        {"municipality_origin_code": "28079", "municipality_destination_code": "08019", "daily_people": 17.4},
    ])
    converted, summary = _apply_people_to_bev_conversion(
        flows,
        car_mode_share=0.849,
        occupancy_rate=1.74,
        ev_penetration_rate=0.10,
        bev_fraction=0.60,
    )

    assert math.isclose(converted.loc[0, "daily_car_travelers"], 17.4 * 0.849)
    assert math.isclose(converted.loc[0, "daily_vehicle_trips"], 8.49)
    assert math.isclose(converted.loc[0, "daily_bev_trips"], 0.5094)
    assert summary["car_mode_share"] == 0.849
    assert summary["occupancy_rate"] == 1.74
    assert summary["ev_penetration_rate"] == 0.10
    assert summary["bev_fraction"] == 0.60


def test_build_stitched_segment_topology_connects_same_road_and_crossing_road():
    roads = gpd.GeoDataFrame(
        [
            {
                "segment_id": 1,
                "raw_segment_id": 1,
                "road_name": "A-1",
                "road_type": "A",
                "PK_inicio": 0.0,
                "PK_fin": 1.0,
                "length_km": 1.0,
                "daily_bev_traffic_2027": 100.0,
                "geometry": LineString([(0.0, 0.0), (1.0, 0.0)]),
            },
            {
                "segment_id": 2,
                "raw_segment_id": 2,
                "road_name": "A-1",
                "road_type": "A",
                "PK_inicio": 1.0,
                "PK_fin": 2.0,
                "length_km": 1.0,
                "daily_bev_traffic_2027": 100.0,
                "geometry": LineString([(1.0, 0.0), (2.0, 0.0)]),
            },
            {
                "segment_id": 3,
                "raw_segment_id": 3,
                "road_name": "N-2",
                "road_type": "N",
                "PK_inicio": 0.0,
                "PK_fin": 2.0,
                "length_km": 2.0,
                "daily_bev_traffic_2027": 80.0,
                "geometry": LineString([(1.0, -1.0), (1.0, 1.0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )

    topology = _build_stitched_segment_topology(roads)
    junctions = topology["junction_nodes"]
    edges = topology["road_edges"]
    exchange_nodes = junctions[junctions["node_type"] == "road_exchange"]

    assert not exchange_nodes.empty
    mid_node = exchange_nodes.iloc[0]["node_id"]
    connected_edges = edges[(edges["from_node_id"] == mid_node) | (edges["to_node_id"] == mid_node)]
    assert connected_edges["road_name"].nunique() == 2
    assert "A-1" in set(connected_edges["road_name"])
    assert "N-2" in set(connected_edges["road_name"])
