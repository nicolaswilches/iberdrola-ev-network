"""Tests for hybrid OD prior helpers in the real Spain network loader."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generation.spanish_network import (
    _base_municipality_code,
    _build_municipality_hub_crosswalk,
    _evaluate_hybrid_hub_prior,
    _province_code_from_municipality_code,
)


def test_base_municipality_code():
    assert _base_municipality_code("01017_AM") == "01017"
    assert _base_municipality_code("0200301") == "02003"
    assert _base_municipality_code("28079") == "28079"
    assert _base_municipality_code("") == ""
    assert _base_municipality_code(None) == ""


def test_province_code_from_municipality_code():
    assert _province_code_from_municipality_code("28079") == "28"
    assert _province_code_from_municipality_code("01017_AM") == "01"
    assert _province_code_from_municipality_code(None) == ""


def test_build_municipality_hub_crosswalk_assigns_exact_and_fallback_hubs():
    node_df = pd.DataFrame([
        {
            "node_id": "MAD",
            "display_name": "Madrid",
            "municipality_code": "28079",
            "province_code": "28",
            "latitude": 40.416,
            "longitude": -3.703,
            "node_population": 3300000,
        },
        {
            "node_id": "BCN",
            "display_name": "Barcelona",
            "municipality_code": "08019",
            "province_code": "08",
            "latitude": 41.385,
            "longitude": 2.173,
            "node_population": 1600000,
        },
    ])
    municipality_df = pd.DataFrame([
        {"municipality_code": "28079", "province_code": "28", "nombre": "Madrid", "provincia": "Madrid", "pob_2025": 3300000, "latitude": 40.416, "longitude": -3.703},
        {"municipality_code": "28092", "province_code": "28", "nombre": "Mostoles", "provincia": "Madrid", "pob_2025": 214817, "latitude": 40.322, "longitude": -3.865},
        {"municipality_code": "19130", "province_code": "19", "nombre": "Guadalajara", "provincia": "Guadalajara", "pob_2025": 87600, "latitude": 40.633, "longitude": -3.166},
    ])
    province_centroids = pd.DataFrame([
        {"province_code": "28", "province_name": "Madrid", "centroid_lat": 40.4, "centroid_lon": -3.7},
        {"province_code": "19", "province_name": "Guadalajara", "centroid_lat": 40.63, "centroid_lon": -3.17},
    ])

    crosswalk = _build_municipality_hub_crosswalk(node_df, municipality_df, province_centroids).set_index("municipality_code")

    assert crosswalk.loc["28079", "assigned_node"] == "MAD"
    assert crosswalk.loc["28079", "assignment_rule"] == "exact_hub_municipality"
    assert crosswalk.loc["28092", "assigned_node"] == "MAD"
    assert crosswalk.loc["28092", "assignment_rule"] == "same_province_official_coordinate_nearest_hub"
    assert crosswalk.loc["19130", "assigned_node"] == "MAD"
    assert crosswalk.loc["19130", "assignment_rule"] == "official_coordinate_nearest_hub"


def test_evaluate_hybrid_hub_prior_builds_prior_and_missing_hubs():
    node_df = pd.DataFrame([
        {"node_id": "MAD", "display_name": "Madrid", "municipality_code": "28079", "province_code": "28", "latitude": 40.416, "longitude": -3.703, "node_population": 3300000},
        {"node_id": "BCN", "display_name": "Barcelona", "municipality_code": "08019", "province_code": "08", "latitude": 41.385, "longitude": 2.173, "node_population": 1600000},
        {"node_id": "VAL", "display_name": "Valencia", "municipality_code": "46250", "province_code": "46", "latitude": 39.470, "longitude": -0.376, "node_population": 800000},
    ])
    municipality_df = pd.DataFrame([
        {"municipality_code": "28079", "province_code": "28", "nombre": "Madrid", "provincia": "Madrid", "pob_2025": 3300000, "latitude": 40.416, "longitude": -3.703},
        {"municipality_code": "08019", "province_code": "08", "nombre": "Barcelona", "provincia": "Barcelona", "pob_2025": 1600000, "latitude": 41.385, "longitude": 2.173},
        {"municipality_code": "46250", "province_code": "46", "nombre": "Valencia", "provincia": "Valencia", "pob_2025": 800000, "latitude": 39.470, "longitude": -0.376},
        {"municipality_code": "03014", "province_code": "03", "nombre": "Alicante/Alacant", "provincia": "Alicante", "pob_2025": 366000, "latitude": 38.345, "longitude": -0.490},
    ])
    province_centroids = pd.DataFrame([
        {"province_code": "28", "province_name": "Madrid", "centroid_lat": 40.4, "centroid_lon": -3.7},
        {"province_code": "08", "province_name": "Barcelona", "centroid_lat": 41.4, "centroid_lon": 2.1},
        {"province_code": "46", "province_name": "Valencia", "centroid_lat": 39.47, "centroid_lon": -0.37},
        {"province_code": "03", "province_name": "Alicante", "centroid_lat": 38.35, "centroid_lon": -0.48},
    ])
    crosswalk_df = _build_municipality_hub_crosswalk(node_df, municipality_df, province_centroids)
    travel_df = pd.DataFrame([
        {"residence_area": "28079", "overnight_stay_area": "08019", "people": 10.0},
        {"residence_area": "08019", "overnight_stay_area": "28079", "people": 5.0},
        {"residence_area": "28079", "overnight_stay_area": "46250", "people": 7.0},
        {"residence_area": "03014", "overnight_stay_area": "28079", "people": 20.0},
    ])

    result = _evaluate_hybrid_hub_prior(
        node_df=node_df,
        municipality_df=municipality_df,
        travel_df=travel_df,
        crosswalk_df=crosswalk_df,
        mapped_flow_share_threshold=0.20,
        major_missing_top_n=5,
    )

    assert result["summary"]["audit_passed"] is True
    assert result["summary"]["prior_pair_count"] == 2

    prior_pairs = result["prior_pairs"].set_index(["origin_node", "destination_node"])
    assert abs(prior_pairs.loc[("BCN", "MAD"), "raw_people"] - 15.0) < 1e-9
    assert abs(prior_pairs.loc[("MAD", "VAL"), "raw_people"] - 27.0) < 1e-9
    assert abs(result["summary"]["mapped_mainland_flow_share"] - 1.0) < 1e-9
    assert abs(result["summary"]["direct_hub_mainland_flow_share"] - (22.0 / 42.0)) < 1e-9

    top_missing = result["missing_municipalities"].iloc[0]
    assert top_missing["municipality_code"] == "03014"
    assert top_missing["nombre"] == "Alicante/Alacant"
    assert top_missing["assigned_node"] == "VAL"


def test_evaluate_hybrid_hub_prior_fails_below_threshold():
    node_df = pd.DataFrame([
        {"node_id": "MAD", "display_name": "Madrid", "municipality_code": "28079", "province_code": "28", "latitude": 40.416, "longitude": -3.703, "node_population": 3300000},
        {"node_id": "BCN", "display_name": "Barcelona", "municipality_code": "08019", "province_code": "08", "latitude": 41.385, "longitude": 2.173, "node_population": 1600000},
    ])
    municipality_df = pd.DataFrame([
        {"municipality_code": "28079", "province_code": "28", "nombre": "Madrid", "provincia": "Madrid", "pob_2025": 3300000, "latitude": 40.416, "longitude": -3.703},
        {"municipality_code": "08019", "province_code": "08", "nombre": "Barcelona", "provincia": "Barcelona", "pob_2025": 1600000, "latitude": 41.385, "longitude": 2.173},
        {"municipality_code": "03014", "province_code": "03", "nombre": "Alicante/Alacant", "provincia": "Alicante", "pob_2025": 366000, "latitude": 38.345, "longitude": -0.490},
        {"municipality_code": "03065", "province_code": "03", "nombre": "Elx/Elche", "provincia": "Alicante", "pob_2025": 245000, "latitude": 38.266, "longitude": -0.698},
    ])
    province_centroids = pd.DataFrame([
        {"province_code": "28", "province_name": "Madrid", "centroid_lat": 40.4, "centroid_lon": -3.7},
        {"province_code": "08", "province_name": "Barcelona", "centroid_lat": 41.4, "centroid_lon": 2.1},
        {"province_code": "03", "province_name": "Alicante", "centroid_lat": 38.35, "centroid_lon": -0.48},
    ])
    crosswalk_df = _build_municipality_hub_crosswalk(node_df, municipality_df, province_centroids)
    travel_df = pd.DataFrame([
        {"residence_area": "28079", "overnight_stay_area": "08019", "people": 5.0},
        {"residence_area": "03014", "overnight_stay_area": "03065", "people": 45.0},
    ])

    result = _evaluate_hybrid_hub_prior(
        node_df=node_df,
        municipality_df=municipality_df,
        travel_df=travel_df,
        crosswalk_df=crosswalk_df,
        mapped_flow_share_threshold=1.01,
        major_missing_top_n=5,
    )

    assert result["summary"]["audit_passed"] is False
    assert result["summary"]["reason"] == "mapped_flow_share_below_threshold"
    assert abs(result["summary"]["mapped_mainland_flow_share"] - 1.0) < 1e-9
