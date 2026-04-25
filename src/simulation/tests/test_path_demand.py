"""Tests for path-demand calibration diagnostics."""

import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from calibration.path_demand import (
    PathCandidate,
    _od_flow_report,
    _od_target_df,
    _is_major_city_node,
    _path_distance_bucket,
    _segment_od_coverage_export,
    _weighted_od_conservation_system,
    build_path_candidates,
    build_segment_competition_diagnostics,
    build_segment_od_coverage_diagnostics,
)
from models.demand import ODPair
from models.network import RoadEdge, RoadNetwork, RoadNode


def _synthetic_diagnostic_inputs():
    candidates = pd.DataFrame([
        {
            "path_id": "P1",
            "od_pair_id": "OD_MAJOR_LONG",
            "origin": "MAD",
            "destination": "BCN",
            "distance_km": 300.0,
            "road_signature": "R_MAIN",
            "source_segment_ids": "1|1",
        },
        {
            "path_id": "P2",
            "od_pair_id": "OD_MAJOR_LONG",
            "origin": "MAD",
            "destination": "BCN",
            "distance_km": 310.0,
            "road_signature": "ALT_CORR",
            "source_segment_ids": "5",
        },
        {
            "path_id": "P3",
            "od_pair_id": "OD_GEO_LOCAL",
            "origin": "GEO_R_MAIN_START",
            "destination": "VLC",
            "distance_km": 80.0,
            "road_signature": "R_MAIN",
            "source_segment_ids": "1|2|2",
        },
        {
            "path_id": "P4",
            "od_pair_id": "OD_GEO_LONG",
            "origin": "GEO_R_GEO_START",
            "destination": "SEV",
            "distance_km": 300.0,
            "road_signature": "R_GEO",
            "source_segment_ids": "3",
        },
        {
            "path_id": "P5",
            "od_pair_id": "OD_MIXED",
            "origin": "GEO_R_MIX_START",
            "destination": "MAD",
            "distance_km": 80.0,
            "road_signature": "R_MIX",
            "source_segment_ids": "4",
        },
        {
            "path_id": "P6",
            "od_pair_id": "OD_MIXED",
            "origin": "GEO_R_MIX_START",
            "destination": "MAD",
            "distance_km": 82.0,
            "road_signature": "ALT_MIX",
            "source_segment_ids": "6",
        },
    ])
    path_flows = pd.DataFrame([
        {"path_id": "P1", "calibrated_daily_bev_flow": 50.0},
        {"path_id": "P2", "calibrated_daily_bev_flow": 100.0},
        {"path_id": "P3", "calibrated_daily_bev_flow": 20.0},
        {"path_id": "P4", "calibrated_daily_bev_flow": 40.0},
        {"path_id": "P5", "calibrated_daily_bev_flow": 30.0},
        {"path_id": "P6", "calibrated_daily_bev_flow": 60.0},
    ])
    segment_report = pd.DataFrame([
        {
            "segment_id": 1,
            "route_segment": "R_MAIN",
            "target_daily_bev_traffic_2027": 80.0,
            "calibrated_daily_bev_flow": 70.0,
            "abs_error": 10.0,
            "abs_error_pct": 12.5,
        },
        {
            "segment_id": 2,
            "route_segment": "R_MAIN",
            "target_daily_bev_traffic_2027": 150.0,
            "calibrated_daily_bev_flow": 20.0,
            "abs_error": 130.0,
            "abs_error_pct": 86.6666667,
        },
        {
            "segment_id": 3,
            "route_segment": "R_GEO",
            "target_daily_bev_traffic_2027": 200.0,
            "calibrated_daily_bev_flow": 40.0,
            "abs_error": 160.0,
            "abs_error_pct": 80.0,
        },
        {
            "segment_id": 4,
            "route_segment": "R_MIX",
            "target_daily_bev_traffic_2027": 120.0,
            "calibrated_daily_bev_flow": 30.0,
            "abs_error": 90.0,
            "abs_error_pct": 75.0,
        },
    ])
    return candidates, path_flows, segment_report


def _base_segment_diagnostics(segment_report: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    df = segment_report.merge(
        coverage[["segment_id", "candidate_path_count", "positive_path_count", "positive_path_flow_sum"]],
        on="segment_id",
        how="left",
    )
    df["max_positive_path_flow"] = df["positive_path_flow_sum"]
    df["is_underfit"] = df["calibrated_daily_bev_flow"] < df["target_daily_bev_traffic_2027"]
    df["is_overfit"] = df["calibrated_daily_bev_flow"] > df["target_daily_bev_traffic_2027"]
    df["fit_bucket"] = "underfit"
    return df


def test_is_major_city_node():
    assert _is_major_city_node("MAD") is True
    assert _is_major_city_node("BCN") is True
    assert _is_major_city_node("GEO_A_7_START") is False
    assert _is_major_city_node("AP-7N_EC001") is False


def test_path_distance_bucket():
    assert _path_distance_bucket(99.9) == "short"
    assert _path_distance_bucket(100.0) == "medium"
    assert _path_distance_bucket(249.9) == "medium"
    assert _path_distance_bucket(250.0) == "long"


def test_segment_coverage_deduplicates_segment_ids_within_path():
    candidates, path_flows, segment_report = _synthetic_diagnostic_inputs()
    coverage = build_segment_od_coverage_diagnostics(candidates, path_flows, segment_report).set_index("segment_id")

    seg1 = coverage.loc[1]
    assert seg1["candidate_path_count"] == 2
    assert seg1["candidate_od_pair_count"] == 2
    assert seg1["candidate_major_city_od_pair_count"] == 1
    assert seg1["candidate_long_distance_od_pair_count"] == 1
    assert seg1["positive_path_count"] == 2
    assert seg1["positive_od_pair_count"] == 2
    assert math.isclose(seg1["positive_path_flow_sum"], 70.0)
    assert math.isclose(seg1["long_distance_positive_flow_sum"], 50.0)
    assert seg1["short_distance_positive_path_count"] == 1
    assert seg1["long_distance_positive_path_count"] == 1


def test_competition_diagnostics_and_labels():
    candidates, path_flows, segment_report = _synthetic_diagnostic_inputs()
    coverage = build_segment_od_coverage_diagnostics(candidates, path_flows, segment_report)
    competition = build_segment_competition_diagnostics(candidates, path_flows, segment_report).set_index("segment_id")

    seg1 = competition.loc[1]
    assert seg1["top_competing_road"] == "ALT_CORR"
    assert math.isclose(seg1["top_competing_road_flow"], 100.0)
    assert math.isclose(seg1["parallel_competition_score"], 100.0 / 70.0)

    base = _base_segment_diagnostics(segment_report, coverage)
    final = _segment_od_coverage_export(base, coverage, competition.reset_index()).set_index("segment_id")

    assert final.loc[1, "diagnostic_label"] == "parallel_competition"
    assert final.loc[2, "diagnostic_label"] == "candidate_scarcity"
    assert final.loc[3, "diagnostic_label"] == "major_city_scarcity"
    assert final.loc[4, "diagnostic_label"] == "mixed"


def test_weighted_od_conservation_system_and_od_flow_report():
    candidates = [
        PathCandidate(
            path_id="P1",
            od_pair_id="MAD_BCN",
            origin="MAD",
            destination="BCN",
            rank=0,
            nodes=("MAD", "BCN"),
            road_signature=("A-2",),
            distance_km=600.0,
            travel_time_min=360.0,
            segment_ids=(1,),
        ),
        PathCandidate(
            path_id="P2",
            od_pair_id="MAD_BCN",
            origin="MAD",
            destination="BCN",
            rank=1,
            nodes=("MAD", "ZAR", "BCN"),
            road_signature=("AP-2",),
            distance_km=620.0,
            travel_time_min=350.0,
            segment_ids=(2,),
        ),
        PathCandidate(
            path_id="P3",
            od_pair_id="LOCALOD_A-2_000_000_001_FWD",
            origin="A",
            destination="B",
            rank=0,
            nodes=("A", "B"),
            road_signature=("A-2",),
            distance_km=10.0,
            travel_time_min=8.0,
            segment_ids=(3,),
        ),
    ]
    od_targets = _od_target_df([
        ODPair(origin="MAD", destination="BCN", daily_bev_trips=120.0),
        ODPair(origin="MAD", destination="VAL", daily_bev_trips=80.0),
    ])

    matrix, targets, weights, summary = _weighted_od_conservation_system(
        candidates,
        od_targets,
        objective="sqrt",
        relative_error_floor=100.0,
        od_conservation_weight=2.0,
    )

    assert summary["n_targeted_od_pairs"] == 2
    assert summary["n_constrained_od_pairs"] == 1
    assert summary["n_unconstrained_od_pairs"] == 1
    assert matrix is not None
    assert targets is not None
    assert matrix.shape == (1, 3)
    dense = matrix.toarray()
    expected_weight = 2.0 / math.sqrt(120.0)
    assert math.isclose(weights[0], expected_weight)
    assert math.isclose(dense[0, 0], expected_weight)
    assert math.isclose(dense[0, 1], expected_weight)
    assert math.isclose(dense[0, 2], 0.0)
    assert math.isclose(targets[0], 120.0 * expected_weight)

    path_flows = pd.DataFrame([
        {"path_id": "P1", "od_pair_id": "MAD_BCN", "calibrated_daily_bev_flow": 70.0},
        {"path_id": "P2", "od_pair_id": "MAD_BCN", "calibrated_daily_bev_flow": 40.0},
        {"path_id": "P3", "od_pair_id": "LOCALOD_A-2_000_000_001_FWD", "calibrated_daily_bev_flow": 15.0},
    ])
    report = _od_flow_report(path_flows, od_targets).set_index("od_pair_id")
    assert math.isclose(report.loc["MAD_BCN", "calibrated_daily_bev_flow"], 110.0)
    assert report.loc["MAD_BCN", "candidate_path_count"] == 2
    assert report.loc["MAD_BCN", "positive_path_count"] == 2
    assert math.isclose(report.loc["MAD_BCN", "abs_error"], 10.0)
    assert math.isclose(report.loc["MAD_VAL", "calibrated_daily_bev_flow"], 0.0)
    assert report.loc["MAD_VAL", "candidate_path_count"] == 0


def test_od_target_df_aggregates_duplicate_od_pair_ids():
    od_targets = _od_target_df([
        ODPair(origin="MAD", destination="BCN", daily_bev_trips=120.0, purpose="work"),
        ODPair(origin="MAD", destination="BCN", daily_bev_trips=30.0, purpose="work"),
        ODPair(origin="LOCAL_A", destination="LOCAL_B", daily_bev_trips=10.0, purpose="leisure"),
    ]).set_index("od_pair_id")

    assert math.isclose(od_targets.loc["MAD_BCN", "target_daily_bev_trips"], 150.0)
    assert od_targets.loc["MAD_BCN", "source_pair_count"] == 2
    assert od_targets.loc["MAD_BCN", "purpose"] == "work"


def test_build_path_candidates_keeps_geometry_paths_without_segment_ids():
    network = RoadNetwork()
    for node_id, lat, lon, node_type in (
        ("MUNI_A", 40.0, -3.0, "municipality"),
        ("J1", 40.0, -3.0, "road_junction"),
        ("J2", 40.1, -3.1, "road_junction"),
        ("MUNI_B", 40.1, -3.1, "municipality"),
    ):
        network.add_node(RoadNode(node_id=node_id, name=node_id, latitude=lat, longitude=lon, node_type=node_type))

    network.add_undirected_road(RoadEdge(
        edge_id="C1",
        from_node="MUNI_A",
        to_node="J1",
        distance_km=0.01,
        travel_time_min=0.1,
        road_type="connector",
        road_name="CONNECTOR_A",
        geometry_backed=True,
    ))
    network.add_undirected_road(RoadEdge(
        edge_id="R1",
        from_node="J1",
        to_node="J2",
        distance_km=5.0,
        travel_time_min=5.0,
        road_type="A",
        road_name="A-TEST",
        geometry_backed=True,
    ))
    network.add_undirected_road(RoadEdge(
        edge_id="C2",
        from_node="J2",
        to_node="MUNI_B",
        distance_km=0.01,
        travel_time_min=0.1,
        road_type="connector",
        road_name="CONNECTOR_B",
        geometry_backed=True,
    ))

    od_pair = ODPair(origin="MUNI_A", destination="MUNI_B", daily_bev_trips=10.0)
    candidates = build_path_candidates(
        network,
        [od_pair],
        max_paths_per_od=3,
        include_roadspan_candidates=False,
        include_local_access_candidates=False,
    )

    assert len(candidates) == 1
    assert candidates[0].od_pair_id == "MUNI_A_MUNI_B"
    assert candidates[0].segment_ids == tuple()
    assert candidates[0].road_signature == ("A-TEST",)
