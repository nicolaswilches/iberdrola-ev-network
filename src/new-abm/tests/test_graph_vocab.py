"""Invariant tests for the canonical graph vocabulary."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_generation.graph_vocab import (  # noqa: E402
    ALLOWED_ENDPOINTS,
    EDGE_ID_PREFIX,
    EdgeKind,
    GraphVocabError,
    NODE_ID_PREFIX,
    NodeKind,
    make_edge,
    make_node,
    validate_graph,
    validate_network,
)
from models.network import RoadEdge, RoadNetwork, RoadNode  # noqa: E402


# ---------------------------------------------------------------------------
# Closed-set / prefix coverage
# ---------------------------------------------------------------------------

def test_every_node_kind_has_an_id_prefix():
    assert set(NODE_ID_PREFIX.keys()) == set(NodeKind)


def test_every_edge_kind_has_an_id_prefix_and_allowed_endpoints():
    assert set(EDGE_ID_PREFIX.keys()) == set(EdgeKind)
    assert set(ALLOWED_ENDPOINTS.keys()) == set(EdgeKind)
    for kind, pairs in ALLOWED_ENDPOINTS.items():
        assert pairs, f"{kind} has no allowed endpoints"


def test_id_prefixes_are_unique():
    node_prefixes = list(NODE_ID_PREFIX.values())
    edge_prefixes = list(EDGE_ID_PREFIX.values())
    assert len(set(node_prefixes)) == len(node_prefixes)
    assert len(set(edge_prefixes)) == len(edge_prefixes)


# ---------------------------------------------------------------------------
# make_node
# ---------------------------------------------------------------------------

def test_make_node_sets_canonical_kind_string():
    n = make_node(NodeKind.MUNICIPALITY, node_id="MUNI_28079", name="Madrid",
                  latitude=40.416, longitude=-3.703, population=3_300_000)
    assert n.node_type == "municipality"
    assert n.population == 3_300_000


def test_make_node_rejects_wrong_prefix_for_kind():
    with pytest.raises(GraphVocabError):
        make_node(NodeKind.MUNICIPALITY, node_id="JUNC_0001", name="x",
                  latitude=0.0, longitude=0.0)


def test_make_node_accepts_unknown_prefix_verbatim():
    n = make_node(NodeKind.GEO_ENDPOINT, node_id="RAW_LEGACY_42", name="x",
                  latitude=0.0, longitude=0.0)
    assert n.node_id == "RAW_LEGACY_42"
    assert n.node_type == "geo_endpoint"


# ---------------------------------------------------------------------------
# make_edge
# ---------------------------------------------------------------------------

def _muni(code: str = "28079") -> RoadNode:
    return make_node(NodeKind.MUNICIPALITY, node_id=f"MUNI_{code}", name=code,
                     latitude=40.0, longitude=-3.7)


def _anchor(suffix: str = "1") -> RoadNode:
    return make_node(NodeKind.ROAD_ANCHOR, node_id=f"ANCH_{suffix}", name=f"anch{suffix}",
                     latitude=40.01, longitude=-3.71)


def _junction(suffix: str = "1", *, lat: float = 40.02, lon: float = -3.72) -> RoadNode:
    return make_node(NodeKind.ROAD_JUNCTION, node_id=f"JUNC_{suffix}", name=f"j{suffix}",
                     latitude=lat, longitude=lon)


def test_make_edge_valid_pair_builds_edge_and_falls_back_on_distance():
    u, v = _junction("A"), _junction("B", lat=40.5, lon=-3.1)
    e = make_edge(EdgeKind.ROAD_SEGMENT, u, v, road_name="A-1")
    assert e.edge_id.startswith("ROADSEG_")
    assert e.distance_km > 0  # haversine fallback
    assert e.travel_time_min >= 0.1
    assert e.road_name == "A-1"


def test_make_edge_rejects_illegal_pair():
    muni = _muni()
    with pytest.raises(GraphVocabError):
        make_edge(EdgeKind.GAP_BRIDGE, muni, _junction())


def test_anchor_link_allows_municipality_to_anchor():
    e = make_edge(EdgeKind.ANCHOR_LINK, _muni(), _anchor(), distance_km=0.5)
    assert e.edge_id.startswith("ANCHORLINK_")
    assert e.distance_km == 0.5
    assert e.road_type == "connector"


def test_municipality_connector_allows_muni_to_junction():
    e = make_edge(EdgeKind.MUNICIPALITY_CONNECTOR, _muni(), _junction(), distance_km=0.01)
    assert e.edge_id.startswith("MUNIEND_")


# ---------------------------------------------------------------------------
# validate_network
# ---------------------------------------------------------------------------

def _build_minimal_network() -> RoadNetwork:
    net = RoadNetwork()
    for n in (_junction("A"), _junction("B"), _muni()):
        net.add_node(n)
    net.add_edge(make_edge(EdgeKind.ROAD_SEGMENT, net.nodes["JUNC_A"], net.nodes["JUNC_B"]))
    net.add_edge(make_edge(EdgeKind.MUNICIPALITY_CONNECTOR, net.nodes["MUNI_28079"], net.nodes["JUNC_A"],
                           distance_km=0.01))
    return net


def test_validate_network_passes_for_clean_network():
    validate_network(_build_minimal_network())


def test_validate_network_flags_unknown_node_type():
    net = _build_minimal_network()
    net.nodes["JUNC_A"] = RoadNode(node_id="JUNC_A", name="bad", latitude=0, longitude=0,
                                   node_type="bogus_kind")
    with pytest.raises(GraphVocabError) as exc:
        validate_network(net)
    assert "bogus_kind" in str(exc.value)


def test_validate_network_flags_illegal_pair_via_prefix():
    net = _build_minimal_network()
    # Forge a ROADSEG_ edge that isn't a valid pair by forcing through raw dataclass.
    bad = RoadEdge(edge_id="ROADSEG_bad", from_node="MUNI_28079", to_node="JUNC_A",
                   distance_km=0, travel_time_min=0.1, road_type="")
    # MUNI -> JUNC is legal for ROAD_SEGMENT in our allowed set, so pick a truly illegal
    # pair: GAP_BRIDGE between MUNI and JUNC.
    net.edges["GAPBRIDGE_bad"] = RoadEdge(
        edge_id="GAPBRIDGE_bad", from_node="MUNI_28079", to_node="JUNC_A",
        distance_km=0, travel_time_min=0.1, road_type="",
    )
    with pytest.raises(GraphVocabError) as exc:
        validate_network(net)
    assert "gap_bridge" in str(exc.value)
    assert bad.edge_id  # keep unused-var quiet


# ---------------------------------------------------------------------------
# validate_graph (DataFrame)
# ---------------------------------------------------------------------------

def test_validate_graph_dataframe_happy_path():
    nodes = pd.DataFrame([
        {"node_id": "JUNC_A", "node_type": "road_junction"},
        {"node_id": "JUNC_B", "node_type": "road_junction"},
    ])
    edges = pd.DataFrame([
        {"edge_id": "ROADSEG_x", "from_node": "JUNC_A", "to_node": "JUNC_B"},
    ])
    validate_graph(nodes, edges)


def test_validate_graph_flags_orphan_edge():
    nodes = pd.DataFrame([{"node_id": "JUNC_A", "node_type": "road_junction"}])
    edges = pd.DataFrame([
        {"edge_id": "ROADSEG_x", "from_node": "JUNC_A", "to_node": "JUNC_MISSING"},
    ])
    with pytest.raises(GraphVocabError) as exc:
        validate_graph(nodes, edges)
    assert "JUNC_MISSING" in str(exc.value)


def test_validate_graph_flags_duplicate_node_id():
    nodes = pd.DataFrame([
        {"node_id": "JUNC_A", "node_type": "road_junction"},
        {"node_id": "JUNC_A", "node_type": "road_junction"},
    ])
    edges = pd.DataFrame([], columns=["edge_id", "from_node", "to_node"])
    with pytest.raises(GraphVocabError) as exc:
        validate_graph(nodes, edges)
    assert "duplicate" in str(exc.value)


def test_validate_graph_missing_required_columns():
    nodes = pd.DataFrame([{"node_id": "x"}])
    edges = pd.DataFrame([], columns=["edge_id", "from_node", "to_node"])
    with pytest.raises(GraphVocabError):
        validate_graph(nodes, edges)
