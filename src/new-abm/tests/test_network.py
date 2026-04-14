"""Tests for road network models and path finding."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models.network import RoadNode, RoadEdge, RoadNetwork


def make_simple_network() -> RoadNetwork:
    """Build a 3-node test network: A → B → C."""
    net = RoadNetwork()
    net.add_node(RoadNode("A", "City A", 40.0, -3.0, "city", 100000))
    net.add_node(RoadNode("B", "City B", 41.0, -3.0, "city", 50000))
    net.add_node(RoadNode("C", "City C", 42.0, -3.0, "city", 80000))

    net.add_edge(RoadEdge("E1", "A", "B", distance_km=100.0, travel_time_min=60.0, road_type="AP"))
    net.add_edge(RoadEdge("E2", "B", "C", distance_km=120.0, travel_time_min=72.0, road_type="AP"))
    return net


def test_node_count():
    net = make_simple_network()
    assert net.node_count == 3


def test_edge_count():
    net = make_simple_network()
    assert net.edge_count == 2


def test_shortest_path():
    net = make_simple_network()
    path = net.shortest_path("A", "C", weight="travel_time_min")
    assert path == ["A", "B", "C"]


def test_no_path():
    net = make_simple_network()
    path = net.shortest_path("C", "A")  # no reverse edges
    assert path == []


def test_path_distance():
    net = make_simple_network()
    path = net.shortest_path("A", "C")
    dist = net.subpath_distance_km(path)
    assert abs(dist - 220.0) < 0.01


def test_path_travel_time():
    net = make_simple_network()
    path = net.shortest_path("A", "C")
    t = net.subpath_travel_time_min(path)
    assert abs(t - 132.0) < 0.01


def test_undirected_road():
    net = RoadNetwork()
    net.add_node(RoadNode("X", "X", 40.0, -3.0, "city"))
    net.add_node(RoadNode("Y", "Y", 41.0, -3.0, "city"))
    edge = RoadEdge("E_XY", "X", "Y", 50.0, 30.0, "A")
    net.add_undirected_road(edge)

    assert net.edge_count == 2
    assert net.shortest_path("X", "Y") != []
    assert net.shortest_path("Y", "X") != []


def test_segment_distance():
    net = make_simple_network()
    assert net.segment_distance_km("A", "B") == 100.0
    assert net.segment_distance_km("A", "C") == float("inf")  # not adjacent


def test_subpath_up_to_node():
    net = make_simple_network()
    path = ["A", "B", "C"]
    sub = net.subpath_up_to_node(path, "B")
    assert sub == ["A", "B"]


def test_subpath_from_node():
    net = make_simple_network()
    path = ["A", "B", "C"]
    sub = net.subpath_from_node(path, "B")
    assert sub == ["B", "C"]


def test_add_edge_with_missing_node():
    net = RoadNetwork()
    net.add_node(RoadNode("A", "A", 40.0, -3.0, "city"))
    with pytest.raises(ValueError):
        net.add_edge(RoadEdge("E1", "A", "MISSING", 100.0, 60.0, "AP"))


def test_has_path():
    net = make_simple_network()
    assert net.has_path("A", "C")
    assert not net.has_path("C", "A")
