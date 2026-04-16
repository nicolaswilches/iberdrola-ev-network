"""Road network data models.

Represents Spain's interurban road network as a directed graph.
Nodes are cities / junctions; edges are road segments with distance,
travel time, and road type.  Real data from MiTMA / IGN can be loaded
by replacing the synthetic builder (data_generation/synthetic.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RoadNode:
    """A node in the road network (city, junction, or intermediate waypoint)."""

    node_id: str
    name: str
    latitude: float
    longitude: float
    node_type: str  # "city" | "junction" | "waypoint"
    population: int = 0

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RoadNode) and self.node_id == other.node_id

    def __repr__(self) -> str:
        return f"RoadNode({self.node_id!r}, {self.name!r})"


@dataclass
class RoadEdge:
    """A directed edge representing a road segment between two adjacent nodes."""

    edge_id: str
    from_node: str
    to_node: str
    distance_km: float
    travel_time_min: float   # at free-flow (typical interurban) speed
    road_type: str           # "AP" (motorway) | "A" (national) | "N" (secondary)
    speed_limit_kmh: float = 120.0
    slope_grade: float = 0.0  # positive = uphill from→to, negative = downhill
    toll_eur: float = 0.0    # one-way toll cost in EUR for this segment

    @property
    def free_flow_speed_kmh(self) -> float:
        if self.travel_time_min > 0:
            return self.distance_km / (self.travel_time_min / 60.0)
        return self.speed_limit_kmh

    def __repr__(self) -> str:
        return (
            f"RoadEdge({self.edge_id!r}, "
            f"{self.from_node}→{self.to_node}, "
            f"{self.distance_km:.0f} km)"
        )


# ---------------------------------------------------------------------------
# Network container
# ---------------------------------------------------------------------------

class RoadNetwork:
    """
    Directed road network backed by a NetworkX DiGraph.

    Provides domain-specific helpers for energy-aware routing,
    segment queries, and spatial lookups.  All weight attributes on
    the internal graph are kept in sync with the RoadEdge objects.

    To load real data:
        network = RoadNetwork()
        for node in load_nodes_from_csv("mitma_nodes.csv"):
            network.add_node(node)
        for edge in load_edges_from_csv("mitma_edges.csv"):
            network.add_edge(edge)
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes: Dict[str, RoadNode] = {}
        self.edges: Dict[str, RoadEdge] = {}

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_node(self, node: RoadNode) -> None:
        """Register a node in the network."""
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            name=node.name,
            latitude=node.latitude,
            longitude=node.longitude,
            node_type=node.node_type,
            population=node.population,
        )

    def add_edge(self, edge: RoadEdge) -> None:
        """Register a directed edge in the network."""
        if edge.from_node not in self.nodes:
            raise ValueError(f"from_node {edge.from_node!r} not in network")
        if edge.to_node not in self.nodes:
            raise ValueError(f"to_node {edge.to_node!r} not in network")
        self.edges[edge.edge_id] = edge
        self.graph.add_edge(
            edge.from_node,
            edge.to_node,
            edge_id=edge.edge_id,
            distance_km=edge.distance_km,
            travel_time_min=edge.travel_time_min,
            road_type=edge.road_type,
            speed_limit_kmh=edge.speed_limit_kmh,
            slope_grade=edge.slope_grade,
            toll_eur=edge.toll_eur,
            weight=edge.travel_time_min,  # default weight = travel time (no toll)
        )

    def add_undirected_road(self, edge: RoadEdge) -> None:
        """Add a road in both directions (with negated slope for return)."""
        self.add_edge(edge)
        reverse = RoadEdge(
            edge_id=edge.edge_id + "_rev",
            from_node=edge.to_node,
            to_node=edge.from_node,
            distance_km=edge.distance_km,
            travel_time_min=edge.travel_time_min,
            road_type=edge.road_type,
            speed_limit_kmh=edge.speed_limit_kmh,
            slope_grade=-edge.slope_grade,
        )
        self.add_edge(reverse)

    # ------------------------------------------------------------------
    # Path finding
    # ------------------------------------------------------------------

    def shortest_path(
        self,
        origin: str,
        destination: str,
        weight: str = "travel_time_min",
    ) -> List[str]:
        """Shortest path by *weight* attribute (default: travel time).

        Returns empty list if no path exists.
        """
        try:
            return nx.shortest_path(self.graph, origin, destination, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            logger.warning("No path %s → %s: %s", origin, destination, exc)
            return []

    def shortest_path_gc(
        self,
        origin: str,
        destination: str,
        vot_eur_per_hour: float,
        max_comfortable_speed_kmh: Optional[float] = None,
    ) -> List[str]:
        """Shortest path by generalized cost: travel_time + toll_eur / vot_per_min.

        Agents with high value-of-time pay tolls to use faster AP- motorways;
        budget-conscious agents accept slower A-/N- roads to avoid tolls.

        If max_comfortable_speed_kmh is provided, each edge's travel time is
        recomputed as distance / min(road_speed, comfort_speed) so that an
        agent who self-selects a cruise speed below the road limit gets no
        benefit from using a faster road — and therefore no incentive to pay
        a toll for the speed premium they won't use.
        Falls back to travel-time-only routing when VoT is zero.
        """
        if vot_eur_per_hour <= 0:
            return self.shortest_path(origin, destination)
        vot_per_min = vot_eur_per_hour / 60.0

        def _gc(_u: str, _v: str, d: dict) -> float:
            if max_comfortable_speed_kmh is not None:
                road_speed = d.get("speed_limit_kmh", 120.0)
                eff_speed = min(road_speed, max_comfortable_speed_kmh)
                dist = d.get("distance_km", 0.0)
                travel_time = (dist / eff_speed * 60.0) if eff_speed > 0 else float("inf")
            else:
                travel_time = d.get("travel_time_min", 0.0)
            return travel_time + d.get("toll_eur", 0.0) / vot_per_min

        try:
            return nx.shortest_path(self.graph, origin, destination, weight=_gc)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            logger.warning("No GC path %s → %s: %s", origin, destination, exc)
            return []

    def shortest_path_length(
        self,
        origin: str,
        destination: str,
        weight: str = "travel_time_min",
    ) -> float:
        """Shortest path length (sum of *weight*). Returns inf if no path."""
        try:
            return nx.shortest_path_length(
                self.graph, origin, destination, weight=weight
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float("inf")

    # ------------------------------------------------------------------
    # Segment / path queries
    # ------------------------------------------------------------------

    def get_edge_attrs(self, from_node: str, to_node: str) -> Optional[Dict]:
        """Raw NetworkX edge attribute dict between adjacent nodes."""
        return self.graph.get_edge_data(from_node, to_node)

    def segment_distance_km(self, from_node: str, to_node: str) -> float:
        data = self.graph.get_edge_data(from_node, to_node)
        return data["distance_km"] if data else float("inf")

    def segment_travel_time_min(self, from_node: str, to_node: str) -> float:
        data = self.graph.get_edge_data(from_node, to_node)
        return data["travel_time_min"] if data else float("inf")

    def segment_slope_grade(self, from_node: str, to_node: str) -> float:
        data = self.graph.get_edge_data(from_node, to_node)
        return data.get("slope_grade", 0.0) if data else 0.0

    def subpath_distance_km(self, nodes: List[str]) -> float:
        """Sum of distances along a list of consecutive nodes."""
        return sum(
            self.segment_distance_km(nodes[i], nodes[i + 1])
            for i in range(len(nodes) - 1)
        )

    def subpath_travel_time_min(self, nodes: List[str]) -> float:
        """Sum of travel times along a list of consecutive nodes."""
        return sum(
            self.segment_travel_time_min(nodes[i], nodes[i + 1])
            for i in range(len(nodes) - 1)
        )

    def iter_edges_on_path(self, path: List[str]) -> Iterator[Tuple[str, str, Dict]]:
        """Iterate (from_node, to_node, attrs) for every edge on *path*."""
        for i in range(len(path) - 1):
            attrs = self.get_edge_attrs(path[i], path[i + 1]) or {}
            yield path[i], path[i + 1], attrs

    def subpath_up_to_node(self, full_path: List[str], target_node: str) -> List[str]:
        """Return the prefix of *full_path* ending at *target_node* (inclusive)."""
        try:
            idx = full_path.index(target_node)
            return full_path[: idx + 1]
        except ValueError:
            return full_path

    def subpath_from_node(self, full_path: List[str], start_node: str) -> List[str]:
        """Return the suffix of *full_path* starting from *start_node* (inclusive)."""
        try:
            idx = full_path.index(start_node)
            return full_path[idx:]
        except ValueError:
            return full_path

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    def get_successors(self, node_id: str) -> List[str]:
        return list(self.graph.successors(node_id))

    def has_path(self, origin: str, destination: str) -> bool:
        return nx.has_path(self.graph, origin, destination)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        return (
            f"RoadNetwork({self.node_count} nodes, "
            f"{self.edge_count} directed edges)"
        )
