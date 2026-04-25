"""Canonical node/edge vocabulary for the graph-building layer.

Single source of truth for:
- Which node kinds exist (``NodeKind``)
- Which edge kinds exist (``EdgeKind``)
- The id-prefix rule for each kind
- Which ``(u_kind, v_kind)`` pairs are legal for each edge kind
- Factories (``make_node`` / ``make_edge``) that hide id formation and defaults
- A ``validate_network`` / ``validate_graph`` seam callable before CSV export

Downstream phases (calibration, simulation, optimization) should import kinds
from here rather than typing string literals.

See RFC: https://github.com/nicolaswilches/iberdrola-ev-network/issues/7
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple

import pandas as pd

from models.network import RoadEdge, RoadNetwork, RoadNode


class GraphVocabError(ValueError):
    """Raised when a node/edge violates the vocabulary's invariants."""


class NodeKind(str, Enum):
    MUNICIPALITY = "municipality"
    ROAD_JUNCTION = "road_junction"
    ROAD_EXCHANGE = "road_exchange"
    ROAD_ANCHOR = "road_anchor"
    GEO_ENDPOINT = "geo_endpoint"
    SERVICE_STATION = "service_station"


class EdgeKind(str, Enum):
    ROAD_SEGMENT = "road_segment"
    GAP_BRIDGE = "gap_bridge"
    MUNICIPALITY_CONNECTOR = "municipality_connector"
    ANCHOR_LINK = "anchor_link"


NODE_ID_PREFIX: Dict[NodeKind, str] = {
    NodeKind.MUNICIPALITY: "MUNI_",
    NodeKind.ROAD_JUNCTION: "JUNC_",
    NodeKind.ROAD_EXCHANGE: "EXCH_",
    NodeKind.ROAD_ANCHOR: "ANCH_",
    NodeKind.GEO_ENDPOINT: "GEO_",
    NodeKind.SERVICE_STATION: "SVC_",
}

EDGE_ID_PREFIX: Dict[EdgeKind, str] = {
    EdgeKind.ROAD_SEGMENT: "ROADSEG_",
    EdgeKind.GAP_BRIDGE: "GAPBRIDGE_",
    EdgeKind.MUNICIPALITY_CONNECTOR: "MUNIEND_",
    EdgeKind.ANCHOR_LINK: "ANCHORLINK_",
}

_ROAD_TERMINAL_KINDS: FrozenSet[NodeKind] = frozenset({
    NodeKind.ROAD_JUNCTION,
    NodeKind.ROAD_EXCHANGE,
    NodeKind.GEO_ENDPOINT,
    NodeKind.ROAD_ANCHOR,
})
_ATTACHMENT_KINDS: FrozenSet[NodeKind] = frozenset({
    NodeKind.MUNICIPALITY,
    NodeKind.SERVICE_STATION,
})
_ANY_GRAPH_KIND: FrozenSet[NodeKind] = _ROAD_TERMINAL_KINDS | _ATTACHMENT_KINDS


def _undirected_pairs(lefts: Iterable[NodeKind], rights: Iterable[NodeKind]) -> FrozenSet[Tuple[NodeKind, NodeKind]]:
    pairs = set()
    for left in lefts:
        for right in rights:
            pairs.add((left, right))
            pairs.add((right, left))
    return frozenset(pairs)


ALLOWED_ENDPOINTS: Dict[EdgeKind, FrozenSet[Tuple[NodeKind, NodeKind]]] = {
    # Road segments can touch any graph node; attachments (municipality,
    # service_station) appear when a road endpoint falls inside an attachment.
    EdgeKind.ROAD_SEGMENT: _undirected_pairs(_ANY_GRAPH_KIND, _ANY_GRAPH_KIND),
    # Gap bridges link two road terminals on the same road.
    EdgeKind.GAP_BRIDGE: _undirected_pairs(_ROAD_TERMINAL_KINDS, _ROAD_TERMINAL_KINDS),
    # Municipality connectors link an attachment node to a road terminal at
    # the same geometric point (zero-length glue).
    EdgeKind.MUNICIPALITY_CONNECTOR: _undirected_pairs(_ATTACHMENT_KINDS, _ROAD_TERMINAL_KINDS),
    # Anchor links either attach a municipality to its anchor point, or
    # connect an anchor to the surrounding road terminals when a segment is split.
    EdgeKind.ANCHOR_LINK: (
        _undirected_pairs(_ATTACHMENT_KINDS, {NodeKind.ROAD_ANCHOR})
        | _undirected_pairs({NodeKind.ROAD_ANCHOR}, _ROAD_TERMINAL_KINDS)
    ),
}


def _as_node_kind(value: Any) -> NodeKind:
    if isinstance(value, NodeKind):
        return value
    try:
        return NodeKind(str(value))
    except ValueError as exc:
        raise GraphVocabError(f"Unknown node kind: {value!r}") from exc


def _as_edge_kind(value: Any) -> EdgeKind:
    if isinstance(value, EdgeKind):
        return value
    try:
        return EdgeKind(str(value))
    except ValueError as exc:
        raise GraphVocabError(f"Unknown edge kind: {value!r}") from exc


def make_node(
    kind: NodeKind,
    *,
    node_id: str,
    name: str,
    latitude: float,
    longitude: float,
    population: int = 0,
) -> RoadNode:
    """Build a ``RoadNode`` whose ``node_type`` is the canonical kind string.

    ``node_id`` is accepted verbatim — callers already have deterministic id
    schemes (``MUNI_{code}``, ``GEO_{road}_{seg}_{suffix}``). The factory
    verifies the id prefix matches the kind when the id starts with a known
    prefix; otherwise it accepts the id as-is (so legacy ids like
    ``RAWROAD_...`` keep working).
    """
    kind = _as_node_kind(kind)
    expected_prefix = NODE_ID_PREFIX[kind]
    if any(node_id.startswith(p) for p in NODE_ID_PREFIX.values()):
        if not node_id.startswith(expected_prefix):
            raise GraphVocabError(
                f"node_id {node_id!r} has a known kind prefix that does not match "
                f"declared kind {kind.value!r} (expected prefix {expected_prefix!r})"
            )
    return RoadNode(
        node_id=node_id,
        name=name,
        latitude=float(latitude),
        longitude=float(longitude),
        node_type=kind.value,
        population=int(population),
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def make_edge(
    kind: EdgeKind,
    u: RoadNode,
    v: RoadNode,
    *,
    edge_id: Optional[str] = None,
    distance_km: Optional[float] = None,
    travel_time_min: Optional[float] = None,
    speed_kmh: float = 100.0,
    road_type: Optional[str] = None,
    road_name: str = "",
    from_road_km: float = 0.0,
    to_road_km: float = 0.0,
    source_segment_ids: Tuple[int, ...] = (),
    target_daily_bev_traffic_2027: float = 0.0,
    geometry_backed: bool = False,
    toll_eur: float = 0.0,
) -> RoadEdge:
    """Build a ``RoadEdge`` and validate the endpoint-kind pair.

    If ``distance_km`` is omitted, falls back to haversine between the two
    node coords. If ``travel_time_min`` is omitted, it's derived from distance
    and ``speed_kmh`` (minimum 0.1 min to avoid zero-weight edges).
    """
    kind = _as_edge_kind(kind)
    u_kind = _as_node_kind(u.node_type)
    v_kind = _as_node_kind(v.node_type)
    if (u_kind, v_kind) not in ALLOWED_ENDPOINTS[kind]:
        raise GraphVocabError(
            f"edge kind {kind.value!r} does not allow endpoints "
            f"({u_kind.value}, {v_kind.value}); u={u.node_id!r}, v={v.node_id!r}"
        )

    if distance_km is None:
        distance_km = _haversine_km(u.latitude, u.longitude, v.latitude, v.longitude)
    distance_km = float(distance_km)

    if travel_time_min is None:
        travel_time_min = max(distance_km / speed_kmh * 60.0, 0.1)
    travel_time_min = float(travel_time_min)

    if edge_id is None:
        edge_id = f"{EDGE_ID_PREFIX[kind]}{u.node_id}__{v.node_id}"

    if road_type is None:
        road_type = "connector" if kind in (EdgeKind.MUNICIPALITY_CONNECTOR, EdgeKind.ANCHOR_LINK) else ""

    return RoadEdge(
        edge_id=edge_id,
        from_node=u.node_id,
        to_node=v.node_id,
        distance_km=distance_km,
        travel_time_min=travel_time_min,
        road_type=road_type,
        road_name=road_name,
        from_road_km=float(from_road_km),
        to_road_km=float(to_road_km),
        source_segment_ids=tuple(source_segment_ids),
        target_daily_bev_traffic_2027=float(target_daily_bev_traffic_2027),
        geometry_backed=bool(geometry_backed),
        toll_eur=float(toll_eur),
    )


def validate_network(network: RoadNetwork) -> None:
    """Validate a built ``RoadNetwork`` against the vocabulary's invariants.

    Raises ``GraphVocabError`` listing all violations, not just the first.
    Checks:
      1. Every node's ``node_type`` is a known ``NodeKind``.
      2. Every edge endpoint exists in ``network.nodes``.
      3. Every edge whose id has a known prefix respects ``ALLOWED_ENDPOINTS``.
    """
    violations: list[str] = []

    for node_id, node in network.nodes.items():
        try:
            NodeKind(node.node_type)
        except ValueError:
            violations.append(
                f"node {node_id!r}: unknown node_type {node.node_type!r} "
                f"(valid: {[k.value for k in NodeKind]})"
            )

    kind_by_prefix = {prefix: kind for kind, prefix in EDGE_ID_PREFIX.items()}
    for edge_id, edge in network.edges.items():
        if edge.from_node not in network.nodes:
            violations.append(f"edge {edge_id!r}: from_node {edge.from_node!r} missing")
            continue
        if edge.to_node not in network.nodes:
            violations.append(f"edge {edge_id!r}: to_node {edge.to_node!r} missing")
            continue
        matched_kind: Optional[EdgeKind] = None
        for prefix, kind in kind_by_prefix.items():
            if edge_id.startswith(prefix):
                matched_kind = kind
                break
        if matched_kind is None:
            continue  # legacy/unprefixed edges pass
        u_kind_raw = network.nodes[edge.from_node].node_type
        v_kind_raw = network.nodes[edge.to_node].node_type
        try:
            u_kind = NodeKind(u_kind_raw)
            v_kind = NodeKind(v_kind_raw)
        except ValueError:
            continue  # already reported above
        if (u_kind, v_kind) not in ALLOWED_ENDPOINTS[matched_kind]:
            violations.append(
                f"edge {edge_id!r}: kind {matched_kind.value!r} does not allow "
                f"({u_kind.value}, {v_kind.value})"
            )

    if violations:
        summary = "\n  - ".join(violations[:20])
        more = f"\n  ... and {len(violations) - 20} more" if len(violations) > 20 else ""
        raise GraphVocabError(f"Graph validation failed ({len(violations)} issues):\n  - {summary}{more}")


def validate_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Validate exported node/edge DataFrames.

    Expects at least columns:
      - nodes: ``node_id``, ``node_type``
      - edges: ``edge_id``, ``from_node``, ``to_node``
    """
    violations: list[str] = []

    required_node_cols = {"node_id", "node_type"}
    required_edge_cols = {"edge_id", "from_node", "to_node"}
    missing_node_cols = required_node_cols - set(nodes.columns)
    missing_edge_cols = required_edge_cols - set(edges.columns)
    if missing_node_cols:
        raise GraphVocabError(f"nodes DataFrame missing columns: {sorted(missing_node_cols)}")
    if missing_edge_cols:
        raise GraphVocabError(f"edges DataFrame missing columns: {sorted(missing_edge_cols)}")

    valid_kinds = {k.value for k in NodeKind}
    kind_by_node: Dict[str, str] = dict(zip(nodes["node_id"].astype(str), nodes["node_type"].astype(str)))

    dup = nodes["node_id"].astype(str).duplicated()
    if dup.any():
        for node_id in nodes.loc[dup, "node_id"].astype(str).head(10):
            violations.append(f"duplicate node_id {node_id!r}")

    for node_id, node_type in kind_by_node.items():
        if node_type not in valid_kinds:
            violations.append(f"node {node_id!r}: unknown node_type {node_type!r}")

    kind_by_prefix = {prefix: kind for kind, prefix in EDGE_ID_PREFIX.items()}
    for row in edges.itertuples(index=False):
        edge_id = str(row.edge_id)
        u, v = str(row.from_node), str(row.to_node)
        if u not in kind_by_node:
            violations.append(f"edge {edge_id!r}: from_node {u!r} missing from nodes")
            continue
        if v not in kind_by_node:
            violations.append(f"edge {edge_id!r}: to_node {v!r} missing from nodes")
            continue
        matched_kind: Optional[EdgeKind] = None
        for prefix, kind in kind_by_prefix.items():
            if edge_id.startswith(prefix):
                matched_kind = kind
                break
        if matched_kind is None:
            continue
        u_kind_raw, v_kind_raw = kind_by_node[u], kind_by_node[v]
        if u_kind_raw not in valid_kinds or v_kind_raw not in valid_kinds:
            continue
        pair = (NodeKind(u_kind_raw), NodeKind(v_kind_raw))
        if pair not in ALLOWED_ENDPOINTS[matched_kind]:
            violations.append(
                f"edge {edge_id!r}: kind {matched_kind.value!r} does not allow "
                f"({u_kind_raw}, {v_kind_raw})"
            )

    if violations:
        summary = "\n  - ".join(violations[:20])
        more = f"\n  ... and {len(violations) - 20} more" if len(violations) > 20 else ""
        raise GraphVocabError(f"Graph validation failed ({len(violations)} issues):\n  - {summary}{more}")


__all__ = [
    "GraphVocabError",
    "NodeKind",
    "EdgeKind",
    "NODE_ID_PREFIX",
    "EDGE_ID_PREFIX",
    "ALLOWED_ENDPOINTS",
    "make_node",
    "make_edge",
    "validate_network",
    "validate_graph",
]
