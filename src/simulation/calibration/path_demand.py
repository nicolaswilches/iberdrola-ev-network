"""Path-flow calibration for the geometry-backed ABM graph.

This module estimates non-negative demand on candidate ABM paths so aggregate
path usage matches the BEV segment targets in ``demand_per_segment.csv``.
It keeps multiple geometry-backed alternatives per OD pair, so parallel roads
such as A-2 and AP-2 can receive separate calibrated flows.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix, vstack

from models.demand import ODMatrix, ODPair
from models.network import RoadNetwork

logger = logging.getLogger(__name__)

_POSITIVE_FLOW_EPS = 1e-6
_LONG_DISTANCE_THRESHOLD_KM = 250.0
_LOW_VOLUME_MAJOR_CITY_THRESHOLD = 100.0
_PARALLEL_COMPETITION_THRESHOLD = 0.35


@dataclass(frozen=True)
class PathCandidate:
    """One geometry-backed candidate path for an OD pair."""

    path_id: str
    od_pair_id: str
    origin: str
    destination: str
    rank: int
    nodes: Tuple[str, ...]
    road_signature: Tuple[str, ...]
    distance_km: float
    travel_time_min: float
    segment_ids: Tuple[int, ...]


def calibrate_path_demand(
    network: RoadNetwork,
    od_matrix: ODMatrix,
    demand_df: pd.DataFrame,
    max_paths_per_od: int = 8,
    max_od_pairs: int | None = None,
    solver_max_iter: int = 500,
    solver_tol: float = 1e-4,
    out_dir: Path | None = None,
    candidate_mode: str = "diagnostic",
    calibration_objective: str = "sqrt",
    relative_error_floor: float = 100.0,
    top_contributors_per_segment: int = 5,
    od_conservation_weight: float = 0.5,
) -> dict:
    """Calibrate non-negative candidate path flows against segment demand.

    Targets come from ``demand_df.daily_bev_traffic_2027`` keyed by
    ``segment_id``. Candidate paths come from k-shortest geometry-backed graph
    paths for existing ABM OD pairs, deduplicated by road-sequence signature.
    """
    if candidate_mode not in {"diagnostic", "boundary", "behavioral"}:
        raise ValueError("candidate_mode must be 'diagnostic', 'boundary', or 'behavioral'")
    if calibration_objective not in {"absolute", "relative", "sqrt"}:
        raise ValueError("calibration_objective must be 'absolute', 'relative', or 'sqrt'")
    if od_conservation_weight < 0.0:
        raise ValueError("od_conservation_weight must be non-negative")

    targets = _segment_targets(demand_df)
    raw_od_pairs = _selected_od_pairs(od_matrix.pairs, max_od_pairs=max_od_pairs)
    od_pairs, excluded_synthetic_pairs = _filter_od_pairs_for_mode(raw_od_pairs, candidate_mode)
    candidates = build_path_candidates(
        network,
        od_pairs,
        max_paths_per_od=max_paths_per_od,
        include_roadspan_candidates=(candidate_mode == "diagnostic"),
        include_local_access_candidates=(candidate_mode == "boundary"),
    )
    od_targets = _od_target_df(od_pairs)
    path_df = _path_candidates_df(candidates, od_targets=od_targets)

    if not candidates:
        raise RuntimeError("No path candidates generated; cannot calibrate ABM path demand")

    matrix, row_segment_ids, covered_targets = _build_incidence_matrix(candidates, targets)
    segment_solve_matrix, segment_solve_targets, row_weights = _weighted_calibration_system(
        matrix,
        covered_targets,
        objective=calibration_objective,
        relative_error_floor=relative_error_floor,
    )
    od_solve_matrix, od_solve_targets, od_row_weights, od_constraint_summary = _weighted_od_conservation_system(
        candidates,
        od_targets,
        objective=calibration_objective,
        relative_error_floor=relative_error_floor,
        od_conservation_weight=od_conservation_weight,
    )
    solve_matrix = segment_solve_matrix
    solve_targets = segment_solve_targets
    if od_solve_matrix is not None and od_solve_targets is not None:
        solve_matrix = vstack([segment_solve_matrix, od_solve_matrix]).tocsr()
        solve_targets = np.concatenate([segment_solve_targets, od_solve_targets])
    solution = lsq_linear(
        solve_matrix,
        solve_targets,
        bounds=(0.0, np.inf),
        lsmr_tol="auto",
        max_iter=solver_max_iter,
        tol=solver_tol,
    )
    flows = np.asarray(solution.x, dtype=float)
    simulated = np.asarray(matrix @ flows, dtype=float)

    segment_report = _segment_report(row_segment_ids, covered_targets, simulated, demand_df)
    path_flows = path_df.copy()
    path_flows["calibrated_daily_bev_flow"] = flows
    od_flow_report = _od_flow_report(path_flows, od_targets)
    segment_diagnostics = _segment_calibration_diagnostics(
        candidates,
        flows,
        segment_report,
    )
    coverage_diagnostics = build_segment_od_coverage_diagnostics(
        path_df,
        path_flows,
        segment_report,
    )
    competition_diagnostics = build_segment_competition_diagnostics(
        path_df,
        path_flows,
        segment_report,
    )
    segment_od_coverage_diagnostics = _segment_od_coverage_export(
        segment_diagnostics,
        coverage_diagnostics,
        competition_diagnostics,
    )
    top_contributors = _top_segment_path_contributors(
        candidates,
        path_flows,
        row_segment_ids,
        top_n=top_contributors_per_segment,
    )
    road_summary = _road_calibration_summary(segment_diagnostics)
    road_od_coverage_summary = _road_od_coverage_summary(
        segment_od_coverage_diagnostics,
        competition_diagnostics,
    )

    uncovered = sorted(set(targets) - set(row_segment_ids))
    uncovered_target = float(sum(targets[sid] for sid in uncovered))
    covered_target = float(np.sum(covered_targets))
    total_target = covered_target + uncovered_target
    accepted = segment_report[
        (segment_report["target_daily_bev_traffic_2027"] <= 25.0)
        | (segment_report["abs_error_pct"] <= 10.0)
    ]
    summary = {
        "candidate_mode": candidate_mode,
        "n_od_pairs": len(od_pairs),
        "n_unique_od_pairs": int(len(od_targets)),
        "n_synthetic_endpoint_od_pairs_excluded": excluded_synthetic_pairs,
        "n_path_candidates": len(candidates),
        "n_target_segments": len(targets),
        "n_covered_segments": int(len(row_segment_ids)),
        "n_uncovered_segments": int(len(uncovered)),
        "covered_target_daily_bev": covered_target,
        "uncovered_target_daily_bev": uncovered_target,
        "covered_target_share_pct": (covered_target / total_target * 100.0) if total_target else 0.0,
        "weighted_mape_pct": _weighted_mape(segment_report),
        "weighted_mape_by_objective_pct": _objective_weighted_mape(
            segment_report,
            objective=calibration_objective,
            relative_error_floor=relative_error_floor,
        ),
        "segments_within_10pct_or_25bev": int(len(accepted)),
        "segments_within_10pct_or_25bev_pct": float(len(accepted) / len(segment_report) * 100.0) if len(segment_report) else 0.0,
        "solver_status": int(solution.status),
        "solver_converged": bool(solution.success),
        "solver_message": str(solution.message),
        "cost": float(solution.cost),
        "calibration_objective": calibration_objective,
        "relative_error_floor": float(relative_error_floor),
        "row_weight_min": float(np.min(row_weights)) if len(row_weights) else 0.0,
        "row_weight_max": float(np.max(row_weights)) if len(row_weights) else 0.0,
        "od_conservation_weight": float(od_conservation_weight),
        "n_od_conservation_targets": int(od_constraint_summary["n_targeted_od_pairs"]),
        "n_od_conservation_targets_with_candidates": int(od_constraint_summary["n_constrained_od_pairs"]),
        "n_od_conservation_targets_without_candidates": int(od_constraint_summary["n_unconstrained_od_pairs"]),
        "od_row_weight_min": float(np.min(od_row_weights)) if len(od_row_weights) else 0.0,
        "od_row_weight_max": float(np.max(od_row_weights)) if len(od_row_weights) else 0.0,
        "od_weighted_mape_pct": _od_weighted_mape(od_flow_report),
    }
    summary.update(_od_coverage_summary_fields(segment_od_coverage_diagnostics, road_od_coverage_summary))

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path_df.to_csv(out_dir / "abm_path_candidates.csv", index=False)
        path_flows.to_csv(out_dir / "abm_calibrated_path_flows.csv", index=False)
        od_flow_report.to_csv(out_dir / "abm_od_flow_calibration.csv", index=False)
        segment_report.to_csv(out_dir / "abm_segment_flow_calibration.csv", index=False)
        segment_diagnostics.to_csv(out_dir / "abm_segment_calibration_diagnostics.csv", index=False)
        segment_od_coverage_diagnostics.to_csv(out_dir / "abm_segment_od_coverage_diagnostics.csv", index=False)
        top_contributors.to_csv(out_dir / "abm_segment_top_path_contributors.csv", index=False)
        road_summary.to_csv(out_dir / "abm_road_calibration_summary.csv", index=False)
        road_od_coverage_summary.to_csv(out_dir / "abm_road_od_coverage_summary.csv", index=False)
        pd.DataFrame({"segment_id": uncovered, "target_daily_bev_traffic_2027": [targets[s] for s in uncovered]}).to_csv(
            out_dir / "abm_uncovered_segment_targets.csv", index=False
        )
        (out_dir / "abm_path_calibration_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    return {
        "summary": summary,
        "path_candidates": path_df,
        "path_flows": path_flows,
        "od_flow_report": od_flow_report,
        "segment_report": segment_report,
        "segment_diagnostics": segment_diagnostics,
        "segment_od_coverage_diagnostics": segment_od_coverage_diagnostics,
        "top_contributors": top_contributors,
        "road_summary": road_summary,
        "road_od_coverage_summary": road_od_coverage_summary,
        "uncovered_segments": uncovered,
    }


def build_path_candidates(
    network: RoadNetwork,
    od_pairs: Sequence[ODPair],
    max_paths_per_od: int = 8,
    include_roadspan_candidates: bool = True,
    include_local_access_candidates: bool = False,
) -> List[PathCandidate]:
    """Generate k geometry-backed paths for each OD pair, preserving AP/A alternatives."""
    candidates: List[PathCandidate] = []
    seen_candidates: set[Tuple[str, str, Tuple[str, ...], Tuple[int, ...]]] = set()
    for pair_idx, pair in enumerate(od_pairs):
        paths = _k_geometry_paths(network, pair.origin, pair.destination, max_paths_per_od)
        seen_signatures: set[Tuple[str, ...]] = set()
        rank = 0
        for nodes in paths:
            if not _path_is_geometry_backed(network, nodes):
                continue
            signature = _road_signature(network, nodes)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            seg_ids = _path_segment_ids(network, nodes)
            key = (pair.origin, pair.destination, signature, seg_ids)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidates.append(
                PathCandidate(
                    path_id=f"PATH_{len(candidates):07d}",
                    od_pair_id=pair.pair_id,
                    origin=pair.origin,
                    destination=pair.destination,
                    rank=rank,
                    nodes=tuple(nodes),
                    road_signature=signature,
                    distance_km=network.subpath_distance_km(nodes),
                    travel_time_min=network.subpath_travel_time_min(nodes),
                    segment_ids=seg_ids,
                )
            )
            rank += 1
    if include_roadspan_candidates:
        candidates.extend(_road_span_candidates(network, len(candidates), seen_candidates))
    if include_local_access_candidates:
        candidates.extend(_local_access_candidates(network, len(candidates), seen_candidates))
    return candidates


def _local_access_candidates(
    network: RoadNetwork,
    start_idx: int,
    seen_candidates: set[Tuple[str, str, Tuple[str, ...], Tuple[int, ...]]],
) -> List[PathCandidate]:
    """Add same-road local-access OD paths between real graph waypoints."""
    candidates: List[PathCandidate] = []
    for road, component_nodes in _same_road_node_components(network).items():
        for component_idx, nodes in enumerate(component_nodes):
            if len(nodes) < 2:
                continue
            for i in range(len(nodes) - 1):
                for j in range(i + 1, len(nodes)):
                    for direction_idx, path_nodes in enumerate((nodes[i : j + 1], list(reversed(nodes[i : j + 1])))):
                        signature = _road_signature(network, path_nodes)
                        seg_ids = _path_segment_ids(network, path_nodes)
                        key = (path_nodes[0], path_nodes[-1], signature, seg_ids)
                        if key in seen_candidates:
                            continue
                        seen_candidates.add(key)
                        path_idx = start_idx + len(candidates)
                        suffix = "FWD" if direction_idx == 0 else "REV"
                        candidates.append(
                            PathCandidate(
                                path_id=f"PATH_{path_idx:07d}",
                                od_pair_id=f"LOCALOD_{road}_{component_idx:03d}_{i:03d}_{j:03d}_{suffix}",
                                origin=path_nodes[0],
                                destination=path_nodes[-1],
                                rank=direction_idx,
                                nodes=tuple(path_nodes),
                                road_signature=signature,
                                distance_km=network.subpath_distance_km(path_nodes),
                                travel_time_min=network.subpath_travel_time_min(path_nodes),
                                segment_ids=seg_ids,
                            )
                        )
    return candidates


def _same_road_node_components(network: RoadNetwork) -> dict[str, List[List[str]]]:
    nodes_by_road: dict[str, dict[str, float]] = defaultdict(dict)
    for u, v, attrs in network.graph.edges(data=True):
        road = str(attrs.get("road_name", ""))
        if not road or not attrs.get("geometry_backed", False):
            continue
        _record_road_km(nodes_by_road[road], u, attrs.get("from_road_km"))
        _record_road_km(nodes_by_road[road], v, attrs.get("to_road_km"))

    components: dict[str, List[List[str]]] = {}
    for road, node_km in sorted(nodes_by_road.items()):
        ordered_nodes = [
            node for node, _km in sorted(node_km.items(), key=lambda item: (item[1], item[0]))
        ]
        components[road] = _same_road_components(network, road, ordered_nodes)
    return components


def _road_span_candidates(
    network: RoadNetwork,
    start_idx: int,
    seen_candidates: set[Tuple[str, str, Tuple[str, ...], Tuple[int, ...]]],
) -> List[PathCandidate]:
    """Add same-road calibration candidates so every geometry-backed span can carry demand."""
    candidates: List[PathCandidate] = []
    for road, component_nodes in _same_road_node_components(network).items():
        for component_idx, nodes in enumerate(component_nodes):
            if len(nodes) < 2:
                continue
            for direction_idx, path_nodes in enumerate((nodes, list(reversed(nodes)))):
                if not _path_is_geometry_backed(network, path_nodes):
                    continue
                signature = _road_signature(network, path_nodes)
                seg_ids = _path_segment_ids(network, path_nodes)
                key = (path_nodes[0], path_nodes[-1], signature, seg_ids)
                if key in seen_candidates:
                    continue
                seen_candidates.add(key)
                path_idx = start_idx + len(candidates)
                suffix = "FWD" if direction_idx == 0 else "REV"
                candidates.append(
                    PathCandidate(
                        path_id=f"PATH_{path_idx:07d}",
                        od_pair_id=f"ROADSPAN_{road}_{component_idx:03d}_{suffix}",
                        origin=path_nodes[0],
                        destination=path_nodes[-1],
                        rank=direction_idx,
                        nodes=tuple(path_nodes),
                        road_signature=signature,
                        distance_km=network.subpath_distance_km(path_nodes),
                        travel_time_min=network.subpath_travel_time_min(path_nodes),
                        segment_ids=seg_ids,
                    )
                )
    return candidates


def _record_road_km(node_km: dict[str, float], node: str, raw_km: object) -> None:
    try:
        km = float(raw_km)
    except (TypeError, ValueError):
        return
    current = node_km.get(node)
    if current is None or km < current:
        node_km[node] = km


def _same_road_components(network: RoadNetwork, road: str, ordered_nodes: Sequence[str]) -> List[List[str]]:
    components: List[List[str]] = []
    current: List[str] = []
    for node in ordered_nodes:
        if not current:
            current = [node]
            continue
        attrs = network.graph.get_edge_data(current[-1], node)
        if attrs and attrs.get("road_name") == road and attrs.get("geometry_backed", False):
            current.append(node)
            continue
        if len(current) >= 2:
            components.append(current)
        current = [node]
    if len(current) >= 2:
        components.append(current)
    return components


def _selected_od_pairs(od_pairs: Sequence[ODPair], max_od_pairs: int | None) -> List[ODPair]:
    pairs = sorted(od_pairs, key=lambda p: p.daily_bev_trips, reverse=True)
    return pairs[:max_od_pairs] if max_od_pairs else pairs


def _filter_od_pairs_for_mode(
    od_pairs: Sequence[ODPair],
    candidate_mode: str,
) -> Tuple[List[ODPair], int]:
    if candidate_mode in {"diagnostic", "boundary"}:
        return list(od_pairs), 0
    filtered = [
        pair for pair in od_pairs
        if not (_is_synthetic_endpoint(pair.origin) or _is_synthetic_endpoint(pair.destination))
    ]
    return filtered, len(od_pairs) - len(filtered)


def _is_synthetic_endpoint(node_id: str) -> bool:
    return str(node_id).startswith("GEO_")


def _k_geometry_paths(
    network: RoadNetwork,
    origin: str,
    destination: str,
    k: int,
) -> List[List[str]]:
    try:
        gen = nx.shortest_simple_paths(network.graph, origin, destination, weight="travel_time_min")
        paths = []
        for path in gen:
            if _path_is_geometry_backed(network, path):
                paths.append(path)
            if len(paths) >= k:
                break
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def _path_is_geometry_backed(network: RoadNetwork, nodes: Sequence[str]) -> bool:
    return all(attrs.get("geometry_backed", False) for _, _, attrs in network.iter_edges_on_path(list(nodes)))


def _road_signature(network: RoadNetwork, nodes: Sequence[str]) -> Tuple[str, ...]:
    roads: List[str] = []
    for _, _, attrs in network.iter_edges_on_path(list(nodes)):
        if str(attrs.get("road_type", "")) == "connector":
            continue
        road = str(attrs.get("road_name", ""))
        if road and (not roads or roads[-1] != road):
            roads.append(road)
    return tuple(roads)


def _path_segment_ids(network: RoadNetwork, nodes: Sequence[str]) -> Tuple[int, ...]:
    ids: set[int] = set()
    for _, _, attrs in network.iter_edges_on_path(list(nodes)):
        raw = attrs.get("source_segment_ids", ())
        for sid in raw:
            try:
                ids.add(int(sid))
            except (TypeError, ValueError):
                continue
    return tuple(sorted(ids))


def _segment_targets(demand_df: pd.DataFrame) -> Dict[int, float]:
    required = {"segment_id", "daily_bev_traffic_2027"}
    if not required.issubset(demand_df.columns):
        raise ValueError(f"demand_df must contain {required}")
    clean = demand_df[list(required)].dropna()
    return {
        int(row["segment_id"]): float(row["daily_bev_traffic_2027"])
        for _, row in clean.iterrows()
    }


def _build_incidence_matrix(
    candidates: Sequence[PathCandidate],
    targets: Dict[int, float],
):
    covered_segments = sorted(set().union(*(set(c.segment_ids) for c in candidates)) & set(targets))
    row_by_segment = {sid: i for i, sid in enumerate(covered_segments)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for col, candidate in enumerate(candidates):
        for sid in candidate.segment_ids:
            row = row_by_segment.get(sid)
            if row is None:
                continue
            rows.append(row)
            cols.append(col)
            data.append(1.0)
    matrix = coo_matrix((data, (rows, cols)), shape=(len(covered_segments), len(candidates))).tocsr()
    target_vec = np.array([targets[sid] for sid in covered_segments], dtype=float)
    return matrix, covered_segments, target_vec


def _is_support_od_pair_id(od_pair_id: str) -> bool:
    od_id = str(od_pair_id or "")
    return od_id.startswith("ROADSPAN_") or od_id.startswith("LOCALOD_")


def _row_weights_for_targets(
    targets: np.ndarray,
    objective: str,
    relative_error_floor: float,
) -> np.ndarray:
    if objective == "absolute":
        row_scale = np.ones_like(targets, dtype=float)
    elif objective == "sqrt":
        row_scale = np.sqrt(np.maximum(targets, 1.0))
    else:
        row_scale = np.maximum(targets, float(relative_error_floor))
    row_scale = np.maximum(row_scale, 1e-9)
    return 1.0 / row_scale


def _weighted_calibration_system(
    matrix,
    targets: np.ndarray,
    objective: str,
    relative_error_floor: float,
):
    row_weights = _row_weights_for_targets(targets, objective, relative_error_floor)
    return matrix.multiply(row_weights[:, None]), targets * row_weights, row_weights


def _od_target_df(od_pairs: Sequence[ODPair]) -> pd.DataFrame:
    rows = []
    for pair in od_pairs:
        rows.append({
            "od_pair_id": pair.pair_id,
            "origin": pair.origin,
            "destination": pair.destination,
            "purpose": pair.purpose,
            "target_daily_bev_trips": float(pair.daily_bev_trips),
            "is_support_od": _is_support_od_pair_id(pair.pair_id),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "od_pair_id",
            "origin",
            "destination",
            "purpose",
            "target_daily_bev_trips",
            "is_support_od",
            "source_pair_count",
        ])
    df = pd.DataFrame(rows)
    grouped = df.groupby(["od_pair_id", "origin", "destination"], dropna=False).agg(
        purpose=("purpose", lambda s: str(s.iloc[0]) if s.nunique(dropna=False) == 1 else "mixed"),
        target_daily_bev_trips=("target_daily_bev_trips", "sum"),
        is_support_od=("is_support_od", "max"),
        source_pair_count=("od_pair_id", "size"),
    ).reset_index()
    grouped["is_support_od"] = grouped["is_support_od"].fillna(False).astype(bool)
    grouped["source_pair_count"] = grouped["source_pair_count"].fillna(0).astype(int)
    return grouped


def _weighted_od_conservation_system(
    candidates: Sequence[PathCandidate],
    od_targets: pd.DataFrame,
    objective: str,
    relative_error_floor: float,
    od_conservation_weight: float,
):
    if od_targets.empty:
        return None, None, np.array([], dtype=float), {
            "n_targeted_od_pairs": 0,
            "n_constrained_od_pairs": 0,
            "n_unconstrained_od_pairs": 0,
        }

    od_targets = od_targets.copy()
    od_targets = od_targets[~od_targets["is_support_od"].fillna(False)].copy()
    if od_targets.empty:
        return None, None, np.array([], dtype=float), {
            "n_targeted_od_pairs": 0,
            "n_constrained_od_pairs": 0,
            "n_unconstrained_od_pairs": 0,
        }

    row_by_od = {str(od_id): idx for idx, od_id in enumerate(od_targets["od_pair_id"].astype(str))}
    candidate_counts: dict[str, int] = defaultdict(int)
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for col, candidate in enumerate(candidates):
        row = row_by_od.get(str(candidate.od_pair_id))
        if row is None:
            continue
        rows.append(row)
        cols.append(col)
        data.append(1.0)
        candidate_counts[str(candidate.od_pair_id)] += 1

    od_targets["candidate_path_count"] = od_targets["od_pair_id"].astype(str).map(candidate_counts).fillna(0).astype(int)
    constrained = od_targets[od_targets["candidate_path_count"] > 0].copy()
    summary = {
        "n_targeted_od_pairs": int(len(od_targets)),
        "n_constrained_od_pairs": int(len(constrained)),
        "n_unconstrained_od_pairs": int(len(od_targets) - len(constrained)),
    }
    if constrained.empty:
        return None, None, np.array([], dtype=float), summary

    constrained_row_by_od = {
        str(od_id): idx for idx, od_id in enumerate(constrained["od_pair_id"].astype(str))
    }
    filtered_rows: List[int] = []
    filtered_cols: List[int] = []
    filtered_data: List[float] = []
    for row, col, value in zip(rows, cols, data):
        od_id = str(od_targets.iloc[row]["od_pair_id"])
        constrained_row = constrained_row_by_od.get(od_id)
        if constrained_row is None:
            continue
        filtered_rows.append(constrained_row)
        filtered_cols.append(col)
        filtered_data.append(value)

    matrix = coo_matrix(
        (filtered_data, (filtered_rows, filtered_cols)),
        shape=(len(constrained), len(candidates)),
    ).tocsr()
    target_vec = constrained["target_daily_bev_trips"].to_numpy(dtype=float)
    row_weights = _row_weights_for_targets(
        target_vec,
        objective=objective,
        relative_error_floor=relative_error_floor,
    ) * float(max(od_conservation_weight, 0.0))
    return matrix.multiply(row_weights[:, None]), target_vec * row_weights, row_weights, summary


def _is_geo_node(node_id: str) -> bool:
    return str(node_id).startswith("GEO_")


def _is_major_city_node(node_id: str) -> bool:
    """Canonical OD matrix city nodes are compact uppercase tokens like MAD/BCN."""
    node = str(node_id or "").strip()
    if not node or _is_geo_node(node):
        return False
    if re.search(r"_EC\d+$", node):
        return False
    if "_" in node:
        return False
    return bool(re.fullmatch(r"[A-Z]{2,5}", node))


def _path_distance_bucket(distance_km: float) -> str:
    distance = float(distance_km)
    if distance < 100.0:
        return "short"
    if distance < _LONG_DISTANCE_THRESHOLD_KM:
        return "medium"
    return "long"


def _is_long_distance(distance_km: float) -> bool:
    return float(distance_km) >= _LONG_DISTANCE_THRESHOLD_KM


def _parse_segment_ids(raw: object) -> Tuple[int, ...]:
    values: List[int] = []
    if isinstance(raw, (list, tuple, set, np.ndarray)):
        items = raw
    else:
        items = str(raw or "").split("|")
    for item in items:
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(set(values)))


def _parse_road_signature(raw: object) -> Tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        items = [str(item).strip() for item in raw]
    else:
        items = [part.strip() for part in str(raw or "").split("|")]
    roads: List[str] = []
    for road in items:
        if road and road not in roads:
            roads.append(road)
    return tuple(roads)


def _od_flow_report(
    path_flows: pd.DataFrame,
    od_targets: pd.DataFrame,
) -> pd.DataFrame:
    base = od_targets.copy()
    if base.empty:
        return pd.DataFrame(columns=[
            "od_pair_id",
            "origin",
            "destination",
            "purpose",
            "target_daily_bev_trips",
            "candidate_path_count",
            "positive_path_count",
            "calibrated_daily_bev_flow",
            "abs_error",
            "abs_error_pct",
            "is_support_od",
        ])

    flow_df = path_flows.copy()
    flow_df["od_pair_id"] = flow_df["od_pair_id"].astype(str)
    flow_df["calibrated_daily_bev_flow"] = pd.to_numeric(
        flow_df["calibrated_daily_bev_flow"],
        errors="coerce",
    ).fillna(0.0)
    grouped = flow_df.groupby("od_pair_id", dropna=False).agg(
        candidate_path_count=("path_id", "count"),
        positive_path_count=("calibrated_daily_bev_flow", lambda s: int((s > _POSITIVE_FLOW_EPS).sum())),
        calibrated_daily_bev_flow=("calibrated_daily_bev_flow", "sum"),
    ).reset_index()
    report = base.merge(grouped, on="od_pair_id", how="left")
    report["candidate_path_count"] = report["candidate_path_count"].fillna(0).astype(int)
    report["positive_path_count"] = report["positive_path_count"].fillna(0).astype(int)
    report["calibrated_daily_bev_flow"] = pd.to_numeric(
        report["calibrated_daily_bev_flow"],
        errors="coerce",
    ).fillna(0.0)
    report["abs_error"] = (report["calibrated_daily_bev_flow"] - report["target_daily_bev_trips"]).abs()
    report["abs_error_pct"] = (
        report["abs_error"]
        / report["target_daily_bev_trips"].replace(0, np.nan)
        * 100.0
    )
    return report.sort_values("abs_error", ascending=False)


def _od_weighted_mape(od_flow_report: pd.DataFrame) -> float:
    if od_flow_report.empty:
        return 0.0
    target = pd.to_numeric(od_flow_report["target_daily_bev_trips"], errors="coerce").fillna(0.0)
    abs_error = pd.to_numeric(od_flow_report["abs_error"], errors="coerce").fillna(0.0)
    denom = float(target.sum())
    return float(abs_error.sum() / denom * 100.0) if denom > 0 else 0.0


def _prepared_candidate_path_frame(
    candidates: pd.DataFrame | Sequence[PathCandidate],
    path_flows: pd.DataFrame,
) -> pd.DataFrame:
    candidate_df = candidates.copy() if isinstance(candidates, pd.DataFrame) else _path_candidates_df(candidates)
    if candidate_df.empty:
        return pd.DataFrame(columns=[
            "path_id",
            "od_pair_id",
            "origin",
            "destination",
            "distance_km",
            "road_signature",
            "source_segment_ids",
            "calibrated_daily_bev_flow",
        ])

    required = {
        "path_id",
        "od_pair_id",
        "origin",
        "destination",
        "distance_km",
        "road_signature",
        "source_segment_ids",
    }
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(f"candidate rows missing columns: {sorted(missing)}")

    flow_df = path_flows.copy()
    if "path_id" not in flow_df.columns or "calibrated_daily_bev_flow" not in flow_df.columns:
        raise ValueError("path_flows must contain path_id and calibrated_daily_bev_flow")

    candidate_df["path_id"] = candidate_df["path_id"].astype(str)
    flow_df["path_id"] = flow_df["path_id"].astype(str)
    flow_series = pd.to_numeric(
        flow_df.set_index("path_id")["calibrated_daily_bev_flow"],
        errors="coerce",
    ).fillna(0.0)
    candidate_df["calibrated_daily_bev_flow"] = candidate_df["path_id"].map(flow_series).fillna(0.0)
    candidate_df["distance_km"] = pd.to_numeric(candidate_df["distance_km"], errors="coerce").fillna(0.0)
    candidate_df["segment_ids_parsed"] = candidate_df["source_segment_ids"].apply(_parse_segment_ids)
    candidate_df["road_signature_parsed"] = candidate_df["road_signature"].apply(_parse_road_signature)
    endpoint_nodes = set(candidate_df["origin"].astype(str)) | set(candidate_df["destination"].astype(str))
    major_city_nodes = {
        node for node in endpoint_nodes
        if _is_major_city_node(node)
    }
    candidate_df["is_major_city_od"] = (
        candidate_df["origin"].astype(str).isin(major_city_nodes)
        & candidate_df["destination"].astype(str).isin(major_city_nodes)
    )
    candidate_df["is_long_distance"] = candidate_df["distance_km"].apply(_is_long_distance)
    candidate_df["is_major_city_long_distance_od"] = (
        candidate_df["is_major_city_od"] & candidate_df["is_long_distance"]
    )
    candidate_df["distance_bucket"] = candidate_df["distance_km"].apply(_path_distance_bucket)
    candidate_df["is_positive_flow"] = candidate_df["calibrated_daily_bev_flow"] > _POSITIVE_FLOW_EPS
    return candidate_df


def build_segment_od_coverage_diagnostics(
    candidates: pd.DataFrame | Sequence[PathCandidate],
    path_flows: pd.DataFrame,
    segment_report: pd.DataFrame,
) -> pd.DataFrame:
    candidate_df = _prepared_candidate_path_frame(candidates, path_flows)
    metrics: dict[int, dict[str, object]] = {}

    def _entry(segment_id: int) -> dict[str, object]:
        return metrics.setdefault(segment_id, {
            "candidate_paths": set(),
            "candidate_ods": set(),
            "candidate_major_city_ods": set(),
            "candidate_long_distance_ods": set(),
            "candidate_major_city_long_distance_ods": set(),
            "positive_paths": set(),
            "positive_ods": set(),
            "positive_major_city_ods": set(),
            "positive_long_distance_ods": set(),
            "positive_major_city_long_distance_ods": set(),
            "positive_path_flow_sum": 0.0,
            "major_city_positive_flow_sum": 0.0,
            "long_distance_positive_flow_sum": 0.0,
            "major_city_long_distance_positive_flow_sum": 0.0,
            "short_distance_positive_path_count": 0,
            "medium_distance_positive_path_count": 0,
            "long_distance_positive_path_count": 0,
        })

    for row in candidate_df.itertuples(index=False):
        segment_ids = set(getattr(row, "segment_ids_parsed", ()))
        if not segment_ids:
            continue
        flow = float(row.calibrated_daily_bev_flow)
        is_positive = bool(getattr(row, "is_positive_flow"))
        for segment_id in segment_ids:
            entry = _entry(int(segment_id))
            entry["candidate_paths"].add(str(row.path_id))
            entry["candidate_ods"].add(str(row.od_pair_id))
            if bool(getattr(row, "is_major_city_od")):
                entry["candidate_major_city_ods"].add(str(row.od_pair_id))
            if bool(getattr(row, "is_long_distance")):
                entry["candidate_long_distance_ods"].add(str(row.od_pair_id))
            if bool(getattr(row, "is_major_city_long_distance_od")):
                entry["candidate_major_city_long_distance_ods"].add(str(row.od_pair_id))
            if not is_positive:
                continue
            entry["positive_paths"].add(str(row.path_id))
            entry["positive_ods"].add(str(row.od_pair_id))
            entry["positive_path_flow_sum"] += flow
            if bool(getattr(row, "is_major_city_od")):
                entry["positive_major_city_ods"].add(str(row.od_pair_id))
                entry["major_city_positive_flow_sum"] += flow
            if bool(getattr(row, "is_long_distance")):
                entry["positive_long_distance_ods"].add(str(row.od_pair_id))
                entry["long_distance_positive_flow_sum"] += flow
            if bool(getattr(row, "is_major_city_long_distance_od")):
                entry["positive_major_city_long_distance_ods"].add(str(row.od_pair_id))
                entry["major_city_long_distance_positive_flow_sum"] += flow
            bucket = str(getattr(row, "distance_bucket"))
            entry[f"{bucket}_distance_positive_path_count"] += 1

    rows: List[Dict[str, object]] = []
    for row in segment_report.itertuples(index=False):
        segment_id = int(row.segment_id)
        entry = metrics.get(segment_id, {})
        segment_flow = float(getattr(row, "calibrated_daily_bev_flow", 0.0) or 0.0)
        rows.append({
            "segment_id": segment_id,
            "candidate_path_count": len(entry.get("candidate_paths", set())),
            "candidate_od_pair_count": len(entry.get("candidate_ods", set())),
            "candidate_major_city_od_pair_count": len(entry.get("candidate_major_city_ods", set())),
            "candidate_long_distance_od_pair_count": len(entry.get("candidate_long_distance_ods", set())),
            "candidate_major_city_long_distance_od_pair_count": len(entry.get("candidate_major_city_long_distance_ods", set())),
            "positive_path_count": len(entry.get("positive_paths", set())),
            "positive_od_pair_count": len(entry.get("positive_ods", set())),
            "positive_major_city_od_pair_count": len(entry.get("positive_major_city_ods", set())),
            "positive_long_distance_od_pair_count": len(entry.get("positive_long_distance_ods", set())),
            "positive_major_city_long_distance_od_pair_count": len(entry.get("positive_major_city_long_distance_ods", set())),
            "positive_path_flow_sum": float(entry.get("positive_path_flow_sum", 0.0)),
            "major_city_positive_flow_sum": float(entry.get("major_city_positive_flow_sum", 0.0)),
            "long_distance_positive_flow_sum": float(entry.get("long_distance_positive_flow_sum", 0.0)),
            "major_city_long_distance_positive_flow_sum": float(entry.get("major_city_long_distance_positive_flow_sum", 0.0)),
            "major_city_flow_share": float(entry.get("major_city_positive_flow_sum", 0.0)) / segment_flow if segment_flow > 0 else 0.0,
            "long_distance_flow_share": float(entry.get("long_distance_positive_flow_sum", 0.0)) / segment_flow if segment_flow > 0 else 0.0,
            "major_city_long_distance_flow_share": float(entry.get("major_city_long_distance_positive_flow_sum", 0.0)) / segment_flow if segment_flow > 0 else 0.0,
            "short_distance_positive_path_count": int(entry.get("short_distance_positive_path_count", 0)),
            "medium_distance_positive_path_count": int(entry.get("medium_distance_positive_path_count", 0)),
            "long_distance_positive_path_count": int(entry.get("long_distance_positive_path_count", 0)),
        })

    return pd.DataFrame(rows)


def build_segment_competition_diagnostics(
    candidates: pd.DataFrame | Sequence[PathCandidate],
    path_flows: pd.DataFrame,
    segment_report: pd.DataFrame,
) -> pd.DataFrame:
    candidate_df = _prepared_candidate_path_frame(candidates, path_flows)
    candidate_ods_by_segment: dict[int, set[str]] = defaultdict(set)
    positive_paths_by_od: dict[str, List[object]] = defaultdict(list)

    for row in candidate_df.itertuples(index=False):
        segment_ids = set(getattr(row, "segment_ids_parsed", ()))
        for segment_id in segment_ids:
            candidate_ods_by_segment[int(segment_id)].add(str(row.od_pair_id))
        if bool(getattr(row, "is_positive_flow")):
            positive_paths_by_od[str(row.od_pair_id)].append(row)

    rows: List[Dict[str, object]] = []
    for row in segment_report.itertuples(index=False):
        segment_id = int(row.segment_id)
        own_road = str(getattr(row, "route_segment", "") or "")
        competing_flow: dict[str, float] = defaultdict(float)
        competing_path_count = 0
        competing_path_flow_sum = 0.0
        for od_pair_id in candidate_ods_by_segment.get(segment_id, set()):
            for alt_path in positive_paths_by_od.get(od_pair_id, []):
                alt_segments = set(getattr(alt_path, "segment_ids_parsed", ()))
                if segment_id in alt_segments:
                    continue
                competing_roads = {
                    road for road in getattr(alt_path, "road_signature_parsed", ())
                    if road and road != own_road
                }
                if not competing_roads:
                    continue
                flow = float(alt_path.calibrated_daily_bev_flow)
                competing_path_count += 1
                competing_path_flow_sum += flow
                for road in competing_roads:
                    competing_flow[str(road)] += flow

        ordered = sorted(competing_flow.items(), key=lambda item: (-item[1], item[0]))
        top_road, top_road_flow = ("", 0.0)
        if ordered:
            top_road, top_road_flow = ordered[0]
        top_3 = ordered[:3]
        segment_flow = float(getattr(row, "calibrated_daily_bev_flow", 0.0) or 0.0)
        rows.append({
            "segment_id": segment_id,
            "competing_positive_path_count": int(competing_path_count),
            "competing_positive_path_flow_sum": float(competing_path_flow_sum),
            "competing_road_count": len(ordered),
            "top_competing_road": top_road,
            "top_competing_road_flow": float(top_road_flow),
            "top_3_competing_roads": "|".join(road for road, _flow in top_3),
            "top_3_competing_roads_flow": float(sum(flow for _road, flow in top_3)),
            "competing_road_signature": "|".join(f"{road}:{flow:.3f}" for road, flow in ordered),
            "competing_road_flow_map": dict(ordered),
            "parallel_competition_score": float(top_road_flow) / max(segment_flow, 1e-9),
        })

    return pd.DataFrame(rows)


def _path_candidates_df(
    candidates: Sequence[PathCandidate],
    od_targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target_map: dict[str, float] = {}
    if od_targets is not None and not od_targets.empty:
        target_map = {
            str(row.od_pair_id): float(row.target_daily_bev_trips)
            for row in od_targets.itertuples(index=False)
        }
    return pd.DataFrame([
        {
            "path_id": c.path_id,
            "od_pair_id": c.od_pair_id,
            "origin": c.origin,
            "destination": c.destination,
            "rank": c.rank,
            "distance_km": c.distance_km,
            "travel_time_min": c.travel_time_min,
            "road_signature": "|".join(c.road_signature),
            "node_path": "|".join(c.nodes),
            "source_segment_ids": "|".join(str(sid) for sid in c.segment_ids),
            "od_target_daily_bev_trips": target_map.get(c.od_pair_id, np.nan),
            "is_support_od": _is_support_od_pair_id(c.od_pair_id),
        }
        for c in candidates
    ])


def _segment_report(
    segment_ids: Sequence[int],
    target: np.ndarray,
    simulated: np.ndarray,
    demand_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "segment_id": segment_ids,
        "target_daily_bev_traffic_2027": target,
        "calibrated_daily_bev_flow": simulated,
    })
    if demand_df is not None and not demand_df.empty and "segment_id" in demand_df.columns:
        optional = [
            col for col in ("route_segment", "length_km", "imd_total", "is_tent", "tent_tier")
            if col in demand_df.columns
        ]
        if optional:
            attrs = demand_df[["segment_id"] + optional].copy()
            attrs["segment_id"] = attrs["segment_id"].astype(int)
            df = df.merge(attrs, on="segment_id", how="left")
    df["abs_error"] = (df["calibrated_daily_bev_flow"] - df["target_daily_bev_traffic_2027"]).abs()
    df["abs_error_pct"] = df["abs_error"] / df["target_daily_bev_traffic_2027"].replace(0, np.nan) * 100.0
    return df.sort_values("abs_error", ascending=False)


def _segment_calibration_diagnostics(
    candidates: Sequence[PathCandidate],
    flows: np.ndarray,
    segment_report: pd.DataFrame,
) -> pd.DataFrame:
    candidate_counts: dict[int, int] = defaultdict(int)
    positive_counts: dict[int, int] = defaultdict(int)
    positive_flow: dict[int, float] = defaultdict(float)
    max_path_flow: dict[int, float] = defaultdict(float)

    for candidate, flow in zip(candidates, flows):
        unique_segments = set(candidate.segment_ids)
        for sid in unique_segments:
            candidate_counts[sid] += 1
            if flow > _POSITIVE_FLOW_EPS:
                positive_counts[sid] += 1
                positive_flow[sid] += float(flow)
                max_path_flow[sid] = max(max_path_flow[sid], float(flow))

    df = segment_report.copy()
    df["candidate_path_count"] = df["segment_id"].map(candidate_counts).fillna(0).astype(int)
    df["positive_path_count"] = df["segment_id"].map(positive_counts).fillna(0).astype(int)
    df["positive_path_flow_sum"] = df["segment_id"].map(positive_flow).fillna(0.0)
    df["max_positive_path_flow"] = df["segment_id"].map(max_path_flow).fillna(0.0)
    df["is_underfit"] = df["calibrated_daily_bev_flow"] < df["target_daily_bev_traffic_2027"]
    df["is_overfit"] = df["calibrated_daily_bev_flow"] > df["target_daily_bev_traffic_2027"]
    df["fit_bucket"] = np.select(
        [
            df["candidate_path_count"] == 0,
            df["positive_path_count"] == 0,
            df["abs_error_pct"] <= 10.0,
            df["is_underfit"],
            df["is_overfit"],
        ],
        [
            "uncovered",
            "covered_zero_flow",
            "within_10pct",
            "underfit",
            "overfit",
        ],
        default="other",
    )
    return df.sort_values("abs_error", ascending=False)


def _segment_diagnostic_label(row: pd.Series) -> str:
    if int(row.get("candidate_path_count", 0) or 0) == 0:
        return "uncovered"
    if int(row.get("positive_path_count", 0) or 0) == 0:
        return "no_positive_flow"

    candidate_scarcity = (
        int(row.get("candidate_long_distance_od_pair_count", 0) or 0) == 0
        or bool(row.get("below_road_median_candidate_od", False))
    )
    major_city_scarcity = (
        int(row.get("candidate_major_city_od_pair_count", 0) or 0) == 0
        and float(row.get("target_daily_bev_traffic_2027", 0.0) or 0.0) > _LOW_VOLUME_MAJOR_CITY_THRESHOLD
    )
    parallel_competition = (
        float(row.get("parallel_competition_score", 0.0) or 0.0) >= _PARALLEL_COMPETITION_THRESHOLD
        and float(row.get("competing_positive_path_flow_sum", 0.0) or 0.0)
        > float(row.get("calibrated_daily_bev_flow", 0.0) or 0.0)
    )
    if (candidate_scarcity or major_city_scarcity) and parallel_competition:
        return "mixed"
    if candidate_scarcity:
        return "candidate_scarcity"
    if major_city_scarcity:
        return "major_city_scarcity"
    if parallel_competition:
        return "parallel_competition"
    return "well_covered"


def _segment_od_coverage_export(
    segment_diagnostics: pd.DataFrame,
    coverage_diagnostics: pd.DataFrame,
    competition_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    if segment_diagnostics.empty:
        return pd.DataFrame()

    df = segment_diagnostics.copy()
    coverage_extra = coverage_diagnostics.drop(columns=[
        col for col in ("candidate_path_count", "positive_path_count", "positive_path_flow_sum")
        if col in coverage_diagnostics.columns and col in df.columns
    ], errors="ignore")
    competition_extra = competition_diagnostics.drop(columns=["competing_road_flow_map"], errors="ignore")
    df = df.merge(coverage_extra, on="segment_id", how="left")
    df = df.merge(competition_extra, on="segment_id", how="left")

    road_col = "route_segment" if "route_segment" in df.columns else None
    if road_col is not None:
        medians = df.groupby(road_col, dropna=False).agg(
            road_median_candidate_od_pair_count=("candidate_od_pair_count", "median"),
            road_median_long_distance_od_pair_count=("candidate_long_distance_od_pair_count", "median"),
            road_median_positive_od_pair_count=("positive_od_pair_count", "median"),
        ).reset_index()
        df = df.merge(medians, on=road_col, how="left")
        df["below_road_median_candidate_od"] = (
            df["candidate_od_pair_count"] < df["road_median_candidate_od_pair_count"]
        )
        df["below_road_median_long_distance_od"] = (
            df["candidate_long_distance_od_pair_count"] < df["road_median_long_distance_od_pair_count"]
        )
        df["below_road_median_positive_od"] = (
            df["positive_od_pair_count"] < df["road_median_positive_od_pair_count"]
        )
    else:
        df["road_median_candidate_od_pair_count"] = np.nan
        df["road_median_long_distance_od_pair_count"] = np.nan
        df["road_median_positive_od_pair_count"] = np.nan
        df["below_road_median_candidate_od"] = False
        df["below_road_median_long_distance_od"] = False
        df["below_road_median_positive_od"] = False

    for col in (
        "candidate_od_pair_count",
        "candidate_major_city_od_pair_count",
        "candidate_long_distance_od_pair_count",
        "candidate_major_city_long_distance_od_pair_count",
        "positive_od_pair_count",
        "positive_major_city_od_pair_count",
        "positive_long_distance_od_pair_count",
        "positive_major_city_long_distance_od_pair_count",
        "major_city_positive_flow_sum",
        "long_distance_positive_flow_sum",
        "major_city_long_distance_positive_flow_sum",
        "major_city_flow_share",
        "long_distance_flow_share",
        "major_city_long_distance_flow_share",
        "short_distance_positive_path_count",
        "medium_distance_positive_path_count",
        "long_distance_positive_path_count",
        "competing_positive_path_count",
        "competing_positive_path_flow_sum",
        "competing_road_count",
        "top_competing_road_flow",
        "top_3_competing_roads_flow",
        "parallel_competition_score",
    ):
        if col not in df.columns:
            continue
        if pd.api.types.is_integer_dtype(df[col]) or col.endswith("_count"):
            df[col] = df[col].fillna(0).astype(int)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ("top_competing_road", "top_3_competing_roads", "competing_road_signature"):
        if col in df.columns:
            df[col] = df[col].fillna("")

    for col in (
        "below_road_median_candidate_od",
        "below_road_median_long_distance_od",
        "below_road_median_positive_od",
    ):
        df[col] = df[col].fillna(False).astype(bool)

    df["diagnostic_label"] = df.apply(_segment_diagnostic_label, axis=1)
    return df.sort_values(
        ["abs_error", "target_daily_bev_traffic_2027"],
        ascending=[False, False],
    )


def _top_segment_path_contributors(
    candidates: Sequence[PathCandidate],
    path_flows: pd.DataFrame,
    segment_ids: Sequence[int],
    top_n: int = 5,
) -> pd.DataFrame:
    flow_by_path = dict(zip(path_flows["path_id"], path_flows["calibrated_daily_bev_flow"]))
    rows: List[Dict] = []
    wanted = set(segment_ids)
    for candidate in candidates:
        flow = float(flow_by_path.get(candidate.path_id, 0.0) or 0.0)
        if flow <= _POSITIVE_FLOW_EPS:
            continue
        for sid in set(candidate.segment_ids) & wanted:
            rows.append({
                "segment_id": sid,
                "path_id": candidate.path_id,
                "od_pair_id": candidate.od_pair_id,
                "origin": candidate.origin,
                "destination": candidate.destination,
                "rank": candidate.rank,
                "road_signature": "|".join(candidate.road_signature),
                "distance_km": candidate.distance_km,
                "travel_time_min": candidate.travel_time_min,
                "calibrated_daily_bev_flow": flow,
                "node_path": "|".join(candidate.nodes),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        ["segment_id", "calibrated_daily_bev_flow"], ascending=[True, False]
    )
    df["contributor_rank"] = df.groupby("segment_id").cumcount() + 1
    return df[df["contributor_rank"] <= int(top_n)]


def _road_calibration_summary(segment_diagnostics: pd.DataFrame) -> pd.DataFrame:
    if segment_diagnostics.empty:
        return pd.DataFrame()
    road_col = "route_segment" if "route_segment" in segment_diagnostics.columns else None
    if road_col is None:
        return pd.DataFrame()
    grouped = segment_diagnostics.groupby(road_col, dropna=False).agg(
        segments=("segment_id", "count"),
        target_daily_bev=("target_daily_bev_traffic_2027", "sum"),
        calibrated_daily_bev=("calibrated_daily_bev_flow", "sum"),
        abs_error=("abs_error", "sum"),
        avg_abs_error_pct=("abs_error_pct", "mean"),
        candidate_path_count=("candidate_path_count", "sum"),
        positive_path_count=("positive_path_count", "sum"),
        underfit_segments=("is_underfit", "sum"),
        overfit_segments=("is_overfit", "sum"),
    ).reset_index()
    grouped["weighted_mape_pct"] = (
        grouped["abs_error"] / grouped["target_daily_bev"].replace(0, np.nan) * 100.0
    )
    grouped["signed_error"] = grouped["calibrated_daily_bev"] - grouped["target_daily_bev"]
    return grouped.sort_values("abs_error", ascending=False)


def _road_od_coverage_summary(
    segment_od_coverage_diagnostics: pd.DataFrame,
    competition_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    if segment_od_coverage_diagnostics.empty or "route_segment" not in segment_od_coverage_diagnostics.columns:
        return pd.DataFrame()

    grouped = segment_od_coverage_diagnostics.groupby("route_segment", dropna=False).agg(
        segments=("segment_id", "count"),
        target_daily_bev=("target_daily_bev_traffic_2027", "sum"),
        calibrated_daily_bev=("calibrated_daily_bev_flow", "sum"),
        abs_error=("abs_error", "sum"),
        median_candidate_od_pair_count=("candidate_od_pair_count", "median"),
        median_long_distance_od_pair_count=("candidate_long_distance_od_pair_count", "median"),
        median_major_city_od_pair_count=("candidate_major_city_od_pair_count", "median"),
        median_positive_od_pair_count=("positive_od_pair_count", "median"),
        share_segments_candidate_scarcity=("diagnostic_label", lambda s: float((s == "candidate_scarcity").mean())),
        share_segments_major_city_scarcity=("diagnostic_label", lambda s: float((s == "major_city_scarcity").mean())),
        share_segments_parallel_competition=("diagnostic_label", lambda s: float((s == "parallel_competition").mean())),
        share_segments_mixed=("diagnostic_label", lambda s: float((s == "mixed").mean())),
    ).reset_index()
    grouped["weighted_mape_pct"] = (
        grouped["abs_error"] / grouped["target_daily_bev"].replace(0, np.nan) * 100.0
    )

    competition_by_segment = competition_diagnostics.merge(
        segment_od_coverage_diagnostics[["segment_id", "route_segment"]],
        on="segment_id",
        how="left",
    )
    road_competing_flow: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in competition_by_segment.itertuples(index=False):
        road = str(getattr(row, "route_segment", "") or "")
        flow_map = getattr(row, "competing_road_flow_map", {}) or {}
        if not road or not isinstance(flow_map, dict):
            continue
        for competing_road, flow in flow_map.items():
            road_competing_flow[road][str(competing_road)] += float(flow)

    top_competing_road_by_flow: List[str] = []
    top_competing_road_flow: List[float] = []
    for road in grouped["route_segment"].astype(str):
        ordered = sorted(
            road_competing_flow.get(road, {}).items(),
            key=lambda item: (-item[1], item[0]),
        )
        if ordered:
            top_competing_road_by_flow.append(ordered[0][0])
            top_competing_road_flow.append(float(ordered[0][1]))
        else:
            top_competing_road_by_flow.append("")
            top_competing_road_flow.append(0.0)

    grouped["top_competing_road_by_flow"] = top_competing_road_by_flow
    grouped["top_competing_road_flow"] = top_competing_road_flow
    return grouped.sort_values("abs_error", ascending=False)


def _json_safe_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _top_json_records(
    df: pd.DataFrame,
    columns: Sequence[str],
    sort_by: Sequence[str],
    ascending: Sequence[bool],
    top_n: int = 5,
) -> List[dict]:
    if df.empty:
        return []
    ranked = df.sort_values(list(sort_by), ascending=list(ascending)).head(int(top_n))
    rows: List[dict] = []
    for _, row in ranked.iterrows():
        rows.append({col: _json_safe_value(row[col]) for col in columns if col in ranked.columns})
    return rows


def _od_coverage_summary_fields(
    segment_od_coverage_diagnostics: pd.DataFrame,
    road_od_coverage_summary: pd.DataFrame,
) -> dict:
    labels = segment_od_coverage_diagnostics.get("diagnostic_label", pd.Series(dtype=object))
    return {
        "n_segments_candidate_scarcity": int((labels == "candidate_scarcity").sum()),
        "n_segments_major_city_scarcity": int((labels == "major_city_scarcity").sum()),
        "n_segments_parallel_competition": int((labels == "parallel_competition").sum()),
        "n_segments_mixed": int((labels == "mixed").sum()),
        "roads_with_highest_competition": _top_json_records(
            road_od_coverage_summary,
            columns=[
                "route_segment",
                "share_segments_parallel_competition",
                "share_segments_mixed",
                "top_competing_road_by_flow",
                "top_competing_road_flow",
                "abs_error",
            ],
            sort_by=[
                "share_segments_parallel_competition",
                "share_segments_mixed",
                "top_competing_road_flow",
                "abs_error",
            ],
            ascending=[False, False, False, False],
        ),
        "roads_with_lowest_long_distance_coverage": _top_json_records(
            road_od_coverage_summary,
            columns=[
                "route_segment",
                "median_long_distance_od_pair_count",
                "median_candidate_od_pair_count",
                "abs_error",
            ],
            sort_by=[
                "median_long_distance_od_pair_count",
                "median_candidate_od_pair_count",
                "abs_error",
            ],
            ascending=[True, True, False],
        ),
    }


def _weighted_mape(report: pd.DataFrame) -> float:
    if report.empty:
        return 0.0
    denom = report["target_daily_bev_traffic_2027"].sum()
    if denom <= 0:
        return 0.0
    return float(report["abs_error"].sum() / denom * 100.0)


def _objective_weighted_mape(
    report: pd.DataFrame,
    objective: str,
    relative_error_floor: float,
) -> float:
    if report.empty:
        return 0.0
    target = report["target_daily_bev_traffic_2027"].to_numpy(dtype=float)
    error = report["abs_error"].to_numpy(dtype=float)
    if objective == "absolute":
        weights = np.ones_like(target)
    elif objective == "sqrt":
        weights = 1.0 / np.sqrt(np.maximum(target, 1.0))
    else:
        weights = 1.0 / np.maximum(target, float(relative_error_floor))
    denom = float(np.sum(target * weights))
    if denom <= 0:
        return 0.0
    return float(np.sum(error * weights) / denom * 100.0)
