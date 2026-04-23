"""Path-flow calibration for the geometry-backed ABM graph.

This module estimates non-negative demand on candidate ABM paths so aggregate
path usage matches the BEV segment targets in ``demand_per_segment.csv``.
It keeps multiple geometry-backed alternatives per OD pair, so parallel roads
such as A-2 and AP-2 can receive separate calibrated flows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix

from models.demand import ODMatrix, ODPair
from models.network import RoadNetwork

logger = logging.getLogger(__name__)


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
    path_df = _path_candidates_df(candidates)

    if not candidates:
        raise RuntimeError("No path candidates generated; cannot calibrate ABM path demand")

    matrix, row_segment_ids, covered_targets = _build_incidence_matrix(candidates, targets)
    solve_matrix, solve_targets, row_weights = _weighted_calibration_system(
        matrix,
        covered_targets,
        objective=calibration_objective,
        relative_error_floor=relative_error_floor,
    )
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
    segment_diagnostics = _segment_calibration_diagnostics(
        candidates,
        flows,
        segment_report,
    )
    top_contributors = _top_segment_path_contributors(
        candidates,
        path_flows,
        row_segment_ids,
        top_n=top_contributors_per_segment,
    )
    road_summary = _road_calibration_summary(segment_diagnostics)

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
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path_df.to_csv(out_dir / "abm_path_candidates.csv", index=False)
        path_flows.to_csv(out_dir / "abm_calibrated_path_flows.csv", index=False)
        segment_report.to_csv(out_dir / "abm_segment_flow_calibration.csv", index=False)
        segment_diagnostics.to_csv(out_dir / "abm_segment_calibration_diagnostics.csv", index=False)
        top_contributors.to_csv(out_dir / "abm_segment_top_path_contributors.csv", index=False)
        road_summary.to_csv(out_dir / "abm_road_calibration_summary.csv", index=False)
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
        "segment_report": segment_report,
        "segment_diagnostics": segment_diagnostics,
        "top_contributors": top_contributors,
        "road_summary": road_summary,
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
            if not seg_ids:
                continue
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
                        if not seg_ids:
                            continue
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
                if not seg_ids:
                    continue
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


def _weighted_calibration_system(
    matrix,
    targets: np.ndarray,
    objective: str,
    relative_error_floor: float,
):
    if objective == "absolute":
        row_scale = np.ones_like(targets, dtype=float)
    elif objective == "sqrt":
        row_scale = np.sqrt(np.maximum(targets, 1.0))
    else:
        row_scale = np.maximum(targets, float(relative_error_floor))
    row_scale = np.maximum(row_scale, 1e-9)
    row_weights = 1.0 / row_scale
    return matrix.multiply(row_weights[:, None]), targets * row_weights, row_weights


def _path_candidates_df(candidates: Sequence[PathCandidate]) -> pd.DataFrame:
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
            if flow > 1e-6:
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
        if flow <= 1e-6:
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
