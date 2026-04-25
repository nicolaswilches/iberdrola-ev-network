#!/usr/bin/env python3
"""Print hotspot diagnostics from calibration coverage exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _print_section(title: str, df: pd.DataFrame, columns: list[str], limit: int) -> None:
    print(f"\n## {title}")
    if df.empty:
        print("(none)")
        return
    view = df.loc[:, [col for col in columns if col in df.columns]].head(limit)
    print(view.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/tmp/abm_path_calibration")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    segment_path = out_dir / "abm_segment_od_coverage_diagnostics.csv"
    road_path = out_dir / "abm_road_od_coverage_summary.csv"

    if not segment_path.exists() or not road_path.exists():
        raise FileNotFoundError(
            "Missing calibration diagnostics. Expected "
            f"{segment_path} and {road_path}"
        )

    segment_df = pd.read_csv(segment_path, keep_default_na=False)
    road_df = pd.read_csv(road_path, keep_default_na=False)

    for label in (
        "candidate_scarcity",
        "major_city_scarcity",
        "parallel_competition",
        "mixed",
    ):
        _print_section(
            f"Top Segments: {label}",
            segment_df[segment_df["diagnostic_label"] == label].sort_values(
                ["abs_error", "target_daily_bev_traffic_2027"],
                ascending=[False, False],
            ),
            [
                "segment_id",
                "route_segment",
                "target_daily_bev_traffic_2027",
                "calibrated_daily_bev_flow",
                "abs_error",
                "candidate_od_pair_count",
                "candidate_long_distance_od_pair_count",
                "candidate_major_city_od_pair_count",
                "parallel_competition_score",
                "top_competing_road",
            ],
            args.top_n,
        )

    _print_section(
        "Top Roads: Competition",
        road_df.sort_values(
            ["share_segments_parallel_competition", "share_segments_mixed", "top_competing_road_flow", "abs_error"],
            ascending=[False, False, False, False],
        ),
        [
            "route_segment",
            "segments",
            "abs_error",
            "median_long_distance_od_pair_count",
            "share_segments_parallel_competition",
            "share_segments_mixed",
            "top_competing_road_by_flow",
            "top_competing_road_flow",
        ],
        args.top_n,
    )

    _print_section(
        "Top Roads: Weak Long-Distance Coverage",
        road_df.sort_values(
            ["median_long_distance_od_pair_count", "median_candidate_od_pair_count", "abs_error"],
            ascending=[True, True, False],
        ),
        [
            "route_segment",
            "segments",
            "abs_error",
            "median_candidate_od_pair_count",
            "median_long_distance_od_pair_count",
            "median_major_city_od_pair_count",
            "share_segments_candidate_scarcity",
            "share_segments_major_city_scarcity",
        ],
        args.top_n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
