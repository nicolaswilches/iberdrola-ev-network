"""
Level 1 feedback loop: detailed ABM → NB07 charger sizing.

For each of the 8 proposed stations, read the ABM baseline observed peak
queue. Adjust ``n_chargers_proposed`` upward within the NB07 regulatory caps
(MAX_CHARGERS_HIGH_TRAFFIC = 12 for high-IMD TEN-T, MAX_CHARGERS_STANDARD = 8
otherwise). Iterate: re-run the ABM with the adjusted stations, observe the
new peak queues, re-adjust. Stop when no station is adjusted, or after N
iterations (cap reached).

Iteration outputs are written to src/simulation/feedback_loop/iter_<i>/ and a
summary log is appended to feedback_loop/log.csv. The original committed
proposed_stations.csv is restored at the end if --no-commit is passed.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROPOSED = REPO / "data/processed/proposed_stations.csv"
DEMAND = REPO / "data/processed/demand_per_segment.csv"
BACKUP = REPO / "data/processed/proposed_stations.backup.csv"
LOOP_DIR = REPO / "src/simulation/feedback_loop"
RUN_DEMO = REPO / "src/simulation/run_demo.py"
PY = Path(sys.executable)

# From src/constants.py
MAX_CHARGERS_HIGH_TRAFFIC = 12
MAX_CHARGERS_STANDARD = 8
HIGH_TRAFFIC_IMD_THRESHOLD = 20_000

# Feedback rule parameters
TARGET_PEAK_QUEUE = 20
QUEUE_DIVISOR = 30  # one extra connector per 30 peak-queue units above target


def max_chargers_for_station(row: pd.Series, demand: pd.DataFrame) -> int:
    """Return the regulatory cap on n_chargers for a given proposed station."""
    seg = demand.loc[demand["segment_id"] == row["source_segment_id"]]
    imd = float(seg["imd_total"].iloc[0]) if not seg.empty else 0.0
    return MAX_CHARGERS_HIGH_TRAFFIC if imd > HIGH_TRAFFIC_IMD_THRESHOLD else MAX_CHARGERS_STANDARD


def adjust_chargers(row: pd.Series, peak_queue: int, cap: int) -> int:
    """Map observed peak queue to a new n_chargers, respecting the regulatory cap."""
    current = int(row["n_chargers_proposed"])
    excess = max(0, peak_queue - TARGET_PEAK_QUEUE)
    extra = math.ceil(excess / QUEUE_DIVISOR) if excess > 0 else 0
    return min(cap, current + extra)


def run_abm(out_dir: Path, agents: int = 25_000) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PY), str(RUN_DEMO),
        "--agents", str(agents),
        "--output-dir", str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO), capture_output=True)
    return out_dir / "baseline_station_summary.csv"


def load_observations(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    prop = df[df["station_id"].str.startswith("STA_")].copy()
    prop = prop.rename(columns={"station_id": "location_id"})
    return prop[["location_id", "total_sessions", "peak_queue", "num_connectors"]]


def write_iteration_log(path: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--agents", type=int, default=25_000)
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Restore the original proposed_stations.csv at the end.",
    )
    args = parser.parse_args()

    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOOP_DIR / "log.csv"
    if log_path.exists():
        log_path.unlink()

    # Backup originals.
    shutil.copy(PROPOSED, BACKUP)
    original_proposed = pd.read_csv(PROPOSED)
    demand = pd.read_csv(DEMAND)

    # Compute per-station regulatory caps once.
    caps = {
        row["location_id"]: max_chargers_for_station(row, demand)
        for _, row in original_proposed.iterrows()
    }

    proposed = original_proposed.copy()
    any_adjustment_last_iter = True

    for it in range(args.iterations):
        iter_dir = LOOP_DIR / f"iter_{it:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot the proposed file for this iteration.
        proposed.to_csv(PROPOSED, index=False)
        proposed.to_csv(iter_dir / "proposed_stations.csv", index=False)

        # Run the ABM.
        print(f"[iter {it}] running ABM at {args.agents} agents...")
        summary_path = run_abm(iter_dir, agents=args.agents)
        obs = load_observations(summary_path)

        # Decide new charger counts.
        adjusted_rows = []
        any_adjustment = False
        for _, row in proposed.iterrows():
            lid = row["location_id"]
            match = obs.loc[obs["location_id"] == lid]
            peak = int(match["peak_queue"].iloc[0]) if not match.empty else 0
            sessions = int(match["total_sessions"].iloc[0]) if not match.empty else 0
            cap = caps[lid]
            old_n = int(row["n_chargers_proposed"])
            new_n = adjust_chargers(row, peak, cap)
            if new_n != old_n:
                any_adjustment = True
            adjusted_rows.append({
                "iteration": it,
                "location_id": lid,
                "route": row["route_segment"],
                "is_tent": bool(row["is_tent"]),
                "cap": cap,
                "n_before": old_n,
                "peak_queue": peak,
                "sessions": sessions,
                "n_after": new_n,
            })
            proposed.loc[proposed["location_id"] == lid, "n_chargers_proposed"] = new_n

        # Append to log.
        for r in adjusted_rows:
            write_iteration_log(log_path, r)

        # Print iteration summary.
        print(f"[iter {it}] station adjustments:")
        for r in adjusted_rows:
            marker = " *" if r["n_after"] != r["n_before"] else "  "
            print(
                f"{marker} {r['location_id']}  {r['route']:8s}  cap={r['cap']}  "
                f"peak={r['peak_queue']:4d}  sessions={r['sessions']:5d}  "
                f"n: {r['n_before']} -> {r['n_after']}"
            )

        if not any_adjustment:
            print(f"[iter {it}] converged: no further adjustments possible.")
            break

    # Final state: save adjusted file under feedback_loop/ for inspection.
    proposed.to_csv(LOOP_DIR / "proposed_stations_final.csv", index=False)

    # Restore or keep.
    if args.no_commit:
        shutil.copy(BACKUP, PROPOSED)
        print(f"\n[done] restored original proposed_stations.csv (backup at {BACKUP}).")
    else:
        proposed.to_csv(PROPOSED, index=False)
        print(f"\n[done] updated proposed_stations.csv kept. Backup at {BACKUP}.")

    print(f"[done] iteration outputs in {LOOP_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
