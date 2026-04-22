"""
QA harness for visualization/abm_animation/trajectories.json.

Checks per-trip and per-corridor inter-waypoint geometry continuity. Exits 0
iff every trip has max inter-waypoint jump < TRIP_JUMP_KM_HARD and every
corridor display fragment has max internal jump < CORRIDOR_JUMP_KM_HARD.

Usage:
    .venv/bin/python visualization/abm_animation/qa_trajectories.py
    .venv/bin/python visualization/abm_animation/qa_trajectories.py --json path/to/trajectories.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# These pair with `_PATH_MAX_SEGMENT_KM` (3.0) and `_DISPLAY_MAX_SEGMENT_KM`
# (5.0) in export_trajectories.py — WARN below, HARD just above cap so a
# clean export always passes but a regression (e.g. a re-introduced >5 km
# routing jump) trips the guard.
TRIP_JUMP_KM_WARN = 2.0
TRIP_JUMP_KM_HARD = 4.0
CORRIDOR_JUMP_KM_WARN = 3.0
CORRIDOR_JUMP_KM_HARD = 6.0


def haversine_km(a, b):
    lon1, lat1 = a
    lon2, lat2 = b
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(h))


def max_jump(path):
    m = 0.0
    idx = -1
    for k in range(1, len(path)):
        if path[k - 1] == path[k]:
            continue
        j = haversine_km(path[k - 1], path[k])
        if j > m:
            m = j
            idx = k
    return m, idx


def bucket(jumps, thresholds=(5.0, 10.0, 30.0, 60.0)):
    buckets = [0] * (len(thresholds) + 1)
    for j in jumps:
        placed = False
        for i, t in enumerate(thresholds):
            if j < t:
                buckets[i] += 1
                placed = True
                break
        if not placed:
            buckets[-1] += 1
    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path,
                        default=Path(__file__).parent / "trajectories.json")
    parser.add_argument("--top", type=int, default=10,
                        help="Show top-N worst trips and corridors")
    args = parser.parse_args()

    if not args.json.exists():
        print(f"ERROR: {args.json} not found", file=sys.stderr)
        sys.exit(2)

    data = json.loads(args.json.read_text())
    trips = data.get("trips", [])
    corridors = data.get("corridors", [])

    # --- Trips --------------------------------------------------------------
    trip_jumps = []
    worst_trips = []
    for i, t in enumerate(trips):
        m, idx = max_jump(t["path"])
        trip_jumps.append(m)
        if m > TRIP_JUMP_KM_WARN:
            worst_trips.append((m, i, idx, t["path"][max(0, idx - 1)], t["path"][idx] if idx >= 0 else None))

    tb = bucket(trip_jumps)
    print(f"=== Trips ({len(trips)} total) ===")
    print(f"  max jump <{TRIP_JUMP_KM_HARD:>4} km : {tb[0]:>5}")
    print(f"  max jump  5–10 km : {tb[1]:>5}")
    print(f"  max jump 10–30 km : {tb[2]:>5}")
    print(f"  max jump 30–60 km : {tb[3]:>5}")
    print(f"  max jump  >60 km  : {tb[4]:>5}")
    if trip_jumps:
        print(f"  overall max       : {max(trip_jumps):>5.1f} km")

    if worst_trips:
        worst_trips.sort(reverse=True)
        print(f"\nTop {min(args.top, len(worst_trips))} worst trips (>{TRIP_JUMP_KM_WARN} km):")
        for m, i, idx, a, b in worst_trips[: args.top]:
            print(f"  trip#{i:<5} max={m:>6.1f} km at idx {idx}: {a} -> {b}")

    # --- Corridor fragments -------------------------------------------------
    corr_jumps = []
    worst_corrs = []
    for c in corridors:
        road = c.get("road", "?")
        m, idx = max_jump(c["path"])
        corr_jumps.append(m)
        if m > CORRIDOR_JUMP_KM_WARN:
            worst_corrs.append((m, road, idx, c["path"][max(0, idx - 1)], c["path"][idx] if idx >= 0 else None))

    cb = bucket(corr_jumps, thresholds=(2.0, 5.0, 10.0, 20.0))
    print(f"\n=== Corridor fragments ({len(corridors)} total) ===")
    print(f"  max internal jump <{CORRIDOR_JUMP_KM_HARD:>4} km : {cb[0]:>5}")
    print(f"  max jump 2–5 km  : {cb[1]:>5}")
    print(f"  max jump 5–10 km : {cb[2]:>5}")
    print(f"  max jump 10–20 km : {cb[3]:>5}")
    print(f"  max jump >20 km  : {cb[4]:>5}")
    if corr_jumps:
        print(f"  overall max       : {max(corr_jumps):>5.1f} km")

    if worst_corrs:
        worst_corrs.sort(reverse=True)
        print(f"\nTop {min(args.top, len(worst_corrs))} worst corridor fragments (>{CORRIDOR_JUMP_KM_WARN} km):")
        for m, road, idx, a, b in worst_corrs[: args.top]:
            print(f"  {road:<10} max={m:>6.1f} km at idx {idx}: {a} -> {b}")

    # --- Exit code ----------------------------------------------------------
    trip_max = max(trip_jumps) if trip_jumps else 0.0
    corr_max = max(corr_jumps) if corr_jumps else 0.0
    hard_fail = trip_max > TRIP_JUMP_KM_HARD or corr_max > CORRIDOR_JUMP_KM_HARD

    print(f"\n=== Verdict ===")
    print(f"  trip max {trip_max:.1f} km (hard limit {TRIP_JUMP_KM_HARD} km)")
    print(f"  corridor max {corr_max:.1f} km (hard limit {CORRIDOR_JUMP_KM_HARD} km)")
    if hard_fail:
        print("  FAIL — geometry has jumps above hard thresholds")
        sys.exit(1)
    print("  PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
