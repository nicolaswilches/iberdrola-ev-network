# ABM Scaling Debug Task

**Last updated:** 2026-04-15
**Status:** Pending — not yet started

## Goal

Fix the new ABM (`src/new-abm/`) so it runs correctly at 627,000 agents (matching the real daily BEV demand from `demand_per_segment.csv`).

**Success criteria:**
- Strand rate < 5% at 627,000 agents
- Peak queue length ≤ ~50 agents per station node at any 30-minute window (queue should scale with `num_connectors`, not blow up to thousands)
- Agents that DO strand should be ones for whom the trip is genuinely infeasible given their battery — not an artefact of missing stations or a too-short sim window

**Observed symptoms at 50,000 agents:**
- Excessive stranding (exact % TBD — needs measurement)
- Queue lengths up to ~2,000 per station node

---

## Likely root causes (in priority order)

### 1. Impossible OD pairs
Some corridor endpoint pairs span 1,000+ km (e.g. A Coruña → Murcia via the corridor graph). A 51 kWh MG4 Standard (255 km effective range) cannot make such a trip even with charging if intermediate station coverage is thin.

**Fix:** In `_build_od_matrix()` in `spanish_network.py`, filter out OD pairs where `network.shortest_path_length(origin, dest, weight='distance_km') > 700`. These ultra-long-haul trips are not realistic daily interurban BEV journeys and should be split or dropped.

### 2. Station coverage gaps on long corridor edges
The road name matching in `_cluster_chargers_on_road()` is exact-string (`nearest_road == road_name`). This means "AP-7" stations are not assigned to the "A-7" corridor, and "A-7S" stations are not assigned to "A-7". Many corridor edges > 150 km end up with no waypoint stations, making them impassable for small-battery vehicles.

**Fix:** Add normalised name matching — strip directional suffixes (N/S/E/W), try stripping the `P` from `AP-`, and try prefix matching. Example normalisation: `"A-7S" → "A-7"`, `"AP-7" → "A-7"`.

```python
def _normalise_road(name: str) -> str:
    name = name.strip().rstrip("NSEWnsew")
    name = re.sub(r'^AP-', 'A-', name)
    return name
```

Use normalised name for both the `nearest_road` column and the corridor key when matching.

### 3. Connector count inflation
`_cluster_chargers_on_road()` sums `n_connectors` across all stations in a cluster. On heavy corridors (AP-2, AP-7) this produces single nodes with 400+ connectors, which effectively makes those nodes have infinite capacity. The queue model never fires there, but stations elsewhere get overloaded.

**Fix:** Cap per-cluster connector count at 20 (reflecting a large HPC hub, not hundreds of stations collapsed into one point):
```python
total_connectors = min(int(grp["n_connectors"].fillna(2).astype(int).sum()), 20)
```

### 4. Too few station clusters per road — increase to 6–8
Currently `max_existing_clusters_per_road=4` is the default. At 627k agents, spreading load across only 4 waypoints per road causes hotspots.

**Fix:** Change the default from 4 to 6 in `build_spain_real_network()`, and pass 8 from `run_demo.py`:
```python
network, stations, od_matrix = build_spain_real_network(
    data_dir=data_dir, rng=rng,
    max_existing_clusters_per_road=8,
)
```

### 5. Sim window too short — 1440 min (24 h) cuts off late-evening trips
Agents departing at 20:00 on a Madrid→Barcelona trip (3.5 h drive + 30 min charge) complete at minute 1440 + ~10 min — just past the cutoff. They are marked stranded even though they would have made it.

**Fix:** In `config/base_config.yaml`:
```yaml
simulation:
  sim_duration_min: 2880   # 48 hours — lets late-evening trips complete
```

---

## Debugging procedure

Run at increasing agent counts. At each step inspect `outputs/debug_N/baseline_trip_records.csv`.

```bash
PYTHON=/opt/anaconda3/envs/iberdrola_abm/bin/python

for N in 500 5000 50000 200000 627000; do
    $PYTHON src/new-abm/run_demo.py \
        --agents $N \
        --no-plots \
        --output-dir src/new-abm/outputs/debug_$N
done
```

At each level check:
1. `status` column — what fraction is `stranded` vs `completed`?
2. `failure_reason` column — is it "no_route", "no_station_reachable", or "timeout"?
3. `vehicle_type` of stranded agents — are they disproportionately small-battery models?
4. Station summary: what is `peak_queue` on the busiest nodes?

When strand rate jumps between two consecutive levels, the root cause is the factor that scales with agent count between those levels (connector saturation if queue grows, OD infeasibility if stranding is flat across agent counts).

---

## Files to edit

| File | Change |
|---|---|
| `src/new-abm/data_generation/spanish_network.py` | Fuzzy road name matching; cap connectors at 20; filter infeasible OD pairs (>700 km); increase default `max_clusters` to 6 |
| `src/new-abm/config/base_config.yaml` | `sim_duration_min: 2880` |
| `src/new-abm/run_demo.py` | Pass `max_existing_clusters_per_road=8` |

---

## Conda environment

All ABM work runs in `iberdrola_abm`:
```bash
/opt/anaconda3/envs/iberdrola_abm/bin/python src/new-abm/run_demo.py [args]
```
