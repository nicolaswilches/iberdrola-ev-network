# Project State

**Last updated:** 2026-04-25
**Status:** Main notebook pipeline remains complete and unchanged. New ABM municipality-demand refactor is now in progress in `src/simulation/`: official municipality OD input is converted explicitly from people → private-car travelers → vehicle trips → 2027 BEV trips; the municipality-road graph is stitched from processed/Hermes geometry; municipality nodes are attached directly or via anchors; segment/road OD coverage diagnostics and competition diagnostics are exported; and top-1000 municipality OD pairs on the stitched graph now have 100% candidate-path reachability. Calibration quality is still poor because the current sample only covers ~15% of municipality OD demand and candidate diversity is still shallow, so the next work is demand scaling and richer candidate generation rather than more solver tuning.

## What is done

| Component | Status | Notes |
|---|---|---|
| `src/constants.py` | ✅ Done | Demand sizing now uses the same `2,498,159` EV baseline reported in `File_1.csv` |
| `src/geo_utils.py` | ✅ Done | `find_nearest_substation()` supports uncapped nearest lookup for remote-site DSO attribution |
| `src/grid_analysis.py` | ✅ Done | Added safe `consolidate_substations()` using distributor + name + coordinates |
| `src/optimization.py` | ✅ Done | Coverage gaps are now evaluated across every contiguous route component; service-area centroids projected correctly before BallTree siting |
| `04_existing_chargers_baseline.ipynb` | ✅ Executed | Corrected AFIR baseline: 8 uncovered stretches across 8 routes |
| `05_grid_capacity_consolidation.ipynb` | ✅ Executed | `grid_consolidated.csv` regenerated with 2,147 physical substations |
| `06_demand_modeling.ipynb` | ✅ Executed | `demand_per_segment.csv` regenerated (1,295 rows, total charger demand = 4,053) |
| `06a`–`06d` demand notebooks | ✅ Executed | Deterministic / calibration / stochastic / reconciliation notebooks rerun and aligned with the mandatory EV baseline |
| `07_network_optimization.ipynb` | ✅ Executed | `proposed_stations.csv` regenerated: 8 stations, 26 chargers, 0 remaining AFIR gaps |
| `07b_abm_validation.ipynb` | ✅ Executed | `station_validation_metrics.csv` regenerated: 0 overloaded stations, 1 high-queue-risk station, 0 post-placement gaps |
| `08_grid_viability_friction.ipynb` | ✅ Executed | `stations_with_grid_status.csv` and `friction_points.csv` regenerated: 8/8 stations are grid-constrained |
| `09_output_generation.ipynb` | ✅ Executed | `File_1.csv`, `File_2.csv`, `File_3.csv`, and `dso_investment_summary.csv` regenerated; all compliance checks passed with no manual DSO overrides |
| `10_visualization_export.ipynb` | ✅ Executed | `visualization/bi_map.html` regenerated successfully |

## Current pipeline metrics

- Interurban road segments: 1,295
- Existing interurban stations (all powers): 6,065
- Fast chargers ≥50 kW: 3,246
- Baseline AFIR gaps: 8 stretches across 8 routes
- Gap mix: 3 TEN-T Core, 2 TEN-T Comprehensive, 3 general interurban
- Longest baseline gap: `N-435`, 149.27 km
- Proposed stations: 8
- Total proposed chargers: 26
- Remaining AFIR gaps after placement: 0
- Grid substations after safe consolidation: 2,147
- Grid status split: 1,759 Congested, 85 Moderate, 303 Sufficient
- Friction points: 8 / 8 proposed stations
- Remote grid sites beyond 25 km: `N-502` (29.6 km, Endesa) and `AP-9` (98.4 km, Viesgo)

## Submission outputs

- `output/File_1.csv` — 1 row; `total_proposed_stations = 8`, `total_existing_stations_baseline = 6065`, `total_friction_points = 8`, `total_ev_projected_2027 = 2498159`
- `output/File_2.csv` — 8 proposed stations across `A-23`, `AP-2`, `AP-9`, `N-322`, `N-433`, `N-435`, `N-502`, `N-621`
- `output/File_3.csv` — 8 friction points; all `Congested`; valid DSOs only
- `output/dso_investment_summary.csv` — Endesa 2.4 MW, Viesgo 0.9 MW, i-DE 0.6 MW
- `visualization/bi_map.html` — self-contained deck.gl map (built by `scripts/build_bi_map.py`). Primary colour encoding is `grid_status` (Green=Sufficient / Amber=Moderate / Red=Congested) across all point layers. Additional layers: congestion heatmap underlay, TEN-T tier-styled corridors, station→substation connection lines (distance-graded), friction "!" badges on File_3 stations, KPI stat panel (proposed MW, DSO investment, AFIR gaps)

## Remaining deliverables

- `report/analytical_report.pdf`
- `presentation/pitch.pdf`

## New-ABM municipality calibration status

- Municipality demand nodes restricted to the `2,435` mainland municipalities that actually appear in the raw OD parquet.
- Raw municipality people flows are now converted with explicit national factors:
  - `car_mode_share = 0.849`
  - `occupancy = 1.74`
  - `EV_PENETRATION_RATE = EV_FLEET_2027 / TOTAL_VEHICLE_FLEET`
  - `BEV_FRACTION = 0.60`
- Full processed road network is stitched into a routable graph with:
  - road junction nodes
  - inter-road exchange nodes at geometric intersections
  - same-road gap bridges for adjacent processed segments
  - municipality anchors for non-endpoint municipalities
- Current stitched municipality graph diagnostics:
  - `2435` municipality nodes
  - `1295` processed road segments
  - `1379` road junction nodes
  - `1261` road exchange nodes
  - `1867` road anchor nodes
  - `5` weakly connected components
  - `0` unreachable municipality nodes
- Candidate generation on the stitched graph now gives `1000 / 1000` candidate-covered ODs for the current top-1000 demand slice.
- Current calibration diagnosis:
  - `728 / 1295` target segments covered
  - `70.5%` covered target share
  - most covered target flow is still labeled `candidate_scarcity` or `major_city_scarcity`
  - OD conservation sweep shows expected tradeoff, but errors remain high because the run only calibrates `1000` ODs (`~15.2%` of total municipality BEV OD demand) and `68.2%` of ODs still have only one candidate path
