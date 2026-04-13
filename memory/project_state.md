# Project State

**Last updated:** 2026-04-13
**Status:** Full pipeline executed (NB01–NB10) with corrected gap logic, aligned demand sizing, safe grid consolidation, regenerated submission files, and exported visualization. Report and pitch deck remain pending.

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
- `visualization/bi_map.html` — self-contained Folium map with proposed stations, friction points, and baseline chargers

## Remaining deliverables

- `report/analytical_report.pdf`
- `presentation/pitch.pdf`
