# Project Structure

```
iberdrola-ev-network/
├── notebooks/                  # Main pipeline notebooks + auxiliary demand notebooks
│   ├── 01_data_ingestion_and_cleaning.ipynb   ✅ executed
│   ├── 03_road_network_analysis.ipynb         ✅ executed
│   ├── 04_existing_chargers_baseline.ipynb    ✅ executed — corrected baseline = 8 AFIR gaps
│   ├── 05_grid_capacity_consolidation.ipynb   ✅ executed — 2,147 safe physical substations
│   ├── 06_demand_modeling.ipynb               ✅ executed — authoritative ABM demand, 1,295 rows
│   ├── 06a_demand_deterministic.ipynb         ✅ executed — lower-bound deterministic benchmark
│   ├── 06b_abm_calibration.ipynb              ✅ executed — B1 / SOC / seasonal sensitivity
│   ├── 06c_abm_demand_simulation.ipynb        ✅ executed — stochastic cross-check
│   ├── 06d_demand_reconciliation.ipynb        ✅ executed — NB06 formally designated authoritative
│   ├── 07_network_optimization.ipynb          ✅ executed — 8 stations, 26 chargers, 0 AFIR gaps
│   ├── 07b_abm_validation.ipynb               ✅ executed — utilization + AFIR re-check
│   ├── 08_grid_viability_friction.ipynb       ✅ executed — 8 friction points
│   ├── 09_output_generation.ipynb             ✅ executed — final submission CSVs
│   ├── 10_visualization_export.ipynb          ✅ executed — `bi_map.html`
│   └── test.ipynb                             ⚠️ exploratory notebook, decision pending
│
├── src/                        # Shared Python modules
│   ├── constants.py            ✅ Submission baseline + model parameters
│   ├── abm_demand.py           ✅ Behavioral demand model
│   ├── optimization.py         ✅ Component-aware AFIR gaps + greedy placement
│   ├── geo_utils.py            ✅ Substation matching + road snapping
│   ├── grid_analysis.py        ✅ Grid consolidation + status logic
│   └── data_loading.py         ✅ CSV / parquet locale helpers
│
├── src/new-abm/                # New ABM + municipality calibration workstream
│   ├── calibration/
│   │   └── path_demand.py      ✅ Segment/road OD diagnostics, OD-conservation term, municipality candidate fixes
│   ├── data_generation/
│   │   ├── spanish_network.py  ✅ Hub OD prior + official municipality / ministry ingestion helpers
│   │   └── municipality_graph.py  ✅ Municipality-attached stitched road graph builder
│   ├── tools/
│   │   ├── calibrate_path_demand.py               ✅ Hub/corridor calibration CLI
│   │   ├── calibrate_municipality_path_demand.py  ✅ Municipality calibration CLI
│   │   ├── audit_municipality_road_graph.py       ✅ Municipality-road graph audit/export
│   │   └── report_calibration_hotspots.py         ✅ Hotspot diagnostic reporter
│   └── tests/
│       ├── test_path_demand.py
│       ├── test_spanish_network_od_prior.py
│       └── test_municipality_graph.py
│
├── data/
│   ├── raw/                    # Original downloads
│   └── processed/              # Regenerated pipeline artifacts
│       ├── demand_per_segment.csv
│       ├── demand_per_segment_deterministic.csv
│       ├── demand_per_segment_stochastic.csv
│       ├── demand_reconciliation_report.csv
│       ├── proposed_stations.csv
│       ├── station_validation_metrics.csv
│       ├── stations_with_grid_status.csv
│       ├── friction_points.csv
│       ├── grid_consolidated.csv
│       └── baseline_kpi.csv
│
├── output/                     # Submission deliverables
│   ├── File_1.csv              ✅ generated
│   ├── File_2.csv              ✅ generated
│   ├── File_3.csv              ✅ generated
│   └── dso_investment_summary.csv  ✅ generated
│
├── visualization/
│   └── bi_map.html             ✅ generated
│
├── references/
│   ├── assumptions.md          # Source-of-truth assumptions register
│   ├── glossary.md
│   ├── sources.md
│   └── data_gap_audit.md
│
├── memory/
│   ├── project_state.md
│   ├── task_board.md
│   ├── decisions_log.md
│   ├── blockers.md
│   ├── lessons_learned.md
│   ├── abm_scaling_task.md
│   └── project_structure.md
│
├── CLAUDE.md
└── requirements.txt
```

## Data Flow

```
NB01 / NB03 / NB04 / NB05
    ↓
NB06 (+ 06a–06d validation stack)
    ↓
NB07 / NB07b
    ↓
NB08
    ↓
NB09
    ↓
NB10
```
