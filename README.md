# IE Sustainability Datathon March 2026 — Intelligent Electric Mobility (Iberdrola)

**Team:** Greenlabs

Optimal placement of EV charging stations along Spain's interurban road network for 2027, incorporating electrical grid capacity constraints from three distribution network operators: i-DE (Iberdrola), Endesa, and Viesgo.

---

## Challenge Overview

Design an optimal EV charging network along Spain's interurban roads (autopistas, autovías, carreteras nacionales) for a 2027 horizon. The solution must cross-reference mobility demand data with electrical grid capacity constraints.

**Deliverables:**
- `File_1.csv` — Global Network KPIs (single-row summary scorecard)
- `File_2.csv` — Proposed Charging Locations
- `File_3.csv` — Friction Points (Moderate/Congested only)
- `visualization/bi_map.html` — Self-contained interactive map
- `report/analytical_report.pdf` — 3-5 page executive summary
- `presentation/pitch.pdf` — Max 5 min pitch deck

---

## Repository Structure

```
iberdrola-ev-network/
├── notebooks/
│   ├── 00_environment_setup.ipynb       # Dependency setup (reference only)
│   ├── 01_data_ingestion_and_cleaning.ipynb
│   ├── 02_ev_projection_fork.ipynb      # SARIMA EV forecast (reference only)
│   ├── 03_road_network_analysis.ipynb
│   ├── 04_existing_chargers_baseline.ipynb
│   ├── 05_grid_capacity_consolidation.ipynb
│   ├── 06_demand_modeling.ipynb         # Authoritative ABM demand
│   ├── 06a_demand_deterministic.ipynb   # Deterministic benchmark
│   ├── 06b_abm_calibration.ipynb        # B1/SOC/seasonal sensitivity
│   ├── 06c_abm_demand_simulation.ipynb  # Monte Carlo cross-check
│   ├── 06d_demand_reconciliation.ipynb  # Three-way reconciliation
│   ├── 07_network_optimization.ipynb    # Greedy AFIR-compliant placement
│   ├── 07b_abm_validation.ipynb         # Utilization + AFIR re-check
│   ├── 08_grid_viability_friction.ipynb
│   ├── 09_output_generation.ipynb       # Generates File_1/2/3
│   └── 10_visualization_export.ipynb    # Generates bi_map.html
│
├── src/
│   ├── constants.py       # All thresholds, spacing rules, output schemas
│   ├── data_loading.py    # CSV/parquet loaders, locale helpers, UTM→WGS84
│   ├── geo_utils.py       # Substation matching, road snapping
│   ├── grid_analysis.py   # Grid consolidation + status classification
│   ├── abm_demand.py      # Behavioral demand model (B1=12%, SOC distribution)
│   ├── optimization.py    # AFIR gap detection + greedy station placement
│   └── new-abm/           # Agent-based simulation + Level-1 feedback loop
│       ├── feedback_loop.py   # ABM → NB07 charger-count tuning (3 iterations)
│       ├── feedback_loop/     # Iteration outputs (iter_00, iter_01, iter_02)
│       ├── run_demo.py
│       ├── run_scenarios.py
│       ├── behavior/
│       ├── models/
│       ├── simulation/
│       ├── scenarios/
│       └── config/
│
├── data/
│   ├── raw/               # Original downloads — immutable
│   └── processed/         # Cleaned pipeline artifacts
│
├── output/
│   ├── File_1.csv         # 8 stations, 3,246 baseline, 8 friction pts, 2,498,159 EVs
│   ├── File_2.csv         # 8 proposed stations, 28 chargers (ABM-tuned)
│   ├── File_3.csv         # 8 friction points, all Congested
│   └── dso_investment_summary.csv
│
├── visualization/
│   └── bi_map.html        # Self-contained Folium map
│
├── references/
│   ├── assumptions.md     # 25+ assumptions with citations
│   ├── glossary.md        # 62 domain-specific terms
│   ├── sources.md         # All data sources
│   └── data_gap_audit.md
│
├── memory/                # Project intelligence (agent protocol)
│   ├── project_state.md
│   ├── task_board.md
│   ├── decisions_log.md
│   ├── blockers.md
│   ├── lessons_learned.md
│   └── project_structure.md
│
├── report/
│   └── analytical_report_draft.docx
│
├── CLAUDE.md
├── requirements.txt
└── .gitignore
```

---

## Data Sources

See [references/sources.md](references/sources.md) for full citations.

**Mandatory:**
1. Road Routes — Ministry of Transport and Sustainable Mobility
2. EV Charging Points — National Access Point (NAP)
3. Route to Electrification — datos.gob.es GitHub Fork
4. i-DE Consumption Capacity Map — Iberdrola Group
5. e-distribución Historical Access Capacity — Endesa
6. Viesgo Interactive Grid Map — Viesgo Distribución
7. DGT Vehicle Registrations — Dirección General de Tráfico

**Additional:**
8. DGT IMD Traffic Counts
9. INE Population & Tourism Data
10. OpenStreetMap Rest Areas
11. EU TEN-T Core Network Corridors
12. AFIR Regulation Requirements

---

## Methodology

1. **Data Ingestion** (NB01): Clean road network, charger dataset, grid capacity, traffic counts
2. **Road Network** (NB03): Filter to AP-, A-, N- interurban roads; map geometry + TEN-T tiers
3. **Charger Baseline** (NB04): AFIR gap detection via linear referencing → 8 uncovered stretches
4. **Grid Consolidation** (NB05): Deduplicate 4,990 records → 2,147 physical substations
5. **Demand Modeling** (NB06): ABM with B1=12% charging probability → chargers per segment
6. **Network Optimization** (NB07): Greedy AFIR-compliant placement → 8 stations, 28 chargers
7. **ABM Feedback Loop** (`src/simulation/feedback_loop.py`): 3 iterations tune `n_chargers_proposed` from observed peak queues → STA_0003 AP-2 raised 4→6 connectors
8. **Grid Viability** (NB08): Station↔substation matching → grid_status + friction points
9. **Output** (NB09): Generate File_1.csv, File_2.csv, File_3.csv
10. **Visualization** (NB10): Folium interactive map → bi_map.html

---

## How to Run

Run pipeline notebooks in order: **NB01 → NB03 → NB04 → NB05 → NB06 → NB07 → NB08 → NB09 → NB10**.

NB00 (environment setup) and NB02 (SARIMA fork) are reference-only and not required for pipeline execution.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

## Current Results

| Metric | Value |
|--------|-------|
| Proposed stations | 8 |
| Proposed chargers | 28 (ABM-tuned) |
| AFIR gaps covered | 8 / 8 (0 remaining) |
| Friction points | 8 / 8 Congested |
| EV fleet 2027 | 2,498,159 |
| Baseline fast chargers (≥50 kW) | 3,246 |
| Grid substations | 2,147 physical |
| Congested substations | 81.9% |

---

## Key Assumptions

See [references/assumptions.md](references/assumptions.md) for full documentation.

- 150 kW per charger (mandated by datathon rules)
- AFIR spacing: 60 km TEN-T Core, 100 km TEN-T Comprehensive, 120 km general interurban
- Average EV range: 340 km WLTP (255 km effective highway range)
- Charging probability: 12% (B1, calibrated from IONITY empirical data)
- Grid thresholds: ≥5 MW = Sufficient, 1–5 MW = Moderate, <1 MW = Congested

---

*IE Sustainability Datathon March 2026 — Team Greenlabs*
