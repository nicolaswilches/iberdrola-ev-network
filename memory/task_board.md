# Task Board

**Last updated:** 2026-04-22

## Done

- [x] Fix shared AFIR gap logic so every contiguous route component is checked
- [x] Align NB06 demand sizing to the mandatory `2,498,159` EV submission baseline
- [x] Replace unsafe substation-name-only consolidation with safe physical-substation grouping
- [x] Update NB07 to preserve placement traceability (`source_segment_id`, component, placement km, threshold)
- [x] Update NB07b to validate stations against their source demand segment instead of route-average demand when available
- [x] Update NB08 so remote sites inherit the nearest valid DSO label while remaining `Congested`
- [x] Remove NB09’s manual DSO override path and replace it with strict schema validation
- [x] Repair malformed notebook code strings introduced during JSON edits
- [x] Re-execute NB04–NB10 plus auxiliary NB06a–NB06d with the corrected logic
- [x] Regenerate `File_1.csv`, `File_2.csv`, `File_3.csv`, `dso_investment_summary.csv`, and `visualization/bi_map.html`
- [x] Re-audit the regenerated network: 8 stations, 26 chargers, 0 remaining AFIR gaps
- [x] BI map: swap primary palette to `grid_status` (green/amber/red), add congestion heatmap, TEN-T tier styling, station↔substation links, friction badges, KPI dashboard
- [x] ABM animation corridor geometry QA: eliminate off-corridor trip jumps (92.7 km → 3.0 km max); capped all 248 display fragments at 5 km; added continuity-checked stitcher, MAD-anchored multi-hop, and jump-guard reject log
- [x] v2 MIP model (Core 4): enriched candidate set (2,238 candidates), PuLP+CBC solver, AFIR+demand+grid+DSO-equity constraints, post-placement greedy AFIR closer → v2 network: 225 stations / 785 chargers / 117.8 MW; 0 AFIR gaps; unmet 427; DSO split i-DE 47% / Endesa 43% / Viesgo 10%

## Current Result

- [x] Baseline compliance gap audit: 8 true uncovered stretches
- [x] Final network optimization: 8 proposed stations, 26 chargers
- [x] Grid viability pass: 8 / 8 friction points, 2 remote grid sites
- [x] Submission validation: all CSV schema and cross-count checks pass

## Pending Deliverables

- [ ] Write `report/analytical_report.pdf` (wait until v2 MIP validated; v1 = Phase 1, v2 = full 2027 target narrative)
- [ ] Create `presentation/pitch.pdf`
- [ ] Decide whether to keep or remove `notebooks/test.ipynb`

## ABM Scaling (new-abm)

- [ ] Fix stranding and queue overflow in `src/new-abm/` at 627k agents — see `memory/abm_scaling_task.md` for full debugging plan and root-cause breakdown
