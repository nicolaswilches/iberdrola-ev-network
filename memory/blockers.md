# Blockers

## Active Blockers

_(none)_

## Recently Resolved

### [RESOLVED 2026-04-13] Shared AFIR gap logic was overstating and understating the network in different places

**Issue:** The notebook and shared optimization logic had drifted. One version still reported stale route counts, while the production gap function only kept the longest contiguous component of each road. That meant the repo could simultaneously overstate the baseline narrative and miss real uncovered stretches.

**Fix:** Rewrote shared gap detection to evaluate every contiguous route component, apply the correct threshold per component (`60/100/120 km`), and reuse that same logic in NB04, NB07, and NB07b.

**Result:** Corrected baseline = 8 uncovered stretches across 8 routes. Corrected final network = 8 proposed stations, 26 chargers, 0 remaining AFIR gaps.

---

### [RESOLVED 2026-04-13] Demand sizing was inconsistent with the mandatory submission EV fleet

**Issue:** NB06 had been sizing the network with a conservative `2,000,000` EV base while `File_1.csv` reported `2,498,159`.

**Fix:** Set `EV_FLEET_DEMAND_BASE = EV_FLEET_2027`, updated notebook text, and reran NB06 plus the auxiliary demand notebooks.

**Result:** `demand_per_segment.csv` is now aligned to the same fleet figure used in the final submission.

---

### [RESOLVED 2026-04-13] Grid consolidation was merging distinct substations with the same name

**Issue:** NB05 had been effectively treating `substation_name` as the physical asset key, which merged different substations that happened to share common names.

**Fix:** Added safe consolidation by distributor + name + rounded coordinates and regenerated `grid_consolidated.csv`.

**Result:** Physical substation count is 2,147, not 2,137. National grid status counts are now internally consistent.

---

### [RESOLVED 2026-04-13] Notebook reruns were blocked by environment-specific execution issues

**Issue:** Jupyter kernel execution segfaulted in this environment, and Matplotlib `plt.show()` blocked on the macOS GUI backend.

**Fix:** Re-executed notebook code paths headlessly and reran plotting notebooks with `MPLBACKEND=Agg`.

**Result:** NB04–NB10 and NB06a–NB06d all completed, and the regenerated CSV/HTML outputs are current.
