#!/usr/bin/env python3
"""Assemble the Colab submission notebook from source notebooks and src/ modules.

Output: notebooks/datathon_submission.ipynb (a build artifact — do not hand-edit).

The builder concatenates cells from individual pipeline notebooks into a single
Colab deliverable, tags every cell with `origin` metadata for audit traceability,
and materializes src/ + src/new-abm/ via %%writefile cells so judges can read
every line of our code inline while still being able to re-run the pipeline.

Run:
    python scripts/build_colab_notebook.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"
OUT_PATH = NOTEBOOKS / "datathon_submission.ipynb"


def _head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


HEAD_SHA = _head_sha()


# -----------------------------------------------------------------------------
# Cell builders
# -----------------------------------------------------------------------------

def md(text: str, section: str | None = None, origin: str = "builder") -> nbformat.NotebookNode:
    cell = new_markdown_cell(text)
    cell.metadata["origin"] = {"source": origin, "section": section}
    return cell


def code(src: str, section: str | None = None, origin: str = "builder",
         cell_index: int | None = None) -> nbformat.NotebookNode:
    cell = new_code_cell(src)
    cell.metadata["origin"] = {"source": origin, "section": section, "cell_index": cell_index}
    return cell


def writefile(path: str, content: str, section: str) -> nbformat.NotebookNode:
    """Produce a %%writefile cell that materializes `path` with `content`."""
    body = f"%%writefile {path}\n{content}"
    return code(body, section=section, origin=path)


def inline_notebook(nb_path: Path, section: str, gate: str | None = None) -> list[nbformat.NotebookNode]:
    """Read a source notebook, merge its code cells into a single gated block.

    Markdown cells are kept separate (as documentation). Code cells are merged
    into one Python block wrapped in `if <gate>:` so the whole notebook can be
    skipped/re-enabled with a single flag. All cells carry `origin` metadata
    pointing back to the source notebook.
    """
    nb = nbformat.read(nb_path, as_version=4)
    rel = str(nb_path.relative_to(REPO))
    out: list[nbformat.NotebookNode] = []

    code_blocks: list[str] = []
    code_indices: list[int] = []

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            # Keep markdown cells inline as narrative
            out.append(md(cell.source, section=section, origin=rel))
        elif cell.cell_type == "code":
            # Collect code cells for later merging. Strip `%matplotlib` magics
            # (already handled in Section 0) so they never end up indented
            # inside an `if RUN_*:` gate.
            lines = [l for l in cell.source.splitlines()
                     if not l.strip().startswith("%matplotlib")]
            src = "\n".join(lines)
            if src.strip():
                code_blocks.append(f"# --- cell {idx} ---\n{src}")
                code_indices.append(idx)

    merged = "\n\n".join(code_blocks) if code_blocks else "pass"

    if gate:
        body = f"if {gate}:\n" + "\n".join("    " + line for line in merged.splitlines())
        body += f'\nelse:\n    print("Skipped {rel} (gate: {gate})")'
    else:
        body = merged

    out.append(code(body, section=section, origin=rel,
                    cell_index=f"merged[{','.join(str(i) for i in code_indices)}]"))
    return out


def walk_writefiles(base_dir: Path, section: str, include_exts: tuple = (".py", ".yaml")) -> list[nbformat.NotebookNode]:
    """Emit %%writefile cells for every source file under base_dir (recursive)."""
    cells: list[nbformat.NotebookNode] = []
    files = sorted(p for p in base_dir.rglob("*") if p.is_file() and p.suffix in include_exts)
    for p in files:
        rel = p.relative_to(REPO)
        content = p.read_text()
        cells.append(md(f"**`{rel}`**", section=section, origin=str(rel)))
        cells.append(writefile(str(rel), content, section=section))
    return cells


# -----------------------------------------------------------------------------
# Section 0 — Front matter + bootstrap
# -----------------------------------------------------------------------------

def section_0() -> list[nbformat.NotebookNode]:
    title = f"""# Iberdrola EV Charging Network — Team Greenlabs

**IE Sustainability Datathon, March 2026**

Optimal placement of EV charging stations along Spain's interurban road network for
2027, cross-referenced against electrical grid capacity from three DSOs: i-DE
(Iberdrola), Endesa, and Viesgo.

**Submission artifacts (shown in Section 1 below):**
- `output/File_1.csv` — Global network KPIs (single row)
- `output/File_2.csv` — 8 proposed stations, 28 chargers (ABM-tuned)
- `output/File_3.csv` — 8 friction points, all `Congested`
- `visualization/bi_map.html` — Interactive Folium map

**Build:** commit `{HEAD_SHA}` (branch `new-abm`). This notebook is a build artifact
generated by `scripts/build_colab_notebook.py`. The repo is the source of truth.

---

### How to read this notebook

1. **Section 1** displays the final committed outputs — judges land here.
2. **Section 2** exposes four flags that gate optional re-execution of the pipeline.
3. **Sections 3–7** expose every line of our code and every pipeline step for audit.

Every code cell carries an `origin` tag in its metadata pointing back to the source
file in the repo, so any result can be traced to its exact origin.
"""
    bootstrap = """# Colab detection + repo acquisition (public repo, no PAT required)
import os
import sys
from pathlib import Path

try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

if IS_COLAB:
    REPO = Path('/content/iberdrola-ev-network')
    if not REPO.exists():
        print("Cloning repo...")
        os.system(
            'git clone -b new-abm --depth 1 '
            'https://github.com/nicolaswilches/iberdrola-ev-network.git '
            '/content/iberdrola-ev-network 2>&1 | tail -3'
        )
    os.chdir(REPO)
else:
    REPO = Path.cwd()
    # Expect user to have opened this notebook from the repo root
    assert (REPO / 'src' / 'constants.py').exists(), (
        f"Not in repo root: {REPO}. Open this notebook from the iberdrola-ev-network directory."
    )

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src' / 'new-abm'))  # ABM package uses flat imports
print(f"Repo:     {REPO}")
print(f"Colab:    {IS_COLAB}")
print(f"Branch:   new-abm")
"""
    install = """# Install Python dependencies (Colab only — local envs should already have them)
if IS_COLAB:
    print("Installing requirements...")
    rc = os.system(f'pip install -q -r {REPO}/requirements.txt 2>&1 | tail -3')
    print(f"pip install exit code: {rc}")
else:
    print("Local env — skipping pip install. Ensure requirements.txt is satisfied.")
"""
    setup = """# Matplotlib inline display (Colab default). Individual notebooks can override.
%matplotlib inline
import matplotlib.pyplot as plt

# Display helpers
import pandas as pd
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

# Rerun flags — judges can flip these to regenerate outputs from raw
RUN_FULL_PIPELINE = False   # Re-run NB01→NB10 from raw data (~12–15 min)
RUN_NB02          = False   # Re-run SARIMA EV projection (needs pmdarima; output locked at 2,498,159)
RUN_ABM_FEEDBACK  = False   # Re-run the ABM→LP feedback loop (~20 min; iter_00/01/02 committed)
RUN_VALIDATION    = False   # Re-run the demand validation stack (06a–06d + 07b, ~10 min)

print(f"RUN_FULL_PIPELINE = {RUN_FULL_PIPELINE}")
print(f"RUN_NB02          = {RUN_NB02}")
print(f"RUN_ABM_FEEDBACK  = {RUN_ABM_FEEDBACK}")
print(f"RUN_VALIDATION    = {RUN_VALIDATION}")
"""

    return [
        md(title, section="0.1"),
        md("## 0.2 — Bootstrap", section="0.2"),
        code(bootstrap, section="0.2"),
        md("## 0.3 — Install dependencies", section="0.3"),
        code(install, section="0.3"),
        md("## 0.4 — Environment setup and rerun flags", section="0.4"),
        code(setup, section="0.4"),
    ]


# -----------------------------------------------------------------------------
# Section 1 — Results (outputs-first)
# -----------------------------------------------------------------------------

def section_1() -> list[nbformat.NotebookNode]:
    intro = """# Section 1 — Results

The four submission artifacts, loaded directly from the committed outputs in
`output/` and `visualization/`. These reflect the final ABM-tuned network:
**8 stations, 28 chargers, 8 friction points, 2,498,159 EVs projected for 2027.**
"""
    file1 = """# File_1.csv — Global Network KPIs
import pandas as pd
from IPython.display import display, Markdown

file1 = pd.read_csv('output/File_1.csv')
display(Markdown("### File_1.csv — Global KPIs"))
display(file1)
"""
    file2 = """# File_2.csv — Proposed Charging Stations
file2 = pd.read_csv('output/File_2.csv')
display(Markdown(f"### File_2.csv — {len(file2)} proposed stations, {file2['n_chargers_proposed'].sum()} total chargers"))
display(file2)
"""
    file3 = """# File_3.csv — Friction Points (Moderate + Congested)
file3 = pd.read_csv('output/File_3.csv')
display(Markdown(f"### File_3.csv — {len(file3)} friction points"))
display(file3)
"""
    dso = """# Per-DSO investment summary (supplementary)
from pathlib import Path as _Path
dso_path = _Path('output/dso_investment_summary.csv')
if dso_path.exists():
    dso = pd.read_csv(dso_path)
    display(Markdown("### DSO Investment Summary"))
    display(dso)
else:
    print("dso_investment_summary.csv not found — run Section 4 to regenerate")
"""
    map_cell = """# Interactive map — bi_map.html
from IPython.display import IFrame, display, Markdown
display(Markdown("### Interactive Map (`visualization/bi_map.html`)"))
display(IFrame(src='visualization/bi_map.html', width='100%', height=700))
"""
    kpis = """# Executive KPI summary
kpi_md = f'''
### Executive KPI summary

| Metric | Value |
|---|---|
| Proposed stations | {len(file2)} |
| Proposed chargers (ABM-tuned) | {file2['n_chargers_proposed'].sum()} |
| Total installed power | {file2['n_chargers_proposed'].sum() * 150 / 1000:.1f} MW |
| AFIR gaps covered | 8 / 8 (0 remaining) |
| Friction points (grid-constrained) | {len(file3)} / {len(file2)} |
| EV fleet projected 2027 | {file1.iloc[0]['total_ev_projected_2027']:,} |
| Baseline fast chargers (≥50 kW) | {file1.iloc[0]['total_existing_stations_baseline']:,} |

**ABM feedback loop:** 3 iterations brought STA_0003 (AP-2) from 4 → 6 connectors.
Evidence lives in `src/new-abm/feedback_loop/iter_00/01/02/` (shown in Section 5).
'''
display(Markdown(kpi_md))
"""
    checks = """# Compliance checklist — validates that the committed outputs satisfy the datathon brief
import pandas as pd

checks = []

# File_1
f1 = pd.read_csv('output/File_1.csv')
checks.append(('File_1 has exactly 1 row', len(f1) == 1))
expected_f1 = {'total_proposed_stations', 'total_existing_stations_baseline',
               'total_friction_points', 'total_ev_projected_2027'}
checks.append(('File_1 columns match schema', set(f1.columns) == expected_f1))
checks.append(('EV fleet = 2,498,159', int(f1.iloc[0]['total_ev_projected_2027']) == 2_498_159))

# File_2
f2 = pd.read_csv('output/File_2.csv')
expected_f2 = ['location_id', 'latitude', 'longitude', 'route_segment',
               'n_chargers_proposed', 'grid_status']
checks.append(('File_2 column order matches schema', list(f2.columns) == expected_f2))
checks.append(('File_2 station IDs unique', f2['location_id'].is_unique))
checks.append(('File_2 latitudes inside Spain', f2['latitude'].between(35, 44).all()))

# File_3
f3 = pd.read_csv('output/File_3.csv')
expected_f3 = ['bottleneck_id', 'latitude', 'longitude', 'route_segment',
               'distributor_network', 'estimated_demand_kw', 'grid_status']
checks.append(('File_3 column order matches schema', list(f3.columns) == expected_f3))
checks.append(('File_3 contains only Moderate + Congested',
               f3['grid_status'].isin(['Moderate', 'Congested']).all()))
checks.append(('File_3 DSOs are valid',
               f3['distributor_network'].isin(['i-DE', 'Endesa', 'Viesgo']).all()))

print(f"{'Check':<55} {'Pass?':<6}")
print('-' * 62)
for name, passed in checks:
    marker = 'OK' if passed else 'FAIL'
    print(f"{name:<55} {marker:<6}")

all_pass = all(p for _, p in checks)
print('-' * 62)
print(f"Total: {sum(p for _, p in checks)}/{len(checks)} {'PASS' if all_pass else 'FAIL'}")
"""
    return [
        md(intro, section="1.0"),
        md("## 1.1 — File_1.csv", section="1.1"),
        code(file1, section="1.1"),
        md("## 1.2 — File_2.csv", section="1.2"),
        code(file2, section="1.2"),
        md("## 1.3 — File_3.csv", section="1.3"),
        code(file3, section="1.3"),
        md("## 1.4 — DSO investment summary", section="1.4"),
        code(dso, section="1.4"),
        md("## 1.5 — Interactive map", section="1.5"),
        code(map_cell, section="1.5"),
        md("## 1.6 — KPI summary", section="1.6"),
        code(kpis, section="1.6"),
        md("## 1.7 — Compliance checklist", section="1.7"),
        code(checks, section="1.7"),
    ]


# -----------------------------------------------------------------------------
# Section 2 — One-click reproducibility
# -----------------------------------------------------------------------------

def section_2() -> list[nbformat.NotebookNode]:
    intro = """# Section 2 — One-click reproducibility

Section 1 displays the **committed** outputs from the `new-abm` branch. Sections
4–6 can regenerate them end-to-end from raw / processed data. Re-execution is
gated by four flags (set in Section 0.4):

| Flag | Default | What it does | Runtime |
|---|---|---|---|
| `RUN_FULL_PIPELINE` | `False` | Executes NB01→NB10 (core pipeline) | ~12–15 min |
| `RUN_NB02` | `False` | Re-runs the SARIMA EV projection (requires `pmdarima`) | ~5 min |
| `RUN_ABM_FEEDBACK` | `False` | Re-runs the ABM → LP feedback loop (3 iters) | ~20 min |
| `RUN_VALIDATION` | `False` | Re-runs the demand validation stack (06a–06d + 07b) | ~10 min |

**Default `Run All` behaviour:** Section 1 displays the final committed outputs
instantly; Sections 3 and 7 execute quickly (audit materialisation + reference
rendering). Sections 4/5/6 print skip notices. Flip a flag in Section 0.4 and
re-run to regenerate from raw.

**Why `RUN_ABM_FEEDBACK = False` by default:** the loop already ran to
convergence on the `new-abm` branch and the committed `File_2.csv` already
reflects the ABM-tuned 28 chargers (STA_0003 4→6). Re-running produces the same
result and adds ~20 min. Evidence is displayed from `src/new-abm/feedback_loop/`
in Section 5 without re-execution.

**Why `RUN_NB02 = False` by default:** the brief mandates
`total_ev_projected_2027 = 2,498,159`. This value is locked in
`src/constants.py` as `EV_FLEET_2027`. Re-running SARIMA with newer data
produces ~0.98% drift (2,522,552) which would break the submission's fixed
baseline.
"""
    return [md(intro, section="2.0")]


# -----------------------------------------------------------------------------
# Section 3 — src/ materialization (audit)
# -----------------------------------------------------------------------------

def section_3() -> list[nbformat.NotebookNode]:
    intro = """# Section 3 — `src/` audit materialization

Every module under `src/` is materialized below via `%%writefile` cells. This
serves two purposes: (1) judges can read every line of our code inline; (2) when
the notebook runs on a fresh Colab VM, the files get written to disk so that
`from src.<module> import ...` in Section 4 resolves exactly to what is shown
here. Content is identical to the repo at commit `{sha}`.
""".format(sha=HEAD_SHA)
    out: list[nbformat.NotebookNode] = [md(intro, section="3.0")]
    src_dir = REPO / "src"
    files = [
        "__init__.py",
        "constants.py",
        "data_loading.py",
        "geo_utils.py",
        "grid_analysis.py",
        "abm_demand.py",
        "optimization.py",
    ]
    for i, fname in enumerate(files, start=1):
        p = src_dir / fname
        if not p.exists():
            continue
        rel = f"src/{fname}"
        out.append(md(f"## 3.{i} — `{rel}`", section=f"3.{i}"))
        out.append(writefile(rel, p.read_text(), section=f"3.{i}"))
    return out


# -----------------------------------------------------------------------------
# Section 4 — Core pipeline (NB01–NB10)
# -----------------------------------------------------------------------------

def section_4() -> list[nbformat.NotebookNode]:
    intro = """# Section 4 — Core pipeline (NB01 → NB10)

Each subsection corresponds to one pipeline notebook from the repo. Code cells
are merged into a single Python block per notebook, wrapped in
`if RUN_FULL_PIPELINE:` so the whole section can be toggled from Section 0.4.
Markdown cells from the original notebooks are kept inline as narrative.

**Execution order: NB01 → NB03 → NB04 → NB05 → NB06 → NB07 → NB08 → NB09 → NB10.**
NB02 is reference-only (the SARIMA output is locked in `constants.py`).
"""
    out: list[nbformat.NotebookNode] = [md(intro, section="4.0")]

    # 4.1 — NB01 (gated on RUN_FULL_PIPELINE and presence of raw data)
    out.append(md(
        "## 4.1 — NB01 Data Ingestion & Cleaning\n\n"
        "**Skip guard:** NB01 reads large raw files under `data/raw/` (notably "
        "`dgt_registrations/`, ~600 MB). These are intentionally not shipped with "
        "the public repo — the downstream notebooks only need `data/processed/` "
        "which **is** committed. When raw data is absent NB01 prints a skip "
        "notice and NB03→NB10 still run to completion.",
        section="4.1",
    ))
    nb01 = NOTEBOOKS / "01_data_ingestion_and_cleaning.ipynb"
    out += inline_notebook(
        nb01, section="4.1",
        gate="RUN_FULL_PIPELINE and (Path('data/raw/dgt_registrations').exists() or Path('data/raw/rutas_por_carretera').exists())",
    )

    # 4.2 — NB02 (gated on RUN_FULL_PIPELINE AND RUN_NB02)
    out.append(md(
        "## 4.2 — NB02 EV Projection (SARIMA fork) — reference only\n\n"
        "The mandatory datathon value `total_ev_projected_2027 = 2,498,159` is "
        "captured as `EV_FLEET_2027` in `src/constants.py`. NB02 is the SARIMA "
        "model that produced this number. **Re-running it with newer data "
        "yields ~0.98% drift, which would break the fixed submission baseline**, "
        "so it is guarded behind `RUN_NB02 = False`. The cell below is "
        "auditable but does not execute by default.",
        section="4.2",
    ))
    nb02 = NOTEBOOKS / "02_ev_projection_fork.ipynb"
    out += inline_notebook(nb02, section="4.2", gate="RUN_FULL_PIPELINE and RUN_NB02")

    # 4.3–4.10 — NB03..NB10 (all gated on RUN_FULL_PIPELINE)
    pipeline = [
        ("4.3", "NB03 Road Network Analysis", "03_road_network_analysis.ipynb"),
        ("4.4", "NB04 Existing Chargers Baseline (AFIR gap detection)", "04_existing_chargers_baseline.ipynb"),
        ("4.5", "NB05 Grid Capacity Consolidation", "05_grid_capacity_consolidation.ipynb"),
        ("4.6", "NB06 Demand Modeling (authoritative)", "06_demand_modeling.ipynb"),
        ("4.7", "NB07 Network Optimization", "07_network_optimization.ipynb"),
        ("4.8", "NB08 Grid Viability & Friction", "08_grid_viability_friction.ipynb"),
        ("4.9", "NB09 Output Generation (File_1/2/3)", "09_output_generation.ipynb"),
        ("4.10", "NB10 Visualization Export (bi_map.html)", "10_visualization_export.ipynb"),
    ]
    for sec, title, fname in pipeline:
        out.append(md(f"## {sec} — {title}\n\n*Source:* `notebooks/{fname}`", section=sec))
        nb_path = NOTEBOOKS / fname
        out += inline_notebook(nb_path, section=sec, gate="RUN_FULL_PIPELINE")

    return out


# -----------------------------------------------------------------------------
# Section 5 — ABM feedback loop (new-abm)
# -----------------------------------------------------------------------------

def section_5() -> list[nbformat.NotebookNode]:
    intro = """# Section 5 — ABM → LP feedback loop (`src/new-abm/`)

The Level-1 feedback loop reads observed peak queues from an ABM baseline run,
adjusts each station's `n_chargers_proposed` within the NB07 regulatory caps
(`MAX_CHARGERS_HIGH_TRAFFIC=12`, `MAX_CHARGERS_STANDARD=8`), and re-runs the
ABM. The loop stops when no station is adjusted, or after 3 iterations.

**Rule:** add 1 connector per 30 peak-queue units above a target of 20.

**Result on the `new-abm` branch:** the loop converged in 2 effective iterations.
`STA_0003` (AP-2, high-IMD TEN-T Core) was raised from 4 → 6 connectors. All
other stations were already correctly sized by NB07. The final tuned
`proposed_stations.csv` drives `File_2.csv` in Section 1.

Every file under `src/new-abm/` is materialized below via `%%writefile` cells
(directory walk, including YAML configs). Re-execution of the loop is gated on
`RUN_ABM_FEEDBACK`.
"""
    out: list[nbformat.NotebookNode] = [md(intro, section="5.0")]

    # 5.1 — Walk src/new-abm/ tree and emit writefile cells
    abm_dir = REPO / "src" / "new-abm"
    out.append(md("## 5.1 — `src/new-abm/` package tree (audit materialization)", section="5.1"))
    out += walk_writefiles(abm_dir, section="5.1", include_exts=(".py", ".yaml"))

    # 5.2 — Display committed iter_*/summary evidence
    display_cell = """# Display the committed feedback loop evidence
import pandas as pd
from pathlib import Path as _Path
from IPython.display import display, Markdown

loop_dir = _Path('src/new-abm/feedback_loop')

log_csv = loop_dir / 'log.csv'
if log_csv.exists():
    display(Markdown("### Feedback loop log"))
    display(pd.read_csv(log_csv))
else:
    print(f"Missing {log_csv}")

final_csv = loop_dir / 'proposed_stations_final.csv'
if final_csv.exists():
    display(Markdown("### Final ABM-tuned stations (`proposed_stations_final.csv`)"))
    df = pd.read_csv(final_csv)
    display(df)
    total = df['n_chargers_proposed'].sum()
    display(Markdown(f"**Total chargers after ABM tuning: {total}**"))
else:
    print(f"Missing {final_csv}")

print("\\nPer-iteration output dirs:")
for d in sorted(loop_dir.glob('iter_*')):
    files = sorted(p.name for p in d.iterdir() if p.is_file())
    print(f"  {d.name}/  →  {len(files)} files")
"""
    out.append(md("## 5.2 — Committed feedback loop evidence", section="5.2"))
    out.append(code(display_cell, section="5.2"))

    # 5.3 — Optional re-execution, gated on RUN_ABM_FEEDBACK
    rerun_cell = """# Re-execute the feedback loop (disabled by default — results are already committed)
if RUN_ABM_FEEDBACK:
    import subprocess
    print("Running feedback loop (3 iterations, 25k agents)...")
    rc = subprocess.run(
        [sys.executable, 'src/new-abm/feedback_loop.py', '--agents', '25000', '--max-iters', '3'],
        cwd=str(REPO),
        check=False,
    )
    print(f"Exit code: {rc.returncode}")
else:
    print("ABM feedback loop skipped (RUN_ABM_FEEDBACK=False).")
    print("Committed results in src/new-abm/feedback_loop/ shown above.")
"""
    out.append(md("## 5.3 — Optional re-execution", section="5.3"))
    out.append(code(rerun_cell, section="5.3"))

    return out


# -----------------------------------------------------------------------------
# Section 6 — Validation stack
# -----------------------------------------------------------------------------

def section_6() -> list[nbformat.NotebookNode]:
    intro = """# Section 6 — Demand validation stack (NB06a–d, NB07b)

These notebooks are **not** in the primary `File_1/2/3` pipeline. They
independently cross-check the authoritative NB06 demand model (`B1 = 12%`
charging probability) via three methods — deterministic closed-form, parameter
sensitivity sweep, and Monte Carlo — and reconcile the three. NB07b then
validates placement against per-station utilization + AFIR spacing.

Gated on `RUN_VALIDATION = False`. Outputs (`demand_reconciliation_report.csv`,
`station_validation_metrics.csv`, figures) are committed in `data/processed/`
and `output/figures/` — set the flag to regenerate from scratch.
"""
    out: list[nbformat.NotebookNode] = [md(intro, section="6.0")]

    validation = [
        ("6.1", "NB06a Deterministic Demand Baseline", "06a_demand_deterministic.ipynb"),
        ("6.2", "NB06b ABM Calibration (B1 / SOC / seasonal sensitivity)", "06b_abm_calibration.ipynb"),
        ("6.3", "NB06c Monte Carlo Stochastic Demand", "06c_abm_demand_simulation.ipynb"),
        ("6.4", "NB06d Three-way Reconciliation", "06d_demand_reconciliation.ipynb"),
        ("6.5", "NB07b ABM Placement Validation", "07b_abm_validation.ipynb"),
    ]
    for sec, title, fname in validation:
        out.append(md(f"## {sec} — {title}\n\n*Source:* `notebooks/{fname}`", section=sec))
        nb_path = NOTEBOOKS / fname
        out += inline_notebook(nb_path, section=sec, gate="RUN_VALIDATION")

    return out


# -----------------------------------------------------------------------------
# Section 7 — References
# -----------------------------------------------------------------------------

def section_7() -> list[nbformat.NotebookNode]:
    intro = """# Section 7 — References

Project documentation rendered inline from the repo. These files define every
assumption, term, and data source used above.
"""
    refs = [
        ("7.1", "Assumptions", "references/assumptions.md"),
        ("7.2", "Glossary", "references/glossary.md"),
        ("7.3", "Data sources", "references/sources.md"),
        ("7.4", "Data gap audit", "references/data_gap_audit.md"),
        ("7.5", "Decisions log (engineering narrative)", "memory/decisions_log.md"),
    ]
    out: list[nbformat.NotebookNode] = [md(intro, section="7.0")]
    for sec, title, path in refs:
        out.append(md(f"## {sec} — {title}", section=sec))
        render_cell = f"""from pathlib import Path as _Path
from IPython.display import Markdown, display
_p = _Path('{path}')
if _p.exists():
    display(Markdown(_p.read_text()))
else:
    print(f"Missing {{_p}}")
"""
        out.append(code(render_cell, section=sec, origin=path))
    return out


# -----------------------------------------------------------------------------
# Assembly
# -----------------------------------------------------------------------------

def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = (
        section_0()
        + section_1()
        + section_2()
        + section_3()
        + section_4()
        + section_5()
        + section_6()
        + section_7()
    )
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": "3.10"},
        "build": {"commit": HEAD_SHA, "script": "scripts/build_colab_notebook.py"},
    })
    return nb


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build()
    nbformat.write(nb, OUT_PATH)
    n_cells = len(nb.cells)
    md_cells = sum(1 for c in nb.cells if c.cell_type == "markdown")
    code_cells = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"Wrote {OUT_PATH.relative_to(REPO)}")
    print(f"  commit: {HEAD_SHA}")
    print(f"  cells:  {n_cells} total ({md_cells} md, {code_cells} code)")


if __name__ == "__main__":
    main()
