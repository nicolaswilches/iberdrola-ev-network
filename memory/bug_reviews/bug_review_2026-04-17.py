"""Generates bug_review_2026-04-17.pdf for team triage."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)
from pathlib import Path

OUT = Path(__file__).parent / "bug_review_2026-04-17.pdf"

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=1.8*cm, bottomMargin=1.8*cm,
    title="ABM bug review, 2026-04-17",
)

styles = getSampleStyleSheet()
NAVY = HexColor("#003366")
RED = HexColor("#b03a2e")
ORANGE = HexColor("#c57a1e")
GREY = HexColor("#5d6d7e")

title_style = ParagraphStyle("title", parent=styles["Title"],
    fontName="Helvetica-Bold", fontSize=16, textColor=NAVY,
    spaceAfter=4, alignment=TA_LEFT)
subtitle_style = ParagraphStyle("subtitle", parent=styles["Normal"],
    fontSize=10, textColor=GREY, spaceAfter=16)
h1 = ParagraphStyle("h1", parent=styles["Heading1"],
    fontName="Helvetica-Bold", fontSize=13, textColor=NAVY,
    spaceBefore=14, spaceAfter=6)
h2_crit = ParagraphStyle("h2c", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=11.5, textColor=RED,
    spaceBefore=12, spaceAfter=4)
h2_maj = ParagraphStyle("h2m", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=11.5, textColor=ORANGE,
    spaceBefore=12, spaceAfter=4)
h2_min = ParagraphStyle("h2min", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=10.5, textColor=GREY,
    spaceBefore=10, spaceAfter=3)
body = ParagraphStyle("body", parent=styles["Normal"],
    fontSize=9.7, leading=13, spaceAfter=4)
note = ParagraphStyle("note", parent=body, fontSize=9, textColor=GREY)


def issue(story, tag, title_text, style, fields):
    """Render one issue block. fields: list of (label, text)."""
    story.append(Paragraph(f"{tag}. {title_text}", style))
    table_rows = []
    for label, text in fields:
        table_rows.append([
            Paragraph(f"<b>{label}</b>", body),
            Paragraph(text, body),
        ])
    t = Table(table_rows, colWidths=[3.1*cm, 13.1*cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*cm))


story = []

# ---- Title ----
story.append(Paragraph("ABM bug review, 17 April 2026 (v2)", title_style))
story.append(Paragraph(
    "Pre-existing issues found during a full-repo audit and independently verified against the code "
    "and data. Two regressions introduced earlier today (clustering and stale scenario YAML) were "
    "already fixed and pushed. This document lists the remaining issues for team triage.",
    subtitle_style,
))

# ---- Summary table ----
story.append(Paragraph("At a glance", h1))
summary_rows = [
    ["", "Severity", "Affects", "Decision needed"],
    ["C3", "Critical", "network topology", "Drop the AP to A fuzzy match?"],
    ["M1", "Major", "agent routing", "Accept approximation or improve?"],
    ["M2", "Major", "diagnostics only", "Quick win"],
    ["M5", "Major", "scenario runner", "Clean up demo vs YAML"],
    ["m2", "Minor", "optimization edge case", "Defensive code?"],
    ["m3 to m7", "Minor", "polish and docs", "Optional"],
]
st = Table(summary_rows, colWidths=[1.6*cm, 2.0*cm, 5.0*cm, 7.6*cm])
st.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f6f6f6"), white]),
    ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(st)

# ---- CRITICAL ----
story.append(Paragraph("Critical", h1))

issue(story, "C3", "AP-X chargers falsely appearing on A-X corridor (and vice versa)", h2_crit, [
    ("Where",
     "<font face='Courier' size='8.8'>src/new-abm/data_generation/spanish_network.py</font>, "
     "lines 528 to 532 (<font face='Courier'>_normalise_road</font>) and 110 to 143 (corridor list)."),
    ("What is happening",
     "<font face='Courier'>_normalise_road</font> strips the <font face='Courier'>P</font> from "
     "<font face='Courier'>AP-X</font>, so the fuzzy-match step in "
     "<font face='Courier'>_cluster_chargers_on_road</font> treats AP-2 and A-2 as the same road. "
     "The raw charger data tags them distinctly "
     "(AP-2: 10 chargers, A-2: 409). The fuzzy match causes BOTH corridors to receive ALL 419 chargers."),
    ("Why that is wrong",
     "AP-2 and A-2 are separate parallel highways. AP-2 is the autopista (originally a toll road, "
     "now free); A-2 is a different physical autovía. A driver on AP-2 cannot use A-2 service areas "
     "without exiting and re-entering. Same applies to AP-4 / A-4, AP-1 / A-1, AP-68 / A-68, "
     "AP-9 / A-9, AP-46 / A-46."),
    ("Measured",
     "185 physical charger locations end up represented as multiple station nodes. "
     "Connector capacity at those locations is effectively doubled: the physical chargers get "
     "two independent SimPy queues instead of one."),
    ("<b>Important</b>",
     "The <i>demand</i> side is fine. AP-2 and A-2 have distinct per-segment demand rows in the CSV "
     "(3 vs 20 segments), so the 12 MAD to BCN style 'duplicate' OD pairs are actually correct "
     "parallel-road demand, not duplication. The bug is only in charger assignment."),
    ("History",
     "Present since the first <font face='Courier'>build_spain_real_network</font> commit "
     "(15 April). Not introduced by today's work."),
    ("Options",
     "<b>A.</b> Remove the <font face='Courier'>AP- to A-</font> regex. Keep the NSEW suffix strip "
     "(that handles A-7 / A-7S, which really are the same road). One-line fix.<br/>"
     "<b>B.</b> Leave as is, accept the 2x station capacity on parallel corridors as a modelling "
     "shorthand.<br/>"
     "<b>C.</b> More invasive: ensure every AP / A pair has its own entry in "
     "<font face='Courier'>_ROAD_CORRIDORS</font> with the correct physical chargers. "
     "Today some AP- corridors do not appear at all (no AP-4 entry), so removing fuzzy-match would "
     "orphan their chargers; the corridor list would need to be audited."),
])

# ---- MAJOR ----
story.append(Paragraph("Major", h1))

issue(story, "M1", "Expected wait time over-estimates by 1.4 to 5x", h2_maj, [
    ("Where",
     "<font face='Courier' size='8.8'>src/new-abm/models/station.py</font>, lines 89 to 101 "
     "(<font face='Courier'>expected_wait_time_min</font>)."),
    ("Issue",
     "Formula is <font face='Courier'>(queue + current_users) / num_connectors * avg_session</font>. "
     "When queue is 0 but all c connectors are busy, it returns a full session (22 min). Correct "
     "M/D/c expectation for a new arrival in that state is the residual service time of the first "
     "connector to free, roughly avg_session / (c + 1)."),
    ("Measured bias",
     "queue=0, 4/4 connectors busy: formula 22 min, correct ~4.4 min (<b>5x high</b>).<br/>"
     "queue=4, 4/4 busy: formula 44 min, correct ~26 min (<b>1.67x high</b>).<br/>"
     "queue=8, 4/4 busy: formula 66 min, correct ~48 min (<b>1.36x high</b>).<br/>"
     "Over-estimation is worst when queue is small, which is exactly when the decision matters most."),
    ("Impact",
     "Agents over-estimate waits at busy-but-moving stations and skip to the next stop, which may "
     "itself be more congested. Plausibly contributes to the fat-tail pattern observed today "
     "(90% of charge events have 0 min wait but a tail exceeds hundreds of minutes)."),
    ("Options",
     "<b>A.</b> Change to <font face='Courier'>queue / c * avg_session + avg_session / (c + 1)</font>. "
     "One line.<br/>"
     "<b>B.</b> Track residual service time per connector and return the minimum. More accurate, "
     "more code.<br/>"
     "<b>C.</b> Keep the high bias deliberately as a 'driver pessimism' behavioural layer. "
     "Defensible but should be documented as such."),
])

issue(story, "M2", "Stranding reason never populated in output CSV", h2_maj, [
    ("Where",
     "<font face='Courier' size='8.8'>src/new-abm/simulation/engine.py</font>, "
     "<font face='Courier'>_record_strand()</font> at lines 378 to 399, and the results collector "
     "schema."),
    ("Issue",
     "<font face='Courier'>agent.failure_reason</font> is a dataclass field. It is set to "
     "<font face='Courier'>\"no_path_found\"</font> only in one place (the failed-route path at "
     "line 102). The SOC-depletion strand path "
     "(<font face='Courier'>_record_strand</font>) never writes it, and the output CSV "
     "(<font face='Courier'>baseline_trip_records.csv</font>) has no "
     "<font face='Courier'>failure_reason</font> column at all."),
    ("Impact",
     "Debugging stranded agents requires grepping warning logs. The ABM-scaling debug procedure in "
     "<font face='Courier'>memory/abm_scaling_task.md</font> explicitly asks for breakdown by "
     "<font face='Courier'>failure_reason</font>, which is not currently possible from the CSV alone."),
    ("Options",
     "<b>A.</b> Add <font face='Courier'>agent.failure_reason</font> to the strand paths with values "
     "like <font face='Courier'>soc_depleted</font>, <font face='Courier'>no_reachable_station</font>, "
     "<font face='Courier'>sim_window_timeout</font>. Also add the column to the collector schema. "
     "Self-contained, no downside."),
])

issue(story, "M5", "Two of three demo scenarios are silent no-ops on the real network", h2_maj, [
    ("Where",
     "<font face='Courier' size='8.8'>src/new-abm/scenarios/runner.py</font> "
     "(<font face='Courier'>make_demo_scenarios</font>, lines 178 to 204) and "
     "<font face='Courier' size='8.8'>src/new-abm/config/scenarios/*.yaml</font>."),
    ("Issue",
     "<font face='Courier'>price_reduction</font> and <font face='Courier'>capacity_increase</font> "
     "both reference synthetic-network station IDs like <font face='Courier'>STA_MAD_N</font>, "
     "<font face='Courier'>STA_ZAR_E</font>. These exist only in "
     "<font face='Courier'>build_spain_demo_network()</font>, not in the real pipeline, so the "
     "scenarios match nothing. "
     "<font face='Courier'>high_home_charging</font> is fine because it changes a fleet-wide "
     "parameter, not station IDs."),
    ("Supporting evidence",
     "In the 627k scenario comparison run earlier today, <font face='Courier'>price_reduction</font> "
     "and <font face='Courier'>capacity_increase</font> returned numbers identical to baseline, "
     "consistent with silent no-ops."),
    ("Related",
     "<font face='Courier'>config/scenarios/price_reduction.yaml</font> and "
     "<font face='Courier'>capacity_increase.yaml</font> also reference the same stale synthetic IDs "
     "and are never loaded by <font face='Courier'>run_scenarios.py</font> anyway (only "
     "<font face='Courier'>summer_peak</font> and <font face='Courier'>expand_hubs</font> are)."),
    ("Options",
     "<b>A.</b> Regenerate the two scenarios with real station IDs (like we did for "
     "<font face='Courier'>expand_hubs.yaml</font>). Load them from YAML, delete the hardcoded "
     "versions.<br/>"
     "<b>B.</b> Remove these scenarios entirely if nobody plans to use them.<br/>"
     "<b>C.</b> Keep them but explicitly scope to the synthetic demo mode only and document that."),
])

# ---- MINOR ----
story.append(Paragraph("Minor", h1))

issue(story, "m1", "Stale substation count (2,137 vs 2,147)", h2_min, [
    ("Where",
     "A few historical entries in <font face='Courier' size='8.8'>memory/decisions_log.md</font>."),
    ("Impact",
     "Docs only. Pipeline outputs correctly use 2,147 everywhere. Cosmetic."),
])

issue(story, "m2", "tent_tier fallback edge case", h2_min, [
    ("Where",
     "<font face='Courier' size='8.8'>src/optimization.py</font>, around line 91."),
    ("Issue",
     "If a row had <font face='Courier'>is_tent=True</font> together with "
     "<font face='Courier'>tent_tier in ('none', '')</font>, spacing would fall through to 120 km "
     "instead of 60 km (TEN-T Core)."),
    ("Current state",
     "Verified against <font face='Courier'>demand_per_segment.csv</font>: no such row exists today. "
     "Latent bug, not currently active."),
])

issue(story, "m3", "Misleading 'make this a generator' comments", h2_min, [
    ("Where",
     "<font face='Courier' size='8.8'>src/new-abm/simulation/engine.py</font>, lines 356 and 375 "
     "(<font face='Courier'>_handle_emergency_charge</font>)."),
    ("Issue",
     "Both places have <font face='Courier'>yield  # make this a generator</font> after an "
     "unreachable <font face='Courier'>return</font>. The function is already a generator thanks to "
     "real <font face='Courier'>yield</font>s earlier in the body, so these dead yields serve no "
     "purpose. Code works, but the comments mislead future editors."),
    ("Impact",
     "Cosmetic. Delete the dead <font face='Courier'>yield</font> lines and fix the comments when "
     "convenient."),
])

issue(story, "m4", "Stale 5.7% / 5.71% references in docs", h2_min, [
    ("Where",
     "<font face='Courier' size='8.8'>src/abm_demand.py:88</font> docstring; "
     "<font face='Courier' size='8.8'>references/assumptions.md</font> lines 337, 339, 429; "
     "<font face='Courier' size='8.8'>references/github_roadmap.md:107</font>."),
    ("Issue",
     "Docstring and docs cite EV penetration as 5.71% (based on the old 2 M EV base). Actual "
     "<font face='Courier'>EV_PENETRATION_RATE</font> is 2,498,159 / 35,000,000 = 7.14%. Code uses "
     "the correct value dynamically; only the narrative explanations are stale."),
    ("Impact",
     "Docs only. Can be updated any time."),
])

issue(story, "m5", "<font face='Courier'>apply_scenario</font> silently skips unknown station IDs", h2_min, [
    ("Where",
     "<font face='Courier' size='8.8'>src/new-abm/scenarios/base_scenario.py</font>, "
     "line 118 (<font face='Courier'>.get(sid, 0)</font> pattern)."),
    ("Impact",
     "A typo in a scenario YAML fails silently (no warning, no error). This is why M5 and today's "
     "stale <font face='Courier'>expand_hubs.yaml</font> IDs went unnoticed. Adding a warning when "
     "scenario IDs do not match any real station would surface future regressions immediately."),
])

issue(story, "m6", "No unit test for <font face='Courier'>no_dest_charger_arrival_soc_fraction</font>", h2_min, [
    ("Where",
     "Introduced today in <font face='Courier' size='8.8'>behavior/routing.py</font> and "
     "<font face='Courier' size='8.8'>behavior/en_route.py</font>; no matching test in "
     "<font face='Courier' size='8.8'>src/new-abm/tests/</font>."),
    ("Impact",
     "A future refactor could silently orphan the config key. Low-risk given we validated the "
     "behaviour manually today, but worth adding coverage."),
])

issue(story, "m7", "Silent 100 km fallback in <font face='Courier'>_project_onto_polyline</font>", h2_min, [
    ("Where",
     "<font face='Courier' size='8.8'>src/new-abm/data_generation/spanish_network.py</font> "
     "lines 813 and 838."),
    ("Impact",
     "If a city node is missing from the network, the code uses 100 km as the default segment "
     "distance without logging a warning. Never fires today (all referenced cities exist), but "
     "defensible to at least log a warning."),
])

# ---- Non-bugs ----
story.append(Paragraph("Verified NON-issues (do not spend time on these)", h1))
story.append(Paragraph(
    "<b>Bidirectional OD duplication (previously 'M4')</b>. Withdrawn. AP-2 and A-2 are genuinely "
    "separate parallel highways in Spain carrying distinct traffic. The demand CSV has different "
    "trip counts for each. Having separate OD pairs per corridor is the right model, not a bug.",
    note))
story.append(Paragraph(
    "<b>'return without yield at engine.py:355 breaks the generator'</b>. The function has real "
    "<font face='Courier'>yield</font> statements further down the body, so Python treats it as a "
    "generator regardless. Related code-tidiness concern is tracked as m3.", note))
story.append(Paragraph(
    "<b>'SOC calculation order risky at engine.py:285'</b>. Order "
    "<font face='Courier'>energy_added = target - current</font> then "
    "<font face='Courier'>current = target</font> is mathematically correct.", note))
story.append(Paragraph(
    "<b>'Bidirectional pairs double minority-direction demand'</b>. Demand CSV is per-segment "
    "aggregate (not directional), so splitting it evenly across both directions is the intended "
    "model.", note))

# ---- Recommendations ----
story.append(Paragraph("Suggested order of attack", h1))
story.append(Paragraph(
    "<b>1. M2. Stranding reason.</b> Self-contained, pure diagnostic, unlocks easier ABM debugging. "
    "No modelling implications.", body))
story.append(Paragraph(
    "<b>2. M5 plus m5. Scenario cleanup.</b> Regenerate "
    "<font face='Courier'>price_reduction.yaml</font> and "
    "<font face='Courier'>capacity_increase.yaml</font> against real station IDs, load them from "
    "YAML in <font face='Courier'>run_scenarios.py</font>, add a warning in "
    "<font face='Courier'>apply_scenario</font> when a station ID does not match. Removes a class "
    "of silent failures.", body))
story.append(Paragraph(
    "<b>3. M1. Wait-time formula.</b> Plausibly improves the fat-tail wait pattern. Needs a little "
    "discussion on which approximation to use.", body))
story.append(Paragraph(
    "<b>4. C3. AP / A fuzzy match.</b> Biggest call. Dropping the "
    "<font face='Courier'>AP- to A-</font> regex is mechanically one line, but materially changes "
    "charger distribution on parallel corridors (AP-2 goes from 419 chargers to 10, A-2 stays at "
    "409). All scenario results produced so far would need a re-run and re-interpretation. Worth a "
    "10-minute call before touching.", body))

story.append(Spacer(1, 0.6*cm))
story.append(Paragraph(
    "Already fixed and pushed earlier today: clustering regression (<i>16-floor added back</i>) and "
    "<font face='Courier'>expand_hubs.yaml</font> (<i>regenerated with current station IDs</i>).",
    note))

doc.build(story)
print(f"Saved to {OUT}")
