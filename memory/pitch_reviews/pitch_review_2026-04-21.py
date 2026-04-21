"""Generates pitch_review_2026-04-21.pdf for Theo covering the 5-minute
Iberdrola pitch deck. Reviewed against brief.pdf (Deliverable 5 + B5 rubric)
and the committed submission (File_1/2/3 + dso_investment_summary)."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)

OUT = Path(__file__).parent / "pitch_review_2026-04-21.pdf"

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=1.8*cm, bottomMargin=1.8*cm,
    title="Pitch deck review, 2026-04-21",
)
styles = getSampleStyleSheet()
NAVY = HexColor("#003366")
RED = HexColor("#b03a2e")
ORANGE = HexColor("#c57a1e")
GREEN = HexColor("#1e7e34")
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
h2_ok = ParagraphStyle("h2ok", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=11.5, textColor=GREEN,
    spaceBefore=12, spaceAfter=4)
body = ParagraphStyle("body", parent=styles["Normal"],
    fontSize=9.7, leading=13, spaceAfter=4)
note = ParagraphStyle("note", parent=body, fontSize=9, textColor=GREY)


def issue(story, tag, title_text, style, fields):
    story.append(Paragraph(f"{tag}. {title_text}", style))
    rows = []
    for label, text in fields:
        rows.append([Paragraph(f"<b>{label}</b>", body), Paragraph(text, body)])
    t = Table(rows, colWidths=[3.1*cm, 13.1*cm])
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
story.append(Paragraph("Pitch deck review, 21 April 2026", title_style))
story.append(Paragraph(
    "Review of Theo's 16-slide deck against the datathon brief (Deliverable 5 and "
    "rubric criteria B1, B3, B5, B6) and the committed submission files. Priority: "
    "remove the two foreign-competition slides, fix the charger count to 28, and "
    "complete the empty placeholders before the 5-minute pitch window.",
    subtitle_style,
))

# ---- Review-at-a-glance ----
story.append(Paragraph("Review at a glance", h1))

summary_rows = [
    ["", "Category", "Issue", "Severity"],
    ["1", "Factual accuracy", "Slide 4 says 26 chargers; committed File_2 shows 28", "Critical"],
    ["2", "Template placeholders", "Slides 11 and 13 are layout references from another competition (noted as draft-only)", "Major"],
    ["3", "Completeness", "Slides 7, 8, 10, 12, 14, 15, 16 are placeholders or draft ideas", "Major"],
    ["4", "Redundancy", "Two near-identical title slides (1 and 2)", "Minor"],
    ["5", "Storytelling", "Strong story spine from problem to solution to economics", "Strong"],
    ["6", "Rubric B5", "5-minute pacing achievable if placeholders are resolved", "Manageable"],
]
summary = Table(summary_rows, colWidths=[0.8*cm, 3.7*cm, 9.0*cm, 2.1*cm])
summary.setStyle(TableStyle([
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
story.append(summary)

# ---- CRITICAL ----
story.append(Paragraph("Critical (must fix before submission)", h1))

issue(story, "C1", "Slide 4 shows 26 chargers but the committed File_2 has 28",
      h2_crit, [
    ("What slide 4 says",
     "&quot;Our final plan adds 8 stations and 26 chargers and closes 100% of remaining gaps.&quot;"),
    ("What the committed submission says",
     "8 stations, <b>28 chargers</b>, 4.2 MW total peak demand. Change applied 2026-04-21 "
     "after the ABM feedback loop moved STA_0003 on AP-2 from 4 to 6 chargers."),
    ("Fix",
     "Update slide 4 to &quot;8 stations and 28 chargers&quot;. The total MW on the subsequent "
     "economics slide already works out to roughly 4.2 MW, so nothing else downstream "
     "needs to change. Make sure any visual map legend also reflects the 28-charger total."),
])

# ---- MAJOR ----
story.append(Paragraph("Major (complete before the dry run)", h1))

issue(story, "M0", "Slides 11 and 13: replace the template references with Iberdrola-specific content", h2_maj, [
    ("Context",
     "Martin confirmed these two slides are draft layout references from another "
     "competition (Accuracy Business Cup, SwapFleet robotaxi project) used for formatting "
     "inspiration. They are flagged here only because the template text must be replaced "
     "before the final version is exported."),
    ("Slide 11 (implementation roadmap layout)",
     "Great swimlane layout with years and tracks. Keep the structure. Replace the fleet-"
     "rollout content with Iberdrola station deployment: Q3 2027 for the 2 i-DE stations, "
     "Q4 2027 for the 5 Endesa stations, Q1 2028 for the 2 Viesgo stations. Tracks could "
     "be: Grid Access Requests / Construction / Energisation / Operations."),
    ("Slide 13 (risk-likelihood quadrant layout)",
     "Nice risk matrix layout. Keep the structure. Replace the battery-swap-network risks "
     "with Iberdrola-relevant risks: Endesa or Viesgo connection delays, MOVES subsidy "
     "timing, demand sensitivity to 2027 EV adoption curve, parallel-road demand "
     "distribution between AP and A segments, competitor first-mover risk (Repsol, "
     "IONITY)."),
    ("Why flag it now",
     "It is easy to accidentally export a &quot;draft&quot; version with template text still in "
     "place. A judge page-flipping the PDF will get confused by SwapFleet / Pony.ai "
     "references. Keep a visual reminder in the deck file (e.g. a bright watermark saying "
     "&quot;TEMPLATE -- to replace&quot;) until the content is in."),
])

issue(story, "M1", "Seven slides are placeholders", h2_maj, [
    ("Slides 7, 8",
     "&quot;Recommendation / What to do slide&quot; and &quot;Ideas: Add playing with the html map or "
     "vid/gif; Take cost analysis for each location for certain stations; Show value for "
     "Iberdrola; X many more travelers for these corridors.&quot; "
     "These are the pitch's strategic recommendation slides. They are the answer to "
     "the brief's rubric B3 (Strategic Relevance and Business Value). Currently empty."),
    ("Slide 10 (Implementation Roadmap)",
     "Blank. Needs replacing with the deployment timeline grouped by DSO zone (the report "
     "already has this; lift Table 9 from the analytical report into a visual timeline)."),
    ("Slide 12 (Risks and Mitigations)",
     "Blank. Real risks to add: Endesa / Viesgo connection delays (RDL 29/2021 sanctions "
     "mitigation), subsidy-timing risk for MOVES Corredores, traffic demand sensitivity to "
     "2027 EV adoption curve, parallel-road demand distribution between AP and A segments."),
    ("Slides 14, 15 (Cost and Revenue Analysis)",
     "Both marked &quot;just show excel&quot;. If the plan is to screen-share the Excel during "
     "the pitch, fine, but these slides still need one-liner captions so they make sense "
     "in a static PDF export. Alternatively build a simple summary table for each."),
    ("Slide 16 (Map of coverage per provider)",
     "Blank. If the plan is to show bi_map.html live, add a screenshot as a fallback. "
     "Judges see PDF only if the live demo fails."),
])

issue(story, "M2", "Two title slides are redundant",
      h2_maj, [
    ("What is there",
     "Slide 1 (wireframe with the Iberdrola wave), slide 2 (same title on a filled-green "
     "photo background). Both show the same title, subtitle, and team photos."),
    ("Fix",
     "Keep one. The photo-background version (slide 2) is more visually polished and "
     "would probably land better with judges. Delete slide 1."),
])

issue(story, "M3", "Recommendation slide (7) needs concrete actions, not an idea list",
      h2_maj, [
    ("What a strong recommendation slide looks like",
     "Three to four discrete actions with concrete numbers and owners. Candidates:"),
    ("Recommended structure",
     "1. Start building the 2 i-DE stations (N-322 and STA_0001 unchanged) in Q3 2026 "
     "so they go live Q3 2027.<br/>"
     "2. File Endesa grid-access requests for the 5 Endesa sites (AP-2, N-433, A-23, "
     "N-435, N-502) in Q2 2026 to meet Q4 2027 operational target.<br/>"
     "3. Pre-wire STA_0003 on AP-2 for 8 chargers (already sized at 6 per the ABM-calibrated "
     "feedback loop). It is the highest-demand Madrid-Barcelona gap and the only station "
     "the simulation flagged as approaching capacity.<br/>"
     "4. For Viesgo territory (STA_0006 on N-621 and STA_0008 on AP-9), start the grid "
     "feasibility study immediately given the 98.4 km connection distance at AP-9."),
])

# ---- STRONG POINTS ----
story.append(Paragraph("What is working well (preserve)", h1))

strong_points = [
    ("Slide 3 methodology narrative.",
     "The three-pillar framing (Infrastructure Optimization, ABM, Validation) with the "
     "&quot;tested against real usage&quot; tagline lands exactly the rubric's T3 integration "
     "point. Judges will know the team went beyond a static LP."),
    ("Slide 4 message.",
     "&quot;Spain can close its 2027 charging gaps with only 8 new stations&quot; is a clean "
     "headline that contradicts expectations. Strong rhetorical hook."),
    ("Slide 5 regulated-access framing.",
     "The observation that 7 of 8 sites sit in competitor territory, combined with the "
     "regulated-access process diagram, reframes what could be a weakness (not owning the "
     "grid) into a manageable coordination challenge. This is the pitch-polished version "
     "of the report's Section 6 strategic argument."),
    ("Slide 6 economics.",
     "€1.46M net margin, 1.9-year payback, 123k sessions/year, 3.7 GWh/year, zero "
     "overloaded stations. Concrete, quantified, and tied to the model rather than to "
     "external assumptions. The &quot;grid-access race, not a charger-siting race&quot; line "
     "is the pitch's single strongest soundbite."),
    ("Team credit slides.",
     "Clear team identification with photos. Shows professionalism and ownership without "
     "taking up prime real estate."),
]
for title, text in strong_points:
    story.append(Paragraph(f"<b>{title}</b> {text}", body))

# ---- Economics sanity check ----
story.append(Paragraph("Sanity check on the slide 6 economics", h1))
story.append(Paragraph(
    "Numbers reconcile against the model with rough agreement.",
    body,
))
econ_rows = [
    ["Claim on slide 6", "Reconciliation", "Status"],
    ["2,816 modeled BEVs/day at gap points", "Consistent with daily BEV flow aggregated over the 8 proposed station segments.", "ok"],
    ["338 charging sessions/day", "2,816 x 12% charging probability = 337.9 sessions. Matches.", "ok"],
    ["123k charging sessions/year", "338 x 365 = 123,370. Matches.", "ok"],
    ["3.70 GWh/year energy dispensed", "338 sessions x ~30 kWh per session x 365 days ~ 3.7 GWh. Matches.", "ok"],
    ["€1.85M gross revenue at €0.50/kWh", "3.7 GWh x €0.50 = €1.85M. Matches.", "ok"],
    ["€1.46M net margin after supply and €20k / station OpEx", "Order-of-magnitude consistent. Assumes ~€0.11/kWh supply cost and €20k x 8 = €160k OpEx. Make sure supply cost assumption is cited.", "cite"],
    ["€2.83M mid-case investment, 1.9-year payback", "€2.83M / €1.46M = 1.94 years. Math correct. Investment figure needs sourcing (BNEF, BP Pulse benchmarks, or Iberdrola public numbers).", "cite"],
]
econ = Table(econ_rows, colWidths=[5.0*cm, 9.0*cm, 2.1*cm])
econ.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.8),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f6f6f6"), white]),
    ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
    ("LEFTPADDING", (0, 0), (-1, -1), 3),
    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story.append(econ)
story.append(Paragraph(
    "Action for Theo: add a one-line footnote on slide 6 citing (a) retail price €0.50/kWh "
    "source, (b) wholesale supply cost assumption, (c) €20k/station OpEx source, "
    "(d) mid-case investment source. A judge who asks &quot;where does the €0.50/kWh come "
    "from?&quot; needs an on-slide answer.",
    body,
))

# ---- Rubric audit ----
story.append(Paragraph("Rubric audit (against brief section 6.3)", h1))

rubric_rows = [
    ["", "Criterion", "How the deck scores today", "Priority"],
    ["B1", "Integration technical <-> strategic", "Strong (slide 3 methodology, slide 6 economics tied to model).", "Safe"],
    ["B2", "Coherence of strategic roadmap", "Pending: slide 10 placeholder. Report already has the material.", "Major"],
    ["B3", "Strategic relevance and business value", "Strong where content exists (slide 5, 6). Slide 7 recommendation is empty.", "Major"],
    ["B5", "Clarity and persuasiveness of pitch", "Strong spine. Blocked by foreign-competition slides and empty placeholders.", "Critical"],
    ["B6", "Formal compliance and citation", "Missing sources for €0.50/kWh, OpEx, investment. Otherwise ok.", "Minor"],
]
rubric = Table(rubric_rows, colWidths=[0.9*cm, 4.3*cm, 8.3*cm, 1.8*cm])
rubric.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f6f6f6"), white]),
    ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
    ("LEFTPADDING", (0, 0), (-1, -1), 3),
    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story.append(rubric)

# ---- 5-minute pacing ----
story.append(Paragraph("Suggested 5-minute pacing", h1))

pacing = [
    ("0:00 - 0:30", "Slide 2 (title, photo background). Hook: &quot;Spain needs to comply with AFIR by 2027. We found the 8 stations that do it.&quot;"),
    ("0:30 - 1:00", "Slide 3 methodology. Two-pillar model: LP optimisation plus ABM simulation. &quot;Tested against real driver behaviour.&quot;"),
    ("1:00 - 1:45", "Slide 4 results. 8 stations, 28 chargers, 100% AFIR compliance. Map visual anchors the proposal in geography."),
    ("1:45 - 2:30", "Slide 5 grid challenge. 7 of 8 sites in competitor territory, but regulated access is a solved process. Iberdrola's dual role (CPO + i-DE) is a structural advantage, not a problem."),
    ("2:30 - 3:30", "Slide 6 economics. €1.46M net margin, 1.9-year payback, 123k sessions / year. Soundbite: &quot;This is a grid-access race, not a charger-siting race.&quot;"),
    ("3:30 - 4:15", "Slide 7 recommendation (to be built). Three concrete asks: file grid access Q2 2026, pre-wire STA_0003 on AP-2, launch Viesgo feasibility studies."),
    ("4:15 - 4:45", "Slide 9 thank you. Leave the map on screen for Q&amp;A."),
    ("4:45 - 5:00", "Buffer for transitions."),
]
for time_range, content in pacing:
    story.append(Paragraph(f"<b>{time_range}</b>. {content}", body))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    "Total core narrative: 7 slides (2, 3, 4, 5, 6, 7, 9). Appendix slides (10, 12, 14, 15, "
    "16) are reserved for Q&amp;A backup only and should not be shown during the 5-minute "
    "pitch. If a judge asks about deployment timeline or risks, you can jump to them "
    "during Q&amp;A.",
    note,
))

# ---- Order of attack ----
story.append(Paragraph("Suggested order for Theo", h1))

steps = [
    "<b>Remove slides 11 and 13</b> (foreign-competition content). This is the single highest-risk fix.",
    "Update slide 4 text from &quot;26 chargers&quot; to &quot;28 chargers&quot;. Minor edit.",
    "Merge the two title slides: keep slide 2 (photo), delete slide 1 (wireframe).",
    "Fill slide 7 with three concrete recommendations (use the template in M3).",
    "Replace slide 10 with a simple deployment-timeline visual grouped by DSO (i-DE Q3 27, Endesa Q4 27, Viesgo Q1 28).",
    "Replace slide 12 with an Iberdrola-relevant risks-and-mitigations table (grid-delay risk, subsidy-timing risk, demand sensitivity).",
    "Add one-line source citations to slide 6 economics numbers.",
    "Slides 14, 15, 16 can either stay appendix-only (screen-share Excel live) or get brief captions. Decide based on whether the live demo is reliable.",
]
for i, step in enumerate(steps, 1):
    story.append(Paragraph(f"<b>{i}.</b> {step}", body))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    "If Theo only has time for one change: remove slides 11 and 13. They are a submission "
    "risk on their own. Everything else can be polished in 30 minutes each; those two "
    "slides have to disappear before any version is shown to judges.",
    note,
))

doc.build(story)
print(f"Saved {OUT}")
