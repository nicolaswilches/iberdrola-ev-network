"""Generates report_review_2026-04-21.pdf for David to apply against the
analytical report draft. Reviewed against brief.pdf rubric and actual
committed submission files."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
)

OUT = Path(__file__).parent / "report_review_2026-04-21.pdf"

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=1.8*cm, bottomMargin=1.8*cm,
    title="Report review, 2026-04-21",
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
h2_ok = ParagraphStyle("h2ok", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=11.5, textColor=GREEN,
    spaceBefore=12, spaceAfter=4)
h2_min = ParagraphStyle("h2min", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=10.5, textColor=GREY,
    spaceBefore=10, spaceAfter=3)
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
story.append(Paragraph("Analytical report review, 21 April 2026", title_style))
story.append(Paragraph(
    "Review of David's draft (version sent 2026-04-21) against the datathon brief's "
    "grading rubric and the committed submission files. Priority-ordered list of fixes, "
    "with explicit reference to disqualification risks, tie-breaker items, and pitch-strength "
    "opportunities.",
    subtitle_style,
))

# ---- Executive summary of the review ----
story.append(Paragraph("Review at a glance", h1))

summary_rows = [
    ["", "Category", "Issue", "Severity"],
    ["1", "Factual accuracy", "Numbers in tables 5, 6, 7 do not match File_1/2/3", "Critical"],
    ["2", "Length compliance", "8 pages; brief mandates 3-5 pages", "Major"],
    ["3", "Citations", "Several BOE / MOVES / RDL references need verification", "Major"],
    ["4", "Disqualification checks", "grid_status thresholds justified (ok). 150 kW used (ok). File_3 only Moderate+Congested (ok)", "Safe"],
    ["5", "Rubric strengths", "Grid saturation narrative, DSO split, assumptions register, references", "Strong"],
    ["6", "Rubric gaps", "B3 Iberdrola business value could go further; B5 pitch not produced yet", "Medium"],
]
summary = Table(summary_rows, colWidths=[0.8*cm, 3.4*cm, 8.9*cm, 2.3*cm])
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
story.append(Paragraph("Critical (must fix)", h1))

issue(story, "C1", "Table 5, 6, 7 numbers do not match the committed submission",
      h2_crit, [
    ("What the draft says",
     "9 stations, 32 chargers, 5.1 MW peak demand. Table 6 lists STA_0003 on N-330, "
     "STA_0005 on N-2, and STA_0009 on AP-9. Table 7 shows i-DE 2 stations / 1.2 MW, "
     "Endesa 5 / 2.7 MW, Viesgo 2 / 1.2 MW."),
    ("What File_1 / File_2 / File_3 actually say",
     "<b>8 stations, 28 chargers, 4.2 MW peak demand.</b> Stations: STA_0001 N-322, "
     "STA_0002 N-433, STA_0003 AP-2 (6 chargers after ABM feedback loop), "
     "STA_0004 A-23, STA_0005 N-435, STA_0006 N-621, STA_0007 N-502, STA_0008 AP-9. "
     "DSO split: <b>i-DE 1 station 4 chargers 0.6 MW; Endesa 5 stations 18 chargers "
     "2.7 MW; Viesgo 2 stations 6 chargers 0.9 MW</b>. No N-330, no N-2, no STA_0009."),
    ("Why this is critical",
     "The brief's T4 criterion requires strict consistency between the Analytical Report "
     "and the output datasets. A judge who opens File_2 and compares to Table 6 will "
     "see an immediate mismatch. Under rubric section 6.4, &quot;Output datasets not "
     "following the required structure or field definitions&quot; is listed as a "
     "disqualification ground; consistency between outputs and report falls in the "
     "same spirit."),
    ("Fix",
     "Regenerate tables 5, 6, 7 directly from <font face='Courier'>output/File_1.csv</font>, "
     "<font face='Courier'>output/File_2.csv</font>, <font face='Courier'>"
     "output/dso_investment_summary.csv</font>. "
     "Specifically remove the fake N-330 / N-2 / STA_0009 rows. "
     "Update all downstream references in the executive summary, Section 3.4 (&quot;9 new "
     "stations, 32 chargers&quot;), and Section 6 (&quot;nine stations&quot;, &quot;N-322, N-330&quot; "
     "pairing as i-DE stations). The ONE i-DE station is STA_0001 on N-322."),
])

issue(story, "C2", "Executive summary and Section 6 claim EUR 1.9 to 4.4 million net "
      "investment, not derived from any model output", h2_crit, [
    ("What the draft says",
     "Total net investment EUR 1.9 to 4.4 million under Scenario A (MOVES 70% subsidy "
     "plus RDL 7/2026 depreciation). Tables 8 and 9 break this down by cost component "
     "and DSO zone."),
    ("What the model gives us",
     "The optimisation pipeline produces station coordinates and charger counts. It does "
     "NOT produce euro-denominated CAPEX figures. The cost numbers appear to be David's "
     "own synthesis from external research."),
    ("Why this matters",
     "For the pitch narrative, external cost research is legitimate and encouraged under "
     "T2. However, the specific numbers need sources. The report currently cites RDL 7/2026 "
     "and MOVES Corredores de Recarga without letting the reader verify the quoted CAPEX "
     "bands. A judge who asks &quot;where does the EUR 1.35 to 3.6 million grid reinforcement "
     "figure come from?&quot; must be able to find it in the references."),
    ("Fix",
     "For every euro figure in Section 6, cite a source in the references. Typical sources: "
     "BloombergNEF 2024 Infrastructure Cost Report (charger hardware EUR 80-130k each), "
     "Iberdrola / BP Pulse press releases on AP-68 / Madrid station builds, IDAE MOVES "
     "CTE Corredores CAPEX tables. If a number cannot be traced, soften the claim to "
     "&quot;order-of-magnitude EUR X million per station&quot; with a citation. "
     "Do not invent BOE / Orden TED numbers that were not verified."),
])

issue(story, "C3", "Specific Spanish regulation references need verification",
      h2_crit, [
    ("What the draft cites",
     "RDL 7/2026 (BOE-A-2026-6544, &quot;in force 20 March 2026&quot;). "
     "Orden TED/1477/2025 MOVES Corredores. "
     "BOE-A-2025-25989 (bases MOVES). "
     "RD 1183/2020 (grid access). "
     "RDL 29/2021 DSO sanctions (BOE-A-2021-21096). "
     "November 2025 Royal Decree (Enlit World)."),
    ("Risk",
     "If any of these BOE numbers or BOE dates were fabricated (even partially), a judge "
     "checking them in the BOE database will find nothing. The credibility of the whole "
     "report collapses at that point."),
    ("What to do",
     "David: open the BOE search and verify each citation. For items you cannot verify, "
     "remove them and replace with generic phrasing (e.g., &quot;upcoming MOVES Corredores "
     "subsidy call&quot; instead of &quot;Orden TED/1477/2025&quot;). Keep only citations you have "
     "personally located. This is cheaper than losing credibility on a factual check."),
])

# ---- MAJOR ----
story.append(Paragraph("Major (fix before submission)", h1))

issue(story, "M1", "Length exceeds the 3 to 5 page brief limit", h2_maj, [
    ("Current",
     "8 pages including cover. The brief explicitly states &quot;A clear, professional "
     "report (3-5 pages) detailing the analysis, the methodology used, and the "
     "strategic proposals.&quot; The cover page is typically not counted. Even excluding the "
     "cover, the body is at 7 pages."),
    ("What can be cut fastest (largest savings first)",
     "(a) Section 2 Data Sources to a single paragraph with a 14-item list or compact "
     "table. Saves 0.5 to 1 page. "
     "(b) Section 3.2 Demand Modelling background and the &quot;why 12% matters&quot; box to "
     "one paragraph. Technical detail belongs in the Colab. Saves 0.5 page. "
     "(c) Section 6.4 Smart Charging Layer and 6.5 Competitive Window merged into half a "
     "page. Saves 0.5 page. "
     "(d) Table 9 Deployment Feasibility: keep the table, drop the prose paragraph above "
     "that duplicates its content. Saves 0.25 page."),
    ("What MUST stay in the body (brief-mandated)",
     "Grid thresholds with justification (Table 4). All assumptions (Table 3). "
     "Data source citations (all refs). Strategic roadmap. Limitations. File-2-aligned "
     "station table."),
    ("Note on appendices",
     "The brief does not mention or allow an appendix. Theo suggested moving tables to an "
     "appendix; possible but risky, because if judges only read the 3-5 pages, appendix "
     "content is wasted. Safer to compress rather than appendix-dump."),
])

issue(story, "M2", "Section 3.1 says &quot;16 analytical notebooks&quot;; the repo has 14",
      h2_maj, [
    ("Correction",
     "<font face='Courier'>notebooks/</font> contains: 01, 03, 04, 05, 06, 06a, 06b, 06c, "
     "06d, 07, 07b, 08, 09, 10 = 14 notebooks. "
     "If you count the NB02 SARIMA fork from the mandatory GitHub repository as part of "
     "the pipeline, that makes 15. There is no notebook 15 or 16."),
    ("Fix", "Replace with &quot;14 analytical notebooks plus the mandatory SARIMA fork&quot; or "
     "simply &quot;a pipeline of 14 notebooks&quot;."),
])

issue(story, "M3", "&quot;i-Charging&quot; is not the standard name for Iberdrola's EV charging brand",
      h2_maj, [
    ("Draft phrase",
     "&quot;Iberdrola, as both a charging operator through i-Charging and a distribution "
     "operator through i-DE&quot;"),
    ("Issue",
     "Iberdrola's public EV charging arm in Spain is typically branded &quot;Iberdrola "
     "Movilidad Verde&quot; or simply Iberdrola. &quot;i-Charging&quot; is an unrelated Portuguese "
     "EV-charger manufacturer (not Iberdrola). A judge from the industry will notice. "
     "i-DE is correct for the distribution arm."),
    ("Fix",
     "Replace &quot;i-Charging&quot; with &quot;Iberdrola Movilidad Verde&quot; or just &quot;Iberdrola's "
     "charging operator business&quot;. Keep i-DE for the DSO reference."),
])

issue(story, "M4", "EV penetration 7.14% in report vs 5.7% in references/assumptions.md",
      h2_maj, [
    ("Issue",
     "The report correctly states 7.14% (2,498,159 / 35 million). The team assumption "
     "register at <font face='Courier'>references/assumptions.md</font> still says 5.7% "
     "based on an older 2.0 M EV base. The values are inconsistent between the report "
     "and the assumption register the judges may cross-check."),
    ("Fix",
     "Either the team updates assumptions.md to 7.14% (recommended, matches code and "
     "submitted File_1 value of 2,498,159 EVs), or the report is softened to state "
     "&quot;approximately 7%&quot;. The submitted File_1 number must be consistent with whatever "
     "penetration rate the report cites."),
])

issue(story, "M5", "Seasonal multiplier values inconsistent between Table 3 and Section 3.2",
      h2_maj, [
    ("Issue",
     "Section 3.2 says &quot;2.5x peak in July and August&quot; for Mediterranean and &quot;1.5x&quot; for "
     "Atlantic. Table 3 says &quot;2.5x (July to August)&quot; Mediterranean; Atlantic 1.5x is "
     "mentioned in Section 2 but not in Table 3 assumption register."),
    ("Fix",
     "Add an &quot;Atlantic seasonal multiplier: 1.5x&quot; row to Table 3 with INE source. "
     "Consistency between methodology prose and assumption register matters for B6."),
])

issue(story, "M6", "Section 3.2 says demand formula clamped to &quot;4 chargers on TEN-T Core, 2 on "
      "general&quot; but does not mention the TEN-T Comprehensive tier", h2_maj, [
    ("Issue",
     "The code's tiered AFIR thresholds are Core (60 km, min 4 chargers), Comprehensive "
     "(100 km), and general interurban (120 km, min 2 chargers). Section 3.2 collapses "
     "this into a two-tier picture which contradicts Table 2."),
    ("Fix",
     "In Section 3.2: &quot;clamped to AFIR-mandated minimums (4 chargers on TEN-T Core, "
     "2 chargers on TEN-T Comprehensive and general interurban roads per AFIR Annex II).&quot;"),
])

# ---- STRONG POINTS (acknowledge) ----
story.append(Paragraph("What is strong in the draft (preserve and build on)", h1))

strong_points = [
    ("Grid saturation narrative (central strategic insight).",
     "The framing that 81.9% of 2,147 substations are Congested, with the argument "
     "that grid reinforcement is the real cost driver, is exactly the story the brief "
     "rewards under T3 and B3. Keep this as the pitch spine."),
    ("Assumptions table is well-structured and cited.",
     "Table 3 covers all the mandatory assumption categories the brief explicitly asks "
     "for (EV autonomy, charging behavior, AFIR minimums, EV fleet source, penetration, "
     "seasonal). T1 is strong."),
    ("Grid thresholds justified with quantitative reasoning.",
     "The &quot;5 MW = 8x safety margin for a 4-charger station&quot; logic in Table 4 directly "
     "addresses the brief's mandatory &quot;document and justify grid_status thresholds&quot; "
     "requirement. Explicitly satisfies the disqualification-risk item."),
    ("Iberdrola-specific value proposition.",
     "The dual-role insight (CPO plus DSO) is unique and defensible. Section 6 makes "
     "the argument that Iberdrola has three structural advantages. This is B3 material "
     "done right."),
    ("Strategic roadmap with concrete deployment quarters.",
     "Q3 2027 / Q4 2027 / Q1 2028 phasing grouped by DSO zone. Addresses B2 "
     "feasibility directly. Needs the regulatory citations tightened (see C3)."),
    ("14-dataset sourcing including INE, REE, ACEA, ANFAC, BOE.",
     "Shows T2 innovation beyond the mandatory three datasets. Strong signal of "
     "independent research as the brief encourages."),
]
for title, text in strong_points:
    story.append(Paragraph(f"<b>{title}</b> {text}", body))

# ---- RUBRIC AUDIT ----
story.append(Paragraph("Rubric audit (against brief section 6)", h1))

rubric_rows = [
    ["", "Criterion", "Current state", "Action"],
    ["T1", "Assumptions + credibility", "Strong: Table 3 + Table 4. All mandatory items present.", "Fix M4, M5"],
    ["T2", "Data sources + independent research", "Strong: 14 datasets cited.", "None required"],
    ["T3", "Methodology (optimization + grid)", "Clear; Section 3.3 and 3.4 cover both objectives.", "Fix M6"],
    ["T4", "Output dataset compliance", "Compliant, but REPORT TABLES DO NOT MATCH outputs.", "Fix C1"],
    ["T5", "Code quality / reproducibility", "Not evaluated here (covered in Colab).", "Colab merge pending"],
    ["B1", "Integration technical <-> strategic", "Strong in Section 6. Dependent on fixing C1.", "Fix C1"],
    ["B2", "Strategic roadmap feasibility", "Strong: phased, DSO-grouped.", "Fix C3"],
    ["B3", "Iberdrola business value", "Strong: dual-role CPO+DSO.", "Fix C2, M3"],
    ["B4", "BI Visualization (map)", "Separate deliverable.", "Verify bi_map.html stands"],
    ["B5", "Pitch (5 min max)", "Not yet produced.", "Start after report lock"],
    ["B6", "Formal compliance + citation", "Length over limit; some citations need verification.", "Fix M1, C3"],
]
rubric = Table(rubric_rows, colWidths=[0.9*cm, 4.6*cm, 7.4*cm, 2.4*cm])
rubric.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f6f6f6"), white]),
    ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
    ("LEFTPADDING", (0, 0), (-1, -1), 3),
    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story.append(rubric)

# ---- DISQUALIFICATION CHECKS ----
story.append(Paragraph("Disqualification safeguards (brief section 6.4)", h1))

dq_items = [
    ("Missing or undocumented assumptions",
     "<b>Pass.</b> Table 3 documents all core assumptions with citations."),
    ("Output datasets in required structure",
     "<b>Pass.</b> File_1/2/3 schemas match the brief exactly."),
    ("grid_status thresholds justified in the report",
     "<b>Pass.</b> Table 4 provides quantitative justification for each threshold."),
    ("estimated_demand_kw calculated as n_chargers × 150 kW",
     "<b>Pass.</b> Verified against File_3."),
    ("File_3 only Moderate or Congested (never Sufficient)",
     "<b>Pass.</b> All 8 friction points are Congested."),
    ("Visible code outputs in Colab",
     "Pending. The Colab merge task will produce one notebook with visible outputs."),
    ("BI Visualization: no software install / credentials",
     "Pending verification. bi_map.html should be a self-contained Folium map."),
    ("Analytical Report submitted",
     "In progress."),
    ("Final Presentation submitted",
     "Not yet started."),
]
for title, text in dq_items:
    story.append(Paragraph(f"<b>{title}.</b> {text}", body))

# ---- Suggested order of attack ----
story.append(Paragraph("Suggested order of fixes for David", h1))

order_steps = [
    "Open <font face='Courier'>output/File_2.csv</font> and copy station rows verbatim into Table 6. Replace fabricated stations (N-330, N-2, STA_0009) with the real eight.",
    "Update Table 5 KPIs: 8 stations / 28 chargers / 4.2 MW total / 8 friction points / 2,498,159 EVs.",
    "Update Table 7 DSO splits using <font face='Courier'>output/dso_investment_summary.csv</font>: i-DE 1 station / 4 chargers / 0.6 MW, Endesa 5 / 18 / 2.7 MW, Viesgo 2 / 6 / 0.9 MW.",
    "Update the Executive Summary phrasing: &quot;network of 8 stations with 28 chargers and 4.2 MW of peak demand&quot;. Adjust &quot;seven stations by Q4 2027, one by Q1 2028&quot; etc.",
    "Replace &quot;i-Charging&quot; with &quot;Iberdrola Movilidad Verde&quot; throughout.",
    "Fix the &quot;16 notebooks&quot; claim in Section 3.1 (we have 14).",
    "Verify every BOE and Orden TED number used in Section 6. If uncertain, soften or remove.",
    "Cut 3 pages. Target: Section 2 to half a page, Section 3.2 to half a page, merge 6.4 and 6.5.",
    "Sanity-check Section 3.2 to list all three AFIR tiers (Core, Comprehensive, general).",
]
for i, step in enumerate(order_steps, 1):
    story.append(Paragraph(f"<b>{i}.</b> {step}", body))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    "If fixing C1 (the number mismatch) is the only thing David has time for, do that one. "
    "It is the highest-leverage change. Without it the report looks uncalibrated; with it "
    "the tech-strategy integration (B1) and disqualification safety (T4) are both secured.",
    note,
))

doc.build(story)
print(f"Saved {OUT}")
