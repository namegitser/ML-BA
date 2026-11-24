"""Build the presentation from outputs/results.json and the analysis figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
NAVY = "10233F"
PANEL = "192E4C"
PANEL_LIGHT = "203957"
WHITE = "F7FAFC"
MUTED = "A8B7C9"
TEAL = "12B8A6"
BLUE = "5A9BFF"
ORANGE = "F5B544"
RED = "F47768"
LINE = "304763"
FONT = "Aptos"
WIDTH = 13.333
HEIGHT = 7.5


def color(hex_value: str) -> RGBColor:
    return RGBColor.from_string(hex_value)


def box(slide, x, y, w, h, fill=PANEL, radius=True, line_color=None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(fill)
    shape.line.color.rgb = color(line_color or fill)
    return shape


def text(
    slide,
    value,
    x,
    y,
    w,
    h,
    size=14,
    fill=WHITE,
    bold=False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    font=FONT,
    margin=0,
):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    for index, line in enumerate(str(value).split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.alignment = align
        paragraph.space_after = Pt(0)
        paragraph.space_before = Pt(0)
        paragraph.font.name = font
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color(fill)
    return shape


def background(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(WIDTH), Inches(HEIGHT)
    )
    bg.fill.solid()
    bg.fill.fore_color.rgb = color(NAVY)
    bg.line.color.rgb = color(NAVY)
    return slide


def header(slide, eyebrow, title, subtitle=""):
    text(slide, eyebrow.upper(), 0.64, 0.32, 11.9, 0.22, 9, TEAL, True)
    text(slide, title, 0.64, 0.64, 12.0, 0.55, 25, WHITE, True)
    if subtitle:
        text(slide, subtitle, 0.67, 1.23, 12.0, 0.38, 11, MUTED)


def footer(slide, page):
    text(
        slide,
        "HILLSTROM RCT (2008)  |  TWO-ARM HOLDOUT  |  $5 CONTACT COST IS A SCENARIO ASSUMPTION",
        0.65,
        7.16,
        11.4,
        0.16,
        7.4,
        MUTED,
    )
    text(slide, f"{page:02d} / 08", 12.08, 7.12, 0.6, 0.2, 8, MUTED, align=PP_ALIGN.RIGHT)


def metric_card(slide, x, y, w, label, value, note, accent=TEAL, value_size=24):
    box(slide, x, y, w, 1.17, PANEL)
    box(slide, x, y, 0.055, 1.17, accent, radius=False)
    text(slide, label.upper(), x + 0.2, y + 0.14, w - 0.38, 0.2, 8.5, MUTED, True)
    text(slide, value, x + 0.2, y + 0.39, w - 0.38, 0.38, value_size, WHITE, True)
    text(slide, note, x + 0.2, y + 0.85, w - 0.38, 0.2, 8.5, MUTED)


def money(value, decimals=2):
    return f"-${abs(value):,.{decimals}f}" if value < 0 else f"${value:,.{decimals}f}"


def percent(value, decimals=1):
    return f"{100 * value:.{decimals}f}%"


def method_map(results):
    return {entry["method"]: entry for entry in results["shortlists"]}


def profile_map(results):
    return {entry["method"]: entry for entry in results["profiles"]}


def slide_cover(prs, data, lookup):
    slide = background(prs)
    text(slide, "ML FOR BUSINESS ANALYTICS  /  DECISION SCIENCE", 0.72, 0.48, 7.0, 0.24, 9, TEAL, True)
    text(slide, "WHO GETS THE\nNEXT 1,500?", 0.72, 1.15, 7.2, 1.7, 36, WHITE, True)
    text(
        slide,
        "A randomized test of response ranking,\ncausal uplift, and the cost of being wrong.",
        0.76,
        3.05,
        6.7,
        0.85,
        18,
        MUTED,
    )
    box(slide, 8.28, 0.92, 4.25, 4.1, PANEL)
    text(slide, "THE DECISION", 8.65, 1.28, 3.5, 0.24, 9, MUTED, True)
    text(slide, f"{data['design']['k']:,}", 8.61, 1.7, 3.5, 0.85, 52, WHITE, True)
    text(slide, f"of {data['data']['test_n']:,} held-out prospects", 8.68, 2.58, 3.4, 0.3, 14, MUTED)
    box(slide, 8.62, 3.15, 3.56, 0.02, LINE, radius=False)
    text(slide, "CONTACT COST", 8.68, 3.46, 1.7, 0.2, 8.5, MUTED, True)
    text(slide, money(data["design"]["unit_cost"]), 8.68, 3.77, 2.0, 0.48, 26, ORANGE, True)
    text(slide, "per prospect", 10.44, 3.91, 1.4, 0.22, 10, MUTED)

    rfm = lookup["RFM history"]
    callout = (
        f"RESULT: No shortlist clears the ${data['design']['unit_cost']:.0f} contact-cost "
        f"threshold on held-out gross spend. Best point estimate: RFM at "
        f"{money(rfm['spend_lift_per_customer'])} per selected customer "
        f"(95% CI {money(rfm['spend_ci_low'])} to {money(rfm['spend_ci_high'])})."
    )
    box(slide, 0.75, 5.28, 11.78, 1.02, PANEL_LIGHT)
    box(slide, 0.75, 5.28, 0.06, 1.02, ORANGE, radius=False)
    text(slide, callout, 1.0, 5.5, 11.1, 0.62, 15, WHITE, True)
    text(slide, "A HOLDOUT, NOT A PROMISE OF PROFIT", 0.78, 6.64, 6.8, 0.25, 9, TEAL, True)
    text(
        slide,
        "Email experiment; catalog/voucher economics are a scenario, not an observed intervention.",
        0.78,
        6.89,
        11.5,
        0.18,
        8,
        MUTED,
    )
    footer(slide, 1)
    return slide


def slide_rct(prs, data, balance_path):
    slide = background(prs)
    header(slide, "01  /  Valid experiment", "A randomized test makes targeting measurable",
           "Pre-campaign features rank customers; randomized assignment makes the selected-cohort comparison interpretable.")
    overall = data["overall"]
    metric_card(slide, 0.68, 1.83, 2.55, "Two-arm cohort", f"{data['data']['two_arm_n']:,}", "Womens email vs no email", BLUE)
    metric_card(slide, 3.42, 1.83, 2.55, "Held-out test", f"{data['data']['test_n']:,}", "30% split · seed 42", TEAL)
    metric_card(slide, 6.16, 1.83, 2.55, "Visit lift", f"+{100*overall['visit_lift']:.2f} pp", "15.14% treated vs 10.62% control", ORANGE)
    metric_card(slide, 8.90, 1.83, 3.70, "Overall spend lift", f"{money(overall['spend_lift_per_customer'])} / customer", "vs $5 assumed contact cost", RED, 21)
    box(slide, 0.7, 3.28, 7.25, 3.48, PANEL)
    slide.shapes.add_picture(str(balance_path), Inches(0.91), Inches(3.43), width=Inches(6.83), height=Inches(3.12))
    box(slide, 8.18, 3.28, 4.42, 3.48, PANEL)
    text(slide, "DESIGN CHECK", 8.5, 3.62, 3.5, 0.2, 9, TEAL, True)
    text(slide, f"Max |SMD| = {data['design']['max_absolute_smd']:.3f}", 8.5, 4.0, 3.8, 0.4, 22, WHITE, True)
    text(
        slide,
        f"{data['design']['balance_fields_under_010']} of {data['design']['balance_feature_count']} pre-campaign feature fields are below 0.10. This is a balance diagnostic—not proof that every possible bias is absent.",
        8.5,
        4.55,
        3.6,
        0.88,
        12,
        MUTED,
    )
    text(
        slide,
        "The source intervention is a randomized email. A $5 catalog/voucher intervention is not directly tested.",
        8.5,
        5.78,
        3.55,
        0.68,
        11,
        ORANGE,
        True,
    )
    footer(slide, 2)
    return slide


def slide_methods(prs, data):
    slide = background(prs)
    header(slide, "02  /  Three ranking policies", "Same customers. Three different questions.",
           "Every score uses the same eight pre-campaign features; only the scoring logic changes.")
    cards = [
        (0.72, BLUE, "01  RFM HISTORY", "Who spent most before?", "Score = past 12-month spend\n\nA transparent, non-model baseline."),
        (4.72, ORANGE, "02  CALIBRATED EV", "Who is likely to respond?", "P(visit | X, treated) × log(1 + history)\n\nCalibrated response, weighted by a CLV proxy."),
        (8.72, TEAL, "03  X-LEARNER", "Who changes because of contact?", "Estimated P(visit | X, treated) − P(visit | X, control)\n\nCounterfactual response difference."),
    ]
    for x, accent, label, question, formula in cards:
        box(slide, x, 2.05, 3.83, 3.37, PANEL)
        box(slide, x, 2.05, 3.83, 0.07, accent, radius=False)
        text(slide, label, x + 0.28, 2.39, 3.2, 0.23, 9, accent, True)
        text(slide, question, x + 0.28, 2.83, 3.2, 0.67, 18, WHITE, True)
        text(slide, formula, x + 0.28, 3.73, 3.22, 1.22, 12, MUTED)
    box(slide, 0.74, 5.78, 11.8, 0.86, PANEL_LIGHT)
    text(
        slide,
        "TRAINING SIGNAL: website visit (denser)     |     ECONOMIC TEST: observed spend in the untouched randomized holdout",
        1.0,
        6.02,
        11.25,
        0.32,
        12,
        WHITE,
        True,
        align=PP_ALIGN.CENTER,
    )
    footer(slide, 3)
    return slide


def slide_calibration(prs, data, chart_path):
    slide = background(prs)
    cal = data["calibration"]
    diff = cal["brier_isotonic"] - cal["brier_uncalibrated"]
    header(slide, "03  /  Probability quality", "Calibration is a check, not a magic upgrade",
           "Isotonic calibration was fit only in training folds; the curve and Brier scores are measured on treated holdout customers.")
    box(slide, 0.7, 1.82, 7.63, 4.95, PANEL)
    slide.shapes.add_picture(str(chart_path), Inches(0.91), Inches(2.0), width=Inches(7.2), height=Inches(4.55))
    metric_card(slide, 8.68, 2.03, 3.86, "Raw Brier · lower is better", f"{cal['brier_uncalibrated']:.5f}", "Treated holdout", BLUE, 22)
    metric_card(slide, 8.68, 3.42, 3.86, "Isotonic Brier", f"{cal['brier_isotonic']:.5f}", f"Change: {diff:+.5f}", ORANGE, 22)
    box(slide, 8.68, 5.0, 3.86, 1.23, PANEL_LIGHT)
    text(slide, "READOUT", 8.95, 5.22, 3.2, 0.2, 9, TEAL, True)
    text(
        slide,
        "Here isotonic is slightly worse on Brier. Calibration improves score meaning only when it validates; it does not guarantee better ranking or economics.",
        8.95,
        5.55,
        3.18,
        0.55,
        10,
        WHITE,
    )
    footer(slide, 4)
    return slide


def slide_shortlists(prs, data):
    slide = background(prs)
    profiles = profile_map(data)
    overlap = data["overlap"]
    header(slide, "04  /  Who gets selected?", "The policies don't choose the same customer",
           "Top 1,500 profiles from the held-out cohort; profile differences describe ranking behavior, not causal effects.")
    box(slide, 0.74, 1.88, 3.08, 4.88, PANEL)
    text(slide, "TOP-K OVERLAP", 1.02, 2.2, 2.55, 0.24, 9, TEAL, True)
    text(slide, f"{overlap['calibrated_ev_x_xlearner_n']:,}", 1.02, 2.72, 2.4, 0.76, 43, WHITE, True)
    text(slide, "customers in both lists", 1.04, 3.5, 2.4, 0.28, 12, MUTED)
    text(slide, f"{overlap['calibrated_ev_x_xlearner_pct_of_k']:.1%} of K", 1.04, 4.08, 2.3, 0.4, 20, ORANGE, True)
    text(slide, f"Jaccard similarity: {overlap['jaccard']:.1%}", 1.04, 4.54, 2.4, 0.27, 11, MUTED)
    text(
        slide,
        "EV leans toward recent, high-spend buyers. Uplift prioritizes customers predicted to change response.",
        1.04,
        5.27,
        2.42,
        0.95,
        12,
        WHITE,
    )

    box(slide, 4.1, 1.88, 8.48, 4.88, PANEL)
    text(slide, "SHORTLIST PROFILE  /  TOP 1,500", 4.43, 2.2, 7.3, 0.23, 9, TEAL, True)
    headers = ["Profile", "RFM", "Calibrated EV", "X-learner"]
    columns = [
        ("Past spend", "history_mean", lambda v: money(v)),
        ("Recency (months)", "recency_mean_months", lambda v: f"{v:.1f}"),
        ("Women-category buyers", "womens_buyer_pct", lambda v: f"{v:.1f}%"),
        ("Multichannel buyers", "multichannel_pct", lambda v: f"{v:.1f}%"),
    ]
    x_positions = [4.43, 7.18, 8.82, 10.65]
    widths = [2.67, 1.55, 1.75, 1.56]
    for x, w, value in zip(x_positions, widths, headers):
        text(slide, value, x, 2.73, w, 0.3, 9, MUTED, True)
    profile_lookup = {
        "RFM": profiles["RFM"],
        "Calibrated EV": profiles["Calibrated EV"],
        "X-learner": profiles["X-learner"],
    }
    y = 3.23
    for label, key, formatter in columns:
        box(slide, 4.34, y - 0.07, 7.91, 0.62, PANEL_LIGHT if int((y - 3.23) * 10) % 2 == 0 else PANEL)
        text(slide, label, 4.48, y + 0.08, 2.52, 0.25, 10, WHITE)
        for j, name in enumerate(("RFM", "Calibrated EV", "X-learner")):
            text(slide, formatter(profile_lookup[name][key]), x_positions[j + 1], y + 0.08, widths[j + 1], 0.25, 10, WHITE, name == "X-learner")
        y += 0.74
    box(slide, 4.42, 6.32, 7.84, 0.01, LINE, radius=False)
    text(
        slide,
        "Selection shift is real; whether it creates value is answered by randomized outcomes, not the profile labels.",
        4.47,
        6.43,
        7.8,
        0.22,
        9,
        MUTED,
    )
    footer(slide, 5)
    return slide


def slide_economics(prs, data, qini_path):
    slide = background(prs)
    lookup = method_map(data)
    header(slide, "05  /  Held-out decision", "Visit lift is not the same as dollars earned",
           "Qini ranks policies by incremental visits; the table uses the same locked top-K groups to evaluate observed spend.")
    box(slide, 0.72, 1.76, 11.88, 2.65, PANEL)
    slide.shapes.add_picture(str(qini_path), Inches(0.91), Inches(1.86), width=Inches(11.48), height=Inches(2.42))
    box(slide, 0.72, 4.56, 11.88, 1.95, PANEL)
    cols = [0.98, 3.55, 5.1, 8.25, 10.62]
    widths = [2.45, 1.35, 2.97, 2.15, 1.68]
    table_headers = ["Policy", "Visit lift", "Spend lift / person*", "95% bootstrap CI", "Net contrib.**"]
    for x, w, value in zip(cols, widths, table_headers):
        text(slide, value, x, 4.79, w, 0.23, 8.5, MUTED, True)
    names = [
        ("Random selection", "Random selection", MUTED),
        ("RFM history", "RFM history", BLUE),
        ("Calibrated EV", "Calibrated expected value", ORANGE),
        ("X-learner uplift", "X-learner uplift", TEAL),
    ]
    y = 5.11
    for display, key, accent in names:
        item = lookup[key]
        box(slide, 0.91, y - 0.03, 11.47, 0.31, PANEL_LIGHT)
        text(slide, display, cols[0] + 0.04, y + 0.02, widths[0] - 0.08, 0.2, 9, accent, True)
        text(slide, f"{100*item['visit_lift']:+.1f} pp", cols[1], y + 0.02, widths[1], 0.2, 9, WHITE)
        text(slide, money(item["spend_lift_per_customer"]), cols[2], y + 0.02, 1.25, 0.2, 9, WHITE, True)
        text(
            slide,
            f"[{money(item['spend_ci_low'])}, {money(item['spend_ci_high'])}]",
            cols[2] + 1.34,
            y + 0.02,
            1.62,
            0.2,
            8.5,
            MUTED,
        )
        text(slide, money(item["net_contribution"], 0), cols[3] + 0.05, y + 0.02, widths[3], 0.2, 9, RED, True)
        y += 0.34
    text(
        slide,
        "* Incremental gross spend per selected customer.  ** Net contribution = K × (spend lift − $5); excludes product and operating costs.",
        0.98,
        6.66,
        11.5,
        0.22,
        8,
        MUTED,
    )
    footer(slide, 6)
    return slide


def slide_interpretation(prs, data, lookup):
    slide = background(prs)
    xlearner = lookup["X-learner uplift"]
    rfm = lookup["RFM history"]
    header(slide, "06  /  What the evidence says", "The model wins the visit metric. The business case still fails.",
           "A useful distinction: uplift ranking can find persuadable visitors without identifying profitable spend.")
    cards = [
        (0.75, 1.92, "DENSE SIGNAL", f"+{100*xlearner['visit_lift']:.1f} pp", "X-learner visit lift in top K\n(95% interval stays above zero)", TEAL),
        (4.73, 1.92, "MONETARY SIGNAL", money(xlearner["spend_lift_per_customer"]), "X-learner spend lift per person\n(95% interval spans both signs)", ORANGE),
        (8.71, 1.92, "BREAK-EVEN BAR", money(data["design"]["unit_cost"]), "Assumed contact cost per prospect\n(before COGS or redemption)", RED),
    ]
    for x, y, label, value, note, accent in cards:
        metric_card(slide, x, y, 3.85, label, value, note, accent, 23)

    box(slide, 0.75, 3.55, 5.78, 2.8, PANEL)
    text(slide, "WHY THE METRICS DISAGREE", 1.06, 3.87, 5.05, 0.22, 9, TEAL, True)
    text(
        slide,
        "Visit is a denser learning target than purchase or spend. But a visit is only a proxy: people can visit without buying, and revenue is not contribution margin.",
        1.06,
        4.3,
        5.03,
        1.0,
        14,
        WHITE,
    )
    text(
        slide,
        f"RFM has the highest observed spend lift ({money(rfm['spend_lift_per_customer'])}/person), yet its 95% upper bound is {money(rfm['spend_ci_high'])}—below the ${data['design']['unit_cost']:.0f} contact cost.",
        1.06,
        5.48,
        5.03,
        0.62,
        10,
        ORANGE,
        True,
    )
    box(slide, 6.8, 3.55, 5.78, 2.8, PANEL)
    text(slide, "WHAT A DEFENSIBLE READ REQUIRES", 7.12, 3.87, 5.1, 0.22, 9, BLUE, True)
    guardrails = [
        "1  Rank only with pre-campaign information.",
        "2  Compare treated and control inside the locked shortlist.",
        "3  Report uncertainty, not just point estimates.",
        "4  Do not transfer email results to a catalog without a pilot.",
    ]
    for i, line in enumerate(guardrails):
        text(slide, line, 7.12, 4.31 + i * 0.43, 5.0, 0.32, 11, WHITE if i < 3 else ORANGE, i == 3)
    footer(slide, 7)
    return slide


def slide_recommendation(prs, data, lookup):
    slide = background(prs)
    rfm = lookup["RFM history"]
    header(slide, "07  /  Decision & next step", "Do not scale the $5 campaign on this evidence",
           "Best observed spend estimate is still below the assumed contact cost—and the experiment tested email, not a catalog.")
    box(slide, 0.75, 1.83, 11.83, 1.25, PANEL_LIGHT)
    box(slide, 0.75, 1.83, 0.07, 1.25, RED, radius=False)
    text(slide, "DECISION", 1.05, 2.05, 1.2, 0.22, 9, RED, True)
    text(
        slide,
        f"NO-GO at ${data['design']['unit_cost']:.0f} per contact. RFM's spend-lift 95% CI tops out at {money(rfm['spend_ci_high'])} per customer.",
        2.14,
        1.99,
        9.9,
        0.48,
        17,
        WHITE,
        True,
    )
    steps = [
        ("01", "Measure margin", "Replace gross spend with contribution after COGS, discounts, returns, and fulfillment."),
        ("02", "Run a matched pilot", "Test the actual offer and channel; preserve a randomized no-contact holdout."),
        ("03", "Set a scale gate", "Scale only when expected incremental contribution clears fully loaded contact cost with uncertainty accounted for."),
    ]
    for i, (num, title, detail) in enumerate(steps):
        x = 0.78 + i * 4.0
        box(slide, x, 3.48, 3.72, 2.26, PANEL)
        text(slide, num, x + 0.27, 3.76, 0.72, 0.45, 23, TEAL, True)
        text(slide, title, x + 0.27, 4.35, 3.12, 0.34, 15, WHITE, True)
        text(slide, detail, x + 0.27, 4.87, 3.12, 0.63, 10.5, MUTED)
    text(
        slide,
        "Reproducible: run prospect_selection.py, then slides\\build_slides.py. Fixed seed 42; metrics and intervals are generated from the held-out split.",
        0.84,
        6.18,
        11.7,
        0.35,
        10,
        MUTED,
    )
    text(
        slide,
        "Source: Kevin Hillstrom, MineThatData E-Mail Analytics and Data Mining Challenge (2008), two-arm RCT.",
        0.84,
        6.62,
        11.7,
        0.22,
        8.5,
        MUTED,
    )
    footer(slide, 8)
    return slide


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "outputs" / "results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "slides" / "prospect_selection.pptx")
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = args.results.resolve()
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Missing {results_path}. Run prospect_selection.py before building the slides."
        )
    with results_path.open(encoding="utf-8") as file:
        data = json.load(file)
    figures = results_path.parent
    for name in ("balance.png", "calibration.png", "qini.png"):
        if not (figures / name).is_file():
            raise FileNotFoundError(f"Missing analysis figure: {figures / name}")

    prs = Presentation()
    prs.slide_width = Inches(WIDTH)
    prs.slide_height = Inches(HEIGHT)
    prs.core_properties.title = "Who Gets the Next 1,500? Prospect Selection Under a Budget"
    prs.core_properties.subject = "Hillstrom randomized email experiment: response ranking vs causal uplift"
    prs.core_properties.author = "ML for Business Analytics project"
    lookup = method_map(data)

    slide_cover(prs, data, lookup)
    slide_rct(prs, data, figures / "balance.png")
    slide_methods(prs, data)
    slide_calibration(prs, data, figures / "calibration.png")
    slide_shortlists(prs, data)
    slide_economics(prs, data, figures / "qini.png")
    slide_interpretation(prs, data, lookup)
    slide_recommendation(prs, data, lookup)

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    prs.save(args.output.resolve())
    print(f"Saved {len(prs.slides)} slides to {args.output.resolve()}")


if __name__ == "__main__":
    main()
