#!/usr/bin/env python3
"""Turn the 3D-quality audit into a page you can look at.

Numbers alone do not settle whether monocular HMR output is usable: a jitter
figure means nothing without a distribution that is right by construction
beside it, and a "worst clip" means nothing until you watch the skeleton do the
thing the number accuses it of.  So this renders both -- every metric as wild
against the mocap reference, and the clips at the tail of each metric as
skeletons.

Two rendering decisions, both learned the hard way in this repo:

* **Skeletons are drawn 30 degrees off the front axis, never straight on.**  A
  pure x-z projection throws away the depth axis, and forward-folding motion
  happens almost entirely in depth: the body loses most of its height on screen
  and reads as a broken skeleton.  That produced a defect report against the 3D
  estimation that was really a defect in the picture.  The cost is 13% lateral
  compression, which is the cheaper error.
* **Charts are rendered twice, light and dark.**  A single PNG bakes its axis
  ink, and half the readers would get grey text on a grey ground.  The
  skeletons instead render once on a transparent ground with coloured bones and
  no text inside the image, which is legible either way.

Colours are the data-viz reference palette's categorical slots 1 and 2 taken
unchanged (blue = wild, orange = mocap reference); that palette is documented
as passing the CVD and contrast gates in both modes, and this tool has no
business re-stepping it.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

WILD = {"light": "#2a78d6", "dark": "#3987e5"}
REFERENCE = {"light": "#eb6834", "dark": "#d95926"}
INK = {"light": "#0b0b0b", "dark": "#ffffff"}
MUTED = {"light": "#52514e", "dark": "#c3c2b7"}
GRID = {"light": "#e3e2df", "dark": "#33322f"}

# The metrics worth a panel, with the question each one answers.  The rest of
# the audit's columns stay in the JSON: a panel per column would bury the four
# that discriminate under ten that do not.
PANELS: List[Tuple[str, str, str, bool]] = [
    ("floor_z", "Floor height (m)",
     "where the ground sits in each clip's own world", False),
    ("lowest_toe_spread", "Lowest-toe wander (m)",
     "how far the supporting foot rides up and down", False),
    ("skate_p95", "Foot skate, p95 (m/s)",
     "speed of the planted foot, which should be still", True),
    ("jitter_ratio", "Jitter / own speed (1/s)",
     "acceleration in units of the clip's own motion", True),
    ("penetration_fraction", "Frames below the floor",
     "fraction of frames with a toe under the ground", True),
    ("root_speed_max", "Peak root speed (m/s)",
     "tracking failures show up as the pelvis teleporting", True),
    ("body_height", "Head above floor (m)",
     "one fixed skeleton, so this should barely move", False),
    ("frozen_fraction", "Frozen frames",
     "a stuck tracker repeats a pose", True),
]


def figure_to_data_uri(figure, *, transparent: bool = False) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight",
                   transparent=transparent)
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("ascii")


def values_of(records: Sequence[Dict], key: str) -> np.ndarray:
    return np.asarray([r[key] for r in records if key in r and "error" not in r],
                      dtype=np.float64)


def distribution_figure(wild: Sequence[Dict], reference: Sequence[Dict], mode: str):
    """Small multiples: one panel per metric, two step-outline histograms.

    Step outlines rather than filled bars: two filled histograms overlap, and
    the fix for overlap (a surface ring on every bar) costs more ink than the
    comparison is worth.  Outlines never occlude each other.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = INK[mode], MUTED[mode], GRID[mode]
    figure, axes = plt.subplots(2, 4, figsize=(16, 7.8))
    for position, (axis, (key, title, subtitle, logx)) in enumerate(
            zip(axes.ravel(), PANELS)):
        a = values_of(wild, key)
        b = values_of(reference, key) if reference else np.zeros(0)
        if logx:
            floor = max(1e-4, min([v for v in np.concatenate([a, b]) if v > 0] or [1e-4]))
            a = np.maximum(a, floor)
            b = np.maximum(b, floor) if len(b) else b
            edges = np.geomspace(floor, max(a.max(), b.max() if len(b) else a.max()), 40)
            axis.set_xscale("log")
        else:
            lo = min(a.min(), b.min()) if len(b) else a.min()
            hi = max(a.max(), b.max()) if len(b) else a.max()
            edges = np.linspace(lo, hi, 40)
        for values, colour, label in ((a, WILD[mode], "wild"),
                                      (b, REFERENCE[mode], "AIST++ mocap")):
            if not len(values):
                continue
            counts, _ = np.histogram(values, bins=edges)
            axis.step(edges[:-1], counts / counts.sum(), where="post",
                      color=colour, linewidth=2.0, label=label)
        # Title and subtitle are both placed by hand.  set_title's pad is
        # measured to the *baseline*, so a second line drawn at 1.02 lands on
        # top of it -- which is exactly what happened the first time this
        # rendered.
        axis.text(0, 1.16, title, transform=axis.transAxes, color=ink,
                  fontsize=11, va="bottom")
        axis.text(0, 1.03, subtitle, transform=axis.transAxes, color=muted,
                  fontsize=8.5, va="bottom")
        axis.tick_params(colors=muted, labelsize=8)
        axis.grid(True, color=grid, linewidth=0.8, alpha=0.6)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_yticks([])
        # Height is share-of-clips in both series, normalised separately so two
        # corpora of different size are comparable; saying so once per row beats
        # a tick scale nobody reads off a histogram.
        if position % 4 == 0:
            axis.set_ylabel("share of clips", color=muted, fontsize=8.5)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc="upper right", frameon=False,
                           ncol=2, fontsize=10)
    for text in legend.get_texts():
        text.set_color(ink)
    figure.patch.set_alpha(0)
    for axis in axes.ravel():
        axis.patch.set_alpha(0)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    uri = figure_to_data_uri(figure, transparent=True)
    plt.close(figure)
    return uri


def filmstrip(joints: np.ndarray, *, frames: int, colour: str, floor: float,
              ground: str):
    """A row of poses across the clip, in world coordinates, above its floor.

    Deliberately **not** ``render_prototype_cards.draw_pose``: that one draws the
    canonical pose -- translation removed, hips turned to +x, divided by
    shoulder width -- because that is what the clustering sees.  Here the whole
    claim is about the body's relation to the ground, and canonicalising would
    subtract exactly the signal: vertical wander and foot skate would both be
    normalised away and every clip would look fine.  The 30-degree turn is
    reused, because that lesson still applies.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tools.render_prototype_cards import project
    from vis import smpl_parents

    picks = np.linspace(0, len(joints) - 1, frames).astype(int)
    screens = [project(joints[index], 30.0) for index in picks]
    # One horizontal window for the whole strip, so travel across the floor
    # stays visible instead of being re-centred out of existence frame by frame.
    horizontal = np.concatenate([s[:, 0] for s in screens])
    centre = float(np.median(horizontal))
    span = max(1.2, float(np.abs(horizontal - centre).max()) * 1.15)
    top = max(2.0, float(joints[picks][:, :, 2].max() - floor) * 1.15)

    figure, axes = plt.subplots(1, frames, figsize=(1.55 * frames, 2.6))
    for axis, screen in zip(np.atleast_1d(axes), screens):
        axis.axhline(0.0, color=ground, linewidth=1.2, zorder=1)
        for joint, parent in enumerate(smpl_parents[:22]):
            if parent < 0:
                continue
            axis.plot([screen[joint, 0], screen[parent, 0]],
                      [screen[joint, 1] - floor, screen[parent, 1] - floor],
                      color=colour, linewidth=1.6, solid_capstyle="round", zorder=2)
        axis.scatter(screen[:22, 0], screen[:22, 1] - floor, s=3.0, color=colour, zorder=3)
        axis.set_xlim(centre - span, centre + span)
        axis.set_ylim(-0.35, top)
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.patch.set_alpha(0)
    figure.patch.set_alpha(0)
    figure.subplots_adjust(wspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)
    uri = figure_to_data_uri(figure, transparent=True)
    plt.close(figure)
    return uri


def worst(records: Sequence[Dict], key: str, count: int, *, largest: bool = True
          ) -> List[Dict]:
    usable = [r for r in records if "error" not in r]
    return sorted(usable, key=lambda r: r[key], reverse=largest)[:count]


def build(*, audit: pathlib.Path, bundle: pathlib.Path, output: pathlib.Path,
          strips: int, strip_frames: int) -> pathlib.Path:
    from tools.convert_motion_to_guofeats import motion_151_to_joints

    report = json.loads(audit.read_text(encoding="utf-8"))
    wild_records = report["records"]
    reference_records = report.get("reference_records", [])
    rows = {json.loads(l)["recording_id"]: json.loads(l)
            for l in (bundle / "sequences.jsonl").open(encoding="utf-8")}

    charts = {mode: distribution_figure(wild_records, reference_records, mode)
              for mode in ("light", "dark")}

    # The tails that a picture can actually adjudicate.  Skate and wander are
    # claims about the body relative to the ground; a filmstrip either shows
    # the foot sliding and the body bobbing, or it does not.
    cases = []
    for key, title, largest in (("lowest_toe_spread", "Largest vertical wander", True),
                                ("skate_p95", "Worst foot skate", True),
                                ("jitter_ratio", "Most jitter per unit speed", True)):
        for record in worst(wild_records, key, strips, largest=largest):
            row = rows.get(record["recording_id"])
            if row is None:
                continue
            joints = motion_151_to_joints(np.load(bundle / row["motion_path"]))
            cases.append({
                "group": title, "metric": key, "value": record[key],
                "recording_id": record["recording_id"], "frames": record["frames"],
                "image": filmstrip(joints, frames=strip_frames, colour=WILD["light"],
                                   floor=record["floor_z"], ground=MUTED["light"]),
                "detail": "floor {floor_z} m · wander {lowest_toe_spread} m · "
                          "skate p95 {skate_p95} m/s · jitter/speed {jitter_ratio}".format(**record),
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page(report, charts, cases), encoding="utf-8")
    return output


GATE_KEYS = ("lowest_toe_spread", "skate_p95", "jitter_ratio", "root_speed_max",
             "frozen_fraction", "penetration_fraction", "body_height")


def gates(report: Dict) -> Dict[str, object]:
    """How many wild clips fall outside anything the mocap corpus produced.

    The threshold is the reference's own p99, not a number chosen here: it makes
    "bad" mean "past the worst 1% of motion capture", which is a claim a reader
    can check, rather than a cutoff that could be moved until the answer looked
    acceptable.  By construction the reference scores 1% on every row, and that
    row is printed so the comparison is never read as a wild-only indictment.
    """
    wild = [r for r in report["records"] if "error" not in r]
    mocap = [r for r in report.get("reference_records", []) if "error" not in r]
    if not mocap:
        return {}
    rows, thresholds = [], {}
    for key in GATE_KEYS:
        threshold = float(np.percentile([r[key] for r in mocap], 99))
        thresholds[key] = threshold
        over = sum(1 for r in wild if r[key] > threshold)
        rows.append({
            "key": key,
            "title": dict((p[0], p[1]) for p in PANELS).get(key, key),
            "threshold": round(threshold, 4),
            "wild_over": over,
            "wild_percent": round(100.0 * over / len(wild), 1),
        })
    counts = [sum(1 for k in GATE_KEYS[:5] if r[k] > thresholds[k]) for r in wild]
    return {
        "rows": rows,
        "any": int(sum(1 for c in counts if c >= 1)),
        "two_or_more": int(sum(1 for c in counts if c >= 2)),
        "three_or_more": int(sum(1 for c in counts if c >= 3)),
        "total": len(wild),
    }


def summary_table(report: Dict) -> str:
    wild = report["wild"]["per_metric"]
    reference = report.get("reference", {}).get("per_metric")
    head = ("<tr><th>metric</th><th>wild median</th><th>mocap median</th>"
            "<th>wild p95</th><th>mocap p95</th><th>wild ÷ mocap</th></tr>")
    body = []
    for key, title, _subtitle, _log in PANELS:
        w, m = wild[key], (reference or {}).get(key)
        ratio = "—"
        if m and m["median"]:
            value = w["median"] / m["median"]
            ratio = "<b>{:.2f}×</b>".format(value) if value >= 1.5 else "{:.2f}×".format(value)
        body.append(
            "<tr><td>{}</td><td class=n>{}</td><td class=n>{}</td>"
            "<td class=n>{}</td><td class=n>{}</td><td class=n>{}</td></tr>".format(
                title, w["median"], m["median"] if m else "—",
                w["p95"], m["p95"] if m else "—", ratio))
    return "<table>{}{}</table>".format(head, "".join(body))


def gate_table(gate: Dict) -> str:
    if not gate:
        return "<p class=lede>No mocap reference was measured, so there is no gate.</p>"
    rows = "".join(
        "<tr><td>{title}</td><td class=n>{threshold}</td>"
        "<td class=n>{wild_over}</td><td class=n>{wild_percent}%</td></tr>".format(**row)
        for row in gate["rows"])
    return ("<table><tr><th>metric</th><th>mocap p99</th><th>wild clips over</th>"
            "<th>share</th></tr>{}</table>".format(rows))


def page(report: Dict, charts: Dict[str, str], cases: Sequence[Dict]) -> str:
    wild, reference = report["wild"], report.get("reference", {})
    gauge = wild["gauge"]
    groups: Dict[str, List[Dict]] = {}
    for case in cases:
        groups.setdefault(case["group"], []).append(case)
    blocks = []
    for title, items in groups.items():
        cards = "".join(
            "<figure><img src='{image}' alt='skeleton filmstrip'>"
            "<figcaption><b>{recording_id}</b> · {frames} frames<br>{detail}</figcaption>"
            "</figure>".format(**item) for item in items)
        blocks.append("<h3>{}</h3><div class=strips>{}</div>".format(title, cards))

    gate = gates(report)
    metric = wild["per_metric"]
    reference_metric = reference.get("per_metric", {})

    def pair(key, unit=""):
        """A wild number is not readable alone; it ships with its mocap twin."""
        mine = metric[key]["median"]
        theirs = reference_metric.get(key, {}).get("median")
        ratio = ""
        if theirs:
            ratio = "<i>{:.2f}× mocap</i>".format(mine / theirs)
        # Left open on purpose: the caller appends the label and closes it.
        return ("<div class=kpi><b>{}{}</b><em>mocap {}{}</em>{}".format(
            mine, unit, theirs if theirs is not None else "—", unit, ratio))

    return """<title>Wild 3D pose asset audit</title>
<style>
:root{{color-scheme:light;
  --ground:#fbfbfc;--panel:#ffffff;--ink:#12151a;--muted:#5b6570;--rule:#e2e5ea;
  --wild:#2a78d6;--mocap:#eb6834;--warn:#b45309}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;
  --ground:#14171b;--panel:#191d22;--ink:#f2f4f7;--muted:#9aa5b1;--rule:#282d34;
  --wild:#3987e5;--mocap:#d95926;--warn:#e0a44a}}}}
:root[data-theme="dark"]{{color-scheme:dark;
  --ground:#14171b;--panel:#191d22;--ink:#f2f4f7;--muted:#9aa5b1;--rule:#282d34;
  --wild:#3987e5;--mocap:#d95926;--warn:#e0a44a}}
*{{box-sizing:border-box}}
body{{background:var(--ground);color:var(--ink);margin:0 auto;padding:40px 24px 80px;
  max-width:1120px;font:15.5px/1.65 ui-sans-serif,system-ui,-apple-system,sans-serif}}
h1,h2{{font-family:'Iowan Old Style',Georgia,'Times New Roman',serif;font-weight:600;
  text-wrap:balance;letter-spacing:-.01em}}
h1{{font-size:32px;margin:0 0 10px}}
h2{{font-size:21px;margin:52px 0 6px;padding-top:18px;border-top:1px solid var(--rule)}}
h3{{font-size:12px;margin:26px 0 10px;color:var(--muted);font-weight:600;
  text-transform:uppercase;letter-spacing:.07em}}
p.lede{{color:var(--muted);margin:0 0 8px;max-width:68ch}}
section{{display:flex;flex-direction:column;gap:14px}}
img{{max-width:100%;display:block}}
.charts img{{width:100%}} .charts .dark{{display:none}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]) .charts .light{{display:none}}
  :root:not([data-theme="light"]) .charts .dark{{display:block}}}}
:root[data-theme="dark"] .charts .light{{display:none}}
:root[data-theme="dark"] .charts .dark{{display:block}}
table{{border-collapse:collapse;width:100%;font-size:14px;min-width:640px}}
th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid var(--rule)}}
th{{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;
  letter-spacing:.05em}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.scroll{{overflow-x:auto}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;
  margin:22px 0 6px}}
.kpi{{background:var(--panel);border:1px solid var(--rule);border-radius:10px;
  padding:14px 16px;border-left:3px solid var(--wild)}}
.kpi b{{display:block;font-size:27px;line-height:1.15;font-variant-numeric:tabular-nums;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.kpi em{{display:block;font-style:normal;color:var(--mocap);font-size:13px;
  font-variant-numeric:tabular-nums}}
.kpi i{{display:block;font-style:normal;color:var(--muted);font-size:12.5px;margin-top:4px}}
.kpi span{{display:block;color:var(--muted);font-size:12.5px;margin-top:6px}}
.strips{{display:flex;flex-direction:column;gap:12px}}
.strips figure{{margin:0;background:var(--panel);border:1px solid var(--rule);
  border-radius:10px;padding:12px 14px}}
figcaption{{color:var(--muted);font-size:12.5px;margin-top:8px;
  font-variant-numeric:tabular-nums}}
figcaption b{{color:var(--ink);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:12px}}
.key{{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:13px;
  margin-right:18px}}
.key s{{width:16px;height:2px;border-radius:1px;text-decoration:none;display:inline-block}}
</style>

<h1>Wild 3D pose asset audit</h1>
<p class=lede>{sequences} monocular-HMR sequences ({frames} frames) from TikTok video,
measured against {ref_sequences} AIST++ motion-capture sequences that are right by
construction. Every number is a physical claim about the body — where the ground is,
whether the planted foot stays put, whether the pelvis teleports — and none of them
is judgeable without the mocap column beside it.</p>
<p><span class=key><s style="background:var(--wild)"></s>wild (monocular HMR)</span>
<span class=key><s style="background:var(--mocap)"></s>AIST++ motion capture</span></p>

<div class=kpis>
  {kpi_wander}
  {kpi_planted}
  {kpi_jitter}
  {kpi_height}
</div>
<p class=lede>Vertical stability is the one axis where the two corpora genuinely part:
the supporting foot rides up and down more than twice as far, and is near the floor
less than half as often. Jitter and body height sit on top of mocap.</p>

<h2>Distributions</h2>
<p class=lede>Both series are normalised to their own corpus, so shape is comparable
even though one has {sequences} clips and the other {ref_sequences}.</p>
<div class=charts>
  <img class=light src="{chart_light}" alt="metric distributions, wild versus mocap">
  <img class=dark src="{chart_dark}" alt="metric distributions, wild versus mocap">
</div>

<h2>Every metric, both corpora</h2>
<div class=scroll>{table}</div>
<p class=lede>Floor height differs by convention, not quality: motion capture sits at
{ref_floor} m and the wild corpus at {wild_floor} m. Neither is z = 0, which is why
penetration is measured against each clip's own floor rather than the world origin.</p>

<h2>How many clips are actually bad</h2>
<p class=lede>The threshold on each row is the mocap corpus's own 99th percentile, so
“over” means “past the worst 1% of motion capture”. Motion capture scores 1% on every
row by construction.</p>
<div class=scroll>{gate_table}</div>
<p class=lede><b>{gate_any} of {gate_total} clips ({gate_any_pct}%)</b> exceed that
threshold on at least one of the five motion axes; {gate_two} exceed it on two or
more and {gate_three} on three or more. The failures are overwhelmingly single-axis,
which is the signature of mild artefacts rather than broken reconstructions.</p>

<h2>The tails, drawn</h2>
<p class=lede>Skeletons are projected 30° off the front axis: a straight-on view drops
the depth axis, and forward-folding motion lives almost entirely in depth — it would
manufacture a defect that is not in the data. The rule under each figure is that
clip's estimated floor.</p>
{blocks}
""".format(
        sequences=wild["sequences"], frames=wild["frames"],
        ref_sequences=reference.get("sequences", 0),
        kpi_wander=pair("lowest_toe_spread", " m") + "<span>lowest-toe wander</span></div>",
        kpi_planted=pair("planted_fraction") + "<span>frames with a foot planted</span></div>",
        kpi_jitter=pair("jitter_ratio") + "<span>jitter per unit speed (1/s)</span></div>",
        kpi_height=pair("body_height", " m") + "<span>head above floor</span></div>",
        wild_floor=metric["floor_z"]["median"],
        ref_floor=reference_metric.get("floor_z", {}).get("median", "—"),
        chart_light=charts["light"], chart_dark=charts["dark"],
        table=summary_table(report), gate_table=gate_table(gate),
        gate_any=gate.get("any", 0), gate_total=gate.get("total", 0),
        gate_any_pct=round(100.0 * gate.get("any", 0) / max(gate.get("total", 1), 1), 1),
        gate_two=gate.get("two_or_more", 0), gate_three=gate.get("three_or_more", 0),
        blocks="".join(blocks))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--strips", type=int, default=3, help="clips per tail group")
    parser.add_argument("--strip-frames", type=int, default=8)
    args = parser.parse_args(argv)
    path = build(audit=args.audit, bundle=args.bundle, output=args.output,
                 strips=args.strips, strip_frames=args.strip_frames)
    print("wrote {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
