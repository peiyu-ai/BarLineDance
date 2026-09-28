#!/usr/bin/env python3
"""Assemble the M2 review page: which prototypes to trust, and the samples behind it.

Everything is inlined, so the page survives being copied off the pod.

Design note, 2026-08-21
-----------------------
The first version of this page hid which of each pair of panels was the real
prototype and which was the shuffled control, so a reviewer could judge before
being told.  That was the wrong trade.  Two things went wrong in practice:

* The pairing itself was misread -- two panels were taken for ten members of one
  prototype, and the disagreement between them (which is the *result*) was read
  as the prototype being incoherent.
* Hiding the answer left the reviewer with nothing to act on.  The question was
  never "can you spot the control", it was "which prototypes are good", and a
  blind test answers that only after the reviewer has done the work by hand for
  all 53.

So: **everything is labelled, the real arm is always on the left, and the arm is
burned into the image itself** via ``--title-note`` rather than written in the
HTML around it, because a reviewer reads the picture.  The control stays, because
without it a page of prototype sheets is the gate that cannot fail -- but it is
now labelled evidence rather than a quiz.

The table in section 4 is the deliverable.  The sheets are its samples.
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ROOT = pathlib.Path("/dev/shm/atomicdance-m2-review")
OUT = REPO / "output" / "clean5b5_m2_review.html"
SEED = 20260821


def uri(path: pathlib.Path, mime: str) -> str:
    import base64
    return "data:{};base64,{}".format(
        mime, base64.b64encode(path.read_bytes()).decode("ascii"))


def manifest_of(directory: pathlib.Path, name: str):
    report = json.loads((directory / name).read_text(encoding="utf-8"))
    return {int(row["label"]): row for row in report["manifest"]}


def members_for(labels_dir: pathlib.Path, bundle: pathlib.Path, *,
                subprototypes: int, members: int):
    """Reproduce exactly which segments each sheet drew.

    The renderers do not record their picks, and re-deriving them here rather
    than parsing them out of the images means the listed members are the ones
    actually drawn only if this stays in lockstep with the renderer's selection.
    Both call the same three functions in the same order with the same seed, so
    they do; if the renderer's rule changes, this must change with it.
    """
    from tools.build_atomic_gallery import collect_members, spread_over_uploads
    from tools.render_prototype_cards import select_subprototypes

    grouped = collect_members(labels_dir, bundle)
    chosen, picked = select_subprototypes(grouped, count=subprototypes,
                                          members=members, strategy="spread",
                                          min_uploads=2)
    for label, entries, _ in chosen:
        picked[label] = spread_over_uploads(entries, members)
    return picked


# The renderers' member colours, in draw order, so the table's dots match the
# skeletons on the sheet above it.
MEMBER_COLOURS = ["#4fc3f7", "#ffb74d", "#81c784", "#f06292", "#9575cd"]


def main() -> int:
    global ROOT, OUT, SEED
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--output", type=pathlib.Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    ROOT, OUT, SEED = args.root, args.output, args.seed

    tight = json.loads((ROOT / "pose_space_tightness.json").read_text())
    coherence = json.loads(
        (REPO / "runs/clean5b5_prototype_coherence_bothspaces.json").read_text())
    enrich = {k: json.loads((REPO / "runs" / f).read_text()) for k, f in (
        ("clean5b5", "clean5b5_prototype_enrichment.json"),
        ("subset", "wild_v4_clean5subset_prototype_enrichment.json"),
        ("full", "wild_v4_prototype_enrichment.json"))}

    real_cards = manifest_of(ROOT / "cards_real", "cards.json")
    ctrl_cards = manifest_of(ROOT / "cards_ctrl", "cards.json")
    real_clips = manifest_of(ROOT / "videos_real", "videos.json")
    ctrl_clips = manifest_of(ROOT / "videos_ctrl", "videos.json")

    bundle = ROOT / "bundle"
    picks = {
        ("cards", "real"): members_for(ROOT / "labels", bundle,
                                       subprototypes=24, members=5),
        ("cards", "ctrl"): members_for(ROOT / "labels_shuffled", bundle,
                                       subprototypes=24, members=5),
        ("videos", "real"): members_for(ROOT / "labels", bundle,
                                        subprototypes=12, members=5),
        ("videos", "ctrl"): members_for(ROOT / "labels_shuffled", bundle,
                                        subprototypes=12, members=5),
    }

    parts = []
    add = parts.append
    add("<title>M2 clean5b5 review</title>")
    add(STYLE)
    add("<div class='wrap'>")
    add("<h1>M2 on the clean5 corpus — which prototypes hold up</h1>")
    add("<p class='lede'>Tag <code>clean5b5</code>: K=53 prototypes over 11,971 accepted "
        "segments from 1,999 clips, TMR embeddings, music-beat segmentation. "
        "M2's own quality numbers cannot fail, so every figure here carries the null "
        "it was measured against, and every sample sheet is shown next to a "
        "size-matched control built from shuffled labels. "
        "<b>Section 4 carries a retraction: an earlier verdict about which prototypes "
        "were incoherent came from scoring them in the wrong space.</b></p>")

    # ------------------------------------------------------------------ 1
    add("<h2>1. First, the numbers that prove nothing</h2>")
    add("<p>M2's run report says: acceptance 0.8412, 0 empty prototypes, largest class "
        "3.4%, 225.9 segments per class. All four describe how tight the clusters are "
        "<em>in the TMR embedding K-Means minimised</em>. A K-Means fit to pure noise "
        "scores well on all four. Nothing below rests on them.</p>")

    # ------------------------------------------------------------------ 2
    obs = tight["observed_distance"]["weighted"]
    null = tight["null_distance_mean"]
    floor = tight["floor_distance"]["weighted"]
    add("<h2>2. Does the clustering carry structure at all?</h2>")
    add("<p>Measured in the <b>signature-pose space the sheets below actually draw</b> "
        "— canonical joint positions at up to four motion beats, 264 dims. M2 never "
        "optimised these coordinates, so tightness here is not true by construction. "
        "Cross-upload pairs only, in all three arms.</p>")
    lo, hi = floor - 0.06, null + 0.06
    p = lambda v: 100.0 * (v - lo) / (hi - lo)
    add("<div class='box'><div class='scale'><div class='bar'></div>")
    add("<div class='pin first' style='left:{:.1f}%'>K-Means fitted on pose<b>{:.3f}"
        "</b></div>".format(p(floor), floor))
    add("<div class='pin mid' style='left:{:.1f}%'>M2<b>{:.3f}</b></div>".format(
        p(obs), obs))
    add("<div class='pin last' style='left:{:.1f}%'>shuffled labels<b>{:.3f}</b>"
        "</div>".format(p(null), null))
    add("</div><p class='sub' style='margin-top:2.2rem'>Mean cross-upload distance "
        "inside a prototype — <b>lower is tighter</b>.</p></div>")
    add("<p><b>Yes, corpus-wide.</b> M2 sits <span class='good'>{:.0%}</span> of the "
        "way from no-structure to pose-fitted. 20 shuffles landed in [{:.4f}, {:.4f}] "
        "and M2 is {:.3f} below every one. But this is an average over 53 prototypes, "
        "and section 4 is where it stops being an average.</p>".format(
            tight["recovery_fraction"], null - 2 * tight["null_distance_sd"],
            null + 2 * tight["null_distance_sd"], null - obs))

    # ------------------------------------------------------------------ 3
    add("<h2>3. The cost side: how much is just the account?</h2>")
    add("<p>Whether prototypes concentrate the <b>choreographer account</b>. Here high "
        "is <em>bad</em>: an account is a person, a room and a camera.</p>")
    add("<div class='tablewrap'><table>")
    add("<tr><th>arm</th><th class='n'>K</th><th class='n'>segments</th>"
        "<th class='n'>observed</th><th class='n'>null</th><th class='n'>lift</th></tr>")
    for name, key, note in (("full wild_v4", "full", "25 accounts"),
                            ("same 1,999 clips, old generation", "subset", "5 accounts"),
                            ("clean5b5 (this run)", "clean5b5", "5 accounts")):
        row = enrich[key]
        add("<tr{}><td>{}<br><span class='sub' style='font-size:.82rem'>{}</span></td>"
            "<td class='n'>{}</td><td class='n'>{:,}</td><td class='n'>{:.4f}</td>"
            "<td class='n'>{:.4f}</td><td class='n'>{:.3f}</td></tr>".format(
                " style='font-weight:640'" if key == "clean5b5" else "",
                name, note, row["prototypes"], row["segments"],
                row["observed_purity"], row["null_purity_mean"], row["lift"]))
    add("</table></div>")
    add("<p>For scale the same probe reads <b>2.83</b> on AIST++ dance genre (a "
        "movement attribute, where high is good) and <b>1.045</b> on dancer identity "
        "with genre held fixed. clean5b5's 1.274 sits far closer to the identity "
        "floor. Most of the gap from 1.058 is not new: the old K=100 generation "
        "already read 1.208 on this same clip list.</p>")

    # ------------------------------------------------------------------ 4
    rows = coherence["per_prototype"]
    summary = coherence["summary"]
    add("<h2>4. Which prototypes to trust <span class='hd-note'>— and why this "
        "page no longer answers that</span></h2>")
    add("<div class='box retract'><p><b>Retraction.</b> An earlier version of this "
        "page said <b>12 of 53 prototypes &ldquo;buy nothing&rdquo;</b> and singled "
        "out #8 as a junk drawer. Both claims came from scoring M2's prototypes in "
        "<b>signature-pose space — which is not the space M2 clusters in</b>. Scored "
        "in the TMR embedding it actually optimises, the same 53 prototypes read a "
        "median of <b>{:.4f}</b>, with <b>all 53 below 0.85</b> and "
        "<b>none at or above 1.0</b>. Every one of the 12 flips. #8 reads "
        "<b>{:.3f}</b> in TMR — tighter than the median prototype.</p>"
        "<p>The ratio was never miscomputed. What was missing is a <b>positive "
        "control in the space doing the judging</b>: there was no calibration for "
        "what a good clustering scores in pose space, so &ldquo;0.95 in pose "
        "space&rdquo; was read as &ldquo;not a cluster&rdquo; when it is simply what "
        "a TMR-built clustering looks like seen from pose space. Both nulls sit at "
        "1.00, which is exactly why the error was invisible — the null was right, "
        "the ceiling was never measured.</p></div>".format(
            summary["tmr"]["median"], rows["8"]["tmr_ratio"]))
    add("<p>Both readings are below, side by side, and there is deliberately "
        "<b>no verdict column</b>. The two rankings are <b>uncorrelated</b> across "
        "the 53 (Spearman &rho; = {:.3f}, p = {:.2f}), so a verdict from either one "
        "is a statement about the space, not about the prototype.</p>".format(
            summary["rank_agreement_spearman_rho"],
            summary["rank_agreement_p"]))
    add("<div class='tablewrap'><table>")
    add("<tr><th>reading</th><th class='n'>median</th><th class='n'>&ge;1.0</th>"
        "<th class='n'>&lt;0.85</th><th class='n'>null median</th></tr>")
    for name, label, note in (
            ("tmr", "in TMR space", "what M2 clustered"),
            ("pose", "in pose space", "not what M2 optimised")):
        row = summary[name]
        add("<tr><td><b>{}</b><br><span class='sub' style='font-size:.82rem'>{}</span>"
            "</td><td class='n'>{:.4f}</td><td class='n'>{}</td><td class='n'>{}</td>"
            "<td class='n sub'>{:.4f}</td></tr>".format(
                label, note, row["median"], row["n_at_or_above_1"],
                row["n_below_085"], row["null_median"]))
    add("</table></div>")
    add("<p>What the pose reading <em>does</em> still say is narrower than the "
        "retracted claim: <b>M2's groups do not correspond to similar signature "
        "poses.</b> It groups by whatever TMR encodes. Whether that is the right "
        "thing to group by is a question about the downstream task, and no geometry "
        "on this page settles it.</p>")
    add("<div class='tablewrap'><table class='coh'>")
    add("<tr><th>prototype</th><th class='n'>segments</th><th class='n'>uploads</th>"
        "<th class='n'>ratio in TMR</th><th class='n'>ratio in pose</th>"
        "<th>TMR reading</th></tr>")
    for key, row in sorted(rows.items(), key=lambda kv: kv[1]["tmr_ratio"]):
        width = max(0.0, min(100.0, (row["tmr_ratio"] - 0.6) / (1.05 - 0.6) * 100.0))
        colour = "var(--bad)" if row["tmr_ratio"] >= 1.0 else (
            "var(--good)" if row["tmr_ratio"] < 0.78 else "var(--warn)")
        add("<tr><td><b>#{}</b></td><td class='n'>{}</td><td class='n'>{}</td>"
            "<td class='n'><b>{:.3f}</b></td><td class='n sub'>{:.3f}</td>"
            "<td class='barcell'><span class='minibar' style='width:{:.1f}%;"
            "background:{}'></span></td></tr>".format(
                key, row["n"], row["uploads"], row["tmr_ratio"], row["pose_ratio"],
                width, colour))
    add("</table></div>")
    add("<p class='sub'>Sorted by the TMR reading. The pose column is kept so the "
        "disagreement stays visible rather than being quietly dropped — it is the "
        "evidence for the retraction above.</p>")

    # ------------------------------------------------------------------ 5 & 6
    add("<h2>5. The samples: keyframes</h2>")
    add(PAIR_INTRO)
    order = lambda k: coherence["per_prototype"].get(str(k), {}).get("tmr_ratio", 9)
    for label in sorted(real_cards, key=order):
        if label in ctrl_cards:
            add(pair_block(label, real_cards[label], ctrl_cards[label],
                           ROOT / "cards_real", ROOT / "cards_ctrl",
                           picks[("cards", "real")], picks[("cards", "ctrl")],
                           coherence, "image/png"))

    add("<h2>6. The samples: motion</h2>")
    add("<p>The sheets above show where a segment <em>settles</em>. The paper's first "
        "criterion for an atomic movement is that it &ldquo;involves clear "
        "processes&rdquo;, and a still frame is what that rules out. Same pairing, as "
        "four-second loops, one member per cell.</p>")
    add("<p class='sub'>Deliberately more than the clustering saw: each segment is "
        "canonicalised <b>once</b>, from its first frame, so rotation and travel reach "
        "the screen. A group that looks inconsistent on turns may still be consistent "
        "in the space it was built in.</p>")
    for label in sorted(real_clips, key=order):
        if label in ctrl_clips:
            add(pair_block(label, real_clips[label], ctrl_clips[label],
                           ROOT / "videos_real", ROOT / "videos_ctrl",
                           picks[("videos", "real")], picks[("videos", "ctrl")],
                           coherence, "video/mp4"))

    # ------------------------------------------------------------------ 7
    add("<h2>7. What this page still cannot tell you</h2>")
    add("<ul>"
        "<li><b>Whether a coherent prototype is a <em>movement</em>.</b> Section 4's "
        "ratio scores a tight group of similar static shapes exactly as well. Only "
        "section 6 and a person answer that.</li>"
        "<li><b>Whether K=53 is right.</b> K came from 14,231 segments &divide; 268.57, "
        "the paper's Tab. 2 density. Nothing here tests another K.</li>"
        "<li><b>How much of the 1.274 is style rather than identity.</b> The corpus has "
        "no style label to stratify on — a property of the corpus, not of the "
        "measurement.</li>"
        "<li><b>Captions or the M3b vocabulary.</b> No LLM tags exist for this tag, "
        "which is why every sheet reads &ldquo;no LLM tag&rdquo;.</li></ul>")

    add("<div class='foot'><p>Labels <code>data/wild3d/clean5b5_labels</code> (1,999 "
        "recordings, all resolved in <code>data/wild3d/wild_v4_performance</code>). "
        "Section 4 covers all 53 prototypes; the sheets sample {} and {} of them by "
        "the renderers' <code>spread</code> rule, which walks the size-ordered list so "
        "the page contains the large, the median and the small in the proportion they "
        "occur. Sheets draw from {:,} segments — those both renderers keep "
        "(<code>label &gt; 0</code>, at least 4 frames). Seed {}. Readings: "
        "<code>runs/clean5b5_prototype_coherence.json</code>, "
        "<code>runs/clean5b5_pose_space_tightness.json</code>.</p></div></div>".format(
            len(real_cards), len(real_clips), tight["segments"], SEED))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("wrote {} ({:.1f} MB)".format(OUT, OUT.stat().st_size / 1e6))
    return 0


def pair_block(label, real, ctrl, real_dir, ctrl_dir, real_picks, ctrl_picks,
               coherence, mime) -> str:
    """One prototype: real on the left, control on the right, both named."""
    # No verdict chip. The two spaces rank the prototypes independently
    # (rho = -0.17), so a single label here would be a claim the data does not
    # support -- which is exactly what the retracted section-4 table did.
    row = coherence["per_prototype"].get(str(label), {})
    if row:
        verdict = ("<span class='verdict'>ratio {:.3f} in TMR &middot; {:.3f} in "
                   "pose</span>".format(row["tmr_ratio"], row["pose_ratio"]))
    else:
        verdict = ""

    out = ["<div class='pair'><div class='pairhd'>",
           "<span class='id'>prototype #{}</span>".format(label),
           "<span class='sub'>{} segments in the full group</span>".format(real["size"]),
           verdict, "</div><div class='grid'>"]
    tag = "video" if mime.startswith("video") else "img"
    for kind, row_, directory, picks in (
            ("real", real, real_dir, real_picks),
            ("ctrl", ctrl, ctrl_dir, ctrl_picks)):
        title = ("真实 M2 prototype #{} · REAL — 5 members of this one cluster"
                 .format(label) if kind == "real" else
                 "随机对照 · RANDOM CONTROL — 5 segments drawn at random")
        out.append("<div class='panel {}'>".format(kind))
        out.append("<div class='who'>{}</div>".format(html.escape(title)))
        if tag == "img":
            out.append("<img alt='prototype {} {}' src='{}'>".format(
                label, kind, uri(directory / row_["path"], mime)))
        else:
            out.append("<video autoplay loop muted playsinline src='{}'></video>".format(
                uri(directory / row_["path"], mime)))
        entries = picks.get(label, [])
        if entries:
            out.append(member_table(entries))
        # ``uploads`` is a property of the WHOLE group, not of the five drawn.
        # Labelling it "among these members" said 36 for five rows and was simply
        # false; the five are one per upload by construction of spread_over_uploads.
        out.append("<div class='who-sub'>full group spans {} uploads; the {} shown "
                   "are one per upload</div>".format(row_["uploads"], len(entries)))
        out.append("</div>")
    out.append("</div></div>")
    return "".join(out)


def member_table(entries) -> str:
    rows = ["<table class='mem'><tr><th>#</th><th>upload</th>"
            "<th class='n'>frames</th><th class='n'>sec</th></tr>"]
    for i, e in enumerate(entries, 1):
        upload = str(e["recording_id"]).replace("wild_v4:", "")
        rows.append(
            "<tr><td><span class='dot' style='background:{}'></span>{}</td>"
            "<td><code>{}</code></td><td class='n'>{}&ndash;{}</td>"
            "<td class='n'>{:.2f}</td></tr>".format(
                MEMBER_COLOURS[(i - 1) % len(MEMBER_COLOURS)], i,
                html.escape(upload), e["start"], e["end"],
                (e["end"] - e["start"]) / 30.0))
    rows.append("</table>")
    return "".join(rows)


PAIR_INTRO = """
<div class='box'>
<p><b>How to read a row.</b> Each row is <u>one</u> prototype id, shown twice as
two independent five-member samples:</p>
<ul>
<li><span class='chip real'>LEFT · 真实 M2</span> five members that M2 actually
put in this cluster, each from a different upload.</li>
<li><span class='chip ctrl'>RIGHT · 随机对照</span> five segments drawn at random
and given the same id. The control keeps the group's exact size and the
different-uploads rule; it destroys only the link between a segment's motion and
its group.</li>
</ul>
<p><b>The two panels are not meant to agree with each other.</b> The left panel
looking internally consistent while the right does not is the good case — that
difference is the whole result. A row where both look equally mixed is the bad
case, and section 4 says which rows those are.</p>
<p class='sub'>Every member is listed under its panel with its upload id and
frame span, so any segment can be traced back. Within a panel, the table's
coloured dot matches the skeleton's colour on the sheet above it.</p>
</div>
"""

STYLE = """<style>
:root{--bg:#fbfbfd;--fg:#1a1a1f;--muted:#5b5b6b;--card:#fff;--line:#e3e3ea;
--good:#1a7f5a;--bad:#a8321f;--warn:#8a6d1a;--accent:#2b5fd9}
@media (prefers-color-scheme:dark){:root:not([data-theme='light']){--bg:#111114;
--fg:#f0f0f4;--muted:#9a9aab;--card:#1a1a20;--line:#2c2c36;--good:#4fd1a0;
--bad:#ff8a70;--warn:#e0bc55;--accent:#8fb0ff}}
:root[data-theme='dark']{--bg:#111114;--fg:#f0f0f4;--muted:#9a9aab;--card:#1a1a20;
--line:#2c2c36;--good:#4fd1a0;--bad:#ff8a70;--warn:#e0bc55;--accent:#8fb0ff}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:2rem 1.25rem;
font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif}
.wrap{max-width:1240px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .3rem}
h2{font-size:1.15rem;margin:2.4rem 0 .6rem;padding-top:1.2rem;
border-top:1px solid var(--line)}
.hd-note{font-weight:400;color:var(--muted)}
p{margin:.55rem 0}
.sub{color:var(--muted)}
.lede{color:var(--muted);margin:0 0 1.4rem;max-width:80ch}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em;
background:color-mix(in srgb,var(--fg) 8%,transparent);padding:.08em .32em;
border-radius:4px}
.box{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:1rem 1.1rem;margin:1rem 0}
.tablewrap{overflow-x:auto;margin:.8rem 0}
table{border-collapse:collapse;width:100%;min-width:520px;font-size:.92rem}
th,td{text-align:left;padding:.42rem .6rem;border-bottom:1px solid var(--line)}
th{font-weight:640;color:var(--muted);font-size:.82rem;text-transform:uppercase;
letter-spacing:.04em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
table.coh{min-width:660px}
.barcell{width:180px;padding-right:0}
.minibar{display:block;height:8px;border-radius:4px}
.good{color:var(--good);font-weight:640}
.bad{color:var(--bad);font-weight:640}
.warn{color:var(--warn);font-weight:640}
.scale{margin:1.1rem 0 .4rem;position:relative;height:64px}
.scale .bar{position:absolute;left:0;right:0;top:26px;height:6px;border-radius:3px;
background:linear-gradient(90deg,var(--good),var(--warn),var(--bad))}
.scale .pin{position:absolute;top:6px;text-align:center;font-size:.78rem;
color:var(--muted);white-space:nowrap;transform:translateX(-50%)}
.scale .pin b{display:block;color:var(--fg);font-size:.9rem;
font-variant-numeric:tabular-nums}
.scale .pin::after{content:'';position:absolute;left:50%;top:100%;width:2px;
height:14px;background:var(--fg);transform:translateX(-50%);opacity:.55}
.scale .pin.first{transform:translateX(0);text-align:left}
.scale .pin.first::after{left:1px;transform:none}
.scale .pin.last{transform:translateX(-100%);text-align:right}
.scale .pin.last::after{left:auto;right:1px;transform:none}
.pair{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:.9rem;margin-bottom:1.1rem}
.pairhd{display:flex;gap:.7rem;align-items:baseline;flex-wrap:wrap;
margin-bottom:.6rem}
.pairhd .id{font-weight:680;font-size:1.02rem}
.verdict{font-size:.78rem;padding:.1rem .5rem;border-radius:999px;
border:1px solid var(--line);color:var(--muted);font-variant-numeric:tabular-nums}
.box.retract{border-left:5px solid var(--bad)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:.9rem}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.panel{border:1px solid var(--line);border-radius:8px;padding:.55rem;
background:var(--bg);overflow-x:auto}
.panel.real{border-left:5px solid var(--good)}
.panel.ctrl{border-left:5px solid var(--bad)}
.panel .who{font-size:.84rem;font-weight:640;margin-bottom:.45rem}
.panel.real .who{color:var(--good)}
.panel.ctrl .who{color:var(--bad)}
.who-sub{font-size:.78rem;color:var(--muted);margin-top:.3rem}
table.mem{min-width:0;font-size:.78rem;margin-top:.5rem}
table.mem th,table.mem td{padding:.22rem .4rem}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;
margin-right:.4rem;vertical-align:middle}
.chip{display:inline-block;font-size:.76rem;font-weight:660;padding:.06rem .45rem;
border-radius:999px;margin-right:.3rem}
.chip.real{color:var(--good);border:1px solid var(--good)}
.chip.ctrl{color:var(--bad);border:1px solid var(--bad)}
img,video{max-width:100%;height:auto;display:block;border-radius:6px}
ul{margin:.5rem 0;padding-left:1.2rem}li{margin:.3rem 0}
.foot{color:var(--muted);font-size:.86rem;margin-top:2.5rem;
border-top:1px solid var(--line);padding-top:1rem}
</style>"""


if __name__ == "__main__":
    raise SystemExit(main())
