#!/usr/bin/env python3
"""M4 inspection as still images: the music, the plan's words, and the poses.

Why this replaces the video sheet
--------------------------------
A pose video shows *that* the generated dance is monotonous; it cannot show
*why*, because the two things that decide it -- which atomic movement the
planner asked for, and what the music was doing at that moment -- are invisible
in a skeleton.  This sheet puts all three on one time axis:

    music     onset envelope + the beat grid the segmentation was cut on
    plan      one coloured bar per planned segment, labelled with the atomic
              movement's own caption text (``subprototype_tags.json``)
    poses     thumbnails sampled at fixed times, under the bar that produced
              them

so "the planner spends 67% of the clip on the transition token" stops being a
number and becomes a grey bar that covers two thirds of the row.

Three arms are drawn for every clip because on this checkpoint they disagree,
and the disagreement is itself the finding: ``vote`` and ``centre`` are the
same model and the same seed, differing only in how ``infer_atomic`` fuses the
overlapping window samples.

Everything is PNG.  The page embeds them as data URIs so it renders in an IDE
preview, which will not play <video>.
"""
import base64
import io
import json
import pathlib
import pickle
import sys

import matplotlib
import matplotlib.font_manager
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The in-figure labels are Chinese; DejaVu has no CJK glyphs and matplotlib
# renders them as empty boxes with only a UserWarning, which is exactly the
# kind of defect that survives review because the file still opens.
# "Droid Sans Fallback" is on the list because this host has it and nothing
# else: its cmap was CHECKED, not assumed -- every glyph of 真值生成基线选择器接缝卡点舞蹈
# is present.  The gate is an allowlist rather than a coverage test because the
# labels are the caller's arm names and are not known at import time; adding a
# font here means someone verified its coverage, not that it sounded right.
for _cjk in ("Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Serif CJK JP",
             "Droid Sans Fallback"):
    if any(f.name == _cjk for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_cjk, "DejaVu Sans"]
        break
else:  # pragma: no cover - depends on the host
    raise SystemExit("no CJK font installed; the figure labels would render as boxes")
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import Rectangle

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "output" / "clean5b5_m4_plan"
LABELS = pathlib.Path("/dev/shm/atomicdance-m3a-segmentation/ingroup_llm_v1")
GT_MOTION = REPO / "runs/wild_v4_acct_gt_eval/motion"
AUDIO = REPO / "runs/wild_v4_acct_gt_eval/audio"
FPS = 30.0

# SMPL kinematic tree, same list as tools/render_dance_video.py.
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
           18, 19, 20, 21]
ARMS = [
    ("真值", "gt", "#2b8a3e"),
    ("planner · argmax", "argmax", "#7048e8"),
    ("planner · 随机采样 centre", "centre", "#1c7ed6"),
    ("planner · 随机采样 vote(仓库钉死)", "vote", "#e8590c"),
]
ARM_DIR = {"argmax": REPO / "runs/vis_clean5b5_argmax",
           "centre": REPO / "runs/vis_clean5b5_centre",
           "vote": REPO / "runs/vis_clean5b5_planner"}


def short(clip):
    return clip.replace("wild_v4:", "").replace(":clip", "_c")


def load_tags():
    return {int(k): v for k, v in json.loads((LABELS / "subprototype_tags.json").read_text()).items()}


def gt_label_paths():
    out = {}
    for line in (LABELS / "labels.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        out[row["sequence_id"]] = row["labels_path"]
    return out


def runs_of(labels):
    """[(start, end, label)] for maximal equal-label runs."""
    labels = np.asarray(labels)
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = [0, *edges.tolist(), len(labels)]
    return [(bounds[i], bounds[i + 1], int(labels[bounds[i]]))
            for i in range(len(bounds) - 1)]


def colour_for(label, palette):
    if label == 0:
        return "#d9d9d9"
    return palette[label % len(palette)]


def ground_relative(pose):
    """Remove the horizontal root only; keep height.

    Subtracting the whole root (x, y, z) centres every frame on the pelvis and
    therefore *deletes vertical motion* -- a jump and a stand render
    identically.  A first draft of this sheet did exactly that and made a class
    captioned "high jump explosive" look like it contained no jumps, which
    would have been a conclusion about the projection, not about the data.
    Horizontal translation is the retrieval draft's business and is still
    removed; height is the movement's own and stays.
    """
    pose = np.asarray(pose, float)
    out = pose.copy()
    out[..., 0] -= pose[..., :1, 0]
    out[..., 1] -= pose[..., :1, 1]
    return out


def draw_skeleton(ax, pose, colour):
    """Front view (x lateral, z up) -- the projection a dance reader expects."""
    x, z = pose[:, 0], pose[:, 2]
    for joint, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        ax.plot([x[joint], x[parent]], [z[joint], z[parent]], color=colour, linewidth=1.3)
    ax.scatter(x, z, s=2.0, color="#c92a2a", zorder=3)
    ax.set_axis_off()
    ax.set_aspect("equal")


def clip_figure(clip, tags, gtp, thumbs=9):
    music = np.load(AUDIO / (clip + ".npy"))
    frames = len(music)
    series = {"gt": np.load(LABELS / gtp[clip])}
    poses = {"gt": np.asarray(pickle.load(open(GT_MOTION / (clip + ".pkl"), "rb"))["full_pose"], float)}
    for key, root in ARM_DIR.items():
        payload = pickle.load(open(root / (clip + ".pkl"), "rb"))
        series[key] = np.asarray(payload["atomic_labels"])
        poses[key] = np.asarray(payload["full_pose"], float)
    frames = min([frames] + [len(v) for v in series.values()])

    palette = plt.get_cmap("tab20").colors
    palette = ["#%02x%02x%02x" % tuple(int(255 * c) for c in rgb) for rgb in palette]

    fig = plt.figure(figsize=(17.5, 12.0), dpi=100)
    gs = fig.add_gridspec(9, 1, height_ratios=[1.0, 0.66, 0.66, 0.66, 0.66, 1.35, 1.35, 1.35, 1.35],
                          hspace=0.42, left=0.055, right=0.995, top=0.945, bottom=0.035)

    # --- music -----------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    t = np.arange(frames) / FPS
    ax.fill_between(t, music[:frames, 0], color="#adb5bd", linewidth=0)
    beats = np.flatnonzero(music[:frames, 34] > 0.5) / FPS
    for b in beats:
        ax.axvline(b, color="#495057", linewidth=0.7, alpha=0.85)
    for b in beats[::4]:
        ax.axvline(b, color="#b00", linewidth=1.5, alpha=0.9)
    ax.set_xlim(0, frames / FPS)
    ax.set_yticks([])
    ax.set_ylabel("音乐", rotation=0, ha="right", va="center", fontsize=10, labelpad=12)
    ax.set_title("{}   —   灰=起音包络,细线=每一拍,红线=每 4 拍(M1 就是按红线切的)".format(short(clip)),
                 fontsize=11, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # --- three plan timelines -------------------------------------------
    for row, (title, key, accent) in enumerate(ARMS, start=1):
        ax = fig.add_subplot(gs[row])
        labels = series[key][:frames]
        segs = runs_of(labels)
        transition = float((labels == 0).mean())
        for start, end, label in segs:
            ax.add_patch(Rectangle((start / FPS, 0), (end - start) / FPS, 1,
                                   facecolor=colour_for(label, palette),
                                   edgecolor="white", linewidth=0.8))
            width_s = (end - start) / FPS
            if label != 0 and width_s > 0.9:
                ax.text((start + end) / 2 / FPS, 0.5, tags.get(label, str(label)),
                        ha="center", va="center", fontsize=7.5, color="#111",
                        clip_on=True)
        ax.set_xlim(0, frames / FPS)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(title.split(" · ")[0], rotation=0, ha="right", va="center",
                      fontsize=10, labelpad=12, color=accent)
        ax.set_title("{} — {} 段 / {} 个不同动作 / transition {:.0%}".format(
            title, len(segs), len({l for _, _, l in segs} - {0}), transition),
            fontsize=9.5, loc="left", color=accent, pad=2)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        if row < 4:
            ax.set_xticks([])

    # --- pose storyboards -------------------------------------------------
    times = np.linspace(0, frames - 1, thumbs).astype(int)
    for row, (title, key, accent) in enumerate(ARMS, start=5):
        ax = fig.add_subplot(gs[row])
        ax.set_xlim(0, frames / FPS)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.text(-0.006, 0.5, title.split(" · ")[0], transform=ax.transAxes, rotation=0,
                ha="right", va="center", fontsize=10, color=accent)
        pose = ground_relative(poses[key])
        span = np.percentile(np.abs(pose - pose[:, :1, :]), 99.0) * 2.2
        floor = float(pose[..., 2].min())
        box = ax.get_position()
        width = box.width / thumbs
        for i, frame in enumerate(times):
            f = min(frame, len(pose) - 1)
            sub = fig.add_axes([box.x0 + i * width, box.y0, width * 0.94, box.height])
            rel = pose[f]
            draw_skeleton(sub, rel, accent)
            sub.set_xlim(-span / 2, span / 2)
            sub.set_ylim(floor - 0.05, floor + span)
            if row == 5:
                sub.set_title("{:.1f}s".format(f / FPS), fontsize=7, color="#666", pad=1)
        ax.remove()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def class_member_index(gtp, limit_sequences=None):
    """Every atomic segment whose recording has a decoded ground-truth motion.

    Only recordings with a ``runs/wild_v4_acct_gt_eval/motion`` pkl can be
    drawn, so the index is built from that intersection (840 of the corpus's
    1,999) rather than from the label manifest alone -- otherwise a class would
    look empty when it is merely undrawable.
    """
    index = {}
    for clip, path in gtp.items():
        if not (GT_MOTION / (clip + ".pkl")).exists():
            continue
        labels = np.load(LABELS / path)
        for start, end, label in runs_of(labels):
            if label == 0 or end - start < 20:
                continue
            index.setdefault(label, []).append((clip, start, end))
        if limit_sequences and len(index) > limit_sequences:
            pass
    return index


def _segment_poses(clip, start, end, k=4, cache={}):
    if clip not in cache:
        if len(cache) > 60:
            cache.clear()
        with open(GT_MOTION / (clip + ".pkl"), "rb") as handle:
            cache[clip] = np.asarray(pickle.load(handle)["full_pose"], float)
    pose = cache[clip]
    idx = np.linspace(start, min(end - 1, len(pose) - 1), k).astype(int)
    return pose[idx]


def class_figure(label, tags, index, rng, k=4, members=5):
    """One class's members beside a same-size random control.

    The control is the panel that makes this readable: a viewer asked "do these
    look like one movement?" will say yes to almost anything, so the question
    has to be "do they look *more alike than these*?".  Same number of rows,
    same number of poses, drawn from the whole corpus.
    """
    pool = index.get(label, [])
    by_upload = {}
    for clip, start, end in pool:
        by_upload.setdefault(clip.split(":")[1], []).append((clip, start, end))
    picks = [v[0] for v in by_upload.values()][:members]
    allsegs = [s for segs in index.values() for s in segs]
    control = [allsegs[i] for i in rng.choice(len(allsegs), size=members, replace=False)]

    rows = len(picks) + len(control)
    fig = plt.figure(figsize=(13.6, 1.25 * rows + 0.9), dpi=100)
    gs = fig.add_gridspec(rows, k, hspace=0.05, wspace=0.02,
                          left=0.14, right=0.995, top=0.88, bottom=0.02)
    fig.suptitle('类 {} — "{}"   ·   上 {} 行是这个类的真实成员(各来自不同 upload),'
                 '下 {} 行是全语料随机对照'.format(label, tags.get(label, label),
                                                len(picks), len(control)),
                 fontsize=11.5, x=0.005, ha="left")
    for row, (segs, colour, tag) in enumerate(
            [(p, "#2b8a3e", "成员") for p in picks] + [(c, "#868e96", "随机") for c in control]):
        clip, start, end = segs
        poses = ground_relative(_segment_poses(clip, start, end, k))
        span = np.percentile(np.abs(poses - poses[:, :1, :]), 99.0) * 2.2
        floor = float(poses[..., 2].min())
        for col in range(k):
            ax = fig.add_subplot(gs[row, col])
            rel = poses[col]
            draw_skeleton(ax, rel, colour)
            ax.set_xlim(-span / 2, span / 2)
            ax.set_ylim(floor - 0.05, floor + span)
            if col == 0:
                ax.text(-0.06, 0.5, "{} · {} {:.1f}s".format(tag, clip.split(":")[1][-6:], start / FPS),
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=8, color=colour)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def caption_geometry_table(labels, tags, index, rng, sample=400):
    """Do the words in a class name correspond to anything measurable?

    Three crude geometric quantities per segment -- vertical range of the root
    (does it leave the floor), mean root height (how low the stance is), and
    root-relative joint speed (how fast) -- reported as a percentile against
    400 random segments of the same corpus.  Crude on purpose: the point is not
    to score the vocabulary but to check whether a specific word ("jump") shows
    up in the one quantity that word is about.  A percentile near 50 means the
    class is indistinguishable from a random segment on that axis; it does not
    mean the class is meaningless, because these three features say nothing
    about, for example, arm height.
    """
    allsegs = [s for segs in index.values() for s in segs]
    def feats(clip, start, end):
        with open(GT_MOTION / (clip + ".pkl"), "rb") as handle:
            pose = np.asarray(pickle.load(handle)["full_pose"], float)[start:end]
        root = pose[:, 0, :]
        return (float(root[:, 2].max() - root[:, 2].min()), float(root[:, 2].mean()),
                float(np.abs(np.diff(pose - pose[:, :1, :], axis=0)).mean()))
    base = np.array([feats(*allsegs[i]) for i in rng.choice(len(allsegs), sample, replace=False)])
    rows = []
    for label in labels:
        mem = index.get(label, [])[:40]
        if not mem:
            continue
        v = np.array([feats(*m) for m in mem]).mean(axis=0)
        pct = [(base[:, i] < v[i]).mean() for i in range(3)]
        rows.append((label, tags.get(label, str(label)), len(mem), v, pct))
    return rows, np.median(base, axis=0)


def plan_table(clip, tags, gtp, frames_cap=None):
    rows = {}
    gt = np.load(LABELS / gtp[clip])
    rows["真值"] = runs_of(gt)
    for key in ("argmax", "centre", "vote"):
        payload = pickle.load(open(ARM_DIR[key] / (clip + ".pkl"), "rb"))
        rows["planner · " + key] = runs_of(np.asarray(payload["atomic_labels"]))
    out = []
    for name, segs in rows.items():
        words = ["{}–{:.1f}s <b>{}</b>".format(
            "{:.1f}".format(s / FPS), e / FPS,
            "过渡" if l == 0 else tags.get(l, str(l))) for s, e, l in segs]
        out.append((name, " · ".join(words)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tags = load_tags()
    gtp = gt_label_paths()
    clips = [c.strip() for c in (REPO / "runs/clean5b5_vis6.txt").read_text().split() if c.strip()]

    html = ['<meta charset="utf-8"><title>M4 计划与姿态抽检</title>', """
<style>
 body{font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
      margin:0 auto;max-width:1360px;padding:22px;color:#111;background:#fafafa}
 h1{font-size:22px;margin:0 0 6px} h2{font-size:17px;margin:30px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}
 h3{font-size:15px;margin:26px 0 6px}
 .note{background:#fff;border:1px solid #e3e3e3;border-left:3px solid #b00;padding:12px 14px;margin:12px 0;border-radius:3px}
 .note.good{border-left-color:#0a7} .note b{color:#b00} .note.good b{color:#076}
 img{width:100%;display:block;border:1px solid #e3e3e3;border-radius:3px;background:#fff}
 table{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin:8px 0}
 th,td{border:1px solid #e3e3e3;padding:6px 8px;text-align:right} th{background:#f2f2f2;text-align:left}
 td.l,th.l{text-align:left} tr.gt td{background:#f4faf5}
 .bad{color:#b00;font-weight:600}
 .plan{font-size:12px;line-height:1.9;background:#fff;border:1px solid #e3e3e3;border-radius:3px;padding:8px 10px}
 .plan b{color:#111} code{background:#f0f0f0;padding:1px 4px;border-radius:2px;font-size:12px}
 p.sub{color:#666;margin:4px 0 10px}
</style>"""]
    html.append("<h1>M4 抽检:音乐 / 计划的词 / 姿态,放在同一条时间轴上</h1>")
    html.append("<p class='sub'>planner <code>planner_v1A_s15/planner_step135648.pt</code> → completion "
                "<code>completion_wild_v4_acct/completion_step134496.pt</code>,seed 20260822,6 条 test clip。"
                "全部是静态图(PNG,内嵌),不需要播放器。</p>")
    html.append("""
<div class="note"><b>三个先说清楚的问题:</b>
<ol style="margin:6px 0 0 18px;padding:0">
<li><b>planner 到底看的是什么“音乐”</b> —— 不是 wav,是每帧 35 维的音乐特征
(1 起音包络 + 20 MFCC + 12 chroma + 1 峰值 one-hot + 1 拍点 one-hot)。
第一行画的就是它:灰色是包络,细竖线是拍,<b>红竖线是每 4 拍</b>——
M1 的分割就是按红线切的。wav 本身在 OSS 的 ingest 树里,这次没拉下来,所以视频没有音轨;
要带声音的视频我可以去拉,但<b>模型从头到尾没听过 wav</b>,它只见过这 35 维。</li>
<li><b>计划的词表现在印出来了</b> —— 每个色块上的字就是那一段被计划的原子动作自己的名字
(<code>subprototype_tags.json</code>,820 个类各有一个,由 M3 的 LLM 归纳)。
灰色无字的块是 <b>transition(过渡)</b>,即“这里没有原子动作”。</li>
<li><b>vote 与 centre 是同一个模型、同一个 seed</b>,只差 <code>infer_atomic</code> 怎么融合重叠窗口。
它们差别这么大,本身就是结论的一部分。</li>
</ol></div>""")

    html.append("""
<div class="note good"><b>怎么用这一页判断“问题在哪”——看三件事:</b>
<ol style="margin:6px 0 0 18px;padding:0">
<li><b>灰色占了多少。</b>真值大约三分之一是过渡;planner 那两行是一半到三分之二。
灰色就是“没有动作被计划”,姿态上表现为站着不动。</li>
<li><b>色块换不换、换成什么。</b>真值一条 clip 用 5–8 个不同动作,名字是
“arms swinging legs stepping”“low-level poses”这种有区分度的;
planner 常常整条 clip 只用 1–2 个,而且反复是同一个。</li>
<li><b>色块的边界跟不跟红线。</b>真值的边界基本落在红线(每 4 拍)上——因为 M1 就是这么切的;
planner 的边界与红线没有对应关系,<b>这就是“没学会按音乐 plan”的直接画面</b>。</li>
</ol></div>""")

    for clip in clips:
        png = clip_figure(clip, tags, gtp)
        name = short(clip) + ".png"
        (OUT / name).write_bytes(png)
        html.append("<h3>{}</h3>".format(short(clip)))
        html.append('<img alt="{}" src="data:image/png;base64,{}">'.format(
            short(clip), base64.b64encode(png).decode("ascii")))
        html.append("<div class='plan'>")
        for label, text in plan_table(clip, tags, gtp):
            html.append("<div><b>{}</b> — {}</div>".format(label, text))
        html.append("</div>")

    html.append("<h2>2. 一个“原子动作”类里到底装了什么</h2>")
    html.append("<p class='sub'>上面每个色块都写着一个动作名。这一节问的是那个名字背不背得起来:"
                "<b>把同一个类的真实成员排在一起,它们是不是同一个动作?</b>"
                "每行一个成员(取自不同 upload,避免近重复),行内四张是这一段的起→止。"
                "<b>下半部分是同样张数的全语料随机对照</b> —— 没有对照的话,任何一组人体姿态看起来都像“差不多”。</p>")
    index = class_member_index(gtp)
    rng = np.random.RandomState(20260822)
    used = []
    for clip in clips:
        for _s, _e, label in runs_of(np.load(LABELS / gtp[clip])):
            if label and label not in used and len({c.split(":")[1] for c, _, _ in index.get(label, [])}) >= 5:
                used.append(label)
    for label in used[:4]:
        png = class_figure(label, tags, index, rng)
        (OUT / "class_{}.png".format(label)).write_bytes(png)
        html.append('<img alt="class {}" src="data:image/png;base64,{}">'.format(
            label, base64.b64encode(png).decode("ascii")))
    rows, med = caption_geometry_table(used[:4], tags, index, np.random.RandomState(7))
    html.append("<p class='sub'>肉眼看“像不像同一个动作”很容易自我说服,所以再补一条能失败的:"
                "<b>类名里的词,对不对得上它该对应的那个几何量。</b>括号是在全语料 400 个随机段里的百分位,"
                "<b>50% 附近 = 与随机段没有区别</b>。</p>")
    html.append("<table><tr><th class='l'>类</th><th class='l'>类名</th><th>成员</th>"
                "<th>根节点垂直幅度<br><span style='font-weight:400;color:#888'>“跳”该看这个</span></th>"
                "<th>根节点高度<br><span style='font-weight:400;color:#888'>“低位”该看这个</span></th>"
                "<th>关节速度<br><span style='font-weight:400;color:#888'>“爆发”该看这个</span></th></tr>")
    for label, tag, n, v, pct in rows:
        cells = []
        for i in range(3):
            mark = ' class="bad"' if 0.35 < pct[i] < 0.65 else ""
            cells.append("<td{}>{:.3f} ({:.0f}%)</td>".format(mark, v[i], 100 * pct[i]))
        html.append("<tr><td class='l'>{}</td><td class='l'>{}</td><td>{}</td>{}</tr>".format(
            label, tag, n, "".join(cells)))
    html.append("<tr class='gt'><td class='l' colspan='3'>全语料随机段中位</td>"
                "<td>{:.3f}</td><td>{:.3f}</td><td>{:.4f}</td></tr></table>".format(*med))
    html.append("""
<div class="note"><b>读法:</b>红色 = 与随机段无区别。
名字里带 <b>jump</b> 的两个类,垂直幅度在 59%/64% 百分位 —— 也就是<b>“跳”这个词没有对应的离地</b>;
而 “high jump explosive” 的<b>速度</b>在 91% 百分位,<b>“explosive”是真的</b>。
所以这些类不是空的,它们抓住了“快/慢”,但没抓住人读到那个名字时以为的那件事。
这与全语料的 caption 接地读数同向:<code>body_action=jump</code> AUC 0.62、
<code>dynamics=explosive</code> 0.69(随机 0.50)。</div>""")
    html.append("""
<div class="note"><b>这一节对应的数字是已经量过的:</b>子原型的组内相干性
(中立的 TMR 空间,跨 upload 配对,分母是它所属的 M2 母组)读 <b>0.9870</b>,
而“把母组随机切成同样大小”的地板是 <b>1.0002</b>、同形状 k-means 的天花板是 <b>0.8791</b> ——
也就是说 <b>LLM 的分组只买到可得区间的 11%</b>。
上面这些图是那个 11% 的样子:<b>成员之间确实共享一个粗糙的共性(手臂位置、快慢),
但腿、重心、朝向各不相同,与随机对照的差距肉眼可见但很小。</b>
下游因此拿不到一个干净的“这个类=这个动作”的映射:同一个类检索出来的草稿彼此不像,
completion 只能把它们平均,平均的结果就是一个站姿。</div>""")

    target = OUT / "index.html"
    target.write_text("\n".join(html), encoding="utf-8")
    print("wrote", target, "({:.1f} MB)".format(target.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
