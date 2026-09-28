#!/usr/bin/env python3
"""Does putting the plan on the music's own bar grid admit more choreography?

What this page is for
---------------------
``tools/render_m4_plan_sheet.py`` showed that the shipped plan spends most of a
clip on the transition token.  This page asks the next question: the boundaries
that plan is trying to predict are **already a closed-form function of its own
input**.  M1 has cut on the music's beat grid since 2026-08-20
(``tools/segment_on_music_beats.py``, every 4 beats), and that grid is channel
34 of the 35-D music feature the planner is conditioned on.  Measured over the
65 M6 clips: **97.3%** of ground-truth boundaries sit on a single 4-beat phase
of that grid against a 3.6% uniform-random control, while the planner's own
boundaries reach **30.7%**.

So the third row here keeps the planner's per-frame opinions and replaces only
the *segmentation*, with no retraining:

    the bar takes the majority non-transition class among the planner's frames
    inside it; it is transition only when the planner named no atomic movement
    anywhere in the bar.

That rule has no free parameter, but it is still **a rule this repository
invented**, and it is a *calibration* -- it decides how much of the vocabulary
is admitted, not whether the classes are right.  Over the 65 clips it moves the
atomic-frame share 0.4737 -> 0.6288 against a ground truth of 0.6786, i.e. a
third more retrieved motion enters, on bars that are 57.0 frames long against
the ground truth's 62.9 and 100% aligned to the grid.

``--arm NAME=DIR`` draws generated motion from a run directory (each clip a
``.pkl`` carrying ``full_pose`` and ``atomic_labels``), so the plan drawn and
the poses drawn come from the same artifact rather than being paired by hand.
With no ``--arm`` the page falls back to the retrieval draft, which is what it
had to draw while M5 was still being trained on this line.

Two things this page cannot show, stated so they are not read into it
--------------------------------------------------------------------
* **A better-shaped plan is not a better plan.**  The class decision is
  untouched, and its ceiling is low: two dancers on one song agree on 4.81% of
  atomic frames.
* **The videos carry the clip's own music**, taken from the ingest tree
  (``data/wild_ingest_v1/<upload>__<clip>/audio.wav``) and muxed by
  ``tools/render_dance_video.py``.  The first version of this page said this
  corpus had no playable audio; that was wrong, and it was wrong because the
  directory the *models* read (``runs/..._gt_eval/audio``) holds the 35-D
  feature arrays rather than sound.

  The second version then claimed the wav and the motion agree to "under one
  frame on both", citing 15.743 s against 472 frames and 22.338 s against 669.
  Both readings are real and both are useless as evidence: they are two of the
  four clips whose ratio is exactly 1.000, so the check was run only where it
  could not fail.  Measured across all six drawn clips on 2026-08-24, **two
  carry a wav a quarter shorter than the motion** -- 13.65 s against 18.17 s
  and 13.38 s against 17.77 s.  The page now reads each clip's own
  ``.render.json`` and names the silent tails instead of asserting alignment,
  because judging whether a dance is on the beat over a silent tail is exactly
  the mistake the sound was added to prevent.

* **All the arms of one clip are drawn into ONE video**, not stacked from
  separately-rendered files.  ``render_dance_video`` derives its axis cube from
  the motion it is handed, so N separate renders are N different
  metres-per-pixel, and since the axes are drawn off there is nothing in the
  frame to say so -- the arm that travels furthest gets the biggest cube, is
  drawn smallest, and its drift reads as the calmest.  Drawn together the
  panels share one half-extent (the largest any of them needs) and each is
  centred on its own motion, so **amplitude** is comparable panel to panel
  while **absolute floor position** is not.

The bottom row is the root's path seen from above, on a shared scale.  It is
there because the rest of the page cannot show it: the pose strips are drawn
root-relative, which deletes exactly the defect measured on 2026-08-23 --
inside a segment the completion's root moves **442%** of the ground truth's
while the draft it was handed moves 103%, and at a plan cut it moves ~15x.
The skeletons look like dancing; the trajectory is what shows the sliding.

Usage::

    python3 tools/render_plan_grid_sheet.py --clips 6 \\
        --arm "vote=runs/gapfill_zero" --arm "bias+grid=runs/biasgrid_test"
"""
import argparse
import base64
import io
import json
import pathlib
import pickle
import re
import subprocess
import sys

import matplotlib
import matplotlib.font_manager
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

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

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset.atomic import labels_to_segments  # noqa: E402
from infer_atomic import (IndexedAtomicMotionLibrary, _load_music,  # noqa: E402
                          _query_retrieval_group_id, _source_safe_draft,
                          decode_motion, infer_plan)
from tools.eval_planner_checkpoint import load_planner  # noqa: E402
from tools.render_m4_plan_sheet import (colour_for, draw_skeleton,  # noqa: E402
                                        ground_relative, runs_of, short)
from tools.segment_on_music_beats import choose_phase  # noqa: E402

OUT = REPO / "output" / "clean5b5_plan_grid"
LABELS = pathlib.Path("/dev/shm/atomicdance-m3a-segmentation/ingroup_llm_v1")
RELEASE = "/dev/shm/atomicdance-m3a-segmentation/clean5b5_windows_v1A_s15"
CKPT = "/dev/shm/atomicdance-m3a-segmentation/planner_v1A_s15/planner_step135648.pt"
GT_MOTION = REPO / "runs/wild_v4_acct_gt_eval/motion"
AUDIO = REPO / "runs/wild_v4_acct_gt_eval/audio"
FPS = 30.0
K = 4

# (row title, key, colour, short label for the left margin -- the long titles
# all begin "planner", so splitting them would label two different rows the same)
GT_ROW = ("真值", "gt", "#2b8a3e", "真值")
# One accent per arm, in the order the --arm flags were given.  The page was
# fixed at exactly two arms until 2026-08-24, and that cost something concrete:
# a third configuration could only be compared by opening an earlier render, so
# the velocity term's effect on the root -- the one axis this page exists to
# show -- was never on a page beside the arm it had to be compared against.
ARM_ACCENTS = ("#e8590c", "#1c7ed6", "#862e9c", "#0b7285", "#5c940d")
# The rows used when no --arm is given: the poses are then the retrieval draft
# and the two rows differ only in how the plan was segmented.
DRAFT_ARMS = (("planner · vote(仓库钉死)", "vote", "vote"),
              ("planner · 压上小节网格", "grid", "小节网格"))


def margin_tag(name):
    """The left-margin label for an arm row.

    The margin is narrow and the row titles are not: an arm called
    ``vel4(网格+CFG+速度项4.0)`` runs off the left edge of the figure and is drawn
    over the plot area.  Everything up to the first bracket labels the row
    (``vel4``) while the row's own title, drawn inside the axes where there is
    room, keeps the whole name.  A name with no bracket is used whole rather
    than truncated -- truncating is how two different rows end up labelled the
    same, which is what the hand-written short tags existed to prevent.
    """
    head = re.split(r"[(（]", name, maxsplit=1)[0].strip()
    return head or name


def build_rows(arms):
    """[(title, key, accent, short tag)] -- ground truth, then one row per arm."""
    if len(arms) > len(ARM_ACCENTS):
        raise SystemExit("this page has {} accents, got {} arm(s)".format(
            len(ARM_ACCENTS), len(arms)))
    return [GT_ROW] + [(title, key, ARM_ACCENTS[i], tag)
                       for i, (title, key, tag) in enumerate(arms)]


def _root_title_size(rows):
    """Four root panels on one row need a smaller title than three do."""
    return 9 if len(rows) <= 3 else 7.5


POSE_NOTE_DRAFT = (
    "<p><b>姿态画的是检索出来的 draft,不是生成动作。</b>写这一版时 completion 还是从另一份 "
    "release 借来的,它只保留 draft 45% 的能量,画它的输出等于画那个借来的模型。"
    "灰色骨架 = 该帧计划判为 transition,draft 在那里是空的。</p>")

def pose_note_generated(arms):
    """The note above the strips, naming the checkpoints this page actually drew.

    Read out of each arm's own ``manifest.json`` rather than typed in.  The
    typed version of this paragraph named ``completion_clean5b5_v1A_s15`` on a
    page that by then was drawing two other checkpoints, and nothing could catch
    that because the sentence and the artifact had no link between them.
    """
    items = []
    for title, directory in arms:
        manifest = pathlib.Path(directory) / "manifest.json"
        if not manifest.is_file():
            items.append("<li><b>{}</b> — <code>{}</code> 里没有 manifest.json,"
                         "checkpoint 无从查证</li>".format(title, directory))
            continue
        flat = {}

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, dict):
                        walk(value)
                    else:
                        flat.setdefault(key, value)

        walk(json.loads(manifest.read_text(encoding="utf-8")))
        items.append(
            "<li><b>{}</b> — M5 <code>{}</code> · planner <code>{}</code> "
            "@ guidance {} · 小节网格{}</li>".format(
                title, flat.get("completion_checkpoint", "?"),
                flat.get("planner_checkpoint", "?"),
                flat.get("planner_guidance_weight", "?"),
                "开" if flat.get("plan_bar_grid") else "关"))
    return ("<p><b>姿态是真实生成动作</b>,每条 clip 的计划与姿态取自同一份产物。"
            "下面这份 checkpoint 清单读自每条臂自己的 <code>manifest.json</code>,"
            "不是手写的:</p><ul>" + "".join(items) + "</ul>"
            "<p><b>最后一行的根轨迹是本页唯一能看见根的地方。</b>姿态条带是去根画的,"
            "而去根恰好删掉了 2026-08-23 量到的那处缺陷:段内 completion 的根走了真值的 "
            "<b>442%</b>,它拿到的 draft 只走 103%,切点处约 15 倍。"
            "(那组数是在<b>还没有速度项</b>的本线 M5 上量的;本页各臂自己的根路径见下表。)"
            "骨架看着像在跳舞,轨迹才看得出在满地滑。</p>")


def _path_length(track):
    """Total horizontal distance the root travelled, in metres."""
    track = np.asarray(track, float)
    if len(track) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(track, axis=0), axis=-1).sum())


def root_path_summary(rows, arms, clips):
    """Root path over every clip the arms share, not only the ones drawn below.

    Same instrument as the figure's bottom row (``_path_length`` on joint 0's
    xy), so the table and the picture cannot disagree.  Every sequence of one
    clip is truncated to the shortest of {ground truth, each arm}: without that
    an arm that generated more frames is charged for a longer path it never
    danced differently, which is a property of the generation length, not of the
    root.
    """
    per_arm = {key: [] for _, key, _, _ in rows}
    used = 0
    for clip in clips:
        gt_pkl = GT_MOTION / (clip + ".pkl")
        if not gt_pkl.is_file():
            continue
        if any(not (pathlib.Path(d) / (clip + ".pkl")).is_file() for _, d in arms):
            continue
        tracks = {"gt": np.asarray(pickle.load(gt_pkl.open("rb"))["full_pose"], float)}
        for (_, key, _, _), (_, directory) in zip(rows[1:], arms):
            payload = pickle.load((pathlib.Path(directory) / (clip + ".pkl")).open("rb"))
            tracks[key] = np.asarray(payload["full_pose"], float)
        frames = min(len(v) for v in tracks.values())
        if frames < 2:
            continue
        for key, pose in tracks.items():
            per_arm[key].append(_path_length(pose[:frames, 0, :2]))
        used += 1
    if not used:
        return 0, []
    truth = np.asarray(per_arm["gt"], float)
    table = []
    for _, key, _, tag in rows:
        lengths = np.asarray(per_arm[key], float)
        ratio = 100.0 * lengths / np.where(truth > 1e-9, truth, np.nan)
        table.append((tag, float(np.median(lengths)),
                      float(np.nanmedian(ratio)), float(np.nanpercentile(ratio, 90))))
    return used, table


def delta_note(table, rows):
    """The "improvement is not uniform" paragraph, computed from the table below it.

    The typed version of this paragraph quoted 57%→24% / 63%→25% / 23%→23%:
    numbers from a two-arm page whose columns were ``vote`` and ``grid``.  Once
    the page could take three arms those columns stopped existing, and the
    sentence went on describing a comparison the table under it did not contain.
    So it is derived from ``table`` -- first arm against last -- and says which
    clip each number came from.
    """
    first, last = rows[1], rows[-1]
    ranked = sorted(((s[first[1]]["transition"] - s[last[1]]["transition"], clip, s)
                     for clip, _, s in table), reverse=True)
    if not ranked:
        return ""

    def cell(entry, row):
        return "{:.0%}".format(entry[2][row[1]]["transition"])

    top, bottom = ranked[0], ranked[-1]
    # Two different reasons a clip can barely move, and they are opposite:
    # it was already right, or the planner named nothing for the grid to promote.
    landed = abs(bottom[2][last[1]]["transition"] - bottom[2]["gt"]["transition"])
    verdict = ("它本来就落在真值附近,规则在这里近乎 no-op"
               if landed < 0.10 else
               "它离真值还有 {:.0f} 个点 —— planner 在多数小节里一个动作都没点名,"
               "小节网格按构造救不了它,<b>网格能修的是「切在哪里」,"
               "修不了「有没有东西可放」</b>".format(100 * landed))
    return ("<p><b>提升不是均匀的,别把这几条读成一个平均值。</b>"
            "下表逐条给出每条臂的 transition 占比。从 <b>{}</b> 到 <b>{}</b>,"
            "落差最大的是 {} 的 {} → {},最小的是 {} 的 {} → {},而{}。</p>".format(
                first[0], last[0],
                short(top[1]), cell(top, first), cell(top, last),
                short(bottom[1]), cell(bottom, first), cell(bottom, last), verdict))


PANEL_SLOTS = ("左上", "右上", "左下", "右下", "左三", "右三")


def panel_caption(sources):
    """Which panel is which arm, in the order they were handed to the renderer.

    Derived from ``sources`` rather than typed out, because the panel order IS
    the ``--result`` order: a caption that restates it by hand is a second copy
    that can drift from the first, which is how this page came to name a
    checkpoint it had stopped drawing.
    """
    names = [label for label, _ in sources]
    if len(names) == 1:
        return names[0]
    slots = ("左", "右") if len(names) == 2 else PANEL_SLOTS
    mapped = " / ".join("{} {}".format(slots[i], name)
                        for i, name in enumerate(names))
    # The count is derived, and the footage is excluded from the scale claim by
    # name: it is a camera image with no metres in it, so "same metres per
    # pixel" is true of the skeleton panels and false of that one.  The typed
    # version said "四格" on a page that had just stopped having four.
    skeletons = [n for n in names if n != "原视频"]
    scale = ("{} 个骨架格的<b>米/像素相同</b>(共用半径,每格按自己的动作居中),"
             "所以<b>幅度</b>可以逐格直接比,<b>绝对位置</b>不能".format(len(skeletons)))
    if len(skeletons) != len(names):
        scale += "；<b>原视频那一格是镜头画面,不在这套尺度里</b>,它给的是节奏和长度"
    return mapped + " —— 一条时间轴、一条音轨。" + scale + "。"


def audio_coverage_note(videos):
    """What the wav does and does not cover, per clip, read from the sidecars.

    The module docstring used to assert this alignment was "under one frame on
    both" and cited two clips -- which happen to be two of the four where the
    ratio is 1.000.  The check was run only where it could not fail.  The other
    clips are read here instead of asserted: on this corpus two of six carry a
    wav a quarter shorter than the motion, so their tails are silent, and the
    page's whole reason for having sound is judging whether the dance is on the
    beat.  A reader must not judge a silent tail.
    """
    short_ones = []
    for clip, target in videos:
        sidecar = pathlib.Path(str(target)).with_suffix(".render.json")
        if not sidecar.is_file():
            continue
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        tail = meta.get("silent_tail_seconds")
        if tail and tail > 1.0 / 30.0:
            short_ones.append((short(clip), tail, meta.get("motion_seconds")))
    if not short_ones:
        return ("<p>每条 clip 的音轨都覆盖了它自己的全长(逐条读自各自的 "
                "<code>.render.json</code>),所以音画同步可以整段判。</p>")
    rows = "、".join("{} 的最后 {:.2f} 秒(全长 {:.2f} 秒)".format(name, tail, total)
                     for name, tail, total in short_ones)
    return ("<p><b>有 {} 条 clip 的音轨比动作短,尾巴是静音的:</b>{}。"
            "ingest 树里那份 wav 就是这么长,视频没有被裁掉 —— 是音频补了静音。"
            "这一段不能用来判音画同步。逐条读自各自的 <code>.render.json</code> 的 "
            "<code>silent_tail_seconds</code>,不是写死的。</p>".format(
                len(short_ones), rows))


def video_src(target, embed):
    """A relative filename, or the whole mp4 inlined as a ``data:`` URI.

    The figures on this page have always been base64, so the videos were its
    only relative references -- and in the VS Code HTML preview
    (``george-alisson.html-preview-vscode``) they are the only thing that does
    not play.  That extension injects ``<base href="vscode-resource:...">``,
    the webview scheme VS Code removed in 1.47, so a relative ``src`` resolves
    to an address the webview cannot fetch; its own CSP still allows
    ``media-src ... data:``.  Inlining therefore makes the page play there with
    no server and no port forwarding, at about 4/3 of the mp4 bytes in page
    size.  Off by default: a page opened in a real browser should not carry
    30 MB it does not need.
    """
    if not embed:
        return target.name
    return "data:video/mp4;base64," + base64.b64encode(
        target.read_bytes()).decode("ascii")


def common_frames(music, series):
    """The frames every row of one clip shares.

    The figure has truncated to this since it was written; the table under it
    did not, so the two printed different transition shares for the same clip --
    84% in the table against 78% in the picture on
    ``7438547996335295781_c001``, found 2026-08-24.  One number per clip, taken
    from one place.
    """
    return min([len(music)] + [len(v) for v in series.values()])


def bar_bounds(music, total):
    beats = np.flatnonzero(music[:, 34] > 0.5)
    if len(beats) < K + 2:
        return None, None, None
    phase = choose_phase(beats, music[:, 0], K)["phase"]
    cuts = [c for c in (int(b) for b in beats[phase::K]) if 0 < c < total]
    return [0] + cuts + [total], phase, beats


def snap_to_bars(plan, bounds):
    """One decision per bar; transition only when no class was named in it."""
    out = plan.clone()
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = plan[a:b]
        if not len(seg):
            continue
        nonzero = seg[seg != 0]
        if len(nonzero):
            values, counts = torch.unique(nonzero, return_counts=True)
            out[a:b] = values[counts.argmax()]
        else:
            out[a:b] = 0
    return out


def stats(labels):
    segs = [s for s in labels_to_segments(labels) if s.label]
    return dict(segments=len(segs),
                classes=len({int(s.label) for s in segs}),
                transition=float((labels == 0).float().mean()),
                atomic=float((labels != 0).float().mean()),
                median=float(np.median([s.length for s in segs])) if segs else 0.0)


INGEST = REPO / "data/wild_ingest_v1"


def source_audio(clip):
    """The clip's own wav, or None.

    ``wild_v4:<upload>:clip000`` in the release is ``<upload>__clip000`` in the
    ingest tree.  Returns None rather than guessing when it is absent, so a
    corpus without audio still renders -- silently, and the page says which.
    """
    parts = clip.split(":")
    if len(parts) != 3:
        return None
    candidate = INGEST / "{}__{}".format(parts[1], parts[2]) / "audio.wav"
    return candidate if candidate.is_file() else None


FOOTAGE_CACHE = pathlib.Path("/cache/atomicdance-assets/data/wild_ingest_v1")


def source_footage(clip, fetch=True):
    """The clip's own video: the ingest tree's copy, or pulled from OSS.

    ``clip.mp4`` is evicted for most of this corpus -- 41 of the 65 M6 clips --
    so without a fetch the panel exists for the clips that happen to be local,
    which is a selection nobody chose.  Pulled **one object at a time**: this
    page draws six clips and ``data/wild_ingest_v1`` is thousands, so
    ``oss_assets.pull``, which materialises whole trees, is the wrong call.

    It is written under ``/cache/atomicdance-assets/data/wild_ingest_v1``, which
    is where ``data/wild_ingest_v1`` already points: that path is a symlink into
    local CPFS (same inode, checked), the arrangement CLAUDE.md 1.2 prescribes so
    corpora do not land on the quota-limited NAS.  So the pulled object appears
    at the repo path too, and the local branch above finds it on the next run --
    no NAS bytes are spent either way.  Verified on the first pull: 589 frames /
    19.633 s against that clip's own ``meta.json`` ``num_frames: 589``.

    Returns None rather than raising when the object is not there or the pull
    fails, and the page names which clips got no panel -- because "no footage"
    and "footage that agrees" must not look the same.
    """
    parts = clip.split(":")
    if len(parts) != 3:
        return None
    stem = "{}__{}".format(parts[1], parts[2])
    local = INGEST / stem / "clip.mp4"
    if local.is_file():
        return local
    cached = FOOTAGE_CACHE / stem / "clip.mp4"
    if cached.is_file():
        return cached
    if not fetch:
        return None
    from tools.oss_assets import fetch_object
    key = "AtomicDance/data/wild_ingest_v1/{}/clip.mp4".format(stem)
    try:
        return cached if fetch_object(key, cached) else None
    except Exception as error:  # pragma: no cover - depends on the store
        print("  footage pull failed for {}: {}".format(stem, error), flush=True)
        return None


def footage_note(videos):
    """What the original footage says about each drawn clip, read from sidecars.

    The reading that matters is ``footage_vs_motion_ratio``: the sequence is
    declared at 30 fps, so if its own source video is a quarter shorter then the
    two disagree about how long the dance is, and every panel on this page is a
    claim about a sequence whose clock does not match its footage.  Read per
    clip from ``.render.json`` rather than asserted, for the same reason the
    audio note is.
    """
    have, missing, off = [], [], []
    for clip, target in videos:
        sidecar = pathlib.Path(str(target)).with_suffix(".render.json")
        if not sidecar.is_file():
            continue
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if not meta.get("footage"):
            missing.append(short(clip))
            continue
        ratio = meta.get("footage_vs_motion_ratio")
        have.append(short(clip))
        if ratio is not None and abs(ratio - 1.0) > 1.0 / 30.0:
            off.append((short(clip), ratio, meta.get("footage_seconds"),
                        meta.get("motion_seconds")))
    if not have and not missing:
        return ""
    text = ("<p><b>第一格是这条 clip 自己的原视频</b>(<code>data/wild_ingest_v1/"
            "&lt;clip&gt;/clip.mp4</code>),按<b>时间戳</b>重采样到输出帧率 —— 不是逐帧对拷。"
            "所以如果一条序列声称的帧率与它的素材对不上,两个舞者会当场越走越开,"
            "而不是被重采样成一致。骨架格是世界坐标、原视频是手机镜头,"
            "<b>朝向不可比,节奏可比</b>。</p>")
    if missing:
        text += ("<p>有 {} 条 clip 拿不到原片,它们没有这一格:{}。"
                 "本语料的原片按 §1.2 清到了 OSS,本页会<b>按需逐个对象拉回</b>到 "
                 "<code>/cache</code>(不落回 NAS,那里有目录配额);拉不到才会缺。"
                 "<b>没有这一格 ≠ 对得上</b>。</p>".format(len(missing), "、".join(missing)))
    if off:
        rows = "、".join("{} 的素材 {:.2f}s 对序列 {:.2f}s(比值 {:.3f})".format(
            n, f, m, r) for n, r, f, m in off)
        text += ("<p><b>有 {} 条 clip 的原视频与序列长度对不上:</b>{}。"
                 "这不是「片子短了」——序列是按 30 fps 解释的,素材短了四分之一"
                 "意味着这条序列的时钟和它自己的素材不是一个。"
                 "逐条读自各自 <code>.render.json</code> 的 "
                 "<code>footage_vs_motion_ratio</code>。</p>".format(len(off), rows))
    elif have:
        text += "<p>画出来的这几条里,原视频与序列长度逐条一致。</p>"
    return text


def _slug(text):
    """A filename that survives Chinese arm names and shell quoting."""
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text).strip("_")
    return cleaned or "arm"


def clip_figure(clip, tags, series, poses, music, bounds, rows, thumbs=9,
                drafts=True):
    frames = common_frames(music, series)
    palette = ["#%02x%02x%02x" % tuple(int(255 * c) for c in rgb)
               for rgb in plt.get_cmap("tab20").colors]
    # One strip row and one pose row per entry in ``rows`` (ground truth is the
    # first entry), plus the music on top and the root's path at the bottom.
    # The figure grows with the arm count instead of squeezing them, because the
    # skeleton thumbnails stop being readable below about 1.35 units of height.
    height_ratios = [1.0] + [0.70] * len(rows) + [1.35] * len(rows) + [1.9]
    fig = plt.figure(figsize=(17.5, 1.4365 * sum(height_ratios)), dpi=100)
    gs = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios,
                          hspace=0.44, left=0.075, right=0.995, top=0.95, bottom=0.03)

    ax = fig.add_subplot(gs[0])
    t = np.arange(frames) / FPS
    ax.fill_between(t, music[:frames, 0], color="#adb5bd", linewidth=0)
    for b in np.flatnonzero(music[:frames, 34] > 0.5) / FPS:
        ax.axvline(b, color="#495057", linewidth=0.7, alpha=0.8)
    for c in bounds[1:-1]:
        if c < frames:
            ax.axvline(c / FPS, color="#b00", linewidth=1.6, alpha=0.95)
    ax.set_xlim(0, frames / FPS)
    ax.set_yticks([])
    ax.set_ylabel("音乐", rotation=0, ha="right", va="center", fontsize=10, labelpad=12)
    ax.set_title("{}   —   灰=起音包络,细线=每一拍,红线=M1 的小节切点(每 4 拍,相位由 "
                 "choose_phase 定)".format(short(clip)), fontsize=11, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    for row, (title, key, accent, tag) in enumerate(rows, start=1):
        ax = fig.add_subplot(gs[row])
        labels = series[key][:frames]
        for start, end, label in runs_of(labels.numpy()):
            ax.add_patch(Rectangle((start / FPS, 0), (end - start) / FPS, 1,
                                   facecolor=colour_for(label, palette),
                                   edgecolor="white", linewidth=0.8))
            if label != 0 and (end - start) / FPS > 0.9:
                ax.text((start + end) / 2 / FPS, 0.5, tags.get(label, str(label)),
                        ha="center", va="center", fontsize=7.5, color="#111", clip_on=True)
        s = stats(labels)
        ax.set_xlim(0, frames / FPS)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(tag, rotation=0, ha="right", va="center",
                      fontsize=10, labelpad=12, color=accent)
        ax.set_title("{} — {} 段 / {} 个不同动作 / 灰色 transition {:.0%} / 段长中位 {:.0f} 帧".format(
            title, s["segments"], s["classes"], s["transition"], s["median"]),
            fontsize=9.5, loc="left", color=accent, pad=2)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        if row < len(rows):
            ax.set_xticks([])

    times = np.linspace(0, frames - 1, thumbs).astype(int)
    first_pose_row = 1 + len(rows)
    for row, (title, key, accent, tag) in enumerate(rows, start=first_pose_row):
        ax = fig.add_subplot(gs[row])
        ax.set_xlim(0, frames / FPS)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        caption = tag + ("" if key == "gt" or not drafts else "\n的 draft")
        pose = ground_relative(poses[key])
        span = np.percentile(np.abs(pose - pose[:, :1, :]), 99.0) * 2.2
        floor = float(pose[..., 2].min())
        box = ax.get_position()
        # Written on the figure, not on ``ax``: the storyboard replaces ``ax``
        # with its own sub-axes and removes it, which takes an axes-relative
        # label with it.
        fig.text(box.x0 - 0.008, box.y0 + box.height / 2, caption, ha="right",
                 va="center", fontsize=9.5, color=accent)
        width = box.width / thumbs
        for i, frame in enumerate(times):
            f = min(frame, len(pose) - 1)
            sub = fig.add_axes([box.x0 + i * width, box.y0, width * 0.94, box.height])
            # A frame the plan calls transition has no retrieved prototype at
            # all: the draft is literally zero there, and zero decodes to one
            # fixed pose that looks like a dancer collapsed on the floor.
            # Drawing that skeleton would invent a posture the plan never
            # asked for, so the slot is drawn empty and labelled.
            empty = (drafts and key != "gt"
                     and int(series[key][min(f, len(series[key]) - 1)]) == 0)
            if empty:
                sub.add_patch(Rectangle((0, 0), 1, 1, transform=sub.transAxes,
                                        facecolor="#f1f3f5", edgecolor="#dee2e6",
                                        linewidth=0.8))
                sub.text(0.5, 0.5, "无\n原型", transform=sub.transAxes, ha="center",
                         va="center", fontsize=7.5, color="#adb5bd")
                sub.set_axis_off()
            else:
                draw_skeleton(sub, pose[f], accent)
                sub.set_xlim(-span / 2, span / 2)
                sub.set_ylim(floor - 0.05, floor + span)
            if row == first_pose_row:
                sub.set_title("{:.1f}s".format(f / FPS), fontsize=7, color="#666", pad=1)
        ax.remove()

    # --- the root's path from above, the one thing the strips cannot show ---
    ax = fig.add_subplot(gs[-1])
    ax.set_axis_off()
    box = ax.get_position()
    paths = {k: np.asarray(v, float)[:frames, 0, :2] for k, v in poses.items()}
    span = max(float(np.ptp(v, axis=0).max()) for v in paths.values() if len(v) > 1)
    span = max(span, 0.5) * 0.62
    reference = _path_length(paths["gt"])
    width = box.width / len(rows)
    for i, (title, key, accent, tag) in enumerate(rows):
        sub_ax = fig.add_axes([box.x0 + i * width, box.y0, width * 0.9, box.height])
        track = paths[key]
        centre = track.mean(axis=0) if len(track) else np.zeros(2)
        sub_ax.plot(track[:, 0], track[:, 1], color=accent, linewidth=1.0)
        sub_ax.scatter(track[:1, 0], track[:1, 1], s=18, color="#111", zorder=3)
        sub_ax.set_xlim(centre[0] - span, centre[0] + span)
        sub_ax.set_ylim(centre[1] - span, centre[1] + span)
        sub_ax.set_aspect("equal")
        sub_ax.set_xticks([]); sub_ax.set_yticks([])
        length = _path_length(track)
        note = "" if key == "gt" else "  (真值的 {:.0f}%)".format(100 * length / reference
                                                                 if reference else float("nan"))
        sub_ax.set_title("{} · 根轨迹俯视 · 路径长 {:.1f} m{}".format(tag, length, note),
                         fontsize=_root_title_size(rows), color=accent, pad=3)
    ax.remove()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=int, default=6)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=DIR",
                        help="a run directory of generated .pkl files to draw "
                             "instead of the retrieval draft; repeatable, one "
                             "row per arm, at most as many as ARM_ACCENTS")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--video", action="store_true",
                        help="also render an mp4 per arm beside the ground "
                             "truth's, carrying the clip's own wav from "
                             "data/wild_ingest_v1 when it is there and silent "
                             "when it is not (the directory the *models* read, "
                             "runs/..._gt_eval/audio, holds 35-D feature arrays "
                             "rather than sound -- do not point this at it)")
    parser.add_argument("--embed-video", action="store_true",
                        help="inline each mp4 into the page as a data: URI "
                             "instead of referencing it by name -- needed by "
                             "webview previews that cannot fetch a relative "
                             "file, and it makes the page about 4/3 of the "
                             "video bytes larger.  The .mp4 files are written "
                             "either way, so a later run can drop the flag")
    parser.add_argument("--video-stride", type=int, default=1,
                        help="frame stride for the videos; 2 halves render time "
                             "and plays at half speed")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    from train_atomic import seed_everything

    tags = {int(k): v for k, v in json.loads(
        (LABELS / "subprototype_tags.json").read_text()).items()}
    gtp = {}
    for line in (LABELS / "labels.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        gtp[row["sequence_id"]] = row["labels_path"]
    names = [l.strip() for l in (REPO / "runs/clean5b5_m6_clips.txt").read_text().splitlines()
             if l.strip()]

    arms = []
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit("--arm wants NAME=DIR, got {!r}".format(spec))
        name, directory = spec.split("=", 1)
        arms.append((name, pathlib.Path(directory)))
    if arms:
        row_specs = [(name, "arm{}".format(i), margin_tag(name))
                     for i, (name, _) in enumerate(arms)]
    else:
        row_specs = list(DRAFT_ARMS)
    rows = build_rows(row_specs)
    arm_keys = [key for _, key, _, _ in rows[1:]]

    device = torch.device(args.device)
    planner, targs, _ = load_planner(CKPT, device)
    normalizer = str(pathlib.Path(RELEASE) / "normalizer.pt")
    library = None if arms else IndexedAtomicMotionLibrary(RELEASE, retrieval_rule="duration")

    out_dir = args.out or OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    cards, table = [], []
    # Two arms that differ only in their M5 checkpoint produce the *same plan*,
    # frame for frame, and the page then carries two identical strip rows.  A
    # reader who does not know that reads the agreement as a result.  Counted
    # rather than asserted, because whether it holds depends on which arms were
    # passed -- arms with different planners will not match here.
    identical = {}
    # What the common-length truncation throws away.  It is not a rounding
    # detail: the ground truth's labels stop before the music does on some
    # clips, generation runs the whole music, and the discarded tail is far more
    # transition than the part that is kept -- so a transition share quoted from
    # the whole generated plan is a different number from the one in this table.
    truncation = []
    videos_made = []
    for clip in names:
        if len(cards) >= args.clips:
            break
        if clip not in gtp or not (GT_MOTION / (clip + ".pkl")).is_file():
            continue
        music_t = _load_music(AUDIO / (clip + ".npy"), None, targs.music_dim)
        music = music_t.numpy()
        gt = torch.from_numpy(np.load(LABELS / gtp[clip]).astype(np.int64))
        series = {"gt": gt}
        poses = {"gt": np.asarray(pickle.load(
            open(GT_MOTION / (clip + ".pkl"), "rb"))["full_pose"], float)}
        if arms:
            if any(not (d / (clip + ".pkl")).is_file() for _, d in arms):
                continue
            for key, (_, directory) in zip(arm_keys, arms):
                payload = pickle.load(open(directory / (clip + ".pkl"), "rb"))
                series[key] = torch.from_numpy(payload["atomic_labels"].astype(np.int64))
                poses[key] = np.asarray(payload["full_pose"], float)
            bounds, phase, _ = bar_bounds(music, len(series[arm_keys[0]]))
        else:
            seed_everything(args.seed)
            vote = infer_plan(planner, music_t, targs.seq_len, device,
                              plan_stride=15, plan_fusion="vote")
            bounds, phase, _ = bar_bounds(music, len(vote))
            if bounds is None:
                continue
            series["vote"] = vote
            series["grid"] = snap_to_bars(vote, bounds)
            qid = _query_retrieval_group_id(library, clip)
            for key in ("vote", "grid"):
                draft, _ = _source_safe_draft(library, series[key], 151, qid,
                                              root_continuity="off", gap_fill="zero")
                poses[key] = np.asarray(decode_motion(draft, normalizer)["full_pose"], float)
        if bounds is None:
            continue
        png = clip_figure(clip, tags, series, poses, music, bounds, rows,
                          drafts=not arms)
        path = out_dir / (short(clip) + ".png")
        path.write_bytes(png)
        clips_video = []
        if args.video:
            # ONE video per clip, with every arm drawn into it, rather than one
            # video per arm.  Two reasons, and the second is the load-bearing
            # one.  (a) Four separate <video> elements have four clocks: to
            # compare arms a reader must start them all and hope, and they drift
            # the moment one buffers.  (b) render_dance_video derives its axis
            # cube from the motion it is handed, so four separately-rendered
            # files are four different metres-per-pixel -- and with the axes
            # drawn off, nothing in the frame says so.  The arm that travels
            # furthest gets the largest cube, is therefore drawn smallest, and
            # its drift reads as the calmest of the four.  Drawn together they
            # share one scale by construction.
            sources = [("真值", GT_MOTION / (clip + ".pkl"))]
            sources += [(name, directory / (clip + ".pkl")) for name, directory in arms]
            target = out_dir / (short(clip) + "__panels.mp4")
            if not target.is_file():
                # Not check=True: one clip that will not encode must not take
                # the whole sheet with it -- the page is still worth having with
                # five videos and a named gap, and a sheet that dies at clip
                # four leaves no record of what failed.
                command = [sys.executable, str(REPO / "tools/render_dance_video.py"),
                           "--output", str(target),
                           "--stride", str(args.video_stride)]
                for label, source in sources:
                    command += ["--result", str(source), "--title", label]
                wav = source_audio(clip)
                if wav is not None:
                    command += ["--audio", str(wav)]
                # The original footage goes in ahead of the skeletons when the
                # ingest tree still has it: every other panel is a claim about
                # this clip, and until now the page carried no panel showing
                # what the claim is about.
                mp4 = source_footage(clip)
                if mp4 is not None:
                    command += ["--footage", str(mp4), "--footage-title", "原视频"]
                done = subprocess.run(command, capture_output=True)
                if done.returncode != 0:
                    print("  video failed for {}: {}".format(
                        short(clip),
                        done.stderr.decode("utf-8", "replace").strip()[-300:]),
                        flush=True)
                    target = None
            if target is not None and target.is_file():
                shown = ([("原视频", None)] if source_footage(clip) else []) + sources
                clips_video.append((panel_caption(shown), target))
                videos_made.append((clip, target))
        for a in range(len(arm_keys)):
            for b in range(a + 1, len(arm_keys)):
                hit, seen = identical.get((a, b), (0, 0))
                identical[(a, b)] = (
                    hit + int(torch.equal(series[arm_keys[a]], series[arm_keys[b]])),
                    seen + 1)
        cards.append((clip, base64.b64encode(png).decode("ascii"), clips_video))
        frames = common_frames(music, series)
        table.append((clip, phase,
                      {k: stats(v[:frames]) for k, v in series.items()}))
        longest = max(len(v) for v in series.values())
        if longest > frames:
            tail_key = arm_keys[-1] if arm_keys else "gt"
            tail = series[tail_key][frames:]
            truncation.append((frames, longest, float((tail == 0).float().mean())))
        print("rendered", short(clip), "phase", phase, flush=True)

    table_html = "".join(
        "<tr><td>{}</td><td>{}</td>".format(short(c), ph) +
        "".join("<td>{:.0%}</td><td>{}</td>".format(s[key]["transition"], s[key]["segments"])
                for _, key, _, _ in rows) + "</tr>"
        for c, ph, s in table)
    header_html = ("<tr><th>clip</th><th>相位</th>"
                   + "".join("<th>{} transition</th><th>段</th>".format(tag)
                             for _, _, _, tag in rows) + "</tr>")

    twins = ["<b>{}</b> 与 <b>{}</b>".format(rows[1 + a][0], rows[1 + b][0])
             for (a, b), (hit, seen) in sorted(identical.items())
             if seen and hit == seen]
    same_plan_html = ""
    if twins:
        same_plan_html = (
            "<p><b>有几行的计划条带按构造就是同一条:</b>" + "、".join(twins)
            + " 在画出来的每一条 clip 上逐帧相同 —— 它们共用同一个 planner、同一个种子、"
              "同一套计划开关,<b>只有 M5(completion)不同</b>。"
              "所以那几行之间可比的只有姿态和最后一行的根轨迹;"
              "计划条带一致是构造出来的,不是一个结果。</p>")

    trunc_html = ""
    if truncation:
        # np.median, not ``sorted(...)[len//2]``: on an even count the latter is
        # the upper of the two middles, and this list is routinely two long.
        lost = float(np.median([longest - kept for kept, longest, _ in truncation]))
        tail_share = float(np.median([share for *_, share in truncation]))
        trunc_html = (
            "<p><b>条带和上面这张表都截到「音乐、真值标注、每条臂」里最短的那个长度。</b>"
            "画出来的 {} 条里有 {} 条因此被截,中位少了 {:.0f} 帧({:.1f} 秒):"
            "真值的标注比音乐短,而生成是按整条音乐的长度跑的。"
            "<b>被截掉的尾巴不是随机的一段</b> —— {} 在那段上的 transition 占比中位是 "
            "{:.0%},明显高于留下来的部分,所以<b>整条生成计划的 transition 会比本表高</b>。"
            "本表报的是与真值可比的那一段,因为真值在尾巴上根本没有标注可比。</p>".format(
                len(cards), len(truncation), round(lost), lost / FPS,
                rows[-1][3], tail_share))

    # The root over every clip the arms share, not only the handful drawn below:
    # six clips is a storyboard, and a storyboard is not a measurement.
    summary_html = ""
    if arms:
        used, summary = root_path_summary(rows, arms, names)
        if summary:
            summary_html = (
                "<p><b>根轨迹路径长,全部 {} 条共享 clip —— 不只是下面画出来的 {} 条。</b>"
                "与图最后一行同一把尺子(关节 0 的 xy 逐帧位移求和),"
                "每条 clip 上所有臂与真值统一截到最短的那个长度,"
                "否则多生成了几帧的臂会被算上一段它并没有多跳的路。</p>"
                "<table><tr><th>臂</th><th>路径长中位</th><th>真值的</th>"
                "<th>p90</th></tr>".format(used, len(cards))
                + "".join("<tr><td>{}</td><td>{:.2f} m</td><td>{:.0f}%</td>"
                          "<td>{:.0f}%</td></tr>".format(tag, metres, median, p90)
                          for tag, metres, median, p90 in summary)
                + "</table>")
    def block(clip, encoded, videos):
        out = "<h2>{}</h2>".format(short(clip))
        if videos:
            out += "<div class='vids'>" + "".join(
                "<figure><figcaption>{}</figcaption>"
                "<video src='{}' controls playsinline "
                "preload='metadata'></video></figure>".format(
                    label, video_src(target, args.embed_video))
                for label, target in videos) + "</div>"
        out += "<img src='data:image/png;base64,{}'>".format(encoded)
        return out

    body = "".join(block(*card) for card in cards)
    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>plan on the bar grid</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;max-width:1800px}"
        "img{width:100%;border:1px solid #e5e5e5;margin-bottom:28px}"
        "table{border-collapse:collapse;margin:12px 0 28px}"
        ".vids{display:block;margin:8px 0 16px}"
        ".vids figure{margin:0 0 10px}"
        ".vids video{width:100%;background:#fff;border:1px solid #e5e5e5}"
        ".vids figcaption{font-size:13px;color:#555;padding:2px 0}"
        "td,th{border:1px solid #ddd;padding:4px 10px;font-size:13px}"
        "p{max-width:960px;line-height:1.6}</style>"
        "<h1>把计划压到音乐自己的小节网格上</h1>"
        "<p><b>小节网格规则</b>:保留 planner 的逐帧意见,只把<b>分割</b>换成 M1 自己的规则"
        "(第 34 通道取拍、每 4 拍一刀、相位由 <code>choose_phase</code> 定),不重训。"
        "小节取它内部非 transition 类的多数;只有当 planner 在整个小节里一个动作都没点名时,"
        "这个小节才是 transition —— <b>这条规则没有自由参数,但它是本仓造的,而且它是一次"
        "标定(决定放多少词表进去),不是「类别判得更准」的证据。</b>"
        "哪几条臂开了它,见下面那份读自 manifest 的清单。</p>"
        "<p>规则自己的读数(同一个 planner,vote 分割 → 网格分割,全 65 条 clip):"
        "原子帧占比 0.4737 → 0.6288,真值 0.6786。<b>这一对数说的是网格规则,"
        "不是本页任何两条臂之差</b> —— 本页的臂之间还差着 planner 和 M5。</p>"
        + delta_note(table, rows)
        + (POSE_NOTE_DRAFT if not arms else pose_note_generated(arms))
        + same_plan_html + summary_html + trunc_html
        + (audio_coverage_note(videos_made) if videos_made else "")
        + (footage_note(videos_made) if videos_made else "")
        + "<table>" + header_html + table_html + "</table>" + body, encoding="utf-8")
    page = out_dir / "index.html"
    print("wrote {} ({:.1f} MB{})".format(
        page, page.stat().st_size / 1e6,
        ", videos inlined" if args.embed_video else ", videos referenced by name"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
