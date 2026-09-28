#!/usr/bin/env python3
"""Watch and listen: the clip, its beat grid, and where the body settles.

Two measurements disagreed about this corpus and a person should be able to
check which one is describing reality.  The level of the change signal at beat
frames says nothing (median z = +0.03); the autocorrelation at the clip's own
beat period says the body is locked to it (63.9% of clips beat their own
neighbouring lags, binomial p < 1e-14).  The difference is phase: dancers hit
the beat, the off-beat, or half tempo, and averaging a level across them
cancels.

So this page plays the clip with its audio and draws, on one time axis:

* the joint-speed curve -- the troughs are where a movement settles, which is
  what the paper calls a motion beat;
* the music's beat grid, and the same grid at half tempo, each drawn at the
  phase fitted to that clip;
* the current cut points.

Read it by playing the video and watching which vertical lines the dancer lands
on.  Nothing here is a measurement; it is the thing the measurements are about.
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.probe_segmentation_boundaries import (           # noqa: E402
    canonical_key, change_signal, load_arm, load_bundle_index)


def fitted_grid(change, period):
    """The phase that puts the grid on the troughs; the period is the music's."""
    best, best_cost = np.array([], int), None
    for phase in np.arange(0, period, 1.0):
        grid = np.arange(phase, len(change) - 1, period).round().astype(int)
        grid = grid[(grid > 0) & (grid < len(change))]
        if len(grid) < 3:
            continue
        cost = float(change[grid].mean())
        if best_cost is None or cost < best_cost:
            best, best_cost = grid, cost
    return best


def curve(signal, width, top, span):
    signal = np.asarray(signal, dtype=float)
    if signal.max() > signal.min():
        signal = (signal - signal.min()) / (signal.max() - signal.min())
    return " ".join("{:.1f},{:.1f}".format(i * width / max(1, len(signal) - 1),
                                           top + span - s * span)
                    for i, s in enumerate(signal))


def marks(positions, frames, width, y0, y1, colour, opacity="0.5"):
    return "\n".join(
        '<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{c}" '
        'stroke-opacity="{o}" stroke-width="1"/>'.format(
            x=p * width / max(1, frames), y0=y0, y1=y1, c=colour, o=opacity)
        for p in positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--clips-dir", type=pathlib.Path, required=True)
    parser.add_argument("--arm", required=True, help="name=segmentation.json")
    parser.add_argument("--output-dir", type=pathlib.Path, default=REPO / "output")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    name, path = args.arm.split("=", 1)
    arm = load_arm(pathlib.Path(path))
    index = load_bundle_index(args.bundle)
    music_of = {canonical_key(json.loads(l)["recording_id"]):
                args.bundle / json.loads(l)["assets"]["music_35"]
                for l in open(args.bundle / "sequences.jsonl")}
    eligible = sorted(k for k in set(arm) & set(index) & set(music_of)
                      if (args.clips_dir / (k.replace(":", "__") + ".mp4")).exists())
    rng = np.random.default_rng(args.seed)
    chosen = [eligible[i] for i in rng.choice(len(eligible), args.count, replace=False)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key in chosen:
        stem = key.replace(":", "__")
        motion = np.load(index[key]).astype(np.float32)
        change = change_signal(motion)
        music = np.load(music_of[key]).astype(np.float32)
        beats = np.where(music[:len(change), -1] > 0.5)[0]
        period = float(np.median(np.diff(beats))) if len(beats) > 3 else 15.0
        grid = fitted_grid(change, period)
        half = fitted_grid(change, period * 2)
        cuts = [c for c in arm[key][1:-1]]
        width, frames = 1200, len(change)
        svg = ['<svg viewBox="0 0 {} 210" width="100%">'.format(width),
               marks(beats, frames, width, 0, 150, "#8a8f98", "0.35"),
               marks(grid, frames, width, 0, 150, "#3fa7e0", "0.75"),
               marks(half, frames, width, 0, 150, "#7fd06a", "0.85"),
               marks(cuts, frames, width, 0, 150, "#e0553f", "0.9"),
               '<polyline fill="none" stroke="currentColor" stroke-width="1.2" '
               'points="{}"/>'.format(curve(change, width, 10, 130)),
               '<text x="4" y="172" fill="#8a8f98" font-size="11">音乐拍点（原始）</text>',
               '<text x="150" y="172" fill="#3fa7e0" font-size="11">拍格 · 拟合相位 ({:.1f} 帧 / {:.0f} BPM)</text>'.format(period, 1800 / period),
               '<text x="430" y="172" fill="#7fd06a" font-size="11">半速拍格 ({:.1f} 帧)</text>'.format(period * 2),
               '<text x="620" y="172" fill="#e0553f" font-size="11">当前切点 ({})</text>'.format(len(cuts)),
               '<text x="4" y="196" fill="currentColor" font-size="11" opacity="0.7">'
               '曲线 = 逐帧关节变化率；谷 = 动作落定处（论文的 motion beat）</text>',
               "</svg>"]
        video = base64.b64encode((args.clips_dir / (stem + ".mp4")).read_bytes()).decode()
        html = ["<title>beat alignment {}</title>".format(stem),
                "<style>body{font:14px/1.6 system-ui;margin:0;padding:24px;max-width:1280px;"
                "background:#faf9f7;color:#1a1a1a}"
                "@media(prefers-color-scheme:dark){body{background:#141416;color:#eee}}"
                "video{max-width:520px;width:100%;border-radius:6px}</style>",
                "<h1>{}</h1>".format(stem),
                "<p>{} 帧 &middot; 拍周期 {:.1f} 帧（{:.0f} BPM）&middot; 从 {} 条候选里随机抽（seed {}）</p>".format(
                    frames, period, 1800 / period, len(eligible), args.seed),
                '<video controls src="data:video/mp4;base64,{}"></video>'.format(video),
                "\n".join(svg)]
        target = args.output_dir / "beats_{}.html".format(stem)
        target.write_text("\n".join(html), encoding="utf-8")
        print("{} -> {} ({:.1f} MB)".format(stem, target, target.stat().st_size / 1e6), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
