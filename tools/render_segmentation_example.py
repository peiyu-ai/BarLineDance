#!/usr/bin/env python3
"""One clip, one row per segment, so a person can see whether a cut split a move.

The three statistics in ``probe_segmentation_boundaries`` say where cuts landed
and whether neighbours are near-duplicates.  Neither can say whether the thing
inside a segment reads as *one movement* -- that is a judgement a person makes
by looking, and this page is what they look at.

Layout, and why it is this and not a plot:

* **one row per segment, frames sampled inside it.**  Reading across a row
  answers "is this one movement"; reading down the rows answers "did the cut
  fall between two movements".  A timeline with tick marks answers neither.
* **the arms share the clip, the frames and the row height**, so the only thing
  that differs between the two panels is where the cuts are.
* **the change signal is drawn under both**, with each arm's cuts on it, because
  "the cut is on a turn" and "the segment is one movement" can disagree and the
  reader should see both at once.

The clip is chosen at random from those that have everything (video, features,
motion, and a cut list in every arm), and **the choice is printed and recorded
in the page** -- picking the clip that looks best is the failure mode this note
exists to prevent.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.probe_segmentation_boundaries import (           # noqa: E402
    canonical_key, change_signal, load_arm, load_bundle_index, visual_change)

THUMB_H = 96
FRAMES_PER_ROW = 8


def decode(video: pathlib.Path, height: int = THUMB_H):
    """Every frame of the clip, short side ``height``, as a list of PIL images."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video)],
        capture_output=True, text=True, check=True)
    w, h = (int(x) for x in probe.stdout.strip().split("x"))
    width = max(2, int(round(w * height / h)) // 2 * 2)
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video),
         "-vf", "scale={}:{}".format(width, height),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    return [Image.fromarray(f) for f in frames]


def strip(frames, spans, per_row=FRAMES_PER_ROW):
    """Rows of segments; each row is ``per_row`` frames sampled inside it."""
    if not spans:
        return None
    thumb_w = frames[0].width
    width = per_row * thumb_w + (per_row - 1) * 2
    height = len(spans) * (THUMB_H + 18)
    canvas = Image.new("RGB", (width, height), (16, 16, 20))
    draw = ImageDraw.Draw(canvas)
    for row, (start, end) in enumerate(spans):
        end = min(int(end), len(frames))
        start = int(start)
        if end <= start:
            continue
        picks = np.linspace(start, end - 1, per_row).round().astype(int)
        y = row * (THUMB_H + 18)
        for column, index in enumerate(picks):
            canvas.paste(frames[int(index)], (column * (thumb_w + 2), y))
        draw.text((4, y + THUMB_H + 3),
                  "seg {:>2}  frames {}-{}  {:.2f}s".format(
                      row, start, end, (end - start) / 30.0),
                  fill=(200, 200, 210))
        draw.line([(0, y - 1), (width, y - 1)], fill=(220, 90, 90), width=2)
    return canvas


def as_data_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def timeline(motion_change, visual_ch, arms_cuts, width=1000, height=150):
    """One SVG: both change signals, every arm's cuts drawn over them."""
    frames = len(motion_change)
    def path(signal, top, span):
        signal = np.asarray(signal, dtype=float)
        if signal.max() > signal.min():
            signal = (signal - signal.min()) / (signal.max() - signal.min())
        points = " ".join("{:.1f},{:.1f}".format(i * width / max(1, len(signal) - 1),
                                                 top + span - s * span)
                          for i, s in enumerate(signal))
        return points
    colours = ["#e0553f", "#3fa7e0", "#7fd06a", "#d9a441"]
    out = ['<svg viewBox="0 0 {} {}" width="100%" role="img">'.format(width, height)]
    out.append('<polyline fill="none" stroke="currentColor" stroke-opacity="0.55" '
               'stroke-width="1" points="{}"/>'.format(path(motion_change, 4, 52)))
    out.append('<polyline fill="none" stroke="currentColor" stroke-opacity="0.30" '
               'stroke-width="1" points="{}"/>'.format(path(visual_ch, 62, 52)))
    for index, (name, cuts) in enumerate(arms_cuts.items()):
        y = 120 + index * 12
        colour = colours[index % len(colours)]
        for cut in cuts[1:-1]:
            x = cut * width / max(1, frames)
            out.append('<line x1="{:.1f}" y1="4" x2="{:.1f}" y2="{}" stroke="{}" '
                       'stroke-opacity="0.35" stroke-width="1"/>'.format(x, x, y, colour))
        out.append('<text x="4" y="{}" fill="{}" font-size="9">{}</text>'.format(
            y + 8, colour, name))
    out.append('<text x="4" y="14" fill="currentColor" font-size="9" '
               'opacity="0.7">motion change</text>')
    out.append('<text x="4" y="72" fill="currentColor" font-size="9" '
               'opacity="0.7">visual change</text>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True,
                        help="name=path/to/segmentation.json; repeat")
    parser.add_argument("--clips-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--features-dir", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path, default=REPO / "output")
    parser.add_argument("--clip", help="pick this clip instead of drawing one at random")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        arms[name] = load_arm(pathlib.Path(path))
    motion_index = load_bundle_index(args.bundle)
    eligible = set.intersection(*(set(a) for a in arms.values())) & set(motion_index)
    eligible = sorted(k for k in eligible
                      if (args.clips_dir / (k.replace(":", "__") + ".mp4")).exists())
    if not eligible:
        raise SystemExit("no clip has a video, a motion array and a cut list in every arm")
    rng = np.random.default_rng(args.seed)
    chosen = ([canonical_key(args.clip)] if args.clip
              else [eligible[i] for i in rng.choice(len(eligible), args.count, replace=False)])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in chosen:
        stem = key.replace(":", "__")
        print("clip {} (drawn from {} eligible, seed {})".format(
            stem, len(eligible), args.seed), flush=True)
        frames = decode(args.clips_dir / (stem + ".mp4"))
        motion = np.load(motion_index[key]).astype(np.float32)
        mchange = change_signal(motion)
        vchange = np.zeros_like(mchange)
        if args.features_dir:
            with np.load(args.features_dir / (stem + ".npz"), allow_pickle=False) as b:
                feats = b["features"].astype(np.float32)
            feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
            vchange = visual_change(feats)
        panels = []
        for name, arm in arms.items():
            spans = list(zip(arm[key][:-1], arm[key][1:]))
            image = strip(frames, spans)
            panels.append((name, len(spans), as_data_uri(image)))
        svg = timeline(mchange, vchange[:len(mchange)],
                       {n: arms[n][key] for n in arms})
        html = ["<title>segmentation {}</title>".format(stem),
                "<style>body{font:14px/1.5 system-ui;margin:0;padding:24px;"
                "background:#faf9f7;color:#1a1a1a}"
                "@media(prefers-color-scheme:dark){body{background:#141416;color:#eee}}"
                "img{max-width:100%;height:auto;display:block;border-radius:4px}"
                ".panel{margin:28px 0}h2{font-size:15px;margin:0 0 8px}"
                ".scroll{overflow-x:auto}</style>",
                "<h1>{}</h1>".format(stem),
                "<p>{} frames &middot; drawn at random from {} eligible clips "
                "(seed {}). Each row is one segment; the frames in a row are "
                "sampled inside it. Read across a row for &ldquo;is this one "
                "movement&rdquo;, down the rows for &ldquo;did the cut fall "
                "between two movements&rdquo;.</p>".format(
                    len(frames), len(eligible), args.seed),
                "<div class='panel'>{}</div>".format(svg)]
        for name, count, uri in panels:
            html.append("<div class='panel'><h2>{} &mdash; {} segments</h2>"
                        "<div class='scroll'><img src='{}' alt='{}'></div></div>".format(
                            name, count, uri, name))
        target = args.output_dir / "segmentation_{}.html".format(stem)
        target.write_text("\n".join(html), encoding="utf-8")
        print("  ->", target, "({:.1f} MB)".format(target.stat().st_size / 1e6), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
