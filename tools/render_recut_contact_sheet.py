#!/usr/bin/env python3
"""One page a person can look at to see whether the fps re-cut did the right thing.

The measurements say the re-cut clips cover the span they claim and carry the
audio that belongs to it.  Those are numbers; this is the thing the numbers are
about.  For each sampled clip it lays out two strips of frames on one time axis:

    old   frames from the clip as released, fetched from the object store
    new   frames from the clip as re-cut, from the local tree

Both strips are sampled at the same *fraction* of their own duration, so what
you are comparing is what a viewer of each clip would have seen at the same
point in it.  Where the re-cut changed the span, the two strips show different
material -- that is the defect, made visible: the released clip on a 60 fps
upload holds 9.8 seconds of dancing restamped as 19.6, so its strip runs out of
choreography halfway and repeats what the new one gets through in half the
width.

The caption under each pair carries the numbers the strips cannot show: the
upload's measured rate, the source span before and after, and each clip's real
duration against the span it claims.  A clip that reads 19.6 s while its span
covers 9.8 s of a 60 fps upload is the whole bug in two numbers.

Clips are sampled from the ones the re-cut actually produced, and both strips
must be readable or the clip is skipped and counted -- a page that quietly drops
what it could not fetch reads as a page where everything was fine.

Usage::

    render_recut_contact_sheet.py --sample 12 --output output/recut_contact_sheet.png
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import random
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

GRID_FPS = 30.0
INGEST = pathlib.Path("/cache/atomicdance-assets/data/wild_ingest_v1")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def envelope(video: pathlib.Path, width: int, height: int, marks, duration):
    """The clip's own loudness contour, drawn on the same axis as its strip."""
    import numpy as np
    from PIL import Image, ImageDraw

    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vn",
         "-ac", "1", "-ar", "8000", "-f", "f32le", "-"], capture_output=True)
    panel = Image.new("RGB", (width, height), (26, 27, 32))
    draw = ImageDraw.Draw(panel)
    if raw.returncode == 0 and raw.stdout:
        samples = np.frombuffer(raw.stdout, dtype="<f4")
        if samples.size:
            columns = np.array_split(np.abs(samples), width)
            peak = max(1e-6, float(np.abs(samples).max()))
            for x, column in enumerate(columns):
                if not len(column):
                    continue
                level = float(column.max()) / peak
                bar = int(level * (height - 6))
                draw.line([(x, height - 3), (x, height - 3 - bar)], fill=(92, 106, 130))
    if duration:
        for when in marks:
            x = int(width * when / duration)
            draw.line([(x, 0), (x, height)], fill=(214, 196, 120), width=1)
    return panel


def frames_at(video: pathlib.Path, seconds, height: int):
    """One decoded frame per wall-clock second given, or None if any is past the end."""
    from PIL import Image

    duration = probe_duration(video)
    if not duration:
        return None
    out = []
    for when in seconds:
        if when > duration:
            return None
        when = max(0.0, min(duration - 0.05, when))
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-ss", "{:.3f}".format(when),
             "-i", str(video), "-frames:v", "1", "-f", "image2pipe",
             "-vcodec", "png", "-"], capture_output=True)
        if result.returncode != 0 or not result.stdout:
            return None
        image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        width = max(1, int(image.width * height / image.height))
        out.append(image.resize((width, height), Image.LANCZOS))
    return out


def probe_duration(path: pathlib.Path) -> Optional[float]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float((result.stdout or "").strip())
    except ValueError:
        return None


OSSUTIL = "/opt/data-infra/ossutil64"


def fetch_released(stem: str) -> Optional[pathlib.Path]:
    """The clip as published, straight out of the store.

    Deliberately not ``asset_io.read_bytes``: that resolves local-first, and
    ``data/wild_ingest_v1`` is a symlink into the cache the re-cut just wrote
    to -- so it hands back the *new* clip under the name of the old one, and
    the two strips come out identical for every panel.  That is the whole
    failure this page exists to make visible, reproduced by the page itself.

    The store still holds the released bytes because nothing has been pushed;
    once it has, this comparison is no longer available from OSS.
    """
    from tools import asset_io

    handle, name = tempfile.mkstemp(suffix=".mp4")
    target = pathlib.Path(name)
    import os

    os.close(handle)
    url = "oss://" + asset_io.remote_path(
        "data/wild_ingest_v1/{}/clip.mp4".format(stem))
    argv = [OSSUTIL, "cp", url, str(target), "-f"]
    config = pathlib.Path(__file__).resolve().parents[1] / "ossutilconfig"
    if config.is_file():
        argv += ["-c", str(config)]
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        return None
    return target


gap_hint = 6


def main(argv: Optional[List[str]] = None) -> int:
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--census", type=pathlib.Path,
                        default=pathlib.Path("/cache/atomicdance-assets/scratch/"
                                             "c1/refix/clips_after.json"))
    parser.add_argument("--baseline", type=pathlib.Path,
                        default=pathlib.Path("/cache/atomicdance-assets/scratch/"
                                             "c1/refix/clips_before.json"))
    parser.add_argument("--sample", type=int, default=12)
    parser.add_argument("--columns", type=int, default=8, help="frames per strip")
    parser.add_argument("--thumb-height", type=int, default=132)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    after = json.loads(args.census.read_text(encoding="utf-8"))
    before = json.loads(args.baseline.read_text(encoding="utf-8"))["clips"]
    # Only clips this run produced, that also existed before, so there are two
    # strips to compare.  A clip with no released counterpart has nothing to be
    # laid beside.
    candidates = [name for name, row in after["clips"].items()
                  if row.get("produced_now") and name in before
                  and (INGEST / name / "clip.mp4").is_file()]
    # Spread over distinct source rates rather than taking whatever sorts first:
    # 60 fps is 46% of the corpus and would otherwise be the whole page.
    by_rate: Dict[float, List[str]] = {}
    for name in candidates:
        by_rate.setdefault(after["clips"][name].get("source_fps"), []).append(name)
    random.seed(args.seed)
    for names in by_rate.values():
        random.shuffle(names)
    chosen, rates = [], sorted(by_rate, key=lambda r: (r is None, r))
    while len(chosen) < args.sample and any(by_rate.values()):
        for rate in rates:
            if by_rate[rate] and len(chosen) < args.sample:
                chosen.append(by_rate[rate].pop())
    print("candidates {} over {} distinct rates; sampling {}".format(
        len(candidates), len(by_rate), len(chosen)), flush=True)

    panels, skipped = [], []
    for index, stem in enumerate(chosen, start=1):
        local = INGEST / stem / "clip.mp4"
        released = fetch_released(stem)
        new_seconds = probe_duration(local)
        old_seconds = probe_duration(released) if released else None
        if not new_seconds or not old_seconds:
            if released is not None:
                released.unlink(missing_ok=True)
            skipped.append(stem)
            print("[{}/{}] {} SKIPPED (a duration would not read)".format(
                index, len(chosen), stem), flush=True)
            continue
        # The same wall-clock instants in both, inside whichever ends first.
        # Comparing second N to second N is the only comparison that separates
        # a clip whose picture is time-stretched from one that is not.
        shared = min(new_seconds, old_seconds)
        marks = [shared * (i + 0.5) / args.columns for i in range(args.columns)]
        new_strip = frames_at(local, marks, args.thumb_height)
        old_strip = frames_at(released, marks, args.thumb_height)
        row_after, row_before = after["clips"][stem], before[stem]
        strip_px = (sum(f.width for f in new_strip) + gap_hint * (args.columns - 1)
                    if new_strip else 0)
        new_wave = envelope(local, strip_px, 34, marks, new_seconds) if new_strip else None
        old_wave = envelope(released, strip_px, 34, marks, old_seconds) if old_strip else None
        if released is not None:
            released.unlink(missing_ok=True)
        if not new_strip or not old_strip:
            skipped.append(stem)
            print("[{}/{}] {} SKIPPED (a strip would not decode)".format(
                index, len(chosen), stem), flush=True)
            continue
        panels.append({
            "clip": stem, "old": old_strip, "new": new_strip,
            "old_wave": old_wave, "new_wave": new_wave, "marks": marks,
            "rate": row_after.get("source_fps"),
            "span_before": row_before.get("source_frame_span"),
            "span_after": row_after.get("source_frame_span"),
            "old_seconds": old_seconds, "new_seconds": new_seconds,
            "frames_after": row_after.get("num_frames"),
        })
        print("[{}/{}] {}".format(index, len(chosen), stem), flush=True)

    if not panels:
        raise SystemExit("nothing decoded; no page written")

    pad, label_w, gap, caption_h = 18, 74, gap_hint, 46
    wave_h = 34
    strip_w = max(sum(f.width for f in p["new"]) + gap * (args.columns - 1)
                  for p in panels)
    strip_w = max(strip_w, max(sum(f.width for f in p["old"]) + gap * (args.columns - 1)
                               for p in panels))
    panel_h = (args.thumb_height + wave_h + 4) * 2 + gap + caption_h + 26
    header_h = 96
    width = pad * 2 + label_w + strip_w
    height = header_h + pad + len(panels) * (panel_h + pad)

    canvas = Image.new("RGB", (width, height), (17, 18, 21))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(FONT_BOLD, 25)
    head_font = ImageFont.truetype(FONT, 14)
    label_font = ImageFont.truetype(FONT_BOLD, 14)
    small = ImageFont.truetype(FONT, 13)

    draw.text((pad, pad), "fps re-cut: released clip against re-cut clip",
              font=title_font, fill=(240, 240, 245))
    draw.text((pad, pad + 34),
              "Both strips are the same wall-clock seconds of each clip, marked on the "
              "audio envelope below them.  Second N against second N.",
              font=head_font, fill=(150, 154, 164))
    draw.text((pad, pad + 54),
              "The released clip's picture runs at 30/rate of real speed against its own "
              "music; the re-cut one runs at 1.  Where the strips diverge, that is it.",
              font=head_font, fill=(150, 154, 164))

    y = header_h + pad
    for panel in panels:
        for offset, (key, colour, name) in enumerate(
                ((("old"), (196, 108, 96), "released"), (("new"), (110, 176, 132), "re-cut"))):
            row_y = y + offset * (args.thumb_height + wave_h + 4 + gap)
            draw.text((pad, row_y + args.thumb_height // 2 - 9), name,
                      font=label_font, fill=colour)
            x = pad + label_w
            for frame in panel[key]:
                canvas.paste(frame, (x, row_y))
                x += frame.width + gap
            draw.rectangle([pad + label_w - 3, row_y - 3, x - gap + 2,
                            row_y + args.thumb_height + 2], outline=colour, width=2)
            wave = panel[key + "_wave"]
            if wave is not None:
                canvas.paste(wave, (pad + label_w, row_y + args.thumb_height + 4))
                draw.text((x - gap + 8, row_y + args.thumb_height + 12),
                          "{:.2f} s".format(panel[key + "_seconds"]),
                          font=small, fill=colour)

        caption_y = y + (args.thumb_height + wave_h + 4) * 2 + gap + 8
        rate = panel["rate"]
        draw.text((pad, caption_y), panel["clip"], font=label_font, fill=(226, 228, 234))
        line = ("upload {} fps   span {} -> {}   released {}   re-cut {} "
                "({} frames at 30 fps)").format(
            rate,
            panel["span_before"], panel["span_after"],
            "{:.2f} s".format(panel["old_seconds"]) if panel["old_seconds"] else "?",
            "{:.2f} s".format(panel["new_seconds"]) if panel["new_seconds"] else "?",
            panel["frames_after"])
        draw.text((pad, caption_y + 19), line, font=small, fill=(150, 154, 164))
        span = panel["span_after"]
        if span and rate:
            truth = (span[1] - span[0]) / rate
            draw.text((pad + strip_w + label_w - 300, caption_y + 19),
                      "span is {:.2f} s of real time".format(truth),
                      font=small, fill=(110, 176, 132))
        y += panel_h + pad

    if skipped:
        draw.text((pad, height - 22),
                  "{} clip(s) skipped, a strip would not decode: {}".format(
                      len(skipped), ", ".join(skipped[:4])),
                  font=small, fill=(196, 108, 96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print("\nwrote {}  ({} panels, {} skipped, {}x{})".format(
        args.output, len(panels), len(skipped), width, height))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
