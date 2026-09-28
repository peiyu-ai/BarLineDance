#!/usr/bin/env python3
"""Put the segmentation methods side by side on the same clips, for the eye.

WHY A PICTURE.  ``tools/snap_cuts_to_settle.py`` says it plainly: the claim
"this segmentation is better" needs either a human boundary set, which this
repository does not have, or the operator looking at a contact sheet.  Every
number available here is circular in at least one direction -- a settle-snapped
arm wins a settle criterion by construction, a beat grid wins a beat criterion
by construction -- so the numbers below are printed to characterise the arms,
never to rank them.

WHAT IS DRAWN, per clip:

  * a time panel: smoothed joint speed, the music's beats as thin lines, and one
    lane of tick marks per method.  This is where "does the cut land in a speed
    valley or on a peak" is visible.
  * one row of video frames per method, at that method's own cut frames inside
    the same window.  A boundary the paper would accept shows a pose that has
    LANDED; a boundary drawn to the velocity peak shows a blur mid-swing.

Both halves cover the identical window, so the rows are comparable across
methods rather than each being a tour of its own clip.

AND A VIDEO, which is the form the operator asked for and the better one: a
still cannot show that a cut fell in the middle of a movement.  ``--video``
writes one mp4 per clip, six panels playing at once.  Each panel is that
method's own segments played back to back with a short BLACK GAP inserted at
every cut, so a cut is something you see happen.  A method that punctuates
where the dancer lands reads as phrasing; one that punctuates mid-swing reads
as a stutter, and no number in this file says which is which.

The gaps make the panels different lengths -- more cuts, more inserted black --
so the shorter ones are padded with black to the longest.  That padding is
visible on purpose: a panel that goes dark early is a panel with fewer cuts,
which is itself part of what is being compared.

The video carries no audio.  Inserting the gaps stretches each panel's timeline
by a different amount, so one shared soundtrack would be out of sync with five
of the six panels; a per-panel one cannot be muxed into a single file.  Beat
alignment is what the still sheet's time panel is for.

THE CONTROL ROW IS NOT DECORATION.  ``random`` places the same number of cuts
with the same segment-length distribution at arbitrary offsets.  The repository
already paid for this lesson in numbers: only 22.3% of shipped Alg.1 segments
begin and end within 0.1 s of a settled pose, against 23.8% for the same
segment lengths dropped anywhere -- i.e. Alg.1's boundaries were no better than
chance on the criterion its successor was chosen for, and nobody would have
known without the control.

METHODS
  alg1        the shipped v5 recipe, read from an existing segmentation.json
              (frames_per_cluster 34, min_length 18, index_weight 4.0)
  beat4/beat2 tools/segment_on_music_beats.grid_bounds at k beats, phase from
              choose_phase (a downbeat GUESS scored by onset energy, not a
              downbeat detection -- see that function's docstring)
  settle      local minima of smoothed joint speed.  This is motion_beats.
              find_motion_beats's construction with its ``max_beats`` cap
              removed: that cap exists to pick <=4 signature poses INSIDE a
              short segment and would silently truncate a whole clip.
  alg1_settle alg1's interior cuts moved to the nearest settle within --max-snap
  random      the control described above

``--method-set bar`` swaps in a second, narrower set.  The operator's reading
after round one was that ~1 s segments are too short to hold one move -- two of
them stitched from different recordings is worse than one 2 s move taken whole
-- and that the right shape is the 4-beat grid with the boundary slid onto the
settled pose.  These arms all start from the SAME beat4 grid, so the only thing
that varies is what happens at a bar line:

  beat4              the grid untouched, boundary exactly on the beat, with
                     the phase the music's onset energy picks
  beat4_phase        the same rigid grid with the phase that lands the most
                     boundaries on a settle.  Boundaries stay exactly on beats
                     -- this is the conservative reading of "slide the anchor"
  beat4_settle       boundary moved to the nearest settle within --max-snap,
                     and left on the beat when there is none in reach
  beat4_settle_wide  the same at --wide-snap, i.e. allowed to slide further
  beat8              8 beats instead of 4, for "still too short?"
  beat4_gate         cut at a bar line ONLY when a settle is in reach, so
                     segments are a variable number of whole bars and every
                     boundary is a landed pose
  beat4_shift        THE CONTROL.  Same displacement magnitudes as
                     beat4_settle, shuffled onto the grid.  Without it "the
                     snap helped" cannot be told from "any move off the beat
                     helped", which is the same hole CLAUDE.md 2.1 records.

Usage::

    python3 tools/render_cut_comparison.py --clips runs/eval_clips_txy29.txt \\
        --limit 20 --output-dir output/t_cut_comparison
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import pickle
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.motion_beats import joint_speed  # noqa: E402
from tools.segment_on_music_beats import choose_phase, grid_bounds  # noqa: E402

FPS = 30.0
BEAT_CHANNEL = 34
ONSET_CHANNEL = 0
METHOD_COLOURS = {
    "alg1": "#4C78A8",
    "beat4": "#F58518",
    "beat2": "#E45756",
    "settle": "#54A24B",
    "alg1_settle": "#B279A2",
    "random": "#9D9D9D",
    "beat4_phase": "#72B7B2",
    "beat4_settle": "#54A24B",
    "beat4_settle_wide": "#2E7D32",
    "beat8": "#E45756",
    "beat4_gate": "#B279A2",
    "beat4_shift": "#9D9D9D",
}

METHOD_SETS = {
    # round one: does the cut land where the dancer lands?  Everything against
    # everything, plus the length-matched random control.
    "survey": ["alg1", "beat4", "beat2", "settle", "alg1_settle", "random"],
    # round two: the operator's reading is that ~1 s is too short to hold one
    # move and that the bar grid is the right skeleton, with the boundary slid
    # onto the settled pose.  Every arm here is anchored on the bar grid so the
    # comparison is about HOW FAR the boundary may leave the beat, not about
    # whether to use beats at all.  ``beat4_shift`` is the control that makes
    # the others readable: same displacement magnitudes, shuffled, so "the snap
    # helped" cannot be confused with "any move off the grid helped".
    "bar": ["beat4", "beat4_phase", "beat4_settle", "beat4_gate",
            "beat8", "beat4_shift"],
}


def settle_frames(joints: np.ndarray, *, prominence: float = 0.30,
                  min_separation: int = 15, smooth: int = 3) -> np.ndarray:
    """Every local minimum of smoothed joint speed, deep enough and separated.

    Deliberately NOT ``motion_beats.find_motion_beats``: that function caps the
    result at ``max_beats`` (default 4) because its job is to pick a handful of
    signature poses inside one short segment.  Applied to a 20-second clip the
    cap would return four cuts and look like a finding.  Everything else --
    the smoothing, the strict-entering/loose-leaving minimum test, the relative
    prominence against the clip's own mean, the greedy deepest-first separation
    -- is that function's construction, unchanged.
    """
    speed = joint_speed(joints)
    if len(speed) < 3:
        return np.zeros(0, dtype=int)
    if smooth > 1:
        speed = np.convolve(speed, np.ones(smooth) / smooth, mode="same")
    interior = np.arange(1, len(speed) - 1)
    is_minimum = (speed[1:-1] < speed[:-2]) & (speed[1:-1] <= speed[2:])
    deep_enough = speed[1:-1] <= float(speed.mean()) * (1.0 - prominence)
    minima = interior[is_minimum & deep_enough]
    chosen: List[int] = []
    for index in minima[np.argsort(speed[minima])]:
        if all(abs(int(index) - taken) >= min_separation for taken in chosen):
            chosen.append(int(index))
    return np.asarray(sorted(chosen), dtype=int)


def snap_to(cuts: Sequence[int], targets: np.ndarray, max_snap: int,
            frames: int, min_length: int) -> List[int]:
    """Move each interior cut onto the nearest target within ``max_snap``."""
    if len(targets) == 0:
        return [int(c) for c in cuts]
    out = [int(cuts[0])]
    for cut in list(cuts)[1:-1]:
        window = targets[np.abs(targets - cut) <= max_snap]
        moved = int(window[np.argmin(np.abs(window - cut))]) if len(window) else int(cut)
        if moved - out[-1] >= min_length:
            out.append(moved)
        elif int(cut) - out[-1] >= min_length:
            out.append(int(cut))
    out.append(int(cuts[-1]))
    return out


def bar_bounds(beats: np.ndarray, onset: np.ndarray, frames: int, k: int,
               min_length: int) -> List[int]:
    """The k-beat grid, phase chosen by onset energy.  A downbeat GUESS."""
    if len(beats) <= k:
        return [0, int(frames)]
    phase = choose_phase(beats, onset, k)["phase"]
    return grid_bounds(beats, frames, k, phase, min_length, edges="keep")


def settle_phase(beats: np.ndarray, settles: np.ndarray, frames: int, k: int,
                 min_length: int, tolerance: int) -> Tuple[int, List[float]]:
    """Which beat starts the bar, chosen by settles instead of onset energy.

    This is the rigid reading of "slide the anchor onto the landing": the grid
    stays perfectly periodic and every boundary stays exactly on a beat, only
    the offset of the whole grid moves.  There are only ``k`` choices, so it
    cannot drift the way a per-boundary snap can.

    The hit share of the phase it returns is CIRCULAR -- it is the quantity
    that was maximised.  What is not circular, and is what the caller records,
    is the spread across all ``k`` phases (a grid whose best and worst phase
    score the same is a grid the dancer's landings know nothing about) and
    whether this phase agrees with the one the music's onset energy picks.
    """
    shares = []
    for phase in range(k):
        bounds = grid_bounds(beats, frames, k, phase, min_length, edges="keep")
        share = hit_share(bounds, settles, tolerance)
        shares.append(0.0 if share is None else float(share))
    return int(np.argmax(shares)), shares


def snap_with_offsets(cuts: Sequence[int], targets: np.ndarray, max_snap: int,
                      frames: int, min_length: int) -> Tuple[List[int], List[int]]:
    """``snap_to`` that also reports how far each boundary actually moved.

    The offsets are what the control needs.  Reporting "the snapped arm looks
    better" without them cannot separate *landing on a settle* from *leaving
    the grid at all*, and this repository has already paid for one criterion
    that could not tell those apart (CLAUDE.md 2.1, the boundary-contrast
    ruler that rewarded cutting on the velocity peak).
    """
    out = [int(cuts[0])]
    offsets: List[int] = []
    for cut in list(cuts)[1:-1]:
        cut = int(cut)
        moved = cut
        if len(targets):
            window = targets[np.abs(targets - cut) <= max_snap]
            if len(window):
                moved = int(window[np.argmin(np.abs(window - cut))])
        if moved - out[-1] >= min_length:
            out.append(moved)
            offsets.append(moved - cut)
        elif cut - out[-1] >= min_length:
            out.append(cut)
            offsets.append(0)
    out.append(int(cuts[-1]))
    return out, offsets


def shift_like(cuts: Sequence[int], offsets: Sequence[int], frames: int,
               min_length: int, seed: int) -> List[int]:
    """The control for a snapped arm: same displacements, shuffled.

    Every boundary moves by a distance drawn from the snap's own displacement
    multiset, but to wherever that lands rather than to a settled pose.  An arm
    that beats this one bought something from the settles; an arm that does not
    only bought "off the beat by a few frames".
    """
    interior = [int(c) for c in cuts][1:-1]
    if not interior or not len(offsets):
        return [int(c) for c in cuts]
    generator = np.random.default_rng(seed)
    drawn = np.asarray(list(offsets), dtype=int).copy()
    generator.shuffle(drawn)
    out = [int(cuts[0])]
    for index, cut in enumerate(interior):
        moved = cut + int(drawn[index % len(drawn)])
        moved = int(np.clip(moved, 1, int(frames) - 1))
        if moved - out[-1] >= min_length:
            out.append(moved)
        elif cut - out[-1] >= min_length:
            out.append(cut)
    out.append(int(cuts[-1]))
    return sorted(set(out))


def gate_to(cuts: Sequence[int], targets: np.ndarray, max_snap: int,
            frames: int, min_length: int) -> List[int]:
    """Cut at a bar line only when a settle is near it, and cut on the settle.

    The difference from ``snap_with_offsets`` is what happens at a bar line
    with no settle in reach: snapping keeps the boundary (on the beat, mid
    movement), this drops it and lets the segment run on to the next bar.  So
    every segment is a whole number of bars AND every boundary is a landed
    pose -- at the price of a variable, and sometimes long, segment.  This is
    the arm for "do not cut before the move has finished".
    """
    out = [int(cuts[0])]
    for cut in list(cuts)[1:-1]:
        cut = int(cut)
        if not len(targets):
            continue
        window = targets[np.abs(targets - cut) <= max_snap]
        if not len(window):
            continue
        moved = int(window[np.argmin(np.abs(window - cut))])
        if moved - out[-1] >= min_length:
            out.append(moved)
    out.append(int(cuts[-1]))
    return out


def random_like(bounds: Sequence[int], frames: int, seed: int) -> List[int]:
    """Same number of cuts, same span-length multiset, arbitrary order."""
    spans = np.diff(np.asarray(bounds, dtype=int))
    if len(spans) < 2:
        return [int(bounds[0]), int(bounds[-1])]
    generator = np.random.default_rng(seed)
    generator.shuffle(spans)
    out = [int(bounds[0])]
    for span in spans:
        out.append(min(int(out[-1] + span), int(bounds[-1])))
    out[-1] = int(bounds[-1])
    return sorted(set(out))


def hit_share(cuts: Sequence[int], targets: np.ndarray, tolerance: int) -> Optional[float]:
    interior = [int(c) for c in cuts][1:-1]
    if not len(interior) or not len(targets):
        return None
    distance = np.abs(np.asarray(interior)[:, None] - np.asarray(targets)[None, :]).min(axis=1)
    return float((distance <= tolerance).mean())


def load_boxes(video: pathlib.Path) -> Optional[np.ndarray]:
    """The dancer box the ingest chose, ``[T,4]`` xyxy, or None.

    Cropping to it is what makes the sheet judgeable at all: these are 1080x1920
    studio wides where the dancer is a fifth of the frame height, and a 150-px
    thumbnail of the whole frame shows a silhouette, not a pose.  The box is the
    same one seeded into GVHMR, so the crop follows the person the 3D describes
    rather than a second guess at who the dancer is.
    """
    path = video.parent / "preprocess" / "bbx.pt"
    if not path.is_file():
        return None
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    boxes = payload.get("bbx_xyxy") if isinstance(payload, dict) else None
    if boxes is None:
        return None
    return np.asarray(boxes, dtype=float)


def read_frames(video: pathlib.Path, indices: Sequence[int], height: int,
                boxes: Optional[np.ndarray] = None, pad: float = 0.35):
    """Decode exactly the requested frame indices, in one forward pass."""
    import cv2
    from PIL import Image

    wanted = sorted(set(int(i) for i in indices))
    if not wanted:
        return {}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return {}
    out: Dict[int, object] = {}
    position, cursor = 0, 0
    while cursor < len(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        if position == wanted[cursor]:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if boxes is not None and len(boxes):
                x0, y0, x1, y1 = boxes[min(position, len(boxes) - 1)]
                width, height_box = x1 - x0, y1 - y0
                # Pad, then square up on the taller side: a dancer is a tall box
                # and a tall thumbnail wastes the row, so the crop is widened to
                # the height rather than the height cropped to the width.
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                side = max(width, height_box) * (1.0 + pad)
                left = max(0, int(cx - side / 2)); top = max(0, int(cy - side / 2))
                right = min(image.width, int(cx + side / 2))
                bottom = min(image.height, int(cy + side / 2))
                if right - left > 8 and bottom - top > 8:
                    image = image.crop((left, top, right, bottom))
            scale = height / image.height
            out[position] = image.resize((max(1, int(image.width * scale)), height))
            cursor += 1
        position += 1
    capture.release()
    return out


def time_panel(speed: np.ndarray, beats: np.ndarray, methods: Dict[str, List[int]],
               window: Tuple[int, int], width_px: int, order: Sequence[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    start, end = window
    seconds = np.arange(start, end) / FPS
    figure, axis = plt.subplots(figsize=(width_px / 100.0, 2.6), dpi=100)
    axis.plot(seconds, speed[start:end], color="#333333", linewidth=1.0, zorder=3)
    for beat in beats:
        if start <= beat < end:
            axis.axvline(beat / FPS, color="#BBBBBB", linewidth=0.6, zorder=1)
    top = float(np.nanmax(speed[start:end])) if end > start else 1.0
    lane_height = top * 0.16
    for lane, name in enumerate(order):
        base = -lane_height * (lane + 1)
        axis.axhline(base + lane_height * 0.5, color="#EEEEEE", linewidth=0.4, zorder=0)
        for cut in methods.get(name, []):
            if start <= cut < end:
                axis.vlines(cut / FPS, base, base + lane_height,
                            color=METHOD_COLOURS.get(name, "#000000"), linewidth=2.0, zorder=4)
        axis.text(seconds[0] + (end - start) / FPS * 0.004, base + lane_height * 0.5, name,
                  ha="left", va="center", fontsize=7, fontweight="bold",
                  color=METHOD_COLOURS.get(name, "#000000"),
                  bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))
    axis.set_xlim(seconds[0], seconds[-1] if len(seconds) > 1 else seconds[0] + 1)
    axis.set_ylim(-lane_height * (len(order) + 0.3), top * 1.05)
    axis.set_ylabel("joint speed (m/s)", fontsize=7)
    axis.set_xlabel("seconds; thin vertical lines are the music's beats", fontsize=7)
    axis.tick_params(labelsize=7)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def render_clip(record: str, joints: np.ndarray, music: np.ndarray,
                methods: Dict[str, List[int]], order: Sequence[str],
                video: Optional[pathlib.Path], window: Tuple[int, int],
                destination: pathlib.Path, thumb_height: int = 150,
                max_thumbs: int = 12) -> None:
    from PIL import Image, ImageDraw

    beats = np.flatnonzero(music[:, BEAT_CHANNEL] > 0.5)
    speed = np.convolve(joint_speed(joints), np.ones(3) / 3, mode="same")
    start, end = window
    width_px = 1500
    panel = time_panel(speed, beats, methods, window, width_px, order)

    rows = []
    if video is not None and video.is_file():
        needed: List[int] = []
        per_method: Dict[str, List[int]] = {}
        for name in order:
            inside = [c for c in methods.get(name, []) if start <= c < end]
            if len(inside) > max_thumbs:
                step = len(inside) / max_thumbs
                inside = [inside[int(i * step)] for i in range(max_thumbs)]
            per_method[name] = inside
            needed.extend(inside)
        frames = read_frames(video, needed, thumb_height, load_boxes(video))
        for name in order:
            thumbs = [(c, frames[c]) for c in per_method[name] if c in frames]
            if not thumbs:
                continue
            row = Image.new("RGB", (width_px, thumb_height + 16), (255, 255, 255))
            draw = ImageDraw.Draw(row)
            draw.text((4, 3), "{}  ({} cuts in window)".format(name, len(per_method[name])),
                      fill=METHOD_COLOURS.get(name, "#000000"))
            gap = 3
            unit = max(1, (width_px - gap * (len(thumbs) + 1)) // max(1, len(thumbs)))
            offset = gap
            for cut, image in thumbs:
                scaled = image.resize((unit, thumb_height)) if image.width != unit else image
                row.paste(scaled, (offset, 16))
                mark = ImageDraw.Draw(row)
                mark.rectangle([offset, 16, offset + unit - 1, 16 + thumb_height - 1],
                               outline=METHOD_COLOURS.get(name, "#000000"), width=2)
                mark.rectangle([offset, 16, offset + 46, 28], fill=(0, 0, 0))
                mark.text((offset + 3, 17), "{:.2f}s".format(cut / FPS), fill=(255, 255, 0))
                offset += unit + gap
            rows.append(row)

    total_height = panel.height + sum(r.height + 6 for r in rows) + 22
    sheet = Image.new("RGB", (width_px, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), "{}   window {:.1f}-{:.1f}s".format(record, start / FPS, end / FPS),
              fill=(0, 0, 0))
    sheet.paste(panel, (0, 20))
    y = 20 + panel.height
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + 6
    sheet.save(destination, quality=90)


def write_index(output_dir: pathlib.Path, report: Dict[str, object],
                order: Sequence[str]) -> None:
    """One scrollable page, because the sheets are the deliverable."""
    table = report["table"]
    rows = "".join(
        "<tr><td style='color:{}'><b>{}</b></td><td>{}</td><td>{}</td>"
        "<td>{}</td><td>{}</td></tr>".format(
            METHOD_COLOURS.get(name, "#000"), name,
            table[name]["segments_total"], table[name]["median_seconds_mean"],
            table[name]["cuts_on_beat"], table[name]["cuts_on_settle"])
        for name in order if name in table)
    blocks = []
    for entry in report["clips"]:
        blocks.append("<h3>{}</h3>".format(entry["clip"]))
        if entry.get("mp4"):
            blocks.append("<video src='{}' controls loop preload='none'></video>".format(
                pathlib.Path(entry["mp4"]).name))
        if entry.get("sheet"):
            blocks.append("<img src='{}'>".format(pathlib.Path(entry["sheet"]).name))
    sheets = "".join(blocks)
    html = """<!doctype html><meta charset="utf-8">
<title>cut comparison</title>
<style>body{{font:13px/1.5 system-ui,sans-serif;margin:24px;max-width:1560px}}
img{{width:100%;border:1px solid #ddd;margin-bottom:28px}}
video{{width:960px;max-width:100%;background:#000;margin-bottom:14px}}
table{{border-collapse:collapse;margin:12px 0}}
td,th{{border:1px solid #ccc;padding:4px 10px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}
.note{{background:#fffbe6;border-left:4px solid #f0c000;padding:10px 14px;margin:14px 0}}</style>
<h1>切分方法对比</h1>
<div class="note">{note}</div>
<table><tr><th>method</th><th>segments</th><th>median sec</th>
<th>cuts on beat</th><th>cuts on settle</th></tr>{rows}</table>
{sheets}""".format(note=report["how_to_read"], rows=rows, sheets=sheets)
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def decode_all(video: pathlib.Path, count: int, size: int,
               boxes: Optional[np.ndarray], pad: float = 0.35):
    """Every frame up to ``count``, cropped to the dancer and squared to ``size``."""
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return []
    frames = []
    index = 0
    while index < count:
        ok, frame = capture.read()
        if not ok:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if boxes is not None and len(boxes):
            x0, y0, x1, y1 = boxes[min(index, len(boxes) - 1)]
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            side = max(x1 - x0, y1 - y0) * (1.0 + pad)
            left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
            right, bottom = min(image.width, int(cx + side / 2)), min(image.height, int(cy + side / 2))
            if right - left > 8 and bottom - top > 8:
                image = image.crop((left, top, right, bottom))
        frames.append(image.resize((size, size)))
        index += 1
    capture.release()
    return frames


def blink_plan(bounds: Sequence[int], total: int, gap: int) -> List[Optional[int]]:
    """Frame indices for one panel: each segment, then ``gap`` blanks (None)."""
    plan: List[Optional[int]] = []
    cuts = [int(b) for b in bounds]
    for start, end in zip(cuts[:-1], cuts[1:]):
        start, end = max(0, start), min(total, end)
        if end <= start:
            continue
        plan.extend(range(start, end))
        plan.extend([None] * gap)
    return plan


def render_clip_video(record: str, methods: Dict[str, List[int]], order: Sequence[str],
                      video: pathlib.Path, frames_total: int, destination: pathlib.Path,
                      *, panel: int = 300, gap: int = 4, columns: int = 3,
                      fps: float = FPS) -> Optional[Dict[str, object]]:
    import subprocess
    from PIL import Image, ImageDraw, ImageFont

    decoded = decode_all(video, frames_total, panel, load_boxes(video))
    if not decoded:
        return None
    plans = {name: blink_plan(methods[name], len(decoded), gap) for name in order}
    length = max(len(p) for p in plans.values())
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    # Segment index per output frame, so a panel can say which segment it is in.
    seg_at: Dict[str, List[Optional[int]]] = {}
    for name in order:
        marks: List[Optional[int]] = []
        index = 0
        for value in plans[name]:
            marks.append(None if value is None else index)
            if value is None and (not marks[:-1] or marks[-2] is not None):
                index += 1
        seg_at[name] = marks

    label_h = 24
    rows = (len(order) + columns - 1) // columns
    width, height = panel * columns, (panel + label_h) * rows
    black = Image.new("RGB", (panel, panel), (0, 0, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "{}x{}".format(width, height), "-r", "{:g}".format(fps), "-i", "-",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         str(destination)],
        stdin=subprocess.PIPE)
    try:
        for step in range(length):
            sheet = Image.new("RGB", (width, height), (12, 12, 12))
            draw = ImageDraw.Draw(sheet)
            for slot, name in enumerate(order):
                column, row = slot % columns, slot // columns
                x, y = column * panel, row * (panel + label_h)
                plan = plans[name]
                value = plan[step] if step < len(plan) else None
                sheet.paste(decoded[value] if value is not None else black, (x, y + label_h))
                colour = METHOD_COLOURS.get(name, "#FFFFFF")
                exhausted = step >= len(plan)
                draw.rectangle([x, y, x + panel - 1, y + label_h - 1], fill=(24, 24, 24))
                segment = seg_at[name][step] if step < len(seg_at[name]) else None
                # The source timestamp is on every panel because the gaps put
                # each panel on its own stretched clock: more cuts, more inserted
                # black, so by the end of a clip two panels can be more than a
                # second apart in the dance.  Without this you cannot tell
                # whether two panels are showing the same moment.
                text = "{}  {} cuts".format(name, max(0, len(methods[name]) - 2))
                if segment is not None:
                    text += "   seg {}   t={:.1f}s".format(segment + 1, (value or 0) / fps)
                elif exhausted:
                    text += "   (ended)"
                else:
                    text += "   . cut ."
                draw.text((x + 6, y + 4), text, fill=colour, font=font)
                if value is None and not exhausted:
                    draw.rectangle([x + 1, y + label_h + 1, x + panel - 2, y + panel + label_h - 2],
                                   outline=colour, width=3)
            process.stdin.write(sheet.tobytes())
    finally:
        process.stdin.close()
        process.wait()
    return {"frames": length, "panel": panel, "gap": gap,
            "seconds": round(length / fps, 2)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--motion-dir", type=pathlib.Path,
                        default=pathlib.Path("runs/wild_v5_song_gt_eval/motion"))
    parser.add_argument("--audio-dir", type=pathlib.Path,
                        default=pathlib.Path("runs/wild_v5_song_gt_eval/audio"))
    parser.add_argument("--segmentation", type=pathlib.Path,
                        default=pathlib.Path("/cache/atomicdance-assets/runs/wild_v5_song_seg/segmentation.json"))
    parser.add_argument("--video-root", type=pathlib.Path,
                        default=pathlib.Path("/cache/atomicdance-assets/data/wild_ingest_v1"))
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--min-length", type=int, default=18)
    parser.add_argument("--max-snap", type=int, default=8,
                        help="how far a boundary may leave the grid to reach a "
                             "settle.  At 112 BPM one beat is 16 frames, so 8 "
                             "is half a beat and 6 is three eighths")
    parser.add_argument("--wide-snap", type=int, default=12,
                        help="the same for the beat4_settle_wide arm, there to "
                             "show what letting the boundary slide further "
                             "looks like rather than to argue for it")
    parser.add_argument("--method-set", choices=sorted(METHOD_SETS), default="survey",
                        help="survey = every cutting rule against the "
                             "length-matched random control; bar = four ways of "
                             "placing a boundary on the 4-beat grid, against "
                             "the grid itself and the displacement control")
    parser.add_argument("--settle-prominence", type=float, default=0.30)
    parser.add_argument("--settle-separation", type=int, default=15)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--video", action="store_true",
                        help="also write one mp4 per clip: six panels, each "
                             "playing its own segments with a black gap at "
                             "every cut, padded to the longest panel")
    parser.add_argument("--no-sheets", action="store_true",
                        help="skip the still contact sheets")
    parser.add_argument("--video-panel", type=int, default=300)
    parser.add_argument("--video-gap", type=int, default=4,
                        help="black frames inserted at each cut")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segmentation = json.loads(args.segmentation.read_text(encoding="utf-8"))
    alg1 = {record["sequence"]: record["boundaries"] for record in segmentation["records"]}

    records = [line.strip() for line in args.clips.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = records[: args.limit]
    order = METHOD_SETS[args.method_set]
    summary: Dict[str, List[Dict[str, object]]] = {name: [] for name in order}
    per_clip = []

    for index, record in enumerate(records):
        parts = record.split(":")
        stem = "{}__{}".format(parts[1], parts[2])
        motion_path = args.motion_dir / (record + ".pkl")
        music_path = args.audio_dir / (record + ".npy")
        if not motion_path.is_file() or not music_path.is_file() or stem not in alg1:
            print("skip {}: missing motion/music/segmentation".format(record))
            continue
        payload = pickle.load(motion_path.open("rb"))
        joints = payload["full_pose"] if isinstance(payload, dict) else payload
        music = np.load(music_path)
        frames = int(min(len(joints), len(music)))
        joints, music = joints[:frames], music[:frames]
        beats = np.flatnonzero(music[:, BEAT_CHANNEL] > 0.5)
        settles = settle_frames(joints, prominence=args.settle_prominence,
                                min_separation=args.settle_separation)

        onset = music[:, ONSET_CHANNEL]
        methods: Dict[str, List[int]] = {}
        methods["alg1"] = [int(b) for b in alg1[stem] if b <= frames]
        for name, k in (("beat4", 4), ("beat2", 2), ("beat8", 8)):
            methods[name] = bar_bounds(beats, onset, frames, k, args.min_length)
        methods["settle"] = sorted(set([0] + [int(s) for s in settles] + [frames]))
        methods["alg1_settle"] = snap_to(methods["alg1"], settles, args.max_snap,
                                         frames, args.min_length)
        methods["random"] = random_like(methods["alg1"], frames, args.seed + index)

        # The bar-anchored set.  All four live arms start from the SAME beat4
        # grid, so any difference between them is the boundary rule alone.
        snapped, offsets = snap_with_offsets(methods["beat4"], settles,
                                             args.max_snap, frames, args.min_length)
        methods["beat4_settle"] = snapped
        methods["beat4_settle_wide"] = snap_with_offsets(
            methods["beat4"], settles, args.wide_snap, frames, args.min_length)[0]
        methods["beat4_gate"] = gate_to(methods["beat4"], settles, args.max_snap,
                                        frames, args.min_length)
        methods["beat4_shift"] = shift_like(methods["beat4"], offsets, frames,
                                            args.min_length, args.seed + index)
        phase_settle, phase_shares = settle_phase(beats, settles, frames, 4,
                                                  args.min_length, args.tolerance)
        methods["beat4_phase"] = grid_bounds(beats, frames, 4, phase_settle,
                                             args.min_length, edges="keep") \
            if len(beats) > 4 else [0, frames]

        for name in order:
            cuts = methods[name]
            spans = np.diff(np.asarray(cuts, dtype=float)) / FPS
            interior = np.asarray([int(c) for c in cuts][1:-1], dtype=int)
            off_beat = None
            if len(interior) and len(beats):
                off_beat = float(np.median(
                    np.abs(interior[:, None] - beats[None, :]).min(axis=1)))
            summary[name].append({
                "clip": record,
                "segments": int(len(spans)),
                "median_seconds": float(np.median(spans)) if len(spans) else None,
                "on_beat": hit_share(cuts, beats, args.tolerance),
                "on_settle": hit_share(cuts, settles, args.tolerance),
                # the cost side of a snap: how far the boundary ended up from
                # the nearest beat.  A bar grid that has slid half a beat is
                # no longer a bar grid, and only this column says so.
                "off_beat_frames": off_beat,
            })

        centre = frames // 2
        half = int(args.window_seconds * FPS / 2)
        window = (max(0, centre - half), min(frames, centre + half))
        video = args.video_root / stem / "clip.mp4"
        phase_energy = (int(choose_phase(beats, onset, 4)["phase"])
                        if len(beats) > 4 else None)
        entry = {"clip": record, "video_present": video.is_file(),
                 "frames": frames, "beats": int(len(beats)),
                 "settles": int(len(settles)),
                 # non-circular half of beat4_phase: how much the choice of bar
                 # line can move settle landing at all, and whether the music's
                 # own downbeat guess picks the same line the dancer does.
                 "phase_settle_shares": [round(s, 4) for s in phase_shares],
                 "phase_settle_spread": round(max(phase_shares) - min(phase_shares), 4),
                 "phase_by_settle": phase_settle,
                 "phase_by_onset_energy": phase_energy}
        if not args.no_sheets:
            destination = args.output_dir / "{}.jpg".format(stem)
            render_clip(record, joints, music, methods, order,
                        video if video.is_file() else None, window, destination)
            entry["sheet"] = str(destination)
        if args.video and video.is_file():
            mp4 = args.output_dir / "{}.mp4".format(stem)
            entry["video"] = render_clip_video(
                record, methods, order, video, frames, mp4,
                panel=args.video_panel, gap=args.video_gap)
            entry["mp4"] = str(mp4)
        per_clip.append(entry)
        print("{:2d}/{:2d} {}".format(index + 1, len(records), stem))

    table = {}
    for name in order:
        rows = summary[name]
        if not rows:
            continue
        def mean(key):
            values = [r[key] for r in rows if r[key] is not None]
            return round(float(np.mean(values)), 4) if values else None
        table[name] = {
            "clips": len(rows),
            "segments_total": int(sum(r["segments"] for r in rows)),
            "median_seconds_mean": mean("median_seconds"),
            "cuts_on_beat": mean("on_beat"),
            "cuts_on_settle": mean("on_settle"),
            "off_beat_frames_mean": mean("off_beat_frames"),
        }
    report = {
        "schema_version": "atomicdance-cut-comparison-v1",
        "clips": per_clip,
        "config": {"tolerance_frames": args.tolerance, "max_snap": args.max_snap,
                   "settle_prominence": args.settle_prominence,
                   "settle_separation": args.settle_separation,
                   "min_length": args.min_length, "seed": args.seed},
        "table": table,
        "how_to_read": (
            "cuts_on_beat and cuts_on_settle are CIRCULAR for the arms built on those "
            "targets: beat4/beat2 score 1.0 on beats by construction and settle scores "
            "1.0 on settles by construction. They are here to characterise the arms and "
            "to be read against the `random` row, which carries the same segment-length "
            "distribution placed anywhere. Nothing here ranks the methods; the sheets do."
        ),
        "per_clip": summary,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_index(args.output_dir, report, order)
    print()
    print("{:<12} {:>6} {:>9} {:>13} {:>12} {:>13}".format(
        "method", "clips", "segments", "median sec", "on beat", "on settle"))
    for name in order:
        if name not in table:
            continue
        row = table[name]
        print("{:<12} {:>6} {:>9} {:>13} {:>12} {:>13}".format(
            name, row["clips"], row["segments_total"],
            row["median_seconds_mean"], row["cuts_on_beat"], row["cuts_on_settle"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
