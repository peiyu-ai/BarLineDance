#!/usr/bin/env python3
"""Do the jolts show ON SCREEN?  Pelvis motion measured in the rendered skin panels.

WHY A RENDER-SPACE TOOL.  The operator judges the video, not the pickle.  The
data-space column (tools/score_seam_root_speed.py) says the root braked at every
bar seam in fix2 and does not in fix4, but the skin panels go through a shared
camera that follows the centroid of every row, a VRM retarget and a rasteriser,
any of which could add or hide motion.  This reads the pixels.

THE MARKER.  The avatar wears black shorts; unsaturated near-black pixels
(max channel < 45, max - min < 6) in a skin panel are almost all shorts, with a
few stray hair and sole pixels.  The MEDIAN position of those pixels is a robust
pelvis marker.  Frames with fewer than MIN_PIXELS such pixels are dropped.

THE READINGS, per skin panel, per clip:
    speed     |marker(t+1) - marker(t)| in pixels
    brake     mean speed at 0-1 frames from a bar seam / mean at 13-14 frames
    lurch     max over distances 3-8 / the same denominator
    jolts     frames whose |second difference| exceeds the ground-truth
              panel's own p99.5 in the same video
The panels share one camera, so camera motion is common to all of them; a
difference between two arms' panels at the same frame is the figures'.

THE STRIP IS ROOT-LOCKED BY DEFAULT, AND THAT DECIDES WHAT THIS CAN SEE.
render_sample_strip.py passes --lock-root unless --free-root is given, and
render_avatar_video.py then moves every generated row's HORIZONTAL root onto the
reference's; "Height is never touched".  So in the default strip every panel's
horizontal pelvis motion is ground truth's, and ``brake`` / ``lurch`` -- which are
horizontal-travel readings -- cannot tell arms apart there.  They are printed for
--free-root strips only.  What the default view DOES show is height, so the
column that judges it is ``vjolts``: frames whose |vertical second difference|
exceeds the ground-truth panel's own p99.5 in the same video.

This was learned the hard way on 2026-09-16: the first version reported brake
for fix2's panel as 1.16 against ground truth's 1.11, i.e. "no brake", while the
data-space brake was 0.24 -- not because the tool was blind, but because the view
had replaced fix2's horizontal root with ground truth's.

POSITIVE CONTROL for ``vjolts``: an independent pass over fix2's strip found 24
vertical-jump frames on the fix2 skin, 23 within one frame of a root-height jolt
in the pkl.
"""
import argparse
import json
import math
import pathlib
import pickle
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROW2 = (560, 1045)            # the skin row in render_sample_strip's layout
PANEL_WIDTH = 520
MIN_PIXELS = 300
# A pelvis cannot cross this many pixels in one frame at this framing (a root at
# 0.3 m/s moves 2-3 px/frame; ground truth's own fastest real frames are about
# 10).  Larger steps are the shorts being occluded and the median jumping to a
# sole or a strand of hair.  Measured 2026-09-16: without this, the ground-truth
# panel's own p99.5 read 38-442 px on 4 of 10 clips and made those clips unjudgeable.
MAX_STEP_PX = 30.0


def two_sided(differences):
    d = [x for x in differences if x == x and x != 0.0]
    n = len(d)
    wins = sum(1 for x in d if x > 0)
    k = min(wins, n - wins)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(0, k + 1)) / 2 ** n) if n else 1.0
    return wins, n, p


def markers(video, panels):
    """[T, panels, 2] median (x, y) of the shorts pixels in each skin panel."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "json", str(video)],
                           capture_output=True, text=True, check=True)
    width = json.loads(probe.stdout)["streams"][0]["width"]
    y0, y1 = ROW2
    height = y1 - y0
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                             "crop={}:{}:0:{}".format(width, height, y0),
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    size = width * height * 3
    out = []
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
        hi = frame.max(axis=2)
        lo = frame.min(axis=2)
        mask = (hi < 45) & ((hi.astype(np.int16) - lo) < 6)
        row = []
        for p in range(panels):
            sub = mask[:, p * PANEL_WIDTH:(p + 1) * PANEL_WIDTH]
            ys, xs = np.nonzero(sub)
            if len(ys) < MIN_PIXELS:
                row.append((np.nan, np.nan))
            else:
                row.append((float(np.median(xs)), float(np.median(ys))))
        out.append(row)
    proc.wait()
    track = np.asarray(out, np.float64)
    for p in range(track.shape[1]):
        step = np.linalg.norm(np.diff(track[:, p], axis=0), axis=1)
        bad = np.flatnonzero(step > MAX_STEP_PX)
        track[bad + 1, p] = np.nan          # the frame the marker jumped TO
    return track


def profile(track, seams, max_distance=15):
    speed = np.linalg.norm(np.diff(track, axis=0), axis=1)
    table = {}
    for d in range(max_distance + 1):
        vals = []
        for s in seams:
            for sign in ((1, -1) if d else (1,)):
                i = s + sign * d
                if 0 <= i < len(speed) and speed[i] == speed[i]:
                    vals.append(speed[i])
        table[d] = float(np.mean(vals)) if vals else float("nan")
    far = np.nanmean([table[13], table[14]]) or float("nan")
    brake = np.nanmean([table[0], table[1]]) / far
    lurch = max(table[d] for d in range(3, 9)) / far
    return table, brake, lurch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="directory of strip mp4s")
    ap.add_argument("--clips", default="runs/vis_clips_t10.txt")
    ap.add_argument("--seams-from", required=True, help="arm dir whose bar_bounds define the seams")
    ap.add_argument("--names", default="ground truth,arm 1,arm 2",
                    help="labels of the skin panels, left to right")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    names = [n.strip() for n in args.names.split(",")]
    clips = [c.strip() for c in pathlib.Path(args.clips).read_text().split() if c.strip()]
    per = {n: {"brake": [], "lurch": [], "jolts": [], "vjolts": []} for n in names}
    curves = {n: [] for n in names}
    rows = []
    import torch
    from dataset.atomic import labels_to_segments
    loaded = []
    for clip in clips:
        stem = clip.split(":")[1] + "__" + clip.split(":")[2]
        video = pathlib.Path(args.videos) / (stem + ".mp4")
        if not video.is_file():
            continue
        with open(pathlib.Path(args.seams_from) / (clip + ".pkl"), "rb") as handle:
            record = pickle.load(handle)
        seams = record["prototype_retrieval"]["plan_postprocess"]["bar_bounds"][1:-1]
        label_starts = {seg.start for seg in labels_to_segments(
            torch.as_tensor(np.asarray(record["atomic_labels"])))}
        in_run = [b for b in seams if b not in label_starts]
        loaded.append((clip, stem, seams, in_run, markers(video, len(names))))
    if not loaded:
        raise SystemExit("no strip found")
    # ONE line for every clip: the ground-truth panel's pooled p99.5 of vertical
    # second difference, after the glitch filter.
    pooled = np.concatenate([np.abs(np.diff(t[:, 0, 1], n=2)) for *_, t in loaded])
    vline = float(np.nanpercentile(pooled, 99.5))
    pooled2 = np.concatenate([np.linalg.norm(np.diff(t[:, 0], n=2, axis=0), axis=1) for *_, t in loaded])
    line = float(np.nanpercentile(pooled2, 99.5))
    print("pooled lines from the ground-truth panel: vertical {:.2f} px, total {:.2f} px".format(vline, line))
    for p_, n_ in enumerate(names):
        per[n_]["vseam"] = []
    for clip, stem, seams, in_run, track in loaded:
        acc = np.linalg.norm(np.diff(track, n=2, axis=0), axis=2)      # [T-2, panels]
        vacc = np.abs(np.diff(track[:, :, 1], n=2, axis=0))             # vertical only
        near = np.zeros(len(vacc), bool)
        for b in in_run:
            near[max(0, b - 3):b + 2] = True     # second difference at t is centred on t+1
        row = {"clip": clip, "frames": len(track), "in_run_seams": len(in_run)}
        for p, n in enumerate(names):
            table, brake, lurch = profile(track[:, p], seams)
            hits = vacc[:, p] > vline
            jolts = int(np.nansum(acc[:, p] > line))
            vjolts = int(np.nansum(hits))
            vseam = int(np.nansum(hits & near))
            per[n]["brake"].append(brake); per[n]["lurch"].append(lurch); per[n]["jolts"].append(jolts)
            per[n]["vjolts"].append(vjolts); per[n]["vseam"].append(vseam)
            curves[n].append([table[d] for d in range(16)])
            row[n] = {"brake": brake, "lurch": lurch, "jolts": jolts, "vjolts": vjolts, "vjolts_at_in_run_seams": vseam,
                      "dropped": int(np.isnan(track[:, p, 0]).sum())}
        rows.append(row)
        print("{:<8} in-run seams {:2d}  ".format(stem.split("__")[0][-6:], len(in_run)) + "  ".join(
            "{}: vjolts {:3d} (at in-run seams {:2d})".format(n[:12], row[n]["vjolts"], row[n]["vjolts_at_in_run_seams"])
            for n in names))
    if not rows:
        raise SystemExit("no strip found")
    print("\n{} clips; on-screen pelvis speed by distance to the nearest bar seam (pixels/frame)".format(len(rows)))
    print("{:<22}".format("distance") + "".join("{:>6}".format(d) for d in range(16)))
    for n in names:
        print("{:<22}".format(n[:22]) + "".join("{:6.2f}".format(v) for v in np.nanmean(np.array(curves[n]), axis=0)))
    print("\n{:<22}{:>8}{:>10}{:>8}{:>8}{:>8}   (brake/lurch are horizontal: meaningless in a root-locked strip)".format(
        "panel", "vjolts", "@in-run", "jolts", "brake", "lurch"))
    for n in names:
        print("{:<22}{:8d}{:10d}{:8d}{:8.2f}{:8.2f}".format(n[:22], int(sum(per[n]["vjolts"])), int(sum(per[n]["vseam"])),
                                                           int(sum(per[n]["jolts"])),
                                                           np.nanmean(per[n]["brake"]), np.nanmean(per[n]["lurch"])))
    if len(names) >= 3:
        a, b = names[1], names[2]
        print("\npaired {} minus {}, two-sided sign test".format(b, a))
        for col in ("vjolts", "vseam", "jolts"):
            w, m, p = two_sided([y - x for x, y in zip(per[a][col], per[b][col])])
            print("  {:<6} {:+8.3f}  {} higher on {}/{}  P={:.4f}".format(
                col, float(np.nanmean(np.array(per[b][col], float) - np.array(per[a][col], float))), b[:10], w, m, p))
        for n in names[1:]:
            w, m, p = two_sided([y - x for x, y in zip(per[names[0]]["vjolts"], per[n]["vjolts"])])
            print("  vjolts {} vs {}: higher on {}/{}  P={:.4f}".format(n[:12], names[0][:12], w, m, p))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
