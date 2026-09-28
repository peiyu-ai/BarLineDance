"""How many visible accents the body produces per second.

WHY A THIRD BEAT COLUMN.  ``score_beat_phase_shape``'s ``settle`` has saturated:
measured 2026-09-12, the shipped arm reads +0.0731 and the 21-class arm +0.1387
against ground truth's +0.0594 -- both arms BEAT the dancer, so the column can
no longer rank them.  This one still separates.  On the clip the operator
singled out (7650126416710192357:clip000, "卡点旋律和节奏型不及 gt") the strip's
own readout is ground truth 2.40 accents/s with 1.00 on beat, baseline and
beat-fit 1.99 / 0.59, feet-lead 2.17 / 0.72 -- the ranking the operator gave,
in numbers, on a column that was being printed and never gated.

WHAT AN ACCENT IS.  ``render_plan_strip.accents`` -- a sharp change in whole-body
speed -- imported rather than reimplemented, so the number here is the number
drawn on the strip the operator looked at.  Deliberately kinematic: a boundary
between two similar prototypes changes the plan and nothing on screen.

THE ON-BEAT HALF OF THIS FILE IS REFUTED AND IS KEPT ONLY AS ITS OWN CONTROL.
Rolling ground truth half a beat -- a synthesised "does not land on the beat" --
loses on-beat accents on just 10 of 20 clips, and at every tolerance ground
truth's on-beat FRACTION equals the chance level (tol 2: 0.352 against a chance
of 0.354).  The paper's own motion beat (tools/motion_beats.find_motion_beats)
fails the same control.  So at the level of a single event, not even a real
dancer's accents are beat-coincident above chance on this corpus; the beat lock
that does exist is the continuous modulation ``score_beat_phase_shape`` reads.
Do not judge with ``on-beat/s`` or with the strip's "(X on beat)", which is a
RATE and so is collected faster by any arm that accents more.

WHAT SURVIVES IS THE DENSITY, and it separates: ground truth 2.276 accents/s
against baseline 2.051 (17 of 20 clips below, sign P = 0.0013), beat-fit 2.090
(16/20, P = 0.0059), feet-lead 2.091 (13/16, P = 0.0106).

A LIMIT THE READER HAS TO KNOW: ``accents`` thresholds at the 90th percentile
of the clip's OWN speed changes, so the count is partly scale-free.  A dance
made uniformly duller can keep its count, because the threshold falls with it.
On real ground truth the smoothing control does bite (2.276 -> 1.404), because
smoothing removes the isolated hits and changes the SHAPE of the distribution
rather than its scale -- but on a synthetic trace with evenly spaced impulses it
does not, and a unit test written against such a fixture fails for that reason
and not because the tool is broken.  What the column reliably compares is how
many sharp events a body produces against another body of the same kind.

TWO CONTROLS, and the second one is why this column may never be read alone:
  * smoothing the dance (uniform filter, 9 frames) drops it 2.276 -> 1.404, so
    it is reading movement content -- PASS;
  * adding 1 cm of Gaussian noise raises it 2.276 -> 2.477, so **jitter buys
    accents** -- FAIL.  ``jitter_share`` must be gated in parallel (CLAUDE.md
    13.3: any one-sided "bigger is better" column gets bought this way).

That buying is not what is happening here, and the check is in the table:
**ground truth's jitter is 0.0509, BELOW every arm** (baseline 0.0557,
feet-lead 0.0531, beat-fit 0.0996).  The dancer produces more accents with a
steadier body; we are both duller and slightly shakier.
"""
import argparse
import json
import pathlib
import pickle
import sys
from math import comb

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.render_plan_strip import FPS, accents  # noqa: E402
from tools.score_arm_table import jitter_share  # noqa: E402

BEAT_CHANNEL = 34
TOLERANCE = 2          # frames, the same window the strip draws with


def on_beat(joints, beats, tolerance=TOLERANCE):
    peaks = accents(joints)
    if not len(peaks):
        return 0, 0
    if not len(beats):
        return 0, len(peaks)
    hits = sum(1 for p in peaks if np.min(np.abs(np.asarray(beats) - p)) <= tolerance)
    return hits, len(peaks)


def clip_row(joints, beats, seconds, shift=0):
    if shift:
        joints = np.roll(np.asarray(joints), shift, axis=0)
    hits, total = on_beat(joints, beats)
    return {"rate": hits / max(seconds, 1e-9),
            "fraction": hits / total if total else float("nan"),
            "accents_per_s": total / max(seconds, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--audio-dir", default="runs/txy_t_gt_eval/audio")
    ap.add_argument("--ground-truth-dir", default="runs/txy_t_gt_eval/motion")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--json")
    args = ap.parse_args()

    clips = [c.strip() for c in open(args.clips) if c.strip()]
    truth, grids, seconds = {}, {}, {}
    for clip in clips:
        path = pathlib.Path(args.ground_truth_dir) / (clip + ".pkl")
        audio = pathlib.Path(args.audio_dir) / (clip + ".npy")
        if not path.is_file() or not audio.is_file():
            continue
        joints = np.asarray(pickle.load(open(path, "rb"))["full_pose"], float)
        music = np.load(audio)
        truth[clip] = joints
        grids[clip] = np.flatnonzero(music[:len(joints), BEAT_CHANNEL] > 0.5)
        seconds[clip] = len(joints) / FPS
    if not truth:
        raise SystemExit("no clip had both ground-truth motion and audio")

    rows = {}
    rows["ground truth"] = {c: clip_row(truth[c], grids[c], seconds[c]) for c in truth}
    period = int(round(np.median([np.median(np.diff(grids[c])) for c in truth
                                  if len(grids[c]) > 2])))
    rows["  control: GT half-beat rolled"] = {
        c: clip_row(truth[c], grids[c], seconds[c], shift=max(1, period // 2))
        for c in truth}
    for spec in args.arm:
        name, _, directory = spec.rpartition("=")
        row = {}
        for clip in truth:
            path = pathlib.Path(directory) / (clip + ".pkl")
            if not path.is_file():
                continue
            joints = np.asarray(pickle.load(open(path, "rb"))["full_pose"], float)
            row[clip] = clip_row(joints, grids[clip][grids[clip] < len(joints)],
                                 len(joints) / FPS)
        rows[name] = row

    print("{:<32}{:>5}{:>12}{:>11}{:>12}".format(
        "arm", "n", "on-beat/s", "fraction", "accents/s"))
    for name, row in rows.items():
        if not row:
            continue
        print("{:<32}{:>5}{:>12.3f}{:>11.3f}{:>12.3f}".format(
            name, len(row),
            float(np.mean([v["rate"] for v in row.values()])),
            float(np.nanmean([v["fraction"] for v in row.values()])),
            float(np.mean([v["accents_per_s"] for v in row.values()]))))

    control = rows["  control: GT half-beat rolled"]
    gt = rows["ground truth"]
    worse = sum(1 for c in gt if control[c]["rate"] < gt[c]["rate"])
    print("\non-beat control (REFUTED, printed so the refutation is visible): "
          "rolling ground truth half a beat loses on-beat accents on {}/{} "
          "clips -- chance. Do not judge with on-beat/s.".format(worse, len(gt)))

    # The two controls the DENSITY column has to pass, run every time rather
    # than quoted from a docstring, because a control that is not run is a
    # sentence and not a gate.
    from scipy.ndimage import uniform_filter1d
    rng = np.random.default_rng(0)
    dull, noisy = [], []
    for clip, joints in truth.items():
        seconds_here = seconds[clip]
        dull.append(len(accents(uniform_filter1d(joints, size=9, axis=0))) / seconds_here)
        noisy.append(len(accents(joints + rng.normal(0, 0.01, joints.shape))) / seconds_here)
    base_density = float(np.mean([v["accents_per_s"] for v in gt.values()]))
    print("density controls: smoothed {:.3f} (must be LOWER than {:.3f}) -> {}; "
          "+1cm noise {:.3f} -> jitter BUYS accents, so jitter is gated below"
          .format(float(np.mean(dull)), base_density,
                  "PASS" if np.mean(dull) < base_density else "FAIL",
                  float(np.mean(noisy))))

    print("\njitter, gated in parallel (T-line ground truth 0.0509; the 0.096 in "
          "score_arm_table's docstring is the v5 line)")
    print("{:<32}{:>10}".format("arm", "jitter"))
    for name in rows:
        if name.startswith("  control"):
            continue
        source = truth if name == "ground truth" else None
        values = []
        for clip in rows[name]:
            path = (pathlib.Path(args.ground_truth_dir) if source is not None
                    else pathlib.Path(dict(
                        (s.rpartition("=")[0], s.rpartition("=")[2])
                        for s in args.arm)[name])) / (clip + ".pkl")
            if path.is_file():
                values.append(float(jitter_share(
                    np.asarray(pickle.load(open(path, "rb"))["full_pose"], float))))
        if values:
            print("{:<32}{:>10.4f}".format(name, float(np.nanmean(values))))

    # PAIRED ON THE DENSITY AND AGAINST GROUND TRUTH.  The first version paired
    # on ``rate`` -- the on-beat column refuted six lines above -- and against
    # the first arm rather than the dancer, so it answered "is this arm more
    # like that arm" with a number that measures nothing.  A judgement made with
    # a column the same file has just refuted is the worst shape available.
    arms_only = [n for n in rows if not n.startswith(("ground", "  control"))]
    if arms_only:
        base = "ground truth"
        print("\npaired against GROUND TRUTH, per clip, on accents/s "
              "(negative = duller than the dancer)")
        print("{:<32}{:>5}{:>11}{:>10}{:>10}".format(
            "arm", "n", "mean d", "below", "sign P"))
        for name in arms_only:
            shared = [c for c in rows[name] if c in rows[base]]
            d = np.array([rows[name][c]["accents_per_s"]
                          - rows[base][c]["accents_per_s"] for c in shared])
            moved = d[d != 0]
            wins = int((moved < 0).sum())
            total = len(moved)
            p = (sum(comb(total, k) for k in range(wins, total + 1))
                 / 2.0 ** total) if total else 1.0
            print("{:<32}{:>5}{:>11.3f}{:>6}/{:<3}{:>10.3f}".format(
                name, len(shared), float(d.mean()), wins, total, min(1.0, p)))

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rows, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
