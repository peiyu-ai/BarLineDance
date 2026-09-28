#!/usr/bin/env python3
"""Three defects the operator named on the rendered strips, each with ground truth beside it.

Watching ``output/sample_20260914_floornorm/7618203431723357818__clip000.mp4``
the operator separated what they saw into two piles, and the piles need
different fixes, so this file measures them as three separate columns:

  UPSTREAM (planner + label-to-motion)
    ``settle`` by POSITION IN THE CLIP.  The report was "the middle locks on
    reasonably, the front is so-so, the TAIL is bad" -- a claim about where in
    the clip the beat-lock lives, which no existing column can answer because
    ``score_beat_phase_shape.py`` pools every window in the clip into one
    number.  Here the clip's windows are split into thirds and each third gets
    its own aligned profile.

  DOWNSTREAM (completion / post)
    ``flicker`` -- one frame that leaves the pose and comes straight back.  A
    single frame is invisible to any speed statistic (it is two frames of
    motion that cancel), so it is measured as the SECOND difference: how far
    frame t sits from the midpoint of its neighbours, in units of the clip's
    own median.  A real movement, however fast, is smooth in this ratio.

  ROOT (what every other column here is blind to)
    ``wander`` -- the length of the pelvis path.  Every other criterion in this
    repository is root-RELATIVE, which is why a 24x root displacement once went
    unseen; this one is the root itself.  Reported beside net displacement,
    because the pair is the finding: the same ground covered by a longer path
    is sliding, not travelling.

TWO CRITERIA THAT WERE BUILT HERE AND FAILED, kept with their failures because
deleting them invites the next person to rebuild them (CLAUDE.md 2.1.2):

  * ``pivot`` -- pelvis yaw rate with both feet planted, meant to catch "the
    dancer on a turntable".  DEAD: ground truth's own yaw rate has p99 296
    deg/s and a worst frame of 661, so no threshold separates.  Ground truth
    dancers turn fast on planted feet; that is a spin, not a defect.
  * ``stance collapse`` -- the ankle-to-ankle distance dipping and returning,
    which is what the eye reads during the defect.  DEAD on the "is there any
    room" test: ground truth dips to 0.080 of its own local median against the
    generated arm's 0.099, i.e. ground truth collapses its stance MORE.  Real
    dancers bring their feet together.

AND ONE THAT CANNOT JUDGE AT THIS SAMPLE SIZE: ``settle`` by thirds.  The
half-beat roll -- the synthetic form of the exact defect -- flips the sign on
only 5 of 9 ground-truth clips in every third (P=1.000), so a third's reading
cannot separate an arm from its own control.  It is printed for LOCATING a
complaint inside one clip, which is what it was asked for, and must not be
pooled into a claim about where in a clip beat-lock lives (CLAUDE.md 2.1 rule
3: prove the measurement has power before reporting a zero).

WHY GROUND TRUTH IS IN EVERY COLUMN.  Ground truth is reconstructed by the same
GVHMR that produced the library, so it carries its own jitter and its own
foot-contact error; a threshold picked by eye off the generated arm would fire
on ground truth too and prove nothing.  Every threshold here is stated as
"ground truth's worst clip", and the per-clip paired difference plus a sign
test is the judgement (CLAUDE.md 2.1.2: pooled statistics can be bought by one
component).

CONTACT IS DERIVED GEOMETRICALLY ON BOTH SIDES.  The generated pkl carries a
``contacts`` channel and ground truth's does not.  Using the channel on one
side and geometry on the other is the mismatch 2026-09-12 already paid for
(a filter whose own quantity correlated rho=+0.029 with what it claimed to
select), so both sides get the same geometric definition.
"""
import argparse
import pathlib
import sys
from math import comb

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.score_beat_phase_profile import (BINS, FPS, WINDOW_BEATS,  # noqa: E402
                                            beat_phase, body_speed, joints_of,
                                            profile)

L_HIP, R_HIP = 1, 2
L_ANKLE, R_ANKLE = 7, 8
L_TOE, R_TOE = 10, 11

PLANTED_METRES = 0.08
# A frame this far from its neighbours' midpoint, relative to the clip's own
# median, is not a movement.  THE LINE IS GROUND TRUTH'S WORST: over the 20
# clips of runs/eval_clips_txy_t20.txt the largest single-frame ratio any
# ground-truth clip reaches is 11.8, so anything above it is a shape no
# reconstruction of a real dancer produced.
FLICKER_RATIO = 11.8


def thirds_profile(speed, beats, window_beats=WINDOW_BEATS, bins=BINS):
    """One aligned beat-phase table per third of the clip, by window position.

    Split by the window's MIDPOINT frame, not by beat index: the beat grid is
    not uniform in wild music, and splitting on beat index would put unequal
    amounts of time in the three buckets.
    """
    beats = np.asarray(beats)
    buckets = [[] for _ in range(3)]
    for start in range(0, len(beats) - window_beats):
        first, last = int(beats[start]), int(beats[start + window_beats])
        if last > len(speed) or last - first < bins * 3:
            continue
        table, _, _ = profile(speed[first:last],
                              beat_phase(beats[start:start + window_beats + 1] - first,
                                         last - first),
                              bins)
        if table is None:
            continue
        where = min(2, int(3.0 * ((first + last) / 2.0) / len(speed)))
        buckets[where].append(table)
    return [float(-np.mean(np.asarray(b), axis=0)[0]) if len(b) >= 2 else np.nan
            for b in buckets]


def flicker_frames(joints, ratio=FLICKER_RATIO):
    """Frames whose pose leaves and returns within one frame, and the worst ratio.

    Root-relative, because a root that jumps is the ``pivot``/drift family and
    is measured separately; this column is about the POSE snapping.
    """
    relative = joints - joints[:, :1, :]
    midpoint = 0.5 * (relative[:-2] + relative[2:])
    residual = np.linalg.norm(relative[1:-1] - midpoint, axis=2).mean(1)
    base = np.median(residual)
    if base <= 0:
        return np.array([], int), 0.0, residual
    scaled = residual / base
    return np.flatnonzero(scaled > ratio) + 1, float(scaled.max()), scaled


def planted(joints, metres=PLANTED_METRES):
    """Per frame, whether BOTH feet are within ``metres`` of this clip's floor."""
    feet = joints[:, [L_ANKLE, R_ANKLE, L_TOE, R_TOE], 2]
    floor = np.percentile(feet, 1.0)
    low = feet - floor
    return (low[:, [0, 2]].min(1) < metres) & (low[:, [1, 3]].min(1) < metres)


def pelvis_yaw(joints):
    """Unwrapped yaw of the hip line, radians."""
    across = joints[:, R_HIP] - joints[:, L_HIP]
    return np.unwrap(np.arctan2(across[:, 1], across[:, 0]))


def wander(joints):
    """``(path length, net displacement)`` of the pelvis, in metres.

    The pair is the point.  A dancer who ends two metres from where they began
    has travelled; a dancer whose path is two metres longer than ground truth's
    while ending in the same place has slid.  Only the first of these is
    visible to a root-relative column, which is all the others are.
    """
    ground = joints[:, 0, :2]
    return (float(np.linalg.norm(np.diff(ground, axis=0), axis=1).sum()),
            float(np.linalg.norm(ground[-1] - ground[0])))


def sign_test(differences):
    values = [d for d in differences if not np.isnan(d) and d != 0.0]
    if not values:
        return 0, 0, 1.0
    wins = sum(1 for d in values if d > 0)
    n = len(values)
    p = sum(comb(n, k) for k in range(wins, n + 1)) / (2.0 ** n)
    return wins, n, min(1.0, 2.0 * p)


def read(clip, motion_dir, audio_dir):
    path = pathlib.Path(motion_dir) / (clip + ".pkl")
    audio = pathlib.Path(audio_dir) / (clip + ".npy")
    if not (path.is_file() and audio.is_file()):
        return None
    joints = joints_of(str(path))
    music = np.load(str(audio))
    speed = body_speed(joints)
    length = min(len(music) - 1, len(speed))
    beats = np.flatnonzero(music[:length, 34] > 0.5)
    flicks, worst_flick, _ = flicker_frames(joints)
    path, net = wander(joints)
    front, middle, tail = thirds_profile(speed[:length], beats)
    return {"clip": clip, "frames": len(joints),
            "front": front, "middle": middle, "tail": tail,
            "flicks": flicks, "worst_flick": worst_flick,
            "path": path, "net": net,
            "planted_share": float(planted(joints).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="runs/vis_clips_t10.txt")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--truth", default="runs/txy_t_gt_eval/motion")
    ap.add_argument("--audio", default="runs/txy_t_gt_eval/audio")
    ap.add_argument("--frames", action="store_true",
                    help="print every offending frame, so the strip can be checked there")
    args = ap.parse_args()

    clips = [c.strip() for c in pathlib.Path(args.clips).read_text().split() if c.strip()]
    rows = []
    for clip in clips:
        gen = read(clip, args.arm, args.audio)
        truth = read(clip, args.truth, args.audio)
        if gen and truth:
            rows.append((clip, truth, gen))
    if not rows:
        raise SystemExit("no clip resolved on both sides")

    print("settle-by-thirds is printed to LOCATE a complaint inside one clip; it "
          "fails its\npower check across clips and must not be pooled -- see the "
          "module docstring.\n")
    print(f"{'clip':<22}{'settle front':>15}{'middle':>15}{'tail':>15}"
          f"{'flick n':>10}{'worst x':>10}{'root path m':>14}")
    print(f"{'':<22}{'GT / gen':>15}{'GT / gen':>15}{'GT / gen':>15}"
          f"{'GT/gen':>10}{'GT/gen':>10}{'GT / gen':>14}")
    for clip, truth, gen in rows:
        short = clip.split(":")[1][-6:] + ":" + clip.split(":")[2]
        print(f"{short:<22}"
              + f"{truth['front']:+.3f}/{gen['front']:+.3f}".rjust(15)
              + f"{truth['middle']:+.3f}/{gen['middle']:+.3f}".rjust(15)
              + f"{truth['tail']:+.3f}/{gen['tail']:+.3f}".rjust(15)
              + f"{len(truth['flicks'])}/{len(gen['flicks'])}".rjust(10)
              + f"{truth['worst_flick']:.0f}/{gen['worst_flick']:.0f}".rjust(10)
              + f"{truth['path']:.2f}/{gen['path']:.2f}".rjust(14))

    print()
    for name in ("front", "middle", "tail"):
        diffs = [g[name] - t[name] for _, t, g in rows]
        pooled_t = np.nanmean([t[name] for _, t, _ in rows])
        pooled_g = np.nanmean([g[name] for _, _, g in rows])
        wins, n, p = sign_test(diffs)
        print(f"settle {name:<7} GT {pooled_t:+.4f}  gen {pooled_g:+.4f}  "
              f"paired {np.nanmean(diffs):+.4f}  {wins}/{n}  P={p:.3f}")

    print()
    tflick = sum(len(t["flicks"]) for _, t, _ in rows)
    gflick = sum(len(g["flicks"]) for _, _, g in rows)
    print(f"flicker frames (>{FLICKER_RATIO:.0f}x the clip's own median second difference): "
          f"GT {tflick}, gen {gflick}; worst ratio GT "
          f"{max(t['worst_flick'] for _, t, _ in rows):.1f}, gen "
          f"{max(g['worst_flick'] for _, _, g in rows):.1f}")
    for name, key in (("root path length", "path"), ("net displacement", "net")):
        diffs = [g[key] - t[key] for _, t, g in rows]
        wins, n, p = sign_test(diffs)
        print(f"{name:<18} GT {np.mean([t[key] for _, t, _ in rows]):5.2f} m  gen "
              f"{np.mean([g[key] for _, _, g in rows]):5.2f} m  paired "
              f"{np.mean(diffs):+5.2f} m  {wins}/{n}  P={p:.4f}")
    print("  (a longer path over the same net displacement is sliding, not "
          "travelling; no root-relative column can see either)")

    if args.frames:
        print("\noffending frames (frame / second into the clip):")
        for clip, truth, gen in rows:
            short = clip.split(":")[1][-6:] + ":" + clip.split(":")[2]
            if len(gen["flicks"]):
                at = ", ".join(f"{f}({f / FPS:.1f}s)" for f in gen["flicks"][:14])
                print(f"  {short:<22} gen flick {len(gen['flicks']):>3}: {at}")


if __name__ == "__main__":
    main()
