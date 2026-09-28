"""Refuse to spend a render on two arms that look the same.

WHY.  On 2026-09-13 a ten-clip render (about an hour of GPU-less wall clock,
five workers) was spent comparing ``feet-lead`` against ``feet-lead + inpaint
12``.  The operator watched it and said: "feetlead 和 fl inp12 一模一样啊,
是不是 pipeline 上有什么 bug".  There was no bug -- the only knob between those
two arms is the completion's inpaint width, whose entire effect is about 10 cm
of per-frame joint displacement, while the choice of prototype that separates
either of them from the shipping baseline is about 102 cm, the same order as the
112 cm that separates the baseline from GROUND TRUTH.

So the scale to judge a knob by, before rendering anything:

    ~110 cm   a different dance (ground truth against the baseline)
    ~100 cm   a different retrieval choice (--draft-feet-lead)
    ~40 cm    a different tie-break within the same pool (--draft-beat-fit)
    ~10 cm    the whole seam-repair family (inpaint width, seam blend,
              root-velocity blend) -- BELOW what the eye separates

A column can move a lot inside that last band: the seam family took the sharpness
peak from 122% of ground truth to 102% and the seam share from 21.4% to 12.6%,
all of it invisible.  Measuring is not the problem; spending the one judgement
that counts -- the video -- on a difference the video cannot carry is.
"""
import argparse
import itertools
import pathlib
import pickle

import numpy as np

# Below this, two arms have never been visually separable in this project.
VISIBLE_CM = 20.0


def track(directory, clip):
    path = pathlib.Path(directory) / (clip + ".pkl")
    if not path.is_file():
        return None
    return np.asarray(pickle.load(open(path, "rb"))["full_pose"], float)


def separation_cm(left, right, clips):
    values = []
    for clip in clips:
        a, b = track(left, clip), track(right, clip)
        if a is None or b is None:
            continue
        n = min(len(a), len(b))
        values.append(np.linalg.norm(a[:n] - b[:n], axis=-1).mean(1))
    if not values:
        return float("nan")
    return float(np.concatenate(values).mean() * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--threshold-cm", type=float, default=VISIBLE_CM)
    ap.add_argument("--allow-indistinguishable", action="store_true",
                    help="render anyway; only for a deliberate null comparison")
    args = ap.parse_args()

    clips = [c.strip() for c in open(args.clips) if c.strip()]
    arms = [(s.rpartition("=")[0], s.rpartition("=")[2]) for s in args.arm]

    print("{:<22}{:<22}{:>12}".format("arm", "arm", "mean cm"))
    too_close = []
    for (ln, ld), (rn, rd) in itertools.combinations(arms, 2):
        gap = separation_cm(ld, rd, clips)
        flag = "" if gap >= args.threshold_cm else "   <- indistinguishable"
        print("{:<22}{:<22}{:>12.2f}{}".format(ln, rn, gap, flag))
        if gap < args.threshold_cm:
            too_close.append((ln, rn, gap))

    if too_close and not args.allow_indistinguishable:
        raise SystemExit(
            "these pairs differ by less than {:.0f} cm of per-frame joint "
            "displacement and have never been separable on video in this "
            "project:\n{}\nRendering them side by side spends the only "
            "judgement that counts on a difference the video cannot carry. "
            "Drop one of each pair, or pass --allow-indistinguishable if the "
            "null comparison is the point.".format(
                args.threshold_cm,
                "\n".join("  {} vs {}: {:.2f} cm".format(*t) for t in too_close)))
    print("\nevery pair is separable at >= {:.0f} cm".format(args.threshold_cm))


if __name__ == "__main__":
    main()
