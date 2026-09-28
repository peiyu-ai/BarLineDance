"""Cross-clip novelty: does every clip in an arm dance the same dance?

WHY A NEW FILE.  ``tools/measure_motion_repetition.py`` answers "does this clip
repeat ITSELF" -- for each half-second window it finds the nearest window at
least 1.5 s away IN THE SAME CLIP.  The operator's complaint is a different one
("every clip dances the same"), and the readings quoted throughout
``docs/DANCE_QUALITY_DEFECTS.md`` sections 18.2 and 18.7 -- ground truth 0.4582,
retrieval draft 0.3253, shipped output 0.3827 -- come from a scratch script that
no longer exists in the tree (``grep -rn novelty tools/ tests/`` finds only the
within-clip tool).  So this is a RE-IMPLEMENTATION from those sections' wording,
and it says so; its agreement with the published ground-truth reading is a
runtime check (``--expect-ground-truth``), not an assumption.  If the check
fails, the numbers this file prints are a criterion of its own and must be
labelled that way rather than quoted as the documented column.

THE MEASUREMENT, one step off the within-clip one so the two are comparable:

  * windows          identical to the within-clip tool -- ``window_seconds``
                     (0.5) of root-relative joint positions, flattened, taken
                     every ``stride_frames`` (5).  Root-relative so a clip that
                     travels cannot look novel for that reason.
  * nearest neighbour  for each window of clip A, the closest window belonging
                     to any OTHER clip of the same arm.  There is no time gap to
                     mask: a different clip is already far away in time.
  * scale            the mean of all cross-clip window distances in the arm, so
                     the reading is scale free and an arm that simply moves more
                     cannot win the column for that reason.
  * the statistic    the median over an arm's windows, and beside it the
                     per-clip median so ``clips more stereotyped than truth``
                     can be counted the way sections 18.2/18.7 count it.

  low  = every moment of every clip has a near twin in some other clip
  high = clips are unlike each other

WHY THE SCALE IS POOLED ACROSS THE ARM rather than per clip: the question is
about the arm's whole output, and a per-clip scale would divide each clip by its
own spread, which is exactly the quantity a stereotyped arm has collapsed.

CONTROLS (tests/test_exp_draftonly_cross_novelty.py), each able to fail:

  * NEGATIVE: an arm whose clips are all the SAME motion must read near 0.
  * POSITIVE: an arm of independent random motions must read high (> 0.4), and
    strictly higher than the cloned arm by a wide margin -- without this a
    metric that is merely noisy would pass for one that works.
  * SCALE INVARIANCE: multiplying every clip by a constant must not move the
    reading.
  * TRANSLATION INVARIANCE: adding a per-clip constant root offset must not
    move the reading, since the windows are root-relative.
  * The within-clip tool must NOT reproduce these readings on the same input --
    a clip that repeats itself but differs from the others reads low there and
    high here, which is what makes this a second column and not a copy.
"""

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.measure_motion_repetition import FPS, windows  # noqa: E402


def clip_windows(joints, window_seconds=0.5, stride_frames=5):
    length = max(2, int(round(window_seconds * FPS)))
    joints = np.asarray(joints, float)
    if len(joints) < length:
        return None
    block, _ = windows(joints, length, stride_frames)
    return block if len(block) else None


def cross_novelty(blocks_by_clip):
    """blocks_by_clip: name -> (n_windows, d).  Returns (pooled, per_clip)."""
    names = [n for n, b in blocks_by_clip.items() if b is not None and len(b)]
    if len(names) < 2:
        return float("nan"), {}
    stacked = np.concatenate([blocks_by_clip[n] for n in names], axis=0)
    owner = np.concatenate([np.full(len(blocks_by_clip[n]), i) for i, n in enumerate(names)])
    distance = cdist(stacked, stacked)
    other = owner[:, None] != owner[None, :]
    scale = float(distance[other].mean())
    if scale <= 0:
        return float("nan"), {}
    masked = np.where(other, distance, np.inf)
    nearest = masked.min(axis=1) / scale
    per_clip = {n: float(np.median(nearest[owner == i])) for i, n in enumerate(names)}
    return float(np.median(nearest)), per_clip


def score_arm(directory, clips, **options):
    blocks = {}
    for clip in clips:
        path = pathlib.Path(directory) / (clip + ".pkl")
        if not path.exists():
            continue
        payload = pickle.load(open(path, "rb"))
        blocks[clip] = clip_windows(np.asarray(payload["full_pose"], float), **options)
    pooled, per_clip = cross_novelty(blocks)
    return {"clips": len(per_clip), "novelty": pooled, "per_clip": per_clip}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument("--stride-frames", type=int, default=5)
    parser.add_argument("--expect-ground-truth", type=float, nargs=2, metavar=("LO", "HI"),
                        default=None,
                        help="warn loudly unless ground truth lands in [LO, HI]; sections "
                             "18.2/18.7 report 0.4582")
    parser.add_argument("--out", type=pathlib.Path)
    arguments = parser.parse_args()
    options = {"window_seconds": arguments.window_seconds,
               "stride_frames": arguments.stride_frames}
    clips = [line.strip() for line in arguments.clips.read_text().splitlines() if line.strip()]
    truth = score_arm(arguments.ground_truth, clips, **options)
    report = {"options": options, "ground_truth": truth, "arms": {},
              "provenance": "re-implementation of the cross-clip novelty column of "
                            "docs/DANCE_QUALITY_DEFECTS.md 18.2/18.7; the script that "
                            "produced the published 0.4582/0.3253/0.3827 is not in the tree"}
    if arguments.expect_ground_truth:
        lo, hi = arguments.expect_ground_truth
        report["calibration"] = {"expected_band": [lo, hi],
                                 "ground_truth_read": truth["novelty"],
                                 "passed": bool(lo <= truth["novelty"] <= hi)}
    for entry in arguments.arm:
        name, _, directory = entry.rpartition("=")
        if not pathlib.Path(directory).is_dir():
            raise SystemExit("error: --arm {!r} points at {!r}, which does not exist."
                             .format(name, directory))
        arm = score_arm(directory, clips, **options)
        shared = [c for c in arm["per_clip"] if c in truth["per_clip"]]
        arm["clips_more_stereotyped_than_truth"] = "{}/{}".format(
            sum(1 for c in shared if arm["per_clip"][c] < truth["per_clip"][c]), len(shared))
        report["arms"][name] = arm
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.out:
        arguments.out.write_text(text)
    print("%-24s %10s %8s   %s" % ("arm", "novelty", "clips", "more stereotyped than truth"))
    print("%-24s %10.4f %8d" % ("ground truth", truth["novelty"], truth["clips"]))
    for name, row in report["arms"].items():
        print("%-24s %10.4f %8d   %s" % (name, row["novelty"], row["clips"],
                                         row["clips_more_stereotyped_than_truth"]))
    if arguments.expect_ground_truth and not report["calibration"]["passed"]:
        print("\nWARNING: ground truth reads {:.4f}, outside the published band {}. "
              "These numbers are this file's own criterion, NOT the documented column."
              .format(truth["novelty"], report["calibration"]["expected_band"]),
              file=sys.stderr)


if __name__ == "__main__":
    main()
