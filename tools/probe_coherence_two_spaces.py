#!/usr/bin/env python3
"""M2's own prototypes, scored in BOTH spaces, because one space alone lied.

What this exists to correct
---------------------------
``tools/probe_prototype_coherence.py`` scores M2's prototypes in signature-pose
space and reports a within/between ratio.  On 2026-08-21 that table was used to
tell an operator that **12 of 53 prototypes "buy nothing"** and that prototype 8
was a junk drawer collecting the corpus's most peripheral segments.

Both statements were wrong, and this tool is the measurement that showed it.
M2 clusters **TMR embeddings**.  Scored there, M2's same 53 prototypes read:

===================  ==========  =========  ==========
reading              median      >= 1.0     < 0.85
===================  ==========  =========  ==========
in pose space        0.9437      12         10
in TMR space         0.7703      **0**      **53**
===================  ==========  =========  ==========

Every one of the 12 flips.  Prototype 8 reads 1.038 in pose and **0.748** in TMR,
tighter than the median prototype.  And the two rankings are not merely different
in level, they are **uncorrelated**: Spearman rho = -0.174, p = 0.21 over the 53.
So the pose ranking carried no information about the TMR ranking, and "which
prototypes to trust" derived from it was noise.

The mistake, named
------------------
The ratio was not miscomputed.  What was missing is CLAUDE.md section 2.1 point
2: a criterion needs a **positive control in the same space it will judge in**.
There was no calibration for what a *good* clustering scores in pose space, so
"0.95 in pose space" was read as "not a cluster" when it is simply what a
TMR-built clustering looks like from pose space.  Both nulls sit at 1.00, which
is why the error was invisible: the null was right, the ceiling was never
measured.

What the pose reading does still say
------------------------------------
Not nothing, but something much narrower than was claimed: **M2's groups do not
correspond to similar signature poses.**  It groups by whatever TMR encodes.
Whether that is the right thing to group by is a question about the downstream
task, and no amount of geometry here settles it.

So this tool reports both columns side by side and refuses to emit a verdict
column.  When two rulers disagree this completely, CLAUDE.md section 2.1 point 4
applies: suspect the ruler, do not pick the one that reads nicer.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.probe_prototype_coherence import table  # noqa: E402


def load_tmr(features, embeddings) -> np.ndarray:
    blob = np.load(embeddings, allow_pickle=True)
    index = {(str(r), int(s), int(e)): i for i, (r, s, e) in enumerate(
        zip(blob["recordings"], blob["starts"], blob["ends"]))}
    rows = [index.get((str(r), int(s), int(e))) for r, s, e in
            zip(features["recording"], features["start"], features["end"])]
    missing = sum(1 for r in rows if r is None)
    if missing:
        raise SystemExit(
            "{} of {} segments have no embedding under their (recording, start, end) "
            "key. Rebuild the feature dump with --spans-from: the default unit is "
            "runs of equal label, which fuses adjacent same-prototype spans."
            .format(missing, len(rows)))
    return np.asarray(blob["embeddings"], dtype=np.float64)[np.array(rows)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=pathlib.Path,
                        default=pathlib.Path("/dev/shm/atomicdance-m2-review/"
                                             "segment_features_spans.npz"))
    parser.add_argument("--embeddings", type=pathlib.Path,
                        default=pathlib.Path("/dev/shm/atomicdance-m2-review/"
                                             "clean5b5_tmr_embeddings.npz"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=REPO / "runs/clean5b5_prototype_coherence_bothspaces.json")
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=False)
    pose = data["vector"].astype(np.float64)
    uploads, labels = data["upload"], data["label"]
    tmr = load_tmr(data, args.embeddings)

    rng = np.random.default_rng(args.seed)
    shuffled = labels[rng.permutation(len(labels))]

    spaces = {"pose": pose, "tmr": tmr}
    real = {name: table(vec, labels, uploads, args.seed) for name, vec in spaces.items()}
    null = {name: table(vec, shuffled, uploads, args.seed) for name, vec in spaces.items()}

    per = {}
    for prototype in sorted(real["pose"]):
        per[str(prototype)] = {
            "n": real["pose"][prototype]["n"],
            "uploads": real["pose"][prototype]["uploads"],
            "pose_ratio": real["pose"][prototype]["ratio"],
            "pose_null": null["pose"][prototype]["ratio"],
            "tmr_ratio": real["tmr"][prototype]["ratio"],
            "tmr_null": null["tmr"][prototype]["ratio"],
        }

    pose_r = np.array([v["pose_ratio"] for v in per.values()])
    tmr_r = np.array([v["tmr_ratio"] for v in per.values()])
    from scipy.stats import spearmanr
    rho, pvalue = spearmanr(pose_r, tmr_r)

    summary = {
        "pose": {"median": float(np.median(pose_r)),
                 "n_at_or_above_1": int((pose_r >= 1.0).sum()),
                 "n_below_085": int((pose_r < 0.85).sum()),
                 "null_median": float(np.median(
                     [v["pose_null"] for v in per.values()]))},
        "tmr": {"median": float(np.median(tmr_r)),
                "n_at_or_above_1": int((tmr_r >= 1.0).sum()),
                "n_below_085": int((tmr_r < 0.85).sum()),
                "null_median": float(np.median(
                    [v["tmr_null"] for v in per.values()]))},
        "rank_agreement_spearman_rho": float(rho),
        "rank_agreement_p": float(pvalue),
    }
    report = {
        "statistic": "mean cross-upload distance among members / same to a random "
                     "sample of non-members. below 1 = tighter than an arbitrary set.",
        "spaces": {"pose": "canonical joint positions at up to 4 motion beats (264d) "
                           "-- NOT what M2 optimised",
                   "tmr": "the TMR motion embedding (256d) -- what M2 clustered"},
        "no_verdict_column": "deliberate. The two rankings are uncorrelated "
                             "(rho={:.3f}, p={:.2f}), so any single-space verdict is "
                             "a statement about the space, not about the "
                             "prototype.".format(rho, pvalue),
        "summary": summary,
        "per_prototype": per,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("{:<12}{:>9}{:>9}{:>9}{:>11}".format(
        "", "median", ">=1.0", "<0.85", "null med"))
    for name in ("pose", "tmr"):
        s = summary[name]
        print("{:<12}{:>9.4f}{:>9}{:>9}{:>11.4f}".format(
            "in " + name, s["median"], s["n_at_or_above_1"], s["n_below_085"],
            s["null_median"]))
    print("\nrank agreement between the two spaces: rho={:.3f}, p={:.3g}".format(
        rho, pvalue))
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
