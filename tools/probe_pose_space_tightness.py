#!/usr/bin/env python3
"""Are M2's prototypes tight in a space M2 never optimised?

The number that must not be reported alone
------------------------------------------
M2 is K-Means over TMR embeddings, so "members of a prototype are close in the
TMR embedding" is true of any K-Means result on any data, noise included.  That
is CLAUDE.md section 2's gate that cannot fail, and acceptance rate, cluster
dispersion and largest-class share are all versions of it.

So this measures tightness somewhere else: the **pose space the contact sheets
actually draw** -- ``tools/motion_beats.segment_descriptor``'s signature poses,
canonical joint positions at the motion beats.  K-Means never saw these
coordinates.  Being tight here is not guaranteed by construction, which is the
only reason the reading is worth anything.

This criterion is invented for this review (CLAUDE.md section 2.1 point 1), so
it does not get to make a judgement on its own reading.  It is calibrated
against two points whose answers are known before it runs:

* **Null (known bad).**  Labels permuted over segments with every prototype's
  size preserved exactly -- the same construction as the visual control sheet.
  A partition with no movement structure.  Distance here should be the largest.
* **Floor (known good).**  K-Means with the same K fitted *directly on these
  descriptors*.  This is close to the tightest a 53-way partition of this data
  can be in this space, so it bounds how much tightness is available at all.
  It is not a fair rival to M2 -- it optimises the very quantity being read --
  and it is not meant to be; it is the ruler's top end.

M2 then has somewhere to sit between them, and "recovery" says where.  If M2
lands at the null, the coherence on the sheets is the reader's pattern matching.
If it lands near the floor, the grouping carries pose structure that the
descriptor can see without having been fitted to it.

Cross-upload pairs only
-----------------------
Consecutive segments of one upload resemble each other for reasons that have
nothing to do with the vocabulary, and these prototypes are already known to
concentrate accounts (lift 1.274, ``runs/clean5b5_prototype_enrichment.json``).
Counting same-upload pairs would therefore pay M2 for capturing identity, which
is the thing the account probe says is a *cost*.  Every pair here crosses
uploads, in all three arms.

What it does not answer
-----------------------
Nothing about whether the prototypes are *movements* in the paper's sense --
"complete motion processes" with preparation, execution and recovery.  A tight
group of static shapes would score exactly as well.  That question needs the
video sheet and a person.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.cluster_atomics_tmr import build_row_index, resolve_row  # noqa: E402
from tools.convert_motion_to_guofeats import motion_151_to_joints   # noqa: E402
from tools.motion_beats import canonical_pose, find_motion_beats    # noqa: E402
from tools.recluster_atomics_ingroup import segments_of             # noqa: E402

BODY_JOINTS = 22
BEATS = 4
MIN_FRAMES = 4          # collect_members' floor, kept in lockstep with it


def descriptor(joints: np.ndarray) -> np.ndarray:
    """The signature poses a card row draws, flattened.

    Padded to a fixed beat count by repeating the last beat: a segment that
    settles only twice has two poses, and dropping it instead would quietly
    select for long segments.
    """
    beats = find_motion_beats(joints, max_beats=BEATS)
    if not beats:
        beats = [0]
    poses = [canonical_pose(joints[b])[:BODY_JOINTS] for b in beats[:BEATS]]
    while len(poses) < BEATS:
        poses.append(poses[-1])
    return np.concatenate([pose.reshape(-1) for pose in poses])


def mean_cross_upload_distance(vectors: np.ndarray, uploads: np.ndarray) -> float:
    """Mean Euclidean distance over pairs drawn from two different uploads."""
    if len(vectors) < 2:
        return float("nan")
    gram = vectors @ vectors.T
    square = np.diag(gram)
    distance = np.sqrt(np.maximum(
        square[:, None] + square[None, :] - 2.0 * gram, 0.0))
    cross = uploads[:, None] != uploads[None, :]
    if not cross.any():
        return float("nan")
    return float(distance[cross].mean())


def arm_distance(labels: np.ndarray, vectors: np.ndarray,
                 uploads: np.ndarray) -> Dict[str, float]:
    """Size-weighted mean of the per-prototype cross-upload distance."""
    readings, weights = [], []
    for label in np.unique(labels):
        mask = labels == label
        value = mean_cross_upload_distance(vectors[mask], uploads[mask])
        if not np.isnan(value):
            readings.append(value)
            weights.append(int(mask.sum()))
    readings, weights = np.array(readings), np.array(weights, dtype=np.float64)
    return {"weighted": float((readings * weights).sum() / weights.sum()),
            "unweighted": float(readings.mean()),
            "prototypes_read": int(len(readings))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--null-draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    rows = [json.loads(line) for line
            in (args.labels / "labels.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    sequences = {json.loads(line)["recording_id"]: json.loads(line)
                 for line in (args.bundle / "sequences.jsonl").open(encoding="utf-8")}
    index = build_row_index(sequences)

    vectors: List[np.ndarray] = []
    labels: List[int] = []
    uploads: List[str] = []
    skipped = 0
    for done, row in enumerate(rows, 1):
        sequence = resolve_row(index, row["recording_id"]) or sequences.get(
            row["recording_id"])
        if sequence is None:
            skipped += 1
            continue
        joints = motion_151_to_joints(np.load(args.bundle / sequence["motion_path"]))
        upload = str(row["recording_id"]).rsplit(":", 1)[0]
        arrays = np.load(args.labels / row["labels_path"])
        for start, end, label in segments_of(arrays):
            if label <= 0 or end - start < MIN_FRAMES:
                continue
            vectors.append(descriptor(joints[start:end]))
            labels.append(int(label))
            uploads.append(upload)
        if done % 400 == 0:
            print("  {}/{} recordings, {} segments".format(
                done, len(rows), len(vectors)), flush=True)

    vectors = np.stack(vectors)
    labels = np.array(labels)
    upload_ids = np.array([hash(u) for u in uploads])
    sizes = collections.Counter(labels.tolist())
    print("{} segments, {} prototypes, {} uploads, {} recordings unresolved".format(
        len(vectors), len(sizes), len(set(uploads)), skipped), flush=True)

    observed = arm_distance(labels, vectors, upload_ids)
    print("observed  {:.4f}".format(observed["weighted"]), flush=True)

    rng = np.random.default_rng(args.seed)
    null = []
    for draw in range(args.null_draws):
        # A permutation of the label vector: every prototype keeps its exact
        # size, only membership scatters.  Same null as the control sheet.
        null.append(arm_distance(labels[rng.permutation(len(labels))],
                                 vectors, upload_ids)["weighted"])
        print("  null draw {}/{}  {:.4f}".format(draw + 1, args.null_draws, null[-1]),
              flush=True)
    null = np.array(null)

    from sklearn.cluster import KMeans
    floor_labels = KMeans(n_clusters=len(sizes), n_init=4, random_state=args.seed
                          ).fit_predict(vectors)
    floor = arm_distance(floor_labels, vectors, upload_ids)
    print("floor     {:.4f}".format(floor["weighted"]), flush=True)

    span = null.mean() - floor["weighted"]
    recovery = (null.mean() - observed["weighted"]) / span if span > 0 else float("nan")
    z = (observed["weighted"] - null.mean()) / null.std(ddof=1) if null.std(ddof=1) else float("nan")

    report = {
        "space": "signature poses: canonical joint positions at up to 4 motion beats, "
                 "{}x{}x3 = {} dims".format(BEATS, BODY_JOINTS, vectors.shape[1]),
        "why_this_space": "M2 is K-Means over TMR embeddings; tightness *there* is true "
                          "by construction. These coordinates were never optimised, so "
                          "this reading can come out either way.",
        "pairs": "cross-upload only, in all three arms",
        "segments": int(len(vectors)),
        "prototypes": int(len(sizes)),
        "uploads": int(len(set(uploads))),
        "observed_distance": observed,
        "null_distance_mean": float(null.mean()),
        "null_distance_sd": float(null.std(ddof=1)),
        "null_draws": int(args.null_draws),
        "null": "label vector permuted over segments; every prototype keeps its exact size",
        "floor_distance": floor,
        "floor": "K-Means at the same K fitted directly on these descriptors -- the "
                 "ruler's top end, not a rival to M2",
        "recovery_fraction": float(recovery),
        "z_against_null": float(z),
        "reading": "lower distance is tighter. recovery_fraction is how far M2 travels "
                   "from the no-structure null toward the best a partition of this data "
                   "reaches in this space: 0.0 means the sheets show nothing, 1.0 means "
                   "M2 is as pose-coherent as a clustering fitted on pose directly.",
        "does_not_answer": "whether a prototype is a *movement* (preparation, execution, "
                           "recovery). A tight group of static shapes scores the same.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "observed_distance", "null_distance_mean", "floor_distance",
        "recovery_fraction", "z_against_null")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
