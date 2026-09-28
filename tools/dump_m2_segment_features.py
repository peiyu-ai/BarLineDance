#!/usr/bin/env python3
"""One table of every drawn segment, so the outlier audit reads bytes once.

The audit asks several independent questions of the same 10,652 segments --
how far each sits from its prototype's centre, whether its 3D is broken, where
it falls in its clip, which account it belongs to.  Computing motion beats and
canonical poses costs a few minutes over 621 MB of motion, and doing it once per
question would be four times the work and, worse, four chances for two answers
to disagree because they were built from slightly different segment sets.

So everything is derived here, in one pass, and written to a single npz.

Two families of columns:

**Pose descriptor** -- the same signature poses the contact sheets draw and
``tools/probe_pose_space_tightness.py`` measures: canonical joint positions at
up to four motion beats.  Identical construction, deliberately, so a claim made
here can be checked against that probe's readings.

**3D health** -- cheap invariants that a working reconstruction must satisfy,
recorded per segment so "the outlier is a broken skeleton" becomes a testable
claim rather than a guess:

* ``head_below_hip_frac`` -- fraction of frames with the head lower than the
  pelvis.  A genuine inversion (handstand, floor work) and a flipped
  reconstruction both raise it; it separates *upright* from *not*, and nothing
  more.  Which of the two it is needs the next two columns.
* ``limb_cv`` -- coefficient of variation of a bone length over the segment.
  A body does not change size.  GVHMR emits a per-frame skeleton, so a limb
  that breathes is the reconstruction failing, not the dancer moving.
* ``max_joint_speed`` -- fastest per-frame joint displacement in shoulder
  widths.  Teleporting joints are tracking failures.
* ``nonfinite`` -- any NaN/inf reaching the array at all.

None of these are thresholded here.  Thresholds are judgements and belong to
whoever makes the claim; this file only records the measurements.

Two different units, and why the flag exists
--------------------------------------------
By default a "segment" here is a **run of equal label**, because that is what
``build_atomic_gallery.collect_members`` groups and therefore what the contact
sheets draw.  That is *not* the unit M2 clustered.  M2 embedded the 14,231 spans
the beat grid produced; ``segments_of`` recovers runs, so wherever M2 gave the
same prototype to two adjacent spans they fuse into one longer segment.

Measured on clean5b5: **1,021 of the 10,652 runs are unions of two or more cache
spans** (834 of two, 115 of three, 48 of four, the rest longer).  Every one of
those runs has a span that matches no embedding key, which is how the fusion was
found -- a join against the TMR cache dropped exactly them.

``--spans-from <embeddings.npz>`` switches the unit to the cache's own spans,
each taking the label its first frame carries.  Use it whenever the analysis has
to line up with the embeddings, and note that the two runs are not interchangeable:
the run-based table answers "are the groups the sheets draw coherent", the
span-based one answers "are the things M2 clustered coherent".
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.cluster_atomics_tmr import build_row_index, resolve_row  # noqa: E402
from tools.convert_motion_to_guofeats import motion_151_to_joints   # noqa: E402
from tools.motion_beats import canonical_pose, find_motion_beats    # noqa: E402
from tools.recluster_atomics_ingroup import segments_of             # noqa: E402

BODY_JOINTS = 22
BEATS = 4
MIN_FRAMES = 4
ROOT_J, HEAD_J = 0, 15
L_SHOULDER, R_SHOULDER = 16, 17
# Bones whose length is fixed in any real body; used only for limb_cv.
BONES = [(0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8), (16, 18), (17, 19)]


def descriptor(joints: np.ndarray) -> np.ndarray:
    beats = find_motion_beats(joints, max_beats=BEATS) or [0]
    poses = [canonical_pose(joints[b])[:BODY_JOINTS] for b in beats[:BEATS]]
    while len(poses) < BEATS:
        poses.append(poses[-1])
    return np.concatenate([pose.reshape(-1) for pose in poses])


def health(joints: np.ndarray) -> tuple:
    scale = np.linalg.norm(joints[:, L_SHOULDER] - joints[:, R_SHOULDER], axis=-1)
    scale = float(np.median(scale)) or 1.0
    head_below = float(np.mean(joints[:, HEAD_J, 2] < joints[:, ROOT_J, 2]))
    cvs = []
    for a, b in BONES:
        length = np.linalg.norm(joints[:, a] - joints[:, b], axis=-1)
        mean = float(length.mean())
        if mean > 1e-6:
            cvs.append(float(length.std() / mean))
    limb_cv = float(np.mean(cvs)) if cvs else float("nan")
    if len(joints) > 1:
        step = np.linalg.norm(np.diff(joints, axis=0), axis=-1).max() / scale
    else:
        step = 0.0
    return head_below, limb_cv, float(step), int(not np.isfinite(joints).all())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path("/dev/shm/atomicdance-m2-review"))
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("/dev/shm/atomicdance-m2-review/"
                                             "segment_features.npz"))
    parser.add_argument("--spans-from", type=pathlib.Path, default=None,
                        help="a TMR embedding cache. With it, segments are the spans "
                             "the cache holds -- the unit M2 actually embedded and "
                             "clustered -- instead of runs of equal label. See the "
                             "note on fusion in this file's docstring.")
    args = parser.parse_args()

    labels_dir = args.root / "labels"
    shuffled_dir = args.root / "labels_shuffled"
    bundle = args.root / "bundle"

    spans_by_recording = None
    if args.spans_from is not None:
        blob = np.load(args.spans_from, allow_pickle=True)
        spans_by_recording = {}
        for rec, s0, e0 in zip(blob["recordings"], blob["starts"], blob["ends"]):
            spans_by_recording.setdefault(str(rec), []).append((int(s0), int(e0)))
        print("spans from {}: {} recordings, {} spans".format(
            args.spans_from.name, len(spans_by_recording),
            sum(len(v) for v in spans_by_recording.values())), flush=True)

    rows = [json.loads(line) for line
            in (labels_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    sequences = {json.loads(line)["recording_id"]: json.loads(line)
                 for line in (bundle / "sequences.jsonl").open(encoding="utf-8")}
    index = build_row_index(sequences)

    out = {k: [] for k in ("vector", "label", "shuffled", "upload", "recording",
                           "start", "end", "clip_frames", "head_below", "limb_cv",
                           "max_step", "nonfinite", "segment_index", "segments_in_clip")}
    for done, row in enumerate(rows, 1):
        sequence = resolve_row(index, row["recording_id"]) or sequences.get(
            row["recording_id"])
        if sequence is None:
            continue
        joints = motion_151_to_joints(np.load(bundle / sequence["motion_path"]))
        real = np.load(labels_dir / row["labels_path"])
        fake = np.load(shuffled_dir / row["labels_path"])
        if spans_by_recording is None:
            drawn = [(s, e, l) for s, e, l in segments_of(real)
                     if l > 0 and e - s >= MIN_FRAMES]
        else:
            # The cache's own spans, each carrying the label its first frame holds.
            # Two adjacent spans with the same label stay two segments here, where
            # segments_of would have fused them into one.
            drawn = []
            for s0, e0 in spans_by_recording.get(row["recording_id"], ()):
                if e0 - s0 < MIN_FRAMES or s0 >= len(real):
                    continue
                lab = int(real[s0])
                if lab > 0:
                    drawn.append((s0, min(e0, len(real)), lab))
            drawn.sort()
        for order, (start, end, label) in enumerate(drawn):
            span = joints[start:end]
            hb, cv, step, nf = health(span)
            out["vector"].append(descriptor(span))
            out["label"].append(int(label))
            # The shuffled tree keeps boundaries, so the same span carries the
            # control's label at the same index.
            out["shuffled"].append(int(fake[start]))
            out["upload"].append(str(row["recording_id"]).rsplit(":", 1)[0])
            out["recording"].append(row["recording_id"])
            out["start"].append(int(start))
            out["end"].append(int(end))
            out["clip_frames"].append(int(len(real)))
            out["segment_index"].append(int(order))
            out["segments_in_clip"].append(int(len(drawn)))
            out["head_below"].append(hb)
            out["limb_cv"].append(cv)
            out["max_step"].append(step)
            out["nonfinite"].append(nf)
        if done % 400 == 0:
            print("  {}/{} recordings, {} segments".format(
                done, len(rows), len(out["label"])), flush=True)

    packed = {
        "vector": np.stack(out["vector"]).astype(np.float32),
        "label": np.array(out["label"], dtype=np.int32),
        "shuffled": np.array(out["shuffled"], dtype=np.int32),
        "upload": np.array(out["upload"]),
        "recording": np.array(out["recording"]),
        "start": np.array(out["start"], dtype=np.int32),
        "end": np.array(out["end"], dtype=np.int32),
        "clip_frames": np.array(out["clip_frames"], dtype=np.int32),
        "segment_index": np.array(out["segment_index"], dtype=np.int32),
        "segments_in_clip": np.array(out["segments_in_clip"], dtype=np.int32),
        "head_below": np.array(out["head_below"], dtype=np.float32),
        "limb_cv": np.array(out["limb_cv"], dtype=np.float32),
        "max_step": np.array(out["max_step"], dtype=np.float32),
        "nonfinite": np.array(out["nonfinite"], dtype=np.int8),
    }
    np.savez_compressed(args.output, **packed)

    # A gate, not a print: the label vectors must agree with the two trees the
    # review already published, or every downstream count is of a different set.
    if args.spans_from is None:
        assert len(packed["label"]) == 10652, len(packed["label"])
    real_sizes = np.bincount(packed["label"])[1:]
    fake_sizes = np.bincount(packed["shuffled"])[1:]
    if args.spans_from is None:
        assert sorted(real_sizes[real_sizes > 0]) == sorted(fake_sizes[fake_sizes > 0]), \
            "control is not size-matched in this dump"
    else:
        # Under --spans-from the shuffled tree is read at the same spans, so its
        # group sizes are a permutation of the *span* assignment and need not match
        # the run-based sizes. What must still hold is that both arms cover the
        # same spans.
        assert len(real_sizes) and len(fake_sizes), "an arm produced no labels"
    print("{} segments, {} prototypes, {} uploads -> {}".format(
        len(packed["label"]), int((real_sizes > 0).sum()),
        len(set(out["upload"])), args.output), flush=True)
    print("head_below>0.5: {}  limb_cv>0.05: {}  nonfinite: {}".format(
        int((packed["head_below"] > 0.5).sum()),
        int((packed["limb_cv"] > 0.05).sum()),
        int(packed["nonfinite"].sum())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
