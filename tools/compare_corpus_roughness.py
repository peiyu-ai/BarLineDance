#!/usr/bin/env python3
"""Is the generated motion's gap to ground truth a property of the method or of the corpus?

The wild 340-frame arm's generated motion sits at 3.4x ground truth's jerk and
below it on reach and turn rate.  Two explanations survive that observation and
they call for opposite work:

* the *method* produces rough motion, in which case AIST++ -- where this repo's
  fid_k (23.473) lands in the same band as the paper's own 25.26 / 24.02 -- must
  show the same gap; or
* the *wild corpus* is the difference, its ground truth being a GVHMR
  reconstruction rather than mocap, in which case AIST's gap is small and the
  roughness is something the wild pipeline introduces.

The headline metrics cannot separate these.  ``eval/metrics.py`` standardises
each distribution by its own per-dimension mean and std before FID, so fid_k is
exactly invariant to any per-dimension affine change -- halving every clip's
amplitude leaves it at 0.0.  Roughness and amplitude are therefore unfalsifiable
against the number the paper is compared on, which is why they need this.

Ground truth is decoded from each release's own held-out arrays through the same
inverse-normalisation and forward kinematics the generator's output takes, one
window per recording so overlapping slices cannot vote twice.

Usage::

    python3 tools/compare_corpus_roughness.py \\
        --arm aist:runs/m6_songsplit1286_retrieval/motion:/cache/atomicdance-assets/data/atomic_aistpp/aist_songsplit_llm_release_v1 \\
        --arm wild:runs/t_stride_sweep/s75_seed20260817:/dev/shm/atomicdance-acct/release_w2_340 \\
        --output runs/t8_discovery/corpus_roughness.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from infer_atomic import unnormalize_motion  # noqa: E402
from tools.score_generation_diagnostics import (  # noqa: E402
    _jerk, amplitude_columns, speed_profile,
)

COLUMNS = ("jerk_median", "speed_median_of_frame_mean", "reach_span_shoulders",
           "turn_rate_deg_s", "root_height_span_m", "net_displacement_m")


def _row(pose):
    amp = amplitude_columns(pose)
    speed = speed_profile(pose)
    return {
        "jerk_median": float(np.median(_jerk(pose))),
        "speed_median_of_frame_mean": speed["median_of_frame_mean"],
        "reach_span_shoulders": amp["reach_span_shoulders"],
        "turn_rate_deg_s": amp["turn_rate_deg_s"],
        "root_height_span_m": amp["root_height_span_m"],
        "net_displacement_m": amp["net_displacement_m"],
        "frames": int(len(pose)),
    }


def _generated_rows(directory):
    rows = []
    for path in sorted(pathlib.Path(directory).glob("*.pkl")):
        with open(path, "rb") as handle:
            clip = pickle.load(handle)
        rows.append(_row(np.asarray(clip["full_pose"], dtype=np.float64)))
    if not rows:
        raise SystemExit("error: no .pkl under {}".format(directory))
    return rows


def _ground_truth_rows(release, split, limit):
    from vis import SMPLSkeleton
    from dataset.quaternion import ax_from_6v

    root = pathlib.Path(release)
    motion = np.load(str(root / split / "motion.npy"), mmap_mode="r")
    names = json.load(open(str(root / split / "names.json")))
    first = {}
    for index, name in enumerate(names):
        first.setdefault(name.rsplit("_slice", 1)[0], index)
    chosen = [first[key] for key in sorted(first)][:limit]

    skeleton = SMPLSkeleton()
    rows = []
    for index in chosen:
        raw = unnormalize_motion(
            torch.from_numpy(np.array(motion[index], dtype=np.float32)),
            str(root / "normalizer.pt"))
        rotations = ax_from_6v(raw[:, 7:].reshape(-1, 24, 6))
        pose = skeleton.forward(rotations.unsqueeze(0), raw[:, 4:7].unsqueeze(0))[0]
        rows.append(_row(pose.numpy().astype(np.float64)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True,
                        help="name:generated_dir:release_root")
    parser.add_argument("--split", default="test")
    parser.add_argument("--gt-limit", type=int, default=400,
                        help="recordings drawn for the ground-truth side")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    report = {"split": args.split, "gt_limit": args.gt_limit, "arms": {}}
    for spec in args.arm:
        name, generated, release = spec.split(":", 2)
        gen = _generated_rows(generated)
        truth = _ground_truth_rows(release, args.split, args.gt_limit)
        entry = {"generated_dir": generated, "release": release,
                 "generated_clips": len(gen), "ground_truth_windows": len(truth),
                 "generated": {}, "ground_truth": {}, "ratio": {}}
        for column in COLUMNS:
            g = float(np.median([r[column] for r in gen]))
            t = float(np.median([r[column] for r in truth]))
            entry["generated"][column] = g
            entry["ground_truth"][column] = t
            entry["ratio"][column] = g / t if abs(t) > 1e-9 else None
        entry["generated"]["median_frames"] = float(np.median([r["frames"] for r in gen]))
        entry["ground_truth"]["median_frames"] = float(np.median([r["frames"] for r in truth]))
        report["arms"][name] = entry

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
