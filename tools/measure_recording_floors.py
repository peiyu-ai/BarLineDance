#!/usr/bin/env python3
"""Every recording's own floor height, for ``infer_atomic --draft-floor-normalize``.

Parameterised from the scratch script that wrote
``data/wild3d/txy_t_normalized/recording_floors.json`` (2026-09-12; preserved at
``runs/txy_t2_20260922/preserved_from_516f17be/corpus_floors.py``), whose inputs
and output were hard-coded -- run as-is on a new corpus it would have overwritten
the T file.  The rule is unchanged: the FLOOR of a recording is the 5th percentile
of its lowest foot joint (SMPL 7, 8, 10, 11) in metres, the same rule
``anchor_floor`` and ``render_avatar_video.floor_of`` use, computed after
INVERTING the bundle's normalizer -- forward kinematics on min-max-scaled rot6d
returns numbers that are neither metres nor floors, which is what the first
version of the scratch script did.

Why a recording needs one (the operator, 2026-09-12): per-recording median root
height spans 0.329 m across the T corpus, which is calibration, not dancing, and
retrieval pastes bars across recordings.  ``infer_atomic`` refuses to run when
a library sequence has no floor, so this must cover every one.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FOOT = (7, 8, 10, 11)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sequences", required=True, type=pathlib.Path,
                    help="sequences_normalized.jsonl (absolute motion paths)")
    ap.add_argument("--normalizer", required=True, type=pathlib.Path,
                    help="normalizer.pt those sequences were normalized with")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit("{} exists; floors are written to a new file".format(args.out))

    from dataset.quaternion import ax_from_6v
    from vis import SMPLSkeleton

    records = [json.loads(line) for line in open(args.sequences, encoding="utf-8")]
    normalizer = torch.load(str(args.normalizer), map_location="cpu", weights_only=False)
    norm_min = normalizer["data_min"].float()
    norm_max = normalizer["data_max"].float()
    norm_span = torch.where(norm_max == norm_min, torch.ones_like(norm_max), norm_max - norm_min)
    skeleton = SMPLSkeleton()
    floors, missing = {}, []
    for index, row in enumerate(records):
        path = pathlib.Path(row["motion_path"])
        if not path.is_absolute():
            path = args.sequences.resolve().parent / path
        if not path.exists():
            missing.append(row["sequence_id"])
            continue
        motion = torch.from_numpy(np.load(path).astype(np.float32))
        motion = (motion + 1.0) / 2.0 * norm_span + norm_min
        root = motion[:, 4:7]
        rotations = ax_from_6v(motion[:, 7:].reshape(-1, 24, 6))
        with torch.no_grad():
            joints = skeleton.forward(rotations.unsqueeze(0), root.unsqueeze(0))[0]
        low = joints[:, FOOT, 2].min(dim=1).values.numpy()
        floors[row["sequence_id"]] = float(np.percentile(low, 5))
        if index % 50 == 0:
            print("  {}/{}".format(index, len(records)), flush=True)
    if missing:
        raise SystemExit("{} sequences have no motion file, e.g. {}".format(len(missing), missing[:3]))
    values = np.array(list(floors.values()))
    report = {
        "rule": "5th percentile of the lowest of SMPL joints 7, 8, 10, 11, in METRES, computed after "
                "inverting {}".format(args.normalizer),
        "sequences": len(floors),
        "median": float(np.median(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "spread_p5_p95": float(np.percentile(values, 95) - np.percentile(values, 5)),
        "inputs": {"sequences": str(args.sequences), "normalizer": str(args.normalizer)},
        "tool": "tools/measure_recording_floors.py",
        "floors": floors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print("floors: median {:.3f}  p5 {:.3f}  p95 {:.3f}  spread {:.3f} m  -> {}".format(
        report["median"], report["p5"], report["p95"], report["spread_p5_p95"], args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
