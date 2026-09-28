#!/usr/bin/env python3
"""Convert AtomicDance 151-D motion into the Guo 263-D features TMR expects.

The paper's clustering step (M2) encodes each motion segment with TMR's motion
encoder.  TMR's released checkpoint is ``tmr_humanml3d_guoh3dfeats``: it reads
the 263-D HumanML3D representation of Guo et al. at 20 fps, not this repo's
151-D EDGE representation at 30 fps.  This tool is that bridge, and it delegates
the actual feature construction to TMR's own vendored ``joints_to_guofeats`` so
the arithmetic is upstream's, not a reimplementation.

Four conversions happen here, each of which is a place to be wrong:

* **151-D -> joints.**  Decoded through the same forward kinematics the renderer
  and inference use (``vis.SMPLSkeleton``), so a segment embedding describes the
  same body a rendered clip shows.
* **24 -> 22 joints.**  HumanML3D uses SMPL's first 22.  SMPL's joints 22/23 are
  the hands, which GVHMR cannot observe at all and which this corpus already
  records as identity -- so nothing is lost that was ever measured.
* **z-up -> y-up.**  HumanML3D is y-up.  The mapping ``(x, y, z) -> (x, z, -y)``
  is the exact inverse of TMR's own ``guofeats_to_joints`` tail, which returns
  ``(x, -my, z)`` from ``(x, z, my)``; deriving it from their decoder rather
  than guessing is what keeps the corpus off a mirrored skeleton.
* **30 -> 20 fps.**  TMR was trained at 20 fps.  3:2 is not an integer stride,
  so frames are linearly interpolated rather than dropped.

One thing deliberately *not* copied from ``prepare/compute_guoh3dfeats.py``:
its ``joints[..., 0] *= -1`` flip.  That corrects a storage quirk in the
HumanML3D AMASS ``.npy`` files (saved with Y/Z swapped, giving det = -1); joints
produced here come out of forward kinematics already right-handed, so applying
it would mirror the whole corpus.  ``tests/test_convert_motion_to_guofeats.py``
pins chirality against that.

TMR's ``joints_to_guofeats`` retargets onto the HumanML3D skeleton
(``uniform_skeleton``) before computing features, so differing limb lengths
between SMPL-X and HumanML3D are handled by upstream and need no scaling here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

SOURCE_FPS = 30.0
TARGET_FPS = 20.0
GUOFEATS_DIM = 263
GUOFEATS_JOINTS = 22
MOTION_DIM = 151
TMR_ROOT = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "TMR"


class GuofeatsError(RuntimeError):
    pass


def _load_upstream():
    """Import TMR's vendored transform; the arithmetic must be theirs."""
    if not (TMR_ROOT / "guofeats_ref" / "motion_representation.py").is_file():
        raise GuofeatsError(
            "TMR reference transform missing under {}; run tools/setup_tmr_env.sh".format(TMR_ROOT))
    if str(TMR_ROOT) not in sys.path:
        sys.path.insert(0, str(TMR_ROOT))
    from guofeats_ref import guofeats_to_joints, joints_to_guofeats  # noqa: E402

    return joints_to_guofeats, guofeats_to_joints


def motion_151_to_joints(motion: np.ndarray) -> np.ndarray:
    """[T,151] raw (unnormalized) z-up motion -> [T,24,3] joint positions."""
    import torch

    from dataset.quaternion import ax_from_6v
    from vis import SMPLSkeleton

    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != MOTION_DIM:
        raise GuofeatsError("expected [T,{}], got {}".format(MOTION_DIM, motion.shape))
    tensor = torch.from_numpy(motion)
    _contacts, values = torch.split(tensor, (4, 147), dim=-1)
    root_positions = values[:, :3]
    rotations = ax_from_6v(values[:, 3:].reshape(-1, 24, 6))
    joints = SMPLSkeleton().forward(rotations.unsqueeze(0), root_positions.unsqueeze(0))[0]
    return joints.numpy().astype(np.float64)


def zup_to_humanml3d(joints: np.ndarray) -> np.ndarray:
    """(x, y, z) z-up -> (x, z, -y) y-up, the inverse of TMR's decoder tail."""
    joints = np.asarray(joints, dtype=np.float64)
    return np.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], axis=-1)


def humanml3d_to_zup(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    return np.stack([joints[..., 0], -joints[..., 2], joints[..., 1]], axis=-1)


def resample_joints(joints: np.ndarray, source_fps: float, target_fps: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Linear resample along time; returns joints and the source index of each output frame."""
    joints = np.asarray(joints, dtype=np.float64)
    frames = len(joints)
    if frames < 2:
        raise GuofeatsError("need at least 2 frames to resample, got {}".format(frames))
    if source_fps == target_fps:
        return joints, np.arange(frames, dtype=np.float64)
    duration = (frames - 1) / source_fps
    count = int(np.floor(duration * target_fps)) + 1
    if count < 2:
        raise GuofeatsError(
            "sequence of {} frames at {} fps is too short for {} fps".format(
                frames, source_fps, target_fps))
    source_index = np.arange(count, dtype=np.float64) * (source_fps / target_fps)
    lower = np.floor(source_index).astype(int)
    upper = np.minimum(lower + 1, frames - 1)
    weight = (source_index - lower)[:, None, None]
    resampled = joints[lower] * (1.0 - weight) + joints[upper] * weight
    return resampled, source_index


def motion_151_to_guofeats(motion: np.ndarray, *, source_fps: float = SOURCE_FPS,
                           target_fps: float = TARGET_FPS) -> Dict[str, np.ndarray]:
    """Full bridge.  Returns features plus the 30 fps frame each output row came from."""
    joints_to_guofeats, _ = _load_upstream()
    joints = motion_151_to_joints(motion)[:, :GUOFEATS_JOINTS]
    resampled, source_index = resample_joints(joints, source_fps, target_fps)
    h3d_joints = zup_to_humanml3d(resampled)
    features = np.asarray(joints_to_guofeats(h3d_joints.astype(np.float32)), dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != GUOFEATS_DIM:
        raise GuofeatsError("upstream returned {}, expected [T,{}]".format(
            features.shape, GUOFEATS_DIM))
    # Guo features consume one frame to velocities, so row i describes the step
    # from resampled frame i to i+1; keep the source index of the row's start.
    return {
        "features": features,
        "source_frame_index": source_index[: len(features)].astype(np.float32),
        "source_fps": np.asarray([source_fps], dtype=np.float32),
        "target_fps": np.asarray([target_fps], dtype=np.float32),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True,
                        help="raw-performance bundle with sequences.jsonl")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = args.bundle
    rows = [json.loads(line) for line in (bundle / "sequences.jsonl").open(encoding="utf-8")]
    if args.limit is not None:
        rows = rows[: args.limit]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    written = failed = 0
    for row in rows:
        target = output / (row["sequence_id"].replace("/", "_").replace(":", "_") + ".npz")
        if target.exists():
            continue
        try:
            payload = motion_151_to_guofeats(np.load(bundle / row["motion_path"]))
        except (GuofeatsError, IndexError, ValueError) as error:
            print("FAIL {}: {}".format(row["sequence_id"], error), flush=True)
            failed += 1
            continue
        np.savez_compressed(target, sequence_id=row["sequence_id"], **payload)
        written += 1
        if written % 200 == 0:
            print("converted {}".format(written), flush=True)
    print(json.dumps({"written": written, "failed": failed, "output_dir": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
