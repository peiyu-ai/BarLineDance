#!/usr/bin/env python3
"""Materialise generated motion into the arrays the release scorers read.

M6 reports FID / Div / BAS for a checkpoint, but the paper's R-precision is
computed by ``tools/eval_r_precision.py`` from ``<release>/<split>/{motion,
music,names}`` -- it never opens a checkpoint.  Pointed at a release it scores
the ground truth *inside* that release, which is the vocabulary's ceiling and
not a model's score; ``run_m6_headline.sh`` says exactly that at the point where
the step used to be, and on 2026-08-14 the dry run's "model" R came back
byte-identical to a ground-truth run from three hours earlier.  This tool closes
the gap the only honest way: it puts generated motion into the same arrays so
the same code path scores it.

What it does **not** produce is a training release.  No labels, no splits, no
source manifest, and no normalizer of its own -- the normalizer is copied from
the release the generation was conditioned on, because a bundle normalised
against its own statistics describes a different body from the one the ceiling
was measured on.  ``build.json`` records ``artifact: scoring_bundle`` so a later
reader cannot mistake it for a release.

Three gates, each able to fail:

* **The reconstruction is checked against the artifact's own joints.**
  ``decode_motion`` throws the 151-D array away and stores
  ``smpl_poses``/``smpl_trans``/``contacts``, so it is rebuilt here.  The
  rebuilt array is run back through the same forward kinematics and compared to
  the stored ``full_pose`` -- the array FID was computed from.  Disagreement
  means this bundle is not the motion that was scored, which is a refusal and
  not a warning.
* **The normaliser round trip is checked in the direction the scorer uses it**:
  ``unnormalize(normalize(raw))`` must return ``raw``.  Nothing is clipped to
  [-1, 1]; generated motion may leave the training range, and clipping it would
  move the body rather than report that it left.
* **Music must already be frame-aligned.**  Generation derives its own length
  from the music, so ``len(music) == len(motion)`` holds by construction.  A
  mismatch means this is not the pair that was generated, and padding it to fit
  would put zero-filled audio into the candidate pool.

The window policy is read from the reference release's own ``build.json``
rather than defaulted, because the pool a clip is retrieved from is the whole
split: a bundle windowed at a different stride is not comparable to the ceiling
it is meant to be read against.

Usage:
    python3 tools/materialize_generated_release.py \\
        --motion-dir runs/m6_songsplit630/motion \\
        --reference-release data/atomic_aistpp/aist_songsplit_rg2_v3_release_v1 \\
        --output runs/m6_songsplit630/scoring_bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MOTION_DIM = 151
MUSIC_DIM = 35
# The generated joints come out of the same float32 forward kinematics as the
# stored ones, so the only difference admissible here is float dust.  A metre of
# tolerance would let a genuinely different body through.
JOINT_TOLERANCE_M = 1e-4
NORMALIZER_TOLERANCE = 1e-4


class MaterialisationError(RuntimeError):
    """Raised when the generated motion cannot be scored honestly."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_normalizer(path: pathlib.Path):
    """Load the frozen normalizer under the same contract the scorer enforces."""
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if set(payload) != {"data_min", "data_max"}:
        raise MaterialisationError("normalizer.pt must hold exactly data_min and data_max")
    data_min = np.asarray(payload["data_min"].float().numpy(), dtype=np.float32)
    data_max = np.asarray(payload["data_max"].float().numpy(), dtype=np.float32)
    if data_min.shape != (MOTION_DIM,) or data_max.shape != (MOTION_DIM,):
        raise MaterialisationError("normalizer must be [{}]".format(MOTION_DIM))
    safe_range = np.where(data_max == data_min, np.float32(1.0), data_max - data_min)
    return data_min, data_max, safe_range.astype(np.float32)


def normalize(raw: np.ndarray, data_min: np.ndarray, safe_range: np.ndarray) -> np.ndarray:
    """``apply_motion_normalizer``'s published formula, in its forward direction."""
    return (np.float32(2.0) * (raw.astype(np.float32) - data_min) / safe_range
            - np.float32(1.0)).astype(np.float32)


def unnormalize(normalized: np.ndarray, data_min: np.ndarray,
                safe_range: np.ndarray) -> np.ndarray:
    return ((normalized.astype(np.float32) + np.float32(1.0)) * safe_range / np.float32(2.0)
            + data_min).astype(np.float32)


def rebuild_motion_151(payload: Dict[str, object]) -> np.ndarray:
    """(contacts, trans, axis-angle) -> the 151-D array ``decode_motion`` consumed.

    The layout is not guessed: ``decode_motion`` splits (4, 147), reads the root
    from the first three of the 147 and the 24 rot6d blocks from the rest, so
    the inverse is a concatenation in that order.
    """
    import torch

    from dataset.quaternion import ax_to_6v

    contacts = np.asarray(payload["contacts"], dtype=np.float32)
    trans = np.asarray(payload["smpl_trans"], dtype=np.float32)
    poses = np.asarray(payload["smpl_poses"], dtype=np.float32)
    if contacts.ndim != 2 or contacts.shape[1] != 4:
        raise MaterialisationError("contacts must be [T,4], got {}".format(contacts.shape))
    if trans.shape != (len(contacts), 3):
        raise MaterialisationError("smpl_trans must be [T,3], got {}".format(trans.shape))
    if poses.shape != (len(contacts), 72):
        raise MaterialisationError("smpl_poses must be [T,72], got {}".format(poses.shape))
    rot6d = ax_to_6v(torch.from_numpy(poses).reshape(-1, 24, 3)).reshape(len(poses), 144)
    return np.concatenate([contacts, trans, rot6d.numpy().astype(np.float32)], axis=-1)


def check_against_stored_joints(raw: np.ndarray, stored: np.ndarray) -> float:
    """Forward-kinematic the rebuilt array and compare to the pkl's own joints."""
    from tools.convert_motion_to_guofeats import motion_151_to_joints

    joints = motion_151_to_joints(raw)
    stored = np.asarray(stored, dtype=np.float64)
    if joints.shape != stored.shape:
        raise MaterialisationError(
            "rebuilt joints {} do not match stored full_pose {}".format(
                joints.shape, stored.shape))
    return float(np.abs(joints - stored).max())


_MUSIC_CACHE: Dict[str, np.ndarray] = {}


def music_for(audio_path: pathlib.Path, frames: int) -> np.ndarray:
    """The 35-D features, from the same loader that conditioned the generation.

    Imported rather than reimplemented: a second copy of these three lines is a
    second thing to keep in step with the extractor, and the whole point is that
    this music is the music the model heard.  Memoised because several seeds of
    one song are the common case and the features are a deterministic function
    of the file.
    """
    from infer_atomic import _load_music

    key = str(audio_path)
    if key not in _MUSIC_CACHE:
        _MUSIC_CACHE[key] = _load_music(audio_path).numpy().astype(np.float32)
    features = _MUSIC_CACHE[key]
    if features.shape[1] != MUSIC_DIM:
        raise MaterialisationError("expected {}-D music, got {}".format(
            MUSIC_DIM, features.shape[1]))
    if len(features) != frames:
        raise MaterialisationError(
            "{}: {} music frames against {} motion frames -- generation derives its "
            "length from the music, so this is not the pair that was generated".format(
                audio_path.name, len(features), frames))
    return features


def window_policy(release: pathlib.Path, *, override_length: Optional[int] = None,
                  one_window_per_sequence: bool = False) -> Dict[str, int]:
    """Read the reference release's own window policy; refuse to default it.

    Two overrides, both of which have to be asked for by name, because a bundle
    windowed differently from the ceiling it is read against is not comparable to
    it and nothing downstream can tell:

    ``one_window_per_sequence``
        R-precision's pool is one row per window, and R changes outright with
        pool size -- measured on one AIST set, pool 20 gives 54.0 and pool 128
        gives 7.0, a 7.7x swing on identical data.  So the model-side figure has
        to be scored at a pool the ceiling can also be scored at, and the wild
        arms' published number was taken at one window per generated sequence,
        2,400 rows.  That bundle was built by hand against a stub reference and
        lived in tmpfs; it is gone, and no tool could rebuild it.  This flag is
        that recipe, named.
    ``override_length``
        Only for reading two arms trained at different window lengths on the same
        ruler.  R is comparable across arms only at one clip length, so a
        340-frame arm scored beside a 150-frame one has to be cut to 150 -- the
        cut is the first ``length`` frames of each generated sequence, the same
        frames the 150-frame arm's bundle held.

    Both are recorded in the emitted build.json, so a bundle always says which
    ruler produced it.
    """
    build = json.loads((release / "build.json").read_text(encoding="utf-8"))
    policy = build.get("window_policy") or {}
    length, stride = policy.get("window_length"), policy.get("window_stride")
    if not isinstance(length, int) or not isinstance(stride, int):
        raise MaterialisationError(
            "{}/build.json does not state window_length and window_stride; a bundle "
            "windowed differently from the ceiling it is read against is not "
            "comparable to it".format(release))
    resolved = {"window_length": length, "window_stride": stride,
                "source": "reference release build.json"}
    if override_length is not None:
        if override_length <= 0:
            raise MaterialisationError("--window-length must be positive")
        resolved["window_length"] = int(override_length)
        resolved["window_length_overridden_from"] = length
    if one_window_per_sequence:
        # Larger than any generated sequence, so window_starts yields exactly the
        # first window.  Stated as a sentinel rather than computed per sequence so
        # every row of the bundle is windowed by one declared rule.
        resolved["window_stride"] = 10 ** 9
        resolved["one_window_per_sequence"] = True
        resolved["window_stride_overridden_from"] = stride
    return resolved


def window_starts(frames: int, length: int, stride: int) -> List[int]:
    """Whole windows only -- a short tail is a noisier observation, not a shorter one."""
    if frames < length:
        return []
    return list(range(0, frames - length + 1, stride))


def materialise(motion_dir: pathlib.Path, release: pathlib.Path, output: pathlib.Path,
                split: str, require_headline_eligible: bool = True,
                override_length: Optional[int] = None,
                one_window_per_sequence: bool = False) -> Dict[str, object]:
    policy = window_policy(release, override_length=override_length,
                           one_window_per_sequence=one_window_per_sequence)
    length, stride = policy["window_length"], policy["window_stride"]
    normalizer_path = release / "normalizer.pt"
    data_min, data_max, safe_range = load_normalizer(normalizer_path)

    pkls = sorted(motion_dir.glob("*.pkl"))
    if not pkls:
        raise MaterialisationError("no generated .pkl under {}".format(motion_dir))

    motion_rows: List[np.ndarray] = []
    music_rows: List[np.ndarray] = []
    names: List[str] = []
    sequences: List[Dict[str, object]] = []
    worst_joint_error = 0.0
    worst_normalizer_error = 0.0

    for path in pkls:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        protocol = str(payload.get("generation_protocol", payload.get("plan_source", "")))
        if require_headline_eligible and not payload.get("headline_eligible", False):
            raise MaterialisationError(
                "{} is not headline eligible ({}); scoring an oracle-plan artifact as "
                "a model result is the failure this flag exists to prevent".format(
                    path.name, protocol))

        raw = rebuild_motion_151(payload)
        joint_error = check_against_stored_joints(raw, payload["full_pose"])
        worst_joint_error = max(worst_joint_error, joint_error)
        if joint_error > JOINT_TOLERANCE_M:
            raise MaterialisationError(
                "{}: rebuilt motion differs from the stored full_pose by {:.6f} m -- "
                "this bundle would not be the motion that FID scored".format(
                    path.name, joint_error))

        normalized = normalize(raw, data_min, safe_range)
        round_trip = float(np.abs(unnormalize(normalized, data_min, safe_range) - raw).max())
        worst_normalizer_error = max(worst_normalizer_error, round_trip)
        if round_trip > NORMALIZER_TOLERANCE:
            raise MaterialisationError(
                "{}: normaliser round trip is off by {:.6f}".format(path.name, round_trip))

        audio_path = pathlib.Path(str(payload["audio_path"]))
        music = music_for(audio_path, len(raw))

        starts = window_starts(len(raw), length, stride)
        if not starts:
            raise MaterialisationError(
                "{}: {} frames cannot fill one {}-frame window".format(
                    path.name, len(raw), length))
        for index, start in enumerate(starts):
            motion_rows.append(normalized[start:start + length])
            music_rows.append(music[start:start + length])
            names.append("{}_slice{}".format(path.stem, index))
        sequences.append({
            "pkl": path.name,
            "audio": audio_path.name,
            "frames": int(len(raw)),
            "windows": len(starts),
            "generation_protocol": protocol,
            "max_joint_error_m": round(joint_error, 9),
            "planner_checkpoint": str(payload.get("planner_checkpoint", "")),
            "completion_checkpoint": str(payload.get("completion_checkpoint", "")),
        })
        print("  {}: {} frames -> {} windows (joint error {:.2e} m)".format(
            path.stem, len(raw), len(starts), joint_error), flush=True)

    motion = np.stack(motion_rows).astype(np.float32)
    music = np.stack(music_rows).astype(np.float32)

    staging = pathlib.Path(tempfile.mkdtemp(prefix=".staging-", dir=str(output.parent)))
    try:
        (staging / split).mkdir(parents=True)
        np.save(staging / split / "motion.npy", motion)
        np.save(staging / split / "music.npy", music)
        (staging / split / "names.json").write_text(
            json.dumps(names, indent=2), encoding="utf-8")
        shutil.copy2(normalizer_path, staging / "normalizer.pt")
        build = {
            "artifact": "scoring_bundle",
            "not_a_release": ("no labels, no splits, no source manifest -- this exists so "
                              "release-level scorers can read generated motion, and it must "
                              "never be passed to a trainer"),
            "split": split,
            "motion_dir": str(motion_dir),
            "reference_release": str(release),
            "normalizer_sha256": sha256_file(normalizer_path),
            "window_policy": policy,
            "windows": int(len(motion)),
            "sequences": sequences,
            "gates": {
                "max_joint_error_m": round(worst_joint_error, 9),
                "joint_tolerance_m": JOINT_TOLERANCE_M,
                "max_normalizer_round_trip": round(worst_normalizer_error, 9),
                "normalizer_tolerance": NORMALIZER_TOLERANCE,
                "clipped_to_unit_range": False,
                "motion_outside_training_range_fraction": round(
                    float(np.mean((motion < -1.0) | (motion > 1.0))), 6),
            },
        }
        (staging / "build.json").write_text(json.dumps(build, indent=2), encoding="utf-8")
        # Immutable new directory, atomic rename -- the repo's own convention, and
        # the one an interrupted publish otherwise breaks by leaving a half bundle
        # that every ``[ -e ]`` guard reads as finished.
        if output.exists():
            raise MaterialisationError("{} already exists; bundles are immutable".format(output))
        staging.rename(output)
        staging = None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return build


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--motion-dir", type=pathlib.Path, required=True,
                        help="directory of generated .pkl files (M6's motion/)")
    parser.add_argument("--reference-release", type=pathlib.Path, required=True,
                        help="the release the generation was conditioned on; supplies "
                             "the normalizer and the window policy")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--one-window-per-sequence", action="store_true",
        help="cut exactly one window from each generated sequence. R-precision's "
             "pool is one row per window and R swings 7.7x with pool size, so the "
             "model figure has to be scored at the pool its ceiling was")
    parser.add_argument(
        "--window-length", type=int, default=None,
        help="override the reference release's window length. Only for putting two "
             "arms trained at different lengths on one ruler; recorded in build.json")
    parser.add_argument("--allow-oracle-plans", action="store_true",
                        help="materialise artifacts that are not headline eligible; "
                             "their numbers are not model results")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    build = materialise(args.motion_dir, args.reference_release, args.output,
                        args.split, require_headline_eligible=not args.allow_oracle_plans,
                        override_length=args.window_length,
                        one_window_per_sequence=args.one_window_per_sequence)
    print("{} windows -> {}".format(build["windows"], args.output))
    print("  max joint error {:.2e} m, normaliser round trip {:.2e}".format(
        build["gates"]["max_joint_error_m"], build["gates"]["max_normalizer_round_trip"]))
    print("  {:.2%} of values sit outside the training range (not clipped)".format(
        build["gates"]["motion_outside_training_range_fraction"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
