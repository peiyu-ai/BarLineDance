#!/usr/bin/env python3
"""Prepare camera-decoupled wild-video motions for AtomicDance.

This tool deliberately separates three concerns that were entangled in the
previous Lodge 2D pipeline:

* observations: the DWPose/raw-clean 2D cache remains immutable;
* camera: visual-odometry output is retained as a distinct asset; and
* motion: SMPL pose + world root trajectory is converted to AtomicDance's
  unnormalised 151-D EDGE-compatible representation.

The default backend adapter is WHAM.  Its official custom-video output is a
``wham_output.pkl`` containing ``pose_world`` and ``trans_world``.  Conversion
rejects unprovenanced or camera-space-only output by default: camera/body
disentanglement is a required invariant for the new corpus, not a best-effort
option. DPVO poses are retained only as an *unregistered* visual-odometry
asset; WHAM does not establish a common metric world frame for DPVO and body
translation.

The script has no dependency on PyTorch, SMPL, or a pose-estimation runtime.
WHAM is responsible for fitting SMPL; this adapter only transforms its saved
parameters and can therefore run in the AtomicDance environment.  See
``docs/WILD_3D_PREPROCESSING.md`` for the staged command sequence and license
requirements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA_VERSION = "wild3d-v1"
WHAM_GLOBAL_RUN_SCHEMA_VERSION = "wham-global-run-v1"
GVHMR_PERSON_TRACK_ID = "gvhmr:get_one_track"
GVHMR_SIMPLE_VO = "SimpleVO(sift)"
GVHMR_DPVO = "DPVO"
GVHMR_NO_VO = "none"
ATOMIC_MOTION_DIM = 151
CONTACT_JOINTS = (7, 8, 10, 11)  # EDGE / AtomicDance order: L ankle, R ankle, L toe, R toe.
_WILD_CLIP_ID = re.compile(r"^(?P<recording>.+)__clip(?P<clip_index>\d+)$")

# These are the neutral SMPL offsets and joint tree used by AtomicDance/EDGE
# (``vis.py``).  Keeping them here avoids importing the visualization module,
# which imports PyTorch3D and is intentionally not a preprocessing dependency.
SMPL_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)
SMPL_OFFSETS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.05858135, -0.08228004, -0.01766408],
        [-0.06030973, -0.09051332, -0.01354254],
        [0.00443945, 0.12440352, -0.03838522],
        [0.04345142, -0.38646945, 0.008037],
        [-0.04325663, -0.38368791, -0.00484304],
        [0.00448844, 0.1379564, 0.02682033],
        [-0.01479032, -0.42687458, -0.037428],
        [0.01905555, -0.4200455, -0.03456167],
        [-0.00226458, 0.05603239, 0.00285505],
        [0.04105436, -0.06028581, 0.12204243],
        [-0.03483987, -0.06210566, 0.13032329],
        [-0.0133902, 0.21163553, -0.03346758],
        [0.07170245, 0.11399969, -0.01889817],
        [-0.08295366, 0.11247234, -0.02370739],
        [0.01011321, 0.08893734, 0.05040987],
        [0.12292141, 0.04520509, -0.019046],
        [-0.11322832, 0.04685326, -0.00847207],
        [0.2553319, -0.01564902, -0.02294649],
        [-0.26012748, -0.01436928, -0.03126873],
        [0.26570925, 0.01269811, -0.00737473],
        [-0.26910836, 0.00679372, -0.00602676],
        [0.08669055, -0.01063603, -0.01559429],
        [-0.0887537, -0.00865157, -0.01010708],
    ],
    dtype=np.float64,
)

# EDGE rotates AIST++'s y-up data +90 degrees about X before it builds its
# 151-D representation.  WHAM with ``return_y_up=True`` is likewise y-up, so
# use exactly the same conversion to keep wild data in the released model's
# coordinate convention: (x, y, z) -> (x, -z, y).
Y_UP_TO_EDGE_Z_UP = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()

# Paths used by the official WHAM custom-video demo.  Keeping this preflight
# list explicit prevents an accidentally local-only or randomly initialized
# run from being mistaken for a world-grounded 3D corpus.
WHAM_RUNTIME_CODE = (
    "demo.py",
    "third-party/DPVO",
    "third-party/ViTPose",
)
WHAM_BODY_ASSETS = (
    "dataset/body_models/smpl/SMPL_NEUTRAL.pkl",
    "dataset/body_models/J_regressor_wham.npy",
    "dataset/body_models/J_regressor_h36m.npy",
    "dataset/body_models/J_regressor_feet.npy",
    "dataset/body_models/smpl_mean_params.npz",
)
WHAM_MODEL_ASSETS = (
    "checkpoints/wham_vit_bedlam_w_3dpw.pth.tar",
    "checkpoints/hmr2a.ckpt",
    "checkpoints/dpvo.pth",
    "checkpoints/yolov8x.pt",
    "checkpoints/vitpose-h-multi-coco.pth",
)
_WHAM_WORLD_RUNTIME_CHECK = (
    "from lib.models.preproc.slam import SLAMModel; "
    "from dpvo.dpvo import DPVO; "
    "from lib.models.preproc.detector import DetectionModel; "
    "from lib.models.preproc.extractor import FeatureExtractor"
)


def _stable_path(value: Path) -> Path:
    """Normalised, but neither dereferenced nor forced absolute.

    ``Path.resolve`` does both, and for a path that is going into a manifest
    both are wrong here:

    * It follows symlinks, so ``data/wild3d/ingest_v1_converted`` becomes
      ``/cache/atomicdance-assets/data/wild3d/ingest_v1_converted`` -- the mount
      ``tools/oss_assets.py evict`` is allowed to empty.  The recorded location
      then stops existing the moment the cache is reclaimed, even though the
      tree is still in OSS and one ``pull --cache`` away, which re-creates the
      symlink in place.  That is how the v4 staging manifest died, and why
      stage C of the rebuild driver refuses a manifest holding ``/cache``.
    * It makes the path absolute, which costs the manifest its portability.
      The repo-relative spelling *is* the OSS key: ``data/wild_ingest_v1/<clip>``
      names the checkout path and the object equally, so a relative manifest can
      be published and read back by the OSS-resident stages, whose
      ``publish_rows`` refuses to publish a row containing an absolute path at
      all.  v4's manifests are relative for this reason.

    ``os.path.normpath`` collapses ``..`` and duplicate separators and does
    nothing else, so what is recorded is the path the caller named.
    """
    return Path(os.path.normpath(str(Path(value).expanduser())))


def _as_path(value: Optional[str]) -> Optional[Path]:
    return _stable_path(Path(value)) if value else None


# ``tools/oss_assets.py pull --cache`` puts a tree at
# ``/cache/atomicdance-assets/<repo path>`` and leaves a symlink at the repo
# path, so this prefix is exactly the difference between the two spellings of
# one file.
CACHE_MIRROR_ROOT = "/cache/atomicdance-assets/"


def _uncached(value: str) -> str:
    """Spell a cache-mirror path the way the repo spells it."""
    return value[len(CACHE_MIRROR_ROOT):] if value.startswith(CACHE_MIRROR_ROOT) else value


def _corpus_source_video(
    metadata: Mapping[str, Any],
    clip_id: str,
    upload_root: Optional[Path],
) -> Tuple[Optional[str], Optional[str]]:
    """Name the upload a clip came from in a way that outlives the run.

    ``meta.json``'s ``source`` is whatever file the cutter was pointed at, and
    for the two re-cut families that is a scratch working copy: 2,185 clips
    name ``scratch/c1/refix/uploads/<id>.mp4`` (2026-08-19) and 278 name
    ``scratch/c1/cfr_uploads/<id>.mp4`` (the 2026-08-25 CFR re-encodes).  Both
    directories are working space -- the CFR copies exist only to make
    ``cut_clip`` correct on variable-frame-rate containers -- so a manifest that
    records them names 2,463 files that will not be there later, and stage C in
    ``tools/run_wild_rebuild.sh`` refuses the manifest for it.

    The upload itself does persist, under ``upload_root``.  The mapping used
    here is clip name prefix -> ``<upload_root>/<prefix>.mp4``, and it was
    checked against the corpus rather than assumed (2026-08-25): for all 278
    CFR clips the recorded ``cfr_normalized_from`` de-caches to exactly that
    path, and on 40 sampled clips of the older family the upload's
    ``r_frame_rate`` equals the ``source_fps`` their meta recorded, 40/40.  All
    2,463 uploads are present.

    Returns ``(source_video, working_copy)``: the second is the scratch path
    that was replaced, or ``None`` when nothing was replaced -- dropping it
    would lose the record of which cut a clip came out of.
    """
    recorded = metadata.get("source")
    recorded = str(recorded) if recorded else None
    if not recorded:
        return None, None
    candidate = _uncached(recorded)
    if not candidate.startswith("scratch/") and not candidate.startswith("/"):
        return candidate, None
    if Path(candidate).is_file():
        return candidate, None
    # A re-encode records the file it was made from; that one is a corpus path.
    normalized_from = metadata.get("cfr_normalized_from")
    if normalized_from:
        replacement = _uncached(str(normalized_from))
        if Path(replacement).is_file():
            return replacement, recorded
    if upload_root is not None:
        replacement = upload_root / "{}.mp4".format(clip_id.split("__")[0])
        if replacement.is_file():
            return str(replacement), recorded
    # Nothing better was found.  Report the path as recorded rather than
    # inventing one: a wrong upload is worse than an unreachable one.
    return candidate, None


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_serialized(path: Path) -> Any:
    """Load WHAM/DPVO files saved by joblib, with a pickle fallback."""
    try:
        import joblib  # WHAM's official demo uses joblib.dump.

        return joblib.load(str(path))
    except ImportError:
        pass
    except Exception as joblib_error:
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except Exception as pickle_error:  # pragma: no cover - diagnostic path
            raise RuntimeError(
                "could not load {} as joblib or pickle: {} / {}".format(
                    path, joblib_error, pickle_error
                )
            )
    with path.open("rb") as handle:
        return pickle.load(handle)


def _ensure_array(name: str, value: Any, shape_tail: Tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape_tail) + 1 or tuple(array.shape[1:]) != shape_tail:
        raise ValueError(
            "{} must have shape [T, {}], got {}".format(
                name, ", ".join(str(item) for item in shape_tail), tuple(array.shape)
            )
        )
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("{} is empty or contains non-finite values".format(name))
    return array


def _select_wham_person(
    results: Mapping[Any, Any], person_id: Optional[str]
) -> Tuple[str, Mapping[str, Any]]:
    if not isinstance(results, Mapping) or not results:
        raise ValueError("WHAM output must be a non-empty mapping of track IDs")
    choices = [(str(key), key, value) for key, value in results.items()]
    if person_id is not None:
        selected = [item for item in choices if item[0] == str(person_id)]
        if not selected:
            raise KeyError(
                "person_id {!r} not found; available IDs: {}".format(
                    person_id, ", ".join(item[0] for item in choices)
                )
            )
        _, _, value = selected[0]
        if not isinstance(value, Mapping):
            raise ValueError("selected WHAM track is not a mapping")
        return str(person_id), value
    if len(choices) != 1:
        raise ValueError(
            "WHAM output has {} tracks; pass --person-id explicitly (available: {})".format(
                len(choices), ", ".join(item[0] for item in choices)
            )
        )
    track_id, _, value = choices[0]
    if not isinstance(value, Mapping):
        raise ValueError("selected WHAM track is not a mapping")
    return track_id, value


def matrices_to_rotation6d(matrices: np.ndarray) -> np.ndarray:
    """Match ``pytorch3d.transforms.matrix_to_rotation_6d`` exactly."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must end in [3, 3]")
    return matrices[..., :2, :].reshape(matrices.shape[:-2] + (6,))


def forward_kinematics(
    local_rotations: np.ndarray, root_positions: np.ndarray
) -> np.ndarray:
    """Neutral-SMPL FK used solely to derive EDGE-compatible foot contacts."""
    local_rotations = np.asarray(local_rotations, dtype=np.float64)
    root_positions = np.asarray(root_positions, dtype=np.float64)
    if local_rotations.shape[1:] != (24, 3, 3):
        raise ValueError("local_rotations must have shape [T, 24, 3, 3]")
    if root_positions.shape != (local_rotations.shape[0], 3):
        raise ValueError("root_positions must have shape [T, 3]")

    frames = local_rotations.shape[0]
    positions = np.zeros((frames, 24, 3), dtype=np.float64)
    world_rotations = np.zeros((frames, 24, 3, 3), dtype=np.float64)
    positions[:, 0] = root_positions
    world_rotations[:, 0] = local_rotations[:, 0]
    for joint in range(1, 24):
        parent = int(SMPL_PARENTS[joint])
        positions[:, joint] = positions[:, parent] + np.einsum(
            "tij,j->ti", world_rotations[:, parent], SMPL_OFFSETS[joint]
        )
        world_rotations[:, joint] = np.einsum(
            "tij,tjk->tik", world_rotations[:, parent], local_rotations[:, joint]
        )
    return positions


def contact_labels(
    local_rotations: np.ndarray,
    root_positions: np.ndarray,
    velocity_threshold: float = 0.01,
) -> np.ndarray:
    """Use the exact frame-displacement contact rule from EDGE preprocessing."""
    if velocity_threshold <= 0:
        raise ValueError("velocity_threshold must be positive")
    feet = forward_kinematics(local_rotations, root_positions)[:, CONTACT_JOINTS]
    velocity = np.zeros(feet.shape[:2], dtype=np.float64)
    if len(feet) > 1:
        velocity[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
    return (velocity < velocity_threshold).astype(np.float32)


def _edge_z_up_from_wham(
    pose_world_y_up: np.ndarray, trans_world_y_up: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert WHAM's y-up global SMPL pose to EDGE's z-up representation."""
    pose = _ensure_array("pose_world", pose_world_y_up, (72,)).reshape(-1, 24, 3)
    translation = _ensure_array("trans_world", trans_world_y_up, (3,))
    if len(pose) != len(translation):
        raise ValueError("pose_world and trans_world have different frame counts")

    local_rotations = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    # Body-joint rotations are local and keep their convention.  The root is a
    # global orientation, therefore the coordinate change applies on its left.
    local_rotations[:, 0] = np.einsum(
        "ij,tjk->tik", Y_UP_TO_EDGE_Z_UP, local_rotations[:, 0]
    )
    translation_z_up = np.einsum("ij,tj->ti", Y_UP_TO_EDGE_Z_UP, translation)
    pose_z_up = Rotation.from_matrix(local_rotations.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 24, 3)
    return pose_z_up, translation_z_up, local_rotations


def _normalize_frame_ids(value: Any, expected_frames: int) -> np.ndarray:
    """Validate WHAM track frame IDs without silently rounding or reordering."""
    raw = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(raw) != expected_frames:
        raise ValueError(
            "WHAM frame_ids has {} entries, motion has {} frames".format(len(raw), expected_frames)
        )
    if not np.isfinite(raw).all():
        raise ValueError("WHAM frame_ids contains non-finite values")
    normalized = np.rint(raw).astype(np.int64)
    if not np.array_equal(raw, normalized.astype(np.float64)):
        raise ValueError("WHAM frame_ids must be exact integer frame indices")
    if len(normalized) and normalized[0] < 0:
        raise ValueError("WHAM frame_ids cannot be negative")
    if len(normalized) > 1 and np.any(np.diff(normalized) <= 0):
        raise ValueError("WHAM frame_ids must be strictly increasing")
    return normalized


def _frame_id_summary(frame_ids: np.ndarray) -> Dict[str, Any]:
    frame_ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    if not len(frame_ids):
        raise ValueError("frame_ids cannot be empty")
    gaps = np.diff(frame_ids)
    missing = int(np.maximum(gaps - 1, 0).sum()) if len(gaps) else 0
    return {
        "frames": int(len(frame_ids)),
        "start_frame": int(frame_ids[0]),
        "end_frame_inclusive": int(frame_ids[-1]),
        "span_frames": int(frame_ids[-1] - frame_ids[0] + 1),
        "is_contiguous": bool(np.all(gaps == 1)) if len(gaps) else True,
        "missing_frames_within_span": missing,
    }


def convert_wham_result(
    result: Mapping[str, Any], velocity_threshold: float = 0.01
) -> Dict[str, np.ndarray]:
    """Convert one WHAM track into lossless, AtomicDance-compatible arrays."""
    if "pose_world" not in result or "trans_world" not in result:
        present = ", ".join(sorted(str(key) for key in result.keys()))
        raise ValueError(
            "global WHAM fields pose_world/trans_world are required; found {}. "
            "Do not use camera-space-only motion for the wild corpus.".format(present)
        )
    pose_z_up, root_z_up, rotations_z_up = _edge_z_up_from_wham(
        result["pose_world"], result["trans_world"]
    )
    frame_ids = _normalize_frame_ids(
        result.get("frame_ids", np.arange(len(root_z_up))), len(root_z_up)
    )
    contacts = contact_labels(rotations_z_up, root_z_up, velocity_threshold)
    # EDGE's contact proxy is a one-frame displacement.  A gap makes the
    # displacement across that boundary physically meaningless.  Preserve a
    # finite 151-D candidate for inspection, but mark it invalid and let the
    # corpus validator reject non-contiguous tracks from training.
    contact_valid_mask = np.ones_like(contacts, dtype=bool)
    if len(frame_ids) > 1:
        gap_indices = np.flatnonzero(np.diff(frame_ids) != 1)
        if len(gap_indices):
            contacts[gap_indices] = 0.0
            contact_valid_mask[gap_indices] = False
    rotation6d = matrices_to_rotation6d(rotations_z_up)
    motion_151 = np.concatenate(
        (contacts, root_z_up.astype(np.float32), rotation6d.astype(np.float32).reshape(len(root_z_up), -1)),
        axis=-1,
    ).astype(np.float32)
    if motion_151.shape[1] != ATOMIC_MOTION_DIM:
        raise AssertionError("expected {} features, got {}".format(ATOMIC_MOTION_DIM, motion_151.shape[1]))
    return {
        "motion_151": motion_151,
        "pose_axis_angle_z_up": pose_z_up.astype(np.float32),
        "root_translation_z_up": root_z_up.astype(np.float32),
        "rotation6d_z_up": rotation6d.astype(np.float32),
        "body_rotation6d_local": rotation6d[:, 1:].astype(np.float32),
        "root_rotation6d_world": rotation6d[:, 0].astype(np.float32),
        "contacts": contacts.astype(np.float32),
        "contact_valid_mask": contact_valid_mask,
        "frame_ids": frame_ids,
        "betas": np.asarray(result.get("betas", []), dtype=np.float32),
    }


def motion_quality_summary(converted: Mapping[str, np.ndarray], fps: float) -> Dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    root = np.asarray(converted["root_translation_z_up"], dtype=np.float64)
    rotations = np.asarray(converted["rotation6d_z_up"], dtype=np.float64)
    contacts = np.asarray(converted["contacts"], dtype=np.float64)
    frame_ids = _normalize_frame_ids(converted["frame_ids"], len(root))
    frame_summary = _frame_id_summary(frame_ids)
    contact_valid_mask = np.asarray(
        converted.get("contact_valid_mask", np.ones_like(contacts, dtype=bool)), dtype=bool
    )
    if contact_valid_mask.shape != contacts.shape:
        raise ValueError("contact_valid_mask must have shape [T, 4]")
    if len(root) > 1:
        root_speed = np.linalg.norm(np.diff(root, axis=0), axis=-1) * fps
        rotation_speed = np.linalg.norm(np.diff(rotations, axis=0), axis=(-2, -1)) * fps
    else:
        root_speed = np.zeros(1, dtype=np.float64)
        rotation_speed = np.zeros((1, 24), dtype=np.float64)
    return {
        "schema_version": SCHEMA_VERSION,
        "frames": int(len(root)),
        "fps": float(fps),
        "frame_alignment": frame_summary,
        "finite": bool(np.isfinite(root).all() and np.isfinite(rotations).all()),
        "root_speed_median": float(np.median(root_speed)),
        "root_speed_p95": float(np.quantile(root_speed, 0.95)),
        "joint_rotation_speed_median": float(np.median(rotation_speed)),
        "joint_rotation_speed_p95": float(np.quantile(rotation_speed, 0.95)),
        "contact_fraction": [float(item) for item in contacts.mean(axis=0)],
        "contact_valid_fraction": float(contact_valid_mask.mean()),
        "contact_order": ["left_ankle", "right_ankle", "left_toe", "right_toe"],
        "status": "candidate" if frame_summary["is_contiguous"] else "quarantine",
        "notes": [
            "Contact is the EDGE neutral-SMPL velocity proxy; inspect it before using as a supervision target.",
            "World scale and floor orientation from monocular WHAM remain estimates and require corpus-level QC.",
        ],
    }


def _is_wham_local_only_sentinel(raw: np.ndarray) -> bool:
    """Detect the exact all-frame fallback written by official WHAM demo.py."""
    sentinel = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return bool(len(raw) and np.allclose(raw, sentinel[None, :], rtol=0.0, atol=1e-8))


def _load_slam_camera(path: Path, frame_ids: np.ndarray) -> Dict[str, np.ndarray]:
    """Select a WHAM track's camera rows from the full-video DPVO trajectory.

    DPVO output is deliberately not rotated into the body coordinate system.
    WHAM only consumes its angular velocity and does not register DPVO's SE(3)
    gauge to ``trans_world``.  Treat this file as camera audit evidence, not a
    world-space training feature.
    """
    raw_full = np.asarray(_load_serialized(path), dtype=np.float64)
    if raw_full.ndim != 2 or raw_full.shape[1] != 7:
        raise ValueError("WHAM DPVO trajectory must have shape [T, 7], got {}".format(tuple(raw_full.shape)))
    if not np.isfinite(raw_full).all():
        raise ValueError("camera trajectory contains non-finite values")
    if _is_wham_local_only_sentinel(raw_full):
        raise ValueError(
            "slam_results.pth is WHAM's local-only/SLAM-fallback sentinel; "
            "rerun from a fresh global WHAM job and do not convert this result"
        )
    frame_ids = _normalize_frame_ids(frame_ids, len(frame_ids))
    if int(frame_ids[-1]) >= len(raw_full):
        raise ValueError(
            "camera trajectory has {} full-video frames but track requires frame {}. "
            "Provide the matching full-video slam_results.pth, not a cropped camera cache.".format(
                len(raw_full), int(frame_ids[-1])
            )
        )
    raw_track = raw_full[frame_ids]
    # WHAM/DPVO stores c2w as [tx, ty, tz, qx, qy, qz, qw].  Translation is
    # visual-odometry up-to-scale in an unregistered gauge.  Do not call this
    # y-up or z-up: neither relation to WHAM's body world is established.
    return {
        "dpvo_c2w_unregistered_full_video": raw_full.astype(np.float32),
        "dpvo_c2w_unregistered_track": raw_track.astype(np.float32),
        "camera_frame_ids": frame_ids.astype(np.int64),
        "full_video_frame_count": np.asarray([len(raw_full)], dtype=np.int64),
    }


def _source_cache_frame_alignment(
    source_cache: Optional[Path], frame_ids: np.ndarray
) -> Dict[str, Any]:
    """Record whether WHAM's timeline can be mapped back to Lodge 2D frames."""
    summary = _frame_id_summary(frame_ids)
    if source_cache is None:
        return {
            "available": False,
            "status": "not_provided",
            "track_frame_alignment": summary,
        }
    source_cache = _stable_path(source_cache)
    pose_path = _choose_pose_file(source_cache)
    if pose_path is None:
        return {
            "available": False,
            "status": "source_pose_missing",
            "source_cache": str(source_cache),
            "track_frame_alignment": summary,
        }
    try:
        poses = np.load(str(pose_path), mmap_mode="r")
        if poses.ndim < 1:
            raise ValueError("pose array has no frame dimension")
        source_frames = int(poses.shape[0])
    except Exception as error:
        return {
            "available": False,
            "status": "source_pose_unreadable",
            "source_cache": str(source_cache),
            "source_pose": str(_stable_path(pose_path)),
            "error": str(error),
            "track_frame_alignment": summary,
        }
    within_source = bool(summary["start_frame"] >= 0 and summary["end_frame_inclusive"] < source_frames)
    return {
        "available": True,
        "status": "aligned" if within_source else "out_of_source_range",
        "source_cache": str(source_cache),
        "source_pose": str(_stable_path(pose_path)),
        "source_frames": source_frames,
        "track_frame_alignment": summary,
        "track_frame_ids_within_source": within_source,
        "track_covers_full_source": bool(
            within_source
            and summary["is_contiguous"]
            and summary["start_frame"] == 0
            and summary["frames"] == source_frames
        ),
    }


def _global_run_marker_path(result_dir: Path) -> Path:
    return result_dir / "wild3d_wham_global_run.json"


def record_wham_global_run(
    result_dir: Path,
    *,
    video: Path,
    wham_root: Path,
) -> Dict[str, Any]:
    """Create an audit marker immediately before a fresh global WHAM run.

    WHAM silently reuses ``tracking_results.pth``/``slam_results.pth``.  A
    result without this marker could therefore have been produced in local-only
    mode.  We never remove these files automatically: an operator must inspect
    or explicitly clear a stale partial output before rerunning it.
    """
    result_dir = result_dir.resolve()
    video = video.resolve()
    wham_root = wham_root.resolve()
    if not video.is_file():
        raise FileNotFoundError("source video does not exist: {}".format(video))
    if not (wham_root / "demo.py").is_file():
        raise FileNotFoundError("WHAM demo.py not found under {}".format(wham_root))
    marker = _global_run_marker_path(result_dir)
    stale = [
        name
        for name in ("tracking_results.pth", "slam_results.pth", "wham_output.pkl")
        if (result_dir / name).exists()
    ]
    if stale:
        raise RuntimeError(
            "refusing to reuse potentially local-only WHAM cache at {}: {}. "
            "Inspect and explicitly clear/move it before a fresh global run.".format(
                result_dir, ", ".join(stale)
            )
        )
    expected = {
        "schema_version": WHAM_GLOBAL_RUN_SCHEMA_VERSION,
        "status": "launch_authorized",
        "global_requested": True,
        "estimate_local_only": False,
        "fresh_cache_verified": True,
        "result_dir": str(result_dir),
        "expected_wham_output": str((result_dir / "wham_output.pkl").resolve()),
        "source_video": str(video),
        "wham_root": str(wham_root),
        "camera_requirement": "full-video DPVO slam_results.pth; local-only fallback is rejected at conversion",
    }
    if marker.exists():
        current, error = _read_meta(marker)
        if error or current != expected:
            raise FileExistsError(
                "existing run provenance differs from this requested job: {}".format(marker)
            )
        return current
    _json_dump(marker, expected)
    return expected


def load_wham_global_run_provenance(
    path: Path,
    *,
    wham_output: Path,
    source_video: Optional[Path],
) -> Dict[str, Any]:
    """Require the fresh-global-run marker before converting a WHAM result."""
    path = path.resolve()
    payload, error = _read_meta(path)
    if error:
        raise ValueError(error)
    if payload.get("schema_version") != WHAM_GLOBAL_RUN_SCHEMA_VERSION:
        raise ValueError("run provenance is not a {} marker".format(WHAM_GLOBAL_RUN_SCHEMA_VERSION))
    if payload.get("status") != "launch_authorized":
        raise ValueError("run provenance has unexpected status {!r}".format(payload.get("status")))
    if payload.get("global_requested") is not True or payload.get("estimate_local_only") is not False:
        raise ValueError("run provenance does not attest a global, non-local-only WHAM request")
    if payload.get("fresh_cache_verified") is not True:
        raise ValueError("run provenance does not attest a fresh WHAM cache")
    expected_result = Path(str(payload.get("expected_wham_output", ""))).expanduser().resolve()
    if expected_result != wham_output.resolve():
        raise ValueError(
            "run provenance expects {}, not {}".format(expected_result, wham_output.resolve())
        )
    if source_video is not None:
        expected_video = Path(str(payload.get("source_video", ""))).expanduser().resolve()
        if expected_video != source_video.resolve():
            raise ValueError(
                "run provenance source video {} does not match {}".format(
                    expected_video, source_video.resolve()
                )
            )
    return payload


def save_converted_wham(
    converted: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    track_id: str,
    input_path: Path,
    fps: float,
    source_cache: Optional[Path] = None,
    source_video: Optional[Path] = None,
    slam_path: Optional[Path] = None,
    run_provenance: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    output_dir = output_dir.resolve()
    marker = output_dir / "metadata.json"
    if marker.exists() and not overwrite:
        raise FileExistsError("{} already exists; pass --overwrite after inspection".format(marker))
    if slam_path is not None and run_provenance is None:
        raise ValueError(
            "camera-aware conversion requires fresh global WHAM run provenance; "
            "use convert-wham --run-provenance or load_wham_global_run_provenance"
        )
    if run_provenance is not None and (
        run_provenance.get("schema_version") != WHAM_GLOBAL_RUN_SCHEMA_VERSION
        or run_provenance.get("global_requested") is not True
        or run_provenance.get("estimate_local_only") is not False
        or run_provenance.get("fresh_cache_verified") is not True
    ):
        raise ValueError("run_provenance is not a valid fresh global WHAM marker")

    frame_ids = _normalize_frame_ids(converted["frame_ids"], len(converted["motion_151"]))
    camera: Optional[Dict[str, np.ndarray]] = None
    if slam_path is not None:
        camera = _load_slam_camera(slam_path, frame_ids)
    source_alignment = _source_cache_frame_alignment(source_cache, frame_ids)
    quality = motion_quality_summary(converted, fps)
    quality["source_frame_alignment"] = source_alignment
    if camera is not None:
        quality["camera_frame_alignment"] = {
            "status": "selected_from_full_video_by_track_frame_ids",
            "full_video_frames": int(camera["full_video_frame_count"][0]),
            "track_frames": int(len(frame_ids)),
        }
    if source_alignment.get("available") and not source_alignment.get("track_frame_ids_within_source"):
        quality["status"] = "quarantine"
        quality["notes"].append("WHAM track frame IDs fall outside the source 2D cache timeline.")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "atomic_motion_151.npy", converted["motion_151"])
    np.save(output_dir / "pose_axis_angle_z_up.npy", converted["pose_axis_angle_z_up"])
    np.save(output_dir / "root_translation_z_up.npy", converted["root_translation_z_up"])
    np.save(output_dir / "rotation6d_z_up.npy", converted["rotation6d_z_up"])
    np.save(output_dir / "body_rotation6d_local.npy", converted["body_rotation6d_local"])
    np.save(output_dir / "root_rotation6d_world.npy", converted["root_rotation6d_world"])
    np.save(output_dir / "contacts.npy", converted["contacts"])
    np.save(output_dir / "contact_valid_mask.npy", converted["contact_valid_mask"])
    np.save(output_dir / "frame_ids.npy", frame_ids)
    if len(converted["betas"]):
        np.save(output_dir / "betas.npy", converted["betas"])

    camera_available = camera is not None
    if camera is not None:
        np.savez_compressed(output_dir / "camera.npz", **camera)

    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backend": "wham",
        "backend_track_id": str(track_id),
        "input_wham_output": str(input_path.resolve()),
        "source_cache": str(_stable_path(source_cache)) if source_cache else None,
        "source_video": str(_stable_path(source_video)) if source_video else None,
        "frames": int(len(converted["motion_151"])),
        "fps": float(fps),
        "atomicdance_representation": "[4 contacts, 3 z-up world root translation, 24x6D rotations]",
        "coordinate_convention": {
            "input": "WHAM pose_world/trans_world, y-up world coordinates",
            "output": "EDGE/AtomicDance z-up; +90 degrees about X, (x,y,z)->(x,-z,y)",
            "root_orientation": "world/global root rotation after y-up -> z-up coordinate conversion",
            "body_orientation": "SMPL local joint rotations",
            "camera_relation_to_body_world": "not registered; never coordinate-converted with body motion",
        },
        "camera": {
            "available": camera_available,
            "asset": "camera.npz" if camera_available else None,
            "input_slam_results": str(slam_path.resolve()) if slam_path else None,
            "representation": "[tx, ty, tz, qx, qy, qz, qw] DPVO c2w",
            "interpretation": (
                "raw DPVO camera-to-world in an unregistered visual-odometry gauge; "
                "translation is up-to-scale and is not y-up/z-up body world"
            ),
            "registered_to_body_world": False,
            "training_role": "audit_only",
        },
        "source_frame_alignment": source_alignment,
        "global_run_provenance": dict(run_provenance) if run_provenance is not None else None,
        "raw_observations": "kept at source_cache; never overwritten by this converter",
        "training_eligibility": "candidate_only_until_reprojection_and_corpus_QC_pass",
    }
    _json_dump(output_dir / "quality.json", quality)
    _json_dump(marker, metadata)
    return metadata


def _read_meta(path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping):
            return dict(payload), None
        return {}, "meta.json is not a JSON object"
    except Exception as error:  # pragma: no cover - malformed source data path
        return {}, "cannot read meta.json: {}".format(error)


def _choose_pose_file(cache_dir: Path) -> Optional[Path]:
    for name in ("keypoints_clean2.npy", "keypoints_clean.npy", "keypoints.npy"):
        candidate = cache_dir / name
        if candidate.is_file():
            return candidate
    return None


def _cache_metrics(
    keypoints: np.ndarray,
    scores: Optional[np.ndarray],
    min_score: float,
) -> Dict[str, Any]:
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if keypoints.ndim != 3 or keypoints.shape[-1] != 2:
        raise ValueError("keypoints must have shape [T, J, 2], got {}".format(tuple(keypoints.shape)))
    finite = np.isfinite(keypoints).all(axis=-1)
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != keypoints.shape[:2]:
            raise ValueError(
                "scores shape {} does not match keypoints {}".format(
                    tuple(scores.shape), tuple(keypoints.shape[:2])
                )
            )
        visible = finite & (scores >= min_score)
        finite_scores = scores[np.isfinite(scores)]
        median_score: Optional[float] = float(np.median(finite_scores)) if len(finite_scores) else None
    else:
        visible = finite
        median_score = None
    # Movement is only measurable on joints the detector actually saw in *both*
    # frames.  An unseen joint has no position, and whatever stands in for it --
    # NaN zeroed here, or the frame centre the old pipeline substituted -- is the
    # same value every frame, so it reads as a joint that did not move.  Left
    # unmasked, "frozen" silently becomes a second copy of "not visible":
    # measured on the rebuilt corpus, every one of the 49 quarantined clips
    # tripped both thresholds, and none of them had a stuck tracker -- they had
    # half their joints below the score threshold.  Two names for one defect
    # makes the accounting say a corpus has two problems when it has one.
    if len(keypoints) > 1:
        measurable = finite[:-1] & finite[1:]
        diffs = np.linalg.norm(np.diff(np.nan_to_num(keypoints), axis=0), axis=-1)
        movement = diffs[measurable]
        frozen_fraction = float(np.mean(movement < 1e-7)) if movement.size else 0.0
    else:
        diffs = np.zeros((0, keypoints.shape[1]))
        movement = np.zeros(0)
        frozen_fraction = 0.0
    return {
        "frames": int(keypoints.shape[0]),
        "joints": int(keypoints.shape[1]),
        "finite_joint_fraction": float(finite.mean()),
        "visible_joint_fraction": float(visible.mean()),
        "median_score": median_score,
        "median_2d_frame_displacement": float(np.median(movement)) if movement.size else 0.0,
        "frozen_joint_pair_fraction": frozen_fraction,
        "coordinate_range": [float(np.nanmin(keypoints)), float(np.nanmax(keypoints))],
    }


def inventory_wild_cache(
    cache_root: Path,
    *,
    recursive: bool,
    min_frames: int,
    min_visible_fraction: float,
    min_score: float,
    max_frozen_fraction: float,
    exclude: Optional[Iterable[str]] = None,
    upload_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Every clip directory under ``cache_root``, minus the names in ``exclude``.

    **This function enumerates a directory, and on this corpus that is not the
    same as enumerating the corpus.**  Nothing deletes a clip directory: an
    upload that used to yield three clips and now yields two leaves the third on
    disk, identical in every respect to a current one except that no run
    produces it, and the credentials here cannot remove it (``ossutil rm``
    answers 403 AccessDenied).  Measured 2026-08-25 on the live ingest tree:
    17,985 directories carry a ``meta.json`` while the manifests produce 17,225
    -- **760 orphans** a glob hands back forever.

    So ``exclude`` is not an optimisation.  Without it a rebuild silently
    re-admits clips whose 3D, features and music were all built from bytes the
    corpus no longer produces, and every hash among them agrees, so no
    freshness gate downstream can see it.  The list comes from
    ``tools/refix_wild_fps_clips_census.py`` (per re-cut) or from the ingest
    manifests directly; passing nothing keeps the old behaviour and is recorded
    in the summary as such rather than left to be inferred.
    """
    if not cache_root.is_dir():
        raise FileNotFoundError("cache root does not exist: {}".format(cache_root))
    excluded = {name.strip() for name in (exclude or ()) if name.strip()}
    iterator: Iterable[Path] = cache_root.rglob("meta.json") if recursive else cache_root.glob("*/meta.json")
    records: List[Dict[str, Any]] = []
    for meta_path in sorted(iterator):
        cache_dir = meta_path.parent
        if cache_dir.name in excluded:
            continue
        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "cache_dir": str(_stable_path(cache_dir)),
            "clip_id": cache_dir.name,
            "status": "quarantine",
            "reasons": [],
        }
        metadata, metadata_error = _read_meta(meta_path)
        source_video, working_copy = _corpus_source_video(metadata, cache_dir.name, upload_root)
        record["source_video"] = source_video
        if working_copy is not None:
            record["source_video_working_copy"] = working_copy
        record["fps"] = float(metadata.get("fps", 30.0) or 30.0)
        record["metadata"] = metadata
        if metadata_error:
            record["reasons"].append(metadata_error)
        pose_path = _choose_pose_file(cache_dir)
        score_path = cache_dir / "scores.npy"
        record["pose2d_path"] = str(_stable_path(pose_path)) if pose_path else None
        record["scores_path"] = str(_stable_path(score_path)) if score_path.is_file() else None
        if pose_path is None:
            record["reasons"].append("no keypoints_clean2.npy, keypoints_clean.npy, or keypoints.npy")
            records.append(record)
            continue
        try:
            keypoints = np.load(str(pose_path), mmap_mode="r")
            scores = np.load(str(score_path), mmap_mode="r") if score_path.is_file() else None
            metrics = _cache_metrics(keypoints, scores, min_score)
            record["pose2d_metrics"] = metrics
        except Exception as error:
            record["reasons"].append("cannot load 2D observations: {}".format(error))
            records.append(record)
            continue
        if metrics["frames"] < min_frames:
            record["reasons"].append("too short: {} < {} frames".format(metrics["frames"], min_frames))
        if metrics["visible_joint_fraction"] < min_visible_fraction:
            record["reasons"].append(
                "visible fraction {:.3f} < {:.3f}".format(
                    metrics["visible_joint_fraction"], min_visible_fraction
                )
            )
        if metrics["frozen_joint_pair_fraction"] > max_frozen_fraction:
            record["reasons"].append(
                "frozen fraction {:.3f} > {:.3f}".format(
                    metrics["frozen_joint_pair_fraction"], max_frozen_fraction
                )
            )
        if not record["reasons"]:
            record["status"] = "ready_for_wham"
        records.append(record)
    return records


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("invalid JSONL at {}:{}: {}".format(path, line_number, error))
            if not isinstance(value, Mapping):
                raise ValueError("JSONL item at {}:{} is not an object".format(path, line_number))
            records.append(dict(value))
    return records


def _wild_identity(clip_id: str, corpus: str) -> Tuple[str, str, str]:
    """Map a Lodge-style clip ID to stable recording and sequence identities.

    ``<raw-id>__clipNNN`` is a temporal crop of one uploaded recording, not a
    new retrieval/split unit.  Preserve that distinction before WHAM runs so
    all crops of the same original video stay on one side of any future split.
    """
    if not corpus or any(char.isspace() for char in corpus):
        raise ValueError("corpus must be a non-empty, whitespace-free identifier")
    match = _WILD_CLIP_ID.match(clip_id)
    if match is None:
        raw_recording = clip_id
        clip_token = "clip000"
    else:
        raw_recording = match.group("recording")
        clip_token = "clip{}".format(match.group("clip_index"))
    recording_id = "{}:{}".format(corpus, raw_recording)
    sequence_id = "{}:{}".format(recording_id, clip_token)
    return recording_id, sequence_id, raw_recording


def _stable_source_splits(
    recording_ids: Sequence[str],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> Dict[str, str]:
    """Assign exact source-level train/val/test counts in a stable hash order."""
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be below 1")
    unique_ids = sorted(set(recording_ids))
    if len(unique_ids) != len(recording_ids):
        raise ValueError("recording IDs must be unique before source splitting")
    ranked = sorted(
        unique_ids,
        key=lambda recording_id: (
            hashlib.sha256("{}:{}".format(seed, recording_id).encode("utf-8")).hexdigest(),
            recording_id,
        ),
    )
    train_count = int(len(ranked) * train_fraction)
    val_count = int(len(ranked) * val_fraction)
    assignments: Dict[str, str] = {}
    for index, recording_id in enumerate(ranked):
        assignments[recording_id] = (
            "train" if index < train_count else "val" if index < train_count + val_count else "test"
        )
    return assignments


def build_wild_staging_manifests(
    records: Sequence[Mapping[str, Any]],
    *,
    corpus: str,
    split_seed: int,
    train_fraction: float,
    val_fraction: float,
    ready_only: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Materialize pre-HMR recording/sequence manifests from inventory JSONL.

    These are deliberately *staging* manifests, not training data: a 2D cache
    can establish source identity and a contiguous input-video timeline, but
    cannot establish 3D quality, duplicate-content equivalence, or labels.
    Their split is deterministic and source-level, yet explicitly marked
    provisional until duplicate-content and 3D QC are complete.
    """
    by_recording: Dict[str, List[Dict[str, Any]]] = {}
    sequence_ids: set[str] = set()
    excluded_status: Dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        if ready_only and status != "ready_for_wham":
            excluded_status[status] = excluded_status.get(status, 0) + 1
            continue
        raw_clip_id = record.get("clip_id")
        if not isinstance(raw_clip_id, str) or not raw_clip_id:
            raise ValueError("inventory record lacks a non-empty clip_id")
        recording_id, sequence_id, raw_recording = _wild_identity(raw_clip_id, corpus)
        if sequence_id in sequence_ids:
            raise ValueError("duplicate sequence_id in inventory: {}".format(sequence_id))
        sequence_ids.add(sequence_id)
        metrics = record.get("pose2d_metrics", {})
        metadata = record.get("metadata", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        frames_value = metrics.get("frames", metadata.get("num_frames"))
        try:
            frames = int(frames_value)
        except (TypeError, ValueError):
            raise ValueError("{} lacks a valid 2D frame count".format(raw_clip_id))
        if frames < 1:
            raise ValueError("{} has non-positive 2D frame count {}".format(raw_clip_id, frames))
        try:
            fps = float(record.get("fps", metadata.get("fps", 30.0)) or 30.0)
        except (TypeError, ValueError):
            raise ValueError("{} has invalid fps".format(raw_clip_id))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("{} has invalid fps {}".format(raw_clip_id, fps))
        source_cache = record.get("cache_dir")
        source_video = record.get("source_video")
        sequence = {
            "schema_version": "atomic-sequence-v1",
            "stage": "pre_hmr_candidate",
            "corpus": corpus,
            "recording_id": recording_id,
            "sequence_id": sequence_id,
            "retrieval_group_id": recording_id,
            "duplicate_content_group_id": None,
            "legacy_clip_id": raw_clip_id,
            "source_recording_key": raw_recording,
            "person_track_id": None,
            "timeline": {
                "fps": fps,
                "source_start_frame": 0,
                "source_end_frame_exclusive": frames,
                "frame_count": frames,
                "frame_ids_path": None,
                "input_video_frames_are_contiguous": True,
                "motion_frames_are_contiguous": None,
            },
            "assets": {
                "source_video": str(source_video) if source_video else None,
                "pose2d": str(record.get("pose2d_path")) if record.get("pose2d_path") else None,
                "pose2d_scores": str(record.get("scores_path")) if record.get("scores_path") else None,
                "source_cache": str(source_cache) if source_cache else None,
                "motion_151_raw": None,
                "camera": None,
            },
            "representation": {
                "motion": None,
                "coordinate_system": None,
                "camera_in_model_input": False,
            },
            "qc": {
                "inventory_status": status,
                "hmr_status": "pending",
                "accepted_for_training": False,
                "reason_codes": list(record.get("reasons", [])),
            },
        }
        by_recording.setdefault(recording_id, []).append(sequence)

    sequences: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    assignments = _stable_source_splits(
        sorted(by_recording),
        seed=split_seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    for recording_id in sorted(by_recording):
        source_sequences = sorted(by_recording[recording_id], key=lambda item: item["sequence_id"])
        split = assignments[recording_id]
        for sequence in source_sequences:
            sequence["split"] = split
            sequence["split_status"] = "provisional_pending_duplicate_content_qc"
            sequences.append(sequence)
        sources.append(
            {
                "schema_version": "atomic-source-v1",
                "stage": "pre_hmr_candidate",
                "corpus": corpus,
                "recording_id": recording_id,
                "retrieval_group_id": recording_id,
                "duplicate_content_group_id": None,
                "split": split,
                "split_status": "provisional_pending_duplicate_content_qc",
                "sequence_ids": [item["sequence_id"] for item in source_sequences],
                "source_recording_keys": sorted(
                    {str(item["source_recording_key"]) for item in source_sequences}
                ),
                "assets": {
                    "source_videos": [item["assets"]["source_video"] for item in source_sequences],
                    "content_sha256": None,
                    "hash_status": "not_computed",
                },
                "qc": {
                    "sequence_count": len(source_sequences),
                    "all_ready_for_wham": all(
                        item["qc"]["inventory_status"] == "ready_for_wham"
                        for item in source_sequences
                    ),
                    "accepted_for_training": False,
                },
            }
        )
    split_counts = {name: sum(item["split"] == name for item in sources) for name in ("train", "val", "test")}
    summary = {
        "schema_version": "atomic-wild-staging-manifest-v1",
        "stage": "pre_hmr_candidate",
        "corpus": corpus,
        "ready_only": ready_only,
        "input_records": len(records),
        "excluded_by_inventory_status": excluded_status,
        "recordings": len(sources),
        "sequences": len(sequences),
        "split": {
            "algorithm": "ranked sha256(seed:recording_id)",
            "seed": split_seed,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": 1.0 - train_fraction - val_fraction,
            "recording_counts": split_counts,
            "status": "provisional_pending_duplicate_content_qc",
        },
        "training_eligibility": "none; await global HMR, 3D QC, duplicate-content QC, labels, and frozen split",
    }
    return sources, sequences, summary


def _resolve_source_video(record: Mapping[str, Any], video_root: Optional[Path]) -> Optional[Path]:
    raw = record.get("source_video")
    if raw:
        candidate = Path(str(raw)).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    if video_root is None:
        return None
    candidates: List[Path] = []
    if raw:
        candidates.extend(video_root.rglob(Path(str(raw)).name))
    if not candidates:
        candidates.extend(video_root.rglob("{}.mp4".format(record["clip_id"])))
    regular_files = [candidate for candidate in candidates if candidate.is_file()]
    return regular_files[0].resolve() if len(regular_files) == 1 else None


def build_wham_queue(
    records: Sequence[Mapping[str, Any]],
    *,
    wham_root: Path,
    result_root: Path,
    video_root: Optional[Path],
    python_executable: str,
    max_tasks: Optional[int] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build shell task bodies for a preflight-gated WHAM queue.

    ``python_executable`` deliberately is not expanded into the task body.
    ``write_wham_queue_script`` installs it as the default for the shell
    variable ``WHAM_PYTHON``; every generated invocation below then uses that
    one variable.  This lets an operator use a different authorized WHAM
    environment at launch time without accidentally running provenance or
    ``demo.py`` under a different interpreter than the preflight.
    """
    demo = wham_root / "demo.py"
    if not demo.is_file():
        raise FileNotFoundError("WHAM demo.py not found under {}".format(wham_root))
    adapter = Path(__file__).resolve()
    commands: List[str] = []
    skipped: List[Dict[str, Any]] = []
    for record in records:
        if max_tasks is not None and len(commands) >= max_tasks:
            break
        if record.get("status") != "ready_for_wham":
            continue
        video = _resolve_source_video(record, video_root)
        if video is None:
            skipped.append({"clip_id": record.get("clip_id"), "reason": "source video not resolvable"})
            continue
        output = result_root / str(record["clip_id"])
        # WHAM itself appends the video stem under --output_pth.  A task is
        # resumable only after its *terminal artifact and fresh-global marker*
        # exist.  This refuses unprovenanced historical output rather than
        # silently treating a local-only result as global motion.
        result_dir = output / video.stem
        result_pkl = result_dir / "wham_output.pkl"
        provenance = _global_run_marker_path(result_dir)
        invocation = "(cd {} && \"${{WHAM_PYTHON}}\" {} --video {} --output_pth {} --save_pkl)".format(
            shlex.quote(str(wham_root)),
            shlex.quote(str(demo)),
            shlex.quote(str(video)),
            shlex.quote(str(output)),
        )
        record_invocation = "\"${{WHAM_PYTHON}}\" {} record-wham-provenance --result-dir {} --video {} --wham-root {}".format(
            shlex.quote(str(adapter)),
            shlex.quote(str(result_dir)),
            shlex.quote(str(video)),
            shlex.quote(str(wham_root)),
        )
        commands.append(
            "if [ -s {result} ]; then "
            "if [ -f {provenance} ]; then echo '[skip WHAM] {clip}'; "
            "else echo '[refuse WHAM] unprovenanced result: {result}' >&2; exit 3; fi; "
            "else mkdir -p {result_dir} && {record} && {invoke}; fi".format(
                result=shlex.quote(str(result_pkl)),
                provenance=shlex.quote(str(provenance)),
                clip=str(record["clip_id"]),
                result_dir=shlex.quote(str(result_dir)),
                record=record_invocation,
                invoke=invocation,
            )
        )
    return commands, skipped


def split_round_robin(items: Sequence[str], shards: int) -> List[List[str]]:
    """Deterministically distribute independently resumable commands by shard."""
    if shards < 1:
        raise ValueError("shards must be positive")
    return [list(items[index::shards]) for index in range(shards)]


def _write_wham_preflight_gate(
    handle: Any,
    *,
    wham_root: Path,
    python_executable: str,
    launcher_name: str,
) -> None:
    """Emit the common, fail-closed WHAM readiness gate for a shell launcher."""
    adapter = Path(__file__).resolve()
    ready_check = (
        "import json, sys; "
        "ready = json.load(open(sys.argv[1], encoding='utf-8')).get('ready'); "
        "sys.exit(0 if ready is True else 'WHAM preflight report is not ready')"
    )
    handle.write("# Refuse WHAM's silent camera-local fallback and random/missing-checkpoint runs.\n")
    handle.write("# Override the configured WHAM environment without editing this file, e.g.:\n")
    handle.write("#   WHAM_PYTHON=/opt/wham/bin/python bash {}\n".format(shlex.quote(launcher_name)))
    handle.write("WHAM_ROOT={}\n".format(shlex.quote(str(wham_root.resolve()))))
    handle.write("ATOMIC_WILD3D_ADAPTER={}\n".format(shlex.quote(str(adapter))))
    handle.write("DEFAULT_WHAM_PYTHON={}\n".format(shlex.quote(str(python_executable))))
    handle.write('WHAM_PYTHON="${WHAM_PYTHON:-$DEFAULT_WHAM_PYTHON}"\n')
    handle.write("export WHAM_PYTHON\n")
    handle.write('if ! command -v "$WHAM_PYTHON" >/dev/null 2>&1; then\n')
    handle.write('  echo "WHAM_PYTHON is not an executable command: $WHAM_PYTHON" >&2\n')
    handle.write("  exit 2\nfi\n")
    handle.write('WHAM_PREFLIGHT_REPORT="$(mktemp "${TMPDIR:-/tmp}/atomicdance-wham-preflight.XXXXXX")"\n')
    handle.write('cleanup_wham_preflight() { rm -f "$WHAM_PREFLIGHT_REPORT"; }\n')
    handle.write("trap cleanup_wham_preflight EXIT\n")
    handle.write(
        '"${{WHAM_PYTHON}}" "${{ATOMIC_WILD3D_ADAPTER}}" preflight-wham '
        '--wham-root "${{WHAM_ROOT}}" --python "${{WHAM_PYTHON}}" '
        '--output "${{WHAM_PREFLIGHT_REPORT}}"\n'.format()
    )
    handle.write(
        '"${{WHAM_PYTHON}}" -c {} "${{WHAM_PREFLIGHT_REPORT}}"\n\n'.format(
            shlex.quote(ready_check)
        )
    )


def write_wham_queue_script(
    output: Path,
    commands: Sequence[str],
    *,
    wham_root: Path,
    python_executable: str,
) -> None:
    """Write a queue whose preamble proves WHAM is ready before any task.

    The shell preamble intentionally invokes this adapter through the same
    interpreter used for both ``record-wham-provenance`` and WHAM's
    ``demo.py``.  It calls ``preflight-wham`` once before *any* skip, marker,
    directory creation, or demo invocation.  ``preflight-wham`` returns zero
    only for a complete licensed-asset + world-runtime installation, and the
    JSON result is checked again as a defense against future CLI drift.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        handle.write("# Generated by tools/preprocess_wild_3d.py; one camera-aware WHAM task per line.\n")
        _write_wham_preflight_gate(
            handle,
            wham_root=wham_root,
            python_executable=python_executable,
            launcher_name=output.name,
        )
        for command in commands:
            handle.write(command)
            handle.write("\n")
    os.chmod(str(output), 0o755)


def write_wham_shard_launcher(
    output: Path,
    shard_scripts: Sequence[Path],
    *,
    wham_root: Path,
    python_executable: str,
) -> None:
    """Create an explicit GPU-ID launcher without assuming a host topology."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        handle.write("# Usage: GPU_IDS=0,1,... bash {}\n".format(shlex.quote(output.name)))
        handle.write("# One WHAM shard is assigned to each visible GPU ID.\n")
        _write_wham_preflight_gate(
            handle,
            wham_root=wham_root,
            python_executable=python_executable,
            launcher_name=output.name,
        )
        handle.write('IFS=, read -r -a gpu_ids <<< "${GPU_IDS:-}"\n')
        handle.write("if [ \"${{#gpu_ids[@]}}\" -ne {} ]; then\n".format(len(shard_scripts)))
        handle.write(
            "  echo \"set GPU_IDS to exactly {} comma-separated GPU IDs\" >&2\n".format(
                len(shard_scripts)
            )
        )
        handle.write("  exit 2\nfi\n\n")
        handle.write('script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n')
        handle.write("pids=()\n")
        for index, script in enumerate(shard_scripts):
            handle.write(
                "CUDA_VISIBLE_DEVICES=\"${{gpu_ids[{index}]}}\" bash \"$script_dir/{name}\" &\n"
                "pids+=(\"$!\")\n".format(index=index, name=script.name)
            )
        handle.write("status=0\n")
        handle.write('for pid in "${pids[@]}"; do\n')
        handle.write('  if ! wait "$pid"; then status=1; fi\n')
        handle.write("done\nexit \"$status\"\n")
    os.chmod(str(output), 0o755)


def wham_preflight(wham_root: Path, python_executable: str) -> Dict[str, Any]:
    """Report whether the official world-HMR demo has its required assets.

    This is intentionally read-only.  In particular, it never invokes
    ``fetch_demo_data.sh`` because that script asks for registered SMPL/
    SMPLify credentials and downloads separately licensed body models.
    """
    wham_root = wham_root.resolve()

    def missing(relative_paths: Sequence[str]) -> List[str]:
        return [item for item in relative_paths if not (wham_root / item).exists()]

    missing_code = missing(WHAM_RUNTIME_CODE)
    missing_body = missing(WHAM_BODY_ASSETS)
    missing_models = missing(WHAM_MODEL_ASSETS)
    python_path = shutil.which(python_executable)
    world_runtime: Dict[str, Any] = {"checked": False, "ready": False}
    if python_path and not missing_code:
        try:
            completed = subprocess.run(
                [python_executable, "-c", _WHAM_WORLD_RUNTIME_CHECK],
                cwd=str(wham_root),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            world_runtime = {
                "checked": True,
                "ready": completed.returncode == 0,
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-2000:],
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            world_runtime = {"checked": True, "ready": False, "error": str(error)}
    report = {
        "backend": "WHAM",
        "wham_root": str(wham_root),
        "python_executable": python_executable,
        "python_resolved": python_path,
        "missing_runtime_code": missing_code,
        "missing_body_model_assets": missing_body,
        "missing_model_assets": missing_models,
        "world_runtime": world_runtime,
        "ready": bool(
            python_path
            and not missing_code
            and not missing_body
            and not missing_models
            and world_runtime["ready"]
        ),
        "next_action": (
            "Use an authorized WHAM Python environment and licensed SMPL/SMPLify assets; "
            "then rerun this read-only preflight before the smoke queue."
        ),
    }
    return report


def command_inventory(args: argparse.Namespace) -> int:
    exclude: List[str] = []
    if getattr(args, "exclude", None):
        exclude_path = _as_path(args.exclude)
        assert exclude_path is not None
        if not exclude_path.is_file():
            raise FileNotFoundError(
                "no exclusion list at {}.  An absent list and an empty one read "
                "the same downstream, and on this corpus the difference is 760 "
                "orphan clips".format(exclude_path))
        exclude = [line.strip() for line in
                   exclude_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = inventory_wild_cache(
        _as_path(args.cache_root),
        recursive=args.recursive,
        min_frames=args.min_frames,
        min_visible_fraction=args.min_visible_fraction,
        min_score=args.min_score,
        max_frozen_fraction=args.max_frozen_fraction,
        exclude=exclude,
        # Spelled as given, not resolved: 14,762 of the 17,225 clips already
        # record ``data/wild_videos_20260811/<id>.mp4`` from their own meta, and
        # a rewritten row that named the same file absolutely would read as a
        # different upload to anything that groups by the string.
        upload_root=Path(args.upload_root) if getattr(args, "upload_root", None) else None,
    )
    output = _as_path(args.output)
    assert output is not None
    write_jsonl(output, records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "cache_root": str(_as_path(args.cache_root)),
        # Stated, not inferred: a run with no exclusion list looks exactly like
        # a run whose list happened to be empty, and the two mean opposite
        # things about whether orphans are in this manifest.
        "exclusion_list": str(_as_path(args.exclude)) if getattr(args, "exclude", None) else None,
        "excluded_names": len(exclude),
        "upload_root": args.upload_root if getattr(args, "upload_root", None) else None,
        "records": len(records),
        "ready_for_wham": sum(item["status"] == "ready_for_wham" for item in records),
        "quarantine": sum(item["status"] != "ready_for_wham" for item in records),
        "thresholds": {
            "min_frames": args.min_frames,
            "min_visible_fraction": args.min_visible_fraction,
            "min_score": args.min_score,
            "max_frozen_fraction": args.max_frozen_fraction,
        },
    }
    _json_dump(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_build_wild_staging_manifest(args: argparse.Namespace) -> int:
    inventory_path = _as_path(args.inventory)
    output_dir = _as_path(args.output_dir)
    assert inventory_path is not None and output_dir is not None
    sources_path = output_dir / "sources.jsonl"
    sequences_path = output_dir / "sequences.jsonl"
    split_path = output_dir / "split_candidates_v1.json"
    summary_path = output_dir / "summary.json"
    outputs = (sources_path, sequences_path, split_path, summary_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "staging manifest output already exists: {}; pass --overwrite after inspection".format(
                ", ".join(str(path) for path in existing)
            )
        )
    sources, sequences, summary = build_wild_staging_manifests(
        read_jsonl(inventory_path),
        corpus=args.corpus,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        ready_only=not args.include_quarantine,
    )
    split_manifest = {
        "schema_version": "atomic-source-split-v1",
        "stage": "pre_hmr_candidate",
        "status": "provisional_pending_duplicate_content_qc",
        "corpus": args.corpus,
        "unit": "recording_id",
        "seed": args.split_seed,
        "policy": {
            "algorithm": "ranked sha256(seed:recording_id)",
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "group_by": ["recording_id"],
            "duplicate_content_policy": "must be checked before freezing",
        },
        "assignments": [
            {"recording_id": item["recording_id"], "split": item["split"]}
            for item in sources
        ],
        "training_eligibility": summary["training_eligibility"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(sources_path, sources)
    write_jsonl(sequences_path, sequences)
    _json_dump(split_path, split_manifest)
    _json_dump(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_queue_wham(args: argparse.Namespace) -> int:
    manifest = _as_path(args.manifest)
    output = _as_path(args.output)
    assert manifest is not None and output is not None
    commands, skipped = build_wham_queue(
        read_jsonl(manifest),
        wham_root=_as_path(args.wham_root),
        result_root=_as_path(args.result_root),
        video_root=_as_path(args.video_root),
        python_executable=args.python,
        max_tasks=args.max_tasks,
    )
    write_wham_queue_script(
        output,
        commands,
        wham_root=_as_path(args.wham_root),
        python_executable=args.python,
    )
    _json_dump(
        output.with_suffix(".skipped.json"),
        {"schema_version": SCHEMA_VERSION, "skipped": skipped, "tasks": len(commands)},
    )
    print("wrote {} WHAM tasks to {}; skipped {}".format(len(commands), output, len(skipped)))
    return 0


def command_queue_wham_shards(args: argparse.Namespace) -> int:
    manifest = _as_path(args.manifest)
    output_dir = _as_path(args.output_dir)
    wham_root = _as_path(args.wham_root)
    assert manifest is not None and output_dir is not None and wham_root is not None
    commands, skipped = build_wham_queue(
        read_jsonl(manifest),
        wham_root=wham_root,
        result_root=_as_path(args.result_root),
        video_root=_as_path(args.video_root),
        python_executable=args.python,
        max_tasks=args.max_tasks,
    )
    groups = split_round_robin(commands, args.shards)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_scripts = []
    for index, group in enumerate(groups):
        script = output_dir / "wham_shard_{:02d}.sh".format(index)
        write_wham_queue_script(
            script,
            group,
            wham_root=wham_root,
            python_executable=args.python,
        )
        shard_scripts.append(script)
    launcher = output_dir / "launch_wham_shards.sh"
    write_wham_shard_launcher(
        launcher,
        shard_scripts,
        wham_root=wham_root,
        python_executable=args.python,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "tasks": len(commands),
        "skipped": skipped,
        "shards": args.shards,
        "tasks_per_shard": [len(group) for group in groups],
        "launcher": str(launcher),
        "scripts": [str(path) for path in shard_scripts],
    }
    _json_dump(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_preflight_wham(args: argparse.Namespace) -> int:
    wham_root = _as_path(args.wham_root)
    assert wham_root is not None
    report = wham_preflight(wham_root, args.python)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = _as_path(args.output)
        assert output is not None
        _json_dump(output, report)
    return 0 if report["ready"] else 2


def command_record_wham_provenance(args: argparse.Namespace) -> int:
    result_dir = _as_path(args.result_dir)
    video = _as_path(args.video)
    wham_root = _as_path(args.wham_root)
    assert result_dir is not None and video is not None and wham_root is not None
    provenance = record_wham_global_run(result_dir, video=video, wham_root=wham_root)
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


def command_convert_wham(args: argparse.Namespace) -> int:
    input_path = _as_path(args.wham_output)
    output_dir = _as_path(args.output_dir)
    source_video = _as_path(args.source_video)
    run_provenance_path = _as_path(args.run_provenance)
    assert input_path is not None and output_dir is not None and run_provenance_path is not None
    run_provenance = load_wham_global_run_provenance(
        run_provenance_path,
        wham_output=input_path,
        source_video=source_video,
    )
    results = _load_serialized(input_path)
    track_id, result = _select_wham_person(results, args.person_id)
    converted = convert_wham_result(result, args.contact_velocity_threshold)
    metadata = save_converted_wham(
        converted,
        output_dir,
        track_id=track_id,
        input_path=input_path,
        fps=args.fps,
        source_cache=_as_path(args.source_cache),
        source_video=source_video,
        slam_path=_as_path(args.slam),
        run_provenance=run_provenance,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def command_reconcile_wild_hmr(args: argparse.Namespace) -> int:
    staging_path = _as_path(args.staging_sequences)
    converted_root = _as_path(args.converted_root)
    output = _as_path(args.output)
    assert staging_path is not None and converted_root is not None and output is not None
    if output.exists() and not args.overwrite:
        raise FileExistsError("{} already exists; pass --overwrite after inspection".format(output))
    records, summary = reconcile_wild_hmr_sequences(
        read_jsonl(staging_path), converted_root=converted_root
    )
    write_jsonl(output, records)
    _json_dump(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def validate_converted_output(root: Path) -> Dict[str, Any]:
    """Return a strict validation report for one converted wild 3D sequence."""
    root = root.resolve()
    required = {
        "atomic_motion_151.npy": (ATOMIC_MOTION_DIM,),
        "pose_axis_angle_z_up.npy": (24, 3),
        "root_translation_z_up.npy": (3,),
        "rotation6d_z_up.npy": (24, 6),
        "contacts.npy": (4,),
        "contact_valid_mask.npy": (4,),
        "frame_ids.npy": (),
    }
    errors: List[str] = []
    frames: Optional[int] = None
    arrays: Dict[str, np.ndarray] = {}
    for name, tail in required.items():
        path = root / name
        if not path.is_file():
            errors.append("missing {}".format(name))
            continue
        try:
            array = np.load(str(path), mmap_mode="r")
        except Exception as error:
            errors.append("cannot load {}: {}".format(name, error))
            continue
        arrays[name] = array
        if array.ndim != len(tail) + 1 or tuple(array.shape[1:]) != tail:
            errors.append("{} has shape {}, expected [T, {}]".format(name, tuple(array.shape), tail))
        elif not np.isfinite(array).all():
            errors.append("{} contains non-finite values".format(name))
        elif frames is None:
            frames = len(array)
        elif frames != len(array):
            errors.append("{} has {} frames, expected {}".format(name, len(array), frames))

    frame_ids: Optional[np.ndarray] = None
    if frames is not None and "frame_ids.npy" in arrays:
        try:
            frame_ids = _normalize_frame_ids(arrays["frame_ids.npy"], frames)
            frame_summary = _frame_id_summary(frame_ids)
            if not frame_summary["is_contiguous"]:
                errors.append(
                    "frame_ids are not contiguous ({} missing frames); do not train contacts across track gaps".format(
                        frame_summary["missing_frames_within_span"]
                    )
                )
        except ValueError as error:
            errors.append(str(error))

    metadata_path = root / "metadata.json"
    metadata: Dict[str, Any] = {}
    if not metadata_path.is_file():
        errors.append("missing metadata.json")
    else:
        metadata, error = _read_meta(metadata_path)
        if error:
            errors.append(error)
        elif metadata.get("coordinate_convention", {}).get("output") is None:
            errors.append("metadata lacks coordinate convention")
        else:
            camera_meta = metadata.get("camera", {})
            if not isinstance(camera_meta, Mapping) or camera_meta.get("available") is not True:
                errors.append("metadata does not attest a retained camera audit asset")
            elif camera_meta.get("registered_to_body_world") is not False:
                errors.append("camera metadata must state that the camera gauge is not registered to body world")
            backend = metadata.get("backend")
            if backend == "GVHMR":
                # The GVHMR path has its own audit trail; requiring WHAM's
                # would force the converter to fabricate provenance.
                gvhmr_run = metadata.get("gvhmr_run_provenance")
                if not isinstance(gvhmr_run, Mapping):
                    errors.append("metadata lacks GVHMR run provenance")
                elif (
                    gvhmr_run.get("schema_version") != "gvhmr-extract-v1"
                    or not gvhmr_run.get("checkpoint_sha256_1mb")
                    or "'ay'" not in str(gvhmr_run.get("world_convention"))
                    or gvhmr_run.get("visual_odometry")
                    not in (GVHMR_SIMPLE_VO, GVHMR_DPVO, GVHMR_NO_VO)
                ):
                    errors.append("metadata GVHMR provenance is incomplete")
                if metadata.get("hand_joints_identity") is not True or metadata.get("hand_joints") != [22, 23]:
                    errors.append(
                        "GVHMR metadata must attest identity SMPL hand joints (SMPL-X carries none)")
            else:
                global_run = metadata.get("global_run_provenance")
                if not isinstance(global_run, Mapping):
                    errors.append("metadata lacks global WHAM run provenance")
                elif (
                    global_run.get("schema_version") != WHAM_GLOBAL_RUN_SCHEMA_VERSION
                    or global_run.get("global_requested") is not True
                    or global_run.get("estimate_local_only") is not False
                    or global_run.get("fresh_cache_verified") is not True
                ):
                    errors.append("metadata global WHAM provenance is incomplete or local-only")
            source_alignment = metadata.get("source_frame_alignment")
            if isinstance(source_alignment, Mapping) and source_alignment.get("available"):
                if source_alignment.get("track_frame_ids_within_source") is not True:
                    errors.append("track frame IDs fall outside the source 2D cache timeline")

    camera_path = root / "camera.npz"
    if not camera_path.is_file():
        errors.append("missing camera.npz")
    elif metadata.get("backend") == "GVHMR" and frames is not None and frame_ids is not None:
        try:
            with np.load(str(camera_path), allow_pickle=False) as camera:
                required_camera = {"intrinsics_K_fullimg", "camera_frame_ids",
                                   "full_video_frame_count"}
                vo_name = metadata.get("gvhmr_run_provenance", {}).get("visual_odometry")
                vo_expected = vo_name != "none"
                # Each tracker is checked against what it actually writes:
                # SimpleVO 4x4 matrices, DPVO a 7-vector per frame plus the
                # matrices this repo derives from it.  A shared key name would
                # let one tracker's output pass as the other's.
                is_dpvo = vo_name == GVHMR_DPVO
                full_key, track_key = (
                    ("dpvo_traj_unregistered_full_video", "dpvo_traj_unregistered_track")
                    if is_dpvo
                    else ("simplevo_c2w_unregistered_full_video", "simplevo_c2w_unregistered_track")
                )
                if vo_expected:
                    required_camera |= {full_key, track_key}
                    if is_dpvo:
                        required_camera |= {"dpvo_w2c_unregistered_full_video",
                                            "dpvo_w2c_unregistered_track"}
                missing_camera = sorted(required_camera.difference(camera.files))
                if missing_camera:
                    errors.append("camera.npz missing {}".format(", ".join(missing_camera)))
                else:
                    intrinsics = np.asarray(camera["intrinsics_K_fullimg"], dtype=np.float64)
                    camera_ids = _normalize_frame_ids(camera["camera_frame_ids"], frames)
                    full_count = np.asarray(camera["full_video_frame_count"], dtype=np.int64).reshape(-1)
                    if intrinsics.shape != (frames, 3, 3) or not np.isfinite(intrinsics).all():
                        errors.append("camera intrinsics must be finite [motion_frames,3,3]")
                    if not np.array_equal(camera_ids, frame_ids):
                        errors.append("camera_frame_ids do not match motion frame_ids")
                    if vo_expected:
                        raw_full = np.asarray(camera[full_key], dtype=np.float64)
                        raw_track = np.asarray(camera[track_key], dtype=np.float64)
                        expected_shape = (7,) if is_dpvo else (4, 4)
                        label = GVHMR_DPVO if is_dpvo else "SimpleVO"
                        if (
                            raw_full.ndim != 1 + len(expected_shape)
                            or raw_full.shape[1:] != expected_shape
                            or not np.isfinite(raw_full).all()
                        ):
                            errors.append(
                                "camera full-video {} trajectory must be finite [T,{}]".format(
                                    label, ",".join(str(dim) for dim in expected_shape)))
                        if raw_track.shape != (frames,) + expected_shape or not np.isfinite(raw_track).all():
                            errors.append(
                                "camera track {} trajectory must be finite [motion_frames,{}]".format(
                                    label, ",".join(str(dim) for dim in expected_shape)))
                        if len(full_count) != 1 or int(full_count[0]) != len(raw_full):
                            errors.append("camera full_video_frame_count does not match full trajectory")
                        elif len(raw_full) <= int(frame_ids[-1]):
                            errors.append("camera full trajectory does not cover the motion frame IDs")
                        elif not np.allclose(raw_track, raw_full[frame_ids], rtol=0.0, atol=1e-6):
                            errors.append(
                                "camera track trajectory is not indexed from the full-video "
                                "{} trajectory".format(label))
                        if is_dpvo:
                            derived_full = np.asarray(
                                camera["dpvo_w2c_unregistered_full_video"], dtype=np.float64)
                            derived_track = np.asarray(
                                camera["dpvo_w2c_unregistered_track"], dtype=np.float64)
                            if (
                                derived_full.shape != (len(raw_full), 4, 4)
                                or not np.isfinite(derived_full).all()
                            ):
                                errors.append("derived DPVO w2c must be finite [T,4,4]")
                            elif not np.allclose(
                                derived_track, derived_full[frame_ids], rtol=0.0, atol=1e-6
                            ):
                                errors.append(
                                    "derived DPVO w2c track is not indexed from the full-video matrices")
                            else:
                                rotations = derived_full[:, :3, :3]
                                identity = np.einsum("tij,tkj->tik", rotations, rotations)
                                if not np.allclose(
                                    identity, np.eye(3), rtol=0.0, atol=1e-6
                                ):
                                    errors.append("derived DPVO w2c rotations are not orthonormal")
        except Exception as error:
            errors.append("cannot validate camera.npz: {}".format(error))
    elif frames is not None and frame_ids is not None:
        try:
            with np.load(str(camera_path), allow_pickle=False) as camera:
                required_camera = {
                    "dpvo_c2w_unregistered_full_video",
                    "dpvo_c2w_unregistered_track",
                    "camera_frame_ids",
                    "full_video_frame_count",
                }
                missing_camera = sorted(required_camera.difference(camera.files))
                if missing_camera:
                    errors.append("camera.npz missing {}".format(", ".join(missing_camera)))
                else:
                    raw_full = np.asarray(camera["dpvo_c2w_unregistered_full_video"], dtype=np.float64)
                    raw_track = np.asarray(camera["dpvo_c2w_unregistered_track"], dtype=np.float64)
                    camera_ids = _normalize_frame_ids(camera["camera_frame_ids"], frames)
                    full_count = np.asarray(camera["full_video_frame_count"], dtype=np.int64).reshape(-1)
                    if raw_full.ndim != 2 or raw_full.shape[1] != 7 or not np.isfinite(raw_full).all():
                        errors.append("camera full-video DPVO trajectory must be finite [T,7]")
                    elif _is_wham_local_only_sentinel(raw_full):
                        errors.append("camera trajectory is WHAM's local-only fallback sentinel")
                    if raw_track.shape != (frames, 7) or not np.isfinite(raw_track).all():
                        errors.append("camera track DPVO trajectory must be finite [motion_frames,7]")
                    if not np.array_equal(camera_ids, frame_ids):
                        errors.append("camera_frame_ids do not match motion frame_ids")
                    if len(full_count) != 1 or int(full_count[0]) != len(raw_full):
                        errors.append("camera full_video_frame_count does not match full trajectory")
                    elif len(raw_full) <= int(frame_ids[-1]):
                        errors.append("camera full trajectory does not cover the motion frame IDs")
                    elif not np.allclose(raw_track, raw_full[frame_ids], rtol=0.0, atol=1e-6):
                        errors.append("camera track trajectory is not indexed from the full-video DPVO trajectory")
        except Exception as error:
            errors.append("cannot validate camera.npz: {}".format(error))

    report = {
        "output_dir": str(root),
        "frames": frames,
        "frame_alignment": _frame_id_summary(frame_ids) if frame_ids is not None else None,
        "valid": not errors,
        "errors": errors,
    }
    return report


def reconcile_wild_hmr_sequences(
    staging_sequences: Sequence[Mapping[str, Any]],
    *,
    converted_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Join staged wild sequences with validated 151-D conversions.

    A successful WHAM process alone is not a training acceptance.  This makes
    that distinction explicit: each staging sequence is carried forward as
    ``pending``, ``quarantine``, or ``candidate`` based on the converter's
    strict validation report.  No record becomes trainable here because source
    duplicate QC and atomic label generation still remain separate gates.
    """
    converted_root = _stable_path(converted_root)
    output: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    counts = {"pending": 0, "quarantine": 0, "candidate": 0}
    for staging in staging_sequences:
        sequence_id = staging.get("sequence_id")
        clip_id = staging.get("legacy_clip_id")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("staging sequence lacks non-empty sequence_id")
        if sequence_id in seen_ids:
            raise ValueError("duplicate sequence_id in staging manifest: {}".format(sequence_id))
        seen_ids.add(sequence_id)
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError("{} lacks legacy_clip_id needed to locate conversion".format(sequence_id))

        item = dict(staging)
        item["stage"] = "post_hmr_reconciled"
        item["assets"] = dict(staging.get("assets", {}))
        item["timeline"] = dict(staging.get("timeline", {}))
        item["representation"] = dict(staging.get("representation", {}))
        item["qc"] = dict(staging.get("qc", {}))
        converted_dir = converted_root / clip_id
        item["conversion_output_dir"] = str(converted_dir)
        reason_codes = list(item["qc"].get("reason_codes", []))
        if not converted_dir.is_dir():
            item["qc"].update(
                {
                    "hmr_status": "pending",
                    "conversion_validation": "not_found",
                    "accepted_for_training": False,
                    "reason_codes": reason_codes,
                }
            )
            counts["pending"] += 1
            output.append(item)
            continue

        report = validate_converted_output(converted_dir)
        if not report["valid"]:
            item["qc"].update(
                {
                    "hmr_status": "quarantine",
                    "conversion_validation": "failed",
                    "validation_errors": list(report["errors"]),
                    "accepted_for_training": False,
                    "reason_codes": reason_codes + ["conversion_validation_failed"],
                }
            )
            counts["quarantine"] += 1
            output.append(item)
            continue

        metadata, metadata_error = _read_meta(converted_dir / "metadata.json")
        if metadata_error:  # pragma: no cover - validate_converted_output already guards this
            raise RuntimeError(metadata_error)
        frame_ids = np.load(str(converted_dir / "frame_ids.npy"), mmap_mode="r")
        frame_ids = _normalize_frame_ids(frame_ids, int(report["frames"]))
        frame_summary = _frame_id_summary(frame_ids)
        item["timeline"].update(
            {
                "fps": float(metadata["fps"]),
                "source_start_frame": frame_summary["start_frame"],
                "source_end_frame_exclusive": frame_summary["end_frame_inclusive"] + 1,
                "frame_count": frame_summary["frames"],
                "frame_ids_path": str(converted_dir / "frame_ids.npy"),
                "motion_frames_are_contiguous": frame_summary["is_contiguous"],
            }
        )
        item["assets"].update(
            {
                "motion_151_raw": str(converted_dir / "atomic_motion_151.npy"),
                "camera": str(converted_dir / "camera.npz"),
                "conversion_metadata": str(converted_dir / "metadata.json"),
                "conversion_quality": str(converted_dir / "quality.json"),
            }
        )
        item["representation"].update(
            {
                "motion": "AtomicDance_151D",
                "coordinate_system": "z_up_world_body_only",
                "normalization": "raw",
                "camera_in_model_input": False,
            }
        )
        if metadata.get("backend") == "GVHMR":
            # GVHMR's demo commits to one subject inside the extractor
            # (``Tracker.get_one_track``) and never surfaces a track index, so
            # there is no WHAM-style id to carry.  Naming the selector is
            # honest; inventing a numeric track id would not be.
            item["person_track_id"] = GVHMR_PERSON_TRACK_ID
        else:
            item["person_track_id"] = str(metadata["backend_track_id"])
        item["qc"].update(
            {
                "hmr_status": "candidate",
                "conversion_validation": "passed",
                "conversion_frame_alignment": frame_summary,
                "accepted_for_training": False,
                "reason_codes": reason_codes,
            }
        )
        counts["candidate"] += 1
        output.append(item)
    summary = {
        "schema_version": "atomic-wild-post-hmr-manifest-v1",
        "stage": "post_hmr_reconciled",
        "input_sequences": len(staging_sequences),
        "converted_root": str(converted_root),
        "status_counts": counts,
        "training_eligibility": "none; candidate requires duplicate-content QC, 3D/human QC, labels, and frozen split",
    }
    return output, summary


def command_validate(args: argparse.Namespace) -> int:
    root = _as_path(args.output_dir)
    assert root is not None
    report = validate_converted_output(root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="build a QC-gated manifest from Lodge-style 2D caches")
    inventory.add_argument("--cache-root", required=True)
    inventory.add_argument("--output", required=True, help="output JSONL manifest")
    inventory.add_argument("--recursive", action="store_true")
    inventory.add_argument("--exclude", default=None,
                           help="file of clip names to leave out, one per line: "
                                "the orphans a directory glob would otherwise "
                                "return forever (760 on this corpus, 2026-08-25)")
    inventory.add_argument(
        "--upload-root", default=None,
        help="where the uploads live, e.g. data/wild_videos_20260811.  Re-cut clips "
             "record the scratch working copy they were cut from; this is what they "
             "are rewritten to name instead, by clip-name prefix.")
    inventory.add_argument("--min-frames", type=int, default=180)
    inventory.add_argument("--min-visible-fraction", type=float, default=0.60)
    inventory.add_argument("--min-score", type=float, default=0.30)
    inventory.add_argument("--max-frozen-fraction", type=float, default=0.30)
    inventory.set_defaults(func=command_inventory)

    staging = subparsers.add_parser(
        "build-wild-staging-manifest",
        help="materialize recording/sequence staging manifests from wild inventory JSONL",
    )
    staging.add_argument("--inventory", required=True)
    staging.add_argument("--output-dir", required=True)
    staging.add_argument("--corpus", default="tiktok")
    staging.add_argument("--split-seed", type=int, default=20260805)
    staging.add_argument("--train-fraction", type=float, default=0.80)
    staging.add_argument("--val-fraction", type=float, default=0.10)
    staging.add_argument(
        "--include-quarantine",
        action="store_true",
        help="also emit quarantined inventory records as non-training staging candidates",
    )
    staging.add_argument("--overwrite", action="store_true")
    staging.set_defaults(func=command_build_wild_staging_manifest)

    queue = subparsers.add_parser("queue-wham", help="create resumable, camera-aware WHAM commands")
    queue.add_argument("--manifest", required=True)
    queue.add_argument("--wham-root", required=True, help="official WHAM checkout with weights installed")
    queue.add_argument("--result-root", required=True)
    queue.add_argument("--output", required=True, help="generated executable .sh file")
    queue.add_argument("--video-root", default=None, help="fallback root used to resolve missing source paths")
    queue.add_argument("--python", default=sys.executable)
    queue.add_argument("--max-tasks", type=int, default=None, help="emit at most this many ready tasks")
    queue.set_defaults(func=command_queue_wham)

    queue_shards = subparsers.add_parser(
        "queue-wham-shards", help="create balanced resumable WHAM scripts plus an explicit GPU launcher"
    )
    queue_shards.add_argument("--manifest", required=True)
    queue_shards.add_argument("--wham-root", required=True, help="official WHAM checkout with weights installed")
    queue_shards.add_argument("--result-root", required=True)
    queue_shards.add_argument("--output-dir", required=True, help="directory for shard scripts and launcher")
    queue_shards.add_argument("--shards", type=int, required=True, help="number of independent GPU shards")
    queue_shards.add_argument("--video-root", default=None, help="fallback root used to resolve missing source paths")
    queue_shards.add_argument("--python", default=sys.executable)
    queue_shards.add_argument("--max-tasks", type=int, default=None, help="emit at most this many ready tasks")
    queue_shards.set_defaults(func=command_queue_wham_shards)

    preflight = subparsers.add_parser("preflight-wham", help="read-only check for WHAM code and required model assets")
    preflight.add_argument("--wham-root", required=True)
    preflight.add_argument("--python", default="python", help="WHAM environment Python executable")
    preflight.add_argument("--output", default=None, help="optional JSON report path")
    preflight.set_defaults(func=command_preflight_wham)

    provenance = subparsers.add_parser(
        "record-wham-provenance",
        help="record a fresh, non-local-only WHAM launch immediately before demo.py",
    )
    provenance.add_argument("--result-dir", required=True, help="WHAM's <output_pth>/<video_stem> directory")
    provenance.add_argument("--video", required=True)
    provenance.add_argument("--wham-root", required=True)
    provenance.set_defaults(func=command_record_wham_provenance)

    convert = subparsers.add_parser("convert-wham", help="standardize one global WHAM output to 151-D motion")
    convert.add_argument("--wham-output", required=True, help="WHAM's wham_output.pkl")
    convert.add_argument("--output-dir", required=True)
    convert.add_argument("--person-id", default=None, help="required when WHAM tracks multiple people")
    convert.add_argument(
        "--slam",
        required=True,
        help="matching WHAM slam_results.pth; required to preserve camera/body separation",
    )
    convert.add_argument(
        "--run-provenance",
        required=True,
        help="fresh global-run marker emitted by the generated WHAM queue",
    )
    convert.add_argument("--source-cache", default=None)
    convert.add_argument("--source-video", default=None)
    convert.add_argument("--fps", type=float, default=30.0)
    convert.add_argument("--contact-velocity-threshold", type=float, default=0.01)
    convert.add_argument("--overwrite", action="store_true")
    convert.set_defaults(func=command_convert_wham)

    reconcile = subparsers.add_parser(
        "reconcile-wild-hmr",
        help="join staged wild sequences with validated converted WHAM outputs",
    )
    reconcile.add_argument("--staging-sequences", required=True)
    reconcile.add_argument("--converted-root", required=True)
    reconcile.add_argument("--output", required=True, help="post-HMR sequence JSONL")
    reconcile.add_argument("--overwrite", action="store_true")
    reconcile.set_defaults(func=command_reconcile_wild_hmr)

    validate = subparsers.add_parser("validate", help="verify a standardized wild 3D clip")
    validate.add_argument("--output-dir", required=True)
    validate.set_defaults(func=command_validate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2  # pragma: no cover - parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
