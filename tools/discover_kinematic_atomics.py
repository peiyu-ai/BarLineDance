#!/usr/bin/env python3
"""Produce train-only 3D kinematic atomic pseudo-labels on full timelines.

This is deliberately a *baseline*, not a claim to reproduce AtomicDance's
unreleased I3D/TMR/LLM discovery pipeline.  The public package contains no
video features, TMR checkpoint, or semantic re-clustering artifacts.  What it
does provide is valid 151-D 3D motion, so this producer makes that limitation
explicit and builds a deterministic, auditable alternative:

* body/world motion is represented by a heading-aligned root trajectory,
  contact state, rotation-6D pose, and joint angular velocity;
* full sequences (never overlapping 150-frame windows) are segmented with the
  repository's Algorithm-1-style similarity segmenter on a temporal stride;
* only train-split descriptors fit frame statistics, segment embedding
  statistics, K-Means centers, confidence thresholds, and softmax temperature;
* validation/test sequences only apply those frozen artifacts; and
* rejected frames stay ``-1`` with a false mask.  Label ``0`` is reserved for
  an explicitly configured gutter around a boundary between two accepted
  events, never for an unknown event.

The output is immutable and contains one label timeline per source sequence.
It can be consumed by ``tools/materialize_atomic_windows.py`` after the raw
source manifest has been normalized with the same frozen train-only release.
No legacy ``labels.npy`` file is read anywhere in this tool.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# sys.path[0] is tools/ when this runs as a script, not the repo root; the
# repo-package import below otherwise rides the ambient PYTHONPATH's stray
# trailing colon (which puts CWD on the path) -- a coincidence, not a contract.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.atomic_discovery import adaptive_segment, kmeans  # noqa: E402


SCHEMA_VERSION = "atomicdance-kinematic-atomic-labels-v1"
# v2 introduces sequence-identity-stable segmentation randomness and one
# constant initial-heading canonical frame for root world motion.  It is not
# compatible with the earlier v1 cluster assignments, so both the producer
# and label-space identifiers must change even though the JSONL schema does
# not.
PRODUCER_VERSION = "kinematic-atomic-discovery-v2"
LABEL_SPACE_VERSION = "v2"
MODEL_MOTION_REPRESENTATION = "AtomicDance_151D"
MODEL_COORDINATE_SYSTEM = "z_up_world_body_only"
MODEL_FPS = 30.0
MOTION_DIM = 151
MUSIC_DIM = 35
INVALID_LABEL = -1
ALLOWED_SPLITS = frozenset(("train", "val", "test"))


class KinematicDiscoveryError(ValueError):
    """An input/output contract makes pseudo-label publication unsafe."""


@dataclass(frozen=True)
class SourceIdentity:
    recording_id: str
    retrieval_group_id: str
    duplicate_content_group_id: Optional[str]
    content_sha256: str
    fps: float
    split: str


@dataclass(frozen=True)
class SequenceInput:
    sequence_id: str
    recording_id: str
    retrieval_group_id: str
    duplicate_content_group_id: Optional[str]
    split: str
    frame_count: int
    raw_motion_path: Path
    raw_motion_sha256: str
    normalized_motion_path: Path
    normalized_motion_sha256: str
    normalizer_artifact_sha256: str


@dataclass(frozen=True)
class Event:
    sequence: SequenceInput
    start: int
    end: int
    embedding: np.ndarray

    @property
    def length(self) -> int:
        return self.end - self.start


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_sha256(value: object, context: str) -> str:
    if not _is_sha256(value):
        raise KinematicDiscoveryError("{} must be a SHA-256 hex digest".format(context))
    return str(value).lower()


def _require_string(row: Mapping[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise KinematicDiscoveryError("{} requires non-empty {}".format(context, field))
    return value


def _require_split(row: Mapping[str, Any], context: str) -> str:
    split = _require_string(row, "split", context)
    if split not in ALLOWED_SPLITS:
        raise KinematicDiscoveryError(
            "{} split {!r} is not one of {}".format(context, split, ", ".join(sorted(ALLOWED_SPLITS)))
        )
    return split


def _require_optional_group(row: Mapping[str, Any], field: str, context: str) -> Optional[str]:
    if field not in row:
        raise KinematicDiscoveryError("{} must explicitly contain {} (string or null)".format(context, field))
    value = row[field]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise KinematicDiscoveryError("{} has invalid {}".format(context, field))
    return value


def _require_mapping(row: Mapping[str, Any], field: str, context: str) -> Mapping[str, Any]:
    value = row.get(field)
    if not isinstance(value, Mapping):
        raise KinematicDiscoveryError("{} requires object {}".format(context, field))
    return value


def _require_positive_int(row: Mapping[str, Any], field: str, context: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise KinematicDiscoveryError("{} requires positive integer {}".format(context, field))
    return int(value)


def _require_atomicdance_fps(value: object, context: str) -> float:
    """Require the fixed frame rate assumed by 150-frame AtomicDance windows."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KinematicDiscoveryError("{} requires an explicit numeric fps".format(context))
    fps = float(value)
    if not math.isfinite(fps) or fps != MODEL_FPS:
        raise KinematicDiscoveryError(
            "{} fps must be exactly {} for AtomicDance 150-frame timing, got {!r}".format(
                context, int(MODEL_FPS), value
            )
        )
    return fps


def _require_sequence_fps(row: Mapping[str, Any], context: str) -> float:
    """Read flat or timeline fps while rejecting absent/contradictory aliases."""
    values: List[Tuple[str, object]] = []
    if "fps" in row:
        values.append(("fps", row["fps"]))
    timeline = row.get("timeline")
    if timeline is not None:
        if not isinstance(timeline, Mapping):
            raise KinematicDiscoveryError("{} timeline must be an object when present".format(context))
        if "fps" in timeline:
            values.append(("timeline.fps", timeline["fps"]))
    if not values:
        raise KinematicDiscoveryError("{} requires an explicit sequence fps or timeline.fps".format(context))
    parsed = [(_path, _require_atomicdance_fps(_value, "{} {}".format(context, _path))) for _path, _value in values]
    first = parsed[0][1]
    if any(value != first for _, value in parsed[1:]):
        rendered = ", ".join("{}={!r}".format(path, value) for path, value in parsed)
        raise KinematicDiscoveryError("{} has contradictory fps aliases: {}".format(context, rendered))
    return first


def _read_jsonl(path: Path, kind: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("{} manifest does not exist: {}".format(kind, path))
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise KinematicDiscoveryError(
                    "{} manifest {} line {} is invalid JSON: {}".format(kind, path, line_number, error)
                ) from error
            if not isinstance(row, Mapping):
                raise KinematicDiscoveryError(
                    "{} manifest {} line {} is not an object".format(kind, path, line_number)
                )
            item = dict(row)
            item["__line_number__"] = line_number
            rows.append(item)
    if not rows:
        raise KinematicDiscoveryError("{} manifest is empty: {}".format(kind, path))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _accepted_for_training(row: Mapping[str, Any], context: str) -> bool:
    qc = row.get("qc")
    if not isinstance(qc, Mapping) or qc.get("accepted_for_training") is not True:
        raise KinematicDiscoveryError(
            "{} must explicitly set qc.accepted_for_training=true before label production".format(context)
        )
    for field in ("status", "hmr_status", "conversion_validation"):
        value = qc.get(field)
        if isinstance(value, str) and value.strip().lower() in {"pending", "quarantine", "failed", "rejected", "invalid"}:
            raise KinematicDiscoveryError(
                "{} contradicts accepted_for_training with qc.{}={!r}".format(context, field, value)
            )
    return True


def _resolve_asset(manifest_path: Path, value: object, field: str, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise KinematicDiscoveryError("{} requires non-empty {}".format(context, field))
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        root = manifest_path.parent.resolve()
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise KinematicDiscoveryError("{} {} escapes manifest root".format(context, field)) from error
    if not resolved.is_file():
        raise KinematicDiscoveryError("{} asset does not exist: {}".format(context, resolved))
    return resolved


def _parse_sources(rows: Sequence[Mapping[str, Any]]) -> Dict[str, SourceIdentity]:
    sources: Dict[str, SourceIdentity] = {}
    ownership: Dict[Tuple[str, str], str] = {}
    for row in rows:
        context = "source row {}".format(row.get("__line_number__", "?"))
        recording_id = _require_string(row, "recording_id", context)
        retrieval_group_id = _require_string(row, "retrieval_group_id", context)
        duplicate_group = _require_optional_group(row, "duplicate_content_group_id", context)
        content_sha256 = _require_sha256(row.get("content_sha256"), context + " content_sha256")
        fps = _require_atomicdance_fps(row.get("fps"), context + " fps")
        split = _require_split(row, context)
        _accepted_for_training(row, context)
        if recording_id in sources:
            raise KinematicDiscoveryError("duplicate source recording_id {!r}".format(recording_id))
        for kind, value in (
            ("recording_id", recording_id),
            ("retrieval_group_id", retrieval_group_id),
            ("content_sha256", content_sha256),
        ):
            key = (kind, value)
            prior = ownership.get(key)
            if prior is not None and prior != split:
                raise KinematicDiscoveryError("{} {!r} crosses {} and {}".format(kind, value, prior, split))
            ownership[key] = split
        if duplicate_group is not None:
            key = ("duplicate_content_group_id", duplicate_group)
            prior = ownership.get(key)
            if prior is not None and prior != split:
                raise KinematicDiscoveryError(
                    "duplicate_content_group_id {!r} crosses {} and {}".format(duplicate_group, prior, split)
                )
            ownership[key] = split
        sources[recording_id] = SourceIdentity(
            recording_id=recording_id,
            retrieval_group_id=retrieval_group_id,
            duplicate_content_group_id=duplicate_group,
            content_sha256=content_sha256,
            fps=fps,
            split=split,
        )
    return sources


def _normalized_representation(row: Mapping[str, Any], context: str) -> str:
    representation = _require_mapping(row, "representation", context)
    expected = {
        "motion": MODEL_MOTION_REPRESENTATION,
        "coordinate_system": MODEL_COORDINATE_SYSTEM,
        "normalization": "normalized",
        "camera_in_model_input": False,
        "normalizer_fit_split": "train",
    }
    for field, value in expected.items():
        if representation.get(field) != value:
            raise KinematicDiscoveryError(
                "{} representation.{} must be {!r}, got {!r}".format(
                    context, field, value, representation.get(field)
                )
            )
    return _require_sha256(representation.get("normalizer_artifact_sha256"), context + " normalizer artifact")


def _parse_sequences(
    rows: Sequence[Mapping[str, Any],],
    *,
    sources: Mapping[str, SourceIdentity],
    manifest_path: Path,
) -> List[SequenceInput]:
    sequences: List[SequenceInput] = []
    seen: set[str] = set()
    for row in rows:
        context = "sequence row {}".format(row.get("__line_number__", "?"))
        sequence_id = _require_string(row, "sequence_id", context)
        if sequence_id in seen:
            raise KinematicDiscoveryError("duplicate sequence_id {!r}".format(sequence_id))
        seen.add(sequence_id)
        recording_id = _require_string(row, "recording_id", context)
        source = sources.get(recording_id)
        if source is None:
            raise KinematicDiscoveryError("{} references unknown source {!r}".format(context, recording_id))
        retrieval_group = _require_string(row, "retrieval_group_id", context)
        duplicate_group = _require_optional_group(row, "duplicate_content_group_id", context)
        split = _require_split(row, context)
        if (
            retrieval_group != source.retrieval_group_id
            or duplicate_group != source.duplicate_content_group_id
            or split != source.split
        ):
            raise KinematicDiscoveryError("{} does not inherit source group/split identity".format(context))
        sequence_fps = _require_sequence_fps(row, context)
        if sequence_fps != source.fps:
            raise KinematicDiscoveryError(
                "{} fps {} disagrees with source {!r} fps {}".format(
                    context, sequence_fps, recording_id, source.fps
                )
            )
        _accepted_for_training(row, context)
        normalizer_hash = _normalized_representation(row, context)
        assets = _require_mapping(row, "assets", context)
        # ``apply_motion_normalizer`` publishes a derived manifest in a new
        # directory.  Its immutable raw input can therefore be outside that
        # directory; the normalized manifest records the absolute provenance
        # path under ``normalization``.  Prefer and verify that path rather
        # than accidentally resolving a copied relative raw string against the
        # derived bundle root.
        normalization = _require_mapping(row, "normalization", context)
        raw_provenance = normalization.get("input_motion_151_raw")
        if raw_provenance is None:
            raw_path = _resolve_asset(
                manifest_path, assets.get("motion_151_raw"), "assets.motion_151_raw", context
            )
        else:
            raw_path = _resolve_asset(
                manifest_path, raw_provenance, "normalization.input_motion_151_raw", context
            )
        normalized_value = assets.get("motion_151_model_input", assets.get("motion_151_normalized"))
        normalized_path = _resolve_asset(
            manifest_path, normalized_value, "assets.motion_151_model_input", context
        )
        raw_hash = _sha256_file(raw_path)
        normalized_hash = _sha256_file(normalized_path)
        declared_raw = assets.get("motion_151_raw_sha256")
        if declared_raw is not None and _require_sha256(declared_raw, context + " raw hash") != raw_hash:
            raise KinematicDiscoveryError("{} raw motion hash does not match its asset".format(context))
        provenance_raw_hash = normalization.get("input_motion_151_raw_sha256")
        if provenance_raw_hash is not None and _require_sha256(
            provenance_raw_hash, context + " raw provenance hash"
        ) != raw_hash:
            raise KinematicDiscoveryError("{} raw provenance hash does not match its asset".format(context))
        declared_normalized = assets.get(
            "motion_151_model_input_sha256", assets.get("motion_151_normalized_sha256")
        )
        if declared_normalized is not None and _require_sha256(
            declared_normalized, context + " normalized hash"
        ) != normalized_hash:
            raise KinematicDiscoveryError("{} normalized motion hash does not match its asset".format(context))
        sequences.append(
            SequenceInput(
                sequence_id=sequence_id,
                recording_id=recording_id,
                retrieval_group_id=retrieval_group,
                duplicate_content_group_id=duplicate_group,
                split=split,
                frame_count=_require_positive_int(row, "frame_count", context),
                raw_motion_path=raw_path,
                raw_motion_sha256=raw_hash,
                normalized_motion_path=normalized_path,
                normalized_motion_sha256=normalized_hash,
                normalizer_artifact_sha256=normalizer_hash,
            )
        )
    return sorted(sequences, key=lambda item: item.sequence_id)


def _load_motion(path: Path, *, expected_frames: int, kind: str) -> np.ndarray:
    try:
        values = np.load(str(path), mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise KinematicDiscoveryError("cannot open {} motion {}: {}".format(kind, path, error)) from error
    if values.dtype != np.float32 or tuple(values.shape) != (expected_frames, MOTION_DIM):
        raise KinematicDiscoveryError(
            "{} motion {} must be float32 [{}, {}], got {} {}".format(
                kind, path, expected_frames, MOTION_DIM, values.dtype, tuple(values.shape)
            )
        )
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise KinematicDiscoveryError("{} motion {} has non-finite values".format(kind, path))
    return result


def rotation6d_to_matrix(values: np.ndarray) -> np.ndarray:
    """Pure NumPy Gram-Schmidt conversion; avoids optional PyTorch3D imports."""
    if values.ndim != 3 or values.shape[1:] != (24, 6):
        raise KinematicDiscoveryError("rotation6d must have shape [T,24,6]")
    first = values[..., :3].astype(np.float64, copy=False)
    second = values[..., 3:].astype(np.float64, copy=False)
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm <= 1e-8):
        raise KinematicDiscoveryError("rotation6d first axis is degenerate")
    basis1 = first / first_norm
    projected = second - np.sum(basis1 * second, axis=-1, keepdims=True) * basis1
    projected_norm = np.linalg.norm(projected, axis=-1, keepdims=True)
    if np.any(projected_norm <= 1e-8):
        raise KinematicDiscoveryError("rotation6d second axis is degenerate")
    basis2 = projected / projected_norm
    basis3 = np.cross(basis1, basis2)
    # Match ``pytorch3d.transforms.rotation_6d_to_matrix`` exactly: its first
    # two rows are the stored 6-D representation.  Keeping this convention
    # avoids a silent transpose between discovery and AtomicDance decoding.
    result = np.stack((basis1, basis2, basis3), axis=-2)
    return result.astype(np.float32, copy=False)


def kinematic_frame_descriptor(raw_motion: np.ndarray) -> np.ndarray:
    """Return a camera-free, globally-yaw-canonical frame descriptor.

    The raw 151-D contract contains body/world root translation and the root
    joint's *global* orientation, but no camera parameters.  A camera/world
    coordinate system can still choose an arbitrary constant z-up yaw.  To
    make discovery invariant to that gauge without erasing a dancer's actual
    turns, this function uses the **first-frame** root-facing direction as one
    constant canonical frame for the full sequence.  It applies that same
    transform to root translation and global-root rotation only; child joint
    rotations remain local body pose.  In particular, it does not re-align
    each frame to its instantaneous facing direction, because that would
    remove real turn dynamics from the descriptors.

    The returned descriptor is derived solely from the 151-D body motion:
    contacts, canonical root translation/velocity, canonical rotation-6D
    pose, and joint angular velocity.  It never reads or appends a camera
    estimate.
    """
    if raw_motion.ndim != 2 or raw_motion.shape[1] != MOTION_DIM:
        raise KinematicDiscoveryError("raw motion must have shape [T,151]")
    if len(raw_motion) < 2:
        raise KinematicDiscoveryError("a sequence needs at least two frames for kinematic discovery")
    contacts = np.asarray(raw_motion[:, :4], dtype=np.float32)
    root = np.asarray(raw_motion[:, 4:7], dtype=np.float32)
    rotations = rotation6d_to_matrix(np.asarray(raw_motion[:, 7:], dtype=np.float32).reshape(-1, 24, 6))
    root_rotation = rotations[:, 0]
    # z is up and AtomicDance's root-facing direction is the third rotation
    # column.  Use one *constant* reference yaw so a global coordinate change
    # cannot alter labels while within-sequence yaw/turning remains visible.
    initial_forward = root_rotation[0, :, 2]
    horizontal_norm = float(np.linalg.norm(initial_forward[:2]))
    if horizontal_norm <= 1e-6:
        raise KinematicDiscoveryError(
            "root forward axis has no horizontal component at frame 0; cannot canonicalize z-up yaw"
        )
    reference_heading = float(np.arctan2(initial_forward[1], initial_forward[0]))
    cosine = float(np.cos(reference_heading))
    sine = float(np.sin(reference_heading))
    world_to_initial_heading = np.asarray(
        ((cosine, sine, 0.0), (-sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=np.float32
    )
    canonical_rotations = rotations.copy()
    canonical_rotations[:, 0] = np.matmul(world_to_initial_heading[None, :, :], root_rotation)
    centered = root - root[:1]
    aligned = np.matmul(world_to_initial_heading[None, :, :], centered[..., None]).squeeze(-1)
    velocity = np.zeros_like(aligned)
    velocity[1:] = aligned[1:] - aligned[:-1]
    relative = np.matmul(np.swapaxes(canonical_rotations[:-1], -1, -2), canonical_rotations[1:])
    trace = np.trace(relative, axis1=-2, axis2=-1)
    angles = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)).astype(np.float32)
    angular_velocity = np.zeros((len(raw_motion), 24), dtype=np.float32)
    angular_velocity[1:] = angles
    # Keep the orthonormal first two axes (a valid 6-D rotation representation)
    # for all joints.  Root translation/orientation share one initial-heading
    # frame; no camera value is ever read or appended.
    pose6d = canonical_rotations[..., :2, :].reshape(len(raw_motion), -1).astype(np.float32, copy=False)
    descriptor = np.concatenate((contacts, aligned, velocity, pose6d, angular_velocity), axis=1)
    if not np.isfinite(descriptor).all():  # defensive; raw validation above should imply this
        raise KinematicDiscoveryError("kinematic descriptor contains non-finite values")
    return descriptor.astype(np.float32, copy=False)


def _sequence_seed(seed: int, sequence_id: str) -> int:
    """Derive per-sequence randomness without depending on held-out ordering.

    Segmentation is stochastic through K-Means initialization.  Deriving its
    seed from a stable sequence identity (rather than a position in the
    combined train/val/test list) ensures adding, removing, or reordering
    held-out rows cannot change a train-fitted vocabulary.
    """
    payload = "atomicdance-kinematic-discovery-seed-v2\\0{}\\0{}".format(seed, sequence_id)
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return value & ((1 << 63) - 1)


def _fit_standardizer(descriptors: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    total: Optional[np.ndarray] = None
    squared: Optional[np.ndarray] = None
    count = 0
    for descriptor in descriptors:
        values = np.asarray(descriptor, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise KinematicDiscoveryError("cannot fit statistics from an invalid descriptor")
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
            squared = np.zeros(values.shape[1], dtype=np.float64)
        if values.shape[1] != len(total):
            raise KinematicDiscoveryError("descriptor dimensions disagree during statistics fit")
        total += values.sum(axis=0)
        squared += np.square(values).sum(axis=0)
        count += len(values)
    if total is None or squared is None or count < 1:
        raise KinematicDiscoveryError("no train descriptors were available for statistics fit")
    mean = total / float(count)
    variance = np.maximum(squared / float(count) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    # Constant channels (commonly contacts in a short fixture) should remain
    # usable and map to zero rather than create division by zero.
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), count


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if values.shape[1] != len(mean) or mean.shape != std.shape:
        raise KinematicDiscoveryError("descriptor/statistics shape mismatch")
    return ((values - mean) / std).astype(np.float32, copy=False)


def _enforce_min_segment_length(boundaries: Sequence[int], frames: int, minimum: int) -> List[int]:
    if frames < 1 or minimum < 1:
        raise KinematicDiscoveryError("frames and minimum segment length must be positive")
    result = sorted(set(int(value) for value in boundaries if 0 <= int(value) <= frames))
    if not result or result[0] != 0:
        result.insert(0, 0)
    if result[-1] != frames:
        result.append(frames)
    while len(result) > 2:
        short = next(
            (index for index, (start, end) in enumerate(zip(result[:-1], result[1:])) if end - start < minimum),
            None,
        )
        if short is None:
            break
        # Remove the boundary toward the shorter temporal neighbor.  This is
        # deterministic and leaves a full [0,T) partition.
        if short == 0:
            del result[1]
        elif short == len(result) - 2:
            del result[-2]
        else:
            left = result[short] - result[short - 1]
            right = result[short + 2] - result[short + 1]
            del result[short if left <= right else short + 1]
    return result


def segment_sequence(
    standardized_descriptor: np.ndarray,
    *,
    segment_stride: int,
    target_segment_frames: int,
    minimum_segment_frames: int,
    iterations: int,
    seed: int,
) -> List[Tuple[int, int]]:
    """Segment a complete timeline using stride-sampled similarity features."""
    frames = len(standardized_descriptor)
    if frames < 2:
        raise KinematicDiscoveryError("cannot segment fewer than two frames")
    if segment_stride < 1 or target_segment_frames < 1 or minimum_segment_frames < 1:
        raise KinematicDiscoveryError("segment stride/length parameters must be positive")
    indices = np.arange(0, frames, segment_stride, dtype=np.int64)
    if len(indices) < 2:
        return [(0, frames)]
    sampled = torch.from_numpy(np.asarray(standardized_descriptor[indices], dtype=np.float32))
    target_samples = max(1, int(round(target_segment_frames / float(segment_stride))))
    local_clusters = min(len(sampled), max(2, int(round(len(sampled) / float(target_samples)))))
    minimum_samples = max(1, int(math.ceil(minimum_segment_frames / float(segment_stride))))
    if local_clusters <= 1:
        boundaries = [0, frames]
    else:
        _, cuts = adaptive_segment(
            sampled,
            sampled,
            num_clusters=local_clusters,
            min_length=minimum_samples,
            iterations=iterations,
            seed=seed,
        )
        boundaries = [0, *[int(indices[cut]) for cut in cuts], frames]
    boundaries = _enforce_min_segment_length(boundaries, frames, minimum_segment_frames)
    return list(zip(boundaries[:-1], boundaries[1:]))


def _dct_basis(frames: int, coefficients: int) -> np.ndarray:
    if frames < 1 or coefficients < 1 or coefficients > frames:
        raise KinematicDiscoveryError("invalid DCT frame/coefficient count")
    positions = np.arange(frames, dtype=np.float64) + 0.5
    frequencies = np.arange(coefficients, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * frequencies * positions[None, :] / float(frames))
    basis[0] *= math.sqrt(1.0 / float(frames))
    if coefficients > 1:
        basis[1:] *= math.sqrt(2.0 / float(frames))
    return basis.astype(np.float32)


def segment_embedding(
    standardized_descriptor: np.ndarray,
    start: int,
    end: int,
    *,
    resample_frames: int,
    dct_coefficients: int,
    dct_basis: np.ndarray,
) -> np.ndarray:
    if not (0 <= start < end <= len(standardized_descriptor)):
        raise KinematicDiscoveryError("invalid segment interval [{}, {})".format(start, end))
    if dct_basis.shape != (dct_coefficients, resample_frames):
        raise KinematicDiscoveryError("DCT basis shape mismatch")
    segment = torch.from_numpy(np.asarray(standardized_descriptor[start:end], dtype=np.float32))
    sampled = F.interpolate(
        segment.transpose(0, 1).unsqueeze(0),
        size=resample_frames,
        mode="linear",
        align_corners=True,
    ).squeeze(0).transpose(0, 1).numpy()
    transformed = np.matmul(dct_basis, sampled).reshape(-1)
    duration = np.asarray([math.log(float(end - start))], dtype=np.float32)
    embedding = np.concatenate((transformed.astype(np.float32, copy=False), duration), axis=0)
    if not np.isfinite(embedding).all():
        raise KinematicDiscoveryError("segment embedding is non-finite")
    return embedding.astype(np.float32, copy=False)


def _normalized_embeddings(events: Sequence[Event], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if not events:
        raise KinematicDiscoveryError("no events available for clustering")
    matrix = np.stack([event.embedding for event in events]).astype(np.float32, copy=False)
    if matrix.shape[1] != len(mean) or mean.shape != std.shape:
        raise KinematicDiscoveryError("embedding/statistics shape mismatch")
    standardized = (matrix - mean) / std
    norms = np.linalg.norm(standardized, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise KinematicDiscoveryError("a segment embedding is zero after train-only standardization")
    return (standardized / norms).astype(np.float32, copy=False)


def _cluster_thresholds(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    *,
    keep_quantile: float,
    minimum_support: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if not 0.0 < keep_quantile <= 1.0:
        raise KinematicDiscoveryError("keep_quantile must be in (0, 1]")
    if minimum_support < 1:
        raise KinematicDiscoveryError("minimum cluster support must be positive")
    distances = (embeddings - centers[labels]).norm(dim=-1)
    thresholds = torch.full((len(centers),), float("-inf"), dtype=torch.float32, device=centers.device)
    support = torch.zeros(len(centers), dtype=torch.long, device=centers.device)
    for cluster in range(len(centers)):
        values = distances[labels == cluster]
        support[cluster] = len(values)
        if len(values) < minimum_support:
            continue
        ordered = torch.sort(values).values
        index = min(len(ordered) - 1, max(0, int(math.ceil(keep_quantile * len(ordered))) - 1))
        thresholds[cluster] = ordered[index]
    temperature = float(torch.median(distances).item()) if len(distances) else 0.0
    return distances, thresholds, support, max(temperature, 1e-6)


def _event_predictions(
    embeddings: np.ndarray,
    centers: torch.Tensor,
    thresholds: torch.Tensor,
    support: torch.Tensor,
    *,
    minimum_support: int,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(centers.device)
    distances = torch.cdist(values, centers)
    nearest, labels = distances.min(dim=1)
    if distances.shape[1] > 1:
        second = torch.topk(distances, k=2, largest=False).values[:, 1]
    else:
        second = torch.full_like(nearest, float("inf"))
    probabilities = torch.softmax(-distances / float(temperature), dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    accepted = (support[labels] >= minimum_support) & (nearest <= thresholds[labels])
    return (
        labels.cpu().numpy().astype(np.int64),
        accepted.cpu().numpy().astype(bool),
        probabilities.cpu().numpy().astype(np.float32),
        entropy.cpu().numpy().astype(np.float32),
        torch.stack((nearest, second - nearest), dim=1).cpu().numpy().astype(np.float32),
    )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as error:
        raise KinematicDiscoveryError("invalid torch device {!r}: {}".format(value, error)) from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise KinematicDiscoveryError("CUDA device {!r} was requested but CUDA is unavailable".format(value))
    return device


def _sequence_descriptor(sequence: SequenceInput) -> np.ndarray:
    return kinematic_frame_descriptor(
        _load_motion(sequence.raw_motion_path, expected_frames=sequence.frame_count, kind="raw")
    )


def _fit_train_frame_statistics(sequences: Sequence[SequenceInput]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Read only train raw arrays while fitting descriptor statistics."""
    train = [sequence for sequence in sequences if sequence.split == "train"]
    if not train:
        raise KinematicDiscoveryError("no train sequences are available for discovery fit")
    return _fit_standardizer(_sequence_descriptor(sequence) for sequence in train)


def _all_events(
    sequences: Sequence[SequenceInput],
    *,
    frame_mean: np.ndarray,
    frame_std: np.ndarray,
    segment_stride: int,
    target_segment_frames: int,
    minimum_segment_frames: int,
    segment_iterations: int,
    seed: int,
    embedding_frames: int,
    dct_coefficients: int,
) -> List[Event]:
    basis = _dct_basis(embedding_frames, dct_coefficients)
    events: List[Event] = []
    for sequence in sequences:
        # The model-input asset is deliberately opened even though descriptors
        # come from raw motion: labels must be bound to this exact normalized
        # tensor later consumed by the planner/completion trainer.
        _load_motion(
            sequence.normalized_motion_path,
            expected_frames=sequence.frame_count,
            kind="normalized",
        )
        standardized = _standardize(_sequence_descriptor(sequence), frame_mean, frame_std)
        spans = segment_sequence(
            standardized,
            segment_stride=segment_stride,
            target_segment_frames=target_segment_frames,
            minimum_segment_frames=minimum_segment_frames,
            iterations=segment_iterations,
            seed=_sequence_seed(seed, sequence.sequence_id),
        )
        if not spans or spans[0][0] != 0 or spans[-1][1] != sequence.frame_count:
            raise KinematicDiscoveryError("segmenter did not cover complete sequence {!r}".format(sequence.sequence_id))
        previous = 0
        for start, end in spans:
            if start != previous or end <= start:
                raise KinematicDiscoveryError("segmenter produced a non-contiguous partition")
            if end - start < minimum_segment_frames and len(spans) > 1:
                raise KinematicDiscoveryError("segmenter left a shorter-than-minimum event")
            events.append(
                Event(
                    sequence=sequence,
                    start=start,
                    end=end,
                    embedding=segment_embedding(
                        standardized,
                        start,
                        end,
                        resample_frames=embedding_frames,
                        dct_coefficients=dct_coefficients,
                        dct_basis=basis,
                    ),
                )
            )
            previous = end
    if not events:
        raise KinematicDiscoveryError("segmenter produced no events")
    return events


def _fit_train_embedding_statistics(events: Sequence[Event]) -> Tuple[np.ndarray, np.ndarray, int]:
    train_events = [event for event in events if event.sequence.split == "train"]
    if not train_events:
        raise KinematicDiscoveryError("no train events are available for K-Means fit")
    return _fit_standardizer([np.stack([event.embedding for event in train_events])])


def _new_output_dir(output_dir: Path) -> Tuple[Path, Path]:
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("refusing to overwrite immutable label output {}".format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = output_dir.parent / ".{}.kinematic-labels.lock".format(output_dir.name)
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise KinematicDiscoveryError("another label build holds lock {}".format(lock)) from error
    try:
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError("refusing to overwrite immutable label output {}".format(output_dir))
        staging = Path(tempfile.mkdtemp(prefix=".{}-staging-".format(output_dir.name), dir=str(output_dir.parent)))
    except Exception:
        lock.rmdir()
        raise
    return staging, lock


def _save_array(staging: Path, public: Path, values: np.ndarray) -> Tuple[Path, str]:
    staging.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(staging), values, allow_pickle=False)
    if staging.suffix != ".npy":  # pragma: no cover - callers always pass .npy
        raise AssertionError("label arrays must use .npy")
    if not staging.is_file():  # pragma: no cover - defensive np.save contract check
        raise RuntimeError("failed to write label array {}".format(staging))
    return public, _sha256_file(staging)


def _sequence_label_arrays(
    sequence: SequenceInput,
    events: Sequence[Event],
    *,
    predicted_labels: Mapping[int, int],
    predicted_accepted: Mapping[int, bool],
    predicted_probabilities: Mapping[int, np.ndarray],
    predicted_entropy: Mapping[int, float],
    predicted_distance_margin: Mapping[int, np.ndarray],
    transition_gutter: int,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.full(sequence.frame_count, INVALID_LABEL, dtype=np.int64)
    valid = np.zeros(sequence.frame_count, dtype=bool)
    probabilities = np.zeros((sequence.frame_count, num_classes + 1), dtype=np.float32)
    entropy = np.full(sequence.frame_count, np.nan, dtype=np.float32)
    distance = np.full(sequence.frame_count, np.nan, dtype=np.float32)
    margin = np.full(sequence.frame_count, np.nan, dtype=np.float32)
    for event in events:
        index = id(event)
        if not predicted_accepted[index]:
            continue
        label = predicted_labels[index] + 1
        labels[event.start : event.end] = label
        valid[event.start : event.end] = True
        probabilities[event.start : event.end, 1:] = predicted_probabilities[index]
        entropy[event.start : event.end] = predicted_entropy[index]
        distance[event.start : event.end] = predicted_distance_margin[index][0]
        margin[event.start : event.end] = predicted_distance_margin[index][1]
    if transition_gutter:
        for left, right in zip(events[:-1], events[1:]):
            if not (predicted_accepted[id(left)] and predicted_accepted[id(right)]):
                continue
            boundary = left.end
            start = max(left.start, boundary - transition_gutter)
            end = min(right.end, boundary + transition_gutter)
            labels[start:end] = 0
            valid[start:end] = True
            probabilities[start:end] = 0.0
            probabilities[start:end, 0] = 1.0
            entropy[start:end] = 0.0
            distance[start:end] = 0.0
            margin[start:end] = np.inf
    if np.any(labels[~valid] != INVALID_LABEL):  # defensive invariant
        raise AssertionError("invalid pseudo-label frame was not sentinel -1")
    if np.any((labels[valid] < 0) | (labels[valid] > num_classes)):
        raise AssertionError("valid pseudo-label is outside the declared label space")
    return labels, valid, probabilities, entropy, distance, margin


def discover_kinematic_atomics(
    source_manifest: Path,
    sequence_manifest: Path,
    output_dir: Path,
    *,
    num_classes: int = 100,
    keep_quantile: float = 0.8,
    minimum_cluster_support: int = 3,
    segment_stride: int = 6,
    target_segment_frames: int = 45,
    minimum_segment_frames: int = 18,
    transition_gutter: int = 3,
    segment_iterations: int = 8,
    kmeans_iterations: int = 100,
    embedding_frames: int = 60,
    dct_coefficients: int = 8,
    seed: int = 20260806,
    device: str = "auto",
) -> Dict[str, Any]:
    """Fit a 3D kinematic atomic vocabulary on train and apply it to all splits."""
    if num_classes < 1:
        raise KinematicDiscoveryError("num_classes must be positive")
    if not 0.0 < keep_quantile <= 1.0:
        raise KinematicDiscoveryError("keep_quantile must be in (0, 1]")
    if minimum_cluster_support < 1 or segment_stride < 1 or target_segment_frames < 1:
        raise KinematicDiscoveryError("cluster/segment parameters must be positive")
    if minimum_segment_frames < 2:
        raise KinematicDiscoveryError("minimum_segment_frames must be at least 2")
    if transition_gutter < 0 or segment_iterations < 1 or kmeans_iterations < 1:
        raise KinematicDiscoveryError("gutter and iteration values are invalid")
    if embedding_frames < 2 or not 1 <= dct_coefficients <= embedding_frames:
        raise KinematicDiscoveryError("invalid embedding/DCT dimensions")
    source_manifest = source_manifest.expanduser().resolve()
    sequence_manifest = sequence_manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_hash = _sha256_file(source_manifest)
    sequence_hash = _sha256_file(sequence_manifest)
    torch_device = _resolve_device(device)
    sources = _parse_sources(_read_jsonl(source_manifest, "source"))
    sequences = _parse_sequences(
        _read_jsonl(sequence_manifest, "sequence"), sources=sources, manifest_path=sequence_manifest
    )
    if not any(sequence.split == "val" for sequence in sequences):
        raise KinematicDiscoveryError("source release has no validation sequences")
    if not any(sequence.split == "test" for sequence in sequences):
        raise KinematicDiscoveryError("source release has no test sequences")
    normalizer_hashes = {sequence.normalizer_artifact_sha256 for sequence in sequences}
    if len(normalizer_hashes) != 1:
        raise KinematicDiscoveryError("all label inputs must share one frozen train-only normalizer artifact")

    # Fit the first statistics pass before reading a single val/test raw array.
    frame_mean, frame_std, train_frame_count = _fit_train_frame_statistics(sequences)
    events = _all_events(
        sequences,
        frame_mean=frame_mean,
        frame_std=frame_std,
        segment_stride=segment_stride,
        target_segment_frames=target_segment_frames,
        minimum_segment_frames=minimum_segment_frames,
        segment_iterations=segment_iterations,
        seed=seed,
        embedding_frames=embedding_frames,
        dct_coefficients=dct_coefficients,
    )
    train_events = [event for event in events if event.sequence.split == "train"]
    if len(train_events) < num_classes:
        raise KinematicDiscoveryError(
            "need at least {} train events for {} clusters, found {}".format(
                num_classes, num_classes, len(train_events)
            )
        )
    embedding_mean, embedding_std, train_event_count = _fit_train_embedding_statistics(events)
    train_embeddings = _normalized_embeddings(train_events, embedding_mean, embedding_std)
    train_tensor = torch.from_numpy(train_embeddings).to(torch_device)
    train_cluster_labels, centers = kmeans(
        train_tensor, num_clusters=num_classes, iterations=kmeans_iterations, seed=seed
    )
    _, thresholds, support, temperature = _cluster_thresholds(
        train_tensor,
        train_cluster_labels,
        centers,
        keep_quantile=keep_quantile,
        minimum_support=minimum_cluster_support,
    )
    all_embeddings = _normalized_embeddings(events, embedding_mean, embedding_std)
    labels, accepted, probabilities, entropy, distance_margin = _event_predictions(
        all_embeddings,
        centers,
        thresholds,
        support,
        minimum_support=minimum_cluster_support,
        temperature=temperature,
    )
    event_by_sequence: Dict[str, List[Event]] = {}
    predicted_labels: Dict[int, int] = {}
    predicted_accepted: Dict[int, bool] = {}
    predicted_probabilities: Dict[int, np.ndarray] = {}
    predicted_entropy: Dict[int, float] = {}
    predicted_distance_margin: Dict[int, np.ndarray] = {}
    for event, label, event_accepted, probability, event_entropy, event_distance_margin in zip(
        events, labels, accepted, probabilities, entropy, distance_margin
    ):
        event_by_sequence.setdefault(event.sequence.sequence_id, []).append(event)
        predicted_labels[id(event)] = int(label)
        predicted_accepted[id(event)] = bool(event_accepted)
        predicted_probabilities[id(event)] = np.asarray(probability, dtype=np.float32)
        predicted_entropy[id(event)] = float(event_entropy)
        predicted_distance_margin[id(event)] = np.asarray(event_distance_margin, dtype=np.float32)

    staging, lock = _new_output_dir(output_dir)
    published = False
    try:
        label_root = staging / "labels"
        public_label_root = output_dir / "labels"
        label_root.mkdir()
        producer_path = staging / "producer.npz"
        config = {
            "schema_version": SCHEMA_VERSION,
            "producer_version": PRODUCER_VERSION,
            "label_space_id": "kinematic_atomic_{}_{}".format(num_classes, LABEL_SPACE_VERSION),
            "num_atomic_classes": num_classes,
            "segmenter": {
                "algorithm": "adaptive_similarity_on_train_standardized_3d_kinematic_descriptor",
                "segment_stride": segment_stride,
                "target_segment_frames": target_segment_frames,
                "minimum_segment_frames": minimum_segment_frames,
                "iterations": segment_iterations,
            },
            "embedding": {
                "kind": "resample_dct_3d_kinematic_descriptor_plus_log_duration",
                "frames": embedding_frames,
                "dct_coefficients": dct_coefficients,
            },
            "clustering": {
                "algorithm": "deterministic_kmeans",
                "iterations": kmeans_iterations,
                "seed": seed,
                "fit_split": "train",
                "minimum_cluster_support": minimum_cluster_support,
                "device": str(torch_device),
            },
            "acceptance": {
                "distance_quantile_fit_on_train_only": keep_quantile,
                "transition_gutter_frames": transition_gutter,
                "invalid_label": INVALID_LABEL,
                "invalid_label_policy": "unknown_is_minus_one_with_false_mask_never_transition_zero",
            },
            "representation": {
                "motion": MODEL_MOTION_REPRESENTATION,
                "coordinate_system": MODEL_COORDINATE_SYSTEM,
                "input_normalization": "normalized",
                "camera_in_model_input": False,
            },
        }
        np.savez_compressed(
            str(producer_path),
            frame_mean=frame_mean,
            frame_std=frame_std,
            embedding_mean=embedding_mean,
            embedding_std=embedding_std,
            centers=centers.cpu().numpy().astype(np.float32),
            cluster_distance_thresholds=thresholds.cpu().numpy().astype(np.float32),
            cluster_support=support.cpu().numpy().astype(np.int64),
            softmax_temperature=np.asarray([temperature], dtype=np.float32),
            config_json=np.asarray([json.dumps(config, sort_keys=True, separators=(",", ":"))]),
        )
        producer_hash = _sha256_file(producer_path)
        label_space_id = config["label_space_id"]
        label_rows: List[Dict[str, Any]] = []
        counts_by_split: Dict[str, Dict[str, int]] = {
            split: {"sequences": 0, "frames": 0, "valid_frames": 0, "transition_frames": 0}
            for split in sorted(ALLOWED_SPLITS)
        }
        for sequence in sequences:
            sequence_events = event_by_sequence.get(sequence.sequence_id)
            if not sequence_events:
                raise KinematicDiscoveryError("no events for sequence {!r}".format(sequence.sequence_id))
            arrays = _sequence_label_arrays(
                sequence,
                sequence_events,
                predicted_labels=predicted_labels,
                predicted_accepted=predicted_accepted,
                predicted_probabilities=predicted_probabilities,
                predicted_entropy=predicted_entropy,
                predicted_distance_margin=predicted_distance_margin,
                transition_gutter=transition_gutter,
                num_classes=num_classes,
            )
            labels_array, mask_array, probabilities_array, entropy_array, distance_array, margin_array = arrays
            token = hashlib.sha256(sequence.sequence_id.encode("utf-8")).hexdigest()
            staging_dir = label_root / token
            public_dir = public_label_root / token
            staging_dir.mkdir()
            # ``public_dir`` deliberately points into the not-yet-existing
            # destination.  Do not create it before the atomic rename; it is
            # used only as the future absolute path recorded in the manifest.
            labels_path, labels_hash = _save_array(staging_dir / "labels.npy", public_dir / "labels.npy", labels_array)
            mask_path, mask_hash = _save_array(staging_dir / "label_valid_mask.npy", public_dir / "label_valid_mask.npy", mask_array)
            confidence = np.where(np.isfinite(distance_array), 1.0 / (1.0 + distance_array), 0.0).astype(np.float32)
            confidence_path, confidence_hash = _save_array(
                staging_dir / "confidence.npy", public_dir / "confidence.npy", confidence
            )
            entropy_path, entropy_hash = _save_array(staging_dir / "entropy.npy", public_dir / "entropy.npy", entropy_array)
            probabilities_path, probabilities_hash = _save_array(
                staging_dir / "probabilities.npy", public_dir / "probabilities.npy", probabilities_array
            )
            distance_path, distance_hash = _save_array(
                staging_dir / "nearest_distance.npy", public_dir / "nearest_distance.npy", distance_array
            )
            margin_path, margin_hash = _save_array(staging_dir / "distance_margin.npy", public_dir / "distance_margin.npy", margin_array)
            label_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "accepted",
                    "sequence_id": sequence.sequence_id,
                    "recording_id": sequence.recording_id,
                    "retrieval_group_id": sequence.retrieval_group_id,
                    "duplicate_content_group_id": sequence.duplicate_content_group_id,
                    "split": sequence.split,
                    "labels_path": str(labels_path),
                    "labels_sha256": labels_hash,
                    "label_valid_mask_path": str(mask_path),
                    "label_valid_mask_sha256": mask_hash,
                    "confidence_path": str(confidence_path),
                    "confidence_sha256": confidence_hash,
                    "entropy_path": str(entropy_path),
                    "entropy_sha256": entropy_hash,
                    "probabilities_path": str(probabilities_path),
                    "probabilities_sha256": probabilities_hash,
                    "nearest_distance_path": str(distance_path),
                    "nearest_distance_sha256": distance_hash,
                    "distance_margin_path": str(margin_path),
                    "distance_margin_sha256": margin_hash,
                    "label_space_id": label_space_id,
                    "producer_version": PRODUCER_VERSION,
                    "producer_artifact": str((output_dir / "producer.npz").resolve()),
                    "producer_artifact_sha256": producer_hash,
                    "fit_split": "train",
                    "fit_source_manifest_sha256": source_hash,
                    "fit_sequence_manifest_sha256": sequence_hash,
                    "input_motion_sha256": sequence.normalized_motion_sha256,
                    "input_raw_motion_sha256": sequence.raw_motion_sha256,
                    "input_motion_representation_id": MODEL_MOTION_REPRESENTATION,
                    "input_normalization_state": "normalized",
                    "input_coordinate_system": MODEL_COORDINATE_SYSTEM,
                    "input_normalization_artifact_sha256": sequence.normalizer_artifact_sha256,
                    "frame_count": sequence.frame_count,
                    "valid_frames": int(mask_array.sum()),
                    "transition_frames": int(np.sum(labels_array[mask_array] == 0)),
                }
            )
            split_counts = counts_by_split[sequence.split]
            split_counts["sequences"] += 1
            split_counts["frames"] += sequence.frame_count
            split_counts["valid_frames"] += int(mask_array.sum())
            split_counts["transition_frames"] += int(np.sum(labels_array[mask_array] == 0))
        # Destination paths above are deliberately not created before the
        # atomic rename.  Remove the placeholder nesting that Python created
        # only if it is inside staging (it is not for a public destination).
        # All rows are sorted by a stable sequence identity before publication.
        label_rows.sort(key=lambda row: row["sequence_id"])
        _write_jsonl(staging / "labels.jsonl", label_rows)
        report = {
            "schema_version": SCHEMA_VERSION,
            "producer_version": PRODUCER_VERSION,
            "publication": "immutable_new_directory_only_atomic_rename",
            "input": {
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": source_hash,
                "sequence_manifest": str(sequence_manifest),
                "sequence_manifest_sha256": sequence_hash,
                "source_ids_sha256": _sha256_strings(sorted(sources)),
                "sequence_ids_sha256": _sha256_strings([sequence.sequence_id for sequence in sequences]),
            },
            "fit": {
                "fit_split": "train",
                "train_frame_count": train_frame_count,
                "train_event_count": train_event_count,
                "normalizer_artifact_sha256": next(iter(normalizer_hashes)),
                "producer_artifact": str((output_dir / "producer.npz").resolve()),
                "producer_artifact_sha256": producer_hash,
            },
            "config": config,
            "counts": {
                "events": len(events),
                "events_by_split": {
                    split: sum(event.sequence.split == split for event in events) for split in sorted(ALLOWED_SPLITS)
                },
                "labels_by_split": counts_by_split,
                "supported_clusters": int((support >= minimum_cluster_support).sum().item()),
                "total_clusters": num_classes,
            },
            "artifacts": {
                "producer.npz": producer_hash,
                "labels.jsonl": _sha256_file(staging / "labels.jsonl"),
            },
            "label_policy": config["acceptance"],
        }
        (staging / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.rename(str(staging), str(output_dir))
        published = True
        result = copy.deepcopy(report)
        result["output_dir"] = str(output_dir)
        return result
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
        if lock.exists():
            lock.rmdir()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="frozen source-level sources.jsonl")
    parser.add_argument("--sequences", required=True, help="normalized full-timeline sequences JSONL")
    parser.add_argument("--output-dir", required=True, help="new immutable labels output directory")
    parser.add_argument("--num-classes", type=int, default=100, help="non-transition atomic classes")
    parser.add_argument("--keep-quantile", type=float, default=0.8, help="train-only per-cluster distance quantile")
    parser.add_argument("--minimum-cluster-support", type=int, default=3)
    parser.add_argument("--segment-stride", type=int, default=6)
    parser.add_argument("--target-segment-frames", type=int, default=45)
    parser.add_argument("--minimum-segment-frames", type=int, default=18)
    parser.add_argument("--transition-gutter", type=int, default=3)
    parser.add_argument("--segment-iterations", type=int, default=8)
    parser.add_argument("--kmeans-iterations", type=int, default=100)
    parser.add_argument("--embedding-frames", type=int, default=60)
    parser.add_argument("--dct-coefficients", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="auto", help="auto, cpu, or an explicit Torch device such as cuda:0")
    args = parser.parse_args(argv)
    try:
        report = discover_kinematic_atomics(
            Path(args.sources),
            Path(args.sequences),
            Path(args.output_dir),
            num_classes=args.num_classes,
            keep_quantile=args.keep_quantile,
            minimum_cluster_support=args.minimum_cluster_support,
            segment_stride=args.segment_stride,
            target_segment_frames=args.target_segment_frames,
            minimum_segment_frames=args.minimum_segment_frames,
            transition_gutter=args.transition_gutter,
            segment_iterations=args.segment_iterations,
            kmeans_iterations=args.kmeans_iterations,
            embedding_frames=args.embedding_frames,
            dct_coefficients=args.dct_coefficients,
            seed=args.seed,
            device=args.device,
        )
    except (KinematicDiscoveryError, FileExistsError, FileNotFoundError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
