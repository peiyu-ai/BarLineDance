#!/usr/bin/env python3
"""Apply a frozen train-only AtomicDance motion normalizer immutably.

``tools/fit_motion_normalizer.py`` deliberately only computes statistics.  It
does not change any source sequence, because doing so in place would make it
too easy to lose the raw z-up/body-only record or to apply a re-fitted scaler
to validation/test data.  This companion command is the only supported
materialization step:

* it binds the supplied source and sequence manifests byte-for-byte to the
  normalizer fit report before reading a motion array;
* it audits every source and sequence identity so no recording, retrieval
  group, or known duplicate group spans splits;
* it requires every input sequence to explicitly attest the *raw*
  AtomicDance-151D z-up/body-only contract; and
* it writes a new, immutable bundle.  Raw arrays and their input manifest are
  never modified.

The output manifest retains the non-motion asset *meaning* (including source,
audio, and camera provenance), but canonicalizes declared local paths against
the input manifest before publishing them in the new bundle.  This matters
because an input-relative path would otherwise be interpreted relative to the
new output manifest.  Its model-input motion is a separate
``assets.motion_151_normalized`` artifact with a content hash.  Camera values
are never read or appended to the 151-D tensor.

The transform intentionally has no clipping or repair path.  It is exactly
the released feature-range convention, with an explicit constant-dimension
rule::

    safe_range = 1 where data_max == data_min, else data_max - data_min
    normalized = 2 * (raw - data_min) / safe_range - 1

For a constant training dimension, a raw value equal to that constant maps to
``-1`` and inverses exactly under the same safe range.  Validation/test rows
use the frozen train artifact unchanged and can therefore legitimately fall
outside ``[-1, 1]``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# Invoked as `python tools/apply_motion_normalizer.py`, sys.path[0] is tools/,
# not the repo root; without this line the sibling import below only works when
# the ambient PYTHONPATH happens to include the CWD.  It did, so this ran for
# months and failed the moment it was run under `env -u PYTHONPATH`.
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fit_motion_normalizer import FIT_SPLIT, MOTION_DIM, RAW_REPRESENTATION_CONTRACT  # noqa: E402


SCHEMA_VERSION = "atomicdance-motion-normalizer-apply-v1"
OUTPUT_SEQUENCE_MANIFEST = "sequences_normalized.jsonl"
OUTPUT_REPORT = "report.json"
OUTPUT_MOTION_DIR = "motion_151_normalized"
ALLOWED_SPLITS = frozenset(("train", "val", "test"))
DEFAULT_BLOCK_FRAMES = 8192

# The raw source manifest and the normalized manifest deliberately live in
# different immutable directories.  Only these schema-defined fields are
# local artifact locations; IDs, URIs, labels, and arbitrary provenance text
# must not be treated as paths merely because they are strings.
_TOP_LEVEL_LOCAL_PATH_FIELDS = frozenset(
    (
        "motion_path",
        "music_path",
        "frame_ids_path",
        "camera_path",
        "source_cache",
        "source_video",
        "source_video_path",
        "audio_path",
        "audio_features_path",
        "pose2d_path",
        "pose2d_scores_path",
    )
)
_ASSET_LOCAL_PATH_FIELDS = frozenset(
    (
        "motion_151_raw",
        "motion_151_normalized",
        "motion_151_model_input",
        "music_35",
        "frame_ids",
        "camera",
        "camera_path",
        "source_cache",
        "source_video",
        "source_video_path",
        "audio",
        "audio_path",
        "audio_features",
        "audio_features_path",
        "pose2d",
        "pose2d_path",
        "pose2d_scores",
        "pose2d_scores_path",
    )
)
_TIMELINE_LOCAL_PATH_FIELDS = frozenset(("frame_ids_path",))


class NormalizerApplicationError(ValueError):
    """A normalizer or raw-sequence contract makes publication unsafe."""


@dataclass(frozen=True)
class SourceIdentity:
    recording_id: str
    retrieval_group_id: str
    duplicate_content_group_id: Optional[str]
    split: str


@dataclass(frozen=True)
class RawSequence:
    row: Mapping[str, Any]
    sequence_id: str
    recording_id: str
    retrieval_group_id: str
    duplicate_content_group_id: Optional[str]
    split: str
    raw_motion_value: str


@dataclass(frozen=True)
class FrozenNormalizer:
    bundle_dir: Path
    normalizer_path: Path
    report_path: Path
    artifact_sha256: str
    report_sha256: str
    data_min: np.ndarray
    data_max: np.ndarray
    safe_range: np.ndarray
    report: Mapping[str, Any]


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


def _require_sha256(value: object, *, context: str) -> str:
    if not _is_sha256(value):
        raise NormalizerApplicationError("{} must be a SHA-256 hex digest".format(context))
    return str(value).lower()


def _require_string(record: Mapping[str, Any], field: str, *, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise NormalizerApplicationError("{} lacks non-empty {}".format(context, field))
    return value


def _require_split(record: Mapping[str, Any], *, context: str) -> str:
    split = _require_string(record, "split", context=context)
    if split not in ALLOWED_SPLITS:
        raise NormalizerApplicationError(
            "{} has unsupported split {!r}; expected one of {}".format(
                context, split, ", ".join(sorted(ALLOWED_SPLITS))
            )
        )
    return split


def _require_explicit_optional_group(
    record: Mapping[str, Any], field: str, *, context: str
) -> Optional[str]:
    """Read a nullable group without silently inventing provenance."""
    if field not in record:
        raise NormalizerApplicationError(
            "{} must explicitly include {} (string or null)".format(context, field)
        )
    value = record[field]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise NormalizerApplicationError(
            "{} has invalid {}; expected a non-empty string or null".format(context, field)
        )
    return value


def _require_mapping(record: Mapping[str, Any], field: str, *, context: str) -> Mapping[str, Any]:
    value = record.get(field)
    if not isinstance(value, Mapping):
        raise NormalizerApplicationError("{} lacks object {}".format(context, field))
    return value


def _read_jsonl(path: Path, *, kind: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("{} JSONL manifest does not exist: {}".format(kind, path))
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise NormalizerApplicationError(
                    "{} JSONL is invalid at {}:{}: {}".format(kind, path, line_number, error)
                ) from error
            if not isinstance(value, Mapping):
                raise NormalizerApplicationError(
                    "{} JSONL row {} is not an object".format(kind, line_number)
                )
            records.append(dict(value))
    if not records:
        raise NormalizerApplicationError("{} JSONL manifest is empty: {}".format(kind, path))
    return records


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _register_group_owner(
    owners: Dict[Tuple[str, str], str],
    *,
    kind: str,
    group_id: str,
    split: str,
    context: str,
) -> None:
    key = (kind, group_id)
    previous = owners.get(key)
    if previous is not None and previous != split:
        raise NormalizerApplicationError(
            "source-safety violation: {} {!r} appears in both {!r} and {!r} ({})".format(
                kind, group_id, previous, split, context
            )
        )
    owners[key] = split


def _parse_sources(records: Sequence[Mapping[str, Any]]) -> Dict[str, SourceIdentity]:
    """Require complete source identity and reject every known split collision."""
    sources: Dict[str, SourceIdentity] = {}
    owners: Dict[Tuple[str, str], str] = {}
    for index, record in enumerate(records, 1):
        context = "source manifest row {}".format(index)
        recording_id = _require_string(record, "recording_id", context=context)
        retrieval_group_id = _require_string(record, "retrieval_group_id", context=context)
        duplicate_group_id = _require_explicit_optional_group(
            record, "duplicate_content_group_id", context=context
        )
        split = _require_split(record, context=context)
        if recording_id in sources:
            raise NormalizerApplicationError(
                "source manifest has duplicate recording_id {!r}".format(recording_id)
            )
        _register_group_owner(
            owners,
            kind="recording_id",
            group_id=recording_id,
            split=split,
            context=context,
        )
        _register_group_owner(
            owners,
            kind="retrieval_group_id",
            group_id=retrieval_group_id,
            split=split,
            context=context,
        )
        if duplicate_group_id is not None:
            _register_group_owner(
                owners,
                kind="duplicate_content_group_id",
                group_id=duplicate_group_id,
                split=split,
                context=context,
            )
        sources[recording_id] = SourceIdentity(
            recording_id=recording_id,
            retrieval_group_id=retrieval_group_id,
            duplicate_content_group_id=duplicate_group_id,
            split=split,
        )
    return sources


def _assert_raw_representation(representation: Mapping[str, Any], *, context: str) -> None:
    """Reject normalizer-chain ambiguity rather than trying to infer a state."""
    for field, expected in RAW_REPRESENTATION_CONTRACT.items():
        actual = representation.get(field)
        if actual != expected:
            raise NormalizerApplicationError(
                "{} has unsafe raw representation.{}={!r}; expected {!r}".format(
                    context, field, actual, expected
                )
            )
    # ``normalization_state`` was added to derived manifests after the fitter
    # contract was written.  It is optional for a raw legacy row, but when it
    # is present it must corroborate (not contradict) ``normalization: raw``.
    if "normalization_state" in representation and representation["normalization_state"] != "raw":
        raise NormalizerApplicationError(
            "{} has unsafe raw representation.normalization_state={!r}; expected 'raw'".format(
                context, representation["normalization_state"]
            )
        )
    for field in ("normalizer_artifact_sha256", "normalization_artifact_sha256"):
        if field in representation and representation[field] is not None:
            raise NormalizerApplicationError(
                "{} is declared raw but representation.{} is not null".format(context, field)
            )
    for field in ("normalizer_fit_split", "normalization_fit_split"):
        if field in representation and representation[field] is not None:
            raise NormalizerApplicationError(
                "{} is declared raw but representation.{} is not null".format(context, field)
            )


def _assert_top_level_raw_contract(record: Mapping[str, Any], *, context: str) -> None:
    """Catch mixed old/new manifest fields before a derived asset is written."""
    expected = {
        "motion_representation_id": RAW_REPRESENTATION_CONTRACT["motion"],
        "coordinate_system": RAW_REPRESENTATION_CONTRACT["coordinate_system"],
        "normalization_state": "raw",
    }
    for field, value in expected.items():
        if field in record and record[field] != value:
            raise NormalizerApplicationError(
                "{} has unsafe top-level {}={!r}; expected {!r}".format(
                    context, field, record[field], value
                )
            )
    for field in (
        "normalizer_artifact_sha256",
        "normalizer_fit_split",
        "normalization_artifact_sha256",
        "normalization_fit_split",
    ):
        if field in record and record[field] is not None:
            raise NormalizerApplicationError(
                "{} is declared raw but top-level {} is not null".format(context, field)
            )
    # Some callers expose the same state as a top-level object rather than in
    # ``representation``.  Do not let a contradictory derived-state marker
    # sneak through simply because the nested contract still says ``raw``.
    if "normalization" in record:
        value = record["normalization"]
        if isinstance(value, str):
            if value != "raw":
                raise NormalizerApplicationError(
                    "{} has unsafe top-level normalization={!r}; expected 'raw'".format(context, value)
                )
        elif isinstance(value, Mapping):
            state = value.get("state", value.get("normalization_state"))
            if state != "raw":
                raise NormalizerApplicationError(
                    "{} has unsafe top-level normalization state={!r}; expected 'raw'".format(context, state)
                )
            for field in (
                "normalizer_artifact_sha256",
                "normalization_artifact_sha256",
                "normalizer_fit_split",
                "normalization_fit_split",
            ):
                if field in value and value[field] is not None:
                    raise NormalizerApplicationError(
                        "{} is declared raw but top-level normalization.{} is not null".format(
                            context, field
                        )
                    )
        else:
            raise NormalizerApplicationError(
                "{} top-level normalization must be 'raw' or a raw-state object".format(context)
            )


def _parse_raw_sequences(
    records: Sequence[Mapping[str, Any]], *, sources: Mapping[str, SourceIdentity]
) -> List[RawSequence]:
    """Join every sequence to the frozen source partition before processing."""
    seen_sequences: set[str] = set()
    owners: Dict[Tuple[str, str], str] = {}
    sequences: List[RawSequence] = []
    for index, record in enumerate(records, 1):
        context = "sequence manifest row {}".format(index)
        sequence_id = _require_string(record, "sequence_id", context=context)
        if sequence_id in seen_sequences:
            raise NormalizerApplicationError(
                "sequence manifest has duplicate sequence_id {!r}".format(sequence_id)
            )
        seen_sequences.add(sequence_id)
        recording_id = _require_string(record, "recording_id", context=context)
        retrieval_group_id = _require_string(record, "retrieval_group_id", context=context)
        duplicate_group_id = _require_explicit_optional_group(
            record, "duplicate_content_group_id", context=context
        )
        split = _require_split(record, context=context)
        source = sources.get(recording_id)
        if source is None:
            raise NormalizerApplicationError(
                "{} references recording_id {!r} absent from source manifest".format(context, recording_id)
            )
        for field, actual, expected in (
            ("retrieval_group_id", retrieval_group_id, source.retrieval_group_id),
            ("duplicate_content_group_id", duplicate_group_id, source.duplicate_content_group_id),
            ("split", split, source.split),
        ):
            if actual != expected:
                raise NormalizerApplicationError(
                    "{} {} {!r} disagrees with source-manifest {!r}".format(
                        context, field, actual, expected
                    )
                )
        for kind, group_id in (
            ("recording_id", recording_id),
            ("retrieval_group_id", retrieval_group_id),
        ):
            _register_group_owner(owners, kind=kind, group_id=group_id, split=split, context=context)
        if duplicate_group_id is not None:
            _register_group_owner(
                owners,
                kind="duplicate_content_group_id",
                group_id=duplicate_group_id,
                split=split,
                context=context,
            )
        assets = _require_mapping(record, "assets", context=context)
        raw_motion_value = assets.get("motion_151_raw")
        if not isinstance(raw_motion_value, str) or not raw_motion_value.strip():
            raise NormalizerApplicationError(
                "{} lacks non-empty assets.motion_151_raw".format(context)
            )
        for field in (
            "motion_151_normalized",
            "motion_151_normalized_sha256",
            "motion_151_model_input",
            "motion_151_model_input_sha256",
        ):
            if field in assets and assets[field] is not None:
                raise NormalizerApplicationError(
                    "{} is a raw input manifest but assets.{} is already populated".format(context, field)
                )
        representation = _require_mapping(record, "representation", context=context)
        _assert_raw_representation(representation, context=context)
        _assert_top_level_raw_contract(record, context=context)
        sequences.append(
            RawSequence(
                row=record,
                sequence_id=sequence_id,
                recording_id=recording_id,
                retrieval_group_id=retrieval_group_id,
                duplicate_content_group_id=duplicate_group_id,
                split=split,
                raw_motion_value=raw_motion_value,
            )
        )
    return sorted(sequences, key=lambda item: item.sequence_id)


def _load_json_object(path: Path, *, context: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("{} does not exist: {}".format(context, path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NormalizerApplicationError("cannot read {} {}: {}".format(context, path, error)) from error
    if not isinstance(value, Mapping):
        raise NormalizerApplicationError("{} must contain a JSON object: {}".format(context, path))
    return dict(value)


def _load_frozen_normalizer(
    bundle_dir: Path,
    *,
    supplied_source_manifest_sha256: str,
    supplied_sequence_manifest_sha256: str,
) -> FrozenNormalizer:
    """Load only a fit artifact bound to both frozen input manifests."""
    bundle_dir = bundle_dir.expanduser().resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError("normalizer bundle directory does not exist: {}".format(bundle_dir))
    report_path = bundle_dir / "report.json"
    normalizer_path = bundle_dir / "normalizer.pt"
    report = _load_json_object(report_path, context="normalizer report")
    if report.get("schema_version") != "atomicdance-motion-normalizer-fit-v1":
        raise NormalizerApplicationError(
            "normalizer report has unsupported schema_version {!r}".format(report.get("schema_version"))
        )
    if report.get("fit_split") != FIT_SPLIT:
        raise NormalizerApplicationError(
            "normalizer report fit_split must be {!r}, got {!r}".format(FIT_SPLIT, report.get("fit_split"))
        )
    representation = report.get("representation_contract")
    if not isinstance(representation, Mapping) or dict(representation) != RAW_REPRESENTATION_CONTRACT:
        raise NormalizerApplicationError(
            "normalizer report representation_contract is not the exact raw AtomicDance-151D contract"
        )
    report_input = report.get("input")
    if not isinstance(report_input, Mapping):
        raise NormalizerApplicationError("normalizer report lacks input provenance")
    reported_source_hash = _require_sha256(
        report_input.get("source_manifest_sha256"),
        context="normalizer report input.source_manifest_sha256",
    )
    if reported_source_hash != supplied_source_manifest_sha256:
        raise NormalizerApplicationError(
            "stale normalizer: report source_manifest_sha256 {} does not match supplied source manifest {}".format(
                reported_source_hash, supplied_source_manifest_sha256
            )
        )
    reported_sequence_hash = _require_sha256(
        report_input.get("sequence_manifest_sha256"),
        context="normalizer report input.sequence_manifest_sha256",
    )
    if reported_sequence_hash != supplied_sequence_manifest_sha256:
        raise NormalizerApplicationError(
            "stale normalizer: report sequence_manifest_sha256 {} does not match supplied sequence manifest {}".format(
                reported_sequence_hash, supplied_sequence_manifest_sha256
            )
        )
    normalizer_section = report.get("normalizer")
    if not isinstance(normalizer_section, Mapping):
        raise NormalizerApplicationError("normalizer report lacks normalizer provenance")
    declared_artifact_hash = _require_sha256(
        normalizer_section.get("sha256"), context="normalizer report normalizer.sha256"
    )
    if not normalizer_path.is_file():
        raise FileNotFoundError("normalizer artifact does not exist: {}".format(normalizer_path))
    actual_artifact_hash = _sha256_file(normalizer_path)
    if declared_artifact_hash != actual_artifact_hash:
        raise NormalizerApplicationError(
            "normalizer artifact hash does not match report: {} != {}".format(
                actual_artifact_hash, declared_artifact_hash
            )
        )
    try:
        payload = torch.load(str(normalizer_path), map_location="cpu", weights_only=True)
    except TypeError as error:  # pragma: no cover - old Torch must fail closed rather than unpickle
        raise NormalizerApplicationError(
            "this Torch version lacks safe weights_only normalizer loading; upgrade Torch"
        ) from error
    except Exception as error:
        raise NormalizerApplicationError(
            "cannot safely load normalizer artifact {}: {}".format(normalizer_path, error)
        ) from error
    if not isinstance(payload, Mapping) or set(payload) != {"data_min", "data_max"}:
        raise NormalizerApplicationError(
            "normalizer artifact must contain exactly data_min and data_max"
        )
    values: Dict[str, np.ndarray] = {}
    for key in ("data_min", "data_max"):
        value = payload[key]
        if not isinstance(value, torch.Tensor):
            raise NormalizerApplicationError("normalizer {} is not a tensor".format(key))
        if value.dtype != torch.float32 or tuple(value.shape) != (MOTION_DIM,):
            raise NormalizerApplicationError(
                "normalizer {} must be a float32 [{}] tensor, got {} {}".format(
                    key, MOTION_DIM, value.dtype, tuple(value.shape)
                )
            )
        array = np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
        if not np.isfinite(array).all():
            raise NormalizerApplicationError("normalizer {} contains non-finite values".format(key))
        values[key] = array.copy()
    if np.any(values["data_max"] < values["data_min"]):
        raise NormalizerApplicationError("normalizer data_max is smaller than data_min in at least one dimension")
    data_range = values["data_max"] - values["data_min"]
    if not np.isfinite(data_range).all():
        raise NormalizerApplicationError("normalizer data range is not finite")
    safe_range = np.where(data_range == np.float32(0.0), np.float32(1.0), data_range).astype(
        np.float32, copy=False
    )
    if not np.isfinite(safe_range).all() or np.any(safe_range <= np.float32(0.0)):
        raise NormalizerApplicationError("normalizer safe range is invalid")
    return FrozenNormalizer(
        bundle_dir=bundle_dir,
        normalizer_path=normalizer_path,
        report_path=report_path,
        artifact_sha256=actual_artifact_hash,
        report_sha256=_sha256_file(report_path),
        data_min=values["data_min"],
        data_max=values["data_max"],
        safe_range=safe_range.copy(),
        report=report,
    )


def _resolve_asset(value: str, *, manifest_parent: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_parent / candidate
    return candidate.resolve()


def _canonicalize_local_path(value: object, *, manifest_parent: Path) -> object:
    """Freeze one declared local artifact location into an absolute path.

    A normalizer output is a new bundle, so retaining ``assets/foo.npy``
    verbatim would silently make it mean ``<new-bundle>/assets/foo.npy``.
    Leave non-string values and URI-shaped references alone: this command does
    not fetch remote provenance or reinterpret a logical identifier as a
    local filename.
    """
    if not isinstance(value, str) or not value.strip() or "://" in value:
        return value
    return str(_resolve_asset(value, manifest_parent=manifest_parent))


def _canonicalize_retained_local_paths(
    row: Dict[str, Any], *, manifest_parent: Path, context: str
) -> None:
    """Canonicalize schema-defined retained provenance paths in place.

    The row is already a deep copy of an input row.  We intentionally update
    both flat and nested aliases so a downstream consumer cannot see two
    different interpretations of the same asset after the manifest moves.
    No existence check is performed for optional camera/source provenance;
    their original manifest may legitimately describe an unavailable
    inspection-only artifact.  Required model-input assets are checked by the
    callers that consume them.
    """
    for field in _TOP_LEVEL_LOCAL_PATH_FIELDS:
        if field in row:
            row[field] = _canonicalize_local_path(row[field], manifest_parent=manifest_parent)
    for object_field, path_fields in (
        ("assets", _ASSET_LOCAL_PATH_FIELDS),
        ("timeline", _TIMELINE_LOCAL_PATH_FIELDS),
    ):
        value = row.get(object_field)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise NormalizerApplicationError(
                "{} {} must be an object when present".format(context, object_field)
            )
        canonical = dict(value)
        for field in path_fields:
            if field in canonical:
                canonical[field] = _canonicalize_local_path(
                    canonical[field], manifest_parent=manifest_parent
                )
        row[object_field] = canonical


def _optional_declared_raw_hash(assets: Mapping[str, Any], *, context: str) -> Optional[str]:
    if "motion_151_raw_sha256" not in assets or assets["motion_151_raw_sha256"] is None:
        return None
    return _require_sha256(
        assets["motion_151_raw_sha256"], context="{} assets.motion_151_raw_sha256".format(context)
    )


def _declared_frame_count(row: Mapping[str, Any], *, context: str) -> Optional[int]:
    """Validate optional canonical frame counts without inventing one."""
    candidates: List[Tuple[str, Any]] = []
    if "frame_count" in row:
        candidates.append(("frame_count", row["frame_count"]))
    if "timeline" in row:
        timeline = row["timeline"]
        if not isinstance(timeline, Mapping):
            raise NormalizerApplicationError("{} timeline must be an object when present".format(context))
        if "frame_count" in timeline:
            candidates.append(("timeline.frame_count", timeline["frame_count"]))
    expected: Optional[int] = None
    for field, value in candidates:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise NormalizerApplicationError("{} {} must be a positive integer".format(context, field))
        if expected is not None and expected != value:
            raise NormalizerApplicationError(
                "{} has inconsistent declared frame counts {} and {}".format(context, expected, value)
            )
        expected = int(value)
    return expected


def _normalize_sequence(
    sequence: RawSequence,
    *,
    sequence_manifest_parent: Path,
    artifact_root: Path,
    public_artifact_root: Path,
    normalizer: FrozenNormalizer,
    block_frames: int,
) -> Tuple[Dict[str, Any], int, str, str]:
    """Stream one raw array into a separate normalized .npy artifact."""
    context = "sequence {!r}".format(sequence.sequence_id)
    assets = _require_mapping(sequence.row, "assets", context=context)
    raw_path = _resolve_asset(sequence.raw_motion_value, manifest_parent=sequence_manifest_parent)
    if not raw_path.is_file():
        raise NormalizerApplicationError("{} raw motion asset does not exist: {}".format(context, raw_path))
    try:
        raw_values = np.load(str(raw_path), mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise NormalizerApplicationError(
            "{} cannot memory-map raw motion {}: {}".format(context, raw_path, error)
        ) from error
    if raw_values.ndim != 2 or tuple(raw_values.shape[1:]) != (MOTION_DIM,):
        raise NormalizerApplicationError(
            "{} raw motion has shape {}; expected [T, {}]".format(
                context, tuple(raw_values.shape), MOTION_DIM
            )
        )
    frame_count = int(raw_values.shape[0])
    if frame_count < 1:
        raise NormalizerApplicationError("{} raw motion is empty".format(context))
    if not np.issubdtype(raw_values.dtype, np.floating):
        raise NormalizerApplicationError(
            "{} raw motion dtype {} is not floating".format(context, raw_values.dtype)
        )
    declared_frames = _declared_frame_count(sequence.row, context=context)
    if declared_frames is not None and declared_frames != frame_count:
        raise NormalizerApplicationError(
            "{} declared {} frames but raw motion has {}".format(context, declared_frames, frame_count)
        )
    declared_raw_hash = _optional_declared_raw_hash(assets, context=context)
    raw_hash = _sha256_file(raw_path)
    if declared_raw_hash is not None and declared_raw_hash != raw_hash:
        raise NormalizerApplicationError(
            "{} assets.motion_151_raw_sha256 does not match its raw motion artifact".format(context)
        )

    artifact_name = hashlib.sha256(sequence.sequence_id.encode("utf-8")).hexdigest() + ".npy"
    output_path = artifact_root / artifact_name
    if output_path.exists():  # pragma: no cover - cryptographic collision/race protection
        raise RuntimeError("refusing to overwrite normalized motion artifact {}".format(output_path))
    output_values = np.lib.format.open_memmap(
        str(output_path), mode="w+", dtype=np.float32, shape=(frame_count, MOTION_DIM)
    )
    try:
        float32_limit = np.finfo(np.float32).max
        for start in range(0, frame_count, block_frames):
            stop = min(start + block_frames, frame_count)
            raw_block = np.asarray(raw_values[start:stop])
            if not np.isfinite(raw_block).all():
                raise NormalizerApplicationError(
                    "{} raw motion contains non-finite values in frames [{}, {})".format(context, start, stop)
                )
            # The downstream AtomicDance normalizer is float32.  Refuse an
            # out-of-range source rather than cast it to infinity or repair it.
            if np.any(np.abs(raw_block) > float32_limit):
                raise NormalizerApplicationError(
                    "{} raw motion values exceed the float32 model-input range".format(context)
                )
            working = np.asarray(raw_block, dtype=np.float32)
            normalized = (
                np.float32(2.0) * (working - normalizer.data_min) / normalizer.safe_range
                - np.float32(1.0)
            )
            if not np.isfinite(normalized).all():
                raise NormalizerApplicationError(
                    "{} normalization produced non-finite values; no clipping or repair is permitted".format(
                        context
                    )
                )
            output_values[start:stop] = normalized
    finally:
        # Ensure all mmap writes are flushed before hashing/publishing, even
        # when a later block raises and the enclosing staging tree is removed.
        del output_values
        del raw_values
    normalized_hash = _sha256_file(output_path)
    public_path = (public_artifact_root / artifact_name).resolve()

    output_row: Dict[str, Any] = copy.deepcopy(dict(sequence.row))
    _canonicalize_retained_local_paths(
        output_row, manifest_parent=sequence_manifest_parent, context=context
    )
    output_assets = dict(_require_mapping(output_row, "assets", context=context))
    # ``motion_151_raw`` remains a retained provenance artifact, now anchored
    # to its original source manifest rather than accidentally rebased to the
    # normalized bundle.  The model-input aliases below deliberately point to
    # the new normalized tensor.
    output_assets["motion_151_raw"] = str(raw_path)
    output_assets["motion_151_raw_sha256"] = raw_hash
    output_assets["motion_151_normalized"] = str(public_path)
    output_assets["motion_151_normalized_sha256"] = normalized_hash
    output_assets["motion_151_model_input"] = str(public_path)
    output_assets["motion_151_model_input_sha256"] = normalized_hash
    output_row["assets"] = output_assets
    # Flat aliases describe the actual model input in a normalized manifest.
    # Leaving the raw aliases in place creates a contradictory manifest and
    # could make a materializer consume raw motion after a path-base change.
    output_row["motion_path"] = str(public_path)
    output_row["motion_sha256"] = normalized_hash

    output_representation = dict(_require_mapping(output_row, "representation", context=context))
    output_representation.update(
        {
            "motion": RAW_REPRESENTATION_CONTRACT["motion"],
            "coordinate_system": RAW_REPRESENTATION_CONTRACT["coordinate_system"],
            "normalization": "normalized",
            "normalization_state": "normalized",
            "camera_in_model_input": False,
            "normalizer_artifact_sha256": normalizer.artifact_sha256,
            "normalizer_fit_split": FIT_SPLIT,
            "normalization_artifact_sha256": normalizer.artifact_sha256,
            "normalization_fit_split": FIT_SPLIT,
        }
    )
    output_row["representation"] = output_representation
    # These redundant explicit fields let the window materializer validate a
    # derived manifest without guessing nested representation semantics.
    output_row["motion_representation_id"] = RAW_REPRESENTATION_CONTRACT["motion"]
    output_row["coordinate_system"] = RAW_REPRESENTATION_CONTRACT["coordinate_system"]
    # The mirror was missing exactly one field, and it is the one
    # ``materialize_atomic_windows`` reads at the top level rather than out of
    # ``representation`` (see its ``camera_not_decoupled`` quarantine).  So a
    # normalized wild manifest was rejected sequence by sequence, and the
    # materializer -- which refuses to publish a bundle with no train windows --
    # reported only that it had none.  Measured 2026-08-22: every one of the
    # 1,999 clean5b5 sequences, and the published ``wild_v4_normalized`` tree
    # too, which is why this bridge had never carried the wild corpus.
    output_row["camera_in_model_input"] = False
    # The rest of what ``materialize_atomic_windows._verify_normalizer_provenance``
    # reads off the top level.  The mirror carried the hashes but not the paths,
    # so the check that the artifact on disk *is* the artifact the manifest
    # claims could never run -- it failed on the field's absence instead.
    output_row["normalizer_artifact_path"] = str(normalizer.normalizer_path)
    output_row["normalizer_artifact_sha256"] = normalizer.artifact_sha256
    output_row["normalizer_fit_report_path"] = str(normalizer.report_path)
    output_row["normalizer_fit_report_sha256"] = normalizer.report_sha256
    output_row["normalizer_fit_split"] = FIT_SPLIT
    output_row["normalization_state"] = "normalized"
    output_row["normalization_artifact_sha256"] = normalizer.artifact_sha256
    output_row["normalization_fit_split"] = FIT_SPLIT
    output_row["normalization"] = {
        "schema_version": SCHEMA_VERSION,
        "state": "normalized",
        "formula": "2*(raw-data_min)/safe_range-1",
        "constant_dimension_safe_range": 1.0,
        "clipping": "none",
        "repair": "none",
        "normalizer_artifact": str(normalizer.normalizer_path),
        "normalizer_artifact_sha256": normalizer.artifact_sha256,
        "normalizer_fit_report": str(normalizer.report_path),
        "normalizer_fit_report_sha256": normalizer.report_sha256,
        "fit_split": FIT_SPLIT,
        "fit_source_manifest_sha256": normalizer.report["input"]["source_manifest_sha256"],
        "input_motion_151_raw": str(raw_path),
        "input_motion_151_raw_sha256": raw_hash,
        "output_motion_151_normalized": str(public_path),
        "output_motion_151_normalized_sha256": normalized_hash,
        "frame_count": frame_count,
        "motion_dim": MOTION_DIM,
        "camera_in_model_input": False,
    }
    return output_row, frame_count, raw_hash, normalized_hash


def apply_motion_normalizer(
    sequence_manifest: Path,
    source_manifest: Path,
    normalizer_bundle: Path,
    output_dir: Path,
    *,
    block_frames: int = DEFAULT_BLOCK_FRAMES,
) -> Dict[str, Any]:
    """Atomically publish normalized copies for all raw canonical sequences.

    The normalizer has already been fit on train-only data.  This function
    deliberately applies it to every declared split; it never calls ``fit``
    and never uses validation/test values to change the frozen parameters.
    """
    if isinstance(block_frames, bool) or not isinstance(block_frames, int) or block_frames < 1:
        raise NormalizerApplicationError("block_frames must be a positive integer")
    sequence_manifest = sequence_manifest.expanduser().resolve()
    source_manifest = source_manifest.expanduser().resolve()
    normalizer_bundle = normalizer_bundle.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            "refusing to overwrite existing normalized bundle {}; choose a new versioned output directory".format(
                output_dir
            )
        )
    if not source_manifest.is_file():
        raise FileNotFoundError("source manifest does not exist: {}".format(source_manifest))
    if not sequence_manifest.is_file():
        raise FileNotFoundError("sequence manifest does not exist: {}".format(sequence_manifest))

    source_hash = _sha256_file(source_manifest)
    sequence_hash = _sha256_file(sequence_manifest)
    # Check stale/re-split provenance before parsing or reading any target
    # motion.  Equivalent JSON with different bytes is still a new frozen
    # artifact and needs a new normalizer fit; this applies to both manifests.
    normalizer = _load_frozen_normalizer(
        normalizer_bundle,
        supplied_source_manifest_sha256=source_hash,
        supplied_sequence_manifest_sha256=sequence_hash,
    )
    sources = _parse_sources(_read_jsonl(source_manifest, kind="source"))
    sequences = _parse_raw_sequences(
        _read_jsonl(sequence_manifest, kind="sequence"), sources=sources
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".{}-staging-".format(output_dir.name), dir=str(output_dir.parent))
    )
    try:
        staging_motion_dir = staging / OUTPUT_MOTION_DIR
        staging_motion_dir.mkdir()
        output_rows: List[Dict[str, Any]] = []
        frame_count_by_split = {split: 0 for split in sorted(ALLOWED_SPLITS)}
        sequence_count_by_split = {split: 0 for split in sorted(ALLOWED_SPLITS)}
        raw_hashes: List[str] = []
        normalized_hashes: List[str] = []
        for sequence in sequences:
            output_row, frames, raw_hash, normalized_hash = _normalize_sequence(
                sequence,
                sequence_manifest_parent=sequence_manifest.parent,
                artifact_root=staging_motion_dir,
                public_artifact_root=output_dir / OUTPUT_MOTION_DIR,
                normalizer=normalizer,
                block_frames=block_frames,
            )
            output_rows.append(output_row)
            frame_count_by_split[sequence.split] += frames
            sequence_count_by_split[sequence.split] += 1
            raw_hashes.append(raw_hash)
            normalized_hashes.append(normalized_hash)
        _write_jsonl(staging / OUTPUT_SEQUENCE_MANIFEST, output_rows)
        output_manifest_hash = _sha256_file(staging / OUTPUT_SEQUENCE_MANIFEST)
        report: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "publication": "immutable_new_directory_only_atomic_rename",
            "input": {
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": source_hash,
                "sequence_manifest": str(sequence_manifest),
                "sequence_manifest_sha256": sequence_hash,
            },
            "normalizer": {
                "bundle": str(normalizer.bundle_dir),
                "fit_report": str(normalizer.report_path),
                "fit_report_sha256": normalizer.report_sha256,
                "artifact": str(normalizer.normalizer_path),
                "artifact_sha256": normalizer.artifact_sha256,
                "fit_split": FIT_SPLIT,
                "fit_source_manifest_sha256": source_hash,
                "raw_representation_contract": dict(RAW_REPRESENTATION_CONTRACT),
            },
            "output": {
                "sequence_manifest": str((output_dir / OUTPUT_SEQUENCE_MANIFEST).resolve()),
                "sequence_manifest_sha256": output_manifest_hash,
                "motion_directory": str((output_dir / OUTPUT_MOTION_DIR).resolve()),
                "layout": "motion_151_normalized/<sha256(sequence_id)>.npy",
            },
            "counts": {
                "sequences": len(output_rows),
                "sequences_by_split": sequence_count_by_split,
                "frames": int(sum(frame_count_by_split.values())),
                "frames_by_split": frame_count_by_split,
            },
            "transform": {
                "motion_dim": MOTION_DIM,
                "output_dtype": "float32",
                "feature_range": [-1.0, 1.0],
                "formula": "2*(raw-data_min)/safe_range-1",
                "safe_range": "1 where data_max == data_min, else data_max-data_min",
                "clipping": "none",
                "repair": "none",
                "camera_in_model_input": False,
                "constant_dimensions": int(np.sum(normalizer.data_max == normalizer.data_min)),
                "block_frames": block_frames,
            },
            "provenance": {
                "sequence_id_ordering": "lexicographic sequence_id",
                "sequence_ids_sha256": _sha256_strings([item.sequence_id for item in sequences]),
                "input_raw_motion_hashes_sha256": _sha256_strings(raw_hashes),
                "output_normalized_motion_hashes_sha256": _sha256_strings(normalized_hashes),
                "val_test_used_frozen_train_normalizer": True,
            },
        }
        with (staging / OUTPUT_REPORT).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if output_dir.exists():  # pragma: no cover - race protection
            raise FileExistsError("refusing to overwrite existing normalized bundle {}".format(output_dir))
        os.rename(str(staging), str(output_dir))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-manifest",
        required=True,
        help="full raw sequence JSONL with explicit source identity, raw 151D asset, and raw representation",
    )
    parser.add_argument(
        "--source-manifest",
        required=True,
        help="the exact frozen source JSONL used to fit the normalizer",
    )
    parser.add_argument(
        "--normalizer-bundle",
        required=True,
        help="immutable directory emitted by tools/fit_motion_normalizer.py",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new immutable normalized bundle directory; it must not already exist",
    )
    parser.add_argument(
        "--block-frames",
        type=int,
        default=DEFAULT_BLOCK_FRAMES,
        help="maximum motion frames normalized at once (default: %(default)s)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = apply_motion_normalizer(
            Path(args.sequence_manifest),
            Path(args.source_manifest),
            Path(args.normalizer_bundle),
            Path(args.output_dir),
            block_frames=args.block_frames,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit("error: {}".format(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
