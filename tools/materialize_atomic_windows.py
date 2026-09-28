#!/usr/bin/env python3
"""Safely materialize source-level labels into legacy AtomicDance arrays.

The released AtomicDance trainer consumes fixed indexed arrays, while the new
data contract keeps one motion/music/label timeline per source sequence.  This
tool is the *only* bridge between those formats.  It deliberately has a narrow
contract:

* source, sequence, and label identities are joined and checked before a
  single output file is published;
* a window is a direct ``[start, end)`` view of one continuous timeline;
* ``0`` remains a known transition only when its mask is true.  The current
  ``AtomicSequenceDataset``/D3PM path has no invalid-label loss support, so
  partially masked windows are quarantined rather than written with ``-1`` or
  silently rewritten as transition ``0``;
* all materialized sequences must use the exact normalized AtomicDance-151D,
  z-up/body-only, camera-decoupled model-input contract.  A single verified
  frozen train-only normalizer is copied into the resulting inference root;
  no normalizer is fitted or applied here.

The result is an immutable directory containing ``train``, ``val``, and
``test`` indexed arrays, canonical ``windows.jsonl``, a transparent
``quarantine.jsonl``, and a hash-rich ``build.json``.  Existing destinations
are never changed; staging is atomically renamed into place only after all
selected windows have been copied and verified bit-for-bit.
"""

from __future__ import annotations

import argparse
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


SCHEMA_VERSION = "atomic-window-materialization-v1"
MATERIALIZER_VERSION = "source-manifest-atomic-window-materializer-v1"
OUTPUT_SPLITS: Tuple[str, ...] = ("train", "val", "test")
INVALID_LABEL = -1
_MISSING = object()
MODEL_MOTION_REPRESENTATION = "AtomicDance_151D"
MODEL_COORDINATE_SYSTEM = "z_up_world_body_only"
MODEL_FPS = 30.0


class MaterializationError(ValueError):
    """An input contract violation that makes publication unsafe."""


class _QuarantineSequence(ValueError):
    """A sequence is well referenced but cannot safely produce windows."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _input_mapping(row: Mapping[str, Any], field: str, context: str) -> Mapping[str, Any]:
    """Read an optional nested input object without changing the source row."""
    value = row.get(field, _MISSING)
    if value is _MISSING or value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MaterializationError("{} has non-object {}".format(context, field))
    return value


def _alias_value(
    row: Mapping[str, Any],
    *,
    canonical_field: str,
    nested_value: object = _MISSING,
    nested_path: str,
    context: str,
    default: object = _MISSING,
) -> object:
    """Resolve flat-or-nested aliases while rejecting contradictory copies.

    The materializer accepts the documented flat source-manifest contract and
    the reconciler's nested wild contract, but it never guesses which of two
    disagreeing values is authoritative.  It returns a value for an internal
    copy only; callers never mutate the input manifest object.
    """
    flat_value = row.get(canonical_field, _MISSING)
    if flat_value is not _MISSING and nested_value is not _MISSING and flat_value != nested_value:
        raise MaterializationError(
            "{} has conflicting {}={!r} and {}={!r}".format(
                context, canonical_field, flat_value, nested_path, nested_value
            )
        )
    if flat_value is not _MISSING:
        return flat_value
    if nested_value is not _MISSING:
        return nested_value
    return default


def _coalesce_aliases(
    row: Mapping[str, Any],
    *,
    canonical_field: str,
    nested_values: Sequence[Tuple[str, object]],
    context: str,
    default: object = _MISSING,
) -> object:
    """Resolve several equivalent spellings, rejecting every disagreement."""
    value = row.get(canonical_field, _MISSING)
    origin = canonical_field if value is not _MISSING else None
    for nested_path, nested_value in nested_values:
        if nested_value is _MISSING:
            continue
        if value is not _MISSING and value != nested_value:
            raise MaterializationError(
                "{} has conflicting {}={!r} and {}={!r}".format(
                    context, origin, value, nested_path, nested_value
                )
            )
        value = nested_value
        origin = nested_path
    return default if value is _MISSING else value


def _canonicalize_source_row(raw_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a read-only-in-spirit flat view over a source manifest row.

    Wild staging/reconcile rows keep source hashes under
    ``assets.content_sha256``; the canonical source contract uses top-level
    ``content_sha256``.  Both spellings are accepted only when they agree.
    """
    result = dict(raw_row)
    context = _row_context("sources", result)
    assets = _input_mapping(raw_row, "assets", context)
    content = _alias_value(
        raw_row,
        canonical_field="content_sha256",
        nested_value=assets.get("content_sha256", _MISSING),
        nested_path="assets.content_sha256",
        context=context,
    )
    if content is not _MISSING:
        result["content_sha256"] = content
    return result


def _canonicalize_sequence_row(raw_row: Mapping[str, Any]) -> Dict[str, Any]:
    """Create a flat internal view over a canonical or nested wild sequence.

    Exact mappings are intentionally local and auditable:

    * ``timeline.{frame_count,source_start_frame,source_end_frame_exclusive,
      frame_ids_path,motion_frames_are_contiguous}`` map to the corresponding
      flat timeline keys (the final item maps to ``is_contiguous``);
    * raw ``assets.motion_151_raw`` or normalized
      ``assets.motion_151_model_input``/``motion_151_normalized`` maps to
      ``motion_path`` according to explicit normalization state; and
      ``assets.music_35`` maps to ``music_path``;
    * ``representation.{motion,coordinate_system,normalization,...}`` maps to
      the explicit motion-representation/normalization contract.

    Missing raw-normalization provenance is normalized internally to explicit
    null fields; missing/contradictory non-raw provenance remains an error.
    """
    result = dict(raw_row)
    context = _row_context("sequences", result)
    timeline = _input_mapping(raw_row, "timeline", context)
    assets = _input_mapping(raw_row, "assets", context)
    representation = _input_mapping(raw_row, "representation", context)
    normalization = _input_mapping(raw_row, "normalization", context)
    timeline_aliases = (
        ("fps", timeline.get("fps", _MISSING), "timeline.fps"),
        ("frame_count", timeline.get("frame_count", _MISSING), "timeline.frame_count"),
        ("source_start_frame", timeline.get("source_start_frame", _MISSING), "timeline.source_start_frame"),
        (
            "source_end_frame_exclusive",
            timeline.get("source_end_frame_exclusive", _MISSING),
            "timeline.source_end_frame_exclusive",
        ),
        ("frame_ids_path", timeline.get("frame_ids_path", _MISSING), "timeline.frame_ids_path"),
        ("is_contiguous", timeline.get("motion_frames_are_contiguous", _MISSING), "timeline.motion_frames_are_contiguous"),
        ("music_path", assets.get("music_35", _MISSING), "assets.music_35"),
        ("music_sha256", assets.get("music_35_sha256", _MISSING), "assets.music_35_sha256"),
        ("motion_representation_id", representation.get("motion", _MISSING), "representation.motion"),
        ("coordinate_system", representation.get("coordinate_system", _MISSING), "representation.coordinate_system"),
        ("normalization_state", representation.get("normalization", _MISSING), "representation.normalization"),
        ("camera_in_model_input", representation.get("camera_in_model_input", _MISSING), "representation.camera_in_model_input"),
    )
    for canonical_field, nested_value, nested_path in timeline_aliases:
        value = _alias_value(
            raw_row,
            canonical_field=canonical_field,
            nested_value=nested_value,
            nested_path=nested_path,
            context=context,
        )
        if value is not _MISSING:
            result[canonical_field] = value
    # The apply-normalizer output contains both a compact representation
    # summary and a richer top-level normalization provenance object.  All
    # aliases must agree, including the source artifact hash used by inference.
    state = _coalesce_aliases(
        raw_row,
        canonical_field="normalization_state",
        nested_values=(
            ("representation.normalization", representation.get("normalization", _MISSING)),
            ("representation.normalization_state", representation.get("normalization_state", _MISSING)),
            ("normalization.state", normalization.get("state", _MISSING)),
        ),
        context=context,
    )
    if state is not _MISSING:
        result["normalization_state"] = state
    normalized = state == "normalized"
    if normalized:
        normalized_motion = _coalesce_aliases(
            {},
            canonical_field="motion_path",
            nested_values=(
                ("assets.motion_151_model_input", assets.get("motion_151_model_input", _MISSING)),
                ("assets.motion_151_normalized", assets.get("motion_151_normalized", _MISSING)),
            ),
            context=context,
        )
        normalized_motion_hash = _coalesce_aliases(
            {},
            canonical_field="motion_sha256",
            nested_values=(
                ("assets.motion_151_model_input_sha256", assets.get("motion_151_model_input_sha256", _MISSING)),
                ("assets.motion_151_normalized_sha256", assets.get("motion_151_normalized_sha256", _MISSING)),
            ),
            context=context,
        )
        # A flat canonical ``motion_path``/``motion_sha256`` describes model
        # input; it must agree with nested normalized assets when both exist.
        for field, nested_value, nested_path in (
            ("motion_path", normalized_motion, "normalized assets"),
            ("motion_sha256", normalized_motion_hash, "normalized asset hashes"),
        ):
            value = _alias_value(
                raw_row,
                canonical_field=field,
                nested_value=nested_value,
                nested_path=nested_path,
                context=context,
            )
            if value is not _MISSING:
                result[field] = value
    else:
        for field, nested_value, nested_path in (
            ("motion_path", assets.get("motion_151_raw", _MISSING), "assets.motion_151_raw"),
            ("motion_sha256", assets.get("motion_151_raw_sha256", _MISSING), "assets.motion_151_raw_sha256"),
        ):
            value = _alias_value(
                raw_row,
                canonical_field=field,
                nested_value=nested_value,
                nested_path=nested_path,
                context=context,
            )
            if value is not _MISSING:
                result[field] = value
    artifact_hash = _coalesce_aliases(
        raw_row,
        canonical_field="normalization_artifact_sha256",
        nested_values=(
            ("representation.normalization_artifact_sha256", representation.get("normalization_artifact_sha256", _MISSING)),
            ("representation.normalizer_artifact_sha256", representation.get("normalizer_artifact_sha256", _MISSING)),
            ("normalization.normalizer_artifact_sha256", normalization.get("normalizer_artifact_sha256", _MISSING)),
        ),
        context=context,
    )
    fit_split = _coalesce_aliases(
        raw_row,
        canonical_field="normalization_fit_split",
        nested_values=(
            ("representation.normalization_fit_split", representation.get("normalization_fit_split", _MISSING)),
            ("representation.normalizer_fit_split", representation.get("normalizer_fit_split", _MISSING)),
            ("normalization.fit_split", normalization.get("fit_split", _MISSING)),
        ),
        context=context,
    )
    if artifact_hash is not _MISSING:
        result["normalization_artifact_sha256"] = artifact_hash
    if fit_split is not _MISSING:
        result["normalization_fit_split"] = fit_split
    artifact_path = _coalesce_aliases(
        raw_row,
        canonical_field="normalizer_artifact_path",
        nested_values=(
            ("normalizer_artifact", raw_row.get("normalizer_artifact", _MISSING)),
            ("normalization.normalizer_artifact", normalization.get("normalizer_artifact", _MISSING)),
        ),
        context=context,
    )
    if artifact_path is not _MISSING:
        result["normalizer_artifact_path"] = artifact_path
    report_path = _coalesce_aliases(
        raw_row,
        canonical_field="normalizer_fit_report_path",
        nested_values=(
            ("normalization.normalizer_fit_report", normalization.get("normalizer_fit_report", _MISSING)),
        ),
        context=context,
    )
    if report_path is not _MISSING:
        result["normalizer_fit_report_path"] = report_path
    report_hash = _coalesce_aliases(
        raw_row,
        canonical_field="normalizer_fit_report_sha256",
        nested_values=(
            ("normalization.normalizer_fit_report_sha256", normalization.get("normalizer_fit_report_sha256", _MISSING)),
        ),
        context=context,
    )
    if report_hash is not _MISSING:
        result["normalizer_fit_report_sha256"] = report_hash
    # Reconcile's raw representation intentionally has no normalizer artifact
    # because it has not been transformed.  Make that absence explicit only in
    # this isolated derived view; the manifest itself stays byte-for-byte
    # untouched.  A normalized row must state both fields and is checked later.
    if result.get("normalization_state") == "raw":
        result.setdefault("normalization_artifact_sha256", None)
        result.setdefault("normalization_fit_split", None)
    return result


@dataclass(frozen=True)
class _CandidateWindow:
    split: str
    window_id: str
    sample_name: str
    sequence_id: str
    recording_id: str
    retrieval_group_id: str
    duplicate_content_group_id: Optional[str]
    start_frame: int
    end_frame_exclusive: int
    array_index: int
    motion_path: Path
    music_path: Path
    labels_path: Path
    label_valid_mask_path: Path
    source_content_sha256: str
    source_motion_sha256: str
    source_music_sha256: str
    label_labels_sha256: str
    label_valid_mask_sha256: str
    input_motion_sha256: str
    label_space_id: str
    producer_version: str
    fit_split: str
    fit_source_manifest_sha256: str
    producer_artifact_sha256: str
    motion_representation_id: str
    normalization_state: str
    normalization_artifact_sha256: Optional[str]
    normalization_fit_split: Optional[str]
    coordinate_system: str


@dataclass(frozen=True)
class _NormalizerProvenance:
    """One frozen fit artifact that may back an indexed model-input bundle."""

    artifact_path: Path
    artifact_sha256: str
    fit_report_path: Path
    fit_report_sha256: str
    fit_source_manifest_sha256: str
    fit_split: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_sha256(row: Mapping[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not _is_sha256(value):
        raise MaterializationError("{} requires {} to be a lowercase-or-uppercase SHA-256 hex string".format(context, field))
    return str(value).lower()


# The source contract asks for "content_sha256, a raw video hash, an audio hash
# or a comparable perceptual hash -- at least one", so that duplicate uploads,
# re-encodes and crops cannot leak across splits.  This tool used to demand the
# first spelling and nothing else, which is stricter than the contract and
# blocked a corpus that satisfies it: 371 of AIST++'s 1,363 recordings come from
# the official-supplement path and carry motion and music hashes but no
# content_sha256.
#
# ``music_sha256`` is deliberately NOT in this list.  It is a *song* identity,
# not a recording identity -- 1,363 rows carry only 158 distinct values, one of
# them shared by 23 recordings, because the supplement copied each music array
# from a same-song donor rather than re-extracting it.  Accepting it would make
# every second dance to the same song look like duplicated content, which is a
# gate that fires on the corpus instead of on a leak.
#
# ``motion_sha256`` is: 1,363 present, 1,363 distinct, and it exists on both
# ingestion paths.
IDENTITY_HASH_FIELDS = ("content_sha256", "motion_sha256")


def _require_identity_hash(row: Mapping[str, Any], context: str) -> Tuple[str, str]:
    """The per-recording identity hash, and which field supplied it.

    Returns the field name too, because the two ingestion paths hash different
    objects -- the rebase path hashes an object inside the upstream release
    package, the supplement path hashes the official motion pickle -- so two
    sources agreeing is only meaningful when they agreed *in the same field*.
    The caller records this rather than flattening it into one value.
    """
    for field in IDENTITY_HASH_FIELDS:
        if _is_sha256(row.get(field)):
            return str(row[field]).lower(), field
    raise MaterializationError(
        "{} requires one of {} to be a SHA-256 hex string; music_sha256 is not "
        "accepted because it identifies the song, not the recording".format(
            context, " or ".join(IDENTITY_HASH_FIELDS)))


def _require_string(row: Mapping[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise MaterializationError("{} requires a non-empty string {}".format(context, field))
    return value


def _require_int(row: Mapping[str, Any], field: str, context: str, *, minimum: Optional[int] = None) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MaterializationError("{} requires integer {}".format(context, field))
    if minimum is not None and value < minimum:
        raise MaterializationError("{} requires {} >= {}".format(context, field, minimum))
    return int(value)


def _require_atomicdance_fps(row: Mapping[str, Any], field: str, context: str) -> float:
    """Require the frame rate encoded by the fixed 150-frame model contract."""
    value = row.get(field, _MISSING)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterializationError("{} requires explicit numeric {}".format(context, field))
    fps = float(value)
    if not math.isfinite(fps) or fps != MODEL_FPS:
        raise MaterializationError(
            "{} {} must be exactly {} for AtomicDance 150-frame timing, got {!r}".format(
                context, field, int(MODEL_FPS), value
            )
        )
    return fps


def _require_optional_group(row: Mapping[str, Any], field: str, context: str) -> Optional[str]:
    if field not in row:
        raise MaterializationError("{} requires explicit {} (string or null)".format(context, field))
    value = row[field]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise MaterializationError("{} requires {} to be a non-empty string or null".format(context, field))
    return value


def _require_split(row: Mapping[str, Any], context: str) -> str:
    value = _require_string(row, "split", context)
    if value not in OUTPUT_SPLITS:
        raise MaterializationError("{} has unsupported split {!r}; expected one of {}".format(context, value, ", ".join(OUTPUT_SPLITS)))
    return value


def _read_jsonl(path: Path, table: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise MaterializationError("{} manifest does not exist: {}".format(table, path))
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as error:
                    raise MaterializationError(
                        "{} manifest {} line {} is not JSON: {}".format(table, path, line_number, error)
                    ) from error
                if not isinstance(parsed, dict):
                    raise MaterializationError(
                        "{} manifest {} line {} must be a JSON object".format(table, path, line_number)
                    )
                parsed["__line_number__"] = line_number
                rows.append(parsed)
    except OSError as error:
        raise MaterializationError("cannot read {} manifest {}: {}".format(table, path, error)) from error
    return rows


def _row_context(table: str, row: Mapping[str, Any]) -> str:
    return "{} row {}".format(table, row.get("__line_number__", "?"))


def _index_unique(rows: Iterable[Dict[str, Any]], field: str, table: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        context = _row_context(table, row)
        identifier = _require_string(row, field, context)
        if identifier in result:
            raise MaterializationError("{} has duplicate {} {!r}".format(table, field, identifier))
        result[identifier] = row
    return result


def _validate_source_groups(sources: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    """Reject all known recording/retrieval/content duplicate split leakage.

    Returns how many sources each identity field covered, so the release
    records that this corpus is identified through two different hashes rather
    than leaving a reader to assume one.
    """
    group_splits: Dict[Tuple[str, str], str] = {}
    identity_fields_used: Dict[str, int] = {}
    for recording_id, source in sources.items():
        context = _row_context("sources", source)
        split = _require_split(source, context)
        _require_atomicdance_fps(source, "fps", context)
        retrieval_group_id = _require_string(source, "retrieval_group_id", context)
        duplicate_group_id = _require_optional_group(source, "duplicate_content_group_id", context)
        content_sha256, identity_field = _require_identity_hash(source, context)
        identity_fields_used[identity_field] = identity_fields_used.get(identity_field, 0) + 1
        # recording_id is unique by indexing, but retain it in this map to
        # make the no-cross-split policy explicit and auditable.
        #
        # The identity group is keyed by the field as well as the value: the
        # two ingestion paths hash different objects, so a collision across
        # them would be a coincidence rather than duplicated content, and
        # keying on the value alone would let one become a cross-split verdict.
        groups = [
            ("recording_id", recording_id),
            ("retrieval_group_id", retrieval_group_id),
            (identity_field, content_sha256),
        ]
        if duplicate_group_id is not None:
            groups.append(("duplicate_content_group_id", duplicate_group_id))
        for kind, group_id in groups:
            key = (kind, group_id)
            prior = group_splits.get(key)
            if prior is not None and prior != split:
                raise MaterializationError(
                    "cross-split leakage: {} {!r} appears in both {!r} and {!r}".format(
                        kind, group_id, prior, split
                    )
                )
            group_splits[key] = split
    return identity_fields_used


def _validate_sequence_foreign_keys(
    sources: Mapping[str, Mapping[str, Any]], sequences: Mapping[str, Mapping[str, Any]]
) -> None:
    for sequence_id, sequence in sequences.items():
        context = _row_context("sequences", sequence)
        recording_id = _require_string(sequence, "recording_id", context)
        source = sources.get(recording_id)
        if source is None:
            raise MaterializationError(
                "{} sequence_id {!r} references unknown recording_id {!r}".format(context, sequence_id, recording_id)
            )
        source_context = _row_context("sources", source)
        source_fps = _require_atomicdance_fps(source, "fps", source_context)
        sequence_fps = _require_atomicdance_fps(sequence, "fps", context)
        if sequence_fps != source_fps:
            raise MaterializationError(
                "{} sequence_id {!r} fps {} disagrees with source {!r} fps {}".format(
                    context, sequence_id, sequence_fps, recording_id, source_fps
                )
            )
        for field in ("retrieval_group_id", "duplicate_content_group_id", "split"):
            if field == "duplicate_content_group_id":
                sequence_value = _require_optional_group(sequence, field, context)
                source_value = _require_optional_group(source, field, source_context)
            elif field == "split":
                sequence_value = _require_split(sequence, context)
                source_value = _require_split(source, source_context)
            else:
                sequence_value = _require_string(sequence, field, context)
                source_value = _require_string(source, field, source_context)
            if sequence_value != source_value:
                raise MaterializationError(
                    "{} sequence_id {!r} {} {!r} disagrees with source {!r}".format(
                        context, sequence_id, field, sequence_value, source_value
                    )
                )


def _validate_label_foreign_keys(
    sequences: Mapping[str, Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]]
) -> None:
    for sequence_id, label in labels.items():
        context = _row_context("labels", label)
        sequence = sequences.get(sequence_id)
        if sequence is None:
            raise MaterializationError(
                "{} references unknown sequence_id {!r}".format(context, sequence_id)
            )
        sequence_context = _row_context("sequences", sequence)
        for field in ("recording_id", "retrieval_group_id", "duplicate_content_group_id", "split"):
            if field == "duplicate_content_group_id":
                label_value = _require_optional_group(label, field, context)
                sequence_value = _require_optional_group(sequence, field, sequence_context)
            elif field == "split":
                label_value = _require_split(label, context)
                sequence_value = _require_split(sequence, sequence_context)
            else:
                label_value = _require_string(label, field, context)
                sequence_value = _require_string(sequence, field, sequence_context)
            if label_value != sequence_value:
                raise MaterializationError(
                    "{} sequence_id {!r} {} {!r} disagrees with sequence {!r}".format(
                        context, sequence_id, field, label_value, sequence_value
                    )
                )


def _resolve_asset(manifest_path: Path, raw_path: object, field: str, context: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise _QuarantineSequence("missing_asset_path", "{} requires non-empty {}".format(context, field))
    candidate = Path(raw_path)
    if candidate.is_absolute():
        # HMR reconciliation manifests intentionally carry absolute, immutable
        # artifact paths.  Accept those explicitly rather than forcing callers
        # to rewrite a frozen provenance record.  Only *relative* paths are
        # constrained to their manifest root, so ``../../`` cannot escape a
        # bundle by surprise.
        resolved = candidate.resolve()
    else:
        root = manifest_path.parent.resolve()
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise _QuarantineSequence(
                "unsafe_asset_path", "{} {} escapes its manifest directory".format(context, field)
            ) from error
    if not resolved.is_file():
        raise _QuarantineSequence("missing_asset", "{} asset does not exist: {}".format(context, resolved))
    return resolved


def _load_npy(path: Path, label: str) -> np.ndarray:
    try:
        return np.load(str(path), mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise _QuarantineSequence("unreadable_array", "cannot open {}: {}".format(label, error)) from error


def _load_contiguous_sequence(
    sequence: Mapping[str, Any],
    *,
    sequences_manifest: Path,
    motion_dim: int,
    music_dim: int,
) -> Tuple[Path, Path, str, str, Dict[str, Any]]:
    """Validate a source timeline and return its safe assets + hashes."""
    context = _row_context("sequences", sequence)
    sequence_id = _require_string(sequence, "sequence_id", context)
    _require_atomicdance_fps(sequence, "fps", context)
    frame_count = _require_int(sequence, "frame_count", context, minimum=1)
    if sequence.get("is_contiguous") is not True:
        raise _QuarantineSequence("non_contiguous_sequence", "{} sequence {!r} is not contiguous".format(context, sequence_id))
    source_start = _require_int(sequence, "source_start_frame", context, minimum=0)
    source_end = _require_int(sequence, "source_end_frame_exclusive", context, minimum=source_start + 1)
    if source_end - source_start != frame_count:
        raise _QuarantineSequence(
            "frame_count_mismatch",
            "{} sequence {!r} frame_count does not equal source range".format(context, sequence_id),
        )
    motion_path = _resolve_asset(sequences_manifest, sequence.get("motion_path"), "motion_path", context)
    music_path = _resolve_asset(sequences_manifest, sequence.get("music_path"), "music_path", context)
    frame_ids_path = _resolve_asset(sequences_manifest, sequence.get("frame_ids_path"), "frame_ids_path", context)
    motion = _load_npy(motion_path, "motion")
    music = _load_npy(music_path, "music")
    frame_ids = _load_npy(frame_ids_path, "frame_ids")
    if motion.dtype != np.float32 or tuple(motion.shape) != (frame_count, motion_dim):
        raise _QuarantineSequence(
            "motion_shape_or_dtype",
            "{} motion must be float32 with shape [{}, {}], got {} {}".format(
                context, frame_count, motion_dim, motion.dtype, tuple(motion.shape)
            ),
        )
    if music.dtype != np.float32 or tuple(music.shape) != (frame_count, music_dim):
        raise _QuarantineSequence(
            "music_shape_or_dtype",
            "{} music must be float32 with shape [{}, {}], got {} {}".format(
                context, frame_count, music_dim, music.dtype, tuple(music.shape)
            ),
        )
    if frame_ids.ndim != 1 or len(frame_ids) != frame_count or not np.issubdtype(frame_ids.dtype, np.integer):
        raise _QuarantineSequence("frame_ids_shape", "{} frame_ids must be an integer vector of length {}".format(context, frame_count))
    expected_frame_ids = np.arange(source_start, source_end, dtype=frame_ids.dtype)
    if not np.array_equal(frame_ids, expected_frame_ids):
        raise _QuarantineSequence(
            "non_contiguous_frame_ids", "{} frame_ids are not the exact continuous source range".format(context)
        )
    if not bool(np.isfinite(motion).all()) or not bool(np.isfinite(music).all()):
        raise _QuarantineSequence("nonfinite_motion_or_music", "{} contains non-finite motion/music".format(context))
    motion_sha256 = _sha256_file(motion_path)
    music_sha256 = _sha256_file(music_path)
    for field, actual in (("motion_sha256", motion_sha256), ("music_sha256", music_sha256)):
        declared = sequence.get(field)
        if declared is not None:
            if not _is_sha256(declared) or str(declared).lower() != actual:
                raise _QuarantineSequence(
                    "artifact_hash_mismatch", "{} {} does not match its asset".format(context, field)
                )
    representation = _representation_contract(sequence)
    return motion_path, music_path, motion_sha256, music_sha256, representation


def _representation_contract(sequence: Mapping[str, Any]) -> Dict[str, Any]:
    """Require explicit, auditable representation semantics; never infer them."""
    context = _row_context("sequences", sequence)
    representation_id = _require_string(sequence, "motion_representation_id", context)
    coordinate_system = _require_string(sequence, "coordinate_system", context)
    if representation_id != MODEL_MOTION_REPRESENTATION:
        raise MaterializationError(
            "{} motion_representation_id must be {!r}, got {!r}".format(
                context, MODEL_MOTION_REPRESENTATION, representation_id
            )
        )
    if coordinate_system != MODEL_COORDINATE_SYSTEM:
        raise MaterializationError(
            "{} coordinate_system must be {!r}, got {!r}".format(
                context, MODEL_COORDINATE_SYSTEM, coordinate_system
            )
        )
    normalization_state = _require_string(sequence, "normalization_state", context)
    if normalization_state not in {"raw", "normalized"}:
        raise _QuarantineSequence(
            "unknown_normalization_state",
            "{} normalization_state must be 'raw' or 'normalized'".format(context),
        )
    if "normalization_artifact_sha256" not in sequence or "normalization_fit_split" not in sequence:
        raise _QuarantineSequence(
            "missing_normalization_contract",
            "{} requires normalization_artifact_sha256 and normalization_fit_split".format(context),
        )
    artifact = sequence["normalization_artifact_sha256"]
    fit_split = sequence["normalization_fit_split"]
    if normalization_state == "raw":
        if artifact is not None or fit_split is not None:
            raise _QuarantineSequence(
                "invalid_raw_normalization_contract",
                "{} raw motion must have null normalization artifact and fit split".format(context),
            )
        # A raw sequence is a valid input to apply_motion_normalizer, but it
        # is never a valid indexed model root: the released AtomicDance
        # trainer/inference path consumes already-normalized tensors.
        raise MaterializationError(
            "{} is raw; apply the frozen train-only normalizer before indexed materialization".format(context)
        )
    else:
        if not _is_sha256(artifact) or fit_split != "train":
            raise _QuarantineSequence(
                "invalid_normalized_contract",
                "{} normalized motion requires a SHA-256 artifact fitted on train".format(context),
            )
        artifact_value = str(artifact).lower()
        fit_value = "train"
    camera_in_model_input = sequence.get("camera_in_model_input", _MISSING)
    if camera_in_model_input is not False:
        raise _QuarantineSequence(
            "camera_not_decoupled",
            "{} must explicitly declare camera_in_model_input=false".format(context),
        )
    return {
        "motion_representation_id": representation_id,
        "coordinate_system": coordinate_system,
        "normalization_state": normalization_state,
        "normalization_artifact_sha256": artifact_value,
        "normalization_fit_split": fit_value,
        "camera_in_model_input": False,
    }


def _load_accepted_labels(
    label: Mapping[str, Any],
    sequence: Mapping[str, Any],
    representation: Mapping[str, Any],
    *,
    labels_manifest: Path,
    expected_motion_sha256: str,
    expected_fit_source_manifest_sha256: str,
    num_classes: int,
) -> Tuple[Path, Path, str, str, str, str, str, str, str]:
    """Validate a full-timeline accepted label artifact without coercion."""
    context = _row_context("labels", label)
    if label.get("status") != "accepted":
        raise _QuarantineSequence("label_status_not_accepted", "{} status is not accepted".format(context))
    fit_split = _require_string(label, "fit_split", context)
    if fit_split != "train":
        raise _QuarantineSequence("label_not_train_only", "{} fit_split must be train".format(context))
    fit_source_manifest_sha256 = _require_sha256(label, "fit_source_manifest_sha256", context)
    if fit_source_manifest_sha256 != expected_fit_source_manifest_sha256:
        raise _QuarantineSequence(
            "label_fit_source_manifest_mismatch",
            "{} fit_source_manifest_sha256 does not bind to the supplied sources manifest".format(context),
        )
    producer_artifact_sha256 = _require_sha256(label, "producer_artifact_sha256", context)
    input_motion_sha256 = _require_sha256(label, "input_motion_sha256", context)
    if input_motion_sha256 != expected_motion_sha256:
        raise _QuarantineSequence(
            "label_motion_hash_mismatch", "{} was not produced from this exact motion asset".format(context)
        )
    for field, expected in (
        ("input_motion_representation_id", representation["motion_representation_id"]),
        ("input_normalization_state", representation["normalization_state"]),
        ("input_coordinate_system", representation["coordinate_system"]),
        ("input_normalization_artifact_sha256", representation["normalization_artifact_sha256"]),
    ):
        if field not in label or label[field] != expected:
            raise _QuarantineSequence(
                "label_representation_mismatch",
                "{} {} does not match the sequence representation contract".format(context, field),
            )
    label_space_id = _require_string(label, "label_space_id", context)
    producer_version = _require_string(label, "producer_version", context)
    labels_path = _resolve_asset(labels_manifest, label.get("labels_path"), "labels_path", context)
    valid_mask_path = _resolve_asset(
        labels_manifest, label.get("label_valid_mask_path"), "label_valid_mask_path", context
    )
    labels_array = _load_npy(labels_path, "labels")
    mask_array = _load_npy(valid_mask_path, "label_valid_mask")
    frame_count = _require_int(sequence, "frame_count", _row_context("sequences", sequence), minimum=1)
    if labels_array.ndim != 1 or len(labels_array) != frame_count or not np.issubdtype(labels_array.dtype, np.signedinteger):
        raise _QuarantineSequence(
            "labels_shape_or_dtype", "{} labels must be signed integer vector of length {}".format(context, frame_count)
        )
    if mask_array.dtype != np.bool_ or mask_array.ndim != 1 or len(mask_array) != frame_count:
        raise _QuarantineSequence(
            "label_valid_mask_shape_or_dtype", "{} label_valid_mask must be bool vector of length {}".format(context, frame_count)
        )
    valid = np.asarray(mask_array, dtype=bool)
    labels_values = np.asarray(labels_array)
    if np.any((labels_values[valid] < 0) | (labels_values[valid] >= num_classes)):
        raise _QuarantineSequence(
            "valid_label_out_of_range", "{} valid labels must be in [0, {})".format(context, num_classes)
        )
    # The sentinel is intentionally strict.  If an upstream producer used 0
    # for unknown, it is ambiguous and must be repaired at the producer, not
    # guessed here.
    if np.any(labels_values[~valid] != INVALID_LABEL):
        raise _QuarantineSequence(
            "invalid_label_not_sentinel",
            "{} invalid-mask labels must remain sentinel {} (never transition 0)".format(context, INVALID_LABEL),
        )
    labels_sha256 = _sha256_file(labels_path)
    mask_sha256 = _sha256_file(valid_mask_path)
    for field, actual in (("labels_sha256", labels_sha256), ("label_valid_mask_sha256", mask_sha256)):
        declared = label.get(field)
        if declared is not None and (not _is_sha256(declared) or str(declared).lower() != actual):
            raise _QuarantineSequence("artifact_hash_mismatch", "{} {} does not match its asset".format(context, field))
    return (
        labels_path,
        valid_mask_path,
        labels_sha256,
        mask_sha256,
        label_space_id,
        producer_version,
        fit_split,
        fit_source_manifest_sha256,
        producer_artifact_sha256,
    )


def _resolve_provenance_file(manifest_path: Path, raw_path: object, field: str, context: str) -> Path:
    """Resolve an immutable provenance asset without allowing relative escape."""
    if not isinstance(raw_path, str) or not raw_path:
        raise MaterializationError("{} requires non-empty {}".format(context, field))
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        root = manifest_path.parent.resolve()
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise MaterializationError("{} {} escapes its manifest directory".format(context, field)) from error
    if not resolved.is_file():
        raise MaterializationError("{} {} does not exist: {}".format(context, field, resolved))
    return resolved


def _read_json_object(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializationError("cannot read {} {}: {}".format(context, path, error)) from error
    if not isinstance(value, Mapping):
        raise MaterializationError("{} must be a JSON object: {}".format(context, path))
    return value


def _verify_normalizer_payload(path: Path, context: str) -> None:
    """Ensure the copied inference normalizer is a safe 151-D min/max object."""
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as error:  # pragma: no cover - old Torch must fail closed
        raise MaterializationError("{} requires Torch weights_only support".format(context)) from error
    except Exception as error:
        raise MaterializationError("cannot safely load {} {}: {}".format(context, path, error)) from error
    if not isinstance(payload, Mapping) or set(payload) != {"data_min", "data_max"}:
        raise MaterializationError("{} must contain exactly data_min and data_max".format(context))
    values: Dict[str, np.ndarray] = {}
    for name in ("data_min", "data_max"):
        value = payload[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or tuple(value.shape) != (151,):
            raise MaterializationError("{} {} must be a float32 [151] tensor".format(context, name))
        array = np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
        if not np.isfinite(array).all():
            raise MaterializationError("{} {} contains non-finite values".format(context, name))
        values[name] = array
    if np.any(values["data_max"] < values["data_min"]):
        raise MaterializationError("{} data_max is below data_min".format(context))


def _verify_normalizer_provenance(
    sequence: Mapping[str, Any],
    representation: Mapping[str, Any],
    *,
    sequences_manifest: Path,
    expected_source_manifest_sha256: str,
) -> _NormalizerProvenance:
    """Bind an applied normalized sequence to one frozen train-only artifact.

    The application manifest stores a normalizer artifact path/hash inside its
    top-level ``normalization`` object.  A fit report is derived only from the
    artifact's immutable bundle sibling ``report.json`` (or may be declared
    explicitly if it resolves to that exact same file).  This avoids accepting
    a correct-looking hash paired with an unrelated provenance report.
    """
    context = _row_context("sequences", sequence)
    if representation.get("normalization_state") != "normalized":
        raise MaterializationError(
            "{} is raw or has unknown normalization; apply the frozen normalizer before indexed materialization".format(
                context
            )
        )
    if representation.get("normalization_fit_split") != "train":
        raise MaterializationError("{} normalized motion must declare normalization_fit_split='train'".format(context))
    declared_artifact_hash = representation.get("normalization_artifact_sha256")
    if not _is_sha256(declared_artifact_hash):
        raise MaterializationError("{} normalized motion lacks a valid normalizer artifact SHA-256".format(context))
    if representation.get("camera_in_model_input") is not False:
        raise MaterializationError(
            "{} must explicitly set camera_in_model_input=false for model-input motion".format(context)
        )
    artifact_path = _resolve_provenance_file(
        sequences_manifest, sequence.get("normalizer_artifact_path"), "normalizer_artifact_path", context
    )
    actual_artifact_hash = _sha256_file(artifact_path)
    if actual_artifact_hash != str(declared_artifact_hash).lower():
        raise MaterializationError(
            "{} normalizer artifact hash does not match normalization provenance".format(context)
        )
    _verify_normalizer_payload(artifact_path, "normalizer artifact")
    expected_report_path = (artifact_path.parent / "report.json").resolve()
    explicit_report_value = sequence.get("normalizer_fit_report_path", _MISSING)
    if explicit_report_value is _MISSING:
        report_path = expected_report_path
    else:
        report_path = _resolve_provenance_file(
            sequences_manifest, explicit_report_value, "normalizer_fit_report_path", context
        )
        if report_path != expected_report_path:
            raise MaterializationError(
                "{} normalizer_fit_report_path is ambiguous; it must be the artifact bundle report.json".format(context)
            )
    if not report_path.is_file():
        raise MaterializationError("{} normalizer fit report does not exist: {}".format(context, report_path))
    report_sha256 = _sha256_file(report_path)
    declared_report_hash = sequence.get("normalizer_fit_report_sha256", _MISSING)
    if declared_report_hash is not _MISSING:
        if not _is_sha256(declared_report_hash) or str(declared_report_hash).lower() != report_sha256:
            raise MaterializationError("{} normalizer fit report hash does not match provenance".format(context))
    report = _read_json_object(report_path, "normalizer fit report")
    if report.get("schema_version") != "atomicdance-motion-normalizer-fit-v1":
        raise MaterializationError("{} has unsupported normalizer fit report schema".format(context))
    if report.get("fit_split") != "train":
        raise MaterializationError("{} normalizer fit report was not fit on train".format(context))
    report_input = report.get("input")
    if not isinstance(report_input, Mapping) or not _is_sha256(report_input.get("source_manifest_sha256")):
        raise MaterializationError("{} normalizer fit report lacks source-manifest provenance".format(context))
    fit_source_hash = str(report_input["source_manifest_sha256"]).lower()
    if fit_source_hash != expected_source_manifest_sha256:
        raise MaterializationError(
            "{} normalizer fit report source manifest does not match this frozen sources.jsonl".format(context)
        )
    report_normalizer = report.get("normalizer")
    if not isinstance(report_normalizer, Mapping) or not _is_sha256(report_normalizer.get("sha256")):
        raise MaterializationError("{} normalizer fit report lacks artifact hash".format(context))
    if str(report_normalizer["sha256"]).lower() != actual_artifact_hash:
        raise MaterializationError("{} normalizer fit report artifact hash mismatch".format(context))
    raw_contract = report.get("representation_contract")
    expected_raw_contract = {
        "motion": MODEL_MOTION_REPRESENTATION,
        "coordinate_system": MODEL_COORDINATE_SYSTEM,
        "normalization": "raw",
        "camera_in_model_input": False,
    }
    if not isinstance(raw_contract, Mapping) or dict(raw_contract) != expected_raw_contract:
        raise MaterializationError("{} normalizer fit report has an unsafe raw representation contract".format(context))
    return _NormalizerProvenance(
        artifact_path=artifact_path,
        artifact_sha256=actual_artifact_hash,
        fit_report_path=report_path,
        fit_report_sha256=report_sha256,
        fit_source_manifest_sha256=fit_source_hash,
        fit_split="train",
    )


def _ensure_new_output(output_dir: Path) -> Tuple[Path, Path]:
    if output_dir.exists() or output_dir.is_symlink():
        raise MaterializationError("refusing to overwrite immutable dataset bundle: {}".format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = output_dir.parent / ".{}.materialize.lock".format(output_dir.name)
    try:
        lock_dir.mkdir()
    except FileExistsError as error:
        raise MaterializationError("another materialization holds lock {}".format(lock_dir)) from error
    try:
        if output_dir.exists() or output_dir.is_symlink():
            raise MaterializationError("refusing to overwrite immutable dataset bundle: {}".format(output_dir))
        staging = Path(tempfile.mkdtemp(prefix=".{}.staging-".format(output_dir.name), dir=str(output_dir.parent)))
    except Exception:
        lock_dir.rmdir()
        raise
    return staging, lock_dir


def _publish_staging(staging: Path, output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise MaterializationError("refusing to overwrite immutable dataset bundle: {}".format(output_dir))
    os.rename(str(staging), str(output_dir))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _copy_and_verify(candidate: _CandidateWindow, out_motion: np.ndarray, out_music: np.ndarray, out_labels: np.ndarray, out_mask: np.ndarray) -> None:
    """Copy exact slices only, then verify no dtype/value transformation happened."""
    selection = slice(candidate.start_frame, candidate.end_frame_exclusive)
    motion = _load_npy(candidate.motion_path, "motion")
    music = _load_npy(candidate.music_path, "music")
    labels = _load_npy(candidate.labels_path, "labels")
    valid_mask = _load_npy(candidate.label_valid_mask_path, "label_valid_mask")
    source_motion = np.asarray(motion[selection])
    source_music = np.asarray(music[selection])
    source_labels = np.asarray(labels[selection])
    source_mask = np.asarray(valid_mask[selection])
    if not bool(np.all(source_mask)) or np.any(source_labels == INVALID_LABEL):
        raise MaterializationError(
            "internal selection invariant failed for {}; partially valid labels must not enter indexed arrays".format(
                candidate.window_id
            )
        )
    out_motion[candidate.array_index] = source_motion
    out_music[candidate.array_index] = source_music
    out_labels[candidate.array_index] = source_labels
    out_mask[candidate.array_index] = source_mask
    if not (
        np.array_equal(out_motion[candidate.array_index], source_motion)
        and np.array_equal(out_music[candidate.array_index], source_music)
        and np.array_equal(out_labels[candidate.array_index], source_labels)
        and np.array_equal(out_mask[candidate.array_index], source_mask)
    ):
        raise MaterializationError("copy verification failed for {}".format(candidate.window_id))


def _manifest_hashes(paths: Mapping[str, Path]) -> Dict[str, str]:
    return {name: _sha256_file(path) for name, path in sorted(paths.items())}


def _accepted_for_training(row: Mapping[str, Any]) -> bool:
    """Acceptance is an explicit QC decision, never inferred from labels."""
    qc = row.get("qc")
    return isinstance(qc, Mapping) and qc.get("accepted_for_training") is True


def materialize_atomic_windows(
    sources_path: Path,
    sequences_path: Path,
    labels_path: Path,
    output_dir: Path,
    *,
    window_length: int = 150,
    window_stride: int = 150,
    motion_dim: int = 151,
    music_dim: int = 35,
    num_classes: int = 101,
    min_label_valid_fraction: float = 1.0,
) -> Dict[str, Any]:
    """Build a source-safe, immutable indexed AtomicDance dataset bundle.

    ``min_label_valid_fraction`` is audited for every proposed window.  The
    current legacy indexed format can only materialize a window when the mask
    is exactly all-valid; lowering the threshold therefore makes partial
    windows visible in ``quarantine.jsonl`` but cannot make them trainable by
    an unmasked D3PM loss.  The final root is deliberately stricter than the
    raw preprocessing stage: it needs at least one all-valid train/val/test
    window and one verified normalized model-input contract.
    """
    if window_length < 1 or window_stride < 1 or window_stride > window_length:
        raise MaterializationError("window_length/window_stride must satisfy 1 <= stride <= length")
    if motion_dim < 1 or music_dim < 1 or num_classes < 1:
        raise MaterializationError("motion_dim, music_dim, and num_classes must be positive")
    if not isinstance(min_label_valid_fraction, (int, float)) or isinstance(min_label_valid_fraction, bool):
        raise MaterializationError("min_label_valid_fraction must be a number")
    min_label_valid_fraction = float(min_label_valid_fraction)
    if not 0.0 < min_label_valid_fraction <= 1.0:
        raise MaterializationError("min_label_valid_fraction must be in (0, 1]")

    sources_path = Path(sources_path).resolve()
    sequences_path = Path(sequences_path).resolve()
    labels_path = Path(labels_path).resolve()
    output_dir = Path(output_dir).resolve()
    source_rows = [_canonicalize_source_row(row) for row in _read_jsonl(sources_path, "sources")]
    sequence_rows = [_canonicalize_sequence_row(row) for row in _read_jsonl(sequences_path, "sequences")]
    label_rows = _read_jsonl(labels_path, "labels")
    sources = _index_unique(source_rows, "recording_id", "sources")
    sequences = _index_unique(sequence_rows, "sequence_id", "sequences")
    labels = _index_unique(label_rows, "sequence_id", "labels")
    identity_fields_used = _validate_source_groups(sources)
    _validate_sequence_foreign_keys(sources, sequences)
    _validate_label_foreign_keys(sequences, labels)
    sources_manifest_sha256 = _sha256_file(sources_path)

    staging, lock_dir = _ensure_new_output(output_dir)
    published = False
    try:
        candidates_by_split: Dict[str, List[_CandidateWindow]] = {split: [] for split in OUTPUT_SPLITS}
        quarantine: List[Dict[str, Any]] = []
        representation_contracts: Dict[Tuple[Any, ...], int] = {}
        normalizer_provenances: Dict[Tuple[str, str, str, str], _NormalizerProvenance] = {}
        # Count all candidate grid positions per recording before output
        # filtering, so names remain unique even when a preceding window is
        # rejected.  Retrieval safety does not depend on this name convention:
        # every selected window carries an explicit retrieval_group_id sidecar.
        recording_ordinals: Dict[str, int] = {}
        for sequence_id in sorted(sequences):
            sequence = sequences[sequence_id]
            context = _row_context("sequences", sequence)
            recording_id = _require_string(sequence, "recording_id", context)
            split = _require_split(sequence, context)
            label = labels.get(sequence_id)
            if label is None:
                quarantine.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "sequence",
                        "sequence_id": sequence_id,
                        "recording_id": recording_id,
                        "split": split,
                        "reason_code": "missing_label_row",
                    }
                )
                continue
            if label.get("status") != "accepted":
                quarantine.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "sequence",
                        "sequence_id": sequence_id,
                        "recording_id": recording_id,
                        "split": split,
                        "reason_code": "label_status_not_accepted",
                        "label_status": label.get("status"),
                    }
                )
                continue
            source = sources[recording_id]
            if not _accepted_for_training(source):
                quarantine.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "sequence",
                        "sequence_id": sequence_id,
                        "recording_id": recording_id,
                        "split": split,
                        "reason_code": "source_not_accepted_for_training",
                    }
                )
                continue
            if not _accepted_for_training(sequence):
                quarantine.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "sequence",
                        "sequence_id": sequence_id,
                        "recording_id": recording_id,
                        "split": split,
                        "reason_code": "sequence_not_accepted_for_training",
                    }
                )
                continue
            try:
                motion_path, music_path, motion_sha256, music_sha256, representation = _load_contiguous_sequence(
                    sequence,
                    sequences_manifest=sequences_path,
                    motion_dim=motion_dim,
                    music_dim=music_dim,
                )
                normalizer_provenance = _verify_normalizer_provenance(
                    sequence,
                    representation,
                    sequences_manifest=sequences_path,
                    expected_source_manifest_sha256=sources_manifest_sha256,
                )
                (
                    sequence_labels_path,
                    valid_mask_path,
                    labels_sha256,
                    valid_mask_sha256,
                    label_space_id,
                    producer_version,
                    fit_split,
                    fit_source_manifest_sha256,
                    producer_artifact_sha256,
                ) = _load_accepted_labels(
                    label,
                    sequence,
                    representation,
                    labels_manifest=labels_path,
                    expected_motion_sha256=motion_sha256,
                    expected_fit_source_manifest_sha256=sources_manifest_sha256,
                    num_classes=num_classes,
                )
            except _QuarantineSequence as error:
                quarantine.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "sequence",
                        "sequence_id": sequence_id,
                        "recording_id": recording_id,
                        "split": split,
                        "reason_code": error.code,
                        "detail": error.detail,
                        "label_status": "accepted",
                    }
                )
                continue
            frame_count = _require_int(sequence, "frame_count", context, minimum=1)
            starts = list(range(0, frame_count - window_length + 1, window_stride))
            if not starts:
                quarantine.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "sequence",
                        "sequence_id": sequence_id,
                        "recording_id": recording_id,
                        "split": split,
                        "reason_code": "sequence_shorter_than_window_length",
                        "frame_count": frame_count,
                        "window_length": window_length,
                    }
                )
                continue
            source_content_sha256, _ = _require_identity_hash(source, _row_context("sources", source))
            label_values = _load_npy(sequence_labels_path, "labels")
            mask_values = _load_npy(valid_mask_path, "label_valid_mask")
            contract_key = (
                representation["motion_representation_id"],
                representation["coordinate_system"],
                representation["normalization_state"],
                representation["normalization_artifact_sha256"],
                representation["normalization_fit_split"],
            )
            normalizer_key = (
                str(normalizer_provenance.artifact_path),
                normalizer_provenance.artifact_sha256,
                str(normalizer_provenance.fit_report_path),
                normalizer_provenance.fit_report_sha256,
            )
            for local_index, start in enumerate(starts):
                end = start + window_length
                next_ordinal = recording_ordinals.get(recording_id, 0)
                recording_ordinals[recording_id] = next_ordinal + 1
                sample_name = "{}_slice{}".format(recording_id, next_ordinal)
                window_id = "{}/window{:06d}".format(sequence_id, local_index)
                valid_fraction = float(np.mean(mask_values[start:end]))
                base_quarantine = {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "window",
                    "window_id": window_id,
                    "sample_name": sample_name,
                    "sequence_id": sequence_id,
                    "recording_id": recording_id,
                    "split": split,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "length": window_length,
                    "label_valid_fraction": valid_fraction,
                    "required_label_valid_fraction": min_label_valid_fraction,
                    "source_content_sha256": source_content_sha256,
                    "labels_sha256": labels_sha256,
                    "label_valid_mask_sha256": valid_mask_sha256,
                }
                if valid_fraction < min_label_valid_fraction:
                    quarantine.append(dict(base_quarantine, reason_code="label_valid_fraction_below_threshold"))
                    continue
                if valid_fraction != 1.0:
                    quarantine.append(
                        dict(base_quarantine, reason_code="partial_label_mask_not_legacy_compatible")
                    )
                    continue
                # This should already follow from the strict full-timeline
                # validation, but recheck each exact selection before it enters
                # the legacy dense array.
                if not bool(np.all(mask_values[start:end])) or np.any(label_values[start:end] == INVALID_LABEL):
                    quarantine.append(dict(base_quarantine, reason_code="window_label_mask_invariant_failed"))
                    continue
                array_index = len(candidates_by_split[split])
                candidates_by_split[split].append(
                    _CandidateWindow(
                        split=split,
                        window_id=window_id,
                        sample_name=sample_name,
                        sequence_id=sequence_id,
                        recording_id=recording_id,
                        retrieval_group_id=_require_string(sequence, "retrieval_group_id", context),
                        duplicate_content_group_id=_require_optional_group(sequence, "duplicate_content_group_id", context),
                        start_frame=start,
                        end_frame_exclusive=end,
                        array_index=array_index,
                        motion_path=motion_path,
                        music_path=music_path,
                        labels_path=sequence_labels_path,
                        label_valid_mask_path=valid_mask_path,
                        source_content_sha256=source_content_sha256,
                        source_motion_sha256=motion_sha256,
                        source_music_sha256=music_sha256,
                        label_labels_sha256=labels_sha256,
                        label_valid_mask_sha256=valid_mask_sha256,
                        input_motion_sha256=motion_sha256,
                        label_space_id=label_space_id,
                        producer_version=producer_version,
                        fit_split=fit_split,
                        fit_source_manifest_sha256=fit_source_manifest_sha256,
                        producer_artifact_sha256=producer_artifact_sha256,
                        motion_representation_id=representation["motion_representation_id"],
                        normalization_state=representation["normalization_state"],
                        normalization_artifact_sha256=representation["normalization_artifact_sha256"],
                        normalization_fit_split=representation["normalization_fit_split"],
                        coordinate_system=representation["coordinate_system"],
                    )
                )
                representation_contracts[contract_key] = representation_contracts.get(contract_key, 0) + 1
                normalizer_provenances[normalizer_key] = normalizer_provenance

        if len(representation_contracts) > 1:
            rendered = [
                {
                    "motion_representation_id": key[0],
                    "coordinate_system": key[1],
                    "normalization_state": key[2],
                    "normalization_artifact_sha256": key[3],
                    "normalization_fit_split": key[4],
                    "windows": count,
                }
                for key, count in sorted(representation_contracts.items(), key=lambda item: repr(item[0]))
            ]
            raise MaterializationError(
                "selected sequences mix motion representation/normalization contracts: {}".format(
                    json.dumps(rendered, sort_keys=True)
                )
            )
        if not candidates_by_split["train"]:
            raise MaterializationError(
                "refusing to publish an indexed bundle with zero materialized train windows; resolve QC/label gates and retry"
            )
        if not candidates_by_split["val"]:
            raise MaterializationError(
                "refusing to publish an indexed bundle with zero materialized val windows; validation selection must not use held-out test"
            )
        if not candidates_by_split["test"]:
            raise MaterializationError(
                "refusing to publish an indexed bundle with zero materialized test windows; train_atomic.py requires test evaluation data"
            )
        if len(normalizer_provenances) != 1:
            raise MaterializationError(
                "selected windows must bind to exactly one verified normalizer artifact/report, found {}".format(
                    len(normalizer_provenances)
                )
            )

        normalizer_provenance = next(iter(normalizer_provenances.values()))
        copied_normalizer_path = staging / "normalizer.pt"
        shutil.copyfile(normalizer_provenance.artifact_path, copied_normalizer_path)
        copied_normalizer_sha256 = _sha256_file(copied_normalizer_path)
        if copied_normalizer_sha256 != normalizer_provenance.artifact_sha256:
            raise MaterializationError("copied normalizer.pt hash does not match verified source artifact")

        windows_rows: List[Dict[str, Any]] = []
        split_artifacts: Dict[str, Dict[str, str]] = {}
        for split in OUTPUT_SPLITS:
            candidates = candidates_by_split[split]
            split_dir = staging / split
            split_dir.mkdir()
            motion_out_path = split_dir / "motion.npy"
            music_out_path = split_dir / "music.npy"
            labels_out_path = split_dir / "labels.npy"
            mask_out_path = split_dir / "label_valid_mask.npy"
            motion_out = np.lib.format.open_memmap(
                str(motion_out_path), mode="w+", dtype=np.float32, shape=(len(candidates), window_length, motion_dim)
            )
            music_out = np.lib.format.open_memmap(
                str(music_out_path), mode="w+", dtype=np.float32, shape=(len(candidates), window_length, music_dim)
            )
            labels_out = np.lib.format.open_memmap(
                str(labels_out_path), mode="w+", dtype=np.int64, shape=(len(candidates), window_length)
            )
            mask_out = np.lib.format.open_memmap(
                str(mask_out_path), mode="w+", dtype=np.bool_, shape=(len(candidates), window_length)
            )
            for candidate in candidates:
                _copy_and_verify(candidate, motion_out, music_out, labels_out, mask_out)
            # Closing is important before hashing and publication, especially
            # for mmap-backed files on long-running preprocessing hosts.
            del motion_out, music_out, labels_out, mask_out
            names = [candidate.sample_name for candidate in candidates]
            retrieval_groups = [candidate.retrieval_group_id for candidate in candidates]
            if len(names) != len(set(names)):
                raise MaterializationError("internal duplicate sample name in split {}".format(split))
            names_path = split_dir / "names.json"
            names_path.write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            # This indexed sidecar is deliberately aligned with the arrays,
            # rather than requiring consumers to reverse-engineer a recording
            # identity from a window name.  In particular, multiple cameras
            # of one performance may have different recording/sample names
            # but the same retrieval group.
            retrieval_groups_path = split_dir / "retrieval_groups.json"
            retrieval_groups_path.write_text(
                json.dumps(retrieval_groups, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            split_artifacts[split] = _manifest_hashes(
                {
                    "motion.npy": motion_out_path,
                    "music.npy": music_out_path,
                    "labels.npy": labels_out_path,
                    "label_valid_mask.npy": mask_out_path,
                    "names.json": names_path,
                    "retrieval_groups.json": retrieval_groups_path,
                }
            )
            for candidate in candidates:
                windows_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "window_id": candidate.window_id,
                        "sample_name": candidate.sample_name,
                        "sequence_id": candidate.sequence_id,
                        "recording_id": candidate.recording_id,
                        "retrieval_group_id": candidate.retrieval_group_id,
                        "duplicate_content_group_id": candidate.duplicate_content_group_id,
                        "split": candidate.split,
                        "array_index": candidate.array_index,
                        "start_frame": candidate.start_frame,
                        "end_frame_exclusive": candidate.end_frame_exclusive,
                        "length": window_length,
                        "motion_path": "{}/motion.npy".format(split),
                        "music_path": "{}/music.npy".format(split),
                        "labels_path": "{}/labels.npy".format(split),
                        "label_valid_mask_path": "{}/label_valid_mask.npy".format(split),
                        "label_valid_fraction": 1.0,
                        "label_space_id": candidate.label_space_id,
                        "source_content_sha256": candidate.source_content_sha256,
                        "source_motion_sha256": candidate.source_motion_sha256,
                        "source_music_sha256": candidate.source_music_sha256,
                        "label_labels_sha256": candidate.label_labels_sha256,
                        "label_valid_mask_sha256": candidate.label_valid_mask_sha256,
                        "input_motion_sha256": candidate.input_motion_sha256,
                        "producer_version": candidate.producer_version,
                        "fit_split": candidate.fit_split,
                        "fit_source_manifest_sha256": candidate.fit_source_manifest_sha256,
                        "producer_artifact_sha256": candidate.producer_artifact_sha256,
                        "motion_representation_id": candidate.motion_representation_id,
                        "coordinate_system": candidate.coordinate_system,
                        "normalization_state": candidate.normalization_state,
                        "normalization_artifact_sha256": candidate.normalization_artifact_sha256,
                        "normalization_fit_split": candidate.normalization_fit_split,
                        "camera_in_model_input": False,
                        "materializer_version": MATERIALIZER_VERSION,
                    }
                )
        windows_rows.sort(key=lambda row: (row["split"], row["array_index"]))
        quarantine.sort(
            key=lambda row: (
                row.get("split", ""),
                row.get("recording_id", ""),
                row.get("sequence_id", ""),
                row.get("start_frame", -1),
                row.get("reason_code", ""),
            )
        )
        _write_jsonl(staging / "windows.jsonl", windows_rows)
        _write_jsonl(staging / "quarantine.jsonl", quarantine)
        selected_contract: Optional[Dict[str, Any]] = None
        if representation_contracts:
            key = next(iter(representation_contracts))
            selected_contract = {
                "motion_representation_id": key[0],
                "coordinate_system": key[1],
                "normalization_state": key[2],
                "normalization_artifact_sha256": key[3],
                "normalization_fit_split": key[4],
                "camera_in_model_input": False,
            }
        build = {
            "schema_version": SCHEMA_VERSION,
            "materializer_version": MATERIALIZER_VERSION,
            "publication": "immutable_new_directory_only_atomic_rename",
            # Which field identified each source.  A release built from one
            # ingestion path shows a single entry; this AIST++ corpus shows two,
            # and that is a fact a later duplicate audit has to know -- hashes
            # from different fields are not comparable to each other.
            "source_identity_hash_fields": dict(sorted(identity_fields_used.items())),
            "input_manifests": {
                "sources.jsonl": {"path": str(sources_path), "sha256": sources_manifest_sha256},
                "sequences.jsonl": {"path": str(sequences_path), "sha256": _sha256_file(sequences_path)},
                "labels.jsonl": {"path": str(labels_path), "sha256": _sha256_file(labels_path)},
            },
            "window_policy": {
                "window_length": window_length,
                "window_stride": window_stride,
                "fps": MODEL_FPS,
                "motion_dim": motion_dim,
                "music_dim": music_dim,
                "num_classes": num_classes,
                "required_label_valid_fraction": min_label_valid_fraction,
                "legacy_indexed_output_requires_all_valid_mask": True,
                "invalid_label_sentinel": INVALID_LABEL,
                "invalid_labels_are_never_rewritten_to_transition_zero": True,
            },
            "representation_contract": selected_contract,
            "normalization_policy": "pre-normalized model-input motion only; no normalization/refit in materializer; frozen train-only normalizer copied into root",
            "normalizer": {
                "source_artifact": str(normalizer_provenance.artifact_path),
                "source_artifact_sha256": normalizer_provenance.artifact_sha256,
                "published_artifact": "normalizer.pt",
                "published_artifact_sha256": copied_normalizer_sha256,
                "fit_report": str(normalizer_provenance.fit_report_path),
                "fit_report_sha256": normalizer_provenance.fit_report_sha256,
                "fit_split": normalizer_provenance.fit_split,
                "fit_source_manifest_sha256": normalizer_provenance.fit_source_manifest_sha256,
                "copy_policy": "verified_byte_copy_from_single_frozen_fit_artifact",
            },
            "fit_provenance_policy": {
                "required_label_status": "accepted",
                "required_fit_split": "train",
                "required_fields": [
                    "fit_source_manifest_sha256",
                    "producer_artifact_sha256",
                    "input_motion_sha256",
                ],
            },
            "counts": {
                "sources": len(sources),
                "sequences": len(sequences),
                "label_rows": len(labels),
                "materialized_windows": {split: len(candidates_by_split[split]) for split in OUTPUT_SPLITS},
                "quarantined_records": len(quarantine),
            },
            "artifacts": {
                "normalizer.pt": copied_normalizer_sha256,
                "splits": split_artifacts,
                "windows.jsonl": _sha256_file(staging / "windows.jsonl"),
                "quarantine.jsonl": _sha256_file(staging / "quarantine.jsonl"),
            },
        }
        build_path = staging / "build.json"
        build_path.write_text(json.dumps(build, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        _publish_staging(staging, output_dir)
        published = True
        result = dict(build)
        result["output_dir"] = str(output_dir)
        return result
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
        if lock_dir.exists():
            lock_dir.rmdir()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="source-level sources.jsonl")
    parser.add_argument("--sequences", required=True, help="source-level sequences.jsonl")
    parser.add_argument("--labels", required=True, help="sequence-level labels.jsonl")
    parser.add_argument("--output-dir", required=True, help="new immutable indexed dataset directory")
    parser.add_argument("--window-length", type=int, default=150)
    parser.add_argument("--window-stride", type=int, default=150)
    parser.add_argument("--motion-dim", type=int, default=151)
    parser.add_argument("--music-dim", type=int, default=35)
    parser.add_argument("--num-classes", type=int, default=101)
    parser.add_argument("--min-label-valid-fraction", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        report = materialize_atomic_windows(
            Path(args.sources),
            Path(args.sequences),
            Path(args.labels),
            Path(args.output_dir),
            window_length=args.window_length,
            window_stride=args.window_stride,
            motion_dim=args.motion_dim,
            music_dim=args.music_dim,
            num_classes=args.num_classes,
            min_label_valid_fraction=args.min_label_valid_fraction,
        )
    except MaterializationError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
