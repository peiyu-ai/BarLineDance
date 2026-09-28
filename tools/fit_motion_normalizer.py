#!/usr/bin/env python3
"""Fit an immutable, train-only min/max normalizer for raw AtomicDance motion.

The released AtomicDance checkpoint expects a Torch file with exactly two
151-dimensional tensors::

    {"data_min": Tensor[151], "data_max": Tensor[151]}

This command creates that artifact only from a frozen source/sequence
manifest pair.  It is intentionally stricter than a generic ``min``/``max``
script:

* every sequence is joined to its authoritative source row and the complete
  recording/retrieval-group/known-duplicate split is audited before any array
  is read;
* only ``split == 'train'`` rows with ``qc.accepted_for_training is True``
  are eligible;
* selected arrays must explicitly attest the raw, z-up, camera-decoupled
  AtomicDance 151-D representation; and
* arrays are memory-mapped and scanned in frame blocks.  No val/test array is
  loaded, no input is normalized, and malformed values are never repaired.

The output is an immutable directory:

    <output-dir>/normalizer.pt
    <output-dir>/fit_sequences.jsonl
    <output-dir>/report.json

``fit_sequences.jsonl`` contains the exact selected sequence IDs and hashes;
``report.json`` binds them to the source and sequence manifest hashes.  The
directory must not exist beforehand, so a fit cannot silently replace a
normalizer used by a checkpoint or evaluation.  Pending or non-train rows may
omit a raw motion asset/representation because they are identity-audited but
never read.  A row cannot become selected without the complete raw-151D
contract below.
"""

from __future__ import annotations

import argparse
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


SCHEMA_VERSION = "atomicdance-motion-normalizer-fit-v1"
MOTION_DIM = 151
FIT_SPLIT = "train"
ALLOWED_SPLITS = frozenset(("train", "val", "test"))
DEFAULT_BLOCK_FRAMES = 8192
OUTPUT_NORMALIZER = "normalizer.pt"
OUTPUT_SEQUENCE_MANIFEST = "fit_sequences.jsonl"
OUTPUT_REPORT = "report.json"

# This is the exact representation emitted by the validated WHAM reconciler.
# A representation that merely has 151 columns is not sufficient: it may be
# normalized already, camera-relative, y-up, or another model's feature order.
RAW_REPRESENTATION_CONTRACT: Dict[str, Any] = {
    "motion": "AtomicDance_151D",
    "coordinate_system": "z_up_world_body_only",
    "normalization": "raw",
    "camera_in_model_input": False,
}


class NormalizerFitError(ValueError):
    """A frozen data contract was not safe to use for a train-only fit."""


@dataclass(frozen=True)
class SourceRow:
    recording_id: str
    retrieval_group_id: str
    duplicate_content_group_id: Optional[str]
    split: str
    qc: Optional[Mapping[str, Any]]


@dataclass(frozen=True)
class SequenceRow:
    sequence_id: str
    recording_id: str
    retrieval_group_id: str
    duplicate_content_group_id: Optional[str]
    split: str
    assets: Optional[Mapping[str, Any]]
    representation: Optional[Mapping[str, Any]]
    qc: Optional[Mapping[str, Any]]


@dataclass(frozen=True)
class FittedSequence:
    sequence_id: str
    recording_id: str
    retrieval_group_id: str
    duplicate_content_group_id: Optional[str]
    split: str
    motion_path: Path
    motion_sha256: str
    frame_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_strings(values: Iterable[str]) -> str:
    """Hash an ordered identity list unambiguously and reproducibly."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("JSONL manifest does not exist: {}".format(path))
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise NormalizerFitError(
                    "invalid JSONL at {}:{}: {}".format(path, line_number, error)
                ) from error
            if not isinstance(value, Mapping):
                raise NormalizerFitError(
                    "JSONL item at {}:{} is not an object".format(path, line_number)
                )
            records.append(dict(value))
    if not records:
        raise NormalizerFitError("JSONL manifest is empty: {}".format(path))
    return records


def _required_id(record: Mapping[str, Any], field: str, *, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise NormalizerFitError("{} lacks non-empty {}".format(context, field))
    return value


def _required_split(record: Mapping[str, Any], *, context: str) -> str:
    split = _required_id(record, "split", context=context)
    if split not in ALLOWED_SPLITS:
        raise NormalizerFitError(
            "{} has unsupported split {!r}; expected one of {}".format(
                context, split, ", ".join(sorted(ALLOWED_SPLITS))
            )
        )
    return split


def _optional_mapping(record: Mapping[str, Any], field: str, *, context: str) -> Optional[Mapping[str, Any]]:
    """Return an optional nested object while rejecting scalar schema drift."""
    if field not in record or record[field] is None:
        return None
    value = record[field]
    if not isinstance(value, Mapping):
        raise NormalizerFitError("{} has non-object {}".format(context, field))
    return value


def _required_optional_group_id(
    record: Mapping[str, Any], field: str, *, context: str
) -> Optional[str]:
    """Read an explicitly declared nullable grouping identity.

    ``null`` is meaningful: it says duplicate-content QC has not assigned a
    group.  Omitting the field is not equivalent, because a producer could
    otherwise silently drop provenance while a train-only fit continues.
    """
    if field not in record:
        raise NormalizerFitError(
            "{} must explicitly include {} (string or null)".format(context, field)
        )
    if record[field] is None:
        return None
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise NormalizerFitError("{} has invalid {}; expected null or non-empty string".format(context, field))
    return value


def _parse_sources(records: Sequence[Mapping[str, Any]]) -> Dict[str, SourceRow]:
    """Validate unique source identities and group-level split ownership."""
    sources: Dict[str, SourceRow] = {}
    retrieval_group_splits: Dict[str, set[str]] = {}
    duplicate_group_splits: Dict[str, set[str]] = {}
    for index, record in enumerate(records, 1):
        context = "source manifest row {}".format(index)
        recording_id = _required_id(record, "recording_id", context=context)
        retrieval_group_id = _required_id(record, "retrieval_group_id", context=context)
        duplicate_content_group_id = _required_optional_group_id(
            record, "duplicate_content_group_id", context=context
        )
        split = _required_split(record, context=context)
        qc = _optional_mapping(record, "qc", context=context)
        if recording_id in sources:
            raise NormalizerFitError(
                "source manifest has duplicate recording_id {!r}".format(recording_id)
            )
        sources[recording_id] = SourceRow(
            recording_id=recording_id,
            retrieval_group_id=retrieval_group_id,
            duplicate_content_group_id=duplicate_content_group_id,
            split=split,
            qc=qc,
        )
        retrieval_group_splits.setdefault(retrieval_group_id, set()).add(split)
        if duplicate_content_group_id is not None:
            duplicate_group_splits.setdefault(duplicate_content_group_id, set()).add(split)

    collisions = {
        group: sorted(splits)
        for group, splits in retrieval_group_splits.items()
        if len(splits) > 1
    }
    if collisions:
        preview = ", ".join(
            "{} in {}".format(group, "/".join(splits))
            for group, splits in sorted(collisions.items())[:8]
        )
        raise NormalizerFitError(
            "source manifest splits retrieval_group_id across splits: {}".format(preview)
        )
    duplicate_collisions = {
        group: sorted(splits)
        for group, splits in duplicate_group_splits.items()
        if len(splits) > 1
    }
    if duplicate_collisions:
        preview = ", ".join(
            "{} in {}".format(group, "/".join(splits))
            for group, splits in sorted(duplicate_collisions.items())[:8]
        )
        raise NormalizerFitError(
            "source manifest splits duplicate_content_group_id across splits: {}".format(preview)
        )
    return sources


def _parse_sequences(
    records: Sequence[Mapping[str, Any]],
    *,
    sources: Mapping[str, SourceRow],
) -> Tuple[List[SequenceRow], Dict[str, int]]:
    """Join all sequence rows to source rows before selecting train data.

    Val/test motion files intentionally are not opened below, but their
    identities are still audited here.  That makes a leaked retrieval group a
    hard error rather than an accidental exclusion from the min/max scan.
    """
    rows: List[SequenceRow] = []
    seen_sequences: set[str] = set()
    retrieval_group_splits: Dict[str, set[str]] = {}
    duplicate_group_splits: Dict[str, set[str]] = {}
    by_split = {split: 0 for split in sorted(ALLOWED_SPLITS)}
    for index, record in enumerate(records, 1):
        context = "sequence manifest row {}".format(index)
        sequence_id = _required_id(record, "sequence_id", context=context)
        recording_id = _required_id(record, "recording_id", context=context)
        retrieval_group_id = _required_id(record, "retrieval_group_id", context=context)
        duplicate_content_group_id = _required_optional_group_id(
            record, "duplicate_content_group_id", context=context
        )
        split = _required_split(record, context=context)
        if sequence_id in seen_sequences:
            raise NormalizerFitError(
                "sequence manifest has duplicate sequence_id {!r}".format(sequence_id)
            )
        seen_sequences.add(sequence_id)
        source = sources.get(recording_id)
        if source is None:
            raise NormalizerFitError(
                "{} references recording_id {!r} absent from source manifest".format(context, recording_id)
            )
        if source.retrieval_group_id != retrieval_group_id:
            raise NormalizerFitError(
                "{} retrieval_group_id {!r} does not match source manifest {!r}".format(
                    context, retrieval_group_id, source.retrieval_group_id
                )
            )
        if source.split != split:
            raise NormalizerFitError(
                "{} split {!r} does not match source manifest {!r}".format(
                    context, split, source.split
                )
            )
        if source.duplicate_content_group_id != duplicate_content_group_id:
            raise NormalizerFitError(
                "{} duplicate_content_group_id {!r} does not match source manifest {!r}".format(
                    context, duplicate_content_group_id, source.duplicate_content_group_id
                )
            )
        # Val/test and pending rows are intentionally not array-validated or
        # opened.  They can lack a raw asset because e.g. their 3D conversion
        # is still pending; identity/split ownership remains fully audited.
        assets = _optional_mapping(record, "assets", context=context)
        representation = _optional_mapping(record, "representation", context=context)
        qc = _optional_mapping(record, "qc", context=context)
        retrieval_group_splits.setdefault(retrieval_group_id, set()).add(split)
        if duplicate_content_group_id is not None:
            duplicate_group_splits.setdefault(duplicate_content_group_id, set()).add(split)
        by_split[split] += 1
        rows.append(
            SequenceRow(
                sequence_id=sequence_id,
                recording_id=recording_id,
                retrieval_group_id=retrieval_group_id,
                duplicate_content_group_id=duplicate_content_group_id,
                split=split,
                assets=assets,
                representation=representation,
                qc=qc,
            )
        )

    collisions = {
        group: sorted(splits)
        for group, splits in retrieval_group_splits.items()
        if len(splits) > 1
    }
    if collisions:
        preview = ", ".join(
            "{} in {}".format(group, "/".join(splits))
            for group, splits in sorted(collisions.items())[:8]
        )
        raise NormalizerFitError(
            "sequence manifest splits retrieval_group_id across splits: {}".format(preview)
        )
    duplicate_collisions = {
        group: sorted(splits)
        for group, splits in duplicate_group_splits.items()
        if len(splits) > 1
    }
    if duplicate_collisions:
        preview = ", ".join(
            "{} in {}".format(group, "/".join(splits))
            for group, splits in sorted(duplicate_collisions.items())[:8]
        )
        raise NormalizerFitError(
            "sequence manifest splits duplicate_content_group_id across splits: {}".format(preview)
        )
    return rows, by_split


def _assert_raw_representation(sequence: SequenceRow) -> None:
    if sequence.representation is None:
        raise NormalizerFitError(
            "{} lacks representation required for an accepted train fit".format(sequence.sequence_id)
        )
    for field, expected in RAW_REPRESENTATION_CONTRACT.items():
        actual = sequence.representation.get(field)
        if actual != expected:
            raise NormalizerFitError(
                "{} has unsafe representation.{}={!r}; expected {!r}".format(
                    sequence.sequence_id, field, actual, expected
                )
            )


def _is_explicitly_accepted_for_training(
    qc: Optional[Mapping[str, Any]], *, context: str
) -> bool:
    """Return true only for an explicit, non-contradictory acceptance.

    A false/missing flag is an ordinary pending candidate and is skipped.  A
    true flag paired with an explicit known failed status is contradictory and
    rejected rather than promoted by this fitter.
    """
    if qc is None:
        return False
    accepted = qc.get("accepted_for_training")
    if accepted is not True:
        return False
    unsafe_statuses = frozenset(("pending", "quarantine", "failed", "rejected", "invalid"))
    for field in ("status", "hmr_status", "conversion_validation", "audio_feature_status"):
        value = qc.get(field)
        if isinstance(value, str) and value.strip().lower() in unsafe_statuses:
            raise NormalizerFitError(
                "{} sets qc.accepted_for_training=true but qc.{}={!r}".format(
                    context, field, value
                )
            )
    return True


def _assert_accepted_train_qc(sequence: SequenceRow) -> bool:
    return _is_explicitly_accepted_for_training(
        sequence.qc, context="sequence {!r}".format(sequence.sequence_id)
    )


def _assert_accepted_train_source_qc(source: SourceRow) -> bool:
    """Require recording-level acceptance in addition to sequence QC.

    A sequence is never enough to promote a recording into a fit.  This
    prevents a stale or over-eager segment-level producer from bypassing a
    source-level quarantine/rejection decision.
    """
    return _is_explicitly_accepted_for_training(
        source.qc, context="source {!r}".format(source.recording_id)
    )


def _selected_motion_asset(sequence: SequenceRow) -> str:
    if sequence.assets is None:
        raise NormalizerFitError(
            "{} lacks assets required for an accepted train fit".format(sequence.sequence_id)
        )
    value = sequence.assets.get("motion_151_raw")
    if not isinstance(value, str) or not value.strip():
        raise NormalizerFitError(
            "{} lacks non-empty assets.motion_151_raw required for an accepted train fit".format(
                sequence.sequence_id
            )
        )
    return value


def _resolve_asset(value: str, *, manifest_parent: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_parent / candidate
    return candidate.resolve()


def _scan_motion_array(
    sequence: SequenceRow,
    *,
    manifest_parent: Path,
    block_frames: int,
) -> Tuple[Path, str, int, np.ndarray, np.ndarray]:
    """Memory-map one raw sequence and return its finite per-dimension range."""
    motion_value = _selected_motion_asset(sequence)
    motion_path = _resolve_asset(motion_value, manifest_parent=manifest_parent)
    if not motion_path.is_file():
        raise NormalizerFitError(
            "{} motion asset does not exist: {}".format(sequence.sequence_id, motion_path)
        )
    try:
        values = np.load(str(motion_path), mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise NormalizerFitError(
            "{} cannot memory-map motion asset {}: {}".format(sequence.sequence_id, motion_path, error)
        ) from error
    if values.ndim != 2 or tuple(values.shape[1:]) != (MOTION_DIM,):
        raise NormalizerFitError(
            "{} motion asset has shape {}; expected [T, {}]".format(
                sequence.sequence_id, tuple(values.shape), MOTION_DIM
            )
        )
    if len(values) < 1:
        raise NormalizerFitError("{} motion asset is empty".format(sequence.sequence_id))
    if not np.issubdtype(values.dtype, np.floating):
        raise NormalizerFitError(
            "{} motion asset dtype {} is not floating".format(sequence.sequence_id, values.dtype)
        )

    minimum = np.full(MOTION_DIM, np.inf, dtype=np.float64)
    maximum = np.full(MOTION_DIM, -np.inf, dtype=np.float64)
    for start in range(0, len(values), block_frames):
        stop = min(start + block_frames, len(values))
        # This materializes only the current frame block, not the sequence nor
        # the corpus.  Converting to float64 preserves a stable accumulator.
        block = np.asarray(values[start:stop], dtype=np.float64)
        if not np.isfinite(block).all():
            raise NormalizerFitError(
                "{} motion asset contains non-finite values in frames [{}, {})".format(
                    sequence.sequence_id, start, stop
                )
            )
        minimum = np.minimum(minimum, np.min(block, axis=0))
        maximum = np.maximum(maximum, np.max(block, axis=0))
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():  # defensive invariant
        raise NormalizerFitError("{} did not yield finite min/max statistics".format(sequence.sequence_id))

    # The consumer is explicitly float32.  Refuse an array whose finite values
    # would become infinity in the saved artifact instead of clamping it.
    float32_limit = np.finfo(np.float32).max
    if np.any(np.abs(minimum) > float32_limit) or np.any(np.abs(maximum) > float32_limit):
        raise NormalizerFitError(
            "{} raw values exceed the float32 range required by normalizer.pt".format(sequence.sequence_id)
        )
    return motion_path, _sha256_file(motion_path), int(len(values)), minimum, maximum


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _fit_manifest_row(fitted: FittedSequence) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence_id": fitted.sequence_id,
        "recording_id": fitted.recording_id,
        "retrieval_group_id": fitted.retrieval_group_id,
        "duplicate_content_group_id": fitted.duplicate_content_group_id,
        "split": fitted.split,
        "motion_151_raw": str(fitted.motion_path),
        "motion_151_raw_sha256": fitted.motion_sha256,
        "frame_count": fitted.frame_count,
        "representation": dict(RAW_REPRESENTATION_CONTRACT),
    }


def fit_motion_normalizer(
    sequence_manifest: Path,
    source_manifest: Path,
    output_dir: Path,
    *,
    block_frames: int = DEFAULT_BLOCK_FRAMES,
) -> Dict[str, Any]:
    """Fit and atomically publish a raw 151-D train-only normalizer bundle.

    ``sequence_manifest`` and ``source_manifest`` are both required rather
    than inferred from neighboring files.  The exact file hashes are part of
    the output provenance, so later training/evaluation can reject a stale or
    re-split normalizer.
    """
    if isinstance(block_frames, bool) or not isinstance(block_frames, int) or block_frames < 1:
        raise NormalizerFitError("block_frames must be a positive integer")
    sequence_manifest = sequence_manifest.expanduser().resolve()
    source_manifest = source_manifest.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            "refusing to overwrite existing normalizer bundle {}; choose a new versioned output directory".format(
                output_dir
            )
        )

    source_records = _read_jsonl(source_manifest)
    sequence_records = _read_jsonl(sequence_manifest)
    source_hash = _sha256_file(source_manifest)
    sequence_hash = _sha256_file(sequence_manifest)
    sources = _parse_sources(source_records)
    sequences, input_by_split = _parse_sequences(sequence_records, sources=sources)

    selected = sorted(
        (
            sequence
            for sequence in sequences
            if sequence.split == FIT_SPLIT
            and _assert_accepted_train_qc(sequence)
            and _assert_accepted_train_source_qc(sources[sequence.recording_id])
        ),
        key=lambda sequence: sequence.sequence_id,
    )
    if not selected:
        raise NormalizerFitError(
            "no explicitly accepted train sequences; source.qc.accepted_for_training and "
            "sequence.qc.accepted_for_training must both be exactly true"
        )

    # Validate every selected representation before writing anything.  This
    # makes a mixed raw/normalized corpus a fail-closed input error, not a
    # partially-published normalizer.
    for sequence in selected:
        _assert_raw_representation(sequence)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".{}-staging-".format(output_dir.name), dir=str(output_dir.parent))
    )
    try:
        corpus_minimum = np.full(MOTION_DIM, np.inf, dtype=np.float64)
        corpus_maximum = np.full(MOTION_DIM, -np.inf, dtype=np.float64)
        fitted_sequences: List[FittedSequence] = []
        for sequence in selected:
            motion_path, motion_hash, frames, minimum, maximum = _scan_motion_array(
                sequence,
                manifest_parent=sequence_manifest.parent,
                block_frames=block_frames,
            )
            corpus_minimum = np.minimum(corpus_minimum, minimum)
            corpus_maximum = np.maximum(corpus_maximum, maximum)
            fitted_sequences.append(
                FittedSequence(
                    sequence_id=sequence.sequence_id,
                    recording_id=sequence.recording_id,
                    retrieval_group_id=sequence.retrieval_group_id,
                    duplicate_content_group_id=sequence.duplicate_content_group_id,
                    split=sequence.split,
                    motion_path=motion_path,
                    motion_sha256=motion_hash,
                    frame_count=frames,
                )
            )
        if not np.isfinite(corpus_minimum).all() or not np.isfinite(corpus_maximum).all():
            raise NormalizerFitError("corpus scan did not yield finite min/max statistics")

        # Do not add an epsilon or otherwise "fix" constant dimensions.  The
        # downstream normalizer already has its own safe zero-range handling;
        # this artifact must remain an exact summary of raw train inputs.
        normalizer = {
            "data_min": torch.from_numpy(corpus_minimum.astype(np.float32, copy=True)),
            "data_max": torch.from_numpy(corpus_maximum.astype(np.float32, copy=True)),
        }
        torch.save(normalizer, str(staging / OUTPUT_NORMALIZER))
        normalizer_hash = _sha256_file(staging / OUTPUT_NORMALIZER)

        fitted_rows = [_fit_manifest_row(item) for item in fitted_sequences]
        _write_jsonl(staging / OUTPUT_SEQUENCE_MANIFEST, fitted_rows)
        fitted_manifest_hash = _sha256_file(staging / OUTPUT_SEQUENCE_MANIFEST)
        sequence_ids = [item.sequence_id for item in fitted_sequences]
        report: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "fit_split": FIT_SPLIT,
            "representation_contract": dict(RAW_REPRESENTATION_CONTRACT),
            "input": {
                "sequence_manifest": str(sequence_manifest),
                "sequence_manifest_sha256": sequence_hash,
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": source_hash,
            },
            "counts": {
                "input_sequences": len(sequences),
                "input_by_split": input_by_split,
                "selected_sequences": len(fitted_sequences),
                "selected_recordings": len({item.recording_id for item in fitted_sequences}),
                "selected_retrieval_groups": len(
                    {item.retrieval_group_id for item in fitted_sequences}
                ),
                "selected_frames": int(sum(item.frame_count for item in fitted_sequences)),
                "train_not_explicitly_accepted": sum(
                    item.split == FIT_SPLIT
                    and (item.qc is None or item.qc.get("accepted_for_training") is not True)
                    for item in sequences
                ),
                "excluded_val_sequences": input_by_split["val"],
                "excluded_test_sequences": input_by_split["test"],
            },
            "selected_sequences": {
                "manifest": str((output_dir / OUTPUT_SEQUENCE_MANIFEST).resolve()),
                "manifest_sha256": fitted_manifest_hash,
                "ids_sha256": _sha256_strings(sequence_ids),
                "ordering": "lexicographic sequence_id",
            },
            "normalizer": {
                "path": str((output_dir / OUTPUT_NORMALIZER).resolve()),
                "sha256": normalizer_hash,
                "keys": ["data_min", "data_max"],
                "dtype": "torch.float32",
                "shape": [MOTION_DIM],
                "constant_dimensions": int(np.sum(corpus_minimum == corpus_maximum)),
            },
            "streaming": {
                "algorithm": "per-dimension finite min/max over memory-mapped frame blocks",
                "block_frames": block_frames,
                "input_arrays_normalized_or_repaired": False,
            },
            "publication": "immutable_new_directory_only_atomic_rename",
        }
        with (staging / OUTPUT_REPORT).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")

        # Check again immediately before publish.  This is also useful in the
        # ordinary case where another job prepared a versioned output between
        # parsing and completion.
        if output_dir.exists():  # pragma: no cover - race protection
            raise FileExistsError("refusing to overwrite existing normalizer bundle {}".format(output_dir))
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
        help="full sequence JSONL with explicit source identity, raw motion asset, representation, and qc",
    )
    parser.add_argument(
        "--source-manifest",
        required=True,
        help="frozen source JSONL used to validate recording/retrieval-group split ownership",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new immutable normalizer bundle directory; it must not already exist",
    )
    parser.add_argument(
        "--block-frames",
        type=int,
        default=DEFAULT_BLOCK_FRAMES,
        help="maximum memory-mapped motion frames scanned at once (default: %(default)s)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = fit_motion_normalizer(
            Path(args.sequence_manifest),
            Path(args.source_manifest),
            Path(args.output_dir),
            block_frames=args.block_frames,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        raise SystemExit("error: {}".format(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
