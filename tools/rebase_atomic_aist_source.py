#!/usr/bin/env python3
"""Publish a raw, source-safe AIST release from AtomicDance's legacy bundle.

The released AtomicDance AIST arrays are already min/max normalised.  They
are useful as a *motion representation* baseline, but are not a suitable
source corpus as-is: their train/test split is at camera-channel granularity,
and their window labels disagree in overlapping frames.  This tool consumes
only the label-free source timelines reconstructed by
``build_source_manifest.py`` and the matching upstream ``normalizer.pt``.

It creates a new immutable release with three important properties:

* every 151-D motion timeline is exactly inverse-minmaxed to raw coordinates;
* a performance (legacy name with its trailing ``_chNN`` removed) is the
  retrieval/split unit, old upstream test performances remain test, and a
  stable SHA-256 partition reserves whole old-train performances for val; and
* no legacy label file is opened, copied, or emitted.

The output is deliberately a *source release*, not a final supervised Atomic
dataset.  It has raw motion, music, timeline, source and split provenance so
that ``fit_motion_normalizer.py`` and ``apply_motion_normalizer.py`` can
consume it directly.  Atomic labels must still be produced from train sources
only before ``materialize_atomic_windows.py`` is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# sys.path[0] is tools/ when this is run as a script, not the repo root; the
# sibling import below otherwise depends on the ambient PYTHONPATH including
# the CWD, which is a coincidence rather than a contract.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.fit_motion_normalizer import MOTION_DIM, RAW_REPRESENTATION_CONTRACT  # noqa: E402


SCHEMA_VERSION = "atomicdance-aist-raw-source-release-v1"
REBASE_VERSION = "atomicdance-aist-upstream-minmax-inverse-v1"
DEFAULT_VAL_FRACTION = 0.10
WINDOW_LENGTH = 150
MUSIC_DIM = 35
_CHANNEL_SUFFIX = re.compile(r"^(?P<performance>.+)_ch\d+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AISTRebaseError(ValueError):
    """The candidate input cannot safely become a raw source release."""


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


def _combined_content_sha256(motion_sha256: str, music_sha256: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"AtomicDance raw AIST source release v1\n")
    digest.update(b"motion:")
    digest.update(motion_sha256.encode("ascii"))
    digest.update(b"\nmusic:")
    digest.update(music_sha256.encode("ascii"))
    digest.update(b"\n")
    return digest.hexdigest()


def _read_json(path: Path, *, kind: str) -> Dict[str, Any]:
    if not path.is_file():
        raise AISTRebaseError("{} does not exist: {}".format(kind, path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AISTRebaseError("cannot read {} {}: {}".format(kind, path, error)) from error
    if not isinstance(value, dict):
        raise AISTRebaseError("{} must be a JSON object: {}".format(kind, path))
    return value


def _read_jsonl(path: Path, *, kind: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise AISTRebaseError("{} does not exist: {}".format(kind, path))
    result: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise AISTRebaseError("{} invalid JSON at line {}: {}".format(kind, number, error)) from error
            if not isinstance(value, dict):
                raise AISTRebaseError("{} row {} is not an object".format(kind, number))
            result.append(value)
    if not result:
        raise AISTRebaseError("{} is empty".format(kind))
    return result


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _require_string(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AISTRebaseError("{} lacks non-empty {}".format(context, field))
    return value


def _require_int(row: Mapping[str, Any], field: str, *, context: str, minimum: int = 0) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AISTRebaseError("{} requires integer {} >= {}".format(context, field, minimum))
    return int(value)


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.lower()) is None:
        raise AISTRebaseError("{} must be a SHA-256 hex digest".format(context))
    return value.lower()


def _resolve_bundle_asset(value: object, *, bundle: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AISTRebaseError("{} lacks a relative asset path".format(context))
    candidate = Path(value)
    if candidate.is_absolute():
        raise AISTRebaseError("{} must be bundle-relative, not absolute: {}".format(context, value))
    resolved = (bundle / candidate).resolve()
    try:
        resolved.relative_to(bundle)
    except ValueError as error:
        raise AISTRebaseError("{} escapes source bundle: {}".format(context, value)) from error
    if not resolved.is_file():
        raise AISTRebaseError("{} asset does not exist: {}".format(context, resolved))
    return resolved


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _safe_dir_name(recording_id: str) -> str:
    # Hash avoids unsafe legacy/source characters while retaining a traceable
    # human-prefix in the manifest rather than in the filesystem path.
    return hashlib.sha256(recording_id.encode("utf-8")).hexdigest()


def _performance_group(legacy_source_name: str) -> str:
    match = _CHANNEL_SUFFIX.fullmatch(legacy_source_name)
    if match is None or not match.group("performance"):
        raise AISTRebaseError(
            "legacy_source_name {!r} must end in '_chNN' to form a performance group".format(
                legacy_source_name
            )
        )
    # Preserve the conventional ``aistpp/`` namespace, while making the
    # identity itself exactly the legacy performance name without its camera
    # channel suffix.  This is the retrieval exclusion unit downstream.
    return "aistpp/{}".format(match.group("performance"))


def _stable_val_group(group_id: str, *, validation_fraction: float) -> bool:
    """Partition an old-train group without depending on row/order/hash seed."""
    bucket = int.from_bytes(hashlib.sha256(group_id.encode("utf-8")).digest()[:8], "big")
    return bucket < int(validation_fraction * (1 << 64))


def _validate_manifest_hashes(bundle: Path) -> Tuple[Dict[str, Any], Dict[str, str]]:
    build = _read_json(bundle / "build.json", kind="source build.json")
    if build.get("label_policy") != "legacy_window_labels_not_read_or_emitted":
        raise AISTRebaseError("source build is not explicitly label-free")
    reconstruction = build.get("reconstruction")
    if not isinstance(reconstruction, Mapping):
        raise AISTRebaseError("source build lacks reconstruction object")
    if reconstruction.get("motion_dim") != MOTION_DIM or reconstruction.get("music_dim") != MUSIC_DIM:
        raise AISTRebaseError("source build representation is not 151D motion / 35D music")
    window_length = reconstruction.get("window_length")
    if isinstance(window_length, bool) or not isinstance(window_length, int) or window_length < 1:
        raise AISTRebaseError("source build window_length must be a positive integer")
    names = ("sources.jsonl", "sequences.jsonl", "windows.jsonl")
    manifests = build.get("manifests")
    if not isinstance(manifests, Mapping):
        raise AISTRebaseError("source build lacks manifest hashes")
    hashes: Dict[str, str] = {}
    for name in names:
        expected = _require_sha256(manifests.get(name), context="source build manifests.{}".format(name))
        actual = _sha256_file(bundle / name)
        if actual != expected:
            raise AISTRebaseError("source build hash mismatch for {}".format(name))
        hashes[name] = actual
    return build, hashes


def _load_upstream_normalizer(
    path: Path, *, expected_sha256: Optional[str]
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], str]:
    if not path.is_file():
        raise AISTRebaseError("upstream normalizer does not exist: {}".format(path))
    sha256 = _sha256_file(path)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise AISTRebaseError("upstream normalizer hash does not match source build provenance")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise AISTRebaseError("cannot load upstream normalizer {}: {}".format(path, error)) from error
    if not isinstance(payload, Mapping) or set(payload) != {"data_min", "data_max", "training_names"}:
        raise AISTRebaseError("upstream normalizer must have exactly data_min, data_max, training_names")
    training_names = payload["training_names"]
    if (
        not isinstance(training_names, list)
        or not training_names
        or not all(isinstance(name, str) and name for name in training_names)
        or len(set(training_names)) != len(training_names)
    ):
        raise AISTRebaseError("upstream normalizer training_names must be a non-empty unique string list")
    values: List[np.ndarray] = []
    for key in ("data_min", "data_max"):
        tensor = payload[key]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or tuple(tensor.shape) != (MOTION_DIM,):
            raise AISTRebaseError("upstream normalizer {} must be tensor[{}]".format(key, MOTION_DIM))
        if not tensor.dtype.is_floating_point:
            raise AISTRebaseError("upstream normalizer {} must be floating".format(key))
        array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(array).all():
            raise AISTRebaseError("upstream normalizer {} contains non-finite values".format(key))
        values.append(array)
    data_min, data_max = values
    if np.any(data_max < data_min):
        raise AISTRebaseError("upstream normalizer data_max is below data_min")
    # The min/max tensor is only meaningful for the exact upstream train
    # sources that produced it.  Return the name roster as a first-class
    # provenance value rather than merely checking that it deserializes.
    return data_min, data_max, tuple(training_names), sha256


def _load_array(path: Path, *, shape_tail: Tuple[int, ...], context: str) -> np.ndarray:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise AISTRebaseError("cannot memory-map {}: {}".format(context, error)) from error
    if value.ndim != len(shape_tail) + 1 or tuple(value.shape[1:]) != shape_tail or len(value) < 1:
        raise AISTRebaseError("{} has shape {}; expected [T, {}]".format(context, tuple(value.shape), ", ".join(map(str, shape_tail))))
    if not np.issubdtype(value.dtype, np.floating):
        raise AISTRebaseError("{} must be floating, got {}".format(context, value.dtype))
    if not np.isfinite(value).all():
        raise AISTRebaseError("{} contains non-finite values".format(context))
    return value


def _validate_input(
    bundle: Path,
    *,
    normalizer_sha256: str,
    normalizer_training_names: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str], int]:
    build, hashes = _validate_manifest_hashes(bundle)
    window_length = int(build["reconstruction"]["window_length"])
    input_provenance = build.get("input")
    if not isinstance(input_provenance, Mapping):
        raise AISTRebaseError("source build lacks input provenance")
    expected_normalizer = _require_sha256(
        input_provenance.get("normalizer_sha256"), context="source build input.normalizer_sha256"
    )
    if expected_normalizer != normalizer_sha256:
        raise AISTRebaseError("input normalizer hash differs from source build input.normalizer_sha256")
    sources = _read_jsonl(bundle / "sources.jsonl", kind="sources.jsonl")
    sequences = _read_jsonl(bundle / "sequences.jsonl", kind="sequences.jsonl")
    windows = _read_jsonl(bundle / "windows.jsonl", kind="windows.jsonl")
    counts = build.get("counts")
    if not isinstance(counts, Mapping) or any(counts.get(key) != len(rows) for key, rows in (
        ("sources", sources), ("sequences", sequences), ("windows", windows)
    )):
        raise AISTRebaseError("source build counts do not match manifest rows")
    source_by_id: Dict[str, Mapping[str, Any]] = {}
    source_groups: Dict[str, str] = {}
    for index, source in enumerate(sources, 1):
        context = "source row {}".format(index)
        recording_id = _require_string(source, "recording_id", context=context)
        if recording_id in source_by_id:
            raise AISTRebaseError("duplicate source recording_id {!r}".format(recording_id))
        if source.get("schema_version") != "source-manifest-v1" or source.get("source_kind") != "aistpp":
            raise AISTRebaseError("{} is not an AIST label-free source-manifest-v1 row".format(context))
        if source.get("split") not in {"train", "test"}:
            raise AISTRebaseError("{} has unsupported legacy split".format(context))
        if source.get("duplicate_content_group_id") is not None:
            raise AISTRebaseError("{} must explicitly have no unverified duplicate content group".format(context))
        legacy_name = _require_string(source, "legacy_source_name", context=context)
        group = _performance_group(legacy_name)
        prior = source_groups.get(group)
        if prior is not None and prior != source["split"]:
            raise AISTRebaseError("performance group {!r} spans legacy train/test".format(group))
        source_groups[group] = source["split"]
        source_by_id[recording_id] = source
    expected_training_names = {
        _require_string(source, "legacy_source_name", context="source normalizer roster")
        for source in sources
        if source.get("split") == "train"
    }
    # ``training_names`` is part of the released min/max artifact.  A hash
    # match against a mutable manifest is necessary but not enough: reject a
    # normalizer whose stated train roster is not exactly the label-free
    # source bundle's legacy train roster.  Set equality is intentional here;
    # source order does not change min/max statistics, while duplicates were
    # rejected during normalizer loading.
    if set(normalizer_training_names) != expected_training_names:
        raise AISTRebaseError(
            "upstream normalizer training_names do not exactly match the source bundle legacy train roster"
        )
    sequence_by_id: Dict[str, Mapping[str, Any]] = {}
    sequence_assets: Dict[str, Tuple[Path, Path, Path]] = {}
    for index, sequence in enumerate(sequences, 1):
        context = "sequence row {}".format(index)
        sequence_id = _require_string(sequence, "sequence_id", context=context)
        if sequence_id in sequence_by_id:
            raise AISTRebaseError("duplicate sequence_id {!r}".format(sequence_id))
        recording_id = _require_string(sequence, "recording_id", context=context)
        source = source_by_id.get(recording_id)
        if source is None:
            raise AISTRebaseError("{} references absent source {!r}".format(context, recording_id))
        if sequence.get("split") != source.get("split") or sequence.get("retrieval_group_id") != recording_id:
            raise AISTRebaseError("{} does not preserve input source split/identity".format(context))
        if sequence.get("duplicate_content_group_id") is not None:
            raise AISTRebaseError("{} must explicitly have no unverified duplicate content group".format(context))
        frame_count = _require_int(sequence, "frame_count", context=context, minimum=1)
        if _require_int(sequence, "source_start_frame", context=context) != 0 or _require_int(
            sequence, "source_end_frame_exclusive", context=context, minimum=1
        ) != frame_count or sequence.get("is_contiguous") is not True:
            raise AISTRebaseError("{} has non-contiguous/incomplete source timeline".format(context))
        motion = _resolve_bundle_asset(sequence.get("motion_path"), bundle=bundle, context=context + " motion_path")
        music = _resolve_bundle_asset(sequence.get("music_path"), bundle=bundle, context=context + " music_path")
        frame_ids = _resolve_bundle_asset(sequence.get("frame_ids_path"), bundle=bundle, context=context + " frame_ids_path")
        if _sha256_file(motion) != _require_sha256(source.get("motion_sha256"), context=context + " source motion_sha256"):
            raise AISTRebaseError("{} motion asset hash does not match source".format(context))
        if _sha256_file(music) != _require_sha256(source.get("music_sha256"), context=context + " source music_sha256"):
            raise AISTRebaseError("{} music asset hash does not match source".format(context))
        if _sha256_file(frame_ids) != _require_sha256(sequence.get("frame_ids_sha256"), context=context + " frame_ids_sha256"):
            raise AISTRebaseError("{} frame-id asset hash does not match sequence".format(context))
        values = _load_array(motion, shape_tail=(MOTION_DIM,), context=context + " normalized motion")
        audio = _load_array(music, shape_tail=(MUSIC_DIM,), context=context + " music")
        try:
            frame_values = np.load(frame_ids, mmap_mode="r", allow_pickle=False)
        except Exception as error:
            raise AISTRebaseError("cannot memory-map {} frame ids: {}".format(context, error)) from error
        if frame_values.ndim != 1 or len(frame_values) != frame_count or not np.issubdtype(frame_values.dtype, np.integer):
            raise AISTRebaseError("{} frame ids do not match declared timeline".format(context))
        if not np.array_equal(frame_values, np.arange(frame_count, dtype=frame_values.dtype)):
            raise AISTRebaseError("{} frame ids are not the contiguous [0,T) timeline".format(context))
        if len(values) != frame_count or len(audio) != frame_count:
            raise AISTRebaseError("{} motion/music frame count does not match manifest".format(context))
        sequence_by_id[sequence_id] = sequence
        sequence_assets[sequence_id] = (motion, music, frame_ids)
    sequence_counts: Dict[str, int] = {}
    for sequence in sequences:
        sequence_counts[sequence["recording_id"]] = sequence_counts.get(sequence["recording_id"], 0) + 1
    if set(sequence_counts) != set(source_by_id) or any(count != 1 for count in sequence_counts.values()):
        raise AISTRebaseError("each AIST source must have exactly one source timeline sequence")
    seen_windows: set[str] = set()
    for index, window in enumerate(windows, 1):
        context = "window row {}".format(index)
        window_id = _require_string(window, "window_id", context=context)
        if window_id in seen_windows:
            raise AISTRebaseError("duplicate window_id {!r}".format(window_id))
        seen_windows.add(window_id)
        sequence = sequence_by_id.get(_require_string(window, "sequence_id", context=context))
        if sequence is None:
            raise AISTRebaseError("{} references absent sequence".format(context))
        if window.get("recording_id") != sequence.get("recording_id") or window.get("split") != sequence.get("split"):
            raise AISTRebaseError("{} does not agree with its sequence source/split".format(context))
        start = _require_int(window, "start_frame", context=context)
        end = _require_int(window, "end_frame_exclusive", context=context, minimum=1)
        if end <= start or end - start != window_length or end > sequence["frame_count"]:
            raise AISTRebaseError("{} does not describe a valid {}-frame source slice".format(context, window_length))
        # This reads only label *metadata* generated by source reconstruction;
        # labels.npy remains outside this tool's dependency graph.
        if window.get("label_space_id") is not None or window.get("label_state") != "unavailable_not_canonical":
            raise AISTRebaseError("{} is not explicitly label-free".format(context))
    return sources, sequences, windows, hashes, window_length


def _inverse_upstream_minmax(values: np.ndarray, data_min: np.ndarray, data_max: np.ndarray, *, context: str) -> np.ndarray:
    """Invert the released normalisation in float32 with no clipping/repair."""
    if np.any(values < np.float32(-1.0)) or np.any(values > np.float32(1.0)):
        raise AISTRebaseError("{} values are outside exact upstream normalized [-1, 1] range".format(context))
    safe_range = np.where(data_max == data_min, np.float32(1.0), data_max - data_min).astype(np.float32)
    constant = data_max == data_min
    if np.any(constant) and not np.array_equal(values[:, constant], np.full((len(values), int(np.sum(constant))), -1.0, dtype=values.dtype)):
        raise AISTRebaseError("{} violates constant-dimension upstream normalized value -1".format(context))
    raw = (np.asarray(values, dtype=np.float32) + np.float32(1.0)) * safe_range / np.float32(2.0) + data_min
    if not np.isfinite(raw).all():
        raise AISTRebaseError("{} inverse-minmax produced non-finite values".format(context))
    return raw.astype(np.float32, copy=False)


def _reserve_output(output_dir: Path) -> Tuple[Path, Path]:
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise AISTRebaseError("refusing to overwrite immutable output directory: {}".format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = output_dir.parent / ".{}.aist-rebase.lock".format(output_dir.name)
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise AISTRebaseError("another AIST rebase holds lock {}".format(lock)) from error
    try:
        if output_dir.exists() or output_dir.is_symlink():
            raise AISTRebaseError("refusing to overwrite immutable output directory: {}".format(output_dir))
        staging = Path(tempfile.mkdtemp(prefix=".{}.staging-".format(output_dir.name), dir=str(output_dir.parent)))
    except Exception:
        lock.rmdir()
        raise
    return staging, lock


def _publish(staging: Path, output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise AISTRebaseError("refusing to overwrite immutable output directory: {}".format(output_dir))
    os.rename(staging, output_dir)


def rebase_atomic_aist_source(
    source_bundle: Path,
    upstream_normalizer: Path,
    output_dir: Path,
    *,
    validation_fraction: float = DEFAULT_VAL_FRACTION,
) -> Dict[str, Any]:
    """Validate, inverse-normalize, source-split and atomically publish AIST.

    ``source_bundle`` must be a label-free ``source_manifest_v3``-style
    bundle.  ``output_dir`` must not exist.  The return value is exactly the
    immutable ``report.json`` content plus ``output_dir`` for caller ergonomics.
    """
    if not (0.0 < validation_fraction < 1.0):
        raise AISTRebaseError("validation_fraction must be strictly between 0 and 1")
    source_bundle = Path(source_bundle).resolve()
    output_dir = Path(output_dir).resolve()
    data_min, data_max, normalizer_training_names, normalizer_sha256 = _load_upstream_normalizer(
        Path(upstream_normalizer).resolve(), expected_sha256=None
    )
    sources, sequences, windows, input_hashes, input_window_length = _validate_input(
        source_bundle,
        normalizer_sha256=normalizer_sha256,
        normalizer_training_names=normalizer_training_names,
    )
    # Validate all source timelines and normalized coordinate ranges before a
    # lock/output is created.  Thus malformed input never leaves a partial
    # release directory behind.
    source_by_id = {row["recording_id"]: row for row in sources}
    group_split: Dict[str, str] = {}
    source_assignment: Dict[str, Tuple[str, str]] = {}
    for source in sources:
        legacy_split = source["split"]
        group = _performance_group(source["legacy_source_name"])
        split = "test" if legacy_split == "test" else ("val" if _stable_val_group(group, validation_fraction=validation_fraction) else "train")
        previous = group_split.setdefault(group, split)
        if previous != split:
            raise AISTRebaseError("deterministic split disagreement inside performance group {!r}".format(group))
        source_assignment[source["recording_id"]] = (group, split)
    assigned_group_counts = {split: sum(assigned == split for assigned in group_split.values()) for split in ("train", "val", "test")}
    if any(assigned_group_counts[split] == 0 for split in ("train", "val", "test")):
        raise AISTRebaseError(
            "frozen AIST release requires non-empty train/val/test performance groups; got {}".format(
                assigned_group_counts
            )
        )
    staging, lock = _reserve_output(output_dir)
    published = False
    try:
        sequence_root = staging / "sequences"
        sequence_root.mkdir()
        output_sources: List[Dict[str, Any]] = []
        output_sequences: List[Dict[str, Any]] = []
        output_windows: List[Dict[str, Any]] = []
        sequence_paths: Dict[str, Tuple[str, str, str, str, str, str]] = {}
        source_provenance: Dict[str, Dict[str, str]] = {}
        for sequence in sorted(sequences, key=lambda row: row["sequence_id"]):
            recording_id = sequence["recording_id"]
            source = source_by_id[recording_id]
            group, split = source_assignment[recording_id]
            input_motion = _resolve_bundle_asset(sequence["motion_path"], bundle=source_bundle, context=sequence["sequence_id"] + " motion")
            input_music = _resolve_bundle_asset(sequence["music_path"], bundle=source_bundle, context=sequence["sequence_id"] + " music")
            input_frame_ids = _resolve_bundle_asset(sequence["frame_ids_path"], bundle=source_bundle, context=sequence["sequence_id"] + " frame ids")
            motion = _load_array(input_motion, shape_tail=(MOTION_DIM,), context=sequence["sequence_id"] + " normalized motion")
            raw = _inverse_upstream_minmax(motion, data_min, data_max, context=sequence["sequence_id"])
            target = sequence_root / _safe_dir_name(recording_id)
            target.mkdir()
            raw_path = target / "motion_151_raw.npy"
            music_path = target / "music_35.npy"
            frame_ids_path = target / "frame_ids.npy"
            np.save(raw_path, raw, allow_pickle=False)
            # Copy immutable source assets byte-for-byte; source v3 hashes
            # have already been checked before this point.
            shutil.copyfile(input_music, music_path)
            shutil.copyfile(input_frame_ids, frame_ids_path)
            raw_sha256 = _sha256_file(raw_path)
            music_sha256 = _sha256_file(music_path)
            frame_ids_sha256 = _sha256_file(frame_ids_path)
            raw_relative = _relative(raw_path, staging)
            music_relative = _relative(music_path, staging)
            frame_relative = _relative(frame_ids_path, staging)
            source_provenance[recording_id] = {
                "raw_sha256": raw_sha256,
                "music_sha256": music_sha256,
                "frame_ids_sha256": frame_ids_sha256,
                "upstream_normalized_motion_sha256": _sha256_file(input_motion),
            }
            sequence_paths[sequence["sequence_id"]] = (
                raw_relative, raw_sha256, music_relative, music_sha256, frame_relative, frame_ids_sha256
            )
            output_sequences.append({
                "schema_version": SCHEMA_VERSION,
                "sequence_id": sequence["sequence_id"],
                "recording_id": recording_id,
                "retrieval_group_id": group,
                "duplicate_content_group_id": None,
                "split": split,
                "split_status": "frozen_performance_group_source_safe",
                "split_note": "old test performance groups retained as test; old train groups stably partitioned into train/val",
                "person_track_id": sequence["person_track_id"],
                "source_start_frame": 0,
                "source_end_frame_exclusive": sequence["frame_count"],
                "frame_count": sequence["frame_count"],
                "fps": sequence["fps"],
                "is_contiguous": True,
                "frame_ids_path": frame_relative,
                "frame_ids_sha256": frame_ids_sha256,
                "motion_path": raw_relative,
                "motion_sha256": raw_sha256,
                "music_path": music_relative,
                "music_sha256": music_sha256,
                "assets": {
                    "motion_151_raw": raw_relative,
                    "motion_151_raw_sha256": raw_sha256,
                    "music_35": music_relative,
                    "music_35_sha256": music_sha256,
                    "frame_ids": frame_relative,
                    "frame_ids_sha256": frame_ids_sha256,
                },
                "representation": dict(RAW_REPRESENTATION_CONTRACT),
                "normalization": {
                    "state": "raw",
                    "inverse_of": "upstream_minmax_normalized",
                    "upstream_normalizer_sha256": normalizer_sha256,
                    "formula": "raw=(normalized+1)*(data_max-data_min_or_1)/2+data_min",
                },
                "preprocess_version": REBASE_VERSION,
                "qc": {
                    "status": "passed_upstream_normalizer_inverse",
                    "accepted_for_training": True,
                    "input_manifest_validation": "passed",
                    "legacy_labels": "not_read_or_emitted",
                },
            })
        for source in sorted(sources, key=lambda row: row["recording_id"]):
            recording_id = source["recording_id"]
            group, split = source_assignment[recording_id]
            assets = source_provenance[recording_id]
            output_sources.append({
                "schema_version": SCHEMA_VERSION,
                "recording_id": recording_id,
                "retrieval_group_id": group,
                "duplicate_content_group_id": None,
                "split": split,
                "split_status": "frozen_performance_group_source_safe",
                "split_note": "old test performance groups retained as test; old train groups stably partitioned into train/val",
                "source_kind": "aistpp",
                "source_variant": "upstream_normalized_inverse_to_raw",
                "raw_uri": source["raw_uri"],
                "content_sha256": _combined_content_sha256(assets["raw_sha256"], assets["music_sha256"]),
                "motion_sha256": assets["raw_sha256"],
                "music_sha256": assets["music_sha256"],
                "upstream_normalized_motion_sha256": assets["upstream_normalized_motion_sha256"],
                "fps": source["fps"],
                "audio_id": source.get("audio_id"),
                "dancer_id": source.get("dancer_id"),
                "legacy_source_name": source["legacy_source_name"],
                "representation": dict(RAW_REPRESENTATION_CONTRACT),
                "provenance": {
                    "rebase_version": REBASE_VERSION,
                    "input_source_manifest_sha256": input_hashes["sources.jsonl"],
                    "input_sequence_manifest_sha256": input_hashes["sequences.jsonl"],
                    "upstream_normalizer_sha256": normalizer_sha256,
                    "legacy_labels": "intentionally_not_read_or_emitted",
                },
                "qc": {
                    "status": "passed_upstream_normalizer_inverse",
                    "accepted_for_training": True,
                    "input_manifest_validation": "passed",
                    "legacy_labels": "not_read_or_emitted",
                },
            })
        for window in sorted(windows, key=lambda row: row["window_id"]):
            sequence_id = window["sequence_id"]
            sequence = next(row for row in output_sequences if row["sequence_id"] == sequence_id)
            output_windows.append({
                "schema_version": SCHEMA_VERSION,
                "window_id": window["window_id"],
                "sequence_id": sequence_id,
                "recording_id": sequence["recording_id"],
                "retrieval_group_id": sequence["retrieval_group_id"],
                "duplicate_content_group_id": None,
                "split": sequence["split"],
                "split_status": sequence["split_status"],
                "split_note": sequence["split_note"],
                "start_frame": window["start_frame"],
                "end_frame_exclusive": window["end_frame_exclusive"],
                "length": input_window_length,
                "motion_path": sequence["motion_path"],
                "motion_sha256": sequence["motion_sha256"],
                "music_path": sequence["music_path"],
                "music_sha256": sequence["music_sha256"],
                "frame_ids_path": sequence["frame_ids_path"],
                "label_space_id": None,
                "label_valid_fraction": 0.0,
                "label_state": "unavailable_not_canonical",
                "builder_version": REBASE_VERSION,
                "qc": {"status": "passed_source_timeline_rebase", "accepted_for_training": True},
                "legacy_window": dict(window.get("legacy_window", {})),
            })
        _write_jsonl(staging / "sources.jsonl", output_sources)
        _write_jsonl(staging / "sequences.jsonl", output_sequences)
        _write_jsonl(staging / "windows.jsonl", output_windows)
        groups_by_split = {split: sorted(group for group, assigned in group_split.items() if assigned == split) for split in ("train", "val", "test")}
        report = {
            "schema_version": SCHEMA_VERSION,
            "rebase_version": REBASE_VERSION,
            "publication": "immutable_new_directory_only_atomic_rename",
            "label_policy": "legacy_labels_not_read_or_emitted",
            "input": {
                "source_bundle": str(source_bundle),
                "source_manifests": input_hashes,
                "upstream_normalizer": str(Path(upstream_normalizer).resolve()),
                "upstream_normalizer_sha256": normalizer_sha256,
            },
            "representation": dict(RAW_REPRESENTATION_CONTRACT),
            "inverse_normalization": "raw=(normalized+1)*(data_max-data_min_or_1)/2+data_min",
            "split_policy": {
                "unit": "legacy source name with trailing _chNN removed",
                "retrieval_group_id": "aistpp/<legacy_without_chNN>",
                "legacy_test": "retained_as_test",
                "legacy_train": "stable_sha256_group_partition",
                "validation_fraction": validation_fraction,
                "hash": "sha256(group_id) first 64 bits < fraction*2^64",
            },
            "counts": {
                "sources": len(output_sources), "sequences": len(output_sequences), "windows": len(output_windows),
                "groups": len(group_split),
                "sources_by_split": {split: sum(row["split"] == split for row in output_sources) for split in ("train", "val", "test")},
                "sequences_by_split": {split: sum(row["split"] == split for row in output_sequences) for split in ("train", "val", "test")},
                "groups_by_split": {split: len(groups_by_split[split]) for split in ("train", "val", "test")},
            },
            "groups": {split: {"ids_sha256": _sha256_strings(groups_by_split[split]), "count": len(groups_by_split[split])} for split in ("train", "val", "test")},
            "manifests": {name: _sha256_file(staging / name) for name in ("sources.jsonl", "sequences.jsonl", "windows.jsonl")},
        }
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        _publish(staging, output_dir)
        published = True
        result = dict(report)
        result["output_dir"] = str(output_dir)
        return result
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
        if lock.exists():
            lock.rmdir()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", required=True, help="label-free source_manifest_v3 directory")
    parser.add_argument("--upstream-normalizer", required=True, help="matching AtomicDance normalizer.pt")
    parser.add_argument("--output-dir", required=True, help="new immutable raw AIST release directory")
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    args = parser.parse_args(argv)
    try:
        report = rebase_atomic_aist_source(
            Path(args.source_bundle), Path(args.upstream_normalizer), Path(args.output_dir),
            validation_fraction=args.validation_fraction,
        )
    except AISTRebaseError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
