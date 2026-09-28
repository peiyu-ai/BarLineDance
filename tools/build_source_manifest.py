#!/usr/bin/env python3
"""Reconstruct continuous legacy AIST motion/music timelines safely.

The released AtomicDance package is indexed as heavily-overlapping
``<recording>_sliceN`` windows.  Its per-window atomic labels are known not to
agree in shared frames, so this builder intentionally never opens
``labels.npy`` and never emits ``labels.jsonl``.  It reconstructs only the
bitwise-consistent motion/music timeline behind each recording and publishes
an immutable source/sequence/window manifest bundle for the later,
train-only labelling pipeline.

Publication is all-or-nothing: an existing output directory is refused, work
is staged beside it, and the finished bundle is renamed into place only after
every source has passed shape, split, overlap, finite-value, and continuity
checks.  The builder therefore cannot silently turn a broken source into a
partial dataset or rewrite a frozen split manifest.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

import numpy as np


SCHEMA_VERSION = "source-manifest-v1"
BUILDER_VERSION = "aist-legacy-source-reconstruction-v1"
LEGACY_SPLIT_STATUS = "provisional_legacy_source_disjoint"
LEGACY_SPLIT_NOTE = (
    "Package train/test recording identities are disjoint, but legacy window labels are not "
    "canonical and this package is not an approved held-out benchmark."
)
_SLICE_NAME = re.compile(r"^(?P<source>.+)_slice(?P<index>\d+)$")


class ManifestBuildError(ValueError):
    """The indexed package cannot safely be promoted to source timelines."""


@dataclass(frozen=True)
class IndexedWindow:
    """One indexed input window, without trusting its label payload."""

    split: str
    source_name: str
    slice_index: int
    row_index: int
    original_name: str


@dataclass(frozen=True)
class IndexedSplit:
    """Memory-mapped motion/music arrays plus their parsed window identities."""

    split: str
    root: Path
    motion: np.ndarray
    music: np.ndarray
    windows_by_source: Mapping[str, Tuple[IndexedWindow, ...]]
    names_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_content_sha256(motion_sha256: str, music_sha256: str) -> str:
    """Hash the two reconstructed modalities without inventing a raw-video hash."""
    digest = hashlib.sha256()
    digest.update(b"AtomicDance legacy reconstructed source content v1\n")
    digest.update(b"motion:")
    digest.update(motion_sha256.encode("ascii"))
    digest.update(b"\nmusic:")
    digest.update(music_sha256.encode("ascii"))
    digest.update(b"\n")
    return digest.hexdigest()


def _read_names(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestBuildError("cannot read {}: {}".format(path, error)) from error
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ManifestBuildError("{} must contain a JSON list of strings".format(path))
    if len(values) != len(set(values)):
        raise ManifestBuildError("{} contains duplicate window names".format(path))
    return values


def _parse_split_names(split: str, names: Sequence[str]) -> Mapping[str, Tuple[IndexedWindow, ...]]:
    grouped: Dict[str, List[IndexedWindow]] = {}
    for row_index, original_name in enumerate(names):
        match = _SLICE_NAME.fullmatch(original_name)
        if match is None:
            raise ManifestBuildError(
                "{} names.json row {} is not a legacy '<recording>_sliceN' identity: {!r}".format(
                    split, row_index, original_name
                )
            )
        source_name = match.group("source")
        if not source_name:
            raise ManifestBuildError("{} names.json row {} has an empty recording identity".format(split, row_index))
        grouped.setdefault(source_name, []).append(
            IndexedWindow(
                split=split,
                source_name=source_name,
                slice_index=int(match.group("index")),
                row_index=row_index,
                original_name=original_name,
            )
        )

    result: Dict[str, Tuple[IndexedWindow, ...]] = {}
    for source_name, windows in grouped.items():
        ordered = tuple(sorted(windows, key=lambda window: window.slice_index))
        indices = [window.slice_index for window in ordered]
        if len(indices) != len(set(indices)):
            raise ManifestBuildError(
                "{} recording {!r} has duplicate legacy slice indices".format(split, source_name)
            )
        expected = list(range(len(indices)))
        if indices != expected:
            raise ManifestBuildError(
                "{} recording {!r} has non-contiguous slice indices {}; expected {}".format(
                    split, source_name, indices, expected
                )
            )
        result[source_name] = ordered
    return result


def _validate_split_name(split: str) -> str:
    if not split or split in {".", ".."} or Path(split).name != split:
        raise ManifestBuildError("invalid split name {!r}".format(split))
    return split


def _load_indexed_split(
    data_root: Path,
    split: str,
    *,
    window_length: int,
    motion_dim: int,
    music_dim: int,
) -> IndexedSplit:
    split = _validate_split_name(split)
    split_root = data_root / split
    motion_path = split_root / "motion.npy"
    music_path = split_root / "music.npy"
    names_path = split_root / "names.json"
    missing = [path for path in (motion_path, music_path, names_path) if not path.is_file()]
    if missing:
        raise ManifestBuildError(
            "{} is missing required motion/music/name assets: {}".format(
                split_root, ", ".join(str(path) for path in missing)
            )
        )
    try:
        motion = np.load(str(motion_path), mmap_mode="r", allow_pickle=False)
        music = np.load(str(music_path), mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ManifestBuildError("cannot memory-map {}: {}".format(split_root, error)) from error
    names = _read_names(names_path)
    expected_motion_shape = (len(names), window_length, motion_dim)
    expected_music_shape = (len(names), window_length, music_dim)
    if tuple(motion.shape) != expected_motion_shape:
        raise ManifestBuildError(
            "{} motion shape {} != expected {}".format(
                split_root, tuple(motion.shape), expected_motion_shape
            )
        )
    if tuple(music.shape) != expected_music_shape:
        raise ManifestBuildError(
            "{} music shape {} != expected {}".format(
                split_root, tuple(music.shape), expected_music_shape
            )
        )
    if not np.issubdtype(motion.dtype, np.floating) or not np.issubdtype(music.dtype, np.floating):
        raise ManifestBuildError(
            "{} motion/music must be floating arrays, got {}/{}".format(
                split_root, motion.dtype, music.dtype
            )
        )
    return IndexedSplit(
        split=split,
        root=split_root,
        motion=motion,
        music=music,
        windows_by_source=_parse_split_names(split, names),
        names_sha256=_sha256_file(names_path),
    )


def _check_source_split_safety(indexed_splits: Iterable[IndexedSplit]) -> None:
    ownership: Dict[str, List[str]] = {}
    for indexed_split in indexed_splits:
        for source_name in indexed_split.windows_by_source:
            ownership.setdefault(source_name, []).append(indexed_split.split)
    collisions = {
        source_name: sorted(splits)
        for source_name, splits in ownership.items()
        if len(splits) > 1
    }
    if collisions:
        preview = ", ".join(
            "{} in {}".format(source_name, "/".join(splits))
            for source_name, splits in sorted(collisions.items())[:8]
        )
        raise ManifestBuildError(
            "recording-level split leakage: {} source(s) appear in multiple input splits ({})".format(
                len(collisions), preview
            )
        )


def _reconstruct_source(
    indexed_split: IndexedSplit,
    source_name: str,
    windows: Sequence[IndexedWindow],
    *,
    window_length: int,
    window_stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stitch one source while requiring exact agreement in every overlap."""
    if not windows:
        raise ManifestBuildError("{} recording {!r} has no windows".format(indexed_split.split, source_name))
    frame_count = window_length + (len(windows) - 1) * window_stride
    motion = np.empty((frame_count, indexed_split.motion.shape[-1]), dtype=indexed_split.motion.dtype)
    music = np.empty((frame_count, indexed_split.music.shape[-1]), dtype=indexed_split.music.dtype)
    occupied = np.zeros(frame_count, dtype=bool)
    for window in windows:
        start = window.slice_index * window_stride
        end = start + window_length
        selection = slice(start, end)
        source_motion = np.asarray(indexed_split.motion[window.row_index])
        source_music = np.asarray(indexed_split.music[window.row_index])
        overlapping = occupied[selection]
        if np.any(overlapping):
            if not np.array_equal(motion[selection][overlapping], source_motion[overlapping]):
                raise ManifestBuildError(
                    "{} recording {!r} has inconsistent motion values in slice {} overlap".format(
                        indexed_split.split, source_name, window.slice_index
                    )
                )
            if not np.array_equal(music[selection][overlapping], source_music[overlapping]):
                raise ManifestBuildError(
                    "{} recording {!r} has inconsistent music values in slice {} overlap".format(
                        indexed_split.split, source_name, window.slice_index
                    )
                )
        write_mask = ~overlapping
        if np.any(write_mask):
            destination_motion = motion[selection]
            destination_music = music[selection]
            destination_motion[write_mask] = source_motion[write_mask]
            destination_music[write_mask] = source_music[write_mask]
        occupied[selection] = True
    if not bool(np.all(occupied)):
        raise ManifestBuildError(
            "{} recording {!r} reconstructed with frame gaps".format(indexed_split.split, source_name)
        )
    if not bool(np.isfinite(motion).all()) or not bool(np.isfinite(music).all()):
        raise ManifestBuildError(
            "{} recording {!r} contains non-finite motion/music values".format(
                indexed_split.split, source_name
            )
        )
    return motion, music


def _safe_sequence_directory(recording_id: str) -> str:
    # Source IDs are user/data supplied.  Keep generated paths independent of
    # their spelling, while retaining the full identity in the JSONL rows.
    return hashlib.sha256(recording_id.encode("utf-8")).hexdigest()


def _recording_id(source_name: str) -> str:
    return "aistpp/{}".format(source_name)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _package_provenance(data_root: Path, indexed_splits: Sequence[IndexedSplit]) -> Dict[str, Any]:
    manifest_path = data_root / "manifest.json"
    normalizer_path = data_root / "normalizer.pt"
    result: Dict[str, Any] = {
        "dataset_root": str(data_root),
        "dataset_manifest_sha256": _sha256_file(manifest_path) if manifest_path.is_file() else None,
        "normalizer_sha256": _sha256_file(normalizer_path) if normalizer_path.is_file() else None,
        "input_names_sha256": {
            indexed_split.split: indexed_split.names_sha256 for indexed_split in indexed_splits
        },
    }
    return result


def _ensure_new_output(output_dir: Path) -> Tuple[Path, Path]:
    """Reserve a target name and return an isolated staging directory + lock."""
    if output_dir.exists() or output_dir.is_symlink():
        raise ManifestBuildError(
            "refusing to overwrite existing immutable manifest bundle: {}".format(output_dir)
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = output_dir.parent / ".{}.source-manifest.lock".format(output_dir.name)
    try:
        lock_dir.mkdir()
    except FileExistsError as error:
        raise ManifestBuildError(
            "another manifest build holds lock {}; retry after it completes".format(lock_dir)
        ) from error
    try:
        if output_dir.exists() or output_dir.is_symlink():
            raise ManifestBuildError(
                "refusing to overwrite existing immutable manifest bundle: {}".format(output_dir)
            )
        staging = Path(tempfile.mkdtemp(prefix=".{}.staging-".format(output_dir.name), dir=str(output_dir.parent)))
    except Exception:
        lock_dir.rmdir()
        raise
    return staging, lock_dir


def _publish_staging(staging: Path, output_dir: Path) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise ManifestBuildError(
            "refusing to overwrite existing immutable manifest bundle: {}".format(output_dir)
        )
    # Both paths share a parent, making this POSIX rename atomic.
    os.rename(str(staging), str(output_dir))


def build_source_manifest(
    data_root: Path,
    output_dir: Path,
    *,
    splits: Sequence[str] = ("train", "test"),
    window_length: int = 150,
    window_stride: int = 15,
    motion_dim: int = 151,
    music_dim: int = 35,
    fps: int = 30,
) -> Dict[str, Any]:
    """Build and atomically publish a label-free source manifest bundle.

    The returned dictionary is the exact content of ``build.json`` plus the
    final output path.  ``labels.npy`` is deliberately neither a dependency
    nor an output of this operation.
    """
    if window_length < 1:
        raise ManifestBuildError("window_length must be positive")
    if window_stride < 1 or window_stride > window_length:
        raise ManifestBuildError("window_stride must be in [1, window_length]")
    if motion_dim < 1 or music_dim < 1 or fps < 1:
        raise ManifestBuildError("motion_dim, music_dim, and fps must be positive")
    normalized_splits = tuple(_validate_split_name(split) for split in splits)
    if not normalized_splits:
        raise ManifestBuildError("at least one input split is required")
    if len(normalized_splits) != len(set(normalized_splits)):
        raise ManifestBuildError("input split names must be unique")
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    indexed_splits = [
        _load_indexed_split(
            data_root,
            split,
            window_length=window_length,
            motion_dim=motion_dim,
            music_dim=music_dim,
        )
        for split in normalized_splits
    ]
    _check_source_split_safety(indexed_splits)
    provenance = _package_provenance(data_root, indexed_splits)
    staging, lock_dir = _ensure_new_output(output_dir)
    published = False
    try:
        sequence_root = staging / "sequences"
        sequence_root.mkdir()
        sources: List[Dict[str, Any]] = []
        sequences: List[Dict[str, Any]] = []
        windows_rows: List[Dict[str, Any]] = []
        for indexed_split in indexed_splits:
            for source_name in sorted(indexed_split.windows_by_source):
                source_windows = indexed_split.windows_by_source[source_name]
                motion, music = _reconstruct_source(
                    indexed_split,
                    source_name,
                    source_windows,
                    window_length=window_length,
                    window_stride=window_stride,
                )
                recording_id = _recording_id(source_name)
                sequence_id = "{}/sequence0".format(recording_id)
                sequence_path = sequence_root / _safe_sequence_directory(recording_id)
                sequence_path.mkdir()
                motion_path = sequence_path / "motion.npy"
                music_path = sequence_path / "music.npy"
                frame_ids_path = sequence_path / "frame_ids.npy"
                np.save(str(motion_path), motion, allow_pickle=False)
                np.save(str(music_path), music, allow_pickle=False)
                np.save(str(frame_ids_path), np.arange(len(motion), dtype=np.int64), allow_pickle=False)
                motion_sha256 = _sha256_file(motion_path)
                music_sha256 = _sha256_file(music_path)
                frame_ids_sha256 = _sha256_file(frame_ids_path)
                content_sha256 = _combined_content_sha256(motion_sha256, music_sha256)
                motion_relative = _relative_path(motion_path, staging)
                music_relative = _relative_path(music_path, staging)
                frame_ids_relative = _relative_path(frame_ids_path, staging)
                frame_count = int(len(motion))
                sources.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "recording_id": recording_id,
                        "retrieval_group_id": recording_id,
                        "duplicate_content_group_id": None,
                        "split": indexed_split.split,
                        "split_status": LEGACY_SPLIT_STATUS,
                        "split_note": LEGACY_SPLIT_NOTE,
                        "source_kind": "aistpp",
                        "source_variant": "legacy_reconstructed",
                        "raw_uri": "atomic_aistpp://{}/{}".format(
                            indexed_split.split, quote(source_name, safe="")
                        ),
                        "content_sha256": content_sha256,
                        "motion_sha256": motion_sha256,
                        "music_sha256": music_sha256,
                        "fps": fps,
                        "audio_id": None,
                        "dancer_id": None,
                        "legacy_source_name": source_name,
                        "provenance": {
                            "builder_version": BUILDER_VERSION,
                            "legacy_window_length": window_length,
                            "legacy_window_stride": window_stride,
                            "legacy_window_count": len(source_windows),
                            "legacy_labels": "intentionally_not_read_or_emitted",
                            "input_names_sha256": indexed_split.names_sha256,
                        },
                    }
                )
                sequences.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sequence_id": sequence_id,
                        "recording_id": recording_id,
                        "retrieval_group_id": recording_id,
                        "duplicate_content_group_id": None,
                        "split": indexed_split.split,
                        "split_status": LEGACY_SPLIT_STATUS,
                        "split_note": LEGACY_SPLIT_NOTE,
                        "person_track_id": "legacy_single_track",
                        "source_start_frame": 0,
                        "source_end_frame_exclusive": frame_count,
                        "frame_count": frame_count,
                        "fps": fps,
                        "is_contiguous": True,
                        "frame_ids_path": frame_ids_relative,
                        "frame_ids_sha256": frame_ids_sha256,
                        "motion_path": motion_relative,
                        "music_path": music_relative,
                        "coordinate_system": "AtomicDance released 151D motion; values retained exactly from upstream windows",
                        "preprocess_version": BUILDER_VERSION,
                        "qc_status": "passed_exact_overlap_reconstruction",
                    }
                )
                for window in source_windows:
                    start = window.slice_index * window_stride
                    end = start + window_length
                    windows_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "window_id": "{}/window{:06d}".format(sequence_id, window.slice_index),
                            "sequence_id": sequence_id,
                            "recording_id": recording_id,
                            "retrieval_group_id": recording_id,
                            "duplicate_content_group_id": None,
                            "split": indexed_split.split,
                            "split_status": LEGACY_SPLIT_STATUS,
                            "split_note": LEGACY_SPLIT_NOTE,
                            "start_frame": start,
                            "end_frame_exclusive": end,
                            "length": window_length,
                            "motion_path": motion_relative,
                            "music_path": music_relative,
                            "label_space_id": None,
                            "label_valid_fraction": 0.0,
                            "label_state": "unavailable_not_canonical",
                            "builder_version": BUILDER_VERSION,
                            "legacy_window": {
                                "input_split": indexed_split.split,
                                "input_row_index": window.row_index,
                                "original_name": window.original_name,
                                "slice_index": window.slice_index,
                            },
                        }
                    )
        sources.sort(key=lambda row: row["recording_id"])
        sequences.sort(key=lambda row: row["sequence_id"])
        windows_rows.sort(key=lambda row: row["window_id"])
        _write_jsonl(staging / "sources.jsonl", sources)
        _write_jsonl(staging / "sequences.jsonl", sequences)
        _write_jsonl(staging / "windows.jsonl", windows_rows)
        build = {
            "schema_version": "source-manifest-build-v1",
            "builder_version": BUILDER_VERSION,
            "publication": "immutable_new_directory_only",
            "label_policy": "legacy_window_labels_not_read_or_emitted",
            "input": provenance,
            "reconstruction": {
                "input_splits": list(normalized_splits),
                "window_length": window_length,
                "window_stride": window_stride,
                "motion_dim": motion_dim,
                "music_dim": music_dim,
                "fps": fps,
                "overlap_policy": "bitwise_exact",
                "source_split_policy": "recording_id_must_belong_to_exactly_one_split",
            },
            "counts": {
                "sources": len(sources),
                "sequences": len(sequences),
                "windows": len(windows_rows),
                "labels": 0,
            },
            "manifests": {
                name: _sha256_file(staging / name)
                for name in ("sources.jsonl", "sequences.jsonl", "windows.jsonl")
            },
        }
        build_path = staging / "build.json"
        build_path.write_text(
            json.dumps(build, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
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


def _parse_splits(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/atomic_aistpp")
    parser.add_argument(
        "--output-dir",
        default="data/atomic_aistpp/source_manifest_v1",
        help="new destination directory; existing directories are never overwritten",
    )
    parser.add_argument("--splits", default="train,test", help="comma-separated indexed splits")
    parser.add_argument("--window-length", type=int, default=150)
    parser.add_argument("--window-stride", type=int, default=15)
    parser.add_argument("--motion-dim", type=int, default=151)
    parser.add_argument("--music-dim", type=int, default=35)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        report = build_source_manifest(
            Path(args.data_root),
            Path(args.output_dir),
            splits=_parse_splits(args.splits),
            window_length=args.window_length,
            window_stride=args.window_stride,
            motion_dim=args.motion_dim,
            music_dim=args.music_dim,
            fps=args.fps,
        )
    except ManifestBuildError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
