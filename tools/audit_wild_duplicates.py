#!/usr/bin/env python3
"""Audit exact raw-video byte duplicates in a wild source staging manifest.

This is intentionally a narrow, conservative audit.  It computes a streaming
SHA-256 of one safely resolved video file per ``recording_id`` and detects only
*byte-identical* files.  It is not perceptual/near-duplicate detection: a
re-encode, crop, trim, or metadata rewrite will generally have a different
hash.  The audit never changes a source manifest or its split assignments.

When ``--raw-video-root`` is supplied, files below it are indexed recursively
by exact filename stem and matched against ``source_recording_keys``.  More
than one match is deliberately reported as ambiguous rather than guessed.  If
there is no raw-root match, a single existing manifest ``source_videos`` entry
may be used as a clearly labelled fallback.  Multiple clip paths are likewise
ambiguous: hashing one crop would not establish recording-level equivalence.

Results are published as a new immutable directory containing JSONL and a
report.  Existing output directories are refused, so a failed or completed
audit cannot silently be replaced.

The CLI requires either a deterministic ``--max-recordings`` bound or an
explicit ``--full`` acknowledgement.  This prevents an exploratory invocation
from unintentionally hashing a multi-terabyte/large-video corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "atomic-wild-exact-content-audit-v1"
DEFAULT_CHUNK_BYTES = 1024 * 1024
VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


def _absolute_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _sha256_file(path: Path, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> str:
    """Hash one file without loading it into memory and reject mid-read mutation."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError("file changed while hashing")
    return digest.hexdigest()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _write_jsonl_new(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write a new JSONL artifact and fail rather than overwriting it."""
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, Sequence):
        return []
    result: List[str] = []
    seen = set()
    for item in value:
        if not isinstance(item, str) or not item:
            continue
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    unique: Dict[str, Path] = {}
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        unique[str(resolved)] = resolved
    return [unique[key] for key in sorted(unique)]


def index_raw_videos(raw_video_root: Path) -> Dict[str, List[Path]]:
    """Index supported video files by their exact filename stem, recursively."""
    if not raw_video_root.is_dir():
        raise ValueError("raw video root is not a directory: {}".format(raw_video_root))
    indexed: Dict[str, List[Path]] = defaultdict(list)
    for path in raw_video_root.rglob("*"):
        try:
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            indexed[path.stem].append(path.resolve())
        except OSError:
            # A concurrent deletion or unreadable entry is treated as absent;
            # the source record receives an explicit unavailable status later.
            continue
    return {stem: _unique_paths(paths) for stem, paths in indexed.items()}


def _source_fields(source: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str], List[str], List[str]]:
    recording_id = source.get("recording_id")
    split = source.get("split")
    assets = source.get("assets", {})
    if not isinstance(assets, Mapping):
        assets = {}
    return (
        recording_id if isinstance(recording_id, str) and recording_id else None,
        split if isinstance(split, str) and split else None,
        _string_list(source.get("source_recording_keys")),
        _string_list(assets.get("source_videos")),
    )


def _resolution_record(
    source: Mapping[str, Any],
    *,
    raw_video_index: Optional[Mapping[str, Sequence[Path]]],
) -> Dict[str, Any]:
    """Resolve one source to exactly one candidate video without guessing."""
    recording_id, split, recording_keys, source_videos = _source_fields(source)
    base: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "recording_id": recording_id,
        "retrieval_group_id": source.get("retrieval_group_id"),
        "split": split,
        "source_recording_keys": recording_keys,
        "source_videos": source_videos,
        "content_sha256": None,
        "duplicate_content_group_id": None,
        "audit_status": None,
        "resolution": {
            "method": None,
            "scope": None,
            "candidate_paths": [],
            "selected_path": None,
            "detail": None,
        },
    }
    if recording_id is None or split is None:
        base["audit_status"] = "invalid_manifest"
        base["resolution"]["detail"] = "source requires non-empty recording_id and split"
        return base
    if base["retrieval_group_id"] != recording_id:
        base["audit_status"] = "invalid_manifest"
        base["resolution"]["detail"] = (
            "staging contract requires retrieval_group_id to equal recording_id"
        )
        return base
    if not recording_keys:
        base["audit_status"] = "invalid_manifest"
        base["resolution"]["detail"] = "source requires source_recording_keys"
        return base

    if raw_video_index is not None:
        raw_candidates = _unique_paths(
            path for key in recording_keys for path in raw_video_index.get(key, ())
        )
        if len(raw_candidates) > 1:
            base["audit_status"] = "ambiguous"
            base["resolution"].update(
                {
                    "method": "raw_video_root_exact_stem",
                    "scope": "raw_recording_file",
                    "candidate_paths": [str(path) for path in raw_candidates],
                    "detail": "multiple raw-root files matched source_recording_keys; no file was chosen",
                }
            )
            return base
        if len(raw_candidates) == 1:
            selected = raw_candidates[0]
            base["resolution"].update(
                {
                    "method": "raw_video_root_exact_stem",
                    "scope": "raw_recording_file",
                    "candidate_paths": [str(selected)],
                    "selected_path": str(selected),
                    "detail": "one raw-root file matched exact source-recording-key stem",
                }
            )
            return base

    manifest_candidates = _unique_paths(
        Path(value) for value in source_videos if Path(value).expanduser().is_file()
    )
    if len(manifest_candidates) > 1:
        base["audit_status"] = "ambiguous"
        base["resolution"].update(
            {
                "method": "manifest_source_video_fallback",
                "scope": "manifest_source_video",
                "candidate_paths": [str(path) for path in manifest_candidates],
                "detail": "multiple existing manifest source_videos; no clip was chosen",
            }
        )
        return base
    if len(manifest_candidates) == 1:
        selected = manifest_candidates[0]
        base["resolution"].update(
            {
                "method": "manifest_source_video_fallback",
                "scope": "manifest_source_video",
                "candidate_paths": [str(selected)],
                "selected_path": str(selected),
                "detail": "one existing manifest source_video used because raw-root resolution was unavailable",
            }
        )
        return base

    base["audit_status"] = "unavailable"
    base["resolution"].update(
        {
            "method": "raw_video_root_exact_stem" if raw_video_index is not None else None,
            "scope": None,
            "candidate_paths": [],
            "detail": "no unique readable raw-root or manifest source video was available",
        }
    )
    return base


def audit_sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    raw_video_root: Optional[Path] = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    max_recordings: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Audit exact bytes for an immutable source-manifest snapshot.

    ``max_recordings`` is intentionally deterministic (sorted recording IDs)
    so a bounded smoke audit is reproducible and does not accidentally hash a
    large corpus during setup.
    """
    if max_recordings is not None and max_recordings < 1:
        raise ValueError("max_recordings must be at least 1 when supplied")
    raw_video_index = index_raw_videos(raw_video_root) if raw_video_root is not None else None

    unique_sources: Dict[str, Mapping[str, Any]] = {}
    for source in sources:
        recording_id = source.get("recording_id")
        if not isinstance(recording_id, str) or not recording_id:
            raise ValueError("source manifest contains a source without recording_id")
        if recording_id in unique_sources:
            raise ValueError("source manifest contains duplicate recording_id: {}".format(recording_id))
        unique_sources[recording_id] = source
    ordered_sources = [unique_sources[key] for key in sorted(unique_sources)]
    if max_recordings is not None:
        ordered_sources = ordered_sources[:max_recordings]

    records: List[Dict[str, Any]] = []
    for source in ordered_sources:
        record = _resolution_record(source, raw_video_index=raw_video_index)
        selected_path = record["resolution"]["selected_path"]
        if selected_path is not None:
            try:
                record["content_sha256"] = _sha256_file(Path(selected_path), chunk_bytes=chunk_bytes)
                record["audit_status"] = "hashed"
            except (OSError, RuntimeError) as error:
                record["audit_status"] = "unavailable"
                record["resolution"]["detail"] = "unable to hash selected path: {}".format(error)
                record["resolution"]["selected_path"] = None
        records.append(record)

    by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        digest = record["content_sha256"]
        if isinstance(digest, str):
            by_hash[digest].append(record)
    duplicate_groups: List[Dict[str, Any]] = []
    cross_split_collisions: List[Dict[str, Any]] = []
    for digest, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        group_id = "exact-sha256:{}".format(digest)
        recording_ids = sorted(str(member["recording_id"]) for member in members)
        splits = sorted({str(member["split"]) for member in members})
        for member in members:
            member["duplicate_content_group_id"] = group_id
        group = {
            "duplicate_content_group_id": group_id,
            "content_sha256": digest,
            "recording_ids": recording_ids,
            "splits": splits,
            "cross_split_collision": len(splits) > 1,
        }
        duplicate_groups.append(group)
        if len(splits) > 1:
            cross_split_collisions.append(group)

    status_counts = Counter(str(record["audit_status"]) for record in records)
    audit_complete = len(records) == len(sources)
    source_resolution_complete = status_counts["hashed"] == len(records)
    no_observed_cross_split_collision = not cross_split_collisions
    full_corpus_exact_hash_qc_passed = (
        audit_complete and source_resolution_complete and no_observed_cross_split_collision
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "audit_kind": "exact_byte_sha256_only",
        "records_in_input_manifest": len(sources),
        "records_audited": len(records),
        "max_recordings": max_recordings,
        "raw_video_root": str(raw_video_root) if raw_video_root is not None else None,
        "raw_video_index": {
            "enabled": raw_video_index is not None,
            "unique_stems": len(raw_video_index) if raw_video_index is not None else 0,
            "video_files": (
                sum(len(paths) for paths in raw_video_index.values()) if raw_video_index is not None else 0
            ),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "hashed_recordings": status_counts["hashed"],
        "unresolved_recordings": len(records) - status_counts["hashed"],
        "audit_complete": audit_complete,
        "source_resolution_complete": source_resolution_complete,
        "no_observed_cross_split_collision": no_observed_cross_split_collision,
        "full_corpus_exact_hash_qc_passed": full_corpus_exact_hash_qc_passed,
        "duplicate_content_qc_complete": False,
        "duplicate_content_qc_completion_reason": (
            "false by design: exact byte hashes cannot establish near-duplicate or semantic-content QC"
        ),
        "duplicate_content_groups": duplicate_groups,
        "cross_split_collision_errors": cross_split_collisions,
        "valid": full_corpus_exact_hash_qc_passed,
        "split_action": "none; source-manifest split assignments are reported, never overwritten or reassigned",
        "training_eligibility": "none; this audit never accepts data for training or changes a split",
        "limitations": [
            "Only byte-identical selected files share a duplicate_content_group_id.",
            "This is not near-duplicate, perceptual, re-encode, crop, trim, or semantic duplicate detection.",
            "A raw-video-root match requires an exact filename stem equal to source_recording_keys.",
            "Unresolved and ambiguous sources are intentionally not assigned a content hash or duplicate group.",
        ],
    }
    return records, report


def publish_audit_bundle(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    source_manifest: Path,
) -> Tuple[Path, Path]:
    """Publish an append-only audit bundle with stage-then-rename semantics."""
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            "audit output already exists and is immutable: {}; choose a new output directory".format(
                output_dir
            )
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".{}.staging-".format(output_dir.name), dir=str(output_dir.parent))
    )
    final_jsonl_path = output_dir / "duplicate_content_audit.jsonl"
    final_report_path = output_dir / "report.json"
    try:
        jsonl_path = staging / "duplicate_content_audit.jsonl"
        report_path = staging / "report.json"
        _write_jsonl_new(jsonl_path, records)
        final_report = dict(report)
        final_report.update(
            {
                "source_manifest": str(source_manifest.expanduser().resolve()),
                "source_manifest_sha256": _sha256_file(source_manifest),
                "audit_jsonl": str(final_jsonl_path),
                "audit_jsonl_sha256": _sha256_file(jsonl_path),
                "bundle_immutable": True,
                "publication": "staged_then_atomic_directory_rename",
            }
        )
        _write_json_new(report_path, final_report)
        # Do not call os.replace: an existing destination must never be
        # replaced.  The preflight and second check avoid ordinary accidental
        # overwrites; staging cleanup prevents a failed write from leaving a
        # destination that blocks a later immutable publication.
        if output_dir.exists():
            raise FileExistsError(
                "audit output already exists and is immutable: {}; choose a new output directory".format(
                    output_dir
                )
            )
        os.rename(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_jsonl_path, final_report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="pre-HMR wild sources.jsonl staging manifest")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new immutable audit bundle directory (refuses an existing destination)",
    )
    parser.add_argument(
        "--raw-video-root",
        default=None,
        help="optional root recursively indexed by exact raw recording-key filename stem",
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--max-recordings",
        type=int,
        help="deterministic bounded audit over sorted recording IDs",
    )
    scope.add_argument(
        "--full",
        action="store_true",
        help="explicitly acknowledge a full-manifest hash audit",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=DEFAULT_CHUNK_BYTES,
        help="streaming SHA-256 chunk size (default: %(default)s)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    source_manifest = _absolute_path(args.sources)
    output_dir = _absolute_path(args.output_dir)
    raw_video_root = _absolute_path(args.raw_video_root) if args.raw_video_root else None
    records, report = audit_sources(
        _load_jsonl(source_manifest),
        raw_video_root=raw_video_root,
        chunk_bytes=args.chunk_bytes,
        max_recordings=None if args.full else args.max_recordings,
    )
    jsonl_path, report_path = publish_audit_bundle(
        output_dir,
        records,
        report,
        source_manifest=source_manifest,
    )
    print(
        json.dumps(
            {
                "audit_jsonl": str(jsonl_path),
                "report": str(report_path),
                "records_audited": report["records_audited"],
                "audit_complete": report["audit_complete"],
                "full_corpus_exact_hash_qc_passed": report["full_corpus_exact_hash_qc_passed"],
                "duplicate_content_qc_complete": report["duplicate_content_qc_complete"],
                "no_observed_cross_split_collision": report["no_observed_cross_split_collision"],
                "cross_split_collision_errors": len(report["cross_split_collision_errors"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    # A bounded smoke audit is operationally successful when it observed no
    # collision, but the report marks it incomplete and never training-ready.
    return 0 if report["no_observed_cross_split_collision"] else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
