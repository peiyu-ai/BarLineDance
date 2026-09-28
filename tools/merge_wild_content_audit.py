#!/usr/bin/env python3
"""Merge a completed exact-content audit into an immutable wild manifest bundle.

``audit_wild_duplicates.py`` deliberately publishes an audit beside, rather
than inside, the pre-HMR staging manifest.  This command is the only bridge
between those two artifacts.  It is intentionally stricter than a generic
JSON join:

* the audit report must bind byte-for-byte to the supplied ``sources.jsonl``;
* every source must have exactly one successful, exact-SHA256 audit row;
* duplicate groups may exist, but every such group must stay in one split;
* source and sequence identities, source ownership, retrieval groups and
  splits are checked before any row is copied; and
* the output is a new, staged-then-renamed directory.  Neither input manifest
  is ever edited or replaced.

An exact raw-file hash catches byte-identical uploads only.  It cannot prove
that re-encodes, trims, crops, or semantically similar videos are distinct.
Consequently this merger *always* leaves the split provisional pending
near-duplicate QC and forces ``qc.accepted_for_training`` to remain false.
It is not a training-release gate.
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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "atomic-wild-content-audit-merge-v1"
AUDIT_SCHEMA_VERSION = "atomic-wild-exact-content-audit-v1"
AUDIT_KIND = "exact_byte_sha256_only"
ALLOWED_SPLITS = frozenset(("train", "val", "test"))
PROVISIONAL_SPLIT_STATUS = "provisional_pending_near_duplicate_content_qc"
ALLOWED_INPUT_SPLIT_STATUSES = frozenset(
    (
        "provisional_pending_duplicate_content_qc",
        PROVISIONAL_SPLIT_STATUS,
    )
)
HASH_STATUS = "exact_raw_recording_sha256_audited"


class ContentAuditMergeError(ValueError):
    """Input artifacts cannot safely be joined into a derived manifest."""


def _absolute_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _output_path(value: str) -> Path:
    """Make an output path absolute without resolving away a symlink target."""
    return Path(value).expanduser().absolute()


def _path_lexists(path: Path) -> bool:
    """Treat a dangling symlink as an occupied immutable destination too."""
    return os.path.lexists(str(path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("JSON file does not exist: {}".format(path))
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ContentAuditMergeError("cannot read JSON {}: {}".format(path, error)) from error
    if not isinstance(value, Mapping):
        raise ContentAuditMergeError("JSON root is not an object: {}".format(path))
    return dict(value)


def _read_jsonl(path: Path, *, artifact_name: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("{} does not exist: {}".format(artifact_name, path))
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ContentAuditMergeError(
                        "invalid JSONL in {} at line {}: {}".format(path, line_number, error)
                    ) from error
                if not isinstance(value, Mapping):
                    raise ContentAuditMergeError(
                        "{} line {} is not an object".format(artifact_name, line_number)
                    )
                records.append(dict(value))
    except OSError as error:
        raise ContentAuditMergeError("cannot read {}: {}".format(path, error)) from error
    if not records:
        raise ContentAuditMergeError("{} is empty: {}".format(artifact_name, path))
    return records


def _write_jsonl_new(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_copy(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy a JSON object without retaining nested mutable input references."""
    return json.loads(json.dumps(value))


def _require_string(record: Mapping[str, Any], field: str, *, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContentAuditMergeError("{} lacks a non-empty {}".format(context, field))
    return value


def _require_split(record: Mapping[str, Any], *, context: str) -> str:
    split = _require_string(record, "split", context=context)
    if split not in ALLOWED_SPLITS:
        raise ContentAuditMergeError(
            "{} has unsupported split {!r}; expected one of {}".format(
                context, split, ", ".join(sorted(ALLOWED_SPLITS))
            )
        )
    return split


def _require_boolean(record: Mapping[str, Any], field: str, *, context: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise ContentAuditMergeError("{} requires boolean {}".format(context, field))
    return value


def _require_false(record: Mapping[str, Any], field: str, *, context: str) -> None:
    if _require_boolean(record, field, context=context) is not False:
        raise ContentAuditMergeError("{} must keep {} false".format(context, field))


def _optional_group_id(record: Mapping[str, Any], field: str, *, context: str) -> Optional[str]:
    if field not in record or record[field] is None:
        return None
    value = record[field]
    if not isinstance(value, str) or not value:
        raise ContentAuditMergeError(
            "{} has invalid {}; expected null or non-empty string".format(context, field)
        )
    return value


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContentAuditMergeError("{} requires a 64-character SHA-256 string".format(context))
    if any(character not in "0123456789abcdef" for character in value):
        raise ContentAuditMergeError("{} requires lowercase hexadecimal SHA-256".format(context))
    return value


def _require_mapping(
    record: Mapping[str, Any], field: str, *, context: str, allow_missing: bool = False
) -> Dict[str, Any]:
    if field not in record or record[field] is None:
        if allow_missing:
            return {}
        raise ContentAuditMergeError("{} lacks object {}".format(context, field))
    value = record[field]
    if not isinstance(value, Mapping):
        raise ContentAuditMergeError("{} has non-object {}".format(context, field))
    return dict(value)


def _require_string_list(record: Mapping[str, Any], field: str, *, context: str) -> List[str]:
    value = record.get(field)
    if not isinstance(value, list) or not value:
        raise ContentAuditMergeError("{} requires non-empty list {}".format(context, field))
    if not all(isinstance(item, str) and item for item in value):
        raise ContentAuditMergeError("{} has invalid {} entries".format(context, field))
    if len(value) != len(set(value)):
        raise ContentAuditMergeError("{} has duplicate {} entries".format(context, field))
    return list(value)


def _validate_pre_merge_status(record: Mapping[str, Any], *, context: str) -> None:
    """Refuse rows that were already (incorrectly) marked training-ready."""
    split_status = _require_string(record, "split_status", context=context)
    if split_status not in ALLOWED_INPUT_SPLIT_STATUSES:
        raise ContentAuditMergeError(
            "{} split_status {!r} is not an allowed provisional pre-near-duplicate status".format(
                context, split_status
            )
        )
    qc = _require_mapping(record, "qc", context=context)
    _require_false(qc, "accepted_for_training", context="{}.qc".format(context))


def _check_existing_hash(value: Any, digest: str, *, context: str) -> None:
    """Allow an absent/null or matching pre-existing digest, never overwrite a conflict."""
    if value is None:
        return
    existing = _require_sha256(value, context=context)
    if existing != digest:
        raise ContentAuditMergeError(
            "{} conflicts with completed audit hash {}".format(context, digest)
        )


def _check_existing_group(value: Any, group_id: Optional[str], *, context: str) -> None:
    if value is None:
        return
    existing = _optional_group_id({"value": value}, "value", context=context)
    if existing != group_id:
        raise ContentAuditMergeError(
            "{} conflicts with completed audit duplicate_content_group_id {!r}".format(
                context, group_id
            )
        )


def _validate_report(
    report: Mapping[str, Any],
    *,
    source_manifest: Path,
    audit_records: Path,
    source_count: int,
) -> None:
    """Check that the audit bundle is complete and binds to these exact inputs."""
    context = "audit report"
    if report.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ContentAuditMergeError(
            "{} schema_version must be {!r}".format(context, AUDIT_SCHEMA_VERSION)
        )
    if report.get("audit_kind") != AUDIT_KIND:
        raise ContentAuditMergeError("{} is not an exact-byte SHA-256 audit".format(context))
    for field in (
        "bundle_immutable",
        "audit_complete",
        "source_resolution_complete",
        "no_observed_cross_split_collision",
        "full_corpus_exact_hash_qc_passed",
        "valid",
    ):
        if _require_boolean(report, field, context=context) is not True:
            raise ContentAuditMergeError("{} must have {} true".format(context, field))
    # An exact byte audit is deliberately not a general duplicate-content
    # audit.  Refuse a malformed report that would try to claim otherwise.
    _require_false(report, "duplicate_content_qc_complete", context=context)
    if report.get("records_in_input_manifest") != source_count:
        raise ContentAuditMergeError(
            "audit report records_in_input_manifest does not match supplied sources.jsonl"
        )
    if report.get("records_audited") != source_count:
        raise ContentAuditMergeError(
            "audit report records_audited does not match supplied sources.jsonl"
        )
    if report.get("hashed_recordings") != source_count:
        raise ContentAuditMergeError(
            "audit report hashed_recordings does not match supplied sources.jsonl"
        )
    if report.get("unresolved_recordings") != 0:
        raise ContentAuditMergeError("audit report has unresolved recordings")
    if report.get("status_counts") != {"hashed": source_count}:
        raise ContentAuditMergeError("audit report status_counts must contain only every source as hashed")
    if report.get("cross_split_collision_errors") != []:
        raise ContentAuditMergeError("audit report contains cross-split collision errors")

    expected_source_hash = _sha256_file(source_manifest)
    reported_source_hash = _require_sha256(
        report.get("source_manifest_sha256"), context="audit report source_manifest_sha256"
    )
    if reported_source_hash != expected_source_hash:
        raise ContentAuditMergeError(
            "audit report source_manifest_sha256 does not bind to supplied sources.jsonl"
        )
    expected_audit_hash = _sha256_file(audit_records)
    reported_audit_hash = _require_sha256(
        report.get("audit_jsonl_sha256"), context="audit report audit_jsonl_sha256"
    )
    if reported_audit_hash != expected_audit_hash:
        raise ContentAuditMergeError(
            "audit report audit_jsonl_sha256 does not bind to supplied duplicate_content_audit.jsonl"
        )


def _validate_sources(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    sources: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records, 1):
        context = "source row {}".format(index)
        recording_id = _require_string(record, "recording_id", context=context)
        retrieval_group_id = _require_string(record, "retrieval_group_id", context=context)
        if retrieval_group_id != recording_id:
            raise ContentAuditMergeError(
                "{} retrieval_group_id must equal recording_id for the wild staging contract".format(
                    context
                )
            )
        _require_split(record, context=context)
        _require_string_list(record, "sequence_ids", context=context)
        _optional_group_id(record, "duplicate_content_group_id", context=context)
        _validate_pre_merge_status(record, context=context)
        assets = _require_mapping(record, "assets", context=context, allow_missing=True)
        if "content_sha256" in assets and assets["content_sha256"] is not None:
            _require_sha256(assets["content_sha256"], context="{}.assets.content_sha256".format(context))
        if "content_sha256" in record and record["content_sha256"] is not None:
            _require_sha256(record["content_sha256"], context="{}.content_sha256".format(context))
        if recording_id in sources:
            raise ContentAuditMergeError("sources.jsonl has duplicate recording_id {!r}".format(recording_id))
        sources[recording_id] = dict(record)
    return sources


def _validate_sequences(
    records: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    sequences: Dict[str, Dict[str, Any]] = {}
    actual_source_sequences: Dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records, 1):
        context = "sequence row {}".format(index)
        sequence_id = _require_string(record, "sequence_id", context=context)
        recording_id = _require_string(record, "recording_id", context=context)
        source = sources.get(recording_id)
        if source is None:
            raise ContentAuditMergeError(
                "{} references recording_id {!r} absent from sources.jsonl".format(context, recording_id)
            )
        retrieval_group_id = _require_string(record, "retrieval_group_id", context=context)
        source_retrieval_group = _require_string(
            source, "retrieval_group_id", context="source {!r}".format(recording_id)
        )
        if retrieval_group_id != source_retrieval_group:
            raise ContentAuditMergeError(
                "{} retrieval_group_id does not match its source".format(context)
            )
        split = _require_split(record, context=context)
        source_split = _require_split(source, context="source {!r}".format(recording_id))
        if split != source_split:
            raise ContentAuditMergeError("{} split does not match its source".format(context))
        sequence_group = _optional_group_id(record, "duplicate_content_group_id", context=context)
        source_group = _optional_group_id(
            source, "duplicate_content_group_id", context="source {!r}".format(recording_id)
        )
        if sequence_group != source_group:
            raise ContentAuditMergeError(
                "{} duplicate_content_group_id does not match its source".format(context)
            )
        _validate_pre_merge_status(record, context=context)
        assets = _require_mapping(record, "assets", context=context, allow_missing=True)
        if "content_sha256" in assets and assets["content_sha256"] is not None:
            _require_sha256(assets["content_sha256"], context="{}.assets.content_sha256".format(context))
        if "content_sha256" in record and record["content_sha256"] is not None:
            _require_sha256(record["content_sha256"], context="{}.content_sha256".format(context))
        source_digest = source.get("content_sha256")
        sequence_digest = record.get("content_sha256")
        if source_digest is not None and sequence_digest is not None and source_digest != sequence_digest:
            raise ContentAuditMergeError("{} content_sha256 does not match its source".format(context))
        if sequence_id in sequences:
            raise ContentAuditMergeError("sequences.jsonl has duplicate sequence_id {!r}".format(sequence_id))
        sequences[sequence_id] = dict(record)
        actual_source_sequences[recording_id].add(sequence_id)

    for recording_id, source in sources.items():
        expected = set(_require_string_list(source, "sequence_ids", context="source {!r}".format(recording_id)))
        actual = actual_source_sequences.get(recording_id, set())
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ContentAuditMergeError(
                "source {!r} sequence_ids do not exactly match sequences.jsonl (missing={}, extra={})".format(
                    recording_id, missing, extra
                )
            )
    return sequences


def _derive_duplicate_groups(
    audit_by_recording: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    by_hash: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in audit_by_recording.values():
        by_hash[str(record["content_sha256"])].append(record)

    groups: List[Dict[str, Any]] = []
    for digest, members in sorted(by_hash.items()):
        member_ids = sorted(str(member["recording_id"]) for member in members)
        group_ids = {member.get("duplicate_content_group_id") for member in members}
        splits = sorted({str(member["split"]) for member in members})
        if len(member_ids) == 1:
            if group_ids != {None}:
                raise ContentAuditMergeError(
                    "audit singleton {} must not have duplicate_content_group_id".format(member_ids[0])
                )
            continue
        expected_group_id = "exact-sha256:{}".format(digest)
        if group_ids != {expected_group_id}:
            raise ContentAuditMergeError(
                "audit records for exact hash {} must all use duplicate_content_group_id {!r}".format(
                    digest, expected_group_id
                )
            )
        if len(splits) != 1:
            raise ContentAuditMergeError(
                "exact-content duplicate group {!r} crosses splits {}".format(
                    expected_group_id, splits
                )
            )
        groups.append(
            {
                "duplicate_content_group_id": expected_group_id,
                "content_sha256": digest,
                "recording_ids": member_ids,
                "splits": splits,
                "cross_split_collision": False,
            }
        )
    return groups


def _validate_report_duplicate_groups(report: Mapping[str, Any], derived: Sequence[Mapping[str, Any]]) -> None:
    value = report.get("duplicate_content_groups")
    if not isinstance(value, list):
        raise ContentAuditMergeError("audit report duplicate_content_groups must be a list")
    expected = {
        (
            group["duplicate_content_group_id"],
            group["content_sha256"],
            tuple(group["recording_ids"]),
            tuple(group["splits"]),
            group["cross_split_collision"],
        )
        for group in derived
    }
    observed = set()
    for index, group in enumerate(value, 1):
        context = "audit report duplicate_content_groups row {}".format(index)
        if not isinstance(group, Mapping):
            raise ContentAuditMergeError("{} is not an object".format(context))
        group_id = _require_string(group, "duplicate_content_group_id", context=context)
        digest = _require_sha256(group.get("content_sha256"), context="{}.content_sha256".format(context))
        recording_ids = group.get("recording_ids")
        splits = group.get("splits")
        if not isinstance(recording_ids, list) or not all(
            isinstance(item, str) and item for item in recording_ids
        ):
            raise ContentAuditMergeError("{} has invalid recording_ids".format(context))
        if recording_ids != sorted(set(recording_ids)):
            raise ContentAuditMergeError("{} recording_ids must be sorted and unique".format(context))
        if not isinstance(splits, list) or not all(item in ALLOWED_SPLITS for item in splits):
            raise ContentAuditMergeError("{} has invalid splits".format(context))
        if splits != sorted(set(splits)):
            raise ContentAuditMergeError("{} splits must be sorted and unique".format(context))
        collision = _require_boolean(group, "cross_split_collision", context=context)
        observed.add((group_id, digest, tuple(recording_ids), tuple(splits), collision))
    if observed != expected:
        raise ContentAuditMergeError(
            "audit report duplicate_content_groups does not exactly agree with audit JSONL"
        )


def _validate_and_index_audit_records(
    records: Sequence[Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    audit_by_recording: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records, 1):
        context = "audit record {}".format(index)
        if record.get("schema_version") != AUDIT_SCHEMA_VERSION:
            raise ContentAuditMergeError(
                "{} schema_version must be {!r}".format(context, AUDIT_SCHEMA_VERSION)
            )
        recording_id = _require_string(record, "recording_id", context=context)
        source = sources.get(recording_id)
        if source is None:
            raise ContentAuditMergeError(
                "{} references recording_id {!r} absent from sources.jsonl".format(context, recording_id)
            )
        if recording_id in audit_by_recording:
            raise ContentAuditMergeError("duplicate audit record for recording_id {!r}".format(recording_id))
        if record.get("audit_status") != "hashed":
            raise ContentAuditMergeError("{} audit_status must be 'hashed'".format(context))
        digest = _require_sha256(record.get("content_sha256"), context="{}.content_sha256".format(context))
        source_split = _require_split(source, context="source {!r}".format(recording_id))
        if _require_split(record, context=context) != source_split:
            raise ContentAuditMergeError("{} split does not match sources.jsonl".format(context))
        source_retrieval_group = _require_string(
            source, "retrieval_group_id", context="source {!r}".format(recording_id)
        )
        if _require_string(record, "retrieval_group_id", context=context) != source_retrieval_group:
            raise ContentAuditMergeError("{} retrieval_group_id does not match sources.jsonl".format(context))
        group_id = _optional_group_id(record, "duplicate_content_group_id", context=context)
        if group_id is not None and not group_id.startswith("exact-sha256:"):
            raise ContentAuditMergeError(
                "{} duplicate_content_group_id is not an exact-SHA256 group".format(context)
            )
        audit_by_recording[recording_id] = dict(record)

    expected_ids = set(sources)
    observed_ids = set(audit_by_recording)
    if observed_ids != expected_ids:
        raise ContentAuditMergeError(
            "audit JSONL must have exactly one record per source (missing={}, unexpected={})".format(
                sorted(expected_ids - observed_ids), sorted(observed_ids - expected_ids)
            )
        )
    return audit_by_recording, _derive_duplicate_groups(audit_by_recording)


def _update_row(
    row: Mapping[str, Any], *, digest: str, group_id: Optional[str], context: str
) -> Dict[str, Any]:
    """Copy a row and apply only attested source-level content identities."""
    _check_existing_hash(row.get("content_sha256"), digest, context="{}.content_sha256".format(context))
    _check_existing_group(
        row.get("duplicate_content_group_id"), group_id, context="{}.duplicate_content_group_id".format(context)
    )
    result = _json_copy(row)
    assets = _require_mapping(result, "assets", context=context, allow_missing=True)
    _check_existing_hash(
        assets.get("content_sha256"), digest, context="{}.assets.content_sha256".format(context)
    )
    result["content_sha256"] = digest
    result["duplicate_content_group_id"] = group_id
    assets["content_sha256"] = digest
    # Existing staging sources previously said ``not_computed``.  Update this
    # explicit status alongside the hash so a consumer cannot mistake the
    # copied row for an unhashed source.
    assets["hash_status"] = HASH_STATUS
    result["assets"] = assets
    result["split_status"] = PROVISIONAL_SPLIT_STATUS
    qc = _require_mapping(result, "qc", context=context)
    _require_false(qc, "accepted_for_training", context="{}.qc".format(context))
    qc["accepted_for_training"] = False
    result["qc"] = qc
    return result


def merge_content_audit(
    sources: Sequence[Mapping[str, Any]],
    sequences: Sequence[Mapping[str, Any]],
    audit_records: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    source_manifest: Path,
    sequence_manifest: Path,
    audit_records_path: Path,
    audit_report_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Validate then derive output rows, without writing or mutating inputs."""
    source_by_id = _validate_sources(sources)
    sequence_by_id = _validate_sequences(sequences, source_by_id)
    _validate_report(
        report,
        source_manifest=source_manifest,
        audit_records=audit_records_path,
        source_count=len(source_by_id),
    )
    audit_by_id, duplicate_groups = _validate_and_index_audit_records(audit_records, source_by_id)
    _validate_report_duplicate_groups(report, duplicate_groups)

    merged_sources: List[Dict[str, Any]] = []
    for index, source in enumerate(sources, 1):
        recording_id = _require_string(source, "recording_id", context="source row {}".format(index))
        audit = audit_by_id[recording_id]
        merged_sources.append(
            _update_row(
                source,
                digest=str(audit["content_sha256"]),
                group_id=_optional_group_id(audit, "duplicate_content_group_id", context="audit"),
                context="source row {}".format(index),
            )
        )

    merged_sequences: List[Dict[str, Any]] = []
    for index, sequence in enumerate(sequences, 1):
        recording_id = _require_string(sequence, "recording_id", context="sequence row {}".format(index))
        audit = audit_by_id[recording_id]
        merged_sequences.append(
            _update_row(
                sequence,
                digest=str(audit["content_sha256"]),
                group_id=_optional_group_id(audit, "duplicate_content_group_id", context="audit"),
                context="sequence row {}".format(index),
            )
        )

    # Re-parse all output ownership/group invariants rather than trusting the
    # update loop.  This also makes any future edits to _update_row fail
    # closed before publication.
    merged_source_by_id = _validate_sources(merged_sources)
    _validate_sequences(merged_sequences, merged_source_by_id)
    expected_groups = {
        group["duplicate_content_group_id"]: (group["content_sha256"], tuple(group["splits"]))
        for group in duplicate_groups
    }
    seen_groups: Dict[str, Tuple[str, set[str]]] = {}
    for source in merged_sources:
        group_id = _optional_group_id(source, "duplicate_content_group_id", context="merged source")
        digest = _require_sha256(source.get("content_sha256"), context="merged source content_sha256")
        if group_id is not None:
            content, splits = seen_groups.setdefault(group_id, (digest, set()))
            if content != digest:
                raise ContentAuditMergeError(
                    "merged duplicate_content_group_id {!r} contains multiple content hashes".format(group_id)
                )
            splits.add(_require_split(source, context="merged source"))
    if set(seen_groups) != set(expected_groups):
        raise ContentAuditMergeError("merged duplicate groups do not match completed audit")
    for group_id, (digest, splits) in seen_groups.items():
        expected_digest, expected_splits = expected_groups[group_id]
        if digest != expected_digest or tuple(sorted(splits)) != expected_splits or len(splits) != 1:
            raise ContentAuditMergeError(
                "merged duplicate_content_group_id {!r} is not split-consistent".format(group_id)
            )

    source_split_counts = Counter(str(item["split"]) for item in merged_sources)
    sequence_split_counts = Counter(str(item["split"]) for item in merged_sequences)
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "pre_hmr_candidate",
        "inputs": {
            "sources_jsonl": str(source_manifest),
            "sources_jsonl_sha256": _sha256_file(source_manifest),
            "sequences_jsonl": str(sequence_manifest),
            "sequences_jsonl_sha256": _sha256_file(sequence_manifest),
            "audit_report": str(audit_report_path),
            "audit_report_sha256": _sha256_file(audit_report_path),
            "duplicate_content_audit_jsonl": str(audit_records_path),
            "duplicate_content_audit_jsonl_sha256": _sha256_file(audit_records_path),
        },
        "exact_content_audit": {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "audit_kind": AUDIT_KIND,
            "audit_complete": True,
            "source_resolution_complete": True,
            "valid": True,
            "no_cross_split_exact_content_collision": True,
            "duplicate_content_group_count": len(duplicate_groups),
            "duplicate_content_groups": duplicate_groups,
        },
        "sources": len(merged_sources),
        "sequences": len(merged_sequences),
        "split": {
            "assignments_preserved": True,
            "source_counts": {split: source_split_counts.get(split, 0) for split in sorted(ALLOWED_SPLITS)},
            "sequence_counts": {
                split: sequence_split_counts.get(split, 0) for split in sorted(ALLOWED_SPLITS)
            },
            "status": PROVISIONAL_SPLIT_STATUS,
        },
        "duplicate_content_qc_complete": False,
        "duplicate_content_qc_completion_reason": (
            "false by design: exact raw-file SHA-256 cannot establish near-duplicate, "
            "re-encode, crop, trim, or semantic-content QC"
        ),
        "accepted_for_training": False,
        "training_eligibility": (
            "none; exact byte-content audit is merged, but near-duplicate QC, 3D QC, "
            "labels, and a frozen release split remain required"
        ),
        "publication": "staged_then_atomic_directory_rename",
        "bundle_immutable": True,
    }
    return merged_sources, merged_sequences, summary


def publish_merged_bundle(
    output_dir: Path,
    sources: Sequence[Mapping[str, Any]],
    sequences: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> Tuple[Path, Path, Path]:
    """Publish all three derived files atomically without replacing a bundle."""
    output_dir = output_dir.expanduser().absolute()
    if _path_lexists(output_dir):
        raise FileExistsError(
            "merge output already exists and is immutable: {}; choose a new output directory".format(
                output_dir
            )
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".{}.staging-".format(output_dir.name), dir=str(output_dir.parent))
    )
    source_path = staging / "sources.jsonl"
    sequence_path = staging / "sequences.jsonl"
    summary_path = staging / "summary.json"
    try:
        _write_jsonl_new(source_path, sources)
        _write_jsonl_new(sequence_path, sequences)
        final_summary = dict(summary)
        final_summary.update(
            {
                "sources_jsonl": str(output_dir / "sources.jsonl"),
                "sources_jsonl_sha256": _sha256_file(source_path),
                "sequences_jsonl": str(output_dir / "sequences.jsonl"),
                "sequences_jsonl_sha256": _sha256_file(sequence_path),
                "summary_json": str(output_dir / "summary.json"),
            }
        )
        _write_json_new(summary_path, final_summary)
        if _path_lexists(output_dir):
            raise FileExistsError(
                "merge output already exists and is immutable: {}; choose a new output directory".format(
                    output_dir
                )
            )
        os.rename(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir / "sources.jsonl", output_dir / "sequences.jsonl", output_dir / "summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="pre-merge wild sources.jsonl")
    parser.add_argument("--sequences", required=True, help="pre-merge wild sequences.jsonl")
    parser.add_argument("--audit-report", required=True, help="immutable audit bundle report.json")
    parser.add_argument(
        "--audit-records", required=True, help="immutable audit duplicate_content_audit.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new immutable merged bundle directory (refuses an existing destination)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    source_manifest = _absolute_path(args.sources)
    sequence_manifest = _absolute_path(args.sequences)
    audit_report_path = _absolute_path(args.audit_report)
    audit_records_path = _absolute_path(args.audit_records)
    output_dir = _output_path(args.output_dir)
    source_rows = _read_jsonl(source_manifest, artifact_name="sources.jsonl")
    sequence_rows = _read_jsonl(sequence_manifest, artifact_name="sequences.jsonl")
    audit_rows = _read_jsonl(audit_records_path, artifact_name="duplicate_content_audit.jsonl")
    report = _read_json(audit_report_path)
    merged_sources, merged_sequences, summary = merge_content_audit(
        source_rows,
        sequence_rows,
        audit_rows,
        report,
        source_manifest=source_manifest,
        sequence_manifest=sequence_manifest,
        audit_records_path=audit_records_path,
        audit_report_path=audit_report_path,
    )
    source_path, sequence_path, summary_path = publish_merged_bundle(
        output_dir, merged_sources, merged_sequences, summary
    )
    print(
        json.dumps(
            {
                "sources": str(source_path),
                "sequences": str(sequence_path),
                "summary": str(summary_path),
                "recordings": len(merged_sources),
                "sequences_count": len(merged_sequences),
                "duplicate_content_qc_complete": False,
                "accepted_for_training": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    sys.exit(main())
