import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.merge_wild_content_audit import ContentAuditMergeError, main


AUDIT_SCHEMA = "atomic-wild-exact-content-audit-v1"
HASH_A = "a" * 64
HASH_C = "c" * 64
PROVISIONAL = "provisional_pending_duplicate_content_qc"
MERGED_PROVISIONAL = "provisional_pending_near_duplicate_content_qc"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _source(recording_id, split, sequence_ids, *, accepted=False):
    return {
        "schema_version": "atomic-source-v1",
        "stage": "pre_hmr_candidate",
        "recording_id": recording_id,
        "retrieval_group_id": recording_id,
        "duplicate_content_group_id": None,
        "split": split,
        "split_status": PROVISIONAL,
        "sequence_ids": sequence_ids,
        "assets": {"source_videos": [], "content_sha256": None, "hash_status": "not_computed"},
        "qc": {"accepted_for_training": accepted},
    }


def _sequence(sequence_id, recording_id, split, *, accepted=False):
    return {
        "schema_version": "atomic-sequence-v1",
        "stage": "pre_hmr_candidate",
        "sequence_id": sequence_id,
        "recording_id": recording_id,
        "retrieval_group_id": recording_id,
        "duplicate_content_group_id": None,
        "split": split,
        "split_status": PROVISIONAL,
        "assets": {"source_video": None, "content_sha256": None},
        "qc": {"accepted_for_training": accepted},
    }


def _audit_row(recording_id, split, digest, group_id=None):
    return {
        "schema_version": AUDIT_SCHEMA,
        "recording_id": recording_id,
        "retrieval_group_id": recording_id,
        "split": split,
        "audit_status": "hashed",
        "content_sha256": digest,
        "duplicate_content_group_id": group_id,
    }


def _report(sources_path, audit_path, source_count, duplicate_groups):
    return {
        "schema_version": AUDIT_SCHEMA,
        "audit_kind": "exact_byte_sha256_only",
        "bundle_immutable": True,
        "audit_complete": True,
        "source_resolution_complete": True,
        "no_observed_cross_split_collision": True,
        "full_corpus_exact_hash_qc_passed": True,
        "valid": True,
        "duplicate_content_qc_complete": False,
        "records_in_input_manifest": source_count,
        "records_audited": source_count,
        "hashed_recordings": source_count,
        "unresolved_recordings": 0,
        "status_counts": {"hashed": source_count},
        "cross_split_collision_errors": [],
        "duplicate_content_groups": duplicate_groups,
        "source_manifest": str(sources_path),
        "source_manifest_sha256": _sha256(sources_path),
        "audit_jsonl": str(audit_path),
        "audit_jsonl_sha256": _sha256(audit_path),
    }


def _write_report(path, report):
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(root, *, split_b="train", accepted_a=False):
    Path(root).mkdir(parents=True, exist_ok=True)
    sources = [
        _source("tiktok:a", "train", ["tiktok:a:clip000", "tiktok:a:clip001"], accepted=accepted_a),
        _source("tiktok:b", split_b, ["tiktok:b:clip000"]),
        _source("tiktok:c", "val", ["tiktok:c:clip000"]),
    ]
    sequences = [
        _sequence("tiktok:a:clip000", "tiktok:a", "train", accepted=accepted_a),
        _sequence("tiktok:a:clip001", "tiktok:a", "train", accepted=accepted_a),
        _sequence("tiktok:b:clip000", "tiktok:b", split_b),
        _sequence("tiktok:c:clip000", "tiktok:c", "val"),
    ]
    sources_path = root / "sources.jsonl"
    sequences_path = root / "sequences.jsonl"
    audit_path = root / "duplicate_content_audit.jsonl"
    report_path = root / "report.json"
    _write_jsonl(sources_path, sources)
    _write_jsonl(sequences_path, sequences)
    group_id = "exact-sha256:{}".format(HASH_A)
    audit_rows = [
        _audit_row("tiktok:a", "train", HASH_A, group_id),
        _audit_row("tiktok:b", split_b, HASH_A, group_id),
        _audit_row("tiktok:c", "val", HASH_C),
    ]
    _write_jsonl(audit_path, audit_rows)
    duplicate_groups = [
        {
            "duplicate_content_group_id": group_id,
            "content_sha256": HASH_A,
            "recording_ids": ["tiktok:a", "tiktok:b"],
            "splits": sorted({"train", split_b}),
            "cross_split_collision": split_b != "train",
        }
    ]
    report = _report(sources_path, audit_path, len(sources), duplicate_groups)
    _write_report(report_path, report)
    return sources_path, sequences_path, audit_path, report_path


def _invoke(sources, sequences, audit, report, output):
    return main(
        [
            "--sources",
            str(sources),
            "--sequences",
            str(sequences),
            "--audit-report",
            str(report),
            "--audit-records",
            str(audit),
            "--output-dir",
            str(output),
        ]
    )


class MergeWildContentAuditTests(unittest.TestCase):
    def test_immutable_merge_propagates_same_split_exact_duplicate_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, audit, report = _fixture(root)
            original_sources = sources.read_bytes()
            original_sequences = sequences.read_bytes()
            output = root / "merged"

            self.assertEqual(_invoke(sources, sequences, audit, report, output), 0)
            self.assertEqual(sources.read_bytes(), original_sources)
            self.assertEqual(sequences.read_bytes(), original_sequences)
            out_sources = {row["recording_id"]: row for row in _read_jsonl(output / "sources.jsonl")}
            out_sequences = {row["sequence_id"]: row for row in _read_jsonl(output / "sequences.jsonl")}
            group_id = "exact-sha256:{}".format(HASH_A)
            self.assertEqual(out_sources["tiktok:a"]["content_sha256"], HASH_A)
            self.assertEqual(out_sources["tiktok:a"]["assets"]["content_sha256"], HASH_A)
            self.assertEqual(
                out_sources["tiktok:a"]["assets"]["hash_status"],
                "exact_raw_recording_sha256_audited",
            )
            self.assertEqual(out_sources["tiktok:a"]["duplicate_content_group_id"], group_id)
            self.assertEqual(out_sources["tiktok:b"]["duplicate_content_group_id"], group_id)
            self.assertIsNone(out_sources["tiktok:c"]["duplicate_content_group_id"])
            self.assertEqual(out_sequences["tiktok:a:clip001"]["content_sha256"], HASH_A)
            self.assertEqual(out_sequences["tiktok:a:clip001"]["assets"]["content_sha256"], HASH_A)
            self.assertEqual(out_sequences["tiktok:a:clip001"]["duplicate_content_group_id"], group_id)
            for row in list(out_sources.values()) + list(out_sequences.values()):
                self.assertEqual(row["split_status"], MERGED_PROVISIONAL)
                self.assertIs(row["qc"]["accepted_for_training"], False)

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["split"]["assignments_preserved"])
            self.assertEqual(summary["split"]["status"], MERGED_PROVISIONAL)
            self.assertFalse(summary["duplicate_content_qc_complete"])
            self.assertFalse(summary["accepted_for_training"])
            self.assertEqual(summary["exact_content_audit"]["duplicate_content_group_count"], 1)
            with self.assertRaises(FileExistsError):
                _invoke(sources, sequences, audit, report, output)

    def test_report_must_bind_to_exact_supplied_source_and_audit_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, audit, report_path = _fixture(root)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["source_manifest_sha256"] = "f" * 64
            _write_report(report_path, report)
            with self.assertRaisesRegex(ContentAuditMergeError, "source_manifest_sha256"):
                _invoke(sources, sequences, audit, report_path, root / "bad-source-binding")

            sources, sequences, audit, report_path = _fixture(root / "audit-tamper")
            audit.write_text(audit.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ContentAuditMergeError, "audit_jsonl_sha256"):
                _invoke(sources, sequences, audit, report_path, root / "bad-audit-binding")

    def test_cross_split_exact_duplicate_is_rejected_even_if_report_claims_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, audit, report = _fixture(root, split_b="test")
            # The report deliberately lies about a clean audit.  The merger
            # derives groups from the rows too, so it cannot be fooled by this.
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["duplicate_content_groups"][0]["cross_split_collision"] = False
            _write_report(report, payload)
            with self.assertRaisesRegex(ContentAuditMergeError, "crosses splits"):
                _invoke(sources, sequences, audit, report, root / "must-not-publish")
            self.assertFalse((root / "must-not-publish").exists())

    def test_audit_requires_exactly_one_successful_row_for_every_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, audit, report_path = _fixture(root)
            rows = _read_jsonl(audit)
            _write_jsonl(audit, [row for row in rows if row["recording_id"] != "tiktok:b"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["audit_jsonl_sha256"] = _sha256(audit)
            _write_report(report_path, report)
            with self.assertRaisesRegex(ContentAuditMergeError, "exactly one record per source"):
                _invoke(sources, sequences, audit, report_path, root / "missing-audit-row")

    def test_previously_training_accepted_input_is_not_silently_downgraded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, sequences, audit, report = _fixture(root, accepted_a=True)
            with self.assertRaisesRegex(ContentAuditMergeError, "accepted_for_training false"):
                _invoke(sources, sequences, audit, report, root / "must-not-downgrade")
            self.assertFalse((root / "must-not-downgrade").exists())


if __name__ == "__main__":
    unittest.main()
