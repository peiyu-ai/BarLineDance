import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import audit_wild_duplicates
from tools.audit_wild_duplicates import audit_sources, main


def _source(recording_id, split, key, source_videos=None):
    return {
        "recording_id": recording_id,
        "retrieval_group_id": recording_id,
        "split": split,
        "source_recording_keys": [key],
        "assets": {"source_videos": source_videos or []},
    }


class WildDuplicateAuditTests(unittest.TestCase):
    def test_exact_raw_byte_duplicates_get_a_group_without_split_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "raw_a.mp4").write_bytes(b"same exact bytes")
            (raw / "raw_b.mp4").write_bytes(b"same exact bytes")
            (raw / "raw_c.mp4").write_bytes(b"different bytes")
            sources = [
                _source("tiktok:a", "train", "raw_a"),
                _source("tiktok:b", "train", "raw_b"),
                _source("tiktok:c", "val", "raw_c"),
            ]

            records, report = audit_sources(sources, raw_video_root=raw, chunk_bytes=3)
            by_id = {record["recording_id"]: record for record in records}
            self.assertTrue(report["valid"])
            self.assertTrue(report["audit_complete"])
            self.assertTrue(report["source_resolution_complete"])
            self.assertTrue(report["full_corpus_exact_hash_qc_passed"])
            self.assertFalse(report["duplicate_content_qc_complete"])
            self.assertEqual(report["status_counts"], {"hashed": 3})
            self.assertEqual(len(report["duplicate_content_groups"]), 1)
            self.assertEqual(by_id["tiktok:a"]["split"], "train")
            self.assertEqual(by_id["tiktok:b"]["split"], "train")
            self.assertEqual(
                by_id["tiktok:a"]["duplicate_content_group_id"],
                by_id["tiktok:b"]["duplicate_content_group_id"],
            )
            self.assertIsNone(by_id["tiktok:c"]["duplicate_content_group_id"])
            self.assertEqual(by_id["tiktok:a"]["resolution"]["scope"], "raw_recording_file")

    def test_cross_split_exact_collision_is_an_error_but_is_not_reassigned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "raw_a.mp4").write_bytes(b"same")
            (raw / "raw_b.mp4").write_bytes(b"same")
            sources = [
                _source("tiktok:a", "train", "raw_a"),
                _source("tiktok:b", "test", "raw_b"),
            ]

            records, report = audit_sources(sources, raw_video_root=raw)
            self.assertFalse(report["valid"])
            self.assertTrue(report["audit_complete"])
            self.assertFalse(report["no_observed_cross_split_collision"])
            self.assertFalse(report["full_corpus_exact_hash_qc_passed"])
            self.assertEqual(len(report["cross_split_collision_errors"]), 1)
            self.assertEqual(
                report["cross_split_collision_errors"][0]["recording_ids"], ["tiktok:a", "tiktok:b"]
            )
            self.assertEqual([record["split"] for record in records], ["train", "test"])
            self.assertEqual(
                report["split_action"],
                "none; source-manifest split assignments are reported, never overwritten or reassigned",
            )

    def test_ambiguous_and_unavailable_sources_are_explicit_and_unhashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "raw_a.mp4").write_bytes(b"one")
            (raw / "raw_a.mov").write_bytes(b"two")
            sources = [
                _source("tiktok:a", "train", "raw_a"),
                _source("tiktok:missing", "val", "missing"),
            ]

            records, report = audit_sources(sources, raw_video_root=raw)
            by_id = {record["recording_id"]: record for record in records}
            self.assertEqual(by_id["tiktok:a"]["audit_status"], "ambiguous")
            self.assertEqual(len(by_id["tiktok:a"]["resolution"]["candidate_paths"]), 2)
            self.assertIsNone(by_id["tiktok:a"]["content_sha256"])
            self.assertEqual(by_id["tiktok:missing"]["audit_status"], "unavailable")
            self.assertEqual(report["status_counts"], {"ambiguous": 1, "unavailable": 1})
            self.assertFalse(report["source_resolution_complete"])
            self.assertFalse(report["full_corpus_exact_hash_qc_passed"])

    def test_manifest_fallback_and_immutable_cli_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "single_clip.mp4"
            video.write_bytes(b"fallback bytes")
            sources_path = root / "sources.jsonl"
            sources_path.write_text(
                json.dumps(_source("tiktok:a", "train", "raw_a", [str(video)])) + "\n",
                encoding="utf-8",
            )
            output = root / "audit"
            self.assertEqual(
                main(
                    [
                        "--sources",
                        str(sources_path),
                        "--output-dir",
                        str(output),
                        "--chunk-bytes",
                        "2",
                        "--max-recordings",
                        "1",
                    ]
                ),
                0,
            )
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            record = json.loads((output / "duplicate_content_audit.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(report["bundle_immutable"])
            self.assertTrue(report["audit_complete"])
            self.assertEqual(record["audit_status"], "hashed")
            self.assertEqual(record["resolution"]["method"], "manifest_source_video_fallback")
            with self.assertRaises(FileExistsError):
                main(
                    [
                        "--sources",
                        str(sources_path),
                        "--output-dir",
                        str(output),
                        "--max-recordings",
                        "1",
                    ]
                )

    def test_bounded_audit_uses_stable_recording_id_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "a.mp4").write_bytes(b"a")
            (raw / "b.mp4").write_bytes(b"b")
            records, report = audit_sources(
                [
                    _source("tiktok:b", "train", "b"),
                    _source("tiktok:a", "val", "a"),
                ],
                raw_video_root=raw,
                max_recordings=1,
            )
            self.assertEqual([record["recording_id"] for record in records], ["tiktok:a"])
            self.assertEqual(report["records_in_input_manifest"], 2)
            self.assertEqual(report["records_audited"], 1)
            self.assertFalse(report["audit_complete"])
            self.assertTrue(report["no_observed_cross_split_collision"])
            self.assertFalse(report["full_corpus_exact_hash_qc_passed"])

    def test_staging_retrieval_group_contract_is_checked_before_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "raw_a.mp4").write_bytes(b"data")
            source = _source("tiktok:a", "train", "raw_a")
            source["retrieval_group_id"] = "wrong-group"
            records, report = audit_sources([source], raw_video_root=raw)
            self.assertEqual(records[0]["audit_status"], "invalid_manifest")
            self.assertIsNone(records[0]["content_sha256"])
            self.assertIn("retrieval_group_id", records[0]["resolution"]["detail"])
            self.assertFalse(report["source_resolution_complete"])

    def test_failed_publication_cleans_staging_without_leaving_a_partial_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "sources.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            output = root / "audit"
            records = [{"recording_id": "tiktok:a", "content_sha256": None}]
            report = {"schema_version": "test"}
            with patch.object(audit_wild_duplicates, "_write_json_new", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    audit_wild_duplicates.publish_audit_bundle(
                        output,
                        records,
                        report,
                        source_manifest=manifest,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".audit.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
