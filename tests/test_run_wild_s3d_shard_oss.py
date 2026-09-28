"""Stage F decides what to extract, and "there is a file" is not the question.

Until 2026-08-19 this shard picked its work with ``stem not in have_features``.
The published store then turned out to hold 3,991 clips whose ``.npz`` existed,
carried the right number of rows, and had been extracted from an older cut of
the video -- so that test called every one of them done, and a re-extraction
driven by it would have printed a clean ``todo=0``.  The judgment it cannot make
on its own is made by ``tools/audit_clip_freshness.py`` and arrives as a list.

These tests pin the two halves: a named stem is extracted even though its
features are present, and a named stem the corpus cannot serve is an error
rather than a quiet drop -- returning fewer clips than the caller asked for is
how a partial re-extraction reports success.
"""

import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import run_wild_s3d_shard_oss as stage_f


class Eligibility(unittest.TestCase):
    def _patch(self, have_3d, have_features):
        """Patch the real function, not sys.modules.

        ``eligible`` does ``import tools.run_gvhmr_ingest_shard_oss as stage_b``,
        and once anything in the suite has imported that module for real the
        name is an attribute of the ``tools`` package -- so a ``patch.dict`` on
        ``sys.modules`` is bypassed and the test passes alone but fails in the
        full run.  Patching the function itself does not care who imported what.
        """
        import tools.run_gvhmr_ingest_shard_oss as stage_b

        stems = mock.patch.object(stage_b, "stems_with", return_value=have_3d)
        listing = mock.patch.object(
            stage_f.asset_io, "list_prefix",
            return_value={name + ".npz": 1 for name in have_features})
        stems.start(); listing.start()
        self.addCleanup(stems.stop); self.addCleanup(listing.stop)

    def test_without_a_list_only_missing_features_are_work(self):
        self._patch(["a", "b", "c"], ["a", "b"])
        self.assertEqual(stage_f.eligible(1, 0), ["c"])

    def test_a_named_stem_is_redone_although_its_features_exist(self):
        self._patch(["a", "b", "c"], ["a", "b", "c"])
        self.assertEqual(stage_f.eligible(1, 0), [])
        self.assertEqual(stage_f.eligible(1, 0, ["b"]), ["b"])

    def test_naming_a_stem_with_no_3d_is_an_error_not_a_silent_drop(self):
        self._patch(["a", "b"], ["a", "b"])
        with self.assertRaises(SystemExit) as raised:
            stage_f.eligible(1, 0, ["b", "ghost"])
        self.assertIn("ghost", str(raised.exception))

    def test_sharding_still_partitions_the_work(self):
        self._patch(["a", "b", "c", "d"], [])
        shards = [stage_f.eligible(2, 0), stage_f.eligible(2, 1)]
        self.assertEqual(sorted(shards[0] + shards[1]), ["a", "b", "c", "d"])
        self.assertEqual(set(shards[0]) & set(shards[1]), set())


class RedoList(unittest.TestCase):
    """The list is a measurement; retyping it by hand is where it stops being one."""

    def _write(self, name, payload):
        path = pathlib.Path(self.workspace.name) / name
        path.write_text(payload, encoding="utf-8")
        return path

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)

    def test_reads_a_plain_list(self):
        path = self._write("stems.txt", "a__clip000\nb__clip001\n\n")
        self.assertEqual(stage_f.load_redo(path), ["a__clip000", "b__clip001"])

    def test_reads_the_freshness_audit_directly(self):
        path = self._write("audit.json", json.dumps(
            {"stale": {"3d": ["x"], "s3d": ["a__clip000", "b__clip001"]}}))
        self.assertEqual(stage_f.load_redo(path), ["a__clip000", "b__clip001"])

    def test_reads_a_census_directly(self):
        path = self._write("census.json", json.dumps(
            {"clips": {"b__clip001": {}, "a__clip000": {}}}))
        self.assertEqual(stage_f.load_redo(path), ["a__clip000", "b__clip001"])

    def test_a_json_file_with_no_stem_list_is_refused(self):
        path = self._write("wrong.json", json.dumps({"summary": "nothing here"}))
        with self.assertRaises(SystemExit):
            stage_f.load_redo(path)


if __name__ == "__main__":
    unittest.main()
