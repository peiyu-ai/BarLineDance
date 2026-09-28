"""One reader for "which clips must be derived again", and what it refuses.

Stage B and stage F both consume this list.  When four tools each parsed a
song's identity their own way this repo spent a day on the disagreement, so the
manifest has one reader; these tests pin what it accepts and, more usefully,
the two cases where returning an empty list would be a lie.
"""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import redo_manifest


class Reading(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)

    def _write(self, name, payload):
        path = pathlib.Path(self.workspace.name) / name
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload),
                        encoding="utf-8")
        return path

    def test_plain_list(self):
        path = self._write("s.txt", "a__clip000\n\nb__clip001\n")
        self.assertEqual(redo_manifest.load(path, "3d"), ["a__clip000", "b__clip001"])

    def test_each_stage_reads_its_own_half_of_an_audit(self):
        path = self._write("audit.json", {"stale": {"3d": ["x"], "s3d": ["y", "z"]}})
        self.assertEqual(redo_manifest.load(path, "3d"), ["x"])
        self.assertEqual(redo_manifest.load(path, "s3d"), ["y", "z"])

    def test_a_stage_with_nothing_stale_is_an_empty_list_not_an_error(self):
        path = self._write("audit.json", {"stale": {"3d": [], "s3d": ["y"]}})
        self.assertEqual(redo_manifest.load(path, "3d"), [])

    def test_an_audit_that_never_covered_the_stage_is_refused(self):
        """Empty and "not measured" are the same value and opposite facts."""
        path = self._write("audit.json", {"stale": {"s3d": ["y"]}})
        with self.assertRaises(redo_manifest.ManifestError) as raised:
            redo_manifest.load(path, "3d")
        self.assertIn("did not cover stage", str(raised.exception))

    def test_a_census_is_read_as_every_clip_it_names(self):
        path = self._write("census.json", {"clips": {"b__clip001": {}, "a__clip000": {}}})
        self.assertEqual(redo_manifest.load(path, "s3d"), ["a__clip000", "b__clip001"])

    def test_a_worklist_of_rows_is_read_as_the_stems_it_names(self):
        """``str()`` over these rows returns stem-shaped nonsense, not stems.

        ``runs/wild_ingest_v1_worklist.json`` stores ``{"clip": ..., "bbx": ...}``
        per clip.  Stringifying the row yields a list of the right *length*, so
        a stage driven by it prints the count it was asked for and matches no
        clip at all.
        """
        path = self._write("worklist.json", {"clips": [{"clip": "b__clip001", "bbx": True},
                                                       {"clip": "a__clip000", "bbx": False}]})
        self.assertEqual(redo_manifest.load(path, "3d"), ["a__clip000", "b__clip001"])

    def test_a_row_without_a_clip_key_is_refused(self):
        path = self._write("worklist.json", {"clips": [{"clip": "a__clip000"}, {"bbx": True}]})
        with self.assertRaises(redo_manifest.ManifestError) as raised:
            redo_manifest.load(path, "3d")
        self.assertIn("no 'clip' key", str(raised.exception))

    def test_a_json_file_that_is_neither_is_refused(self):
        path = self._write("other.json", {"summary": "nothing here"})
        with self.assertRaises(redo_manifest.ManifestError):
            redo_manifest.load(path, "3d")

    def test_an_unknown_stage_is_refused_before_the_file_is_read(self):
        with self.assertRaises(redo_manifest.ManifestError):
            redo_manifest.load(pathlib.Path("/nonexistent"), "motion")


class Partitioning(unittest.TestCase):
    def test_named_stems_are_split_not_filtered(self):
        """Both halves need handling and both are silent failures if merged."""
        inside, outside = redo_manifest.partition(["a", "ghost", "b"], ["a", "b", "c"])
        self.assertEqual(inside, ["a", "b"])
        self.assertEqual(outside, ["ghost"])


if __name__ == "__main__":
    unittest.main()
