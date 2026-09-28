"""What an upload set produced before the fps re-ingest, and what it produces now.

The re-ingest changes which clips exist, not only their contents: the length
bounds are policies about *seconds*, so on a 60 fps upload they cover twice as
many source frames and an upload that used to split into two clips can come
back as one.  Names are ``<upload>__clip%03d``, so a surviving name is
overwritten in place -- these credentials can PUT over an existing key -- while
a name that stops being produced becomes an object they cannot delete.  That
list is the only thing standing between the corpus and a consumer that globs a
directory, so these tests pin it.

The second property is the one that is easy to lose.  ``clip.mp4`` has been
evicted from the local ingest tree, so the baseline knows spans and not bytes.
"Its span did not move" is not "its bytes are the same", and the comparison has
to keep those apart -- the R0 round is the precedent: judging staleness by
frame count found 3,929 clips and judging it by content found 3,991.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import refix_wild_fps_clips_census as census


def _manifest(root, produced):
    """The ingest's own statement of what a run ended up with.

    The census asks this rather than the directory listing, because nothing
    deletes a clip directory: an upload that used to yield four clips and now
    yields two leaves the other two on disk, identical in every respect except
    that no run produces them.
    """
    rows = []
    for upload, names in produced.items():
        rows.append({"upload": upload, "status": "ok", "seconds": 30.0,
                     "spans": [], "ingested_at": 1000.0,
                     "clips": [{"clip": n, "status": "ok"} for n in names]})
    (pathlib.Path(root) / "ingest_shard0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _clip(root, name, span, video=None, frames=450):
    directory = pathlib.Path(root) / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps({
        "num_frames": frames, "fps": 30.0, "source_frame_span": list(span),
        "span_end_reason": "upload_end"}), encoding="utf-8")
    if video is not None:
        (directory / "clip.mp4").write_bytes(video)
    return directory


class Comparison(unittest.TestCase):
    def test_a_name_that_stops_being_produced_is_named_an_orphan(self):
        before = {"clips": {"u__clip000": {"upload": "u", "source_frame_span": [0, 450]},
                            "u__clip001": {"upload": "u", "source_frame_span": [450, 900]}}}
        after = {"clips": {"u__clip000": {"upload": "u", "source_frame_span": [0, 900]}}}
        diff = census.compare(before, after)
        self.assertEqual(diff["orphaned"], ["u__clip001"])
        self.assertEqual(diff["orphan_uploads"], ["u"])
        # And the survivor is reported as proved-different, because its span moved.
        self.assertEqual([c["clip"] for c in diff["changed"]], ["u__clip000"])

    def test_a_new_name_is_not_confused_with_a_changed_one(self):
        before = {"clips": {"u__clip000": {"upload": "u", "source_frame_span": [0, 450]}}}
        after = {"clips": {"u__clip000": {"upload": "u", "source_frame_span": [0, 450]},
                           "u__clip001": {"upload": "u", "source_frame_span": [450, 900]}}}
        diff = census.compare(before, after)
        self.assertEqual(diff["created"], ["u__clip001"])
        self.assertEqual(diff["orphaned"], [])
        self.assertEqual(diff["changed"], [])

    def test_an_uncompared_clip_is_not_reported_as_identical(self):
        """The baseline had no video, so nothing weighed these bytes.

        This is the whole reason the report has three buckets instead of two.
        """
        before = {"clips": {"u__clip000": {"upload": "u", "source_frame_span": [0, 450],
                                           "clip_sha256_1mb": None}}}
        after = {"clips": {"u__clip000": {"upload": "u", "source_frame_span": [0, 450],
                                          "clip_sha256_1mb": "abc"}}}
        diff = census.compare(before, after)
        self.assertEqual(diff["unchanged_span"], ["u__clip000"])
        self.assertEqual(diff["bytes_identical"], [])
        self.assertEqual(diff["unverified_bytes"], ["u__clip000"])

    def test_bytes_are_weighed_when_both_sides_have_them(self):
        before = {"clips": {"a__clip000": {"upload": "a", "source_frame_span": [0, 450],
                                           "clip_sha256_1mb": "same"},
                            "b__clip000": {"upload": "b", "source_frame_span": [0, 450],
                                           "clip_sha256_1mb": "old"}}}
        after = {"clips": {"a__clip000": {"upload": "a", "source_frame_span": [0, 450],
                                          "clip_sha256_1mb": "same"},
                           "b__clip000": {"upload": "b", "source_frame_span": [0, 450],
                                          "clip_sha256_1mb": "new"}}}
        diff = census.compare(before, after)
        self.assertEqual(diff["bytes_identical"], ["a__clip000"])
        # Same span, different bytes -- which is exactly the resampled 30 fps
        # grid of a non-30 fps upload, and would be invisible to a span test.
        self.assertEqual(diff["bytes_differ"], ["b__clip000"])
        self.assertEqual(diff["changed"], [])


class ClipRecord(unittest.TestCase):
    def test_a_clip_without_meta_is_a_problem_not_an_absence(self):
        with tempfile.TemporaryDirectory() as workspace:
            broken = pathlib.Path(workspace) / "u__clip000"
            broken.mkdir()
            record = census.clip_record(broken)
        self.assertEqual(record["status"], "no_meta")

    def test_a_missing_video_is_recorded_as_a_named_absence(self):
        with tempfile.TemporaryDirectory() as workspace:
            directory = _clip(workspace, "u__clip000", (0, 450))
            record = census.clip_record(directory)
        self.assertIsNone(record["clip_sha256_1mb"])
        self.assertFalse(record["video_local"])
        self.assertFalse(record["records_source_fps"])


class Driver(unittest.TestCase):
    def test_a_broken_clip_makes_the_census_exit_non_zero(self):
        """A shrunken baseline is indistinguishable from a corpus that grew."""
        with tempfile.TemporaryDirectory() as workspace:
            base = pathlib.Path(workspace)
            uploads, ingest = base / "uploads", base / "ingest"
            uploads.mkdir(); ingest.mkdir()
            (uploads / "u.mp4").write_bytes(b"not really a video")
            _clip(ingest, "u__clip000", (0, 450))
            (ingest / "u__clip001").mkdir()          # no meta.json
            completed = subprocess.run(
                [sys.executable, "tools/refix_wild_fps_clips_census.py",
                 "--uploads", str(uploads), "--ingest", str(ingest),
                 "--output", str(base / "census.json"), "--no-probe-rate"],
                capture_output=True, text=True,
                cwd=str(pathlib.Path(__file__).resolve().parents[1]))
            self.assertEqual(completed.returncode, 1, completed.stdout + completed.stderr)
            self.assertIn("PROBLEMS", completed.stdout)
            written = json.loads((base / "census.json").read_text(encoding="utf-8"))
        # Written anyway: the operator needs to see which clip, and the broken
        # one is in the census with a status rather than dropped from it.
        self.assertEqual(sorted(written["clips"]), ["u__clip000", "u__clip001"])
        self.assertEqual(written["clips"]["u__clip001"]["status"], "no_meta")


if __name__ == "__main__":
    unittest.main()


class ManifestOutput(unittest.TestCase):
    """The lists are what the next stages consume, so they are written by the
    tool that computed them rather than re-derived by hand downstream."""

    def _corpus(self, base, names, produced=None):
        uploads, ingest = base / "uploads", base / "ingest"
        uploads.mkdir(exist_ok=True); ingest.mkdir(exist_ok=True)
        (uploads / "u.mp4").write_bytes(b"x")
        for index, name in enumerate(names):
            _clip(ingest, name, (index * 450, (index + 1) * 450))
        _manifest(ingest, {"u": list(names if produced is None else produced)})
        return uploads, ingest

    def _run(self, base, args):
        return subprocess.run(
            [sys.executable, "tools/refix_wild_fps_clips_census.py"] + args,
            capture_output=True, text=True,
            cwd=str(pathlib.Path(__file__).resolve().parents[1]))

    def test_orphan_and_audit_lists_are_written_and_disjoint(self):
        with tempfile.TemporaryDirectory() as workspace:
            base = pathlib.Path(workspace)
            uploads, ingest = self._corpus(base, ["u__clip000", "u__clip001"])
            first = self._run(base, ["--uploads", str(uploads), "--ingest", str(ingest),
                                     "--output", str(base / "before.json"),
                                     "--no-probe-rate"])
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)

            # The upload re-splits: clip001 stops being produced.  Its
            # directory stays exactly where it was -- that is the whole
            # difficulty -- so what changes is the manifest, not the tree.
            _manifest(ingest, {"u": ["u__clip000"]})
            second = self._run(base, ["--uploads", str(uploads), "--ingest", str(ingest),
                                      "--output", str(base / "after.json"),
                                      "--compare", str(base / "before.json"),
                                      "--orphan-list", str(base / "orphans.txt"),
                                      "--audit-list", str(base / "audit.txt"),
                                      "--no-probe-rate"])
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            orphans = (base / "orphans.txt").read_text(encoding="utf-8").split()
            audit = (base / "audit.txt").read_text(encoding="utf-8").split()

        self.assertEqual(orphans, ["u__clip001"])
        self.assertEqual(audit, ["u__clip000"])
        # An orphan is consistent with its own derived artifacts, so the
        # freshness audit would call it fresh.  Keeping it out of the audit list
        # is what stops "fresh" from being read as "keep".
        self.assertEqual(set(orphans) & set(audit), set())

    def test_a_clip_still_on_disk_but_no_longer_produced_is_an_orphan(self):
        """The directory listing cannot see this, and it is the common case.

        Before the census asked the manifest, a run that orphaned hundreds of
        clips reported zero, because every one of their directories was still
        exactly where the previous generation left it.
        """
        before = {"clips": {"u__clip000": {"upload": "u", "source_frame_span": [0, 450],
                                           "produced_now": True},
                            "u__clip001": {"upload": "u", "source_frame_span": [450, 900],
                                           "produced_now": True}}}
        after = {"clips": {"u__clip000": {"upload": "u", "source_frame_span": [0, 900],
                                          "produced_now": True},
                           "u__clip001": {"upload": "u", "source_frame_span": [450, 900],
                                          "produced_now": False}}}
        diff = census.compare(before, after)
        self.assertEqual(diff["orphaned"], ["u__clip001"])
        # And it is kept out of the span comparison: its span is unchanged,
        # which is true and would read as "this clip is fine".
        self.assertEqual(diff["unchanged_span"], [])
        self.assertEqual([c["clip"] for c in diff["changed"]], ["u__clip000"])

    def test_an_orphan_list_without_a_baseline_is_refused(self):
        with tempfile.TemporaryDirectory() as workspace:
            base = pathlib.Path(workspace)
            uploads, ingest = self._corpus(base, ["u__clip000"])
            completed = self._run(base, ["--uploads", str(uploads), "--ingest", str(ingest),
                                         "--output", str(base / "c.json"),
                                         "--orphan-list", str(base / "orphans.txt"),
                                         "--no-probe-rate"])
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("needs --compare", completed.stdout + completed.stderr)
        # Nothing written: an empty exclusion list reads exactly like "nothing
        # to exclude", which is the most expensive way for this to be wrong.
        self.assertFalse((base / "orphans.txt").exists())


class FailedCutsAreNotOrphans(unittest.TestCase):
    """Two ways to be absent from this run's output, needing opposite handling.

    An orphan is finished with: nothing may read it again, and on a store that
    cannot delete, the name is the only mechanism there is.  A failed cut is
    work still owed -- the run meant to produce that clip, the directory holds
    last generation's, and it has to be retried.  On 2026-08-19 the census
    reported 989 orphans of which 214 were failed cuts; excluding those by name
    would have retired 214 clips the corpus is supposed to have.
    """

    def test_a_failed_cut_is_reported_separately(self):
        before = {"clips": {"u__clip000": {"upload": "u", "source_frame_span": [0, 450]},
                            "u__clip001": {"upload": "u", "source_frame_span": [450, 900]}}}
        after = {"clips": {
            "u__clip000": {"upload": "u", "source_frame_span": [0, 450],
                           "produced_now": False, "cut_failed": True},
            "u__clip001": {"upload": "u", "source_frame_span": [450, 900],
                           "produced_now": False, "cut_failed": False}}}
        diff = census.compare(before, after)
        self.assertEqual(diff["orphaned"], ["u__clip001"])
        self.assertEqual(diff["failed_to_produce"], ["u__clip000"])
        # Neither belongs in the span comparison; both would read "unchanged".
        self.assertEqual(diff["unchanged_span"], [])


class RateAgreement(unittest.TestCase):
    """The field the end gate reads.

    ``records_source_fps`` says a rate was written down; it does not say the
    rate was the right one.  The 2026-08-19 re-cut wrote ``source_fps`` from
    ``r_frame_rate`` alone, which on a variable-rate container is exactly the
    claim that is wrong, so 17 clips gained the field while their picture went
    to about twice the speed of their sound.  The census therefore records both
    of the container's rate claims and whether they agree.
    """

    def _record(self, extra):
        with tempfile.TemporaryDirectory() as raw:
            directory = pathlib.Path(raw) / "u__clip000"
            directory.mkdir()
            meta = {"num_frames": 450, "fps": 30.0, "source_frame_span": [0, 450]}
            meta.update(extra)
            (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            return census.clip_record(directory)

    def test_a_clip_with_no_rate_pair_is_unmeasured_rather_than_clean(self):
        record = self._record({"source_fps": 60.0})
        self.assertTrue(record["records_source_fps"])
        self.assertIsNone(record["rates_agree"])

    def test_a_clip_cut_from_a_disagreeing_container_is_marked(self):
        record = self._record({"source_fps": 60.0, "source_avg_frame_rate": 30.07,
                               "source_r_frame_rate": 60.0})
        self.assertIs(record["rates_agree"], False)

    def test_a_clip_cut_from_a_constant_rate_re_encode_agrees(self):
        record = self._record({"source_fps": 30.07, "source_avg_frame_rate": 30.07,
                               "source_r_frame_rate": 30.07,
                               "cfr_normalized_from": "/uploads/u.mp4"})
        self.assertIs(record["rates_agree"], True)
        self.assertEqual(record["cfr_normalized_from"], "/uploads/u.mp4")
