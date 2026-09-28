"""The static-cam gate must be able to say no, and must say no for the right reason.

This gate decides whether to assert R_w2c = I on footage that visual odometry
could not solve. Asserting it wrongly fabricates a camera; refusing wrongly
throws away usable material. So the tests here are mostly about REFUSAL: that
it happens, that a rotating clip and an unmeasurable clip are refused
*differently*, and that neither collapses into the accept branch.
"""

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.run_static_cam_recovery import (  # noqa: E402
    ACCEPT, REFUSE, UNMEASURED, decide, outstanding)


def measurement(rotation, *, static=None, reason="within threshold", measured=1.0):
    if static is None:
        static = rotation is not None and rotation <= 3.0
    return {
        "max_rotation_from_first_frame_deg": rotation,
        "static_camera": static,
        "verdict_reason": reason,
        "measured_fraction": measured,
    }


class DecideTests(unittest.TestCase):
    def test_a_still_camera_is_accepted(self):
        got = decide(measurement(0.2132), 3.0)
        self.assertEqual(got["decision"], ACCEPT)
        self.assertAlmostEqual(got["max_rotation_deg"], 0.2132)

    def test_a_rotating_camera_is_refused_with_its_number(self):
        """The number must survive into the verdict: a refusal without its
        measurement is indistinguishable from a crash, and this repo's rule is
        that a gate reports what it compared."""
        got = decide(measurement(12.12, static=False, reason="exceeds threshold"), 3.0)
        self.assertEqual(got["decision"], REFUSE)
        self.assertAlmostEqual(got["max_rotation_deg"], 12.12)
        self.assertIn("12.12", got["why"])

    def test_exactly_at_threshold_is_accepted(self):
        self.assertEqual(decide(measurement(3.0), 3.0)["decision"], ACCEPT)

    def test_just_over_threshold_is_refused(self):
        self.assertEqual(
            decide(measurement(3.0001, static=False, reason="exceeds threshold"), 3.0)["decision"],
            REFUSE)

    def test_threshold_is_honoured_not_hardcoded(self):
        """A clip refused at 3 degrees must be accepted at 15, or the argument
        is decoration."""
        clip = measurement(12.12, static=False, reason="exceeds threshold")
        self.assertEqual(decide(clip, 3.0)["decision"], REFUSE)
        self.assertEqual(decide(clip, 15.0)["decision"], ACCEPT)

    def test_unmeasurable_background_is_not_the_same_as_still(self):
        """The failure this separates: a clip whose background could not be fit
        reports no rotation, and treating that as 0 would declare it static on
        faith -- exactly what measure_camera_rotation.py refuses to do."""
        got = decide(measurement(None, static=False, reason="too few measurable pairs",
                                 measured=0.05), 3.0)
        self.assertEqual(got["decision"], UNMEASURED)
        self.assertNotIn("max_rotation_deg", got)

    def test_a_missing_measurement_is_unmeasured_not_accepted(self):
        self.assertEqual(decide(None, 3.0)["decision"], UNMEASURED)
        self.assertEqual(decide({}, 3.0)["decision"], UNMEASURED)

    def test_the_three_outcomes_are_distinct(self):
        outcomes = {
            decide(measurement(0.5), 3.0)["decision"],
            decide(measurement(9.0, static=False, reason="exceeds threshold"), 3.0)["decision"],
            decide(None, 3.0)["decision"],
        }
        self.assertEqual(len(outcomes), 3)


class OutstandingTests(unittest.TestCase):
    def _tree(self, root, clips, with_result=()):
        for clip in clips:
            (root / "ingest" / clip).mkdir(parents=True, exist_ok=True)
            (root / "ingest" / clip / "clip.mp4").write_bytes(b"\0")
        for clip in with_result:
            (root / "raw" / clip).mkdir(parents=True, exist_ok=True)
            (root / "raw" / clip / "hmr4d_results.pt").write_bytes(b"\0")
        (root / "raw").mkdir(parents=True, exist_ok=True)
        return root / "ingest", root / "raw"

    def test_clips_with_a_result_are_skipped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            ingest, rawroot = self._tree(pathlib.Path(raw), ["a", "b", "c"], with_result=["b"])
            self.assertEqual(outstanding(ingest, rawroot, None), ["a", "c"])

    def test_a_cleaned_failure_marker_does_not_hide_a_clip(self):
        """Selecting on the marker rather than the missing result would make
        this pass silently incomplete for any clip whose marker was cleared."""
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            ingest, rawroot = self._tree(pathlib.Path(raw), ["a"])
            (rawroot / "a").mkdir(parents=True, exist_ok=True)   # no marker, no result
            self.assertEqual(outstanding(ingest, rawroot, None), ["a"])

    def test_an_explicit_clip_list_is_still_filtered_by_result(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            ingest, rawroot = self._tree(pathlib.Path(raw), ["a", "b"], with_result=["a"])
            self.assertEqual(outstanding(ingest, rawroot, ["a", "b"]), ["b"])

    def test_a_clip_without_video_is_not_offered(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            ingest, rawroot = self._tree(root, ["a"])
            (ingest / "novideo").mkdir(parents=True, exist_ok=True)
            self.assertEqual(outstanding(ingest, rawroot, None), ["a"])


if __name__ == "__main__":
    unittest.main()
