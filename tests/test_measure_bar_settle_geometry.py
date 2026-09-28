"""The geometry tool has to be able to tell an aligned corpus from a random one.

Every case here is a synthetic clip whose answer is known before the tool runs,
because the point of the tool is to decide whether a real corpus lets the
operator's bar-grid-plus-snap proposal work.  A tool that reads the same on
aligned and unaligned footage would say "the snap is free" either way.
"""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.measure_bar_settle_geometry import (  # noqa: E402
    GeometryError, measure_clip, null_distances, run)

FPS = 30.0
BEAT_CHANNEL = 34
PERIOD = 16  # frames; 112.5 BPM, the tempo this corpus actually sits at


def music_with_beats(frames, period=PERIOD, offset=0, downbeat_every=4):
    """Beats every ``period`` frames, with the onset envelope peaking on the
    downbeat.  The peak is not decoration: ``choose_phase`` picks the bar line
    by onset energy, and a flat envelope leaves it choosing arbitrarily, which
    would make a test about settle geometry silently a test about that tie.
    """
    music = np.zeros((frames, 35), dtype=np.float32)
    music[offset::period, BEAT_CHANNEL] = 1.0
    music[:, 0] = 0.1
    music[offset::period * downbeat_every, 0] = 1.0
    return music


def joints_settling_at(frames, settle_at, *, joints_count=22, speed=1.0, dip=0.02):
    """A body moving at a constant speed that dips to nearly nothing at each frame in
    ``settle_at``.  Positions are the running integral of that speed."""
    profile = np.full(frames, speed, dtype=np.float64)
    for centre in settle_at:
        for delta in (-1, 0, 1):
            index = centre + delta
            if 0 <= index < frames:
                profile[index] = dip
    positions = np.cumsum(profile)
    out = np.zeros((frames, joints_count, 3), dtype=np.float32)
    out[:, :, 0] = positions[:, None]
    return out


class NullDistanceTests(unittest.TestCase):
    def test_no_points_gives_no_distances(self):
        got = null_distances(np.array([10, 20]), 0, 100,
                             np.random.default_rng(0), draws=3)
        self.assertEqual(len(got), 0)

    def test_draw_count_and_size(self):
        got = null_distances(np.array([10, 20, 30]), 5, 200,
                             np.random.default_rng(0), draws=7)
        self.assertEqual(len(got), 3 * 7)
        self.assertTrue((got >= 0).all())

    def test_asking_for_more_points_than_frames_does_not_raise(self):
        got = null_distances(np.array([2]), 500, 10, np.random.default_rng(0), draws=2)
        self.assertEqual(len(got), 2)


class MeasureClipTests(unittest.TestCase):
    def _measure(self, joints, music, **kwargs):
        options = {"beats_per_segment": 4, "min_length": 8, "prominence": 0.30,
                   "separation": 8, "draws": 200,
                   "generator": np.random.default_rng(20260902)}
        options.update(kwargs)
        return measure_clip(joints, music, **options)

    def test_positive_control_settles_on_every_bar_line(self):
        """The case the tool exists to recognise: the dancer lands exactly on the
        bar lines.  Distance must be ~0 and must beat its own null by a mile."""
        frames = 16 * PERIOD
        bars = list(range(0, frames, 4 * PERIOD))
        row = self._measure(joints_settling_at(frames, bars), music_with_beats(frames))
        self.assertIsNotNone(row)
        self.assertLessEqual(row["distance_median_frames"], 1.0)
        self.assertGreater(row["distance_median_frames_null"],
                           row["distance_median_frames"] + 2.0)
        self.assertEqual(row["beyond_half_beat"], 0.0)
        self.assertGreaterEqual(row["within_3"], 0.99)

    def test_negative_control_settles_half_a_bar_off(self):
        """Same settle count, placed as far from the bar lines as they can be.
        The tool must NOT report these as close, or it cannot fail."""
        frames = 16 * PERIOD
        offbeat = list(range(2 * PERIOD, frames, 4 * PERIOD))
        row = self._measure(joints_settling_at(frames, offbeat), music_with_beats(frames))
        self.assertIsNotNone(row)
        self.assertGreater(row["distance_median_frames"], PERIOD / 2.0)
        self.assertEqual(row["beyond_half_beat"], 1.0)
        self.assertLessEqual(row["within_6"], 0.01)

    def test_aligned_and_offbeat_are_distinguished(self):
        """The two controls above, compared directly: the ordering is the claim."""
        frames = 16 * PERIOD
        aligned = self._measure(
            joints_settling_at(frames, list(range(0, frames, 4 * PERIOD))),
            music_with_beats(frames))
        offbeat = self._measure(
            joints_settling_at(frames, list(range(2 * PERIOD, frames, 4 * PERIOD))),
            music_with_beats(frames))
        self.assertLess(aligned["distance_median_frames"],
                        offbeat["distance_median_frames"])

    def test_beat_period_is_reported_in_frames(self):
        frames = 16 * PERIOD
        row = self._measure(
            joints_settling_at(frames, list(range(0, frames, 4 * PERIOD))),
            music_with_beats(frames))
        self.assertAlmostEqual(row["beat_period_frames"], float(PERIOD))

    def test_too_few_beats_returns_none_rather_than_a_number(self):
        frames = 3 * PERIOD
        music = np.zeros((frames, 35), dtype=np.float32)
        music[0, BEAT_CHANNEL] = 1.0
        self.assertIsNone(self._measure(joints_settling_at(frames, [10]), music))

    def test_a_body_that_never_settles_returns_none(self):
        frames = 16 * PERIOD
        joints = joints_settling_at(frames, [])
        self.assertIsNone(self._measure(joints, music_with_beats(frames)))


class PhaseNullTests(unittest.TestCase):
    """The best of k bar lines is a MAXIMUM, so it reads high on any footage.

    These two cases are the reason the null is computed at all: without it the
    aligned and the unaligned corpus both return "the best phase lands a
    quarter of the boundaries on a settle" and the number looks like a finding
    either way.
    """

    def _measure(self, joints, music):
        return measure_clip(joints, music, beats_per_segment=4, min_length=8,
                            prominence=0.30, separation=8, draws=100,
                            generator=np.random.default_rng(20260902), tolerance=2)

    def test_positive_control_a_phase_that_really_holds_the_landings_beats_its_null(self):
        frames = 24 * PERIOD
        row = self._measure(
            joints_settling_at(frames, list(range(0, frames, 4 * PERIOD))),
            music_with_beats(frames))
        self.assertGreater(row["phase_best_share"], row["phase_best_share_null"] + 0.3)
        self.assertTrue(row["phase_best_beats_its_null"])

    def test_negative_control_scattered_landings_do_not_beat_the_null(self):
        """Settles placed by the same process the null uses.  The best phase
        must NOT look like a finding here."""
        frames = 24 * PERIOD
        generator = np.random.default_rng(7)
        scattered = sorted(generator.choice(np.arange(1, frames - 1), size=6,
                                            replace=False).tolist())
        row = self._measure(joints_settling_at(frames, scattered),
                            music_with_beats(frames))
        self.assertLess(row["phase_best_share"], row["phase_best_share_null"] + 0.25)

    def test_both_phase_columns_carry_their_own_null(self):
        frames = 24 * PERIOD
        row = self._measure(
            joints_settling_at(frames, list(range(0, frames, 4 * PERIOD))),
            music_with_beats(frames))
        for key in ("phase_best_share", "phase_energy_share"):
            self.assertIn(key + "_null", row)
            self.assertIsNotNone(row[key + "_null"])


class RunTests(unittest.TestCase):
    def _write(self, directory, name, joints, music):
        import pickle
        (directory / "motion").mkdir(parents=True, exist_ok=True)
        (directory / "audio").mkdir(parents=True, exist_ok=True)
        with (directory / "motion" / (name + ".pkl")).open("wb") as handle:
            pickle.dump({"full_pose": joints}, handle)
        np.save(directory / "audio" / (name + ".npy"), music)

    def test_run_aggregates_and_keeps_the_null_beside_the_reading(self):
        import tempfile
        frames = 16 * PERIOD
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            for index in range(3):
                self._write(root, "clip{}".format(index),
                            joints_settling_at(frames, list(range(0, frames, 4 * PERIOD))),
                            music_with_beats(frames))
            report = run(["clip0", "clip1", "clip2"], root / "motion", root / "audio",
                         beats_per_segment=4, min_length=8, prominence=0.30,
                         separation=8, draws=50, seed=1)
        self.assertEqual(report["summary"]["clips"], 3)
        self.assertLessEqual(report["summary"]["distance_median_frames"], 1.0)
        self.assertGreater(report["summary"]["distance_median_frames_null"],
                           report["summary"]["distance_median_frames"])
        self.assertEqual(len(report["per_clip"]), 3)

    def test_run_refuses_an_empty_corpus_instead_of_reporting_zero(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            (root / "motion").mkdir()
            (root / "audio").mkdir()
            with self.assertRaises(GeometryError):
                run(["missing"], root / "motion", root / "audio",
                    beats_per_segment=4, min_length=8, prominence=0.30,
                    separation=8, draws=10, seed=1)


if __name__ == "__main__":
    unittest.main()
