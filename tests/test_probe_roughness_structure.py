"""The decompositions have to separate what they claim to separate.

Each test builds a signal whose answer is known by construction, because the
conclusion this probe produced -- that 90% of the whole-body jerk is three of
the 151 dimensions -- is only worth anything if the split is arithmetically what
it says it is.
"""

import pathlib
import pickle
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import probe_roughness_structure as probe


def _clip(root, articulation, frames=200, joints=24, seed=0):
    """A pose whose root and articulation are independently controllable."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames)[:, None]
    base = np.zeros((frames, joints, 3))
    base += 0.3 * np.sin(t[..., None] / 12.0)
    base += articulation * rng.normal(size=(frames, joints, 3))
    base[:, 0] = 0.0
    trajectory = 0.5 * np.sin(t / 20.0) + root * rng.normal(size=(frames, 3))
    return base + trajectory[:, None, :]


def _write(directory, name, pose, labels=None):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"full_pose": pose.astype(np.float32)}
    if labels is not None:
        payload["atomic_labels"] = np.asarray(labels, dtype=np.int64)
    with open(directory / "{}.pkl".format(name), "wb") as handle:
        pickle.dump(payload, handle)


class RootSplitTests(unittest.TestCase):
    def test_a_jittering_root_lands_in_the_root_term_not_the_articulation(self):
        smooth = _clip(root=0.0, articulation=0.004, seed=1)
        jittery = smooth.copy()
        rng = np.random.default_rng(2)
        jitter = 0.02 * rng.normal(size=(len(smooth), 3))
        jittery += jitter[:, None, :]

        clips = {"a": {"full_pose": smooth}, "b": {"full_pose": jittery}}
        smooth_out = probe.root_vs_articulation({"a": clips["a"]}, {})
        jitter_out = probe.root_vs_articulation({"b": clips["b"]}, {})

        self.assertGreater(jitter_out["root_jerk"], 10 * smooth_out["root_jerk"])
        # Adding the same offset to every joint cannot change the pose.
        self.assertAlmostEqual(jitter_out["articulation_jerk"],
                               smooth_out["articulation_jerk"], places=6)

    def test_root_share_reports_where_the_whole_body_number_comes_from(self):
        pose = _clip(root=0.02, articulation=0.001, seed=3)
        out = probe.root_vs_articulation({"a": {"full_pose": pose}}, {})
        self.assertGreater(out["root_share_of_whole_body"], 0.7)

    def test_a_common_mode_offset_flattens_the_world_profile_but_not_the_local_one(self):
        """The measured claim: 'the profile is flat' and 'the root jitters' are one fact."""
        pose = _clip(root=0.0, articulation=0.004, seed=4)
        rng = np.random.default_rng(5)
        shaken = pose + (0.02 * rng.normal(size=(len(pose), 3)))[:, None, :]
        clean = probe.root_vs_articulation({"a": {"full_pose": pose}}, {})
        noisy = probe.root_vs_articulation({"a": {"full_pose": shaken}}, {})
        self.assertLess(noisy["cv_world"], clean["cv_world"])
        self.assertAlmostEqual(noisy["cv_root_relative"], clean["cv_root_relative"], places=6)


class TemporalTests(unittest.TestCase):
    def test_covered_and_transition_frames_are_scored_separately(self):
        pose = _clip(root=0.0, articulation=0.002, seed=6)
        labels = np.zeros(len(pose), dtype=np.int64)
        labels[:100] = 7
        rng = np.random.default_rng(7)
        pose[:100] += 0.01 * rng.normal(size=(100, 24, 3))
        out = probe.temporal({"a": {"full_pose": pose, "atomic_labels": labels}}, 2)
        self.assertGreater(out["covered_jerk_median"], out["transition_jerk_median"])
        self.assertEqual(out["clips_where_covered_is_rougher"], 1)

    def test_frames_next_to_a_label_change_are_excluded_from_both_sides(self):
        pose = _clip(root=0.0, articulation=0.002, seed=8)
        labels = np.zeros(len(pose), dtype=np.int64)
        labels[:100] = 7
        wide = probe.temporal({"a": {"full_pose": pose, "atomic_labels": labels}}, 10)
        narrow = probe.temporal({"a": {"full_pose": pose, "atomic_labels": labels}}, 0)
        wide_frames = wide["per_clip"][0]["covered_frames"] + wide["per_clip"][0]["transition_frames"]
        narrow_frames = narrow["per_clip"][0]["covered_frames"] + narrow["per_clip"][0]["transition_frames"]
        self.assertLess(wide_frames, narrow_frames)

    def test_a_clip_without_a_plan_is_counted_not_silently_dropped(self):
        pose = _clip(root=0.0, articulation=0.002, seed=9)
        labels = np.zeros(len(pose), dtype=np.int64)
        labels[:100] = 7
        out = probe.temporal({"a": {"full_pose": pose, "atomic_labels": labels},
                              "b": {"full_pose": pose}}, 2)
        self.assertEqual(out["clips_without_labels"], 1)

    def test_a_clip_with_no_transition_frames_cannot_produce_a_ratio(self):
        pose = _clip(root=0.0, articulation=0.002, seed=10)
        labels = np.full(len(pose), 7, dtype=np.int64)
        with self.assertRaises(probe.ProbeError):
            probe.temporal({"a": {"full_pose": pose, "atomic_labels": labels}}, 2)


class SpectralTests(unittest.TestCase):
    def test_added_high_frequency_noise_shows_up_in_the_top_band(self):
        pose = _clip(root=0.0, articulation=0.0, seed=11)
        rng = np.random.default_rng(12)
        noisy = pose + 0.01 * rng.normal(size=pose.shape)
        clips = {"a": {"full_pose": noisy}}
        truth = {"a": pose}
        out = probe.spectral(clips, truth, 128)
        ratios = out["band_ratio_generated_over_truth"]
        self.assertGreater(ratios["10-15Hz"], ratios["0-1Hz"])
        self.assertGreater(out["highest_band_over_lowest"], 2.0)

    def test_a_clip_shorter_than_the_segment_refuses(self):
        pose = _clip(root=0.0, articulation=0.002, frames=40, seed=13)
        with self.assertRaises(probe.ProbeError):
            probe.spectral({"a": {"full_pose": pose}}, {"a": pose}, 128)


class PrimitiveTests(unittest.TestCase):
    def test_the_median_is_what_makes_the_pi_flip_harmless(self):
        """Why there is no unwrap in this file.

        A field that flips between the two axis-angle representations of the
        same rotation gives the same median third difference as one that does
        not.  Two unwraps were written and deleted before this test existed;
        both were wrong, and neither was needed.  If the statistic is ever
        changed from a median to a mean, this test goes red and an unwrap has
        to come back with it.
        """
        frames = 240
        angle = np.linspace(0.0, 10 * 2 * np.pi, frames)
        axis = np.tile(np.array([0.0, 0.0, 1.0]), (frames, 1))
        continuous = (angle[:, None] * axis)[:, None, :]
        wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
        flipping = (wrapped_angle[:, None] * axis)[:, None, :]

        smooth = probe.angular_jerk_per_joint(continuous)[0]
        flipped = probe.angular_jerk_per_joint(flipping)[0]
        self.assertAlmostEqual(smooth, flipped, places=6)
        # ... while the maximum, which is not what this file uses, is wrecked.
        raw_max = np.linalg.norm(np.diff(flipping, n=3, axis=0), axis=-1).max()
        smooth_max = np.linalg.norm(np.diff(continuous, n=3, axis=0), axis=-1).max()
        self.assertGreater(raw_max, 100 * max(smooth_max, 1e-9))

    def test_kinematic_depth_matches_the_smpl_tree(self):
        depth = probe.kinematic_depth()
        self.assertEqual(depth[0], 0)
        self.assertEqual(depth[1], 1)      # left hip
        self.assertEqual(depth[20], 7)     # left wrist
        self.assertEqual(len(depth), 24)

    def test_lever_arm_of_the_root_is_zero(self):
        pose = _clip(root=0.01, articulation=0.002, seed=14)
        self.assertAlmostEqual(probe.lever_arms(pose)[0], 0.0, places=9)


class LoadingTests(unittest.TestCase):
    def test_an_empty_run_directory_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(probe.ProbeError):
                probe.load_generated(pathlib.Path(tmp))

    def test_generated_clips_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "run"
            _write(root, "clip000", _clip(root=0.0, articulation=0.002, seed=15))
            clips = probe.load_generated(root)
            self.assertEqual(list(clips), ["clip000"])
            self.assertIn("full_pose", clips["clip000"])


if __name__ == "__main__":
    unittest.main()
