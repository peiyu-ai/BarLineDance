import unittest

import numpy as np

from tools.audit_wild_3d_quality import FPS, TOES, sequence_metrics, summarise


def standing(frames=120, height=1.0):
    """A still figure whose toes sit at ``height`` and head one metre above."""
    joints = np.zeros((frames, 24, 3), dtype=np.float64)
    joints[:, :, 2] = height + 0.9          # everything at hip level by default
    joints[:, TOES, 2] = height             # toes on the floor
    joints[:, 15, 2] = height + 1.6         # head
    return joints


class FloorTests(unittest.TestCase):
    def test_the_floor_is_read_from_the_data_not_assumed_at_zero(self):
        # The corpus floats ~0.33 m above the world origin.  Measuring
        # penetration against z = 0 would report every clip as flying.
        for height in (0.0, 0.33, 1.7):
            metrics = sequence_metrics(standing(height=height))
            self.assertAlmostEqual(metrics["floor_z"], height, places=3)
            self.assertEqual(metrics["penetration_fraction"], 0.0)

    def test_body_height_is_measured_from_the_floor_so_an_offset_cancels(self):
        low = sequence_metrics(standing(height=0.0))["body_height"]
        high = sequence_metrics(standing(height=1.7))["body_height"]
        self.assertAlmostEqual(low, high, places=3)
        self.assertAlmostEqual(low, 1.6, places=3)

    def test_a_brief_punch_through_the_floor_is_counted(self):
        joints = standing(frames=600)
        joints[300:303, TOES[0], 2] -= 0.2          # 0.5% of frames
        metrics = sequence_metrics(joints)
        self.assertGreater(metrics["penetration_fraction"], 0.0)
        self.assertGreater(metrics["penetration_depth_max"], 0.1)
        # A three-frame event does not move the p99 at all, which is why the
        # max is reported beside it.
        self.assertEqual(metrics["penetration_depth_p99"], 0.0)

    def test_sustained_penetration_is_absorbed_by_the_floor_estimate(self):
        # The blind spot, pinned on purpose.  The floor is the 2nd percentile of
        # toe height, so anything below it for more than ~2% of the clip simply
        # redefines the floor.  Without ground truth there is no way to tell a
        # sunken body from a low stage, and pretending otherwise would put a
        # confident zero next to a case this column never examined.
        joints = standing(frames=120)
        joints[:60, TOES[0], 2] -= 0.2              # half the clip, well under
        metrics = sequence_metrics(joints)
        self.assertEqual(metrics["penetration_fraction"], 0.0)
        # ...but it is not invisible: the wander column sees it.
        self.assertGreater(metrics["lowest_toe_spread"], 0.15)


class WanderTests(unittest.TestCase):
    def test_a_planted_figure_has_no_vertical_wander(self):
        self.assertLess(sequence_metrics(standing())["lowest_toe_spread"], 1e-6)

    def test_a_body_riding_up_and_down_opens_the_spread(self):
        # The failure mode monocular root translation actually has: the whole
        # body drifts vertically while the pose is fine.
        joints = standing()
        drift = 0.25 * np.sin(np.linspace(0, 4 * np.pi, len(joints)))
        joints[:, :, 2] += drift[:, None]
        self.assertGreater(sequence_metrics(joints)["lowest_toe_spread"], 0.3)


class SkateTests(unittest.TestCase):
    def test_a_still_planted_foot_scores_zero_skate(self):
        self.assertLess(sequence_metrics(standing())["skate_p95"], 1e-6)

    def test_a_foot_sliding_while_on_the_floor_is_charged(self):
        joints = standing()
        joints[:, TOES[0], 0] = np.linspace(0, 1.0, len(joints))   # 1 m over 4 s
        metrics = sequence_metrics(joints)
        self.assertGreater(metrics["skate_p95"], 0.2)

    def test_a_lifted_foot_moving_fast_is_not_charged_as_skate(self):
        # A swinging foot is supposed to move; charging it would make every
        # fast dance look broken, which is the whole point of gating on
        # "planted".
        joints = standing()
        joints[:, TOES[0], 2] += 0.6                                # lifted well clear
        joints[:, TOES[0], 0] = np.linspace(0, 3.0, len(joints))    # and swinging
        self.assertLess(sequence_metrics(joints)["skate_p95"], 1e-6)


class JitterTests(unittest.TestCase):
    def test_high_frequency_noise_raises_jitter_relative_to_speed(self):
        rng = np.random.default_rng(0)
        clean = standing()
        clean[:, :, 0] = np.linspace(0, 2.0, len(clean))[:, None]   # smooth travel
        noisy = clean.copy()
        noisy[:, :, :] += rng.normal(scale=0.01, size=noisy.shape)
        self.assertGreater(sequence_metrics(noisy)["jitter_ratio"],
                           sequence_metrics(clean)["jitter_ratio"] + 1.0)

    def test_a_fast_clean_dance_is_not_flagged_by_the_ratio(self):
        # Absolute acceleration scales with tempo, so an absolute threshold
        # would flag exactly the clips with the most content.  The ratio must
        # stay put when the same motion is played faster.
        base = standing(frames=240)
        base[:, :, 0] = np.sin(np.linspace(0, 6 * np.pi, len(base)))[:, None]
        fast = standing(frames=240)
        fast[:, :, 0] = np.sin(np.linspace(0, 12 * np.pi, len(fast)))[:, None]
        slow_ratio = sequence_metrics(base)["jitter_ratio"]
        fast_ratio = sequence_metrics(fast)["jitter_ratio"]
        self.assertGreater(sequence_metrics(fast)["jitter_median"],
                           sequence_metrics(base)["jitter_median"])
        self.assertLess(abs(fast_ratio - 2 * slow_ratio), 0.5 * slow_ratio)


class TrackingFailureTests(unittest.TestCase):
    def test_a_repeated_pose_is_counted_as_frozen(self):
        joints = standing()
        joints[:, :, 0] = np.linspace(0, 2.0, len(joints))[:, None]
        joints[40:80] = joints[40]                       # tracker stuck
        self.assertGreater(sequence_metrics(joints)["frozen_fraction"], 0.3)

    def test_a_pelvis_teleport_shows_up_in_peak_root_speed(self):
        joints = standing()
        joints[60:, 0, 0] += 3.0                         # one-frame jump
        self.assertGreater(sequence_metrics(joints)["root_speed_max"], 3.0 * FPS * 0.9)


class SummaryTests(unittest.TestCase):
    def test_failed_sequences_are_counted_not_silently_dropped(self):
        records = [dict(sequence_metrics(standing()), recording_id="a"),
                   {"recording_id": "b", "error": "GuofeatsError()"}]
        summary = summarise(records, "wild")
        self.assertEqual(summary["sequences"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_gauge_reports_spread_across_clips_not_within_one(self):
        # A corpus whose floor moves clip to clip is not one world, and that is
        # a different question from any single clip's internal consistency.
        records = [dict(sequence_metrics(standing(height=h)), recording_id=str(h))
                   for h in (0.20, 0.30, 0.40, 0.50)]
        gauge = summarise(records, "wild")["gauge"]
        self.assertGreater(gauge["floor_z_spread_p95_minus_p05"], 0.2)
        self.assertLess(gauge["body_height_std"], 1e-6)


if __name__ == "__main__":
    unittest.main()
