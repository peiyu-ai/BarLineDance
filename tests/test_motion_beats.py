import unittest

import numpy as np

from tools.motion_beats import (
    canonical_pose,
    describe_pose,
    descriptor_vector,
    find_motion_beats,
    joint_speed,
    segment_descriptor,
)


def _still(frames=30, joints=22):
    pose = np.zeros((joints, 3))
    pose[:, 2] = np.linspace(0.0, 1.7, joints)
    pose[1] = [0.1, 0.1, 0.9]    # L hip
    pose[2] = [0.1, -0.1, 0.9]   # R hip
    pose[16] = [0.0, 0.2, 1.4]   # L shoulder
    pose[17] = [0.0, -0.2, 1.4]  # R shoulder
    return np.repeat(pose[None], frames, axis=0)


def _pulsing(frames=60, joints=22):
    """Two clear slow points: a motion that settles twice."""
    base = _still(frames, joints)
    phase = np.sin(np.linspace(0, 4 * np.pi, frames))
    base[:, 20, 2] = 1.0 + 0.5 * phase   # left hand rises and falls twice
    return base


class SpeedTests(unittest.TestCase):
    def test_a_still_body_has_zero_speed(self):
        np.testing.assert_allclose(joint_speed(_still()), 0.0, atol=1e-12)

    def test_speed_is_reported_for_every_frame(self):
        joints = _pulsing(frames=40)
        self.assertEqual(len(joint_speed(joints)), 40)


class BeatTests(unittest.TestCase):
    def test_a_settling_motion_yields_beats_at_its_slow_points(self):
        beats = find_motion_beats(_pulsing(frames=60), max_beats=4)
        self.assertGreaterEqual(len(beats), 1)
        speed = joint_speed(_pulsing(frames=60))
        # Beats must be slower than the segment's own mean, or they are not
        # where the movement settles.
        self.assertLess(float(speed[beats].mean()), float(speed.mean()))

    def test_beats_are_separated(self):
        beats = find_motion_beats(_pulsing(frames=60), min_separation=5, max_beats=4)
        self.assertTrue(all(b - a >= 5 for a, b in zip(beats, beats[1:])))

    def test_a_monotone_segment_still_reports_its_settling_point(self):
        joints = _still(30)
        joints[:, 20, 2] = np.linspace(1.0, 2.0, 30)  # decelerating never happens
        beats = find_motion_beats(joints)
        self.assertEqual(len(beats), 1)

    def test_never_returns_more_than_requested(self):
        beats = find_motion_beats(_pulsing(frames=120), max_beats=3)
        self.assertLessEqual(len(beats), 3)


class CanonicalPoseTests(unittest.TestCase):
    def test_translation_is_removed(self):
        pose = _still()[0]
        moved = pose + np.array([5.0, -3.0, 2.0])
        np.testing.assert_allclose(canonical_pose(pose), canonical_pose(moved), atol=1e-9)

    def test_rotation_about_the_vertical_is_removed(self):
        pose = _still()[0]
        angle = 0.7
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1.0]])
        np.testing.assert_allclose(canonical_pose(pose), canonical_pose(pose @ rotation.T),
                                   atol=1e-9)

    def test_body_scale_is_removed(self):
        pose = _still()[0]
        np.testing.assert_allclose(canonical_pose(pose), canonical_pose(pose * 1.8), atol=1e-9)


class DescriptorTests(unittest.TestCase):
    def test_vector_width_is_fixed_regardless_of_segment_length(self):
        short = descriptor_vector(_pulsing(frames=20))
        long = descriptor_vector(_pulsing(frames=200))
        self.assertEqual(short.shape, long.shape)
        self.assertTrue(np.isfinite(short).all() and np.isfinite(long).all())

    def test_padding_repeats_a_real_pose_rather_than_inventing_a_collapsed_one(self):
        joints = _still(12)
        joints[:, 20, 2] = np.linspace(1.0, 1.4, 12)   # one beat only
        vector = descriptor_vector(joints, max_beats=4)
        poses = vector[:-6].reshape(4, 22, 3)
        # All four slots must be the same real pose, never zeros.
        for index in range(1, 4):
            np.testing.assert_allclose(poses[index], poses[0], atol=1e-9)
        self.assertGreater(np.abs(poses[0]).sum(), 0.0)

    def test_dynamics_capture_travel_and_rise(self):
        joints = _still(30)
        joints[:, :, 0] += np.linspace(0, 2.0, 30)[:, None]   # travels +x
        parts = segment_descriptor(joints)
        self.assertGreater(parts["dynamics"][4], 1.5)          # floor travel
        self.assertAlmostEqual(float(parts["dynamics"][5]), 0.0, places=6)


def _standing():
    """A canonical-space standing skeleton, in shoulder widths off the pelvis.

    The numbers are the medians measured on the wild corpus, so a test that
    passes here is a statement about poses the captioner will really see rather
    than about a stick figure invented to satisfy the rules.
    """
    pose = np.zeros((24, 3))
    pose[1], pose[2] = [0.15, 0.0, -0.22], [-0.15, 0.0, -0.22]     # hips
    pose[4], pose[5] = [0.20, 0.0, -1.25], [-0.20, 0.0, -1.25]     # knees
    pose[7], pose[8] = [0.22, 0.0, -2.30], [-0.22, 0.0, -2.30]     # ankles
    pose[10], pose[11] = [0.35, -0.2, -2.45], [-0.35, -0.2, -2.45]  # feet
    pose[12], pose[15] = [0.0, 0.0, 1.25], [0.0, 0.0, 1.55]        # neck, head
    pose[16], pose[17] = [0.5, 0.0, 1.17], [-0.5, 0.0, 1.17]       # shoulders
    pose[18], pose[19] = [0.6, 0.0, 0.60], [-0.6, 0.0, 0.60]       # elbows
    pose[20], pose[21] = [0.65, 0.0, 0.10], [-0.65, 0.0, 0.10]     # wrists
    pose[22], pose[23] = pose[20], pose[21]
    return pose


class DescribeTests(unittest.TestCase):
    """The cue goes into the VLM prompt, so a fluent lie is worse than silence."""

    def test_hands_above_the_head_read_as_overhead(self):
        pose = _standing()
        pose[20], pose[21] = [0.3, 0.0, 2.0], [-0.3, 0.0, 2.0]
        self.assertEqual(describe_pose(pose).count("overhead"), 2)

    def test_hands_at_the_hips_are_not_called_overhead(self):
        """The old rule compared against a constant and fired on 69% of poses."""
        text = describe_pose(_standing())
        self.assertNotIn("overhead", text)
        self.assertIn("left arm low", text)

    def test_a_t_pose_reads_as_extended_sideways(self):
        pose = _standing()
        pose[20], pose[21] = [1.7, 0.0, 1.17], [-1.7, 0.0, 1.17]
        self.assertEqual(describe_pose(pose).count("extended sideways"), 2)

    def test_arms_reaching_forward_are_not_called_sideways(self):
        """x is the left-right axis and y is front-back; swapping them yields
        sentences that are fluent and false."""
        pose = _standing()
        pose[20], pose[21] = [0.3, -1.7, 0.9], [-0.3, -1.7, 0.9]
        text = describe_pose(pose)
        self.assertIn("reaching forward", text)
        self.assertNotIn("sideways", text)

    def test_one_arm_up_one_arm_down_is_not_symmetric(self):
        pose = _standing()
        pose[20] = [0.3, 0.0, 2.0]
        text = describe_pose(pose)
        self.assertIn("left arm raised overhead", text)
        self.assertIn("right arm low", text)

    def test_a_crouch_is_reported_and_standing_is_not(self):
        self.assertNotIn("knees bent", describe_pose(_standing()))
        # A real crouch keeps the bones the length they were and folds them:
        # thigh swung forward 45 degrees, shin back the same amount, so the
        # pelvis ends up riding lower over the same feet.
        crouched = _standing()
        crouched[4], crouched[5] = [0.15, -0.73, -0.95], [-0.15, -0.73, -0.95]
        crouched[7], crouched[8] = [0.15, 0.01, -1.69], [-0.15, 0.01, -1.69]
        crouched[10], crouched[11] = [0.35, -0.19, -1.84], [-0.35, -0.19, -1.84]
        self.assertIn("knees bent low", describe_pose(crouched))

    def test_stance_width_reads_the_lateral_axis(self):
        wide = _standing()
        wide[10], wide[11] = [1.3, -0.2, -2.45], [-1.3, -0.2, -2.45]
        self.assertIn("feet wide apart", describe_pose(wide))
        together = _standing()
        together[10], together[11] = [0.1, -0.2, -2.45], [-0.1, -0.2, -2.45]
        self.assertIn("feet together", describe_pose(together))

    def test_a_lifted_foot_is_reported(self):
        pose = _standing()
        pose[10] = [0.35, -0.2, -1.5]
        self.assertIn("one foot lifted", describe_pose(pose))

    def test_the_description_survives_the_canonicalisation_it_is_fed_through(self):
        """describe_pose is always called on canonical_pose output."""
        pose = _standing()
        angle = 1.1
        cos, sin = np.cos(angle), np.sin(angle)
        turned = pose @ np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1.0]]).T
        self.assertEqual(describe_pose(canonical_pose(pose)),
                         describe_pose(canonical_pose(turned)))

    def test_output_is_a_sentence_not_an_empty_string(self):
        self.assertTrue(len(describe_pose(canonical_pose(_still()[0]))) > 0)


if __name__ == "__main__":
    unittest.main()


class ProminenceTests(unittest.TestCase):
    def test_arithmetic_noise_on_a_constant_speed_limb_yields_one_beat(self):
        # np.linspace's last-ulp wobble made this report a full quota of
        # spurious beats before prominence was required.
        joints = _still(30)
        joints[:, 20, 2] = np.linspace(1.0, 2.0, 30)
        self.assertEqual(len(find_motion_beats(joints)), 1)

    def test_a_real_dip_still_registers(self):
        joints = _still(60)
        speed_profile = np.concatenate([
            np.full(20, 0.05), np.full(5, 0.001), np.full(35, 0.05)])
        joints[:, 20, 2] = 1.0 + np.cumsum(speed_profile)
        beats = find_motion_beats(joints)
        self.assertTrue(any(18 <= b <= 30 for b in beats),
                        "the deliberate pause should be found, got {}".format(beats))
