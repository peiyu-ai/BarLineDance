import unittest

import numpy as np

from tools.convert_motion_to_guofeats import (
    GUOFEATS_DIM,
    GuofeatsError,
    humanml3d_to_zup,
    motion_151_to_guofeats,
    motion_151_to_joints,
    resample_joints,
    zup_to_humanml3d,
)

TMR_AVAILABLE = True
try:  # the vendored reference is a runtime dependency, not project source
    from tools.convert_motion_to_guofeats import _load_upstream

    _load_upstream()
except Exception:  # pragma: no cover - environment without third_party/TMR
    TMR_AVAILABLE = False


def _synthetic_motion(frames=60, seed=0):
    """A 151-D clip: binary contacts, a moving root, small joint rotations."""
    rng = np.random.default_rng(seed)
    contacts = (rng.random((frames, 4)) > 0.5).astype(np.float32)
    root = np.stack([
        np.linspace(0.0, 1.0, frames),
        np.linspace(0.0, 0.4, frames),
        0.9 + 0.05 * np.sin(np.linspace(0, 6.0, frames)),
    ], axis=-1).astype(np.float32)
    # 6-D rotations near identity: first two columns of the identity matrix.
    rot6d = np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), (frames, 24, 1))
    rot6d += rng.normal(scale=0.02, size=rot6d.shape).astype(np.float32)
    return np.concatenate([contacts, root, rot6d.reshape(frames, -1)], axis=1)


class AxisConventionTests(unittest.TestCase):
    def test_zup_to_humanml3d_is_invertible(self):
        rng = np.random.default_rng(1)
        joints = rng.normal(size=(7, 22, 3))
        np.testing.assert_allclose(humanml3d_to_zup(zup_to_humanml3d(joints)), joints)

    def test_mapping_matches_tmr_decoder_tail(self):
        # TMR's guofeats_to_joints unbinds (x, z, my) and returns (x, -my, z).
        # Ours must be exactly that inverse, or the corpus lands mirrored.
        joints = np.array([[[1.0, 2.0, 3.0]]])
        np.testing.assert_allclose(zup_to_humanml3d(joints), [[[1.0, 3.0, -2.0]]])

    def test_vertical_axis_moves_from_z_to_y(self):
        standing = np.zeros((1, 22, 3))
        standing[0, :, 2] = np.linspace(0.0, 1.7, 22)  # z-up: height is z
        converted = zup_to_humanml3d(standing)
        self.assertGreater(converted[0, -1, 1], 1.6)   # y-up: height is y
        np.testing.assert_allclose(converted[0, :, 2], 0.0)

    def test_conversion_preserves_chirality(self):
        # A left-right asymmetry must keep its sign; a mirrored corpus is the
        # failure mode of copying HumanML3D's AMASS-only x-flip.
        joints = np.zeros((1, 22, 3))
        joints[0, 16] = [0.2, 0.0, 1.4]   # left shoulder at +x
        joints[0, 17] = [-0.2, 0.0, 1.4]  # right shoulder at -x
        converted = zup_to_humanml3d(joints)
        self.assertGreater(converted[0, 16, 0], converted[0, 17, 0])


class ResampleTests(unittest.TestCase):
    def test_thirty_to_twenty_fps_keeps_duration_and_interpolates(self):
        joints = np.zeros((31, 22, 3))
        joints[:, 0, 0] = np.arange(31)  # ramp so interpolation is checkable
        resampled, index = resample_joints(joints, 30.0, 20.0)
        self.assertEqual(len(resampled), 21)          # 1.0 s at 20 fps + 1
        np.testing.assert_allclose(index[:3], [0.0, 1.5, 3.0])
        np.testing.assert_allclose(resampled[1, 0, 0], 1.5)

    def test_identical_rates_are_a_passthrough(self):
        joints = np.random.default_rng(2).normal(size=(10, 22, 3))
        resampled, index = resample_joints(joints, 30.0, 30.0)
        np.testing.assert_array_equal(resampled, joints)
        np.testing.assert_array_equal(index, np.arange(10))

    def test_one_frame_cannot_be_resampled(self):
        with self.assertRaises(GuofeatsError):
            resample_joints(np.zeros((1, 22, 3)), 30.0, 20.0)


class DecodeTests(unittest.TestCase):
    def test_151d_decodes_to_24_joints(self):
        joints = motion_151_to_joints(_synthetic_motion(frames=12))
        self.assertEqual(joints.shape, (12, 24, 3))
        self.assertTrue(np.isfinite(joints).all())

    def test_wrong_width_is_refused(self):
        with self.assertRaises(GuofeatsError):
            motion_151_to_joints(np.zeros((10, 150), dtype=np.float32))


@unittest.skipUnless(TMR_AVAILABLE, "third_party/TMR reference transform not installed")
class BridgeTests(unittest.TestCase):
    def test_produces_263d_features_at_the_expected_length(self):
        motion = _synthetic_motion(frames=61)          # 2.0 s at 30 fps
        payload = motion_151_to_guofeats(motion)
        features = payload["features"]
        self.assertEqual(features.shape[1], GUOFEATS_DIM)
        # 2.0 s at 20 fps is 41 samples; guofeats consumes one to velocity.
        self.assertEqual(len(features), 40)
        self.assertTrue(np.isfinite(features).all())

    def test_every_row_records_the_source_frame_it_came_from(self):
        payload = motion_151_to_guofeats(_synthetic_motion(frames=61))
        index = payload["source_frame_index"]
        self.assertEqual(len(index), len(payload["features"]))
        self.assertAlmostEqual(float(index[0]), 0.0)
        self.assertTrue(np.all(np.diff(index) > 0))
        self.assertLess(float(index[-1]), 61)

    def test_records_both_frame_rates(self):
        payload = motion_151_to_guofeats(_synthetic_motion(frames=61))
        self.assertAlmostEqual(float(payload["source_fps"][0]), 30.0)
        self.assertAlmostEqual(float(payload["target_fps"][0]), 20.0)


if __name__ == "__main__":
    unittest.main()
