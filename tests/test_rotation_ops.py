"""Pin ``dataset.rotation_ops`` against an independent rotation implementation.

``scipy.spatial.transform.Rotation`` is the oracle: it is a separately authored
library, so agreement to machine precision is real evidence that the pure-torch
replacements carry pytorch3d's conventions rather than a self-consistent but
wrong convention.  A silent transpose or quaternion-order flip here would
corrupt every decoded pose in the 151-D representation, and nothing downstream
would raise.
"""

import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dataset import rotation_ops as ro

SAMPLES = 4096
TOL = 1e-12


def _reference(seed):
    """Random rotations with matching rotvec / matrix / real-first quaternion."""
    rotation = Rotation.random(SAMPLES, random_state=seed)
    quat_xyzw = rotation.as_quat()
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)
    # scipy's quaternion sign is arbitrary; rotation_ops standardises to w >= 0.
    quat_wxyz = np.where(quat_wxyz[:, :1] < 0, -quat_wxyz, quat_wxyz)
    return rotation.as_rotvec(), rotation.as_matrix(), quat_wxyz


class RotationOpsTests(unittest.TestCase):
    def setUp(self):
        self.rotvec, self.matrix, self.quat = _reference(20260808)
        self.t_rotvec = torch.tensor(self.rotvec, dtype=torch.float64)
        self.t_matrix = torch.tensor(self.matrix, dtype=torch.float64)
        self.t_quat = torch.tensor(self.quat, dtype=torch.float64)

    def _assert_close(self, actual, expected, label):
        error = float(np.abs(np.asarray(actual) - np.asarray(expected)).max())
        self.assertLess(error, TOL, "{} max abs error {:.3e}".format(label, error))

    def test_axis_angle_to_matrix(self):
        self._assert_close(ro.axis_angle_to_matrix(self.t_rotvec), self.matrix, "aa->matrix")

    def test_matrix_to_axis_angle(self):
        self._assert_close(ro.matrix_to_axis_angle(self.t_matrix), self.rotvec, "matrix->aa")

    def test_axis_angle_to_quaternion(self):
        self._assert_close(ro.axis_angle_to_quaternion(self.t_rotvec), self.quat, "aa->quat")

    def test_matrix_to_quaternion(self):
        self._assert_close(ro.matrix_to_quaternion(self.t_matrix), self.quat, "matrix->quat")

    def test_quaternion_to_matrix(self):
        self._assert_close(ro.quaternion_to_matrix(self.t_quat), self.matrix, "quat->matrix")

    def test_quaternion_to_axis_angle(self):
        # Regression: an earlier version divided by the half angle instead of
        # the full angle and returned exactly half of every rotation vector.
        self._assert_close(ro.quaternion_to_axis_angle(self.t_quat), self.rotvec, "quat->aa")

    def test_rotation_6d_stores_first_two_rows(self):
        # The row (not column) convention is what ties this module to the
        # numpy implementations in tools/ and to the released 151-D data.
        expected = self.matrix[:, :2, :].reshape(SAMPLES, 6)
        self._assert_close(ro.matrix_to_rotation_6d(self.t_matrix), expected, "matrix->6d")

    def test_rotation_6d_round_trip(self):
        six = ro.matrix_to_rotation_6d(self.t_matrix)
        self._assert_close(ro.rotation_6d_to_matrix(six), self.matrix, "6d->matrix")

    def test_rotation_6d_gram_schmidt_orthonormalises(self):
        """A non-orthonormal 6-D input must still produce a valid rotation."""
        perturbed = ro.matrix_to_rotation_6d(self.t_matrix) + 0.1
        result = ro.rotation_6d_to_matrix(perturbed)
        identity = torch.eye(3, dtype=torch.float64).expand_as(result)
        gram = result @ result.transpose(-1, -2)
        self._assert_close(gram, identity, "6d orthonormality")
        determinant = torch.linalg.det(result)
        self._assert_close(determinant, np.ones(SAMPLES), "6d determinant")

    def test_quaternion_apply_matches_matrix_product(self):
        points = torch.tensor(np.random.RandomState(11).randn(SAMPLES, 3))
        expected = np.einsum("nij,nj->ni", self.matrix, points.numpy())
        self._assert_close(ro.quaternion_apply(self.t_quat, points), expected, "quat apply")

    def test_quaternion_multiply_matches_matrix_composition(self):
        other_rotvec, other_matrix, other_quat = _reference(7)
        composed = ro.quaternion_multiply(self.t_quat, torch.tensor(other_quat))
        expected = np.einsum("nij,njk->nik", self.matrix, other_matrix)
        self._assert_close(ro.quaternion_to_matrix(composed), expected, "quat multiply")

    def test_singularities(self):
        """Identity, near-zero and near-pi angles are the unstable branches."""
        axes = np.random.RandomState(3).randn(256, 3)
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        cases = {
            "identity": np.zeros((8, 3)),
            "tiny": np.random.RandomState(5).randn(256, 3) * 1e-9,
            "near_pi": axes * (np.pi - 1e-9),
            "axis_pi": np.eye(3) * np.pi,
        }
        for label, rotvec in cases.items():
            with self.subTest(case=label):
                expected = Rotation.from_rotvec(rotvec).as_matrix()
                tensor = torch.tensor(rotvec, dtype=torch.float64)
                self._assert_close(ro.axis_angle_to_matrix(tensor), expected, label)
                # Round trip must land on the same rotation, though the raw
                # vector may differ by the 2*pi wrap at exactly pi.
                back = ro.matrix_to_axis_angle(torch.tensor(expected, dtype=torch.float64))
                regenerated = Rotation.from_rotvec(back.numpy()).as_matrix()
                self._assert_close(regenerated, expected, label + " round trip")

    def test_euler_angles_to_matrix_matches_scipy_intrinsic(self):
        """pytorch3d composes R[c0] @ R[c1] @ R[c2], i.e. scipy's uppercase seq."""
        angles = np.random.RandomState(4).uniform(-np.pi, np.pi, size=(SAMPLES, 3))
        tensor = torch.tensor(angles, dtype=torch.float64)
        for convention in ("XYZ", "YXZ", "ZYX", "YZX", "XZY", "ZXY"):
            with self.subTest(convention=convention):
                expected = Rotation.from_euler(convention, angles).as_matrix()
                self._assert_close(
                    ro.euler_angles_to_matrix(tensor, convention), expected, convention
                )

    def test_euler_rejects_repeated_adjacent_axis(self):
        with self.assertRaises(ValueError):
            ro.euler_angles_to_matrix(torch.zeros(4, 3), "XXY")

    def test_so3_exp_and_log_map(self):
        self._assert_close(ro.so3_exp_map(self.t_rotvec), self.matrix, "so3_exp_map")
        self._assert_close(ro.so3_log_map(self.t_matrix), self.rotvec, "so3_log_map")
        self._assert_close(
            ro.so3_exp_map(ro.so3_log_map(self.t_matrix)), self.matrix, "so3 round trip"
        )

    def test_so3_maps_reject_wrong_shapes(self):
        # pytorch3d's so3 maps are strictly [N, 3] / [N, 3, 3]; accepting extra
        # batch dims here would silently reshape a caller's data.
        with self.assertRaises(ValueError):
            ro.so3_exp_map(torch.zeros(2, 4, 3))
        with self.assertRaises(ValueError):
            ro.so3_log_map(torch.zeros(3, 3))

    def test_standardize_quaternion_preserves_rotation(self):
        negated = -self.t_quat
        self._assert_close(
            ro.quaternion_to_matrix(ro.standardize_quaternion(negated)),
            self.matrix,
            "standardize",
        )


class DecodeMotionIntegrationTests(unittest.TestCase):
    """The consumers must import successfully without pytorch3d installed."""

    def test_dataset_quaternion_helpers_round_trip(self):
        from dataset.quaternion import ax_from_6v, ax_to_6v

        rotvec, matrix, _ = _reference(99)
        six = ax_to_6v(torch.tensor(rotvec, dtype=torch.float64))
        expected = matrix[:, :2, :].reshape(SAMPLES, 6)
        self.assertLess(float(np.abs(six.numpy() - expected).max()), TOL)
        recovered = ax_from_6v(six)
        self.assertLess(float(np.abs(recovered.numpy() - rotvec).max()), TOL)

    def test_smpl_skeleton_forward_kinematics_matches_numpy(self):
        """vis.SMPLSkeleton (torch/quaternion) vs tools/ (numpy/matrix) FK."""
        from tools.preprocess_wild_3d import forward_kinematics
        from vis import SMPLSkeleton

        frames = 40
        rotvec = Rotation.random(frames * 24, random_state=17).as_rotvec()
        rotvec = rotvec.reshape(frames, 24, 3)
        root = np.random.RandomState(2).randn(frames, 3)

        torch_pose = SMPLSkeleton().forward(
            torch.tensor(rotvec, dtype=torch.float32).unsqueeze(0),
            torch.tensor(root, dtype=torch.float32).unsqueeze(0),
        )[0].numpy()

        matrices = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()
        numpy_pose = forward_kinematics(matrices.reshape(frames, 24, 3, 3), root)

        error = float(np.abs(torch_pose.astype(np.float64) - numpy_pose).max())
        self.assertLess(error, 1e-4, "FK mismatch {:.3e}".format(error))


if __name__ == "__main__":
    unittest.main()
