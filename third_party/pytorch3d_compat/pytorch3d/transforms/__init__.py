"""``pytorch3d.transforms`` surface, backed by ``dataset.rotation_ops``.

Every name here is the repo's own implementation, verified elementwise against
``scipy.spatial.transform.Rotation`` in ``tests/test_rotation_ops.py``.  The
conventions are pytorch3d's: real-first quaternions, matrices acting on column
vectors, and a 6-D representation holding the first two rows.
"""

from dataset.rotation_ops import (  # noqa: F401
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_raw_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    so3_exp_map,
    so3_log_map,
    standardize_quaternion,
)

__all__ = [
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "euler_angles_to_matrix",
    "matrix_to_axis_angle",
    "matrix_to_quaternion",
    "matrix_to_rotation_6d",
    "quaternion_apply",
    "quaternion_invert",
    "quaternion_multiply",
    "quaternion_raw_multiply",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "rotation_6d_to_matrix",
    "so3_exp_map",
    "so3_log_map",
    "standardize_quaternion",
]
