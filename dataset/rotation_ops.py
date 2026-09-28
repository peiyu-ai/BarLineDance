"""Pure-PyTorch rotation conversions with ``pytorch3d.transforms`` semantics.

Upstream AtomicDance imports these from ``pytorch3d``.  That package has no
wheel for this machine's Python 3.12 / Torch 2.8 / CUDA 12.9 (Blackwell
``sm_120``) environment, and building it from source is more expensive than
re-deriving ten well-defined functions.  Without them ``decode_motion`` raises
on import, so inference writes zero files.

Conventions match pytorch3d exactly, because the released 151-D representation
was produced under them and a silent transpose or quaternion-order change would
corrupt every decoded pose:

* quaternions are **real-first** ``(w, x, y, z)``;
* rotation matrices act on **column** vectors, i.e. ``matrix @ point``;
* the 6-D representation stores the first two **rows** of the matrix
  (Zhou et al. 2019), matching ``matrix[..., :2, :].reshape(..., 6)``;
* ``matrix_to_quaternion`` returns a standardised quaternion (non-negative
  real part), so ``matrix_to_axis_angle`` yields the canonical rotation vector
  with angle in ``[0, pi]`` -- the same convention as ``scipy``'s ``as_rotvec``.

``tools/preprocess_wild_3d.py`` and ``tools/discover_kinematic_atomics.py``
already carry numpy versions of the 6-D convention; this module is the torch
counterpart used by the training/inference path.  ``tests/test_rotation_ops.py``
pins every function against ``scipy.spatial.transform.Rotation`` as an
independent oracle.
"""

from typing import Union

import torch
import torch.nn.functional as F

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

Device = Union[str, torch.device]


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """``sqrt(max(0, x))`` with a zero subgradient where ``x <= 0``."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert real-first quaternions to rotation matrices ``[..., 3, 3]``."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to standardised real-first quaternions.

    Uses the branch-free formulation that builds all four candidate
    quaternions and selects the one whose component is largest in magnitude,
    which stays numerically stable near the 180-degree singularities where a
    naive ``trace``-based formula loses precision.
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError("invalid rotation matrix shape {}".format(matrix.shape))

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # Four candidate encodings; each row is exact when its own |component| is
    # the largest, so the argmax below always picks a well-conditioned one.
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Pick the representative with non-negative real part (``q ~ -q``)."""
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, without standardisation."""
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product followed by standardisation."""
    return standardize_quaternion(quaternion_raw_multiply(a, b))


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """Conjugate, which inverts a unit quaternion."""
    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Rotate ``point`` (``[..., 3]``) by ``quaternion`` (``[..., 4]``)."""
    if point.size(-1) != 3:
        raise ValueError("points must have shape [..., 3], got {}".format(point.shape))
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors to real-first quaternions."""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # Taylor expansion of sin(x/2)/x near x = 0, where the quotient is 0/0.
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert real-first quaternions to rotation vectors."""
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    # Divide by the full angle, not the half angle: the vector part of the
    # quaternion is sin(theta/2) * axis and the result must be theta * axis.
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # Taylor expansion of sin(x/2)/x near x = 0, where the quotient is 0/0.
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors to rotation matrices."""
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to rotation vectors with angle in [0, pi]."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 6-D representation to rotation matrix, by Gram-Schmidt.

    The two 3-vectors become the first two **rows** of the result, matching
    ``matrix_to_rotation_6d`` and the numpy versions used by the wild-3D and
    kinematic-discovery tools.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Drop the last row of the rotation matrix; it is recoverable by cross."""
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _elementary_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Rotation about a single named axis, for column vectors."""
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("axis must be one of X, Y, Z; got {!r}".format(axis))

    return torch.stack(flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """Compose three elementary rotations named by ``convention`` (e.g. "YXZ").

    The product is ``R = R[c0] @ R[c1] @ R[c2]``, i.e. the intrinsic
    interpretation, matching ``scipy``'s uppercase sequence strings.
    """
    if euler_angles.shape[-1] != 3 or len(convention) != 3:
        raise ValueError("expected three angles and a three-letter convention")
    if convention[0] == convention[1] or convention[1] == convention[2]:
        raise ValueError("convention {!r} repeats an adjacent axis".format(convention))

    matrices = [
        _elementary_rotation(axis, angle)
        for axis, angle in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def so3_exp_map(log_rot: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Exponential map from so(3) to SO(3).

    Mathematically identical to :func:`axis_angle_to_matrix`; provided under
    pytorch3d's name because callers import it that way.  ``eps`` is accepted
    for signature compatibility -- the quaternion route is already stable at
    small angles via its Taylor branch, so no extra clamping is needed.
    """
    if log_rot.ndim != 2 or log_rot.shape[-1] != 3:
        raise ValueError("so3_exp_map expects shape [N, 3], got {}".format(tuple(log_rot.shape)))
    return axis_angle_to_matrix(log_rot)


def so3_log_map(rotation: torch.Tensor, eps: float = 1e-4, cos_bound: float = 1e-4) -> torch.Tensor:
    """Logarithm map from SO(3) to so(3), returning angles in ``[0, pi]``.

    Mathematically identical to :func:`matrix_to_axis_angle`.  The quaternion
    route stays accurate near ``pi``, where pytorch3d's own trace-based
    implementation is documented to lose precision.
    """
    if rotation.ndim != 3 or rotation.shape[-2:] != (3, 3):
        raise ValueError("so3_log_map expects shape [N, 3, 3], got {}".format(tuple(rotation.shape)))
    return matrix_to_axis_angle(rotation)
