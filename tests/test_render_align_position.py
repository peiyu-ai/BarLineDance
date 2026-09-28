"""``align_row_positions``: put every row where the reference stands, rigidly.

WHY.  The operator, 2026-09-12, on a frame of 7608191311518369137:clip000:
"这个其实帧,人体不在相机的fov 内,被截断了,后处理能不能修啊".  Measured over
the 20 T eval clips, generated against ground truth: the generated mean root
sits **0.877 m** from ground truth's and the worst single frame **1.508 m**,
while each side's own travel is comparable (0.609 m against 0.725 m) -- so it is
a displacement, not a difference in how far the dancer moves.

Why that crops rather than merely offsetting.  ``_camera_pose`` backs the camera
off by ``spread``, the largest row-to-row root distance, so that every row fits;
but it enters the formula as a LATERAL half-extent, and a row displaced toward
the lens is drawn LARGER at the same framing.  Cutting the spread in half both
uncrops the body and keeps it large, which CLAUDE.md 1.5 requires (the body must
fill 60% of the frame or "动作到不到位" cannot be judged).

THE JUSTIFICATION IS ``align_heading``'S, ONE LINE UP.  That one rotates rows
onto the reference's heading because "the world heading of a monocular
reconstruction is arbitrary ... nothing fixes front".  Nothing fixes where in
the room either.  What must survive is the same thing heading alignment keeps:
every metre of travel WITHIN the clip, which is what is being judged.

POSITIVE CONTROL is ``test_the_rows_actually_move``; the guard that this does
not become a cheat is ``test_travel_within_a_row_is_untouched``, which is the
assertion that would fail if a future version re-centred per frame.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.render_avatar_video import align_row_positions


def _row(offset, travel, frames=40, vertices=True):
    """Joints for one row: a root that walks ``travel`` from ``offset``."""
    joints = np.zeros((frames, 24, 3))
    walk = np.linspace(0, 1, frames)[:, None] * np.asarray(travel)
    joints[:, 0, :2] = np.asarray(offset) + walk
    for j in range(1, 24):                       # a rigid body around the root
        joints[:, j, :2] = joints[:, 0, :2] + j * 0.01
        joints[:, j, 2] = j * 0.05
    verts = joints[:, :10].copy() if vertices else None
    return (verts, None, joints)


def test_the_rows_actually_move():
    """POSITIVE CONTROL: without this the other assertions pass on a no-op."""
    rows = [_row((0.0, 0.0), (1.0, 0.0)), _row((3.0, -2.0), (1.0, 0.0))]
    before = np.asarray(rows[1][2])[:, 0, :2].mean(0).copy()
    align_row_positions(rows)
    after = np.asarray(rows[1][2])[:, 0, :2].mean(0)
    assert np.linalg.norm(after - before) > 1.0
    assert np.allclose(after, np.asarray(rows[0][2])[:, 0, :2].mean(0))


def test_travel_within_a_row_is_untouched():
    """The whole point: it removes an offset, NOT a difference in travel.

    A version that re-centred every frame would pass the test above and fail
    this one, and it would be exactly the defect render_dance_video's header
    warns about -- the arm that travels furthest drawn as the calmest.
    """
    rows = [_row((0.0, 0.0), (1.0, 0.0)), _row((3.0, -2.0), (0.2, 0.4))]
    before = np.asarray(rows[1][2])[:, 0, :2].copy()
    align_row_positions(rows)
    after = np.asarray(rows[1][2])[:, 0, :2]
    assert np.allclose(after - after.mean(0), before - before.mean(0), atol=1e-12)


def test_the_reference_row_is_never_moved():
    rows = [_row((0.5, 0.5), (1.0, 0.0)), _row((3.0, -2.0), (1.0, 0.0))]
    before = np.asarray(rows[0][2]).copy()
    align_row_positions(rows)
    assert np.array_equal(np.asarray(rows[0][2]), before)


def test_height_is_not_touched():
    """The floor is the reference's on purpose -- a body that hovers must keep
    hovering (docs 42), so this may only move in the ground plane."""
    rows = [_row((0.0, 0.0), (1.0, 0.0)), _row((3.0, -2.0), (1.0, 0.0))]
    before = np.asarray(rows[1][2])[:, :, 2].copy()
    align_row_positions(rows)
    assert np.array_equal(np.asarray(rows[1][2])[:, :, 2], before)


def test_vertices_move_with_the_joints():
    """The joint gate compares the mesh against full_pose to a millimetre; a
    shift applied to one and not the other would trip it."""
    rows = [_row((0.0, 0.0), (1.0, 0.0)), _row((3.0, -2.0), (1.0, 0.0))]
    shifts = align_row_positions(rows)
    assert np.allclose(np.asarray(rows[1][0])[..., :2].mean((0, 1))
                       - (np.asarray(_row((3.0, -2.0), (1.0, 0.0))[0])[..., :2].mean((0, 1))),
                       shifts[1])


def test_a_row_without_vertices_survives():
    """The stick-figure path passes ``None`` for vertices."""
    rows = [_row((0.0, 0.0), (1.0, 0.0)),
            _row((3.0, -2.0), (1.0, 0.0), vertices=False)]
    align_row_positions(rows)
    assert rows[1][0] is None


def test_the_shift_is_reported_per_row():
    rows = [_row((0.0, 0.0), (1.0, 0.0)), _row((3.0, -2.0), (1.0, 0.0))]
    shifts = align_row_positions(rows)
    assert set(shifts) == {1}
    assert np.allclose(shifts[1], (-3.0, 2.0))
