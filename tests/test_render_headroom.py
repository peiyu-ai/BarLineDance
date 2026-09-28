"""The camera must leave room for a body that is off the ground or reaching up.

WHY.  The operator, 2026-09-12, on a frame of 7608191311518369137:clip000:
"这个其实帧,人体不在相机的fov 内,被截断了".  At that frame the generated
lowest foot is 0.735 m against ground truth's 0.371, i.e. the body is 0.36 m in
the air -- and hovering is NOT the defect: measured over the 20 T eval clips the
lowest foot is more than 0.15 m above the clip's own floor on **28.5% of ground
truth's frames and 28.2% of the generated ones** (p95 0.217 m and 0.229 m, every
paired difference inside noise).  Both dancers leave the ground; only one of
them was on screen when it happened.

The cause is the framing.  ``_camera_pose`` solves the distance from ``fill``
alone -- "the body should fill about 72% of the frame height" -- which is the
right shot for a body standing on the floor with its arms down and leaves no
headroom for anything else.  There was already a LATERAL guarantee (``spread``
backs the camera off until the widest row-to-row separation fits) and no
vertical one.  ``reach`` is that guarantee.

POSITIVE CONTROL is ``test_a_tall_reach_backs_the_camera_off``; the guard
against over-correcting is ``test_a_body_that_stays_low_is_not_shrunk``, which
is what would fail if the rule fired on every clip and made every body smaller.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.render_avatar_video import FOV, _camera_pose

STATURE = 1.75
BASE = {"centre": (0.0, 0.0), "floor": 0.0, "radius": 1.0, "stature": STATURE,
        "fill": 0.72, "facing": np.array([0.0, 1.0, 0.0])}


def _distance(spec):
    pose = _camera_pose(spec, view="front")
    eye = pose[:3, 3]
    return float(np.linalg.norm(eye[[0, 2]]))          # y-up render space


def test_a_tall_reach_backs_the_camera_off():
    """POSITIVE CONTROL: without the rule the two distances are equal."""
    near = _distance(dict(BASE))
    far = _distance(dict(BASE, reach=1.4 * STATURE))
    assert far > near * 1.05, (near, far)


def test_a_body_that_stays_low_is_not_shrunk():
    """A clip whose highest joint is the head of a standing body needs nothing,
    and firing anyway would cost every clip the 60%-of-frame rule."""
    plain = _distance(dict(BASE))
    modest = _distance(dict(BASE, reach=1.0 * STATURE))
    assert modest == plain


def test_the_top_of_the_reach_is_inside_the_frame():
    """The property, stated directly: the half-frame at the target must cover
    the distance from the camera's target height up to the highest joint."""
    for factor in (1.0, 1.2, 1.5, 1.8):
        reach = factor * STATURE
        spec = dict(BASE, reach=reach)
        half = _distance(spec) * np.tan(FOV / 2)
        assert half >= reach - 0.55 * STATURE - 1e-9, (factor, half)


def test_the_floor_stays_inside_the_frame_too():
    """Backing off for reach must never crop the feet instead."""
    for factor in (1.0, 1.5, 2.0):
        half = _distance(dict(BASE, reach=factor * STATURE)) * np.tan(FOV / 2)
        assert half >= 0.55 * STATURE - 1e-9


def test_reach_and_spread_do_not_cancel():
    """Whichever constraint is tighter wins; neither may undo the other."""
    wide = _distance(dict(BASE, spread=4.0))
    both = _distance(dict(BASE, spread=4.0, reach=2.0 * STATURE))
    assert both >= wide
