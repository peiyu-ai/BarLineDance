"""The camera-coverage gate, and the control that proves it can fail.

The gate exists because the shot, not the model, was hiding the dancer: over
100 clips the generated root's peak separation from the ground truth's is a
median 1.198 m against a half-frame of about 1.25 m.  A gate written after the
fact is worth nothing unless it fails on the code that produced the defect, so
the first test rebuilds the old reference-following camera and requires a fail.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.render_avatar_video import FOV, _camera_pose, _z_up_to_y_up, stage
from tools.check_camera_coverage import visible_fraction


def _body(frames=60, offset=(0.0, 0.0)):
    """A still, upright, 1.75 m stick of joints at a fixed ground position."""
    joints = np.zeros((frames, 24, 3))
    heights = np.linspace(0.0, 1.75, 24)
    joints[:, :, 2] = heights
    joints[:, :, 0] = offset[0]
    joints[:, :, 1] = offset[1]
    # joints 1 and 2 are the hips; body_facing needs them non-degenerate
    joints[:, 1, 0] = offset[0] - 0.1
    joints[:, 2, 0] = offset[0] + 0.1
    return joints


def _visible_following_row_zero(rows, margin=0.90):
    """The pre-2026-08-30 rule, rebuilt: aim at row 0's root, no spread."""
    roots = np.stack([np.asarray(j)[:, 0, :] for j in rows])
    spec = stage(rows, 0.0)
    target = roots[0][:, :2]
    inside = np.zeros(len(rows))
    for frame in range(roots.shape[1]):
        pose = _camera_pose(spec, look_at=target[frame], view="front")
        world_to_camera = np.linalg.inv(pose)
        for row in range(len(rows)):
            camera = world_to_camera @ (_z_up_to_y_up() @ np.append(roots[row, frame], 1.0))
            depth = -camera[2]
            half = depth * np.tan(FOV / 2)
            if depth > 0 and abs(camera[0]) <= margin * half and abs(camera[1]) <= margin * half:
                inside[row] += 1
    return inside / roots.shape[1]


def test_the_old_camera_fails_on_the_separation_that_was_measured():
    rows = [_body(offset=(0.0, 0.0)), _body(offset=(2.0, 0.0))]
    old = _visible_following_row_zero(rows)
    assert old[0] == 1.0            # the row it follows is always fine
    assert old[1] < 0.05            # and the other one is simply not in shot


def test_the_centroid_camera_keeps_both_rows_in_shot():
    rows = [_body(offset=(0.0, 0.0)), _body(offset=(2.0, 0.0))]
    assert visible_fraction(rows, floor=0.0).min() >= 0.95


def test_a_single_row_is_unaffected():
    assert visible_fraction([_body()], floor=0.0)[0] == 1.0


def test_the_smoothing_no_longer_pulls_the_shot_to_the_origin():
    """``np.convolve(..., mode='same')`` zero-pads, so a body standing still at
    x = 3 m had its camera target dragged toward x = 0 over the last 22 frames
    -- measured 0.737 m of error on the final frame of a real clip."""
    track = np.full((60, 2), 3.0)
    window, half = 45, 22
    bad = np.stack([np.convolve(track[:, a], np.ones(window) / window, mode="same")
                    for a in (0, 1)], 1)
    padded = np.pad(track, ((half, window - 1 - half), (0, 0)), mode="edge")
    good = np.stack([np.convolve(padded[:, a], np.ones(window) / window, mode="valid")
                     for a in (0, 1)], 1)
    assert abs(bad[-1, 0] - 3.0) > 1.0
    assert np.allclose(good, 3.0)
