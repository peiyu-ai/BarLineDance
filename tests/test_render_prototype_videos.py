"""Tests for the sub-prototype video renderer.

The load-bearing decision here is that a segment is canonicalised once, from its
first frame, instead of per frame the way the descriptor does it.  That is what
lets a turn stay a turn on screen, so it is what the tests pin down.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.motion_beats import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, ROOT, canonical_pose  # noqa: E402
from tools.render_prototype_videos import (  # noqa: E402
    build_parser,
    canonical_segment,
    frame_extent,
    segment_transform,
)


def _turning_dancer(frames=60, turn=2 * np.pi, travel=0.0):
    """A body that keeps its shape and rotates about the vertical axis."""
    base = np.zeros((24, 3))
    base[L_HIP] = [0.1, 0.0, 0.0]
    base[R_HIP] = [-0.1, 0.0, 0.0]
    base[L_SHOULDER] = [0.18, 0.0, 0.5]
    base[R_SHOULDER] = [-0.18, 0.0, 0.5]
    base[5] = [0.0, 0.35, 0.2]        # something off the hip axis, to see rotation
    sequence = []
    for step in range(frames):
        angle = turn * step / max(frames - 1, 1)
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        pose = base @ rotation.T
        pose[:, 1] += travel * step / max(frames - 1, 1)
        sequence.append(pose)
    return np.stack(sequence)


def test_per_frame_canonicalisation_erases_a_turn():
    # The property that makes canonical_pose wrong for video: re-aligning the
    # hips every frame turns a full revolution into a body that never moves.
    segment = _turning_dancer()
    per_frame = np.stack([canonical_pose(frame) for frame in segment])
    spread = per_frame.reshape(len(per_frame), -1).std(axis=0).max()
    assert spread < 1e-6


def test_one_transform_per_segment_keeps_the_turn():
    segment = _turning_dancer()
    fixed = canonical_segment(segment)
    spread = fixed.reshape(len(fixed), -1).std(axis=0).max()
    # Same motion, same normalisation family, but the rotation survives.
    assert spread > 0.1


def test_the_transform_comes_from_the_first_frame():
    segment = _turning_dancer()
    fixed = canonical_segment(segment)
    # Frame 0 is canonical by construction: root at the origin, hips on +x.
    assert np.allclose(fixed[0][ROOT], 0.0, atol=1e-9)
    hips = fixed[0][L_HIP] - fixed[0][R_HIP]
    assert abs(hips[1]) < 1e-9
    assert hips[0] > 0


def test_travel_survives_the_transform():
    still = canonical_segment(_turning_dancer(turn=0.0, travel=0.0))
    moving = canonical_segment(_turning_dancer(turn=0.0, travel=3.0))
    # Only the root translation of the *first* frame is removed, so a dancer who
    # crosses the floor still crosses it on screen.
    assert np.abs(still[:, ROOT]).max() < 1e-9
    assert np.abs(moving[:, ROOT]).max() > 1.0


def test_scale_is_shoulder_width_so_bodies_are_comparable():
    small = _turning_dancer()
    large = small * 2.5
    a, b = canonical_segment(small), canonical_segment(large)
    # Two dancers of different build must land at the same size, or the grid
    # would be comparing body proportions rather than movement.
    assert np.allclose(a, b, atol=1e-9)


def test_segment_transform_survives_a_degenerate_scale():
    segment = _turning_dancer()
    segment[:, L_SHOULDER] = segment[:, R_SHOULDER]      # zero shoulder width
    _, _, scale = segment_transform(segment)
    assert scale == 1.0
    assert np.isfinite(canonical_segment(segment)).all()


def test_frame_extent_is_shared_across_cells():
    tight = canonical_segment(_turning_dancer(turn=0.0))
    wide = canonical_segment(_turning_dancer(turn=0.0, travel=6.0))
    span, (low, high) = frame_extent([tight, wide], 30.0)
    # One box for every cell: per-cell autoscaling would draw a small tight move
    # and a large travelling one at the same apparent size.
    assert span >= 2.2 and high > low
    alone, _ = frame_extent([tight], 30.0)
    assert span >= alone


def test_parser_defaults():
    args = build_parser().parse_args(["--labels", "l", "--bundle", "b", "--output", "o"])
    assert args.seconds == 4.0
    assert args.view_azimuth == 30.0
    assert args.select == "spread"
    assert args.min_uploads == 2
