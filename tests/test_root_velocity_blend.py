"""``--draft-root-velocity-blend``: the join needs matching SPEED, not just place.

WHY.  ``--draft-root-continuity`` adds a constant offset so each segment's first
frame sits on the previous segment's last.  That makes the root POSITION
continuous and leaves the body's speed and direction changing in a single frame.
Measured 2026-09-06 on the eval drafts with ``xy`` on: the worst single-frame
root step sits AT a join on **18 of 18 clips**, median distance 1 frame, at
0.13-0.37 m against ground truth's 0.0221 -- while every seam column in this
repository is computed on ROOT-RELATIVE joints and cannot see any of it.

The correction is eased in with a raised cosine rather than a line: a linear
blend removes the step in velocity and leaves one in acceleration, which is the
mistake this codebase has now made twice at other levels (the seam cross-fade's
ramp, and its envelope).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_atomic import CONTACT_CHANNELS, ROOT_POSITION_DIMS


def _blend(values, previous_root, previous_velocity, span, columns):
    """The transform build_draft applies, isolated so it can be asserted."""
    import math
    out = values.clone()
    out[:, columns] += previous_root - out[0, columns]
    if span and previous_velocity is not None and len(out) > 1:
        span = min(int(span), len(out) - 1)
        delta = out[1:span + 1, columns] - out[:span, columns]
        correction = previous_velocity - delta[0]
        for i in range(span):
            weight = 0.5 + 0.5 * math.cos(math.pi * (i + 1) / (span + 1))
            out[i + 1:, columns] += correction * weight
    return out


COLUMNS = list(ROOT_POSITION_DIMS) if not isinstance(ROOT_POSITION_DIMS, int) \
    else [CONTACT_CHANNELS, CONTACT_CHANNELS + 1, CONTACT_CHANNELS + 2]


def _segment(start, velocity, n=20, dim=12):
    v = torch.zeros(n, dim)
    for i in range(n):
        for k, c in enumerate(COLUMNS):
            v[i, c] = start[k] + velocity[k] * i
    return v


def test_position_is_continuous_with_or_without_the_blend():
    prev_root = torch.tensor([1.0, 2.0, 0.0])
    seg = _segment([5.0, 5.0, 0.0], [0.10, 0.0, 0.0])
    for span in (0, 6):
        out = _blend(seg, prev_root, torch.tensor([0.01, 0.0, 0.0]), span, COLUMNS)
        assert torch.allclose(out[0, COLUMNS], prev_root, atol=1e-6)


def test_without_the_blend_the_velocity_steps_at_the_join():
    """The defect: the previous segment crawls, the new one sprints."""
    prev_root = torch.tensor([1.0, 2.0, 0.0])
    prev_v = torch.tensor([0.005, 0.0, 0.0])
    seg = _segment([5.0, 5.0, 0.0], [0.10, 0.0, 0.0])
    out = _blend(seg, prev_root, prev_v, 0, COLUMNS)
    first = (out[1, COLUMNS] - out[0, COLUMNS])
    assert abs(float(first[0]) - 0.10) < 1e-6
    assert abs(float(first[0]) - float(prev_v[0])) > 0.09     # a 20x step


def test_with_the_blend_the_first_step_matches_the_previous_velocity():
    prev_root = torch.tensor([1.0, 2.0, 0.0])
    prev_v = torch.tensor([0.005, 0.0, 0.0])
    seg = _segment([5.0, 5.0, 0.0], [0.10, 0.0, 0.0])
    out = _blend(seg, prev_root, prev_v, 6, COLUMNS)
    first = float((out[1, COLUMNS] - out[0, COLUMNS])[0])
    assert abs(first - float(prev_v[0])) < 0.02, first


def test_the_segment_recovers_its_own_travel_after_the_window():
    """The blend must fade OUT: past the window the prototype travels as itself."""
    prev_root = torch.tensor([1.0, 2.0, 0.0])
    prev_v = torch.tensor([0.005, 0.0, 0.0])
    seg = _segment([5.0, 5.0, 0.0], [0.10, 0.0, 0.0], n=30)
    out = _blend(seg, prev_root, prev_v, 6, COLUMNS)
    late = float((out[20, COLUMNS] - out[19, COLUMNS])[0])
    assert abs(late - 0.10) < 1e-6


def test_the_correction_is_monotone_and_has_no_corner():
    """A linear ease would leave a step in acceleration; assert the shape."""
    import math
    span = 8
    w = [0.5 + 0.5 * math.cos(math.pi * (i + 1) / (span + 1)) for i in range(span)]
    assert w[0] > w[-1]
    assert all(w[i] >= w[i + 1] for i in range(len(w) - 1))
    second = max(abs(w[i - 1] - 2 * w[i] + w[i + 1]) for i in range(1, len(w) - 1))
    assert second < 0.1, second


def test_span_zero_reproduces_the_published_behaviour():
    prev_root = torch.tensor([1.0, 2.0, 0.0])
    seg = _segment([5.0, 5.0, 0.0], [0.10, 0.0, 0.0])
    a = _blend(seg, prev_root, torch.tensor([0.005, 0.0, 0.0]), 0, COLUMNS)
    b = seg.clone(); b[:, COLUMNS] += prev_root - b[0, COLUMNS]
    assert torch.equal(a, b)
