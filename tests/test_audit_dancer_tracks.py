"""Tests for the dancer-track continuity audit.

What these pin is the distinction the tool exists for: a detector wobbling is
not the track moving onto a different person, and only the second one means the
clip splices two dancers.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.audit_dancer_tracks import (HEIGHT_CHANGE, MIN_TRACKED_FRAMES,  # noqa: E402
                                       analyse)


def track(frames=200, x=300.0, width=60.0, height=160.0):
    boxes = np.zeros((frames, 4), dtype=np.float32)
    boxes[:, 0] = x
    boxes[:, 1] = 100.0
    boxes[:, 2] = x + width
    boxes[:, 3] = 100.0 + height
    return boxes


def test_a_steady_dancer_has_no_switch():
    boxes = track()
    boxes[:, [0, 2]] += (10 * np.sin(np.arange(len(boxes)) / 5.0))[:, None]
    result = analyse(boxes)
    assert result["decidable"]
    assert result["switches"] == 0


def test_a_wobble_is_not_a_switch():
    """The centre jumps and comes back; the box is the same size throughout."""
    boxes = track()
    boxes[100, [0, 2]] += 200.0        # one frame far away, same box size
    result = analyse(boxes)
    assert result["jump_frames"] > 0, "the jump should be a candidate"
    assert result["switches"] == 0, "same box height means the same person"


def test_a_persistent_move_to_a_different_sized_box_is_a_switch():
    boxes = track(frames=200)
    # From frame 100 the track sits somewhere else on a much taller box: a
    # different person, at a different depth.
    boxes[100:, 0] += 200.0
    boxes[100:, 2] += 200.0
    boxes[100:, 3] = boxes[100:, 1] + 160.0 * 1.6
    result = analyse(boxes)
    assert result["switches"] >= 1
    assert result["max_height_change"] > HEIGHT_CHANGE
    assert result["switch_frames"][0] == 100


def test_a_bridged_gap_is_not_charged_as_a_teleport():
    """Half a second of absence then the same dancer, moved -- the feature."""
    boxes = track(frames=200)
    boxes[100:112] = np.nan
    boxes[112:, [0, 2]] += 60.0        # a body width over twelve frames
    result = analyse(boxes)
    assert result["switches"] == 0


def test_a_clip_too_sparsely_tracked_is_undecidable_not_clean():
    """"We could not tell" and "it is fine" must not collapse."""
    boxes = track(frames=200)
    boxes[MIN_TRACKED_FRAMES - 10:] = np.nan
    result = analyse(boxes)
    assert not result["decidable"]
    assert result["switches"] == 0     # reported, but the flag above says why
