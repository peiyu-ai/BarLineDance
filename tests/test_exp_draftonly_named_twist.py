"""Controls for the named-frames twist split (tools/exp_draftonly_named_twist.py).

The four checks CLAUDE.md 2.1 asks of a criterion before it may judge:

  * PROVENANCE / INSTRUMENT.  The split must be the SAME measure as
    ``score_arm_table.torso_twist_rate``, not a second one that happens to
    correlate; a clip with no filler must make the two agree bit for bit.
  * POSITIVE, with the direction fixed in advance.  A clip that twists at a
    known rate on its named frames and is frozen on its filler frames must read
    that rate on ``twist_named``, ~0 on ``twist_filler``, and something strictly
    between on ``twist_all``.
  * A ZERO MUST BE PROVABLE.  An all-filler clip must read NaN on
    ``twist_named``, never 0.0 -- otherwise an arm with nothing named would
    silently score.
  * THE MIXTURE MUST RECONSTRUCT.  named/filler/crossing steps must partition
    the clip's steps and their weighted mean must equal ``twist_all``, so the
    reader can check the whole-clip column against the split rather than
    trusting it.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import exp_draftonly_named_twist as m  # noqa: E402
from tools.score_arm_table import torso_twist_rate  # noqa: E402

FPS = 30.0


def _clip(twist_per_frame_deg, frames, *, seed=0):
    """A skeleton whose shoulder line rotates by a fixed angle each frame.

    Only joints 1, 2 (hips), 16, 17 (shoulders) and 0 (root) matter to the
    measure; the rest are filled with a fixed pose so the array has the shape
    the loader produces.
    """
    rng = np.random.default_rng(seed)
    joints = np.tile(rng.normal(size=(1, 24, 3)), (frames, 1, 1))
    angle = np.deg2rad(twist_per_frame_deg) * np.arange(frames)
    joints[:, 0] = 0.0
    joints[:, 1] = np.stack([np.full(frames, -0.1), np.zeros(frames), np.zeros(frames)], 1)
    joints[:, 2] = np.stack([np.full(frames, 0.1), np.zeros(frames), np.zeros(frames)], 1)
    joints[:, 16] = np.stack([-0.2 * np.cos(angle), -0.2 * np.sin(angle), np.full(frames, 0.4)], 1)
    joints[:, 17] = np.stack([0.2 * np.cos(angle), 0.2 * np.sin(angle), np.full(frames, 0.4)], 1)
    return joints


def _freeze_after(joints, first_frozen):
    out = joints.copy()
    out[first_frozen:] = out[first_frozen]
    return out


def test_all_named_reproduces_the_arm_table_column_exactly():
    """The instrument check: this is the arm table's twist, only partitioned."""
    joints = _clip(2.0, 200)
    row = m.split_by_label(joints, np.ones(200, int))
    assert row["twist_all"] == pytest.approx(torso_twist_rate(joints), rel=1e-12)
    assert row["twist_named"] == pytest.approx(torso_twist_rate(joints), rel=1e-12)
    assert row["steps_filler"] == 0 and row["steps_crossing"] == 0


def test_named_reads_the_planted_rate_and_filler_reads_zero():
    """Positive control, direction pinned before running: named >> all >> filler."""
    frames = 200
    joints = _freeze_after(_clip(2.0, frames), 100)
    labels = np.concatenate([np.full(100, 7), np.zeros(100, int)])
    row = m.split_by_label(joints, labels)
    # 2 deg per frame at 30 fps is 60 deg/s, and that is what the named half must read.
    assert row["twist_named"] == pytest.approx(60.0, rel=1e-6)
    assert row["twist_filler"] == pytest.approx(0.0, abs=1e-9)
    assert row["twist_filler"] < row["twist_all"] < row["twist_named"]


def test_the_whole_clip_column_is_the_mixture_of_the_three_populations():
    joints = _freeze_after(_clip(2.0, 200), 100)
    labels = np.concatenate([np.full(100, 7), np.zeros(100, int)])
    row = m.split_by_label(joints, labels)
    assert row["steps_named"] + row["steps_filler"] + row["steps_crossing"] == row["steps"]
    total = 0.0
    for value, weight in ((row["twist_named"], row["steps_named"]),
                          (row["twist_filler"], row["steps_filler"]),
                          (row["twist_crossing"], row["steps_crossing"])):
        if weight:
            total += value * weight
    assert total / row["steps"] == pytest.approx(row["twist_all"], rel=1e-9)


def test_an_all_filler_clip_reads_nan_not_zero():
    joints = _clip(2.0, 120)
    row = m.split_by_label(joints, np.zeros(120, int))
    assert np.isnan(row["twist_named"]), "an arm with nothing named must not score 0.0"
    assert row["steps_named"] == 0
    assert row["twist_filler"] == pytest.approx(row["twist_all"], rel=1e-12)


def test_a_short_label_track_truncates_both_rather_than_defaulting_to_named():
    """A ground-truth annotation may end before the decoded motion does."""
    joints = _freeze_after(_clip(2.0, 200), 100)
    row = m.split_by_label(joints, np.full(100, 7))
    assert row["steps"] == 99
    assert row["steps_filler"] == 0
    assert row["twist_named"] == pytest.approx(60.0, rel=1e-6)


def test_crossing_steps_belong_to_neither_population():
    joints = _clip(2.0, 60)
    labels = np.array([7] * 30 + [0] * 30)
    row = m.split_by_label(joints, labels)
    assert row["steps_crossing"] == 1
    assert row["steps_named"] == 29
    assert row["steps_filler"] == 29
