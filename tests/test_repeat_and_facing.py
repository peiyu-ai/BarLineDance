"""The two columns that catch what the operator catches by watching.

Both defects were named from the video and both are PER CLIP: pooled over the
twenty eval clips the selector's replay share reads 2.6% against ground truth's
2.5% -- invisible -- while the clip the operator singled out replays 25% of its
bars.  These tests pin the properties that make the columns able to see that.
"""
import numpy as np
import pytest

from tools.score_repeat_and_facing import (REPLAY_DISTANCE, canonical, facing,
                                           replay_share)


BAR = 48          # 4 beats of 12 frames -- the blocks below MUST match this,
                  # or a "repeated bar" straddles two blocks and nothing repeats


def music_with_beats(frames, period=12):
    music = np.zeros((frames, 35), dtype=np.float32)
    music[::period, 34] = 1.0
    music[::period * 4, 0] = 1.0
    return music


def dancer(frames, seed=0, joints=24):
    rng = np.random.default_rng(seed)
    t = np.arange(frames)[:, None, None]
    return 0.3 * np.sin(t / 11.0 + rng.normal(size=(1, joints, 3))) + 1.0


def test_a_clip_that_repeats_a_bar_is_caught():
    bar = dancer(BAR, seed=1)
    motion = np.concatenate([bar, dancer(BAR, seed=2), bar, dancer(BAR, seed=3)])
    assert replay_share(motion, music_with_beats(len(motion))) > 0.2


def test_a_clip_with_no_repeat_reads_zero():
    motion = np.concatenate([dancer(BAR, seed=s) for s in range(4)])
    assert replay_share(motion, music_with_beats(len(motion))) == 0.0


def test_the_comparison_is_blind_to_where_the_dancer_stands():
    """A replay that walked two metres to the left is still a replay: the
    operator sees the same movement, not the same coordinates."""
    bar = dancer(BAR, seed=1)
    moved = bar + np.array([2.0, 0.0, 0.0])
    motion = np.concatenate([bar, dancer(BAR, seed=2), moved, dancer(BAR, seed=3)])
    assert replay_share(motion, music_with_beats(len(motion))) > 0.2


def test_the_comparison_is_blind_to_body_scale():
    """Canonicalisation divides by shoulder width, so a taller reconstruction of
    the same movement does not read as a different movement."""
    bar = dancer(BAR, seed=1)
    canon_small = canonical(bar)
    canon_big = canonical(bar * 1.4)
    assert np.abs(canon_small - canon_big).max() < 1e-6


def test_facing_reads_the_share_and_the_longest_stretch():
    frames = 300
    motion = dancer(frames)
    # shoulders across the x axis -> chest normal points +y -> faces the lens
    motion[:, 16, :2] = [-0.2, 0.0]
    motion[:, 17, :2] = [0.2, 0.0]
    share, longest = facing(motion)
    assert share == 0.0 and longest == 0.0
    # turn the second half around
    motion[150:, 16, :2] = [0.2, 0.0]
    motion[150:, 17, :2] = [-0.2, 0.0]
    share, longest = facing(motion)
    assert share == pytest.approx(0.5, abs=0.02)
    assert longest == pytest.approx(5.0, abs=0.1)


def test_a_brief_turn_is_not_a_long_one():
    """The defect the operator names is a LONG stretch; a dancer who turns for
    half a second is dancing, not facing away."""
    motion = dancer(300)
    motion[:, 16, :2] = [-0.2, 0.0]
    motion[:, 17, :2] = [0.2, 0.0]
    motion[100:110, 16, :2] = [0.2, 0.0]
    motion[100:110, 17, :2] = [-0.2, 0.0]
    _share, longest = facing(motion)
    assert longest < 0.5


def test_the_replay_threshold_is_documented_where_it_is_used():
    assert 0 < REPLAY_DISTANCE < 1.0
