"""The release builder and inference must pool a bar the SAME way.

THE DEFECT THIS PINS.  Until 2026-09-08 ``tools/build_bar_planner_release.py``
pooled a bar with its own function -- mean everywhere, SUM on channel 33, the
onset-peak count -- while ``dataset/bar_tokens.pool_music``, which is what
``infer_atomic`` pools with, meaned every channel.  Measured on the first bar of
the first validation window (58 frames): the stored release holds 15.0000 on
channel 33 and inference produced 0.2586, a factor of 58, on the one channel
that carries accent density.  Every other channel agreed to 1e-5.

So ``planner_txy_t_bar_s20260902`` was trained on a rhythm channel whose
inference-time value was about 1/58 of what it learned from, and nothing in the
pipeline could report it: both sides were internally consistent and the widths
matched, so the checkpoint's own ``music_dim`` gate passed.

``dataset/bar_tokens``' module docstring already said "Two definitions of the
pooling is precisely the train/test mismatch this repository keeps paying for
... Import from here instead."  It happened anyway, which is why this is a test
and not a comment.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.bar_tokens import COUNT_CHANNELS, pool_music
from tools.build_bar_planner_release import pool_bar_music

MUSIC_DIM = 35


def _bar(frames=58, seed=0):
    """A bar whose count channel is a sparse one-hot, like the real feature."""
    rng = np.random.default_rng(seed)
    music = rng.normal(size=(frames, MUSIC_DIM))
    music[:, 33] = 0.0
    music[rng.choice(frames, 15, replace=False), 33] = 1.0
    return music


def test_the_builder_and_inference_agree_on_every_channel():
    music = _bar()
    built = pool_bar_music(music)
    inferred = pool_music(music, [0, len(music)], "mean").numpy()[0]
    assert np.abs(built - inferred).max() < 1e-4, (
        "builder and inference disagree on channels {}"
        .format(np.flatnonzero(np.abs(built - inferred) > 1e-4).tolist()))


def test_the_count_channel_is_summed_not_meaned():
    """POSITIVE CONTROL: the two rules must give different answers here.

    Without this, an implementation that meaned on BOTH sides would pass the
    agreement test above while throwing away the accent count -- agreement is
    only evidence when the alternative is distinguishable.
    """
    music = _bar()
    pooled = pool_music(music, [0, len(music)], "mean").numpy()[0]
    assert pooled[33] == 15.0, pooled[33]
    meaned = pool_music(music, [0, len(music)], "mean", count_channels=())[0]
    assert abs(float(meaned[33]) - 15.0 / len(music)) < 1e-5
    assert abs(float(pooled[33]) - float(meaned[33])) > 1.0, (
        "sum and mean are indistinguishable on this fixture, so the agreement "
        "test above proves nothing")


def test_non_count_channels_are_still_meaned():
    music = _bar()
    pooled = pool_music(music, [0, len(music)], "mean").numpy()[0]
    for channel in (0, 1, 21, 34):
        assert channel not in COUNT_CHANNELS
        assert abs(float(pooled[channel]) - float(music[:, channel].mean())) < 1e-5


def test_a_span_shorter_than_the_channel_count_does_not_index_out_of_range():
    """Narrow music must not raise: pool_music is called on 2-D fixtures in tests."""
    pooled = pool_music(torch.ones(4, 3), [0, 4], "mean").numpy()[0]
    assert pooled.shape == (3,)
    assert np.allclose(pooled, 1.0)


def test_mean_std_keeps_the_summed_count_in_its_mean_half():
    music = _bar()
    pooled = pool_music(music, [0, len(music)], "mean_std").numpy()[0]
    assert pooled.shape == (2 * MUSIC_DIM,)
    assert pooled[33] == 15.0
    assert abs(float(pooled[MUSIC_DIM + 33]) - float(music[:, 33].std())) < 1e-5
