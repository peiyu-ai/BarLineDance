"""The music channels reach both stages through one bare Linear, and their raw
scales span three orders of magnitude.  These tests hold the mechanism, the
opt-in, and the one property that makes it un-forgettable.

Measured on the shipped release (42,090,600 frames, tools/fit_music_normalizer):
MFCC c1 carries **62.10%** of the input variance and the beat one-hot carries
**0.00050%**.  That is the whole of "the model does not dance to the beat" at
the input layer -- no appeal to capacity or objective is needed.
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.atomic_planner import AtomicPlannerTransformer, MusicNormalization
from model.atomic_completion import AtomicCompletionDecoder
from tools.fit_music_normalizer import energy_share, BEAT_CHANNEL


def _stats(dim=35):
    std = np.full(dim, 0.05, np.float32)
    std[1:21] = 50.0          # MFCC, the loud ones
    std[BEAT_CHANNEL] = 0.23  # the beat one-hot, the quiet one that matters
    return {"mean": torch.zeros(dim), "std": torch.tensor(std)}


def test_the_beat_channel_is_negligible_before_and_equal_after():
    """The positive control for the whole change: a criterion that reads the
    defect on the raw scales and reads it fixed on the normalized ones."""
    std = _stats()["std"].numpy()
    before = energy_share(std)
    assert before[BEAT_CHANNEL] < 1e-4          # 0.004% here, 0.0005% on the corpus
    assert before[1:21].sum() > 0.99            # MFCC owns essentially all of it
    after = energy_share(np.ones_like(std))
    assert after[BEAT_CHANNEL] == pytest.approx(1.0 / len(std))


def test_normalization_equalises_the_channels():
    """Not "the beat gets louder in absolute terms" -- every channel comes out
    at unit scale, which is the point: the beat stops being 1/200th of a MFCC
    coefficient.  Asserted as the RATIO between the loud and the quiet channel,
    which is what the bare Linear actually sees."""
    layer = MusicNormalization(_stats())
    raw = torch.randn(64, 32, 35) * _stats()["std"]
    before = raw.std(dim=(0, 1))
    after = layer(raw).std(dim=(0, 1))
    assert float(before[1] / before[BEAT_CHANNEL]) > 100
    assert float(after[1] / after[BEAT_CHANNEL]) == pytest.approx(1.0, abs=0.1)


def test_a_constant_channel_is_left_alone_not_epsilon_inflated():
    stats = _stats()
    stats["std"][7] = 0.0
    layer = MusicNormalization(stats)
    assert layer.music_std[7] == 1.0
    raw = torch.zeros(1, 4, 35)
    assert torch.isfinite(layer(raw)).all()


def _planner(**kw):
    return AtomicPlannerTransformer(num_atomic_classes=4, music_dim=35, latent_dim=16,
                                    num_layers=1, num_heads=2, ff_size=16,
                                    max_seq_len=8, **kw)


def _completion(**kw):
    return AtomicCompletionDecoder(motion_dim=12, seq_len=8, music_dim=35, latent_dim=16,
                                   ff_size=16, num_layers=1, num_heads=2, **kw)


@pytest.mark.parametrize("build", [_planner, _completion])
def test_it_is_off_unless_asked_for(build):
    assert build().music_normalization is None
    assert build(music_stats=_stats()).music_normalization is not None


@pytest.mark.parametrize("build", [_planner, _completion])
def test_a_checkpoint_cannot_be_run_without_its_own_statistics(build):
    """The reason the numbers live in the state_dict rather than behind a path
    in ``args``: a path some call site forgets to read is the exact defect
    shape this repository paid for three times on 2026-08-30 -- a flag recorded
    in the manifest, computed correctly, and dropped before use, with the
    output byte-identical to the default and nothing warning.  Strict loading
    turns that silent case into a refusal, in both directions."""
    with_stats = build(music_stats=_stats()).state_dict()
    without = build().state_dict()
    with pytest.raises(RuntimeError):
        build().load_state_dict(with_stats)
    with pytest.raises(RuntimeError):
        build(music_stats=_stats()).load_state_dict(without)
    build(music_stats=_stats()).load_state_dict(with_stats)


def test_the_planner_output_changes_when_the_scales_change():
    torch.manual_seed(0)
    plain = _planner()
    torch.manual_seed(0)
    scaled = _planner(music_stats=_stats())
    labels = torch.zeros(1, 8, dtype=torch.long)
    music = torch.randn(1, 8, 35) * _stats()["std"]
    steps = torch.zeros(1, dtype=torch.long)
    assert not torch.allclose(plain(labels, music, steps), scaled(labels, music, steps))
