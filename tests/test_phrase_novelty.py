"""--draft-continue-phrase-novelty (H series): phrases start where the MUSIC changes.  Synthetic music: four sections of
distinct timbre, 4 bars each (bar = 30 frames); the phrase starts must be the section boundaries."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic as ia  # noqa: E402


def _music(sections, bars_per=4, bar=30, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for s in range(sections):
        base = rng.normal(0, 1, 35) * 3
        blocks.append(base + rng.normal(0, 0.1, (bars_per * bar, 35)))
    return np.concatenate(blocks)


def test_starts_fall_on_section_changes():
    music = _music(4)
    bounds = list(range(0, 16 * 30 + 1, 30))                 # 16 bars
    starts = ia._phrase_starts(music, bounds, quantile=0.7, min_len=2, max_len=8)
    assert {4, 8, 12} <= starts
    assert all(k % 4 == 0 for k in starts)


def test_long_unchanging_music_is_still_cut():
    music = _music(1, bars_per=20)
    bounds = list(range(0, 20 * 30 + 1, 30))
    starts = ia._phrase_starts(music, bounds, quantile=0.99, min_len=2, max_len=8)
    gaps = np.diff([0] + sorted(starts) + [20])
    assert gaps.max() <= 8
