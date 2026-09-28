"""The frozen k8 labeller must reproduce the rules the published labels were made with.

The published space (music_aligned_cca_k8_v1) was refit on 2026-09-22 and its
three label arrays reproduced byte-for-byte; these pin the two rules that
reproduction depends on and that a tidy-up would most likely "fix".
"""
import pathlib
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.label_bars_music_aligned import normalize_like_the_fit, window_features  # noqa: E402


def test_a_row_with_one_short_bar_is_zero_filled_whole():
    motion = np.random.default_rng(0).normal(size=(200, 151))
    music = np.random.default_rng(1).normal(size=(200, 35))
    feats, tunes, ok = window_features(motion, music, [[0, 40], [40, 80], [80, 89], [89, 130]])
    assert not ok
    assert feats.shape == (4, 11) and not feats.any() and not tunes.any()
    feats, tunes, ok = window_features(motion, music, [[0, 40], [40, 80], [80, 120], [120, 160]])
    assert ok and feats.shape == (4, 11) and tunes.shape == (4, 16) and feats.any()


def test_a_bar_past_the_music_end_is_unusable_even_if_motion_covers_it():
    motion = np.zeros((200, 151)); music = np.zeros((150, 35))
    _, _, ok = window_features(motion, music, [[0, 40], [40, 80], [80, 120], [120, 160]])
    assert not ok


def test_normalization_matches_apply_motion_normalizer_arithmetic():
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(50, 151)).astype(np.float64)
    data_min = torch.from_numpy(raw.min(0).astype(np.float32) - 0.1)
    data_max = torch.from_numpy(raw.max(0).astype(np.float32) + 0.1)
    data_max[3] = data_min[3]                       # a constant channel -> safe range 1
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "normalizer.pt"
        torch.save({"data_min": data_min, "data_max": data_max}, path)
        got = normalize_like_the_fit(raw, path)
    dmin, dmax = data_min.numpy(), data_max.numpy()
    safe = np.where(dmax - dmin == 0, np.float32(1.0), dmax - dmin).astype(np.float32)
    want = np.float32(2.0) * (raw.astype(np.float32) - dmin) / safe - np.float32(1.0)
    assert got.dtype == np.float32 and np.array_equal(got, want)
