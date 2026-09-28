"""--draft-rhythm-scorer / -keep / -shift: the learned music-motion alignment in retrieval (DEFECTS §93).

Pinned here without a checkpoint (the scorer is replaced by a stub that reads the candidate's start):
  * the keep filter narrows to the best-scoring ceil(keep * n) candidates, never fewer than two, and hands
    them back in their ORIGINAL order (the join band and the selector downstream see a subset, not a
    re-ranking);
  * it is a no-op without music, with too few candidates, or with keep 0 -- and counts what it did;
  * the phase shift keeps the unit's length, stays inside its window, and picks the best-scoring offset;
  * the kinetics the scorer reads are blind to where the dancer stands and which way they face (a turn of
    the whole body or a walk must not look like a different rhythm), and a real scorer reads nothing but
    (music, kinetics) -- no target motion reaches it (CLAUDE.md 1.6).
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic  # noqa: E402
from model import rhythm_scorer  # noqa: E402

Lib = infer_atomic.IndexedAtomicMotionLibrary
MUSIC = np.zeros((48, 35), dtype=np.float32)


def _stub(keep=0.0, shift=0, score=lambda c: -abs(c[1] - 7)):
    lib = Lib.__new__(Lib)
    lib.rhythm_keep = keep
    lib.rhythm_shift = shift
    lib.labels = np.zeros((1, 100))
    lib._rhythm_scores = lambda candidates, music: np.array([score(c) for c in candidates], dtype=float)
    return lib


def test_keep_narrows_to_the_best_quarter_in_original_order():
    tied = [(0, s, s + 40, "g") for s in (20, 7, 0, 9, 6, 30, 8, 1)]
    kept = _stub(keep=0.25)._prefer_rhythm(tied, MUSIC)
    assert kept == [(0, 7, 47, "g"), (0, 6, 46, "g")]            # scores 0 and -1; order as given


def test_keep_never_leaves_fewer_than_two():
    tied = [(0, s, s + 40, "g") for s in (7, 20, 30, 40)]
    assert len(_stub(keep=0.01)._prefer_rhythm(tied, MUSIC)) == 2


def test_keep_is_a_no_op_without_music_or_candidates_and_counts_what_it_did():
    tied = [(0, s, s + 40, "g") for s in (7, 20, 30, 40)]
    lib = _stub(keep=0.25)
    assert lib._prefer_rhythm(tied, None) == tied
    assert lib._prefer_rhythm(tied[:2], MUSIC) == tied[:2]
    assert _stub(keep=0.0)._prefer_rhythm(tied, MUSIC) == tied
    assert not hasattr(lib, "rhythm_slots")
    lib._prefer_rhythm(tied, MUSIC)
    assert lib.rhythm_slots == 1 and lib.rhythm_applied == 1


def test_phase_shift_keeps_length_stays_in_the_window_and_takes_the_best_offset():
    lib = _stub(shift=6)
    moved, shift = lib._rhythm_phase((0, 3, 43, "g"), MUSIC)
    assert (moved, shift) == ((0, 7, 47, "g"), 4)
    # at the window's start only non-negative offsets exist; the best reachable one is taken
    lib = _stub(shift=6, score=lambda c: -c[1])
    assert lib._rhythm_phase((0, 2, 42, "g"), MUSIC) == ((0, 0, 40, "g"), -2)
    # at its end, only offsets that keep end <= frames
    lib = _stub(shift=6, score=lambda c: c[1])
    assert lib._rhythm_phase((0, 50, 97, "g"), MUSIC) == ((0, 53, 100, "g"), 3)
    assert lib.rhythm_shift_slots == 1 and lib.rhythm_shift_moved == 1


def test_phase_shift_is_off_without_the_flag_or_the_music():
    assert _stub(shift=0)._rhythm_phase((0, 3, 43, "g"), MUSIC) == ((0, 3, 43, "g"), 0)
    assert _stub(shift=6)._rhythm_phase((0, 3, 43, "g"), None) == ((0, 3, 43, "g"), 0)


def _dancer(frames=40, seed=0):
    rng = np.random.default_rng(seed)
    j = np.cumsum(rng.normal(0, 0.01, (frames, 24, 3)), axis=0) + rng.normal(0, 0.3, (1, 24, 3))
    j[:, 1] = j[:, 0] + [-0.1, 0.0, 0.0]         # left hip
    j[:, 2] = j[:, 0] + [0.1, 0.0, 0.0]          # right hip: hip axis along world x
    return j


def test_kinetics_do_not_see_where_the_dancer_stands_or_faces():
    j = _dancer()
    base = rhythm_scorer.kinetic_frames(j)
    assert base.shape == (40, rhythm_scorer.KIN_CHANNELS)
    moved = j + np.array([3.0, -2.0, 0.0])
    assert np.allclose(rhythm_scorer.kinetic_frames(moved), base, atol=1e-5)
    a = np.deg2rad(70.0)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    assert np.allclose(rhythm_scorer.kinetic_frames(j @ rot.T), base, atol=1e-5)


def test_the_scorer_reads_music_and_kinetics_only():
    model = rhythm_scorer.RhythmScorer(width=16, blocks=(1, 2), music_channels=[0, 33, 34]).eval()
    kin = torch.from_numpy(rhythm_scorer.kinetic_frames(_dancer(rhythm_scorer.CROP)))[None]
    kin = rhythm_scorer.normalise_kinetics(kin)
    music = torch.randn(1, rhythm_scorer.CROP, rhythm_scorer.MUSIC_CHANNELS)
    with torch.no_grad():
        a = model(music, kin)
        # timbre/harmony channels are not read: changing them leaves the score unchanged
        other = music.clone(); other[..., 1:33] = torch.randn(1, rhythm_scorer.CROP, 32)
        b = model(other, kin)
        c = model(music, kin.roll(6, dims=1))
    assert a.shape == (1,) and torch.allclose(a, b)
    assert not torch.allclose(a, c)                     # but when the motion moves against the beat, it does
