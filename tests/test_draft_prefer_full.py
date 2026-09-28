"""--draft-prefer-full: keep the fullest fraction of the candidates before the join band and selector.

Why (DEFECTS §92, 2026-09-23): inside a slot's filtered pool, fullness spreads 1.9 z from p10 to p90 and
the shipped chain picks at the 41st percentile.  Keeping the fullest quarter by the ARRIVAL score
('stops') made moves land straighter on test-20 and val-30 (elbow at the stops +7 / +10 deg, 15/20 and
24/30 clips) inside the energy band; the all-frames score ('frames') also lifted the peaks but cost
energy (1.20 / 1.22x ground truth).
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic  # noqa: E402

Lib = infer_atomic.IndexedAtomicMotionLibrary


def _stub(mode, frames):
    lib = Lib.__new__(Lib)
    lib.prefer_full = 0.25
    lib.prefer_full_mode = mode
    lib._full_cache = {}
    flat = frames[..., :3].reshape(-1, 3)
    lib._full_frames = (frames, flat.mean(0), flat.std(0) + 1e-6)
    if frames.ndim == 4:
        speed = frames[..., 3].reshape(-1)
        lib._full_speed_stats = (float(speed.mean()), float(speed.std()) + 1e-6)
    return lib


def test_frames_mode_keeps_the_fullest_quarter_and_at_least_two():
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(8, 40, 3)).astype(np.float32)
    for k in range(8):
        frames[k] += k                       # sample k is uniformly "fuller" than k-1
    lib = _stub("frames", frames)
    pool = [(k, 0, 40, "g") for k in range(8)]
    kept = lib._prefer_full(pool)
    assert [c[0] for c in kept] == [7, 6]    # ceil(8 * 0.25) = 2, fullest first
    assert lib.prefer_full_slots == 1 and lib.prefer_full_applied == 1
    assert lib._prefer_full(pool[:2]) == pool[:2]     # fewer than three: untouched


def test_stops_mode_scores_the_arrivals_not_the_sweep():
    # two candidates with the same peak pose over their frames; only one ARRIVES at it (speed dips there)
    frames = np.zeros((2, 30, 2, 4), dtype=np.float32)
    frames[:, :, :, 3] = 0.03                               # steady 3 cm/frame of wrist travel
    frames[0, 14:17, :, :3] = [1.0, 170.0, 1.0]             # sample 0 sweeps THROUGH the full pose...
    frames[1, 20, :, :3] = [1.0, 170.0, 1.0]                # ...sample 1 LANDS on it:
    frames[1, 20, :, 3] = 0.001                             # a stop after >= 5 cm of travel
    lib = _stub("stops", frames)
    assert lib._fullness((1, 0, 30, "g")) > lib._fullness((0, 0, 30, "g"))
