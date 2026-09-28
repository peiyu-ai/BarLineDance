"""--draft-motif-return: a move comes back on the phrase grid (DEFECTS §94).

Pinned on a stub library (no release on disk): which earlier bar may come back, and which may not --
  * lags are tried in the given order (4 before 2 by default);
  * a bar that was itself a return is never returned again (no chains), nor is a source returned twice;
  * the source must fit the slot inside the duration band every retrieval rule uses;
  * label 0 (filler / transition) is never brought back;
  * the options are validated (probability in [0, 1], lags >= 2).
"""
import collections
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic  # noqa: E402

Lib = infer_atomic.IndexedAtomicMotionLibrary
Seg = collections.namedtuple("Seg", "start end length label")


def _stub(lags=(4, 2)):
    lib = Lib.__new__(Lib)
    lib.motif_lags = lags
    labels = np.full((6, 200), 3)
    labels[5] = 0                               # sample 5 is filler
    lib.labels = labels
    return lib


def _unit(sample, length=60):
    return (sample, 0, length, "g%d" % sample)


def test_lags_are_tried_in_order():
    lib = _stub()
    placed = {1: (_unit(1), False), 3: (_unit(3), False)}
    seg = Seg(300, 360, 60, 3)
    assert lib._motif_candidate(5, seg, placed, set()) == (_unit(1), 1, 4)
    assert _stub(lags=(2, 4))._motif_candidate(5, seg, placed, set()) == (_unit(3), 3, 2)


def test_no_chains_and_no_second_return_of_a_source():
    lib = _stub()
    seg = Seg(300, 360, 60, 3)
    assert lib._motif_candidate(5, seg, {1: (_unit(1), True)}, set()) is None
    assert lib._motif_candidate(5, seg, {1: (_unit(1), False)}, {(1, 0, 60)}) is None


def test_duration_band_and_filler():
    lib = _stub()
    assert lib._motif_candidate(5, Seg(300, 400, 100, 3), {1: (_unit(1, 60), False)}, set()) is None
    assert lib._motif_candidate(5, Seg(300, 360, 60, 3), {1: (_unit(5), False)}, set()) is None


def test_options_are_validated():
    with pytest.raises(ValueError, match="motif-return is a probability"):
        Lib.__init__(Lib.__new__(Lib), data_root="/nonexistent", motif_return=1.5)
    with pytest.raises(ValueError, match="motif-lags must be >= 2"):
        Lib.__init__(Lib.__new__(Lib), data_root="/nonexistent", motif_return=0.2, motif_lags=(1,))


def test_lively_pick_skips_a_quieter_than_median_source():
    lib = _stub()
    lib.motif_pick = "lively"
    energy = {1: 0.01, 2: 0.05, 3: 0.04}                  # sample -> mean speed
    lib._unit_energy = lambda c: energy[c[0]]
    placed = {1: (_unit(1), False), 2: (_unit(2), False), 3: (_unit(3), False)}
    seg = Seg(300, 360, 60, 3)
    # lag 4 -> bar 1 is the quiet one (0.01 < median 0.04): skipped; lag 2 -> bar 3 (0.04) qualifies
    assert lib._motif_candidate(5, seg, placed, set()) == (_unit(3), 3, 2)
    assert lib.motif_quiet_skipped == 1
