"""The retrieval floor must remove frozen exemplars and nothing else.

Measured defect it exists for (2026-09-01, the 90 shipped clips): 31 of 1,277
prototype pastes are under 0.20 m/s, covering 1,214 of 45,178 output frames and
touching 25 of 90 clips; the worst clip is 37.1% frozen frames and reads
generated energy 0.200 against a ground truth of 0.699.

The design risk is the opposite error, and it is the one these tests are mostly
about: 27 of those 31 came from classes that are low-energy BY NATURE (M3 tags
"standing pose sustained", "arms crossed pose").  An absolute floor would empty
those classes and silently substitute a different movement.  A within-class
quantile must leave every class non-empty and still return that class.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


class FakeLibrary:
    """Only the two methods under test, over a synthetic motion array."""

    from infer_atomic import IndexedAtomicMotionLibrary as _real
    _segment_energy = _real._segment_energy
    _energy_floor = _real._energy_floor

    def __init__(self, motion):
        self.motion = motion
        self._energy_cache = {}


def library_with(amplitudes, length=30):
    """One candidate per amplitude, each a sinusoid of that amplitude."""
    from infer_atomic import ROOT_POSITION_START
    dims = ROOT_POSITION_START + 3 + 12
    motion = np.zeros((len(amplitudes), length, dims), np.float32)
    phase = np.linspace(0, 4 * np.pi, length)[:, None]
    for index, amplitude in enumerate(amplitudes):
        motion[index, :, ROOT_POSITION_START + 3:] = amplitude * np.sin(phase)
    candidates = tuple((i, 0, length, "group{}".format(i)) for i in range(len(amplitudes)))
    return FakeLibrary(motion), candidates


def test_energy_orders_candidates_by_their_actual_amplitude():
    library, candidates = library_with([0.0, 0.01, 0.1, 1.0])
    values = [library._segment_energy(c) for c in candidates]
    assert values == sorted(values)
    assert values[0] == 0.0


def test_the_frozen_tail_is_dropped():
    """Note the units: _segment_energy is a mean first difference, so a sinusoid
    of amplitude A reads ~0.27A, not A.  The assertion is therefore against the
    measured energies, not against the amplitudes that produced them -- my first
    version asserted on amplitude and failed for that reason alone."""
    library, candidates = library_with([0.0, 0.0, 0.01, 0.5, 0.6, 0.7, 0.8, 1.0])
    energies = {c: library._segment_energy(c) for c in candidates}
    kept = library._energy_floor(1, candidates, 0.25)
    assert len(kept) < len(candidates)
    # the two frozen ones are gone, and nothing above the cut was touched
    assert all(energies[c] > 0.0 for c in kept)
    assert min(energies[c] for c in kept) > min(energies.values())
    assert set(kept) <= set(candidates)


def test_off_by_default_reproduces_the_old_candidate_set_exactly():
    """Every artifact generated before this flag must stay reproducible."""
    library, candidates = library_with([0.0, 0.1, 0.5, 1.0])
    for quantile in (None, 0.0):
        assert library._energy_floor(1, candidates, quantile) is candidates


def test_a_class_that_is_low_energy_by_nature_is_never_emptied():
    """The failure an ABSOLUTE floor would cause: 27 of the 31 bad pastes came
    from pose classes, and emptying them would substitute a different movement
    while the manifest still claimed the planned one."""
    library, candidates = library_with([0.001, 0.002, 0.003, 0.004, 0.005])
    energies = {c: library._segment_energy(c) for c in candidates}
    kept = library._energy_floor(1, candidates, 0.5)
    assert kept, "a pose class must still return a prototype"
    assert set(kept) <= set(candidates)
    # it keeps that class's own better half, not some global survivor: every
    # survivor is above the class's own median, and the class is still served
    # even though every one of its members would fail any absolute floor.
    assert min(energies[c] for c in kept) > np.median(list(energies.values()))
    assert max(energies.values()) < 0.01


def test_a_tiny_pool_is_left_alone():
    """With 3 candidates a quartile is one sample; dropping on that is noise,
    and the pool is thin exactly where substitution hurts most."""
    library, candidates = library_with([0.0, 0.5, 1.0])
    assert library._energy_floor(1, candidates, 0.5) is candidates


def test_a_quantile_of_one_still_returns_something():
    """Able to fail loudly rather than hand back an empty tuple, which would
    surface far downstream as 'no training prototype for atomic label N'."""
    library, candidates = library_with([0.1, 0.2, 0.3, 0.4, 0.5])
    assert library._energy_floor(1, candidates, 1.0)


def test_higher_quantile_drops_more():
    library, candidates = library_with(list(np.linspace(0.0, 1.0, 20)))
    sizes = [len(library._energy_floor(1, candidates, q)) for q in (0.0, 0.25, 0.5, 0.75)]
    assert sizes == sorted(sizes, reverse=True), sizes


def test_the_cache_key_separates_floored_and_unfloored_runs():
    """Two runs differing only in the floor must not share retrieval cache
    entries -- that would make the second run silently reuse the first's picks,
    which reads exactly like 'the flag did nothing'."""
    import inspect
    from infer_atomic import IndexedAtomicMotionLibrary
    source = inspect.getsource(IndexedAtomicMotionLibrary.retrieve)
    key_line = source[source.index("key = ("):source.index("if self.retrieval_rule ==")]
    assert "energy_floor_quantile" in key_line
