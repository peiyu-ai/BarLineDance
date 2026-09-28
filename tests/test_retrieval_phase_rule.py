"""Retrieval by CONTEXT: where in the bar a prototype came from, not just its length.

MEASURED DEFECT (2026-09-01, 405 plan segments over 45 clips).  The shipped
``duration`` rule places a prototype at a median beat-phase error of **0.2381**
against the query's grid -- the uniform-random expectation is 0.250, so the rule
has no phase alignment at all -- and only 4.9% of picks land within 0.05 of the
right phase.  68.1% arrive at a tempo more than 10% away, after which
``_values_at`` linearly stretches them and destroys their internal timing.

WHY IT IS FIXABLE: the training release carries ``music.npy`` frame-aligned with
``motion.npy``, so every prototype knows the beat grid of the recording it was
cut from.  Census over the same segments: the median candidate pool is 481, of
which 26 sit within phase +-0.10 AND tempo +-10%; 87.0% of segments have at
least one such candidate and 84.5% have at least three, so the rule can be
selective and still leave the recurrence draw somewhere to go.

MEASURED EFFECT, and it is stated here because it is smaller than the defect
above: with the seams masked out, the pasted CONTENT's settle goes +0.0522 ->
+0.0655 against ground truth's +0.0772.  Real, but the content was never where
the phase inversion lived -- unmasked the draft reads -0.2356 and masked
-0.0011, so essentially all of it is in the seams.  This rule is therefore a
second-order fix, and these tests exist to keep it honest, not to sell it.
"""
import json
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from infer_atomic import (IndexedAtomicMotionLibrary, _grid_phase,  # noqa: E402
                          beat_grid_of)

BEAT_CHANNEL = 34
PERIOD = 15


def build_release(root, phases, length=30, frames=150):
    """A tiny release whose N samples each hold one prototype of label 1,
    starting at a chosen phase of their own beat grid."""
    root = pathlib.Path(root)
    train = root / "train"
    train.mkdir(parents=True)
    n = len(phases)
    motion = np.zeros((n, frames, 151), np.float32)
    labels = np.zeros((n, frames), np.int64)
    music = np.zeros((n, frames, 35), np.float32)
    starts = []
    for index, phase in enumerate(phases):
        music[index, ::PERIOD, BEAT_CHANNEL] = 1.0
        start = int(round(PERIOD * 3 + phase * PERIOD))
        starts.append(start)
        labels[index, start:start + length] = 1
        # a signature so the test can tell which prototype came back
        motion[index, :, 7:] = float(index + 1)
    np.save(train / "motion.npy", motion)
    np.save(train / "labels.npy", labels)
    np.save(train / "music.npy", music)
    (train / "names.json").write_text(json.dumps(["s{}".format(i) for i in range(n)]))
    torch.save({"data_min": torch.full((151,), -1.0),
                "data_max": torch.full((151,), 1.0)}, root / "normalizer.pt")
    return starts


def which(values):
    """Recover the prototype index from the signature written above."""
    return int(round(float(values[0, 7]))) - 1


# ------------------------------------------------------------- the primitives

@pytest.mark.parametrize("frame,expected", [(0, 0.0), (5, 1/3), (10, 2/3), (15, 0.0)])
def test_grid_phase_reads_the_position_inside_the_beat(frame, expected):
    grid = np.arange(0, 150, PERIOD)
    phase, period = _grid_phase(grid, frame)
    assert phase == pytest.approx(expected, abs=1e-9)
    assert period == pytest.approx(PERIOD)


def test_grid_phase_returns_nothing_rather_than_a_number_without_a_grid():
    """Every rule but ``phase`` must stay byte-identical, and that rests on this
    returning None instead of a default."""
    assert _grid_phase(None, 10) == (None, None)
    assert _grid_phase(np.array([0, 15]), 10)[0] is None


def test_beat_grid_of_reads_channel_34():
    music = np.zeros((150, 35), np.float32)
    music[::PERIOD, BEAT_CHANNEL] = 1.0
    assert np.array_equal(beat_grid_of(music), np.arange(0, 150, PERIOD))


def test_beat_grid_of_refuses_a_grid_too_thin_to_use():
    music = np.zeros((150, 35), np.float32)
    music[0, BEAT_CHANNEL] = 1.0
    assert beat_grid_of(music) is None


# ------------------------------------------------------------------ the rule

def test_it_picks_the_prototype_that_starts_at_the_query_s_phase(tmp_path):
    build_release(tmp_path, [0.0, 0.25, 0.5, 0.75])
    library = IndexedAtomicMotionLibrary(str(tmp_path), retrieval_rule="phase")
    for target, expected in ((0.0, 0), (0.25, 1), (0.5, 2), (0.75, 3)):
        library._retrieval_cache.clear()
        values = library.retrieve(1, 30, target_phase=target, target_beat_period=PERIOD)
        assert which(values) == expected, target


def test_the_phase_distance_wraps(tmp_path):
    """0.95 and 0.05 are a tenth of a beat apart, not nine tenths.  Without the
    wrap the rule would reject the best candidate exactly at the downbeat --
    the position that matters most."""
    build_release(tmp_path, [0.05, 0.45])
    library = IndexedAtomicMotionLibrary(str(tmp_path), retrieval_rule="phase")
    values = library.retrieve(1, 30, target_phase=0.95, target_beat_period=PERIOD)
    assert which(values) == 0


def test_with_no_query_phase_it_is_the_duration_rule(tmp_path):
    """So a run that supplies no beat grid reproduces the shipped behaviour."""
    build_release(tmp_path, [0.0, 0.5])
    phase_lib = IndexedAtomicMotionLibrary(str(tmp_path), retrieval_rule="phase")
    duration_lib = IndexedAtomicMotionLibrary(str(tmp_path), retrieval_rule="duration")
    assert torch.equal(phase_lib.retrieve(1, 30), duration_lib.retrieve(1, 30))


def test_the_duration_rule_ignores_a_phase_it_is_given(tmp_path):
    """The flag must not leak: every artifact made before this rule existed has
    to reproduce from its own command line."""
    build_release(tmp_path, [0.0, 0.5])
    library = IndexedAtomicMotionLibrary(str(tmp_path), retrieval_rule="duration")
    plain = library.retrieve(1, 30)
    library._retrieval_cache.clear()
    with_phase = library.retrieve(1, 30, target_phase=0.5, target_beat_period=PERIOD)
    assert torch.equal(plain, with_phase)


def test_duration_stays_a_hard_filter(tmp_path):
    """A perfectly phased prototype that would need a 50% stretch is not the
    answer: stretching destroys the internal timing this rule exists to keep."""
    root = pathlib.Path(tmp_path)
    train = root / "train"
    train.mkdir(parents=True)
    motion = np.zeros((2, 150, 151), np.float32)
    labels = np.zeros((2, 150), np.int64)
    music = np.zeros((2, 150, 35), np.float32)
    for index, (start, length) in enumerate(((45, 60), (49, 30))):
        music[index, ::PERIOD, BEAT_CHANNEL] = 1.0
        labels[index, start:start + length] = 1
        motion[index, :, 7:] = float(index + 1)
    np.save(train / "motion.npy", motion)
    np.save(train / "labels.npy", labels)
    np.save(train / "music.npy", music)
    (train / "names.json").write_text(json.dumps(["a", "b"]))
    torch.save({"data_min": torch.full((151,), -1.0),
                "data_max": torch.full((151,), 1.0)}, root / "normalizer.pt")
    library = IndexedAtomicMotionLibrary(str(root), retrieval_rule="phase")
    # sample 0 is exactly on phase 0 but 60 frames long; sample 1 is 30 frames
    # and off phase.  Asking for 30 must not return the 60-frame one.
    values = library.retrieve(1, 30, target_phase=0.0, target_beat_period=PERIOD)
    assert which(values) == 1


def test_tempo_breaks_a_phase_tie(tmp_path):
    root = pathlib.Path(tmp_path)
    train = root / "train"
    train.mkdir(parents=True)
    motion = np.zeros((2, 180, 151), np.float32)
    labels = np.zeros((2, 180), np.int64)
    music = np.zeros((2, 180, 35), np.float32)
    for index, step in enumerate((PERIOD, PERIOD * 2)):
        music[index, ::step, BEAT_CHANNEL] = 1.0
        labels[index, step * 2:step * 2 + 30] = 1      # phase 0.0 in both
        motion[index, :, 7:] = float(index + 1)
    np.save(train / "motion.npy", motion)
    np.save(train / "labels.npy", labels)
    np.save(train / "music.npy", music)
    (train / "names.json").write_text(json.dumps(["a", "b"]))
    torch.save({"data_min": torch.full((151,), -1.0),
                "data_max": torch.full((151,), 1.0)}, root / "normalizer.pt")
    library = IndexedAtomicMotionLibrary(str(root), retrieval_rule="phase")
    assert which(library.retrieve(1, 30, target_phase=0.0, target_beat_period=PERIOD)) == 0
    library._retrieval_cache.clear()
    assert which(library.retrieve(1, 30, target_phase=0.0,
                                  target_beat_period=PERIOD * 2)) == 1


# --------------------------------------------------------------- the plumbing

def test_the_cache_key_separates_two_segments_at_different_phases():
    """Same class, same length, different position in the bar -- the whole point
    of the rule.  Sharing a cache entry would make it look like it did nothing."""
    import inspect

    source = inspect.getsource(IndexedAtomicMotionLibrary.retrieve)
    key = source[source.index("key = ("):source.index("if self.retrieval_rule ==")]
    assert "target_phase" in key and "target_beat_period" in key


def test_a_repeat_draws_from_the_phase_equivalent_set_not_the_duration_band(tmp_path):
    """Recurrence variety must survive the rule, and must not undo it: about
    60% of segments are repeats, so drawing from the duration band again would
    throw the alignment away on most of them."""
    build_release(tmp_path, [0.0, 0.0, 0.0, 0.5, 0.5])
    library = IndexedAtomicMotionLibrary(str(tmp_path), retrieval_rule="phase")
    seen = set()
    for occurrence in range(1, 8):
        library._retrieval_cache.clear()
        values = library.retrieve(1, 30, target_phase=0.0, target_beat_period=PERIOD,
                                  occurrence=occurrence,
                                  variety_rng=np.random.default_rng(occurrence))
        seen.add(which(values))
    assert seen <= {0, 1, 2}, seen        # never the phase-0.5 prototypes
    assert len(seen) > 1, seen            # and it really did vary
