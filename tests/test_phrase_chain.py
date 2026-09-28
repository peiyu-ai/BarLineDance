"""--draft-phrase-chain (H series): at a phrase start, keep the candidates whose source dancer can be continued
through the REST of the phrase, so the phrase does not break where an upload ends or its next bar misses the band.

Pinned on a hand-built library stub (no release on disk):
  * the chain is walked with ``_next_source_bar`` -- the same test the draft applies when it continues;
  * the longest available chain wins, and at least two candidates survive (a narrowing, never an empty pool);
  * off (the default) and outside a phrase start (no remaining bar lengths) the pool is returned untouched.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic  # noqa: E402

Lib = infer_atomic.IndexedAtomicMotionLibrary


def _stub(on=True):
    lib = Lib.__new__(Lib)
    # three one-window recordings: A carries three more bars, B one more, C none (its upload ends)
    lib._window_origin = {0: ("A", 0), 1: ("B", 0), 2: ("C", 0), 3: ("D", 0)}
    lib._source_bars = {
        ("A", 50): [(0, 50, 100, 1)], ("A", 100): [(0, 100, 150, 1)], ("A", 150): [(0, 150, 200, 1)],
        ("B", 50): [(1, 50, 100, 1)], ("B", 100): [(1, 100, 170, 1)],     # B's second bar is 70 long: out of band
        ("D", 50): [(3, 50, 100, 1)], ("D", 100): [(3, 100, 150, 1)], ("D", 150): [(3, 150, 200, 1)],
    }
    lib.retrieval_group_ids = ["gA", "gB", "gC", "gD"]
    lib.names = ["A", "B", "C", "D"]
    lib.continue_any_label = True
    lib._yaw_steps = lib._speed_spikes = None
    lib.max_yaw_step = lib.max_speed_spike = None
    lib.phrase_chain = on
    return lib


TIED = [(0, 0, 50, "gA"), (1, 0, 50, "gB"), (2, 0, 50, "gC")]


def test_the_candidate_that_carries_the_whole_phrase_is_kept():
    lib = _stub()
    kept = lib._prefer_phrase_chain(TIED + [(3, 0, 50, "gD")], label=1, lengths=[50, 50, 50])
    assert kept == [(0, 0, 50, "gA"), (3, 0, 50, "gD")]
    assert lib.phrase_chain_full == 2 and lib.phrase_chain_seen == 4 and lib.phrase_chain_applied == 1


def test_at_least_two_survive_ordered_by_reach():
    # A reaches 3, B reaches 1 (its second bar misses the band), C reaches 0: A alone is best, B is the runner-up
    assert _stub()._prefer_phrase_chain(TIED, label=1, lengths=[50, 50, 50]) == [(0, 0, 50, "gA"), (1, 0, 50, "gB")]


def test_off_or_not_a_phrase_start_leaves_the_pool_alone():
    assert _stub(on=False)._prefer_phrase_chain(TIED, label=1, lengths=[50, 50, 50]) == TIED
    assert _stub()._prefer_phrase_chain(TIED, label=1, lengths=[]) == TIED
