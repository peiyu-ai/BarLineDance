"""The circuit breaker that stops a shard before it marks a whole corpus.

Stage B's failure markers cannot be deleted -- the credentials this project
holds are write-only against the bucket -- so a shard that keeps working
through a broken environment converts an environment fault into permanent
corpus state, and does it *faster* than a healthy shard converts video into
motion, because a failure returns in seconds and a success takes a minute.

Both of 2026-08-13's incidents had that shape: a pod rebuild dropped GVHMR's
pip dependencies (666 markers), and a numpy upgrade broke torch's numpy bridge
(1,669 markers in 24 minutes).  The breaker exists for those.  Its first
version counted every failure and tripped within ten clips on eight genuine
visual-odometry divergences, which is what these tests are here to prevent
coming back.
"""

from tools.run_gvhmr_ingest_shard_oss import consecutive_unattributed


def test_a_success_resets_the_counter():
    assert consecutive_unattributed(7, {"ok": True, "seconds": 51.3}) == 0


def test_an_unexplained_failure_advances_it():
    record = {"ok": False, "exit": 1, "reason": "no EXTRACT_FAIL line; exit=1"}
    assert consecutive_unattributed(0, record) == 1
    assert consecutive_unattributed(6, record) == 7


def test_a_diverged_clip_does_not_count_however_many_arrive():
    """Eight in a row is what tripped the first version, on shard 1, wrongly.

    The recovery run reopens every clip that failed before, so the population
    it walks is enriched for exactly this failure -- a breaker that counts it
    fires on the corpus rather than on the environment.
    """
    diverged = {"ok": False, "exit": 1, "attributed": True,
                "reason": "EXTRACT_FAIL x visual odometry diverged: non-finite camera track"}
    streak = 0
    for _ in range(8):
        streak = consecutive_unattributed(streak, diverged)
    assert streak == 0


def test_a_timeout_counts_as_explained():
    # A hang is a fact about the clip: the worker knows what happened to it,
    # and the marker it writes can be acted on later.
    record = {"ok": False, "exit": 124, "attributed": True,
              "reason": "timeout after 1800s"}
    assert consecutive_unattributed(5, record) == 0


def test_a_convert_step_that_dies_is_not_attributed():
    # This is the path the numpy break arrived through, and it names an
    # exception -- which is not the same as attributing the failure to the
    # clip.  Anything the extractor itself did not diagnose has to count.
    record = {"ok": False, "exit": 1,
              "reason": "convert failed: RuntimeError: Numpy is not available"}
    assert consecutive_unattributed(0, record) == 1


def test_one_success_between_two_faults_clears_the_streak():
    # Otherwise a shard that limps -- succeeding occasionally on cached work
    # while the environment is broken -- would still be shut down, and the
    # breaker would be a slow failure rather than a fast one.
    broken = {"ok": False, "reason": "no EXTRACT_FAIL line; exit=1"}
    streak = 0
    for _ in range(4):
        streak = consecutive_unattributed(streak, broken)
    streak = consecutive_unattributed(streak, {"ok": True, "seconds": 60.0})
    assert streak == 0
    assert consecutive_unattributed(streak, broken) == 1


# ---------------------------------------------------------------------------
# Re-deriving clips whose 3D already exists.
#
# The skip sets here are built from what is *present* in the store -- a clip
# with a quality.json is done -- and on a corpus whose failure mode is contents
# that is the wrong question.  After the fps re-cut, 3D extracted from the old
# video sits under the right name with the right shape, and both the freeze and
# the shard pass over it.  The redo list is how a measurement (which clips were
# built from bytes the clip no longer has) reaches those two decisions.
# ---------------------------------------------------------------------------

import json as _json
import pathlib as _pathlib
import sys as _sys
import tempfile as _tempfile
from unittest import mock as _mock

import pytest as _pytest

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

from tools import run_gvhmr_ingest_shard_oss as _stage_b


class _Store:
    """The two prefixes stage B reads, and the one object it writes."""

    def __init__(self, worklist, done=(), failed=()):
        self.worklist = list(worklist)
        self.done = set(done)
        self.failed = set(failed)
        self.written = {}

    def read_json(self, path):
        if path == _stage_b.WORKLIST:
            return {"clips": [{"clip": c, "bbx": True} for c in self.worklist]}
        return self.written[path]

    def write_json(self, path, obj, indent=2):
        self.written[path] = obj
        return path

    def stems_with(self, prefix, filename):
        return set(self.done if prefix == _stage_b.CONVERTED_ROOT else self.failed)

    def patch(self):
        return (
            _mock.patch.object(_stage_b.asset_io, "read_json", self.read_json),
            _mock.patch.object(_stage_b.asset_io, "write_json", self.write_json),
            _mock.patch.object(_stage_b, "stems_with", self.stems_with),
        )


def _freeze(store, **kwargs):
    patches = store.patch()
    for patch in patches:
        patch.start()
    try:
        return _stage_b.freeze_todo(4, **kwargs)
    finally:
        for patch in patches:
            patch.stop()


def test_without_a_redo_list_a_converted_clip_stays_skipped():
    store = _Store(worklist=["a", "b", "c"], done=["a", "b"])
    assert _freeze(store) == ["c"]


def test_a_named_clip_returns_to_the_list_although_its_3d_exists():
    store = _Store(worklist=["a", "b", "c"], done=["a", "b"])
    assert _freeze(store, redo=["b"]) == ["b", "c"]


def test_a_named_clip_carrying_a_failure_marker_is_retried_too():
    """The marker cannot be deleted, so ignoring it means never redoing it."""
    store = _Store(worklist=["a", "b"], done=[], failed=["b"])
    assert _freeze(store) == ["a"]
    assert _freeze(store, redo=["b"]) == ["a", "b"]


def test_a_name_the_worklist_never_had_is_added_not_dropped():
    """A re-cut re-splits uploads, so clip names appear that predate no list.

    Subtraction alone can never reach them, and dropping them silently is how a
    re-derivation covers part of a corpus and reports success.
    """
    store = _Store(worklist=["a", "b"], done=["a", "b"])
    assert _freeze(store, redo=["b", "b__new"]) == ["b", "b__new"]
    frozen = store.written[_stage_b.TODO_LIST]
    assert frozen["redo_absent_from_worklist"] == ["b__new"]
    assert frozen["redo_named"] == ["b", "b__new"]


def test_the_shard_reads_the_redo_set_out_of_the_frozen_list():
    """A shard flag could disagree with the freeze; the frozen list cannot."""
    store = _Store(worklist=["a", "b", "c", "d"], done=["a", "b"])
    _freeze(store, redo=["b"])
    frozen = store.written[_stage_b.TODO_LIST]
    assert frozen["clips"] == ["b", "c", "d"]
    assert set(frozen["redo_named"]) == {"b"}


def test_redo_without_freeze_is_refused_rather_than_ignored():
    with _tempfile.TemporaryDirectory() as workspace:
        listing = _pathlib.Path(workspace) / "redo.json"
        listing.write_text(_json.dumps({"stale": {"3d": ["b"], "s3d": []}}),
                           encoding="utf-8")
        argv = ["--redo", str(listing), "--shard", "0", "--num-shards", "1"]
        with _mock.patch.object(_sys, "argv", ["prog"] + argv), \
             _pytest.raises(SystemExit) as raised:
            _stage_b.main()
    assert "only takes effect at --freeze" in str(raised.value)
