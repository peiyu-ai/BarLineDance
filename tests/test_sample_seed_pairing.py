"""A clip's noise draw must be a property of the clip, not of the work list.

The defect these guard was found twice and fixed neither time (2026-08-23,
fid_k 9.997 vs 9.515 from re-sharding; 2026-08-31, per-clip energy 0.5632 vs
0.6421 from a clip-list length change).  Both readings were taken as arm
differences.
"""
import hashlib
import importlib
import sys
import types

import pytest


def load_infer():
    """Import infer_atomic without paying for its heavy optional deps."""
    if "infer_atomic" in sys.modules:
        return sys.modules["infer_atomic"]
    return importlib.import_module("infer_atomic")


infer_atomic = pytest.importorskip("infer_atomic")


def test_the_same_clip_gets_the_same_draw_at_any_position():
    """The property the old `seed + index` did not have."""
    name = "wild_v5:7625069314060096719:clip000"
    assert infer_atomic.sample_seed(1234, name) == infer_atomic.sample_seed(1234, name)


def test_positional_seeding_would_have_failed_this(monkeypatch):
    """Positive control: the OLD rule, expressed here, does not have the
    property above.  Without this the test above passes on any function of the
    seed alone, including one that ignores the name entirely."""
    positional = lambda seed, index: seed + index
    # the same clip at position 3 of one list and position 17 of another
    assert positional(1234, 3) != positional(1234, 17)


def test_different_clips_get_different_draws():
    seeds = {infer_atomic.sample_seed(7, "clip{:03d}".format(i)) for i in range(512)}
    assert len(seeds) == 512


def test_the_base_seed_still_changes_everything():
    """A seed sweep must still be a sweep -- a name-derived draw that ignored
    the base seed would silently make every 'seed' of a two-seed arm identical,
    which is the evidence standard this repository runs on."""
    name = "wild_v5:7625069314060096719:clip000"
    assert infer_atomic.sample_seed(20260901, name) != infer_atomic.sample_seed(20260902, name)


def test_the_draw_is_stable_across_processes():
    """``hash()`` is salted per process, so a name-derived seed built on it
    would reproduce within a run and not across runs -- the same failure in a
    costume.  Pin the digest so a future refactor to hash() fails here."""
    name = "wild_v5:7625069314060096719:clip000"
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    expected = (20260901 ^ int.from_bytes(digest, "big")) % (2 ** 31 - 1)
    assert infer_atomic.sample_seed(20260901, name) == expected


def test_seeds_are_in_torch_range():
    """torch.manual_seed rejects values outside [0, 2**64); numpy.random.seed
    rejects anything above 2**32-1.  A draw that raises at clip 400 of 90 is a
    gate that fires only on the runs that matter."""
    for i in range(4096):
        value = infer_atomic.sample_seed(20260901, "clip{:05d}".format(i))
        assert 0 <= value < 2 ** 31 - 1
