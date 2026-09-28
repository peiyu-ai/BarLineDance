"""--draft-hold-by-music: hold in the quiet bars, move in the busy ones.

THE TARGET, measured on the twenty eval clips and reading only the music: per
bar, the share of frames holding a shape against that bar's onset-peak density
(channel 33) is NEGATIVE for ground truth on 16 of 20 clips, mean -0.225,
two-sided sign test P=0.0118.  The shipped arm reads -0.094 (11/20, P=0.82).

These tests are behavioural.  The first two build a pool where holding and
moving prototypes are tied on duration -- the situation today's rule cannot
tell apart -- and check that the filter picks by the music.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from infer_atomic import IndexedAtomicMotionLibrary  # noqa: E402

SOURCE = pathlib.Path("infer_atomic.py").read_text()


class Fake(IndexedAtomicMotionLibrary):
    def __init__(self, on, shares):
        self.hold_by_music = bool(on)
        self._shares = shares

    def _hold_share(self, candidate):
        return self._shares[candidate]


HOLDS = (0, 0, 60, 0)
MOVES = (1, 0, 60, 0)
MIDDLE = (2, 0, 60, 0)
SHARES = {HOLDS: 0.40, MIDDLE: 0.10, MOVES: 0.00}
POOL = [HOLDS, MIDDLE, MOVES]


def test_the_pool_is_tied_on_duration_which_is_the_defect():
    lengths = {c[2] - c[1] for c in POOL}
    assert len(lengths) == 1, "the fixture must reproduce the tie today's rule cannot break"


def test_a_quiet_bar_keeps_the_holding_candidates():
    kept = Fake(True, SHARES)._prefer_hold_by_music(POOL, True)
    assert HOLDS in kept and MOVES not in kept


def test_a_busy_bar_keeps_the_moving_candidates():
    kept = Fake(True, SHARES)._prefer_hold_by_music(POOL, False)
    assert MOVES in kept and HOLDS not in kept


def test_off_is_a_no_op():
    assert Fake(False, SHARES)._prefer_hold_by_music(POOL, True) == POOL


def test_no_density_is_counted_not_guessed():
    library = Fake(True, SHARES)
    assert library._prefer_hold_by_music(POOL, None) == POOL
    assert getattr(library, "hold_music_blind", 0) == 1


def test_a_quiet_bar_with_nothing_that_holds_keeps_the_whole_tie():
    library = Fake(True, {HOLDS: 0.0, MIDDLE: 0.0, MOVES: 0.0})
    assert library._prefer_hold_by_music(POOL, True) == POOL
    assert getattr(library, "hold_music_empty", 0) == 1


def test_the_rot6d_hold_share_reads_a_real_hold():
    """The cheap reading must see a hold that is actually there."""
    frames = 60
    motion = np.zeros((1, frames, 151), np.float32)
    rng = np.random.default_rng(0)
    motion[0, :, 7:] = np.cumsum(rng.normal(size=(frames, 144)), axis=0)
    motion[0, 20:40, 7:] = motion[0, 20, 7:]          # held for 20 frames

    class Lib(IndexedAtomicMotionLibrary):
        def __init__(self):
            self.motion = motion
            self._rest_cache = {}
    share = Lib()._hold_share((0, 0, frames, 0))
    assert share > 0.15, share


def test_it_is_wired_at_every_site_and_recorded():
    assert SOURCE.count("_prefer_hold_by_music(") == 4          # def + 3 sites
    assert '"--draft-hold-by-music"' in SOURCE
    assert "hold_by_music=draft_hold_by_music" in SOURCE
    assert "draft_hold_by_music=options.draft_hold_by_music" in SOURCE
    assert "target_quiet=segment_quiet[position]" in SOURCE
    for key in ("draft_hold_music_slots", "draft_hold_music_applied",
                "draft_hold_music_empty", "draft_hold_music_blind"):
        assert '"{}"'.format(key) in SOURCE, key
