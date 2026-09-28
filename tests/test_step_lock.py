"""--draft-step-lock-keep (J series): prefer the units whose FEET land on this song's beats when played into the slot.

Pinned without a release on disk:
  * a landing is a foot that travelled (> 0.6 m/s) and stopped (< 0.25 m/s); standing still or drifting is not one;
  * a landing is scored where the draft will PLAY it (stretched into the slot, align_corners), on the slot's own beats:
    0.2 beat after a beat scores +1, half a beat later -1, no landings 0;
  * the filter keeps the best fraction (at least two), scores the phrase continuation will play at a phrase start,
    and is inert when off or when the slot has no beat grid.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic  # noqa: E402
from infer_atomic import _foot_plant_frames, _step_lock_score  # noqa: E402

Lib = infer_atomic.IndexedAtomicMotionLibrary


def _joints_with_steps(frames, landings):
    j = np.zeros((frames, 24, 3))
    x = 0.0
    for t in range(frames):
        if any(0 < land - t <= 6 for land in landings):
            x += 1.0 / 30.0                    # 1.0 m/s for the 6 frames before each landing
        j[t, 10, 0] = x
    return j


def test_a_foot_that_travels_and_stops_is_one_landing_per_step():
    p = _foot_plant_frames(_joints_with_steps(90, [30, 60]))
    assert len(p) == 2 and abs(p[0] - 30) <= 3 and abs(p[1] - 60) <= 3
    assert len(_foot_plant_frames(np.zeros((90, 24, 3)))) == 0


def test_score_is_read_where_the_draft_plays_the_landing():
    beats = [0, 20, 40, 60]
    assert abs(_step_lock_score([4], 81, 81, beats) - 1.0) < 1e-6       # 0.2 beat after beat 0
    assert abs(_step_lock_score([14], 81, 81, beats) + 1.0) < 1e-6      # 0.7 beat: half a beat off
    # the same landing in a unit twice as long, stretched into the slot: frame 8 of 161 plays at frame 4
    assert abs(_step_lock_score([8], 161, 81, beats) - 1.0) < 1e-6
    assert _step_lock_score([], 81, 81, beats) == 0.0 and _step_lock_score([4], 81, 81, [10]) == 0.0


def _stub(keep):
    lib = Lib.__new__(Lib)
    lib.step_lock_keep = keep
    lib._source_bars = None
    plants = {0: np.array([4, 24, 44]), 1: np.array([14, 34]), 2: np.array([], dtype=int), 3: np.array([4])}
    lib._plants_of = lambda c: plants[c[0]]
    return lib


TIED = [(0, 0, 81, "g0"), (1, 0, 81, "g1"), (2, 0, 81, "g2"), (3, 0, 81, "g3")]


def test_keeps_the_units_whose_feet_land_on_the_beats():
    kept = _stub(0.5)._prefer_step_lock(TIED, 81, [0, 20, 40, 60], 1, None, [])
    assert kept == [(0, 0, 81, "g0"), (3, 0, 81, "g3")]


def test_inert_when_off_or_without_beats():
    assert _stub(0.0)._prefer_step_lock(TIED, 81, [0, 20, 40, 60], 1, None, []) == TIED
    assert _stub(0.5)._prefer_step_lock(TIED, 81, [10], 1, None, []) == TIED
