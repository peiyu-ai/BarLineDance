"""--draft-continue-source: a bar line becomes the source dancer's own instead of a cut (DEFECTS §92).

Pinned here, on a hand-built library stub (no release on disk):
  * the next source bar is found by where the unit ENDS in its recording, even when that bar lives
    in another (overlapping) window;
  * the planned label must match unless --draft-continue-any-label; the duration band always applies;
  * the snap filters are applied WITHOUT the counting ``_reject_*`` helpers, which hand back the whole
    input when nothing passes (a single failing candidate would come back "accepted");
  * the continued bar is played so it starts one entry-rate step after the previous frame and ends
    exactly on its own last source frame (so the next bar line stays exact).
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic  # noqa: E402

Lib = infer_atomic.IndexedAtomicMotionLibrary


def _stub(any_label=False, yaw=None):
    lib = Lib.__new__(Lib)
    # window 0 starts at frame 100 of seq A, window 1 at frame 145 of the same sequence
    lib._window_origin = {0: ("A", 100), 1: ("A", 145)}
    lib._source_bars = {
        ("A", 150): [(1, 5, 55, 3)],       # the bar right after a unit ending at local 50 of window 0
        ("A", 200): [(1, 55, 100, 4)],
    }
    lib.retrieval_group_ids = ["gA", "gA"]
    lib.names = ["A0", "A1"]
    lib.continue_any_label = any_label
    lib._yaw_steps = yaw
    lib.max_yaw_step = 25.0
    lib._speed_spikes = None
    lib.max_speed_spike = None
    return lib


def test_next_bar_is_found_across_windows_with_the_planned_label():
    lib = _stub()
    assert lib._next_source_bar((0, 10, 50, "gA"), label=3, target_length=50) == (1, 5, 55, "gA")


def test_label_must_match_unless_any_label():
    assert _stub()._next_source_bar((0, 10, 50, "gA"), label=5, target_length=50) is None
    assert _stub(any_label=True)._next_source_bar((0, 10, 50, "gA"), label=5, target_length=50) \
        == (1, 5, 55, "gA")


def test_duration_band_always_applies():
    # band = max(2, 0.15 * 30) = 4.5 frames; the bar is 50 long
    assert _stub(any_label=True)._next_source_bar((0, 10, 50, "gA"), label=3, target_length=30) is None


def test_a_failing_snap_filter_rejects_the_single_candidate():
    series = {"A1": np.full(200, 40.0)}           # every frame turns faster than 25 deg
    lib = _stub(yaw=series)
    assert lib._next_source_bar((0, 10, 50, "gA"), label=3, target_length=50) is None
    # and the counting helper would NOT have rejected it -- the reason it is not used
    lib.yaw_slots = lib.yaw_rejected = lib.yaw_exhausted = 0
    assert lib._reject_yaw_snaps(((1, 5, 55, "gA"),)) == ((1, 5, 55, "gA"),)


def test_continued_bar_starts_one_step_on_and_ends_on_its_last_frame():
    lib = Lib.__new__(Lib)
    ramp = np.arange(40, dtype=np.float32)[:, None].repeat(151, 1)   # value == source frame index
    lib.motion = ramp[None]
    lib._local_floor = None
    lib._sample_floor = None
    lib.retrieval_log = [{}]
    lib._candidate_floor = lambda candidate: None
    values = lib._values_continued((0, 0, 40, "g"), target_length=46, entry_rate=1.3)
    frames = values[:, 10].double()
    assert abs(float(frames[0]) - 0.3) < 1e-5                        # previous frame was -1 -> -1 + 1.3
    assert abs(float(frames[-1]) - 39.0) < 1e-5                      # lands on its own last frame
    steps = torch.diff(frames)
    assert abs(float(steps[0]) - 1.3) < 0.05                         # leaves at the entry rate
    assert torch.all(steps > 0)
    assert lib.retrieval_log[-1]["entry_rate"] == 1.3
    assert abs(lib.retrieval_log[-1]["exit_rate"] - float(steps[-1])) < 1e-6


def test_settle_warp_dips_the_rate_after_the_line_and_still_ends_on_the_bar():
    lib = Lib.__new__(Lib)
    lib.motion = np.arange(60, dtype=np.float32)[:, None].repeat(151, 1)[None]
    lib._local_floor = None
    lib._sample_floor = None
    lib.retrieval_log = [{}]
    lib._candidate_floor = lambda candidate: None
    lib.continue_settle = 0.8
    frames = lib._values_continued((0, 0, 60, "g"), target_length=60, entry_rate=1.0)[:, 10].double()
    steps = torch.diff(frames)
    assert abs(float(frames[-1]) - 59.0) < 1e-5                 # the bar still ends on its own last frame
    slowest = int(torch.argmin(steps))
    assert abs(slowest - Lib.SETTLE_PEAK) <= 1                  # the dip sits ~3 frames after the line
    assert float(steps[slowest]) < 0.4 * float(steps[30:].mean())
    assert torch.all(steps > 0)                                 # slows, never reverses
