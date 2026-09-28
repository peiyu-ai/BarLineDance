"""--draft-seam-lead: the incoming unit brings its own approach into the downbeat (F series, 2026-09-24).

Pinned on a stub library (no release on disk):
  * the lead frames are the unit's OWN recording before its start, at the unit's playback rate;
  * after the lead, the unit is what ``_values_at`` plays (so the downbeat frame is the source's first frame);
  * a unit near its window's start takes the lead it has, and none when fewer than two frames exist;
  * the flag refuses to combine with --seam-transition after.
"""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic  # noqa: E402

Lib = infer_atomic.IndexedAtomicMotionLibrary


def _stub():
    lib = Lib.__new__(Lib)
    frames = np.arange(150, dtype=np.float32)
    lib.motion = np.stack([np.stack([frames + 1000 * s + d for d in range(8)], axis=1) for s in range(2)])
    lib._candidate_floor = lambda candidate: None
    return lib


def test_lead_is_the_units_own_approach_at_the_same_rate():
    lib = _stub()
    unit = (1, 40, 100, "g1")                       # 60 source frames
    values, lead = lib._values_with_lead(unit, 60, 8)
    assert lead == 8 and values.shape == (68, 8)
    assert torch.allclose(values[:, 0], torch.arange(1032, 1100, dtype=torch.float32))
    # the bar line (index = lead) is the unit's first frame: its dancer's downbeat
    assert abs(float(values[lead, 0]) - 1040.0) < 1e-4
    assert torch.allclose(values[lead:], lib._values_at(unit, 60), atol=1e-4)


def test_stretched_unit_keeps_its_rate_through_the_lead():
    lib = _stub()
    unit = (0, 50, 110, "g0")                       # 60 source frames played over 30 slot frames: 2 per frame
    values, lead = lib._values_with_lead(unit, 30, 6)
    assert lead == 6 and values.shape == (36, 8)
    assert abs(float(values[lead, 0]) - 50.0) < 1e-3
    assert abs(float(values[0, 0]) - (50.0 - 6 * 59 / 29)) < 1e-3   # 6 slot frames back at the unit's own spacing


def test_lead_is_limited_by_the_window_start():
    lib = _stub()
    values, lead = lib._values_with_lead((1, 3, 63, "g1"), 60, 8)
    assert lead == 3 and abs(float(values[lead, 0]) - 1003.0) < 1e-4
    assert lib._values_with_lead((1, 1, 61, "g1"), 60, 8) == (None, 0)
    assert lib._values_with_lead((1, 40, 100, "g1"), 60, 0) == (None, 0)


def test_refuses_seam_transition_after():
    source = (pathlib.Path(__file__).resolve().parents[1] / "infer_atomic.py").read_text()
    assert "--draft-seam-lead moves the cut BEFORE the bar line" in source


def test_a_unit_at_its_window_head_takes_its_lead_from_the_window_before():
    lib = _stub()
    frames = np.arange(300, dtype=np.float32)       # one recording, windows at stride 15: window 1 = frames 15..165
    lib.motion = np.stack([np.stack([frames[s:s + 150] + d for d in range(8)], axis=1) for s in (0, 15)])
    lib._window_before_table = {1: (0, 15)}
    values, lead = lib._values_with_lead((1, 0, 60, "g"), 60, 8)
    assert lead == 8
    assert torch.allclose(values[:, 0], torch.arange(7, 75, dtype=torch.float32), atol=1e-4)


def test_source_span_is_in_the_recordings_own_frames():
    lib = _stub()
    lib._window_origin = {1: ("wild_v5:u:clip000", 30)}
    assert lib._source_span((1, 40, 100, "g")) == ("wild_v5:u:clip000", 70, 130)
    assert lib._source_span((0, 40, 100, "g")) is None
    assert lib._source_span(None) is None
