"""--draft-energy-follow (K series): the dance calmer where the song is quiet, busier where it is loud.

Pinned without a release on disk:
  * the SPEC parses with defaults and refuses unknown keys / features;
  * the intensity plan ranks each bar's (smoothed) loudness WITHIN the song and maps it onto [lo, hi]; frames outside
    the bar grid take the nearest bar;
  * the filter keeps the candidates whose played speed sits nearest the bar's target percentile, and is inert when off.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic  # noqa: E402
from infer_atomic import _energy_targets, _parse_energy_follow  # noqa: E402

Lib = infer_atomic.IndexedAtomicMotionLibrary


def test_spec_parses_with_defaults_and_refuses_unknown_keys():
    spec = _parse_energy_follow("keep=0.3,smooth=0")
    assert spec["keep"] == 0.3 and spec["smooth"] == 0 and spec["lo"] == 0.15 and spec["feature"] == "loudness"
    assert _parse_energy_follow("") is None
    with pytest.raises(ValueError):
        _parse_energy_follow("keeep=0.3")
    with pytest.raises(ValueError):
        _parse_energy_follow("feature=tempo")


def test_plan_ranks_bars_within_the_song():
    track = np.zeros((100, 35)); track[0:20, 1] = 1.0; track[20:40, 1] = 5.0; track[40:60, 1] = 3.0; track[60:80, 1] = 2.0
    tau = _energy_targets(track, [10, 30, 50, 70, 90], {"smooth": 0, "lo": 0.0, "hi": 1.0, "feature": "loudness"})
    # bar means: [10,30) 3.0, [30,50) 4.0, [50,70) 2.5, [70,90) 1.0 -> ranks 2,3,1,0 of 4 -> (r + .5) / 4
    assert tau[15] == pytest.approx(0.625) and tau[35] == pytest.approx(0.875)
    assert tau[55] == pytest.approx(0.375) and tau[75] == pytest.approx(0.125)
    assert tau[0] == tau[15] and tau[99] == tau[75]          # outside the grid: nearest bar


def _stub(spec, tau):
    lib = Lib.__new__(Lib)
    lib.energy_follow = spec
    lib._energy_tau = tau
    lib._source_bars = None
    lib._speed_cdf = np.linspace(0.0, 1.0, 101)              # speed s sits at percentile ~s
    speeds = {0: 0.1, 1: 0.5, 2: 0.9, 3: 0.2}
    lib._unit_speed = lambda c, length: speeds[c[0]]
    lib._energy_cdf = lambda: lib._speed_cdf
    return lib


TIED = [(0, 0, 40, "a"), (1, 0, 40, "b"), (2, 0, 40, "c"), (3, 0, 40, "d")]


def test_quiet_bar_keeps_the_calm_units_and_loud_bar_the_busy_ones():
    spec = {"keep": 0.5}
    quiet, loud = np.full(200, 0.15), np.full(200, 0.85)
    assert _stub(spec, quiet)._prefer_energy_follow(TIED, 40, 0, 1, []) == [(0, 0, 40, "a"), (3, 0, 40, "d")]
    assert _stub(spec, loud)._prefer_energy_follow(TIED, 40, 0, 1, []) == [(1, 0, 40, "b"), (2, 0, 40, "c")]


def test_inert_when_off():
    assert _stub(None, np.full(200, 0.15))._prefer_energy_follow(TIED, 40, 0, 1, []) == TIED
