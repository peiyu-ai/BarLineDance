"""The two calibration points, and the two instrument bugs this probe had.

A grounding score is only readable if chance is pinned and direction is fixed
in advance.  Both of this probe's early readings were wrong in ways that look
exactly like a finding:

* three tests came back reliably *backwards* because the normaliser -- the
  straight hip-to-ankle distance -- shortens when the knee bends, so it was a
  function of the crouch it was dividing out; and
* the null was drawn once per test, and a single shuffle of a 24-member group
  has a standard error near 0.06, so the instrument check fired on its own
  noise.

Each has a test here, because both produced a confident number before anyone
noticed the ruler was the thing that moved.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.probe_caption_grounding import auc, measures, run_tests  # noqa: E402


def _skeleton(frames, *, knee_angle=0.0, drift=0.0, yaw=0.0):
    """A stick figure whose thigh and shank are always 0.5, foot always on z=0.

    Bending the knee by ``knee_angle`` therefore lowers the pelvis -- a real
    crouch -- while leaving the leg's total length untouched.  This is the
    configuration that separates the two normalisers: the straight hip-to-ankle
    distance equals the pelvis height here, so dividing by it reports every
    crouch as exactly 1.0, while thigh + shank stays 1.0 and lets the height
    fall.
    """
    joints = np.zeros((frames, 24, 3), dtype=np.float64)
    t = np.linspace(0.0, 1.0, frames)
    hip_z = np.cos(knee_angle)
    for f in range(frames):
        angle = yaw * t[f]
        offset = np.array([drift * t[f], 0.0, 0.0])
        for hip, knee, ankle, side in ((1, 4, 7, 0.1), (2, 5, 8, -0.1)):
            base = np.array([side * np.cos(angle), side * np.sin(angle), hip_z])
            joints[f, hip] = base + offset
            joints[f, knee] = base + offset + np.array(
                [0.0, 0.5 * np.sin(knee_angle), -0.5 * np.cos(knee_angle)])
            joints[f, ankle] = base + offset + np.array([0.0, 0.0, -hip_z])
        joints[f, 0] = np.array([0.0, 0.0, hip_z]) + offset
    return joints


def test_a_bent_knee_does_not_inflate_the_normalised_height():
    """The bug that reported ``legs=crouched`` as reliably backwards.

    Same pelvis height, same limb lengths, only the knee folded.  A normaliser
    built on the straight hip-to-ankle distance shrinks here, so the crouched
    figure scores as *taller*; the sum of thigh and shank does not move.
    """
    straight = measures(_skeleton(12, knee_angle=0.0))
    bent = measures(_skeleton(12, knee_angle=0.9))

    # The crouch has to read LOWER.  Under the old normaliser both read 1.0,
    # because the divisor fell exactly as fast as the pelvis did.
    assert bent["height"] < straight["height"] - 0.2
    assert abs(straight["height"] - 1.0) < 0.02

    hips = _skeleton(12, knee_angle=0.9)[0, 1]
    ankle = _skeleton(12, knee_angle=0.9)[0, 7]
    old_normaliser = float(np.linalg.norm(hips - ankle))
    assert abs(hips[2] / old_normaliser - 1.0) < 0.02


def test_chance_is_one_half_whatever_the_class_balance():
    """93.7% of this corpus carries one travel value, so accuracy is unusable."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=4000)
    for share in (0.5, 0.9, 0.95):
        member = rng.random(4000) < share
        assert abs(auc(values, member) - 0.5) < 0.05


def test_a_constant_measure_reads_exactly_one_half():
    """Mid-ranks for ties.  Without them a column that never varies reads
    whatever order the sort happened to leave it in, which is not chance."""
    member = np.array([True] * 50 + [False] * 50)
    assert auc(np.zeros(100), member) == 0.5


def test_a_group_below_the_size_floor_is_not_scored():
    values = np.arange(100.0)
    member = np.array([True] * 5 + [False] * 95)
    assert np.isnan(auc(values, member))


def _fields(n_member, n_other, field, value, other):
    return ([{field: value} for _ in range(n_member)]
            + [{field: other} for _ in range(n_other)])


def test_the_declared_direction_decides_pass_from_fail():
    """A caption that is reliably backwards must score BELOW chance.

    Reporting the distance from 0.5 in either direction would pay a captioner
    for saying "in place" exactly when the dancer travels -- the shape of the
    2026-08-19 boundary-contrast defect.
    """
    rng = np.random.default_rng(3)
    n = 400
    # "in_place" segments given the LARGEST path length: exactly wrong.
    table = {"path_length": np.concatenate([np.full(n, 9.0), np.full(n, 1.0)]),
             "displacement": np.zeros(2 * n), "rotation": np.zeros(2 * n),
             "vertical_range": np.zeros(2 * n), "height": np.zeros(2 * n),
             "height_range": np.zeros(2 * n), "speed": np.zeros(2 * n),
             "ankle_speed": np.zeros(2 * n), "peak_acceleration": np.zeros(2 * n),
             "speed_variation": np.zeros(2 * n), "jerk": np.zeros(2 * n),
             "speed_trend": np.zeros(2 * n), "oscillation": np.zeros(2 * n),
             "stopped_fraction": np.zeros(2 * n)}
    fields = _fields(n, n, "travel", "in_place", "forward")

    rows = {(row["field"], tuple(row["values"]), row["measure"]): row
            for row in run_tests(fields, table, rng, rounds=20)}
    backwards = rows[("travel", ("in_place",), "path_length")]

    assert backwards["auc"] < 0.05
    assert backwards["z"] < 0

    # And the same labelling with the geometry the right way round passes.
    table["path_length"] = np.concatenate([np.full(n, 1.0), np.full(n, 9.0)])
    rows = {(row["field"], tuple(row["values"]), row["measure"]): row
            for row in run_tests(fields, table, rng, rounds=20)}
    assert rows[("travel", ("in_place",), "path_length")]["auc"] > 0.95


def test_the_null_is_averaged_not_drawn_once():
    """A one-draw null on a small group is noisier than the effect measured.

    Twelve members out of 800: a single shuffle lands anywhere near 0.5 +/- 0.15
    and the instrument check fires on nothing.  The averaged null has to sit on
    chance.
    """
    rng = np.random.default_rng(11)
    n = 800
    table = {key: rng.normal(size=n) for key in
             ("path_length", "displacement", "rotation", "vertical_range",
              "height", "height_range", "speed", "ankle_speed",
              "peak_acceleration", "speed_variation", "jerk",
              "speed_trend", "oscillation", "stopped_fraction")}
    fields = _fields(40, n - 40, "travel", "rotating", "in_place")

    rows = run_tests(fields, table, rng, rounds=300)
    nulls = np.array([row["null"] for row in rows])

    assert np.abs(nulls - 0.5).max() < 0.02
