"""The non-dance criterion, and the controls that decide what it may claim.

The defect it exists for is the 2026-09-01 one: ``wild_v5:7424041427996298534:
clip000``, in the TEST split and in the ten-clip review set, is a close-up of a
man eating with chopsticks, and its "ground truth" is a standing figure with
both hands frozen near the face.  Nothing measured whether the tracked person
was dancing, because "dance" entered this corpus as a property of the ACCOUNT.

Three of these tests are controls rather than unit tests, and they are the point
of the file:

* ``test_a_detector_that_accepts_everything_fails_this_file`` -- CLAUDE.md
  section 2 opens with the gate that never triggers.  A criterion is worth
  nothing unless a criterion that flags nothing is measurably worse, so the
  degenerate detector is built here and asserted to lose.
* ``test_the_inverted_criterion_fails`` -- direction is not implied by "it can
  fail" (section 2.1 gate 2: the 2026-08-19 positive control was, in the paper's
  own semantics, the WORST possible answer and was used as an upper bound for a
  day).  Flipping the comparison must destroy the reading.
* ``test_metres_do_not_depend_on_the_clips_own_amplitude`` -- docs section 12.1:
  four earlier rulers divided by the clip's own median speed and all four read
  backwards.  Scaling a clip's motion up must move this statistic up.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.audit_non_dance_clips import (DEFAULT_THRESHOLD_M, confusion, decide,
                                         leg_excursion, pose_excursion,
                                         body_speed, read_labels)

LABELS = pathlib.Path(__file__).resolve().parents[1] / "runs/nondance_handlabels_v1.tsv"
CENSUS = pathlib.Path(__file__).resolve().parents[1] / "runs/nondance_census_v1.jsonl"


# --------------------------------------------------------------------------
# synthetic motions, in metres on the SMPL skeleton's own scale
# --------------------------------------------------------------------------

def _standing(frames=150, jitter=0.002, seed=0):
    """A body that never leaves one posture -- the defect, in synthetic form."""
    rng = np.random.default_rng(seed)
    rest = rng.normal(scale=0.35, size=(24, 3))
    rest[0] = 0.0
    return (rest[None, :, :] + rng.normal(scale=jitter, size=(frames, 24, 3)))[None]


def _dancing(frames=150, amplitude=0.25, period=45, seed=0):
    """A body that swings through distinct postures, root travelling as well."""
    rng = np.random.default_rng(seed)
    rest = rng.normal(scale=0.35, size=(24, 3))
    rest[0] = 0.0
    phase = 2 * np.pi * np.arange(frames)[:, None, None] / period
    swing = amplitude * np.sin(phase + rng.uniform(0, 2 * np.pi, size=(1, 24, 1)))
    motion = rest[None] + swing
    motion[:, 0, :] = 0.0
    # the root TRANSLATES every joint, the way forward kinematics does -- the
    # first version of this helper moved joint 0 alone, which made every other
    # joint carry MINUS the travel once the statistic de-rooted, and a 1.2 m
    # ramp then swamped the swing it was supposed to be measuring.  A synthetic
    # that does not have the shape of the real data tests nothing.
    travel = np.stack([np.linspace(0, 1.2, frames),
                       np.zeros(frames), np.zeros(frames)], axis=-1)
    motion = motion + travel[:, None, :]
    return motion[None]


def test_the_statistic_separates_a_held_pose_from_a_swinging_one():
    still = float(pose_excursion(_standing())[0])
    moving = float(pose_excursion(_dancing())[0])
    assert still < DEFAULT_THRESHOLD_M < moving
    assert moving > 10 * still


def test_the_root_does_not_leak_into_the_statistic():
    """A dancer who travels 1.2 m across a static pose must still read 'still'.

    The statistic is de-rooted on purpose: travel is the camera's problem and
    the dancer's, and a clip of somebody WALKING while holding one posture is
    exactly the spectator case this file exists to cut.  If the root leaked in,
    the walking spectator would pass.
    """
    still = _standing()
    walking = still.copy()
    walking[:, :, :, 0] += np.linspace(0, 1.2, still.shape[1])[None, :, None]
    assert pose_excursion(walking)[0] == pytest.approx(pose_excursion(still)[0], rel=1e-9)


def test_leg_excursion_is_reported_but_not_the_gate():
    """A torso-close-up hand dance has almost no leg motion and IS dance.

    Measured, 2026-09-01: ``wild_v5:7113140130243775756:clip000`` is a hand
    dance filmed from the chest up; its leg excursion is 0.0201 m -- BELOW the
    operator's confirmed non-dance clip's own 0.0185 m neighbourhood -- while
    its pose excursion is 0.0601 m, 1.6x the threshold.  Gating on legs would
    have cut it.  This test pins the shape of that: a synthetic arms-only dance
    must read low on legs and be KEPT by the gate.
    """
    arms_only = _standing(jitter=0.001)
    swing = 0.25 * np.sin(2 * np.pi * np.arange(150) / 45)
    for joint in (18, 19, 20, 21, 22, 23):
        arms_only[:, :, joint, 0] += swing
    assert float(leg_excursion(arms_only)[0]) < 0.01
    assert decide({"pose_excursion_m": float(pose_excursion(arms_only)[0])}) == "dance"


def test_metres_do_not_depend_on_the_clips_own_amplitude():
    """docs/DANCE_QUALITY_DEFECTS.md 12.1: no within-clip normalization.

    Four earlier rulers divided by the clip's own median speed and all four
    reported "generated >= ground truth" while the operator's eye said the
    opposite; the defect they were chasing WAS the per-clip amplitude, and the
    division removed it.  So: damping a clip by 0.4 must LOWER this reading, and
    a scale-invariant statistic would leave it alone.
    """
    loud = _dancing(amplitude=0.25)
    quiet = _dancing(amplitude=0.10)
    assert pose_excursion(quiet)[0] < 0.5 * pose_excursion(loud)[0]
    assert body_speed(quiet)[0] < 0.5 * body_speed(loud)[0]


# --------------------------------------------------------------------------
# the controls
# --------------------------------------------------------------------------

def _hand_labelled_rows():
    """The hand-labelled clips joined to the census the tool actually wrote."""
    if not (LABELS.is_file() and CENSUS.is_file()):
        pytest.skip("hand labels or census not present")
    labels = read_labels(LABELS)
    rows = {}
    for line in CENSUS.read_text().splitlines():
        row = json.loads(line)
        if row["sequence"] in labels:
            rows[row["sequence"]] = row
    return rows, labels


def test_the_criterion_cuts_the_operators_clip_and_keeps_the_random_dance_clips():
    """Both directions, on real data, from the census on disk.

    Negative: the clip the operator found by eye.  Positive: the 40 clips drawn
    at random from the release (tranche A of the label file), every one of them
    hand-confirmed dance.  A criterion that only has one of these is the
    2026-08-19 failure.
    """
    rows, labels = _hand_labelled_rows()
    # the verdict is recomputed with ``decide`` rather than read out of the
    # census's own ``decision`` column, so that a change to the criterion is
    # caught here even when the census on disk is stale.  The stored column is
    # then checked against it, which is what catches a stale census.
    operator = rows["wild_v5:7424041427996298534:clip000"]
    assert decide(operator) == "not_dance"
    assert operator["decision"] == "not_dance", "census on disk is stale"
    random_draw = [key for key, hand in labels.items() if hand["tranche"] == "A"]
    assert len(random_draw) >= 40
    for key in random_draw:
        assert labels[key]["label"] == "dance"
        assert decide(rows[key]) == "dance", key
        assert rows[key]["decision"] == "dance", key


def test_a_detector_that_accepts_everything_fails_this_file():
    """CLAUDE.md section 2: a gate that never triggers is worse than no gate.

    The degenerate detector is built here rather than described, so that the
    number the real criterion has to beat is computed from the same labels by
    the same code path.
    """
    rows, labels = _hand_labelled_rows()
    real, _, _, _ = confusion(rows, labels, DEFAULT_THRESHOLD_M)
    # threshold 0 => nothing is ever below it => nothing is ever cut
    accept_everything, _, _, _ = confusion(rows, labels, 0.0)
    assert accept_everything["tp"] == 0
    assert accept_everything["fp"] == 0
    assert real["tp"] >= 20
    assert real["tp"] > accept_everything["tp"]


def test_the_inverted_criterion_fails():
    """Flip the comparison and the reading must collapse.

    'It can fail' does not imply 'it points the right way' -- section 2.1 gate 2.
    Inverted, the rule cuts the clips with the MOST postural change, which is
    every hand-confirmed dance and none of the confirmed non-dance clips.
    """
    rows, labels = _hand_labelled_rows()
    inverted_tp = inverted_fp = 0
    for key, hand in labels.items():
        if key not in rows or hand["label"] == "unsure":
            continue
        cut = rows[key]["pose_excursion_m"] >= DEFAULT_THRESHOLD_M
        if hand["label"] == "not_dance" and cut:
            inverted_tp += 1
        elif hand["label"] == "dance" and cut:
            inverted_fp += 1
    real, _, _, _ = confusion(rows, labels, DEFAULT_THRESHOLD_M)
    assert inverted_fp >= 60          # it cuts essentially every real dance
    assert real["fp"] == 0            # the right way round cuts none of them
    assert inverted_tp < real["tp"]


def test_the_threshold_does_not_sit_on_a_labelled_clip():
    """A threshold a rounding error can flip is not a threshold.

    The first version sat at 0.035 and a 0.3% difference between two
    implementations of the same forward kinematics moved
    ``wild_v5:7628609331936482789:clip000`` (a choreographer talking to camera)
    across it.  The margin asserted here, 1 mm, is ~8x the 0.33% agreement
    measured between the release-array FK path and the raw-joint pickles at this
    magnitude, so a threshold that clears it cannot be moved by the instrument.
    """
    rows, _ = _hand_labelled_rows()
    values = np.array([row["pose_excursion_m"] for row in rows.values()])
    gap = np.min(np.abs(values - DEFAULT_THRESHOLD_M))
    assert gap > 0.001, "threshold is {} m from a labelled clip".format(gap)


def test_the_census_reports_what_it_could_not_separate():
    """The band 0.038..0.08 m is mixed and the summary has to say so.

    Silence about the overlap is how a criterion gets read as stronger than it
    is; CLAUDE.md section 4.4 -- details may be dropped, the reasoning may not.
    """
    summary = json.loads((CENSUS.parent / "nondance_census_v1_summary.json").read_text())
    assert summary["hand_label_confusion"]["fp"] == 0
    assert summary["hand_label_confusion"]["fn"] > 0, (
        "a criterion with no misses on 30 confirmed negatives would mean the "
        "negatives were chosen by the criterion itself")
    # the band is not prose in the summary, it is a measurement: the lowest and
    # highest reading among the confirmed non-dance clips the gate KEEPS.
    low, high = summary["unseparated_band_m"]
    assert low > summary["threshold_m"]
    assert high > low
    assert len(summary["hand_label_misses"]) == summary["hand_label_confusion"]["fn"]
