"""The stillness criterion must REFUSE what it cannot measure, identically
everywhere.

WHAT IS BEING TESTED.  "Sustained stillness" is the deciding column for "does
the generated dancer ever land": the share of frames (or of 150-frame windows
containing a frame) whose smoothed root-relative joint speed is below 0.25 x
the clip's OWN median smoothed speed.  Being relative to the clip's own speed is
what makes it scale free -- and it is also where it breaks, because at median 0
the threshold is 0 and ``speed < 0`` is unsatisfiable: a body that never moves
reads "no stillness at all".

Until 2026-09-04 three tools implemented this and disagreed about that input:
``measure_motion_dynamics`` dropped the clip silently (``clips: 16`` where 20
were asked for, no names), ``exp_guidance_stillness`` printed 0.0, and
``exp_draftonly_stillness_windows`` printed False.  These tests pin the
refusal, pin a POSITIVE CONTROL that must NOT be refused, and pin that the
tools now answer the same on the same input.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import exp_draftonly_stillness_windows as draftonly  # noqa: E402
from tools import exp_guidance_stillness as guidance  # noqa: E402
from tools import measure_motion_dynamics as dynamics_tool  # noqa: E402
from tools import stillness_criterion as sc  # noqa: E402

FPS = sc.FPS


def frozen_body(frames=600, joints=24):
    """The degenerate input: every frame identical.

    This is the shape ``infer_atomic._source_safe_draft`` returns when a clip
    has no retrieval group -- an all-zero draft -- which is what two of the
    twenty T-line eval clips got (DANCE_QUALITY_DEFECTS.md 23.9)."""
    return np.zeros((frames, joints, 3))


def settling_body(frames=600, joints=24, move_s=3.0, hold_s=1.0, seed=0,
                  noise=0.0005):
    """THE POSITIVE CONTROL: travel 3 s, hold 1 s, repeat.

    Known-good answer with a known DIRECTION (CLAUDE.md 2.1 rule 2): a body
    with real landings must read a sizeable stillness share, close to the 25%
    duty cycle of its holds, and must survive the 9-frame low-pass that exists
    to tell a landing from a zero crossing."""
    return guidance.synthetic_move_then_hold(frames=frames, joints=joints,
                                             seed=seed, move_s=move_s,
                                             hold_s=hold_s, noise=noise)


# ------------------------------------------------------------- the refusal

def test_frozen_body_is_refused_not_read_as_zero():
    speed = sc.joint_speed(sc.low_pass(frozen_body()))
    assert float(np.median(speed)) == 0.0, "fixture must actually be frozen"
    with pytest.raises(sc.DegenerateMotion) as refusal:
        sc.held_frames(speed)
    assert "median smoothed speed" in str(refusal.value)
    with pytest.raises(sc.DegenerateMotion):
        sc.dynamic_range(speed)
    assert sc.is_measurable(speed) is False


def test_half_frozen_body_is_refused_too_and_iqr_would_have_missed_it():
    """The degeneracy is the MEDIAN vanishing, not the body being uniformly
    still.  ``wild_v5:7148005618245242151:clip000``'s draft has 348 of 664
    smoothed frames exactly zero: its median is 0 (refused) while its speed IQR
    is 0.2989 m/s, larger than fourteen of the twenty ground-truth clips'.  An
    IQR floor, one of the two candidate tests considered, would have passed it."""
    rng = np.random.default_rng(0)
    joints = np.zeros((600, 24, 3))
    velocity = rng.normal(size=(24, 3)) * 0.01
    velocity[0] = 0.0
    # 40% of frames move fast, 60% are frozen: median exactly 0, wide IQR.
    position = np.zeros((24, 3))
    for f in range(240):
        position = position + velocity
        joints[f] = position
    joints[240:] = position
    speed = sc.joint_speed(sc.low_pass(joints))
    assert float(np.median(speed)) == 0.0
    q1, q3 = np.percentile(speed, [25, 75])
    assert q3 - q1 > 0.1, "the IQR is wide, which is why an IQR floor fails here"
    with pytest.raises(sc.DegenerateMotion):
        sc.held_frames(speed)


def test_p90_over_p10_is_non_monotone_and_so_was_rejected_as_the_test():
    """Rejected candidate, with its numbers (CLAUDE.md 2.1 rule 4).  A "p90/p10
    below a bound" test catches the fully frozen body (ratio 0 under the old
    clamp) and MISSES the half-frozen one (ratio ~1e11), though both have a
    vanished median.  Recorded so the rejected ruler cannot quietly come back."""
    frozen = sc.joint_speed(sc.low_pass(frozen_body()))
    rng = np.random.default_rng(1)
    half = np.zeros((600, 24, 3))
    velocity = rng.normal(size=(24, 3)) * 0.01
    velocity[0] = 0.0
    position = np.zeros((24, 3))
    for f in range(240):
        position = position + velocity
        half[f] = position
    half[240:] = position
    half_speed = sc.joint_speed(sc.low_pass(half))

    old_ratio = lambda s: float(np.percentile(s, 90) / max(float(np.percentile(s, 10)), 1e-9))
    assert old_ratio(frozen) == 0.0            # would be caught by "below a bound"
    assert old_ratio(half_speed) > 1e7         # would NOT be caught
    # The shipped test catches both, because it asks the question the threshold
    # is actually built from.
    for speed in (frozen, half_speed):
        assert sc.is_measurable(speed) is False


def test_the_floor_admits_everything_a_real_body_does():
    """The floor is float32 storage noise (1e-4 m/s = 3 microns per frame at
    30 fps), not a fit.  A body drifting a thousandth of a metre per second --
    below anything measured in the corpus (smallest real reading 0.001482 m/s)
    -- is still MEASURED, not refused, so the floor cannot be quietly widened
    into a discriminator."""
    joints = np.zeros((600, 24, 3))
    joints[:, 1:, 0] = np.arange(600)[:, None] * (0.001 / FPS)
    speed = sc.joint_speed(joints)
    assert 5e-4 < float(np.median(speed)) < 2e-3
    assert sc.is_measurable(speed) is True


# ------------------------------------------------- the positive control

def test_positive_control_is_measured_and_reads_its_duty_cycle():
    speed = sc.joint_speed(sc.low_pass(settling_body()))
    assert sc.is_measurable(speed) is True
    share = float(sc.held_frames(speed).mean())
    assert 0.15 <= share <= 0.35, share  # the hold is 25% of the period


def test_positive_control_is_not_dropped_by_any_of_the_three_tools():
    joints = settling_body()

    row = dynamics_tool.dynamics(sc.joint_speed(sc.low_pass(joints)))
    assert 0.15 <= row["hold_share"] <= 0.35

    g = guidance.score_clip(joints, np.random.default_rng([7, 1]), 40)
    assert g["clip_measurable"] is True
    assert g["windows_dropped"] == 0
    assert g["window_share"] >= 0.95  # 150-frame windows all overlap a hold

    d = draftonly.score_clip(joints, windows_per_clip=40, seed=7)
    assert d["clip_measurable"] is True
    assert d["windows_dropped"] == 0
    assert d["clip_has_sustained_hold"] is True


# ---------------------------------------------- the two tools must agree

def test_the_three_tools_agree_on_the_frozen_body():
    """The conflict this file exists to close: same input, same verdict.

    BEFORE: measure_motion_dynamics returned None and its caller skipped the
    clip without a name; exp_guidance_stillness returned window_share 0.0 /
    clip_sustained_hold_share 0.0; exp_draftonly_stillness_windows returned
    clip_has_sustained_hold False.  Three different answers, none of them a
    refusal."""
    joints = frozen_body()

    with pytest.raises(sc.DegenerateMotion):
        dynamics_tool.dynamics(sc.joint_speed(sc.low_pass(joints)))

    g = guidance.score_clip(joints, np.random.default_rng([7, 1]), 40)
    d = draftonly.score_clip(joints, windows_per_clip=40, seed=7)

    assert g["clip_measurable"] is False
    assert d["clip_measurable"] is False
    assert g["clip_measurable"] == d["clip_measurable"]
    # No number is emitted anywhere -- None, never 0.0.
    assert g["window_share"] is None and g["clip_sustained_hold_share"] is None
    assert d["clip_has_sustained_hold"] is None and d["p90_over_p10_median"] is None
    # Every drawn window is accounted for as dropped, not as "no stillness".
    assert g["windows_dropped"] == g["windows"] == 40
    assert d["windows_dropped"] == d["windows_drawn"] == 40
    assert g["windows_with_stillness"] == d["windows_with_hold"] == 0


def test_the_two_window_tools_agree_window_for_window_on_the_same_starts():
    """Agreement has to hold in the measurable direction too, otherwise the
    test above would pass for a tool that refuses everything.

    Compared on IDENTICAL window starts, because the two tools seed their
    samplers differently (``default_rng([seed, clip_index])`` vs
    ``default_rng(seed)``) and so draw different windows from the same clip --
    a real difference in sampling that is not a difference in the criterion,
    and one that would blur this test if it went through the samplers."""
    joints = np.concatenate([settling_body(seed=4, frames=300),
                             guidance.synthetic_constant_velocity(frames=300, seed=5)])
    starts = np.array([0, 40, 120, 200, 300, 380, 440])

    mine, _, dropped, _ = guidance.score_windows(joints, starts)
    theirs = [draftonly.window_contains_hold(joints[s:s + 150])[0] for s in starts]
    assert dropped == 0
    assert list(mine) == theirs
    assert any(theirs) and not all(theirs), (
        "the fixture must contain BOTH kinds of window, or agreeing means "
        "nothing")


def test_the_two_window_tools_agree_on_the_same_clip_through_their_own_samplers():
    joints = settling_body(seed=4)
    g = guidance.score_clip(joints, np.random.default_rng([11, 1]), 60)
    d = draftonly.score_clip(joints, windows_per_clip=60, seed=11)
    assert g["clip_measurable"] == d["clip_measurable"] is True
    assert g["windows_dropped"] == d["windows_dropped"] == 0
    assert g["window_share"] == pytest.approx(
        d["windows_with_hold"] / d["windows_measured"], abs=0.1)


# ------------------------------------------------------- the drop accounting

def test_every_arm_report_names_the_clips_it_refused(tmp_path):
    """A JSON that hides its drops is the failure this task exists to fix, so
    the header is asserted rather than trusted: counts AND names."""
    import pickle

    directory = tmp_path / "arm"
    directory.mkdir()
    for name, joints in (("good_a", settling_body(seed=1)),
                         ("good_b", settling_body(seed=2)),
                         ("frozen_one", frozen_body())):
        with open(directory / (name + ".pkl"), "wb") as handle:
            pickle.dump({"full_pose": joints}, handle)
    clips = ["good_a", "good_b", "frozen_one", "never_written"]

    arm = guidance.score_arm(str(directory), clips, seed=5, draws=20)
    assert arm["clips_measured"] == 2
    assert arm["clips_dropped"] == 1
    assert list(arm["dropped_clips"]) == ["frozen_one"]
    assert arm["missing"] == ["never_written"]
    # The pooled denominator excludes the frozen clip's windows entirely.
    assert arm["windows_drawn"] == 60 and arm["windows_measured"] == 40
    assert arm["windows_dropped"] == 20

    other = draftonly.score_arm(str(directory), clips, windows_per_clip=20,
                                seed=5)
    assert other["clips_measured"] == 2
    assert list(other["dropped_clips"]) == ["frozen_one"]
    assert other["absent_clips"] == ["never_written"]
    assert other["windows_dropped"] == 20

    row = dynamics_tool.score(str(directory), clips)
    assert row["clips_measured"] == 2
    assert list(row["dropped_clips"]) == ["frozen_one"]
    assert row["absent_clips"] == ["never_written"]


def test_an_arm_that_is_entirely_frozen_exits_rather_than_reporting(tmp_path):
    import pickle

    directory = tmp_path / "arm"
    directory.mkdir()
    for name in ("a", "b"):
        with open(directory / (name + ".pkl"), "wb") as handle:
            pickle.dump({"full_pose": frozen_body()}, handle)

    for scorer in (lambda: guidance.score_arm(str(directory), ["a", "b"], seed=1, draws=5),
                   lambda: draftonly.score_arm(str(directory), ["a", "b"],
                                               windows_per_clip=5, seed=1),
                   lambda: dynamics_tool.score(str(directory), ["a", "b"])):
        with pytest.raises(SystemExit) as failure:
            scorer()
        assert "a" in str(failure.value) and "b" in str(failure.value)


def test_the_ratio_to_ground_truth_uses_a_matched_clip_set(tmp_path):
    """``hold_share_vs_truth`` divides an arm's median over the clips IT could
    measure by ground truth's median over the clips GROUND TRUTH could measure.
    Those are different sets whenever the arm drops anything, and the arm that
    drops is always the one whose clips are unusual -- so the ratio is biased
    by exactly the clips that were removed.  ``hold_share_vs_truth_matched``
    re-scores ground truth on the arm's own measured clips.

    On runs/txy_t_draft_ep12_index the published 2.346 is the mismatched form;
    matched it is 3.17.  Here the fixture makes the two differ by construction:
    the clip the arm cannot measure is one where ground truth holds a lot."""
    import pickle

    def write(directory, name, joints):
        directory.mkdir(exist_ok=True)
        with open(directory / (name + ".pkl"), "wb") as handle:
            pickle.dump({"full_pose": joints}, handle)

    arm, truth = tmp_path / "arm", tmp_path / "truth"
    # Ground truth's three clips have deliberately different duty cycles, and
    # the one the arm cannot measure is ground truth's HEAVIEST holder -- so
    # leaving it in the denominator changes ground truth's median.
    write(truth, "hold_a", settling_body(seed=1, move_s=7.0, hold_s=0.5))
    write(truth, "hold_b", settling_body(seed=2, move_s=1.5, hold_s=1.0))
    write(truth, "frozen", settling_body(seed=3, move_s=1.0, hold_s=3.0))
    write(arm, "hold_a", settling_body(seed=4))
    write(arm, "hold_b", settling_body(seed=5))
    write(arm, "frozen", frozen_body())

    clips = ["hold_a", "hold_b", "frozen"]
    report = dynamics_tool.run([("arm", str(arm))], clips, str(truth))
    row = report["arms"]["arm"]
    assert row["clips_dropped"] == 1
    assert list(row["dropped_clips"]) == ["frozen"]
    assert report["measurability"]["arm"]["clips_dropped"] == 1
    # Ground truth measured all three, so the two ratios cannot be equal.
    assert report["ground_truth"]["clips_measured"] == 3
    assert row["ground_truth_on_matched_clips"]["clips_measured"] == 2
    mismatched = report["ground_truth"]["hold_share"]
    matched = row["ground_truth_on_matched_clips"]["hold_share"]
    assert matched != mismatched, (matched, mismatched)
    assert row["hold_share_vs_truth"] != row["hold_share_vs_truth_matched"]
