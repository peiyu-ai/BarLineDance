"""``--fix-foot-skate``: move the BODY, not the foot, and never the joint angles.

WHY.  The operator, 2026-09-11 and again 2026-09-12: "脚步时不时有滑动 ...
脚滑也交给后处理来修".  ``tools/score_arm_table.foot_skate`` -- mean horizontal
foot speed over the frames where that foot is at its own ground level -- reads
ground truth 0.295 m/s against the shipped arm's 0.420, and the two arms that
buy reach are worse still (0.471 with ``--draft-whole-units``, 0.556 with the
seam-aware ranking off).  **Ground truth is the target, not zero**: wild
reconstructions skate a little and a dancer whose feet never move is its own
defect, which is why the strength is a knob and not a switch.

THE DRIVING SIGNAL IS THE MODEL'S OWN CONTACT CHANNELS, not the metric's
"grounded" test.  Repairing a defect using the criterion that scores it makes
the score improve by construction, which CLAUDE.md 2.1 records as a way this
repository has already misled itself once (``boundary contrast`` rewarding the
very cut it was meant to forbid).

POSITIVE CONTROL is ``test_a_planted_foot_that_slides_is_brought_to_rest``.
The assertion that stops this becoming a cheat is
``test_joint_angles_are_bit_identical``: the fix may only translate.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_atomic import FOOT_JOINTS, fix_foot_skate

FRAMES = 90


def _clip(slide_per_frame=0.01, planted=True):
    """A body standing still whose feet nonetheless travel ``slide_per_frame``."""
    joints = np.zeros((FRAMES, 24, 3))
    drift = np.arange(FRAMES)[:, None] * np.asarray([slide_per_frame, 0.0])
    for j in range(24):
        joints[:, j, :2] = drift
        joints[:, j, 2] = 0.05 * j
    contacts = np.full((FRAMES, len(FOOT_JOINTS)), 1.0 if planted else 0.0)
    return {"full_pose": joints, "smpl_trans": joints[:, 0].copy(),
            "smpl_poses": np.arange(FRAMES * 72, dtype=float).reshape(FRAMES, 72),
            "contacts": contacts}


def _foot_speed(result):
    j = np.asarray(result["full_pose"])
    return float(np.linalg.norm(np.diff(j[:, FOOT_JOINTS[0], :2], axis=0), axis=1).mean())


def test_a_planted_foot_that_slides_is_brought_to_rest():
    """POSITIVE CONTROL: without it, a no-op implementation passes the rest."""
    clip = _clip(slide_per_frame=0.01)
    before = _foot_speed(clip)
    after = _foot_speed(fix_foot_skate(clip, strength=1.0, smooth=1))
    assert before == pytest.approx(0.01, abs=1e-9)
    assert after < before / 20.0, (before, after)


def test_joint_angles_are_bit_identical():
    """Only a translation is allowed.  A version that re-posed the legs would
    change the dance, which is the one thing this may not do."""
    clip = _clip()
    fixed = fix_foot_skate(clip, strength=1.0)
    assert np.array_equal(fixed["smpl_poses"], clip["smpl_poses"])


def test_the_body_keeps_its_shape():
    """Root-relative joints are untouched: the correction is one vector per
    frame applied to every joint alike."""
    clip = _clip()
    fixed = fix_foot_skate(clip, strength=1.0)
    def relative(r):
        j = np.asarray(r["full_pose"])
        return j - j[:, :1, :]
    assert np.allclose(relative(fixed), relative(clip), atol=1e-12)


def test_full_pose_and_smpl_trans_move_together():
    """The renderer's joint gate stops the render if the mesh and the joints
    disagree by a millimetre, so these two may never drift apart."""
    clip = _clip()
    fixed = fix_foot_skate(clip, strength=1.0)
    shift_joints = np.asarray(fixed["full_pose"])[:, 0, :2] - np.asarray(clip["full_pose"])[:, 0, :2]
    shift_trans = np.asarray(fixed["smpl_trans"])[:, :2] - np.asarray(clip["smpl_trans"])[:, :2]
    assert np.allclose(shift_joints, shift_trans, atol=1e-12)


def test_a_foot_in_the_air_is_not_corrected():
    """A swinging foot travels by design; correcting it would delete the step."""
    clip = _clip(slide_per_frame=0.01, planted=False)
    fixed = fix_foot_skate(clip, strength=1.0)
    assert np.allclose(fixed["full_pose"], clip["full_pose"], atol=1e-12)


def test_strength_zero_is_an_exact_no_op():
    clip = _clip()
    fixed = fix_foot_skate(clip, strength=0.0)
    assert np.allclose(fixed["full_pose"], clip["full_pose"], atol=1e-12)


def test_height_is_never_touched():
    """The floor anchor owns z, and a body that hovers must keep hovering."""
    clip = _clip()
    fixed = fix_foot_skate(clip, strength=1.0)
    assert np.array_equal(np.asarray(fixed["full_pose"])[:, :, 2],
                          np.asarray(clip["full_pose"])[:, :, 2])


def test_it_refuses_a_result_without_contacts():
    """Fail closed: silently skipping would be recorded in the manifest as
    though the fix had run."""
    clip = _clip()
    del clip["contacts"]
    with pytest.raises(ValueError, match="contact"):
        fix_foot_skate(clip, strength=1.0)


def test_the_correction_is_reported():
    clip = _clip(slide_per_frame=0.01)
    fixed = fix_foot_skate(clip, strength=1.0, smooth=1)
    report = fixed["foot_skate_fix"]
    assert report["planted_frame_share"] == pytest.approx(1.0)
    assert report["correction_max_m"] > 0.1
