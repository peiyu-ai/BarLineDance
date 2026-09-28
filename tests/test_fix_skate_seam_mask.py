"""--fix-skate-seam-mask must actually stop the seam ramp reaching the root.

THE DEFECT.  ``_blend_draft_seams`` cross-fades the POSE channels across +-N
frames at every bar seam while the contact channels stay on through the ramp.
``fix_foot_skate``'s rule -- "a foot down on both frames that moved is sliding"
-- reads that pose morph as slide and translates the whole body to cancel it,
and the pelvis carries every millimetre.  Measured on the fixed ten by
recovering the strength-0 pelvis path exactly from two arms differing only in
strength (0.5 and 1.0, pose channels bit-identical): 4.94 m at strength 0
against ground truth's 4.85 (+0.09, 5/10 -- flat) and 6.36 at strength 1.0
(+1.42 against strength 0, 10/10, P=0.002).

THE TEST IS BEHAVIOURAL, not a wiring grep.  DEFECTS 77 is this repository
losing three days to a flag that was recorded in the manifest, named in the
tests and dead in the code; the only check that could not be fooled was the
output bytes.  So the first test below builds a clip whose ONLY foot motion is
a seam ramp and asserts the correction goes to zero.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from infer_atomic import fix_foot_skate  # noqa: E402

SOURCE = pathlib.Path("infer_atomic.py").read_text()


def clip_with_a_seam_ramp(frames=90, seam=45, half=8, slide=0.01):
    """Feet planted throughout; the only horizontal motion is a ramp at ``seam``."""
    joints = np.zeros((frames, 24, 3))
    drift = np.zeros(frames)
    ramp = np.arange(-half, half + 1)
    drift[seam - half:seam + half + 1] = np.cumsum(np.ones(len(ramp))) * slide
    drift[seam + half + 1:] = drift[seam + half]
    for joint in (7, 8, 10, 11):
        joints[:, joint, 0] = drift
    joints[:, 0, 0] = drift                      # the pelvis rides with them
    return {"full_pose": joints, "smpl_trans": joints[:, 0, :].copy(),
            "contacts": np.ones((frames, 4))}


def test_the_seam_ramp_is_charged_to_the_root_without_the_mask():
    out = fix_foot_skate(clip_with_a_seam_ramp(), 1.0)
    assert out["foot_skate_fix"]["correction_max_m"] > 0.02, (
        "the fixture is meant to reproduce the defect; if this is small the "
        "fixture stopped exercising it and the next assertion proves nothing")
    assert out["foot_skate_fix"]["seam_frames_masked"] == 0


def test_the_mask_removes_it():
    seam, half = 45, 8
    frames = np.arange(seam - half, seam + half + 1)
    loose = fix_foot_skate(clip_with_a_seam_ramp(), 1.0)
    tight = fix_foot_skate(clip_with_a_seam_ramp(), 1.0, seam_frames=frames)
    assert tight["foot_skate_fix"]["seam_frames_masked"] == len(frames)
    assert tight["foot_skate_fix"]["correction_max_m"] < 0.1 * loose["foot_skate_fix"]["correction_max_m"], (
        "masking the seam frames must stop the ramp reaching the root: "
        "{:.4f} against {:.4f} m".format(tight["foot_skate_fix"]["correction_max_m"],
                                         loose["foot_skate_fix"]["correction_max_m"]))


def test_real_slide_outside_the_seam_is_still_corrected():
    """The repair must not be traded away: the operator asked for skate to be
    fixed in post, and ground truth skates 0.295 m/s rather than zero."""
    clip = clip_with_a_seam_ramp(seam=20)
    clip["full_pose"][60:, 7, 0] += np.arange(len(clip["full_pose"]) - 60) * 0.01
    clip["full_pose"][60:, 10, 0] += np.arange(len(clip["full_pose"]) - 60) * 0.01
    masked = fix_foot_skate(clip, 1.0, seam_frames=np.arange(12, 29))
    assert masked["foot_skate_fix"]["correction_max_m"] > 0.02, (
        "slide away from any seam must still be corrected")


def test_zero_frames_reproduces_the_old_behaviour_exactly():
    a = fix_foot_skate(clip_with_a_seam_ramp(), 1.0)
    b = fix_foot_skate(clip_with_a_seam_ramp(), 1.0, seam_frames=[])
    assert np.array_equal(np.asarray(a["smpl_trans"]), np.asarray(b["smpl_trans"]))


def test_the_flag_is_wired_and_recorded():
    assert '"--fix-skate-seam-mask"' in SOURCE
    assert "fix_skate_seam_blend=options.fix_skate_seam_blend" in SOURCE
    assert '"fix_skate_seam_blend": fix_skate_seam_blend' in SOURCE, (
        "an unrecorded switch is one nobody can tell ran; this repository has "
        "shipped eight of them")
    assert "seam_frames=seam_frames" in SOURCE
