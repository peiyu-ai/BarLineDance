"""``--seam-transition after``: finish the arriving move, THEN change units.

WHY.  The operator, 2026-09-22, on the 2D renders of fix7: "动作不够舒展到位，很多
pose 都是做到一半就收了" -- and asked whether completion's seam window was
covering normal motion.  Measured on the ten vis clips (wrist speed / clip
median, by frame offset from the bar seam, which is a downbeat):

                    -4    -3    -2    -1     0    +1
    ground truth   1.06  1.00  1.00  1.06  1.13  1.19   decelerates INTO the beat
    fix7 draft     0.86  0.90  1.53  2.41  1.48  0.82   the arriving unit stops
    fix7 final     1.28  1.30  1.20  1.04  1.03  1.18   fastest where it should stop

The centred window (+-8 around the seam, and the draft's centred cross-fade)
regenerates the arriving unit's last frames, so the pose on the downbeat is a
compromise between two units' poses.  "after" keeps everything up to the bar
line and makes the transition in the frames after it.

WHAT THESE TESTS PIN.  The keep mask holds the arrival (frames <= seam-3 kept
exactly) and still frees the seam itself; the draft fade touches nothing before
the seam and starts from the arriving unit COASTING to a stop, not a frozen
frame (a freeze is a velocity step at the bar line); a stagger, which "after"
cannot perform, is refused; and the switch is threaded, not merely accepted.

WHAT WAS TRIED AND DROPPED.  Letting the limbs follow the arriving source
dancer's own frames past the cut (the window usually has them) measured seam
jerk 1.54-1.78 against coasting's 0.69, and on 7618203431723357818 seam 320 it
showed the start of that dancer's NEXT move -- a lunge -- and abandoned it.
"""
import ast
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infer_atomic
from infer_atomic import SEAM_AFTER_LEAD, _blend_draft_seams, infer_completion

DIM = 151
SEAM = 30
WIDTH = 8


class Recorder:
    """sample() returns a constant far from the draft, so a frame equal to the
    draft can only have come through the keep mask."""

    def __init__(self):
        self.keep_masks = []

    def sample(self, music, draft, noise_mask, keep_mask=None, **kwargs):
        self.keep_masks.append(keep_mask.clone())
        generated = torch.full_like(draft, -7.0)
        return keep_mask * draft + (1.0 - keep_mask) * generated


def _keep(side):
    frames = 64
    draft = torch.arange(frames * DIM, dtype=torch.float32).reshape(frames, DIM) / 1000.0
    recorder = Recorder()
    out = infer_completion(recorder, torch.zeros(frames, 4), draft,
                           torch.ones(frames, DIM), frames, frames, "cpu",
                           seam_frames=[SEAM], inpaint_seam_width=WIDTH,
                           inpaint_seam_side=side)
    return recorder.keep_masks[0][0, :, 0], out, draft


def test_the_arrival_is_held_and_only_the_last_frames_loosen():
    keep, _, _ = _keep("after")
    assert torch.all(keep[:SEAM - SEAM_AFTER_LEAD] == 1.0)
    assert abs(float(keep[SEAM - 1]) - 0.25) < 1e-6
    assert abs(float(keep[SEAM - 2]) - 0.75) < 1e-6
    assert float(keep[SEAM]) == 0.0
    after = keep[SEAM:SEAM + WIDTH + 1]
    assert torch.all(after[1:] >= after[:-1] - 1e-6), "the ramp after the seam must rise"
    assert torch.all(keep[SEAM + WIDTH:] == 1.0)


def test_the_centred_default_is_what_frees_the_arrival():
    """Positive control for the one above: the published mask DOES free the
    frames before the seam, so the "after" assertions are not vacuous."""
    keep, _, _ = _keep("centred")
    assert float(keep[SEAM - 4]) < 1.0 and float(keep[SEAM - 3]) < 1.0


def test_the_seam_is_still_generated_not_copied():
    """The trap atomic_completion records: a mask that keeps everything reads as
    a perfect result while the model does nothing."""
    _, out, draft = _keep("after")
    assert torch.allclose(out[:SEAM - SEAM_AFTER_LEAD], draft[:SEAM - SEAM_AFTER_LEAD])
    assert not torch.allclose(out[SEAM], draft[SEAM], atol=1e-3)


def _two_units(frames=60):
    draft = torch.zeros(frames, DIM)
    draft[:SEAM] = 1.0 + torch.linspace(0, 0.5, SEAM).unsqueeze(1)   # unit A, moving
    draft[SEAM:] = -2.0                                               # unit B
    return draft, torch.ones(frames, 1), torch.ones(frames, dtype=torch.long)


def test_the_fade_touches_nothing_before_the_bar_line():
    draft, mask, labels = _two_units()
    before = draft.clone()
    _blend_draft_seams(draft, mask, labels, WIDTH, seams=[SEAM], side="after")
    assert torch.equal(draft[:SEAM], before[:SEAM])
    assert torch.equal(draft[SEAM + WIDTH:], before[SEAM + WIDTH:])
    # starts at the arriving pose, ends in the next unit
    assert torch.allclose(draft[SEAM], before[SEAM - 1], atol=0.2)
    assert torch.allclose(draft[SEAM + WIDTH - 1], before[SEAM + WIDTH - 1], atol=0.2)


def test_a_stagger_it_cannot_perform_is_refused():
    with pytest.raises(SystemExit) as caught:
        infer_atomic.infer_directory(
            audio_dir="/nonexistent", output_dir="/nonexistent",
            planner_checkpoint="/nonexistent", completion_checkpoint="/nonexistent",
            data_root="/nonexistent", draft_seam_blend=8, draft_seam_stagger=True,
            seam_transition="after")
    assert "--draft-seam-stagger" in str(caught.value)


def test_the_switch_is_threaded_to_both_the_draft_and_the_completion():
    tree = ast.parse(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "infer_atomic.py")).read())
    names = {"_source_safe_draft": "seam_transition",
             "infer_completion": "inpaint_seam_side"}
    for func, keyword in names.items():
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == func]
        assert calls, func
        for call in calls:
            if func == "infer_completion" and not any(
                    k.arg == "inpaint_seam_width" for k in call.keywords):
                continue      # the draft-only / oracle paths free nothing
            assert keyword in {k.arg for k in call.keywords}, (func, call.lineno)


def test_the_arriving_unit_coasts_instead_of_freezing():
    """Freezing the arriving frame -- which is still moving -- is a velocity
    step at the bar line; the remaining seam jerk sat on the frames s-3..s+1
    until every channel kept its velocity and eased it out."""
    # The next unit continues the same ramp, so the fade itself pulls nowhere
    # and the first frame after the bar line shows only what replaced the
    # arriving frame: a freeze gives ~0 velocity there, a coast ~0.9 of it.
    frames = 60
    draft = torch.linspace(0, 3, frames).unsqueeze(1).repeat(1, DIM)
    before = draft.clone()
    _blend_draft_seams(draft, torch.ones(frames, 1), torch.ones(frames, dtype=torch.long),
                       16, seams=[SEAM], side="after")
    v_in = float(before[SEAM - 1, 4] - before[SEAM - 2, 4])
    v_out = float(draft[SEAM, 4] - before[SEAM - 1, 4])
    assert v_in > 0 and v_out > 0.5 * v_in, (v_in, v_out)   # no velocity step to zero
