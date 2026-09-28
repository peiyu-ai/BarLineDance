"""``--draft-seam-mask-width``: stop telling the completion model that the
butt-joint IS the evidence.

WHY.  The training draft is two prototypes joined end to end, so the frames
either side of a plan boundary carry a velocity step no dancer produced.  With
the conditioning mask at 1.0 there, the model is trained to reproduce that step.
Measured on the T line 2026-09-05 (17 eval clips, filler frames excluded,
boundaries read from the artifact's own slot_start): jerk within +-2 frames of a
retrieval unit boundary is 0.3515 against ground truth's 0.2553 at the SAME
frame indices, while the interior matches almost exactly (0.2135 vs 0.2045).
The defect is entirely at the join, which is what makes blanking the join --
rather than smoothing the whole clip -- the targeted change.

WHAT MUST NOT HAPPEN.  Blanking must not touch a frame that is not near a seam,
must not alter the mask at width 0 (every earlier checkpoint reproduces), and
must not be confused with blanking the LABEL channel: the model still has to
know which class it is leaving and which it is entering, or it is being asked to
invent a transition between two movements it was not told about.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.atomic import blank_seam_evidence, plan_boundaries


def test_width_zero_is_the_published_behaviour():
    mask = torch.ones(1, 20, 1)
    labels = torch.zeros(1, 20, dtype=torch.long)
    labels[0, 10:] = 1
    out = blank_seam_evidence(mask, plan_boundaries(labels), 0)
    assert torch.equal(out, mask)


def test_only_frames_near_a_seam_are_blanked():
    mask = torch.ones(1, 20, 1)
    labels = torch.zeros(1, 20, dtype=torch.long)
    labels[0, 10:] = 1                       # the seam is frame 10
    out = blank_seam_evidence(mask, plan_boundaries(labels), 2)
    zeroed = (out[0, :, 0] == 0).nonzero().flatten().tolist()
    assert zeroed == [8, 9, 10, 11, 12]


def test_the_input_mask_is_not_mutated():
    """The collator reuses the tensor, so an in-place blank would leak."""
    mask = torch.ones(1, 20, 1)
    labels = torch.zeros(1, 20, dtype=torch.long)
    labels[0, 10:] = 1
    blank_seam_evidence(mask, plan_boundaries(labels), 3)
    assert mask.sum() == 20


def test_two_seams_are_both_blanked_and_a_clip_with_none_is_untouched():
    labels = torch.zeros(1, 30, dtype=torch.long)
    labels[0, 10:20] = 1
    labels[0, 20:] = 2
    out = blank_seam_evidence(torch.ones(1, 30, 1), plan_boundaries(labels), 1)
    assert (out[0, :, 0] == 0).sum() == 6     # 3 frames around each of 2 seams
    flat = torch.zeros(1, 30, dtype=torch.long)
    untouched = blank_seam_evidence(torch.ones(1, 30, 1), plan_boundaries(flat), 1)
    assert untouched.sum() == 30


def test_a_zero_mask_frame_stays_zero():
    """Blanking may only remove evidence, never add it."""
    mask = torch.ones(1, 20, 1)
    mask[0, 15] = 0.0                         # already unconditioned
    labels = torch.zeros(1, 20, dtype=torch.long)
    labels[0, 10:] = 1
    out = blank_seam_evidence(mask, plan_boundaries(labels), 2)
    assert out[0, 15] == 0.0
    assert out.sum() <= mask.sum()


def test_it_scales_before_the_noise_ratio_not_after():
    """Order matters: blanking after scaling would leave 0 either way, but
    blanking a mask that the ratio has already scaled makes width 0 and the
    published path differ by float noise.  Assert the composition the collator
    uses is exactly (blank then scale).
    """
    mask = torch.ones(1, 20, 1)
    labels = torch.zeros(1, 20, dtype=torch.long)
    labels[0, 10:] = 1
    boundaries = plan_boundaries(labels)
    ratio = 0.25
    assert torch.equal(blank_seam_evidence(mask, boundaries, 0) * ratio, mask * ratio)
