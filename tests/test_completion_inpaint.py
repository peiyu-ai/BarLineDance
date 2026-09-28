"""``--completion-inpaint-seam-width``: make the draft a constraint, not a hint.

WHY.  Measured on the T line 2026-09-05, three draft-side fixes that plainly
worked on the draft reached the final dance almost not at all:

    seam-aware retrieval   draft seam jerk 0.3515 -> 0.2030    transmitted 10.6%
    facing continuity      draft jump 143.9 -> 5.7 deg         transmitted  5.3%
    --index-filler         draft filler 0.000 -> 0.6388 m/s    transmitted 13.1%

and no lever moved that: guidance 1.0-4.5 (a single axis trading richness for
jerk), a 600-epoch retrain with --draft-timing-align (5.3->7.1%, 13.1->17.3%),
--completion-start-step 100->30 (output only 4.6% closer to its own draft) and
--completion-reproject-every 10 (8.6%).  The draft, once filler is indexed, is
already at ground truth's corner on every column except the seam -- interior
jerk 0.2200 against 0.2045, adjacent-unit pose distance 1.6811 against 1.3253 --
so keeping it and regenerating only the joins is the change the data argues for.

THE TRAP THIS MUST NOT BE.  ``atomic_completion`` already records one:
reprojecting at step 0 made the output BE the draft, "read as a spectacular
result" while the model did nothing.  The distinguishing property is that the
SEAM frames are not kept, so the model still has to produce every transition.
``test_seam_frames_are_still_generated`` is that guard, and it is the reason
this file exists rather than a count of kept frames.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.atomic_completion import AtomicCompletionDiffusion


class _Echo(torch.nn.Module):
    """A denoiser that always predicts a constant far from any draft.

    Deterministic, so any frame equal to the draft got there by the keep mask
    and not by the model happening to agree.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def guided_forward(self, motion, music, timesteps, draft, mask, weight,
                       labels=None, draft_guidance_weight=None):
        return torch.full_like(motion, -0.75)


def _diffusion(dim=6, steps=10):
    d = AtomicCompletionDiffusion.__new__(AtomicCompletionDiffusion)
    torch.nn.Module.__init__(d)
    betas = torch.linspace(1e-4, 0.2, steps)
    alphas = 1.0 - betas
    bars = torch.cumprod(alphas, 0)
    d.register_buffer("betas", betas)
    d.register_buffer("alphas", alphas)
    d.register_buffer("alpha_bars", bars)
    prev = torch.cat([bars.new_ones(1), bars[:-1]])
    d.register_buffer("posterior_variance", betas * (1 - prev) / (1 - bars))
    d.register_buffer("posterior_mean_coef1", betas * prev.sqrt() / (1 - bars))
    d.register_buffer("posterior_mean_coef2",
                      (1 - prev) * alphas.sqrt() / (1 - bars))
    d.register_buffer("sqrt_alpha_bars", bars.sqrt())
    d.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - bars).sqrt())
    d.diffusion_steps = steps
    d.num_steps = steps
    d.model = _Echo(dim)
    d.guidance_weight = 1.0
    return d


def _inputs(frames=40, dim=6, seam=20):
    """A draft that is +0.5 throughout, conditioned everywhere but a gap.

    The conditioning mask changes at ``seam``, which is what the keep mask
    treats as a join.
    """
    draft = torch.full((1, frames, dim), 0.5)
    mask = torch.ones(1, frames, 1)
    mask[:, seam:seam + 4] = 0.0            # a gap: the mask changes twice
    music = torch.zeros(1, frames, 4)
    return music, draft, mask


def _keep(mask, width, labels=None):
    """The keep mask exactly as infer_completion builds it.

    ``labels`` is not optional in production -- see
    ``test_a_fully_conditioned_clip_still_has_seams``, which is the case the
    first version of this file failed to cover and the reason the real run came
    back bit-identical to its draft.
    """
    conditioned = mask[..., :1] > 0
    edge = torch.zeros_like(conditioned)
    edge[:, 1:] |= conditioned[:, 1:] != conditioned[:, :-1]
    if labels is not None:
        edge[:, 1:] |= (labels[:, 1:] != labels[:, :-1]).unsqueeze(-1)
    import math
    width = max(1, width)
    e = edge.float().squeeze(-1).unsqueeze(1)
    distance = torch.full_like(e, float(width))
    for d in range(width, 0, -1):
        hit = torch.nn.functional.max_pool1d(e, kernel_size=2 * d + 1, stride=1, padding=d) > 0
        distance = torch.where(hit, torch.full_like(distance, float(d)), distance)
    distance = torch.where(e > 0, torch.zeros_like(distance), distance)
    ramp = 0.5 - 0.5 * torch.cos(math.pi * (distance / float(width)).clamp(0, 1))
    return (ramp.squeeze(1).unsqueeze(-1) * conditioned.float()).expand(-1, -1, 6)


def test_off_by_default_the_draft_is_only_a_condition():
    torch.manual_seed(0)
    music, draft, mask = _inputs()
    out = _diffusion().sample(music, draft, mask)
    assert not torch.allclose(out, draft, atol=0.2), "the model was ignored"


def test_kept_frames_come_back_as_the_draft():
    torch.manual_seed(0)
    music, draft, mask = _inputs()
    keep = _keep(mask, 3)
    out = _diffusion().sample(music, draft, mask, keep_mask=keep)
    full = keep > 0.999
    assert full.any()
    assert torch.allclose(out[full], draft[full], atol=1e-5)


def test_seam_frames_are_still_generated():
    """THE GUARD.  If these matched the draft too, the model did nothing and the
    result would be the trap this file's docstring describes."""
    torch.manual_seed(0)
    music, draft, mask = _inputs()
    keep = _keep(mask, 3)
    out = _diffusion().sample(music, draft, mask, keep_mask=keep)
    free = keep < 0.001
    assert free.any()
    assert not torch.allclose(out[free], draft[free], atol=0.2)


def test_the_gap_itself_is_free():
    """Unconditioned frames are never kept -- there is no draft to keep."""
    music, draft, mask = _inputs()
    keep = _keep(mask, 1)
    assert float(keep[0, 20:24].max()) == 0.0


def test_width_zero_frees_only_the_boundary_frames():
    music, draft, mask = _inputs()
    keep = _keep(mask, 1)
    free = (keep[..., 0] < 0.001)[0].nonzero().flatten().tolist()
    assert free == [20, 21, 22, 23, 24]


def test_a_wider_window_frees_strictly_more():
    music, draft, mask = _inputs()
    narrow, wide = _keep(mask, 1), _keep(mask, 5)
    assert float(wide.sum()) < float(narrow.sum()), "wider must free more weight"
    assert bool((wide > narrow + 1e-6).sum() == 0), "wider must not KEEP more anywhere"


def test_a_fully_conditioned_clip_still_has_seams():
    """THE CASE THE FIRST VERSION MISSED, and what it cost.

    With --index-filler every frame carries a prototype, so the conditioning
    mask never changes.  Keying the seam on mask edges therefore found none,
    kept the whole sequence, and the real run came back BIT-IDENTICAL to its
    draft on all three widths -- 0.4311 / 0.2200 / 1.6811 / 0.6072, exactly the
    draft's own numbers -- while the unit tests passed, because the fixture had
    a gap in the mask and production does not.

    A seam is a PLAN boundary: where one prototype ends and the next begins.
    That is also the definition --draft-seam-mask-width uses in training, so the
    two sides now agree.
    """
    frames, dim = 40, 6
    mask = torch.ones(1, frames, 1)              # every frame conditioned
    labels = torch.ones(1, frames, dtype=torch.long)
    labels[:, 20:] = 2                           # one plan boundary at 20
    assert float(_keep(mask, 2).min()) == 1.0, "mask edges alone find no seam here"
    keep = _keep(mask, 2, labels)
    freed = (keep[..., 0] < 0.999)[0].nonzero().flatten().tolist()
    assert freed == [19, 20, 21], freed


def test_adjacent_bars_sharing_a_label_are_not_a_seam():
    """A limit of the plan-boundary definition, asserted rather than hidden.

    --draft-bar-prototypes gives each BAR its own prototype, so two adjacent
    bars with the same label are two prototypes with a join the label track
    cannot see.  Those joins stay kept.  Ground truth's own adjacent bars share
    a label 21.6% of the time, so this is not rare -- it bounds what this switch
    can fix, and is written down so the bound is not rediscovered as a bug.
    """
    mask = torch.ones(1, 40, 1)
    labels = torch.ones(1, 40, dtype=torch.long)   # one label, two bars
    assert float(_keep(mask, 2, labels).min()) == 1.0


def test_the_keep_weight_ramps_and_has_no_corner():
    """THE DEFECT THIS REPLACES, asserted on the ramp itself.

    The first version was boolean, so the constraint switched from 1 to 0 in one
    frame.  Measured 2026-09-06 by profiling jerk against distance from a plan
    boundary: ground truth is flat at 0.21-0.27 everywhere, and the boolean
    version spiked to 9.65 at -10 frames and 5.06/5.32 at +6/+8 -- the EDGES of
    the keep window -- worst 16.85 against ground truth's 0.90, while frames
    -6..+4 stayed clean.  That is why the seam column, sampled at +-2, read
    BETTER than ground truth while the operator watched the video and reported
    jumps between movements.
    """
    mask = torch.ones(1, 60, 1)
    labels = torch.ones(1, 60, dtype=torch.long)
    labels[:, 30:] = 2
    w = _keep(mask, 8, labels)[0, :, 0]
    assert float(w[30]) == 0.0                       # free at the boundary
    assert float(w[10]) == 1.0                       # fully kept far away
    seg = w[30:39]
    assert torch.all(seg[1:] >= seg[:-1] - 1e-6), "the ramp must be monotone"
    second = (seg[2:] - 2 * seg[1:-1] + seg[:-2]).abs().max()
    assert float(second) < 0.25, "a corner would show up as large curvature"
