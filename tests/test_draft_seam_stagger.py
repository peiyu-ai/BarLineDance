"""``--draft-seam-stagger``: stop the four limbs changing on the same frame.

WHY.  The operator, 2026-09-08: "表现动作可以用四肢,腰跨,肩部,等等,表现力的
丰富度不足".  The measurement that agrees is ``lag0`` -- the share of limb pairs
whose speed envelopes correlate best at zero lag.  Measured 2026-09-08 on the 20
T-series eval clips: ground truth 0.417, the shipped arm 0.500, and
``--draft-beat-anchor 1.4`` makes it WORSE at 0.617, because the anchor lands the
phase by braking every part on the same frame.

THE MECHANISM, and it is not new.  ``_blend_draft_seams`` has carried the
``stagger`` argument since the blend landed, and until this file nothing outside
the function passed it -- no CLI switch, no call site.  A single library
prototype already has a real dancer's limb structure (lag0 0.000 on limb pairs);
butt-joining two reads 1.000, because every seam is one instant at which all
four limbs change together.  A synchronous cross-fade softens the jolt and
leaves that at 1.000; only offsetting each limb's fade separates them.

THE POSITIVE CONTROL is ``test_the_four_limbs_stop_changing_on_one_frame``: a
draft built from two prototypes that differ in every limb, where the
synchronous blend must put every limb's change on the same frames and the
staggered one must not.  Without it a do-nothing implementation passes
everything else here.

WHAT IT MAY NOT DO.  It may not change the draft's length, and it may not touch
frames outside the widest staggered window -- the stagger moves WHEN a limb
crosses over, never which slot owns the frame.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infer_atomic
from infer_atomic import (CONTACT_CHANNELS, LIMB_DIMS, LIMB_STAGGER,
                          ROOT_POSITION_DIMS, _blend_draft_seams)

FEATURE_DIM = CONTACT_CHANNELS + ROOT_POSITION_DIMS + 24 * 6
HALF = 8


def _two_prototypes(frames=60, seam=30):
    """A draft that is constant on each side of one seam and differs everywhere.

    Constant sides mean every frame whose value changes is a frame the blend
    touched, so "where did each limb change" is exact rather than inferred.
    """
    draft = torch.zeros(frames, FEATURE_DIM, dtype=torch.float32)
    draft[seam:] = 1.0
    mask = torch.ones(frames, 1, dtype=torch.float32)
    labels = torch.zeros(frames, dtype=torch.long)
    labels[seam:] = 1                      # one prototype-to-prototype seam
    return draft, mask, labels, seam


def _changed_frames(draft, columns):
    moving = (draft[:, list(columns)].diff(dim=0).abs().sum(dim=1) > 1e-9)
    return set(np.flatnonzero(moving.numpy()).tolist())


def test_the_four_limbs_stop_changing_on_one_frame():
    """POSITIVE CONTROL: synchronous puts every limb on one frame set, staggered does not."""
    draft, mask, labels, _ = _two_prototypes()
    _blend_draft_seams(draft, mask, labels, HALF, stagger=False, window="cosine")
    together = [_changed_frames(draft, LIMB_DIMS[name]) for name in LIMB_DIMS]
    assert all(f == together[0] for f in together), (
        "the synchronous blend is supposed to move all four limbs on the same "
        "frames; if it does not, this fixture no longer tests what it claims")

    draft, mask, labels, _ = _two_prototypes()
    _blend_draft_seams(draft, mask, labels, HALF, stagger=True, window="cosine")
    apart = [_changed_frames(draft, LIMB_DIMS[name]) for name in LIMB_DIMS]
    assert not all(f == apart[0] for f in apart), "the stagger changed nothing"
    # Every distinct offset in LIMB_STAGGER must produce a distinct window.
    starts = {name: min(_changed_frames(draft, LIMB_DIMS[name])) for name in LIMB_DIMS}
    assert len(set(starts.values())) == len(set(LIMB_STAGGER.values())), starts


def test_the_stagger_offsets_follow_LIMB_STAGGER():
    draft, mask, labels, seam = _two_prototypes()
    _blend_draft_seams(draft, mask, labels, HALF, stagger=True, window="cosine")
    for name, offset in LIMB_STAGGER.items():
        frames = _changed_frames(draft, LIMB_DIMS[name])
        centre = (min(frames) + max(frames) + 1) / 2.0
        assert abs(centre - (seam + offset * HALF)) <= 1.0, (name, centre)


def test_length_is_unchanged_and_far_frames_are_untouched():
    draft, mask, labels, seam = _two_prototypes()
    before = draft.clone()
    _blend_draft_seams(draft, mask, labels, HALF, stagger=True, window="cosine")
    assert draft.shape == before.shape
    reach = int(HALF * (1 + max(abs(v) for v in LIMB_STAGGER.values()))) + 1
    far = list(range(0, seam - reach)) + list(range(seam + reach, len(draft)))
    assert torch.equal(draft[far], before[far]), "the stagger reached outside its window"


def test_stagger_without_a_blend_width_is_refused_not_ignored():
    """A flag that cannot act must fail, not be recorded as if it had acted."""
    import pytest
    with pytest.raises(SystemExit) as caught:
        infer_atomic.infer_directory(
            audio_dir="/nonexistent", output_dir="/nonexistent",
            planner_checkpoint="/nonexistent", completion_checkpoint="/nonexistent",
            data_root="/nonexistent", draft_seam_blend=0, draft_seam_stagger=True)
    assert "--draft-seam-blend" in str(caught.value)


def test_the_switch_reaches_the_draft_builder_from_the_cli():
    """AST, not a run: the parameter must be threaded, not merely accepted.

    This is the [[manifest-omits-the-flag-that-named-the-arm]] shape -- a switch
    that parses, is recorded, and reaches nothing.
    """
    import ast
    tree = ast.parse(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "infer_atomic.py")).read())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "_source_safe_draft"]
    assert calls, "no _source_safe_draft call sites found"
    for call in calls:
        keywords = {k.arg for k in call.keywords}
        assert "seam_stagger" in keywords, (
            "a _source_safe_draft call site at line {} does not pass "
            "seam_stagger".format(call.lineno))
