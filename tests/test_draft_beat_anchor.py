"""``--draft-beat-anchor``: land the prototype's settle points ON the query's beats.

WHY.  The operator, 2026-09-07, on the union arm:
"动作卡点还是不如 gt,虽然动作节奏有,但是不够舒展到位".

THE READING THAT AGREES.  ``settle`` -- how much the motion decelerates INTO
the beat -- is +0.0569 for ground truth and -0.0769 for the shipped arm.  Ground
truth arrives at a shape and rests on the beat; the generated arm is still
accelerating through it.  "有节奏但不到位" is exactly that: the accents exist,
they do not LAND.

THE MECHANISM.  ``_values_at`` fills a slot with one global linear resample, so
a prototype's internal accents land wherever the stretch puts them.

PROVENANCE (CLAUDE.md 2.1 rule 1).  Neither anchor set is invented here.  The
source anchors are the paper's own motion beat -- "local minima of segment-wise
joint velocities", atomicDance 3.2, implemented as ``motion_accent_frames`` --
and the targets are the music beat grid the bar planner already cuts the plan
on.  ``warp_to_anchors`` was written for this use ("the music's beat frames at
inference") and until now nothing outside its own tests called it.

THE POSITIVE CONTROL (2.1 rule 2) is ``test_a_known_offbeat_accent_is_moved_
onto_the_beat``: a prototype whose single settle point is a known number of
frames off the beat, which the warp must move ONTO it.  Without that, a
do-nothing implementation would pass every other test here.

WHAT IT MAY NOT DO.  It may not change the slot's length, its start or its end
(the length guarantee from 2026-09-05: stretch median 1.000, no unit over the
library ceiling), and it may not stretch any segment past the cap -- an
over-cap pairing is dropped, not clamped, because a clamped anchor no longer
lands where it claims to.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.atomic import ROT6D_START, motion_accent_frames, warp_to_anchors

FEATURE_DIM = ROT6D_START + 6


def _with_settle_at(frames, settle):
    """A span that moves steadily except for one clear settle at ``settle``.

    Built as a cumulative sum of a speed profile, so the rot6d change rate has
    its minimum exactly at ``settle`` and ``motion_accent_frames`` must find it
    there -- asserted below before anything else uses this fixture.
    """
    # A strictly convex speed profile: the ONLY local minimum is at ``settle``.
    # A flat profile will not do -- motion_accent_frames reads a plateau as a
    # run of minima (speed[i] <= both neighbours holds everywhere on it), so a
    # constant-speed fixture reports settles all over the span and a do-nothing
    # warp would appear to work.  The first version of this fixture had exactly
    # that defect and the control's own guard caught it.
    index = np.arange(frames, dtype=np.float64)
    speed = 0.02 + 0.0008 * (index - settle) ** 2
    motion = torch.zeros(frames, FEATURE_DIM)
    motion[:, ROT6D_START:] = torch.from_numpy(
        np.cumsum(speed)[:, None].repeat(6, axis=1)).float()
    return motion


def test_the_fixture_really_does_settle_where_it_claims():
    """Instrument check before any judgement rests on it (CLAUDE.md 2.2)."""
    motion = _with_settle_at(60, 30)
    found = motion_accent_frames(motion)
    assert len(found), "no settle found at all"
    assert min(abs(int(f) - 30) for f in found) <= 2, found


def test_a_known_offbeat_accent_is_moved_onto_the_beat():
    """THE POSITIVE CONTROL.  Answer known in advance: the settle is at 30, the
    beat is at 40, so after the warp the settle must be at 40."""
    motion = _with_settle_at(80, 30)
    before = motion_accent_frames(motion)
    assert min(abs(int(f) - 40) for f in before) > 5, (
        "the fixture already settles on the beat; the control proves nothing")
    warped = warp_to_anchors(motion, before, np.array([40]), max_stretch=1.6)
    after = motion_accent_frames(warped)
    assert min(abs(int(f) - 40) for f in after) <= 2, (before, after)


def test_it_does_not_change_the_slot_length():
    """The length guarantee: a warp that resized a slot would reintroduce the
    slow-motion defect the bar prototypes fixed."""
    motion = _with_settle_at(80, 30)
    warped = warp_to_anchors(motion, motion_accent_frames(motion),
                             np.array([40]), max_stretch=1.6)
    assert warped.shape == motion.shape


def test_the_endpoints_are_pinned_so_seams_are_untouched():
    """The first and last frame must survive, or every join moves with it and
    the seam work is undone."""
    motion = _with_settle_at(80, 30)
    warped = warp_to_anchors(motion, motion_accent_frames(motion),
                             np.array([40]), max_stretch=1.6)
    assert torch.allclose(warped[0], motion[0], atol=1e-5)
    assert torch.allclose(warped[-1], motion[-1], atol=1e-5)


def test_an_over_cap_pairing_is_dropped_rather_than_clamped():
    """Asking to move a settle further than the cap allows must leave the span
    ALONE, not deliver a half-moved accent that lands nowhere."""
    motion = _with_settle_at(80, 30)
    # 30 -> 78 is a 2.6x stretch of the leading segment, far past the cap
    warped = warp_to_anchors(motion, motion_accent_frames(motion),
                             np.array([78]), max_stretch=1.2)
    assert torch.equal(warped, motion)


def test_no_anchors_means_no_change():
    """A slot with no beat inside it is left exactly as retrieval produced it."""
    motion = _with_settle_at(60, 30)
    assert torch.equal(
        warp_to_anchors(motion, motion_accent_frames(motion),
                        np.array([], dtype=np.int64)), motion)


def test_the_flag_is_wired_to_every_draft_call_site():
    """The defect this repository keeps paying for is a flag recorded in the
    manifest but never reaching the code -- it has happened eight times on this
    line of work, and twice it made a P0 inert in production because the
    batched path (the shipping default, --inference-batch-size 4) did not
    forward the switch.  So this is asserted statically rather than trusted.
    """
    import ast
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree = ast.parse(open(os.path.join(here, "infer_atomic.py")).read())
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", getattr(func, "attr", None))
        if name in ("build_draft", "_source_safe_draft"):
            sites.append((name, node.lineno,
                          [k.arg for k in node.keywords]))
    assert len(sites) >= 4, sites
    missing = [(n, line) for n, line, kw in sites if "beat_anchor" not in kw]
    assert not missing, "beat_anchor never reaches: {}".format(missing)
