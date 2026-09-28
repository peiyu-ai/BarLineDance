"""``--index-filler``: let the transition class be retrieved like any other.

THE DEFECT.  The index builder excluded label 0 with a bare truthiness test and
no comment, so filler could never be RETRIEVED -- only bridged by
``--draft-gap-fill``, which draws a straight line or holds a pose.  Filler is
30% of the plan's frames; the line reads 0.015 m/s where ground truth on the
SAME frames moves at 0.628 m/s.

HOW IT WAS FINALLY MEASURED, after four instruments read clean.  Absolute speed
floors, floors at a fraction of the clip's own median, per-limb medians and
local dips all ask "is this clip slow for itself", and none can ask "should it
be moving right now".  Comparing against the same clip's ground truth AT THE
SAME FRAME, smoothed over one second: the draft has windows at literally zero
speed and 24.7% of its windows run below 0.6x ground truth.  Ground truth
against itself is the positive control and reads 1.000.

"Filler" names a span the vocabulary did not classify.  It does not name a span
where the dancer stopped -- ground truth's filler frames move as fast as its
classified ones, which is the whole evidence for this change.
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_atomic import CONTACT_CHANNELS, IndexedAtomicMotionLibrary

FEATURE_DIM = 8


def _library(tmpdir, **kwargs):
    """One recording: 20 frames of class 1, 20 of filler, 20 of class 2."""
    train = os.path.join(tmpdir, "train")
    os.makedirs(train)
    motion = np.zeros((1, 60, FEATURE_DIM), dtype=np.float32)
    motion[0, :, CONTACT_CHANNELS:] = np.linspace(
        0.0, 1.0, 60, dtype=np.float32)[:, None]
    labels = np.zeros((1, 60), dtype=np.int64)
    labels[0, :20] = 1
    labels[0, 40:] = 2
    np.save(os.path.join(train, "motion.npy"), motion)
    np.save(os.path.join(train, "labels.npy"), labels)
    with open(os.path.join(train, "names.json"), "w") as handle:
        json.dump(["src_0_slice0"], handle)
    torch.save({"data_min": torch.full((FEATURE_DIM,), -1.0),
                "data_max": torch.full((FEATURE_DIM,), 1.0)},
               os.path.join(tmpdir, "normalizer.pt"))
    return IndexedAtomicMotionLibrary(tmpdir, **kwargs)


def test_off_by_default_filler_is_not_in_the_index():
    with tempfile.TemporaryDirectory() as d:
        library = _library(d)
        assert set(library.index) == {1, 2}


def test_on_it_is_indexed_with_the_frames_it_actually_spans():
    with tempfile.TemporaryDirectory() as d:
        library = _library(d, index_filler=True)
        assert set(library.index) == {0, 1, 2}
        assert [(c[1], c[2]) for c in library.index[0]] == [(20, 40)]


def test_off_the_draft_leaves_filler_unconditioned():
    """The published behaviour: those frames get no prototype and no mask."""
    with tempfile.TemporaryDirectory() as d:
        library = _library(d)
        plan = torch.zeros(60, dtype=torch.long)
        plan[:20] = 1
        plan[40:] = 2
        draft, mask = library.build_draft(plan, FEATURE_DIM)
        assert float(mask[20:40].sum()) == 0.0


def test_on_the_draft_fills_them_from_the_library():
    """The point of the switch: those frames become retrieved motion.

    Asserted on the MASK as well as the values, because a draft that filled the
    frames without marking them conditioned would leave the completion model
    told to invent motion that is already there.
    """
    with tempfile.TemporaryDirectory() as d:
        library = _library(d, index_filler=True)
        plan = torch.zeros(60, dtype=torch.long)
        plan[:20] = 1
        plan[40:] = 2
        draft, mask = library.build_draft(plan, FEATURE_DIM)
        assert float(mask[20:40].sum()) == 20.0
        assert not torch.equal(draft[20:40], torch.zeros(20, FEATURE_DIM))


def test_the_filled_span_is_not_a_straight_line():
    """The defect this replaces is a LINE; anything retrieved must not be one.

    A straight line has zero second difference everywhere.  The prototype used
    here is itself a ramp, so the test asserts the span came from retrieval by
    its VALUES matching the library's own frames rather than by curvature.
    """
    with tempfile.TemporaryDirectory() as d:
        library = _library(d, index_filler=True)
        plan = torch.zeros(60, dtype=torch.long)
        plan[:20] = 1
        plan[40:] = 2
        draft, _ = library.build_draft(plan, FEATURE_DIM)
        source = torch.from_numpy(np.array(library.motion[0, 20:40], copy=True))
        assert torch.allclose(draft[20:40], source, atol=1e-5)


def test_the_retrieval_log_counts_the_filler_unit():
    """It is motion the draft played, so the stretch gate must see it too."""
    with tempfile.TemporaryDirectory() as d:
        library = _library(d, index_filler=True)
        plan = torch.zeros(60, dtype=torch.long)
        plan[:20] = 1
        plan[40:] = 2
        library.build_draft(plan, FEATURE_DIM)
        assert [r["label"] for r in library.retrieval_log] == [1, 0, 2]


# ---------------------------------------------------------------------------
# --index-filler widens what the mask may cover, so it must widen the
# denominator of safe_draft_condition_fraction too.


def test_the_condition_fraction_denominator_follows_index_filler():
    """A fraction above one is a broken instrument, not a reading.

    ``safe_draft_condition_fraction`` is mask.sum() over the frames a prototype
    was ALLOWED to cover.  With --index-filler the transition class is retrieved
    like any other, so the mask covers filler frames -- while the denominator
    counted only ``labels != 0``.  Measured 2026-09-06 across the 20 eval clips,
    the field then read up to 7.792.

    This field is not decorative: its own docstring records the 2026-08-16 case
    where 40 clips reported 1.0 beside a present retrieval group while 21 of
    them had a plan that was 100% transition, and the fid_k that produced was
    read as reproducing the paper.
    """
    import torch
    from infer_atomic import _safe_draft_condition_fraction

    labels = torch.zeros(10, dtype=torch.long)
    labels[:4] = 1                                  # 4 atomic, 6 filler
    mask = torch.ones(10, 1)                        # every frame conditioned
    assert _safe_draft_condition_fraction(labels, mask, index_filler=True) == 1.0
    assert _safe_draft_condition_fraction(labels, mask) == 2.5   # the old reading


def test_the_condition_fraction_is_none_when_nothing_was_retrievable():
    """Zero of zero is not "everything was conditioned" -- see the docstring."""
    import torch
    from infer_atomic import _safe_draft_condition_fraction

    labels = torch.zeros(10, dtype=torch.long)      # all filler
    mask = torch.zeros(10, 1)
    assert _safe_draft_condition_fraction(labels, mask) is None
    # with filler indexed those frames ARE retrievable, so zero is a real zero
    assert _safe_draft_condition_fraction(labels, mask, index_filler=True) == 0.0
