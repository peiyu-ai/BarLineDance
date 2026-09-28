"""``--draft-whole-units``: do not retrieve a prototype the window cut in half.

WHY.  Every library unit is supposed to be FOUR BEATS of its own song -- the T
line's segmentation is ``runs/txy_t_seg_beat4`` with ``mode: grid,
beats_per_segment: 4`` -- and that is exactly what makes the draft's linear
stretch legitimate: four beats resampled onto four beats sends beat k to beat k.

But the candidates are label runs found INSIDE a 150-frame release window, and a
window that lands in the middle of a segment indexes the part it happens to hold
as though that were a unit.  Measured 2026-09-11 over the T library:

    candidates                    15,203
    touching a window edge        10,375   (68.2%)
    whole-run length     median 61 frames, p10 48, p90 80
    edge-run length      median 40 frames, p10  9, p90 103

and the ``duration`` rule ranks by ``min |frames difference|``, which cannot
tell three beats of a slow song from four beats of a fast one.  So for a short
slot the pool is mostly fragments and the winner is likely to be one.

WHAT THE DEFECT COSTS, measured with tools/score_beat_phase_profile.py on the 19
scorable eval clips (window-level gain against other clips' beat grids as the
null, bootstrap over clips): ground truth is beat-locked at +0.0338
[0.0048, 0.0659] and the shipped arm reads -0.0034 [-0.0192, 0.0146], with the
loss already present in the DRAFT (+0.0031) rather than introduced by the
completion.  Offline, substituting WHOLE ground-truth bars through the same
assembly keeps +0.021 of that lock, so the assembly can carry one.

POSITIVE CONTROL is ``test_the_flag_actually_removes_the_edge_runs``: without
it, every other test here would pass on an implementation that accepted the
argument and ignored it.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_atomic import IndexedAtomicMotionLibrary

FEATURE_DIM = 8


def _library(tmpdir, label_rows, **kwargs):
    """``label_rows`` is a list of per-window label arrays, written verbatim."""
    train = os.path.join(tmpdir, "train")
    os.makedirs(train, exist_ok=True)
    width = len(label_rows[0])
    labels = np.asarray(label_rows, dtype=np.int64)
    motion = np.zeros((len(label_rows), width, FEATURE_DIM), dtype=np.float32)
    np.save(os.path.join(train, "motion.npy"), motion)
    np.save(os.path.join(train, "labels.npy"), labels)
    with open(os.path.join(train, "names.json"), "w") as handle:
        json.dump(["src_{}_slice0".format(i) for i in range(len(label_rows))], handle)
    torch.save({"data_min": torch.full((FEATURE_DIM,), -1.0),
                "data_max": torch.full((FEATURE_DIM,), 1.0)},
               os.path.join(tmpdir, "normalizer.pt"))
    return IndexedAtomicMotionLibrary(tmpdir, **kwargs)


def _window(width, runs):
    """One window: ``runs`` are (label, start, end) written into a zero array."""
    row = np.zeros(width, dtype=np.int64)
    for label, start, end in runs:
        row[start:end] = label
    return row


def _spans(library, label):
    return sorted((entry[1], entry[2]) for entry in library.index.get(label, []))


def test_the_flag_actually_removes_the_edge_runs():
    """POSITIVE CONTROL: an ignored argument would leave the index identical."""
    width = 20
    rows = [_window(width, [(1, 0, 6), (1, 8, 14), (1, 16, width)])]
    with tempfile.TemporaryDirectory() as d:
        every = _library(d, rows)
    with tempfile.TemporaryDirectory() as d:
        whole = _library(d, rows, whole_units=True)
    assert _spans(every, 1) == [(0, 6), (8, 14), (16, 20)]
    # Only the run that touches neither edge survives.
    assert _spans(whole, 1) == [(8, 14)]


def test_a_run_that_merely_ends_early_is_kept():
    """The rule is about the WINDOW edge, not about being short.

    A four-frame run in the middle of a window is a real short segment and has
    to stay, or the flag would be a length filter wearing another name.
    """
    width = 20
    rows = [_window(width, [(1, 3, 7), (1, 9, 18)])]
    with tempfile.TemporaryDirectory() as d:
        whole = _library(d, rows, whole_units=True)
    assert _spans(whole, 1) == [(3, 7), (9, 18)]


def test_the_default_is_the_published_behaviour():
    """Off by default: every artifact made before this existed must reproduce."""
    width = 20
    rows = [_window(width, [(1, 0, 6), (1, 8, 14)])]
    with tempfile.TemporaryDirectory() as d:
        library = _library(d, rows)
    assert library.whole_units is False
    assert _spans(library, 1) == [(0, 6), (8, 14)]


def test_it_refuses_rather_than_emptying_a_class():
    """Fail closed.  An emptied pool would be filled by the caller's fallback
    and recorded in the manifest as though the flag had worked."""
    width = 12
    rows = [_window(width, [(1, 0, width)])]      # one run, spanning the window
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="whole-units"):
            _library(d, rows, whole_units=True)


def test_filler_is_filtered_on_the_same_rule():
    """``--index-filler`` puts label 0 in the pool; it is cut by windows too --
    measured 14.0% of its candidates survive, the lowest of the 21 classes."""
    width = 20
    rows = [_window(width, [(0, 0, 5), (1, 5, 9), (0, 9, 15), (1, 15, width)])]
    with tempfile.TemporaryDirectory() as d:
        whole = _library(d, rows, whole_units=True, index_filler=True)
    assert _spans(whole, 0) == [(9, 15)]
    assert _spans(whole, 1) == [(5, 9)]


def test_the_drop_count_is_recorded():
    """A flag that silently removed two thirds of the library would be worse
    than one that did nothing, so the count is available to the caller."""
    width = 20
    rows = [_window(width, [(1, 0, 6), (1, 8, 14), (1, 16, width)])]
    with tempfile.TemporaryDirectory() as d:
        whole = _library(d, rows, whole_units=True)
    assert whole.dropped_fragments == 2
