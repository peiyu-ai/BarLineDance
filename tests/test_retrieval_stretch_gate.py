"""The gate that says a plan slot asked for more than its class can supply.

WHY THIS EXISTS.  ``IndexedAtomicMotionLibrary._values_at`` resamples the chosen
prototype into the slot unconditionally, so a slot longer than anything in its
class is filled with slow motion rather than refused.  The ceiling is structural
rather than incidental, though NOT for the reason first written here.  The T
corpus is cut on a 4-beat music grid, so one atomic movement is one bar: median
2.00 s, min 1.23 s, max 2.97 s (runs/txy_t_seg_beat4/segmentation.json).  The
150-frame (5.00 s) figure this docstring used to call "the library ceiling" is
an artefact of adjacent same-label BARS being merged and the merged run then
truncated by the release's 150-frame window -- which is why 17 of 20 classes
"top out" at exactly 150.  Nothing in the corpus is longer than 2.97 s, so any
slot longer than that is by construction several bars and was always meant to be
filled by several prototypes.  Measured 2026-09-05 on the 18 sourced eval clips, 9 of 88 retrieval units
(10.2%) asked for more; the worst asked for 312 frames, was handed 150, and
played the last 10.4 s of a 14.3 s clip at 0.48x speed.

WHY A COUNT AND NOT A RATE.  Over those same 88 units the pooled stretch median
is 1.000 and the p90 is 1.041, because the duration rule finds an exact length
match 80.7% of the time out of a pool whose median size is 690.  Every summary
statistic therefore reads clean while one unit halves the tempo of two thirds of
a clip.  ``units_over_library_ceiling`` is a count for that reason.

WHAT THIS GATE DOES NOT CLAIM.  A long label RUN is not by itself illegitimate:
ground truth's own runs exceed 5.0 s on 4.4% of runs and reach 12.30 s.  But a
run is not a retrieval unit -- ground truth's 12.30 s run is six consecutive bars
that happen to share a label, danced as six different instances.  The defect is
answering "I need 10.4 s" with "here is one prototype at half speed" in silence,
so the gate flags unfillable slots and says nothing about whether the plan should
have asked.
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_atomic import IndexedAtomicMotionLibrary


FEATURE_DIM = 8  # 4 contacts + 3 root + 1 stand-in rotation, as elsewhere in tests


def _library(tmpdir, spans):
    """A release whose class 1 holds prototypes of exactly the given lengths.

    ``spans`` are lengths in frames.  Each becomes its own window padded to the
    longest, with the label track marking only the first ``span`` frames, so the
    indexed segment for that window has precisely that length.
    """
    train = os.path.join(tmpdir, "train")
    os.makedirs(train)
    width = max(spans)
    motion = np.zeros((len(spans), width, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros((len(spans), width), dtype=np.int64)
    for row, span in enumerate(spans):
        # A ramp on the rotation channel, so a resample is visible in the values.
        motion[row, :span, 7] = np.linspace(0.0, 1.0, span, dtype=np.float32)
        labels[row, :span] = 1
    np.save(os.path.join(train, "motion.npy"), motion)
    np.save(os.path.join(train, "labels.npy"), labels)
    with open(os.path.join(train, "names.json"), "w") as handle:
        json.dump(["src_{}_slice0".format(i) for i in range(len(spans))], handle)
    return IndexedAtomicMotionLibrary(tmpdir)


def test_a_slot_longer_than_every_prototype_is_flagged():
    """The failing case: nothing in the class is long enough."""
    with tempfile.TemporaryDirectory() as directory:
        library = _library(directory, [4, 6])
        library.retrieve(1, 12)                      # class tops out at 6
        summary = library.retrieval_stretch_summary()
        assert summary["units"] == 1
        assert summary["units_over_library_ceiling"] == 1
        assert summary["stretch_max"] == 2.0
        assert summary["playback_min"] == 0.5
        worst = summary["worst"]
        assert worst["slot_frames"] == 12
        assert worst["native_frames"] == 6
        assert worst["pool_max_frames"] == 6
        assert worst["slot_exceeds_pool_max"] is True


def test_positive_control_a_fillable_slot_reads_clean():
    """The known-good case: a slot the class can supply exactly must not trip.

    Without this the gate could be a constant True and every test above would
    still pass.
    """
    with tempfile.TemporaryDirectory() as directory:
        library = _library(directory, [4, 6])
        library.retrieve(1, 6)
        summary = library.retrieval_stretch_summary()
        assert summary["units"] == 1
        assert summary["units_over_library_ceiling"] == 0
        assert summary["stretch_max"] == 1.0
        assert summary["playback_min"] == 1.0
        assert summary["worst"]["slot_exceeds_pool_max"] is False


def test_a_stretched_but_fillable_slot_is_not_called_unfillable():
    """Stretch and unfillability are different claims and must not be conflated.

    Asking for 5 when the class holds 4 and 6 picks the 6 and SQUEEZES it
    (stretch < 1).  That is a stretch, but the class could supply the length, so
    a different candidate rule could have avoided it -- the count must stay 0.
    """
    with tempfile.TemporaryDirectory() as directory:
        library = _library(directory, [4, 6])
        library.retrieve(1, 5)
        summary = library.retrieval_stretch_summary()
        assert summary["units_over_library_ceiling"] == 0
        assert summary["stretch_max"] != 1.0


def test_a_cached_repeat_still_counts_as_a_unit():
    """A cache hit is motion the draft played, so it must appear in the log.

    The duration rule caches on (label, target_length, ...), so a clip that uses
    one class twice at the same length takes the cache path the second time.  If
    that path did not log, a clip whose every slot is unfillable would report one
    unit and the count gate would under-read by the repeat factor.
    """
    with tempfile.TemporaryDirectory() as directory:
        library = _library(directory, [4, 6])
        first = library.retrieve(1, 12)
        second = library.retrieve(1, 12)
        assert torch.equal(first, second)            # genuinely the cached pick
        summary = library.retrieval_stretch_summary()
        assert summary["units"] == 2
        assert summary["units_over_library_ceiling"] == 2


def test_the_log_is_empty_before_anything_is_retrieved():
    """No units retrieved must read as "not measured", never as "measured clean".

    A summary of ``{}`` with a zero count would let a caller that never retrieved
    anything pass the gate, which is the failure mode ``check_disk_headroom``'s
    docstring is about.
    """
    with tempfile.TemporaryDirectory() as directory:
        library = _library(directory, [4, 6])
        assert library.retrieval_log == []
        assert library.retrieval_stretch_summary() is None


def test_resetting_the_log_makes_the_count_per_clip():
    """Inference clears the log per clip; the count must follow."""
    with tempfile.TemporaryDirectory() as directory:
        library = _library(directory, [4, 6])
        library.retrieve(1, 12)
        library.retrieval_log = []
        library.retrieve(1, 6)
        summary = library.retrieval_stretch_summary()
        assert summary["units"] == 1
        assert summary["units_over_library_ceiling"] == 0


def test_the_record_says_where_the_unit_starts():
    """``slot_start`` exists because cumulating ``slot_frames`` is WRONG.

    Filler spans are never retrieved, so they never enter this log.  A reader
    that reconstructs unit boundaries by cumulating ``slot_frames`` therefore
    drifts earlier by the total filler length after the clip's first filler
    span -- and a seam measurement built on that reconstruction reads the wrong
    frames on exactly the clips with the most filler.  Two units retrieved at
    known, non-adjacent positions must come back with those positions.
    """
    with tempfile.TemporaryDirectory() as directory:
        library = _library(directory, [4, 6])
        library.retrieve(1, 6, slot_start=0)
        library.retrieve(1, 4, slot_start=40)     # a filler gap sits between
        starts = [r["slot_start"] for r in library.retrieval_log]
        assert starts == [0, 40]
        assert sum(r["slot_frames"] for r in library.retrieval_log) == 10
        # the point: cumulating would have said the second unit starts at 6
        assert starts[1] != library.retrieval_log[0]["slot_frames"]


def test_slot_start_is_absent_rather_than_wrong_when_not_supplied():
    """A caller that does not pass it must yield None, never a guess."""
    with tempfile.TemporaryDirectory() as directory:
        library = _library(directory, [4, 6])
        library.retrieve(1, 6)
        assert library.retrieval_log[0]["slot_start"] is None
