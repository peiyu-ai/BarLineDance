"""``--draft-bar-units``: index the library by SOURCE BAR, not by label run.

WHY.  Candidates were label runs found inside 150-frame release windows, and a
run the window edge clips is a fragment whose cut lands mid-bar.  Measured
2026-09-22 on fix7's 101 retrieved units: 31 were such fragments (the same label
continues past the edge in the source recording), and their cut edges sat a
median 7 frames from the source's nearest bar line, 45% of them 10 or more
(a bar is ~55 frames) -- against a median of 0 and 99% within 2 frames for
edges at a run boundary.  A unit that starts or stops mid-bar starts or stops
MID-MOVE, which is the operator's "做到一半就收".

``--draft-whole-units`` answers the same question by dropping those runs, and
it cost the beat count: multi-bar runs stay whole, four-beat candidates run
short, 17 of 101 slots found none of the right beat count and 16.8% of bars came
back a wrong whole number of beats (0.0% before).  Replacing the run index with
bars read 17.8% too, and every one of those was a SLOT that is not four beats
(a clip's partial edge bar, an odd bar of the query's grid) -- which only a
fragment can fill.  So bars are a second index: four-beat slots take whole
bars, the rest keep the runs.

POSITIVE CONTROL: ``test_a_run_clipped_by_the_window_is_not_a_candidate`` -- an
implementation that accepted the path and kept the run index passes every other
test here.
"""
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_atomic import IndexedAtomicMotionLibrary

FEATURE_DIM = 8
WIDTH = 30          # window length
STRIDE = 10


def _library(tmpdir, sequence_labels, boundaries, **kwargs):
    """One source sequence cut into overlapping windows (stride 10), the way
    the release does it, with its bar boundaries in a segmentation file."""
    train = os.path.join(tmpdir, "train")
    os.makedirs(train, exist_ok=True)
    sequence_labels = np.asarray(sequence_labels, dtype=np.int64)
    starts = list(range(0, len(sequence_labels) - WIDTH + 1, STRIDE))
    labels = np.stack([sequence_labels[s:s + WIDTH] for s in starts])
    np.save(os.path.join(train, "motion.npy"),
            np.zeros((len(starts), WIDTH, FEATURE_DIM), dtype=np.float32))
    np.save(os.path.join(train, "labels.npy"), labels)
    with open(os.path.join(train, "names.json"), "w") as handle:
        json.dump(["wild_v5:111:clip000_slice{}".format(i) for i in range(len(starts))], handle)
    with open(os.path.join(tmpdir, "windows.jsonl"), "w") as handle:
        for index, start in enumerate(starts):
            handle.write(json.dumps({"split": "train", "array_index": index,
                                     "sequence_id": "wild_v5:111:clip000",
                                     "start_frame": start}) + "\n")
    segmentation = os.path.join(tmpdir, "segmentation.json")
    with open(segmentation, "w") as handle:
        json.dump({"records": [{"sequence": "111__clip000", "boundaries": boundaries}]}, handle)
    torch.save({"data_min": torch.full((FEATURE_DIM,), -1.0),
                "data_max": torch.full((FEATURE_DIM,), 1.0)},
               os.path.join(tmpdir, "normalizer.pt"))
    return IndexedAtomicMotionLibrary(tmpdir, bar_units_path=segmentation, **kwargs), starts


def _sequence():
    # bars at 0|12|24|36|48|60: labels 1,1,2,2,3 -- two two-bar runs and one bar
    track = np.zeros(60, dtype=np.int64)
    track[0:24] = 1
    track[24:48] = 2
    track[48:60] = 3
    return track, [0, 12, 24, 36, 48, 60]


def _source_spans(library, starts, label):
    """Bar candidates in SEQUENCE frames, so overlapping windows can be compared."""
    return sorted((starts[w] + s, starts[w] + e) for (w, s, e, _) in library.bar_index.get(label, []))


def test_every_candidate_is_one_whole_bar(tmp_path):
    track, bars = _sequence()
    library, starts = _library(str(tmp_path), track, bars)
    assert _source_spans(library, starts, 1) == [(0, 12), (12, 24)]
    assert _source_spans(library, starts, 2) == [(24, 36), (36, 48)]
    assert _source_spans(library, starts, 3) == [(48, 60)]
    assert library.bar_units_indexed == 5 and library.bar_units_mixed == 0


def test_a_run_clipped_by_the_window_is_not_a_candidate(tmp_path):
    """POSITIVE CONTROL.  The window starting at frame 10 holds label 1 over
    [0, 14) of its own frames -- a run the edge cut two frames into a bar.  The
    run index takes it; the bar index must not."""
    track, bars = _sequence()
    library, starts = _library(str(tmp_path), track, bars)
    every = [span for label in library.bar_index for span in _source_spans(library, starts, label)]
    assert all(start in bars and end in bars for start, end in every), every
    assert (10, 24) not in every


def test_a_bar_whole_in_several_windows_is_indexed_once(tmp_path):
    track, bars = _sequence()
    library, starts = _library(str(tmp_path), track, bars)
    spans = [span for label in library.bar_index for span in _source_spans(library, starts, label)]
    assert len(spans) == len(set(spans))


def test_a_bar_the_label_track_splits_is_counted_not_guessed(tmp_path):
    track, bars = _sequence()
    track[40:48] = 4                     # bar [36, 48) now carries two labels
    library, starts = _library(str(tmp_path), track, bars)
    assert library.bar_units_mixed == 1
    assert (36, 48) not in _source_spans(library, starts, 2)


def test_whole_units_and_bar_units_are_refused_together(tmp_path):
    track, bars = _sequence()
    with pytest.raises(ValueError):
        _library(str(tmp_path), track, bars, whole_units=True)


def _picked(library, starts, target_length, beat_period):
    library.retrieval_log = []
    library.retrieve(1, target_length, target_beat_period=beat_period)
    w, s, e = library.retrieval_log[-1]["source"][:3]
    return starts[w] + s, starts[w] + e


def test_a_four_beat_slot_takes_a_whole_bar_and_a_partial_slot_keeps_the_runs(tmp_path):
    """The replacement version sent EVERY slot to whole bars, and the slots that
    are not four beats -- a clip's partial first bar, an odd bar of the query's
    grid -- came back the wrong number of beats.  Here a 12-frame slot at a
    3-frame beat is four beats (a bar); a 6-frame slot is two (not a bar)."""
    track, bars = _sequence()
    library, starts = _library(str(tmp_path), track, bars)
    start, end = _picked(library, starts, 12, 3.0)
    assert start in bars and end in bars
    library.retrieval_log = []
    library.retrieve(1, 6, target_beat_period=3.0)
    assert library.bar_units_fallback == 1 and library.bar_units_slots == 1
