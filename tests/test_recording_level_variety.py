"""One clip may not replay one SOURCE RECORDING, not merely one exact window.

WHY.  The operator, 2026-09-07, on
``output/sample_20260907_union/7618203431723357818__clip000.mp4``:
"大的段落我发现有重复,9 秒附近和 14 秒附近的动作是一模一样的".

WHAT THE ARTIFACT SAID.  Those two bars drew library windows 3222 and 3223 --
``wild_v5:7627866330033205883:clip000_slice13`` and the SAME recording's
``_slice14``.  Two adjacent windows of one performance, both class 5, five
seconds apart.

WHY NOTHING REPORTED IT.  ``--draft-recurrence-variety`` kept a set of
``(window, start, end)`` triples.  3222 != 3223, so the pair passed as
"distinct" and the reuse counter read 0.0% while the screen showed the same
move twice.  The trap had already been paid for once: the previous round's
fix note records occurrence 4 drawing (2972, 0, 67) and occurrence 5 drawing
(2973, 0, 52) -- again adjacent slices of one recording.  Excluding the exact
triple only moved the replay one slice over, which is why the key here is the
RECORDING and not the window.

Across the 20 eval clips the triple-keyed rule left 16 of 200 units (8.0%)
replaying a recording the same clip had already used, in 9 clips.

THE FALLBACK MATTERS AS MUCH AS THE RULE.  A slot must always be filled -- a
repeat beats a hole -- so the last test here drives a library that has only one
recording to offer and asserts both slots still come out.
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


def _library(tmpdir, rows, groups, labels_of=None):
    """``rows[i]`` is a prototype; ``groups[i]`` is the recording it came from."""
    train = os.path.join(tmpdir, "train")
    os.makedirs(train)
    width = max(len(r) for r in rows)
    motion = np.zeros((len(rows), width, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros((len(rows), width), dtype=np.int64)
    for i, row in enumerate(rows):
        motion[i, :len(row)] = row
        labels[i, :len(row)] = 1 if labels_of is None else labels_of[i]
    np.save(os.path.join(train, "motion.npy"), motion)
    np.save(os.path.join(train, "labels.npy"), labels)
    with open(os.path.join(train, "names.json"), "w") as handle:
        json.dump(["{}_slice{}".format(groups[i], i) for i in range(len(rows))],
                  handle)
    # The file whose ABSENCE would silently disable this whole rule: without it
    # _read_retrieval_groups hands back None for every window and every
    # candidate looks like the same group.  The release in use carries it (5329
    # entries, 143 distinct groups, no Nones) -- asserted in
    # test_a_release_without_groups_degrades_to_the_old_rule below.
    with open(os.path.join(train, "retrieval_groups.json"), "w") as handle:
        json.dump(list(groups), handle)
    torch.save({"data_min": torch.full((FEATURE_DIM,), -1.0),
                "data_max": torch.full((FEATURE_DIM,), 1.0)},
               os.path.join(tmpdir, "normalizer.pt"))
    return IndexedAtomicMotionLibrary(tmpdir)


def _ramp(start, step, n):
    row = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    for k in range(n):
        row[k, CONTACT_CHANNELS:] = start + step * k
    return row


def _groups_drawn(library):
    return [rec["source"][0] for rec in library.retrieval_log]


def test_the_operators_case_two_slices_of_one_recording_are_not_both_played():
    """The regression, in the shape the artifact recorded it.

    Class 1 offers two windows of recording A (the 3222/3223 pair) and two of
    other recordings, all the same length so the duration band cannot decide.
    Two bars of class 1 must not both come from A.
    """
    with tempfile.TemporaryDirectory() as d:
        rows = [_ramp(1.0, 0.1, 10), _ramp(1.05, 0.1, 10),
                _ramp(1.1, 0.1, 10), _ramp(1.15, 0.1, 10)]
        groups = ["recA", "recA", "recB", "recC"]
        library = _library(d, rows, groups)
        plan = torch.tensor([1] * 20)
        library.build_draft(plan, feature_dim=FEATURE_DIM,
                            bar_bounds=[0, 10, 20],
                            recurrence_variety=True, seam_aware_retrieval=True)
        drawn = _groups_drawn(library)
        assert len(drawn) == 2, drawn
        names = [library.retrieval_group_ids[i] for i in drawn]
        assert names[0] != names[1], (
            "both bars came from {}; this is the 3222/3223 defect".format(names))


def test_the_same_recording_is_refused_across_DIFFERENT_labels_too():
    """A performance showing up under two class labels is just as visible.

    The old rule only ran its exclusion on repeat occurrences of ONE label, so
    this case had no guard at all.
    """
    with tempfile.TemporaryDirectory() as d:
        rows = [_ramp(1.0, 0.1, 10), _ramp(1.05, 0.1, 10),
                _ramp(1.1, 0.1, 10), _ramp(1.15, 0.1, 10)]
        groups = ["recA", "recA", "recB", "recC"]
        # windows 0 and 1 are recA under labels 1 and 2 respectively
        library = _library(d, rows, groups, labels_of=[1, 2, 1, 2])
        plan = torch.tensor([1] * 10 + [2] * 10)
        library.build_draft(plan, feature_dim=FEATURE_DIM,
                            bar_bounds=[0, 10, 20],
                            recurrence_variety=True, seam_aware_retrieval=True)
        names = [library.retrieval_group_ids[i] for i in _groups_drawn(library)]
        assert names[0] != names[1], names


def test_a_slot_is_still_filled_when_only_one_recording_can_supply_it():
    """The fallback.  A repeat beats a hole.

    Every window of class 1 belongs to recording A, so the preference cannot be
    satisfied; both bars must still be filled rather than left empty or raising.
    """
    with tempfile.TemporaryDirectory() as d:
        rows = [_ramp(1.0, 0.1, 10), _ramp(1.05, 0.1, 10)]
        library = _library(d, rows, ["recA", "recA"])
        plan = torch.tensor([1] * 20)
        draft, mask = library.build_draft(plan, feature_dim=FEATURE_DIM,
                                          bar_bounds=[0, 10, 20],
                                          recurrence_variety=True,
                                          seam_aware_retrieval=True)
        assert len(library.retrieval_log) == 2
        assert bool(mask.any()), "the fallback left the slots unconditioned"


def test_a_release_without_groups_degrades_to_the_old_rule_rather_than_stalling():
    """No retrieval_groups.json -> every group is None.

    That must not make the rule believe the whole library is one recording and
    fall through to an empty candidate list.  Both slots still come out.
    """
    with tempfile.TemporaryDirectory() as d:
        rows = [_ramp(1.0, 0.1, 10), _ramp(1.05, 0.1, 10), _ramp(1.1, 0.1, 10)]
        library = _library(d, rows, ["recA", "recB", "recC"])
        os.remove(os.path.join(d, "train", "retrieval_groups.json"))
        library = IndexedAtomicMotionLibrary(d)
        assert set(library.retrieval_group_ids) == {None}
        plan = torch.tensor([1] * 20)
        library.build_draft(plan, feature_dim=FEATURE_DIM,
                            bar_bounds=[0, 10, 20],
                            recurrence_variety=True, seam_aware_retrieval=True)
        assert len(library.retrieval_log) == 2


def test_the_set_is_reset_between_clips():
    """Without the reset, clip 2 would be starved by clip 1's history."""
    with tempfile.TemporaryDirectory() as d:
        rows = [_ramp(1.0, 0.1, 10), _ramp(1.05, 0.1, 10)]
        library = _library(d, rows, ["recA", "recB"])
        plan = torch.tensor([1] * 20)
        for _ in range(2):
            library.build_draft(plan, feature_dim=FEATURE_DIM,
                                bar_bounds=[0, 10, 20],
                                recurrence_variety=True, seam_aware_retrieval=True)
            names = [library.retrieval_group_ids[i]
                     for i in _groups_drawn(library)[-2:]]
            assert names[0] != names[1], names
