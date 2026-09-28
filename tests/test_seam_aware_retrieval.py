"""``--draft-seam-aware-retrieval``: pick the prototype that CONTINUES the last one.

WHY.  The operator, 2026-09-05, after the stretch and repeat defects were
fixed: "motion 之前选择不是很连贯,会给拼接造成难度 ... draft 选 motion 时看前面一
小节的动作".  The measurement that agrees: seam jerk 0.3515 against ground
truth's 0.2553 at the same frame indices, 17 eval clips, filler frames excluded,
boundaries read from the artifact's own slot_start.

THE CRITERION IS THIS REPOSITORY'S OWN, not the paper's and not fitted, so per
CLAUDE.md 2.1 it must be shown to work on a case whose answer is already known
BEFORE it is allowed to choose anything.  That is what
``test_positive_control_*`` below is: a library in which exactly one candidate
is the true continuation, and the rule must find it.

WHAT IT MAY NOT CHANGE.  It picks from the SAME duration band the plain rule
uses -- max(2, 0.15 * target_length) -- so the length guarantee established on
2026-09-05 (units_over_library_ceiling == 0, stretch median 1.000) is untouched.
The band test below is the one that would catch a version that widened it to
find a better join.
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


def _library(tmpdir, rows):
    """``rows`` are [frames, FEATURE_DIM] arrays, each one prototype of class 1."""
    train = os.path.join(tmpdir, "train")
    os.makedirs(train)
    width = max(len(r) for r in rows)
    motion = np.zeros((len(rows), width, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros((len(rows), width), dtype=np.int64)
    for i, row in enumerate(rows):
        motion[i, :len(row)] = row
        labels[i, :len(row)] = 1
    np.save(os.path.join(train, "motion.npy"), motion)
    np.save(os.path.join(train, "labels.npy"), labels)
    with open(os.path.join(train, "names.json"), "w") as handle:
        json.dump(["src_{}_slice0".format(i) for i in range(len(rows))], handle)
    # _join_cost unnormalizes, so the library needs its normalizer.  Identity
    # (min -1, max 1 gives scale 1, offset 0) so the fixture's numbers ARE the
    # raw ones and a failure cannot be an artefact of the scaling.
    torch.save({"data_min": torch.full((FEATURE_DIM,), -1.0),
                "data_max": torch.full((FEATURE_DIM,), 1.0)},
               os.path.join(tmpdir, "normalizer.pt"))
    return IndexedAtomicMotionLibrary(tmpdir)


def _ramp(start, step, n):
    """A prototype travelling at constant velocity from a known pose."""
    row = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    for k in range(n):
        row[k, CONTACT_CHANNELS:] = start + step * k
    return row


def test_positive_control_the_true_continuation_is_found():
    """Three candidates of equal length; exactly one continues the tail.

    Without this the rule could be a constant "take the first" and every other
    test here would still pass.
    """
    with tempfile.TemporaryDirectory() as d:
        # tail ends at pose 1.0 travelling +0.1 per frame
        good = _ramp(1.1, 0.1, 10)        # continues pose AND velocity
        wrong_pose = _ramp(5.0, 0.1, 10)  # right velocity, far pose
        wrong_vel = _ramp(1.1, -0.4, 10)  # right pose, opposite travel
        library = _library(d, [wrong_pose, wrong_vel, good])
        tail = torch.zeros(2, FEATURE_DIM)
        tail[0, CONTACT_CHANNELS:] = 0.9
        tail[1, CONTACT_CHANNELS:] = 1.0
        costs = [library._join_cost(c, tail) for c in library.index[1]]
        assert int(np.argmin(costs)) == 2, costs


def test_the_cost_separates_pose_error_from_velocity_error():
    """Both terms must be live; a rule using only pose would tie the last two."""
    with tempfile.TemporaryDirectory() as d:
        library = _library(d, [_ramp(1.1, 0.1, 10)])
        tail = torch.zeros(2, FEATURE_DIM)
        tail[0, CONTACT_CHANNELS:] = 0.9
        tail[1, CONTACT_CHANNELS:] = 1.0
        same = library._join_cost(library.index[1][0], tail)
        # a tail with the same END pose but standing still
        still = tail.clone(); still[0, CONTACT_CHANNELS:] = 1.0
        assert library._join_cost(library.index[1][0], still) > same


def test_contacts_are_ignored_because_they_are_supposed_to_flip():
    """A candidate that swaps support must not be penalised for it.

    The four contact channels are binary and flip on 11.61% of adjacent
    training frames -- that IS the weight transfer the operator says ground
    truth is full of.  Scoring them would rank a dancer who never changes feet
    above one who does.
    """
    with tempfile.TemporaryDirectory() as d:
        row = _ramp(1.1, 0.1, 10)
        flipped = row.copy(); flipped[:, :CONTACT_CHANNELS] = 1.0
        library = _library(d, [row, flipped])
        tail = torch.zeros(2, FEATURE_DIM)
        tail[0, CONTACT_CHANNELS:] = 0.9
        tail[1, CONTACT_CHANNELS:] = 1.0
        a, b = (library._join_cost(c, tail) for c in library.index[1])
        assert a == b


def test_it_picks_from_the_duration_band_and_does_not_widen_it():
    """The length guarantee must survive: a great join that is the wrong length
    may not be taken.

    target 10, band is max(2, 1.5) = 2 frames, so the 20-frame candidate is out
    of the band no matter how perfectly it continues the tail.
    """
    with tempfile.TemporaryDirectory() as d:
        in_band = _ramp(9.0, 0.1, 10)          # length 10, a poor join
        out_of_band = _ramp(1.1, 0.1, 20)      # length 20, a perfect join
        library = _library(d, [in_band, out_of_band])
        tail = torch.zeros(2, FEATURE_DIM)
        tail[0, CONTACT_CHANNELS:] = 0.9
        tail[1, CONTACT_CHANNELS:] = 1.0
        values = library.retrieve(1, 10, join_tail=tail)
        record = library.retrieval_log[-1]
        assert record["native_frames"] == 10
        assert record["stretch"] == 1.0


def test_off_by_default_reproduces_the_plain_rule():
    """join_top_k = 1 here so the assertion is about the RULE, not the draw.

    With the shipping top-k of 4 and a two-candidate fixture the draw can land
    on the plain rule's own pick, and this test would fail for a reason that has
    nothing to do with what it is checking.
    """
    with tempfile.TemporaryDirectory() as d:
        library = _library(d, [_ramp(9.0, 0.1, 10), _ramp(1.1, 0.1, 10)])
        library.join_top_k = 1
        tail = torch.zeros(2, FEATURE_DIM)
        tail[0, CONTACT_CHANNELS:] = 0.9
        tail[1, CONTACT_CHANNELS:] = 1.0
        plain = library.retrieve(1, 10)
        library.retrieval_log = []
        library._retrieval_cache = {}
        aware = library.retrieve(1, 10, join_tail=tail)
        assert not torch.equal(plain, aware), "the switch changed nothing"


def test_a_repeat_still_gets_a_different_prototype():
    """Variety and continuity must be bought together, not traded.

    With the seam-aware rule the second occurrence takes the best-joining
    candidate that is NOT the first occurrence's pick, so the repeat defect
    (27.1% of shipping units replayed one) does not come back through this door.
    """
    with tempfile.TemporaryDirectory() as d:
        library = _library(d, [_ramp(1.1, 0.1, 10), _ramp(1.2, 0.1, 10),
                               _ramp(9.0, 0.1, 10)])
        library.join_top_k = 1
        tail = torch.zeros(2, FEATURE_DIM)
        tail[0, CONTACT_CHANNELS:] = 0.9
        tail[1, CONTACT_CHANNELS:] = 1.0
        first = library.retrieve(1, 10, join_tail=tail, occurrence=0)
        library._retrieval_cache = {}
        second = library.retrieve(1, 10, join_tail=tail, occurrence=1)
        assert not torch.equal(first, second)


def test_the_draw_is_over_the_top_k_and_not_argmax():
    """The documented remedy for the failure --retrieval-rule phase hit.

    A hard sort narrows the pool to one answer per query and squeezes out
    variety: measured 2026-09-05 with argmax, adjacent-over-all-pairs pose
    distance fell to 0.957 against ground truth's 0.992, undoing part of what
    the recurrence fix had just bought.  So the rule draws from the best few.
    Five near-equal joins and many draws must not all return the same one.
    """
    with tempfile.TemporaryDirectory() as d:
        rows = [_ramp(1.1 + 0.001 * k, 0.1, 10) for k in range(5)]
        library = _library(d, rows)
        assert library.join_top_k > 1, "the shipping default must not be argmax"
        tail = torch.zeros(2, FEATURE_DIM)
        tail[0, CONTACT_CHANNELS:] = 0.9
        tail[1, CONTACT_CHANNELS:] = 1.0
        seen = set()
        for seed in range(24):
            library._retrieval_cache = {}
            values = library.retrieve(
                1, 10, join_tail=tail,
                variety_rng=np.random.default_rng(seed))
            seen.add(tuple(library.retrieval_log[-1]["source"]))
        assert len(seen) > 1, "the draw collapsed to argmax"
        assert len(seen) <= library.join_top_k, seen


def test_top_k_still_never_leaves_the_duration_band():
    """The draw widens WHICH join is taken, never which lengths are allowed."""
    with tempfile.TemporaryDirectory() as d:
        library = _library(d, [_ramp(9.0, 0.1, 10), _ramp(1.1, 0.1, 20)])
        tail = torch.zeros(2, FEATURE_DIM)
        tail[0, CONTACT_CHANNELS:] = 0.9
        tail[1, CONTACT_CHANNELS:] = 1.0
        for seed in range(12):
            library._retrieval_cache = {}
            library.retrieve(1, 10, join_tail=tail,
                             variety_rng=np.random.default_rng(seed))
            assert library.retrieval_log[-1]["native_frames"] == 10


# ---------------------------------------------------------------------------
# The variety draw must exclude EVERYTHING the clip has already played, not
# only the first occurrence's pick.


def test_the_draw_never_replays_used_prototypes_with_seam_awareness_on():
    """The SEAM-AWARE branch, which is the one production takes.

    The plain-variety branch was fixed first and the run came back unchanged to
    the byte, because --draft-seam-aware-retrieval routes through a different
    branch that only excluded the first occurrence's pick.
    """
    with tempfile.TemporaryDirectory() as d:
        rows = [_ramp(1.0 + 0.01 * k, 0.1, 10) for k in range(5)]
        library = _library(d, rows)
        tail = torch.zeros(2, FEATURE_DIM)
        tail[0, CONTACT_CHANNELS:] = 0.9
        tail[1, CONTACT_CHANNELS:] = 1.0
        seen = []
        for occurrence in range(4):
            library._retrieval_cache = {}
            library.retrieve(1, 10, join_tail=tail, occurrence=occurrence,
                             variety_rng=np.random.default_rng(7))
            seen.append(tuple(library.retrieval_log[-1]["source"]))
        assert len(set(seen)) == len(seen), seen


def test_the_draw_never_replays_anything_used_earlier_in_the_clip():
    """Measured defect, 2026-09-06, wild_v5:7608191311518369137:clip000.

    Its plan names two classes over 20 s, so one class is retrieved six times.
    Excluding only occurrence 0's pick left occurrence 4 free to draw
    occurrence 1's -- ``(2972, 0, 67)`` twice -- and occurrence 5 to draw
    occurrence 2's.  Two of ten units were byte-identical replays: 20% reuse,
    the highest of the twenty eval clips, and the one the operator singled out
    by eye as 动作段落高度重复.
    """
    with tempfile.TemporaryDirectory() as d:
        rows = [_ramp(1.0 + 0.01 * k, 0.1, 10) for k in range(5)]
        library = _library(d, rows)
        seen = []
        for occurrence in range(4):
            library._retrieval_cache = {}
            library.retrieve(1, 10, occurrence=occurrence,
                             variety_rng=np.random.default_rng(7))
            seen.append(tuple(library.retrieval_log[-1]["source"]))
        assert len(set(seen)) == len(seen), seen


def test_it_falls_back_rather_than_failing_when_the_band_runs_out():
    """A repeat beats an empty slot: with two candidates and four draws the
    rule must still return something."""
    with tempfile.TemporaryDirectory() as d:
        library = _library(d, [_ramp(1.0, 0.1, 10), _ramp(1.01, 0.1, 10)])
        for occurrence in range(4):
            library._retrieval_cache = {}
            values = library.retrieve(1, 10, occurrence=occurrence,
                                      variety_rng=np.random.default_rng(3))
            assert values.shape[0] == 10


def test_the_used_set_is_per_clip():
    """build_draft resets it; two clips must not starve each other."""
    with tempfile.TemporaryDirectory() as d:
        library = _library(d, [_ramp(1.0 + 0.01 * k, 0.1, 10) for k in range(3)])
        library.retrieve(1, 10, occurrence=1, variety_rng=np.random.default_rng(1))
        assert library._used_this_clip
        plan = torch.ones(10, dtype=torch.long)
        library.build_draft(plan, FEATURE_DIM)
        assert len(library._used_this_clip) <= 1
