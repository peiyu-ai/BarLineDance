"""The gate that keeps unconditioned clips out of an arm's pooled number.

WHAT IS BEING GUARDED.  ``prototype_retrieval.safe_draft_condition_fraction``
is written into every generated clip's pickle by ``infer_atomic``: the share of
the plan's atomic frames that got a source-safe retrieved prototype.  It reads
0.0 when ``_source_safe_draft`` fell closed to an all-zero draft because the
clip has no retrieval group, and those clips are the completion model running on
music alone -- not M5 output.  Two of the twenty T-line eval clips are like
that, and every table published before this module pooled them in.

WHY THESE THREE CASES.  Per CLAUDE.md 2.1 a criterion may not judge until it has
been validated in the direction it will be used:

  * it must FIRE on a real 0.0 and NAME the clip (not just count it),
  * it must RAISE when the field is absent -- a gate that cannot fire reads like
    "checked", which is the failure CLAUDE.md 2 is about, and an arm produced by
    an older code path is exactly the arm most likely to need the check,
  * POSITIVE CONTROL: on an arm where every clip reads 1.0 it must change
    NOTHING -- same clip list, and a pooled statistic identical to the
    unfiltered one.  Without that, "the filter improved the number" would be
    indistinguishable from "the filter drops clips it does not like".
"""

import pathlib
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.arm_sourcing import (ArmSourcingError, DEFAULT_THRESHOLD,  # noqa: E402
                                format_header, load_arm_sourcing, select_clips)

CLIPS = ["clipA", "clipB", "clipC"]


def write_clip(directory, clip, fraction, drop_field=False, drop_section=False):
    """One arm pickle, shaped like ``infer_atomic._write_generated_result``."""
    payload = {"full_pose": np.zeros((10, 24, 3)), "atomic_labels": np.zeros(10, int)}
    if not drop_section:
        section = {"policy": "EXCLUDE_QUERY_RETRIEVAL_GROUP_FAIL_CLOSED",
                   "query_retrieval_group_id": None if fraction == 0.0 else clip}
        if not drop_field:
            section["safe_draft_condition_fraction"] = fraction
        payload["prototype_retrieval"] = section
    path = pathlib.Path(directory) / (clip + ".pkl")
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return path


def arm(tmp_path, name, fractions, **kwargs):
    directory = tmp_path / name
    directory.mkdir()
    for clip, fraction in fractions.items():
        write_clip(directory, clip, fraction, **kwargs)
    return directory


def test_a_zero_fraction_clip_is_found_and_named(tmp_path):
    """0.0 is the fail-closed reading, and the header has to say WHICH clip."""
    directory = arm(tmp_path, "shipping", {"clipA": 1.0, "clipB": 0.0, "clipC": 1.0})
    sourcing = load_arm_sourcing(directory, CLIPS, name="shipping")
    assert sourcing.unsourced == ["clipB"]
    assert sourcing.excluded == ["clipB"]
    assert sourcing.fractions == {"clipA": 1.0, "clipB": 0.0, "clipC": 1.0}

    kept, header = select_clips(CLIPS, [("shipping", directory)])
    assert kept == ["clipA", "clipC"]
    assert header["unsourced_excluded_count"] == 1
    assert header["unsourced_excluded_clips"] == ["clipB"]
    assert "clipB" in "\n".join(format_header(header))


def test_missing_field_raises_rather_than_reading_as_a_pass(tmp_path):
    """The whole point: absent must not be indistinguishable from sourced."""
    directory = arm(tmp_path, "old_code_path", {"clipA": 1.0, "clipB": 1.0, "clipC": 1.0},
                    drop_field=True)
    with pytest.raises(ArmSourcingError) as caught:
        load_arm_sourcing(directory, CLIPS)
    assert "safe_draft_condition_fraction" in str(caught.value)


def test_missing_section_raises_and_says_ground_truth_is_not_an_arm(tmp_path):
    """A ground-truth pickle carries ``full_pose`` and nothing else."""
    directory = arm(tmp_path, "ground_truth", {"clipA": 1.0}, drop_section=True)
    with pytest.raises(ArmSourcingError) as caught:
        load_arm_sourcing(directory, ["clipA"])
    assert "prototype_retrieval" in str(caught.value)


def test_positive_control_fully_sourced_arm_changes_nothing(tmp_path):
    """KNOWN-GOOD INPUT, direction checked: every clip 1.0 => identical output.

    The pooled statistic is computed both ways from the same per-clip values and
    must be EQUAL, not merely close: a filter that quietly dropped a clip here
    would move the mean, and the whole use of this module is to trust that a
    filtered table and an unfiltered one differ only because of the excluded
    clips.
    """
    directory = arm(tmp_path, "all_sourced", {"clipA": 1.0, "clipB": 1.0, "clipC": 1.0})
    sourcing = load_arm_sourcing(directory, CLIPS)
    assert sourcing.unsourced == []
    assert sourcing.undefined == []

    kept, header = select_clips(CLIPS, [("all_sourced", directory)])
    assert kept == CLIPS
    assert header["unsourced_excluded_count"] == 0
    assert header["unsourced_detected_clips"] == []

    values = {"clipA": 0.31, "clipB": 0.47, "clipC": 0.22}
    filtered = float(np.mean([values[c] for c in kept]))
    unfiltered = float(np.mean([values[c] for c in CLIPS]))
    assert filtered == unfiltered


def test_none_fraction_is_undefined_not_a_pass(tmp_path):
    """``None`` = the plan named no atomic frame: 0/0, which is not 1.0.

    ``infer_atomic._safe_draft_condition_fraction`` returns ``None`` there on
    purpose, because returning 1.0 gave the check its best possible reading in
    the one case it exists to catch (the AIST arm where 21 of 40 plans were
    100% transition and all 40 recorded 1.0).  Comparing ``None <= 0.0`` would
    also raise TypeError in Python 3, so this is a real crash the module has to
    absorb -- ``runs/txy_t_iso_oracle150`` contains such a clip today.
    """
    directory = arm(tmp_path, "empty_plan", {"clipA": 1.0, "clipB": None, "clipC": 1.0})
    sourcing = load_arm_sourcing(directory, CLIPS)
    assert sourcing.undefined == ["clipB"]
    assert sourcing.unsourced == []
    assert sourcing.excluded == ["clipB"]
    kept, header = select_clips(CLIPS, [("empty_plan", directory)])
    assert kept == ["clipA", "clipC"]
    assert header["unsourced_detected_clips"] == ["clipB"]


def test_include_flag_keeps_them_but_still_declares_them(tmp_path):
    """Backward comparison stays possible; a silent unfiltered number does not."""
    directory = arm(tmp_path, "shipping", {"clipA": 1.0, "clipB": 0.0, "clipC": 1.0})
    kept, header = select_clips(CLIPS, [("shipping", directory)], include_unsourced=True)
    assert kept == CLIPS
    assert header["sourcing_policy"] == "INCLUDE_UNSOURCED_BACKWARD_COMPARISON"
    assert header["unsourced_excluded_count"] == 0
    # The detection is reported even when nothing is dropped, so an unfiltered
    # JSON cannot be mistaken for a clean one.
    assert header["unsourced_detected_clips"] == ["clipB"]
    assert "KEPT" in "\n".join(format_header(header))


def test_exclusion_is_the_union_over_arms(tmp_path):
    """Arms are compared to each other, so they must be scored on ONE clip set."""
    good = arm(tmp_path, "good", {"clipA": 1.0, "clipB": 1.0, "clipC": 1.0})
    bad = arm(tmp_path, "bad", {"clipA": 1.0, "clipB": 0.0, "clipC": 1.0})
    kept, header = select_clips(CLIPS, [("good", good), ("bad", bad)])
    assert kept == ["clipA", "clipC"]
    assert header["unsourced_excluded_clips"] == ["clipB"]
    assert [row["arm"] for row in header["per_arm"]] == ["good", "bad"]


def test_a_partially_sourced_clip_survives_the_default_threshold(tmp_path):
    """Default is fail-closed ONLY: 0.5 is still M5 output and is kept.

    Direction check on the threshold itself -- raising it to 0.99 must drop the
    same clip, so the knob does what its help text says.
    """
    directory = arm(tmp_path, "partial", {"clipA": 1.0, "clipB": 0.5, "clipC": 1.0})
    assert DEFAULT_THRESHOLD == 0.0
    kept, _ = select_clips(CLIPS, [("partial", directory)])
    assert kept == CLIPS
    strict, header = select_clips(CLIPS, [("partial", directory)], threshold=0.99)
    assert strict == ["clipA", "clipC"]
    assert header["sourcing_threshold"] == 0.99


def test_a_directory_with_none_of_the_clips_raises(tmp_path):
    """A gate that read nothing is not a gate that passed."""
    directory = arm(tmp_path, "elsewhere", {"clipZ": 1.0})
    with pytest.raises(ArmSourcingError) as caught:
        load_arm_sourcing(directory, CLIPS)
    assert "read nothing" in str(caught.value)


def test_missing_pickles_are_reported_not_confused_with_unsourced(tmp_path):
    """"No pickle" and "zero fraction" are different facts and print differently."""
    directory = arm(tmp_path, "partial_run", {"clipA": 1.0, "clipC": 0.0})
    sourcing = load_arm_sourcing(directory, CLIPS)
    assert sourcing.missing == ["clipB"]
    assert sourcing.unsourced == ["clipC"]
    kept, header = select_clips(CLIPS, [("partial_run", directory)])
    assert kept == ["clipA", "clipB"]
    assert header["per_arm"][0]["missing_pickles"] == ["clipB"]
    assert "no pickle" in "\n".join(format_header(header))
