"""``tools/score_bar_plan.py``: does the scorer read what it claims to read?

THE POINT OF THIS FILE.  The bar-resolution planner experiment is decided by one
number -- how often a plan changes label from one bar to the next, against
ground truth's 78.4% -- so before that number judges anything, the scorer has to
be shown to read it.  CLAUDE.md 2.1 asks for four things and this file supplies
three of them (the fourth, provenance, is in the tool's docstring):

* a POSITIVE control whose known-good answer is checked in the right direction:
  an artifact whose per-frame labels ARE ground truth's bar labels must score
  exactly the ground-truth row, 78.4% on the corpus and 80.9% on the 18 eval
  clips.  A scorer that could not see a win would return something else here;
* a NEGATIVE control: one label held for the whole clip reads 0% change and one
  run, not a low-but-plausible number;
* the two-sided reading: a plan that changes on EVERY bar line reads 100% and is
  reported as +19 points of error, not as a better score.
"""

import json
import os
import pathlib
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import score_bar_plan  # noqa: E402

BAR_RELEASE = pathlib.Path("/cache/atomicdance-assets/scratch/txy_t/release_bar_v1")
SEGMENTATION = pathlib.Path(
    "/cache/atomicdance-assets/runs/txy_t_seg_beat4/segmentation.json")
CLIPS18 = pathlib.Path(
    os.environ.get("CLIPS18_LIST", "/nonexistent/clips18_fullname.txt"))

REAL = BAR_RELEASE.exists() and SEGMENTATION.exists()
needs_real = pytest.mark.skipif(not REAL, reason="the T-line release is not on this machine")


# --------------------------------------------------------------- fixtures ---

def write_artifact(directory, clip, labels, retrieval_stretch=None):
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"atomic_labels": np.asarray(labels, dtype=np.int64)}
    if retrieval_stretch is not None:
        payload["prototype_retrieval"] = {"retrieval_stretch": retrieval_stretch}
    with open(str(directory / (clip + ".pkl")), "wb") as handle:
        pickle.dump(payload, handle)


def fake_segmentation(tmp_path, sequences):
    """``segmentation.json``'s shape, with ``__`` ids like the real one."""
    records = []
    for name, spans in sequences.items():
        records.append({
            "sequence": name,
            "boundaries": [spans[0][0]] + [end for _, end in spans],
            "segments": [{"start": s, "end": e, "frames": e - s} for s, e in spans],
        })
    path = tmp_path / "segmentation.json"
    path.write_text(json.dumps({"records": records}))
    return path


def bars(count, frames=48, offset=0):
    return [(offset + i * frames, offset + (i + 1) * frames) for i in range(count)]


# ------------------------------------------------- synthetic controls --------

def test_positive_control_a_plan_that_changes_every_bar_reads_one_hundred(tmp_path):
    spans = bars(6)
    segmentation = fake_segmentation(tmp_path, {"up__clip000": spans})
    grid = score_bar_plan.load_bar_grid(segmentation)
    labels = np.concatenate([np.full(48, i + 1) for i in range(6)])
    write_artifact(tmp_path / "arm", "wild_v5:up:clip000", labels)
    stats = score_bar_plan.score_arm(tmp_path / "arm", ["wild_v5:up:clip000"], grid)
    assert stats["change_rate"] == 1.0
    assert stats["runs"] == 6 and stats["runs_ge_3"] == 0.0


def test_negative_control_one_held_label_reads_zero_change(tmp_path):
    spans = bars(6)
    segmentation = fake_segmentation(tmp_path, {"up__clip000": spans})
    grid = score_bar_plan.load_bar_grid(segmentation)
    write_artifact(tmp_path / "arm", "wild_v5:up:clip000", np.full(288, 7))
    stats = score_bar_plan.score_arm(tmp_path / "arm", ["wild_v5:up:clip000"], grid)
    assert stats["change_rate"] == 0.0
    assert stats["runs"] == 1 and stats["runs_ge_5"] == 1.0
    assert stats["top_class_share"] == 1.0 and stats["distinct_classes"] == 1


def test_the_criterion_is_two_sided_not_higher_is_better():
    reference = {"change_rate": 0.809, "runs_ge_3": 0.016, "runs_ge_5": 0.0,
                 "filler_bar_share": 0.215}
    too_sticky = {"change_rate": 0.588, "runs_ge_3": 0.126, "runs_ge_5": 0.032,
                  "filler_bar_share": 0.262}
    too_busy = {"change_rate": 1.0, "runs_ge_3": 0.0, "runs_ge_5": 0.0,
                "filler_bar_share": 0.0}
    sticky = score_bar_plan.compare(too_sticky, reference)
    busy = score_bar_plan.compare(too_busy, reference)
    assert sticky["change_rate_error"] < 0 and busy["change_rate_error"] > 0
    # A plan that changes on every bar line is 19 points WRONG, not 19 better.
    assert busy["change_rate_abs_error"] == pytest.approx(0.191)


def test_out_of_grid_frames_are_not_bars_but_do_count_as_filler_frames(tmp_path):
    """The two filler columns must not be the same number, and here is why."""
    spans = bars(4, offset=60)
    segmentation = fake_segmentation(tmp_path, {"up__clip000": spans})
    grid = score_bar_plan.load_bar_grid(segmentation)
    labels = np.concatenate([np.zeros(60, dtype=int),
                             np.concatenate([np.full(48, i + 1) for i in range(4)])])
    write_artifact(tmp_path / "arm", "wild_v5:up:clip000", labels)
    stats = score_bar_plan.score_arm(tmp_path / "arm", ["wild_v5:up:clip000"], grid)
    assert stats["filler_bar_share"] == 0.0
    assert stats["filler_frame_share"] == pytest.approx(60 / 252)


def test_a_bar_running_past_the_plan_is_truncated_and_counted(tmp_path):
    spans = bars(4)
    segmentation = fake_segmentation(tmp_path, {"up__clip000": spans})
    grid = score_bar_plan.load_bar_grid(segmentation)
    write_artifact(tmp_path / "arm", "wild_v5:up:clip000",
                   np.concatenate([np.full(48, i + 1) for i in range(3)] + [np.full(20, 4)]))
    stats = score_bar_plan.score_arm(tmp_path / "arm", ["wild_v5:up:clip000"], grid)
    assert stats["bars"] == 4 and stats["bars_truncated_by_plan_end"] == 1


def test_the_retrieval_gate_is_carried_through(tmp_path):
    spans = bars(3)
    segmentation = fake_segmentation(tmp_path, {"up__clip000": spans})
    grid = score_bar_plan.load_bar_grid(segmentation)
    write_artifact(tmp_path / "arm", "wild_v5:up:clip000", np.full(144, 5),
                   retrieval_stretch={"units_over_library_ceiling": 3, "playback_min": 0.48})
    stats = score_bar_plan.score_arm(tmp_path / "arm", ["wild_v5:up:clip000"], grid)
    assert stats["unfillable_retrieval_units"] == 3
    assert stats["worst_playback_speed"] == 0.48


def test_a_missing_artifact_refuses_instead_of_scoring_the_rest(tmp_path):
    spans = bars(3)
    segmentation = fake_segmentation(tmp_path, {"up__clip000": spans, "vp__clip000": spans})
    grid = score_bar_plan.load_bar_grid(segmentation)
    write_artifact(tmp_path / "arm", "wild_v5:up:clip000", np.full(144, 5))
    with pytest.raises(FileNotFoundError):
        score_bar_plan.score_arm(tmp_path / "arm",
                                 ["wild_v5:up:clip000", "wild_v5:vp:clip000"], grid)


def test_a_clip_with_no_bar_grid_refuses(tmp_path):
    spans = bars(3)
    segmentation = fake_segmentation(tmp_path, {"up__clip000": spans})
    grid = score_bar_plan.load_bar_grid(segmentation)
    write_artifact(tmp_path / "arm", "wild_v5:zz:clip000", np.full(144, 5))
    with pytest.raises(KeyError):
        score_bar_plan.score_arm(tmp_path / "arm", ["wild_v5:zz:clip000"], grid)


def test_a_release_cut_on_other_bars_than_the_grid_refuses():
    with pytest.raises(ValueError, match="cut .* differently"):
        score_bar_plan.assert_same_grid([(0, 48)], [(0, 47), (47, 95)], "wild_v5:up:clip000")


# --------------------------------------------------- the real corpus ---------

@needs_real
def test_the_ground_truth_row_reproduces_the_published_reading():
    """PROOF OF POWER: the scorer reads 78.4% on the corpus it was published on."""
    grid = score_bar_plan.load_bar_grid(SEGMENTATION)
    labels, spans = score_bar_plan.ground_truth_bars(BAR_RELEASE)
    for recording, recording_spans in spans.items():
        score_bar_plan.assert_same_grid(recording_spans, grid[recording], recording)
    from tools.build_bar_planner_release import bar_shape_stats

    stats = bar_shape_stats(list(labels.values()))
    # 258, not the published 270: ``bars.jsonl`` only lists bars that fall in a
    # 4-bar window, and 12 recordings hold exactly 3 bars.  Those 12 carry 24 of
    # the corpus's 1,568 adjacent bar pairs, and dropping them moves the reading
    # by 0.05 points -- 78.50% here against the 78.44% the release's own report
    # computes over all 270.  Asserted rather than glossed: if the two ever
    # diverged, the reference row would silently be a different corpus.
    assert stats["recordings"] == 258
    assert stats["change_rate"] == pytest.approx(0.785, abs=0.002)
    assert stats["runs_ge_3"] == pytest.approx(0.037, abs=0.002)
    assert stats["runs_ge_5"] == pytest.approx(0.005, abs=0.002)


@needs_real
@pytest.mark.skipif(not CLIPS18.exists(), reason="the 18-clip eval list is not here")
def test_positive_control_on_real_data_a_perfect_plan_scores_the_reference_row(tmp_path):
    """An arm whose frames carry ground truth's OWN bar labels must score 80.9%.

    This is the control that makes a null reportable: if a trained bar planner
    reads 60%, this test is the evidence that the scorer would have read the
    ground-truth row had the model produced it.
    """
    clips = [line.strip() for line in open(CLIPS18) if line.strip()]
    grid = score_bar_plan.load_bar_grid(SEGMENTATION)
    labels, spans = score_bar_plan.ground_truth_bars(BAR_RELEASE)
    arm = tmp_path / "oracle"
    for clip in clips:
        length = max(end for _, end in grid[clip])
        track = np.zeros(length, dtype=np.int64)
        for label, (start, end) in zip(labels[clip], spans[clip]):
            track[start:end] = label
        write_artifact(arm, clip, track)
    scored = score_bar_plan.score_arm(arm, clips, grid)
    reference = score_bar_plan.score_ground_truth(BAR_RELEASE, clips, grid)
    assert scored["change_rate"] == pytest.approx(reference["change_rate"])
    assert scored["change_rate"] == pytest.approx(0.809, abs=0.002)
    assert scored["filler_bar_share"] == pytest.approx(reference["filler_bar_share"])
    assert score_bar_plan.compare(scored, reference)["change_rate_abs_error"] == 0.0
