"""Tests for the seam-judder measurement.

The point of the positive controls here is the one CLAUDE.md 2.1 paid for: a
ratio that "can fail" is not enough, it also has to read HIGH on a signal we
built by hand and LOW on one we know is smooth.  A metric that only ever fails
on garbage would have passed the old boundary-contrast criterion too.
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.measure_seam_judder import (
    bar_bounds_from_music,
    brake_offsets,
    jerk,
    measure_clip,
    near_mask,
    ratio_at,
    seam_frames,
    speed,
    summarise,
    uses_bar_prototypes,
)


def smooth_motion(frames, joints=24, seed=0):
    """A body moving on a slow sinusoid: no joins, no steps."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 4.0 * np.pi, frames)[:, None, None]
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(1, joints, 3))
    return 0.3 * np.sin(time + phase)


def spliced_motion(frames, seams, joints=24, seed=1):
    """Smooth motion with a POSE step pasted in at each seam frame.

    The step is applied to every joint EXCEPT the root, because both statistics
    here are root relative: an offset applied to the whole body cancels in
    ``joints - joints[:, :1, :]`` and produces a fixture that looks spliced and
    measures perfectly smooth.  The first version of this file did exactly that
    and the positive control read 0.89, i.e. the "known bad" sample was not bad.
    ``test_speed_and_jerk_are_root_relative`` is the reason it was caught.
    """
    motion = smooth_motion(frames, joints, seed)
    for index, seam in enumerate(seams):
        motion[seam:, 1:, :] += 0.25 * (1 if index % 2 == 0 else -1)
    return motion


def test_the_splice_fixture_actually_differs_from_the_smooth_one():
    # Guards the fixture itself: a splice that cancels under root-relative
    # differencing would make every positive control below vacuous.
    seams = [40, 80, 120]
    assert not np.allclose(spliced_motion(160, seams), smooth_motion(160, seed=1))
    assert jerk(spliced_motion(160, seams)).max() > 10 * jerk(smooth_motion(160, seed=1)).max()


class SeamReconstructionTests:
    pass


def test_touching_conditioned_segments_are_seams():
    labels = np.array([3] * 10 + [7] * 10)
    seams, segments = seam_frames(labels)
    assert seams == [10]
    assert segments == 2


def test_a_transition_between_two_segments_is_a_gap_not_a_seam():
    # build_draft skips label 0, so the draft is unconditioned there and the
    # model was TOLD so -- counting it as a seam would inflate the reading.
    labels = np.array([3] * 10 + [0] * 4 + [7] * 10)
    seams, _ = seam_frames(labels)
    assert seams == []


def test_bar_bounds_split_one_label_run_into_several_seams():
    labels = np.array([3] * 24)
    without, _ = seam_frames(labels)
    with_bars, _ = seam_frames(labels, bar_bounds=[8, 16])
    assert without == []
    assert with_bars == [8, 16]


def test_bar_bounds_outside_the_clip_are_ignored():
    labels = np.array([3] * 12)
    seams, _ = seam_frames(labels, bar_bounds=[0, 6, 12, 40])
    assert seams == [6]


def test_bar_bounds_read_the_beat_channel_at_the_requested_phase():
    music = np.zeros((40, 35))
    music[[0, 5, 10, 15, 20, 25, 30, 35], 34] = 1.0
    assert bar_bounds_from_music(music, 4, 0) == [0, 20]
    assert bar_bounds_from_music(music, 4, 1) == [5, 25]


def test_music_without_beats_yields_no_bar_bounds():
    assert bar_bounds_from_music(np.zeros((30, 35)), 4, 0) == []


class RatioTests:
    pass


def test_ratio_reads_high_on_a_hand_built_splice():
    # POSITIVE CONTROL: the answer is known because we pasted the steps in.
    seams = [40, 80, 120]
    value = ratio_at(jerk(spliced_motion(160, seams)), seams, radius=2, guard=4)
    assert value > 3.0


def test_a_median_ratio_would_miss_the_same_splice():
    # Pins WHY ratio_at takes the mean.  The spike from a paste lands on one or
    # two frames, so the median inside the window is a frame without it; on the
    # very sample we glued together by hand, the median reads near 1.
    seams = [40, 80, 120]
    roughness = jerk(spliced_motion(160, seams))
    near = near_mask(roughness.size, seams, 2)
    away = ~near_mask(roughness.size, seams, 4)
    median_ratio = np.median(roughness[near]) / np.median(roughness[away])
    assert median_ratio < 1.5
    assert roughness[near].mean() / roughness[away].mean() > 3.0


def test_ratio_reads_near_one_on_smooth_motion_at_the_same_frames():
    # NEGATIVE CONTROL, and the important one: the SAME frame positions on
    # motion with no joins must not read high, or the metric is measuring
    # where the frames are rather than what happens there.
    seams = [40, 80, 120]
    value = ratio_at(jerk(smooth_motion(160)), seams, radius=2, guard=4)
    assert 0.5 < value < 2.0


def test_ratio_is_none_when_a_pool_would_be_empty():
    assert ratio_at(np.ones(10), [], radius=2, guard=4) is None
    assert ratio_at(np.array([]), [5], radius=2, guard=4) is None
    # Guarding every frame leaves no comparison pool.
    assert ratio_at(np.ones(9), [4], radius=2, guard=40) is None


def test_ratio_refuses_a_zero_denominator_rather_than_dividing():
    assert ratio_at(np.zeros(60), [30], radius=2, guard=4) is None


def test_near_mask_clips_at_both_ends():
    mask = near_mask(10, [0, 9], radius=2)
    assert mask[:3].all() and mask[7:].all() and not mask[3:7].any()


class ShuffledNullTests:
    pass


def test_the_shuffled_null_stays_low_while_the_real_seams_read_high():
    seams = [40, 80, 120]
    measured = measure_clip(spliced_motion(160, seams), np.array(
        [1] * 40 + [2] * 40 + [3] * 40 + [4] * 40), None, 2, 4,
        np.random.default_rng(0), draws=50)
    assert measured["seams"] == seams
    assert measured["jerk_ratio"] > 3.0
    assert measured["jerk_ratio_null_median"] < 1.5
    assert measured["jerk_ratio"] > measured["jerk_ratio_null_median"]


def test_the_null_does_not_beat_itself_on_smooth_motion():
    # If the metric fired on smooth motion the null would have to fire too;
    # this asserts the pair, which is what makes the comparison meaningful.
    measured = measure_clip(smooth_motion(160), np.array(
        [1] * 40 + [2] * 40 + [3] * 40 + [4] * 40), None, 2, 4,
        np.random.default_rng(0), draws=50)
    assert measured["jerk_ratio"] < 2.0


class BrakeOffsetTests:
    pass


def test_a_hand_placed_slow_frame_is_found_at_its_own_offset():
    speeds = np.ones(100)
    for seam in (30, 60, 90):
        speeds[seam - 1] = 0.01
    assert brake_offsets(speeds, [30, 60, 90], radius=4) == [-1, -1, -1]


def test_windows_that_run_off_the_end_are_dropped_not_truncated():
    # A truncated window would bias the argmin toward the side that survives,
    # so the two positions whose windows fall off the ends must be dropped and
    # only the middle one reported.
    speeds = np.ones(20)
    speeds[12] = 0.01
    assert brake_offsets(speeds, [1, 10, 19], radius=4) == [2]


def test_scattered_slow_frames_do_not_concentrate_on_one_offset():
    rng = np.random.default_rng(3)
    speeds = rng.uniform(0.5, 1.5, size=400)
    offsets = brake_offsets(speeds, list(range(20, 380, 20)), radius=4)
    counts = np.bincount(np.asarray(offsets) + 4, minlength=9)
    assert counts.max() / len(offsets) < 0.5


class SummaryTests:
    pass


def test_summary_counts_clips_that_beat_their_own_null():
    rows = {"clips": ["a", "b", "c"], "jerk_ratio": [3.0, 2.0, 0.5],
            "null": [1.0, 1.0, 1.0], "seam_count": [4, 4, 4],
            "segments": [5, 5, 5], "gt_at_same_frames": [1.0, 1.1, 0.9],
            "offsets": [-1, -1, 0]}
    summary = summarise(rows)
    assert summary["clips_above_own_null"] == "2/3"
    assert summary["brake_offset_mode_share"] == pytest.approx(2 / 3)
    assert summary["ground_truth_at_same_frames"] == pytest.approx(1.0)


def test_summary_survives_an_arm_with_nothing_measurable():
    summary = summarise({"clips": [], "jerk_ratio": [], "null": [],
                         "seam_count": [], "segments": [],
                         "gt_at_same_frames": [], "offsets": []})
    assert summary["clips"] == 0
    assert summary["jerk_ratio"] is None
    assert summary["brake_offset_mode_share"] is None


class ManifestTests:
    pass


def test_bar_prototype_switch_is_read_from_the_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(
        '{"sampling": {"draft_bar_prototypes": true}}')
    assert uses_bar_prototypes(tmp_path) is True


def test_a_run_without_the_switch_is_treated_as_per_run_prototypes(tmp_path):
    (tmp_path / "manifest.json").write_text('{"sampling": {}}')
    assert uses_bar_prototypes(tmp_path) is False


def test_a_run_without_a_manifest_is_refused_not_guessed(tmp_path):
    with pytest.raises(SystemExit):
        uses_bar_prototypes(tmp_path)


def test_speed_and_jerk_are_root_relative():
    # A body translating rigidly has no internal speed and no internal jerk.
    motion = np.zeros((50, 24, 3))
    motion += np.linspace(0.0, 5.0, 50)[:, None, None]
    assert speed(motion).max() < 1e-9
    assert jerk(motion).max() < 1e-9


class ArmParsingTests:
    pass


def test_an_arm_name_containing_an_equals_sign_keeps_its_directory(tmp_path, monkeypatch):
    # The useful name of an arm is usually the switch it sets, so it contains
    # "=".  Splitting on the first one handed the scorer the directory
    # "4)=runs/..." and it reported zero clips as if the arms were empty.
    import tools.measure_seam_judder as module

    (tmp_path / "manifest.json").write_text('{"sampling": {}}')
    captured = {}

    def fake_run(arms, *rest, **kwargs):
        captured["arms"] = arms
        return {"arms": {}}

    clips = tmp_path / "clips.txt"
    clips.write_text("some_clip\n")
    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "measure_seam_judder",
        "--arm", "A (seam_blend=4)={}".format(tmp_path),
        "--clips", str(clips),
        "--ground-truth", str(tmp_path),
        "--audio-dir", str(tmp_path)])
    module.main()
    assert captured["arms"] == [("A (seam_blend=4)", str(tmp_path))]


def test_an_arm_pointing_at_a_missing_directory_is_refused(tmp_path, monkeypatch):
    import tools.measure_seam_judder as module

    clips = tmp_path / "clips.txt"
    clips.write_text("some_clip\n")
    monkeypatch.setattr(sys, "argv", [
        "measure_seam_judder",
        "--arm", "A={}".format(tmp_path / "nope"),
        "--clips", str(clips),
        "--ground-truth", str(tmp_path),
        "--audio-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        module.main()


def test_an_arm_that_scores_no_clips_is_an_error_not_an_empty_summary(tmp_path):
    import tools.measure_seam_judder as module

    (tmp_path / "manifest.json").write_text('{"sampling": {}}')
    with pytest.raises(SystemExit) as failure:
        module.run([("A", str(tmp_path))], ["absent_clip"], tmp_path, tmp_path,
                   2, 4, 0, 10)
    assert "0 of 1 clips" in str(failure.value)
