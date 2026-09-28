"""Unit tests for tools/build_rewindowed_planner_release.py.

THE POSITIVE CONTROL is a synthetic corpus small enough to rewindow by hand:
three recordings of 10, 8 and 6 frames, cut into 6-frame source windows at
stride 2 (so they overlap exactly as ``release_v3``'s do), with music set to
``music[t, c] = 100 * t + c`` so that every frame is distinguishable from every
other frame and a one-frame misalignment cannot hide inside a plausible number.
The expected 4-frame windows are written out literally rather than recomputed
with the tool's own ``window_starts``, because a control that shares the code it
is controlling checks nothing (CLAUDE.md section 2.1).

THE BAR METRIC IS CONTROLLED IN BOTH DIRECTIONS: one hand-built recording whose
bars must read 100% change and one whose bars must read 0%.  A shape metric that
only ever reports "high" would pass a one-sided check and still be useless.

THE NEGATIVE CONTROLS are the other half: overlapping source windows that
disagree must raise instead of letting one silently win, a corrupted written
window must be caught by the identity check with the exact magnitude it was
corrupted by, and a recording moved between splits must raise.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_rewindowed_planner_release import (  # noqa: E402
    MOTION_DIM,
    MUSIC_DIM,
    bar_labels_of,
    bar_shape_stats,
    build,
    coverage_of,
    read_root_tracks,
    recording_id_from_segmentation,
    verify_identity,
    verify_splits,
    window_name,
    window_starts,
)

SOURCE_WINDOW = 6
SOURCE_STRIDE = 2

# Hand-chosen so the bars below are never mixed and the readings are known:
#   A -> bars [1, 2, 3, 0, 1]  ->  4 changes in 4 pairs  = 100%
#   B -> bars [5, 5, 5, 5]     ->  0 changes in 3 pairs  =   0%
TRACKS = {
    "wild_v5:1001:clip000": [1, 1, 2, 2, 3, 3, 0, 0, 1, 1],
    "wild_v5:1002:clip000": [5, 5, 5, 5, 5, 5, 5, 5],
    "wild_v5:1003:clip000": [7, 7, 4, 4, 4, 4],
}
SPLIT_OF = {
    "wild_v5:1001:clip000": "train",
    "wild_v5:1002:clip000": "val",
    "wild_v5:1003:clip000": "test",
}


def music_track(frames):
    """``music[t, c] = 100 * t + c`` -- every frame distinguishable by eye."""
    return (100.0 * np.arange(frames)[:, None] + np.arange(MUSIC_DIM)[None, :]).astype(np.float32)


def motion_track(frames):
    return (1000.0 * np.arange(frames)[:, None] + np.arange(MOTION_DIM)[None, :]).astype(np.float32)


def make_source_release(root, tracks=None, split_of=None, corrupt_overlap=None):
    """Write a miniature indexed release with overlapping source windows.

    ``corrupt_overlap`` is ``(recording, window_index, frame, label)`` and edits
    one label inside one source window only, so two overlapping windows then
    disagree about that frame -- the negative control.
    """
    tracks = TRACKS if tracks is None else tracks
    split_of = SPLIT_OF if split_of is None else split_of
    root = Path(root)
    per_split = {}
    rows = []
    for recording in sorted(tracks):
        labels = np.asarray(tracks[recording], dtype=np.int64)
        frames = labels.shape[0]
        music = music_track(frames)
        motion = motion_track(frames)
        split = split_of[recording]
        bucket = per_split.setdefault(split, {"labels": [], "music": [], "motion": [], "names": []})
        starts = list(range(0, frames - SOURCE_WINDOW + 1, SOURCE_STRIDE))
        assert starts[-1] + SOURCE_WINDOW == frames, "the synthetic source must tile"
        for order, start in enumerate(starts):
            stop = start + SOURCE_WINDOW
            window_labels = labels[start:stop].copy()
            if corrupt_overlap and corrupt_overlap[0] == recording and corrupt_overlap[1] == order:
                window_labels[corrupt_overlap[2]] = corrupt_overlap[3]
            rows.append({
                "array_index": len(bucket["labels"]),
                "split": split,
                "sequence_id": recording,
                "recording_id": recording,
                "retrieval_group_id": recording.rsplit(":", 1)[0],
                "start_frame": int(start),
                "end_frame_exclusive": int(stop),
                "window_id": "{}/window{:06d}".format(recording, order),
            })
            bucket["labels"].append(window_labels)
            bucket["music"].append(music[start:stop])
            bucket["motion"].append(motion[start:stop])
            bucket["names"].append("{}_slice{}".format(recording, order))
    for split, bucket in per_split.items():
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        np.save(str(directory / "labels.npy"), np.stack(bucket["labels"]))
        np.save(str(directory / "music.npy"), np.stack(bucket["music"]))
        np.save(str(directory / "motion.npy"), np.stack(bucket["motion"]))
        np.save(str(directory / "label_valid_mask.npy"),
                np.ones((len(bucket["labels"]), SOURCE_WINDOW), dtype=bool))
        (directory / "names.json").write_text(json.dumps(bucket["names"]))
    with open(str(root / "windows.jsonl"), "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return root


def make_segmentation(path, recordings=("wild_v5:1001:clip000", "wild_v5:1002:clip000")):
    """Two-frame bars over the recordings named, in segmentation.json's own ids."""
    records = []
    for recording in recordings:
        frames = len(TRACKS[recording])
        upload, clip = recording.split(":")[1], recording.split(":")[2]
        records.append({
            "sequence": "{}__{}".format(upload, clip),
            "segments": [{"start": s, "end": s + 2} for s in range(0, frames, 2)],
        })
    Path(path).write_text(json.dumps({"records": records}))
    return str(path)


# ------------------------------------------------------------ small pieces ---

def test_recording_id_mapping():
    assert recording_id_from_segmentation("7029295350347287812__clip000") == \
        "wild_v5:7029295350347287812:clip000"
    with pytest.raises(ValueError):
        recording_id_from_segmentation("7029295350347287812-clip000")


def test_window_starts_by_hand():
    assert window_starts(10, 4, 2) == [0, 2, 4, 6]
    assert window_starts(10, 4, 3) == [0, 3, 6]
    assert window_starts(6, 8, 2) == []          # shorter than the window
    assert window_starts(10, None, 2) == [0]     # whole-recording arm
    with pytest.raises(ValueError):
        window_starts(10, 1, 2)
    with pytest.raises(ValueError):
        window_starts(10, 4, 0)


def test_coverage_of_reports_the_gap_a_wide_stride_leaves():
    assert coverage_of([0, 2, 4, 6], 4, 10).all()
    partial = coverage_of([0], 4, 10)
    assert partial[:4].all() and not partial[4:].any()


def test_window_name_distinguishes_the_whole_clip_arm():
    assert window_name("wild_v5:1:clip000", 30, False) == "wild_v5:1:clip000_w000030"
    assert window_name("wild_v5:1:clip000", 0, True) == "wild_v5:1:clip000"


def test_bar_shape_stats_both_directions():
    alternating = bar_shape_stats([[1, 2, 3, 0, 1]])
    assert alternating["change_rate"] == 1.0
    assert alternating["runs"] == 5 and alternating["runs_ge_3"] == 0.0
    constant = bar_shape_stats([[5, 5, 5, 5]])
    assert constant["change_rate"] == 0.0
    assert constant["runs"] == 1 and constant["runs_ge_3"] == 1.0
    together = bar_shape_stats([[1, 2, 3, 0, 1], [5, 5, 5, 5]])
    assert together["changes"] == 4 and together["adjacent_pairs"] == 7
    assert together["change_rate"] == pytest.approx(4 / 7)
    assert together["filler_share"] == pytest.approx(1 / 9)
    assert together["distinct_classes"] == 5
    assert together["top_class_share"] == pytest.approx(4 / 9)


def test_bar_labels_drop_rather_than_clip_a_bar_past_the_track_end():
    track = np.array([1, 1, 2, 2, 3], dtype=np.int64)
    bars, dropped = bar_labels_of(track, [{"start": 0, "end": 2}, {"start": 2, "end": 4},
                                          {"start": 4, "end": 6}], num_classes=8)
    assert bars == [1, 2] and dropped == 1


# --------------------------------------------------- the rewindow itself ----

def test_rewindow_is_the_hand_computed_cut(tmp_path):
    source = make_source_release(tmp_path / "src")
    out = tmp_path / "w4"
    out.mkdir()
    report, rows, _, _ = build(source, out, window_frames=4, stride_frames=2)

    # Written out literally: 10 frames at window 4 stride 2 is four windows.
    expected_starts = [0, 2, 4, 6]
    recording = "wild_v5:1001:clip000"
    names = [row["name"] for row in rows if row["recording_id"] == recording]
    assert names == ["{}_w{:06d}".format(recording, s) for s in expected_starts]

    labels = np.asarray(TRACKS[recording])
    music = music_track(len(labels))
    for start in expected_starts:
        stem = "{}_w{:06d}".format(recording, start)
        written = np.load(str(out / "train" / "labels" / (stem + ".npy")))
        assert written.tolist() == labels[start:start + 4].tolist()
        written_music = np.load(str(out / "train" / "music" / (stem + ".npy")))
        assert np.array_equal(written_music, music[start:start + 4])
        placeholder = np.load(str(out / "train" / "motion" / (stem + ".npy")))
        assert placeholder.shape == (4, 1) and not placeholder.any()

    assert report["counts"]["windows"] == {"test": 2, "train": 4, "val": 3}
    assert report["counts"]["frames_unique"] == 10 + 8 + 6


def test_whole_clip_arm_emits_one_window_per_recording(tmp_path):
    source = make_source_release(tmp_path / "src")
    out = tmp_path / "full"
    out.mkdir()
    report, rows, _, _ = build(source, out, window_frames=None)
    assert report["counts"]["total_windows"] == 3
    assert {row["name"] for row in rows} == set(TRACKS)
    written = np.load(str(out / "train" / "labels" / "wild_v5:1001:clip000.npy"))
    assert written.tolist() == TRACKS["wild_v5:1001:clip000"]
    assert report["counts"]["frames_emitted"] == report["counts"]["frames_unique"]


def test_identity_holds_and_a_corrupted_window_is_caught(tmp_path):
    source = make_source_release(tmp_path / "src")
    out = tmp_path / "w4"
    out.mkdir()
    report, rows, source_tracks, _ = build(source, out, window_frames=4, stride_frames=2)
    identity = report["identity"]["rewindow"]
    assert identity["music_max_abs_delta"] == 0.0
    assert identity["label_disagreements"] == 0
    assert identity["label_disagreement_rate"] == 0.0
    assert identity["frames_compared"] == 10 + 8 + 6

    # NEGATIVE CONTROL: shift one music value by exactly 7.0 and one label, and
    # the check must report exactly that magnitude rather than "something moved".
    victim = out / rows[0]["music_path"]
    array = np.load(str(victim))
    array[0, 0] += 7.0
    np.save(str(victim), array)
    labels_path = out / rows[0]["labels_path"]
    label_array = np.load(str(labels_path))
    label_array[0] = 19
    np.save(str(labels_path), label_array)
    broken = verify_identity(out, rows, source_tracks)
    assert broken["music_max_abs_delta"] == pytest.approx(7.0)
    assert broken["label_disagreements"] == 1


def test_overlapping_source_windows_that_disagree_raise(tmp_path):
    # Frame 2 of the recording is carried by source windows 0 and 1; window 1
    # is edited so the two disagree.  First-writer-wins must not swallow it.
    source = make_source_release(
        tmp_path / "src", corrupt_overlap=("wild_v5:1001:clip000", 1, 0, 9))
    out = tmp_path / "w4"
    out.mkdir()
    with pytest.raises(ValueError, match="disagree"):
        build(source, out, window_frames=4, stride_frames=2)


def test_split_is_never_changed(tmp_path):
    source = make_source_release(tmp_path / "src")
    out = tmp_path / "w4"
    out.mkdir()
    report, rows, _, _ = build(source, out, window_frames=4, stride_frames=2)
    assert report["splits"]["recordings_per_split"] == {"train": 1, "val": 1, "test": 1}
    assert report["splits"]["recordings_per_split"] == report["splits"]["source_recordings_per_split"]
    assert report["splits"]["changed"] == []
    with pytest.raises(AssertionError, match="changed split"):
        verify_splits(rows, dict(SPLIT_OF, **{"wild_v5:1002:clip000": "train"}))


def test_a_recording_shorter_than_the_window_is_dropped_and_counted(tmp_path):
    source = make_source_release(tmp_path / "src")
    out = tmp_path / "w8"
    out.mkdir()
    report, rows, _, _ = build(source, out, window_frames=8, stride_frames=2)
    dropped = report["counts"]["recordings_dropped_short"]
    assert [entry["recording"] for entry in dropped] == ["wild_v5:1003:clip000"]
    assert dropped[0]["frames"] == 6
    assert report["counts"]["recordings_kept"] == 2
    assert "test" not in report["counts"]["windows"]


def test_a_stride_that_leaves_a_hole_is_refused(tmp_path):
    source = make_source_release(tmp_path / "src")
    out = tmp_path / "gap"
    out.mkdir()
    # 10 frames, window 4, stride 5: starts [0, 5] cover [0,4) and [5,9) --
    # frame 4 and frame 9 are in no window, so the cut would silently drop data.
    with pytest.raises(AssertionError, match="in no window"):
        build(source, out, window_frames=4, stride_frames=5)


def test_bar_change_survives_the_rewindow_and_a_wrong_reference_fails(tmp_path):
    source = make_source_release(tmp_path / "src")
    segmentation = make_segmentation(tmp_path / "seg.json")
    out = tmp_path / "w4"
    out.mkdir()
    report, _, _, _ = build(source, out, window_frames=4, stride_frames=2,
                            segmentation_path=segmentation, num_classes=21,
                            bar_change_reference=4 / 7, bar_change_tolerance=1e-9)
    bars = report["bar_metric"]
    assert bars["matched_recordings"] == 2
    assert bars["recordings_without_grid"] == 1
    assert bars["changes"] == 4 and bars["adjacent_pairs"] == 7
    assert bars["change_rate"] == pytest.approx(4 / 7)
    assert bars["reproduces_reference"] is True

    other = tmp_path / "w4b"
    other.mkdir()
    with pytest.raises(AssertionError, match="chase that"):
        build(source, other, window_frames=4, stride_frames=2,
              segmentation_path=segmentation, bar_change_reference=0.784,
              bar_change_tolerance=0.01)


def test_real_motion_arm_reproduces_the_source_motion(tmp_path):
    source = make_source_release(tmp_path / "src")
    out = tmp_path / "real"
    out.mkdir()
    build(source, out, window_frames=4, stride_frames=2, motion_kind="real")
    written = np.load(str(out / "train" / "motion" / "wild_v5:1001:clip000_w000004.npy"))
    assert written.shape == (4, MOTION_DIM)
    assert np.array_equal(written, motion_track(10)[4:8])


def test_music_stats_are_the_train_split_pooled_as_the_loader_draws_it(tmp_path):
    import torch

    source = make_source_release(tmp_path / "src")
    out = tmp_path / "w4"
    out.mkdir()
    report, rows, _, _ = build(source, out, window_frames=4, stride_frames=2)
    payload = torch.load(str(out / "music_stats.pt"), map_location="cpu")
    stacked = np.concatenate([
        np.load(str(out / row["music_path"])) for row in rows if row["split"] == "train"
    ]).astype(np.float64)
    assert report["music_stats"]["train_frames_pooled"] == stacked.shape[0]
    assert np.allclose(payload["mean"].numpy(), stacked.mean(0), atol=1e-4)
    assert np.allclose(payload["std"].numpy(), stacked.std(0), atol=1e-4)


def test_the_written_root_loads_and_refuses_global_music(tmp_path):
    from dataset.atomic_dataset import AtomicSequenceDataset, collate_atomic_sequences

    source = make_source_release(tmp_path / "src")
    out = tmp_path / "full"
    out.mkdir()
    build(source, out, window_frames=None)
    dataset = AtomicSequenceDataset(str(out), split="train")
    sample = dataset[0]
    assert sample["labels"].shape[0] == sample["music"].shape[0] == sample["motion"].shape[0]
    assert sample["music"].shape[1] == MUSIC_DIM and sample["motion"].shape[1] == 1
    batch = collate_atomic_sequences([sample])
    assert batch["padding_mask"].shape == (1, sample["labels"].shape[0])
    # Documented, not incidental: windows.jsonl exists in this root and the
    # per-file layout still refuses --global-music, because the refusal is
    # raised before anything looks for it.
    assert (out / "windows.jsonl").is_file()
    with pytest.raises(ValueError, match="per-file layout"):
        AtomicSequenceDataset(str(out), split="train", global_music=True)


def test_read_root_tracks_reassembles_what_was_written(tmp_path):
    source = make_source_release(tmp_path / "src")
    out = tmp_path / "w4"
    out.mkdir()
    _, rows, _, _ = build(source, out, window_frames=4, stride_frames=2)
    tracks = read_root_tracks(out, rows)
    labels, music, covered = tracks["wild_v5:1001:clip000"]
    assert covered.all()
    assert labels.tolist() == TRACKS["wild_v5:1001:clip000"]
    assert np.array_equal(music, music_track(10))


def test_bar_filler_share_is_not_the_frame_level_filler_share():
    """Pin the definition that the 30.4% in DANCE_QUALITY_DEFECTS 27.2 is NOT.

    Section 27.2's table puts four columns side by side.  Three of them --
    bar-to-bar change 78.4%, runs>=3 3.7%, runs>=5 0.5% -- are bar-level over
    the 270 gridded recordings, and ``verify_bar_metric`` reproduces all three
    exactly off the rewindowed roots.  The fourth, ``filler 30.4%``, does not
    come from that measurement at all: it is ``tools/score_plan_shape.py``'s
    reading (line 59 ``(labels == 0).mean()`` per clip, line 72 ``median`` over
    clips) on the 20-clip eval set -- **frame** level, on a **different
    population**.  Measured the way ``bar_shape_stats`` measures it, the same
    ground truth reads 15.7%.

    So ``bar_metric.filler_share`` must never be read against 30.4%.  The two
    numbers are free to move in opposite directions on the same data, and this
    test shows them doing exactly that in both directions, so that a future
    reader who reconciles them has to break a test to do it.
    """
    # Filler spread thin: no bar's majority is 0, so bar-level < frame-level.
    thin = np.array([1, 1, 1, 0, 2, 2, 2, 0], dtype=np.int64)
    bars, dropped = bar_labels_of(
        thin, [{"start": 0, "end": 4}, {"start": 4, "end": 8}], num_classes=8)
    assert dropped == 0 and bars == [1, 2]
    assert float((thin == 0).mean()) == pytest.approx(0.25)
    assert bar_shape_stats([bars])["filler_share"] == pytest.approx(0.0)

    # Filler clustered into one bar: bar-level > frame-level.
    clumped = np.array([0, 0, 0, 1, 2, 2, 2, 2], dtype=np.int64)
    bars, dropped = bar_labels_of(
        clumped, [{"start": 0, "end": 4}, {"start": 4, "end": 8}], num_classes=8)
    assert dropped == 0 and bars == [0, 2]
    assert float((clumped == 0).mean()) == pytest.approx(0.375)
    assert bar_shape_stats([bars])["filler_share"] == pytest.approx(0.5)
