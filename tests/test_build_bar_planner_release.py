"""Unit tests for tools/build_bar_planner_release.py.

The positive control is a synthetic corpus whose bar labels are chosen by hand,
so the bar-to-bar change rate, the run lengths, the filler share and the pooled
music are all known before the tool runs.  Two of them are deliberately opposite
readings -- an alternating corpus that must read 100% change and a constant one
that must read 0% -- because a shape metric that only ever reports "high" would
have passed every check in CLAUDE.md section 2.1 and still been useless.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_bar_planner_release import (  # noqa: E402
    BEAT_CHANNEL,
    MUSIC_DIM,
    ONSET_PEAK_CHANNEL,
    bar_shape_stats,
    bar_tokenise,
    build,
    covered_segments,
    majority_label,
    pool_bar_music,
    recording_id_from_segmentation,
    window_starts,
)

BEAT_OFFSETS = (0, 7, 15, 22)


# --------------------------------------------------------------- identity ---

def test_recording_id_mapping():
    assert (recording_id_from_segmentation("7029295350347287812__clip000")
            == "wild_v5:7029295350347287812:clip000")


@pytest.mark.parametrize("bad", ["7029295350347287812", "__clip000", "abc__"])
def test_recording_id_mapping_refuses_malformed(bad):
    with pytest.raises(ValueError):
        recording_id_from_segmentation(bad)


# ---------------------------------------------------------------- pooling ---

def _bar_frames(num_frames, beats, peaks):
    frames = np.zeros((num_frames, MUSIC_DIM), dtype=np.float64)
    frames[:, 0] = np.linspace(0.0, 1.0, num_frames)
    for beat in beats:
        frames[beat, BEAT_CHANNEL] = 1.0
    for peak in peaks:
        frames[peak, ONSET_PEAK_CHANNEL] = 1.0
    return frames


def test_beat_channel_sum_is_constant_but_the_mean_is_the_tempo():
    """The measurement that chose mean over count for channel 34.

    Two bars, both exactly four beats (which the grid guarantees), different
    durations.  A count cannot tell them apart -- it is 4 for both.  The mean
    is 4/frames, which is the bar's tempo, and it does tell them apart.
    """
    slow = _bar_frames(60, BEAT_OFFSETS, peaks=())
    fast = _bar_frames(40, (0, 5, 10, 15), peaks=())
    assert slow[:, BEAT_CHANNEL].sum() == fast[:, BEAT_CHANNEL].sum() == 4.0
    slow_pooled = pool_bar_music(slow)
    fast_pooled = pool_bar_music(fast)
    assert slow_pooled[BEAT_CHANNEL] == pytest.approx(4 / 60, rel=1e-6)
    assert fast_pooled[BEAT_CHANNEL] == pytest.approx(4 / 40, rel=1e-6)
    assert fast_pooled[BEAT_CHANNEL] > slow_pooled[BEAT_CHANNEL]


def test_onset_peak_channel_is_summed_not_meaned():
    frames = _bar_frames(60, BEAT_OFFSETS, peaks=(1, 4, 9, 30, 55))
    pooled = pool_bar_music(frames)
    assert pooled[ONSET_PEAK_CHANNEL] == pytest.approx(5.0)


def test_other_channels_are_meaned():
    frames = _bar_frames(50, BEAT_OFFSETS, peaks=())
    frames[:, 3] = np.arange(50, dtype=np.float64)
    pooled = pool_bar_music(frames)
    assert pooled[3] == pytest.approx(np.arange(50).mean())


def test_pooling_refuses_to_count_the_beat_channel():
    frames = _bar_frames(60, BEAT_OFFSETS, peaks=())
    with pytest.raises(ValueError, match="constant"):
        pool_bar_music(frames, count_channels=(BEAT_CHANNEL,))


def test_pooling_refuses_an_empty_bar():
    with pytest.raises(ValueError):
        pool_bar_music(np.zeros((0, MUSIC_DIM)))


# ----------------------------------------------------------------- labels ---

def test_majority_label_and_tie_break():
    assert majority_label(np.array([3, 3, 3, 5]), 21) == 3
    assert majority_label(np.array([5, 5, 3, 3]), 21) == 3  # lower index on a tie
    with pytest.raises(ValueError):
        majority_label(np.array([-1, 3]), 21)
    with pytest.raises(ValueError):
        majority_label(np.array([], dtype=np.int64), 21)


# ------------------------------------------------------------------ shape ---

def test_bar_shape_stats_positive_and_negative_control():
    """Known answers in both directions.

    ``alternating`` changes on every one of its 5 adjacent pairs -> 100%, and
    every run is length 1 -> no run reaches 3.  ``constant`` never changes ->
    0%, and its single run of 6 is >= 5.  A metric that read the second as
    anything but 0 would flatter any planner that emits one label forever.
    """
    alternating = [1, 2, 1, 2, 1, 2]
    constant = [7, 7, 7, 7, 7, 7]

    high = bar_shape_stats([alternating])
    assert high["adjacent_pairs"] == 5
    assert high["changes"] == 5
    assert high["change_rate"] == pytest.approx(1.0)
    assert high["runs"] == 6
    assert high["runs_ge_3"] == pytest.approx(0.0)
    assert high["longest_run"] == 1

    low = bar_shape_stats([constant])
    assert low["change_rate"] == pytest.approx(0.0)
    assert low["runs"] == 1
    assert low["runs_ge_3"] == pytest.approx(1.0)
    assert low["runs_ge_5"] == pytest.approx(1.0)
    assert low["longest_run"] == 6

    both = bar_shape_stats([alternating, constant])
    assert both["adjacent_pairs"] == 10          # pairs never cross a recording
    assert both["changes"] == 5
    assert both["change_rate"] == pytest.approx(0.5)
    assert both["bars"] == 12

    filler = bar_shape_stats([[0, 1, 0, 1]])
    assert filler["filler_bar_share"] == pytest.approx(0.5)


def test_window_starts():
    assert window_starts(6, 4, 1) == [0, 1, 2]
    assert window_starts(6, 4, 2) == [0, 2]
    assert window_starts(3, 4, 1) == []
    assert window_starts(4, 4, 1) == [0]
    with pytest.raises(ValueError):
        window_starts(6, 1, 1)
    with pytest.raises(ValueError):
        window_starts(6, 4, 0)


def test_covered_segments_drops_only_an_uncovered_tail():
    segments = [{"start": 0, "end": 30}, {"start": 30, "end": 60},
                {"start": 60, "end": 90}]
    kept, dropped = covered_segments(segments, 90)
    assert dropped == 0 and len(kept) == 3
    kept, dropped = covered_segments(segments, 89)
    assert dropped == 1 and [s["end"] for s in kept] == [30, 60]


def test_covered_segments_refuses_a_hole():
    """Uncovered bars may only be at the tail; a gap is a different failure.

    Out-of-order bars are the only way to produce one, and they mean the grid
    does not tile the recording.  Trimming them silently would hand the release
    a discontiguous bar sequence whose "adjacent" pairs are not adjacent.
    """
    segments = [{"start": 0, "end": 30}, {"start": 30, "end": 500},
                {"start": 500, "end": 60}]
    with pytest.raises(ValueError, match="prefix"):
        covered_segments(segments, 100)


def test_bar_tokenise_roundtrip_is_exact_when_bars_are_pure():
    labels = np.concatenate([np.full(30, 4), np.full(30, 9)])
    music = np.concatenate([_bar_frames(30, BEAT_OFFSETS, peaks=(2,)),
                            _bar_frames(30, BEAT_OFFSETS, peaks=(2, 5))])
    segments = [{"start": 0, "end": 30}, {"start": 30, "end": 60}]
    bar_labels, bar_music, spans, checks = bar_tokenise(
        labels, music, segments, 21, 4)
    assert bar_labels.tolist() == [4, 9]
    assert checks["roundtrip_wrong"] == 0
    assert checks["mixed_bars"] == 0
    assert checks["off_beat_starts"] == 0
    assert checks["bars_off_expected_beats"] == 0
    assert bar_music[0][ONSET_PEAK_CHANNEL] == pytest.approx(1.0)
    assert bar_music[1][ONSET_PEAK_CHANNEL] == pytest.approx(2.0)
    assert spans == [(0, 30), (30, 60)]


def test_bar_tokenise_reports_a_mixed_bar_instead_of_hiding_it():
    labels = np.concatenate([np.full(20, 4), np.full(10, 9)])
    music = _bar_frames(30, BEAT_OFFSETS, peaks=())
    bar_labels, _, _, checks = bar_tokenise(
        labels, music, [{"start": 0, "end": 30}], 21, 4)
    assert bar_labels.tolist() == [4]
    assert checks["mixed_bars"] == 1
    assert checks["roundtrip_wrong"] == 10


# ------------------------------------------------- synthetic end-to-end -----

def _write_synthetic_release(root, recordings, window=60, stride=30):
    """A miniature release_v3 with a bar grid whose labels are known.

    ``recordings`` maps ``"<upload>__clip000"`` -> ``(split, [bar labels],
    [onset peak counts], bar_frames)``.  Every bar holds exactly four beats, the
    first on its own first frame, so the tool's grid assertions apply exactly as
    they do on the corpus.  ``bar_frames`` differs between recordings on purpose:
    a fixed-tempo corpus pools channel 34 to the single constant ``4/bar_frames``
    and the tool refuses it -- see
    ``test_build_refuses_a_corpus_whose_tempo_never_moves``.
    """
    root = Path(root)
    per_split = {}
    windows = []
    grids = []
    for name, (split, bar_labels, peak_counts, bar_frames) in sorted(recordings.items()):
        beat_offsets = (0, bar_frames // 4, bar_frames // 2, (3 * bar_frames) // 4)
        upload, clip = name.split("__")
        sequence_id = "wild_v5:{}:{}".format(upload, clip)
        length = bar_frames * len(bar_labels)
        labels = np.concatenate(
            [np.full(bar_frames, label, dtype=np.int64) for label in bar_labels])
        music = np.zeros((length, MUSIC_DIM), dtype=np.float32)
        music[:, 1] = np.arange(length, dtype=np.float32)
        for index in range(len(bar_labels)):
            base = index * bar_frames
            for offset in beat_offsets:
                music[base + offset, BEAT_CHANNEL] = 1.0
            for peak in range(peak_counts[index]):
                music[base + 1 + peak, ONSET_PEAK_CHANNEL] = 1.0
        rows = per_split.setdefault(split, {"labels": [], "music": []})
        segments = [{"start": i * bar_frames, "end": (i + 1) * bar_frames,
                     "frames": bar_frames} for i in range(len(bar_labels))]
        grids.append({"sequence": name, "boundaries": [s["start"] for s in segments],
                      "segments": segments})
        for start in range(0, length - window + 1, stride):
            index = len(rows["labels"])
            rows["labels"].append(labels[start:start + window])
            rows["music"].append(music[start:start + window])
            windows.append({
                "array_index": index, "split": split, "sequence_id": sequence_id,
                "recording_id": sequence_id,
                "retrieval_group_id": "wild_v5:{}".format(upload),
                "start_frame": start, "end_frame_exclusive": start + window,
                "length": window,
                "window_id": "{}/window{:06d}".format(sequence_id, index),
            })
    for split, rows in per_split.items():
        base = root / split
        base.mkdir(parents=True, exist_ok=True)
        labels = np.stack(rows["labels"])
        np.save(str(base / "labels.npy"), labels)
        np.save(str(base / "music.npy"), np.stack(rows["music"]))
        np.save(str(base / "label_valid_mask.npy"), np.ones_like(labels, dtype=bool))
    root.mkdir(parents=True, exist_ok=True)
    with open(str(root / "windows.jsonl"), "w") as handle:
        for row in windows:
            handle.write(json.dumps(row) + "\n")
    segmentation = root / "segmentation.json"
    with open(str(segmentation), "w") as handle:
        json.dump({"config": {"mode": "grid", "beats_per_segment": 4},
                   "records": grids}, handle)
    return segmentation


def test_end_to_end_positive_control(tmp_path):
    """Known-by-construction corpus; every headline number is checked by hand.

    train: 1,2,1,2,1,2  -> 5 pairs, 5 changes
    val:   3,3,3,3,3    -> 4 pairs, 0 changes
    test:  0,4,4,0      -> 3 pairs, 2 changes
    Overall 7 of 12 adjacent pairs change, so the tool must read 7/12.
    """
    source = tmp_path / "frames"
    recordings = {
        "1000__clip000": ("train", [1, 2, 1, 2, 1, 2], [1, 2, 3, 4, 5, 6], 30),
        "2000__clip000": ("val", [3, 3, 3, 3, 3], [2, 2, 2, 2, 2], 42),
        "3000__clip000": ("test", [0, 4, 4, 0], [0, 1, 2, 3], 45),
    }
    segmentation = _write_synthetic_release(source, recordings)
    out = tmp_path / "bars"
    report = build(source, segmentation, out, window_bars=3, stride_bars=1,
                   num_classes=21, beats_per_bar=4, expected_recordings=3)

    shape = report["bar_shape_by_recording"]["all"]
    assert shape["bars"] == 15
    assert shape["adjacent_pairs"] == 12
    assert shape["changes"] == 7
    assert shape["change_rate"] == pytest.approx(7 / 12)
    assert shape["filler_bar_share"] == pytest.approx(2 / 15)
    assert report["roundtrip"]["wrong"] == 0
    assert report["roundtrip"]["frames"] == 6 * 30 + 5 * 42 + 4 * 45
    assert report["reconstruction"]["label_conflicts"] == 0
    assert report["reconstruction"]["music_max_overlap_delta"] == 0.0
    assert report["grid_alignment"]["off_beat_bar_starts"] == 0
    assert report["grid_alignment"]["beat_count_histogram"] == {"4": 15}

    # windows: 6 bars -> 4 starts, 5 -> 3, 4 -> 2, at K=3 stride 1
    assert report["windows_per_split"] == {"test": 2, "train": 4, "val": 3}

    # splits are preserved per recording, not re-drawn
    assert report["recordings_per_split"] == {"test": 1, "train": 1, "val": 1}

    train_labels = np.load(str(out / "train" / "labels.npy"))
    assert train_labels.tolist() == [[1, 2, 1], [2, 1, 2], [1, 2, 1], [2, 1, 2]]
    val_labels = np.load(str(out / "val" / "labels.npy"))
    assert val_labels.tolist() == [[3, 3, 3]] * 3
    test_labels = np.load(str(out / "test" / "labels.npy"))
    assert test_labels.tolist() == [[0, 4, 4], [4, 4, 0]]

    music = np.load(str(out / "train" / "music.npy"))
    assert music.shape == (4, 3, MUSIC_DIM)
    # channel 34 pooled to 4/30 in this recording's bars, all 30 frames long,
    # while the val recording's 42-frame bars pool to 4/42 -- the tempo, which
    # a beat *count* (always 4) could not have carried.
    assert np.allclose(music[:, :, BEAT_CHANNEL], 4 / 30)
    val_music = np.load(str(out / "val" / "music.npy"))
    assert np.allclose(val_music[:, :, BEAT_CHANNEL], 4 / 42)
    # channel 33 is the onset-peak count of each bar, here 1..6 by construction
    assert music[0, :, ONSET_PEAK_CHANNEL].tolist() == [1.0, 2.0, 3.0]
    assert music[3, :, ONSET_PEAK_CHANNEL].tolist() == [4.0, 5.0, 6.0]

    motion = np.load(str(out / "train" / "motion.npy"))
    assert motion.shape == (4, 3, 1)
    assert not motion.any(), "the motion placeholder must be exactly zeros"
    assert report["motion_is_placeholder"] is True

    names = json.loads((out / "train" / "names.json").read_text())
    groups = json.loads((out / "train" / "retrieval_groups.json").read_text())
    assert len(names) == len(groups) == 4
    assert names[0] == "wild_v5:1000:clip000_bars0000"
    assert groups == ["wild_v5:1000"] * 4

    bars = [json.loads(line) for line in (out / "bars.jsonl").read_text().splitlines()]
    assert len(bars) == 9
    first = [row for row in bars if row["split"] == "train"][0]
    assert first["bar_frame_spans"] == [[0, 30], [30, 60], [60, 90]]
    assert first["bar_labels"] == [1, 2, 1]
    assert first["seconds"] == pytest.approx(3.0)

    # no build.json: the trainer must keep taking its legacy, non-headline path
    assert not (out / "build.json").exists()
    assert (out / "bar_release_build.json").is_file()


def test_end_to_end_negative_control_reads_zero_change(tmp_path):
    """A corpus with one label everywhere must read 0% change, not a floor."""
    source = tmp_path / "frames"
    segmentation = _write_synthetic_release(
        source, {"1000__clip000": ("train", [5] * 6, [0, 1, 2, 3, 4, 5], 30),
                 "2000__clip000": ("val", [5] * 6, [5, 4, 3, 2, 1, 0], 42),
                 "3000__clip000": ("test", [5] * 6, [1, 3, 2, 4, 0, 5], 45)})
    report = build(source, segmentation, tmp_path / "bars", window_bars=3,
                   stride_bars=1, num_classes=21, expected_recordings=3)
    shape = report["bar_shape_by_recording"]["all"]
    assert shape["change_rate"] == pytest.approx(0.0)
    assert shape["runs_ge_5"] == pytest.approx(1.0)
    assert shape["distinct_classes"] == 1
    assert shape["top_class_share"] == pytest.approx(1.0)


def test_the_dataset_class_can_read_the_written_release(tmp_path):
    from dataset.atomic_dataset import AtomicSequenceDataset

    source = tmp_path / "frames"
    segmentation = _write_synthetic_release(
        source, {"1000__clip000": ("train", [1, 2, 1, 2, 1, 2], [0, 1, 2, 3, 4, 5], 30),
                 "2000__clip000": ("train", [3, 4, 3, 4, 3, 4], [5, 4, 3, 2, 1, 0], 42)})
    out = tmp_path / "bars"
    report = build(source, segmentation, out, window_bars=3, stride_bars=1,
                   expected_recordings=2)
    # 6 bars -> 4 windows, plus the 42-frame recording whose last bar ends past
    # its last whole 60-frame window and is dropped, 5 bars -> 3 windows.
    assert report["tail_bars_dropped_uncovered"]["bars"] == 1
    dataset = AtomicSequenceDataset(str(out), split="train")
    assert len(dataset) == 7
    sample = dataset[0]
    assert tuple(sample["labels"].shape) == (3,)
    assert tuple(sample["music"].shape) == (3, MUSIC_DIM)
    assert tuple(sample["motion"].shape) == (3, 1)
    assert sample["retrieval_group_id"] == "wild_v5:1000"


def test_music_stats_are_refit_on_the_pooled_bars(tmp_path):
    import torch

    source = tmp_path / "frames"
    segmentation = _write_synthetic_release(
        source, {"1000__clip000": ("train", [1, 2, 1, 2, 1, 2], [0, 1, 2, 3, 4, 5], 30),
                 "2000__clip000": ("train", [3, 4, 3, 4, 3, 4], [1] * 6, 42)})
    out = tmp_path / "bars"
    report = build(source, segmentation, out, window_bars=3, stride_bars=1,
                   expected_recordings=2)
    payload = torch.load(str(out / "music_stats.pt"), map_location="cpu", weights_only=True)
    train_music = np.load(str(out / "train" / "music.npy")).reshape(-1, MUSIC_DIM)
    assert payload["mean"].numpy() == pytest.approx(train_music.mean(0), rel=1e-5)
    assert payload["std"].numpy() == pytest.approx(
        np.where(train_music.std(0) > 0, train_music.std(0), 1.0), rel=1e-5)
    # A channel that is constant across the pooled bars keeps std 1 rather than
    # an epsilon-inflated one: scaling up a channel's noise invents information
    # (the rule in tools/fit_motion_normalizer.py).
    constant = np.flatnonzero(train_music.std(0) == 0)
    assert constant.size
    assert payload["std"].numpy()[constant] == pytest.approx(1.0)
    assert report["music_stats"]["fit_split"] == "train"


# ------------------------------------------------------------- refusals -----

def test_build_refuses_when_overlapping_windows_disagree(tmp_path):
    source = tmp_path / "frames"
    segmentation = _write_synthetic_release(
        source, {"1000__clip000": ("train", [1, 2, 1, 2, 1, 2], [0, 1, 2, 3, 4, 5], 30),
                 "2000__clip000": ("train", [3, 4, 3, 4, 3, 4], [5, 4, 3, 2, 1, 0], 42)})
    labels = np.load(str(source / "train" / "labels.npy"))
    labels[1, 0] = 17           # a frame two windows both cover
    np.save(str(source / "train" / "labels.npy"), labels)
    with pytest.raises(AssertionError, match="disagree"):
        build(source, segmentation, tmp_path / "bars", window_bars=3,
              stride_bars=1, expected_recordings=2)


def test_build_refuses_a_bar_that_is_not_four_beats(tmp_path):
    source = tmp_path / "frames"
    segmentation = _write_synthetic_release(
        source, {"1000__clip000": ("train", [1, 2, 1, 2, 1, 2], [0, 1, 2, 3, 4, 5], 30),
                 "2000__clip000": ("train", [3, 4, 3, 4, 3, 4], [5, 4, 3, 2, 1, 0], 42)})
    music = np.load(str(source / "train" / "music.npy"))
    music[:, :, BEAT_CHANNEL] = 0.0
    music[:, 0, BEAT_CHANNEL] = 1.0
    np.save(str(source / "train" / "music.npy"), music)
    with pytest.raises(AssertionError, match="beats"):
        build(source, segmentation, tmp_path / "bars", window_bars=3,
              stride_bars=1, expected_recordings=2)


def test_build_refuses_an_unexpected_recording_count(tmp_path):
    source = tmp_path / "frames"
    segmentation = _write_synthetic_release(
        source, {"1000__clip000": ("train", [1, 2, 1, 2, 1, 2], [0, 1, 2, 3, 4, 5], 30),
                 "2000__clip000": ("train", [3, 4, 3, 4, 3, 4], [5, 4, 3, 2, 1, 0], 42)})
    with pytest.raises(AssertionError, match="expected 270"):
        build(source, segmentation, tmp_path / "bars", window_bars=3,
              stride_bars=1, expected_recordings=270)


def test_build_refuses_a_release_recording_with_no_bar_grid(tmp_path):
    source = tmp_path / "frames"
    segmentation = _write_synthetic_release(
        source, {"1000__clip000": ("train", [1, 2, 1, 2, 1, 2], [0, 1, 2, 3, 4, 5], 30),
                 "2000__clip000": ("val", [3, 4, 3, 4, 3, 4], [5, 4, 3, 2, 1, 0], 42)})
    payload = json.loads(Path(segmentation).read_text())
    payload["records"] = [r for r in payload["records"] if r["sequence"] != "2000__clip000"]
    Path(segmentation).write_text(json.dumps(payload))
    with pytest.raises(AssertionError, match="no bar grid"):
        build(source, segmentation, tmp_path / "bars", window_bars=3,
              stride_bars=1, expected_recordings=None)


def test_build_refuses_a_corpus_whose_tempo_never_moves(tmp_path):
    """The pooling gate, fired on purpose.

    Every bar the same number of frames means channel 34 pools to the single
    constant ``4/bar_frames``, so the beat is *gone* from the release even
    though it varies frame to frame.  The build must refuse rather than hand a
    planner a dead channel.  (The real corpus spans 85.7-180 BPM, so this
    cannot happen there -- which is exactly why the gate needs its own case.)
    """
    source = tmp_path / "frames"
    segmentation = _write_synthetic_release(
        source, {"1000__clip000": ("train", [1, 2, 1, 2, 1, 2], [1, 2, 3, 4, 5, 6], 30),
                 "2000__clip000": ("val", [3, 4, 3, 4, 3, 4], [6, 5, 4, 3, 2, 1], 30)})
    with pytest.raises(AssertionError, match="beat"):
        build(source, segmentation, tmp_path / "bars", window_bars=3,
              stride_bars=1, expected_recordings=2)
