"""Tests for the whole-track music summary the paper's planner conditions on.

The summary has one job -- carry song-level context a 150-frame window cannot
see -- and two ways to be quietly wrong: summarising a differently-weighted
track (windows overlap ten deep, so averaging them is not averaging the track),
and degenerating into a song id that a model memorises.  Both are asserted here,
the first directly and the second through the derangement that is its null.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.atomic_dataset import (  # noqa: E402
    AtomicSequenceDataset,
    collate_atomic_sequences,
)
from dataset.global_music import GlobalMusicError, track_summaries  # noqa: E402

MUSIC_DIM = 3
MOTION_DIM = 151


def _release(tmp_path, tracks, window=10, stride=5, split="train", drop_windows=0):
    """A release whose music is a known ramp per track, so the mean is known."""
    root = tmp_path / "release"
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    music_rows, names, records = [], [], []
    index = 0
    for track, length in tracks.items():
        timeline = np.arange(length, dtype=np.float32)[:, None] * np.ones(
            (1, MUSIC_DIM), dtype=np.float32)
        starts = list(range(0, length - window + 1, stride))
        if starts[-1] != length - window:
            starts.append(length - window)
        for start in starts:
            music_rows.append(timeline[start:start + window])
            names.append("{}_slice{}".format(track, start // stride))
            records.append({"split": split, "recording_id": track, "array_index": index,
                            "start_frame": start, "end_frame_exclusive": start + window})
            index += 1
    count = len(music_rows)
    np.save(str(directory / "music.npy"), np.stack(music_rows).astype(np.float32))
    np.save(str(directory / "motion.npy"),
            np.zeros((count, window, MOTION_DIM), dtype=np.float32))
    np.save(str(directory / "labels.npy"), np.zeros((count, window), dtype=np.int64))
    (directory / "names.json").write_text(json.dumps(names), encoding="utf-8")
    kept = records[:len(records) - drop_windows] if drop_windows else records
    (root / "windows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in kept) + "\n", encoding="utf-8")
    return root


def test_the_summary_is_the_track_mean_not_the_mean_of_its_windows():
    """Windows overlap ten deep, so their mean over-weights the middle.

    A 0..19 ramp has mean 9.5.  Averaging the overlapping windows of that ramp
    does not: the first and last frames appear in one window each while the
    middle appears in four, so the window mean is pulled toward the centre.
    The test pins the *track* value, which is the only one that summarises the
    track the model is being told about.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        root = _release(pathlib.Path(scratch), {"trackA": 20}, window=10, stride=5)
        table, sequences, _ = track_summaries(str(root), "train")
        assert sequences == ["trackA"]
        assert table.shape[1] == 2 * MUSIC_DIM
        np.testing.assert_allclose(table[:, :MUSIC_DIM], 9.5, rtol=1e-6)
        np.testing.assert_allclose(table[:, MUSIC_DIM:],
                                   np.std(np.arange(20, dtype=np.float64)), rtol=1e-6)
        # Every window of a track carries the same summary -- that is what
        # "broadcast to every frame" means at the window level.
        assert len({tuple(row) for row in table}) == 1


def test_two_tracks_get_two_summaries(tmp_path):
    root = _release(tmp_path, {"trackA": 20, "trackB": 40}, window=10, stride=5)
    table, sequences, index_of = track_summaries(str(root), "train")
    assert sequences == ["trackA", "trackB"]
    rows = {tuple(np.round(row[:MUSIC_DIM], 4)) for row in table}
    assert rows == {(9.5, 9.5, 9.5), (19.5, 19.5, 19.5)}
    assert index_of["trackB"] == 1


def test_the_window_count_must_match_the_array(tmp_path):
    """Dropping records makes the record and the array describe different sets."""
    root = _release(tmp_path, {"trackA": 40}, window=10, stride=5, drop_windows=2)
    with pytest.raises(GlobalMusicError, match="but the array holds"):
        track_summaries(str(root), "train")


def test_a_frame_covered_by_no_window_is_refused(tmp_path):
    """A gap makes the mean an average over an unstated subset of the track.

    Constructed so the *counts* still agree -- two windows, two rows -- because
    the count guard fires first and would otherwise mask this one.  Frames 10-19
    of a 30-frame track are covered by nothing.
    """
    root = tmp_path / "release"
    (root / "train").mkdir(parents=True)
    timeline = np.arange(30, dtype=np.float32)[:, None] * np.ones((1, MUSIC_DIM), np.float32)
    np.save(str(root / "train" / "music.npy"),
            np.stack([timeline[0:10], timeline[20:30]]).astype(np.float32))
    np.save(str(root / "train" / "motion.npy"), np.zeros((2, 10, MOTION_DIM), np.float32))
    np.save(str(root / "train" / "labels.npy"), np.zeros((2, 10), np.int64))
    (root / "train" / "names.json").write_text(
        json.dumps(["trackA_slice0", "trackA_slice1"]), encoding="utf-8")
    (root / "windows.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"split": "train", "recording_id": "trackA", "array_index": 0,
         "start_frame": 0, "end_frame_exclusive": 10},
        {"split": "train", "recording_id": "trackA", "array_index": 1,
         "start_frame": 20, "end_frame_exclusive": 30},
    ]) + "\n", encoding="utf-8")
    with pytest.raises(GlobalMusicError, match="covered by no window"):
        track_summaries(str(root), "train")


def test_a_release_without_the_window_record_is_refused(tmp_path):
    root = _release(tmp_path, {"trackA": 20})
    (root / "windows.jsonl").unlink()
    with pytest.raises(GlobalMusicError, match="no windows.jsonl"):
        track_summaries(str(root), "train")


def test_the_shuffle_is_a_derangement_so_no_song_keeps_its_own(tmp_path):
    """The null for "is this a song id?".

    A permutation that fixed a song would leave that song correctly conditioned,
    and the null would be weaker than it claims by exactly the fixed points.
    """
    root = _release(tmp_path, {"track{}".format(index): 20 + 10 * index
                               for index in range(6)}, window=10, stride=5)
    honest, sequences, _ = track_summaries(str(root), "train")
    shuffled, _, _ = track_summaries(str(root), "train", shuffle_seed=11)
    assert honest.shape == shuffled.shape
    seen = {}
    for row_honest, row_shuffled in zip(honest, shuffled):
        seen[tuple(row_honest)] = tuple(row_shuffled)
    assert all(key != value for key, value in seen.items())
    # Still the same multiset of summaries, only reassigned.
    assert sorted(seen) == sorted(seen.values())


def test_the_null_preserves_how_many_summaries_a_song_sees():
    """A null must not be a *stronger* condition than the thing it nulls.

    The honest summary is largely shared within a song, so deranging across
    sequences hands one song several vectors -- a better sequence id than the
    honest conditioning, and the 630-class arm's training loss showed it (0.169
    against 0.459 at epoch 100).  Deranging across *songs* keeps the number of
    distinct vectors each song's sequences see, so only the song-to-summary
    mapping is broken.
    """
    from dataset.global_music import derange_by_song

    # Two AIST songs, three recordings each; within a song the summaries agree.
    sequences = ["gBR_sBM_cAll_d04_mBR0_ch0{}".format(index) for index in range(1, 4)]
    sequences += ["gBR_sBM_cAll_d05_mBR1_ch0{}".format(index) for index in range(1, 4)]
    per_sequence = np.array([[1.0]] * 3 + [[2.0]] * 3, dtype=np.float32)
    shuffled = derange_by_song(per_sequence, sequences, seed=5)
    # Each song's three sequences still see one shared vector -- not three.
    assert len(set(shuffled[:3].ravel().tolist())) == 1
    assert len(set(shuffled[3:].ravel().tolist())) == 1
    # And it is the other song's.
    assert shuffled[0][0] == 2.0 and shuffled[3][0] == 1.0


def test_a_single_song_cannot_be_nulled(tmp_path):
    from dataset.global_music import derange_by_song

    with pytest.raises(GlobalMusicError, match="cannot be reassigned"):
        derange_by_song(np.zeros((2, 1), dtype=np.float32),
                        ["gBR_sBM_cAll_d04_mBR0_ch01", "gBR_sBM_cAll_d04_mBR0_ch02"],
                        seed=1)


def test_the_dataset_attaches_the_summary_and_the_collate_stacks_it(tmp_path):
    root = _release(tmp_path, {"trackA": 20, "trackB": 20}, window=10, stride=5)
    dataset = AtomicSequenceDataset(str(root), split="train", global_music=True)
    sample = dataset[0]
    assert sample["global_music"].shape == (2 * MUSIC_DIM,)
    batch = collate_atomic_sequences([dataset[index] for index in range(3)])
    assert batch["global_music"].shape == (3, 2 * MUSIC_DIM)


def test_a_batch_may_not_mix_samples_with_and_without_a_summary(tmp_path):
    root = _release(tmp_path, {"trackA": 20}, window=10, stride=5)
    with_summary = AtomicSequenceDataset(str(root), split="train", global_music=True)[0]
    without = AtomicSequenceDataset(str(root), split="train")[0]
    with pytest.raises(ValueError, match="carry a whole-track summary"):
        collate_atomic_sequences([with_summary, without])


def test_the_planner_refuses_to_run_a_global_head_on_nothing():
    """Zero-filling would train the head against a constant and report nothing.

    The entire question this path exists to answer is whether the whole-track
    summary carries anything; a silent zero answers it with a model that never
    received one.
    """
    from model.atomic_planner import AtomicPlannerTransformer

    model = AtomicPlannerTransformer(num_atomic_classes=4, music_dim=MUSIC_DIM,
                                     latent_dim=16, num_layers=1, num_heads=2,
                                     ff_size=16, max_seq_len=8, global_music=True)
    labels = torch.zeros((2, 8), dtype=torch.long)
    music = torch.zeros((2, 8, MUSIC_DIM))
    steps = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="whole-track summary is required"):
        model(labels, music, steps)
    logits = model(labels, music, steps, global_music=torch.zeros((2, 2 * MUSIC_DIM)))
    assert logits.shape == (2, 8, 5)


def test_a_checkpoint_trained_without_the_head_still_loads():
    """Off by default, and the default must stay loadable.

    Every artifact produced so far was trained without this path; a change that
    made those checkpoints unreadable would silently retire them.
    """
    from model.atomic_planner import AtomicPlannerTransformer

    plain = AtomicPlannerTransformer(num_atomic_classes=4, music_dim=MUSIC_DIM,
                                     latent_dim=16, num_layers=1, num_heads=2,
                                     ff_size=16, max_seq_len=8)
    assert plain.global_projection is None
    twin = AtomicPlannerTransformer(num_atomic_classes=4, music_dim=MUSIC_DIM,
                                    latent_dim=16, num_layers=1, num_heads=2,
                                    ff_size=16, max_seq_len=8)
    twin.load_state_dict(plain.state_dict())
