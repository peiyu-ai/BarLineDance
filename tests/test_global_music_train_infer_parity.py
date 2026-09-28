"""The whole-track summary must be the SAME vector at training and inference.

WHY THIS FILE EXISTS.  ``--global-music`` conditions the planner on a 70-D
``concat(mean, std)`` of a track's 35-D music.  That vector is built by two
different pieces of code:

* training  -- ``dataset.global_music.track_summaries`` stitches the track back
  together from ``windows.jsonl`` (overlapping windows cannot be averaged) and
  takes the two moments in float64 with numpy;
* inference -- ``infer_atomic.track_summary`` takes them with torch over the
  clip's own music array, in float32, with ``unbiased=False``.

Nothing in either file references the other.  If they drift -- a different
``ddof``, one of them normalising, one of them padding to ``seq_len`` before
summarising -- a global-music checkpoint is conditioned at inference on a vector
it never saw in training, and the failure is silent: the plan still generates,
the artifact still writes, and every downstream number is a reading of a model
being fed something else.  That is the defect class CLAUDE.md 2 is about, so it
gets a gate rather than a comment.

THE POSITIVE CONTROL is ``test_a_different_track_moves_the_summary``: the same
comparison run on two genuinely different tracks must FAIL to match.  Without it
a parity assertion that compared, say, two all-zero vectors would pass forever.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.global_music import track_summaries  # noqa: E402
from infer_atomic import track_summary  # noqa: E402

MUSIC_DIM = 35
MOTION_DIM = 151


def _release(tmp_path, timelines, window=10, stride=5, split="train"):
    """A release built from known per-track timelines, windowed with overlap."""
    root = tmp_path / "release"
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    music_rows, names, records = [], [], []
    index = 0
    for track, timeline in timelines.items():
        length = len(timeline)
        starts = list(range(0, length - window + 1, stride))
        if starts[-1] != length - window:
            starts.append(length - window)
        for start in starts:
            music_rows.append(timeline[start:start + window])
            names.append("{}_slice{}".format(track, start))
            records.append({"split": split, "recording_id": track,
                            "array_index": index, "start_frame": start,
                            "end_frame_exclusive": start + window})
            index += 1
    count = len(music_rows)
    np.save(str(directory / "music.npy"), np.stack(music_rows).astype(np.float32))
    np.save(str(directory / "motion.npy"),
            np.zeros((count, window, MOTION_DIM), dtype=np.float32))
    np.save(str(directory / "labels.npy"), np.zeros((count, window), dtype=np.int64))
    (directory / "names.json").write_text(json.dumps(names), encoding="utf-8")
    (root / "windows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    return root


def _timeline(seed, length):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(length, MUSIC_DIM)).astype(np.float32)


def test_training_and_inference_build_the_same_summary(tmp_path):
    """The gate: same track in, same 70-D vector out of both code paths.

    Compares ``track_summaries`` (training, stitched from windows.jsonl) with
    ``track_summary`` (inference, over the clip's music array) on one track.
    Tolerance is float32 rounding, not a fudge: training reduces in float64 and
    inference in float32, so the two agree to ~1e-6 and not bitwise.
    """
    timeline = _timeline(11, 97)
    root = _release(tmp_path, {"trackA": timeline}, window=10, stride=5)
    table, sequences, index_of = track_summaries(str(root), "train")
    assert sequences == ["trackA"]
    trained = table[index_of["trackA"]]
    inferred = track_summary(timeline).numpy()
    assert trained.shape == inferred.shape == (2 * MUSIC_DIM,)
    np.testing.assert_allclose(trained, inferred, rtol=0, atol=1e-5)


def test_a_different_track_moves_the_summary(tmp_path):
    """POSITIVE CONTROL: the comparison above can fail.

    Two tracks drawn from different seeds must NOT produce the same summary --
    otherwise the parity assertion is comparing something constant and would
    pass whatever the two implementations did.
    """
    timeline = _timeline(11, 97)
    other = _timeline(12, 97)
    root = _release(tmp_path, {"trackA": timeline}, window=10, stride=5)
    table, _, index_of = track_summaries(str(root), "train")
    trained = table[index_of["trackA"]]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(trained, track_summary(other).numpy(),
                                   rtol=0, atol=1e-5)


def test_inference_summarises_real_frames_not_padding(tmp_path):
    """Padding a clip to seq_len and summarising after would move the vector.

    ``track_summary``'s docstring says it is taken over the real frames only.
    Asserted rather than trusted, because the planner's inference path pads
    every window to ``window_size`` a few lines away from where it calls this.
    """
    timeline = _timeline(13, 60)
    padded = np.concatenate(
        (timeline, np.zeros((90, MUSIC_DIM), dtype=np.float32)), axis=0)
    real = track_summary(timeline).numpy()
    with_padding = track_summary(padded).numpy()
    assert not np.allclose(real, with_padding, atol=1e-5)
    root = _release(tmp_path, {"trackA": timeline}, window=10, stride=5)
    table, _, index_of = track_summaries(str(root), "train")
    np.testing.assert_allclose(table[index_of["trackA"]], real, rtol=0, atol=1e-5)


def test_the_std_half_uses_the_population_convention(tmp_path):
    """A ddof mismatch is the drift this file is most likely to catch.

    numpy's ``std`` and torch's ``std(unbiased=False)`` are both population; a
    change to either side that reached for the sample convention would shift the
    second half of the vector by a factor of sqrt(n/(n-1)).  Pinned explicitly
    so the failure names the cause.
    """
    timeline = _timeline(17, 41)
    inferred = track_summary(timeline).numpy()
    np.testing.assert_allclose(inferred[:MUSIC_DIM], timeline.mean(axis=0),
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(inferred[MUSIC_DIM:], timeline.std(axis=0, ddof=0),
                               rtol=0, atol=1e-5)
    sample = timeline.std(axis=0, ddof=1)
    assert not np.allclose(inferred[MUSIC_DIM:], sample, atol=1e-5)


def test_the_planner_head_is_read_off_the_module_not_a_flag():
    """A global-music checkpoint must announce itself, or inference feeds none.

    ``planner_wants_global_music`` is what decides whether the summary is passed
    at inference.  If it read a flag instead of the module, a checkpoint trained
    with the head would be planned without it and the model would raise -- or
    worse, a future zero-fill would make it silently a constant.
    """
    from infer_atomic import planner_wants_global_music
    from model.atomic_planner import AtomicPlannerTransformer
    from model.atomic_planner import UniformD3PM

    def build(global_music):
        model = AtomicPlannerTransformer(
            num_atomic_classes=5, music_dim=MUSIC_DIM, latent_dim=16,
            num_layers=1, num_heads=2, ff_size=16, dropout=0.0,
            max_seq_len=32, global_music=global_music)
        return UniformD3PM(model, num_steps=4)

    assert planner_wants_global_music(build(True)) is True
    assert planner_wants_global_music(build(False)) is False


def test_a_global_head_fed_nothing_raises_rather_than_conditioning_on_zero():
    """The whole arm is meaningless if the summary can quietly become constant."""
    from model.atomic_planner import AtomicPlannerTransformer

    model = AtomicPlannerTransformer(
        num_atomic_classes=5, music_dim=MUSIC_DIM, latent_dim=16, num_layers=1,
        num_heads=2, ff_size=16, dropout=0.0, max_seq_len=32, global_music=True)
    labels = torch.zeros((2, 8), dtype=torch.long)
    music = torch.zeros((2, 8, MUSIC_DIM))
    steps = torch.zeros((2,), dtype=torch.long)
    with pytest.raises(ValueError, match="whole-track"):
        model(labels, music, steps)
