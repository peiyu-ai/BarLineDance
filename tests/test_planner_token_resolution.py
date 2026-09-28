"""``--planner-token-resolution``: the declaration, and the gate behind it.

WHAT IS BEING TESTED AND WHY IT IS A GATE.  ``infer_atomic`` reads
``planner_token_resolution`` off a checkpoint's saved training arguments and
refuses ``--plan-bar-tokens`` unless it says ``"bar"``.  The declaration is made
in ``train_atomic.declare_planner_token_resolution``, which does not take the
author's word for it: it measures the training rows.  The mistake it exists to
catch is pointing ``--planner-token-resolution bar`` at the per-FRAME release,
whose rows are 150 tokens that change label on 1.24% of adjacent pairs; a bar
release's 4 tokens change on about 77%.  Nothing downstream could see that
error -- ``MusicNormalization``'s buffers are ``(music_dim,)`` either way.

Both directions are covered: a POSITIVE control on the real bar release (and on
a synthetic bar-shaped one, so the file still tests the gate off-machine), and
a NEGATIVE control on frame-shaped rows.
"""

import argparse
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import train_atomic  # noqa: E402

BAR_RELEASE = pathlib.Path("/cache/atomicdance-assets/scratch/txy_t/release_bar_v1")


class FakeDataset:
    """The two fields the gate reads, in the shape the real dataset returns."""

    def __init__(self, labels, music_dim=35):
        self.rows = [torch.tensor(row, dtype=torch.long) for row in labels]
        self.music_dim = music_dim

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        labels = self.rows[index]
        return {"labels": labels,
                "music": torch.zeros(len(labels), self.music_dim)}


def make_args(**overrides):
    args = argparse.Namespace(
        stage="planner",
        seq_len=4,
        music_dim=35,
        planner_token_resolution="bar",
        planner_bar_pooling="mean",
        planner_bar_beats=4,
        music_phase_features=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def bar_shaped_rows(rows=32, tokens=4):
    """Rows whose adjacent tokens nearly always differ, like a real bar release."""
    return [[(index + position) % 20 + 1 for position in range(tokens)]
            for index in range(rows)]


def frame_shaped_rows(rows=8, tokens=150, changes_per_row=2):
    """Rows that hold one label for a long time -- the per-frame release's shape."""
    out = []
    for index in range(rows):
        row = []
        for change in range(changes_per_row + 1):
            row.extend([(index + change) % 20 + 1] * (tokens // (changes_per_row + 1)))
        out.append((row + [row[-1]] * tokens)[:tokens])
    return out


# ------------------------------------------------------------- positive ------

def test_bar_shaped_release_is_accepted_and_stamps_its_evidence():
    args = make_args()
    evidence = train_atomic.declare_planner_token_resolution(args, FakeDataset(bar_shaped_rows()))
    assert args.planner_token_resolution == "bar"
    assert evidence["tokens_per_row"] == 4
    assert evidence["adjacent_token_change_rate"] > 0.9
    # It travels inside the checkpoint, which is the whole point.
    assert args.planner_token_resolution_evidence == evidence


def test_frame_resolution_is_the_default_and_touches_nothing():
    args = make_args(planner_token_resolution="frame", seq_len=150)
    assert train_atomic.declare_planner_token_resolution(
        args, FakeDataset(frame_shaped_rows())) is None
    assert args.planner_token_resolution == "frame"
    assert not hasattr(args, "planner_token_resolution_evidence")


@pytest.mark.skipif(not BAR_RELEASE.exists(), reason="the bar release is not on this machine")
def test_the_real_bar_release_passes_the_gate():
    from dataset.atomic_dataset import AtomicSequenceDataset

    dataset = AtomicSequenceDataset(str(BAR_RELEASE), split="train")
    args = make_args()
    evidence = train_atomic.declare_planner_token_resolution(args, dataset)
    assert evidence["tokens_per_row"] == 4
    # The corpus reading, not a synthetic one: ground truth changes label from
    # bar to bar 78.4% of the time (docs/DANCE_QUALITY_DEFECTS.md 27.2).
    assert 0.6 < evidence["adjacent_token_change_rate"] < 0.95


@pytest.mark.skipif(not BAR_RELEASE.exists(), reason="the bar release is not on this machine")
def test_the_real_frame_release_is_refused_by_the_same_call():
    """The negative control on real data, which is the mistake this gate is for."""
    from dataset.atomic_dataset import AtomicSequenceDataset

    frame_release = pathlib.Path("/cache/atomicdance-assets/scratch/txy_t/release_v3")
    if not frame_release.exists():
        pytest.skip("the frame release is not on this machine")
    dataset = AtomicSequenceDataset(str(frame_release), split="train")
    args = make_args(seq_len=150)
    with pytest.raises(ValueError, match="not tokenised by bars"):
        train_atomic.declare_planner_token_resolution(args, dataset, sample_limit=64)


# ------------------------------------------------------------- negative ------

def test_frame_shaped_rows_are_refused():
    args = make_args(seq_len=150)
    with pytest.raises(ValueError, match="not tokenised by bars"):
        train_atomic.declare_planner_token_resolution(args, FakeDataset(frame_shaped_rows()))


def test_a_row_length_that_is_not_seq_len_is_refused():
    args = make_args(seq_len=8)
    with pytest.raises(ValueError, match="counts BARS"):
        train_atomic.declare_planner_token_resolution(args, FakeDataset(bar_shaped_rows()))


def test_a_music_width_that_contradicts_the_pooling_is_refused():
    args = make_args(planner_bar_pooling="mean_std", music_dim=70)
    with pytest.raises(ValueError, match="70-D music, but the release"):
        train_atomic.declare_planner_token_resolution(args, FakeDataset(bar_shaped_rows()))


def test_the_completion_stage_is_refused():
    args = make_args(stage="completion")
    with pytest.raises(ValueError, match="placeholder motion"):
        train_atomic.declare_planner_token_resolution(args, FakeDataset(bar_shaped_rows()))


def test_music_phase_features_is_refused_in_bar_mode():
    args = make_args(music_phase_features=True)
    with pytest.raises(ValueError, match="beat density"):
        train_atomic.declare_planner_token_resolution(args, FakeDataset(bar_shaped_rows()))


def test_an_unknown_resolution_is_refused():
    args = make_args(planner_token_resolution="beat")
    with pytest.raises(ValueError, match="must be one of"):
        train_atomic.declare_planner_token_resolution(args, FakeDataset(bar_shaped_rows()))


def test_an_unknown_pooling_is_refused():
    args = make_args(planner_bar_pooling="median")
    with pytest.raises(ValueError, match="planner-bar-pooling"):
        train_atomic.declare_planner_token_resolution(args, FakeDataset(bar_shaped_rows()))


def test_an_empty_split_is_refused():
    args = make_args()
    with pytest.raises(ValueError, match="nothing to declare"):
        train_atomic.declare_planner_token_resolution(args, FakeDataset([]))


def test_single_token_rows_carry_no_transition_and_are_refused():
    args = make_args(seq_len=1)
    with pytest.raises(ValueError, match="no adjacent token pair"):
        train_atomic.declare_planner_token_resolution(args, FakeDataset([[3], [7]]))


def test_the_checkpoint_the_shipping_planner_was_trained_from_reads_frame():
    """A checkpoint saved before this flag existed must still read 'frame'."""
    from infer_atomic import planner_token_resolution

    assert planner_token_resolution(argparse.Namespace(seq_len=150)) == "frame"
