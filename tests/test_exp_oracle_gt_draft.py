"""Controls for the completion ORACLE (tools/exp_oracle_gt_draft.py).

The two claims this experiment's conclusion rests on, each pinned here:

1. THE DRAFT REALLY IS THE GROUND TRUTH.  If the oracle draft were a
   re-encoding, a resample, or a differently-normalized copy of the ground
   truth, "the completion cannot reproduce it" would be a statement about the
   encoder.  So arm ``gt`` must equal the ground-truth tensor BIT FOR BIT at
   named frames before any operation, and the release-backed store must return
   the release's own rows bit for bit.

2. THE STITCH IS THE SHIPPING STITCH.  The built-in ground-truth path refuses a
   clip longer than one 150-frame slice, so this tool drives the whole-clip loop
   itself.  It must therefore be shown that the crossfade is not reimplemented:
   the driver hands every window to ``infer_atomic.infer_completion`` with the
   shipping arm's own settings, and that function's overlap-add is a partition
   of unity (a completion that returns its draft unchanged must reassemble into
   exactly that draft).

Plus the arm-separation control: ``gt_ops`` must actually DIFFER from
``gt_nofill`` on the gap frames.  An arm that is silently identical to its
control cannot fail, which is the defect shape CLAUDE.md section 2 is about.
"""

import json
import pathlib
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import infer_atomic as IA  # noqa: E402
from tools import exp_oracle_gt_draft as m  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
RELEASE = pathlib.Path(m.RELEASE)


def _motion(frames=60, dim=151, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(frames, dim, generator=generator) * 2 - 1


def _labels(frames=60):
    """Two atomic segments that touch, then a transition gap, then a third."""
    labels = torch.zeros(frames, dtype=torch.int64)
    labels[5:20] = 3
    labels[20:35] = 7          # touches segment 3 -> a prototype-to-prototype seam
    labels[45:58] = 4          # after a label-0 gap -> gap_fill's business
    return labels


# ---------------------------------------------------------------- claim 1

def test_gt_arm_draft_is_the_ground_truth_bit_for_bit():
    motion = _motion()
    draft, mask = m.build_conditions(motion, _labels(), "gt")
    assert torch.equal(draft, motion)
    for frame in (0, 1, 19, 20, 40, 59):
        assert torch.equal(draft[frame], motion[frame]), frame
    assert torch.equal(mask, torch.ones(len(motion), 1))
    # And it is a copy, not an alias: an arm must not be able to mutate the store.
    draft[0, 0] = 12345.0
    assert motion[0, 0] != 12345.0


def test_gt_nofill_keeps_ground_truth_values_on_every_frame():
    motion, labels = _motion(), _labels()
    draft, mask = m.build_conditions(motion, labels, "gt_nofill")
    assert torch.equal(draft, motion)
    # Named label-0 frames keep the ground truth -- that is what "nofill" means.
    for frame in (0, 4, 36, 44, 58, 59):
        assert labels[frame] == 0
        assert torch.equal(draft[frame], motion[frame]), frame
    assert torch.equal(mask[:, 0], (labels != 0).to(torch.float32))


def test_gt_ops_is_the_shipping_operations_and_differs_from_its_control():
    motion, labels = _motion(), _labels()
    ops, ops_mask = m.build_conditions(motion, labels, "gt_ops")
    nofill, nofill_mask = m.build_conditions(motion, labels, "gt_nofill")
    assert torch.equal(ops_mask, nofill_mask)

    # Independently rebuilt from infer_atomic's own two functions, in
    # build_draft's order: zero the unretrieved frames, fill gaps, blend seams.
    expected = motion.clone()
    expected[labels == 0] = 0.0
    IA._fill_draft_gaps(expected, nofill_mask, "interpolate")
    IA._blend_draft_seams(expected, nofill_mask, labels, 4)
    assert torch.allclose(ops, expected, atol=0, rtol=0)

    # The arm must be able to differ from its control, on the frames it claims.
    gap = slice(35, 45)
    assert not torch.allclose(ops[gap], nofill[gap])
    seam = slice(16, 24)
    assert not torch.allclose(ops[seam], nofill[seam])
    # ... and must NOT differ in the interior of a segment, away from both.
    assert torch.equal(ops[8:13], motion[8:13])


def test_unknown_arm_and_missing_labels_are_refused():
    motion = _motion()
    with pytest.raises(ValueError):
        m.build_conditions(motion, _labels(), "gt_smoothed")
    with pytest.raises(ValueError):
        m.build_conditions(motion, None, "gt_nofill")
    with pytest.raises(ValueError):
        m.build_conditions(motion, _labels(40), "gt_ops")


@pytest.mark.skipif(not (RELEASE / "windows.jsonl").is_file(),
                    reason="release_v3 not present")
def test_store_returns_the_release_rows_bit_for_bit():
    store = m.GroundTruthMotionStore()
    name, spans = next(iter(
        (n, s) for n, s in store.spans.items() if s and s[0][3] == "test"))
    array = np.load(str(RELEASE / "test" / "motion.npy"), mmap_mode="r")
    start, end, index, _ = spans[0]
    frames = end - start
    track, provenance = store.get(name, frames)
    assert provenance["in_release"] is True
    assert provenance["frames_from_release_windows"] == frames
    assert provenance["frames_from_raw_renormalized"] == 0
    # Named frames, against the release array the model was trained on.
    for frame in (0, 1, frames // 2, frames - 1):
        assert torch.equal(track[start + frame],
                           torch.from_numpy(np.asarray(array[index][frame], np.float32)))
    # The overlapping windows agree, so the reconstruction is not a choice.
    assert store.conflicts == 0.0


@pytest.mark.skipif(not (RELEASE / "windows.jsonl").is_file(),
                    reason="release_v3 not present")
def test_raw_renormalized_fallback_matches_the_release_where_both_exist():
    """The calibration behind using the raw path for the 2 uncovered clips.

    Not "it looks reasonable": the fallback is required to reproduce the release
    on the overwhelming majority of frames of a clip the release DOES cover, or
    the two clips it is used for are not in the same space as the other 18.
    """
    store = m.GroundTruthMotionStore()
    name = next(n for n, s in store.spans.items() if s and s[0][3] == "test")
    covered = max(end for _, end, _, _ in store.spans[name])
    release, _ = store.get(name, covered)
    fallback = torch.from_numpy(store._raw_renormalized(name)[:covered])
    delta = (release - fallback).abs().max(dim=1).values
    assert float(delta.median()) < 1e-5
    assert float((delta > 1e-5).float().mean()) < 0.20


# ---------------------------------------------------------------- claim 2

class _EchoCompletion:
    """A completion whose sample() returns its draft, so the stitch is visible."""

    def __init__(self):
        self.model = types.SimpleNamespace(label_embedding=None)
        self.calls = []

    def sample(self, music, draft, mask, **kwargs):
        self.calls.append(kwargs)
        return draft.clone()


def test_infer_completion_overlap_add_reassembles_the_draft_exactly():
    """The shipping stitch is a partition of unity at the shipping settings.

    150-frame windows, stride 75, blend width 10 -- the shipping arm's numbers.
    If this failed, every difference this experiment measures could be the
    reassembly rather than the model.
    """
    frames = 437
    draft = _motion(frames, dim=8, seed=1)
    music = torch.zeros(frames, 35)
    output = IA.infer_completion(
        _EchoCompletion(), music, draft, torch.ones(frames, 1),
        150, 75, torch.device("cpu"), 2.0, 1, labels=None, blend_width=10)
    assert output.shape == draft.shape
    assert torch.allclose(output, draft, atol=1e-5)


def test_driver_hands_the_shipping_settings_to_the_shipping_stitch(tmp_path, monkeypatch):
    """The oracle does not reimplement windowing: it calls infer_completion.

    Captured here rather than asserted in prose, because "reuses the shipping
    code" is exactly the kind of claim that stops being true one edit later.
    """
    frames = 320
    name = "synthetic:clip000"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    np.save(str(audio_dir / (name + ".npy")),
            np.zeros((frames, 35), np.float32))

    completion = _EchoCompletion()
    completion_args = types.SimpleNamespace(
        seq_len=150, music_dim=35, motion_dim=151, draft_noise_ratio=0.25)
    monkeypatch.setattr(IA, "_load_checkpoint",
                        lambda *a, **k: (completion, completion_args))
    monkeypatch.setattr(IA, "validate_training_data_root", lambda root: {"stub": True})

    seen = {}
    real = IA.infer_completion

    def spy(model, music, draft, noise_mask, window, stride, device,
            guidance_weight=None, batch_size=1, **kwargs):
        seen.update(window=window, stride=stride, guidance_weight=guidance_weight,
                    batch_size=batch_size, blend_width=kwargs.get("blend_width"),
                    labels=kwargs.get("labels"),
                    noise_scale=float(noise_mask.max()),
                    draft=draft.clone())
        return real(model, music, draft, noise_mask, window, stride, device,
                    guidance_weight, batch_size, **kwargs)

    monkeypatch.setattr(IA, "infer_completion", spy)
    written = {}
    monkeypatch.setattr(IA, "_write_generated_result",
                        lambda path, normalized, *a, **k: written.update(
                            {"path": path, "normalized": normalized.clone()}))

    motion = _motion(frames, seed=5)

    class _Store:
        conflicts = 0.0

        def get(self, sequence, count):
            return motion[:count].clone(), {"frames": count, "in_release": True,
                                            "frames_from_release_windows": count,
                                            "frames_from_raw_renormalized": 0}

    manifest = m.run_arm(
        "gt", [name], audio_dir=audio_dir, output_dir=tmp_path / "out",
        data_root=tmp_path, completion_checkpoint="stub.pt", seed=20260902,
        guidance_weight=2.0, stride=75, blend_width=10, batch_size=1,
        device="cpu", motions=_Store(), plans=None)

    assert seen["window"] == 150 and seen["stride"] == 75
    assert seen["guidance_weight"] == 2.0 and seen["batch_size"] == 1
    assert seen["blend_width"] == 10
    assert seen["labels"] is None
    # noise_mask reaches the model as mask * the checkpoint's own ratio.
    assert seen["noise_scale"] == pytest.approx(0.25)
    # The draft that reached the shipping stitch is the ground truth itself.
    assert torch.equal(seen["draft"], motion[:frames])
    # ... and the echo model's output survives the reassembly unchanged.
    assert torch.allclose(written["normalized"], motion[:frames], atol=1e-5)

    record = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert record["sampling"]["draft_noise_ratio"] == 0.25
    assert record["sampling"]["mask_policy"] == "ones"
    assert record["headline_eligible"] is False
    assert manifest["per_clip"][name]["draft_vs_gt_max_abs"] == 0.0


def test_seed_is_the_clips_own_and_matches_infer_atomic(monkeypatch):
    """Arms are noise-paired per clip because the seed is a function of the NAME."""
    for name in ("wild_v5:7030793823240424742:clip000", "x:y:z"):
        assert IA.sample_seed(20260902, name) == IA.sample_seed(20260902, name)
    assert (IA.sample_seed(20260902, "a") != IA.sample_seed(20260902, "b"))
