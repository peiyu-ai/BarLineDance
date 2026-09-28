"""--completion-keep-root: the root channels come back from the draft.

The positive control is the one that matters here and it is stated as a rule:
with the flag on, the OUTPUT's root channels must equal the DRAFT's on every
conditioned frame, and the pose channels must NOT (or the flag has quietly
turned the completion off).  Both halves are asserted, because a keep mask that
is accidentally 1 everywhere passes the first and is the exact failure this
repository has already shipped once -- 2026-09-05, when the inpaint mask keyed
on mask edges, --index-filler made every frame conditioned, and the output came
back bit-identical to the draft.
"""
import ast
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import infer_atomic
from infer_atomic import (ROOT_POSITION_START, ROOT_POSITION_DIMS,
                          _hold_root_to_draft, infer_completion)

ROOT = slice(ROOT_POSITION_START, ROOT_POSITION_START + ROOT_POSITION_DIMS)
DIM = 151


class Recorder:
    """A completion whose sample() returns pure noise, so anything that survives
    to the output can only have come through the keep mask."""

    def __init__(self):
        self.keep_masks = []

    def sample(self, music, draft, noise_mask, keep_mask=None, **kwargs):
        self.keep_masks.append(None if keep_mask is None else keep_mask.clone())
        generated = torch.full_like(draft, -7.0)
        if keep_mask is not None:
            generated = keep_mask * draft + (1.0 - keep_mask) * generated
        return generated


def _run(keep_root, conditioned=True, inpaint_seam_width=None):
    frames = 48
    music = torch.zeros(frames, 4)
    draft = torch.arange(frames * DIM, dtype=torch.float32).reshape(frames, DIM) / 100.0
    mask = torch.full((frames, DIM), 1.0 if conditioned else 0.0)
    recorder = Recorder()
    # One window covering the clip, so the assertions read the mask this call
    # built rather than the average of two overlapping ones.
    out = infer_completion(recorder, music, draft, mask, frames, frames, "cpu",
                           seam_frames=[24], inpaint_seam_width=inpaint_seam_width,
                           keep_root=keep_root)
    return out, draft, recorder


def test_root_channels_come_back_from_the_draft():
    out, draft, _ = _run(keep_root=True)
    assert torch.allclose(out[:, ROOT], draft[:, ROOT], atol=1e-5)


def test_the_pose_channels_are_still_generated():
    """The other half of the control: keeping the root must not keep the dance."""
    out, draft, _ = _run(keep_root=True)
    pose = out[:, ROOT_POSITION_START + ROOT_POSITION_DIMS:]
    reference = draft[:, ROOT_POSITION_START + ROOT_POSITION_DIMS:]
    assert not torch.allclose(pose, reference, atol=1e-3)
    assert torch.allclose(pose, torch.full_like(pose, -7.0), atol=1e-5)


def test_off_by_default_leaves_the_root_generated():
    out, draft, recorder = _run(keep_root=False)
    assert recorder.keep_masks == [None]
    assert not torch.allclose(out[:, ROOT], draft[:, ROOT], atol=1e-3)


def test_an_unconditioned_frame_has_no_root_to_hold():
    out, draft, _ = _run(keep_root=True, conditioned=False)
    assert not torch.allclose(out[:, ROOT], draft[:, ROOT], atol=1e-3)


def test_it_composes_with_the_seam_ramp_without_writing_through_the_expand():
    """``keep_mask`` from the seam ramp is an ``expand`` view with a zero stride.

    Assigning one channel of such a view writes every channel.  If that
    happened the pose channels would be pinned to the draft too, i.e. the
    completion would be off -- so this asserts the seam ramp is still shaped
    like a ramp after the root assignment.
    """
    out, draft, recorder = _run(keep_root=True, inpaint_seam_width=8)
    keep = recorder.keep_masks[0]
    pose = keep[..., ROOT_POSITION_START + ROOT_POSITION_DIMS:]
    assert float(pose.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(pose.max()) == pytest.approx(1.0, abs=1e-6)
    assert float(keep[..., ROOT].min()) == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(out[:, ROOT], draft[:, ROOT], atol=1e-5)


def test_helper_does_not_mutate_the_mask_it_was_given():
    given = torch.zeros(2, 8, DIM)
    noise = torch.ones(2, 8, DIM)
    returned = _hold_root_to_draft(given, noise, DIM)
    assert float(given[..., ROOT].max()) == 0.0
    assert float(returned[..., ROOT].min()) == 1.0


def test_both_completion_call_sites_forward_the_flag():
    """The defect shape this repo keeps paying for: a flag that is parsed,
    recorded in the manifest, and never reaches the code.  Assert statically
    that every call that runs a completion carries it."""
    tree = ast.parse(pathlib.Path(infer_atomic.__file__).read_text())
    forwarding = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        if name == "infer_completion" and node.args:
            assert any(k.arg == "keep_root" for k in node.keywords), \
                "infer_completion call does not forward keep_root"
            forwarding += 1
        if name == "sample" and any(k.arg == "reproject_every" for k in node.keywords):
            assert any(k.arg == "keep_mask" for k in node.keywords), \
                "the batched completion.sample call does not pass a keep mask"
            forwarding += 1
    assert forwarding >= 2, forwarding


def test_the_manifest_records_the_flag_and_index_filler():
    source = pathlib.Path(infer_atomic.__file__).read_text()
    assert '"completion_keep_root": completion_keep_root,' in source
    assert '"index_filler": index_filler,' in source
