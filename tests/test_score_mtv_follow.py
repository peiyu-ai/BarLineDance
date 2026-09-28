"""The MTV follow score must tell a following render from a static one, and the
base gate must refuse a backbone with no motion path.

Both defects this guards against were silent: the first MTV renders stood still
on a base with 0 motion tensors and every pipeline signal was green.
"""
import json
import pathlib
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import score_mtv_follow as follow  # noqa: E402


def _dance(frames=120, seed=0):
    """Camera-frame joints (mm, y-down, depth ~1.8 m) with limbs that swing."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 150.0, size=(24, 3))
    base[:, 2] = 1800.0 + rng.normal(0.0, 30.0, size=24)
    base[12, 1], base[7, 1], base[8, 1] = -500.0, 400.0, 400.0     # neck up, ankles down
    t = np.arange(frames)[:, None]
    joints = np.repeat(base[None], frames, axis=0)
    for j in (18, 19, 20, 21, 4, 5):
        phase = rng.uniform(0, 2 * np.pi)
        joints[:, j, 0] += 250.0 * np.sin(2 * np.pi * t[:, 0] / 17.0 + phase)
        joints[:, j, 1] += 200.0 * np.cos(2 * np.pi * t[:, 0] / 23.0 + phase)
    joints[:, :, 0] += 300.0 * np.sin(2 * np.pi * t / 60.0)          # the root travels
    return joints


def _detections(points, scale=900.0, offset=(240.0, 400.0), noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for f in range(len(points)):
        xy = points[f] * scale + np.asarray(offset) + rng.normal(0.0, noise, size=points[f].shape)
        out[f] = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
    return out


def test_following_render_beats_its_shifted_pose():
    projected = follow.project(_dance())
    result = follow.score(_detections(projected, noise=1.0), projected)
    null = follow.score(_detections(projected, noise=1.0), projected, shift=len(projected) // 2)
    assert result["follow"] < 0.02
    assert result["follow"] < 0.5 * null["follow"]
    assert result["moves_render"] == pytest.approx(result["moves_pose"], rel=0.2)


def test_static_render_reads_no_motion_and_does_not_follow():
    projected = follow.project(_dance())
    still = np.repeat(projected[:1], len(projected), axis=0)
    result = follow.score(_detections(still, noise=0.5), projected)
    moving = follow.score(_detections(projected, noise=0.5), projected)
    assert result["moves_render"] < 0.1 * result["moves_pose"]
    assert result["follow"] > 3 * moving["follow"]


def test_scale_and_offset_are_free():
    """MTV picks its own framing; a different scale and position must not count as error."""
    projected = follow.project(_dance())
    a = follow.score(_detections(projected, scale=600.0, offset=(100.0, 50.0)), projected)
    b = follow.score(_detections(projected, scale=1400.0, offset=(300.0, 700.0)), projected)
    assert a["follow"] < 1e-6 and b["follow"] < 1e-6


def test_low_confidence_joints_are_ignored():
    projected = follow.project(_dance())
    dets = _detections(projected)
    for f in dets:
        dets[f][3, :2] += 5000.0          # a wild wrist...
        dets[f][3, 2] = 0.05              # ...that the detector itself did not trust
    assert follow.score(dets, projected)["follow"] < 1e-6


def test_detect_returns_face_points_for_the_quality_column():
    import inspect
    src = inspect.getsource(follow.detect)
    assert "faces[index] = body[FACE]" in src and follow.FACE == [0, 14, 15, 16, 17]


def test_unconfident_bodies_are_not_people():
    projected = follow.project(_dance(frames=10))
    dets = _detections(projected)
    for f in range(0, 10, 2):
        dets[f][:, 2] = 0.2          # the detector "found" a body in a collapsed frame
    kept = follow.people(dets)
    assert sorted(kept) == [1, 3, 5, 7, 9]


def _fake_safetensors(path, names):
    header = {n: {"dtype": "F16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]}
              for i, n in enumerate(names)}
    blob = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * (2 * len(names)))


def test_gate_refuses_a_base_without_the_motion_path(tmp_path, monkeypatch):
    from render2d import comfy_mtv
    monkeypatch.setattr(comfy_mtv, "COMFY_ROOT", tmp_path)
    root = tmp_path / "models" / "diffusion_models"
    _fake_safetensors(root / "plain.safetensors", ["blocks.0.self_attn.q.weight"])
    _fake_safetensors(root / "mtv.safetensors",
                      ["blocks.0.self_attn.q.weight", "blocks.0.motion_attn.q.weight",
                       "blocks.0.norm4.weight"])
    _fake_safetensors(root / "adapter.safetensors", ["blocks.0.motion_attn.q.weight"])
    with pytest.raises(SystemExit, match="0 of its 1 tensors"):
        comfy_mtv.assert_motion_path("plain.safetensors", "adapter.safetensors")
    got = comfy_mtv.assert_motion_path("mtv.safetensors", None)
    assert got == {"base_tensors": 3, "base_motion_tensors": 2, "adapter_motion_tensors": 0}
    with pytest.raises(SystemExit, match="MTV_ADAPTER=none"):
        comfy_mtv.assert_motion_path("mtv.safetensors", "adapter.safetensors")


def test_rope_patch_detection(tmp_path, monkeypatch):
    from render2d import comfy_mtv
    target = tmp_path / "nodes_sampler.py"
    monkeypatch.setattr(comfy_mtv, "ROPE_PATCH", target)
    target.write_text("            mtv_freqs = mtv_freqs.to(device, dtype)\n")
    assert not comfy_mtv.rope_patched()
    # the rope half alone is not the patch: motion guidance must be there too
    target.write_text("            mtv_freqs = mtv_freqs.to(device)\n")
    assert not comfy_mtv.rope_patched()
    target.write_text("            mtv_freqs = mtv_freqs.to(device)\n"
                      "        cfg = mtv_input.get(\"motion_cfg\", 1.0)\n")
    assert not comfy_mtv.rope_patched()   # the window-reference switch is part of it too
    # The markers are the KEY NAMES, not a line of code: the window-reference switch
    # grew a third mode and an exact-line marker made the gate reject a patched tree.
    target.write_text("            mtv_freqs = mtv_freqs.to(device)\n"
                      "        cfg = mtv_input.get(\"motion_cfg\", 1.0)\n"
                      "        mode = mtv_input.get(\"window_reference\", \"on\")\n")
    assert comfy_mtv.rope_patched()


def test_mtv_camera_frame_faces_the_lens():
    """A world clip whose chest faces +y (what align_heading produces) must reach
    MTV with its left shoulder on +x and its wrists nearer the lens than its
    pelvis -- the way the training mean faces.  The first version showed MTV the
    dancer's back."""
    from render2d import mtv_motion
    mean, std = mtv_motion.load_stats()
    frames = 8
    world = np.zeros((frames, 24, 3))
    world[:, :, 2] = 1.0
    world[:, 16, 0], world[:, 17, 0] = -0.2, 0.2      # left shoulder at -x: chest faces +y
    world[:, 20:22, 1] = 0.3                         # wrists held forward (+y)
    world[:, 0, 2] = 0.9
    camera = mtv_motion.to_camera_frame(world, mean, std)
    assert mtv_motion.assert_faces_camera(camera, mean) > 0
    assert np.median(camera[:, 20:22, 2]) < np.median(camera[:, 0, 2])
    mirrored = camera.copy()
    mirrored[..., 0] *= -1
    with pytest.raises(SystemExit, match="its back"):
        mtv_motion.assert_faces_camera(mirrored, mean)
