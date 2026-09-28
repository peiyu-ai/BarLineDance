"""Tests for the generated-motion scoring bundle.

The tool's whole claim is that the arrays it writes hold the same motion the
checkpoint produced and FID scored, so the tests are built around that claim:
the rebuilt 151-D must invert ``decode_motion``'s layout, the check against the
artifact's own joints must fire when the motion is not the same motion, and the
gates that stand in front of the expensive work must refuse before they reach it.
"""

import json
import pathlib
import pickle
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.convert_motion_to_guofeats import motion_151_to_joints  # noqa: E402
from tools.materialize_generated_release import (  # noqa: E402
    MaterialisationError,
    check_against_stored_joints,
    materialise,
    normalize,
    rebuild_motion_151,
    unnormalize,
    window_policy,
    window_starts,
)


def _synthetic_generation(frames=20, seed=0):
    """A raw 151-D array and the pkl payload ``decode_motion`` would write for it."""
    from dataset.quaternion import ax_from_6v, ax_to_6v

    rng = np.random.default_rng(seed)
    contacts = rng.uniform(0.0, 1.0, size=(frames, 4)).astype(np.float32)
    trans = rng.normal(size=(frames, 3)).astype(np.float32) * 0.1
    axis_angle = rng.normal(size=(frames, 24, 3)).astype(np.float32) * 0.3
    rot6d = ax_to_6v(torch.from_numpy(axis_angle)).reshape(frames, 144).numpy()
    raw = np.concatenate([contacts, trans, rot6d], axis=-1).astype(np.float32)
    payload = {
        "contacts": contacts,
        "smpl_trans": trans,
        "smpl_poses": ax_from_6v(
            torch.from_numpy(rot6d).reshape(-1, 24, 6)).reshape(frames, 72).numpy(),
        "full_pose": motion_151_to_joints(raw),
        "headline_eligible": True,
        "generation_protocol": "SELF_DRIVEN_PLANNER",
        "audio_path": "/nowhere/mBR2.wav",
    }
    return raw, payload


def _release(tmp_path, policy={"window_length": 150, "window_stride": 15}):
    root = tmp_path / "release"
    root.mkdir(parents=True)
    body = {"window_policy": dict(policy)} if policy is not None else {}
    (root / "build.json").write_text(json.dumps(body), encoding="utf-8")
    torch.save({"data_min": torch.zeros(151), "data_max": torch.ones(151)},
               root / "normalizer.pt")
    return root


def test_rebuild_inverts_the_layout_decode_motion_reads():
    raw, payload = _synthetic_generation()
    rebuilt = rebuild_motion_151(payload)
    assert rebuilt.shape == raw.shape
    # The rot6d round trip goes through a rotation matrix, so the tolerance sits
    # where float32 trigonometry lives rather than at exact equality.
    assert np.abs(rebuilt - raw).max() < 1e-5


def test_rebuilt_motion_agrees_with_the_artifacts_own_joints():
    _, payload = _synthetic_generation(seed=1)
    error = check_against_stored_joints(rebuild_motion_151(payload), payload["full_pose"])
    assert error < 1e-4


def test_the_joint_check_fires_when_the_motion_is_not_the_same_motion():
    _, payload = _synthetic_generation(seed=2)
    moved = np.asarray(payload["full_pose"], dtype=np.float64).copy()
    moved[:, 0, :] += 0.05          # 5 cm of root drift, well inside "looks fine"
    error = check_against_stored_joints(rebuild_motion_151(payload), moved)
    assert error > 1e-4


def test_the_joint_check_refuses_a_shape_it_cannot_compare():
    _, payload = _synthetic_generation(seed=3)
    with pytest.raises(MaterialisationError, match="do not match"):
        check_against_stored_joints(rebuild_motion_151(payload),
                                    np.zeros((5, 24, 3)))


def test_rebuild_refuses_arrays_that_are_not_the_expected_shape():
    _, payload = _synthetic_generation(seed=4)
    payload["smpl_poses"] = payload["smpl_poses"][:, :69]
    with pytest.raises(MaterialisationError, match="smpl_poses"):
        rebuild_motion_151(payload)


def test_normaliser_round_trips_in_the_direction_the_scorer_uses_it():
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(7, 151)).astype(np.float32)
    data_min = rng.normal(size=151).astype(np.float32)
    safe_range = np.abs(rng.normal(size=151)).astype(np.float32) + np.float32(0.5)
    normalized = normalize(raw, data_min, safe_range)
    assert np.abs(unnormalize(normalized, data_min, safe_range) - raw).max() < 1e-4


def test_values_outside_the_training_range_survive_normalisation():
    # Generated motion may leave the range the normaliser was fitted on.  Clipping
    # would move the body and report nothing, so the round trip has to hold there
    # too -- that is what makes the reported out-of-range fraction meaningful.
    data_min = np.zeros(151, dtype=np.float32)
    safe_range = np.ones(151, dtype=np.float32)
    raw = np.full((2, 151), 3.0, dtype=np.float32)
    normalized = normalize(raw, data_min, safe_range)
    assert normalized.max() > 1.0
    assert np.abs(unnormalize(normalized, data_min, safe_range) - raw).max() < 1e-5


def test_window_starts_drops_the_tail_rather_than_shortening_a_window():
    assert window_starts(300, 150, 15) == list(range(0, 151, 15))
    assert window_starts(149, 150, 15) == []
    assert window_starts(150, 150, 15) == [0]


def test_window_policy_is_read_from_the_release_and_never_defaulted(tmp_path):
    policy = window_policy(_release(tmp_path))
    assert (policy["window_length"], policy["window_stride"]) == (150, 15)
    assert policy["source"] == "reference release build.json"
    with pytest.raises(MaterialisationError, match="window_length and window_stride"):
        window_policy(_release(tmp_path / "bare", policy=None))


def test_one_window_per_sequence_is_a_named_override_that_records_itself(tmp_path):
    """R swings 7.7x with pool size, so the pool has to be declared, not inherited.

    The stride is pushed past any generated sequence so window_starts yields the
    first window only -- and both the new value and the one it replaced land in
    the returned policy, so a bundle always says which ruler cut it.
    """
    release = _release(tmp_path)
    policy = window_policy(release, one_window_per_sequence=True)
    assert policy["one_window_per_sequence"] is True
    assert policy["window_stride_overridden_from"] == 15
    assert window_starts(2000, policy["window_length"], policy["window_stride"]) == [0]
    # Untouched without the flag: an override nobody asked for is the failure
    # this whole function exists to prevent.
    assert "one_window_per_sequence" not in window_policy(release)


def test_window_length_override_is_recorded_and_validated(tmp_path):
    release = _release(tmp_path)
    policy = window_policy(release, override_length=340)
    assert policy["window_length"] == 340
    assert policy["window_length_overridden_from"] == 150
    with pytest.raises(MaterialisationError, match="must be positive"):
        window_policy(release, override_length=0)


def test_materialise_refuses_an_artifact_that_is_not_headline_eligible(tmp_path):
    _, payload = _synthetic_generation(frames=200, seed=6)
    payload["headline_eligible"] = False
    payload["generation_protocol"] = "ORACLE_GROUND_TRUTH_PLAN"
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    with open(motion_dir / "mBR2_s1.pkl", "wb") as handle:
        pickle.dump(payload, handle)
    # The refusal has to come before the audio is opened, which is also why this
    # test can run without any: an oracle-plan artifact must never reach the
    # arrays a model result is read from.
    with pytest.raises(MaterialisationError, match="not headline eligible"):
        materialise(motion_dir, _release(tmp_path), tmp_path / "bundle", "test")
    assert not (tmp_path / "bundle").exists()


def test_materialise_refuses_when_there_is_nothing_to_materialise(tmp_path):
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    with pytest.raises(MaterialisationError, match="no generated"):
        materialise(motion_dir, _release(tmp_path), tmp_path / "bundle", "test")
