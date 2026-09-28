"""The output-side filter: it must preserve the body, and it must be able to fail.

Smoothing can always succeed at the wrong thing -- a strong enough filter drives
every roughness statistic to ground truth by deleting the dance.  So the two
things fixed here are the two that make the tool honest: the skeleton survives,
and the cost gate rejects a filter that bought smoothness with motion.
"""
import pathlib
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.smooth_generated_motion import smooth_payload

REPO = pathlib.Path(__file__).resolve().parents[1]
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def _bone_lengths(joints):
    return np.stack([
        np.linalg.norm(joints[:, index] - joints[:, SMPL_PARENTS[index]], axis=-1)
        for index in range(1, 24)
    ])


def _clip(frames=64, seed=0):
    rng = np.random.default_rng(seed)
    # a slow trajectory plus the high-frequency component the filter is for
    time = np.linspace(0.0, 4.0, frames)[:, None, None]
    poses = 0.4 * np.sin(time) * np.ones((1, 24, 3)) + 0.05 * rng.standard_normal((frames, 24, 3))
    trans = np.stack([np.sin(time[:, 0, 0]), np.cos(time[:, 0, 0]), np.ones(frames)], axis=-1)
    return {"smpl_poses": poses.reshape(frames, 72).astype(np.float32),
            "smpl_trans": trans.astype(np.float32),
            "full_pose": np.zeros((frames, 24, 3), dtype=np.float32),
            "atomic_labels": np.zeros(frames, dtype=np.int64)}


def test_the_skeleton_survives_because_rotations_are_what_get_filtered():
    payload = _clip()
    smoothed = smooth_payload(payload, window=9, polyorder=3)
    lengths = _bone_lengths(np.asarray(smoothed["full_pose"], dtype=np.float64))
    # forward kinematics from filtered rotations keeps every bone constant
    assert float(np.max(lengths.std(axis=1) / lengths.mean(axis=1))) < 1e-5

    # the naive alternative -- filtering joint positions directly -- does not,
    # which is why this tool does not do it.
    from scipy.signal import savgol_filter
    naive = savgol_filter(np.asarray(smoothed["full_pose"], dtype=np.float64), 9, 3, axis=0)
    naive_lengths = _bone_lengths(naive)
    assert float(np.max(naive_lengths.std(axis=1) / naive_lengths.mean(axis=1))) > 1e-4


def test_filtering_removes_high_frequency_power():
    from tools.smooth_generated_motion import _high_frequency_share
    payload = _clip()
    rough = np.asarray(payload["smpl_trans"], dtype=np.float64) + \
        0.02 * np.random.default_rng(1).standard_normal(payload["smpl_trans"].shape)
    payload["smpl_trans"] = rough.astype(np.float32)
    smoothed = smooth_payload(payload, window=9, polyorder=3)
    before = _high_frequency_share(payload["smpl_trans"])
    after = _high_frequency_share(smoothed["smpl_trans"])
    assert after < before


def test_a_clip_shorter_than_the_window_is_returned_untouched():
    payload = _clip(frames=5)
    smoothed = smooth_payload(payload, window=9, polyorder=3)
    assert np.array_equal(smoothed["smpl_poses"], payload["smpl_poses"])
    assert "postprocess" not in smoothed


def test_the_cost_gate_rejects_a_filter_that_deleted_the_dance():
    """A window wide enough to flatten the motion must exit non-zero."""
    with tempfile.TemporaryDirectory() as directory:
        run = pathlib.Path(directory) / "run"
        run.mkdir()
        for index in range(3):
            payload = _clip(frames=120, seed=index)
            # real joint positions, so the reported speed is a real speed
            payload = smooth_payload(payload, window=3, polyorder=1)
            with open(run / "clip{}.pkl".format(index), "wb") as handle:
                pickle.dump(payload, handle)
        common = [sys.executable, "tools/smooth_generated_motion.py", "--run", str(run)]
        gentle = subprocess.run(
            common + ["--output", str(pathlib.Path(directory) / "gentle"),
                      "--window", "5", "--polyorder", "3", "--max-speed-loss", "0.90"],
            cwd=REPO, capture_output=True, text=True)
        assert gentle.returncode == 0
        harsh = subprocess.run(
            common + ["--output", str(pathlib.Path(directory) / "harsh"),
                      "--window", "61", "--polyorder", "3", "--max-speed-loss", "0.001"],
            cwd=REPO, capture_output=True, text=True)
        assert harsh.returncode == 1
        assert "SMOOTHING REJECTED" in harsh.stderr


def test_the_argument_checks_refuse_a_filter_that_cannot_be_what_was_asked_for():
    for flags in (["--window", "8"], ["--window", "1"], ["--window", "5", "--polyorder", "5"]):
        result = subprocess.run(
            [sys.executable, "tools/smooth_generated_motion.py", "--run", ".", "--output", "."] + flags,
            cwd=REPO, capture_output=True, text=True)
        assert result.returncode == 2
