"""tools/score_floor_alignment.py -- every column with a known answer.

The repository's rule is that a criterion must be validated BEFORE it judges
(CLAUDE.md 2.1): it needs a positive sample whose reading is known, not just an
input on which it fails.  The three cases here are the three things this tool
claims to separate, and each has an answer that arithmetic fixes in advance:

  * ground truth against ITSELF must read exactly zero everywhere -- the
    identity is the positive control;
  * a RIGID vertical shift must move ``floor`` and nothing else, because every
    other column is measured relative to the clip's own floor.  This is the
    property ``--floor-anchor`` relies on ("because it is rigid, it cannot
    change the dance"), so a tool that let a rigid shift leak into ``hover`` or
    ``drift`` would blame the anchor for the stitching's defect;
  * an ACCUMULATING drift must move ``drift`` while leaving ``floor`` alone,
    which is the separation the whole 2026-09-14 diagnosis rests on.
"""
import pathlib
import pickle
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "score_floor_alignment.py"
CLIP = "wild_v5:1234567890123456789:clip000.pkl"


def _dance(frames=300, seed=0):
    """A body that stands, steps and lifts a foot -- not a constant pose."""
    rng = np.random.default_rng(seed)
    joints = np.zeros((frames, 24, 3))
    time = np.arange(frames) / 30.0
    joints[:, 0, 2] = 0.95 + 0.03 * np.sin(2 * np.pi * time)          # pelvis
    for index in (7, 10):                                              # left foot
        joints[:, index, 2] = 0.05 + 0.20 * np.clip(np.sin(2 * np.pi * time), 0, None)
    for index in (8, 11):                                              # right foot
        joints[:, index, 2] = 0.05 + 0.20 * np.clip(-np.sin(2 * np.pi * time), 0, None)
    joints += 0.001 * rng.standard_normal(joints.shape)
    return joints


def _write(directory, joints):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / CLIP, "wb") as handle:
        pickle.dump({"full_pose": joints}, handle)


def _run(truth, **arms):
    command = [sys.executable, str(TOOL), "--truth", str(truth)]
    for name, directory in arms.items():
        command += ["--arm", "{}={}".format(name, directory)]
    out = subprocess.run(command, capture_output=True, text=True, check=True,
                         cwd=str(REPO)).stdout
    readings = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 3 and parts[0] in ("floor_err", "hover", "drift",
                                           "pelvis", "foot"):
            value = parts[2] if parts[0] == "floor_err" else parts[3]
            readings[parts[0]] = float(value.rstrip("%m").rstrip())
    return out, readings


def test_identity_reads_zero(tmp_path):
    truth = tmp_path / "truth"
    _write(truth, _dance())
    arm = tmp_path / "arm"
    _write(arm, _dance())
    out, readings = _run(truth, same=arm)
    for column in ("floor_err", "hover", "drift", "pelvis", "foot"):
        assert abs(readings[column]) < 1e-6, (column, readings[column], out)


def test_rigid_shift_moves_only_the_floor(tmp_path):
    truth = tmp_path / "truth"
    _write(truth, _dance())
    arm = tmp_path / "arm"
    joints = _dance()
    joints[:, :, 2] += 0.20                       # the whole body, every frame
    _write(arm, joints)
    out, readings = _run(truth, shifted=arm)
    assert abs(readings["floor_err"] - 0.20) < 1e-3, (readings, out)
    for column in ("hover", "drift", "pelvis", "foot"):
        assert abs(readings[column]) < 1e-6, (column, readings[column], out)


def test_accumulating_drift_shows_in_drift_not_in_floor(tmp_path):
    truth = tmp_path / "truth"
    _write(truth, _dance())
    arm = tmp_path / "arm"
    joints = _dance()
    # The stitching failure in miniature: each unit starts where the last one
    # ended, a little higher every time, with nothing pulling it back.
    ramp = np.linspace(0.0, 0.40, len(joints))[:, None, None]
    joints[:, :, 2:3] += ramp
    _write(arm, joints)
    out, readings = _run(truth, drifting=arm)
    assert readings["drift"] > 0.30, (readings, out)
    # The 5th percentile of the lowest foot still sits near the start of the
    # ramp, so the anchor's own column barely moves: the defect is invisible to
    # it, which is exactly why it needed a column of its own.
    assert abs(readings["floor_err"]) < 0.10, (readings, out)
