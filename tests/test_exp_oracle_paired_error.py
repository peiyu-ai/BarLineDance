"""Controls for the paired-to-ground-truth columns (tools/exp_oracle_paired_error.py).

These two columns are the only ones in the experiment that are NEW, so they are
the ones CLAUDE.md 2.1 says must be checked in both directions before they are
allowed to decide anything:

* the positive with the right direction -- a body that IS the ground truth must
  read mpjpe 0.000 and speed correlation 1.000, and a body that is the ground
  truth shifted in space must read 0.000 root-relative and non-zero global, or
  the root-relative column is quietly measuring travel;
* the negative -- an unrelated body must read a large mpjpe, and a time-reversed
  ground truth must lose the correlation while keeping the SAME distribution of
  speeds, which is the case a distributional scorer cannot see and this one must;
* the power proof -- the seam mask must actually land on the completion window
  junctions, otherwise "the error is not at the seams" would be a statement
  about an empty set.
"""

import pathlib
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import exp_oracle_paired_error as m  # noqa: E402


def _walk(frames=300, seed=0, hz=0.7):
    """A body whose joints move smoothly and whose root travels.

    ``hz`` is a parameter and not a constant because the first version of this
    fixture held it fixed and only re-rolled the per-joint PHASES: averaged over
    24 joints that leaves one shared 1.4 Hz speed envelope, so two "unrelated"
    bodies correlated at 0.945 and the negative control failed.  The tool was
    right and the fixture was wrong -- two dances at the same tempo with
    different phases DO share a speed envelope.  A genuinely unrelated body
    needs a different tempo.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(frames)[:, None, None] / m.FPS
    phase = rng.uniform(0, 2 * np.pi, size=(1, 24, 3))
    joints = 0.4 * np.sin(2 * np.pi * hz * time + phase)
    joints[:, 0, :2] += np.stack([0.3 * time[:, 0, 0], 0.1 * time[:, 0, 0]], 1)
    return joints


def _write(directory, name, joints):
    directory.mkdir(parents=True, exist_ok=True)
    with open(str(directory / (name + ".pkl")), "wb") as handle:
        pickle.dump({"full_pose": np.asarray(joints, np.float32)}, handle)


def test_identity_reads_zero_error_and_unit_correlation(tmp_path):
    truth = tmp_path / "truth"
    _write(truth, "a", _walk())
    row = m.score_arm(truth, ["a"], truth, labels_root=None, window=150,
                      stride=75, seam_radius=5)
    assert row["mpjpe_rootrel"] == pytest.approx(0.0, abs=1e-9)
    assert row["mpjpe_global"] == pytest.approx(0.0, abs=1e-9)
    assert row["speed_corr"] == pytest.approx(1.0, abs=1e-9)
    assert row["speed_corr_smoothed"] == pytest.approx(1.0, abs=1e-9)


def test_a_translated_body_is_zero_root_relative_and_non_zero_global(tmp_path):
    """The root-relative column must not be a travel column in disguise."""
    truth, arm = tmp_path / "truth", tmp_path / "arm"
    joints = _walk()
    _write(truth, "a", joints)
    shifted = joints.copy()
    shifted[:, :, 0] += 1.25
    _write(arm, "a", shifted)
    row = m.score_arm(arm, ["a"], truth, labels_root=None, window=150,
                      stride=75, seam_radius=5)
    assert row["mpjpe_rootrel"] == pytest.approx(0.0, abs=1e-6)
    assert row["mpjpe_global"] == pytest.approx(1.25, abs=1e-6)
    assert row["speed_corr"] == pytest.approx(1.0, abs=1e-6)


def test_an_unrelated_body_reads_large_error_and_no_correlation(tmp_path):
    truth, arm = tmp_path / "truth", tmp_path / "arm"
    _write(truth, "a", _walk(seed=1, hz=0.7))
    _write(arm, "a", _walk(seed=99, hz=1.9))
    row = m.score_arm(arm, ["a"], truth, labels_root=None, window=150,
                      stride=75, seam_radius=5)
    assert row["mpjpe_rootrel"] > 0.1
    assert abs(row["speed_corr"]) < 0.5


def test_time_reversal_keeps_the_speed_distribution_and_loses_the_correlation(tmp_path):
    """The case a distributional column cannot see, and this one must.

    A reversed clip has EXACTLY the ground truth's set of speeds -- same mean,
    same p90/p10, same hold share -- and none of its timing.  If the correlation
    column could not tell the two apart it would be measuring amplitude again.
    """
    truth, arm = tmp_path / "truth", tmp_path / "arm"
    joints = _walk(seed=7)
    _write(truth, "a", joints)
    _write(arm, "a", joints[::-1].copy())
    row = m.score_arm(arm, ["a"], truth, labels_root=None, window=150,
                      stride=75, seam_radius=5)
    forward = m.joint_speed(joints)
    backward = m.joint_speed(joints[::-1].copy())
    assert float(forward.mean()) == pytest.approx(float(backward.mean()), rel=1e-6)
    assert abs(row["speed_corr"]) < 0.6


def test_seam_mask_lands_on_the_completion_window_junctions():
    """Power proof: the mask must be non-empty and centred where the stitch joins."""
    flag = m.seam_frames(437, 150, 75, 5)
    assert flag.any() and not flag.all()
    # Window starts 0,75,150,225,287; junctions at start + overlap//2 = start + 37.
    for junction in (75 + 37, 150 + 37, 225 + 37):
        assert flag[junction]
        assert flag[junction - 5] and flag[junction + 5]
        assert not flag[junction - 6]
    # A clip shorter than one window has no junction at all, and says so.
    assert not m.seam_frames(120, 150, 75, 5).any()


def test_shuffled_null_never_pairs_a_clip_with_itself(tmp_path):
    truth = tmp_path / "truth"
    for index, name in enumerate("abcde"):
        _write(truth, name, _walk(frames=200, seed=index, hz=0.5 + 0.37 * index))
    null = m.shuffled_speed_null(list("abcde"), truth, seed=5)
    assert null["pairs"] == 5
    # Five unrelated synthetic bodies: the null must sit far below 1.0.
    assert abs(null["speed_corr"]) < 0.6
