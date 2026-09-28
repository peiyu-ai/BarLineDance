"""render2d/aapose_video.py --yaw-fold: the pose picture never shows a back view, and never jumps (DEFECTS §96.6).

Pinned on a real frame spun about the vertical:
  * identity up to CAP-25 deg; CAP at 90 deg; a back view (180-a) drawn as the matching front view (a);
  * continuous through +-180 (a full spin moves the drawn yaw by no more than the spin's own step);
  * bone lengths and heights untouched; 0 / >= 90 is off.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from render2d.aapose_video import SMPL, fold_yaw, shoulder_yaw  # noqa: E402


def _frame():
    # a T-posed SMPL-ish skeleton facing the camera (+y chest normal toward the viewer is the renderer's convention)
    rng = np.random.default_rng(0)
    j = rng.normal(0, 0.05, (24, 3))
    j[:, 2] += np.linspace(0, 1.6, 24)
    j[SMPL["pelvis"]] = [0, 0, 0.9]
    j[SMPL["l_shoulder"]] = [0.2, 0, 1.4]
    j[SMPL["r_shoulder"]] = [-0.2, 0, 1.4]
    if abs(float(shoulder_yaw(j[None])[0])) > 90:        # make it face the camera in the renderer's own convention
        j = _spin(j, [180.0])[0]
    return j


def _spin(frame, degrees):
    out = []
    for d in degrees:
        a = math.radians(d)
        c, s = math.cos(a), math.sin(a)
        x = frame.copy()
        p = x[SMPL["pelvis"], :2].copy()
        xy = x[:, :2] - p
        x[:, 0] = p[0] + c * xy[:, 0] - s * xy[:, 1]
        x[:, 1] = p[1] + s * xy[:, 0] + c * xy[:, 1]
        out.append(x)
    return np.stack(out)


def test_bands_and_fold():
    frame = _frame()
    spin = _spin(frame, np.arange(0, 360, 1.0))
    before = shoulder_yaw(spin)
    after = shoulder_yaw(fold_yaw(spin, 60))
    base = before[0]
    rel = (before - base + 180) % 360 - 180
    small = np.abs(rel) < 30
    assert np.allclose(np.abs(after[small] - base), np.abs(rel[small]), atol=1.0)      # identity below the knee
    assert np.abs(after - base).max() <= 60.5                                         # never past the cap
    back = np.abs(np.abs(rel) - 180) < 1
    assert np.all(np.abs(after[back] - base) < 2.0)                                   # a back view drawn facing front


def test_continuous_through_the_back():
    spin = _spin(_frame(), np.arange(0, 720, 1.0))
    after = np.unwrap(np.radians(shoulder_yaw(fold_yaw(spin, 60))))
    assert np.degrees(np.abs(np.diff(after))).max() <= 1.05


def test_geometry_untouched_and_off():
    spin = _spin(_frame(), np.arange(0, 360, 7.0))
    out = fold_yaw(spin, 60)
    assert np.allclose(out[..., 2], spin[..., 2])
    assert np.allclose(np.linalg.norm(out[:, 1] - out[:, 4], axis=-1), np.linalg.norm(spin[:, 1] - spin[:, 4], axis=-1))
    assert np.array_equal(fold_yaw(spin, 0), spin)
    assert np.array_equal(fold_yaw(spin, 90), spin)
