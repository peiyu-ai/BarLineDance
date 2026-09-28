"""``--draft-floor-normalize``: put every recording's floor at the same height.

WHY, and the operator asked for it at this level.  2026-09-12: "还是看到有悬空的
问题,是不是pose 刚开始motion 提取的时候就拿高了呀 ... 不如在前面motion
extraction 那里把z 都挑一致".  Measured, that is exactly what is wrong: a
monocular reconstruction has no absolute height, and over the 295 T sequences
the floor -- the 5th percentile of the lowest foot, the same rule
``anchor_floor`` and ``render_avatar_video.floor_of`` use -- runs from -1.074 m
at the 5th percentile of recordings to -0.883 at the 95th, a **0.192 m spread**.

HOW IT REACHES THE PICTURE.  Retrieval pastes a bar from recording A into a clip
built on recording B, and ``--draft-root-continuity`` including z makes the ROOT
continuous at the seam.  That is the wrong invariant: two prototypes hold their
feet different distances below the root, so aligning the roots lifts the feet off
the floor for the rest of the bar.  Measured on 7412632116703350028:clip001, the
median height of the lowest foot above the render floor:

    ground truth                       0.072 m
    shipped arm                        0.070
    --draft-join-pose-weight 0.25      0.140
    seam-aware ranking off             0.439

and every one of those arms has its own 5th percentile ON the floor, so
``--floor-anchor`` is doing its job and what floats is everything above it.

POSITIVE CONTROL is ``test_the_height_actually_moves``.  The pair that keeps it
honest is ``test_only_the_height_moves`` -- a version that shifted x, y or the
rotations would be changing the dance, not levelling the stage.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_atomic import (CONTACT_CHANNELS, ROOT_POSITION_START,
                          IndexedAtomicMotionLibrary)

FEATURE_DIM = 16
ROOT_Z = ROOT_POSITION_START + 2


def _library(tmpdir, floors=None, sequences=("wild_v5:aaa:clip000",
                                             "wild_v5:bbb:clip000")):
    train = os.path.join(tmpdir, "train")
    os.makedirs(train, exist_ok=True)
    frames = 20
    motion = np.zeros((len(sequences), frames, FEATURE_DIM), dtype=np.float32)
    for i in range(len(sequences)):
        motion[i, :, ROOT_Z] = 0.25 * (i + 1)        # a per-recording height
        motion[i, :, ROOT_POSITION_START] = 0.5      # x, which must not move
    labels = np.ones((len(sequences), frames), dtype=np.int64)
    np.save(os.path.join(train, "motion.npy"), motion)
    np.save(os.path.join(train, "labels.npy"), labels)
    with open(os.path.join(train, "names.json"), "w") as handle:
        json.dump(["{}_slice0".format(s) for s in sequences], handle)
    # Identity normalizer: scale 1, offset 0, so metres and normalized units
    # coincide and a failure cannot be an artefact of the scaling.
    torch.save({"data_min": torch.full((FEATURE_DIM,), -1.0),
                "data_max": torch.full((FEATURE_DIM,), 1.0)},
               os.path.join(tmpdir, "normalizer.pt"))
    path = None
    if floors is not None:
        path = os.path.join(tmpdir, "floors.json")
        with open(path, "w") as handle:
            json.dump({"floors": floors}, handle)
    return IndexedAtomicMotionLibrary(tmpdir, floor_normalize=path)


def _values(library, sample):
    return library._values_at((sample, 0, 20, "g"), 20)


def test_the_height_actually_moves():
    """POSITIVE CONTROL: without it a no-op implementation passes the rest.

    Read INSIDE the temporary directory: ``_normalizer_affine`` loads the
    normalizer lazily, so a library outlives its own files only until the first
    unnormalisation.
    """
    with tempfile.TemporaryDirectory() as d:
        plain = _library(d)
        assert _values(plain, 0)[0, ROOT_Z].item() == pytest.approx(0.25)
    with tempfile.TemporaryDirectory() as d:
        levelled = _library(d, floors={"wild_v5:aaa:clip000": 0.25,
                                       "wild_v5:bbb:clip000": 0.50})
        assert _values(levelled, 0)[0, ROOT_Z].item() == pytest.approx(0.0, abs=1e-6)
        assert _values(levelled, 1)[0, ROOT_Z].item() == pytest.approx(0.0, abs=1e-6)


def test_two_recordings_end_up_on_one_floor():
    """The property: after levelling, the same pose reads the same height
    whichever recording it came from.  Before, they differ by the calibration."""
    with tempfile.TemporaryDirectory() as d:
        plain = _library(d)
        gap_before = abs(_values(plain, 0)[0, ROOT_Z].item()
                         - _values(plain, 1)[0, ROOT_Z].item())
    with tempfile.TemporaryDirectory() as d:
        levelled = _library(d, floors={"wild_v5:aaa:clip000": 0.25,
                                       "wild_v5:bbb:clip000": 0.50})
        gap_after = abs(_values(levelled, 0)[0, ROOT_Z].item()
                        - _values(levelled, 1)[0, ROOT_Z].item())
    assert gap_before == pytest.approx(0.25)
    assert gap_after == pytest.approx(0.0, abs=1e-6)


def test_only_the_height_moves():
    """x, y, the rotations and the contacts are the dance; levelling may not
    touch them or it stops being a change of stage and becomes a change of
    choreography."""
    keep = [c for c in range(FEATURE_DIM) if c != ROOT_Z]
    with tempfile.TemporaryDirectory() as d:
        plain = _values(_library(d), 0)[:, keep].clone()
    with tempfile.TemporaryDirectory() as d:
        levelled = _values(_library(d, floors={"wild_v5:aaa:clip000": 0.25,
                                               "wild_v5:bbb:clip000": 0.50}), 0)
        assert torch.allclose(plain, levelled[:, keep])


def test_it_refuses_a_recording_it_has_no_floor_for():
    """Fail closed.  One sample left at 0.0 would be the single prototype still
    carrying its own calibration, in a library the manifest calls levelled."""
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="floor"):
            _library(d, floors={"wild_v5:aaa:clip000": 0.25})


def test_the_default_is_the_published_behaviour():
    with tempfile.TemporaryDirectory() as d:
        library = _library(d)
        assert library._sample_floor is None
        assert _values(library, 1)[0, ROOT_Z].item() == pytest.approx(0.5)


def test_it_is_refused_with_a_root_continuity_that_includes_z():
    """``--draft-root-continuity xyz`` re-aligns the root at every seam, which
    is the exact operation this exists to stop."""
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run(
        [sys.executable, os.path.join(repo, "infer_atomic.py"),
         "--draft-floor-normalize", "/x.json", "--draft-root-continuity", "xyz",
         "--audio-dir", "/x", "--ingest-root", "/x", "--planner-checkpoint", "/x",
         "--completion-checkpoint", "/x", "--data-root", "/x", "--output-dir", "/x"],
        capture_output=True, text=True)
    assert done.returncode != 0
    assert "--draft-root-continuity" in (done.stderr + done.stdout)
