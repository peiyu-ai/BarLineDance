"""Decoding a release window must reproduce the eval export on the same frames.

THE DEFECT THIS PINS.  The release is normalised to [-1, 1]; ``_normalizer_affine``
is the authority and returns ``(span/2, low + span/2)``.  Three separate places
used the [0, 1] inverse ``x*span + low`` instead, which scales every channel 2x
and shifts it.  Nothing failed: the decode returns a plausible skeleton with
plausible limb lengths, so every reading taken from it looked like a measurement
of the library when it was a measurement of a doubled, shifted impostor.

On 2026-09-16 that produced four wrong findings in one session -- "the library's
prototypes turn less than ground truth", "34.8% of library windows hold as much
as ground truth's p75 bar", "77.7% of library windows exceed ground truth's peak
ankle speed", and a named window's speed -- and it also silently corrupted
``tools/census_release_yaw_steps.py``, whose npz feeds --retrieval-max-yaw-step
and --retrieval-max-speed-spike.

THE CHECK IS END TO END, because that is the only kind this class of error
cannot survive: the eval export and the release are built from the same source,
the release's own music window matches the export's music at the slice offset
EXACTLY, so the decoded joints must match too.  With the right affine the
root-relative pose error is 0.00000 m; with the wrong one it is 0.39959 m.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

RELEASE = pathlib.Path("/cache/atomicdance-assets/scratch/txy_t/release_aligned_k8")
EXPORT = pathlib.Path("runs/txy_t_gt_eval/motion")
AUDIO = pathlib.Path("runs/txy_t_gt_eval/audio")
STRIDE = 15
BEAT = 34


def _pair():
    if not (RELEASE / "test/names.json").is_file() or not EXPORT.is_dir():
        pytest.skip("release or eval export not staged")
    names = json.loads((RELEASE / "test/names.json").read_text())
    for index, name in enumerate(names):
        base, _, tail = name.rpartition("_slice")
        pkl = EXPORT / (base + ".pkl")
        if tail.isdigit() and pkl.is_file():
            return index, name, base, int(tail)
    pytest.skip("no release window has a matching eval export")


def _affine():
    import torch
    normalizer = torch.load(RELEASE / "normalizer.pt", map_location="cpu",
                            weights_only=False)
    low = normalizer["data_min"].numpy()
    span = (normalizer["data_max"].numpy() - low).copy()
    span[span == 0] = 1.0
    return span / 2.0, low + span / 2.0


def test_the_music_window_lands_where_the_slice_index_says():
    """The premise the motion assertion rests on: the frames DO correspond."""
    index, _, base, slice_index = _pair()
    music = np.load(str(RELEASE / "test/music.npy"), mmap_mode="r")[index]
    export = np.load(str(AUDIO / (base + ".npy")))
    start = slice_index * STRIDE
    assert np.abs(np.asarray(music) - export[start:start + len(music)]).max() < 1e-6


def test_decoding_a_release_window_reproduces_the_export():
    from tools.render_dance_video import _decode_raw_151
    from tools.score_beat_phase_profile import joints_of

    index, _, base, slice_index = _pair()
    scale, offset = _affine()
    raw = np.asarray(np.load(str(RELEASE / "test/motion.npy"), mmap_mode="r")[index])
    joints, _ = _decode_raw_151(raw * scale[None, :] + offset[None, :])
    joints = np.asarray(joints, np.float64)

    start = slice_index * STRIDE
    export = joints_of(str(EXPORT / (base + ".pkl")))[start:start + len(joints)]
    if len(export) < len(joints):
        pytest.skip("export shorter than the window")
    relative = lambda x: x - x[:, :1, :]
    error = float(np.linalg.norm(relative(joints) - relative(export), axis=2).mean())
    assert error < 1e-3, (
        "decoded release window differs from the eval export by {:.5f} m per "
        "joint; the [0,1] inverse reads 0.39959 here and the [-1,1] one reads "
        "0.00000".format(error))


def test_the_census_tool_uses_the_authoritative_affine():
    source = pathlib.Path("tools/census_release_yaw_steps.py").read_text()
    assert "span / 2.0" in source and "low + span / 2" in source, (
        "tools/census_release_yaw_steps.py feeds --retrieval-max-yaw-step and "
        "--retrieval-max-speed-spike; decoding it with the [0,1] inverse makes "
        "every threshold a threshold on a doubled skeleton")
    assert "* scale[None, :] + low[None, :]" in source
