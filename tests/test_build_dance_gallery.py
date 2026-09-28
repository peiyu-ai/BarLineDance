"""Tests for the gallery's wild path and its cross-run sampler check.

Three things here can fail silently in a way a viewer cannot see, so those are
what is pinned:

* a wild clip resolving to *no* music still produces a watchable video, and the
  wild eval directory is laid out so that a naive flat lookup finds a file
  (a 35-D ``.npy`` under exactly the flat name) that is not music at all;
* a stack whose top row is missing looks like a stack whose top row is a model;
* two arms sampled with different plan strides look like two checkpoints.
"""

import json
import pathlib
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.build_dance_gallery import (  # noqa: E402
    CONTACT_JOINTS,
    FPS,
    GROUND_BAND_M,
    GalleryError,
    check_sampling,
    decode_ground_truth,
    index_clip_ground_truth,
    motion_diagnostics,
    resolve_audio,
)

CLIP = "wild_v4:7027456421100784900:clip001"


def _sliding_dancer(frames=60, slide_per_frame=0.01, foot_z=0.0):
    """A body whose feet stay on the floor and translate sideways every frame."""
    poses = np.zeros((frames, 24, 3))
    poses[:, :, 2] = 1.0            # everything else well clear of the ground
    poses[:, 0, 2] = 1.0            # root
    for joint in CONTACT_JOINTS:
        poses[:, joint, 2] = foot_z
        poses[:, joint, 0] = np.arange(frames) * slide_per_frame
    return poses


def _write_run(directory, sampling):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps({"samples": 1, "sampling": sampling}), encoding="utf-8"
    )
    return directory


def test_ingest_layout_finds_the_clip_directorys_wav(tmp_path):
    ingest = tmp_path / "wild_ingest_v1" / "7027456421100784900__clip001"
    ingest.mkdir(parents=True)
    (ingest / "audio.wav").write_bytes(b"RIFF")
    found = resolve_audio(tmp_path / "wild_ingest_v1", CLIP, "ingest")
    assert found == ingest / "audio.wav"


def test_ingest_layout_does_not_fall_back_to_the_flat_name(tmp_path):
    """The trap this exists to avoid: the flat name is taken by a feature array.

    ``runs/<tag>_gt_eval/audio`` holds the planner's 35-D conditioning as
    ``<clip>.npy`` -- same directory shape as a WAV directory.  A fallback would
    resolve, fail to mux, and leave a silent video that still looked configured.
    """
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "{}.npy".format(CLIP)).write_bytes(b"\x93NUMPY")
    (audio / "{}.wav".format(CLIP)).write_bytes(b"RIFF")
    assert resolve_audio(audio, CLIP, "ingest") is None
    assert resolve_audio(audio, CLIP, "flat") == audio / "{}.wav".format(CLIP)


def test_a_clip_id_of_another_shape_resolves_to_nothing_rather_than_a_neighbour(tmp_path):
    ingest = tmp_path / "ingest"
    (ingest / "7027456421100784900__clip001").mkdir(parents=True)
    (ingest / "7027456421100784900__clip001" / "audio.wav").write_bytes(b"RIFF")
    assert resolve_audio(ingest, "7027456421100784900__clip001", "ingest") is None


def test_unknown_layout_is_an_error_not_a_silent_miss(tmp_path):
    with pytest.raises(GalleryError):
        resolve_audio(tmp_path, CLIP, "by_song")


def test_missing_ground_truth_is_reported_not_skipped(tmp_path):
    truth = tmp_path / "motion"
    truth.mkdir()
    (truth / "{}.pkl".format(CLIP)).write_bytes(b"")
    found = index_clip_ground_truth(truth, [CLIP, "wild_v4:1:clip000"])
    assert set(found) == {CLIP}
    assert found[CLIP]["motion_path"] == truth / "{}.pkl".format(CLIP)


def test_pkl_ground_truth_is_truncated_to_the_generated_length(tmp_path):
    poses = np.random.default_rng(0).normal(size=(40, 24, 3)).astype(np.float32)
    payload = {
        "full_pose": poses,
        "contacts": np.ones((40, 4), dtype=np.float32),
        "smpl_trans": poses[:, 0, :].copy(),
    }
    path = tmp_path / "{}.pkl".format(CLIP)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    decoded, out = decode_ground_truth({"motion_path": path}, max_frames=25)
    assert decoded.shape == (25, 24, 3)
    # The payload has to be cut with it: motion_diagnostics indexes contacts by
    # frame, so a 40-frame contact channel beside a 25-frame body would misalign
    # the foot-skate mask rather than raise.
    assert out["contacts"].shape == (25, 4)
    assert out["smpl_trans"].shape == (25, 3)
    np.testing.assert_array_equal(decoded, poses[:25])


def test_geometric_skate_exists_where_the_contact_channel_does_not(tmp_path):
    """The reason this statistic was added: the wild ground truth has no contacts.

    ``export_wild_eval_motion`` writes ``full_pose`` and nothing else, so the
    contact-channel column is blank on the only row a reader wants to compare
    against.  The geometric column has to survive that.
    """
    poses = _sliding_dancer(slide_per_frame=0.01)
    stats = motion_diagnostics(poses, {})          # no payload at all
    assert "foot_skate_m_per_s" not in stats       # nothing to be self-consistent with
    assert stats["grounded_fraction_geometric"] == 1.0
    assert stats["foot_skate_geometric_m_per_s"] == pytest.approx(0.01 * FPS, rel=1e-6)


def test_a_foot_above_the_ground_band_is_stepping_not_skating():
    poses = _sliding_dancer(slide_per_frame=0.01, foot_z=GROUND_BAND_M + 0.2)
    stats = motion_diagnostics(poses, {})
    # The floor percentile is taken from this clip's own feet, so lifting every
    # foot together lifts the floor with them -- what must NOT happen is the
    # statistic quietly vanishing, which is how a silent zero would read.
    assert stats["grounded_fraction_geometric"] == 1.0

    # A clip that spends most of its frames airborne relative to its own floor.
    airborne = _sliding_dancer(frames=40, slide_per_frame=0.01)
    airborne[10:, CONTACT_JOINTS, 2] = 1.0
    stats = motion_diagnostics(airborne, {})
    assert stats["grounded_fraction_geometric"] == pytest.approx(10 / 39, abs=1e-3)


def test_the_two_skate_columns_are_computed_on_different_frames(tmp_path):
    """A clip whose contact channel disagrees with where its feet actually are.

    This is the case that makes averaging the two columns wrong, so it is pinned:
    the contact channel says grounded for the airborne half, geometry says
    grounded for the other half, and the two numbers must come out different.
    """
    poses = _sliding_dancer(frames=40, slide_per_frame=0.0)
    poses[20:, CONTACT_JOINTS, 0] = np.arange(20)[:, None] * 0.02   # slides only late
    poses[20:, CONTACT_JOINTS, 2] = 1.0                             # ...and airborne then
    contacts = np.zeros((40, len(CONTACT_JOINTS)))
    contacts[20:] = 1.0                                             # channel says grounded
    stats = motion_diagnostics(poses, {"contacts": contacts})
    assert stats["foot_skate_m_per_s"] > 0.5      # sliding, by its own channel
    assert stats["foot_skate_geometric_m_per_s"] == pytest.approx(0.0, abs=1e-9)


def test_sampler_mismatch_stops_the_stack(tmp_path):
    a = _write_run(tmp_path / "a", {"seed": 1, "plan_stride": 15, "plan_fusion": "vote"})
    b = _write_run(tmp_path / "b", {"seed": 1, "plan_stride": 150, "plan_fusion": "none"})
    with pytest.raises(GalleryError) as error:
        check_sampling([("150", a), ("340", b)])
    assert "plan_stride" in str(error.value)

    report = check_sampling([("150", a), ("340", b)], strict=False)
    assert set(report["differing"]) == {"plan_stride", "plan_fusion"}


def test_two_seeds_of_one_arm_are_a_legitimate_stack(tmp_path):
    a = _write_run(tmp_path / "a", {"seed": 1, "plan_stride": 15, "plan_fusion": "vote"})
    b = _write_run(tmp_path / "b", {"seed": 2, "plan_stride": 15, "plan_fusion": "vote"})
    report = check_sampling([("seed 1", a), ("seed 2", b)])
    assert report["differing"] == {}
    assert report["seeds"] == {"seed 1": 1, "seed 2": 2}


def test_a_run_with_no_manifest_is_named_rather_than_assumed_to_match(tmp_path):
    a = _write_run(tmp_path / "a", {"seed": 1, "plan_stride": 15})
    b = tmp_path / "b"
    b.mkdir()
    report = check_sampling([("150", a), ("340", b)])
    assert report["runs_without_manifest"] == ["340"]


def test_the_plan_shaping_flags_added_in_august_are_checked_too(tmp_path):
    """Two arms differing only in the bar grid used to stack in silence.

    ``SAMPLING_KEYS`` was written before 2026-08-23 and the plan gained five
    more knobs that day, every one of which changes what is on screen.  The
    guard exists so a reader does not credit a sampler difference to the
    checkpoint; a knob it does not name is a knob that difference passes
    through.  ``run_m6_wild.sh`` measures the size of these: the tie-break alone
    was 4.6 of the 22.0 points by which one arm's plan overshot the ground
    truth's transition share.
    """
    base = {"seed": 1, "plan_stride": 15, "plan_fusion": "vote",
            "plan_bar_grid": True, "plan_vote_tie_break": "centre",
            "plan_transition_policy": "protect", "plan_merge_order": "shortest",
            "planner_transition_logit_bias": 0.0}
    for key, other in (("plan_bar_grid", False),
                       ("plan_vote_tie_break", "index"),
                       ("plan_transition_policy", "merge"),
                       ("plan_merge_order", "first"),
                       ("planner_transition_logit_bias", -1.0)):
        a = _write_run(tmp_path / ("a_" + key), dict(base))
        b = _write_run(tmp_path / ("b_" + key), dict(base, **{key: other}))
        with pytest.raises(GalleryError) as error:
            check_sampling([("on", a), ("off", b)])
        assert key in str(error.value)
        report = check_sampling([("on", a), ("off", b)], strict=False)
        assert set(report["differing"]) == {key}


def test_two_arms_that_agree_on_every_plan_flag_still_stack(tmp_path):
    """Positive control: widening the list must not refuse a legitimate stack."""
    base = {"seed": 1, "plan_stride": 15, "plan_fusion": "vote",
            "plan_bar_grid": True, "plan_vote_tie_break": "centre",
            "plan_transition_policy": "protect", "plan_merge_order": "shortest",
            "planner_transition_logit_bias": 0.0}
    a = _write_run(tmp_path / "a", dict(base, seed=1))
    b = _write_run(tmp_path / "b", dict(base, seed=2))
    report = check_sampling([("seed 1", a), ("seed 2", b)])
    assert report["differing"] == {}
