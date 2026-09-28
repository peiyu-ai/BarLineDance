"""Controls for the completion checkpoint sweep's columns and window draw.

Every test here is one of the four things CLAUDE.md 2.1 demands of a criterion
before it is allowed to judge:

  * PROVENANCE      the facing columns come from ``tools/score_facing_spin``'s
                    hip-line azimuth and the stillness column from
                    ``tools/exp_guidance_stillness``'s verified hold criterion;
                    ``test_facing_matches_score_facing_spin`` and
                    ``test_still_share_matches_stillness_tool`` pin that they
                    are the same measurement and not a look-alike.
  * POSITIVE CONTROL a body turning at a known constant rate must read that
                    rate, that travel, and directedness 1.0.
  * PROOF OF POWER  a frozen body must read 0 and be called still; an
                    oscillating body must read a LOW directedness while turning
                    just as fast -- so the column separates the two things the
                    median otherwise merges.
  * THE DISAGREEMENT the legacy x-z plane must FAIL the world-yaw invariance
                    control while the ground plane passes.  If that ever stops
                    being true, sections 24/25's numbers are not what this file
                    says they are.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import exp_completion_ckpt_sweep as sweep  # noqa: E402


def upright_body(frames):
    """A 24-joint skeleton standing at the origin, Z up, facing +y.

    Only the joints the columns read have to be right: hips (1, 2) across x,
    shoulders (16, 17) across x and higher, and a head/foot pair so the body
    has vertical extent.
    """
    joints = np.zeros((frames, 24, 3))
    joints[:, 1, :] = (-0.09, 0.0, 0.0)
    joints[:, 2, :] = (0.09, 0.0, 0.0)
    joints[:, 16, :] = (-0.18, 0.0, 0.45)
    joints[:, 17, :] = (0.18, 0.0, 0.45)
    joints[:, 15, :] = (0.0, 0.0, 0.65)
    joints[:, 10, :] = (-0.09, 0.0, -0.85)
    joints[:, 11, :] = (0.09, 0.0, -0.85)
    return joints


def spin_about_z(joints, angles):
    cos, sin = np.cos(angles), np.sin(angles)
    rotation = np.zeros((len(angles), 3, 3))
    rotation[:, 0, 0] = cos
    rotation[:, 0, 1] = -sin
    rotation[:, 1, 0] = sin
    rotation[:, 1, 1] = cos
    rotation[:, 2, 2] = 1.0
    return np.einsum("tij,tkj->tki", rotation, np.asarray(joints, float))


# --------------------------------------------------------- positive controls

def test_constant_turn_reads_its_own_rate_travel_and_directedness_one():
    """POSITIVE CONTROL with a known answer: 45 deg/s for 4 s is 180 deg."""
    frames = int(4 * sweep.FPS) + 1
    angles = np.radians(np.arange(frames) * 45.0 / sweep.FPS)
    joints = spin_about_z(upright_body(frames), angles)
    turn, travel, directed = sweep.facing_columns(joints)
    assert turn == pytest.approx(45.0, abs=1e-6)
    assert travel == pytest.approx(180.0, abs=1e-6)
    # Every degree turned is a degree of net progress: this is the spin mode.
    assert directed == pytest.approx(1.0, abs=1e-9)


def test_oscillation_turns_as_fast_but_is_not_directed():
    """PROOF OF POWER: the same turning rate, a tenth of the directedness.

    A rate column alone cannot tell a dancer who turns and comes back from a
    body that rotates like a statue, which is exactly the confusion the
    ``directed`` column exists to break.
    """
    frames = 301
    amplitude = np.radians(15.0)
    period = 30
    angles = amplitude * np.sin(2 * np.pi * np.arange(frames) / period)
    joints = spin_about_z(upright_body(frames), angles)
    turn, travel, directed = sweep.facing_columns(joints)
    assert turn > 40.0
    assert travel == pytest.approx(30.0, abs=0.5)
    assert directed < 0.1


def test_frozen_body_reads_zero_turn_zero_spread_and_is_still():
    joints = upright_body(200)
    turn, travel, directed = sweep.facing_columns(joints)
    assert turn == pytest.approx(0.0, abs=1e-12)
    assert travel == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(directed)
    assert sweep.spread(joints) == pytest.approx(0.0, abs=1e-12)
    # THE REFUSAL, pinned.  The hold criterion is RELATIVE (speed < 0.25 x the
    # window's own median), so an exactly frozen body has no scale and
    # ``tools/stillness_criterion`` refuses rather than returning 0 -- which
    # would be inverted, not small.  This column therefore reports None here and
    # the collapse is caught by ``spread`` and ``noise_span``, which read 0.
    assert sweep.has_sustained_hold(joints) is None
    # A body that moves and then settles is the case the column is for.
    moving = joints.copy()
    # Root-relative, so the ARMS have to move, not the whole body.
    moving[:100, [16, 17, 20, 21], 1] += np.linspace(0.0, 0.6, 100)[:, None]
    assert sweep.has_sustained_hold(moving) is True


def test_twist_sd_reads_a_known_shoulder_rotation_and_needs_no_unwrap():
    """A shoulder line swept +-170 deg relative to the hips stays continuous.

    The unwrapped form section 25.1 retracted would inherit a 2*pi jump as the
    angle crosses 180; the signed per-frame angle does not, and its sd is the
    sd of the swept angle itself.
    """
    frames = 400
    joints = upright_body(frames)
    angle = np.radians(np.linspace(-170.0, 170.0, frames))
    across = np.stack([np.cos(angle), np.sin(angle), np.zeros(frames)], axis=1)
    joints[:, 16, :] = -0.18 * across + (0.0, 0.0, 0.45)
    joints[:, 17, :] = 0.18 * across + (0.0, 0.0, 0.45)
    assert sweep.twist_sd(joints) == pytest.approx(np.degrees(angle).std(), abs=1e-6)


# ------------------------------------------------------------- the instrument

def test_ground_plane_is_invariant_to_world_yaw_and_legacy_plane_is_not():
    """THE DISAGREEMENT, pinned.  Rotating the world changes no dance."""
    frames = 201
    rng = np.random.default_rng(0)
    angles = np.radians(np.cumsum(rng.normal(0.0, 2.0, frames)))
    joints = spin_about_z(upright_body(frames), angles)
    joints = joints + rng.normal(0.0, 0.002, joints.shape)
    control = sweep.run_invariance_control(joints)
    assert control["ground_plane_invariant"] is True
    assert control["legacy_plane_invariant"] is False
    # And the legacy plane does not merely wobble: it moves by a lot.
    assert abs(control["legacy_plane_0"] - control["legacy_plane_90"]) > 1.0


def test_facing_matches_score_facing_spin():
    """PROVENANCE: the same hip-line azimuth the repo already ships."""
    from tools import score_facing_spin
    frames = 120
    rng = np.random.default_rng(3)
    joints = spin_about_z(upright_body(frames),
                          np.radians(np.cumsum(rng.normal(0.0, 3.0, frames))))
    assert np.allclose(sweep.facing_yaw(joints), score_facing_spin.facing_yaw(joints))


def test_still_share_matches_stillness_tool():
    """PROVENANCE: the hold criterion is imported, not re-derived."""
    from tools import stillness_criterion
    # Built here rather than imported from ``exp_guidance_stillness`` so this
    # test depends only on the shared criterion module: travel for 3 s, hold
    # for 1 s, repeat -- the move-then-hold control whose direction that
    # module's own tests already pin.
    rng = np.random.default_rng(5)
    frames = 300
    joints = np.zeros((frames, 24, 3))
    position = 0.0
    for frame in range(frames):
        if (frame // 30) % 4 != 3:
            position += 0.02
        joints[frame, 1:, 0] = position
    joints += rng.normal(0.0, 1e-4, joints.shape)
    speed = stillness_criterion.joint_speed(
        stillness_criterion.low_pass(joints, sweep.SMOOTH_WIDTH))
    assert sweep.has_sustained_hold(joints) is bool(
        stillness_criterion.held_frames(speed).any())
    assert sweep.has_sustained_hold(joints) is True
    drifting = np.zeros((300, 24, 3))
    for joint in range(1, 24):
        drifting[:, joint, 0] = np.arange(300) * 0.002 * (1 + joint % 3)
    assert sweep.has_sustained_hold(drifting) is False


# ------------------------------------------------------------ the window draw

def test_stratified_draw_covers_every_clip_equally():
    names = (["a_slice{}".format(i) for i in range(38)]
             + ["b_slice{}".format(i) for i in range(14)]
             + ["c_slice{}".format(i) for i in range(20)])
    picked = sweep.stratified_windows(names, per_clip=3)
    assert len(picked) == 9
    by_clip = {}
    for index in picked:
        by_clip.setdefault(sweep.clip_of(names[index]), []).append(index)
    assert sorted(by_clip) == ["a", "b", "c"]
    assert all(len(v) == 3 for v in by_clip.values())
    # The endpoints of each clip are included, so the draw spans the clip
    # rather than clustering at its head the way consecutive indices did.
    assert by_clip["a"] == [0, 18, 37]


def test_consecutive_indices_would_have_covered_three_clips():
    """The void sweep's draw, reproduced, so the defect stays legible."""
    names = []
    # The real release_v3 test split's first clips, in its own order.
    for clip, count in (("a", 32), ("b", 35), ("c", 38), ("d", 14)):
        names += ["{}_slice{}".format(clip, i) for i in range(count)]
    void = list(range(0, 96, 8))[:12]
    assert len({sweep.clip_of(names[i]) for i in void}) == 3
    stratified = sweep.stratified_windows(names, per_clip=3)
    assert len({sweep.clip_of(names[i]) for i in stratified}) == 4


def test_aggregate_reports_the_spin_count_the_median_hides():
    """A bimodal arm: 8 frozen windows and 2 spinners.  The median says frozen."""
    rows = [{"turn_deg_s": 15.0, "travel_deg": 10.0, "directed": 0.07,
             "spread": 0.05, "twist_sd": 4.0, "legacy_turn_deg_s": 15.0,
             "legacy_travel_deg": 10.0, "legacy_directed": 0.07,
             "legacy_twist_sd": 4.0, "still": True} for _ in range(8)]
    rows += [{"turn_deg_s": 250.0, "travel_deg": 900.0, "directed": 0.9,
              "spread": 0.05, "twist_sd": 4.0, "legacy_turn_deg_s": 250.0,
              "legacy_travel_deg": 900.0, "legacy_directed": 0.9,
              "legacy_twist_sd": 4.0, "still": False} for _ in range(2)]
    rows += [dict(rows[0], still=None)]
    summary = sweep.aggregate(rows, spin_threshold=100.0, noise_spans=[0.1] * 11)
    assert summary["turn_deg_s"] == pytest.approx(15.0)
    assert summary["spins"] == 2
    # The refused window is counted, not averaged in as a zero.
    assert summary["still_unmeasurable"] == 1
    assert summary["still_share"] == pytest.approx(0.8)
    assert summary["noise_span"] == pytest.approx(0.1)


def test_checkpoint_paths_subsamples_and_refuses_an_empty_arm(tmp_path):
    arm = tmp_path / "arm"
    arm.mkdir()
    for epoch in (50, 100, 150, 200):
        (arm / "completion_epoch{}_step{}.pt".format(epoch, epoch * 84)).write_text("x")
    assert len(sweep.checkpoint_paths([str(arm)], every=1)) == 4
    assert len(sweep.checkpoint_paths([str(arm)], every=2)) == 2
    with pytest.raises(FileNotFoundError):
        sweep.checkpoint_paths([str(tmp_path / "empty")], every=1)


def test_decode_batch_matches_decode_motion():
    """The batched decoder must be the shipped decoder, window for window."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("vis")
    release = pathlib.Path("/cache/atomicdance-assets/scratch/txy_t/release_v3")
    if not (release / "normalizer.pt").is_file():
        pytest.skip("release_v3 not present")
    from infer_atomic import decode_motion
    motion = np.load(str(release / "test" / "motion.npy"), mmap_mode="r")[:2]
    batched = sweep.decode_batch(torch.tensor(np.asarray(motion), dtype=torch.float32),
                                 str(release / "normalizer.pt"))
    for index in range(2):
        one = decode_motion(torch.tensor(np.asarray(motion[index]), dtype=torch.float32),
                            str(release / "normalizer.pt"))["full_pose"]
        assert np.allclose(batched[index], one, atol=1e-5)


# ----------------------------------------------------------- the stitch check

def test_stitching_exactly_overlapping_windows_returns_the_original():
    """POSITIVE CONTROL for the stitcher: it must be a no-op on agreeing input.

    This is what makes the ``--stitch-check`` reading interpretable.  If the
    blender itself damaged a signal, a lower reading on the stitched generated
    clip would say nothing about independent draws.
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    stride, size, count = 75, 150, 4
    clip = torch.tensor(rng.normal(size=((count - 1) * stride + size, 7)),
                        dtype=torch.float32)
    windows = torch.stack([clip[i * stride:i * stride + size] for i in range(count)])
    assert torch.allclose(sweep.reference_span(windows, stride), clip, atol=1e-6)
    stitched = sweep.stitch(windows, stride, blend_width=10)
    assert stitched.shape == clip.shape
    assert float((stitched - clip).abs().max()) < 1e-5


def test_stitching_independent_draws_damps_them():
    """PROOF OF POWER: the check can detect the thing it is looking for.

    Two windows with the SAME mean and independent deviations, overlap-added,
    keep only sqrt(w^2+(1-w)^2) of the deviation in the blended band -- the
    mechanism ``_blend_weights``' own docstring describes.  A stitch of
    independent draws must therefore read a smaller range than the draws did.
    """
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(12)
    stride, size, count = 75, 150, 4
    windows = torch.tensor(rng.normal(size=(count, size, 3)), dtype=torch.float32)
    wide = sweep.stitch(windows, stride, blend_width=size - stride)
    assert float(wide.std()) < float(windows.std()) * 0.95
