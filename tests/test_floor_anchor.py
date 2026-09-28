"""``--floor-anchor`` stands the clip on a known floor without touching the dance.

The defect it exists for, measured 2026-09-07 over the 20 T-line eval clips
(floor = 5th percentile of the per-frame lowest foot joint):

    ground truth   median 0.342 m   p5-p95 spread 0.047 m
    generated      median 0.310 m   p5-p95 spread 0.454 m   corr with GT -0.021

``tools/render_avatar_video.py`` puts every arm on the REFERENCE's floor
deliberately, so a wrong standing height is drawn as a body hovering over the
checkerboard (+0.196 m on 7610414564962183545) or sunk through it (-0.243 m on
7287585049111711032). The operator named exactly that.

The property that makes the fix safe is that the shift is RIGID: it must move
where the body stands and nothing else. That is what most of these tests check,
because a correction that also flattened vertical motion would trade a visible
defect for an invisible one -- ground truth's pelvis genuinely travels 0.272 m
up and down, and a per-frame floor correction would delete it.
"""

import numpy as np
import pytest

import infer_atomic


def _clip(frames=120, seed=0):
    rng = np.random.default_rng(seed)
    joints = rng.normal(0.0, 0.05, size=(frames, 24, 3)).astype(np.float32)
    # feet near a floor at 0.40 m, with a couple of frames punched through it
    joints[:, list(infer_atomic.FOOT_JOINTS), 2] = (
        0.40 + np.abs(rng.normal(0.0, 0.06, size=(frames, 4)))).astype(np.float32)
    joints[3, list(infer_atomic.FOOT_JOINTS), 2] = -0.5
    joints[:, 0, 2] = 1.30 + 0.10 * np.sin(np.linspace(0, 8 * np.pi, frames))
    return {"full_pose": joints,
            "smpl_trans": joints[:, 0, :].copy(),
            "smpl_poses": rng.normal(size=(frames, 72)).astype(np.float32),
            "contacts": np.zeros((frames, 4), np.float32)}


def _floor(joints):
    return float(np.percentile(
        np.asarray(joints)[:, list(infer_atomic.FOOT_JOINTS), 2].min(axis=1),
        infer_atomic.FLOOR_PERCENTILE))


def test_the_clip_ends_up_on_the_requested_floor():
    out = infer_atomic.anchor_floor(_clip(), 0.342)
    assert _floor(out["full_pose"]) == pytest.approx(0.342, abs=1e-6)


@pytest.mark.parametrize("start", [0.05, 0.342, 0.75])
def test_it_works_from_any_starting_height(start):
    clip = _clip()
    clip["full_pose"] = clip["full_pose"] + np.array([0, 0, start - _floor(clip["full_pose"])], np.float32)
    out = infer_atomic.anchor_floor(clip, 0.342)
    assert _floor(out["full_pose"]) == pytest.approx(0.342, abs=1e-6)


def test_the_shift_is_rigid_so_the_dance_is_untouched():
    """Root-relative positions must survive to float32 rounding.

    This is the whole safety argument: if it fails, the anchor is editing the
    motion rather than where it stands.

    NOT bit-identical, and the reason is arithmetic rather than a defect --
    ``(j + s) - (j0 + s)`` is not ``j - j0`` in floating point. Measured on this
    fixture the worst root-relative deviation is 1.19e-07 m and the worst
    per-frame joint speed deviation 5.96e-08 m/frame, i.e. a tenth of a micron,
    about six orders of magnitude below the 0.196 m defect this exists to fix.
    The bound below is 1e-6, so it is tight enough to fail if the shift ever
    stopped being uniform.
    """
    clip = _clip()
    out = infer_atomic.anchor_floor(clip, 0.342)
    before = clip["full_pose"] - clip["full_pose"][:, 0:1, :]
    after = out["full_pose"] - out["full_pose"][:, 0:1, :]
    assert np.abs(after - before).max() < 1e-6

    speed = lambda a: np.linalg.norm(np.diff(a, axis=0), axis=2)
    assert np.abs(speed(out["full_pose"]) - speed(clip["full_pose"])).max() < 1e-6


def test_a_non_uniform_shift_would_fail_the_rigidity_check():
    """The rigidity bound must be able to fail -- CLAUDE.md section 2.

    A 1 mm per-frame ramp is far smaller than any real correction would be, and
    the check still catches it. Without this, 1e-6 could be passing because the
    comparison is vacuous rather than because the shift is rigid.
    """
    clip = _clip()
    tampered = dict(clip)
    ramp = np.linspace(0.0, 0.001, len(clip["full_pose"])).astype(np.float32)
    tampered["full_pose"] = clip["full_pose"] + ramp[:, None, None] * np.array([0, 0, 1], np.float32)
    before = clip["full_pose"] - clip["full_pose"][:, 0:1, :]
    after = tampered["full_pose"] - tampered["full_pose"][:, 0:1, :]
    # a per-frame ramp shifts every joint equally WITHIN a frame, so it hides
    # from the root-relative check -- it shows up in the speed check instead.
    speed = lambda a: np.linalg.norm(np.diff(a, axis=0), axis=2)
    assert (np.abs(after - before).max() >= 1e-6
            or np.abs(speed(tampered["full_pose"]) - speed(clip["full_pose"])).max() >= 1e-6)


def test_vertical_motion_survives():
    """A per-frame floor correction would flatten this; a rigid one cannot.

    Ground truth's pelvis really does travel ~0.27 m vertically, so an anchor
    that removed vertical excursion would delete real dance.
    """
    clip = _clip()
    out = infer_atomic.anchor_floor(clip, 0.342)
    span = lambda a: float(a[:, 0, 2].max() - a[:, 0, 2].min())
    assert span(out["full_pose"]) == pytest.approx(span(clip["full_pose"]), abs=1e-6)


def test_horizontal_position_is_untouched():
    clip = _clip()
    out = infer_atomic.anchor_floor(clip, 0.342)
    assert np.array_equal(out["full_pose"][:, :, :2], clip["full_pose"][:, :, :2])


def test_translation_moves_with_the_joints():
    """smpl_trans and full_pose must not drift apart.

    render_avatar_video skins from smpl_trans and then gates the result against
    full_pose to within a millimetre, so shifting one and not the other would
    stop every avatar render.
    """
    clip = _clip()
    out = infer_atomic.anchor_floor(clip, 0.342)
    moved_joint = out["full_pose"][:, 0, 2] - clip["full_pose"][:, 0, 2]
    moved_trans = out["smpl_trans"][:, 2] - clip["smpl_trans"][:, 2]
    assert np.allclose(moved_joint, moved_trans, atol=1e-6)
    assert np.array_equal(out["smpl_poses"], clip["smpl_poses"])


def test_a_foot_punched_through_the_floor_does_not_define_the_plane():
    """Percentile, not min -- wild reconstructions do punch through.

    The fixture puts frame 3's feet at -0.5 m. Using the minimum would put the
    whole clip half a metre too high.
    """
    out = infer_atomic.anchor_floor(_clip(), 0.342)
    lowest = np.asarray(out["full_pose"])[:, list(infer_atomic.FOOT_JOINTS), 2].min(axis=1)
    assert lowest.min() < 0.342          # the punched frame is still below
    assert _floor(out["full_pose"]) == pytest.approx(0.342, abs=1e-6)


def test_the_report_records_what_was_done():
    out = infer_atomic.anchor_floor(_clip(), 0.342)
    report = out["floor_anchor"]
    assert report["target"] == 0.342
    assert report["shift"] == pytest.approx(0.342 - report["measured"], abs=1e-9)


def test_it_does_not_mutate_the_input():
    clip = _clip()
    original = clip["full_pose"].copy()
    infer_atomic.anchor_floor(clip, 0.342)
    assert np.array_equal(clip["full_pose"], original)


def test_off_by_default_and_wired_at_every_call_site():
    """The flag must reach the code from EVERY writer call.

    Eight defects on 2026-09-05/06 were a switch recorded in the manifest but
    applied at only some call sites; this enumerates them with ast instead of
    trusting that the three look alike.
    """
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(infer_atomic.__file__).read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_write_generated_result"]
    assert len(calls) >= 3, "expected every generation path to funnel through the writer"
    unwired = [c.lineno for c in calls
               if not any(k.arg == "floor_anchor" for k in c.keywords)]
    assert not unwired, "_write_generated_result called without floor_anchor at {}".format(unwired)
