"""--draft-root-seam-smooth must stop the draft braking the body at every bar line.

THE DEFECT.  The operator, 2026-09-16, on output/sample_20260916_fix2: "蒙皮上偶发的
位置跳变 ... 视觉上偶尔就会有顿挫感,缺流畅连续".  tools/score_seam_root_speed.py on the
ten vis clips, horizontal root speed / clip median by distance to the nearest bar
seam, ground truth read at the SAME frame indices as the null:

    distance        0     1     2     3     4     5     6     7     8
    ground truth  1.24  1.25  1.25  1.24  1.21  1.19  1.16  1.15  1.14
    shipped       1.15  1.20  1.44  1.60  1.67  1.68  1.59  1.38  1.27   (surge)
    fix2          0.31  0.28  0.65  0.95  1.24  1.43  1.45  1.35  1.28   (brake)

``brake`` (speed at the seam / 13-14 frames away): ground truth 1.19, fix2 0.24,
lower on 10/10 clips, P=0.002.  fix2 and its draft are identical frame for frame, so
the defect is built in the draft.  The shipped arm hid it: --fix-foot-skate injected
travel inside the same window (74% of its path), a surge that cancelled the brake,
and --fix-skate-seam-mask removed the surge and exposed the brake underneath.

TWO MECHANISMS, both in the draft:
1. root continuity put each segment's FIRST frame exactly on the previous segment's
   LAST position -- a zero step across every seam that --draft-root-velocity-blend,
   which corrects frames 1.. of the new segment, never reaches;
2. ``_blend_draft_seams`` blended every channel, root included, toward a ramp between
   two FROZEN anchor frames, dragging the root toward a fixed point for +-half_width.

THE FIX, chosen by simulation against the physical requirement (the speed must pass
from one side's to the other's without dipping below the slower or overshooting the
faster): align the first frame one velocity step ahead, keep the CHAINED root translation (xy under
--draft-root-continuity xy) out of the frozen-anchor blend, and let --draft-root-velocity-blend ease the velocity.
Simulated, speed doubling across a seam: shipped min 0.00 / max 3.21 of the slower
speed; velocity-extrapolated anchors 0.79 / 2.14 (still dips); this fix 1.00 / 2.00.

BEHAVIOURAL TESTS.  The first one asserts the fixture reproduces the shipped brake;
if it stops doing so, the rest prove nothing.
"""
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic as ia  # noqa: E402

SOURCE = pathlib.Path("infer_atomic.py").read_text()
ROOT = list(range(ia.ROOT_POSITION_START, ia.ROOT_POSITION_START + ia.ROOT_POSITION_DIMS))
FRAMES, SEAM, HALF, FEATURES, V = 80, 40, 8, 151, 0.02
XY = ROOT[:2]   # what --draft-root-continuity xy chains, and so what the blend must skip


def travelling_draft():
    draft = torch.zeros(FRAMES, FEATURES)
    t = torch.arange(FRAMES, dtype=torch.float32)
    draft[:, ROOT[0]] = V * t
    draft[:, ROOT[1]] = 0.5 * V * t
    pose = [c for c in range(ia.ROOT_POSITION_START, FEATURES) if c not in ROOT]
    draft[:SEAM, pose] = 0.1
    draft[SEAM:, pose] = -0.1
    labels = torch.zeros(FRAMES, dtype=torch.long)
    labels[SEAM:] = 1
    return draft, torch.ones(FRAMES, 1), labels


def speed(values):
    xy = np.asarray(values)[:, :2] if np.asarray(values).ndim == 2 else np.asarray(values)
    return np.linalg.norm(np.diff(xy, axis=0), axis=-1) if xy.ndim == 2 else np.abs(np.diff(xy))


def test_the_shipped_blend_brakes_the_root_at_the_seam():
    draft, mask, labels = travelling_draft()
    ia._blend_draft_seams(draft, mask, labels, HALF, stagger=True, window="cosine")
    s = speed(draft[:, ROOT[:2]].numpy())
    cruise = float(np.hypot(V, 0.5 * V))
    assert s[SEAM - 1] < 0.5 * cruise, (
        "the fixture must reproduce the brake; speed at the seam {:.4f} vs cruise {:.4f}"
        .format(s[SEAM - 1], cruise))


def test_with_the_flag_the_blend_leaves_the_chained_root_alone():
    draft, mask, labels = travelling_draft()
    before = draft.clone()
    ia._blend_draft_seams(draft, mask, labels, HALF, stagger=True, window="cosine",
                          skip_columns=XY)
    assert torch.equal(draft[:, XY], before[:, XY])
    s = speed(draft[:, XY].numpy())
    assert np.allclose(s, np.hypot(V, 0.5 * V), atol=1e-6), "the chained root must keep cruising"


def test_height_is_still_blended_because_xy_continuity_does_not_chain_it():
    """Skipping z too would leave a raw height jump at every seam."""
    a, mask, labels = travelling_draft()
    a[:SEAM, ROOT[2]] = 0.0
    a[SEAM:, ROOT[2]] = 0.3
    b = a.clone()
    ia._blend_draft_seams(a, mask, labels, HALF, stagger=True, window="cosine")
    ia._blend_draft_seams(b, mask, labels, HALF, stagger=True, window="cosine",
                          skip_columns=XY)
    assert torch.equal(a[:, ROOT[2]], b[:, ROOT[2]])
    assert float(np.abs(np.diff(b[:, ROOT[2]].numpy())).max()) < 0.3, "z must be ramped, not stepped"


@pytest.mark.parametrize("stagger", [True, False])
def test_the_pose_columns_are_bit_identical_with_and_without_the_flag(stagger):
    a, mask, labels = travelling_draft()
    b = a.clone()
    ia._blend_draft_seams(a, mask, labels, HALF, stagger=stagger, window="cosine")
    ia._blend_draft_seams(b, mask, labels, HALF, stagger=stagger, window="cosine",
                          skip_columns=XY)
    pose = [c for c in range(FEATURES) if c not in XY]
    assert torch.equal(a[:, pose], b[:, pose]), (
        "the flag may only change the root translation columns; the pose hold at the "
        "seam sits on the beat and is not this change's to touch")


def test_default_reproduces_the_old_blend_exactly():
    a, mask, labels = travelling_draft()
    b = a.clone()
    ia._blend_draft_seams(a, mask, labels, HALF, stagger=True, window="cosine")
    ia._blend_draft_seams(b, mask, labels, HALF, stagger=True, window="cosine",
                          skip_columns=())
    assert torch.equal(a, b)


def chain(v1, v2, lead, blend):
    """Two constant-velocity segments joined by the repository's own continuity step."""
    first = torch.stack([v1 * torch.arange(SEAM, dtype=torch.float32),
                         torch.zeros(SEAM)], dim=1)
    second = torch.stack([v2 * torch.arange(FRAMES - SEAM, dtype=torch.float32),
                          torch.zeros(FRAMES - SEAM)], dim=1)
    previous_root = first[-1].clone()
    previous_velocity = first[-1] - first[-2]
    joined = ia._continue_root(second, previous_root, previous_velocity,
                               velocity_blend=blend, lead=lead)
    return torch.cat([first, joined])[:, 0].numpy()


def test_the_old_continuity_leaves_a_zero_step_at_the_seam():
    x = chain(V, V, lead=False, blend=0)
    assert abs(x[SEAM] - x[SEAM - 1]) < 1e-9, (
        "the fixture must reproduce the zero step the shipped continuity leaves")


def test_the_lead_removes_the_zero_step():
    x = chain(V, V, lead=True, blend=0)
    assert np.allclose(np.diff(x), V, atol=1e-6)


def test_a_speed_change_passes_between_the_two_speeds():
    x = chain(V, 2 * V, lead=True, blend=8)
    s = np.diff(x)
    assert s.min() >= V - 1e-6, "speed dipped below the slower side: {:.4f}".format(s.min())
    assert s.max() <= 2 * V + 1e-6, "speed overshot the faster side: {:.4f}".format(s.max())


def test_the_continuity_helper_matches_the_old_inline_code_when_lead_is_off():
    """Refactoring the inline block into _continue_root must not move any existing arm."""
    for blend in (0, 8):
        x = chain(V, 2 * V, lead=False, blend=blend)
        first = V * np.arange(SEAM)
        second = 2 * V * np.arange(FRAMES - SEAM)
        second = second + (first[-1] - second[0])
        if blend:
            span = min(blend, len(second) - 1)
            correction = V - (second[1] - second[0])
            for k in range(span):
                weight = 0.5 + 0.5 * math.cos(math.pi * (k + 1) / (span + 1))
                second[k + 1:] += correction * weight
        assert np.allclose(x, np.concatenate([first, second]), atol=1e-6), blend


def zero_step_draft(v1, v2):
    """A draft root as the chaining leaves it: v1, then a ZERO step, then v2."""
    draft = torch.zeros(FRAMES, FEATURES)
    x = np.concatenate([v1 * np.arange(SEAM),
                        v1 * (SEAM - 1) + v2 * np.arange(FRAMES - SEAM)])
    draft[:, XY[0]] = torch.tensor(x, dtype=torch.float32)
    pose = [c for c in range(ia.ROOT_POSITION_START, FEATURES) if c not in ROOT]
    draft[:SEAM, pose] = 0.1
    draft[SEAM:, pose] = -0.1
    labels = torch.zeros(FRAMES, dtype=torch.long)
    labels[SEAM:] = 1
    return draft, torch.ones(FRAMES, 1), labels


def test_the_post_pass_fixture_has_the_zero_step():
    draft, _, _ = zero_step_draft(V, V)
    assert float(draft[SEAM, XY[0]] - draft[SEAM - 1, XY[0]]) == 0.0


def test_the_post_pass_carries_a_constant_speed_through_the_seam():
    draft, mask, labels = zero_step_draft(V, V)
    ia._smooth_root_seams(draft, mask, labels, XY, 8)
    assert np.allclose(np.diff(draft[:, XY[0]].numpy()), V, atol=1e-6)


def test_the_post_pass_passes_monotonically_between_two_speeds():
    draft, mask, labels = zero_step_draft(V, 2 * V)
    ia._smooth_root_seams(draft, mask, labels, XY, 8)
    s = np.diff(draft[:, XY[0]].numpy())
    assert s.min() >= V - 1e-6 and s.max() <= 2 * V + 1e-6, (s.min(), s.max())


def test_the_post_pass_moves_only_the_named_columns():
    a, mask, labels = zero_step_draft(V, 2 * V)
    b = a.clone()
    ia._smooth_root_seams(b, mask, labels, XY, 8)
    others = [c for c in range(FEATURES) if c not in XY]
    assert torch.equal(a[:, others], b[:, others])


def test_the_post_pass_leaves_unconditioned_frames_alone():
    draft, mask, labels = zero_step_draft(V, V)
    mask[60:65] = 0.0
    draft[60:65] = 0.0
    ia._smooth_root_seams(draft, mask, labels, XY, 8)
    assert torch.equal(draft[60:65], torch.zeros(5, FEATURES))


def test_the_flag_is_wired_recorded_and_runs_after_retrieval():
    assert '"--draft-root-seam-smooth", type=int' in SOURCE
    assert "draft_root_seam_smooth=options.draft_root_seam_smooth" in SOURCE
    assert '"draft_root_seam_smooth": draft_root_seam_smooth' in SOURCE
    assert "skip_columns=(continuity_dims if root_seam_smooth else ())" in SOURCE, (
        "build_draft must keep the chained root out of the frozen-anchor blend, or "
        "the flag is recorded and half dead -- DEFECTS 77")
    assert "--draft-root-seam-smooth needs --draft-root-continuity" in SOURCE
    # SELECTION MUST NOT SEE IT.  _join_cost compares absolute root position, so
    # any root change made while chaining reranks candidates: the in-chain version
    # swapped the moves on 2 of 10 clips.  The post-pass must come after the last
    # retrieval, and chaining must not take a lead from this flag.
    assert "lead=root_seam_smooth" not in SOURCE
    body = SOURCE.split("    def build_draft(", 1)[1].split("\n    def ", 1)[0]
    assert body.index("previous_tail = tail * scale + offset") < body.index("_smooth_root_seams("), (
        "the root post-pass must run after the retrieval loop")


def test_a_seam_inside_a_label_run_is_found_when_the_unit_starts_are_given():
    """build_draft chains BAR units; a label run several bars long holds seams a
    label-change scan cannot see (23 of 92 on the ten vis clips)."""
    draft, mask, labels = zero_step_draft(V, V)
    labels[:] = 0
    missed = draft.clone()
    ia._smooth_root_seams(missed, mask, labels, XY, 8)
    assert float(missed[SEAM, XY[0]] - missed[SEAM - 1, XY[0]]) == 0.0, (
        "without the unit starts the in-run seam must be missed -- the fixture's premise")
    ia._smooth_root_seams(draft, mask, labels, XY, 8, starts=[0, SEAM])
    assert np.allclose(np.diff(draft[:, XY[0]].numpy()), V, atol=1e-6)


def test_build_draft_hands_the_post_pass_its_unit_starts():
    # each unit's start AS LAID DOWN: --draft-seam-lead moves a unit's seam before its slot (seam_at); without the
    # flag seam_at maps every start to itself, so this is the same list as before (flags-off output byte-identical)
    assert ("starts=[seam_at.get(int(segment.start), int(segment.start))\n"
            "                                       for segment in segments]") in SOURCE


def in_run_height_step():
    """One label for the whole clip, a height step at a BAR seam inside it."""
    draft, mask, labels = zero_step_draft(V, V)
    labels[:] = 0
    draft[:SEAM, ROOT[2]] = 0.0
    draft[SEAM:, ROOT[2]] = 0.25
    return draft, mask, labels


def test_the_label_scan_never_sees_a_seam_inside_a_run():
    """The premise: the shipped blend leaves an in-run height step untouched."""
    draft, mask, labels = in_run_height_step()
    before = draft.clone()
    ia._blend_draft_seams(draft, mask, labels, HALF, stagger=True, window="cosine")
    assert torch.equal(draft, before)
    assert float(draft[SEAM, ROOT[2]] - draft[SEAM - 1, ROOT[2]]) == pytest.approx(0.25)


def test_explicit_seams_ramp_the_height_and_touch_nothing_else():
    draft, mask, labels = in_run_height_step()
    before = draft.clone()
    ia._blend_draft_seams(draft, mask, labels, HALF, window="cosine",
                          seams=[SEAM], only_columns=(ROOT[2],))
    steps = np.abs(np.diff(draft[:, ROOT[2]].numpy()))
    assert steps.max() < 0.25 * 0.5, "the height step must be ramped: {:.3f}".format(steps.max())
    others = [c for c in range(FEATURES) if c != ROOT[2]]
    assert torch.equal(draft[:, others], before[:, others])


def test_build_draft_ramps_in_run_height_only_under_the_flag():
    body = SOURCE.split("    def build_draft(", 1)[1].split("\\n    def ", 1)[0]
    assert "only_columns=unchained" in body and "seams=in_run" in body
    guard = body.index("if unchained and in_run:")
    assert body.rfind("if root_seam_smooth and continuity_dims:", 0, guard) > body.index(
        "previous_tail = tail * scale + offset"), (
        "the in-run height ramp must sit under the flag and after retrieval")


def test_the_height_ramp_is_counted_into_the_manifest():
    """The flag's value is the same before and after the height ramp was added,
    so only a counter can tell the two builds apart."""
    assert "self.root_seam_height_seams" in SOURCE
    assert '"draft_root_seam_height_seams": getattr(library, "root_seam_height_seams", 0)' in SOURCE
