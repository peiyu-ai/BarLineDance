"""The window stitcher's averaged band must be narrowable without losing frames.

The invariant that matters is not "the ramp looks right" -- it is that every
output frame's weights sum to exactly 1 across the windows covering it.  A
stitcher that quietly leaves a frame at weight 0, or at weight 2, produces
motion that reads as a model defect.
"""
import pytest

torch = pytest.importorskip("torch")
infer_atomic = pytest.importorskip("infer_atomic")

WINDOW = 150
STRIDE = 75


def stitched_weight_sum(length, window, stride, blend_width):
    starts = infer_atomic._window_starts(length, window, stride)
    total = torch.zeros(length, 1)
    overlap = window - stride
    for index, start in enumerate(starts):
        valid = min(window, length - start)
        weights = infer_atomic._blend_weights(
            valid,
            is_first=index == 0,
            is_last=index == len(starts) - 1,
            overlap=min(overlap, valid),
            blend_width=blend_width,
        )
        total[start:start + valid] += weights
    return total


@pytest.mark.parametrize("blend_width", [None, 1, 6, 10, 30, 75, 999])
@pytest.mark.parametrize("length", [150, 225, 300, 450, 600])
def test_every_frame_keeps_unit_weight(length, blend_width):
    total = stitched_weight_sum(length, WINDOW, STRIDE, blend_width)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), \
        "frames at weight {}".format(sorted(set(total.flatten().tolist()))[:5])


def test_none_reproduces_the_shipped_ramp_exactly():
    """Backward compatibility is not a nicety here: every artifact generated
    before 2026-08-31 must stay reproducible from its manifest."""
    for is_first in (True, False):
        for is_last in (True, False):
            new = infer_atomic._blend_weights(WINDOW, is_first, is_last, STRIDE, blend_width=None)
            old = torch.ones(WINDOW, 1)
            if not is_first:
                old[:STRIDE] = torch.linspace(0.0, 1.0, STRIDE + 2)[1:-1, None]
            if not is_last:
                old[-STRIDE:] = torch.linspace(1.0, 0.0, STRIDE + 2)[1:-1, None]
            assert torch.allclose(new, old, atol=1e-7)


def test_the_shipped_default_really_does_average_almost_every_frame():
    """The positive control for the claim this flag exists to address.  Without
    it, the narrowing test below passes on a stitcher that never averaged
    anything and the 79.75% figure goes unchecked."""
    interior = infer_atomic._blend_weights(WINDOW, False, False, STRIDE, blend_width=None)
    mixed = int((interior < 1.0 - 1e-6).sum())
    assert mixed == WINDOW, mixed


def test_narrowing_confines_the_average_to_the_junction():
    interior = infer_atomic._blend_weights(WINDOW, False, False, STRIDE, blend_width=10)
    mixed = int(((interior > 1e-6) & (interior < 1.0 - 1e-6)).sum())
    assert mixed == 20, mixed          # one 10-frame ramp at each junction
    # 150 frames = 32 ceded to the previous window, a 10-frame ramp up, 65
    # frames taken whole, a 10-frame ramp down, 33 ceded to the next.  The two
    # ceded blocks are not lost: the neighbouring window carries them at
    # weight 1, which is what the unit-weight test above verifies.
    whole = int((interior >= 1.0 - 1e-6).sum())
    assert whole == 65, whole
    assert int((interior <= 1e-6).sum()) == 65, "ceded frames"


def test_a_narrow_band_leaves_most_frames_untouched_by_any_average():
    """The quantity the operator cares about: the share of OUTPUT frames that
    are a convex combination of two independent diffusion draws."""
    length = 600
    for blend_width, ceiling in ((None, 1.00), (30, 0.30), (10, 0.12), (6, 0.08)):
        starts = infer_atomic._window_starts(length, WINDOW, STRIDE)
        mixed = torch.zeros(length)
        overlap = WINDOW - STRIDE
        for index, start in enumerate(starts):
            valid = min(WINDOW, length - start)
            weights = infer_atomic._blend_weights(
                valid, index == 0, index == len(starts) - 1,
                min(overlap, valid), blend_width=blend_width).flatten()
            partial = (weights > 1e-6) & (weights < 1.0 - 1e-6)
            mixed[start:start + valid] += partial.float()
        share = float((mixed > 0).float().mean())
        assert share <= ceiling + 1e-6, (blend_width, share)


def test_blend_width_is_clamped_not_crashed():
    """A width wider than the overlap is a configuration mistake, not a reason
    to drop frames; it must degrade to the full ramp."""
    wide = infer_atomic._blend_weights(WINDOW, False, False, STRIDE, blend_width=10_000)
    full = infer_atomic._blend_weights(WINDOW, False, False, STRIDE, blend_width=None)
    assert torch.allclose(wide, full, atol=1e-7)


def test_no_overlap_is_untouched_by_the_flag():
    for blend_width in (None, 1, 10):
        weights = infer_atomic._blend_weights(WINDOW, False, False, 0, blend_width=blend_width)
        assert torch.allclose(weights, torch.ones(WINDOW, 1))
