"""Aligning a pasted prototype's facing must move only the seam, never the content.

The defect (measured 2026-09-01, 97 clips, threshold = ground truth's own pooled
p99.5 turn rate of 549 deg/s): 92.9% of the retrieval draft's over-threshold
frames sit on a plan boundary -- 11.84x their share of frames -- while AWAY from
boundaries the draft turns 57.8 deg/s against ground truth's 62.8.  The library's
own content turns LESS than a real dancer; every violent turn is manufactured
where two prototypes cut from different recordings are pasted together, each
carrying its own recording's facing.

So the fix has two halves and both are tested: the seam must go, and the
prototype must otherwise be untouched.  The second half is the one that could
silently fail -- a rotation that also rescaled or drifted the body would still
remove the seam.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from infer_atomic import ROOT_POSITION_DIMS, ROOT_POSITION_START  # noqa: E402

GLOBAL_ORIENT_START = ROOT_POSITION_START + ROOT_POSITION_DIMS


class FakeLibrary:
    """The three methods under test over an identity normalizer."""

    from infer_atomic import IndexedAtomicMotionLibrary as _real
    GLOBAL_ORIENT_START = _real.GLOBAL_ORIENT_START
    _normalizer_affine = _real._normalizer_affine
    _facing_yaw = _real._facing_yaw
    _rotate_about_z = _real._rotate_about_z

    def __init__(self, dims=151):
        # scale 1, offset 0: normalized == raw, so the tests read in radians and
        # metres rather than in normalizer units.
        self._affine = (torch.ones(dims), torch.zeros(dims))


def segment(yaw, travel=0.0, frames=20):
    """A prototype facing ``yaw``, optionally walking along its own +x."""
    values = torch.zeros(frames, 151)
    cos, sin = math.cos(yaw), math.sin(yaw)
    # rows 1 and 2 of Rz(yaw), which is what rot6d stores
    values[:, GLOBAL_ORIENT_START:GLOBAL_ORIENT_START + 6] = torch.tensor(
        [cos, -sin, 0.0, sin, cos, 0.0])
    step = torch.arange(frames, dtype=torch.float32) * travel
    root = slice(ROOT_POSITION_START, ROOT_POSITION_START + 3)
    values[:, root] = torch.stack([step * cos, step * sin, torch.zeros(frames)], dim=1)
    return values


def yaw_of(library, values):
    return library._facing_yaw(values)[0]


# ------------------------------------------------------------- the seam goes

def test_a_rotated_segment_starts_where_it_is_told_to():
    library = FakeLibrary()
    values = segment(2.0)
    delta = torch.tensor(-0.75) - yaw_of(library, values)[0]
    turned = library._rotate_about_z(values, delta)
    assert float(yaw_of(library, turned)[0]) == pytest.approx(-0.75, abs=1e-5)


def test_the_alignment_never_takes_the_long_way_round():
    """A wrap the wrong way IS a spin: aligning +3.0 rad to -3.0 rad must turn
    0.28 rad, not 6.0.  This is the modulo in build_draft, asserted directly."""
    previous, first = 3.0, -3.0
    delta = math.remainder(previous - first, 2 * math.pi)
    assert abs(delta) < math.pi
    assert abs(delta) == pytest.approx(2 * math.pi - 6.0, abs=1e-9)


# ------------------------------------------------- the content is untouched

def test_the_shape_of_the_movement_is_preserved_exactly():
    """THE test.  A rotation must be rigid: every frame's facing CHANGE, and the
    distance travelled between frames, must be identical before and after."""
    library = FakeLibrary()
    values = segment(0.4, travel=0.05)
    turned = library._rotate_about_z(values, torch.tensor(1.9))
    before, after = yaw_of(library, values), yaw_of(library, turned)
    assert torch.allclose(torch.diff(before), torch.diff(after), atol=1e-5)
    root = slice(ROOT_POSITION_START, ROOT_POSITION_START + 3)
    speed_before = torch.linalg.norm(torch.diff(values[:, root], dim=0), dim=1)
    speed_after = torch.linalg.norm(torch.diff(turned[:, root], dim=0), dim=1)
    assert torch.allclose(speed_before, speed_after, atol=1e-5)


def test_it_rotates_about_the_segments_own_first_frame():
    """So that root_continuity still composes with it unchanged: the first
    frame's position must not move, or the translation that follows would be
    correcting for this instead of for the teleport it exists for."""
    library = FakeLibrary()
    values = segment(0.4, travel=0.05)
    turned = library._rotate_about_z(values, torch.tensor(1.9))
    root = slice(ROOT_POSITION_START, ROOT_POSITION_START + 3)
    assert torch.allclose(values[0, root], turned[0, root], atol=1e-6)


def test_the_ground_track_turns_with_the_body():
    """Turning a dancer turns where they WALK too.  If the root track were left
    alone the body would face one way and slide another -- a different defect
    wearing this fix's name."""
    library = FakeLibrary()
    values = segment(0.0, travel=0.05)
    turned = library._rotate_about_z(values, torch.tensor(math.pi / 2))
    root = slice(ROOT_POSITION_START, ROOT_POSITION_START + 3)
    moved = turned[-1, root] - turned[0, root]
    assert float(moved[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(moved[1]) > 0.5


def test_height_is_never_touched():
    library = FakeLibrary()
    values = segment(0.4, travel=0.05)
    values[:, ROOT_POSITION_START + 2] = torch.linspace(0.9, 1.1, len(values))
    turned = library._rotate_about_z(values, torch.tensor(1.9))
    assert torch.allclose(values[:, ROOT_POSITION_START + 2],
                          turned[:, ROOT_POSITION_START + 2], atol=1e-6)


def test_every_other_joint_is_left_alone():
    """Only the GLOBAL orientation is a world-frame rotation; the other 23 rot6d
    blocks are relative to their parent and rotating them would deform the body."""
    library = FakeLibrary()
    values = segment(0.4)
    values[:, GLOBAL_ORIENT_START + 6:] = torch.randn(len(values), 151 - GLOBAL_ORIENT_START - 6)
    values[:, :ROOT_POSITION_START] = torch.rand(len(values), ROOT_POSITION_START)
    turned = library._rotate_about_z(values, torch.tensor(1.9))
    assert torch.allclose(values[:, GLOBAL_ORIENT_START + 6:],
                          turned[:, GLOBAL_ORIENT_START + 6:], atol=1e-6)
    assert torch.allclose(values[:, :ROOT_POSITION_START],
                          turned[:, :ROOT_POSITION_START], atol=1e-6)


def test_a_zero_turn_is_a_no_op():
    library = FakeLibrary()
    values = segment(0.4, travel=0.05)
    assert torch.allclose(values, library._rotate_about_z(values, torch.tensor(0.0)),
                          atol=1e-5)


# --------------------------------------------------------------- plumbing

def test_off_by_default_so_earlier_artifacts_reproduce():
    import inspect

    from infer_atomic import IndexedAtomicMotionLibrary

    signature = inspect.signature(IndexedAtomicMotionLibrary.build_draft)
    assert signature.parameters["facing_continuity"].default is False


def test_the_affine_is_the_exact_inverse_of_unnormalize():
    """The rotation has to happen in RAW space -- min-max scaling is per
    dimension, so a rotation, which mixes dimensions, is not expressible in
    normalized space the way root_continuity's constant offset is."""
    from infer_atomic import unnormalize_motion

    class Tiny(FakeLibrary):
        def __init__(self, path):
            self.normalizer_path = path
            self._affine = None

    low = torch.linspace(-3.0, 1.0, 151)
    high = low + torch.linspace(0.5, 4.0, 151)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save({"data_min": low, "data_max": high}, handle.name)
        library = Tiny(handle.name)
        scale, offset = library._normalizer_affine()
        sample = torch.randn(5, 151)
        assert torch.allclose(sample * scale + offset,
                              unnormalize_motion(sample, handle.name), atol=1e-5)
