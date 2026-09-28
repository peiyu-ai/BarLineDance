"""``--draft-seam-window``: the envelope of the seam cross-fade.

THE DEFECT.  ``_blend_draft_seams`` multiplies a smooth raised-cosine RAMP by a
TRIANGULAR envelope ``1 - |i - centre| / half_width``.  The ramp was made a
raised cosine on purpose -- its own docstring says a linear fade "removes the
step in the value and leaves one in the first derivative" -- but the envelope it
is multiplied by was left linear, so the corner is still there, one level up.
Multiplying a corner into the signal puts a jerk spike exactly at the seam, and
the corner's slope is 1/half_width, so a NARROW blend spikes harder.

THE EVIDENCE THAT SENT US LOOKING (2026-09-05, 17 eval clips, filler frames
excluded, boundaries read from the artifact's own slot_start rather than
reconstructed):

    ground truth       0.2553      <- the null: no seams, same frame indices
    no blending        0.3515
    half-width  4      0.5139      <- WORSE than not blending at all
    half-width  6      0.3195
    half-width  8      0.2272
    half-width 12      0.1805      <- now smoother than the interior

Non-monotonic in the width, reproduced at four window half-widths.  The
instrument was checked first (CLAUDE.md 2.2): ground truth reads flat across
window widths, so widening the window has no effect of its own.

A MECHANISM THAT WAS PROPOSED AND FALSIFIED, kept here so it is not proposed
again.  The corner above was offered as the CAUSE of the non-monotonicity, on
the reasoning that a sharper corner injects more jerk than the step it removes.
The synthetic control below refutes it: on a pure step, the triangle blend at
half-width 4 takes peak jerk from 12.0 down to 0.99.  It does not make things
worse; it helps enormously.  So the corpus non-monotonicity has some OTHER
cause and remains open -- see DANCE_QUALITY_DEFECTS.  CLAUDE.md 2.2: an
explanation that fits is not evidence, and a falsified one gets discarded
rather than patched.

WHAT THESE TESTS DO PIN, all of which survived that refutation: at equal width
the cosine envelope yields strictly less jerk than the triangle; neither
envelope touches a frame outside the blend window; and triangle stays the
default so earlier artifacts reproduce.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_atomic import _blend_draft_seams

DIM = 6


def _two_prototypes(half_width, length=80, seam=40, step=1.0):
    """A draft that is constant, steps at ``seam``, and is constant again.

    Everything interesting is the step: the interior of each side has zero
    jerk, so any jerk that appears is the blend's doing and nothing else.
    """
    draft = torch.zeros(length, DIM)
    draft[seam:] = step
    mask = torch.ones(length, 1)
    labels = torch.ones(length, dtype=torch.long)
    labels[seam:] = 2                     # one seam, both sides conditioned
    return draft, mask, labels


def _peak_jerk(draft):
    d3 = torch.diff(draft, n=3, dim=0)
    return float(d3.abs().sum(dim=1).max())


def test_the_cosine_envelope_is_flat_where_the_triangle_has_its_corner():
    """A property of the envelopes, not a claim about what causes what.

    At the seam the triangle's slope flips sign in one sample; the cosine's
    derivative is zero there because 0.5-0.5cos(pi*r) is flat at r=1.  Assert
    exactly that and nothing more -- the earlier version of this test asserted
    a curvature RATIO, which is neither what the code depends on nor what the
    corpus measurement showed.
    """
    half = 8
    def tri(i):
        return max(0.0, min(1.0, 1.0 - abs(i) / half))
    def cos(i):
        return 0.5 - 0.5 * math.cos(math.pi * tri(i))
    # one sample either side of the seam
    assert tri(-1) == tri(1) < tri(0)
    assert cos(-1) == cos(1) < cos(0)
    # one frame off the seam the triangle has already given up 1/half of its
    # weight (0.125 at half=8) while the cosine has given up 0.038 -- 3.3x less,
    # which is the whole of what "flat at the top" buys here.  The number is
    # written down rather than bounded loosely, because a looser bound was the
    # first version of this test and it asserted more than the code delivers.
    assert abs((tri(0) - tri(1)) - 0.125) < 1e-9
    assert abs((cos(0) - cos(1)) - 0.0380602337) < 1e-6
    assert (tri(0) - tri(1)) > 3 * (cos(0) - cos(1))


def test_a_narrow_cosine_blend_beats_a_narrow_triangle_blend():
    """THE FALSIFIABLE PREDICTION the mechanism makes.

    If the small-width penalty comes from the envelope's corner, then removing
    the corner must remove the penalty -- most visibly at the narrow widths
    where the corner is sharpest.  If this fails, the explanation is wrong and
    must be discarded rather than patched (CLAUDE.md 2.2).
    """
    for half in (2, 4, 6):
        a, m, l = _two_prototypes(half)
        b = a.clone()
        _blend_draft_seams(a, m, l, half, window="triangle")
        _blend_draft_seams(b, m, l, half, window="cosine")
        assert _peak_jerk(b) < _peak_jerk(a), "half_width {}".format(half)


def test_the_corpus_non_monotonicity_does_NOT_reproduce_on_a_pure_step():
    """The refutation, kept as a test so the theory cannot come back quietly.

    If a narrow triangle blend were intrinsically worse than no blend, this
    would show it.  It shows the opposite by an order of magnitude, which is
    why the envelope corner is NOT the accepted explanation for the measured
    0.5139-at-half-width-4 reading.
    """
    half = 4
    raw, m, l = _two_prototypes(half)
    blended = raw.clone()
    _blend_draft_seams(blended, m, l, half, window="triangle")
    assert _peak_jerk(blended) < 0.2 * _peak_jerk(raw)


def test_the_cosine_envelope_never_makes_a_narrow_blend_worse_than_no_blend():
    half = 4
    raw, m, l = _two_prototypes(half)
    blended = raw.clone()
    _blend_draft_seams(blended, m, l, half, window="cosine")
    assert _peak_jerk(blended) <= _peak_jerk(raw)


def test_nothing_outside_the_blend_window_moves():
    """Rigid: the envelope choice may not touch the prototypes' own interiors."""
    half = 6
    seam = 40
    a, m, l = _two_prototypes(half, seam=seam)
    b = a.clone()
    _blend_draft_seams(a, m, l, half, window="triangle")
    _blend_draft_seams(b, m, l, half, window="cosine")
    outside = torch.ones(len(a), dtype=torch.bool)
    outside[seam - half - 1:seam + half + 1] = False
    assert torch.equal(a[outside], b[outside])


def test_triangle_is_the_default_so_earlier_artifacts_reproduce():
    a, m, l = _two_prototypes(6)
    b = a.clone()
    _blend_draft_seams(a, m, l, 6)
    _blend_draft_seams(b, m, l, 6, window="triangle")
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# The wiring, which was broken on first write and would have been invisible.
#
# The first run of the cosine arms came back BIT-IDENTICAL to the triangle arms
# (max |full_pose difference| = 0.0) while the manifest recorded
# draft_seam_window = "cosine".  The flag reached the artifact's provenance but
# not the code: `_source_safe_draft` -- the path inference actually takes -- was
# not passing it through.  That is the shape this repository fears most, a
# switch whose artifact says it was applied when it was not, so the wiring gets
# its own test rather than being assumed from the CLI's existence.


# Every draft switch, not just the window.  The first version of this test
# checked seam_window alone, and --draft-seam-aware-retrieval was added the same
# afternoon and reached only ONE of the two inference call sites -- so its arms
# came back bit-identical to the control while the manifest said the flag was
# on, exactly the failure this file exists to prevent, a second time.  The list
# is asserted non-empty so it cannot rot into vacuous truth.
DRAFT_SWITCHES = ("seam_blend", "seam_window", "seam_aware_retrieval",
                  "recurrence_variety", "gap_fill", "bar_bounds",
                  "root_velocity_blend", "facing_anchor")


def test_every_call_path_into_build_draft_forwards_every_draft_switch():
    """Read the source: no build_draft/_source_safe_draft call may omit one.

    Parsed with ``ast`` rather than a regex.  The regex version of this test
    reported a false positive within minutes of being written: a call whose
    ``bar_bounds=(bar_bounds_of(...) if ... else None)`` contains a nested
    close-paren was truncated there, so the switches after it looked absent.
    A test that cries wolf about the wiring is worse than none, because the
    next real wiring bug gets waved through as "that test again".
    """
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree = ast.parse(open(os.path.join(root, "infer_atomic.py")).read())
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.attr if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", None))
        if name not in ("build_draft", "_source_safe_draft"):
            continue
        keywords = {k.arg for k in node.keywords if k.arg}
        if "seam_blend" in keywords:
            sites.append((node.lineno, keywords))
    assert len(sites) >= 3, (
        "expected at least three call sites forwarding seam_blend, found "
        "{} -- the pattern has gone stale".format(len(sites)))
    for lineno, keywords in sites:
        missing = [s for s in DRAFT_SWITCHES if s not in keywords]
        assert not missing, (
            "infer_atomic.py:{} forwards seam_blend but not {}".format(
                lineno, ", ".join(missing)))
