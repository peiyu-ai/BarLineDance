import numpy as np
import pytest

from tools.bootstrap_fid_arms import by_clip, clip_of, fid_for


def test_the_seed_suffix_is_stripped_so_a_clip_resamples_as_a_unit():
    """Four seeds of one clip are not four independent draws of a dance.

    Resampling sequences instead of clips would treat them as independent and
    report an interval about four times too narrow -- which is the direction
    that makes an unseparated pair look separated.
    """
    assert clip_of("wild_v4:7319142177966181632:clip000_seed20260816") == \
        "wild_v4:7319142177966181632:clip000"
    assert clip_of("wild_v4:7319142177966181632:clip000") == \
        "wild_v4:7319142177966181632:clip000"


def test_grouping_keeps_every_sequence_and_keys_it_by_its_clip():
    features = np.arange(12, dtype=np.float64).reshape(4, 3)
    stems = ["a_seed1", "a_seed2", "b_seed1", "b_seed2"]
    groups = by_clip(features, stems)
    assert list(groups) == ["a", "b"]
    assert sum(len(v) for v in groups.values()) == 4


def test_fid_is_zero_against_the_distribution_it_came_from():
    """A gate that cannot read zero on identical distributions is not a gate."""
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(400, 6))
    groups = by_clip(reference, [f"c{i}" for i in range(len(reference))])
    value = fid_for(groups, sorted(groups), reference)
    assert value == pytest.approx(0.0, abs=1e-6)


def test_fid_here_is_invariant_to_a_per_dimension_rescale_of_the_features():
    """``eval.metrics.normalize`` standardizes each distribution *separately*.

    So any per-dimension affine transform of the generated features cancels
    before the comparison.  Verified on the real 260-sequence kinetic set on
    2026-08-23: multiplying every feature by 0.25 or by 2.0 leaves fid_k at
    9.760249 to six decimals.

    This is not the same as "blind to a soft dance".  Halving real motion is
    *not* a per-dimension constant on these features -- the ratios spread
    0.29-1.00 -- and it does move the number (4.012 -> 5.661 on a 60/60 split).
    What the standardization removes is the homogeneous part of an amplitude
    difference; what survives is whatever the feature map happens to do
    inhomogeneously, which nobody designed.  Pinned here so the distinction is
    not re-derived as "FID cannot see amplitude".
    """
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(300, 6))
    generated = rng.normal(loc=0.4, scale=1.3, size=(300, 6))
    groups = by_clip(generated, [f"c{i}" for i in range(len(generated))])
    base = fid_for(groups, sorted(groups), reference)
    for factor in (0.25, 2.0):
        scaled = by_clip(generated * factor,
                         [f"c{i}" for i in range(len(generated))])
        assert fid_for(scaled, sorted(scaled), reference) == pytest.approx(base, abs=1e-9)
    assert base > 0.0, "the fixture must not be a zero-distance pair"
