"""Tests for fitting Alg. 1's two unspecified parameters to Fig. 4a."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.calibrate_segmentation import (  # noqa: E402
    histogram,
    paper_shares,
    parse_grid,
    total_variation,
)
from tools.segment_visual_atomics import segment_sequence  # noqa: E402


def test_histogram_shares_sum_to_one():
    shares = histogram([0.5, 0.8, 1.0, 1.2, 2.0, 2.0])
    assert sum(shares.values()) == pytest.approx(1.0)


def test_paper_shares_match_the_published_counts():
    shares = paper_shares()
    assert sum(shares.values()) == pytest.approx(1.0)
    # 1366 of 23194 segments are under 0.7 s in Fig. 4a.
    assert shares["<0.7"] == pytest.approx(1366 / 23194)


def test_total_variation_is_zero_for_an_exact_match_and_one_for_disjoint():
    reference = paper_shares()
    assert total_variation(reference, reference) == pytest.approx(0.0)
    disjoint = {name: 0.0 for name in reference}
    disjoint["<0.7"] = 1.0
    assert total_variation(disjoint, {**{n: 0.0 for n in reference}, ">1.3": 1.0}) == \
        pytest.approx(1.0)


def test_total_variation_reads_as_the_share_in_the_wrong_bucket():
    a = {"<0.7": 0.5, "0.7-0.9": 0.5, "0.9-1.1": 0.0, "1.1-1.3": 0.0, ">1.3": 0.0}
    b = {"<0.7": 0.3, "0.7-0.9": 0.7, "0.9-1.1": 0.0, "1.1-1.3": 0.0, ">1.3": 0.0}
    assert total_variation(a, b) == pytest.approx(0.2)


def test_parse_grid_reads_pairs():
    assert parse_grid("30:12, 40:24") == [(30, 12), (40, 24)]


def test_min_length_is_a_floor_the_segmenter_never_breaks():
    """The bug this whole calibration exists for: L_min = 24 made the paper's
    '<0.7 s' bucket unreachable, so the histogram could never match."""
    rng = np.random.default_rng(0)
    features = np.concatenate([rng.normal(centre, 0.1, size=(50, 8))
                               for centre in range(6)])
    for min_length in (6, 12, 24):
        boundaries, _ = segment_sequence(features, 8, min_length, 4.0, 0)
        lengths = np.diff(boundaries)
        assert lengths.min() >= min_length, (min_length, lengths)


def test_a_smaller_floor_admits_shorter_segments():
    rng = np.random.default_rng(1)
    features = np.concatenate([rng.normal(centre, 0.1, size=(20, 8))
                               for centre in range(10)])
    short, _ = segment_sequence(features, 10, 6, 4.0, 0)
    long, _ = segment_sequence(features, 10, 24, 4.0, 0)
    assert np.diff(short).min() <= np.diff(long).min()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_boundary_contrast_separates_real_cuts_from_clock_cuts():
    """The guard against fitting Fig. 4a by turning Alg. 1 into a uniform chopper."""
    from tools.calibrate_segmentation import boundary_contrast

    rng = np.random.default_rng(0)
    blocks = [np.tile(rng.normal(size=(1, 16)), (40, 1)) + 0.01 * rng.normal(size=(40, 16))
              for _ in range(6)]
    array = np.concatenate(blocks)
    on_content = boundary_contrast(array, [0, 40, 80, 120, 160, 200, 240], window=5,
                                   rng=np.random.default_rng(1))
    on_clock = boundary_contrast(array, [0, 35, 70, 105, 140, 175, 210, 240], window=5,
                                 rng=np.random.default_rng(1))
    assert on_content > 0.5
    assert on_clock < on_content


def test_boundary_contrast_is_none_when_there_is_no_interior_cut():
    from tools.calibrate_segmentation import boundary_contrast

    array = np.random.default_rng(0).normal(size=(30, 8))
    assert boundary_contrast(array, [0, 30], window=5,
                             rng=np.random.default_rng(0)) is None
