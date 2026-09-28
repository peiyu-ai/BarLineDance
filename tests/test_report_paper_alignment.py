"""Tests for the AIST++ paper-alignment report.

The report's whole value is that its comparisons survive a corpus of a
different size, so the tests are built around that: a run that is the paper's
distribution stretched over half the footage must score zero distance, and a
run whose clusters are twice as uneven must not.  The paper-ledger arithmetic
is checked against the published figures by hand, because it is the part that
would silently drift if someone "corrected" a constant.
"""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.report_paper_alignment import (  # noqa: E402
    DURATION_EDGES,
    PAPER_FIG4A,
    build_parser,
    m1_report,
    m2_report,
    main,
    normalise,
    paper_self_consistency,
    prototype_counts,
    shares,
    total_variation,
)


def write_segmentation(path, durations_seconds, frames_total=None):
    """A segmentation.json holding one sequence made of the given segments."""
    frames = [int(round(d * 30)) for d in durations_seconds]
    starts = np.cumsum([0] + frames)
    record = {
        "sequence": "gBR_sBM_c01_d04_mBR0_ch01",
        "encoder": "torchvision/s3d KINETICS400_V1",
        "motion_frames": frames_total if frames_total is not None else int(starts[-1]),
        "boundaries": [int(b) for b in starts],
        "segments": [{"start": int(a), "end": int(b), "frames": int(b - a)}
                     for a, b in zip(starts[:-1], starts[1:])],
    }
    path.write_text(json.dumps({
        "features_dir": "data/aist_visual_s3d",
        "config": {"frames_per_cluster": 36, "min_length_frames": 20},
        "sequences": 1,
        "records": [record],
    }))
    return path


def write_labels(directory, counts, metric="euclidean_raw_mu"):
    """A labels directory whose centres reproduce the given per-prototype counts.

    Each prototype is a distinct axis-aligned direction, so nearest-centre
    assignment is exact and the acceptance thresholds admit everything.
    """
    directory.mkdir(parents=True, exist_ok=True)
    dimension = len(counts)
    centers = np.eye(dimension) * 10.0
    embeddings = np.concatenate([np.repeat(centers[i:i + 1], n, axis=0)
                                 for i, n in enumerate(counts)])
    np.savez(directory / "producer.npz", centers=centers,
             thresholds=np.full(dimension, 1.0), classes=np.array([dimension]))
    (directory / "report.json").write_text(json.dumps(
        {"clustering": {"embedding_metric": metric, "acceptance_rate": 1.0}}))
    cache = directory / "embeddings.npz"
    np.savez(cache, embeddings=embeddings.astype(np.float32))
    return cache


def test_paper_ledger_matches_the_published_figures():
    ledger = paper_self_consistency()
    # 0*1366 + 0.7*7712 + 0.9*8054 + 1.1*2095 + 1.3*3967, by hand.
    assert ledger["fig4a_seconds_at_bucket_floors"] == pytest.approx(20108.6, abs=0.1)
    assert ledger["fig4a_segments"] == 23194
    # Fig. 4a does not fit 5.2 h even at the floors, and Fig. 4b claims more
    # clustered segments than Fig. 4a segmented at all.
    assert ledger["fig4a_exceeds_corpus_by"] > 0
    assert ledger["fig4b_over_fig4a"] > 1.0
    assert ledger["fig4c_over_fig4a"] == pytest.approx(1.0, abs=0.01)


def test_distance_is_zero_for_the_paper_distribution_on_a_smaller_corpus(tmp_path):
    """Half the footage, same shape -> no distance.  That is the point."""
    midpoints = {"<0.7": 0.5, "0.7-0.9": 0.8, "0.9-1.1": 1.0, "1.1-1.3": 1.2, ">1.3": 1.5}
    durations = []
    for name, count in PAPER_FIG4A.items():
        durations.extend([midpoints[name]] * (count // 2))
    report = m1_report(write_segmentation(tmp_path / "seg.json", durations))
    # Not exactly zero only because halving odd bucket counts rounds.
    assert report["total_variation_from_fig4a"] == pytest.approx(0.0, abs=1e-3)
    # ... while the raw count is half, which is why the rate is what gets read.
    assert report["segments"] == sum(count // 2 for count in PAPER_FIG4A.values())


def test_segments_per_hour_uses_footage_not_segment_span(tmp_path):
    """Rate is per hour of corpus, so trailing unsegmented frames still count."""
    path = write_segmentation(tmp_path / "seg.json", [1.0] * 1800, frames_total=30 * 3600)
    report = m1_report(path)
    assert report["hours"] == pytest.approx(1.0)
    assert report["segments_per_hour"] == pytest.approx(1800.0)


def test_shape_comparison_ignores_cluster_density(tmp_path):
    """A vocabulary a quarter as dense, with the paper's shape, scores zero."""
    quarter = {"<200": 175 / 4, "200-250": 225 / 4, "250-300": 275 / 4,
               "300-350": 325 / 4, ">350": 375 / 4}
    counts = []
    for name, count in {"<200": 10, "200-250": 31, "250-300": 32,
                        "300-350": 18, ">350": 9}.items():
        counts.extend([int(round(quarter[name]))] * count)
    cache = write_labels(tmp_path / "labels", counts)
    report = m2_report(tmp_path / "labels", cache, hours=1.0)
    assert report["samples_per_prototype"] < 100      # far below the paper's 268.57
    assert report["shape_total_variation_from_fig4b"] < 0.05


def test_uneven_clusters_are_penalised(tmp_path):
    counts = [10] * 50 + [200] * 50
    cache = write_labels(tmp_path / "labels", counts)
    report = m2_report(tmp_path / "labels", cache, hours=1.0)
    assert report["dispersion_sd_over_mean"] > 0.5
    assert report["shape_total_variation_from_fig4b"] > 0.4


def test_cosine_runs_are_scored_on_the_sphere(tmp_path):
    """The metric recorded by the run decides how its embeddings are read."""
    counts = [7] * 100
    cache = write_labels(tmp_path / "labels", counts, metric="cosine_l2_normalized_mu")
    # Centres are 10 units out; unit-normalised embeddings sit at distance ~9
    # from them, so nothing would be accepted if the tool skipped the
    # normalisation -- and everything is accepted if it applies it.
    (tmp_path / "labels" / "producer.npz").unlink()
    np.savez(tmp_path / "labels" / "producer.npz",
             centers=np.eye(100), thresholds=np.full(100, 0.5), classes=np.array([100]))
    counted = prototype_counts(tmp_path / "labels", cache)
    assert counted.sum() == sum(counts)


def test_shares_and_total_variation_agree_with_the_definition():
    values = [0.5, 0.8, 1.0, 1.2, 2.0]
    fractions = shares(values, DURATION_EDGES)
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert total_variation(fractions, fractions) == pytest.approx(0.0)
    assert 0.0 <= total_variation(fractions, normalise(PAPER_FIG4A)) <= 1.0


def test_labels_without_embeddings_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--labels", str(tmp_path)])


def test_parser_accepts_a_segmentation_only_run():
    args = build_parser().parse_args(["--segmentation", "seg.json"])
    assert args.labels is None and args.segmentation == pathlib.Path("seg.json")
