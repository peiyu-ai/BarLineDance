"""Tests for the captioner acceptance gate."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.caption_segments_vlm import FIELDS  # noqa: E402
from tools.probe_caption_quality import (  # noqa: E402
    field_agreement,
    permutation_p,
    probe,
    sample_pairs,
)


def write_captions(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def make_row(prototype, fields, index):
    full = {field: fields.get(field, "unspecified") for field in FIELDS}
    return {
        "recording_id": "r{}".format(index),
        "start": index * 30,
        "end": index * 30 + 30,
        "prototype": prototype,
        "fields": full,
        "summary": "s",
        "caption": "|".join(full[field] for field in FIELDS),
        "model": "test-vlm",
    }


def test_agreement_is_one_for_identical_fields():
    a = {field: "x" for field in FIELDS}
    assert field_agreement(a, dict(a)) == 1.0


def test_agreement_is_zero_when_every_field_differs():
    a = {field: "x" for field in FIELDS}
    b = {field: "y" for field in FIELDS}
    assert field_agreement(a, b) == 0.0


def test_within_pairs_come_from_one_prototype():
    rows = [make_row(1, {}, 0), make_row(1, {}, 1), make_row(2, {}, 2), make_row(2, {}, 3)]
    rng = np.random.default_rng(0)
    for i, j in sample_pairs(rows, same=True, count=50, rng=rng):
        assert rows[i]["prototype"] == rows[j]["prototype"]
        assert i != j


def test_across_pairs_come_from_different_prototypes():
    rows = [make_row(1, {}, 0), make_row(1, {}, 1), make_row(2, {}, 2), make_row(2, {}, 3)]
    rng = np.random.default_rng(0)
    for i, j in sample_pairs(rows, same=False, count=50, rng=rng):
        assert rows[i]["prototype"] != rows[j]["prototype"]


def test_single_prototype_yields_no_across_pairs_instead_of_crashing():
    rows = [make_row(1, {}, i) for i in range(4)]
    rng = np.random.default_rng(0)
    assert sample_pairs(rows, same=False, count=10, rng=rng) == []


def test_permutation_p_is_small_for_a_real_gap():
    rng = np.random.default_rng(0)
    within = np.full(200, 0.9)
    across = np.full(200, 0.2)
    assert permutation_p(within, across, rounds=500, rng=rng) < 0.01


def test_permutation_p_is_large_when_the_two_sides_are_the_same():
    rng = np.random.default_rng(0)
    pooled = rng.normal(size=400)
    p = permutation_p(pooled[:200], pooled[200:], rounds=500, rng=rng)
    assert p > 0.05


def test_a_consistent_and_varied_captioner_passes_both_gates(tmp_path):
    """Each prototype gets its own consistent description."""
    rows = []
    index = 0
    for prototype in range(6):
        for _ in range(10):
            rows.append(make_row(prototype, {
                "body_action": ["step", "jump", "turn", "slide", "spin", "wave"][prototype],
                "arms": ["down", "raised", "crossed", "extended", "circling", "waving"][prototype],
                "legs": "apart", "level": "middle", "travel": "in_place",
                "dynamics": "smooth"}, index))
            index += 1
    path = write_captions(tmp_path / "c.jsonl", rows)
    result = probe(path, pairs=300, rounds=300, seed=0)
    assert result["G2_consistency"]["within_prototype_agreement"] == 1.0
    assert result["G2_consistency"]["across_prototype_agreement"] < 1.0
    assert result["G2_consistency"]["permutation_p"] < 0.05
    assert result["G3_discriminative"]["distinct_captions"] == 6


def test_a_captioner_that_says_the_same_thing_about_everything_is_caught(tmp_path):
    """G2 looks perfect; G3 is what exposes it."""
    rows = [make_row(i % 6, {"body_action": "step", "arms": "down", "legs": "apart",
                             "level": "middle", "travel": "in_place",
                             "dynamics": "smooth"}, i)
            for i in range(60)]
    path = write_captions(tmp_path / "c.jsonl", rows)
    result = probe(path, pairs=300, rounds=300, seed=0)
    assert result["G2_consistency"]["within_prototype_agreement"] == 1.0
    assert result["G3_discriminative"]["distinct_captions"] == 1
    assert result["G3_discriminative"]["largest_caption_share"] == 1.0


def test_unspecified_rate_is_reported_per_field(tmp_path):
    rows = [make_row(i % 3, {"body_action": "step"}, i) for i in range(12)]
    path = write_captions(tmp_path / "c.jsonl", rows)
    result = probe(path, pairs=50, rounds=50, seed=0)
    rates = result["G1_usable"]["unspecified_rate_per_field"]
    assert rates["body_action"] == 0.0
    assert rates["arms"] == 1.0


def test_model_names_are_carried_into_the_result(tmp_path):
    rows = [make_row(i % 3, {"body_action": "step"}, i) for i in range(8)]
    path = write_captions(tmp_path / "c.jsonl", rows)
    assert probe(path, pairs=20, rounds=20, seed=0)["models"] == ["test-vlm"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_cross_agreement_reports_a_shuffled_baseline_next_to_the_match(tmp_path):
    """Two captioners that both answer 'step' constantly agree about nothing."""
    from tools.probe_caption_quality import cross_agreement

    fields = {"body_action": "step", "arms": "down", "legs": "apart",
              "level": "middle", "travel": "in_place", "dynamics": "smooth"}
    left = [{"recording_id": "r", "start": i, "end": i + 10, "prototype": 1,
             "fields": dict(fields), "caption": "c"} for i in range(20)]
    right = [dict(row) for row in left]
    result = cross_agreement(left, right, rng=np.random.default_rng(0))
    assert result["field_agreement"] == 1.0
    # Identical constant captions agree perfectly with themselves *and* with a
    # shuffle, so the lift is 1.0 -- no information.
    assert result["lift_over_chance"] == 1.0


def test_cross_agreement_lifts_when_the_captioners_track_the_segment(tmp_path):
    from tools.probe_caption_quality import cross_agreement

    def row(index, action):
        return {"recording_id": "r", "start": index, "end": index + 10, "prototype": 1,
                "fields": {"body_action": action, "arms": "down", "legs": "apart",
                           "level": "middle", "travel": "in_place", "dynamics": "smooth"},
                "caption": action}

    actions = ["jump", "slide", "spin", "kick", "wave", "roll", "lean", "drop"]
    left = [row(i, actions[i % len(actions)]) for i in range(40)]
    right = [dict(r) for r in left]
    result = cross_agreement(left, right, rng=np.random.default_rng(0))
    assert result["field_agreement"] > result["field_agreement_if_shuffled"]
    assert result["lift_over_chance"] > 1.0


def test_cross_agreement_says_so_when_the_files_do_not_overlap():
    from tools.probe_caption_quality import cross_agreement

    left = [{"recording_id": "a", "start": 0, "end": 10, "prototype": 1,
             "fields": {}, "caption": "x"}]
    right = [{"recording_id": "b", "start": 0, "end": 10, "prototype": 1,
              "fields": {}, "caption": "y"}]
    assert "skipped" in cross_agreement(left, right, rng=np.random.default_rng(0))
