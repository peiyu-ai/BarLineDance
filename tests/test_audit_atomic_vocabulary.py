"""Tests for the discovery audit: does it actually detect a bad vocabulary?

The audit exists to fail when re-clustering produced nothing, so the tests that
matter are the ones where the answer should be "no": a random split of each
prototype must not pass the coherence gate, and a shape-perfect vocabulary must
not pass on shape alone.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.audit_atomic_vocabulary import (  # noqa: E402
    bucket_durations,
    bucket_sizes,
    coherence,
    decode_ids,
    rescale,
    verdict,
)


def test_duration_buckets_are_half_open_and_cover_everything():
    counts = bucket_durations([0.5, 0.7, 0.9, 1.1, 1.3, 4.0])
    assert counts == {"<0.7": 1, "0.7-0.9": 1, "0.9-1.1": 1, "1.1-1.3": 1, ">1.3": 2}
    assert sum(counts.values()) == 6


def test_size_buckets_place_the_edge_in_the_upper_bucket():
    edges = [(0, 20, "<20"), (20, 35, "20-35"), (35, np.inf, ">=35")]
    assert bucket_sizes([19, 20, 34, 35], edges) == {"<20": 1, "20-35": 2, ">=35": 1}


def test_rescale_preserves_shape_and_total():
    scaled = rescale({"a": 1, "b": 3}, 800)
    assert scaled == {"a": 200.0, "b": 600.0}


def test_decode_ids_recovers_prototype_and_subprototype(tmp_path):
    """The published ids are compacted; producer.npz is how they decode back."""
    path = tmp_path / "producer.npz"
    # width 5: raw id = (prototype - 1) * 5 + sub + 1
    np.savez(path, width=np.asarray([5]), used_raw_ids=np.asarray([1, 3, 6, 11]))
    mapping = decode_ids(path)
    assert mapping == {1: (1, 0), 2: (1, 2), 3: (2, 0), 4: (3, 0)}


def _planted(rng, *, separation):
    """Two prototypes, each split into two sub-prototypes `separation` apart."""
    features, ids, prototypes = [], [], []
    label = 1
    for prototype in (1, 2):
        for sub in (0, 1):
            centre = np.array([prototype * 10.0, sub * separation])
            features.append(rng.normal(centre, 0.4, size=(40, 2)))
            ids.extend([label] * 40)
            prototypes.extend([prototype] * 40)
            label += 1
    return np.concatenate(features), np.asarray(ids), np.asarray(prototypes)


def test_coherence_finds_a_real_split():
    rng = np.random.default_rng(0)
    features, ids, prototypes = _planted(rng, separation=6.0)
    result = coherence(features, ids, prototypes, pairs=600, rounds=300, rng=rng)
    assert result["ratio"] > 1.2
    assert result["permutation_p"] < 0.01


def test_coherence_rejects_a_random_split():
    """A vocabulary can have perfect shape statistics and be noise."""
    rng = np.random.default_rng(1)
    features, ids, prototypes = _planted(rng, separation=0.0)   # sub-prototypes coincide
    shuffled = rng.permutation(ids)
    result = coherence(features, shuffled, prototypes, pairs=600, rounds=300, rng=rng)
    assert result["ratio"] < 1.02 or result["permutation_p"] > 0.01


def test_coherence_says_so_when_no_prototype_was_split():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(30, 2))
    ids = np.ones(30, dtype=np.int64)
    prototypes = np.ones(30, dtype=np.int64)
    assert "skipped" in coherence(features, ids, prototypes, pairs=50, rounds=50, rng=rng)


def _report(**overrides):
    base = {
        "mean_subprototypes_per_prototype": 7.0,
        "mean_samples_per_subprototype": 31.0,
        "subprototypes": 700,
        "coherence_tmr": {"ratio": 1.2, "permutation_p": 0.001},
        "samples_per_subprototype": {"ours": {"<20": 100}},
    }
    base.update(overrides)
    return base


def test_verdict_passes_a_vocabulary_shaped_like_the_paper():
    assert verdict(_report())["pass"] is True


def test_verdict_fails_when_coherence_is_absent_even_if_shape_is_perfect():
    """Shape alone must never be enough -- a random split reproduces it."""
    result = verdict(_report(coherence_tmr={"skipped": "x"}))
    assert result["pass"] is False


def test_verdict_fails_a_coherence_gap_that_could_be_chance():
    result = verdict(_report(coherence_tmr={"ratio": 1.19, "permutation_p": 0.4}))
    assert result["pass"] is False


def test_verdict_fails_when_the_vocabulary_collapses_to_one_class_per_prototype():
    result = verdict(_report(mean_subprototypes_per_prototype=1.0))
    assert result["pass"] is False


def test_verdict_fails_when_sub_prototypes_are_mostly_too_small_to_train_on():
    result = verdict(_report(samples_per_subprototype={"ours": {"<20": 500}},
                             subprototypes=700))
    assert result["pass"] is False


def test_every_check_carries_its_threshold_in_the_text():
    for check in verdict(_report())["checks"]:
        assert any(character.isdigit() for character in check["criterion"]), check


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def _captioned(field_by_sub):
    """Segments whose caption fields follow their sub-prototype."""
    captions, owners, ids, prototypes = {}, [], [], []
    for sub, action in field_by_sub.items():
        for index in range(10):
            owner = ("r{}".format(sub), index, index + 10)
            captions[owner] = {"body_action": action, "arms": "down", "legs": "apart",
                               "level": "middle", "travel": "in_place",
                               "dynamics": "smooth"}
            owners.append(owner)
            ids.append(sub)
            prototypes.append(1)
    return captions, owners, np.asarray(ids), np.asarray(prototypes)


def test_caption_purity_is_high_when_the_grouping_followed_the_captions():
    from tools.audit_atomic_vocabulary import caption_purity

    captions, owners, ids, prototypes = _captioned({1: "jump", 2: "slide", 3: "spin"})
    result = caption_purity(captions, owners, ids, prototypes, pairs=400,
                            rng=np.random.default_rng(0))
    assert result["within_subprototype_agreement"] > result["across_subprototype_agreement"]
    assert result["lift"] > 1.0


def test_caption_purity_is_flat_when_the_grouping_ignored_them():
    from tools.audit_atomic_vocabulary import caption_purity

    captions, owners, ids, prototypes = _captioned({1: "jump", 2: "jump", 3: "jump"})
    result = caption_purity(captions, owners, ids, prototypes, pairs=400,
                            rng=np.random.default_rng(0))
    assert result["lift"] == pytest.approx(1.0, abs=1e-6)


def test_caption_purity_says_so_when_nothing_is_captioned():
    from tools.audit_atomic_vocabulary import caption_purity

    _, owners, ids, prototypes = _captioned({1: "jump", 2: "slide"})
    result = caption_purity({}, owners, ids, prototypes, pairs=100,
                            rng=np.random.default_rng(0))
    assert "skipped" in result
