import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.cluster_atomics_tmr import (
    REJECTED,
    TRANSITION,
    ClusterError,
    drop_non_finite,
    kmeans,
    squared_distances,
)


class KMeansTests(unittest.TestCase):
    def test_recovers_well_separated_groups(self):
        rng = np.random.default_rng(0)
        truth = np.array([0] * 40 + [1] * 40 + [2] * 40)
        centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        points = centers[truth] + rng.normal(scale=0.2, size=(120, 2))
        assignment, found = kmeans(points, 3, seed=1)
        # Labels are arbitrary; the partition is what must match.
        for group in range(3):
            members = assignment[truth == group]
            self.assertEqual(len(set(members.tolist())), 1)
        self.assertEqual(len(set(assignment.tolist())), 3)
        self.assertEqual(found.shape, (3, 2))

    def test_is_deterministic_for_a_given_seed(self):
        points = np.random.default_rng(2).normal(size=(80, 5))
        first, _ = kmeans(points, 4, seed=7)
        second, _ = kmeans(points, 4, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_a_different_seed_is_allowed_to_differ(self):
        points = np.random.default_rng(3).normal(size=(80, 5))
        _, centers_a = kmeans(points, 4, seed=7)
        _, centers_b = kmeans(points, 4, seed=8)
        self.assertEqual(centers_a.shape, centers_b.shape)

    def test_refuses_more_clusters_than_points(self):
        with self.assertRaises(ClusterError):
            kmeans(np.zeros((3, 2)), 5, seed=0)

    def test_no_cluster_is_left_empty_when_the_data_can_fill_them(self):
        # Heavily duplicated points strand centres without the re-seed rule.
        # Four distinct locations, so four non-empty clusters are achievable;
        # asking for more clusters than distinct points would not be.
        points = np.repeat(
            np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0], [0.0, 10.0]]), 30, axis=0)
        assignment, centers = kmeans(points, 4, seed=4)
        self.assertEqual(len(set(assignment.tolist())), 4)
        self.assertEqual(len(np.unique(centers, axis=0)), 4)


class LabelConventionTests(unittest.TestCase):
    def test_transition_and_rejected_are_distinct_codes(self):
        # 0 means "between movements"; -1 means "not confidently assignable".
        # Collapsing them would tell the planner a rejected frame is a
        # transition it should learn to place.
        self.assertEqual(TRANSITION, 0)
        self.assertEqual(REJECTED, -1)
        self.assertNotEqual(TRANSITION, REJECTED)


class AcceptQuantileTests(unittest.TestCase):
    def test_thresholds_fitted_on_train_are_applied_to_other_splits(self):
        # Reproduces the tool's acceptance rule in miniature: a val point far
        # from its centre must be rejected by the *train* threshold, not by a
        # threshold recomputed on val (which would accept 85% of val whatever
        # its true spread).
        train_distances = np.array([0.1, 0.2, 0.3, 0.4, 1.0])
        threshold = np.quantile(train_distances, 0.85)
        val_distances = np.array([0.15, 5.0])
        accepted = val_distances <= threshold
        self.assertTrue(accepted[0])
        self.assertFalse(accepted[1])
        # The threshold must come from train: it is 0.4-ish here, far tighter
        # than anything val's own spread would produce.
        self.assertLess(threshold, val_distances[1])


if __name__ == "__main__":
    unittest.main()


class SquaredDistanceTests(unittest.TestCase):
    def test_matches_the_literal_broadcast_form(self):
        from tools.cluster_atomics_tmr import squared_distances

        rng = np.random.default_rng(11)
        features = rng.normal(size=(37, 8))
        centers = rng.normal(size=(5, 8))
        literal = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        np.testing.assert_allclose(squared_distances(features, centers), literal, atol=1e-9)

    def test_never_returns_a_negative_distance(self):
        from tools.cluster_atomics_tmr import squared_distances

        # Identical points make the expansion cancel to ~0, which floating
        # point can push slightly negative before the clip.
        points = np.full((6, 4), 3.14159)
        self.assertTrue((squared_distances(points, points[:2]) >= 0).all())


class DiscardPolicyTests(unittest.TestCase):
    def test_the_two_policies_are_distinguishable_and_recorded(self):
        from tools.cluster_atomics_tmr import REJECTED, TRANSITION

        # The paper says y_i = 0 means "no atomic movement at frame i", so a
        # discarded ambiguous segment belongs at 0.  The older repo convention
        # masks it out instead.  These must stay separable, because under the
        # masking policy the planner receives no transition supervision at all.
        self.assertNotEqual(TRANSITION, REJECTED)
        frames = 10
        as_transition = np.full(frames, TRANSITION, dtype=np.int64)
        as_rejected = np.full(frames, REJECTED, dtype=np.int64)
        mask_transition = np.ones(frames, dtype=bool)
        mask_rejected = np.zeros(frames, dtype=bool)
        self.assertEqual(int(mask_transition.sum()), frames)   # stays in the loss
        self.assertEqual(int(mask_rejected.sum()), 0)          # leaves the loss
        self.assertFalse(np.array_equal(as_transition, as_rejected))


class EmbeddingCacheTests(unittest.TestCase):
    """The cache exists so two metrics can be compared on identical embeddings."""

    def test_round_trip_preserves_embeddings_owners_and_skips(self):
        from tools.cluster_atomics_tmr import load_embedding_cache, save_embedding_cache

        encoded = {
            "embeddings": np.arange(12, dtype=np.float32).reshape(3, 4),
            "owners": [("tiktok:a:clip001", 0, 30), ("tiktok:a:clip001", 30, 61),
                       ("aist/gBR_sBM_cAll_d04", 5, 44)],
            "skipped": {"no_motion": 2, "convert_failed": 0, "too_short": 7},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emb.npz"
            save_embedding_cache(path, encoded, "fingerprint-a")
            restored = load_embedding_cache(path, "fingerprint-a")
        np.testing.assert_array_equal(restored["embeddings"], encoded["embeddings"])
        self.assertEqual(restored["owners"], encoded["owners"])
        self.assertEqual(restored["skipped"], encoded["skipped"])

    def test_refuses_a_cache_built_from_different_inputs(self):
        # Serving embeddings computed from another segmentation would produce a
        # vocabulary that looks healthy while describing spans that moved.
        from tools.cluster_atomics_tmr import load_embedding_cache, save_embedding_cache

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emb.npz"
            save_embedding_cache(
                path,
                {"embeddings": np.zeros((1, 2), dtype=np.float32),
                 "owners": [("x", 0, 1)], "skipped": {}},
                "fingerprint-a")
            with self.assertRaises(ClusterError):
                load_embedding_cache(path, "fingerprint-b")

    def test_fingerprint_tracks_segmentation_content_not_its_name(self):
        from tools.cluster_atomics_tmr import cache_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seg = root / "segmentation.json"
            seg.write_text('{"records": []}', encoding="utf-8")
            args = dict(bundle=root, segmentation_path=seg, normalized_sequences=None,
                        normalizer_bundle=None, min_segment_frames=12, limit=None)
            before = cache_fingerprint(**args)
            seg.write_text('{"records": [1]}', encoding="utf-8")
            self.assertNotEqual(before, cache_fingerprint(**args))


class MetricTests(unittest.TestCase):
    """L2 normalising is a different vocabulary, not a cosmetic switch."""

    @staticmethod
    def _direction_vs_magnitude():
        # Two nearby directions, each present at a small and a large magnitude.
        # The directions are 30 degrees apart, so at radius 30 they are 15.5
        # apart while the two radii are 29 apart: Euclidean k=2 minimises SSE by
        # cutting on magnitude.  Cosine k=2 cannot see magnitude at all and cuts
        # on direction, which is the axis TMR's contrastive objective shaped.
        rng = np.random.default_rng(5)
        angle = np.deg2rad(15.0)
        left = np.array([np.cos(angle), np.sin(angle)])
        right = np.array([np.cos(-angle), np.sin(-angle)])
        points = np.concatenate([
            left * 1.0 + rng.normal(scale=0.01, size=(20, 2)),
            right * 1.0 + rng.normal(scale=0.01, size=(20, 2)),
            left * 30.0 + rng.normal(scale=0.01, size=(20, 2)),
            right * 30.0 + rng.normal(scale=0.01, size=(20, 2)),
        ])
        magnitude_group = np.array([0] * 40 + [1] * 40)
        direction_group = np.array(([0] * 20 + [1] * 20) * 2)
        return points, magnitude_group, direction_group

    @staticmethod
    def _agrees_with(assignment, truth):
        return all(len(set(assignment[truth == g].tolist())) == 1 for g in set(truth.tolist()))

    def test_raw_kmeans_cuts_by_magnitude_and_normalized_cuts_by_direction(self):
        points, magnitude_group, direction_group = self._direction_vs_magnitude()
        raw, _ = kmeans(points, 2, seed=1)
        self.assertTrue(self._agrees_with(raw, magnitude_group))
        self.assertFalse(self._agrees_with(raw, direction_group))

        unit = points / np.linalg.norm(points, axis=1, keepdims=True)
        normalized, _ = kmeans(unit, 2, seed=1)
        self.assertTrue(self._agrees_with(normalized, direction_group))
        self.assertFalse(self._agrees_with(normalized, magnitude_group))

    def test_the_two_runs_declare_different_label_spaces(self):
        # Sharing a label_space_id would let a planner trained on one be scored
        # against the other with nothing to flag it.
        import inspect

        from tools import cluster_atomics_tmr

        source = inspect.getsource(cluster_atomics_tmr.build)
        self.assertIn('"_l2" if l2_normalize else ""', source)
        self.assertIn('"embedding_metric": metric', source)
        self.assertIn("cosine_l2_normalized_mu", source)


class AssetPathTests(unittest.TestCase):
    def test_recorded_paths_are_relative_to_the_manifest(self):
        # Absolute paths would name the .staging directory the atomic publish
        # renames away, leaving a complete-looking bundle whose every row
        # points at nothing.  Relative paths also keep the bundle relocatable.
        import inspect

        from tools import cluster_atomics_tmr

        source = inspect.getsource(cluster_atomics_tmr.build)
        self.assertIn('"labels_path": "labels/{}/labels.npy"', source)
        self.assertIn('"producer_artifact": "producer.npz"', source)
        self.assertNotIn('str((store / "labels.npy").resolve())', source)


class NonFiniteEmbeddingTests(unittest.TestCase):
    """One NaN embedding is enough to empty the whole vocabulary.

    Measured on wild_v5_song, 2026-08-25: one guofeat row of one clip carried
    NaN in the six columns of joint 19's rot6d -- a degenerate cross product in
    TMR's quaternion path, on motion that was itself finite and well scaled
    (max |v| 1.51).  That is 1 row of 238,115.  The run exited 0 and published
    a bundle in which all 7,234,580 frames were "transition", 100 of 100
    prototypes were empty and the acceptance rate was 0.0000, against 0.8437 on
    the previous corpus.  Nothing raised.
    """

    def test_a_nan_center_makes_every_distance_column_nan(self):
        # The mechanism, in miniature: this is why one bad row is not one bad
        # segment.  argmin over a NaN column returns that column for every
        # point, so every segment lands in the same prototype at distance NaN,
        # and `nearest <= threshold` is then False everywhere.
        points = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        centers = np.array([[0.0, 0.0], [np.nan, np.nan]])
        distances = np.sqrt(squared_distances(points, centers))
        assignment = distances.argmin(axis=1)
        nearest = distances[np.arange(len(points)), assignment]
        self.assertTrue(np.isnan(nearest).all())
        self.assertFalse((nearest <= np.array([1.0, 1.0])[assignment]).any())

    def test_the_bad_row_is_dropped_and_the_owners_stay_aligned(self):
        embeddings = np.array([[1.0, 2.0], [np.nan, 0.0], [3.0, 4.0]])
        encoded = {"embeddings": embeddings,
                   "owners": [("a", 0, 10), ("b", 0, 10), ("c", 0, 10)],
                   "skipped": {}}
        self.assertEqual(drop_non_finite(encoded), 1)
        np.testing.assert_array_equal(encoded["embeddings"],
                                      np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(encoded["owners"], [("a", 0, 10), ("c", 0, 10)])
        self.assertEqual(encoded["skipped"]["non_finite_embeddings"], 1)

    def test_an_infinity_counts_too(self):
        encoded = {"embeddings": np.array([[np.inf, 0.0], [1.0, 1.0]]),
                   "owners": [("a", 0, 10), ("b", 0, 10)], "skipped": {}}
        self.assertEqual(drop_non_finite(encoded), 1)
        self.assertEqual(encoded["owners"], [("b", 0, 10)])

    def test_a_clean_run_drops_nothing_and_says_so(self):
        encoded = {"embeddings": np.array([[1.0, 2.0], [3.0, 4.0]]),
                   "owners": [("a", 0, 10), ("b", 0, 10)], "skipped": {}}
        self.assertEqual(drop_non_finite(encoded), 0)
        self.assertEqual(len(encoded["owners"]), 2)
        # Recorded as 0 rather than omitted: "none were dropped" and "nobody
        # looked" read the same in a report that only lists non-zero counts.
        self.assertEqual(encoded["skipped"]["non_finite_embeddings"], 0)

    def test_it_refuses_rather_than_returning_an_empty_set(self):
        encoded = {"embeddings": np.array([[np.nan, np.nan]]),
                   "owners": [("a", 0, 10)], "skipped": {}}
        with self.assertRaises(ClusterError):
            drop_non_finite(encoded)

    def test_a_cache_written_before_this_check_is_still_cleaned(self):
        # The drop runs on whatever `build` holds, encoded or cached: the
        # embedding cache is keyed by its inputs, which have not changed, so a
        # cache carrying the bad row is served again and would fail again.
        cached = {"embeddings": np.array([[1.0, 0.0], [np.nan, 0.0]]),
                  "owners": [("a", 0, 10), ("b", 0, 10)],
                  "skipped": {"no_motion": 3087, "too_short": 77}}
        self.assertEqual(drop_non_finite(cached), 1)
        self.assertEqual(cached["skipped"]["no_motion"], 3087)
        self.assertEqual(cached["skipped"]["non_finite_embeddings"], 1)


class DegenerateVocabularyGuardTests(unittest.TestCase):
    """The invariants the guard rests on, checked rather than asserted.

    Both are exact consequences of the code above it, not tuned bounds, which
    is what makes refusing on them safe: ``kmeans`` re-seeds an empty cluster
    on the worst-explained point, and a cluster's threshold is a quantile of
    its own train members' distances, so it accepts at least the member that
    set it.
    """

    def test_every_cluster_keeps_a_member_so_no_prototype_can_be_empty(self):
        rng = np.random.default_rng(11)
        points = np.concatenate([rng.normal(loc, 0.1, size=(30, 3))
                                 for loc in (0.0, 4.0, 8.0, 12.0)])
        assignment, centers = kmeans(points, 4, seed=5)
        sizes = np.array([(assignment == i).sum() for i in range(4)])
        self.assertTrue(sizes.all())

    def test_the_train_quantile_accepts_at_least_its_own_members(self):
        rng = np.random.default_rng(12)
        points = np.concatenate([rng.normal(loc, 0.1, size=(50, 3))
                                 for loc in (0.0, 6.0)])
        assignment, centers = kmeans(points, 2, seed=6)
        distances = np.sqrt(squared_distances(points, centers))
        nearest = distances[np.arange(len(points)), assignment]
        thresholds = np.array([np.quantile(nearest[assignment == i], 0.85)
                               for i in range(2)])
        accepted = nearest <= thresholds[assignment]
        self.assertGreater(accepted.sum(), 0)
        # ~85% by construction; the guard only refuses at zero, so this is the
        # margin it is refusing against, not the threshold itself.
        self.assertAlmostEqual(accepted.mean(), 0.85, delta=0.05)


class FrozenProducerTests(unittest.TestCase):
    """Adding clips must not re-draw the vocabulary every checkpoint is bound to.

    Re-clustering identical T-line inputs moved 85 of 2,015 segments to another
    prototype (2026-09-22); assigning to the published producer kept 2,012.
    """

    def test_assignment_uses_the_given_thresholds(self):
        from tools.cluster_atomics_tmr import assign_to_centers

        centers = np.array([[0.0, 0.0], [10.0, 0.0]])
        thresholds = np.array([1.0, 0.5])
        points = np.array([[0.5, 0.0], [9.0, 0.0], [10.2, 0.0], [5.0, 5.0]])
        assignment, nearest, accepted = assign_to_centers(points, centers, thresholds)
        np.testing.assert_array_equal(assignment, [0, 1, 1, 0])
        # 9.0 is nearest prototype 1 but outside its tighter threshold.
        np.testing.assert_array_equal(accepted, [True, False, True, False])
        self.assertAlmostEqual(float(nearest[1]), 1.0)

    def test_a_fit_reassigned_to_its_own_producer_is_unchanged(self):
        from tools.cluster_atomics_tmr import assign_to_centers

        rng = np.random.default_rng(5)
        points = np.concatenate([rng.normal(loc, 0.3, size=(40, 3)) for loc in (0.0, 4.0, 8.0)])
        _, centers = kmeans(points, 3, seed=2)
        distances = np.sqrt(squared_distances(points, centers))
        fit_assignment = distances.argmin(axis=1)
        nearest = distances[np.arange(len(points)), fit_assignment]
        thresholds = np.array([np.quantile(nearest[fit_assignment == k], 0.85) for k in range(3)])
        assignment, _, accepted = assign_to_centers(points, centers, thresholds)
        np.testing.assert_array_equal(assignment, fit_assignment)
        np.testing.assert_array_equal(accepted, nearest <= thresholds[fit_assignment])

    def test_a_producer_that_does_not_match_the_run_is_refused(self):
        from tools.cluster_atomics_tmr import load_frozen_producer

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "producer.npz"
            np.savez_compressed(path, centers=np.zeros((3, 2)), thresholds=np.ones(3),
                                classes=np.asarray([3]), seed=np.asarray([1]),
                                accept_quantile=np.asarray([0.85]))
            (Path(tmp) / "report.json").write_text(json.dumps(
                {"clustering": {"embedding_metric": "euclidean_raw_mu"}}))
            ok = load_frozen_producer(path, classes=3, accept_quantile=0.85,
                                      metric="euclidean_raw_mu")
            self.assertEqual(ok["centers"].shape, (3, 2))
            with self.assertRaises(ClusterError):
                load_frozen_producer(path, classes=20, accept_quantile=0.85,
                                     metric="euclidean_raw_mu")
            with self.assertRaises(ClusterError):
                load_frozen_producer(path, classes=3, accept_quantile=0.9,
                                     metric="euclidean_raw_mu")
            with self.assertRaises(ClusterError):
                load_frozen_producer(path, classes=3, accept_quantile=0.85,
                                     metric="cosine_l2_normalized_mu")
