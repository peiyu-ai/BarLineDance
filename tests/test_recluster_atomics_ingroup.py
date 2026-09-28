import json
import pathlib
import tempfile
import unittest

import numpy as np

from tools.recluster_atomics_ingroup import (
    ClusterError,
    TRANSITION,
    clip_span,
    compact_label_ids,
    drop_rare_subprototypes,
    genre_of,
    train_performance_of_recording,
    load_captions,
    load_subprototypes,
    medoid_index,
    nearest_group,
    place_by_llm,
    scale_block,
    segments_of,
    sub_prototype_count,
)


class SegmentTests(unittest.TestCase):
    def test_runs_are_half_open_intervals_carrying_their_label(self):
        self.assertEqual(segments_of(np.array([5, 5, 0, 0, 7])),
                         [(0, 2, 5), (2, 4, 0), (4, 5, 7)])

    def test_empty_input_has_no_segments(self):
        self.assertEqual(segments_of(np.array([], dtype=np.int64)), [])


class GenreTests(unittest.TestCase):
    def test_aist_names_carry_a_genre(self):
        self.assertEqual(genre_of("aistpp/gBR_sBM_cAll_d04_mBR0_ch01"), "gBR")

    def test_wild_names_do_not_and_that_is_reported_not_invented(self):
        self.assertIsNone(genre_of("tiktok:6796814277069114624:clip000"))


class SubPrototypeCountTests(unittest.TestCase):
    def test_count_follows_group_size_at_the_target(self):
        # The paper reports an average of 7.3 sub-prototypes at 31.8 samples
        # each; a group near 100x that ratio should land near that average.
        self.assertEqual(sub_prototype_count(232, 32), 7)
        self.assertEqual(sub_prototype_count(243, 32), 8)

    def test_a_group_never_yields_zero_sub_prototypes(self):
        self.assertEqual(sub_prototype_count(1, 32), 1)
        self.assertEqual(sub_prototype_count(0, 32), 1)

    def test_a_fixed_k_is_not_imposed(self):
        # Different group sizes must produce different counts, or the stage is
        # imposing uniformity the paper's reported spread says is not there.
        self.assertNotEqual(sub_prototype_count(64, 32), sub_prototype_count(320, 32))


class ExpandedLabelIdTests(unittest.TestCase):
    def test_ids_stay_contiguous_and_leave_the_transition_token_reserved(self):
        # Expanded id: (prototype - 1) * width + sub + 1.
        width = 8
        ids = sorted((p - 1) * width + s + 1
                     for p in range(1, 4) for s in range(width))
        self.assertEqual(ids[0], 1)                      # 0 stays transition
        self.assertEqual(ids, list(range(1, 3 * width + 1)))   # no gaps

    def test_two_different_prototypes_never_collide(self):
        width = 8
        first = {(1 - 1) * width + s + 1 for s in range(width)}
        second = {(2 - 1) * width + s + 1 for s in range(width)}
        self.assertEqual(len(first & second), 0)


if __name__ == "__main__":
    unittest.main()


class CompactIdTests(unittest.TestCase):
    def test_padded_layout_wastes_ids_which_is_why_compaction_exists(self):
        # (p-1)*width + s + 1 reserves `width` slots per prototype, but groups
        # differ in size, so most slots stay empty.  On the wild corpus that was
        # 684 real classes scattered across 1300 ids -- a D3PM would spend a
        # third of its vocabulary on tokens that never occur.
        import numpy as np

        width = 13
        prototypes = np.array([1, 1, 2, 3])
        subs = np.array([0, 1, 0, 0])
        raw = (prototypes - 1) * width + subs + 1
        self.assertEqual(sorted(raw.tolist()), [1, 2, 14, 27])   # sparse
        used = np.unique(raw)
        compact = {int(o): i + 1 for i, o in enumerate(used)}
        remapped = [compact[int(v)] for v in raw]
        self.assertEqual(sorted(remapped), [1, 2, 3, 4])          # contiguous
        self.assertNotIn(0, remapped)                             # 0 stays transition

    def test_compaction_is_order_preserving_and_injective(self):
        import numpy as np

        raw = np.array([27, 1, 14, 1, 27])
        used = np.unique(raw)
        compact = {int(o): i + 1 for i, o in enumerate(used)}
        remapped = np.array([compact[int(v)] for v in raw])
        # Equal raw ids stay equal; distinct stay distinct; order is preserved.
        self.assertEqual(remapped[0], remapped[4])
        self.assertEqual(len(set(remapped.tolist())), len(set(raw.tolist())))
        self.assertTrue((np.argsort(raw) == np.argsort(remapped)).all())


class UngroupedCellTests(unittest.TestCase):
    """``--min-cell-size``: a cell can yield no sub-prototype at all.

    The default keeps this repo's floor of one class per non-empty cell, which
    the paper's own procedure does not have -- it stops when the *ungrouped*
    segments fall below a threshold.  On AIST++ that floor supplies 700 of 849
    classes, so the bug that matters here is a dropped group still minting one.
    """

    WIDTH = 4

    def test_an_ungrouped_segment_takes_transition_and_mints_no_class(self):
        prototypes = np.array([1, 1, 2, 2])
        subs = np.array([0, 0, 0, 0])
        dropped = np.array([False, False, True, True])
        ids, used = compact_label_ids(prototypes, subs, dropped, self.WIDTH)
        self.assertEqual(ids.tolist(), [1, 1, TRANSITION, TRANSITION])
        # Prototype 2's raw id would have been 5; it must not appear.
        self.assertEqual(used.tolist(), [1])

    def test_a_dropped_group_does_not_leave_a_gap_in_the_id_space(self):
        # The ids downstream index a D3PM vocabulary, so a hole is a token that
        # can be sampled and never observed.
        prototypes = np.array([1, 2, 3])
        subs = np.array([0, 0, 0])
        dropped = np.array([False, True, False])
        ids, used = compact_label_ids(prototypes, subs, dropped, self.WIDTH)
        self.assertEqual(sorted(v for v in ids.tolist() if v != TRANSITION), [1, 2])
        self.assertEqual(len(used), 2)

    def test_sizes_read_off_the_ids_count_only_formed_groups(self):
        # Fig. 4c is a histogram over exactly this bincount, so an ungrouped
        # segment counted as a class of its own would put mass in the <20
        # bucket that no clustering produced.
        prototypes = np.array([1, 1, 1, 2, 2])
        subs = np.array([0, 0, 1, 0, 0])
        dropped = np.array([False, False, True, False, False])
        ids, _ = compact_label_ids(prototypes, subs, dropped, self.WIDTH)
        sizes = np.bincount(ids)[1:]
        self.assertEqual(sizes.tolist(), [2, 2])

    def test_dropping_nothing_reproduces_the_previous_behaviour(self):
        # The default must be a no-op: every run published before this option
        # existed has to stay reproducible from the same inputs.
        prototypes = np.array([1, 1, 2, 3])
        subs = np.array([0, 1, 0, 0])
        none_dropped = np.zeros(4, dtype=bool)
        ids, used = compact_label_ids(prototypes, subs, none_dropped, self.WIDTH)
        raw = (prototypes - 1) * self.WIDTH + subs + 1
        expected = {int(o): i + 1 for i, o in enumerate(np.unique(raw))}
        self.assertEqual(ids.tolist(), [expected[int(v)] for v in raw])
        self.assertEqual(len(used), 4)

    def test_dropping_everything_yields_no_class_rather_than_one(self):
        # build() turns this into a ClusterError; the failure has to be
        # reachable, because a threshold above the largest cell is a plausible
        # typo and an empty vocabulary that publishes is worse than a crash.
        prototypes = np.array([1, 2])
        subs = np.array([0, 0])
        ids, used = compact_label_ids(prototypes, subs, np.ones(2, dtype=bool), self.WIDTH)
        self.assertEqual(used.tolist(), [])
        self.assertEqual(set(ids.tolist()), {TRANSITION})


class ClippedSpanTests(unittest.TestCase):
    """A cached span that runs past the label array is trimmed, never dropped.

    Dropping it was silent and expensive: the publisher used to seed from the M2
    array, so a segment this stage skipped kept an M2 prototype id -- 1..100 --
    inside an M3 vocabulary of 599, in range for every bound downstream checks.
    Measured 2026-08-13 on aist_songsplit: 958 spans skipped, 716 of them
    M2-accepted, 20,445 frames shipped carrying the wrong label space.
    """

    def test_a_span_past_the_end_is_trimmed_to_the_array(self):
        self.assertEqual(clip_span(336, 359, 345), (336, 345))

    def test_a_span_inside_the_array_is_untouched(self):
        self.assertEqual(clip_span(10, 20, 345), (10, 20))

    def test_a_span_starting_past_the_end_yields_nothing(self):
        self.assertIsNone(clip_span(400, 420, 345))

    def test_a_span_ending_exactly_at_the_end_survives(self):
        self.assertEqual(clip_span(340, 345, 345), (340, 345))


class RecurrenceFilterTests(unittest.TestCase):
    """A sub-prototype seen in one performance is not a recurring movement.

    The rule has to be able to keep and to drop on constructed corpora, because
    the version of it that only ever keeps is exactly the gate this repo has on
    record as worse than no gate.
    """

    def setUp(self):
        self.prototypes = np.array([1, 1, 1, 1])
        self.sub_labels = np.array([0, 0, 1, 1])

    def test_a_class_spanning_two_performances_survives(self):
        ungrouped = np.zeros(4, dtype=bool)
        classes, segments = drop_rare_subprototypes(
            self.prototypes, self.sub_labels, ungrouped,
            ["p/a", "p/b", "p/a", "p/b"], 2)
        self.assertEqual((classes, segments), (0, 0))
        self.assertFalse(ungrouped.any())

    def test_a_class_confined_to_one_performance_becomes_transition(self):
        ungrouped = np.zeros(4, dtype=bool)
        classes, segments = drop_rare_subprototypes(
            self.prototypes, self.sub_labels, ungrouped,
            ["p/a", "p/a", "p/b", "p/c"], 2)
        self.assertEqual((classes, segments), (1, 2))
        self.assertEqual(ungrouped.tolist(), [True, True, False, False])

    def test_held_out_performances_do_not_count_towards_recurrence(self):
        # Two distinct performances, but only one of them is in train, so the
        # class has a single train performance and the source-safe retrieval
        # audit would not credit it either.
        ungrouped = np.zeros(4, dtype=bool)
        classes, _ = drop_rare_subprototypes(
            self.prototypes, self.sub_labels, ungrouped,
            ["p/a", None, "p/b", "p/c"], 2)
        self.assertEqual(classes, 1)
        self.assertEqual(ungrouped.tolist(), [True, True, False, False])

    def test_the_default_of_one_group_drops_nothing(self):
        ungrouped = np.zeros(4, dtype=bool)
        classes, segments = drop_rare_subprototypes(
            self.prototypes, self.sub_labels, ungrouped,
            ["p/a", "p/a", "p/a", "p/a"], 1)
        self.assertEqual((classes, segments), (0, 0))

    def test_an_already_ungrouped_segment_is_not_counted_twice(self):
        ungrouped = np.array([True, False, False, False])
        classes, segments = drop_rare_subprototypes(
            self.prototypes, self.sub_labels, ungrouped,
            ["p/a", "p/a", "p/b", "p/c"], 2)
        self.assertEqual(segments, 1)
        self.assertEqual(ungrouped.tolist(), [True, True, False, False])


class TrainPerformanceLookupTests(unittest.TestCase):
    def _manifest(self, directory, rows):
        bundle = pathlib.Path(directory)
        (bundle / "sources.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return bundle

    def test_held_out_recordings_map_to_none_rather_than_their_group(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._manifest(directory, [
                {"recording_id": "a", "retrieval_group_id": "p/a", "split": "train"},
                {"recording_id": "b", "retrieval_group_id": "p/b", "split": "test"},
            ])
            mapping = train_performance_of_recording(bundle)
            self.assertEqual(mapping, {"a": "p/a", "b": None})

    def test_a_missing_manifest_raises_instead_of_defaulting_to_the_recording(self):
        # Falling back to the recording id would make every sub-prototype look
        # like it spans several performances, so the rule would pass anywhere.
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ClusterError):
                train_performance_of_recording(pathlib.Path(directory))

    def test_a_row_without_a_group_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._manifest(directory, [
                {"recording_id": "a", "retrieval_group_id": "", "split": "train"}])
            with self.assertRaises(ClusterError):
                train_performance_of_recording(bundle)


class CaptionLoadingTests(unittest.TestCase):
    def _write(self, tmp, rows):
        path = tmp / "captions.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_captions_key_on_the_exact_segment(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = pathlib.Path(name)
            path = self._write(tmp, [
                {"recording_id": "a", "start": 0, "end": 30, "caption": "x",
                 "model": "qwen"},
                {"recording_id": "a", "start": 30, "end": 60, "caption": "y",
                 "model": "qwen"},
            ])
            captions, models = load_captions(path)
            self.assertEqual(captions[("a", 0, 30)], "x")
            self.assertEqual(captions[("a", 30, 60)], "y")
            self.assertEqual(models, ["qwen"])

    def test_duplicate_rows_keep_the_first_not_the_last(self):
        """Resumed runs append; the mapping must not depend on file order."""
        with tempfile.TemporaryDirectory() as name:
            tmp = pathlib.Path(name)
            path = self._write(tmp, [
                {"recording_id": "a", "start": 0, "end": 30, "caption": "first"},
                {"recording_id": "a", "start": 0, "end": 30, "caption": "second"},
            ])
            captions, _ = load_captions(path)
            self.assertEqual(captions[("a", 0, 30)], "first")

    def test_two_captioners_are_both_reported_not_collapsed(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = pathlib.Path(name)
            path = self._write(tmp, [
                {"recording_id": "a", "start": 0, "end": 30, "caption": "x",
                 "model": "qwen3"},
                {"recording_id": "b", "start": 0, "end": 30, "caption": "y",
                 "model": "qwen25"},
            ])
            _, models = load_captions(path)
            self.assertEqual(models, ["qwen25", "qwen3"])

    def test_blank_and_empty_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = pathlib.Path(name)
            path = tmp / "c.jsonl"
            path.write_text('\n{"recording_id": "a", "start": 0, "end": 3, '
                            '"caption": "x"}\n\n', encoding="utf-8")
            captions, _ = load_captions(path)
            self.assertEqual(len(captions), 1)


class FeatureBlockTests(unittest.TestCase):
    def test_blocks_of_different_width_carry_equal_total_variance(self):
        """Otherwise the wider block wins on column count, not on signal."""
        rng = np.random.default_rng(0)
        wide = scale_block(rng.normal(size=(200, 512)))
        narrow = scale_block(rng.normal(size=(200, 20)))
        self.assertAlmostEqual(float((wide ** 2).sum(axis=1).mean()),
                               float((narrow ** 2).sum(axis=1).mean()), delta=0.15)

    def test_constant_column_does_not_produce_nan(self):
        block = np.ones((10, 3))
        self.assertTrue(np.isfinite(scale_block(block)).all())


class MedoidTests(unittest.TestCase):
    def test_medoid_is_the_member_nearest_the_centre(self):
        block = np.array([[0.0], [10.0], [5.0]])
        self.assertEqual(medoid_index(block), 2)

    def test_single_member_group_is_its_own_medoid(self):
        self.assertEqual(medoid_index(np.array([[3.0, 4.0]])), 0)


class SubPrototypeLoadingTests(unittest.TestCase):
    def _write(self, tmp, groups):
        path = tmp / "subs.json"
        path.write_text(json.dumps({"model": "qwen", "groups_detail": groups}),
                        encoding="utf-8")
        return path

    def test_placement_maps_caption_to_its_subprototype_index(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = pathlib.Path(name)
            path = self._write(tmp, [{
                "prototype": 3, "genre": "?",
                "subprototypes": [{"tag": "spins", "captions": ["a", "b"]},
                                  {"tag": "steps", "captions": ["c"]}],
            }])
            placement, tags, header = load_subprototypes(path)
            self.assertEqual(placement[(3, "?", "a")], 0)
            self.assertEqual(placement[(3, "?", "b")], 0)
            self.assertEqual(placement[(3, "?", "c")], 1)
            self.assertEqual(tags[(3, "?", 1)], "steps")
            self.assertEqual(header["model"], "qwen")

    def test_the_same_caption_in_two_prototypes_stays_separate(self):
        """Captions repeat across prototypes; the key must carry the group."""
        with tempfile.TemporaryDirectory() as name:
            tmp = pathlib.Path(name)
            path = self._write(tmp, [
                {"prototype": 1, "genre": "?",
                 "subprototypes": [{"tag": "x", "captions": ["shared"]},
                                   {"tag": "y", "captions": ["other"]}]},
                {"prototype": 2, "genre": "?",
                 "subprototypes": [{"tag": "z", "captions": ["first"]},
                                   {"tag": "w", "captions": ["shared"]}]},
            ])
            placement, _, _ = load_subprototypes(path)
            self.assertEqual(placement[(1, "?", "shared")], 0)
            self.assertEqual(placement[(2, "?", "shared")], 1)

    def test_a_grouping_with_no_members_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            tmp = pathlib.Path(name)
            path = self._write(tmp, [{"prototype": 1, "genre": "?", "subprototypes": []}])
            with self.assertRaises(Exception):
                load_subprototypes(path)


class NearestGroupTests(unittest.TestCase):
    def test_unplaced_members_join_the_nearest_subprototype(self):
        features = np.array([[0.0], [0.1], [10.0], [10.1], [9.9]])
        members = np.arange(5)
        assignment = np.array([0, 0, 1, 1, -1])
        placed = nearest_group(features, members, assignment.copy(), 2)
        self.assertEqual(placed[4], 1)

    def test_nothing_placed_at_all_collapses_to_one_subprototype(self):
        """An LLM that matched no caption must not scatter the group."""
        features = np.array([[0.0], [5.0]])
        placed = nearest_group(features, np.arange(2), np.array([-1, -1]), 1)
        self.assertTrue((placed == 0).all())

    def test_a_fully_placed_group_is_returned_untouched(self):
        features = np.array([[0.0], [1.0]])
        assignment = np.array([1, 0])
        placed = nearest_group(features, np.arange(2), assignment.copy(), 2)
        np.testing.assert_array_equal(placed, assignment)

    def test_members_are_global_indices_not_positions(self):
        """features is indexed globally while assignment is per member."""
        features = np.array([[99.0], [0.0], [0.1], [10.0]])
        members = np.array([1, 2, 3])
        placed = nearest_group(features, members, np.array([0, -1, 1]), 2)
        self.assertEqual(placed[1], 0)


class LLMPlacementTests(unittest.TestCase):
    def test_two_genres_sharing_an_index_do_not_collapse_into_one_class(self):
        """The summarizer numbers sub-prototypes per (prototype, genre), so
        index 0 exists once per genre.  Publishing them as one class would
        merge two different moves."""
        group = np.array([0, 1, 2, 3])
        genres = ["gBR", "gBR", "gJZ", "gJZ"]
        captions = ["a", "b", "c", "d"]
        placement = {(7, "gBR", "a"): 0, (7, "gBR", "b"): 1,
                     (7, "gJZ", "c"): 0, (7, "gJZ", "d"): 1}
        assignment, slots = place_by_llm(group, 7, genres, captions, placement)
        self.assertEqual(len(set(assignment.tolist())), 4)
        self.assertEqual(len(slots), 4)

    def test_members_of_one_subprototype_share_a_slot(self):
        group = np.array([0, 1, 2])
        placement = {(1, "?", "x"): 0, (1, "?", "y"): 0, (1, "?", "z"): 1}
        assignment, slots = place_by_llm(group, 1, ["?"] * 3, ["x", "y", "z"], placement)
        self.assertEqual(assignment[0], assignment[1])
        self.assertNotEqual(assignment[0], assignment[2])
        self.assertEqual(len(slots), 2)

    def test_an_uncaptioned_segment_is_left_for_the_keyframe_fallback(self):
        group = np.array([0, 1])
        assignment, slots = place_by_llm(
            group, 1, ["?", "?"], ["x", ""], {(1, "?", "x"): 0})
        self.assertEqual(assignment[0], 0)
        self.assertEqual(assignment[1], -1)

    def test_a_genreless_grouping_still_matches_genred_segments(self):
        """The summarizer ran without a genre map; the segments still carry one."""
        group = np.array([0])
        assignment, slots = place_by_llm(
            group, 3, ["gLO"], ["x"], {(3, "?", "x"): 2})
        self.assertEqual(assignment[0], 0)
        self.assertEqual(slots, {("?", 2): 0})

    def test_slots_are_numbered_contiguously_from_zero(self):
        group = np.array([0, 1])
        placement = {(1, "?", "x"): 5, (1, "?", "y"): 9}
        assignment, slots = place_by_llm(group, 1, ["?"] * 2, ["x", "y"], placement)
        self.assertEqual(sorted(slots.values()), [0, 1])
        self.assertEqual(sorted(assignment.tolist()), [0, 1])


def test_pre_split_key_comes_from_metadata_when_supplied(tmp_path):
    """The paper reads genre from AIST's sequence id; TikTok has no such field.

    A VLM asked to supply one answered at chance (0.083 against 0.10 on ten
    balanced classes), so the key comes from metadata instead -- the
    choreographer account, which every upload carries.  Both recording-id shapes
    used in this repo must resolve, or the pre-split silently collapses to one
    group with nothing reporting it.
    """
    import json as jsonlib

    from tools.recluster_atomics_ingroup import genre_of, load_group_keys

    path = tmp_path / "groups.json"
    path.write_text(jsonlib.dumps({"7203621409615088908": "studio-A"}), encoding="utf-8")
    groups = load_group_keys(path)

    assert genre_of("wild_v4:7203621409615088908:clip000", groups) == "studio-A"
    assert genre_of("7203621409615088908__clip000", groups) == "studio-A"
    # AIST keeps working: its own field is the fallback, not overridden.
    assert genre_of("gBR_sBM_c01_d04_mBR0_ch01", groups) == "gBR"
    # An upload with no metadata is None, not a fabricated group.
    assert genre_of("9999999999__clip000", groups) is None


def test_no_group_keys_means_the_aist_field_only():
    from tools.recluster_atomics_ingroup import genre_of, load_group_keys

    assert load_group_keys(None) == {}
    assert genre_of("gJS_sFM_c01_d01_mJS0_ch01") == "gJS"
    assert genre_of("7203621409615088908__clip000") is None


def test_merging_folds_small_subprototypes_into_their_nearest_sibling():
    """The LLM's grouping is far finer than the paper's, and merging contracts it.

    wild_v4: 4,159 classes at 41.6 per prototype against the paper's 730 at 7.3,
    with 69.8% of classes under 20 segments where Fig. 4c has 14.9%.  A class
    with two members is a caption phrased differently, not a discovered movement.
    """
    import numpy as np

    from tools.recluster_atomics_ingroup import merge_small_subprototypes

    # Three slots: two large and far apart, one tiny sitting next to the first.
    features = np.array([[0.0]] * 6 + [[10.0]] * 6 + [[0.5]] * 2, dtype=np.float64)
    members = np.arange(len(features))
    assignment = np.array([0] * 6 + [1] * 6 + [2] * 2)
    merged, count, runs = merge_small_subprototypes(features, members, assignment, 3, 5)
    assert runs == 1 and count == 2
    # The tiny slot joined slot 0, which is the nearer centroid -- and no segment
    # was dropped, which is the difference from --min-cell-size.
    assert len(merged) == len(assignment)
    assert set(merged[-2:].tolist()) == set(merged[:6].tolist())
    assert sorted(set(merged.tolist())) == [0, 1]


def test_merging_repeats_until_every_survivor_clears_the_floor():
    """One pass would leave two small classes merged into each other and still small."""
    import numpy as np

    from tools.recluster_atomics_ingroup import merge_small_subprototypes

    features = np.array([[0.0]] * 8 + [[1.0]] * 2 + [[1.1]] * 2 + [[1.2]] * 2,
                        dtype=np.float64)
    members = np.arange(len(features))
    assignment = np.array([0] * 8 + [1] * 2 + [2] * 2 + [3] * 2)
    merged, count, runs = merge_small_subprototypes(features, members, assignment, 4, 6)
    sizes = np.bincount(merged)
    # The invariant is what matters: no survivor is left under the floor.
    assert (sizes[sizes > 0] >= 6).all()
    # Two merges, and the count is checked because it pins the *order*: the three
    # small slots fold into each other (2+2 then +2 = 6) rather than each being
    # dragged to the large distant one.  A single pass would have stopped after
    # the first merge with a 4-member class still below the floor.
    assert runs == 2 and count == 2


def test_merging_never_empties_the_prototype():
    import numpy as np

    from tools.recluster_atomics_ingroup import merge_small_subprototypes

    features = np.array([[0.0]] * 2 + [[5.0]] * 2, dtype=np.float64)
    members = np.arange(4)
    assignment = np.array([0, 0, 1, 1])
    merged, count, _ = merge_small_subprototypes(features, members, assignment, 2, 100)
    # A floor above the prototype's whole population leaves exactly one class,
    # never zero: every segment still needs a label.
    assert count == 1
    assert sorted(set(merged.tolist())) == [0]


def test_the_floor_is_off_by_default_so_published_vocabularies_reproduce():
    from tools.recluster_atomics_ingroup import build_parser

    assert build_parser().parse_args(
        ["--labels", "l", "--bundle", "b", "--output-dir", "o"]
    ).min_subprototype_size == 0


def test_merging_relabels_to_a_dense_id_space():
    """A gap becomes an empty class, and an empty class is a softmax row that
    never receives gradient."""
    import numpy as np

    from tools.recluster_atomics_ingroup import merge_small_subprototypes

    features = np.array([[0.0]] * 5 + [[9.0]] + [[0.1]] * 5, dtype=np.float64)
    members = np.arange(len(features))
    assignment = np.array([0] * 5 + [1] + [2] * 5)
    merged, count, _ = merge_small_subprototypes(features, members, assignment, 3, 3)
    assert sorted(set(merged.tolist())) == list(range(count))
