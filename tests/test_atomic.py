import json
import os
import tempfile
import unittest

import numpy as np
import torch

from dataset.atomic import (
    AtomicMotionLibrary,
    labels_to_segments,
    majority_vote,
    merge_short_segments,
    plan_boundaries,
    refine_plan,
    source_id_from_name,
)
from dataset.atomic_dataset import AtomicSequenceDataset, collate_atomic_sequences
from model.atomic_completion import AtomicCompletionDecoder, AtomicCompletionDiffusion
from model.atomic_planner import AtomicPlannerTransformer, UniformD3PM
from train_atomic import completion_conditions


class AtomicPlanTests(unittest.TestCase):
    def test_segments_and_refinement(self):
        labels = torch.tensor([1, 1, 2, 1, 1, 0, 0, 3, 0, 0])
        voted = majority_vote(labels, window_size=3)
        self.assertEqual(voted.tolist()[:5], [1, 1, 1, 1, 1])
        refined = refine_plan(labels, vote_window=3, min_length=3)
        self.assertTrue(all(segment.length >= 3 for segment in labels_to_segments(refined)))

    def test_duration_nearest_retrieval_and_draft(self):
        library = AtomicMotionLibrary(
            {1: [torch.ones(2, 3), torch.full((5, 3), 5.0)], 2: [torch.full((3, 3), 2.0)]}
        )
        retrieved = library.retrieve(1, 4)
        self.assertEqual(tuple(retrieved.shape), (4, 3))
        self.assertTrue(torch.allclose(retrieved, torch.full((4, 3), 5.0)))
        draft, mask = library.build_draft(torch.tensor([1, 1, 0, 2, 2, 2]), feature_dim=3)
        self.assertEqual(mask.squeeze(-1).tolist(), [1, 1, 0, 1, 1, 1])
        self.assertTrue(torch.equal(draft[2], torch.zeros(3)))
        self.assertEqual(
            plan_boundaries(torch.tensor([1, 1, 0, 2, 2])).tolist(),
            [False, False, True, True, False],
        )

    def test_source_safe_retrieval_excludes_overlapping_video_windows(self):
        motions = [
            torch.full((4, 2), 1.0),
            torch.full((4, 2), 2.0),
        ]
        labels = [torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 1, 1])]
        library = AtomicMotionLibrary.from_sequences(
            motions,
            labels,
            names=["source_a_slice0", "source_b_slice0"],
        )
        retrieved = library.retrieve(1, 4, exclude_source_ids={"source_a"})
        self.assertTrue(torch.allclose(retrieved, torch.full((4, 2), 2.0)))
        draft, mask = library.build_draft(
            labels[0],
            2,
            exclude_source_ids={source_id_from_name("source_a_slice1")},
        )
        self.assertTrue(torch.allclose(draft, torch.full((4, 2), 2.0)))
        self.assertTrue(torch.equal(mask, torch.ones(4, 1)))

    def test_state_roundtrip_preserves_source_provenance(self):
        library = AtomicMotionLibrary.from_sequences(
            [torch.full((3, 2), 1.0), torch.full((3, 2), 2.0)],
            [torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1])],
            names=["source_a_slice0", "source_b_slice0"],
        )
        restored = AtomicMotionLibrary.from_state_dict(library.state_dict())
        prototype = restored.motions[1][0]
        self.assertEqual(prototype.source_id, "source_a")
        self.assertEqual(prototype.sample_name, "source_a_slice0")
        self.assertEqual(prototype.start, 0)
        self.assertEqual(prototype.end, 3)
        retrieved = restored.retrieve(1, 3, exclude_source_ids={"source_a"})
        self.assertTrue(torch.allclose(retrieved, torch.full((3, 2), 2.0)))

    def test_unknown_provenance_is_rejected_when_source_is_excluded(self):
        # Legacy tensor-only state remains usable for no-exclusion callers,
        # but its lost provenance cannot pass a source-safe query.
        library = AtomicMotionLibrary.from_state_dict(
            {1: [torch.full((3, 2), 7.0)]}
        )
        self.assertTrue(torch.allclose(library.retrieve(1, 3), torch.full((3, 2), 7.0)))
        with self.assertRaises(KeyError):
            library.retrieve(1, 3, exclude_source_ids="query_source")

    def test_allow_missing_leaves_no_safe_prototype_as_zero_mask(self):
        labels = torch.tensor([1, 1, 1])
        library = AtomicMotionLibrary.from_sequences(
            [torch.full((3, 2), 7.0)],
            [labels],
            names=["query_source_slice0"],
        )
        draft, mask = library.build_draft(
            labels,
            2,
            exclude_source_ids={"query_source"},
            allow_missing=True,
        )
        self.assertTrue(torch.equal(draft, torch.zeros(3, 2)))
        self.assertTrue(torch.equal(mask, torch.zeros(3, 1)))

    def test_safe_draft_coverage_counts_atomic_frames_not_transitions(self):
        library = AtomicMotionLibrary.from_sequences(
            [torch.ones(4, 2), torch.full((4, 2), 2.0)],
            [torch.tensor([1, 1, 0, 0]), torch.tensor([1, 1, 0, 0])],
            names=["source_a_slice0", "source_b_slice0"],
            retrieval_group_ids=["performance/a", "performance/b"],
        )
        draft, mask, boundaries, coverage = completion_conditions(
            torch.tensor([[1, 1, 0, 0]]),
            ["performance/a"],
            library,
            motion_dim=2,
            noise_ratio=0.25,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(draft.shape), (1, 4, 2))
        self.assertEqual(tuple(mask.shape), (1, 4, 1))
        self.assertEqual(tuple(boundaries.shape), (1, 4))
        self.assertEqual(coverage, 1.0)

    def test_retrieval_group_excludes_all_camera_views_of_one_performance(self):
        """Different recording names must not permit same-performance retrieval."""
        library = AtomicMotionLibrary.from_sequences(
            [
                torch.full((3, 2), 1.0),
                torch.full((3, 2), 2.0),
                torch.full((3, 2), 3.0),
            ],
            [torch.ones(3, dtype=torch.long)] * 3,
            names=[
                "show17_cam_front_slice0",
                "show17_cam_side_slice0",
                "show18_cam_front_slice0",
            ],
            retrieval_group_ids=["performance/show17", "performance/show17", "performance/show18"],
        )
        # The legacy source-name exclusion would leave show17's side camera
        # eligible.  The explicit group exclusion leaves only show18.
        values = library.retrieve(
            1,
            3,
            exclude_retrieval_group_ids={"performance/show17"},
        )
        self.assertTrue(torch.allclose(values, torch.full((3, 2), 3.0)))
        draft, mask = library.build_draft(
            torch.ones(3, dtype=torch.long),
            2,
            exclude_retrieval_group_ids={"performance/show17"},
        )
        self.assertTrue(torch.allclose(draft, torch.full((3, 2), 3.0)))
        self.assertTrue(torch.equal(mask, torch.ones(3, 1)))

    def test_vectorised_retrieval_matches_the_python_scan(self):
        """The fast index must pick the same prototype, not an equally good one.

        ``retrieve`` no longer walks a label's candidates in Python; it argmins
        over precomputed length/provenance vectors.  Both implementations resolve
        a duration tie by taking the *first* candidate in insertion order, so the
        risk is not that the fast path is worse -- it is that it silently returns
        a different-but-plausible prototype and quietly changes every draft the
        completion stage trains on.

        ``_reference_retrieve`` below is the pre-index implementation, kept
        verbatim so this test can fail.  The fixture is built to make ties and
        exclusions common rather than rare: durations are drawn from a small set
        so several candidates share a distance, and half the prototypes carry
        unknown provenance, which an exclusion must reject.
        """

        def _reference_retrieve(library, label, target_length, excluded_groups=(), excluded_sources=()):
            candidates = library.motions.get(int(label), ())
            excluded_groups = set(excluded_groups)
            excluded_sources = set(excluded_sources)
            if excluded_groups:
                candidates = tuple(
                    candidate
                    for candidate in candidates
                    if isinstance(candidate.retrieval_group_id, str)
                    and candidate.retrieval_group_id
                    and candidate.retrieval_group_id not in excluded_groups
                )
            elif excluded_sources:
                candidates = tuple(
                    candidate
                    for candidate in candidates
                    if isinstance(candidate.source_id, str)
                    and candidate.source_id
                    and candidate.source_id not in excluded_sources
                )
            if not candidates:
                raise KeyError(label)
            return min(
                candidates,
                key=lambda prototype: abs(prototype.motion.shape[0] - target_length),
            ).motion

        generator = torch.Generator().manual_seed(20260816)
        durations = [2, 3, 5, 8]
        motions, labels, names, groups = [], [], [], []
        for index in range(60):
            duration = durations[index % len(durations)]
            frames = duration * 2
            motions.append(torch.rand(frames, 4, generator=generator))
            plan = torch.full((frames,), (index % 5) + 1, dtype=torch.long)
            plan[duration:] = 0  # one atomic segment of `duration`, then transition
            labels.append(plan)
            names.append("recording{}_slice0".format(index))
            # Half the corpus has unknown provenance: `None` here, which an
            # exclusion must reject rather than treat as safe.
            groups.append("performance/{}".format(index % 7) if index % 2 else None)
        library = AtomicMotionLibrary.from_sequences(motions, labels, names=names)
        grouped = AtomicMotionLibrary.from_sequences(
            motions,
            labels,
            names=names,
            retrieval_group_ids=[group or "performance/unknown{}".format(index)
                                 for index, group in enumerate(groups)],
        )

        compared = 0
        refused = 0
        for label in range(1, 6):
            for target in range(1, 12):
                expected = _reference_retrieve(library, label, target)
                self.assertTrue(
                    torch.equal(
                        library.retrieve(label, target),
                        AtomicMotionLibrary._resample(expected, target),
                    ),
                    "unfiltered retrieval diverged at label={} target={}".format(label, target),
                )
                compared += 1
                for excluded in ("performance/1", "performance/3"):
                    try:
                        expected = _reference_retrieve(
                            grouped, label, target, excluded_groups={excluded}
                        )
                    except KeyError:
                        with self.assertRaises(KeyError):
                            grouped.retrieve(label, target, exclude_retrieval_group_ids={excluded})
                        refused += 1
                        continue
                    self.assertTrue(
                        torch.equal(
                            grouped.retrieve(label, target, exclude_retrieval_group_ids={excluded}),
                            AtomicMotionLibrary._resample(expected, target),
                        ),
                        "group-excluded retrieval diverged at label={} target={} excluded={}".format(
                            label, target, excluded
                        ),
                    )
                    compared += 1
                expected = _reference_retrieve(
                    library, label, target, excluded_sources={"recording0", "recording1"}
                )
                self.assertTrue(
                    torch.equal(
                        library.retrieve(
                            label, target, exclude_source_ids={"recording0", "recording1"}
                        ),
                        AtomicMotionLibrary._resample(expected, target),
                    ),
                    "source-excluded retrieval diverged at label={} target={}".format(label, target),
                )
                compared += 1
        # A comparison that never ran is not a comparison that passed.
        self.assertGreater(compared, 150)

    def test_unknown_provenance_stays_ineligible_under_the_fast_index(self):
        library = AtomicMotionLibrary.from_sequences(
            [torch.ones(3, 2), torch.full((3, 2), 2.0)],
            [torch.ones(3, dtype=torch.long)] * 2,
            names=["a_slice0", "b_slice0"],
        )
        # Neither prototype declares a retrieval group, so no candidate can be
        # proved external and the request must fail closed rather than pick one.
        with self.assertRaises(KeyError):
            library.retrieve(1, 3, exclude_retrieval_group_ids={"performance/x"})

    def test_fill_draft_writes_the_same_values_as_build_draft(self):
        library = AtomicMotionLibrary.from_sequences(
            [torch.rand(6, 3), torch.rand(6, 3)],
            [torch.tensor([1, 1, 0, 2, 2, 2]), torch.tensor([1, 1, 1, 0, 2, 2])],
            names=["a_slice0", "b_slice0"],
            retrieval_group_ids=["performance/a", "performance/b"],
        )
        plan = torch.tensor([1, 1, 0, 2, 2, 2])
        expected_draft, expected_mask = library.build_draft(
            plan, 3, exclude_retrieval_group_ids={"performance/a"}, allow_missing=True
        )
        batch_draft = torch.zeros(2, 6, 3)
        batch_mask = torch.zeros(2, 6, 1)
        library.fill_draft(
            plan,
            batch_draft[1],
            batch_mask[1],
            exclude_retrieval_group_ids={"performance/a"},
            allow_missing=True,
        )
        self.assertTrue(torch.equal(batch_draft[1], expected_draft))
        self.assertTrue(torch.equal(batch_mask[1], expected_mask))
        # The untouched row must stay zero: a shared buffer that bleeds between
        # samples would condition one sequence on another's prototypes.
        self.assertTrue(torch.equal(batch_draft[0], torch.zeros(6, 3)))
        self.assertTrue(torch.equal(batch_mask[0], torch.zeros(6, 1)))


class D3PMTests(unittest.TestCase):
    def test_training_and_sampling_shapes(self):
        model = AtomicPlannerTransformer(
            num_atomic_classes=4,
            music_dim=6,
            latent_dim=16,
            num_layers=1,
            num_heads=4,
            ff_size=32,
            dropout=0.0,
            max_seq_len=12,
        )
        diffusion = UniformD3PM(model, num_steps=4)
        labels = torch.randint(0, 5, (2, 12))
        music = torch.randn(2, 12, 6)
        output = diffusion.training_step(labels, music, timesteps=torch.tensor([0, 3]))
        self.assertEqual(tuple(output.logits.shape), (2, 12, 5))
        self.assertTrue(torch.isfinite(output.loss))
        sampled = diffusion.sample(music, deterministic=True)
        self.assertEqual(tuple(sampled.shape), (2, 12))
        self.assertTrue(torch.all((sampled >= 0) & (sampled < 5)))


class CompletionTests(unittest.TestCase):
    def test_completion_loss_and_sampling_shapes(self):
        decoder = AtomicCompletionDecoder(
            motion_dim=7,
            seq_len=8,
            music_dim=6,
            latent_dim=16,
            ff_size=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        diffusion = AtomicCompletionDiffusion(decoder, num_steps=3)
        clean = torch.randn(2, 8, 7)
        music = torch.randn(2, 8, 6)
        draft = torch.randn(2, 8, 7)
        mask = torch.full((2, 8, 1), 0.25)
        boundaries = torch.zeros(2, 8, dtype=torch.bool)
        boundaries[:, 4] = True
        losses = diffusion.training_step(
            clean, music, draft, mask, boundaries, timesteps=torch.tensor([0, 2])
        )
        self.assertTrue(torch.isfinite(losses.total))
        sampled = diffusion.sample(music, draft, mask, guidance_weight=1.0)
        self.assertEqual(tuple(sampled.shape), tuple(clean.shape))


class AtomicDatasetTests(unittest.TestCase):
    def test_indexed_array_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            split = "{}/train".format(directory)
            os.makedirs(split)
            np.save("{}/motion.npy".format(split), np.zeros((2, 3, 7), dtype=np.float32))
            np.save("{}/music.npy".format(split), np.zeros((2, 3, 6), dtype=np.float32))
            np.save("{}/labels.npy".format(split), np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint8))
            with open("{}/names.json".format(split), "w") as handle:
                json.dump(["first", "second"], handle)
            with open("{}/retrieval_groups.json".format(split), "w") as handle:
                json.dump(["performance/first", "performance/second"], handle)

            dataset = AtomicSequenceDataset(directory, split="train")
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[1]["name"], "second")
            self.assertEqual(dataset[1]["retrieval_group_id"], "performance/second")
            batch = collate_atomic_sequences([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["motion"].shape), (2, 3, 7))
            self.assertEqual(batch["labels"].dtype, torch.long)
            self.assertEqual(batch["retrieval_group_ids"], ["performance/first", "performance/second"])


if __name__ == "__main__":
    unittest.main()


class TransitionPolicyTests(unittest.TestCase):
    """The minimum-duration merge's two readings of label 0.

    ``infer_atomic`` -- the path that feeds the completion stage -- used
    ``"merge"`` until 2026-08-23, while ``tools/postprocess_atomic_plan.py``
    had ``"protect"`` written in its docstring and its tests.  The rule that
    ran was never the rule that was documented, and these are the two cases
    that tell them apart.
    """

    def test_protect_keeps_a_short_transition_between_two_atomic_movements(self):
        labels = torch.tensor([7] * 30 + [0] * 4 + [9] * 30)
        protected = merge_short_segments(labels, 6, transition_policy="protect")
        self.assertEqual(int((protected == 0).sum()), 4)
        # The published behaviour deleted it, handing the gap to a neighbour.
        merged = merge_short_segments(labels, 6, transition_policy="merge")
        self.assertEqual(int((merged == 0).sum()), 0)

    def test_protect_will_not_feed_an_atomic_fragment_to_a_longer_transition(self):
        # Fragment 8 sits between a 10-frame atomic movement and a 40-frame
        # transition.  Ranking neighbours by length alone gives it away.
        labels = torch.tensor([7] * 10 + [8] * 4 + [0] * 40)
        protected = merge_short_segments(labels, 6, transition_policy="protect")
        self.assertEqual(int((protected == 0).sum()), 40)
        self.assertEqual(protected[12].item(), 7)
        merged = merge_short_segments(labels, 6, transition_policy="merge")
        self.assertEqual(int((merged == 0).sum()), 44)
        self.assertEqual(merged[12].item(), 0)

    def test_a_fragment_with_only_transition_neighbours_still_becomes_transition(self):
        labels = torch.tensor([0] * 10 + [5] * 2 + [0] * 10)
        protected = merge_short_segments(labels, 6, transition_policy="protect")
        self.assertTrue(bool((protected == 0).all()))

    def test_a_plan_shorter_than_the_minimum_is_left_alone(self):
        """One segment is not an over-segmentation error, and has no neighbour.

        Turning it into transition would delete the only movement a short clip
        names, and the fraction that reports an empty plan would read 1.0 with
        nothing to point at.
        """
        labels = torch.tensor([4, 4, 4])
        self.assertTrue(
            torch.equal(merge_short_segments(labels, 6, transition_policy="protect"),
                        labels))

    def test_an_unknown_policy_is_refused(self):
        with self.assertRaises(ValueError):
            merge_short_segments(torch.tensor([1, 1]), 6, transition_policy="keep")

    def test_refine_plan_carries_the_policy_through(self):
        labels = torch.tensor([7] * 30 + [0] * 4 + [9] * 30)
        self.assertEqual(
            int((refine_plan(labels, 1, 6, transition_policy="protect") == 0).sum()), 4)
        self.assertEqual(
            int((refine_plan(labels, 1, 6, transition_policy="merge") == 0).sum()), 0)

    def test_the_merge_order_is_its_own_axis_not_part_of_the_policy(self):
        """Binding the two together cost a reproduction on 2026-08-23.

        ``transition_policy="merge"`` restores the old *transition* rule; the
        inference path also resolved offenders left-to-right rather than
        shortest-first, and the two orders disagree on 1.44% of frames over
        4,000 synthetic transition-heavy plans -- enough to move a FID over 260
        generated clips, which is how the discrepancy was found.  So a
        pre-2026-08-23 inference artifact needs both.
        """
        # 2 transition, 2 atomic, 2 transition, 10 atomic; minimum 6 frames.
        # First-found resolves the leading 2-frame run and cascades leftward;
        # shortest-first reaches the 10-frame anchor and pulls everything to it.
        labels = torch.tensor([0] * 2 + [3] * 2 + [0] * 2 + [3] * 10)
        first = merge_short_segments(labels, 6, transition_policy="merge",
                                     merge_order="first")
        shortest = merge_short_segments(labels, 6, transition_policy="merge",
                                        merge_order="shortest")
        self.assertEqual(first.tolist(), [0] * 6 + [3] * 10)
        self.assertEqual(shortest.tolist(), [3] * 16)
        # Six frames of the plan hang on an argument that was not, until now,
        # an argument at all.
        self.assertEqual(int((first != shortest).sum()), 6)

    def test_an_unknown_merge_order_is_refused(self):
        with self.assertRaises(ValueError):
            merge_short_segments(torch.tensor([1, 1, 2, 2]), 6, merge_order="longest")
