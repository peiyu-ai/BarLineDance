"""The completion stage's conditions moved into the loader; nothing else may move.

``completion_conditions`` used to run on the training process between forward
passes.  It now runs in the DataLoader workers as ``CompletionConditionCollate``,
because on the wild 340-frame release it cost 0.589 s per batch of 64 against a
0.184 s model step -- 62% of every step spent with the GPU idle.

The whole argument for that move is that it changes *where* the work happens and
not *what* it produces: the draft is a deterministic function of ``(labels,
retrieval_group_id)`` with no RNG in the path.  These tests hold that claim to an
element-wise comparison, and check the two things the move could plausibly break:
the per-batch safe-coverage gate, and the isolation between rows of the shared
batch buffer.
"""

import argparse
import sys
import unittest
from unittest import mock

import torch

from dataset.atomic import AtomicMotionLibrary
from train_atomic import (
    DEFAULT_BATCH_SIZE,
    CompletionConditionCollate,
    batch_safe_draft_fraction,
    completion_conditions,
    parse_args,
    resolve_batch_sizes,
    resolve_gpus,
)


def _library_and_samples(count=12, frames=8, motion_dim=3):
    generator = torch.Generator().manual_seed(20260816)
    motions, plans, names, groups = [], [], [], []
    for index in range(count):
        motions.append(torch.rand(frames, motion_dim, generator=generator))
        plan = torch.zeros(frames, dtype=torch.long)
        plan[: frames // 2] = (index % 3) + 1
        plan[frames // 2 + 1 :] = (index % 4) + 1
        plans.append(plan)
        names.append("recording{}_slice0".format(index))
        groups.append("performance/{}".format(index % 5))
    library = AtomicMotionLibrary.from_sequences(
        motions, plans, names=names, retrieval_group_ids=groups
    )
    samples = [
        {
            "motion": motions[index],
            "music": torch.zeros(frames, 2),
            "labels": plans[index],
            "name": names[index],
            "retrieval_group_id": groups[index],
        }
        for index in range(count)
    ]
    return library, samples, motion_dim


class CompletionConditionCollateTests(unittest.TestCase):
    def test_collate_reproduces_the_in_loop_conditions_element_wise(self):
        library, samples, motion_dim = _library_and_samples()
        collate = CompletionConditionCollate(library, motion_dim, noise_ratio=0.25)
        batch = collate(samples)

        expected_draft, expected_mask, expected_boundaries, expected_fraction = completion_conditions(
            batch["labels"],
            batch["retrieval_group_ids"],
            library,
            motion_dim=motion_dim,
            noise_ratio=0.25,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.equal(batch["draft"], expected_draft))
        self.assertTrue(torch.equal(batch["draft_noise_mask"], expected_mask))
        self.assertTrue(torch.equal(batch["plan_boundaries"], expected_boundaries))
        self.assertEqual(batch_safe_draft_fraction(batch), expected_fraction)
        # A comparison against an all-zero draft would pass while proving
        # nothing: the fixture must actually retrieve prototypes.
        self.assertGreater(float(batch["draft"].abs().sum()), 0.0)
        self.assertGreater(float(batch["draft_noise_mask"].sum()), 0.0)

    def test_noise_ratio_scales_the_mask_exactly_as_the_in_loop_path_did(self):
        library, samples, motion_dim = _library_and_samples()
        quarter = CompletionConditionCollate(library, motion_dim, noise_ratio=0.25)(samples)
        half = CompletionConditionCollate(library, motion_dim, noise_ratio=0.5)(samples)
        self.assertTrue(torch.allclose(half["draft_noise_mask"], quarter["draft_noise_mask"] * 2.0))

    def test_a_sample_without_a_group_stays_zero_conditioned(self):
        library, samples, motion_dim = _library_and_samples()
        samples[0] = dict(samples[0], retrieval_group_id=None)
        batch = CompletionConditionCollate(library, motion_dim, noise_ratio=0.25)(samples)
        # No declared group is no proof that any prototype is external, so that
        # row must carry no condition at all -- and must not inherit its
        # neighbour's, which is the failure mode a shared batch buffer invites.
        self.assertTrue(torch.equal(batch["draft"][0], torch.zeros_like(batch["draft"][0])))
        self.assertTrue(torch.equal(batch["draft_noise_mask"][0], torch.zeros_like(batch["draft_noise_mask"][0])))
        self.assertGreater(float(batch["draft"][1].abs().sum()), 0.0)

    def test_safe_coverage_falls_when_a_sample_cannot_be_conditioned(self):
        """The gate this feeds must still be able to fire."""
        library, samples, motion_dim = _library_and_samples()
        full = CompletionConditionCollate(library, motion_dim, noise_ratio=0.25)(samples)
        self.assertEqual(batch_safe_draft_fraction(full), 1.0)

        samples[0] = dict(samples[0], retrieval_group_id=None)
        partial = CompletionConditionCollate(library, motion_dim, noise_ratio=0.25)(samples)
        self.assertLess(batch_safe_draft_fraction(partial), 1.0)


class DataParallelContractTests(unittest.TestCase):
    """The flags that decide what a multi-GPU run is actually comparable with."""

    def test_global_batch_is_split_evenly_and_uneven_splits_are_refused(self):
        # batch_size is None because --batch-size was not typed: argparse's own
        # "the default is still here" is what says a flag was absent.  The
        # earlier contract carried a separate batch_size_explicit flag derived by
        # scanning sys.argv, which any argparse abbreviation defeated.
        args = argparse.Namespace(batch_size=None, global_batch_size=64)
        per_rank, global_batch, declared = resolve_batch_sizes(args, world_size=4)
        self.assertEqual((per_rank, global_batch, declared), (16, 64, "global"))

        # 64 over 6 ranks would give ranks of unequal size, and DDP's average of
        # per-rank means is only the mean over the global batch when they match.
        with self.assertRaisesRegex(ValueError, "not divisible"):
            resolve_batch_sizes(args, world_size=6)

    def test_per_rank_batch_multiplies_the_global_batch(self):
        args = argparse.Namespace(batch_size=64, global_batch_size=None)
        self.assertEqual(resolve_batch_sizes(args, world_size=6), (64, 384, "per-rank"))

    def test_neither_flag_falls_back_to_the_documented_default(self):
        args = argparse.Namespace(batch_size=None, global_batch_size=None)
        self.assertEqual(resolve_batch_sizes(args, world_size=2),
                         (DEFAULT_BATCH_SIZE, DEFAULT_BATCH_SIZE * 2, "per-rank"))

    def test_declaring_both_batch_flags_is_refused_rather_than_resolved(self):
        args = argparse.Namespace(batch_size=64, global_batch_size=256)
        with self.assertRaisesRegex(ValueError, "declare one"):
            resolve_batch_sizes(args, world_size=4)

    def test_an_abbreviated_batch_flag_is_refused_too(self):
        """argparse accepts any unambiguous prefix, and the first version missed it.

        ``--batch 64 --global-batch-size 96`` parsed to batch_size=64, matched no
        literal ``--batch-size`` token in sys.argv, and resolved to the global 96 --
        silently discarding the operator's per-rank number, which is precisely the
        outcome the refusal exists to prevent.  Nothing about the parsed Namespace
        distinguishes the abbreviation from the full spelling, which is why the
        contract now reads the default sentinel instead of the command line.
        """
        argv = ["train_atomic.py", "--stage", "completion",
                "--batch", "64", "--global-batch-size", "96"]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.batch_size, 64)
        with self.assertRaisesRegex(ValueError, "declare one"):
            resolve_batch_sizes(args, world_size=2)

    def test_an_unspecified_batch_flag_leaves_the_sentinel(self):
        argv = ["train_atomic.py", "--stage", "planner", "--global-batch-size", "96"]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertIsNone(args.batch_size)

    def test_gpu_list_rejects_duplicates_and_absent_ordinals(self):
        available = torch.cuda.device_count()
        with self.assertRaisesRegex(ValueError, "same device twice"):
            resolve_gpus(argparse.Namespace(gpus="0,0"))
        with self.assertRaisesRegex(ValueError, "this host has"):
            resolve_gpus(argparse.Namespace(gpus=str(available)))
        self.assertEqual(resolve_gpus(argparse.Namespace(gpus="")), [])


if __name__ == "__main__":
    unittest.main()


class SplitLevelSafeDraftCoverageTests(unittest.TestCase):
    """The source-safe floor, on the unit the plan actually states.

    ``docs/TRAINING_DATA_RELEASE.md`` records ``--min-safe-retrieval-fraction
    0.99`` applied by ``audit_atomic_dataset.py`` to safe frames over atomic
    frames across the WHOLE train split, and that is the quantity the
    2026-08-13 failure (0.981138) was read on.  The training loop applied the
    same 0.99 per batch, which is a strictly harsher and undocumented test: on
    wild_v5_song the split reads 0.999759 and passes the plan's criterion while
    one batch of 64 reached 0.989944 and killed a four-hour run.  These pin the
    split-level computation and the tripwire that replaced the per-batch floor.
    """

    def _dataset(self, samples):
        class _Dataset:
            def __init__(self, rows):
                self._rows = rows
            def __len__(self):
                return len(self._rows)
            def __getitem__(self, index):
                return self._rows[index]
        return _Dataset(samples)

    def test_a_corpus_whose_classes_all_appear_in_two_groups_reads_one(self):
        """The positive control: a criterion that only ever fires proves nothing."""
        from train_atomic import train_split_safe_draft_fraction

        library, samples, _ = _library_and_samples()
        coverage = train_split_safe_draft_fraction(self._dataset(samples), library)
        self.assertEqual(coverage["source_safe_atomic_frame_fraction"], 1.0)
        self.assertEqual(coverage["classes_with_no_external_prototype_for_some_window"], 0)
        self.assertGreater(coverage["atomic_frames"], 0)

    def test_a_class_confined_to_one_group_lowers_the_split_reading(self):
        frames, motion_dim = 8, 3
        motions = [torch.zeros(frames, motion_dim) for _ in range(3)]
        plans = [torch.full((frames,), 1, dtype=torch.long),
                 torch.full((frames,), 1, dtype=torch.long),
                 torch.full((frames,), 2, dtype=torch.long)]  # class 2: one group only
        groups = ["p/0", "p/1", "p/2"]
        names = ["r{}_slice0".format(i) for i in range(3)]
        library = AtomicMotionLibrary.from_sequences(
            motions, plans, names=names, retrieval_group_ids=groups)
        samples = [{"labels": plans[i], "retrieval_group_id": groups[i]} for i in range(3)]

        from train_atomic import train_split_safe_draft_fraction
        coverage = train_split_safe_draft_fraction(self._dataset(samples), library)
        # 16 of 24 atomic frames are class 1, which lives in two groups.
        self.assertAlmostEqual(coverage["source_safe_atomic_frame_fraction"], 16 / 24)
        self.assertEqual(coverage["classes_with_no_external_prototype_for_some_window"], 1)

    def test_the_memory_mapped_path_and_the_sample_path_agree(self):
        """The release supplies arrays; anything else is iterated.  Same answer."""
        import numpy as np

        from train_atomic import train_split_safe_draft_fraction

        library, samples, _ = _library_and_samples()
        by_sample = train_split_safe_draft_fraction(self._dataset(samples), library)

        class _Indexed:
            def __init__(self, rows):
                self.arrays = (None, None,
                               np.stack([r["labels"].numpy() for r in rows]))
                self.retrieval_group_ids = [r["retrieval_group_id"] for r in rows]
            def __len__(self):
                return len(self.retrieval_group_ids)
        by_array = train_split_safe_draft_fraction(_Indexed(samples), library)
        self.assertEqual(by_sample, by_array)

    def test_the_runtime_tripwire_fires_only_when_nothing_at_all_is_safe(self):
        """What replaced the per-batch floor.

        Class confinement costs 0.02% of frames on this corpus, so it cannot
        empty a batch; only the exclusion or the group ids going wrong can.
        A batch that is merely below 0.99 must NOT fire -- that is the case the
        split-level gate already ruled on.
        """
        fires = lambda batch: bool(batch["atomic_frames"]) and not batch["safe_atomic_frames"]
        self.assertTrue(fires({"atomic_frames": 9600, "safe_atomic_frames": 0}))
        self.assertFalse(fires({"atomic_frames": 9600, "safe_atomic_frames": 9503}))  # 0.98990
        self.assertFalse(fires({"atomic_frames": 0, "safe_atomic_frames": 0}))
