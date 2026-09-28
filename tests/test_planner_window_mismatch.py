"""A planner whose window is longer than the completion's, and when that is safe.

WHY THIS EXISTS.  ``infer_directory`` has always refused a planner and a
completion whose ``seq_len`` differ.  That refusal is right whenever the
completion consumes the plan: the plan would have been produced in windows the
completion was never trained to consume, and nothing downstream would say so.

It is wrong in exactly one case.  Under ``--draft-only`` the completion is never
called -- ``infer_directory`` writes the retrieval draft as the generated motion
-- so the completion checkpoint is read for its ``motion_dim`` and its
normalizer and for nothing else.  Without the exemption, a context-length arm
(a planner trained on a 300-frame window against the shipped 150-frame
completion) cannot be evaluated at all without first training a matching
completion, which would put two treatments on one arm.

THE SECOND HALF of the change is the ``short_batch_mode`` conjunct.  That
batched path pads the draft to ``completion_args.seq_len`` and calls the
completion unconditionally, so a 300-frame planner window over clips shorter
than 300 frames would have truncated the plan to 150 frames -- silently, and
under ``--draft-only`` too.  The conjunct sends a mismatched run down the
per-clip path instead.  It is a no-op for every run whose windows match, which
is every run that could exist before this change; the last test here is the
positive control that pins that down.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from infer_atomic import infer_directory


class PlannerStub:
    def sample(self, music, padding_mask=None, temperature=1.0, deterministic=False,
               guidance_weight=1.0, transition_logit_bias=0.0):
        return torch.ones(music.shape[:2], dtype=torch.long, device=music.device)


class CompletionStub:
    """Records every call, so "the completion never ran" is a real assertion."""

    def __init__(self):
        self.calls = 0

    def sample(self, music, draft, noise_mask, guidance_weight=None, **options):
        self.calls += 1
        return draft


def _fixture(directory, frames):
    """A legacy-layout root, one audio stem, and a place to write."""
    train = os.path.join(directory, "dataset", "train")
    os.makedirs(train)
    np.save(os.path.join(train, "motion.npy"), np.ones((1, 4, 2), dtype=np.float32))
    np.save(os.path.join(train, "labels.npy"), np.ones((1, 4), dtype=np.uint8))
    with open(os.path.join(train, "names.json"), "w") as handle:
        json.dump(["query_slice0"], handle)
    open(os.path.join(directory, "dataset", "normalizer.pt"), "wb").close()
    audio_dir = os.path.join(directory, "audio")
    os.makedirs(audio_dir)
    open(os.path.join(audio_dir, "query.wav"), "wb").close()
    return audio_dir, os.path.join(directory, "dataset"), frames


def _run(directory, planner_seq_len, completion_seq_len, frames, draft_only):
    audio_dir, data_root, frames = _fixture(directory, frames)
    planner_args = SimpleNamespace(music_dim=2, seq_len=planner_seq_len,
                                   draft_noise_ratio=0.25, motion_dim=2)
    completion_args = SimpleNamespace(music_dim=2, seq_len=completion_seq_len,
                                      draft_noise_ratio=0.25, motion_dim=2)
    completion = CompletionStub()

    def load_checkpoint(_, expected_stage, __):
        if expected_stage == "completion":
            return completion, completion_args
        return PlannerStub(), planner_args

    written = []

    def record_result(*positional, **kwargs):
        # ``_write_generated_result`` takes (output_path, motion, labels, ...)
        # positionally, so the plan is in ``positional``, not in ``kwargs``.
        written.append({"positional": positional, "kwargs": kwargs})

    with (
        mock.patch("infer_atomic._load_checkpoint", side_effect=load_checkpoint),
        mock.patch("infer_atomic._load_music", return_value=torch.zeros(frames, 2)),
        mock.patch("infer_atomic._write_generated_result", side_effect=record_result),
    ):
        manifest = infer_directory(
            audio_dir=audio_dir,
            output_dir=os.path.join(directory, "output"),
            data_root=data_root,
            planner_checkpoint="planner.pt",
            completion_checkpoint="completion.pt",
            device="cpu",
            completion_stride=completion_seq_len,
            draft_only=draft_only,
        )
    return manifest, completion, written


class PlannerWindowMismatchTests(unittest.TestCase):
    def test_a_mismatched_window_is_still_refused_when_the_completion_runs(self):
        """The gate can fail, and its message names both windows."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as caught:
                _run(directory, planner_seq_len=8, completion_seq_len=4,
                     frames=16, draft_only=False)
        self.assertIn("8", str(caught.exception))
        self.assertIn("4", str(caught.exception))
        self.assertIn("--draft-only", str(caught.exception))

    def test_draft_only_runs_a_longer_planner_window_without_the_completion(self):
        """POSITIVE CONTROL: the case the exemption exists for actually works."""
        with tempfile.TemporaryDirectory() as directory:
            manifest, completion, written = _run(
                directory, planner_seq_len=8, completion_seq_len=4,
                frames=16, draft_only=True)
        self.assertEqual(completion.calls, 0)
        self.assertEqual(len(written), 1)
        self.assertEqual(manifest["sampling"]["draft_only"], True)
        # The plan covers the whole clip, not one completion window of it.
        self.assertEqual(len(written[0]["positional"][2]), 16)

    def test_the_exemption_does_not_excuse_a_run_whose_completion_consumes_the_plan(self):
        """Direction check: matched windows still run, mismatched ones still stop.

        Without this the previous test could pass because the gate had been
        removed rather than narrowed.
        """
        with tempfile.TemporaryDirectory() as directory:
            manifest, completion, written = _run(
                directory, planner_seq_len=4, completion_seq_len=4,
                frames=16, draft_only=False)
        self.assertEqual(len(written), 1)
        self.assertGreaterEqual(completion.calls, 1)

    def test_a_clip_shorter_than_the_planner_window_does_not_take_the_batched_path(self):
        """The ``short_batch_mode`` conjunct, and its positive control.

        A 8-frame planner window over a 6-frame clip satisfies
        ``frames <= planner_args.seq_len``, so before the conjunct this run
        entered the batched path, padded the draft to the COMPLETION's 4 frames
        and called the completion despite ``--draft-only``.  It must now take
        the per-clip path: completion untouched, plan the clip's own length.
        """
        with tempfile.TemporaryDirectory() as directory:
            manifest, completion, written = _run(
                directory, planner_seq_len=8, completion_seq_len=4,
                frames=6, draft_only=True)
        self.assertEqual(completion.calls, 0)
        self.assertEqual(len(written[0]["positional"][2]), 6)

    def test_matched_windows_still_take_the_batched_path_for_a_short_clip(self):
        """POSITIVE CONTROL for the conjunct: it changed nothing when equal.

        Same short clip, same everything, but the two windows agree -- the run
        must still go through the batched path, which is what calling the
        completion here proves.
        """
        with tempfile.TemporaryDirectory() as directory:
            manifest, completion, written = _run(
                directory, planner_seq_len=8, completion_seq_len=8,
                frames=6, draft_only=False)
        self.assertEqual(completion.calls, 1)


if __name__ == "__main__":
    unittest.main()
