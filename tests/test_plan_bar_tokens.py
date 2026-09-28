"""Planning at BAR resolution: the grid, the pooling, the expansion, the gates.

WHAT IS BEING BUILT AND WHY.  M3's vocabulary is a 4-beat music grid:
``data/wild3d/txy_t_labels/report.json`` records
``encoders.segmentation = "music-beat-grid"`` and the segmentation it was built
from validates with ``boundary_off_grid 0`` / ``span_not_k_beats 0`` over 2,015
segments whose duration is 2.00 s median and 2.97 s at the MAXIMUM.  The planner
is nevertheless trained on the per-frame label track, where 98.76% of the tokens
it must emit (784,147 of 794,021 adjacent frame pairs on ``train/labels.npy``)
are "same as the previous frame".  ``--plan-bar-tokens`` is the inference half of
the experiment that asks whether that copy operation is where the capacity went:
the query's music is pooled over the same bars ``--plan-bar-grid`` snaps to, the
planner runs over those tokens, and each bar's label is expanded back across its
frames.

The four things this file pins, one per class below:

1.  The grid and the expansion are EXACT.  An off-by-one in the expansion
    mislabels every frame after it and no downstream reader could see it; this
    repository has a recorded case of exactly that shape (a boundary contrast
    read at ``change[f-1]``, section 2.2 of the defect log, which systematically
    under-read its own arm by 1.394 -> 1.826).
2.  A KNOWN bar plan produces a KNOWN frame track -- the positive control, with
    the planner replaced by a fixture that emits labels chosen in advance.
3.  The frame-level post-process REFUSES rather than being reinterpreted.  A
    5-frame vote over tokens that are whole bars is a five-BAR low-pass.
4.  A frame-trained checkpoint run in bar mode FAILS LOUDLY.  It cannot fail on
    its own: ``MusicNormalization``'s buffers are ``(music_dim,)`` whether they
    were fit on frames or on pooled bars, so ``load_state_dict`` is strict about
    the shape and blind to the distribution -- which the last test in
    ``CheckpointGateTests`` demonstrates rather than asserts.
"""

import hashlib
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import infer_atomic  # noqa: E402
from dataset.bar_tokens import (  # noqa: E402
    bar_bounds,
    bar_lines,
    expand_labels,
    pool_music,
    pooled_dim,
)
from infer_atomic import (  # noqa: E402
    bar_grid_bounds,
    check_planner_bar_tokens,
    infer_plan,
    planner_token_resolution,
)
from model.atomic_planner import AtomicPlannerTransformer, UniformD3PM  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
FPS = 30


def beat_music(frames=200, beat_every=10, dim=35, downbeat_every=4):
    """A synthetic clip whose bar grid is known by construction.

    The onset channel carries energy on every ``downbeat_every``-th beat, so
    ``choose_phase`` picks phase 0 for the reason the rule exists rather than
    arbitrarily -- a flat envelope leaves it free, which is how the fixture in
    ``test_infer_atomic.py`` once ended up on phase 3.
    """
    music = np.zeros((frames, dim), dtype=np.float32)
    music[:, 0] = 1.0
    beats = np.arange(0, frames, beat_every)
    music[beats, infer_atomic.BEAT_CHANNEL] = 1.0
    music[beats[0::downbeat_every], 0] = 10.0
    return torch.from_numpy(music), beats


class FixedPlanner:
    """A planner that emits labels chosen in advance.

    It exists so the expansion can be checked against a KNOWN answer instead of
    against whatever a randomly initialised transformer happens to draw.  It
    records the music it was handed, which is how the pooling is checked at the
    same time.
    """

    model = None

    def __init__(self, bar_labels):
        self.bar_labels = list(bar_labels)
        self.seen = None

    def sample(self, music, padding_mask=None, **kwargs):
        self.seen = music.clone()
        window = music.shape[1]
        row = torch.zeros(window, dtype=torch.long)
        row[: len(self.bar_labels)] = torch.tensor(self.bar_labels, dtype=torch.long)
        return row[None].expand(music.shape[0], window).clone()


class GridAndExpansionTests(unittest.TestCase):
    def test_the_bounds_cover_every_frame_and_start_at_zero(self):
        music, beats = beat_music(frames=200, beat_every=10)
        bounds, phase = bar_grid_bounds(music, 4)
        self.assertEqual(phase, 0)
        self.assertEqual(bounds, [0, 40, 80, 120, 160, 200])
        self.assertEqual(bounds[0], 0)
        self.assertEqual(bounds[-1], len(music))

    def test_the_head_and_tail_are_tokens_not_dropped(self):
        """Phase 2 puts the first bar line at frame 20: the head is a token."""
        beats = np.arange(0, 200, 10)
        bounds = bar_bounds(beats, 4, 2, 200)
        self.assertEqual(bounds, [0, 20, 60, 100, 140, 180, 200])
        # 6 tokens: a 20-frame head, four full bars, a 20-frame tail.
        self.assertEqual(len(bounds) - 1, 6)

    def test_the_grid_is_the_same_slice_retrieval_cuts_on(self):
        """The plan's bars and --draft-bar-prototypes' bars must be one grid."""
        music, _ = beat_music(frames=200, beat_every=10)
        bounds, phase = bar_grid_bounds(music, 4)
        retrieval = infer_atomic.bar_bounds_of(music, 4, phase)
        self.assertEqual([b for b in bounds if 0 < b < 200],
                         [b for b in retrieval if 0 < b < 200])

    def test_expansion_is_exact_at_every_boundary(self):
        bounds = [0, 40, 80, 120, 160, 200]
        frames = expand_labels([7, 0, 3, 3, 11], bounds)
        self.assertEqual(len(frames), 200)
        self.assertEqual(frames[0].item(), 7)
        self.assertEqual(frames[39].item(), 7)
        # The off-by-one that would be invisible downstream:
        self.assertEqual(frames[40].item(), 0)
        self.assertEqual(frames[79].item(), 0)
        self.assertEqual(frames[80].item(), 3)
        self.assertEqual(frames[159].item(), 3)
        self.assertEqual(frames[160].item(), 11)
        self.assertEqual(frames[199].item(), 11)

    def test_expansion_refuses_a_label_count_that_does_not_fill_the_spans(self):
        with self.assertRaises(ValueError):
            expand_labels([1, 2], [0, 40, 80, 120])

    def test_pooling_mean_is_the_bar_mean(self):
        music = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        pooled = pool_music(music, [0, 2, 6], "mean")
        self.assertEqual(tuple(pooled.shape), (2, 2))
        self.assertTrue(torch.allclose(pooled[0], music[0:2].mean(dim=0)))
        self.assertTrue(torch.allclose(pooled[1], music[2:6].mean(dim=0)))

    def test_pooling_mean_std_is_the_two_moments_and_doubles_the_width(self):
        music = torch.randn(9, 5)
        pooled = pool_music(music, [0, 4, 9], "mean_std")
        self.assertEqual(tuple(pooled.shape), (2, 10))
        self.assertEqual(pooled_dim(5, "mean_std"), 10)
        self.assertEqual(pooled_dim(5, "mean"), 5)
        self.assertTrue(torch.allclose(pooled[1][:5], music[4:9].mean(dim=0), atol=1e-6))
        self.assertTrue(torch.allclose(pooled[1][5:],
                                       music[4:9].std(dim=0, unbiased=False), atol=1e-6))

    def test_pooling_a_one_frame_span_reads_std_zero_not_nan(self):
        pooled = pool_music(torch.ones(3, 2) * 4.0, [0, 1, 3], "mean_std")
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertEqual(float(pooled[0][2]), 0.0)

    def test_an_unknown_pooling_is_refused(self):
        with self.assertRaises(ValueError):
            pool_music(torch.zeros(4, 2), [0, 4], "median")

    def test_pooling_the_beat_channel_turns_a_one_hot_into_a_density(self):
        """The measurement the checkpoint gate exists for, stated as a number.

        Frame-level channel 34 is a spike on 1 frame in 10 here; pooled over a
        4-beat bar it is a near-constant 0.1.  Statistics fit on one are wrong
        for the other, and nothing about the ARRAY SHAPE changes -- which is why
        a strict ``load_state_dict`` cannot catch the swap.
        """
        music, _ = beat_music(frames=200, beat_every=10)
        bounds, _ = bar_grid_bounds(music, 4)
        pooled = pool_music(music, bounds, "mean")
        frame_std = float(music[:, infer_atomic.BEAT_CHANNEL].std(unbiased=False))
        pooled_std = float(pooled[:, infer_atomic.BEAT_CHANNEL].std(unbiased=False))
        self.assertGreater(frame_std, 0.29)
        self.assertLess(pooled_std, 0.01)
        self.assertEqual(pooled.shape[1], music.shape[1])


class PlannedTrackTests(unittest.TestCase):
    """The positive control: known bar labels, known frame track."""

    def plan(self, planner, music, window=8, **kwargs):
        stats = {}
        labels = infer_plan(planner, music, window, "cpu", vote_window=1,
                            min_segment_length=1, plan_bar_tokens=True,
                            plan_bar_beats=4, stats=stats, **kwargs)
        return labels, stats

    def test_known_bar_labels_expand_to_the_known_frame_track(self):
        music, _ = beat_music(frames=200, beat_every=10)
        planner = FixedPlanner([7, 0, 3, 3, 11])
        labels, stats = self.plan(planner, music)
        expected = expand_labels([7, 0, 3, 3, 11], [0, 40, 80, 120, 160, 200])
        self.assertTrue(torch.equal(labels, expected))
        self.assertEqual(len(labels), len(music))
        self.assertEqual(stats["bar_tokens"], 5)
        self.assertEqual(stats["bar_bounds"], [0, 40, 80, 120, 160, 200])
        self.assertEqual(stats["bar_grid_phase"], 0)
        self.assertTrue(stats["plan_bar_tokens"])
        self.assertEqual(stats["windows"], 1)

    def test_the_planner_is_handed_one_row_per_bar_not_per_frame(self):
        music, _ = beat_music(frames=200, beat_every=10)
        planner = FixedPlanner([1, 2, 3, 4, 5])
        self.plan(planner, music)
        # [batch, window_bars, music_dim] -- 8 bar slots, 5 real and 3 padded.
        self.assertEqual(tuple(planner.seen.shape), (1, 8, 35))
        bounds = [0, 40, 80, 120, 160, 200]
        self.assertTrue(torch.allclose(planner.seen[0, :5],
                                       pool_music(music, bounds, "mean"), atol=1e-6))
        self.assertTrue(torch.equal(planner.seen[0, 5:], torch.zeros(3, 35)))

    def test_every_frame_of_a_bar_carries_one_label(self):
        music, _ = beat_music(frames=200, beat_every=10)
        labels, stats = self.plan(FixedPlanner([1, 2, 3, 4, 5]), music)
        for start, end in zip(stats["bar_bounds"][:-1], stats["bar_bounds"][1:]):
            self.assertEqual(len(set(labels[start:end].tolist())), 1)

    def test_more_bars_than_one_window_are_planned_in_order(self):
        music, _ = beat_music(frames=400, beat_every=10)
        planner = FixedPlanner([1, 2])
        labels, stats = self.plan(planner, music, window=2)
        self.assertEqual(stats["bar_tokens"], 10)
        self.assertEqual(stats["windows"], 5)
        self.assertEqual(labels[:40].tolist(), [1] * 40)
        self.assertEqual(labels[40:80].tolist(), [2] * 40)
        self.assertEqual(labels[80:120].tolist(), [1] * 40)

    def test_the_bar_grid_flag_is_a_no_op_on_a_bar_token_plan(self):
        """--plan-bar-grid snaps to the grid the plan already sits on."""
        music, _ = beat_music(frames=200, beat_every=10)
        plain, _ = self.plan(FixedPlanner([7, 0, 3, 3, 11]), music)
        snapped, stats = self.plan(FixedPlanner([7, 0, 3, 3, 11]), music,
                                   plan_bar_grid=True)
        self.assertTrue(torch.equal(plain, snapped))
        self.assertTrue(stats["plan_bar_grid"])

    def test_an_unretrievable_class_becomes_transition_and_is_counted_in_frames(self):
        music, _ = beat_music(frames=200, beat_every=10)
        labels, stats = self.plan(FixedPlanner([7, 5, 3, 3, 11]), music,
                                  available_labels=[3, 7, 11])
        self.assertEqual(stats["bars_rewritten_to_transition"], 1)
        # Bar 1 is frames 40..80, so one rewritten BAR is forty rewritten frames.
        self.assertEqual(stats["frames_rewritten_to_transition"], 40)
        self.assertEqual(labels[40:80].tolist(), [0] * 40)

    def test_a_clip_with_no_bar_grid_refuses_instead_of_planning_one_token(self):
        silent = torch.zeros(200, 35)
        with self.assertRaises(ValueError) as caught:
            self.plan(FixedPlanner([1]), silent)
        self.assertIn("no bar grid", str(caught.exception))

    def test_a_real_planner_module_plans_over_bars(self):
        """The plumbing, against the actual model class rather than a stub."""
        torch.manual_seed(0)
        model = AtomicPlannerTransformer(num_atomic_classes=20, music_dim=35,
                                         latent_dim=16, num_layers=1, num_heads=2,
                                         ff_size=32, dropout=0.0)
        planner = UniformD3PM(model, num_steps=2).eval()
        music, _ = beat_music(frames=200, beat_every=10)
        labels, stats = self.plan(planner, music, window=8)
        self.assertEqual(len(labels), 200)
        self.assertEqual(stats["bar_tokens"], 5)
        for start, end in zip(stats["bar_bounds"][:-1], stats["bar_bounds"][1:]):
            self.assertEqual(len(set(labels[start:end].tolist())), 1)


class PostProcessRefusalTests(unittest.TestCase):
    """The frame post-process is refused in bar mode, one offender at a time."""

    def bad(self, **kwargs):
        music, _ = beat_music(frames=200, beat_every=10)
        options = dict(vote_window=1, min_segment_length=1, plan_bar_tokens=True)
        options.update(kwargs)
        with self.assertRaises(ValueError) as caught:
            infer_plan(FixedPlanner([1, 2, 3, 4, 5]), music, 8, "cpu", **options)
        return str(caught.exception)

    def test_a_frame_stride_is_refused_by_name(self):
        self.assertIn("--plan-stride (15 frames)", self.bad(plan_stride=15))

    def test_window_fusion_is_refused_by_name(self):
        self.assertIn("--plan-fusion vote", self.bad(plan_fusion="vote"))

    def test_the_five_frame_vote_is_refused_by_name(self):
        message = self.bad(vote_window=5)
        self.assertIn("--plan-vote-window 5", message)
        self.assertIn("five-BAR low-pass", message)

    def test_the_six_frame_minimum_segment_is_refused_by_name(self):
        self.assertIn("--plan-min-segment 6", self.bad(min_segment_length=6))

    def test_the_shipping_frame_flags_all_refuse_together(self):
        message = self.bad(plan_stride=15, plan_fusion="vote", vote_window=5,
                           min_segment_length=6)
        for flag in ("--plan-stride", "--plan-fusion", "--plan-vote-window",
                     "--plan-min-segment"):
            self.assertIn(flag, message)

    def test_the_accepted_combination_runs(self):
        music, _ = beat_music(frames=200, beat_every=10)
        labels = infer_plan(FixedPlanner([1, 2, 3, 4, 5]), music, 8, "cpu",
                            vote_window=1, min_segment_length=1,
                            plan_bar_tokens=True)
        self.assertEqual(len(labels), 200)

    def test_the_stats_do_not_claim_a_post_process_that_did_not_run(self):
        music, _ = beat_music(frames=200, beat_every=10)
        stats = {}
        infer_plan(FixedPlanner([1, 2, 3, 4, 5]), music, 8, "cpu", vote_window=1,
                   min_segment_length=1, plan_bar_tokens=True, stats=stats)
        for key in ("plan_vote_window", "plan_min_segment_length",
                    "plan_vote_tie_break", "plan_transition_policy",
                    "plan_merge_order"):
            self.assertIsNone(stats[key], key)

    def test_frame_mode_stats_still_say_which_mode_they_came_from(self):
        torch.manual_seed(0)
        model = AtomicPlannerTransformer(num_atomic_classes=4, music_dim=35,
                                         latent_dim=16, num_layers=1, num_heads=2,
                                         ff_size=32, dropout=0.0)
        stats = {}
        infer_plan(UniformD3PM(model, num_steps=2).eval(), torch.randn(200, 35),
                   150, "cpu", stats=stats)
        self.assertFalse(stats["plan_bar_tokens"])
        self.assertEqual(stats["plan_vote_window"], 5)


def planner_args(**overrides):
    args = dict(stage="planner", num_classes=20, music_dim=35, latent_dim=16,
                layers=1, heads=2, ff_size=32, dropout=0.0, seq_len=16,
                diffusion_steps=2)
    args.update(overrides)
    return args


def stub_checkpoint(tmp, name, stats_scale=1.0, **overrides):
    """A tiny planner checkpoint on disk, built here rather than trained.

    Written so the loading path is exercised end to end without waiting on the
    training stage: ``_load_checkpoint`` rebuilds the module from ``args`` and
    ``load_state_dict``s the weights, which is exactly what a real run does.
    """
    import train_atomic

    stats_path = tmp / (name + "_stats.pt")
    torch.save({"mean": torch.zeros(overrides.get("music_dim", 35)),
                "std": torch.full((overrides.get("music_dim", 35),), stats_scale)},
               str(stats_path))
    args = planner_args(music_stats=str(stats_path), **overrides)
    model = train_atomic.planner_model(SimpleNamespace(**args))
    path = tmp / (name + ".pt")
    torch.save({"stage": "planner", "step": 1, "epoch": 1,
                "model": model.state_dict(), "optimizer": {}, "args": args,
                "metrics": {}, "dataset_provenance": None}, str(path))
    return path, args


class CheckpointGateTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory(prefix="bar_tokens_")
        self.tmp = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_checkpoint_without_the_key_is_a_frame_planner(self):
        self.assertEqual(planner_token_resolution(SimpleNamespace()), "frame")

    def test_a_frame_checkpoint_in_bar_mode_fails_loudly(self):
        path, _ = stub_checkpoint(self.tmp, "frame")
        planner, args = infer_atomic._load_checkpoint(path, "planner", "cpu")
        with self.assertRaises(SystemExit) as caught:
            check_planner_bar_tokens(planner, args, 35, True, 4)
        message = str(caught.exception)
        self.assertIn("token-resolution mismatch", message)
        self.assertIn("frame", message)
        self.assertIn("--plan-bar-tokens", message)

    def test_a_bar_checkpoint_without_the_flag_fails_loudly(self):
        path, _ = stub_checkpoint(self.tmp, "bar", planner_token_resolution="bar",
                                  planner_bar_pooling="mean", planner_bar_beats=4)
        planner, args = infer_atomic._load_checkpoint(path, "planner", "cpu")
        with self.assertRaises(SystemExit) as caught:
            check_planner_bar_tokens(planner, args, 35, False, 4)
        self.assertIn("token-resolution mismatch", str(caught.exception))

    def test_a_bar_checkpoint_in_bar_mode_returns_its_pooling(self):
        path, _ = stub_checkpoint(self.tmp, "bar", planner_token_resolution="bar",
                                  planner_bar_pooling="mean", planner_bar_beats=4)
        planner, args = infer_atomic._load_checkpoint(path, "planner", "cpu")
        self.assertEqual(check_planner_bar_tokens(planner, args, 35, True, 4), "mean")

    def test_a_pooling_its_own_width_contradicts_is_refused(self):
        """mean_std doubles the width, so a 35-D projection cannot be mean_std."""
        path, _ = stub_checkpoint(self.tmp, "wide", planner_token_resolution="bar",
                                  planner_bar_pooling="mean_std")
        planner, args = infer_atomic._load_checkpoint(path, "planner", "cpu")
        with self.assertRaises(SystemExit) as caught:
            check_planner_bar_tokens(planner, args, 35, True, 4)
        self.assertIn("70-D", str(caught.exception))

    def test_a_mean_std_checkpoint_of_the_right_width_is_accepted(self):
        path, _ = stub_checkpoint(self.tmp, "wide_ok", music_dim=70,
                                  planner_token_resolution="bar",
                                  planner_bar_pooling="mean_std")
        planner, args = infer_atomic._load_checkpoint(path, "planner", "cpu")
        self.assertEqual(check_planner_bar_tokens(planner, args, 35, True, 4),
                         "mean_std")

    def test_a_different_bar_length_is_refused(self):
        path, _ = stub_checkpoint(self.tmp, "bar8", planner_token_resolution="bar",
                                  planner_bar_beats=8)
        planner, args = infer_atomic._load_checkpoint(path, "planner", "cpu")
        with self.assertRaises(SystemExit) as caught:
            check_planner_bar_tokens(planner, args, 35, True, 4)
        self.assertIn("8-beat bars", str(caught.exception))

    def test_derived_phase_features_are_refused_on_pooled_music(self):
        path, _ = stub_checkpoint(self.tmp, "phase", planner_token_resolution="bar",
                                  music_phase_features=True)
        planner, args = infer_atomic._load_checkpoint(path, "planner", "cpu")
        with self.assertRaises(SystemExit) as caught:
            check_planner_bar_tokens(planner, args, 35, True, 4)
        self.assertIn("beat density", str(caught.exception))

    def test_state_dict_alone_cannot_catch_the_swap(self):
        """WHY the gate has to exist -- demonstrated, not asserted.

        Two checkpoints differing only in the music statistics they carry load
        into each other's module without a murmur, because the buffers are
        ``(35,)`` either way.  Strictness is about shape; the mismatch is about
        distribution.
        """
        import train_atomic

        frame_path, frame_args = stub_checkpoint(self.tmp, "f", stats_scale=0.23)
        bar_path, bar_args = stub_checkpoint(
            self.tmp, "b", stats_scale=0.004, planner_token_resolution="bar")
        frame_state = torch.load(str(frame_path), map_location="cpu")["model"]
        bar_model = train_atomic.planner_model(SimpleNamespace(**bar_args))
        bar_model.load_state_dict(frame_state)          # no error: same shapes
        self.assertEqual(
            tuple(bar_model.model.music_normalization.music_std.shape), (35,))
        self.assertNotEqual(planner_token_resolution(SimpleNamespace(**frame_args)),
                            planner_token_resolution(SimpleNamespace(**bar_args)))



class CommandLineSurfaceTests(unittest.TestCase):
    """The flag exists, defaults OFF, and refuses before a 228 MB read.

    ``infer_plan`` already refuses the frame post-process, but by then the
    planner and completion checkpoints have been loaded -- minutes, and on the
    shipping pair 228 MB and 1.1 GB of I/O -- so ``infer_directory`` runs the
    SAME check (``_bar_token_option_error``, one definition) before it opens
    anything.  The positive control is the second test: the accepted
    combination gets past this gate and fails later, on the data root, which is
    how we know the gate is not simply refusing everything.
    """

    def parse(self, *extra):
        argv = ["infer_atomic.py", "--audio-dir", "audio", *extra]
        original = sys.argv
        sys.argv = argv
        try:
            return infer_atomic.parse_args()
        finally:
            sys.argv = original

    def test_the_flag_is_off_unless_asked_for(self):
        self.assertFalse(self.parse().plan_bar_tokens)
        self.assertTrue(self.parse("--plan-bar-tokens").plan_bar_tokens)

    def run_directory(self, **kwargs):
        options = dict(
            audio_dir=str(self.tmp), output_dir=str(self.tmp / "out"),
            planner_checkpoint=str(self.tmp / "no_planner.pt"),
            completion_checkpoint=str(self.tmp / "no_completion.pt"),
            data_root=str(self.tmp / "no_release"), plan_bar_tokens=True)
        options.update(kwargs)
        return infer_atomic.infer_directory(**options)

    def setUp(self):
        import tempfile

        self.tmpdir = tempfile.TemporaryDirectory(prefix="bar_tokens_cli_")
        self.tmp = pathlib.Path(self.tmpdir.name)
        self.addCleanup(self.tmpdir.cleanup)

    def test_the_shipping_flags_are_refused_before_any_checkpoint_is_read(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_directory(plan_stride=15, plan_fusion="vote",
                               plan_vote_window=5, plan_min_segment_length=6)
        message = str(caught.exception)
        self.assertIn("--plan-bar-tokens is incompatible", message)
        for flag in ("--plan-stride", "--plan-fusion", "--plan-vote-window",
                     "--plan-min-segment"):
            self.assertIn(flag, message)

    def test_the_accepted_combination_passes_this_gate(self):
        """Positive control: it gets past the gate and fails on the checkpoint.

        Without this the first test would pass just as well against a gate that
        refused every bar-mode call; what it shows is that the refusal is about
        the frame flags and nothing else, because the same call with
        ``--plan-vote-window 1 --plan-min-segment 1`` and no stride/fusion runs
        on to ``_load_checkpoint`` and dies there instead.
        """
        with self.assertRaises(FileNotFoundError) as caught:
            self.run_directory(plan_vote_window=1, plan_min_segment_length=1)
        self.assertIn("completion checkpoint", str(caught.exception))
        self.assertNotIn("--plan-bar-tokens is incompatible", str(caught.exception))


# The plan a real eval clip's music produces in FRAME mode, with the shipping
# post-process and a deterministically initialised stub planner.  Recorded from
# the code as it stood BEFORE --plan-bar-tokens existed (git HEAD 125ea71,
# captured by scratchpad/b2_bartokens/fixture_equiv.py); the bar-token change
# refactors ``snap_plan_to_bar_grid``'s grid construction into
# ``bar_grid_bounds``, and this is the assertion that the refactor moved no
# frame.  Regenerate ONLY against a version of the code that predates the
# change -- recomputing it from the current code would make the test agree with
# whatever it does.
EQUIVALENCE_CLIP = "wild_v5:7030793823240424742:clip000"
EQUIVALENCE_SHA = "63f34ef7766ac2aa2c3a469eaeefd507dceae62f5df7071c3d7b236bb9efcde0"
EQUIVALENCE_FRAMES = 619


def shipping_frame_plan(module):
    """The shipped draft-only planning call, on one real clip's real music.

    The planner is a stub with fixed initialisation rather than the 228 MB
    checkpoint: what this pins is the PLAN PIPELINE (windowing at stride 15,
    the vote fusion, the bar-grid snap, the refine), which is the code the bar
    path touches.  The real checkpoint is checked separately, outside the test
    suite, because loading it takes minutes.
    """
    path = REPO / "runs/txy_t_gt_eval/audio" / (EQUIVALENCE_CLIP + ".npy")
    music = torch.from_numpy(np.load(str(path)).astype(np.float32))
    torch.manual_seed(0)
    model = AtomicPlannerTransformer(num_atomic_classes=20, music_dim=35,
                                     latent_dim=32, num_layers=2, num_heads=4,
                                     ff_size=64, dropout=0.0)
    planner = UniformD3PM(model, num_steps=3).eval()
    torch.manual_seed(20260902)
    labels = module.infer_plan(
        planner, music, 150, "cpu", deterministic=False, temperature=1.0,
        vote_window=5, min_segment_length=6, available_labels=list(range(1, 21)),
        plan_stride=15, plan_fusion="vote", plan_vote_tie_break="centre",
        plan_transition_policy="protect", plan_merge_order="shortest",
        planner_guidance_weight=1.0, plan_bar_grid=True, plan_bar_beats=4)
    return labels.numpy().astype(np.int64)


class EquivalenceTest(unittest.TestCase):
    """With the flag off, the plan is the plan this code produced yesterday."""

    def test_frame_mode_reproduces_the_recorded_fixture(self):
        path = REPO / "runs/txy_t_gt_eval/audio" / (EQUIVALENCE_CLIP + ".npy")
        if not path.is_file():
            self.skipTest("the eval clip's music is not in this working tree")
        plan = shipping_frame_plan(infer_atomic)
        self.assertEqual(len(plan), EQUIVALENCE_FRAMES)
        self.assertEqual(hashlib.sha256(plan.tobytes()).hexdigest(), EQUIVALENCE_SHA)


if __name__ == "__main__":
    unittest.main()
