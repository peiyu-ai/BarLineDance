"""Pin the D3PM reverse parameterizations in ``model.atomic_planner``.

The ``x0`` path exists because the original ``eq3`` reading trains the network
to predict ``y_{t-1}`` -- itself a noisy sample -- from ``y_t``.  A single
forward step perturbs only a ``1 - alpha_t`` fraction of tokens, so the optimal
predictor is almost the identity, and a model trained that way learns to copy
rather than to generate.  These tests fix the two behaviours apart so that
difference cannot silently regress.
"""

import unittest

import torch
import torch.nn.functional as F

from model.atomic_planner import AtomicPlannerTransformer, UniformD3PM

CLASSES = 6
MUSIC_DIM = 5
STEPS = 8


def make_planner(parameterization):
    torch.manual_seed(20260808)
    backbone = AtomicPlannerTransformer(
        num_atomic_classes=CLASSES,
        music_dim=MUSIC_DIM,
        latent_dim=32,
        num_layers=1,
        num_heads=2,
        ff_size=32,
        dropout=0.0,
        max_seq_len=16,
    )
    return UniformD3PM(backbone, num_steps=STEPS, parameterization=parameterization)


class ParameterizationTests(unittest.TestCase):
    def test_rejects_unknown_parameterization(self):
        with self.assertRaises(ValueError):
            make_planner("posterior-ish")

    def test_x0_target_is_the_clean_label_sequence(self):
        """The whole point: the regression target must not depend on t."""
        planner = make_planner("x0")
        labels = torch.randint(CLASSES + 1, (3, 12))
        music = torch.randn(3, 12, MUSIC_DIM)
        for step in (0, STEPS // 2, STEPS - 1):
            timesteps = torch.full((3,), step, dtype=torch.long)
            output = planner.training_step(labels, music, None, timesteps)
            self.assertTrue(torch.equal(output.target_labels, labels))

    def test_eq3_target_is_a_noisy_sequence(self):
        """Documents the degenerate behaviour rather than hiding it."""
        planner = make_planner("eq3")
        labels = torch.randint(CLASSES + 1, (4, 16))
        music = torch.randn(4, 16, MUSIC_DIM)
        timesteps = torch.full((4,), STEPS - 1, dtype=torch.long)
        output = planner.training_step(labels, music, None, timesteps)
        # At the noisiest step the target has drifted away from the clean labels.
        self.assertFalse(torch.equal(output.target_labels, labels))

    def test_eq3_target_is_near_identical_to_its_input(self):
        """Why eq3 collapses to copying: one step barely changes anything."""
        planner = make_planner("eq3")
        labels = torch.randint(CLASSES + 1, (128, 16))
        music = torch.randn(128, 16, MUSIC_DIM)
        timesteps = torch.zeros(128, dtype=torch.long)
        output = planner.training_step(labels, music, None, timesteps)
        agreement = (output.target_labels == output.noisy_labels).float().mean()
        self.assertGreater(
            float(agreement), 0.8, "a single forward step should perturb few tokens"
        )


class PosteriorTests(unittest.TestCase):
    """``posterior_logits`` must match the uniform-kernel posterior exactly."""

    def _brute_force(self, planner, noisy, x0_probs, step):
        classes = planner.num_classes
        alpha = float(planner.alphas[step])
        alpha_bar_prev = float(planner.alpha_bars[step - 1]) if step > 0 else 1.0

        result = torch.zeros(noisy.shape + (classes,), dtype=torch.float64)
        for b in range(noisy.shape[0]):
            for f in range(noisy.shape[1]):
                observed = int(noisy[b, f])
                for k in range(classes):
                    # q(y_t | y_{t-1} = k)
                    likelihood = (1.0 - alpha) / classes + (alpha if observed == k else 0.0)
                    # sum_{y0} p(y0) q(y_{t-1} = k | y0)
                    prior = 0.0
                    for zero in range(classes):
                        transition = (1.0 - alpha_bar_prev) / classes + (
                            alpha_bar_prev if k == zero else 0.0
                        )
                        prior += float(x0_probs[b, f, zero]) * transition
                    result[b, f, k] = likelihood * prior
        return result

    def test_matches_brute_force(self):
        planner = make_planner("x0")
        torch.manual_seed(3)
        noisy = torch.randint(planner.num_classes, (2, 4))
        x0_logits = torch.randn(2, 4, planner.num_classes, dtype=torch.float64)
        x0_probs = x0_logits.softmax(dim=-1)

        for step in (0, 1, STEPS // 2, STEPS - 1):
            with self.subTest(step=step):
                got = planner.posterior_logits(noisy, x0_logits, step).exp().double()
                expected = self._brute_force(planner, noisy, x0_probs, step)
                # Compare as distributions; posterior_logits is unnormalised.
                got = got / got.sum(-1, keepdim=True)
                expected = expected / expected.sum(-1, keepdim=True)
                error = float((got - expected).abs().max())
                # The schedule buffers (alphas, alpha_bars) are float32, so the
                # agreement floor is float32 epsilon rather than float64's.
                self.assertLess(error, 1e-6, "step {} error {:.3e}".format(step, error))

    def test_posterior_is_a_valid_distribution(self):
        planner = make_planner("x0")
        noisy = torch.randint(planner.num_classes, (3, 7))
        x0_logits = torch.randn(3, 7, planner.num_classes)
        for step in range(STEPS):
            probs = planner.posterior_logits(noisy, x0_logits, step).softmax(dim=-1)
            self.assertTrue(torch.all(probs >= 0))
            self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(3, 7), atol=1e-5))

    def test_confident_prediction_at_final_step_recovers_it(self):
        """With abar_0 ~ 1 and a peaked y_0 belief, the posterior follows it."""
        planner = make_planner("x0")
        noisy = torch.randint(planner.num_classes, (1, 5))
        target = torch.full((1, 5), 2)
        x0_logits = torch.full((1, 5, planner.num_classes), -30.0)
        x0_logits.scatter_(-1, target.unsqueeze(-1), 30.0)
        argmax = planner.posterior_logits(noisy, x0_logits, 1).argmax(dim=-1)
        self.assertTrue(torch.equal(argmax, target))


class SamplingTests(unittest.TestCase):
    def test_sample_shapes_and_range(self):
        for parameterization in ("x0", "eq3"):
            with self.subTest(parameterization=parameterization):
                planner = make_planner(parameterization)
                music = torch.randn(2, 10, MUSIC_DIM)
                sample = planner.sample(music, None, deterministic=True)
                self.assertEqual(sample.shape, (2, 10))
                self.assertTrue(int(sample.min()) >= 0)
                self.assertTrue(int(sample.max()) < planner.num_classes)

    def test_padding_is_zeroed(self):
        planner = make_planner("x0")
        music = torch.randn(2, 10, MUSIC_DIM)
        padding = torch.zeros(2, 10, dtype=torch.bool)
        padding[:, 6:] = True
        sample = planner.sample(music, padding, deterministic=True)
        self.assertTrue(torch.all(sample[padding] == 0))


if __name__ == "__main__":
    unittest.main()


class ClassifierFreeGuidanceTests(unittest.TestCase):
    """The planner had no way to be guided; the completion has had one all along.

    The failure this addresses is measured, not assumed: on 2026-08-23 the
    released planner's x0 head at t=99 predicted P(transition) 0.1781 on train
    music and 0.4295 on test music, and *shuffling the music between songs of
    the same split changed neither* (0.1782 / 0.4299) while changing 96.6% of
    the argmax labels.  Zeroing the music gave 0.7142 on both.  So on unseen
    music the model backs off toward an unconditional prior that is 71%
    transition, which is exactly what guidance exists to push away from.
    """

    def build(self, cond_drop_prob=0.0):
        model = AtomicPlannerTransformer(
            num_atomic_classes=6, music_dim=4, latent_dim=32, num_layers=1,
            num_heads=2, ff_size=32, max_seq_len=64, cond_drop_prob=cond_drop_prob)
        return UniformD3PM(model, num_steps=4)

    def test_a_planner_without_the_null_gains_no_parameters(self):
        """Otherwise every checkpoint saved before this change fails to load."""
        plain = self.build().state_dict()
        self.assertFalse([k for k in plain if "null" in k])
        self.assertIn("model.null_music", self.build(0.25).state_dict())

    def test_guidance_is_refused_when_there_is_nothing_to_guide_away_from(self):
        """Silently conditioning twice would report 'guidance had no effect'."""
        with self.assertRaises(ValueError):
            self.build().sample(torch.randn(2, 8, 4), guidance_weight=2.0)

    def test_weight_one_is_exactly_plain_conditional_sampling(self):
        planner = self.build(0.25).eval()
        music = torch.randn(2, 8, 4)
        torch.manual_seed(0)
        guided = planner.sample(music, guidance_weight=1.0)
        torch.manual_seed(0)
        plain = planner.sample(music)
        self.assertTrue(torch.equal(guided, plain))

    def test_sampling_never_inherits_the_training_drop_rate(self):
        """``forward`` defaults to the module's rate; sampling must not.

        A guided-capable planner sampled through the default would drop its own
        condition on a quarter of the sequences at inference, and the damage
        would read as a bad model rather than a bad call.
        """
        planner = self.build(1.0).eval()   # drop everything, if inherited
        music = torch.randn(4, 8, 4)
        with torch.no_grad():
            timesteps = torch.zeros(4, dtype=torch.long)
            dropped = planner.model(torch.zeros(4, 8, dtype=torch.long), music,
                                    timesteps)
            kept = planner.model(torch.zeros(4, 8, dtype=torch.long), music,
                                 timesteps, cond_drop_prob=0.0)
        self.assertFalse(torch.allclose(dropped, kept),
                         "the fixture must actually distinguish the two paths")
        torch.manual_seed(0)
        a = planner.sample(music)
        torch.manual_seed(0)
        b = planner.sample(music)
        self.assertTrue(torch.equal(a, b), "sampling must be condition-stable")

    def test_guidance_moves_the_logits_away_from_the_unconditional(self):
        planner = self.build(0.25).eval()
        music = torch.randn(3, 8, 4)
        labels = torch.randint(0, 7, (3, 8))
        timesteps = torch.full((3,), 3, dtype=torch.long)
        with torch.no_grad():
            uncond = planner.model(labels, music, timesteps, cond_drop_prob=1.0)
            cond = planner.model(labels, music, timesteps, cond_drop_prob=0.0)
            guided = planner.model.guided_forward(labels, music, timesteps, 3.0)
        self.assertTrue(torch.allclose(guided, uncond + 3.0 * (cond - uncond), atol=1e-5))


class FactorisedHeadTests(unittest.TestCase):
    """Transition competes with 821 classes in one softmax, and loses to nobody.

    Measured on the released planner over 65 test clips, raw per-window output:
    40% of clips come out at 0.188 transition against a ground truth of 0.276,
    while 35% come out at 0.736 with 70% of their bars naming no class at all
    and 2.4 distinct classes in the whole clip.  That failing third carries
    113% of the corpus's total excess and the rest carries -13%, so the defect
    is per-clip collapse rather than global miscalibration -- which is also why
    a logit bias fitted on val overshot on test.
    """

    def build(self, head):
        model = AtomicPlannerTransformer(
            num_atomic_classes=5, music_dim=4, latent_dim=32, num_layers=1,
            num_heads=2, ff_size=32, max_seq_len=64, head=head)
        return UniformD3PM(model, num_steps=5)

    def test_it_adds_no_parameters_so_old_checkpoints_still_load(self):
        self.assertEqual(set(self.build("joint").state_dict()),
                         set(self.build("factorised").state_dict()))

    def test_an_unknown_head_is_refused(self):
        with self.assertRaises(ValueError):
            AtomicPlannerTransformer(num_atomic_classes=5, music_dim=4, latent_dim=8,
                                     num_layers=1, num_heads=2, ff_size=8,
                                     max_seq_len=16, head="binary")

    def test_the_output_is_still_a_distribution_over_the_same_labels(self):
        """Everything downstream reads these through softmax/log_softmax, so a
        head that returned an unnormalised vector would silently change the
        D3PM posterior rather than fail."""
        planner = self.build("factorised")
        labels = torch.randint(0, 6, (2, 8))
        logits = planner.model(labels, torch.randn(2, 8, 4), torch.zeros(2, dtype=torch.long))
        self.assertEqual(logits.shape[-1], 6)
        self.assertTrue(torch.allclose(logits.logsumexp(-1), torch.zeros(2, 8), atol=1e-5))

    def test_class_confidence_no_longer_drains_transition_mass(self):
        """The whole point, stated as the property that must hold."""
        flat = torch.zeros(1, 1, 6)
        flat[..., 0] = 1.0
        confident = flat.clone()
        confident[..., 1:] = 2.0          # the model becomes sure of a class

        joint = lambda z: float(F.softmax(z, dim=-1)[..., 0])
        factorised = lambda z: float(torch.sigmoid(z[..., 0]))
        self.assertLess(joint(confident), joint(flat) / 2,
                        "the joint head must actually show the coupling")
        self.assertAlmostEqual(factorised(confident), factorised(flat), places=6)

    def test_the_two_heads_run_the_same_reverse_chain(self):
        for head in ("joint", "factorised"):
            planner = self.build(head).eval()
            torch.manual_seed(0)
            labels = planner.sample(torch.randn(2, 8, 4))
            self.assertEqual(tuple(labels.shape), (2, 8))
            self.assertTrue(int(labels.max()) < 6)
