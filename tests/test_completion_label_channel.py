"""The completion stage's optional label channel: it must be opt-in, it must
actually change the output, and both directions of misuse must fail loudly."""
import inspect
import unittest
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model.atomic_completion import AtomicCompletionDecoder, AtomicCompletionDiffusion
import train_atomic


def _decoder(num_classes=None):
    return AtomicCompletionDecoder(motion_dim=16, seq_len=8, music_dim=5,
                                   latent_dim=32, ff_size=32, num_layers=1,
                                   num_heads=2, num_classes=num_classes)


def test_label_channel_is_absent_unless_asked_for():
    assert _decoder().label_embedding is None
    assert _decoder(num_classes=7).label_embedding is not None
    # num_classes=0 is "no vocabulary", not "a vocabulary of none"
    assert _decoder(num_classes=0).label_embedding is None


def test_the_adapter_widens_by_exactly_the_label_dimension():
    plain = _decoder().input_adapter[0].in_features
    labelled = AtomicCompletionDecoder(motion_dim=16, seq_len=8, music_dim=5,
                                       latent_dim=32, ff_size=32, num_layers=1,
                                       num_heads=2, num_classes=7,
                                       label_dim=11).input_adapter[0].in_features
    assert labelled - plain == 11


def test_a_different_plan_changes_the_prediction():
    """The point of the channel: two plans must not produce the same output.

    Without it the plan reaches the model only through the retrieved draft, and
    on the wild arm that draft is followed at 4-5% over chance.  Here the draft
    is held fixed and only the labels move, so any difference is the channel.
    """
    torch.manual_seed(0)
    model = _decoder(num_classes=7).eval()
    motion = torch.randn(2, 8, 16)
    music = torch.randn(2, 8, 5)
    draft = torch.randn(2, 8, 16)
    mask = torch.zeros(2, 8, 1)
    steps = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        a = model(motion, music, steps, draft, mask, labels=torch.zeros(2, 8, dtype=torch.long))
        b = model(motion, music, steps, draft, mask, labels=torch.full((2, 8), 3, dtype=torch.long))
    assert not torch.allclose(a, b), "the label channel must reach the prediction"


def test_both_directions_of_misuse_are_refused():
    args = (torch.randn(2, 8, 16), torch.randn(2, 8, 5), torch.zeros(2, dtype=torch.long),
            torch.randn(2, 8, 16), torch.zeros(2, 8, 1))
    labels = torch.zeros(2, 8, dtype=torch.long)
    for model, kwargs, expected in ((_decoder(num_classes=7), {}, "labels are required"),
                                    (_decoder(), {"labels": labels}, "no label channel")):
        try:
            model(*args, **kwargs)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError("expected a refusal for: " + expected)
    # a label grid of the wrong shape is a bug, not something to broadcast
    try:
        _decoder(num_classes=7)(*args, labels=torch.zeros(2, 9, dtype=torch.long))
    except ValueError as error:
        assert "shape [batch, frames]" in str(error)
    else:
        raise AssertionError("a mis-shaped label grid must be refused")


def test_the_flag_is_off_by_default_and_builds_the_plain_model():
    source = pathlib.Path(train_atomic.__file__).read_text(encoding="utf-8")
    assert '"--completion-label-channel"' in source
    assert 'action="store_true"' in source
    import argparse
    args = argparse.Namespace(motion_dim=16, seq_len=8, music_dim=5, latent_dim=32,
                              ff_size=32, layers=1, heads=2, dropout=0.0,
                              diffusion_steps=4, transition_weight=1.0,
                              cond_drop_prob=0.25, guidance_weight=2.0,
                              num_classes=7)
    assert train_atomic.completion_model(args).model.label_embedding is None
    args.completion_label_channel = True
    assert train_atomic.completion_model(args).model.label_embedding is not None


class VelocityLossTests(unittest.TestCase):
    """The term EDGE has and this completion did not.

    Measured 2026-08-23 on ``completion_clean5b5_v1A_s15``: fed the ground
    truth noised to t=0 -- asked to reproduce an essentially clean input -- the
    model returns a root trajectory 397% as long (5.386 m against 1.356 m over
    a 5 s window, median of 64 test windows).  The full reverse chain reaches
    541%, so most of it is the model rather than the sampler, and root position
    is 3 of 151 dimensions so an unweighted MSE barely notices.
    """

    def build(self, velocity_weight):
        decoder = AtomicCompletionDecoder(motion_dim=8, seq_len=12, music_dim=4,
                                          latent_dim=16, num_layers=1, num_heads=2,
                                          ff_size=16)
        return AtomicCompletionDiffusion(decoder, num_steps=5,
                                         velocity_weight=velocity_weight)

    def test_the_term_adds_no_parameters_so_old_checkpoints_still_load(self):
        self.assertEqual(set(self.build(0.0).state_dict()),
                         set(self.build(2.0).state_dict()))

    def test_zero_weight_is_the_published_objective_exactly(self):
        model = self.build(0.0)
        torch.manual_seed(0)
        out = model.training_step(torch.randn(2, 12, 8), torch.randn(2, 12, 4),
                                  torch.zeros(2, 12, 8), torch.zeros(2, 12, 1))
        self.assertEqual(float(out.velocity), 0.0)
        self.assertTrue(torch.allclose(out.total, out.denoising))

    def test_the_weight_enters_the_total_linearly_and_leaves_denoising_alone(self):
        off, on = self.build(0.0), self.build(2.0)
        on.load_state_dict(off.state_dict())
        args = (torch.randn(2, 12, 8), torch.randn(2, 12, 4),
                torch.zeros(2, 12, 8), torch.zeros(2, 12, 1))
        torch.manual_seed(1)
        a = off.training_step(*args)
        torch.manual_seed(1)
        b = on.training_step(*args)
        self.assertTrue(torch.allclose(a.denoising, b.denoising))
        self.assertTrue(torch.allclose(b.total - a.total, 2.0 * b.velocity))

    def test_it_scores_a_jittery_sequence_worse_than_a_smooth_one(self):
        """A term that cannot separate these would not be worth training on."""
        frames = torch.linspace(0, 1, 40)[None, :, None].repeat(1, 1, 3)
        jitter = frames + 0.05 * torch.sin(torch.arange(40, dtype=torch.float32) * 3.0
                                           )[None, :, None]
        smooth = AtomicCompletionDiffusion.velocity_loss(frames, frames)
        rough = AtomicCompletionDiffusion.velocity_loss(jitter, frames)
        self.assertEqual(float(smooth), 0.0)
        self.assertGreater(float(rough), 0.0)
        # and plain MSE barely sees it, which is the whole point
        import torch.nn.functional as F
        self.assertLess(float(F.mse_loss(jitter, frames)), float(rough))


# ---------------------------------------------------------------------------
# Manifold reprojection.  Its whole justification is that the denoiser is
# unbiased ON the data manifold and biased OFF it, and that its own iterates
# leave and never return -- so the correction has to happen mid-chain and the
# chain has to keep denoising afterwards.  The first version fired at step 0
# too, which replaced the iterate with the draft one step before the end; the
# outputs of ``every 25`` and ``every 50`` then landed 0.0103 m apart on real
# clips, which cannot happen if the steps between corrections do any work.
# ---------------------------------------------------------------------------

def _diffusion(steps=20):
    torch.manual_seed(0)
    return AtomicCompletionDiffusion(_decoder(), num_steps=steps)


def _sample(diff, **kw):
    torch.manual_seed(1234)
    music = torch.zeros(1, 8, 5)
    draft = torch.randn(1, 8, 16)
    mask = torch.zeros(1, 8, 1)
    return diff.sample(music, draft, mask, **kw)


def _reprojection_steps(diff, **kw):
    """The timesteps the chain reprojects at, recorded from ``q_sample`` itself.

    Asserting on the OUTPUT cannot see this invariant: with an untrained
    decoder one denoising step scrambles the iterate, so a chain that ends by
    replacing everything with the draft and a chain that does not produce
    equally arbitrary tensors.  The first version of these tests did assert on
    the output and passed against the very bug they were written for.
    """
    seen = []
    original = type(diff).q_sample

    def recording(self, clean, timesteps, noise=None):
        seen.append(int(timesteps.reshape(-1)[0]))
        return original(self, clean, timesteps, noise)

    type(diff).q_sample = recording
    try:
        _sample(diff, **kw)
    finally:
        type(diff).q_sample = original
    return seen


def test_reprojection_never_fires_on_the_last_step():
    """Step 0 would replace the iterate with the draft one step before the end,
    and the output would BE the draft -- which is how it first read as a
    spectacular arm-span result."""
    steps = _reprojection_steps(_diffusion(steps=20), reproject_every=10)
    assert 0 not in steps
    assert steps == [10]


def test_the_interval_is_the_interval():
    assert _reprojection_steps(_diffusion(steps=20), reproject_every=5) == [15, 10, 5]
    assert _reprojection_steps(_diffusion(steps=30), reproject_every=10) == [20, 10]


def test_reprojection_is_off_by_default():
    diff = _diffusion(steps=20)
    assert _reprojection_steps(diff) == []
    assert _reprojection_steps(diff, reproject_every=0) == []
    assert torch.allclose(_sample(diff), _sample(diff, reproject_every=None))


# ---------------------------------------------------------------------------
# Respaced sampling.  The falsification test for the single-axis reading of the
# sampler switches, so its own default must be provably inert.
# ---------------------------------------------------------------------------

def _model_calls(diff, **kw):
    seen = []
    original = diff.model.guided_forward

    def recording(*a, **k):
        seen.append(int(a[2].reshape(-1)[0]))
        return original(*a, **k)

    diff.model.guided_forward = recording
    try:
        _sample(diff, **kw)
    finally:
        diff.model.guided_forward = original
    return seen


def test_respacing_is_inert_unless_asked_for():
    diff = _diffusion(steps=20)
    full = list(range(19, -1, -1))
    assert _model_calls(diff) == full
    assert _model_calls(diff, sample_steps=None) == full
    assert _model_calls(diff, sample_steps=0) == full
    # asking for more steps than the chain has cannot invent any
    assert _model_calls(diff, sample_steps=40) == full


def test_respacing_shortens_the_chain_and_keeps_both_ends():
    diff = _diffusion(steps=20)
    chain = _model_calls(diff, sample_steps=5)
    assert len(chain) == 5
    assert chain[0] == 19 and chain[-1] == 0
    assert chain == sorted(chain, reverse=True)


def test_a_respaced_chain_still_lands_in_range():
    """The respaced posterior is recomputed from alpha_bars rather than read
    off the trained buffers; a wrong alpha there blows the iterate up, and the
    output is clamped nowhere."""
    diff = _diffusion(steps=20)
    out = _sample(diff, sample_steps=5)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 100.0


# ---------------------------------------------------------------------------
# The EDGE auxiliary objective.  Its danger is being quietly wrong -- a
# geometric loss against a wrong skeleton is a plausible gradient toward a
# subtly wrong body -- so what is tested is agreement with the one decode path
# the rest of the repository already trusts, and the refusals.
# ---------------------------------------------------------------------------

def _normalizer(dim=151):
    return {"data_min": -torch.ones(dim), "data_max": torch.ones(dim)}


def test_geometry_is_refused_without_the_normalizer():
    with pytest.raises(ValueError):
        AtomicCompletionDiffusion(_decoder(), num_steps=4, fk_weight=1.0)


def test_geometry_losses_are_off_by_default_and_change_the_loss_when_on():
    torch.manual_seed(0)
    plain = AtomicCompletionDiffusion(_decoder151(), num_steps=4)
    torch.manual_seed(0)
    geometric = AtomicCompletionDiffusion(_decoder151(), num_steps=4,
                                          fk_weight=1.0, fk_velocity_weight=1.0,
                                          contact_weight=1.0, normalizer=_normalizer())
    torch.manual_seed(7)
    batch = (torch.randn(2, 8, 151), torch.randn(2, 8, 5),
             torch.randn(2, 8, 151), torch.zeros(2, 8, 1))
    torch.manual_seed(1)
    a = plain.training_step(*batch)
    torch.manual_seed(1)
    b = geometric.training_step(*batch)
    assert torch.allclose(a.denoising, b.denoising)
    assert not torch.allclose(a.total, b.total)


def test_the_normalizer_travels_in_the_state_dict():
    with_geo = AtomicCompletionDiffusion(_decoder151(), num_steps=4, fk_weight=1.0,
                                         normalizer=_normalizer()).state_dict()
    assert any("geometry" in key for key in with_geo)
    plain = AtomicCompletionDiffusion(_decoder151(), num_steps=4)
    with pytest.raises(RuntimeError):
        plain.load_state_dict(with_geo)


def test_contact_gate_is_detached():
    """The cheap escape from the contact term is predicting no contact at all;
    detaching the gate removes that gradient path.  Asserted on the gradient
    itself: the contact channels of the prediction must receive gradient ONLY
    through the denoising term, i.e. the same gradient as with the term off."""
    geo = AtomicCompletionDiffusion(_decoder151(), num_steps=4,
                                    contact_weight=100.0, normalizer=_normalizer())
    motion = torch.randn(1, 8, 151, requires_grad=True)
    joints = geo.geometry.joints(motion)
    feet = joints[:, :, list(geo.geometry.FEET), :]
    velocity = feet[:, 1:] - feet[:, :-1]
    gate = geo.geometry.contacts(motion)[:, 1:].detach().clamp(0, 1)
    (velocity.pow(2).sum(-1) * gate).mean().backward()
    assert motion.grad[..., :4].abs().max() == 0.0


import pytest


def _decoder151():
    return AtomicCompletionDecoder(motion_dim=151, seq_len=8, music_dim=5,
                                   latent_dim=32, ff_size=32, num_layers=1,
                                   num_heads=2)


def test_energy_match_gets_worse_under_damping_not_better():
    """The property that makes this term different from every FK MSE: an
    under-driven prediction cannot lower it by damping further.  Asserted
    directly -- the same target, one prediction at full amplitude and one at
    half, the half-amplitude one must carry the LARGER energy term."""
    norm = _normalizer()
    geo = AtomicCompletionDiffusion(_decoder151(), num_steps=4,
                                    energy_match_weight=1.0, normalizer=norm)
    torch.manual_seed(0)
    target = torch.randn(1, 8, 151)

    def energy_term(prediction):
        joints_p = geo.geometry.joints(prediction)
        joints_t = geo.geometry.joints(target)
        rp = joints_p - joints_p[..., :1, :]
        rt = joints_t - joints_t[..., :1, :]
        sp = (rp[:, 1:] - rp[:, :-1]).norm(dim=-1).mean(dim=(1, 2))
        st = (rt[:, 1:] - rt[:, :-1]).norm(dim=-1).mean(dim=(1, 2))
        return float(((sp - st).abs() / st.clamp_min(1e-4)).mean())

    matched = energy_term(target.clone())
    # Damping must be TEMPORAL, not a scalar on the vector: the rot6d
    # representation Gram-Schmidt-normalizes, so scaling the whole 151-D vector
    # by 0.5 leaves every rotation -- and therefore every joint speed --
    # unchanged.  The first version of this test damped that way and read
    # 2e-7 where it asserted > 0.2; the same invariance is how a real model
    # damps too (rotations drawn toward their temporal mean), so smooth in time.
    damped_input = target.clone()
    damped_input[:, 1:-1] = (target[:, :-2] + target[:, 1:-1] + target[:, 2:]) / 3
    damped = energy_term(damped_input)
    assert matched < 1e-6
    assert damped > 0.05
