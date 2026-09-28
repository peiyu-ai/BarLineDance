"""Classifier-free guidance must actually reach the plan.

THE DEFECT THESE GUARD.  Until 2026-09-01 ``guided_forward`` built its
difference from two calls that differed only in ``cond_drop_prob``, and that
flag reaches only ``cond_projection`` -- the music.  The draft was fused into
``x`` before the denoiser, so it appeared identically in both terms and
cancelled out of the difference exactly.  Guidance weight 2.0 amplified the
music by 2.0 and the plan by 1.0.

Measured consequence (8 clips, one window, root-relative joints in metres,
everything else byte-identical): re-rolling only the NOISE moved the output
0.4030 m; replacing the ENTIRE draft moved it 0.2282 m (0.57x).  Nothing-changed
read 0.0000.

The assertions below are on INVARIANTS -- which forwards happen, with which
flags, and what the state_dict contains -- not on output values.  docs
DANCE_QUALITY_DEFECTS.md 6.2: the reprojection tests asserted on outputs and
passed on the buggy code too, because an untrained decoder scrambles the
difference; only the invariant version caught it.
"""
import pytest

torch = pytest.importorskip("torch")

from model.atomic_completion import (AtomicCompletionDecoder,      # noqa: E402
                                     AtomicCompletionDiffusion)

MOTION, FRAMES, MUSIC, BATCH = 151, 16, 35, 3


def decoder(**kwargs):
    return AtomicCompletionDecoder(motion_dim=MOTION, seq_len=FRAMES, music_dim=MUSIC,
                                   latent_dim=32, ff_size=32, num_layers=1, num_heads=2,
                                   **kwargs)


def batch():
    return (torch.randn(BATCH, FRAMES, MOTION),
            torch.randn(BATCH, FRAMES, MUSIC),
            torch.zeros(BATCH, dtype=torch.long),
            torch.randn(BATCH, FRAMES, MOTION),
            torch.rand(BATCH, FRAMES, 1))


# ---------------------------------------------------------------- the defect

def test_old_guidance_cancels_the_draft_exactly():
    """The positive control for the whole file: on a model WITHOUT draft
    guidance, two different drafts must give guidance differences that are
    identical, because the draft cancels.  If this ever stops holding, the
    premise of the fix is wrong and everything below is aimed at nothing."""
    model = decoder().eval()
    x, music, t, draft_a, mask = batch()
    draft_b = torch.randn_like(draft_a)
    with torch.no_grad():
        ua = model(x, music, t, draft_a, mask, cond_drop_prob=1.0)
        ca = model(x, music, t, draft_a, mask, cond_drop_prob=0.0)
        ub = model(x, music, t, draft_b, mask, cond_drop_prob=1.0)
        cb = model(x, music, t, draft_b, mask, cond_drop_prob=0.0)
    # Each term moves with the draft ...
    assert not torch.allclose(ca, cb, atol=1e-5)
    # ... but the guidance DIFFERENCE does not move nearly as much, because the
    # draft enters both terms the same way.  Stated as the ratio so the test
    # says how badly the draft is cancelled, not merely that it is.
    term_shift = (ca - cb).abs().mean()
    diff_shift = ((ca - ua) - (cb - ub)).abs().mean()
    assert diff_shift < term_shift, (float(diff_shift), float(term_shift))


# ------------------------------------------------------- the contract, both ways

def test_null_draft_is_in_the_state_dict_only_when_asked_for():
    assert "null_draft" not in decoder().state_dict()
    assert "null_draft" in decoder(draft_guidance=True).state_dict()


def test_a_guided_checkpoint_refuses_to_load_into_a_plain_model():
    guided = decoder(draft_guidance=True)
    with pytest.raises(RuntimeError):
        decoder().load_state_dict(guided.state_dict())


def test_a_plain_checkpoint_refuses_to_load_into_a_guided_model():
    plain = decoder()
    with pytest.raises(RuntimeError):
        decoder(draft_guidance=True).load_state_dict(plain.state_dict())


def test_dropping_the_draft_without_a_null_is_refused_not_defaulted():
    model = decoder()
    x, music, t, draft, mask = batch()
    with pytest.raises(ValueError, match="without draft guidance"):
        model(x, music, t, draft, mask, draft_drop_prob=0.5)


def test_training_refuses_draft_dropout_on_a_model_that_cannot_drop():
    with pytest.raises(ValueError, match="draft_guidance=True"):
        AtomicCompletionDiffusion(decoder(), num_steps=4, draft_drop_prob=0.25)


def test_draft_guidance_at_inference_refuses_an_untrained_checkpoint():
    model = decoder()
    x, music, t, draft, mask = batch()
    with pytest.raises(ValueError, match="--draft-drop-prob"):
        model.guided_forward(x, music, t, draft, mask, 2.0, draft_guidance_weight=3.0)


# ------------------------------------------------------------ the new behaviour

def test_the_null_draft_replaces_whole_samples_not_frames():
    """Per-sample, not per-frame: dropping frames would teach 'the plan is
    sometimes missing here', a different and easier condition than 'there is no
    plan', and only the second gives guidance a term to amplify."""
    model = decoder(draft_guidance=True).eval()
    seen = {}
    original = model.input_adapter.forward

    def spy(value):
        seen["draft"] = value[..., MOTION:2 * MOTION].clone()
        return original(value)

    model.input_adapter.forward = spy
    x, music, t, draft, mask = batch()
    torch.manual_seed(0)
    with torch.no_grad():
        model(x, music, t, draft, mask, draft_drop_prob=1.0)
    dropped = seen["draft"]
    # every frame of every sample is the same null row
    assert torch.allclose(dropped, dropped[:, :1].expand_as(dropped), atol=1e-6)
    assert torch.allclose(dropped[0, 0], model.null_draft[0, 0], atol=1e-6)


def test_the_mask_travels_with_the_draft():
    """A null draft is trusted nowhere.  Leaving the noise mask behind would
    hand the unconditional branch a map of the plan's own segment boundaries,
    which is most of what the draft carries."""
    model = decoder(draft_guidance=True).eval()
    seen = {}
    original = model.input_adapter.forward

    def spy(value):
        seen["mask"] = value[..., 2 * MOTION:2 * MOTION + 1].clone()
        return original(value)

    model.input_adapter.forward = spy
    x, music, t, draft, mask = batch()
    with torch.no_grad():
        model(x, music, t, draft, mask, draft_drop_prob=1.0)
    assert torch.count_nonzero(seen["mask"]) == 0


def test_three_forwards_with_the_right_flags():
    """The composition is e(-,-) + w_m*(e(m,-) - e(-,-)) + w_d*(e(m,d) - e(m,-)).
    Asserted on the CALLS, so a refactor that reorders or reuses a term fails."""
    model = decoder(draft_guidance=True).eval()
    calls = []
    original = model.forward

    def spy(*args, **kwargs):
        calls.append((kwargs.get("cond_drop_prob", 0.0), kwargs.get("draft_drop_prob", 0.0)))
        return original(*args, **kwargs)

    model.forward = spy
    x, music, t, draft, mask = batch()
    with torch.no_grad():
        model.guided_forward(x, music, t, draft, mask, 2.0, draft_guidance_weight=3.0)
    assert calls == [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]


def test_draft_weight_one_reproduces_the_old_two_term_result():
    """The shipped behaviour is the w_d = 1 point of the new family, which is
    exactly why the draft was never amplified.  Algebra:
    e(-,-) + w*(e(m,-) - e(-,-)) + 1*(e(m,d) - e(m,-)).  This pins that the
    implementation matches the identity claimed in the docstring."""
    model = decoder(draft_guidance=True).eval()
    x, music, t, draft, mask = batch()
    with torch.no_grad():
        neither = model(x, music, t, draft, mask, cond_drop_prob=1.0, draft_drop_prob=1.0)
        music_only = model(x, music, t, draft, mask, cond_drop_prob=0.0, draft_drop_prob=1.0)
        both = model(x, music, t, draft, mask, cond_drop_prob=0.0, draft_drop_prob=0.0)
        got = model.guided_forward(x, music, t, draft, mask, 2.0, draft_guidance_weight=1.0)
    expected = neither + 2.0 * (music_only - neither) + 1.0 * (both - music_only)
    assert torch.allclose(got, expected, atol=1e-6)


def test_draft_guidance_moves_the_output_with_the_weight():
    """Able to fail: if the weight did nothing, this reads equal."""
    model = decoder(draft_guidance=True).eval()
    x, music, t, draft, mask = batch()
    with torch.no_grad():
        low = model.guided_forward(x, music, t, draft, mask, 2.0, draft_guidance_weight=1.0)
        high = model.guided_forward(x, music, t, draft, mask, 2.0, draft_guidance_weight=4.0)
    assert not torch.allclose(low, high, atol=1e-4)


def test_default_path_is_untouched():
    """Every checkpoint trained before this must sample as the model it was."""
    model = decoder().eval()
    x, music, t, draft, mask = batch()
    with torch.no_grad():
        unc = model(x, music, t, draft, mask, cond_drop_prob=1.0)
        con = model(x, music, t, draft, mask, cond_drop_prob=0.0)
        got = model.guided_forward(x, music, t, draft, mask, 2.0)
    assert torch.allclose(got, unc + 2.0 * (con - unc), atol=1e-6)


def test_training_step_passes_the_dropout_through():
    model = decoder(draft_guidance=True)
    diffusion = AtomicCompletionDiffusion(model, num_steps=4, draft_drop_prob=0.25)
    seen = {}
    original = model.forward

    def spy(*args, **kwargs):
        seen["draft_drop_prob"] = kwargs.get("draft_drop_prob")
        return original(*args, **kwargs)

    model.forward = spy
    x, music, _, draft, mask = batch()
    diffusion.training_step(x, music, draft, mask)
    assert seen["draft_drop_prob"] == 0.25
