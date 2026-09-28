"""The velocity term's budget, and the four channels that were eating it.

WHY THIS FILE EXISTS.  ``velocity_loss`` was written over all 151 dimensions on
the stated grounds that weighting them by hand would be "a choice this
repository has no measurement to justify".  The measurement now exists and says
the unweighted term is itself a weighting: on the training set, differences
taken inside a window, the four foot-contact channels carry **86.03%** of the
squared first-difference this term sums, the 23 joints 13.35%, global
orientation **0.60%** and root translation 0.02%.  Those four channels are
strictly binary and flip on 11.61% of frame pairs, so ``--velocity-weight 4.0``
-- what the shipped checkpoint was trained with -- spends most of its budget
making a foot land gradually.

These tests hold the arithmetic of the fix, not the claim that it helps.  That
claim needs a training run with criteria fixed in advance and a parallel
"still has to be a dance" gate; see section 13.3 for what happens without one.
"""
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.atomic_completion import (  # noqa: E402
    AtomicCompletionDecoder, AtomicCompletionDiffusion, MotionGeometry)


def test_skip_contact_removes_exactly_the_contact_channels():
    """Not "roughly" -- the remainder must equal the loss computed on the same
    tensors with those columns sliced off by hand."""
    torch.manual_seed(0)
    a = torch.randn(3, 12, 151)
    b = torch.randn(3, 12, 151)
    contacts = MotionGeometry.CONTACTS
    by_hand = torch.nn.functional.mse_loss(
        a[:, 1:, contacts:] - a[:, :-1, contacts:],
        b[:, 1:, contacts:] - b[:, :-1, contacts:])
    assert torch.allclose(
        AtomicCompletionDiffusion.velocity_loss(a, b, skip_contact=True), by_hand)


def test_a_binary_flag_that_flips_dominates_the_unskipped_term():
    """The mechanism the census describes, on a fixture: a channel that toggles
    between -1 and +1 contributes a squared difference of 4 per frame pair,
    while a smoothly moving joint contributes ~0.  Without the skip the flag
    decides the loss; with it, it cannot."""
    frames = 20
    prediction = torch.zeros(1, frames, 151)
    target = torch.zeros(1, frames, 151)
    flips = torch.tensor([1.0 if i % 2 else -1.0 for i in range(frames)])
    target[0, :, 0] = flips                      # a contact channel, flipping
    prediction[0, :, 0] = 0.0                    # the model refuses to flip
    ramp = torch.linspace(0.0, 0.1, frames)
    target[0, :, 10] = ramp                      # a joint, moving smoothly
    prediction[0, :, 10] = ramp * 0.5

    full = float(AtomicCompletionDiffusion.velocity_loss(prediction, target))
    skipped = float(AtomicCompletionDiffusion.velocity_loss(
        prediction, target, skip_contact=True))
    assert full > 100 * skipped, (full, skipped)


def test_default_is_off_so_old_checkpoints_reproduce():
    a = torch.randn(2, 8, 151)
    b = torch.randn(2, 8, 151)
    assert torch.equal(AtomicCompletionDiffusion.velocity_loss(a, b),
                       AtomicCompletionDiffusion.velocity_loss(a, b, skip_contact=False))
    import inspect
    signature = inspect.signature(AtomicCompletionDiffusion.__init__)
    assert signature.parameters["velocity_skip_contact"].default is False


def test_the_flag_reaches_the_training_step():
    """A flag that is recorded but not applied is the defect shape this
    repository keeps paying for, so the wiring is asserted rather than assumed."""
    decoder = AtomicCompletionDecoder(motion_dim=151, music_dim=35, seq_len=8,
                                      latent_dim=32, num_layers=1, num_heads=2,
                                      ff_size=32)
    model = AtomicCompletionDiffusion(decoder, num_steps=4, velocity_weight=1.0,
                                      velocity_skip_contact=True)
    assert model.velocity_skip_contact is True
    seen = {}
    original = AtomicCompletionDiffusion.velocity_loss

    def spy(prediction, target, skip_contact=False):
        seen["skip_contact"] = skip_contact
        return original(prediction, target, skip_contact=skip_contact)

    model.velocity_loss = spy
    motion = torch.randn(2, 8, 151)
    model.training_step(motion, torch.randn(2, 8, 35), motion.clone(),
                        torch.ones(2, 8, 1))
    assert seen.get("skip_contact") is True
