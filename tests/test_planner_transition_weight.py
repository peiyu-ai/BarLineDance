"""``--planner-transition-weight``: the filler class in the PLANNER's own loss.

WHY IT EXISTS.  Ground truth spends 30.4% of its bars in the transition/filler
class; the plans the rhythm planner writes spend 12.3%.  The generated dancer is
named-moving almost all the time while the real one keeps stepping out of the
vocabulary, and that texture gap is what the operator hears as the dance never
breathing.  This is the loss-side lever for it.

THE DEFECT THIS FILE ALSO PINS.  ``--transition-weight`` already existed and
looks like the same thing.  It is not: it is passed only to
``AtomicCompletionDecoder`` and reaches nothing in a planner run.  Measured
2026-09-09 -- two planner runs, one given ``--transition-weight 2.5`` and one
1.0, produced checkpoints recording 2.5 and 1.0 whose weights were
**bit-identical, max difference 0.0**, and whose validation curves agreed to
four decimals at every one of 13 checkpoints.  An arm named by a flag that did
nothing is the family this repository keeps paying for
([[manifest-omits-the-flag-that-named-the-arm]]), so a planner run given that
flag now fails instead.

POSITIVE CONTROL is ``test_the_weight_changes_the_loss``: the two weights must
disagree, otherwise every other assertion here would also pass on an
implementation that ignored the argument -- which is exactly how the completion
flag went unnoticed.
"""
import os
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.atomic_planner import AtomicPlannerTransformer, UniformD3PM

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _model():
    torch.manual_seed(0)
    return AtomicPlannerTransformer(
        num_atomic_classes=20, music_dim=51, latent_dim=32,
        num_layers=1, num_heads=2, ff_size=32, max_seq_len=4)


def _labels_with_filler(seed=1):
    """Labels that CONTAIN class 0.

    ``torch.randint(0, 21, (8, 4))`` draws 32 tokens from 21 classes and misses
    class 0 entirely about one time in five; on such a batch the filler weight
    has nothing to act on and a working implementation reads identical to a
    broken one.  The first version of this file used exactly that draw and its
    own positive control failed, which is the only reason the fixture is
    pinned here.
    """
    torch.manual_seed(seed)
    labels = torch.randint(1, 21, (8, 4))
    labels[:, 0] = 0
    assert (labels == 0).any()
    return labels


def _loss(model, weight, seed=1):
    diffusion = UniformD3PM(model, num_steps=10, transition_weight=weight)
    labels = _labels_with_filler(seed)
    torch.manual_seed(seed)
    music = torch.randn(8, 4, 51)
    torch.manual_seed(seed + 100)
    return float(diffusion.training_step(labels, music).loss)


def test_the_weight_changes_the_loss():
    """POSITIVE CONTROL: without this, an ignored argument passes everything."""
    model = _model()
    assert _loss(model, 1.0) != _loss(model, 2.5)


def test_the_default_reproduces_the_unweighted_loss_exactly():
    """Every existing checkpoint must rebuild into the module it trained as."""
    model = _model()
    diffusion = UniformD3PM(model, num_steps=10)
    labels = _labels_with_filler(1)
    torch.manual_seed(1)
    music = torch.randn(8, 4, 51)
    torch.manual_seed(101)
    assert float(diffusion.training_step(labels, music).loss) == _loss(model, 1.0)


def test_weighting_the_filler_class_raises_the_loss_on_filler_heavy_labels():
    """The weight must act on class 0 specifically, not scale the loss overall.

    All-filler labels must be affected and no-filler labels must not, which a
    global multiplier would fail.
    """
    model = _model()
    torch.manual_seed(3)
    music = torch.randn(6, 4, 51)
    all_filler = torch.zeros(6, 4, dtype=torch.long)
    no_filler = torch.randint(1, 21, (6, 4))

    def loss(labels, weight):
        diffusion = UniformD3PM(model, num_steps=10, transition_weight=weight)
        torch.manual_seed(7)
        return float(diffusion.training_step(labels, music).loss)

    assert loss(all_filler, 3.0) > loss(all_filler, 1.0)
    # cross_entropy normalises by the summed weights, so a batch with no filler
    # is untouched by the filler's weight.
    assert loss(no_filler, 3.0) == pytest.approx(loss(no_filler, 1.0), rel=1e-6)


def test_a_non_positive_weight_is_refused():
    model = _model()
    with pytest.raises(ValueError):
        UniformD3PM(model, num_steps=10, transition_weight=0.0)


def test_the_completion_flag_is_refused_on_a_planner_run():
    """--transition-weight reaches nothing here, so it must fail, not be ignored."""
    completed = subprocess.run(
        [sys.executable, os.path.join(REPO, "train_atomic.py"),
         "--stage", "planner", "--data-root", "/nonexistent",
         "--output-dir", "/nonexistent", "--transition-weight", "2.5"],
        capture_output=True, text=True)
    assert completed.returncode != 0
    assert "--planner-transition-weight" in completed.stderr


def test_the_completion_stage_still_accepts_its_own_flag():
    """The guard must be stage-scoped: completion runs are unaffected."""
    completed = subprocess.run(
        [sys.executable, os.path.join(REPO, "train_atomic.py"),
         "--stage", "completion", "--data-root", "/nonexistent",
         "--output-dir", "/nonexistent", "--transition-weight", "2.5"],
        capture_output=True, text=True)
    assert "--planner-transition-weight" not in completed.stderr
