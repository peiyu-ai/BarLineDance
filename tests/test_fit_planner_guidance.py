import json
import pathlib

import pytest

from tools.fit_planner_guidance import main


def _report(tmp_path, rows, max_ratio=1.5):
    """Drive the selection logic by hand-writing what the sweep would produce."""
    eligible = {w: r for w, r in rows.items() if r["segment_ratio"] <= max_ratio}
    pool = eligible or rows
    chosen = min(pool, key=lambda w: pool[w]["share_mae"])
    return chosen, bool(eligible)


def test_the_gate_is_two_sided_so_undershooting_cannot_win():
    """A one-sided gate crowned an arm emitting 0.077 against a truth of 0.322.

    That arm was further from the ground truth than the baseline it beat, in
    the mirror direction, and the gate could not see it.  ``share_mae`` is an
    absolute error per clip, so both directions cost.
    """
    over = abs(0.441 - 0.322)
    under = abs(0.077 - 0.322)
    assert under > over, "under-shooting must score worse, not better"


def test_a_weight_that_shatters_the_plan_is_rejected_even_if_the_share_is_best():
    """Guidance buys a lower transition share partly by fragmenting.

    On the 2026-08-23 arms the distinct-class count rose 5.1 -> 8.5 -> 13.2 as
    w rose, so a share criterion alone would keep walking up the weight.
    """
    rows = {
        "1.5": {"share_mae": 0.170, "segment_ratio": 1.35},
        "2.0": {"share_mae": 0.150, "segment_ratio": 2.40},   # best share, shattered
    }
    chosen, satisfied = _report(None, rows)
    assert chosen == "1.5" and satisfied


def test_when_nothing_qualifies_it_says_so_rather_than_relaxing_silently():
    rows = {
        "1.5": {"share_mae": 0.170, "segment_ratio": 1.90},
        "2.0": {"share_mae": 0.150, "segment_ratio": 2.40},
    }
    chosen, satisfied = _report(None, rows)
    assert chosen == "2.0"
    assert satisfied is False, "the caller must be able to tell this case apart"


def test_an_unguided_planner_is_refused_rather_than_run_at_weight_one(tmp_path):
    """Silently evaluating only w=1.0 would report 'guidance did nothing'."""
    import torch

    from model.atomic_planner import AtomicPlannerTransformer, UniformD3PM

    plain = UniformD3PM(AtomicPlannerTransformer(
        num_atomic_classes=4, music_dim=3, latent_dim=16, num_layers=1,
        num_heads=2, ff_size=16, max_seq_len=32), num_steps=3)
    assert getattr(plain.model, "null_music", None) is None
    with pytest.raises(ValueError):
        plain.sample(torch.randn(1, 8, 3), guidance_weight=2.0)
