"""Tests for the seed-stability reader.

The point of this tool is to be able to say "the winner changed when the seed
changed".  So the tests it needs are the ones that prove it CAN say that, and
that it does not quietly rank a checkpoint the gate rejected.
"""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.exp_plannerckpt_stability import (compare, plan_files,  # noqa: E402
                                             rank_of, reseed_disagreement)


def _report(rows, winner, seed):
    return {"rows": {r["checkpoint"]: r for r in rows},
            "verdicts": {r["checkpoint"]: {"checkpoint": r["checkpoint"],
                                           "passes_gate": r.pop("passes_gate")}
                         for r in rows},
            "winner": winner, "seed": seed}


def test_only_gate_survivors_get_a_rank():
    rows = {"a": {"js_train": 0.10}, "b": {"js_train": 0.05}, "c": {"js_train": 0.20}}
    verdicts = {"a": {"passes_gate": True}, "b": {"passes_gate": False},
                "c": {"passes_gate": True}}
    ranks = rank_of(rows, verdicts)
    # b has the best objective and is still unranked: the gate comes first.
    assert ranks == {"a": 1, "c": 2}
    assert "b" not in ranks


def test_a_changed_winner_is_reported_as_unstable():
    left = _report([{"checkpoint": "a", "js_train": 0.10, "passes_gate": True},
                    {"checkpoint": "b", "js_train": 0.12, "passes_gate": True}],
                   winner="a", seed=1)
    right = _report([{"checkpoint": "a", "js_train": 0.13, "passes_gate": True},
                     {"checkpoint": "b", "js_train": 0.11, "passes_gate": True}],
                    winner="b", seed=2)
    result = compare({"s1": left, "s2": right})
    assert result["winner_stable"] is False
    assert result["rank_flips"] == ["a", "b"]
    assert result["gate_flips"] == []


def test_a_stable_pick_is_reported_as_stable_and_the_spread_is_carried():
    left = _report([{"checkpoint": "a", "js_train": 0.10, "passes_gate": True},
                    {"checkpoint": "b", "js_train": 0.20, "passes_gate": True}],
                   winner="a", seed=1)
    right = _report([{"checkpoint": "a", "js_train": 0.11, "passes_gate": True},
                     {"checkpoint": "b", "js_train": 0.19, "passes_gate": True}],
                    winner="a", seed=2)
    result = compare({"s1": left, "s2": right})
    assert result["winner_stable"] is True
    assert result["rank_flips"] == [] and result["gate_flips"] == []
    spread = {r["checkpoint"]: r["js_train_spread"] for r in result["rows"]}
    assert spread["a"] == pytest.approx(0.01)


def test_a_gate_verdict_that_flips_between_seeds_is_named():
    left = _report([{"checkpoint": "a", "js_train": 0.10, "passes_gate": True}],
                   winner="a", seed=1)
    right = _report([{"checkpoint": "a", "js_train": 0.10, "passes_gate": False}],
                    winner=None, seed=2)
    result = compare({"s1": left, "s2": right})
    assert result["gate_flips"] == ["a"]


def test_reseed_disagreement_is_zero_for_identical_plans_and_one_for_disjoint(tmp_path):
    a = {"clip0": np.array([1, 1, 2, 2]), "clip1": np.array([3, 3, 0, 0])}
    np.savez(tmp_path / "ck.pt.seed1.npz", **a)
    np.savez(tmp_path / "ck.pt.seed2.npz", **a)
    files = plan_files(tmp_path)
    assert set(files) == {"ck.pt"} and set(files["ck.pt"]) == {1, 2}
    pairs = reseed_disagreement(files["ck.pt"])
    assert pairs[0]["reseed_disagree"] == pytest.approx(0.0)

    np.savez(tmp_path / "ck.pt.seed3.npz",
             clip0=np.array([9, 9, 9, 9]), clip1=np.array([9, 9, 9, 9]))
    pairs = {tuple(p["seeds"]): p for p in reseed_disagreement(plan_files(tmp_path)["ck.pt"])}
    assert pairs[(1, 3)]["reseed_disagree"] == pytest.approx(1.0)


def test_a_single_report_is_refused_by_the_cli(tmp_path):
    from tools.exp_plannerckpt_stability import main
    path = tmp_path / "one.json"
    path.write_text(json.dumps({"rows": [], "decision": {"verdicts": [], "winner": None}}))
    with pytest.raises(SystemExit):
        main(["--report", "s1={}".format(path), "--out", str(tmp_path / "out.json")])


def test_a_refusal_report_is_not_silently_ranked(tmp_path):
    from tools.exp_plannerckpt_stability import load_report
    path = tmp_path / "refused.json"
    path.write_text(json.dumps({"refused": True, "reason": "missing column"}))
    with pytest.raises(SystemExit):
        load_report(path)
