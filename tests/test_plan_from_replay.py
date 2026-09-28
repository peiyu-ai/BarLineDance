"""--plan-from replays an earlier run's plan: labels AND the plan report (bar grid phase).

Why: --stochastic-planner draws a different plan under GPU load (DEFECTS 90.8), so arms
launched in parallel compare different dances unless the plan is pinned.  End to end on the
fixed ten (2026-09-23): replaying a run onto itself was byte-identical 10/10, and with
--seed 7 the planner's own draw differed on 10/10 clips while the replayed labels and
bar_grid_phase matched 10/10.
"""
import pickle
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic  # noqa: E402


def _write(tmp_path, name, labels, phase):
    blob = {"atomic_labels": np.asarray(labels, dtype=np.int64),
            "prototype_retrieval": {"plan_postprocess": {"bar_grid_phase": phase, "frames": len(labels)}}}
    with open(tmp_path / (name + ".pkl"), "wb") as handle:
        pickle.dump(blob, handle)


def test_replay_returns_the_saved_labels_and_report(tmp_path):
    _write(tmp_path, "wild_v5:1:clip000", [3, 3, 1, 1, 0], 2)
    drawn = torch.tensor([5, 5, 5, 5, 5])
    labels, report = infer_atomic._replay_plan(tmp_path, "wild_v5:1:clip000", drawn)
    assert labels.tolist() == [3, 3, 1, 1, 0] and labels.dtype == drawn.dtype
    assert report["bar_grid_phase"] == 2


def test_replay_refuses_a_different_length(tmp_path):
    _write(tmp_path, "wild_v5:1:clip000", [3, 3, 1], 0)
    with pytest.raises(ValueError):
        infer_atomic._replay_plan(tmp_path, "wild_v5:1:clip000", torch.zeros(5, dtype=torch.long))
