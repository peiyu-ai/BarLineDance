"""Gate v2 must say, in every artifact, whether it was read against a control.

The defect this pins: the same-song rate statistic is satisfied by anything
that follows the beat.  A metronome with random labels scores p = 2.0e-12 on
clean5b5 train -- better than the ground truth (6.1e-03) and better than the
planner (1.2e-04).  A report that does not carry that fact invites the reader
to treat a pass as evidence of music-conditioned choreography.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _plans(path, songs, seed, segments):
    rng = np.random.RandomState(seed)
    plans = []
    for song_index, song in enumerate(songs):
        for take in range(3):
            labels = np.zeros(300, dtype=int)
            cuts = np.linspace(0, 300, segments[song_index] + 1).astype(int)
            for a, b in zip(cuts[:-1], cuts[1:]):
                labels[a:b] = int(rng.randint(1, 40))
            plans.append({"sequence": "w:{}{}:clip000".format(song_index, take),
                          "song": song, "labels": labels.tolist()})
    path.write_text(json.dumps({"plans": plans}))
    return path


def _run(args):
    return subprocess.run([sys.executable, str(REPO / "tools/probe_structure_conditioning.py"),
                           *args], capture_output=True, text=True, timeout=600)


def test_report_says_when_no_control_was_supplied(tmp_path):
    songs = ["s{}".format(i) for i in range(12)]
    plans = _plans(tmp_path / "plans.json", songs, 1, [3 + i % 5 for i in range(12)])
    out = tmp_path / "report.json"
    result = _run(["--plans", str(plans), "--output", str(out)])
    assert result.returncode == 0, result.stderr[-800:]
    report = json.loads(out.read_text())
    control = report["positive_control"]
    assert control.get("supplied") is False
    # The number has to travel with the artifact, not live in a worklog.
    assert "2.0e-12" in control["reading"]
    assert "make_beat_grid_control" in control["reading"]


def test_control_is_scored_with_the_same_statistic(tmp_path):
    songs = ["s{}".format(i) for i in range(12)]
    plans = _plans(tmp_path / "plans.json", songs, 1, [3 + i % 5 for i in range(12)])
    control = _plans(tmp_path / "control.json", songs, 2, [3 + i % 5 for i in range(12)])
    out = tmp_path / "report.json"
    result = _run(["--plans", str(plans), "--control-plans", str(control),
                   "--output", str(out)])
    assert result.returncode == 0, result.stderr[-800:]
    report = json.loads(out.read_text())
    control_block = report["positive_control"]
    assert "structure" in control_block
    assert control_block["explained_by_tempo"] in (True, False, None)
    # Same statistic on both arms: whatever the main arm reports, the control
    # reports the same keys, or the comparison is between two different rulers.
    assert set(control_block["structure"]) >= {"checked"}


def test_beat_grid_control_carries_no_transition(tmp_path):
    """The control must always be dancing.

    A control that also emitted transitions would differ from the arm in two
    ways at once, and the comparison would not say which one moved the p-value.
    """
    from tools.make_beat_grid_control import build

    root = tmp_path / "release"
    (root / "train").mkdir(parents=True)
    frames, windows = 150, 4
    music = np.zeros((windows, frames, 35), dtype=np.float32)
    music[:, ::15, 34] = 1.0
    np.save(root / "train" / "music.npy", music)
    (root / "train" / "names.json").write_text(json.dumps(
        ["w:{}:clip000_slice0".format(i) for i in range(windows)]))
    (root / "build.json").write_text(json.dumps({"window_policy": {"num_classes": 40}}))
    like = tmp_path / "like.json"
    like.write_text(json.dumps({"plans": [
        {"sequence": "w:{}:clip000".format(i), "song": "s{}".format(i),
         "labels": [0] * frames} for i in range(windows)]}))

    report = build(root, "train", like, seed=5)
    assert len(report["plans"]) == windows
    for plan in report["plans"]:
        labels = np.asarray(plan["labels"])
        assert labels.min() > 0, "the control must never emit the transition token"
        assert len(set(labels.tolist())) > 1, "a beat grid must cut more than once"
