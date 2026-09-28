"""The ceiling measurement has to be right before anything is read against it.

Every test here pins a way the number could be silently wrong rather than
absent: a sign error in the lag would deflate the ceiling and make a model look
closer to it; a silently empty pair list would report a ceiling of nothing; a
coarse label map that is not applied would compare 820-class agreement against
a 53-class probe.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.same_song_agreement import measure  # noqa: E402


def _labels_dir(tmp_path, timelines):
    root = tmp_path / "labels_bundle"
    (root / "labels").mkdir(parents=True)
    rows = []
    for name, values in timelines.items():
        rel = "labels/{}/labels.npy".format(name.replace(":", "_"))
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(root / rel, np.asarray(values, dtype=np.int64))
        rows.append({"sequence_id": name, "labels_path": rel})
    (root / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return root


def _pairs(tmp_path, entries):
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return path


def test_identical_timelines_read_one(tmp_path):
    line = ([1] * 40 + [0] * 40 + [2] * 40) * 3
    labels = _labels_dir(tmp_path, {"w:a:clip000": line, "w:b:clip000": line})
    pairs = _pairs(tmp_path, [{"left": "w:a:clip000", "right": "w:b:clip000", "lag_frames": 0}])
    report = measure(labels, pairs, control_pairs=5)
    assert report["same_song"]["all_frames"] == pytest.approx(1.0)
    assert report["same_song"]["atomic_frames"] == pytest.approx(1.0)


def test_lag_sign_is_honoured(tmp_path):
    """A shifted copy scores 1.0 at its own lag and much less at lag 0.

    This is the test the ceiling most needs: reading the lag backwards would
    still produce a plausible number, just a smaller one, and a smaller ceiling
    makes every model look better against it.
    """
    base = list(np.repeat(np.arange(1, 13), 20))
    shift = 20
    shifted = base[shift:] + base[:shift]
    labels = _labels_dir(tmp_path, {"w:a:clip000": base, "w:b:clip000": shifted})
    at_lag = measure(labels, _pairs(tmp_path, [
        {"left": "w:a:clip000", "right": "w:b:clip000", "lag_frames": shift}]),
        control_pairs=5)
    at_zero = measure(labels, _pairs(tmp_path, [
        {"left": "w:a:clip000", "right": "w:b:clip000", "lag_frames": 0}]),
        control_pairs=5)
    assert at_lag["same_song"]["all_frames"] == pytest.approx(1.0)
    assert at_zero["same_song"]["all_frames"] < 0.2


def test_short_overlap_is_dropped_and_counted(tmp_path):
    long_line = [1] * 200
    labels = _labels_dir(tmp_path, {"w:a:clip000": long_line, "w:b:clip000": [1] * 200})
    pairs = _pairs(tmp_path, [
        {"left": "w:a:clip000", "right": "w:b:clip000", "lag_frames": 180},
        {"left": "w:a:clip000", "right": "w:b:clip000", "lag_frames": 0},
    ])
    report = measure(labels, pairs, control_pairs=5)
    assert report["dropped_short_overlap"] == 1
    assert report["same_song"]["pairs"] == 1


def test_pair_list_naming_nothing_refuses(tmp_path):
    labels = _labels_dir(tmp_path, {"w:a:clip000": [1] * 200})
    pairs = _pairs(tmp_path, [{"left": "other:1:clip000", "right": "other:2:clip000",
                               "lag_frames": 0}])
    with pytest.raises(SystemExit):
        measure(labels, pairs, control_pairs=5)


def test_label_map_is_applied(tmp_path):
    """Coarsening two disagreeing labels into one group must raise agreement.

    Without this the probe could be scored at 53 classes against a ceiling
    measured at 820 -- the exact mismatch that would make a coarse vocabulary
    look worse than it is.
    """
    a = [1] * 120
    b = [2] * 120
    labels = _labels_dir(tmp_path, {"w:a:clip000": a, "w:b:clip000": b})
    pairs = _pairs(tmp_path, [{"left": "w:a:clip000", "right": "w:b:clip000", "lag_frames": 0}])
    fine = measure(labels, pairs, control_pairs=5)
    coarse = measure(labels, pairs, label_map=np.array([0, 1, 1]), control_pairs=5)
    assert fine["same_song"]["atomic_frames"] == pytest.approx(0.0)
    assert coarse["same_song"]["atomic_frames"] == pytest.approx(1.0)


def test_majority_rule_is_off_by_default():
    """The withdrawn gate must not fire unless a caller asks for it.

    Asserted on the parser default rather than on a full probe run: the defect
    was a *default*, and a test that only exercised an explicit value would
    pass while the default kept failing every corpus.
    """
    source = (Path(__file__).resolve().parents[1] / "tools/probe_label_predictability.py").read_text()
    block = source.split('"--min-accuracy-over-majority"', 1)[1].split(")", 1)[0]
    assert "default=None" in block, "the majority rule must default to not-enforced"
    assert "WITHDRAWN" in block


def test_torch_label_map_is_accepted(tmp_path):
    """The probe hands this a torch tensor; numpy-only indexing broke on it.

    Found by the first end-to-end run after the rewrite, not by the unit tests
    above, which all passed numpy arrays -- so the type the real caller uses is
    now pinned.
    """
    torch = pytest.importorskip("torch")
    labels = _labels_dir(tmp_path, {"w:a:clip000": [1] * 120, "w:b:clip000": [2] * 120})
    pairs = _pairs(tmp_path, [{"left": "w:a:clip000", "right": "w:b:clip000", "lag_frames": 0}])
    report = measure(labels, pairs, label_map=torch.tensor([0, 1, 1]), control_pairs=5)
    assert report["same_song"]["atomic_frames"] == pytest.approx(1.0)
