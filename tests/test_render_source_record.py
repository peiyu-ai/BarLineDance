"""Every rendered video must say which arm it came from.

The cost of not doing this is recorded in ``write_source_record``'s docstring:
the operator labelled two rendered clips -- the only kind of judgement that
counts in this repository -- and the labelling could not be used, because four
arms had plausible timestamps and the two most likely ones disagree about which
clip is better.
"""
import json
import pathlib
import pickle

import pytest

from tools.render_avatar_video import write_source_record

STATS = {"rows": 2, "frames": 300, "style": "realistic", "body": "smpl", "floor": 0.34}


def make_arm(tmp_path, name, sampling):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "clip.pkl").write_bytes(pickle.dumps({"full_pose": [1, 2, 3]}))
    (directory / "manifest.json").write_text(json.dumps({
        "output_dir": "/somewhere/" + name,
        "planner_checkpoint": "/ckpt/planner_" + name + ".pt",
        "completion_checkpoint": "/ckpt/completion.pt",
        "dataset_provenance": {"data_root": "/release/" + name},
        "sampling": sampling,
    }))
    return directory


def test_the_record_names_the_arm_behind_each_panel(tmp_path):
    a = make_arm(tmp_path, "armA", {"draft_beat_anchor": 1.2})
    b = make_arm(tmp_path, "armB", {"draft_beat_anchor": 0.0})
    out = tmp_path / "compare.mp4"
    sidecar = write_source_record(out, [("A", str(a / "clip.pkl")),
                                        ("B", str(b / "clip.pkl"))], STATS)
    record = json.loads(pathlib.Path(sidecar).read_text())
    assert [r["title"] for r in record["rows"]] == ["A", "B"]
    anchors = [r["arm"]["sampling"]["draft_beat_anchor"] for r in record["rows"]]
    assert anchors == [1.2, 0.0], "the value that NAMES the arm has to survive"
    assert record["rows"][0]["arm"]["planner_checkpoint"].endswith("armA.pt")


def test_two_arms_that_differ_only_in_one_flag_are_distinguishable(tmp_path):
    """The failure being prevented: two renders that look alike on disk and
    cannot be told apart afterwards."""
    a = make_arm(tmp_path, "a", {"draft_beat_anchor": 1.2, "seed": 7})
    b = make_arm(tmp_path, "b", {"draft_beat_anchor": 1.6, "seed": 7})
    ra = json.loads(pathlib.Path(
        write_source_record(tmp_path / "a.mp4", [("x", str(a / "clip.pkl"))], STATS)).read_text())
    rb = json.loads(pathlib.Path(
        write_source_record(tmp_path / "b.mp4", [("x", str(b / "clip.pkl"))], STATS)).read_text())
    assert ra["rows"][0]["arm"]["sampling"] != rb["rows"][0]["arm"]["sampling"]


def test_the_pickle_is_hashed_so_a_rebuilt_arm_is_not_mistaken_for_the_old_one(tmp_path):
    a = make_arm(tmp_path, "a", {})
    first = json.loads(pathlib.Path(
        write_source_record(tmp_path / "v1.mp4", [("x", str(a / "clip.pkl"))], STATS)).read_text())
    (a / "clip.pkl").write_bytes(pickle.dumps({"full_pose": [9, 9, 9]}))
    second = json.loads(pathlib.Path(
        write_source_record(tmp_path / "v2.mp4", [("x", str(a / "clip.pkl"))], STATS)).read_text())
    assert first["rows"][0]["sha256"] != second["rows"][0]["sha256"]


def test_ground_truth_panels_have_no_manifest_and_that_is_not_an_error(tmp_path):
    truth = tmp_path / "gt"
    truth.mkdir()
    (truth / "clip.pkl").write_bytes(pickle.dumps({"full_pose": [0]}))
    record = json.loads(pathlib.Path(
        write_source_record(tmp_path / "o.mp4", [("gt", str(truth / "clip.pkl"))],
                            STATS)).read_text())
    assert "arm" not in record["rows"][0]
    assert record["rows"][0]["sha256"]


def test_a_missing_source_still_writes_a_record(tmp_path):
    """A record that refuses to exist when one panel is missing would leave the
    other panels unattributed too."""
    record = json.loads(pathlib.Path(
        write_source_record(tmp_path / "o.mp4", [("x", str(tmp_path / "nope.pkl"))],
                            STATS)).read_text())
    assert record["rows"][0]["path"].endswith("nope.pkl")
    assert "sha256" not in record["rows"][0]


def test_a_directory_source_is_recorded_not_a_crash(tmp_path):
    """Ground truth is always a directory (its eval pickle holds joints and no
    SMPL parameters).  The first version opened the source as a file, crashed
    on that row, and under `set -e` took the whole render driver down AFTER the
    video had been written -- one clip rendered out of ten."""
    import numpy as np
    truth = tmp_path / "gt_dir"
    truth.mkdir()
    np.save(truth / "atomic_motion_151.npy", np.zeros((4, 151), dtype=np.float32))
    record = json.loads(pathlib.Path(
        write_source_record(tmp_path / "o.mp4", [("ground truth", str(truth))],
                            STATS)).read_text())
    row = record["rows"][0]
    assert row["kind"] == "converted-151d-directory"
    assert row["sha256"] and row["hashed"].endswith("atomic_motion_151.npy")
