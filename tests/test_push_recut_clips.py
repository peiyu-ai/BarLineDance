"""The re-cut publisher, exercised where the irreversible part can go wrong.

Two things here are worth a test rather than a read-through.  The selection
decides what an undeletable store gets written to, and the one name it must
never spend a write on -- an orphan, a clip the re-cut stopped producing -- is
handed back by every enumeration forever, so dropping it has to be a property
of the code rather than of the caller's input file.  And ``verdict`` is the
gate: it has to separate "already pushed" from "pushed and still wrong", which
are the same shape (a comparison that came back equal, or did not) read in
opposite directions.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import redo_manifest  # noqa: E402
from tools.push_recut_clips import select_stems, verdict  # noqa: E402


def _freshness(tmp_path, stale_3d):
    path = tmp_path / "freshness.json"
    path.write_text(json.dumps({
        "generated_by": "tools/audit_clip_freshness.py",
        "stages": 2,
        "stale": {"3d": list(stale_3d), "s3d": list(stale_3d)},
        "missing_derivative": {"3d": [], "s3d": []},
    }), encoding="utf-8")
    return path


def _row(**files):
    return {"clip": "c", "files": {name: {"local": local, "remote": remote}
                                   for name, (local, remote) in files.items()}}


def test_orphans_are_dropped_by_name(tmp_path):
    manifest = _freshness(tmp_path, ["a__clip000", "b__clip001", "c__clip002"])
    orphans = tmp_path / "orphans.txt"
    orphans.write_text("b__clip001\nz__clip999\n", encoding="utf-8")

    selection = select_stems(manifest, "3d", orphans)

    assert selection["stems"] == ["a__clip000", "c__clip002"]
    assert selection["orphans_dropped"] == ["b__clip001"]
    assert selection["named"] == 3


def test_no_orphan_file_keeps_everything(tmp_path):
    manifest = _freshness(tmp_path, ["a__clip000"])
    selection = select_stems(manifest, "3d", None)
    assert selection["stems"] == ["a__clip000"]
    assert selection["orphans_dropped"] == []


def test_manifest_that_does_not_describe_the_stage_raises(tmp_path):
    """An empty push is a real answer; a mis-typed path must not look like one."""
    path = tmp_path / "census.json"
    path.write_text(json.dumps({"clips": ["a__clip000"]}), encoding="utf-8")
    # A census names clips for every stage, so it loads; a freshness audit that
    # carries no such stage is the one that must raise.
    broken = tmp_path / "partial.json"
    broken.write_text(json.dumps({"stale": {"s3d": []}}), encoding="utf-8")
    with pytest.raises(redo_manifest.ManifestError):
        select_stems(broken, "3d", None)


def test_verdict_separates_the_four_states():
    assert verdict(_row(**{"clip.mp4": ("aa", "bb")})) == "differs"
    assert verdict(_row(**{"clip.mp4": ("aa", "aa")})) == "already_current"
    assert verdict(_row(**{"clip.mp4": ("aa", None)})) == "missing_remote"
    assert verdict(_row(**{"clip.mp4": (None, "bb")})) == "no_local"


def test_one_matching_file_does_not_make_a_clip_current():
    """The offset defect lives in audio.wav, the speed defect in clip.mp4.

    A clip whose video happened to be re-published while its audio was not is
    exactly the half-pushed state this tool exists to end, so agreement has to
    be required of every verified file, not any of them.
    """
    row = _row(**{"clip.mp4": ("aa", "aa"), "audio.wav": ("bb", "cc")})
    assert verdict(row) == "differs"
