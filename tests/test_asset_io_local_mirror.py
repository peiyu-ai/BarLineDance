"""Keeping a parked read copy in step with what was just published.

``read_bytes``, ``load_npy`` and ``fetch_dir`` all resolve local-first, and
three of this repo's asset trees are parked under ``/cache`` and symlinked back
into the checkout.  So until 2026-08-20 a stage that re-derived a clip and
published it left this pod reading the *previous* generation: stage B
re-extracted 317 re-cut clips, the store's ``metadata.json`` recorded the new
video hash for 10 of 10 sampled, the parked copy recorded the old one for 10 of
10 -- and ``audit_clip_freshness`` reported that nothing had changed.

The second test is the one that keeps this from becoming a different bug.  The
rule in CLAUDE.md §1.2 is that the checkout holds code, not corpora; a hook
that *created* local copies on every publish would put the 221 GB raw tree on
this pod one stage at a time.  Refreshing what is already there restores an
invariant; creating one breaks a different rule.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import asset_io  # noqa: E402


def _publish_source(tmp_path, **files):
    source = tmp_path / "scratch"
    source.mkdir()
    for name, text in files.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return source


def test_an_existing_parked_copy_is_updated(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_io, "REPO_ROOT", tmp_path)
    parked = tmp_path / "data" / "tree" / "clip000"
    parked.mkdir(parents=True)
    (parked / "metadata.json").write_text("old", encoding="utf-8")

    source = _publish_source(tmp_path, **{"metadata.json": "new"})
    updated = asset_io.refresh_local_mirror(source, "data/tree/clip000")

    assert updated == 1
    assert (parked / "metadata.json").read_text(encoding="utf-8") == "new"


def test_nothing_is_created_where_no_copy_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_io, "REPO_ROOT", tmp_path)
    source = _publish_source(tmp_path, **{"metadata.json": "new"})

    updated = asset_io.refresh_local_mirror(source, "data/tree/clip000")

    assert updated == 0
    assert not (tmp_path / "data" / "tree" / "clip000").exists()


def test_nested_files_reach_the_copy(tmp_path, monkeypatch):
    """A clip directory is not flat -- ``preprocess/bbx.pt`` sits under it."""
    monkeypatch.setattr(asset_io, "REPO_ROOT", tmp_path)
    parked = tmp_path / "data" / "tree" / "clip000"
    (parked / "preprocess").mkdir(parents=True)
    (parked / "preprocess" / "bbx.pt").write_text("old", encoding="utf-8")

    source = _publish_source(tmp_path, **{"preprocess/bbx.pt": "new"})
    assert asset_io.refresh_local_mirror(source, "data/tree/clip000") == 1
    assert (parked / "preprocess" / "bbx.pt").read_text(encoding="utf-8") == "new"


def test_write_bytes_updates_an_existing_parked_manifest(tmp_path, monkeypatch):
    """The single-object path needs the same invariant as the directory one.

    ``runs/`` is a parked tree here, so stage C could merge a corrected
    17,015-row inventory to the store and have ``staging`` -- reading
    local-first -- rebuild from the 17,790-row copy beside it and report the
    old counts as a success.
    """
    monkeypatch.setattr(asset_io, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(asset_io, "_ossutil", lambda *a, **k: None)
    parked = tmp_path / "runs" / "inventory.jsonl"
    parked.parent.mkdir(parents=True)
    parked.write_bytes(b"old\n")

    asset_io.write_bytes("runs/inventory.jsonl", b"new\n")

    assert parked.read_bytes() == b"new\n"


def test_write_bytes_creates_no_parked_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_io, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(asset_io, "_ossutil", lambda *a, **k: None)

    asset_io.write_bytes("runs/inventory.jsonl", b"new\n")

    assert not (tmp_path / "runs" / "inventory.jsonl").exists()


def test_skipped_patterns_are_not_mirrored(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_io, "REPO_ROOT", tmp_path)
    parked = tmp_path / "data" / "tree" / "clip000"
    parked.mkdir(parents=True)

    source = _publish_source(tmp_path, **{"a.pyc": "x", "metadata.json": "new"})
    assert asset_io.refresh_local_mirror(source, "data/tree/clip000") == 1
    assert not (parked / "a.pyc").exists()


def test_publishing_out_of_the_mirror_itself_is_a_no_op_not_a_crash():
    """``data/`` trees are symlinks into /cache on this pod.

    So a publisher handed a directory inside the ingest tree has source and
    destination as the same file, and ``shutil.copyfile`` raises SameFileError.
    That aborted the 2026-08-25 re-cut publish after one clip.  There is nothing
    to refresh in that case -- the mirror already holds the bytes that were just
    published -- so it has to be a skip, and a skip that does not count as work.
    """
    import pathlib
    import tempfile
    from unittest import mock

    from tools import asset_io

    with tempfile.TemporaryDirectory() as raw:
        root = pathlib.Path(raw)
        tree = root / "data" / "tree" / "clip000"
        tree.mkdir(parents=True)
        (tree / "clip.mp4").write_bytes(b"bytes")
        with mock.patch.object(asset_io, "REPO_ROOT", root):
            updated = asset_io.refresh_local_mirror(tree, "data/tree/clip000", ())
        assert updated == 0
        assert (tree / "clip.mp4").read_bytes() == b"bytes"
