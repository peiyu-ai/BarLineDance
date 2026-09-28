"""Tests for the visual gallery's member selection and its index page.

The rendering itself needs video files and is exercised by running the tool;
what is tested here is the part that decides *what gets shown*, because a
gallery that quietly shows six consecutive segments of one clip would make any
vocabulary look coherent.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.build_atomic_gallery import spread_over_uploads, write_index  # noqa: E402


def member(upload, index):
    return {"recording_id": "{}:clip{:03d}".format(upload, index), "upload": upload,
            "start": index * 30, "end": index * 30 + 30, "frame_ids_path": "f.npy"}


def test_members_are_drawn_from_as_many_uploads_as_possible():
    entries = ([member("a", i) for i in range(10)] + [member("b", 0)] + [member("c", 0)])
    picked = spread_over_uploads(entries, 3)
    assert sorted(entry["upload"] for entry in picked) == ["a", "b", "c"]


def test_one_upload_can_still_fill_the_sheet_when_it_is_all_there_is():
    entries = [member("a", i) for i in range(5)]
    picked = spread_over_uploads(entries, 3)
    assert len(picked) == 3
    assert {entry["upload"] for entry in picked} == {"a"}


def test_never_returns_more_members_than_asked_for():
    entries = [member(chr(ord("a") + i), 0) for i in range(9)]
    assert len(spread_over_uploads(entries, 4)) == 4


def test_asking_for_more_than_exists_returns_everything_once():
    entries = [member("a", 0), member("b", 0)]
    picked = spread_over_uploads(entries, 10)
    assert len(picked) == 2
    keys = {(entry["upload"], entry["start"]) for entry in picked}
    assert len(keys) == 2


def test_selection_is_deterministic():
    entries = [member("a", i) for i in range(4)] + [member("b", i) for i in range(4)]
    assert spread_over_uploads(entries, 5) == spread_over_uploads(entries, 5)


def test_index_flags_a_subprototype_that_is_really_one_clip(tmp_path):
    """A sub-prototype drawn from one upload is memorisation, not a prototype."""
    manifest = [
        {"prototype": 1, "subprototype": 0, "label": 1, "segments": 30, "uploads": 12,
         "upload_share_of_largest": 0.2, "tag": "side steps", "sheet": "sheets/a.jpg"},
        {"prototype": 1, "subprototype": 1, "label": 2, "segments": 30, "uploads": 2,
         "upload_share_of_largest": 0.9, "tag": "odd", "sheet": "sheets/b.jpg"},
    ]
    write_index(tmp_path, manifest, pathlib.Path("bundle_name"))
    page = (tmp_path / "index.html").read_text()
    assert "90% from one upload" in page
    assert page.count("<figure>") == 2
    assert "side steps" in page


def test_index_groups_sheets_under_their_prototype(tmp_path):
    manifest = [
        {"prototype": 2, "subprototype": 0, "label": 3, "segments": 10, "uploads": 5,
         "upload_share_of_largest": 0.2, "tag": "", "sheet": "sheets/c.jpg"},
        {"prototype": 1, "subprototype": 0, "label": 1, "segments": 10, "uploads": 5,
         "upload_share_of_largest": 0.2, "tag": "", "sheet": "sheets/a.jpg"},
    ]
    write_index(tmp_path, manifest, pathlib.Path("b"))
    page = (tmp_path / "index.html").read_text()
    assert page.index("prototype 1") < page.index("prototype 2")
    assert "(no tag)" in page


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
