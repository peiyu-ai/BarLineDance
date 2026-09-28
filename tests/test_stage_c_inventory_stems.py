"""Where stage C's clip set comes from, now that it is not a prefix listing.

The defect this pins is not hypothetical and not cheap: the published
``runs/wild_v4_inventory.jsonl`` holds 17,790 records because the stage derived
its clip set by enumerating ``data/wild_ingest_v1`` -- and 775 of those stems
are fps re-cut orphans, names the store can never delete and no consumer may
read.  The listing kept handing them back and the stage kept inventorying them.

So the tests are about the two ways a clip set can be wrong while looking
right: a name that should not be there (orphan, still in the listing), and a
name that should be there and is not (the manifest asked for it, the prefix
cannot serve it -- which a filter would turn into a quietly smaller corpus).
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.run_wild_stage_c_oss import inventory_stems  # noqa: E402


def _contents(*stems, without_meta=()):
    return {stem: ({"keypoints.npy"} if stem in without_meta
                   else {"meta.json", "keypoints.npy"})
            for stem in stems}


def _manifest(tmp_path, stems):
    path = tmp_path / "clips.txt"
    path.write_text("\n".join(stems) + "\n", encoding="utf-8")
    return path


def test_a_manifest_excludes_an_orphan_the_listing_still_returns(tmp_path):
    contents = _contents("a__clip000", "b__clip001", "orphan__clip009")
    manifest = _manifest(tmp_path, ["a__clip000", "b__clip001"])

    assert inventory_stems(contents, manifest, False) == ["a__clip000", "b__clip001"]


def test_the_prefix_path_still_returns_the_orphan(tmp_path):
    """The escape hatch is not a fixed version of the same thing."""
    contents = _contents("a__clip000", "orphan__clip009")
    assert inventory_stems(contents, None, True) == ["a__clip000", "orphan__clip009"]


def test_no_manifest_and_no_escape_hatch_is_refused(tmp_path):
    with pytest.raises(SystemExit) as raised:
        inventory_stems(_contents("a__clip000"), None, False)
    assert "--clips" in str(raised.value)


def test_a_named_stem_the_prefix_cannot_serve_is_an_error(tmp_path):
    """Not a filter: the caller believes it named work that will happen."""
    contents = _contents("a__clip000")
    manifest = _manifest(tmp_path, ["a__clip000", "gone__clip000"])
    with pytest.raises(SystemExit) as raised:
        inventory_stems(contents, manifest, False)
    assert "gone__clip000" in str(raised.value)


def test_a_stem_present_but_without_meta_json_counts_as_unservable(tmp_path):
    """``inventory_wild_cache`` needs meta.json; a directory alone is not a clip."""
    contents = _contents("a__clip000", "b__clip001", without_meta=("b__clip001",))
    manifest = _manifest(tmp_path, ["a__clip000", "b__clip001"])
    with pytest.raises(SystemExit) as raised:
        inventory_stems(contents, manifest, False)
    assert "b__clip001" in str(raised.value)


def test_a_census_manifest_is_read_through_the_shared_reader(tmp_path):
    path = tmp_path / "census.json"
    path.write_text(json.dumps({"clips": ["b__clip001", "a__clip000"]}), encoding="utf-8")
    contents = _contents("a__clip000", "b__clip001", "orphan__clip009")
    assert inventory_stems(contents, path, False) == ["a__clip000", "b__clip001"]
