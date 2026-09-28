"""The uploader-account map, restricted to one bundle's uploads.

The property under test is almost entirely the *refusal*.  Everything else this
tool does is a dict comprehension; what earns it a place in ``tools/`` is that
the failure it guards against is silent everywhere downstream:
``load_group_keys`` drops falsy values and ``genre_of`` returns ``None`` on a
miss, so an upload with no recorded account joins a nameless group and the
pre-split still runs, still reports cells, still writes a vocabulary.  That is
CLAUDE.md §2's gate-that-cannot-fire, so the test that matters is the one that
proves this gate fires.
"""

import json
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import derive_group_keys as deriver


def _bundle(monkeypatch, recording_ids):
    rows = [{"recording_id": name} for name in recording_ids]
    monkeypatch.setattr(deriver.asset_io, "read_jsonl", lambda key: iter(rows))


def _source(root, mapping):
    path = pathlib.Path(root) / "source.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return path


def test_an_upload_with_no_account_is_refused_not_left_nameless(monkeypatch):
    _bundle(monkeypatch, ["wild_v5:u0:clip000", "wild_v5:u1:clip000"])
    with tempfile.TemporaryDirectory() as raw:
        source = _source(raw, {"u0": "acct-a"})
        with pytest.raises(SystemExit) as excinfo:
            deriver.derive("data/wild3d/b_performance", source)
    # The message has to name the missing upload: the answer lives in the
    # ingest metadata, and "1 of 2 uploads" alone does not say which one.
    assert "u1" in str(excinfo.value)


def test_a_falsy_account_counts_as_missing(monkeypatch):
    # load_group_keys drops "" as well, so accepting it here would hand the
    # downstream lookup exactly the hole this gate exists to catch.
    _bundle(monkeypatch, ["wild_v5:u0:clip000"])
    with tempfile.TemporaryDirectory() as raw:
        source = _source(raw, {"u0": ""})
        with pytest.raises(SystemExit):
            deriver.derive("data/wild3d/b_performance", source)


def test_the_generation_prefix_is_stripped_before_the_lookup(monkeypatch):
    # The map is keyed by bare upload id; the same upload appears as
    # "wild_v4:u0:clip000" in one generation and "wild_v5:u0:clip000" in the
    # next, and neither of those is a key.
    _bundle(monkeypatch, ["wild_v5:u0:clip000", "wild_v4:u0:clip007"])
    with tempfile.TemporaryDirectory() as raw:
        source = _source(raw, {"u0": "acct-a"})
        result = deriver.derive("data/wild3d/b_performance", source)
    assert result["mapping"] == {"u0": "acct-a"}
    assert result["report"]["uploads"] == 1
    assert result["report"]["clips"] == 2


def test_the_cache_directory_shape_resolves_to_the_same_upload(monkeypatch):
    _bundle(monkeypatch, ["u0__clip000", "wild_v5:u0:clip001"])
    with tempfile.TemporaryDirectory() as raw:
        source = _source(raw, {"u0": "acct-a"})
        result = deriver.derive("data/wild3d/b_performance", source)
    assert result["report"]["uploads"] == 1


def test_uploads_this_generation_dropped_do_not_enter_the_output(monkeypatch):
    # The v4 map carries 1,607 uploads wild_v5 no longer holds.  Copying the
    # file under the v5 name would leave every one of them readable to a v5
    # consumer, which is the same class of mistake as globbing a prefix.
    _bundle(monkeypatch, ["wild_v5:u0:clip000"])
    with tempfile.TemporaryDirectory() as raw:
        source = _source(raw, {"u0": "acct-a", "u9": "acct-b", "u8": "acct-b"})
        result = deriver.derive("data/wild3d/b_performance", source)
    assert result["mapping"] == {"u0": "acct-a"}
    assert result["report"]["source_uploads_not_in_this_bundle"] == 2


def test_the_report_names_the_largest_account(monkeypatch):
    # Reported next to the count on purpose: on this corpus one account holds
    # 23.7% of the uploads, so "25 accounts" alone reads as a balanced
    # pre-split that does not exist.
    _bundle(monkeypatch, ["wild_v5:u{}:clip000".format(i) for i in range(5)])
    with tempfile.TemporaryDirectory() as raw:
        source = _source(raw, {"u0": "big", "u1": "big", "u2": "big",
                               "u3": "small", "u4": "other"})
        result = deriver.derive("data/wild3d/b_performance", source)
    assert result["report"]["accounts"] == 3
    assert result["report"]["largest_account"] == "big"
    assert result["report"]["largest_account_uploads"] == 3
