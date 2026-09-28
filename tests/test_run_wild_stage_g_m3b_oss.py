"""The pairing ``--bundle-tag`` creates, and the two ways it used to go wrong.

M3b reads six things.  Four of them belong to the run (M2's labels, the
embedding cache, the captions, every output) and two belong to whichever tag
published the ingest (the account map, the stage-E performance bundle).  Before
2026-08-21 one tag had to own all six, which for clean5b5 means a 404: it
published labels and captions but no stage-E tree.

Both tests below are about the *silent* half, because the 404 announces itself:

* ``recluster_atomics_ingroup.build`` skips a label row it cannot resolve in the
  bundle instead of failing, so the wrong bundle re-clusters the intersection
  and reports a clean run over it.  The gate reads the bundle's manifest and
  refuses first.
* the staging root is shared with M3a and across tags, and ``fetch_tree_once``
  returns a populated directory untouched, so an unqualified ``labels/`` was
  whichever tag staged there first.  Measured on the pod that day: it held
  clean5b4's tree beside ``clean5b4_tmr_embeddings.npz``, and the cache name
  carried the tag while the tree's did not.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools import run_wild_stage_g_m3a_oss as m3a              # noqa: E402
from tools import run_wild_stage_g_m3b_oss as m3b              # noqa: E402


def _labels(tmp_path, recordings):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    with (labels_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for name in recordings:
            handle.write(json.dumps({"recording_id": name,
                                     "labels_path": "x.npy"}) + "\n")
    np.save(labels_dir / "x.npy", np.zeros(8, dtype=np.int64))
    return labels_dir


def _bundle(monkeypatch, recordings):
    rows = [{"recording_id": name, "motion_path": "m.npy"} for name in recordings]
    monkeypatch.setattr(m3b.asset_io, "read_jsonl", lambda key: iter(rows))


def test_only_the_account_map_follows_the_bundle_tag(tmp_path):
    """Outputs written under the tag they borrowed inputs from would be
    unrecoverable: this store overwrites but never deletes."""
    keys = m3b.m3b_keys("clean5b5", "wild_v4")

    assert keys["group_keys"] == "runs/wild_v4_group_keys.json"
    for name in ("subprototypes", "cells", "bundle", "report"):
        assert "clean5b5" in keys[name] and "wild_v4" not in keys[name]
    assert m3b.m3b_keys("clean5b5")["group_keys"] == "runs/clean5b5_group_keys.json"


def test_the_gate_resolves_the_other_id_shape_rather_than_failing_on_it(
        tmp_path, monkeypatch, capsys):
    """``<video>__clipNNN`` in a label tree against ``corpus:video:clipNNN`` in
    the bundle is the repo's own aliasing, and ``build`` resolves it -- so a
    gate that did not would refuse a pairing that works."""
    labels_dir = _labels(tmp_path, ["7029295350347287812__clip000"])
    _bundle(monkeypatch, ["wild_v4:7029295350347287812:clip000"])

    assert m3b.check_bundle_covers_labels(labels_dir, "wild_v4") == 1
    assert "1 of 1" in capsys.readouterr().out


def test_a_bundle_missing_recordings_is_refused_with_the_count(
        tmp_path, monkeypatch):
    """The failure that otherwise publishes: two of three rows resolve, and
    ``build`` would cluster those two and report a clean run."""
    labels_dir = _labels(tmp_path, ["wild_v4:1:clip000", "wild_v4:2:clip000",
                                    "wild_v4:3:clip000"])
    _bundle(monkeypatch, ["wild_v4:1:clip000", "wild_v4:2:clip000"])

    with pytest.raises(SystemExit) as failure:
        m3b.check_bundle_covers_labels(labels_dir, "wild_v4")

    message = str(failure.value)
    assert "1 of 3" in message and "wild_v4:3:clip000" in message


def test_a_tag_with_no_stage_e_tree_is_named_not_stacktraced(
        tmp_path, monkeypatch):
    """``data/wild3d/clean5b5_performance/`` holds 0 objects, and the store
    reports that as a transfer error several frames down."""
    labels_dir = _labels(tmp_path, ["wild_v4:1:clip000"])

    def explode(key):
        raise RuntimeError("ossutil: object does not exist")

    monkeypatch.setattr(m3b.asset_io, "read_jsonl", explode)

    with pytest.raises(SystemExit) as failure:
        m3b.check_bundle_covers_labels(labels_dir, "clean5b5")

    assert "clean5b5" in str(failure.value) and "--bundle-tag" in str(failure.value)


def test_two_tags_staged_into_one_root_do_not_share_a_label_tree(
        tmp_path, monkeypatch):
    """The trap that was live on the pod: ``fetch_tree_once`` leaves a populated
    directory alone, so an unqualified name is the first tag that ran here."""
    fetched = {}

    def fake_fetch(key, dest):
        fetched[key] = dest
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    monkeypatch.setattr(m3a, "fetch_tree_once", fake_fetch)
    monkeypatch.setattr(m3a.asset_io, "read_bytes", lambda key: b"")

    first = m3a.stage_segmentation("clean5b4", tmp_path)
    second = m3a.stage_segmentation("clean5b5", tmp_path)

    assert first["labels"] != second["labels"]
    assert "clean5b4" in first["labels"].name and "clean5b5" in second["labels"].name
    assert first["cache"] != second["cache"]


def test_one_tag_refitted_does_not_reuse_the_previous_vocabulary(tmp_path, monkeypatch):
    """The 2026-08-26 case: same tag, refitted M2, republished to the same keys.

    Adding the tag to the staged name fixed the cross-tag trap above and not
    this one.  The union re-split forced M2 to refit under the unchanged tag
    ``wild_v5_song``; ``verify`` then read the pre-refit tree still sitting in
    /dev/shm -- 201,806 clustered spans against the store's 201,913, no error.
    Resuming M3a instead would have captioned seven cards against a superseded
    vocabulary, and the coverage gate would have passed, because coverage is
    computed against that same stale tree.
    """
    fetched = []

    def fake_fetch(key, dest):
        fetched.append(dest)
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    report = {"value": b'{"accepted_segments": 201806}'}
    monkeypatch.setattr(m3a, "fetch_tree_once", fake_fetch)
    monkeypatch.setattr(m3a.asset_io, "read_bytes",
                        lambda key: report["value"] if key.endswith("report.json") else b"")

    before = m3a.stage_segmentation("wild_v5_song", tmp_path)
    again = m3a.stage_segmentation("wild_v5_song", tmp_path)
    assert before["labels"] == again["labels"]          # unchanged store, same staging
    assert before["generation"] == again["generation"]

    report["value"] = b'{"accepted_segments": 201913}'  # M2 refit, same keys
    after = m3a.stage_segmentation("wild_v5_song", tmp_path)
    assert after["labels"] != before["labels"]
    assert after["cache"] != before["cache"]
    assert after["generation"] != before["generation"]
    # And it actually re-fetched rather than returning the populated directory.
    assert fetched[-1] == after["labels"]


def test_a_local_checkpoint_path_carries_the_generation(tmp_path):
    """The object keys were stamped and this local path was not.

    On 2026-08-22 that made the v2 summarising run resume from v1's checkpoint,
    compute nothing, and republish v1's cells under the v2 key -- byte for byte
    -- while every shard printed ``published 7 cell(s)`` and exited 0.
    """
    import argparse

    from tools.run_wild_stage_g_m3b_oss import cells_dir

    flat = argparse.Namespace(genre_split=False, generation="")
    stamped = argparse.Namespace(genre_split=False, generation="v2")
    presplit = argparse.Namespace(genre_split=True, generation="v2")

    assert cells_dir(flat) == "summarise_cells"
    assert cells_dir(stamped) == "summarise_cells_v2"
    assert cells_dir(presplit) == "summarise_cells_presplit_v2"


def test_a_checkpoint_built_from_other_captions_is_refused(tmp_path):
    """A name only protects against the mistakes someone thought of.

    This is the criterion behind it: what the cells were actually built from.
    """
    import pytest

    from tools.run_wild_stage_g_m3b_oss import guard_checkpoint_source

    captions = tmp_path / "captions.jsonl"
    captions.write_text('{"a": 1}\n', encoding="utf-8")
    checkpoint = tmp_path / "cells" / "shard-00.jsonl"
    checkpoint.parent.mkdir()

    # A fresh generation gets a fresh directory, so the first pass records.
    guard_checkpoint_source(checkpoint, captions)
    checkpoint.write_text('{"prototype": 1}\n', encoding="utf-8")
    # Same captions: resuming is what the checkpoint is for.
    guard_checkpoint_source(checkpoint, captions)

    other = tmp_path / "captions_v2.jsonl"
    other.write_text('{"a": 2}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        guard_checkpoint_source(checkpoint, other)
    assert "re-summarise" in str(raised.value)


def test_a_checkpoint_with_no_recorded_source_is_refused(tmp_path):
    """The dangerous case is a checkpoint whose provenance nobody wrote down.

    Refusing is the answer rather than adopting it, because adopting is exactly
    what happened on 2026-08-22: cells of unknown origin were resumed and
    republished as a new generation.
    """
    import pytest

    from tools.run_wild_stage_g_m3b_oss import guard_checkpoint_source

    captions = tmp_path / "captions.jsonl"
    captions.write_text('{"a": 1}\n', encoding="utf-8")
    legacy = tmp_path / "cells" / "shard-00.jsonl"
    legacy.parent.mkdir()
    legacy.write_text('{"prototype": 1}\n', encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        guard_checkpoint_source(legacy, captions)
    assert "unrecorded" in str(raised.value)


def test_an_empty_checkpoint_is_not_treated_as_provenance(tmp_path):
    """A shard that has not written a cell yet must not be refused."""
    from tools.run_wild_stage_g_m3b_oss import guard_checkpoint_source

    captions = tmp_path / "captions.jsonl"
    captions.write_text('{"a": 1}\n', encoding="utf-8")
    checkpoint = tmp_path / "cells" / "shard-00.jsonl"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    guard_checkpoint_source(checkpoint, captions)


def test_every_local_name_this_stage_writes_carries_the_generation():
    """One rule, checked once, instead of four call sites checked never.

    On 2026-08-22 the object keys were stamped with the caption generation and
    four local paths were not: the summarise checkpoint, the per-shard report,
    the assembled grouping and the recluster output tree.  The fourth one made
    ``recluster`` read v2's grouping while clustering v1's captions and emit 53
    sub-prototypes in place of 820, reporting ``ungrouped_segments: 0``.
    """
    import inspect

    from tools import run_wild_stage_g_m3b_oss as driver

    assert driver.stamped("summarise_cells", "") == "summarise_cells"
    assert driver.stamped("summarise_cells", "v2") == "summarise_cells_v2"
    assert driver.stamped("subprototypes_llm", "v2", ".json") == \
        "subprototypes_llm_v2.json"
    assert driver.stamped("subprototypes_llm", "", ".json") == \
        "subprototypes_llm.json"

    # And no local path bypasses it.  A bare ``root / "name"`` is how each of
    # the four got written in the first place.
    source = inspect.getsource(driver)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "stamped(" in line:
            continue
        assert 'root / "subprototypes' not in line, line
        assert 'root / "summarise_cells' not in line, line


def test_the_staged_grouping_carries_the_generation(tmp_path):
    """The local name M3c reads from, not only the object key it came from.

    Until 2026-08-27 this path was ``subprototypes_llm.json`` for every
    generation, and the ``not is_file()`` reuse guard then handed a v5union run
    the file a clean5b5 run had left there five days earlier.
    """
    import argparse

    from tools import run_wild_stage_g_m3b_oss as stage

    args = argparse.Namespace(root=str(tmp_path), genre_split=False,
                              generation="v5union")
    name = stage.stamped("subprototypes_llm", args.generation, ".json")
    assert name == "subprototypes_llm_v5union.json"
    assert stage.stamped("subprototypes_llm", "", ".json") == "subprototypes_llm.json"
    presplit = stage.stamped("subprototypes_llm_presplit", args.generation, ".json")
    assert presplit == "subprototypes_llm_presplit_v5union.json"
    # The two configurations and the two generations are four distinct names.
    assert len({name, presplit,
                stage.stamped("subprototypes_llm", "v2", ".json"),
                stage.stamped("subprototypes_llm_presplit", "v2", ".json")}) == 4


def test_a_grouping_summarised_from_other_captions_is_refused(tmp_path):
    """The criterion behind the name, because a name only stops known mistakes.

    The failure it catches is silent: re-clustering runs to completion against
    a grouping from another corpus and publishes a vocabulary.
    """
    import json as _json

    import pytest as _pytest

    from tools import run_wild_stage_g_m3b_oss as stage

    captions = tmp_path / "captions_v5union.jsonl"
    captions.write_text("", encoding="utf-8")
    grouping = tmp_path / "subprototypes_llm_v5union.json"

    grouping.write_text(_json.dumps(
        {"captions": "/dev/shm/whatever/captions_v2.jsonl", "groups": 53}),
        encoding="utf-8")
    with _pytest.raises(SystemExit) as excinfo:
        stage.guard_subprototype_source(grouping, captions)
    # Both generations have to appear, or the message cannot be acted on.
    assert "captions_v2.jsonl" in str(excinfo.value)
    assert "captions_v5union.jsonl" in str(excinfo.value)

    # A report with no captions field is refused too: it is a grouping whose
    # provenance is unknown, not a grouping that is fine.
    grouping.write_text(_json.dumps({"groups": 53}), encoding="utf-8")
    with _pytest.raises(SystemExit):
        stage.guard_subprototype_source(grouping, captions)

    # The staging root is shared and rebuilt, so only the basename may matter.
    grouping.write_text(_json.dumps(
        {"captions": "/some/other/root/captions_v5union.jsonl"}), encoding="utf-8")
    stage.guard_subprototype_source(grouping, captions)
