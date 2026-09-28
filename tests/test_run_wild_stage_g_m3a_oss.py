"""The segmentation M3a captions on, and the two rules that are not the same one.

``caption_segments_vlm`` will take its boundaries from either the M2 embedding
cache or from runs of equal frame label, depending on whether the driver passes
``--embedding-cache``.  The wild_v4 run did not pass it, and the two rules
disagreed on 20.4% of M2's segments -- which is invisible until M3b tries to
look a caption up by ``(recording_id, start, end)`` and finds nothing.

These pin the replay in ``clustered_spans`` against ``iter_segments``' own
arithmetic, and they pin the *disagreement*, because a test that only checked
the happy path would pass on the code that shipped the defect.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.recluster_atomics_ingroup import segments_of        # noqa: E402
from tools.run_wild_stage_g_m3a_oss import clustered_spans, m3a_keys  # noqa: E402


def build(tmp_path, labels, spans, recording="clip_a"):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    np.save(labels_dir / "clip_a.npy", np.asarray(labels, dtype=np.int64))
    (labels_dir / "labels.jsonl").write_text(json.dumps(
        {"recording_id": recording, "labels_path": "clip_a.npy"}) + "\n")
    cache = tmp_path / "cache.npz"
    np.savez(cache,
             recordings=np.asarray([recording] * len(spans)),
             starts=np.asarray([s for s, _ in spans], dtype=np.int64),
             ends=np.asarray([e for _, e in spans], dtype=np.int64))
    return labels_dir, cache


def test_clustered_spans_keeps_what_m2_clustered(tmp_path):
    labels_dir, cache = build(tmp_path, [1] * 10 + [2] * 10, [(0, 10), (10, 20)])
    assert clustered_spans(labels_dir, cache) == {"clip_a": {(0, 10), (10, 20)}}


def test_adjacent_equal_labels_are_one_span_by_label_runs_and_two_by_m2(tmp_path):
    """The defect, in the smallest corpus that shows it.

    M2 clustered two segments and gave them the same prototype.  Runs of equal
    frame label see one 20-frame segment; M2 sees two of 10.  Neither key
    matches the other, so a caption written under the first rule attaches to
    nothing under the second."""
    labels_dir, cache = build(tmp_path, [1] * 20, [(0, 10), (10, 20)])

    by_m2 = clustered_spans(labels_dir, cache)["clip_a"]
    by_label_runs = {(start, end) for start, end, label
                     in segments_of(np.load(labels_dir / "clip_a.npy"))
                     if label > 0 and end - start >= 4}

    assert by_m2 == {(0, 10), (10, 20)}
    assert by_label_runs == {(0, 20)}
    assert not (by_m2 & by_label_runs)


def test_transitions_and_short_spans_are_dropped_like_the_captioner(tmp_path):
    # label 0 is transition; the 3-frame span is under the captioner's
    # --min-frames default of 4.
    labels_dir, cache = build(tmp_path, [0] * 10 + [3] * 3 + [4] * 10,
                              [(0, 10), (10, 13), (13, 23)])
    assert clustered_spans(labels_dir, cache) == {"clip_a": {(13, 23)}}


def test_spans_are_clipped_to_the_label_array_not_dropped(tmp_path):
    """The segmentation's last span can run past the motion array, and the
    encoder that built the cache sliced with numpy, which clips.  Dropping the
    span instead left 745 of aist_v1's M2-accepted segments uncaptioned."""
    labels_dir, cache = build(tmp_path, [1] * 15, [(0, 15), (15, 40)])
    # (15, 40) clips to (15, 15), which is empty and carries no label.
    assert clustered_spans(labels_dir, cache) == {"clip_a": {(0, 15)}}


def test_a_span_clipped_below_min_frames_is_dropped(tmp_path):
    labels_dir, cache = build(tmp_path, [1] * 12, [(0, 10), (10, 30)])
    assert clustered_spans(labels_dir, cache) == {"clip_a": {(0, 10)}}


def test_parts_suffix_names_a_generation_without_touching_the_first(tmp_path):
    """The store cannot delete, so the first generation's objects are
    permanent; a second one has to be addressable separately rather than
    overwrite in place.

    Until 2026-08-22 this test ended ``assert first["captions"] ==
    second["captions"]`` -- the *merged* file was deliberately shared, on the
    reading that it names the current release while history lives in the parts.
    That reading is right for a re-run that corrects the same thing: CLAUDE.md
    1.1 blesses recomputing in place precisely because it leaves no orphan.

    It is wrong for a generation that is a *different* thing.  Schema v2 is a
    second vocabulary over the same segments, and sharing the merged name means
    publishing it destroys the caption set the released ``_ingroup_llm`` bundle
    was built from -- while every downstream reader still names only
    ``captions.jsonl``, so nothing records which generation any bundle came
    from.  Passing ``--parts-suffix`` is the operator saying "this is another
    generation"; a re-run that means to correct in place simply passes nothing.
    """
    first, second = m3a_keys("wild_v4"), m3a_keys("wild_v4", "keyed")
    assert first["parts"] == "runs/wild_v4_captions/parts"
    assert second["parts"] == "runs/wild_v4_captions/parts_keyed"
    assert first["parts"] != second["parts"]
    assert first["captions"] != second["captions"]
    # The unsuffixed names are exactly what they were, so every reader written
    # before the suffix existed still resolves the published corpus.
    assert first["captions"] == "runs/wild_v4_captions/captions.jsonl"


def test_a_shard_that_captioned_nothing_does_not_exit_zero():
    """``SHARD_DONE ... captions=0`` reads like success and used to return 0.

    On 2026-08-22 shard 6 lost both batches to CUDA OOM and signed off exactly
    that way.  The coverage gate two stages later would have caught it -- 0.8815
    against a 0.90 floor -- but with 0.019 of margin, which a smaller shard or a
    single lost batch would slide under.
    """
    import inspect

    from tools import run_wild_stage_g_m3a_oss as driver

    body = inspect.getsource(driver.cmd_run)
    # Asserted on the function body rather than through a full run: reaching
    # this line needs a staged corpus, a GPU and the model.  What has to be true
    # is that the empty case is separated from the successful one *before* the
    # return, which a test that only exercised the happy path would not pin.
    assert "SHARD_EMPTY" in body
    empty = body.index("SHARD_EMPTY")
    assert "return 1" in body[empty: empty + 400]


def test_clustered_span_prototypes_carries_the_label_the_span_starts_on(tmp_path):
    labels_dir, cache = build(tmp_path, [1] * 10 + [2] * 10, [(0, 10), (10, 20)])
    from tools.run_wild_stage_g_m3a_oss import clustered_span_prototypes

    assert clustered_span_prototypes(labels_dir, cache) == {
        "clip_a": {(0, 10): 1, (10, 20): 2}}
    # The span view stays exactly what it was, so every existing caller reads
    # the same thing it read before the prototypes were carried alongside.
    assert clustered_spans(labels_dir, cache) == {"clip_a": {(0, 10), (10, 20)}}


def _verify(tmp_path, monkeypatch, labels, spans, caption_rows):
    """Run cmd_verify against a synthetic label tree and a synthetic store."""
    import argparse

    from tools import run_wild_stage_g_m3a_oss as stage

    labels_dir, cache = build(tmp_path, labels, spans)
    monkeypatch.setattr(stage, "stage_segmentation",
                        lambda tag, root: {"labels": labels_dir, "cache": cache})
    monkeypatch.setattr(stage, "published_keys",
                        lambda key, staging: {"clip_a": caption_rows})
    args = argparse.Namespace(tag="t", parts_suffix="g",
                              segmentation_root=str(tmp_path / "seg"),
                              min_caption_coverage=0.9)
    return stage.cmd_verify(args)


def test_verify_refuses_captions_keyed_to_another_m2_run(tmp_path, monkeypatch, capsys):
    """The gate coverage alone cannot fire on.

    ``--reuse-parts`` republishes an earlier generation's rows once their span
    survives, so after an M2 refit the caption is right and its ``prototype``
    is another vocabulary's.  M3b builds its cells straight off that field.
    On 2026-08-27 this was 80,720 of 201,913 rows at coverage 0.9997.
    """
    rows = {(0, 10): {"prototype": 1, "caption": "a"},
            (10, 20): {"prototype": 96, "caption": "b"}}   # 96 is the stale one
    rc = _verify(tmp_path, monkeypatch, [1] * 10 + [2] * 10, [(0, 10), (10, 20)], rows)
    out = capsys.readouterr().out
    assert rc == 1
    assert "VERIFY_FAIL" in out
    assert "1 of 2 captions" in out
    # The message has to name the repair: re-keying rewrites an integer, while
    # re-captioning the same spans is card hours.
    assert "rekey_captions_to_labels.py" in out


def test_verify_passes_when_every_prototype_is_this_vocabularys(tmp_path, monkeypatch,
                                                                capsys):
    """The positive control.

    A gate that only fires is not evidence it is measuring the right thing --
    this is the reading it has to give on captions that are in fact current.
    """
    rows = {(0, 10): {"prototype": 1, "caption": "a"},
            (10, 20): {"prototype": 2, "caption": "b"}}
    rc = _verify(tmp_path, monkeypatch, [1] * 10 + [2] * 10, [(0, 10), (10, 20)], rows)
    out = capsys.readouterr().out
    assert rc == 0
    assert "VERIFY_OK" in out
    assert "captions whose prototype is not this vocabulary's : 0" in out


def test_a_caption_row_with_no_prototype_field_is_not_counted_stale(tmp_path,
                                                                    monkeypatch):
    """Absent is not wrong.

    Older rows predate the field; convicting them would make the gate fire on
    a corpus with nothing to repair, and a gate that cries wolf gets bypassed.
    """
    rows = {(0, 10): {"caption": "a"}, (10, 20): {"caption": "b"}}
    assert _verify(tmp_path, monkeypatch, [1] * 10 + [2] * 10,
                   [(0, 10), (10, 20)], rows) == 0
