"""Re-keying captions onto a second vocabulary, and the three refusals.

A caption describes a segment; the ``prototype`` on the row describes whichever
M2 run the captioner was pointed at.  Summarising the same captions against a
different vocabulary without rewriting that field is silent -- every row carries
some integer and the summarizer groups by it happily -- so each way the rewrite
could be wrong has to be able to stop it.
"""

import json
import pathlib

import numpy as np
import pytest

from tools.rekey_captions_to_labels import RekeyError, load_label_rows, rekey


def _labels_bundle(tmp_path, per_recording):
    """{recording: (label array, split)} -> a bundle rekey() can read."""
    root = tmp_path / "labels"
    (root / "labels").mkdir(parents=True)
    rows = []
    for index, (recording, (array, split)) in enumerate(per_recording.items()):
        rel = "labels/{}.npy".format(index)
        np.save(root / rel, np.asarray(array, dtype=np.int64))
        rows.append({"recording_id": recording, "labels_path": rel, "split": split})
    (root / "labels.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


def _caption(recording, start, end, prototype, split="train"):
    return {"recording_id": recording, "start": start, "end": end,
            "prototype": prototype, "split": split, "caption": "a person turns"}


class TestRekey:
    def test_the_prototype_and_split_come_from_the_target_vocabulary(self, tmp_path):
        root = _labels_bundle(tmp_path, {"r/a": ([7, 7, 7, 7], "test")})
        rows, report = rekey([_caption("r/a", 0, 4, 96)], root, load_label_rows(root))
        assert rows[0]["prototype"] == 7          # not the caption's 96
        assert rows[0]["split"] == "test"         # not the caption's "train"
        assert report["prototype_changed"] == 1
        assert rows[0]["caption"] == "a person turns"   # the description is untouched

    def test_a_segment_the_target_did_not_accept_is_dropped_not_zeroed(self, tmp_path):
        # Acceptance is a train-fitted quantile and moves between runs.  Writing
        # 0 would file the segment under the transition token and invent a cell
        # the clustering never produced.
        root = _labels_bundle(tmp_path, {"r/a": ([0, 0, 0, 0], "train")})
        rows, report = rekey([_caption("r/a", 0, 4, 96)], root, load_label_rows(root))
        assert rows == []
        assert report["dropped"]["not_accepted_by_target"] == 1
        assert report["captions_out"] == 0

    def test_a_segment_past_the_end_of_the_target_labels_is_dropped(self, tmp_path):
        root = _labels_bundle(tmp_path, {"r/a": ([5, 5], "train")})
        rows, report = rekey([_caption("r/a", 9, 12, 96)], root, load_label_rows(root))
        assert rows == []
        assert report["dropped"]["segment_outside_labels"] == 1

    def test_a_recording_the_target_never_saw_raises(self, tmp_path):
        # Two runs that segmented different corpora must not be merged quietly.
        root = _labels_bundle(tmp_path, {"r/a": ([5, 5], "train")})
        with pytest.raises(RekeyError, match="absent from the target labels"):
            rekey([_caption("r/b", 0, 2, 96)], root, load_label_rows(root))

    def test_a_caption_whose_span_the_target_splits_is_dropped(self, tmp_path):
        # The caption describes frames 0..4; the target vocabulary changes
        # prototype in the middle of them, so the caption belongs to neither
        # segment.  Checking only labels[start] would have kept it as class 7.
        root = _labels_bundle(tmp_path, {"r/a": ([7, 7, 9, 9], "train")})
        rows, report = rekey([_caption("r/a", 0, 4, 96)], root, load_label_rows(root))
        assert rows == []
        assert report["dropped"]["span_crosses_a_target_boundary"] == 1
        assert report["segmentations_agree"] is False

    def test_a_caption_whose_span_the_target_agrees_on_is_kept(self, tmp_path):
        root = _labels_bundle(tmp_path, {"r/a": ([7, 7, 7, 7], "train")})
        rows, report = rekey([_caption("r/a", 0, 4, 96)], root, load_label_rows(root))
        assert len(rows) == 1 and rows[0]["prototype"] == 7
        assert report["segmentations_agree"] is True

    def test_a_span_running_past_the_target_array_is_dropped(self, tmp_path):
        # Only the start index used to be bounds-checked, so a caption ending 38
        # frames past the array was kept without a drop or an error.
        root = _labels_bundle(tmp_path, {"r/a": ([7, 7], "train")})
        rows, report = rekey([_caption("r/a", 0, 40, 96)], root, load_label_rows(root))
        assert rows == []
        assert report["dropped"]["segment_outside_labels"] == 1

    def test_an_unchanged_prototype_is_counted_as_unchanged(self, tmp_path):
        root = _labels_bundle(tmp_path, {"r/a": ([7, 7], "train")})
        _, report = rekey([_caption("r/a", 0, 2, 7)], root, load_label_rows(root))
        assert report["prototype_changed"] == 0
        assert report["captions_out"] == 1

    def test_an_empty_label_manifest_raises(self, tmp_path):
        root = tmp_path / "labels"
        root.mkdir()
        (root / "labels.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(RekeyError, match="no rows"):
            load_label_rows(root)
