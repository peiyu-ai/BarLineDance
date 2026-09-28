"""The controls that have to hold before M3's ratio is allowed to judge anything.

The 2026-08-21 retraction came from a ratio that was computed correctly, had a
correct null, and still produced a wrong verdict, because nobody had measured
what a *good* grouping scores in the space the ratio was read in.  So the tests
here are not about the arithmetic of the ratio.  They pin the two calibration
points that make it readable:

* a random split of a parent has to read 1.00 -- not approximately, by
  construction, since the mean pairwise distance of a random subset is an
  unbiased estimator of the parent's; and
* a planted split, whose answer is known to be good, has to read clearly below
  it in the same run.

The third test pins the account control: two dancers whose segments are
identical within a dancer and unrelated across them are a group that captures
identity, not movement, and the cross-upload restriction is what refuses to pay
for it.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.probe_prototype_coherence import cross_upload_mean  # noqa: E402
from tools.probe_subprototype_coherence import (  # noqa: E402
    score_partition, split_random, summarise)


def _parent(vectors, uploads):
    return cross_upload_mean(vectors, vectors, uploads, uploads, same_set=True)


def _two_planted_blobs(rng, per_blob=40):
    """Two well-separated blobs, each spread over four uploads.

    Four uploads per blob rather than one: a blob that is also an upload would
    be scored on no cross-upload pairs at all, and the arm would silently drop
    it instead of measuring it.
    """
    vectors, uploads, planted = [], [], []
    for blob, centre in enumerate(([0.0] * 8, [10.0] * 8)):
        for index in range(per_blob):
            vectors.append(np.asarray(centre) + rng.normal(scale=0.3, size=8))
            uploads.append("u{}".format(index % 8))
            planted.append(blob)
    return (np.asarray(vectors), np.asarray(uploads), np.asarray(planted))


def test_a_random_split_reads_one_because_that_is_what_it_must_read():
    rng = np.random.default_rng(7)
    vectors, uploads, _ = _two_planted_blobs(rng)
    parent = _parent(vectors, uploads)

    ratios = []
    for seed in range(12):
        assignment = split_random(len(vectors), [20, 20, 20, 20],
                                  np.random.default_rng(seed))
        rows = score_partition(vectors, uploads, assignment, parent)
        ratios.append(summarise(rows)["segment_weighted_ratio"])

    assert abs(float(np.mean(ratios)) - 1.0) < 0.02


def test_the_same_instrument_reads_a_planted_split_far_below_the_floor():
    """The positive control, in the same run as the negative one.

    Without this the tool can only say "below 1.0", which is the reading that
    was mistaken for a verdict once already.
    """
    rng = np.random.default_rng(7)
    vectors, uploads, planted = _two_planted_blobs(rng)
    parent = _parent(vectors, uploads)

    good = summarise(score_partition(vectors, uploads, planted, parent))
    chance = summarise(score_partition(
        vectors, uploads, split_random(len(vectors), [40, 40], rng), parent))

    assert good["segment_weighted_ratio"] < 0.2
    assert chance["segment_weighted_ratio"] > 0.9
    assert good["n_at_or_above_1"] == 0


def test_a_split_that_only_separates_dancers_is_not_paid_for_it():
    """Each group is one upload, so it holds no cross-upload pair at all.

    Same-upload pairs would make these groups look perfect -- the segments
    inside one are identical.  ``score_partition`` has to drop them rather than
    report a ratio near zero, which is what tells the caller the arm measured
    nothing instead of measuring success.
    """
    rng = np.random.default_rng(3)
    vectors, uploads, groups = [], [], []
    for index, upload in enumerate(["a", "b", "c", "d"]):
        centre = rng.normal(scale=5.0, size=8)
        for _ in range(10):
            vectors.append(centre + rng.normal(scale=0.01, size=8))
            uploads.append(upload)
            groups.append(index)
    vectors, uploads, groups = (np.asarray(vectors), np.asarray(uploads),
                                np.asarray(groups))

    parent = _parent(vectors, uploads)
    rows = score_partition(vectors, uploads, groups, parent)

    assert rows == []
    assert summarise(rows) == {"subprototypes": 0}


def test_groups_below_the_pair_floor_are_dropped_not_scored():
    rng = np.random.default_rng(11)
    vectors, uploads, _ = _two_planted_blobs(rng, per_blob=10)
    parent = _parent(vectors, uploads)

    assignment = np.array([0, 0, 0] + [1] * (len(vectors) - 3))
    rows = score_partition(vectors, uploads, assignment, parent)

    assert [row["n"] for row in rows] == [len(vectors) - 3]


def _write_pair(tmp_path, spans, caption_spans=None):
    """An embedding cache and a caption file over the given (rec, start, end)."""
    import json

    caption_spans = spans if caption_spans is None else caption_spans
    cache = tmp_path / "tmr.npz"
    np.savez(cache,
             recordings=np.array([s[0] for s in spans]),
             starts=np.array([s[1] for s in spans]),
             ends=np.array([s[2] for s in spans]),
             embeddings=np.zeros((len(spans), 3)))
    captions = tmp_path / "captions.jsonl"
    with captions.open("w", encoding="utf-8") as handle:
        for recording, start, end in caption_spans:
            handle.write(json.dumps({"recording_id": recording, "start": start,
                                     "end": end, "prototype": 0,
                                     "caption": "a"}) + "\n")
    return cache, captions


def test_a_caption_file_written_against_another_segmentation_is_refused(tmp_path):
    """The 20.4%-unkeyable case, which is a broken pairing and not a smaller corpus."""
    import pytest

    from tools.probe_subprototype_coherence import load_captions, tmr_vectors

    spans = [("r", index, index + 10) for index in range(0, 100, 10)]
    shifted = [("r", index + 1, index + 11) for index in range(0, 100, 10)]
    cache, captions = _write_pair(tmp_path, spans, caption_spans=shifted)

    with pytest.raises(SystemExit) as raised:
        tmr_vectors(cache, load_captions(captions), 0.01)
    assert "different segmentations" in str(raised.value)


def test_a_few_stragglers_are_dropped_and_counted_rather_than_refused(tmp_path):
    from tools.probe_subprototype_coherence import load_captions, tmr_vectors

    spans = [("r", index, index + 10) for index in range(0, 1000, 10)]
    with_one_extra = spans + [("r", 9999, 10009)]
    cache, captions = _write_pair(tmp_path, spans, caption_spans=with_one_extra)

    vectors, keyed = tmr_vectors(cache, load_captions(captions), 0.01)

    assert len(vectors) == len(spans)
    assert keyed.sum() == len(spans) and not keyed[-1]


def test_unlabelled_frames_do_not_become_a_sub_prototype(tmp_path):
    """Label 0 means "no class here", not class zero.

    Read as a class it would collect every unlabelled segment in the corpus
    into one group and score it -- and because that group is arbitrary, it
    would drag the llm arm toward the floor while the random arm, built from
    its sizes, would not notice.
    """
    import json

    from tools.probe_subprototype_coherence import load_subprototype_labels

    labels = np.array([0, 0, 3, 3, 3, 0, 7, 7])
    (tmp_path / "labels").mkdir()
    np.save(tmp_path / "labels" / "r.npy", labels)
    (tmp_path / "labels.jsonl").write_text(json.dumps(
        {"recording_id": "wild:r:clip000", "labels_path": "labels/r.npy"}) + "\n",
        encoding="utf-8")

    rows = [{"recording": "wild:r:clip000", "start": start, "end": start + 2,
             "prototype": 0, "caption": "a"} for start in (0, 2, 6, 99)]

    assert load_subprototype_labels(tmp_path, rows) == [-1, 3, 7, -1]
