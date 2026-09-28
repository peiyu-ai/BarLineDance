#!/usr/bin/env python3
"""Paper M3: split each prototype into finer, semantically coherent sub-prototypes.

Quoting the method: "we conduct semantical re-clustering for each prototype
P_j... we first pre-split each atomic movement prototype P_j by dance genre...
we use a summarizing LLM to iteratively: (i) identify a subset of mutually
similar captions as a sub-prototype, and (ii) distill it into a concise semantic
tag... On average, each group yields 7.3 sub-prototypes with 31.8 samples each."

The paper's Tab. 2 prices this stage: at cluster base 100, FID_k goes 32.68
without it, 30.11 with re-clustering but no LLM, and 25.26 with the LLM.  So it
is the single largest term in that table, and the two rows are *different
methods*, not a quality knob.

Both rows are implemented here and each run is labelled with the one it ran.
Without ``--captions`` it is the ``w/o LLM`` row, which needs no external
dependency:

* segments are re-clustered **within** each prototype using the paper's own
  keyframes -- signature poses at motion beats, plus movement dynamics (see
  ``tools/motion_beats.py``);
* the number of sub-prototypes per group is chosen from a target group size
  rather than fixed, because the paper reports an *average* of 7.3 with a
  spread, and a fixed k would impose uniformity the data does not have;
* genre pre-split happens only when a genre map is supplied.  AIST encodes
  genre in the sequence name; TikTok has none, and inventing one would be worse
  than recording its absence, so the report says which applied.

``--subprototypes`` switches on the ``w/ LLM`` row (25.26).  It takes the
grouping written by ``tools/summarize_subprototypes_llm.py`` -- the paper's
summarizing LLM, which reads the VLM's captions and decides the membership
itself -- and reads it back onto the segments.  ``--captions`` comes with it,
because the grouping is keyed by caption text.  Segments the grouping does not
cover fall back to the keyframe criterion above, and the report counts them.

``--captions`` alone is a third thing, and is labelled as such: VLM captions as
in ``w/ LLM``, but grouped by k-means over caption embeddings rather than by the
paper's LLM.  It exists because it needs no second model, and its report says
plainly that it sits between the two published rows rather than on either.

Deviations from the paper, recorded rather than hidden:

* Every non-empty pre-split cell yields at least one sub-prototype here, and the
  paper's procedure carries no such floor -- it iterates until the *ungrouped*
  segments fall below a threshold it does not publish.  On AIST++ that floor is
  700 of 849 classes (``tools/audit_genre_presplit.py``), so it is the dominant
  term in the vocabulary size.  ``--min-cell-size`` removes it; the default
  keeps it, because dropping segments to transition is a cost M4 pays.

* Gemini-2.5-Pro is unreachable from this network, so a local Qwen writes the
  captions, and a local Qwen also plays the summarizing LLM.  Both names ride
  in the report and in every label row.
* The paper's step (ii) semantic tag is the LLM's on the ``--subprototypes``
  path.  Without it, the tag falls back to the medoid caption, which is a real
  observed description rather than a generated one, and is named
  ``medoid_caption`` in the report so nobody reads it as LLM-authored.

Whichever row ran is written into the report and every label row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.cluster_atomics_tmr import (  # noqa: E402
    REJECTED,
    TRANSITION,
    ClusterError,
    build_row_index,
    kmeans,
    resolve_row,
    sha256_file,
)
from tools.convert_motion_to_guofeats import motion_151_to_joints  # noqa: E402
from tools.motion_beats import descriptor_vector  # noqa: E402

SCHEMA_VERSION = "atomicdance-kinematic-atomic-labels-v1"
PRODUCER_VERSION = "tmr-ingroup-recluster-v1"
_GENRE = re.compile(r"^(g[A-Z]{2})_")


def segments_of(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        return []
    edges = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(labels)]])
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends)]


def genre_of(name: str, groups: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The key the prototype is pre-split by, before the summarizing LLM runs.

    The paper pre-splits by dance genre "as motions from different genres tend
    to be semantically distinct".  On AIST++ that is a dataset field, written
    into the sequence id -- the paper never predicts it, it reads it.  TikTok
    carries no such field, and a VLM asked to supply one answered at chance
    (0.083 against 0.10 on ten balanced classes, and 0.113 even through the
    video path), so predicting it would scatter one movement across ten groups.

    ``groups`` supplies the key from metadata instead: on this corpus the
    choreographer account, which every upload has and which is arguably a
    *tighter* grouping than "hip-hop" -- one choreographer's style is more
    specific than a genre.  Same epistemic status as AIST's field: recorded, not
    inferred.  Whether it actually groups semantically is a separate question
    with its own answer -- the coherence permutation test in
    ``audit_atomic_vocabulary.py`` -- and not something to assume.
    """
    stem = name.rsplit("/", 1)[-1]
    if groups:
        # Recording ids appear in two shapes across this repo -- the bundle's
        # "corpus:upload:clip" and the cache directory's "upload__clipNNN" --
        # and the key is a property of the upload in both.  Trying each is
        # cheaper than making every caller normalise first, and a lookup that
        # silently missed would drop the pre-split to a single group with no
        # error anywhere.
        candidates = [stem, stem.split("__clip")[0]]
        if ":" in stem:
            candidates.append(stem.split(":")[-2] if stem.count(":") >= 2 else stem)
        for candidate in candidates:
            key = groups.get(candidate)
            if key:
                return key
    match = _GENRE.match(stem)
    return match.group(1) if match else None


def load_group_keys(path: Optional[pathlib.Path]) -> Dict[str, str]:
    """``{clip or upload id: group key}`` from a JSON mapping, or empty."""
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in payload.items() if v}


def clip_span(start: int, end: int, length: int) -> Optional[Tuple[int, int]]:
    """A cached span, trimmed to the label array, or None if nothing is left.

    The segmentation's final span for a recording can run past the motion array
    -- 958 of 16,275 on AIST++, median overshoot 6 frames.  The encoder that
    built the embedding cache sliced with numpy, which clips silently, so the
    cached embedding already describes the trimmed span.  Skipping such a span
    instead (the behaviour until 2026-08-13) left 716 M2-accepted segments
    un-re-clustered and their frames carrying M2 ids inside the M3 id space.
    """
    end = min(int(end), int(length))
    start = int(start)
    if end - start < 1:
        return None
    return start, end


def sub_prototype_count(members: int, target_size: int) -> int:
    """How many sub-prototypes a group of this size should yield."""
    return max(1, int(round(members / float(target_size))))


def compact_label_ids(prototypes: np.ndarray, sub_labels: np.ndarray,
                      ungrouped: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-segment class ids, contiguous from 1, with 0 left for transition.

    Laying sub-prototypes out as ``(p-1)*width + s + 1`` reserves ``width``
    slots per prototype, and groups differ in size, so most slots stay empty --
    on the wild corpus that was 684 real classes scattered across 1300 ids.  A
    D3PM over the padded space would spend a third of its vocabulary on tokens
    that never occur and its uniform noise prior would put mass there.

    Ungrouped segments take ``TRANSITION`` and, more importantly, are excluded
    before the ids are counted: leaving them in would mint a class for a group
    that was never formed, and every size that Fig. 4c is read from comes out
    of this array.
    """
    raw_ids = (prototypes - 1) * width + sub_labels + 1
    used_ids = np.unique(raw_ids[~ungrouped])
    compact = {int(old): index + 1 for index, old in enumerate(used_ids)}
    compact_ids = np.asarray([TRANSITION if drop else compact[int(value)]
                              for value, drop in zip(raw_ids, ungrouped)], dtype=np.int64)
    return compact_ids, used_ids


def load_captions(path: pathlib.Path) -> Tuple[Dict[Tuple[str, int, int], str], List[str]]:
    """Read caption rows keyed by the segment they describe, plus the models used.

    A segment captioned twice (two shards overlapping, a resumed run) keeps the
    first caption rather than the last, so the mapping does not depend on file
    order.  The model names travel out because a bundle built from two different
    captioners is a fact the report has to state, not one to average over.
    """
    captions: Dict[Tuple[str, int, int], str] = {}
    models = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("caption")
            if not text:
                continue
            models.add(str(row.get("model", "unknown")))
            captions.setdefault(
                (row["recording_id"], int(row["start"]), int(row["end"])), text)
    return captions, sorted(models)


def load_subprototypes(path: pathlib.Path) -> Tuple[
        Dict[Tuple[int, str, str], int], Dict[Tuple[int, str, int], str], Dict[str, object]]:
    """Read the summarizing LLM's grouping into caption -> sub-prototype lookups.

    The summarizer works on *distinct* captions, so its output is a mapping from
    a sentence to the sub-prototype it was placed in, per (prototype, genre)
    group.  Both the membership and the LLM's semantic tag come back: the tag is
    the paper's step (ii) and is the only part of this pipeline that is a
    genuine natural-language label rather than a number.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    placement: Dict[Tuple[int, str, str], int] = {}
    tags: Dict[Tuple[int, str, int], str] = {}
    for group in report.get("groups_detail", []):
        prototype, genre = int(group["prototype"]), str(group["genre"])
        for index, sub in enumerate(group["subprototypes"]):
            tags[(prototype, genre, index)] = str(sub.get("tag", "untagged"))
            for caption in sub.get("captions", []):
                placement.setdefault((prototype, genre, str(caption)), index)
    if not placement:
        raise ClusterError("{} contains no sub-prototype memberships".format(path))
    header = {key: value for key, value in report.items() if key != "groups_detail"}
    return placement, tags, header


def place_by_llm(group: np.ndarray, prototype: int, genres: Sequence[str],
                 caption_texts: Sequence[str],
                 placement: Dict[Tuple[int, str, str], int]
                 ) -> Tuple[np.ndarray, Dict[Tuple[str, int], int]]:
    """Read the summarizing LLM's grouping onto one group's segments.

    The summarizer works per (prototype, genre), so its indices restart at 0 for
    every genre.  Without ``--genre-split`` this group spans several genres, and
    taking those indices at face value would publish (p, gBR) sub-prototype 0
    and (p, gJZ) sub-prototype 0 as one class.  So a local slot is keyed by the
    *pair*, and each distinct pair gets its own slot.

    Returns the per-member slot (-1 where the LLM placed nothing) and the slot
    table, which the caller needs to attach each sub-prototype's semantic tag.
    """
    slots: Dict[Tuple[str, int], int] = {}
    assignment = np.full(len(group), -1, dtype=np.int64)
    for local, position in enumerate(group):
        caption = caption_texts[position]
        genre = str(genres[position])
        index = placement.get((prototype, genre, caption))
        if index is None and (prototype, "?", caption) in placement:
            genre, index = "?", placement[(prototype, "?", caption)]
        if index is None:
            continue
        key = (genre, int(index))
        if key not in slots:
            slots[key] = len(slots)
        assignment[local] = slots[key]
    return assignment, slots


def nearest_group(features: np.ndarray, members: np.ndarray,
                  assignment: np.ndarray, count: int) -> np.ndarray:
    """Place unassigned members (marked -1) with the sub-prototype nearest them.

    A segment can miss the LLM's grouping two ways: it was never captioned, or
    its caption is one the summarizer never saw.  Neither is a reason to drop
    the segment -- it is a real span of frames that needs a label -- so it falls
    back to the criterion the ``w/o LLM`` row uses, the keyframe descriptor, and
    the report counts how many landed that way.
    """
    unknown = np.flatnonzero(assignment < 0)
    if len(unknown) == 0:
        return assignment
    centres = []
    for index in range(count):
        placed = members[assignment == index]
        centres.append(features[placed].mean(axis=0) if len(placed)
                       else np.full(features.shape[1], np.inf))
    centre_stack = np.stack(centres)
    if not np.isfinite(centre_stack).all(axis=1).any():
        assignment[unknown] = 0
        return assignment
    distances = ((features[members[unknown]][:, None, :] - centre_stack[None]) ** 2).sum(axis=2)
    assignment[unknown] = np.argmin(distances, axis=1)
    return assignment


def merge_small_subprototypes(features: np.ndarray, members: np.ndarray,
                              assignment: np.ndarray, count: int,
                              min_size: int) -> Tuple[np.ndarray, int, int]:
    """Fold sub-prototypes below ``min_size`` into their nearest surviving sibling.

    The ``--subprototypes`` path hands the class count to the summarizing LLM
    (``count = max(len(slots), 1)``), and ``--target-size`` does not bind there
    at all.  What the LLM produces is far finer than the paper's: on wild_v4,
    4,159 sub-prototypes at 41.6 per prototype against the paper's 730 at 7.3,
    with **69.8% of classes holding under 20 segments** where Fig. 4c has 14.9%,
    and a dispersion of 2.03 against the paper's much tighter shape.  A class
    with two samples is not a discovered movement; it is a caption the LLM
    happened to phrase differently.

    Merging rather than dropping, because the alternative costs the segments.
    ``min_cell_size`` already exists and drops a *cell* whose members then become
    transition -- a real cost to the planner.  Here every segment keeps a label;
    only the label's granularity changes, so the vocabulary contracts without
    the corpus shrinking.

    The merge target is the nearest surviving sub-prototype **inside the same
    prototype**, by centroid in the same feature space ``nearest_group`` uses.
    That containment matters: merging across prototypes would undo M2's
    clustering, which is the paper's step the repo does *not* have a quarrel
    with.

    Smallest-first, one at a time, recomputing after each merge -- a single pass
    that reassigned every small class at once would leave two small classes
    merged into each other and still below the floor.

    Returns the relabelled assignment, the new count, and how many merges ran.
    """
    if min_size <= 1 or count <= 1:
        return assignment, count, 0
    merges = 0
    while True:
        sizes = np.bincount(assignment[assignment >= 0], minlength=count)
        alive = [index for index in range(count) if sizes[index] > 0]
        if len(alive) <= 1:
            break
        smallest = min(alive, key=lambda index: (sizes[index], index))
        if sizes[smallest] >= min_size:
            break
        centres = {}
        for index in alive:
            placed = members[assignment == index]
            centres[index] = features[placed].mean(axis=0)
        others = [index for index in alive if index != smallest]
        target = min(others, key=lambda index: float(
            ((centres[index] - centres[smallest]) ** 2).sum()))
        assignment[assignment == smallest] = target
        merges += 1
    # Relabel to a contiguous range so the caller's ``offset`` arithmetic keeps
    # producing a dense id space; a gap here becomes an empty class in the
    # release, and an empty class is a softmax row that never receives gradient.
    used = sorted(set(assignment[assignment >= 0].tolist()))
    remap = {old: new for new, old in enumerate(used)}
    relabelled = assignment.copy()
    for old, new in remap.items():
        relabelled[assignment == old] = new
    return relabelled, len(used), merges


def scale_block(block: np.ndarray) -> np.ndarray:
    """Standardise a feature block and equalise its total variance.

    Concatenating a 512-D caption embedding with a ~200-D keyframe descriptor
    would otherwise let the caption side dominate purely by having more columns,
    which is a dimension-count artefact rather than a statement about which
    signal matters.
    """
    block = (block - block.mean(axis=0)) / (block.std(axis=0) + 1e-8)
    return block / np.sqrt(block.shape[1])


def embed_captions(texts: Sequence[str], *, device: str) -> np.ndarray:
    """Embed caption sentences with TMR's text encoder.

    The closed caption vocabulary makes the sentences highly repetitive, so
    unique sentences are encoded once and reused -- on the order of thousands
    of forward passes instead of twenty thousand.
    """
    from tools.tmr_runtime import TMREncoder

    unique = sorted(set(texts))
    encoder = TMREncoder(device=device, text=True)
    vectors = encoder.encode_text(unique)
    lookup = {text: index for index, text in enumerate(unique)}
    return np.stack([vectors[lookup[text]] for text in texts]).astype(np.float64)


def medoid_index(block: np.ndarray) -> int:
    """Index of the member closest to its group's centre."""
    centre = block.mean(axis=0, keepdims=True)
    return int(np.argmin(((block - centre) ** 2).sum(axis=1)))


def train_performance_of_recording(bundle: pathlib.Path) -> Dict[str, Optional[str]]:
    """``recording_id -> retrieval_group_id``, or ``None`` outside the train split.

    Recurrence is counted on train performances only, for the same reason M2
    fits K-Means on train segments only: a class supported by a held-out
    performance is a class the model cannot learn and the source-safe retrieval
    audit -- which reads the *train* split -- will not credit.  Counting every
    split instead would keep classes that hold one train performance and two
    test ones, and the audit would still fail them.

    Raises rather than falling back to the recording id.  A recording-level
    fallback would make every sub-prototype look like it spans several
    performances, so the recurrence rule below would pass on every corpus --
    a gate that cannot fail, which this repo has on record as worse than none.
    """
    path = bundle / "sources.jsonl"
    if not path.is_file():
        raise ClusterError(
            "--min-retrieval-groups needs {} to know which performance a recording "
            "belongs to".format(path))
    mapping: Dict[str, Optional[str]] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        group = row.get("retrieval_group_id")
        if not isinstance(group, str) or not group:
            raise ClusterError(
                "{} has no retrieval_group_id for {}".format(path, row.get("recording_id")))
        mapping[row["recording_id"]] = group if row.get("split") == "train" else None
    return mapping


def drop_rare_subprototypes(prototypes: np.ndarray, sub_labels: np.ndarray,
                            ungrouped: np.ndarray,
                            owner_performance: Sequence[Optional[str]],
                            min_retrieval_groups: int) -> Tuple[int, int]:
    """Mark every sub-prototype seen in too few train performances as ungrouped.

    ``None`` in ``owner_performance`` means the segment's recording is held out,
    and it is not counted: a class supported only by val/test performances is
    not one the planner can learn from, and the source-safe retrieval audit
    reads the train split alone.  Mutates ``ungrouped`` and returns
    ``(sub-prototypes dropped, segments dropped)``.
    """
    dropped_classes = dropped_segments = 0
    for prototype in sorted(set(np.asarray(prototypes).tolist())):
        in_prototype = np.asarray(prototypes) == prototype
        for sub in sorted(set(np.asarray(sub_labels)[in_prototype].tolist())):
            member = np.flatnonzero(in_prototype & (np.asarray(sub_labels) == sub) & ~ungrouped)
            if member.size == 0:
                continue
            seen = {owner_performance[position] for position in member.tolist()}
            seen.discard(None)
            if len(seen) < min_retrieval_groups:
                ungrouped[member] = True
                dropped_classes += 1
                dropped_segments += int(member.size)
    return dropped_classes, dropped_segments


def build(*, labels_dir: pathlib.Path, bundle: pathlib.Path, output_dir: pathlib.Path,
          target_size: int, seed: int, max_beats: int, min_cell_size: int = 0,
          min_subprototype_size: int = 0,
          min_retrieval_groups: int = 1,
          genre_split: bool = False, group_keys: Optional[Dict[str, str]] = None,
          captions_path: Optional[pathlib.Path] = None,
          caption_weight: float = 1.0, device: str = "cpu",
          min_caption_coverage: float = 0.9,
          subprototypes_path: Optional[pathlib.Path] = None,
          embedding_cache: Optional[pathlib.Path] = None) -> Dict[str, object]:
    if output_dir.exists():
        raise ClusterError("{} exists; label bundles publish into a new directory".format(output_dir))
    if subprototypes_path is not None and captions_path is None:
        raise ClusterError(
            "--subprototypes groups captions, so --captions is required with it: the "
            "grouping is keyed by caption text and there is no way to tell which "
            "sub-prototype a segment belongs to without knowing its caption")

    rows = {json.loads(l)["recording_id"]: json.loads(l)
            for l in (bundle / "sequences.jsonl").open(encoding="utf-8")}
    index = build_row_index(rows)
    label_rows = [json.loads(l) for l in (labels_dir / "labels.jsonl").open(encoding="utf-8")]

    # Where the segments come from, which is not a detail.
    #
    # Reading them back as runs of equal label -- `segments_of` -- silently
    # merges two adjacent M1 segments whenever M2 put them in the same
    # prototype, because in the frame array they are one unbroken stretch of the
    # same id.  Measured on AIST++ aist_v1 that is 4,413 of 13,686 accepted
    # segments, 32.2%, absorbed into a neighbour; none of the loss is the
    # `< 4 frames` guard below, which drops nothing.  The paper's M3 re-clusters
    # the segments M2 accepted, so clustering merged runs instead measures a
    # different object: a merged run spans a longer motion and its keyframes
    # describe that longer motion.
    #
    # The embedding cache carries (recording, start, end) for every segment as
    # the clusterer actually saw it, so passing it restores the paper's unit.
    # tools/report_paper_alignment.py avoids the same trap the same way, and
    # says so in `prototype_counts`.  Without the cache the old behaviour is
    # kept rather than guessed at, and the report records which was used.
    spans: Optional[Dict[str, List[Tuple[int, int]]]] = None
    if embedding_cache is not None:
        cached = np.load(embedding_cache, allow_pickle=True)
        for key in ("recordings", "starts", "ends"):
            if key not in cached:
                raise ClusterError(
                    "{} has no '{}'; it is not a segment-level embedding cache".format(
                        embedding_cache, key))
        spans = {}
        for name, start, end in zip(cached["recordings"], cached["starts"], cached["ends"]):
            spans.setdefault(str(name), []).append((int(start), int(end)))

    def segments_for(recording: str, labels: np.ndarray):
        """(start, end, prototype) for one recording, from whichever source."""
        if spans is None:
            yield from segments_of(labels)
            return
        for start, end in spans.get(recording, ()):
            # Clip rather than skip.  The segmentation's last span can run past
            # the motion array -- 958 of 16,275 on AIST++, always a recording's
            # final segment, median overshoot 6 frames.  The encoder that built
            # the cache sliced the same way numpy does, so the cached embedding
            # already describes the clipped span; dropping it here instead left
            # 716 M2-accepted segments unre-clustered, and the publisher below
            # used to keep their source values, shipping raw M2 prototype ids
            # 1..100 inside the M3 id space over 20,445 frames.
            clipped = clip_span(start, end, len(labels))
            if clipped is None:
                continue
            yield clipped[0], clipped[1], int(labels[clipped[0]])

    # Gather every accepted segment with its prototype, descriptor and owner.
    descriptors: List[np.ndarray] = []
    owners: List[Tuple[str, int, int]] = []
    prototypes: List[int] = []
    genres: List[str] = []
    for entry in label_rows:
        row = resolve_row(index, entry["recording_id"]) or rows.get(entry["recording_id"])
        if row is None:
            continue
        labels = np.load(labels_dir / entry["labels_path"])
        joints = motion_151_to_joints(np.load(bundle / row["motion_path"]))
        for start, end, label in segments_for(entry["recording_id"], labels):
            if label <= 0 or end - start < 4:
                continue
            descriptors.append(descriptor_vector(joints[start:end], max_beats=max_beats))
            owners.append((entry["recording_id"], start, end))
            prototypes.append(label)
            genres.append(genre_of(entry["recording_id"], group_keys) or "?")
    if not descriptors:
        raise ClusterError("no accepted segments to re-cluster")

    features = np.stack(descriptors).astype(np.float64)
    prototypes = np.asarray(prototypes)
    genres = np.asarray(genres)
    # Standardise so the six dynamics numbers cannot be swamped by hundreds of
    # pose coordinates, nor vice versa.
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    placement = tag_lookup = summary_header = None
    if subprototypes_path is not None:
        placement, tag_lookup, summary_header = load_subprototypes(subprototypes_path)

    caption_stats: Optional[Dict[str, object]] = None
    caption_texts: List[str] = []
    if captions_path is not None:
        captions, caption_models = load_captions(captions_path)
        caption_texts = [captions.get(owner, "") for owner in owners]
        covered = np.asarray([bool(text) for text in caption_texts])
        coverage = float(covered.mean())
        if coverage < min_caption_coverage:
            raise ClusterError(
                "captions cover {:.1%} of segments, below the {:.0%} floor; an "
                "under-captioned run would silently mix two different clustering "
                "criteria across the corpus".format(coverage, min_caption_coverage))
        if placement is None:
            # Uncaptioned segments cannot be placed in caption space at all.
            # Rather than inventing an embedding for them, give them the mean
            # caption vector, which leaves them to be separated by their
            # keyframe half -- the w/o-LLM criterion -- and count them so the
            # report says how many.
            embedded = embed_captions([text for text in caption_texts if text], device=device)
            vectors = np.zeros((len(caption_texts), embedded.shape[1]), dtype=np.float64)
            vectors[covered] = embedded
            if (~covered).any():
                vectors[~covered] = embedded.mean(axis=0)
            features = np.concatenate(
                [caption_weight * scale_block(vectors), scale_block(features)], axis=1)
        caption_stats = {
            "path": str(captions_path.resolve()),
            "coverage": round(coverage, 4),
            "captioned_segments": int(covered.sum()),
            "uncaptioned_segments": int((~covered).sum()),
            "distinct_captions": len(set(text for text in caption_texts if text)),
            # With the LLM grouping in play the captions are read as text, not
            # embedded: the sub-prototypes are the ones the summarizer wrote, so
            # an embedding would only re-derive a grouping that already exists.
            "embedding": "tmr_text_encoder" if placement is None else "none (llm grouping)",
            "caption_weight": caption_weight if placement is None else None,
            "model": "+".join(caption_models),
            "substitutes_for": "gemini-2.5-pro (paper M3)",
        }

    sub_labels = np.zeros(len(features), dtype=np.int64)
    ungrouped = np.zeros(len(features), dtype=bool)
    merges_run = 0
    per_group: Dict[int, int] = {}
    group_tags: Dict[Tuple[int, int], str] = {}
    llm_placed = 0
    for prototype in sorted(set(prototypes.tolist())):
        selection = np.flatnonzero(prototypes == prototype)
        groups = ([np.flatnonzero((prototypes == prototype) & (genres == g))
                   for g in sorted(set(genres[selection].tolist()))]
                  if genre_split else [selection])
        offset = 0
        for group in groups:
            if len(group) == 0:
                continue
            if min_cell_size and len(group) < min_cell_size:
                # The paper's grouping has no floor.  Its loop stops when the
                # *ungrouped* segments fall below a threshold, so a cell too
                # small to form a coherent group contributes none -- whereas
                # ``max(1, round(n / target))`` below makes every non-empty cell
                # a sub-prototype no matter how few segments it holds.  On
                # AIST++ aist_v1 that floor is 700 of 849 classes, so the
                # vocabulary's size is set by the pre-split rather than by the
                # clustering, and 128 classes hold a single sample.
                #
                # This is a *proxy* for the paper's rule, not the rule: the
                # paper drops segments its summarizer never groups, which is
                # not the same predicate as a cell size, and it publishes no
                # threshold.  Hence off by default, applied on the LLM path too
                # (a cell this small is below the floor whoever grouped it),
                # and the segments it drops are counted in the report rather
                # than absorbed silently -- they become transition, which is a
                # real cost to M4 and belongs to whoever asks for it.
                ungrouped[group] = True
                continue
            if placement is not None:
                # The LLM already decided the membership; this loop only reads
                # it back onto the segments and hands the leftovers to the
                # keyframe fallback.
                #
                assignment, slots = place_by_llm(
                    group, prototype, genres, caption_texts, placement)
                llm_placed += int((assignment >= 0).sum())
                count = max(len(slots), 1)
                assignment = nearest_group(features, group, assignment, count)
                for (genre, index), slot in slots.items():
                    tag = tag_lookup.get((prototype, genre, index))
                    if tag:
                        group_tags[(prototype, offset + slot)] = tag
                if min_subprototype_size > 1:
                    # The semantic tags are dropped for a merged prototype
                    # rather than carried on the survivor: two sub-prototypes
                    # the LLM named differently have become one class, and
                    # keeping one of the two names would publish a label that
                    # describes part of what the class now holds.
                    assignment, count, merged = merge_small_subprototypes(
                        features, group, assignment, count, min_subprototype_size)
                    if merged:
                        merges_run += merged
                        for slot in range(count + merged):
                            group_tags.pop((prototype, offset + slot), None)
                sub_labels[group] = assignment + offset
                offset += count
                continue
            count = min(sub_prototype_count(len(group), target_size), len(group))
            if count <= 1:
                sub_labels[group] = offset
                offset += 1
                continue
            assignment, _ = kmeans(features[group], count, seed + prototype)
            sub_labels[group] = assignment + offset
            offset += count
        per_group[prototype] = offset

    # A sub-prototype seen in only one performance is not a recurring movement.
    #
    # The paper's premise is recurrence -- prototypes are "recurring atomic
    # movement" -- and a class whose every sample comes from one performance
    # describes that performance's idiosyncrasy instead.  Measured on the
    # aist_v1_norm release, 135 of 830 classes were confined that way, and the
    # audit's permutation null puts the ceiling at 0.999999: if performance
    # membership were independent of class, essentially every atomic frame
    # would have its class somewhere else.  So the 1.9% shortfall is real
    # concentration, not the vocabulary being fine-grained.
    #
    # Off by default (1 = keep everything), because the segments it drops
    # become transition and that cost belongs to whoever asks for it.
    single_performance_subprototypes = 0
    single_performance_segments = 0
    if min_retrieval_groups > 1:
        performances = train_performance_of_recording(bundle)
        missing = sorted({recording for recording, _, _ in owners} - set(performances))
        if missing:
            raise ClusterError(
                "{} recording(s) are absent from the source manifest, e.g. {}".format(
                    len(missing), ", ".join(missing[:3])))
        single_performance_subprototypes, single_performance_segments = drop_rare_subprototypes(
            prototypes, sub_labels, ungrouped,
            [performances[recording] for recording, _, _ in owners],
            min_retrieval_groups)

    width = max(per_group.values())
    if width < 1:
        raise ClusterError("re-clustering produced no sub-prototypes")

    compact_ids, used_ids = compact_label_ids(prototypes, sub_labels, ungrouped, width)
    if len(used_ids) == 0:
        raise ClusterError(
            "--min-cell-size {} left every one of the {} cells ungrouped".format(
                min_cell_size, len(ungrouped)))

    if placement is not None:
        method = "llm_summarized_subprototypes"
        paper_row = ("Tab.2 'w/ LLM' (25.26): a VLM tags the segments and a summarizing "
                     "LLM groups the tags, both roles run by a local Qwen in place of "
                     "the paper's Gemini-2.5-Pro")
    elif captions_path is not None:
        method = "caption_and_keyframe_kmeans_vlm"
        paper_row = ("between Tab.2's two rows: VLM captions as in 'w/ LLM', but the "
                     "grouping is k-means, not the paper's summarizing LLM")
    else:
        method = "keyframe_pose_dynamics_kmeans_no_llm"
        paper_row = "Tab.2 'w/o LLM' (30.11); the 'w/ LLM' 25.26 row needs a captioner"

    # The paper's step (ii) distils a semantic tag per sub-prototype.  With the
    # summarizing LLM in the loop those tags are the ones it wrote.  Without it,
    # the medoid caption stands in: the caption of the member nearest its
    # group's centre, so it is a real observed description rather than a
    # generated summary, and it is deterministic.  The two are never mixed, and
    # the report says which kind the file holds.
    # TRANSITION is skipped in both branches.  np.unique(compact_ids) contains 0
    # whenever anything was ungrouped, and an ungrouped segment keeps its stale
    # sub_labels entry -- drop_rare_subprototypes marks it without clearing the
    # slot, and group_tags was built before the drop -- so the transition token
    # would inherit the tag of a sub-prototype that was thrown away, and the
    # published file would carry one more tag than the vocabulary has classes.
    # --min-retrieval-groups 2 guarantees ungrouped segments, so this fires on
    # exactly the runs the LLM path is used for.
    tags: Dict[int, str] = {}
    if placement is not None:
        for compact_id in np.unique(compact_ids):
            if int(compact_id) == TRANSITION:
                continue
            member = int(np.flatnonzero(compact_ids == compact_id)[0])
            tag = group_tags.get((int(prototypes[member]), int(sub_labels[member])))
            if tag:
                tags[int(compact_id)] = tag
    elif caption_texts:
        for compact_id in np.unique(compact_ids):
            if int(compact_id) == TRANSITION:
                continue
            members = np.flatnonzero(compact_ids == compact_id)
            local = medoid_index(features[members])
            tags[int(compact_id)] = caption_texts[members[local]]
    # A tag file that describes more things than exist is a claim this stage can
    # check, so it does.
    if tags and max(tags) > len(used_ids):
        raise ClusterError(
            "semantic tags reach id {} but the vocabulary holds {} classes".format(
                max(tags), len(used_ids)))

    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "labels").mkdir(parents=True)
    # ``subprototype_sizes`` is the segment count of each sub-prototype, in the
    # same order as ``used_raw_ids``.  It is published rather than left to be
    # recovered downstream because the only other route to it is counting runs
    # in the frame labels, and that undercounts: two adjacent segments that land
    # in the same sub-prototype merge into one run there.  ``prototype_counts``
    # in tools/report_paper_alignment.py carries the same warning for M2, and
    # solves it by reading the clusterer's own artifact -- this is that artifact.
    # Fig. 4c is a histogram over exactly this quantity.
    np.savez_compressed(staging / "producer.npz",
                        width=np.asarray([width]), target_size=np.asarray([target_size]),
                        seed=np.asarray([seed]), max_beats=np.asarray([max_beats]),
                        used_raw_ids=used_ids,
                        subprototype_sizes=np.bincount(compact_ids)[1:])
    producer_sha = sha256_file(staging / "producer.npz")

    by_recording: Dict[str, List[int]] = {}
    for position, (recording, _, _) in enumerate(owners):
        by_recording.setdefault(recording, []).append(position)

    allowed_ids = {TRANSITION, REJECTED} | {int(v) for v in np.unique(compact_ids)}
    published = []
    counts = {"sequences": 0, "segments": len(owners), "valid_frames": 0,
              "transition_frames": 0, "rejected_frames": 0}
    for entry in label_rows:
        recording = entry["recording_id"]
        source = np.load(labels_dir / entry["labels_path"])
        mask = np.load(labels_dir / entry["label_valid_mask_path"])
        # Start from transition, not from the M2 array.  Copying the source
        # means any frame this stage does not re-cluster keeps an id from a
        # *different* label space -- an M2 prototype in 1..100 sitting inside an
        # M3 vocabulary of 599, indistinguishable from a real class and inside
        # the range every downstream bound checks.  Segments this stage drops
        # are documented as becoming transition; this makes that true of every
        # frame rather than only of the ones it noticed dropping.
        expanded = np.zeros_like(source)
        expanded[source == REJECTED] = REJECTED
        for position in by_recording.get(recording, []):
            _, start, end = owners[position]
            expanded[start:end] = int(compact_ids[position])
        store = staging / "labels" / hashlib.sha256(recording.encode()).hexdigest()
        store.mkdir(parents=True, exist_ok=True)
        # The criterion that would have caught the M2-id leak, kept where the
        # array is written rather than in anyone's memory: every published id is
        # either transition, rejected, or one this stage actually minted.
        stray = set(np.unique(expanded).tolist()) - allowed_ids
        if stray:
            raise ClusterError(
                "{} would publish label id(s) {} that this stage never minted; "
                "the id space is 0..{} plus {}".format(
                    recording, sorted(stray)[:5], len(used_ids), REJECTED))
        np.save(store / "labels.npy", expanded)
        np.save(store / "label_valid_mask.npy", mask)
        counts["sequences"] += 1
        counts["valid_frames"] += int((expanded > 0).sum())
        counts["transition_frames"] += int((expanded == TRANSITION).sum())
        counts["rejected_frames"] += int((expanded == REJECTED).sum())
        published.append({
            **{k: v for k, v in entry.items()
               if k not in ("labels_path", "labels_sha256", "label_valid_mask_path",
                            "label_valid_mask_sha256", "producer_artifact",
                            "producer_artifact_sha256", "producer_version",
                            "label_space_id", "valid_frames", "transition_frames")},
            "producer_version": PRODUCER_VERSION,
            "producer_artifact": "producer.npz",
            "producer_artifact_sha256": producer_sha,
            "label_space_id": "tmr_atomic_ingroup_{}_v1".format(len(used_ids)),
            "valid_frames": int((expanded > 0).sum()),
            "transition_frames": int((expanded == TRANSITION).sum()),
            "labels_path": "labels/{}/labels.npy".format(store.name),
            "labels_sha256": sha256_file(store / "labels.npy"),
            "label_valid_mask_path": "labels/{}/label_valid_mask.npy".format(store.name),
            "label_valid_mask_sha256": sha256_file(store / "label_valid_mask.npy"),
            "recluster_method": method,
            "recluster_genre_presplit": bool(genre_split),
            "recluster_captioner": (caption_stats or {}).get("model", "none"),
        })

    if tags:
        (staging / "subprototype_tags.json").write_text(
            json.dumps({str(k): v for k, v in sorted(tags.items())},
                       indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (staging / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for entry in published:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    sizes = np.bincount(compact_ids)[1:]
    # Counted off the *published* ids, not off `per_group`.  `per_group` holds
    # how many sub-prototypes each prototype formed, which stops being the
    # number it kept as soon as anything is dropped after the loop -- and
    # --min-retrieval-groups drops 219 of 818 on this corpus.  Reporting the
    # formed count would put "8.18 per prototype" next to a 599-class
    # vocabulary, and nothing in the artifact would contradict it.
    surviving: Dict[int, int] = {prototype: 0 for prototype in per_group}
    for prototype, class_id in zip(prototypes.tolist(), compact_ids.tolist()):
        if class_id != TRANSITION:
            surviving.setdefault(int(prototype), 0)
    for prototype in surviving:
        selection = (prototypes == prototype) & (compact_ids != TRANSITION)
        surviving[prototype] = int(len(np.unique(compact_ids[selection])))
    subs = np.asarray(list(surviving.values()))
    report = {
        "schema_version": SCHEMA_VERSION,
        "producer_version": PRODUCER_VERSION,
        "counts": counts,
        "recluster": {
            "method": method,
            "paper_row": paper_row,
            "captions": caption_stats,
            "summarizing_llm": summary_header,
            # "matched", not "placed by the LLM": the summarizer attaches its own
            # leftovers by field agreement before writing the grouping, so a
            # caption being *in* the grouping does not mean the LLM chose it.
            # Its own share is `summarizing_llm.segments_in_llm_formed_subprototypes`.
            "segments_matched_to_llm_grouping": llm_placed if placement is not None else None,
            "segments_placed_by_keyframe_fallback": (
                len(owners) - llm_placed if placement is not None else None),
            "semantic_tags": (
                ("llm_tag per sub-prototype (paper step ii)" if placement is not None
                 else "medoid_caption per sub-prototype (not LLM-authored)")
                if tags else None),
            # Which unit was clustered.  A run-derived bundle and a
            # cache-derived one are not comparable to each other or to Fig. 4c,
            # so the distinction rides in the artifact rather than in a memory.
            "segment_source": ("embedding_cache" if spans is not None
                               else "runs_of_equal_label (merges adjacent segments "
                                    "sharing a prototype)"),
            "genre_presplit": bool(genre_split),
            "genre_presplit_key": ("metadata_group_keys" if group_keys
                                   else "aist_sequence_id") if genre_split else None,
            "target_group_size": target_size,
            # A run with a floor and a run without one are different methods,
            # not the same method at two settings: the second guarantees one
            # class per cell and the first does not, so their class counts are
            # not on the same axis.  Both numbers ride in the artifact.
            "min_cell_size": min_cell_size,
            "min_subprototype_size": min_subprototype_size,
            "subprototypes_merged_as_too_small": merges_run,
            # Same reasoning as min_cell_size: a run that requires recurrence
            # across performances is a different method, not a setting.
            "min_retrieval_groups": min_retrieval_groups,
            "single_performance_subprototypes_dropped": single_performance_subprototypes,
            "single_performance_segments_dropped": single_performance_segments,
            "ungrouped_segments": int(ungrouped.sum()),
            "ungrouped_fraction": round(float(ungrouped.mean()), 4),
            "prototypes_with_no_subprototype": int(sum(1 for v in surviving.values() if v == 0)),
            "subprototypes_formed_before_any_drop": int(sum(per_group.values())),
            "max_beats": max_beats,
            "seed": seed,
            "prototypes": len(per_group),
            "mean_subprototypes_per_prototype": round(float(subs.mean()), 2),
            "median_subprototypes_per_prototype": float(np.median(subs)),
            "mean_samples_per_subprototype": round(float(sizes.mean()), 2),
            "median_samples_per_subprototype": float(np.median(sizes)),
            "total_subprototypes": int(len(used_ids)),
            "label_ids": "contiguous 1..{} (0 reserved for transition)".format(len(used_ids)),
            "padded_id_space_avoided": int(width * len(per_group)),
            "paper_reference": {"subprototypes_per_prototype": 7.3, "samples_each": 31.8},
        },
        "input": {"labels": str(labels_dir.resolve()), "bundle": str(bundle.resolve())},
        "publication": "immutable_new_directory_only_atomic_rename",
    }
    (staging / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    os.rename(staging, output_dir)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=pathlib.Path, required=True,
                        help="M2 label bundle to refine")
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--target-size", type=int, default=32,
                        help="target samples per sub-prototype (paper reports 31.8)")
    parser.add_argument("--min-retrieval-groups", type=int, default=1,
                        help="a sub-prototype whose segments come from fewer "
                             "distinct retrieval groups (performances) than this "
                             "yields no class and its segments become transition. "
                             "1 (default) keeps every sub-prototype. 2 enforces "
                             "the paper's own premise -- prototypes are recurring "
                             "movements -- and is what makes a release pass the "
                             "source-safe retrieval bound in "
                             "tools/audit_atomic_dataset.py")
    parser.add_argument("--min-subprototype-size", type=int, default=0,
                        help="fold sub-prototypes below this many segments into the "
                             "nearest surviving sibling of the same prototype.  Only "
                             "meaningful on --subprototypes, where the class count is "
                             "the LLM's and --target-size does not bind: wild_v4 came "
                             "out at 4,159 classes with 69.8%% under 20 segments against "
                             "the paper's 730 and 14.9%%.  Merging keeps every segment "
                             "labelled, unlike --min-cell-size which drops them to "
                             "transition")
    parser.add_argument("--min-cell-size", type=int, default=0,
                        help="a pre-split cell holding fewer segments than this "
                             "yields no sub-prototype and its segments become "
                             "transition. 0 (default) keeps the floor of one "
                             "class per non-empty cell, which the paper's own "
                             "procedure does not have -- it stops when the "
                             "ungrouped segments fall below an unpublished "
                             "threshold. Changes the method, not a setting: see "
                             "runs/aist_v1_presplit_audit.json")
    parser.add_argument("--max-beats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--group-keys", type=pathlib.Path, default=None,
                        help="JSON {clip or upload id: key} supplying the pre-split "
                             "key from metadata (this corpus: the choreographer "
                             "account). Without it the key is read from an AIST "
                             "sequence id, and wild ids have none.")
    parser.add_argument("--genre-split", action="store_true",
                        help="pre-split by genre; only meaningful where names carry one")
    parser.add_argument("--captions", type=pathlib.Path, default=None,
                        help="segment caption JSONL from tools/caption_segments_vlm.py")
    parser.add_argument("--subprototypes", type=pathlib.Path, default=None,
                        help="grouping written by tools/summarize_subprototypes_llm.py; "
                             "switches on the paper's w/ LLM row (needs --captions)")
    parser.add_argument("--caption-weight", type=float, default=1.0,
                        help="weight of the caption block against the keyframe block")
    parser.add_argument("--min-caption-coverage", type=float, default=0.9,
                        help="refuse to run below this caption coverage")
    parser.add_argument("--device", default="cpu",
                        help="device for the TMR text encoder")
    parser.add_argument("--embedding-cache", type=pathlib.Path, default=None,
                        help="the segment-level TMR embeddings M2 clustered. Supplies "
                             "the segment boundaries, so two adjacent segments sharing "
                             "a prototype stay two samples instead of merging into one "
                             "run -- 32.2%% of them do on AIST++ aist_v1. Strongly "
                             "recommended: without it the sub-prototype sizes are not "
                             "comparable to the paper's Fig. 4c.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build(labels_dir=args.labels, bundle=args.bundle, output_dir=args.output_dir,
                       target_size=args.target_size, seed=args.seed, max_beats=args.max_beats,
                       min_cell_size=args.min_cell_size,
                       min_subprototype_size=args.min_subprototype_size,
                       min_retrieval_groups=args.min_retrieval_groups,
                       genre_split=args.genre_split,
                       group_keys=load_group_keys(args.group_keys),
                       captions_path=args.captions,
                       caption_weight=args.caption_weight, device=args.device,
                       min_caption_coverage=args.min_caption_coverage,
                       subprototypes_path=args.subprototypes,
                       embedding_cache=args.embedding_cache)
    except (ClusterError, FileNotFoundError) as error:
        raise SystemExit("error: {}".format(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
