#!/usr/bin/env python3
"""Do M2's prototypes group by anything outside the embedding they were cut from?

M2's own report says how tight the clustering is -- acceptance rate, dispersion,
largest prototype -- and none of that can fail in a useful direction.  K-means
minimises within-cluster distance by construction, so "members of a prototype
are closer to each other than to other prototypes" is a criterion that is true
of noise.  CLAUDE.md §2.1 is explicit that a criterion must be shown to give a
good reading on a case whose answer is known before it is allowed to judge.

This probe measures one thing -- **how concentrated an external attribute is
inside a prototype** -- on two corpora where the desired direction is opposite,
which is what makes it an instrument rather than a number:

* **AIST++, attribute = genre (positive control).**  Genre is a recorded field
  and a strong motion prior: break and waacking do not share a movement
  vocabulary.  A clustering that found movement must concentrate genre above
  chance.  If it does not, the probe has failed to detect a signal that is
  certainly there, and no reading it gives on the wild corpus means anything.
* **The wild corpus, attribute = uploader account (leakage check).**  Here high
  is *bad*.  The account is a person, a room, an outfit and a camera; a
  prototype that concentrates one account is grouping identity, not movement.
  Chance-level is the good outcome.

**The null has to hold the corpus's own structure fixed.**  Segments of one
recording resemble each other for reasons that have nothing to do with the
attribute, so a null that shuffles the attribute over *segments* would be
trivially beaten by any clustering that keeps a recording together.  The
attribute is a property of an entity -- a recording on AIST, an upload in the
wild -- so the permutation reassigns attribute values **across entities**,
leaving every prototype's size, every recording's internal cohesion and the
attribute's marginal distribution exactly as observed.  What is left to explain
is only the alignment between the partition and the attribute.

Usage::

    probe_prototype_enrichment.py --labels data/wild3d/clean5b5_labels \\
        --embedding-cache runs/clean5b5_tmr_embeddings.npz \\
        --attribute account --group-keys runs/wild_v4_group_keys.json \\
        --output runs/clean5b5_prototype_enrichment.json

    probe_prototype_enrichment.py --labels data/atomic_aistpp/aist_atomic_p1_labels \\
        --embedding-cache runs/aist_p1_tmr_embeddings.npz \\
        --attribute aist-genre --output runs/aist_p1_prototype_enrichment.json

**``--restrict-to`` holds the corpus fixed while the clustering changes.**  Two
M2 generations read on their own corpora differ in four things at once -- K, the
segmentation, which clips are in, and how many accounts there are -- so the two
lifts cannot be subtracted.  Restricting the older generation's labels to the
newer corpus's clip list removes three of those four, leaving K and the
segmentation as the only difference, which is the comparison that answers
whether the *clustering* changed how much identity it concentrates:

    probe_prototype_enrichment.py --labels data/wild3d/wild_v4_labels \\
        --embedding-cache runs/wild_v4_tmr_embeddings.npz --attribute account \\
        --restrict-to runs/clean5/clips.txt \\
        --output runs/wild_v4_clean5subset_prototype_enrichment.json
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import pathlib
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import asset_io  # noqa: E402
from tools.run_wild_stage_g_m3a_oss import clip_stem  # noqa: E402

MIN_FRAMES = 4
AIST_GENRE = re.compile(r"/(g[A-Z]{2})_")
# The dancer id in the same name.  AIST++ has the same choreography performed
# by different dancers and the same dancer across genres, so this is identity
# with the movement attribute held roughly constant -- the third calibration
# point that says whether a lift is style or a person.
AIST_DANCER = re.compile(r"_(d\d{2})_")


def spans_from_cache(path: str) -> Dict[str, List[Tuple[int, int]]]:
    """``{recording: [(start, end)]}`` -- the segments M2 actually encoded.

    From the embedding cache rather than the segmentation because the cache is
    what M2 keyed its labels to; re-deriving the spans some other way answers a
    question nobody asked, which is how the 2026-08-13 near-retraction happened.
    """
    local = REPO / path
    source = local if local.is_file() else io.BytesIO(asset_io.read_bytes(path))
    with np.load(source, allow_pickle=True) as cached:
        for key in ("recordings", "starts", "ends"):
            if key not in cached:
                raise SystemExit(
                    "{} has no '{}'; it is not a segment-level cache".format(path, key))
        spans: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
        for name, start, end in zip(cached["recordings"], cached["starts"],
                                    cached["ends"]):
            spans[str(name)].append((int(start), int(end)))
    return dict(spans)


def segment_prototypes(labels_dir: pathlib.Path,
                       spans: Dict[str, List[Tuple[int, int]]]
                       ) -> Tuple[List[str], List[int], List[Tuple[int, int]]]:
    """One prototype per accepted segment, replaying M3a's own arithmetic.

    Same three rules ``clustered_spans`` applies -- clip to the array, read the
    label at the start, drop anything under ``MIN_FRAMES`` or labelled <= 0 --
    so this cannot disagree with what the captioner will be given.

    The clipped spans come back as a third list so a renderer can put the same
    segments on screen that this probe measured; measuring one set of segments
    and showing another is how a page comes to illustrate a number it does not
    describe.
    """
    recordings: List[str] = []
    prototypes: List[int] = []
    kept_spans: List[Tuple[int, int]] = []
    with (labels_dir / "labels.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            owned = spans.get(entry["recording_id"])
            if not owned:
                continue
            labels = np.load(labels_dir / entry["labels_path"])
            for start, end in owned:
                end = min(int(end), len(labels))
                start = int(start)
                if end - start < MIN_FRAMES or start >= len(labels):
                    continue
                value = int(labels[start])
                if value <= 0:
                    continue
                recordings.append(entry["recording_id"])
                prototypes.append(value)
                kept_spans.append((start, end))
    return recordings, prototypes, kept_spans


def restrict(recordings: Sequence[str], prototypes: Sequence[int],
             stems: Sequence[str]) -> Tuple[List[str], List[int]]:
    """Keep the segments whose clip is named in ``stems``.

    The two id shapes in this repo are ``wild_v4:<upload>:clip000`` in a label
    tree and ``<upload>__clip000`` in a corpus list, and ``clip_stem`` is the
    mapping M3a already uses -- reused rather than rewritten, because a second
    copy of it would be a second thing that can drift and the symptom of a drift
    here is an empty intersection reported as a small corpus.
    """
    wanted = set(stems)
    kept = [(recording, prototype) for recording, prototype
            in zip(recordings, prototypes) if clip_stem(recording) in wanted]
    return [r for r, _ in kept], [p for _, p in kept]


def entity_of(recording_id: str, attribute: str) -> str:
    """The thing the attribute is a property of, and therefore what to permute.

    AIST names a recording; the wild corpus names an upload, which is one
    performance a choreographer posted and may be cut into several clips.
    Permuting over clips instead would let a clustering that merely keeps one
    upload together beat the null.
    """
    if attribute in ("aist-genre", "aist-dancer"):
        return recording_id
    parts = str(recording_id).split(":")
    return ":".join(parts[:2]) if len(parts) >= 3 else recording_id


def attribute_of(recording_id: str, attribute: str,
                 group_keys: Optional[Dict[str, str]]) -> Optional[str]:
    if attribute == "aist-genre":
        found = AIST_GENRE.search(recording_id)
        return found.group(1) if found else None
    if attribute == "aist-dancer":
        found = AIST_DANCER.search(recording_id)
        return found.group(1) if found else None
    upload = str(recording_id).split(":")[1] if ":" in recording_id else recording_id
    return (group_keys or {}).get(upload)


def purity(prototypes: Sequence[int], values: Sequence[str]) -> float:
    """Share of segments sitting in their prototype's most common value."""
    tally: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for prototype, value in zip(prototypes, values):
        tally[prototype][value] += 1
    hits = sum(counter.most_common(1)[0][1] for counter in tally.values())
    return hits / max(1, len(prototypes))


def probe(recordings: Sequence[str], prototypes: Sequence[int],
          attributes: Dict[str, str], entities: Sequence[str],
          rounds: int, seed: int,
          strata: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Observed purity against a permutation that holds the corpus's structure.

    ``strata`` restricts the permutation to run *within* a stratum, and it is
    what makes an identity reading meaningful.  Measured on 2026-08-20: all 30
    AIST++ dancers appear in exactly one genre each, three per genre, so dancer
    is nested inside genre by construction and an unstratified dancer lift of
    2.22 is genre's 2.83 read through a finer label -- not evidence that the
    clustering encodes a person.  Permuting the dancer only among recordings of
    the same genre leaves genre structure untouched, so whatever lift survives
    is dancer-specific.
    """
    values = [attributes[entity] for entity in entities]
    observed = purity(prototypes, values)

    unique = sorted(set(entities))
    pool = [attributes[entity] for entity in unique]
    index = {entity: position for position, entity in enumerate(unique)}
    rows = np.fromiter((index[entity] for entity in entities), dtype=np.int64,
                       count=len(entities))

    blocks: List[np.ndarray]
    if strata is None:
        blocks = [np.arange(len(unique))]
    else:
        grouped: Dict[str, List[int]] = collections.defaultdict(list)
        for position, entity in enumerate(unique):
            grouped[strata[entity]].append(position)
        blocks = [np.asarray(members) for members in grouped.values()]

    rng = np.random.default_rng(seed)
    null = np.empty(rounds, dtype=float)
    shuffled = np.array(pool, dtype=object)
    for round_index in range(rounds):
        for members in blocks:
            if len(members) > 1:
                shuffled[members] = rng.permutation(shuffled[members])
        null[round_index] = purity(prototypes, shuffled[rows])
    hits = int((null >= observed).sum())
    return {
        "stratified_by": None if strata is None else "given",
        "strata": None if strata is None else len(blocks),
        "segments": len(prototypes),
        "prototypes": len(set(prototypes)),
        "entities": len(unique),
        "attribute_values": len(set(pool)),
        "observed_purity": round(observed, 4),
        "null_purity_mean": round(float(null.mean()), 4),
        "null_purity_sd": round(float(null.std()), 4),
        "lift": round(observed / max(1e-9, float(null.mean())), 4),
        "permutations": rounds,
        "p_value": (hits + 1) / (rounds + 1),
        "null": "attribute values reassigned across entities; prototype sizes, "
                "within-entity cohesion and the attribute's marginal "
                "distribution all held at their observed values",
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True,
                        help="a label bundle directory (labels.jsonl + arrays)")
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--attribute", required=True,
                        choices=("aist-genre", "aist-dancer", "account"))
    parser.add_argument("--group-keys", default="runs/wild_v4_group_keys.json")
    parser.add_argument("--stratify", default=None,
                        choices=("aist-genre", "aist-dancer", "account"),
                        help="permute the attribute only within entities that "
                             "share this attribute's value.  Use it when the "
                             "two are nested: AIST++ dancers each appear in "
                             "exactly one genre, so an unstratified dancer "
                             "reading is genre's reading in disguise")
    parser.add_argument("--restrict-to", default=None,
                        help="a file of clip stems (runs/clean5/clips.txt "
                             "shape): keep only the segments of those clips.  "
                             "Use it to read an older generation's labels on a "
                             "newer corpus, so the two lifts differ by the "
                             "clustering alone and not by who is in the corpus")
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    labels_dir = pathlib.Path(args.labels)
    if not labels_dir.is_dir():
        staged = pathlib.Path("/dev/shm/atomicdance-enrichment") / labels_dir.name
        print("staging {} -> {}".format(args.labels, staged), flush=True)
        asset_io.fetch_dir(args.labels, staged)
        labels_dir = staged

    group_keys = None
    if args.attribute == "account":
        group_keys = json.loads(
            pathlib.Path(args.group_keys).read_text(encoding="utf-8"))

    spans = spans_from_cache(args.embedding_cache)
    recordings, prototypes, _ = segment_prototypes(labels_dir, spans)
    print("{} accepted segment(s) over {} prototype(s)".format(
        len(prototypes), len(set(prototypes))), flush=True)

    restriction = None
    if args.restrict_to:
        stems = [line.strip() for line in pathlib.Path(args.restrict_to)
                 .read_text(encoding="utf-8").splitlines() if line.strip()]
        sample = recordings[0] if recordings else "-"
        recordings, prototypes = restrict(recordings, prototypes, stems)
        restriction = {"restricted_to": args.restrict_to,
                       "stems_requested": len(stems),
                       "recordings_matched": len(set(recordings)),
                       "segments_kept": len(prototypes)}
        print("restricted to {}: {} of {} clip(s) matched, {} segment(s) left"
              .format(args.restrict_to, restriction["recordings_matched"],
                      len(stems), len(prototypes)), flush=True)
        if not prototypes:
            raise SystemExit(
                "no label in {} carries a stem from {}; the shapes are '{}' and "
                "'{}'".format(args.labels, args.restrict_to, sample,
                              stems[0] if stems else "-"))

    attributes: Dict[str, str] = {}
    keep_rec, keep_proto, keep_entity = [], [], []
    unresolved = 0
    for recording, prototype in zip(recordings, prototypes):
        value = attribute_of(recording, args.attribute, group_keys)
        if value is None:
            unresolved += 1
            continue
        entity = entity_of(recording, args.attribute)
        attributes[entity] = value
        keep_rec.append(recording)
        keep_proto.append(prototype)
        keep_entity.append(entity)
    if unresolved:
        print("{} segment(s) have no {} and are dropped".format(
            unresolved, args.attribute), flush=True)
    if not keep_proto:
        raise SystemExit("no segment carries the attribute; nothing to measure")

    strata = None
    if args.stratify:
        strata = {}
        for recording, entity in zip(keep_rec, keep_entity):
            value = attribute_of(recording, args.stratify, group_keys)
            if value is None:
                raise SystemExit(
                    "cannot stratify: {} has no {}".format(recording, args.stratify))
            strata[entity] = value

    report = probe(keep_rec, keep_proto, attributes, keep_entity,
                   args.rounds, args.seed, strata=strata)
    report["stratified_by"] = args.stratify
    report["attribute"] = args.attribute
    report["labels"] = args.labels
    report["unresolved_segments"] = unresolved
    report["restriction"] = restriction
    report["reading"] = (
        "high is GOOD for aist-genre (a recorded movement attribute: a "
        "clustering that found movement must concentrate it) and BAD for "
        "account (a person, a room and a camera: concentrating it means the "
        "prototypes encode identity). The same statistic, opposite directions.")

    for key in ("segments", "prototypes", "entities", "attribute_values",
                "observed_purity", "null_purity_mean", "null_purity_sd",
                "lift", "p_value"):
        print("  {:22s} {}".format(key, report[key]))
    if args.output:
        target = pathlib.Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                          encoding="utf-8")
        print("wrote {}".format(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
