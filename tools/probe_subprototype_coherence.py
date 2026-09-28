#!/usr/bin/env python3
"""M3's gate: do the caption-formed sub-prototypes group motion, or only text?

The question, and why it is not M2's question
---------------------------------------------
M3b forms sub-prototypes by handing an LLM the *captions* of one M2 prototype
and asking it to group them; ``recluster_atomics_ingroup`` then reads that
grouping back onto the segments.  Nothing in that loop ever sees a motion
vector.  So a sub-prototype could be a genuine finer movement class, or it
could be a synonym set -- two ways of writing the same caption -- and the
segments underneath it no tighter than the prototype they came from.

That makes the TMR embedding a **neutral** ruler here in a way it was not for
M2.  M2 is fitted in TMR, so reading M2 in TMR is a home game (2026-08-21
worklog: M2 reads 0.7703 there and 0.9437 in pose, and the two rankings are
uncorrelated).  M3's grouping was fitted in caption space, so TMR is a space
it never optimised.

Signature-pose space is **not** neutral for M3 and is not offered here: every
caption carries a ``posescript_cue`` computed by ``motion_beats.describe_pose``
from the very joint positions pose space is built from, so the LLM has partial
sight of that space through the text.  Scoring there would pay M3 for a
correlation it was handed.

The statistic
-------------
Per sub-prototype ``s`` inside parent prototype ``P``::

    ratio(s) = mean cross-upload distance among members of s
               ------------------------------------------------
               mean cross-upload distance among members of P

The denominator is the parent, not the corpus: the sub-prototype's whole job is
to be tighter than the prototype it was cut out of, and dividing by a corpus
mean would pay it a second time for whatever M2 already achieved.

Cross-upload pairs only, as in ``probe_prototype_coherence``: consecutive
segments of one dancer resemble each other for reasons unrelated to the
vocabulary, and these prototypes concentrate accounts (measured lift 1.24
before de-accounting), so same-upload pairs would pay a group for capturing
identity.

The floor is known analytically, which is what makes it checkable
-----------------------------------------------------------------
The mean pairwise distance of a *random* subset is an unbiased estimator of the
mean pairwise distance of the set it is drawn from.  So splitting P at random
into subgroups of any sizes must read **1.00**, and it must do so independently
of how big the subgroups are.  ``random`` is that arm.  If it comes back away
from 1.00, the instrument is wrong and no other column may be read -- which is
the property CLAUDE.md 2.1 asks for and the pose-space reading of M2 lacked.

The ceiling is measured, not assumed
------------------------------------
CLAUDE.md 2.1 point 2, which is the exact hole that produced the 2026-08-21
retraction: a criterion needs a sample whose answer is known to be good, in the
space it will judge in.  ``kmeans`` splits each parent in TMR space into the
same number of subgroups the LLM produced.  It is the tightest any split of
that shape can be *in this space*, so it is the ceiling -- and the gap from
1.00 down to it is the entire range a caption-formed grouping could recover.
Reporting M3 as a fraction of that gap is the difference between "0.95, which
sounds close to nothing" and "0.95, which is 40% of everything available".

The baseline that decides whether the LLM earned its cards
----------------------------------------------------------
``caption`` groups the parent's segments by **exact caption string** -- no
model, no rounds, no GPU.  The LLM's whole contribution over that is merging
synonyms and separating homonyms.  If ``llm`` does not beat ``caption``, the
summarising model bought nothing that ``sort | uniq`` did not, and that is a
finding about the stage rather than about the corpus.

What this does not answer
-------------------------
Whether a tight sub-prototype is *one movement*.  A group of similar static
shapes reads exactly as well here as a group of kicks.  That needs the contact
sheet and a person, and the M3 acceptance criterion in the plan says so.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.probe_prototype_coherence import cross_upload_mean  # noqa: E402

MIN_SUB = 4          # a sub-prototype needs enough cross-upload pairs to mean anything


def _scorable(assignment: np.ndarray) -> np.ndarray:
    """Which members sit in a group this instrument can actually score."""
    counts = collections.Counter(assignment.tolist())
    return np.array([counts[value] >= MIN_SUB for value in assignment.tolist()])


def load_captions(path: pathlib.Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({
                "recording": str(row["recording_id"]),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "prototype": int(row["prototype"]),
                "caption": str(row["caption"]),
            })
    return rows


def load_subprototype_labels(labels_dir: pathlib.Path,
                             rows: Sequence[dict]) -> List[int]:
    """The sub-prototype each segment carries in the published bundle.

    Read at the span's start frame, which is how ``clustered_spans`` and the
    captioner key a segment; a span whose frame array is shorter than its end
    is skipped rather than clipped, so a mismatch shows up as a coverage number
    instead of a silently different partition.
    """
    per_recording: Dict[str, np.ndarray] = {}
    for line in (labels_dir / "labels.jsonl").open(encoding="utf-8"):
        entry = json.loads(line)
        per_recording[str(entry["recording_id"])] = np.load(
            labels_dir / entry["labels_path"])
    out = []
    for row in rows:
        labels = per_recording.get(row["recording"])
        if labels is None or row["start"] >= len(labels):
            out.append(-1)
            continue
        # 0 is "no class here", the same convention ``clustered_spans`` reads
        # with ``labels[start] <= 0``.  Treating it as class 0 would build one
        # enormous pseudo-sub-prototype out of every unlabelled frame in the
        # corpus and hand it to every arm alike -- a group that is tight for no
        # reason anyone chose.
        value = int(labels[row["start"]])
        out.append(value if value > 0 else -1)
    return out


def tmr_vectors(embeddings: pathlib.Path, rows: Sequence[dict],
                max_unkeyed: float) -> tuple:
    """Vectors for the captioned segments, and which of them got one.

    The failure this guards is not a few stragglers, it is a whole caption file
    written against a *different* segmentation -- runs of equal frame label
    instead of M2's clustered spans, which on wild_v4 left 20.4% of segments
    unkeyable and would have been read here as a smaller corpus rather than as
    a broken pairing.  So the threshold is stated rather than implied: above
    ``max_unkeyed`` this refuses, below it the unmatched rows are dropped and
    counted out loud.  Measured 2026-08-21: clean5b5 0.0000, wild_v4 0.0036.
    """
    blob = np.load(embeddings, allow_pickle=True)
    index = {(str(r), int(s), int(e)): i for i, (r, s, e) in enumerate(
        zip(blob["recordings"], blob["starts"], blob["ends"]))}
    picked = [index.get((row["recording"], row["start"], row["end"])) for row in rows]
    keyed = np.array([value is not None for value in picked])
    share = 1.0 - (keyed.sum() / len(picked)) if picked else 1.0
    if share > max_unkeyed:
        raise SystemExit(
            "{} of {} captioned segments ({:.2%}) have no TMR embedding under "
            "their (recording, start, end) key, over the {:.2%} allowed -- the "
            "caption file and the embedding cache are different segmentations"
            .format(int((~keyed).sum()), len(picked), share, max_unkeyed))
    if share:
        print("dropping {} of {} captioned segment(s) ({:.2%}) with no TMR "
              "embedding".format(int((~keyed).sum()), len(picked), share))
    order = np.array([value for value in picked if value is not None])
    return np.asarray(blob["embeddings"], dtype=np.float64)[order], keyed


def split_random(size: int, sizes: Sequence[int], rng) -> np.ndarray:
    assignment = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    return assignment[rng.permutation(size)]


def split_kmeans(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    k = max(1, min(k, len(vectors)))
    if k == 1:
        return np.zeros(len(vectors), dtype=int)
    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(vectors)


def score_partition(vectors: np.ndarray, uploads: np.ndarray,
                    assignment: np.ndarray, parent_within: float,
                    accounts: np.ndarray = None) -> List[dict]:
    rows = []
    for value in np.unique(assignment):
        inside = np.flatnonzero(assignment == value)
        if len(inside) < MIN_SUB:
            continue
        within = cross_upload_mean(vectors[inside], vectors[inside],
                                   uploads[inside], uploads[inside], same_set=True)
        if not np.isfinite(within) or not parent_within:
            continue
        row = {"n": int(len(inside)), "within": within,
               "ratio": within / parent_within}
        if accounts is not None:
            # Modal-account share.  Excluding same-upload pairs stops a group
            # from being paid for one dancer's consecutive segments, but not
            # for one *choreographer's* several uploads -- and 2026-08-21
            # measured that as a lift of 1.24 on M2, about half of which
            # survives de-accounting.  A grouping that is tighter because its
            # groups are more account-pure has found the person, not the move,
            # and the size-matched random arm is what says which one this is.
            modal = collections.Counter(accounts[inside].tolist()).most_common(1)
            row["account_purity"] = modal[0][1] / len(inside)
        rows.append(row)
    return rows


def summarise(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"subprototypes": 0}
    ratios = np.array([row["ratio"] for row in rows])
    weights = np.array([row["n"] for row in rows], dtype=float)
    out = {
        "subprototypes": int(len(rows)),
        "segments": int(weights.sum()),
        "median_ratio": float(np.median(ratios)),
        "segment_weighted_ratio": float((ratios * weights).sum() / weights.sum()),
        "n_at_or_above_1": int((ratios >= 1.0).sum()),
    }
    if rows and "account_purity" in rows[0]:
        purity = np.array([row["account_purity"] for row in rows])
        out["account_purity"] = float((purity * weights).sum() / weights.sum())
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=pathlib.Path, required=True,
                        help="M3a's merged captions.jsonl; supplies each "
                             "segment's parent prototype and caption text")
    parser.add_argument("--ingroup", type=pathlib.Path, required=True,
                        help="the bundle recluster published (labels.jsonl "
                             "plus per-recording arrays of sub-prototype ids)")
    parser.add_argument("--embeddings", type=pathlib.Path, required=True,
                        help="segment-level TMR cache -- the neutral space")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--group-keys", type=pathlib.Path, default=None,
                        help="runs/<tag>_group_keys.json; adds the modal-account "
                             "share per arm, the one column no arm optimises")
    parser.add_argument("--max-unkeyed", type=float, default=0.01,
                        help="share of captions allowed to key onto no TMR "
                             "span before this refuses; see tmr_vectors")
    args = parser.parse_args()

    rows = load_captions(args.captions)
    vectors, keyed = tmr_vectors(args.embeddings, rows, args.max_unkeyed)
    rows = [row for row, ok in zip(rows, keyed) if ok]
    uploads = np.array([row["recording"].rsplit(":", 1)[0] for row in rows])
    parents = np.array([row["prototype"] for row in rows])
    captions = np.array([row["caption"] for row in rows])
    subs = np.array(load_subprototype_labels(args.ingroup, rows))

    accounts = None
    if args.group_keys:
        from tools.recluster_atomics_ingroup import genre_of, load_group_keys
        groups = load_group_keys(args.group_keys)
        accounts = np.array([genre_of(row["recording"], groups) for row in rows])
        unresolved = int((accounts == "?").sum())
        print("account resolved for       : {} of {} segment(s)".format(
            len(accounts) - unresolved, len(accounts)))

    resolved = int((subs >= 0).sum())
    print("segments captioned            : {}".format(len(rows)))
    print("segments with a sub-prototype : {}".format(resolved))
    if resolved < len(rows):
        print("note: {} segment(s) fell outside the published label arrays and "
              "are excluded from every arm alike".format(len(rows) - resolved))

    rng = np.random.default_rng(args.seed)
    arms: Dict[str, List[dict]] = collections.defaultdict(list)
    parent_rows = []
    for prototype in np.unique(parents):
        member = np.flatnonzero((parents == prototype) & (subs >= 0))
        if len(member) < 2 * MIN_SUB:
            continue
        vec, upl = vectors[member], uploads[member]
        acct = None if accounts is None else accounts[member]
        parent_within = cross_upload_mean(vec, vec, upl, upl, same_set=True)
        if not np.isfinite(parent_within):
            continue

        llm = subs[member]
        sizes = [int(count) for count in collections.Counter(llm.tolist()).values()]
        by_caption = np.unique(captions[member], return_inverse=True)[1]

        caption_sizes = [int(count) for count
                         in collections.Counter(by_caption.tolist()).values()]

        parent_rows.append({"prototype": int(prototype), "n": int(len(member)),
                            "subprototypes": int(len(sizes)),
                            "within": parent_within})
        arms["llm"] += score_partition(vec, upl, llm, parent_within, acct)
        arms["caption"] += score_partition(vec, upl, by_caption, parent_within, acct)
        # One floor per compared arm, drawn with *that* arm's group sizes.  The
        # theory says a random split reads 1.00 whatever the sizes -- the mean
        # pairwise distance of a random subset is unbiased -- but the caption
        # arm has roughly twice the groups of the llm arm, and comparing the two
        # against a single floor would make "does group size flatter this ruler"
        # an assumption instead of a reading.  Both floors are printed; if they
        # differ, the comparison between llm and caption is not available.
        arms["random_llm"] += score_partition(
            vec, upl, split_random(len(member), sizes, rng), parent_within, acct)
        arms["random_caption"] += score_partition(
            vec, upl, split_random(len(member), caption_sizes, rng),
            parent_within, acct)
        arms["kmeans"] += score_partition(
            vec, upl, split_kmeans(vec, len(sizes), args.seed), parent_within, acct)

        # The two compared arms do not cover the same segments.  A group under
        # ``MIN_SUB`` holds too few cross-upload pairs to score and is dropped,
        # and the exact-caption partition is mostly rare captions, so it scores
        # a minority of the corpus while the llm partition scores nearly all of
        # it.  Measured on clean5b5 that is 7,249 segments against 11,566 -- so
        # "caption 0.9718 beats llm 0.9870" would be a comparison between two
        # different subsets, and the segments it silently drops are exactly the
        # unusual ones.  Everything below is re-scored on the intersection,
        # denominator included, so the two partitions answer over one corpus.
        common = _scorable(llm) & _scorable(by_caption)
        if common.sum() >= 2 * MIN_SUB:
            sub_vec, sub_upl = vec[common], upl[common]
            common_within = cross_upload_mean(sub_vec, sub_vec, sub_upl, sub_upl,
                                              same_set=True)
            if np.isfinite(common_within):
                common_sizes = [int(count) for count in collections.Counter(
                    llm[common].tolist()).values()]
                sub_acct = None if acct is None else acct[common]
                arms["llm_common"] += score_partition(
                    sub_vec, sub_upl, llm[common], common_within, sub_acct)
                arms["caption_common"] += score_partition(
                    sub_vec, sub_upl, by_caption[common], common_within, sub_acct)
                arms["random_common"] += score_partition(
                    sub_vec, sub_upl,
                    split_random(int(common.sum()), common_sizes, rng),
                    common_within, sub_acct)

    report = {"parents": len(parent_rows),
              "arms": {name: summarise(rows_) for name, rows_ in arms.items()}}

    floors = {"llm": report["arms"]["random_llm"]["segment_weighted_ratio"],
              "caption": report["arms"]["random_caption"]["segment_weighted_ratio"]}
    floor = floors["llm"]
    ceiling = report["arms"]["kmeans"]["segment_weighted_ratio"]
    for name in ("llm", "caption"):
        value = report["arms"][name]["segment_weighted_ratio"]
        span = floors[name] - ceiling
        report["arms"][name]["fraction_of_available_gap"] = (
            float((floors[name] - value) / span) if span else float("nan"))
        report["arms"][name]["size_matched_floor"] = floors[name]

    print()
    print("{:<15} {:>6} {:>8} {:>9} {:>9} {:>8} {:>10} {:>8}".format(
        "arm", "groups", "segs", "median", "weighted", ">=1.0", "gap share",
        "acct"))
    for name in ("random_llm", "random_caption", "caption", "llm", "kmeans",
                 "random_common", "caption_common", "llm_common"):
        row = report["arms"].get(name)
        if not row or not row.get("subprototypes"):
            continue
        print("{:<15} {:>6} {:>8} {:>9.4f} {:>9.4f} {:>8} {:>10} {:>8}".format(
            name, row["subprototypes"], row["segments"], row["median_ratio"],
            row["segment_weighted_ratio"], row["n_at_or_above_1"],
            "{:.3f}".format(row["fraction_of_available_gap"])
            if "fraction_of_available_gap" in row else "--",
            "{:.3f}".format(row["account_purity"])
            if "account_purity" in row else "--"))

    # The instrument check comes last so it is the line next to the verdict:
    # a random split of a set has the same expected mean pairwise distance as
    # the set, so this arm has to read 1.00 or nothing above it may be quoted.
    print()
    for name, value in sorted(floors.items()):
        if abs(value - 1.0) > 0.02:
            print("INSTRUMENT_FAIL the random split at {} sizes reads {:.4f}, "
                  "not 1.00 -- the denominator or the pairing is wrong".format(
                      name, value), flush=True)
            return 1
    print("instrument ok: random split reads {:.4f} at llm sizes and {:.4f} at "
          "caption sizes (expected 1.00 for both, and equal to each other)"
          .format(floors["llm"], floors["caption"]))
    if ceiling >= floor:
        print("CEILING_FAIL k-means in TMR is not tighter than random ({:.4f} vs "
              "{:.4f}); there is no gap for M3 to recover and no reading below "
              "is interpretable".format(ceiling, floor), flush=True)
        return 1
    print("ceiling: k-means in TMR reaches {:.4f}".format(ceiling))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
