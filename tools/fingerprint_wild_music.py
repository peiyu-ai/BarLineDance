#!/usr/bin/env python3
"""Recover a backing-track identity for the wild corpus, from features we have.

AIST++ records which song each recording was danced to.  TikTok does not, and
that absence is load-bearing in three places:

* ``assign_account_disjoint_split`` can prove the split is account-disjoint and
  cannot say whether the same track is danced on both sides of it;
* ``eval_r_precision --exclude-same-music`` falls back to the *upload* as the
  audio group, which is sound but a floor -- two uploads on one track stay in
  each other's candidate pools;
* MultiModality has no ground-truth ceiling on this corpus, because the ceiling
  is "several dances to one track" and we cannot currently name one.

The obvious identity is already in the bundle and is useless: ``music_sha256``
hashes the clip's own 35-D array, so it is unique per clip by construction
(13,781 distinct over 13,783 clips).  Two cuts of one song at different offsets
hash differently.  It answers "is this the same file", not "is this the same
track".

What is usable is inside the 35-D array itself.  The released extractor is
``[envelope(1) | mfcc(20) | chroma_cens(12) | peak(1) | beat(1)]``, and
chroma CENS is the standard cover/version-identification feature: it is
normalised for loudness and smoothed over time, so it tracks harmony rather
than production.  So no audio is fetched here at all -- the 408 GiB ingest tree
stays untouched and this runs on the music features the corpus already
published.

Two stages, because they answer two different questions
-------------------------------------------------------
**Stage 1, profile.**  Time-averaged CENS (harmony) and MFCC (timbre) per clip,
compared by cosine.  Cheap enough for all pairs, and it is a *blocking* step:
it proposes candidates and decides nothing.  Two different songs in the same
key with the same production sit close here.

**Stage 2, alignment.**  For each candidate, the peak of the lag-wise mean
cosine between the two CENS sequences.  Two cuts of one track that overlap in
time align at a lag; two different tracks do not.  This is the verdict.

Both thresholds come from a calibration that can fail, not from taste
------------------------------------------------------------------
The corpus supplies its own labelled pairs, at no cost and with no circularity:

* **positive, overlapping** -- two windows cut from *one clip's own* feature
  array with a known offset.  Same track, same recording, known lag.  Stage 2
  must find these.
* **positive, disjoint** -- the first and last thirds of one clip.  Same track,
  no shared content.  This is the case stage 2 *cannot* see and stage 1 must
  carry, and reporting its rate is how the tool states its own ceiling.
* **negative** -- windows from clips of different accounts.  Two dancers on the
  same viral sound are a real possibility, so a handful of these scoring high
  is expected rather than alarming, and the report prints the score
  distribution instead of asserting a clean separation.

The operating point is chosen as the score that holds negatives at or under
``--max-false-positive-rate``, and the achieved rates ride in the report.  A
threshold picked after looking at how many cross-split matches it produces
would be a threshold tuned to the answer, so the calibration runs first and its
output is written before any corpus pair is scored.

Usage::

    python3 tools/fingerprint_wild_music.py calibrate \\
        --bundle data/wild3d/wild_v4_performance \\
        --report runs/wild_v4_music_calibration.json

    python3 tools/fingerprint_wild_music.py group \\
        --bundle data/wild3d/wild_v4_performance \\
        --calibration runs/wild_v4_music_calibration.json \\
        --output runs/wild_v4_music_groups.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# The released 35-D layout, restated here because reading the wrong slice would
# silently fingerprint the onset envelope and still produce a plausible number.
ENVELOPE = slice(0, 1)
MFCC = slice(1, 21)
CHROMA = slice(21, 33)
PEAK = slice(33, 34)
BEAT = slice(34, 35)
FEATURE_DIM = 35
FPS = 30.0

SCHEMA_VERSION = "atomicdance-wild-music-fingerprint-v1"


class FingerprintError(RuntimeError):
    pass


def load_sequences(bundle: pathlib.Path) -> List[Dict[str, Any]]:
    path = bundle / "sequences.jsonl"
    if not path.is_file():
        raise FingerprintError("missing {}".format(path))
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_music(bundle: pathlib.Path, row: Dict[str, Any]) -> np.ndarray:
    array = np.load(bundle / row["music_path"])
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM:
        raise FingerprintError("{} is not a {}-D music array: {}".format(
            row.get("recording_id"), FEATURE_DIM, array.shape))
    return np.asarray(array, dtype=np.float32)


def _unit(rows: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(rows, axis=axis, keepdims=True)
    return rows / np.maximum(norm, 1e-8)


def profile(music: np.ndarray) -> np.ndarray:
    """One vector per clip: harmony over time, then timbre over time.

    Both halves are unit-normalised *separately* before being concatenated, so
    a clip with loud percussion cannot let its MFCC magnitude outvote the
    chroma; the cosine that follows would otherwise be a timbre match wearing
    the name of a track match.
    """
    chroma = _unit(np.asarray(music[:, CHROMA]).mean(axis=0), axis=0)
    mfcc = np.asarray(music[:, MFCC]).mean(axis=0)
    mfcc = _unit(mfcc - mfcc.mean(), axis=0)
    return np.concatenate([chroma, mfcc]).astype(np.float32)


def align_chroma(a: np.ndarray, b: np.ndarray, *, center: Optional[np.ndarray] = None,
                 min_overlap: int = 45) -> Tuple[float, int]:
    """Peak lag-wise mean cosine between two CENS sequences, and its lag.

    Frames are unit-normalised first, so the per-frame product is a cosine and
    the mean over an overlap is comparable between overlaps of different
    lengths.  Overlaps shorter than ``min_overlap`` (1.5 s at 30 fps) are not
    scored at all: a two-frame overlap can hit 1.0 on any two clips, and a
    maximum taken over such lags is a maximum over noise.

    ``center`` is the corpus-mean chroma vector, and passing it is what makes
    this score discriminative rather than merely high.  CENS frames are
    non-negative and smoothed, so any two clips share a large common component:
    measured on wild_v4, different-upload pairs score a **median 0.853 and a
    maximum 0.972** raw, which leaves an operating point that only re-finds
    byte-identical audio -- exactly what ``music_sha256`` already does for free.
    Subtracting the corpus mean first drops those same negatives to a median
    0.235 and a maximum 0.854 while overlapping positives stay at 1.000.  Both
    stages must use the *same* centre, which is why it is written into the
    calibration report rather than recomputed per call.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if center is not None:
        a = a - center
        b = b - center
    a = _unit(a)
    b = _unit(b)
    if len(a) < min_overlap or len(b) < min_overlap:
        return 0.0, 0
    best, best_lag = -1.0, 0
    for lag in range(-(len(b) - min_overlap), len(a) - min_overlap + 1):
        a0 = max(0, lag)
        b0 = max(0, -lag)
        width = min(len(a) - a0, len(b) - b0)
        if width < min_overlap:
            continue
        score = float(np.einsum("ij,ij->", a[a0:a0 + width], b[b0:b0 + width]) / width)
        if score > best:
            best, best_lag = score, lag
    return best, best_lag


def alignment_score(left: np.ndarray, right: np.ndarray, *,
                    center: Optional[np.ndarray] = None,
                    min_overlap: int = 45) -> Tuple[float, int]:
    """``align_chroma`` on two full 35-D arrays, so callers cannot mis-slice."""
    return align_chroma(np.asarray(left[:, CHROMA]), np.asarray(right[:, CHROMA]),
                        center=center, min_overlap=min_overlap)


def _windows_for_calibration(music: np.ndarray, rng: np.random.Generator
                             ) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Overlapping and disjoint window pairs cut from one clip's own features."""
    frames = len(music)
    if frames < 300:
        return None
    third = frames // 3
    width = max(150, third)
    offset = int(rng.integers(width // 4, width // 2))
    if offset + width > frames:
        return None
    return {
        "overlapping": (music[:width], music[offset:offset + width]),
        "disjoint": (music[:third], music[-third:]),
    }


def cmd_calibrate(args: argparse.Namespace) -> int:
    bundle = pathlib.Path(args.bundle)
    rows = load_sequences(bundle)
    rng = np.random.default_rng(args.seed)
    picked = rng.permutation(len(rows))[:args.sample]

    positives_overlap: List[float] = []
    positives_disjoint_profile: List[float] = []
    positives_disjoint_align: List[float] = []
    negatives_align: List[float] = []
    negatives_profile: List[float] = []
    cache: List[Tuple[str, np.ndarray]] = []

    # Two passes, because the centre has to exist before any score does.  A
    # centre computed from the pairs being scored would move with them.
    for index in picked:
        row = rows[int(index)]
        music = read_music(bundle, row)
        if _windows_for_calibration(music, rng) is None:
            continue
        cache.append((str(row["recording_id"]), music))
    if not cache:
        raise FingerprintError("no sampled clip is long enough to cut calibration windows from")
    center = np.mean(np.concatenate([np.asarray(m[:, CHROMA]) for _, m in cache]),
                     axis=0).astype(np.float32)

    for _, music in cache:
        pair = _windows_for_calibration(music, rng)
        if pair is None:
            continue
        left, right = pair["overlapping"]
        positives_overlap.append(alignment_score(left, right, center=center)[0])
        left, right = pair["disjoint"]
        positives_disjoint_align.append(alignment_score(left, right, center=center)[0])
        positives_disjoint_profile.append(float(profile(left) @ profile(right)))

    # Negatives: different uploads, so no shared track by construction of the
    # corpus -- except where two uploads really do share one, which is the
    # thing being measured and the reason the rate is reported rather than
    # assumed to be zero.
    for _ in range(args.negatives):
        i, j = rng.integers(0, len(cache), size=2)
        if i == j:
            continue
        name_i, music_i = cache[int(i)]
        name_j, music_j = cache[int(j)]
        if name_i.split(":")[1] == name_j.split(":")[1]:
            continue
        negatives_align.append(alignment_score(music_i, music_j, center=center)[0])
        negatives_profile.append(float(profile(music_i) @ profile(music_j)))

    def quantiles(values: Sequence[float]) -> Dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            return {}
        return {name: round(float(np.quantile(array, q)), 4)
                for name, q in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95),
                                ("p99", 0.99), ("max", 1.0))}

    negatives = np.asarray(negatives_align, dtype=np.float64)
    threshold = float(np.quantile(negatives, 1.0 - args.max_false_positive_rate))
    recall = float(np.mean(np.asarray(positives_overlap) >= threshold))
    achieved_fpr = float(np.mean(negatives >= threshold))

    profile_negatives = np.asarray(negatives_profile, dtype=np.float64)
    profile_threshold = float(np.quantile(profile_negatives, 1.0 - args.block_rate))

    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "calibrate",
        "bundle": str(bundle.resolve()),
        "sample_clips": len(cache),
        "operating_point": {
            "alignment_threshold": round(threshold, 4),
            "profile_block_threshold": round(profile_threshold, 4),
            "chosen_by": ("the alignment score that holds different-upload pairs at "
                          "--max-false-positive-rate; picked before any corpus pair "
                          "was scored"),
            "max_false_positive_rate": args.max_false_positive_rate,
            "achieved_false_positive_rate": round(achieved_fpr, 5),
            "recall_on_overlapping_positives": round(recall, 4),
            # Part of the operating point, not a note about it: scoring with a
            # different centre than the one these thresholds were drawn against
            # is scoring a different quantity.
            "chroma_center": [round(float(value), 6) for value in center],
        },
        "scores": {
            "positive_overlapping_alignment": quantiles(positives_overlap),
            "positive_disjoint_alignment": quantiles(positives_disjoint_align),
            "positive_disjoint_profile": quantiles(positives_disjoint_profile),
            "negative_alignment": quantiles(negatives_align),
            "negative_profile": quantiles(negatives_profile),
        },
        "what_this_cannot_do": (
            "positive_disjoint_alignment is the ceiling: two cuts of one track that "
            "share no audio cannot be aligned, so this tool links overlapping "
            "excerpts and re-uploads, not every appearance of a track. Its output is "
            "therefore a lower bound on shared music, which is the useful direction "
            "for a leakage check and the wrong one for a completeness claim."
        ),
    }
    path = pathlib.Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_group(args: argparse.Namespace) -> int:
    bundle = pathlib.Path(args.bundle)
    calibration = json.loads(pathlib.Path(args.calibration).read_text(encoding="utf-8"))
    point = calibration["operating_point"]
    align_threshold = float(point["alignment_threshold"])
    block_threshold = float(point["profile_block_threshold"])
    if "chroma_center" not in point:
        raise FingerprintError(
            "{} predates the corpus-centred score and carries no chroma_center; "
            "re-run calibrate rather than scoring with an implicit centre of "
            "zero, which is a different quantity from the one it thresholded"
            .format(args.calibration))
    center = np.asarray(point["chroma_center"], dtype=np.float32)

    rows = load_sequences(bundle)
    names = [str(row["recording_id"]) for row in rows]
    split_of = {str(row["recording_id"]): str(row.get("split")) for row in rows}
    profiles = np.zeros((len(rows), 32), dtype=np.float32)
    music: List[np.ndarray] = []
    for index, row in enumerate(rows):
        array = read_music(bundle, row)
        profiles[index] = profile(array)
        music.append(np.asarray(array[:, CHROMA], dtype=np.float32))
        if (index + 1) % 2000 == 0:
            print("  profiled {}/{}".format(index + 1, len(rows)), flush=True)

    # Blocking, in blocks: the full 13,783^2 similarity is 190M floats and does
    # not need to exist at once.
    candidates: List[Tuple[int, int]] = []
    block = args.block_size
    for start in range(0, len(rows), block):
        stop = min(start + block, len(rows))
        similarity = profiles[start:stop] @ profiles.T
        for local, absolute in enumerate(range(start, stop)):
            hits = np.nonzero(similarity[local] >= block_threshold)[0]
            for other in hits:
                if int(other) > absolute:
                    candidates.append((absolute, int(other)))
        print("  blocked {}/{}, {} candidate pair(s)".format(
            stop, len(rows), len(candidates)), flush=True)
        if len(candidates) > args.max_candidates:
            raise FingerprintError(
                "blocking produced over {} candidates; the profile threshold from "
                "{} is too loose for this corpus".format(args.max_candidates, args.calibration))

    parent = list(range(len(rows)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    matched: List[Dict[str, Any]] = []
    for left, right in candidates:
        score, lag = align_chroma(music[left], music[right], center=center)
        if score < align_threshold:
            continue
        matched.append({"left": names[left], "right": names[right],
                        "score": round(score, 4), "lag_frames": int(lag),
                        "splits": sorted({split_of[names[left]], split_of[names[right]]})})
        a, b = find(left), find(right)
        if a != b:
            parent[a] = b

    groups: Dict[int, List[str]] = {}
    for index, name in enumerate(names):
        groups.setdefault(find(index), []).append(name)
    multi = {root: members for root, members in groups.items() if len(members) > 1}
    crossing = [pair for pair in matched if len(pair["splits"]) > 1]

    # A positive control the calibration cannot supply: two cuts of one upload
    # are the same track by construction, so the share of those pairs this
    # verifies is recall on *real* pairs rather than on windows cut from one
    # array.  It is a floor on recall for the same reason the tool is a floor
    # overall -- consecutive cuts often share no audio at all.
    upload_of = {name: name.split(":")[1] for name in names}
    same_upload_candidates = [pair for pair in candidates
                              if upload_of[names[pair[0]]] == upload_of[names[pair[1]]]]
    verified_pairs = {(pair["left"], pair["right"]) for pair in matched}
    same_upload_verified = sum(
        1 for left, right in same_upload_candidates
        if (names[left], names[right]) in verified_pairs)

    # Connected components chain: a --- b --- c links a and c even when a and c
    # never scored above threshold, and on a corpus of viral sounds that merges
    # whole neighbourhoods.  Measured rather than assumed, and the grouping is
    # marked unusable as a track id when it happens, because a 2,335-clip
    # "track" published as one would put 17% of the corpus under one music key
    # and every same-music control built on it would be wrong in the direction
    # that flatters the model.
    largest = max((len(members) for members in multi.values()), default=0)
    chains = largest > args.max_track_fraction * len(rows)

    track_of = {}
    for order, (root, members) in enumerate(sorted(multi.items())):
        for member in members:
            track_of[member] = "wildtrack{:06d}".format(order)

    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": "group",
        "bundle": str(bundle.resolve()),
        "calibration": str(pathlib.Path(args.calibration).resolve()),
        "operating_point": point,
        "clips": len(rows),
        "candidate_pairs": len(candidates),
        "verified_pairs": len(matched),
        "clips_in_a_multi_clip_track": sum(len(m) for m in multi.values()),
        "multi_clip_tracks": len(multi),
        "pairs_crossing_a_split": len(crossing),
        "crossing_examples": crossing[:20],
        "positive_control": {
            "same_upload_candidate_pairs": len(same_upload_candidates),
            "same_upload_verified": same_upload_verified,
            "recall_on_same_upload_pairs": round(
                same_upload_verified / len(same_upload_candidates), 4)
            if same_upload_candidates else None,
            "what_it_is": ("two cuts of one upload are the same track by construction; "
                           "this is recall on real pairs, and a floor on it, because "
                           "consecutive cuts often share no audio to align"),
        },
        "transitive_closure": {
            "largest_track_clips": largest,
            "largest_track_fraction": round(largest / len(rows), 4),
            "chains": bool(chains),
            "track_of_is_usable_as_a_music_key": not chains,
            "why": ("components are single-linkage: a--b--c groups a with c even when "
                    "that pair never scored above threshold. When this chains, the "
                    "pair list is the finding and the grouping is not a track id"),
        },
        "track_of": {} if chains else track_of,
        "reading": (
            "verified pairs are a lower bound on shared music: alignment cannot link "
            "two cuts of one track that share no audio. A zero here means no *provable* "
            "leak, not no leak."
        ),
    }
    pairs_path = pathlib.Path(args.output).with_name(
        pathlib.Path(args.output).stem + "_pairs.jsonl")
    with pairs_path.open("w", encoding="utf-8") as handle:
        for pair in matched:
            handle.write(json.dumps(pair, sort_keys=True) + "\n")
    report["verified_pairs_path"] = str(pairs_path)
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")
    summary = {key: report[key] for key in
               ("clips", "candidate_pairs", "verified_pairs", "multi_clip_tracks",
                "clips_in_a_multi_clip_track", "pairs_crossing_a_split")}
    summary["positive_control"] = report["positive_control"]
    summary["transitive_closure"] = report["transitive_closure"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="pick both thresholds from self-supervised pairs")
    calibrate.add_argument("--bundle", required=True)
    calibrate.add_argument("--report", required=True)
    calibrate.add_argument("--sample", type=int, default=400)
    calibrate.add_argument("--negatives", type=int, default=2000)
    calibrate.add_argument("--max-false-positive-rate", type=float, default=0.001)
    calibrate.add_argument("--block-rate", type=float, default=0.002,
                           help="fraction of different-upload pairs the profile stage "
                                "may pass to the alignment stage")
    calibrate.add_argument("--seed", type=int, default=20260816)
    calibrate.set_defaults(func=cmd_calibrate)

    group = sub.add_parser("group", help="link clips into tracks and report split crossings")
    group.add_argument("--bundle", required=True)
    group.add_argument("--calibration", required=True)
    group.add_argument("--output", required=True)
    group.add_argument("--block-size", type=int, default=512)
    group.add_argument("--max-candidates", type=int, default=5_000_000)
    group.add_argument("--max-track-fraction", type=float, default=0.02,
                       help="a component larger than this share of the corpus means "
                            "single-linkage chaining, and the grouping is published "
                            "as unusable rather than as a track id")
    group.set_defaults(func=cmd_group)
    return parser


def main(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except FingerprintError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
