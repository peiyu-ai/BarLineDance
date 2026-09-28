#!/usr/bin/env python3
"""M6b: R-precision, the paper's own structure-consistency metric.

This is not HumanML3D's text-motion R-precision, and using that one instead
would silently answer a different question.  The paper proposes its own and
defines it in one sentence:

    "To evaluate the structural consistency of the dance generated, we propose
    R-precision.  We split the music into segments and calculate the music
    features of each clip.  We examine whether the motion feature of the dances
    in every pair of the most-similar-music-clip is among the three highest
    most-similar-dance-clip."

So it asks one thing: when two pieces of music are the most similar pair in the
corpus, are the dances on them among each other's most similar dances?  It needs
no text encoder and no external retrieval model -- only paired music and motion
features, both of which this repo already produces.

It is the column this repo has been missing, and it is the column that decides
M3.  Tab. 2 prices the LLM re-clustering row at R 26.6 against 23.3 without it,
and the audit gate this repo does have (motion-space coherence) rejects the LLM
row.  Neither of those settles the question; R does, because it is the number
the paper actually claims.

Three parameters the sentence does not fix, and how each is handled:

* **Clip length.**  Not stated.  Calibrated against the one observable the paper
  publishes for this metric -- ground truth R = 42.1 on AIST++ (Tab. 4, Tab. 5)
  -- exactly as L_min was calibrated against Fig. 4a.  ``--calibrate`` sweeps it
  and reports the distance to 42.1 so the choice is visible rather than tuned in
  private.
* **Candidate pool.**  Every clip in the evaluated split, which is the reading
  that makes "the most-similar-music-clip" well defined.  Pool size changes the
  number outright, so it is reported alongside R together with the chance rate
  ``top_k / (pool - 1)``; an R quoted without its pool size is not comparable to
  anything.
* **Similarity.**  Features are standardised per dimension and compared by
  Euclidean distance, matching how ``eval/metrics.py`` already treats these same
  feature spaces for FID and diversity.

One trap the definition walks into, measured rather than argued: consecutive
clips of the *same* song are both musically and motionally near-identical, so
they can supply a large share of the hits without the metric having learned
anything about music-motion structure.  ``--exclude-same-sequence`` removes
them, and both numbers are always reported.  If R collapses when neighbours are
excluded, then R was mostly measuring adjacency, and that is a fact about the
metric worth publishing next to it.

Usage:
    python3 tools/eval_r_precision.py \\
        --release data/wild3d/wild_v2_release_nollm --split test \\
        --output runs/r_precision_wild_test.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

PAPER_GROUND_TRUTH_R = 42.1
TOP_K = 3
FPS = 30.0


class RPrecisionError(RuntimeError):
    """Raised when the inputs cannot support an honest R-precision."""


def standardise(features: np.ndarray) -> np.ndarray:
    """Z-score each dimension, leaving constant dimensions at zero.

    A constant dimension carries no information about which clips resemble each
    other; dividing it by its own (zero) spread would turn floating point dust
    into the loudest signal in the space.
    """
    features = np.asarray(features, dtype=np.float64)
    spread = features.std(axis=0)
    spread = np.where(spread < 1e-10, 1.0, spread)
    return (features - features.mean(axis=0)) / spread


def pairwise_distances(features: np.ndarray) -> np.ndarray:
    """Euclidean distances, computed by expansion rather than broadcasting.

    A pool of a few thousand clips times a few hundred dimensions is a tensor
    that does not need to exist: ``|a-b|^2 = |a|^2 + |b|^2 - 2ab`` keeps it to
    one Gram matrix.  Negatives from cancellation are clipped before the square
    root, which is the only place they can appear.
    """
    square = (features ** 2).sum(axis=1)
    gram = features @ features.T
    squared = square[:, None] + square[None, :] - 2.0 * gram
    return np.sqrt(np.maximum(squared, 0.0))


def r_precision(music: np.ndarray, motion: np.ndarray, *, top_k: int = TOP_K,
                groups: Optional[Sequence[str]] = None,
                exclude_same_group: bool = False,
                exclude_pairs: Optional[Sequence[FrozenSet[str]]] = None,
                pair_keys: Optional[Sequence[str]] = None,
                keep_ranks: bool = False) -> Dict[str, object]:
    """The paper's R-precision over paired music and motion clip features.

    For every clip the nearest *other* clip in music space is found, and the
    metric asks whether that same clip is within the ``top_k`` nearest in motion
    space.  Ties in motion space are resolved against the metric -- a candidate
    tied at the boundary counts as outside -- because the alternative silently
    inflates R on degenerate features that make every distance equal.

    Two ways to remove a candidate, and they are not interchangeable.
    ``exclude_same_group`` removes every clip sharing a key, which is the right
    shape when the key *is* the shared thing (a sequence, an AIST track, a wild
    upload).  ``exclude_pairs`` removes named pairs, which is the right shape
    when what is known is "these two share audio" and no key groups them --
    exactly the output of an audio fingerprint, whose components chain under
    single linkage and therefore make a bad key while making a sound pair list.
    Both write into the same eligibility matrix and compose.
    """
    music = np.asarray(music, dtype=np.float64)
    motion = np.asarray(motion, dtype=np.float64)
    if music.ndim != 2 or motion.ndim != 2:
        raise RPrecisionError("music and motion features must both be matrices")
    if len(music) != len(motion):
        raise RPrecisionError("{} music clips against {} motion clips: R-precision "
                              "is defined on pairs".format(len(music), len(motion)))
    count = len(music)
    if count < top_k + 2:
        raise RPrecisionError("{} clips cannot support top-{} retrieval".format(count, top_k))
    if exclude_same_group and groups is None:
        raise RPrecisionError("--exclude-same-sequence needs a group per clip")

    music_distance = pairwise_distances(standardise(music))
    motion_distance = pairwise_distances(standardise(motion))

    eligible = np.ones((count, count), dtype=bool)
    np.fill_diagonal(eligible, False)
    if exclude_same_group:
        keys = np.asarray([str(group) for group in groups])
        eligible &= keys[:, None] != keys[None, :]
    applied_pairs = 0
    if exclude_pairs:
        if pair_keys is None:
            raise RPrecisionError("exclude_pairs needs a key per clip to match against")
        if len(pair_keys) != count:
            raise RPrecisionError("{} pair keys for {} clips".format(len(pair_keys), count))
        rows_by_key: Dict[str, List[int]] = {}
        for row, key in enumerate(pair_keys):
            rows_by_key.setdefault(str(key), []).append(row)
        for pair in exclude_pairs:
            left, right = sorted(pair)
            left_rows = rows_by_key.get(left)
            right_rows = rows_by_key.get(right)
            # A pair naming a clip this split does not hold is not an error: the
            # fingerprint runs over the whole corpus, so most of its pairs live
            # in train.  Silently applying *none* of them would be the error,
            # which is why the applied count is returned and the caller refuses
            # a control that touched nothing.
            if not left_rows or not right_rows:
                continue
            eligible[np.ix_(left_rows, right_rows)] = False
            eligible[np.ix_(right_rows, left_rows)] = False
            applied_pairs += 1

    hits, considered, pool_sizes, ranks = 0, 0, [], []
    for index in range(count):
        candidates = np.flatnonzero(eligible[index])
        # A clip whose only companions are its own siblings has no comparison to
        # make.  Counting it as a miss would report the corpus's shape as the
        # model's failure, so it is dropped and the drop is reported.
        if len(candidates) < top_k + 1:
            continue
        pool_sizes.append(len(candidates))
        nearest_music = candidates[np.argmin(music_distance[index, candidates])]
        # The count includes the candidate itself, so an unrivalled nearest
        # motion clip scores 1.  Anything tied with it is counted too, which is
        # what resolves ties against the metric: if more than top_k candidates
        # are at most this far away, this one is not reliably inside the cut.
        rank = int((motion_distance[index, candidates]
                    <= motion_distance[index, nearest_music]).sum())
        if rank <= top_k:
            hits += 1
        ranks.append((rank - 1) / max(len(candidates) - 1, 1))
        considered += 1
    if considered == 0:
        raise RPrecisionError("no clip had enough candidates to score")

    mean_pool = float(np.mean(pool_sizes))
    value = 100.0 * hits / considered
    chance = 100.0 * top_k / mean_pool
    # R is a top-3 indicator, so on a pool of a few hundred it resolves only a
    # handful of hits and two conditions a few hits apart are indistinguishable.
    # The mean normalised rank of the same music-nearest neighbour uses every
    # comparison rather than three of them: 0.5 is chance, lower is better, and
    # it separates conditions that R reports as identical.  It is a diagnostic,
    # not a substitute -- the paper's number is R.
    rank_array = np.asarray(ranks, dtype=np.float64)
    return {
        "R": round(value, 2),
        "hits": int(hits),
        "mean_normalised_rank": round(float(rank_array.mean()), 4),
        "mean_normalised_rank_stderr": round(
            float(rank_array.std(ddof=1) / np.sqrt(len(rank_array)))
            if len(rank_array) > 1 else float("nan"), 4),
        "rank_chance": 0.5,
        "scored_clips": int(considered),
        "clips": int(count),
        "dropped_for_small_pool": int(count - considered),
        "mean_candidate_pool": round(mean_pool, 1),
        "chance_R": round(chance, 2),
        "lift_over_chance": round(value / chance, 2) if chance > 0 else None,
        "top_k": int(top_k),
        "excluded_same_sequence": bool(exclude_same_group),
        # Zero here with a pair list supplied means the list named no two clips
        # this split holds -- a control that ran and removed nothing, which is
        # indistinguishable in R from one that was never asked for.  Reported
        # rather than inferred, and the caller refuses it.
        "excluded_verified_pairs": int(applied_pairs),
        # Off by default because the pool sweep calls this hundreds of times and
        # would carry a list per repeat.  On, it is what makes two conditions
        # comparable *per clip*: they are scored on the same clips against the
        # same music, so their difference is paired, and a paired difference is
        # an order of magnitude tighter than two means each carrying their own
        # standard error.  Comparing the means is how two conditions look
        # indistinguishable when they are not -- and how they can look
        # distinguishable when the difference is one unusual clip.
        **({"normalised_ranks": [round(float(r), 6) for r in rank_array]}
           if keep_ranks else {}),
    }


def r_precision_at_pool(music: np.ndarray, motion: np.ndarray, *, pool_size: int,
                        repeats: int = 20, seed: int = 20260811, top_k: int = TOP_K,
                        groups: Optional[Sequence[str]] = None,
                        exclude_same_group: bool = False) -> Dict[str, object]:
    """R-precision on random sub-pools of a fixed size.

    R is not comparable across corpora at different pool sizes, and this is not a
    small effect: on the wild test split the same clips score 33.3 in a pool of
    60 and 12.8 in a pool of 690, because "among the three nearest" is a far
    harder target when there are ten times as many candidates to beat.  Neither
    is the lift over chance invariant -- it moves the other way, since a larger
    pool also supplies a genuinely better nearest-music match.

    So there is no pool-independent form of this number, and the only honest way
    to compare against a published R is to evaluate at the same pool size.  The
    paper does not state its pool, so whatever is assumed for it must be stated
    explicitly alongside any comparison; this function makes that assumption a
    parameter instead of an accident of how much data a split happens to hold.

    Sub-pools are drawn without replacement and repeated, so the spread across
    draws is reported too: a mean quoted without it would hide that a small pool
    is also a noisy one.
    """
    count = len(music)
    if pool_size > count:
        raise RPrecisionError("cannot draw a pool of {} from {} clips".format(pool_size, count))
    if pool_size < top_k + 2:
        raise RPrecisionError("a pool of {} cannot support top-{} retrieval".format(
            pool_size, top_k))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        picked = rng.choice(count, size=pool_size, replace=False)
        subset_groups = [groups[index] for index in picked] if groups is not None else None
        try:
            values.append(r_precision(music[picked], motion[picked], top_k=top_k,
                                      groups=subset_groups,
                                      exclude_same_group=exclude_same_group)["R"])
        except RPrecisionError:
            # A draw where exclusion leaves nothing scoreable is a property of
            # that draw, not a failure of the run; it is counted, not fatal.
            continue
    if not values:
        raise RPrecisionError("no sub-pool of {} could be scored".format(pool_size))
    array = np.asarray(values, dtype=np.float64)
    return {
        "pool_size": int(pool_size),
        "draws_scored": int(len(values)),
        "draws_requested": int(repeats),
        "R_mean": round(float(array.mean()), 2),
        "R_std": round(float(array.std()), 2),
        "R_min": round(float(array.min()), 2),
        "R_max": round(float(array.max()), 2),
        "chance_R": round(100.0 * top_k / (pool_size - 1), 2),
        "excluded_same_sequence": bool(exclude_same_group),
        "seed": int(seed),
    }


def unnormalize(motion: np.ndarray, normalizer_path: pathlib.Path) -> np.ndarray:
    """Invert the release's min-max normalisation, exactly as inference does.

    Kinetic and manual features need a real body: metres, and rot6d columns that
    still describe a rotation.  Min-max scaled 151-D has neither, so features
    taken from it would describe a body that does not exist.  This is the same
    conclusion the label pipeline reached -- bind against the normalized
    manifest, compute on the inverted array.
    """
    import torch

    payload = torch.load(str(normalizer_path), map_location="cpu", weights_only=False)
    if set(payload) != {"data_min", "data_max"}:
        raise RPrecisionError("normalizer.pt must hold exactly data_min and data_max")
    data_min = payload["data_min"].float()
    data_max = payload["data_max"].float()
    safe_range = torch.where(data_max == data_min,
                             torch.ones_like(data_max - data_min), data_max - data_min)
    # np.array rather than asarray: the caller usually hands over a slice of a
    # memory-mapped release, which is read-only, and torch refuses to share
    # storage with it.
    tensor = torch.as_tensor(np.array(motion, dtype=np.float32), dtype=torch.float32)
    return ((tensor + 1.0) * safe_range / 2.0 + data_min).numpy()


def motion_clip_feature(joints: np.ndarray, kind: str) -> np.ndarray:
    """Kinetic or manual features for one clip of joint positions.

    Positions arrive z-up from this repo's ``SMPLSkeleton`` while the upstream
    extractors assume y-up -- ``KineticFeatures`` splits energy into horizontal
    and vertical against that axis, so feeding z-up data swaps two of its three
    channels per joint.  The conversion is explicit here rather than assumed.

    The clip is also re-centred on its own first root position, matching
    ``eval/extract_aist_features.py``: without it, the dominant term in the
    distance is where in the world the dancer happened to be standing.
    """
    from eval.utils.kinetic import extract_kinetic_features
    from eval.utils.manual import extract_manual_features
    from tools.convert_motion_to_guofeats import zup_to_humanml3d

    positions = zup_to_humanml3d(np.asarray(joints, dtype=np.float64))
    positions = positions - positions[:1, :1]
    if kind == "kinetic":
        return np.asarray(extract_kinetic_features(positions), dtype=np.float64)
    if kind == "manual":
        return np.asarray(extract_manual_features(positions), dtype=np.float64)
    raise RPrecisionError("unknown motion feature space {}".format(kind))


def music_clip_feature(music: np.ndarray, aggregate: str) -> np.ndarray:
    """Summarise a clip's per-frame 35-D music features.

    ``mean_std`` is the default because a clip's musical character is not only
    its average: two clips can share a mean spectrum and differ entirely in how
    much they move within it, and that variation is what "similar phrasing or
    rhythm" refers to.  ``mean`` is kept so the choice can be shown to matter or
    not rather than asserted.
    """
    music = np.asarray(music, dtype=np.float64)
    if aggregate == "mean":
        return music.mean(axis=0)
    if aggregate == "mean_std":
        return np.concatenate([music.mean(axis=0), music.std(axis=0)])
    raise RPrecisionError("unknown music aggregation {}".format(aggregate))


def split_into_clips(window_length: int, clip_frames: int) -> List[Tuple[int, int]]:
    """Cut a window into whole clips, refusing a remainder rather than keeping it.

    A trailing part-clip has fewer frames than the rest, and every feature here
    is an average over time, so a short clip is not a shorter observation of the
    same quantity -- it is a noisier one that would sit in the same pool as if it
    were comparable.
    """
    if clip_frames < 2:
        raise RPrecisionError("a clip needs at least two frames to have motion")
    if clip_frames > window_length:
        raise RPrecisionError("clip of {} frames does not fit a {}-frame window".format(
            clip_frames, window_length))
    count = window_length // clip_frames
    return [(index * clip_frames, (index + 1) * clip_frames) for index in range(count)]


def sequence_key(name: str) -> str:
    """The recording a window came from, so siblings can be excluded together."""
    base = str(name)
    marker = base.rfind("_slice")
    if marker > 0 and base[marker + len("_slice"):].isdigit():
        return base[:marker]
    return base


# ``wild_v4:7316544444038237474:clip003`` -- corpus, upload, cut.  Also matched
# in the ``<upload>__clip003`` form the source-video names use.
WILD_CLIP = re.compile(r"^(?P<corpus>[A-Za-z][A-Za-z0-9]*(?:_v\d+)?):(?P<upload>\d+):clip(?P<cut>\d+)$")
WILD_LEGACY_CLIP = re.compile(r"^(?P<upload>\d+)__clip(?P<cut>\d+)$")


def music_key(name: str) -> Optional[str]:
    """The group of dances that provably share audio, or None when unknowable.

    Excluding same-*sequence* pairs still leaves same-*song* pairs in the pool,
    and for generated motion that is not a small residue: re-sampling one song
    under several seeds produces sequences whose music features are identical
    bit for bit, so the nearest music clip is a sibling seed by construction and
    R would report re-sampling stability as music-motion structure.  The same
    hole is open on ground truth, where AIST records one song with several
    dancers.  Only the id is parsed, and unparseable names are reported rather
    than silently grouped into one bucket.

    On AIST the group is a track id, because the corpus records one.  On the
    wild corpus **it is the upload**, and the two are not the same claim:

    * clips of one upload are cuts of one video, so they carry one backing
      track by construction -- that direction is certain;
    * two uploads may still dance to the same track, and this corpus has no
      track id and no audio fingerprint, so those pairs are *invisible here,
      not absent*.

    So a wild exclusion built on this key is a **floor**: it removes every pair
    we can prove shares audio and leaves the ones we cannot see, which makes the
    resulting R an upper bound rather than a measurement.  Callers report it
    that way.  Returning None instead -- the behaviour before 2026-08-16, on the
    reasoning that a wild clip carries no music id -- was right about the id and
    wrong about the exclusion: it left the *provable* same-track pairs sitting
    in the pool, which is the artifact the flag exists to remove.  A grouping
    key is not a track name; it only has to be sound in the direction it is
    used.
    """
    text = str(name)
    for field in text.split("_"):
        if (len(field) == 4 and field[0] == "m"
                and field[1:3].isalpha() and field[1:3].isupper() and field[3].isdigit()):
            return field
    # A generated sample is named for what it was generated from plus its seed
    # (``<clip>_s20260816``); the release window adds ``_sliceN``.  Both suffixes
    # are stripped so a sample and the window it should be compared against land
    # in one group rather than two.
    stem = sequence_key(text)
    marker = stem.rfind("_s")
    if marker > 0 and stem[marker + 2:].isdigit():
        stem = stem[:marker]
    stem = stem.rsplit("/", 1)[-1]
    for pattern in (WILD_CLIP, WILD_LEGACY_CLIP):
        match = pattern.match(stem)
        if match is not None:
            corpus = match.groupdict().get("corpus") or "wild"
            return "{}:{}".format(corpus, match.group("upload"))
    return None


def load_verified_music_pairs(path: pathlib.Path) -> Tuple[List[FrozenSet[str]], Dict[str, object]]:
    """Read ``fingerprint_wild_music.py``'s verified pair list.

    That tool publishes two things and only one of them is usable here.  Its
    ``track_of`` grouping is a single-linkage closure -- ``a--b--c`` puts ``a``
    with ``c`` on a pair that never scored above threshold -- and it says so
    itself (``track_of_is_usable_as_a_music_key: false``, largest component
    2,335 clips = 17% of the corpus).  Its **pair list** carries no such
    transitivity: each row is one alignment that cleared the calibrated
    operating point.  Excluding candidates pairwise needs exactly that and never
    needs a track id, so the unusable artifact is skipped and the sound one is
    read.

    What this buys, stated so it is not over-read: the upload key already
    removes cuts of one video, which share a track by construction.  These pairs
    remove *cross-upload* same-track pairs, which the key cannot see.  Together
    they are still a floor -- the fingerprint's own positive control recalls
    0.564 of known-true pairs, and its disjoint-alignment control sits below
    threshold -- so an R with both removed remains an upper bound, just a
    tighter one.
    """
    rows = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    pairs, scores = [], []
    for row in rows:
        left, right = str(row["left"]), str(row["right"])
        if left == right:
            continue
        pairs.append(frozenset((left, right)))
        if "score" in row:
            scores.append(float(row["score"]))
    unique = sorted(set(pairs), key=lambda pair: sorted(pair))
    if not unique:
        raise RPrecisionError("{} carries no usable pair".format(path))
    provenance = {
        "path": str(path),
        "rows": len(rows),
        "distinct_pairs": len(unique),
        "min_score": round(min(scores), 4) if scores else None,
        "what_it_is": ("cross-upload pairs whose 35-D chroma-CENS alignment cleared the "
                       "calibrated operating point; a lower bound on shared music, not a "
                       "track id"),
    }
    return unique, provenance


def clip_key(name: str) -> str:
    """The clip a window belongs to, with any seed suffix removed.

    ``sequence_key`` stops at the slice: a generated sample is
    ``<clip>_s20260816_slice3`` and it returns ``<clip>_s20260816``.  That is the
    right unit for excluding a sample's own slices, and the wrong one for
    matching a fingerprint pair list, whose keys are clips -- so every pair
    missed and the same-music control silently removed nothing.  It did not stay
    silent: the applied-pair guard refused the run rather than publishing an
    uncontrolled R as a controlled one, which is what that guard exists for.
    """
    stem = sequence_key(name)
    marker = stem.rfind("_s")
    if marker > 0 and stem[marker + 2:].isdigit():
        stem = stem[:marker]
    return stem.rsplit("/", 1)[-1]


def stored_window_policy(release: pathlib.Path) -> Optional[Dict[str, int]]:
    """``window_length`` / ``window_stride`` off the release's own build.json, or None.

    Returned rather than defaulted: a stride guessed here and changed in the
    materialiser is the failure shape this repo has already hit four times, and
    the only thing the guess buys is speed.  Absent policy means the slow path,
    not an assumed one.
    """
    build = release / "build.json"
    if not build.exists():
        return None
    try:
        policy = json.loads(build.read_text(encoding="utf-8")).get("window_policy") or {}
    except (ValueError, OSError):
        return None
    length, stride = policy.get("window_length"), policy.get("window_stride")
    if not isinstance(length, int) or not isinstance(stride, int):
        return None
    if length <= 0 or stride <= 0 or stride >= length:
        # stride >= length means the windows do not overlap, so there is no
        # redundant frame to remove and the fast path would only add risk.
        return None
    return {"window_length": length, "window_stride": stride}


def sequence_blocks(names: Sequence[str]) -> Optional[List[Tuple[int, int]]]:
    """Contiguous runs of one sequence's windows, as (start_index, count), or None.

    ``None`` means the layout is not the one the fast path assumes -- windows of a
    sequence adjacent and its slice indices running 0..n-1.  A release that has
    been filtered or reordered still scores correctly; it just scores the slow
    way.  Detecting that here rather than trusting the names is the point: a
    stitched sequence built from a reordered array would be a *different dance*,
    and every number downstream would be confidently wrong.
    """
    blocks: List[Tuple[int, int]] = []
    index = 0
    while index < len(names):
        key = sequence_key(names[index])
        run = 0
        while index + run < len(names) and sequence_key(names[index + run]) == key:
            suffix = names[index + run].rsplit("_slice", 1)
            if len(suffix) != 2 or not suffix[1].isdigit() or int(suffix[1]) != run:
                return None
            run += 1
        blocks.append((index, run))
        index += run
    seen = {sequence_key(names[start]) for start, _ in blocks}
    if len(seen) != len(blocks):
        # One sequence appearing in two runs means the array is interleaved.
        return None
    return blocks


def stitch_sequence(motion, start: int, count: int, length: int, stride: int) -> np.ndarray:
    """Rebuild a sequence's normalized frames from its overlapping windows.

    Window ``k`` is asserted -- not assumed -- to begin ``stride`` frames after
    window ``k-1``: the overlap has to match bit for bit, which it does because
    every window stores the same source frame through the same normalisation.
    A release whose windows were produced some other way fails here loudly
    instead of yielding a plausible dance nobody generated.
    """
    overlap = length - stride
    first = np.asarray(motion[start])
    parts = [first]
    previous = first
    for offset in range(1, count):
        current = np.asarray(motion[start + offset])
        if not np.array_equal(previous[stride:], current[:overlap]):
            raise RPrecisionError(
                "window {} does not overlap window {} by {} frames; the release's "
                "stated window_stride does not describe this array".format(
                    start + offset, start + offset - 1, overlap))
        parts.append(current[overlap:])
        previous = current
    return np.concatenate(parts, axis=0) if len(parts) > 1 else first


def build_clip_features(release: pathlib.Path, split: str, *, clip_frames: int,
                        feature: str, music_aggregate: str,
                        limit: Optional[int] = None) -> Dict[str, object]:
    """Load a release split and reduce it to one music and one motion vector per clip.

    Motion features need joint positions, and joint positions cost one SMPL
    forward pass.  Windows at stride 15 overlap ten deep, so featurising each
    window separately runs that pass over every frame about eight times --
    423,000 window-frames over 47,892 real ones on the AIST scoring bundle,
    measured.

    What that costs depends entirely on how busy the box is, and getting this
    wrong once is why the numbers below are spelled out.  Per 150-frame window,
    same bundle, same code:

        torch threads     unnormalize    SMPL FK    kinetic feature
        1                     0.16 ms    3.47 ms         102.62 ms
        128 (box saturated)   2.75 ms     ~350 ms        ~130    ms

    So on an idle box the forward pass is 3% of the window and the *kinetic
    feature* is 96% -- and the kinetic feature is genuinely per window, nothing
    dedupes it.  Under the load this step actually runs at, right behind an M6
    inference that saturates every core, torch's intra-op pool turns the same
    forward pass into 350 ms and it becomes the whole cost.  Hence both changes
    below, and neither alone: the threads are pinned, *and* the pass is run once
    per sequence instead of once per window.

    (An earlier version of this docstring said the forward kinematics was
    "essentially all" of 0.291 s per window.  That 0.291 s was measured with the
    box saturated and torch spreading a 24-joint tree over 128 threads; it is not
    what this code costs, and it named the wrong dominant term.  The remaining
    102 ms of kinetic feature per window is untouched by anything here -- at
    61,164 windows that is still ~104 core-minutes, and reducing it means
    parallelising across windows, not sharing work between them.)

    When the release states its window policy, this rebuilds each sequence from
    its windows, runs the forward pass once over the sequence, and slices the
    windows back out.  Two properties make that an identity rather than an
    approximation, and both are checked rather than argued:

    * SMPL forward kinematics is frame-independent, so ``FK(sequence)[a:b]`` and
      ``FK(sequence[a:b])`` agree to the last bit -- verified in
      ``tests/test_eval_r_precision.py``;
    * overlapping windows store a shared source frame identically, because it
      went through the same min-max map, so the stitch is exact -- and
      ``stitch_sequence`` refuses the run if any overlap disagrees.

    The path taken is reported in ``featurisation`` so a slow run says why it was
    slow instead of only being slow.
    """
    import torch

    from tools.convert_motion_to_guofeats import motion_151_to_joints

    directory = release / split
    motion = np.load(directory / "motion.npy", mmap_mode="r")
    music = np.load(directory / "music.npy", mmap_mode="r")
    names = json.loads((directory / "names.json").read_text(encoding="utf-8"))
    if len(motion) != len(music) or len(motion) != len(names):
        raise RPrecisionError("{}: motion, music and names disagree on window count".format(
            directory))
    if limit is not None:
        motion, music, names = motion[:limit], music[:limit], names[:limit]

    spans = split_into_clips(motion.shape[1], clip_frames)
    normalizer = release / "normalizer.pt"
    window_length = int(motion.shape[1])
    # Pinned for the duration, then restored.  This step runs directly behind an
    # M6 inference that leaves every core busy, and torch's intra-op pool spends
    # more time coordinating a 24-joint kinematic tree across 128 threads than
    # walking it on one: 3.47 ms per window against ~350 ms, measured on the AIST
    # bundle.  The features themselves are numpy and unaffected either way.
    parent_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    policy = stored_window_policy(release)
    blocks = sequence_blocks(names) if policy else None
    if policy and policy["window_length"] != window_length:
        # The array's own second dimension is the authority on how long a window
        # is; a build.json that disagrees describes some other artifact.
        blocks = None
    stride = policy["window_stride"] if policy else 0

    music_rows, motion_rows, groups, clips = [], [], [], []

    def emit(joints: np.ndarray, index: int) -> None:
        window_music = np.asarray(music[index])
        for start, end in spans:
            motion_rows.append(motion_clip_feature(joints[start:end], feature))
            music_rows.append(music_clip_feature(window_music[start:end], music_aggregate))
            groups.append(sequence_key(names[index]))
            clips.append(clip_key(names[index]))

    if blocks is not None:
        mode = "per_sequence"
        done = 0
        for start, count in blocks:
            stitched = stitch_sequence(motion, start, count, window_length, stride)
            joints = motion_151_to_joints(unnormalize(stitched, normalizer))
            for offset in range(count):
                head = offset * stride
                emit(joints[head:head + window_length], start + offset)
            done += count
            if done % 500 < count:
                print("  featurised {}/{} windows ({} sequences)".format(
                    done, len(motion), len(blocks)), flush=True)
    else:
        mode = "per_window"
        print("  featurising per window: {}".format(
            "release states no usable window_policy" if policy is None
            else "window layout is not one contiguous 0..n-1 run per sequence"), flush=True)
        for index in range(len(motion)):
            emit(motion_151_to_joints(unnormalize(np.asarray(motion[index]), normalizer)), index)
            if (index + 1) % 100 == 0:
                print("  featurised {}/{} windows".format(index + 1, len(motion)), flush=True)

    torch.set_num_threads(parent_threads)

    return {
        "music": np.stack(music_rows),
        "motion": np.stack(motion_rows),
        "groups": groups,
        "clips": clips,
        "music_groups": [music_key(group) for group in groups],
        "windows": int(len(motion)),
        "clips_per_window": len(spans),
        "clip_seconds": round(clip_frames / FPS, 3),
        "featurisation": {
            "mode": mode,
            "sequences": len(blocks) if blocks is not None else None,
            "window_stride": stride or None,
            "what_it_is": "per_sequence runs SMPL forward kinematics once per sequence and "
                          "slices the windows out of it; per_window runs it once per window. "
                          "The two are bit-identical -- FK is frame-independent and the "
                          "overlap is checked -- so this records cost, not method.",
        },
    }


def evaluate(release: pathlib.Path, split: str, *, clip_frames: int, feature: str,
             music_aggregate: str, limit: Optional[int] = None,
             pool_sizes: Sequence[int] = (), pool_repeats: int = 20,
             exclude_same_music: bool = False,
             music_pairs: Optional[pathlib.Path] = None) -> Dict[str, object]:
    """R-precision for one split, reported both with and without same-song pairs."""
    built = build_clip_features(release, split, clip_frames=clip_frames, feature=feature,
                                music_aggregate=music_aggregate, limit=limit)
    result = {
        "release": str(release),
        "split": split,
        "clip_frames": int(clip_frames),
        "clip_seconds": built["clip_seconds"],
        "windows": built["windows"],
        "clips_per_window": built["clips_per_window"],
        # Carried into the report so a slow run says why it was slow.  The two
        # modes are bit-identical, so this records cost and never method -- but
        # this repo has already paid for one defect whose only symptom was being
        # slow and which nothing reported.
        "featurisation": built["featurisation"],
        "motion_feature": feature,
        "music_feature": "35-D baseline, aggregated by {}".format(music_aggregate),
        "similarity": "per-dimension z-score, then Euclidean distance",
        "paper_ground_truth_R": PAPER_GROUND_TRUTH_R,
        "definition": ("for each clip, the nearest other clip in music space must be "
                       "among the top-3 nearest in motion space (paper, Sec. 4.2)"),
    }
    for label, exclude in (("pooled", False), ("excluding_same_sequence", True)):
        result[label] = r_precision(built["music"], built["motion"],
                                    groups=built["groups"], exclude_same_group=exclude)
    result["adjacency_share"] = round(
        result["pooled"]["R"] - result["excluding_same_sequence"]["R"], 2)
    if exclude_same_music:
        music_groups = built["music_groups"]
        unparsed = sorted({group for group, key in zip(built["groups"], music_groups)
                           if key is None})
        if unparsed:
            raise RPrecisionError(
                "{} of {} sequences name no audio group ({}...); grouping them "
                "together would exclude unrelated songs from each other's pools".format(
                    len(unparsed), len(set(built["groups"])), unparsed[0]))
        result["excluding_same_music"] = r_precision(
            built["music"], built["motion"], groups=music_groups, exclude_same_group=True)
        # The unit this control actually achieved, written next to its number:
        # a control whose achieved unit is not published is one nobody can audit,
        # which is how ``music.roll(1)`` passed for a mispaired-music control
        # while handing every window another take of its own song.
        result["music_group_unit"] = {
            "parsed_from": "AIST music id token, or the wild upload, in the window name",
            "distinct_music_groups": len(set(music_groups)),
            "distinct_sequences": len(set(built["groups"])),
            # On AIST the group is the track.  On wild it is the upload, which
            # provably shares a track but does not catch two uploads dancing to
            # the same one, so the exclusion is a floor and this R is an upper
            # bound.  Stated in the report because a control whose achieved unit
            # is not published is one nobody can audit.
            "is_a_floor": any(str(key).count(":") == 1 for key in music_groups if key),
        }
        if music_pairs is not None:
            pairs, provenance = load_verified_music_pairs(music_pairs)
            tightened = r_precision(built["music"], built["motion"], groups=music_groups,
                                    exclude_same_group=True, exclude_pairs=pairs,
                                    pair_keys=built["clips"])
            # A control that removed nothing is a control that did not run, and
            # in R it is indistinguishable from one that was never asked for.
            # This is the shape CLAUDE.md 2 forbids -- a gate that cannot fire --
            # so the pair list naming no two clips of this split is an error, not
            # a quiet pass-through.
            if tightened["excluded_verified_pairs"] == 0:
                raise RPrecisionError(
                    "{} names no two clips of the {} split, so excluding its pairs "
                    "removed nothing; a control that cannot fire must not be "
                    "reported as one that passed".format(music_pairs, split))
            result["excluding_same_music_and_verified_pairs"] = tightened
            result["music_group_unit"]["verified_pairs"] = dict(
                provenance,
                applied_within_split=int(tightened["excluded_verified_pairs"]),
                pool_shrank_by=round(
                    result["excluding_same_music"]["mean_candidate_pool"]
                    - tightened["mean_candidate_pool"], 1),
                reading=("the upload key removes cuts of one video; these pairs remove "
                         "cross-upload same-track pairs it cannot see.  Both directions "
                         "are provable, neither is complete, so this R is a tighter "
                         "upper bound rather than a measurement"),
            )
    elif music_pairs is not None:
        raise RPrecisionError(
            "--music-pairs refines the same-music control, so it needs "
            "--exclude-same-music; on its own it would remove named pairs while "
            "leaving every clip's own siblings in the pool")
    if pool_sizes:
        result["fixed_pool"] = [
            r_precision_at_pool(built["music"], built["motion"], pool_size=size,
                                repeats=pool_repeats, groups=built["groups"])
            for size in pool_sizes]
        result["fixed_pool_note"] = (
            "R falls as the pool grows, so a published R is only comparable at the "
            "same pool size; the paper does not state its own, which is why these "
            "are reported as a curve rather than a single figure")
        if exclude_same_music:
            result["fixed_pool_excluding_same_music"] = [
                r_precision_at_pool(built["music"], built["motion"], pool_size=size,
                                    repeats=pool_repeats, groups=built["music_groups"],
                                    exclude_same_group=True)
                for size in pool_sizes]
            # The curve above excludes nothing, and on generated motion it passes
            # through the paper's 42.1 at a small pool -- built entirely from other
            # takes of the same song.  Calibrating a pool size against 42.1 on that
            # curve would land on the artifact, so the controlled curve is computed
            # next to it rather than left for a reader to ask for.
            result["fixed_pool_control_note"] = (
                "fixed_pool excludes nothing; fixed_pool_excluding_same_music is the "
                "same sweep with same-song pairs removed, and it is the one a pool "
                "size may be calibrated on")
    return result


def calibrate(release: pathlib.Path, split: str, *, clip_lengths: Sequence[int],
              feature: str, music_aggregate: str, limit: Optional[int] = None
              ) -> Dict[str, object]:
    """Sweep the one parameter the paper leaves free, against the one number it prints.

    The target is the ground-truth row, R = 42.1, and it is only a valid target
    on AIST++ ground-truth motion -- that is the corpus and the motion the paper
    measured.  Running this sweep on wild motion reports the shape of the curve
    but must not be read as reproducing 42.1.
    """
    rows = []
    for clip_frames in clip_lengths:
        try:
            measured = evaluate(release, split, clip_frames=clip_frames, feature=feature,
                                music_aggregate=music_aggregate, limit=limit)
        except RPrecisionError as error:
            rows.append({"clip_frames": int(clip_frames), "error": str(error)})
            continue
        rows.append({
            "clip_frames": int(clip_frames),
            "clip_seconds": measured["clip_seconds"],
            "R": measured["pooled"]["R"],
            "R_excluding_same_sequence": measured["excluding_same_sequence"]["R"],
            "chance_R": measured["pooled"]["chance_R"],
            "clips": measured["pooled"]["clips"],
            "distance_to_paper_ground_truth": round(
                abs(measured["pooled"]["R"] - PAPER_GROUND_TRUTH_R), 2),
        })
        print("  clip {:>4} frames ({:.2f}s): R {:.2f} (excl {:.2f}, chance {:.2f})".format(
            clip_frames, rows[-1]["clip_seconds"], rows[-1]["R"],
            rows[-1]["R_excluding_same_sequence"], rows[-1]["chance_R"]), flush=True)
    scored = [row for row in rows if "R" in row]
    best = min(scored, key=lambda row: row["distance_to_paper_ground_truth"]) if scored else None
    return {
        "release": str(release),
        "split": split,
        "motion_feature": feature,
        "music_feature": "35-D baseline, aggregated by {}".format(music_aggregate),
        "target": PAPER_GROUND_TRUTH_R,
        "target_note": ("the paper's ground-truth R on AIST++; it is a calibration "
                        "target only when this sweep runs on AIST++ ground-truth motion"),
        "sweep": rows,
        "closest": best,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=pathlib.Path, required=True,
                        help="a materialized release directory holding motion.npy, "
                             "music.npy and names.json per split")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--clip-frames", type=int, default=150,
                        help="clip length in frames; the paper does not state one, so "
                             "this is calibrated, not assumed (see --calibrate)")
    parser.add_argument("--feature", default="kinetic", choices=("kinetic", "manual"),
                        help="motion feature space; kinetic and manual are the same two "
                             "spaces FID_k and FID_g are computed in")
    parser.add_argument("--music-aggregate", default="mean_std", choices=("mean", "mean_std"))
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N windows, for a quick read")
    parser.add_argument("--calibrate", default=None,
                        help="comma-separated clip lengths to sweep against the paper's "
                             "ground-truth R instead of reporting a single value")
    parser.add_argument("--pool-sizes", default=None,
                        help="comma-separated candidate-pool sizes to additionally report "
                             "R at; R is only comparable to a published figure at the "
                             "same pool size, so this is how a comparison is made")
    parser.add_argument("--pool-repeats", type=int, default=20,
                        help="random sub-pools drawn per size, so the spread is reported "
                             "rather than one lucky draw")
    parser.add_argument("--exclude-same-music", action="store_true",
                        help="additionally report R with every pair from the same song "
                             "removed; required for generated motion, where several "
                             "seeds of one song share their music features exactly")
    parser.add_argument("--music-pairs", type=pathlib.Path, default=None,
                        help="fingerprint_wild_music.py's verified pair list "
                             "(runs/<tag>_music_groups_pairs.jsonl); refines "
                             "--exclude-same-music by also removing cross-upload pairs "
                             "proven to share a track, which the upload key cannot see")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.calibrate:
            lengths = [int(value) for value in args.calibrate.split(",") if value.strip()]
            report = calibrate(args.release, args.split, clip_lengths=lengths,
                               feature=args.feature, music_aggregate=args.music_aggregate,
                               limit=args.limit)
        else:
            pool_sizes = [int(value) for value in (args.pool_sizes or "").split(",")
                          if value.strip()]
            report = evaluate(args.release, args.split, clip_frames=args.clip_frames,
                              feature=args.feature, music_aggregate=args.music_aggregate,
                              limit=args.limit, pool_sizes=pool_sizes,
                              pool_repeats=args.pool_repeats,
                              exclude_same_music=args.exclude_same_music,
                              music_pairs=args.music_pairs)
    except RPrecisionError as error:
        print("R-precision refused: {}".format(error), file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print("wrote {}".format(args.output))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
