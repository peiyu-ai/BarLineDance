#!/usr/bin/env python3
"""M6c: MultiModality -- how far apart are several dances for the *same* music.

The last empty cell in the module audit.  The paper reports it alongside FID,
Div and BAS; this repo had no implementation, so the column could not be filled
even in principle.

The metric itself is one line -- mean pairwise distance inside a music, averaged
over musics -- and that is exactly why it needs the scaffolding around it:

* **Alone it is not a quality signal.**  A model that ignores its conditioning
  entirely maximises MultiModality, and would beat any model that responds to
  the music.  Reporting MM without what it is being compared against is
  reporting a number that improves when the model gets worse.
* **So it is reported with Diversity, in the same units.**  Same distance, same
  standardisation, computed on the same set: Div spreads over every sample, MM
  only over samples that share a music.  ``MM / Div`` is the interpretable
  quantity.  1.0 means samples for one music are as far apart as samples for
  different musics -- unconditioned.  Well below 1.0 means the music is
  constraining the dance, which is the thing the model is for.
* **And with a permutation null.**  Shuffling the music labels and recomputing
  gives what the ratio looks like when the grouping carries no information.
  The observed ratio has to be read against that, not against 1.0, because a
  finite sample of unequal group sizes does not land on 1.0 by itself.

Ground truth calibration, which is what this tool can do *today*:

    AIST++ pairs several choreographies with each piece of music, so the ground
    truth has a measurable MultiModality -- the spread real dancers produce on
    one song.  That is the reference a model should be read against, the same
    way R-precision is calibrated against the paper's GT 42.1 rather than
    against zero.  A model far below GT is copying one answer; far above it is
    ignoring the music.

    The wild corpus cannot supply this: every clip carries its own audio, so
    each music has exactly one dance and MultiModality is undefined on it.  The
    tool says so rather than returning a number built from nothing.

Usage::

    # ground-truth calibration on AIST++
    python3 tools/eval_multimodality.py --release data/atomic_aistpp/aist_kinematic_release_v1 \\
        --split test --output runs/multimodality_aist_gt.json

    # a model's samples, five per music, laid out like the eval feature roots
    python3 tools/eval_multimodality.py --feature-root runs/samples_v1 \\
        --output runs/multimodality_model_v1.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.metrics import (  # noqa: E402
    calc_multimodality,
    calc_multimodality_pooled,
    calculate_avg_distance,
    normalize_separately,
)

FPS = 30.0
# The genre control.  On AIST++ one song belongs to one genre, so "same music"
# and "same genre" are nested -- a low MM/Div by music could be nothing but
# genre.  Grouping by genre instead answers how much of it is, and the
# difference is what music-specific choreography is actually worth.
# Not anchored at the string start: release names carry a corpus prefix
# ("aistpp/gHO_sBM_..."), so ^ matched nothing.  The genre field is always
# followed by the subject field, which is what pins it.
AIST_GENRE = re.compile(r"(g[A-Z]{2})_s")


class MultiModalityError(RuntimeError):
    pass


def music_key(name: str) -> Optional[str]:
    """The music a sample was danced to, or None when the name does not say.

    Borrowed from ``eval_r_precision`` rather than written again.  This module
    had its own ``_(m[A-Z]{2}\\d+)_`` regex, which requires the id to sit
    *between* underscores -- true of a release window
    (``gBR_sBM_cAll_d04_mBR0_ch01``) and false of a generated sample, whose name
    starts with the song it was generated for (``mBR2_s20260808``).  Every one
    of the 40 generated samples therefore parsed as None, and the tool refused
    to run.  Refusing was the good outcome; the same divergence in the other
    direction would have pooled all forty into one bucket and reported a ratio
    for a grouping that never happened.

    Two tools that must agree on "which dances share a song" now share the one
    function, so a name either parses for both or for neither.  ``AIST_GENRE``
    stays a regex here because ``eval_r_precision`` has no genre control.
    """
    from tools.eval_r_precision import music_key as shared_music_key

    return shared_music_key(name)


def genre_key(name: str) -> Optional[str]:
    """The dance genre, for the control that separates music from genre."""
    match = AIST_GENRE.search(str(name))
    return match.group(1) if match else None


def group_key(name: str, group_by: str) -> Optional[str]:
    return music_key(name) if group_by == "music" else genre_key(name)


def verified_track_groups(music_pairs: pathlib.Path, split_of: Dict[str, str],
                          split: str) -> Dict[str, object]:
    """Wild clips that provably share a track, grouped without assuming transitivity.

    This is the grouping this module said it could not have.  Its docstring
    still records why the obvious two do not work, and both remain true:

    * the **upload** groups cuts of one video, so "several dances for one music"
      degenerates into "several slices of one performance" -- that measures how
      smooth a take is, not how many dances a song admits;
    * the fingerprint's **connected components** chain.  ``a--b--c`` groups
      ``a`` with ``c`` on a pair that never cleared threshold, and on this
      corpus the largest component swallows 2,335 clips (17%).  The tool that
      produced them says so itself: ``track_of_is_usable_as_a_music_key: false``.

    What does work is the subset of those components that are **complete
    graphs** -- every pair inside the group scored above the calibrated
    operating point, so no transitivity is assumed anywhere.  Groups are further
    required to span **at least two uploads**, because a complete group inside
    one upload is the first failure again.  On wild_v4's test split that leaves
    144 groups over 320 clips (121 pairs, 15 triples, 7 quads, one group of 5) --
    different choreographers dancing to one track, which is AIST's unit and the
    paper's.

    The cost is stated rather than hidden: this is a *subset* of the split
    chosen by a property of the evidence, so it is a measurement of the clips a
    fingerprint could link, not of the corpus.  Its recall is the fingerprint's,
    0.564 on known-true pairs.
    """
    from itertools import combinations

    from tools.eval_r_precision import load_verified_music_pairs

    pairs, provenance = load_verified_music_pairs(music_pairs)
    edges = set()
    adjacency: Dict[str, set] = {}
    for pair in pairs:
        left, right = sorted(pair)
        if split_of.get(left) != split or split_of.get(right) != split:
            continue
        edges.add((left, right))
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    seen, components = set(), []
    for node in adjacency:
        if node in seen:
            continue
        stack, component = [node], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        seen |= component
        components.append(component)

    group_of: Dict[str, str] = {}
    kept, sizes = 0, []
    for component in sorted(components, key=lambda members: sorted(members)):
        if any(tuple(sorted(pair)) not in edges for pair in combinations(component, 2)):
            continue
        uploads = {member.split(":")[1] for member in component if member.count(":") >= 2}
        if len(uploads) < 2:
            continue
        name = "track:{}".format(sorted(component)[0])
        for member in component:
            group_of[member] = name
        kept += 1
        sizes.append(len(component))
    if not group_of:
        raise MultiModalityError(
            "{} yields no complete multi-upload group in split {!r}; without one "
            "there is no set of distinct performances of a shared track to "
            "measure, and a grouping built any other way would measure something "
            "else".format(music_pairs, split))
    return {
        "group_of": group_of,
        "groups": kept,
        "clips": len(group_of),
        "sizes": {"min": min(sizes), "max": max(sizes),
                  "mean": round(sum(sizes) / len(sizes), 2)},
        "components_examined": len(components),
        "provenance": provenance,
        "what_it_is": ("complete subgraphs of the verified-pair graph spanning at "
                       "least two uploads: every pair inside a group cleared the "
                       "operating point, so no transitivity is assumed"),
        "what_it_is_not": ("a sample of the corpus -- it is the clips a chroma-CENS "
                           "fingerprint could link, and its recall on known-true "
                           "pairs is 0.564"),
    }


def sequence_of(name: str) -> str:
    """The recording a window came from, so its own slices can be excluded."""
    base = str(name)
    marker = base.rfind("_slice")
    if marker > 0 and base[marker + len("_slice"):].isdigit():
        return base[:marker]
    return base


def permutation_null(features: np.ndarray, groups: Sequence[str], *,
                     repeats: int, seed: int,
                     sequences: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """MM/Div when the grouping is destroyed but the group *sizes* are kept.

    Keeping the sizes matters: mean pairwise distance inside a small group is
    noisier than inside a large one, so a null that also resampled the sizes
    would be answering a different question.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray([str(g) for g in groups])
    diversity = calculate_avg_distance(features)
    ratios = []
    for _ in range(repeats):
        # Shuffle the music labels only.  The recording labels stay attached to
        # their samples, so the cross-recording mask still means the same thing
        # under the null as it does in the observed run.
        shuffled = rng.permutation(labels)
        ratios.append(calc_multimodality(features, shuffled, sequences) / diversity)
    ratios = np.asarray(ratios, dtype=np.float64)
    return {
        "mean": float(ratios.mean()),
        "sd": float(ratios.std(ddof=1)) if len(ratios) > 1 else 0.0,
        "p05": float(np.quantile(ratios, 0.05)),
        "p95": float(np.quantile(ratios, 0.95)),
        "repeats": int(repeats),
        "ratios": ratios,
    }


def score(features: np.ndarray, groups: Sequence[str], *,
          repeats: int = 200, seed: int = 20260812,
          sequences: Optional[Sequence[str]] = None) -> Dict[str, object]:
    features = normalize_separately(np.asarray(features, dtype=np.float64))
    sizes: Dict[str, int] = {}
    for key in groups:
        sizes[str(key)] = sizes.get(str(key), 0) + 1
    usable = {k: v for k, v in sizes.items() if v >= 2}
    if not usable:
        raise MultiModalityError(
            "every music has exactly one dance, so MultiModality is undefined "
            "here -- it needs several dances per music, which the wild corpus "
            "does not have and AIST++ does")
    diversity = calculate_avg_distance(features)
    multimodality = calc_multimodality(features, groups, sequences)
    pooled = calc_multimodality_pooled(features, groups, sequences)
    null = permutation_null(features, groups, repeats=repeats, seed=seed,
                            sequences=sequences)
    ratio = multimodality / diversity
    # The standard permutation p-value, with the observed value counted as one
    # of the draws.  Comparing against a 5th-percentile cut instead would call
    # a null-consistent result "conditioned" one time in twenty by
    # construction, which is too loose for a number meant to settle whether the
    # music is doing anything at all.
    draws = np.asarray(null.pop("ratios"), dtype=np.float64)
    p_value = float((1 + int((draws <= ratio).sum())) / (len(draws) + 1))
    counts = np.asarray(sorted(usable.values()))
    return {
        "p_value": round(p_value, 5),
        "samples": int(len(features)),
        "musics_total": int(len(sizes)),
        "musics_used": int(len(usable)),
        "samples_per_music": {
            "min": int(counts.min()), "median": int(np.median(counts)),
            "max": int(counts.max()), "mean": round(float(counts.mean()), 2),
        },
        "singleton_musics_dropped": int(len(sizes) - len(usable)),
        "cross_recording_pairs_only": sequences is not None,
        "multimodality": round(multimodality, 4),
        "diversity": round(diversity, 4),
        "mm_over_div": round(ratio, 4),
        "mm_pooled": round(pooled, 4),
        "mm_pooled_over_div": round(pooled / diversity, 4),
        "pooled_note": ("averaged over pairs rather than over groups; the one "
                        "that can compare nested groupings, because a coarser "
                        "grouping must admit every pair the finer one did"),
        "permutation_null_mm_over_div": {k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in null.items()},
        "conditioned": bool(p_value < 0.01),
        "reading": ("p_value is the fraction of label shuffles whose mm_over_div "
                    "is at least as small as the observed one, so small means the "
                    "music constrains the dance.  At the null, it does not -- and "
                    "a high MultiModality then measures indifference to the "
                    "conditioning rather than useful variety, which is why this "
                    "column must never be read without diversity beside it"),
    }


def features_from_release(release: pathlib.Path, split: str, *,
                          feature: str, limit: Optional[int],
                          group_by: str = "music",
                          group_of: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Ground-truth features and their music ids, straight from a release."""
    from tools.convert_motion_to_guofeats import motion_151_to_joints
    from tools.eval_r_precision import motion_clip_feature, unnormalize

    directory = release / split
    motion = np.load(directory / "motion.npy", mmap_mode="r")
    names = json.loads((directory / "names.json").read_text(encoding="utf-8"))
    if len(motion) != len(names):
        raise MultiModalityError("{}: motion and names disagree".format(directory))
    if limit is not None:
        motion, names = motion[:limit], names[:limit]
    normalizer = release / "normalizer.pt"
    # Both keys, so the run can say whether the genre control is informative
    # here at all.  On the test split AIST++ happens to give one music per
    # genre, and the two groupings are then the *same partition* -- the control
    # returns an identical number and looks like a confirmation when it is a
    # tautology.
    distinct_music = {k for k in (music_key(n) for n in names) if k}
    distinct_genre = {k for k in (genre_key(n) for n in names) if k}
    rows, groups, seqs, unnamed = [], [], [], 0
    for index in range(len(motion)):
        if group_of is not None:
            # An explicit grouping replaces the name-derived one entirely: a
            # clip outside it is not "unnamed", it is a clip this measurement is
            # not about, and mixing the two rules would put some clips in a
            # proven-shared-track group and others in an upload group.
            key = group_of.get(sequence_of(names[index]))
        else:
            key = group_key(names[index], group_by)
        if key is None:
            unnamed += 1
            continue
        raw = unnormalize(np.asarray(motion[index]), normalizer)
        rows.append(motion_clip_feature(motion_151_to_joints(raw), feature))
        groups.append(key)
        seqs.append(sequence_of(names[index]))
        if (index + 1) % 100 == 0:
            print("  featurised {}/{}".format(index + 1, len(motion)), flush=True)
    if not rows:
        raise MultiModalityError(
            "no sample name carried a music id; MultiModality needs to know "
            "which dances share a song")
    return {"features": np.stack(rows), "groups": groups, "sequences": seqs,
            "unnamed": unnamed, "distinct_music": len(distinct_music), "distinct_genre": len(distinct_genre),
            "genre_control_degenerate": len(distinct_music) == len(distinct_genre)}


def _without_seed(name: str) -> str:
    """``<clip>_s20260816`` -> ``<clip>``: the clip a sample was generated for."""
    marker = name.rfind("_s")
    return name[:marker] if marker > 0 and name[marker + 2:].isdigit() else name


def features_from_root(root: pathlib.Path, directory: str,
                       group_of: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """Pre-extracted features laid out like the eval roots, grouped by music.

    With ``group_of``, a generated sample is keyed by the *clip* it came from --
    seed suffix stripped -- so a model can be scored on the same grouping the
    ground truth was.  That matters more than it looks: the ground truth's MM is
    "two choreographers on one track", and a model scored by its own default
    grouping is measuring "four seeds of one clip".  Putting those two numbers
    in one table is comparing two questions.
    """
    paths = sorted((root / directory).glob("*.npy"))
    if not paths:
        raise MultiModalityError("no .npy features under {}".format(root / directory))
    rows, groups, seqs, unnamed = [], [], [], 0
    for path in paths:
        key = (group_of.get(_without_seed(path.stem)) if group_of is not None
               else music_key(path.stem))
        if key is None:
            unnamed += 1
            continue
        rows.append(np.asarray(np.load(str(path)), dtype=np.float64).reshape(-1))
        groups.append(key)
        # The recording a sample belongs to.  With an explicit grouping this is
        # the clip, not the clip-plus-seed: the ground truth excluded pairs from
        # one recording, so the model has to as well, or its MM counts four
        # seeds of one clip as four distinct dances for the track.
        seqs.append(_without_seed(path.stem) if group_of is not None
                    else sequence_of(path.stem))
    if not rows:
        raise MultiModalityError(
            "no filename under {} carried a music id such as mBR0".format(root / directory))
    widths = {row.shape for row in rows}
    if len(widths) != 1:
        raise MultiModalityError("inconsistent feature widths under {}".format(directory))
    return {"features": np.stack(rows), "groups": groups, "sequences": seqs,
            "unnamed": unnamed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--release", type=pathlib.Path,
                        help="a training release; features are computed from its motion")
    source.add_argument("--feature-root", type=pathlib.Path,
                        help="a directory of pre-extracted .npy features")
    parser.add_argument("--split", default="test")
    parser.add_argument("--feature", default="kinetic", choices=("kinetic", "manual"))
    parser.add_argument("--feature-dir", default="kinetic_features",
                        help="subdirectory of --feature-root to read")
    parser.add_argument("--group-by", default="music", choices=("music", "genre"),
                        help="genre is the control: on AIST++ one song is one "
                             "genre, so a low ratio by music may be genre alone")
    parser.add_argument("--limit", type=int, default=None)
    # On ground truth this is not a refinement, it is the difference between
    # measuring "how many dances a song admits" and "how smooth one performance
    # is": AIST++ windows are slices, and the median music in the test split has
    # two recordings behind sixteen windows.  Model samples are independent, so
    # --all-pairs is the right setting for them.
    parser.add_argument("--all-pairs", action="store_true",
                        help="count pairs from the same recording too; correct "
                             "for model samples, wrong for ground truth")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--music-pairs", type=pathlib.Path, default=None,
                        help="fingerprint_wild_music.py's verified pair list.  On the "
                             "wild corpus this supplies the grouping the name cannot: "
                             "complete subgraphs spanning two or more uploads, i.e. "
                             "different performances of one proven-shared track.  "
                             "Requires --release, because the clip's split has to be "
                             "read from the bundle the release was built from")
    parser.add_argument("--bundle", type=pathlib.Path, default=None,
                        help="performance bundle supplying each clip's split; required "
                             "with --music-pairs")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    grouping = None
    if args.music_pairs is not None:
        if args.bundle is None:
            print("--music-pairs needs --bundle: the grouping is over clips of one "
                  "split, and the split is recorded in the bundle", file=sys.stderr)
            return 2
        split_of = {}
        for line in (args.bundle / "sequences.jsonl").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                split_of[str(row["recording_id"])] = str(row["split"])
        grouping = verified_track_groups(args.music_pairs, split_of, args.split)
    if args.release is not None:
        built = features_from_release(args.release, args.split,
                                      feature=args.feature, limit=args.limit,
                                      group_by=args.group_by,
                                      group_of=None if grouping is None
                                      else grouping["group_of"])
        source = "{}::{}".format(args.release, args.split)
    else:
        built = features_from_root(args.feature_root, args.feature_dir,
                                   group_of=None if grouping is None
                                   else grouping["group_of"])
        source = str(args.feature_root / args.feature_dir)
    report = score(built["features"], built["groups"],
                   repeats=args.permutations, seed=args.seed,
                   sequences=None if args.all_pairs else built.get("sequences"))
    report["source"] = source
    report["feature_space"] = args.feature
    report["group_by"] = "verified_shared_track" if grouping else args.group_by
    if grouping is not None:
        report["music_grouping"] = {k: v for k, v in grouping.items() if k != "group_of"}
    report["samples_without_music_id"] = int(built["unnamed"])
    if "genre_control_degenerate" in built:
        report["distinct_music"] = int(built["distinct_music"])
        report["distinct_genre"] = int(built["distinct_genre"])
        report["genre_control_degenerate"] = bool(built["genre_control_degenerate"])
        if built["genre_control_degenerate"]:
            report["warning"] = (
                "this split has one music per genre, so grouping by music and "
                "grouping by genre are the same partition -- the genre control "
                "cannot separate them here, and an identical number is a "
                "tautology rather than agreement")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
