"""Gate v2: does a label sequence's *structure* respond to the song?

The frame-accuracy gate turned out to be unattainable on AIST++: different
performances of the same song agree on only ~0.10 of frames, so which atom is
danced is choreographic freedom.  What music measurably constrains is the
*structural rhythm* -- same-song performances have significantly more similar
segments-per-second than different-song pairs (Mann-Whitney p = 3.0e-07 on the
visual segmentation), while boundary-beat phase and class distributions show
nothing.  This tool turns that observation into a reusable gate.

Two modes share the same statistics:

* ``--data-root``: score a release's GT labels.  This is the *vocabulary* gate:
  a vocabulary whose segmentation rate does not respond to the song carries no
  music-conditioned structure for a planner to learn.
* ``--plans``: score generated plans (JSON: ``{"plans": [{"song": ..,
  "labels": [...]}, ...]}``).  This is the *planner* gate: sampled plans must
  reproduce the same-song structural coherence, and their per-song rate should
  track the GT rate.

Statistics reported:

1. same-song vs different-song absolute rate difference, Mann-Whitney
   one-sided (same < diff), with a same-genre control -- genre is trivially
   recoverable from music, so a genre-only effect must not masquerade as
   song-level structure conditioning;
2. per-song mean rate split-half reliability (do a song's performances agree
   on a characteristic rate at all);
3. in plans mode, Spearman correlation between generated and GT per-song rates.

What this gate cannot see (2026-08-22)
--------------------------------------
The statistic is a *rate*, so anything that follows the beat passes it.
Measured on clean5b5 train, 483 verified same-song pairs:

    ground truth                      p = 6.1e-03
    planner, stochastic + vote        p = 1.2e-04
    **a metronome with random labels**  **p = 2.0e-12**

The metronome -- ``tools/make_beat_grid_control.py``, a cut every four beats
(M1's own rule) filled with uniformly random classes -- carries zero
information about *which* movement happens, and beats both.  So a pass here
means the plan's segmentation rate tracks tempo; it does not mean the planner
learned music-conditioned choreography, and a tempo-only model would pass it
more convincingly than the truth does.

``--control-plans`` scores such a control alongside the arm and records
``explained_by_tempo``.  Without it the report says the control was not
supplied, because a criterion whose ceiling was never measured is the defect
CLAUDE.md 2.1 is about, and an artifact that does not say so invites the same
mistake twice.
"""

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SLICE = re.compile(r"_slice(\d+)$")
FPS = 30.0

# The third tool to need "which dances share a song", and the third parser it
# would have grown.  ``_(m[A-Za-z]{2}\d+)_`` -- underscores required on both
# sides -- is the exact shape that made ``eval_multimodality`` refuse every
# generated sample on 2026-08-15, and on the wild corpus it matches nothing at
# all.  That failure is worse here than there: a sequence whose song will not
# parse is skipped, so an unparseable corpus yields *no pairs* and the gate
# reports ``checked: False`` instead of failing -- a gate that cannot fire
# (CLAUDE.md 2).  One shared parser, so three tools cannot disagree.
from tools.eval_r_precision import music_key  # noqa: E402


def genre_of_song(song):
    """The AIST genre letters behind a music id, or None when there are none.

    ``song[1:3]`` was the rule, and it is only a rule on AIST: applied to a wild
    key it slices two letters out of ``wild_v4:7195…`` and invents a genre.  The
    control is skipped rather than faked when the corpus does not name one.
    """
    text = str(song)
    if len(text) == 4 and text[0] == "m" and text[1:3].isalpha() and text[3].isdigit():
        return text[1:3]
    return None


def sequence_rates_from_release(release, split):
    """Segments-per-second of every full sequence, rebuilt from windows.

    Windows overlap (15-frame stride), so rates are computed on stitched
    full-sequence label timelines rather than on windows, which would count
    every boundary up to ten times.
    """
    labels = np.load(Path(release) / split / "labels.npy", mmap_mode="r")
    names = json.loads((Path(release) / split / "names.json").read_text(encoding="utf-8"))

    timelines = {}
    for index, window in enumerate(names):
        base = window.split("/")[-1]
        sequence = _SLICE.sub("", base)
        start = int(_SLICE.search(base).group(1)) * 15
        end = start + labels.shape[1]
        row = np.asarray(labels[index])
        current = timelines.setdefault(sequence, {})
        # Overlapping windows carry identical labels on shared frames for a
        # window release built from full-sequence labels; last write wins.
        current[start] = (end, row)

    output, skipped = {}, []
    for sequence, chunks in timelines.items():
        length = max(end for end, _ in chunks.values())
        timeline = np.zeros(length, dtype=np.int64)
        for start, (end, row) in sorted(chunks.items()):
            timeline[start:end] = row
        song = music_key(sequence)
        if song is None:
            skipped.append(sequence)
            continue
        output[sequence] = (song, segment_rate(timeline))
    # Silence here used to mean "no pairs", which the gate reports as
    # ``checked: False``.  Naming the corpus that could not be parsed turns a
    # gate that quietly does not fire into one that says why.
    if skipped:
        print("structure probe: {} of {} sequence(s) name no song, e.g. {}".format(
            len(skipped), len(timelines), skipped[0]))
    return output


def segment_rate(timeline):
    """Label-change boundaries per second of one full label timeline."""
    changes = int((np.diff(timeline) != 0).sum())
    return (changes + 1) / (len(timeline) / FPS)


def rates_from_plans(plans_path):
    """{sequence: (song, rate)} for generated plans.

    Keyed by the sequence a plan was generated for, not by its position in the
    file.  ``plan00042`` is a name no pair list can refer to, so ``--music-pairs``
    would match nothing and the same-song arm would come back empty -- reported
    as "too few pairs", which reads as a corpus too small rather than as two
    naming schemes that never met.  A duplicated sequence is an error rather
    than a silent overwrite: two plans for one sequence means several seeds, and
    the statistic counts each sequence once by construction.
    """
    payload = json.loads(Path(plans_path).read_text(encoding="utf-8"))
    output = {}
    for index, plan in enumerate(payload["plans"]):
        timeline = np.asarray(plan["labels"], dtype=np.int64)
        key = str(plan.get("sequence") or "plan{:05d}".format(index))
        key = key.rsplit("/", 1)[-1]
        if key in output:
            raise ValueError(
                "{}: two plans for sequence {}; this statistic counts each "
                "sequence once, so several seeds must be scored separately".format(
                    plans_path, key))
        output[key] = (plan["song"], segment_rate(timeline))
    return output


def pair_statistics(rates, seed, genre_of=genre_of_song, same_pairs=None):
    """Same-song vs different-song rate differences plus a genre control.

    ``same_pairs`` replaces the key-defined same-song set, and on the wild
    corpus it is the only way this probe asks the paper's question.  The key
    there is the **upload**, and the several clips of one upload are consecutive
    cuts of a single take -- so "same song" degenerates into "two halves of one
    performance", and the test becomes close to tautological.  AIST's question
    is the other one: *different* performances of one track.  The fingerprint's
    verified pair list names exactly those, cross-upload and proven to share
    audio, so passing it restores the comparison the statistic is named after.

    The two are reported under different names for that reason.  A p-value from
    cuts of one take and a p-value from two choreographers on one track are not
    the same claim, and the file that holds them must say which it is.
    """
    rng = np.random.RandomState(seed)
    sequences = list(rates)
    by_song = {}
    for sequence, (song, rate) in rates.items():
        by_song.setdefault(song, []).append(rate)

    if same_pairs is None:
        same = [
            abs(a - b)
            for members in by_song.values()
            for a, b in combinations(members, 2)
        ]
        same_unit = ("the grouping key: an AIST track, or a wild upload -- and a "
                     "wild upload's clips are cuts of one take, which makes this "
                     "close to tautological on that corpus")
    else:
        same, present = [], set(rates)
        for pair in same_pairs:
            left, right = sorted(pair)
            if left in present and right in present:
                same.append(abs(rates[left][1] - rates[right][1]))
        same_unit = ("verified cross-upload same-track pairs -- different "
                     "performances of one song, which is AIST's unit")
        # A pair list naming nothing here would silently fall through to "too
        # few pairs", which reads as a corpus too small rather than as a control
        # that never ran (CLAUDE.md 2).
        if not same:
            return {"checked": False, "same_song_unit": same_unit,
                    "reason": "the pair list names no two sequences of this split"}
    if len(same) < 8 or len(sequences) < 8:
        return {"checked": False, "same_song_unit": same_unit,
                "reason": "too few same-song pairs"}

    linked = set()
    if same_pairs is not None:
        for pair in same_pairs:
            left, right = sorted(pair)
            linked.add((left, right))

    diff, same_genre, diff_genre = [], [], []
    attempts, budget = 0, 200 * len(same) + 1000
    while len(diff) < len(same):
        attempts += 1
        if attempts > budget:
            return {"checked": False, "same_song_unit": same_unit,
                    "reason": "could not draw enough different-song pairs"}
        a, b = rng.choice(sequences, 2, replace=False)
        song_a, rate_a = rates[a]
        song_b, rate_b = rates[b]
        if song_a == song_b:
            continue
        # With a pair list, two clips of different uploads may still be proven
        # to share a track.  Leaving them in the "different song" arm would put
        # same-song pairs on both sides and drive the difference toward zero --
        # the control would then fail for the one reason that is not a finding.
        if tuple(sorted((a, b))) in linked:
            continue
        delta = abs(rate_a - rate_b)
        diff.append(delta)
        genre_a, genre_b = genre_of(song_a), genre_of(song_b)
        # Explicit, not incidental: on a corpus with no genre both sides are
        # None, and ``None == None`` would quietly file every pair as
        # *same*-genre.  The control would then be skipped for the right reason
        # by accident, and would start reporting a fabricated genre the moment
        # someone changed the bucket rule.
        if genre_a is not None and genre_b is not None:
            (same_genre if genre_a == genre_b else diff_genre).append(delta)

    _, p_song = mannwhitneyu(same, diff, alternative="less")
    genre_control = {"checked": False, "reason": "the corpus names no genre"}
    if len(same_genre) >= 8 and len(diff_genre) >= 8:
        _, p_genre = mannwhitneyu(same_genre, diff_genre, alternative="less")
        genre_control = {
            "checked": True,
            "same_genre_mean": float(np.mean(same_genre)),
            "diff_genre_mean": float(np.mean(diff_genre)),
            "p_value": float(p_genre),
        }

    # Split-half reliability of per-song mean rates: does a song have a
    # characteristic rate at all?
    firsts, seconds = [], []
    for members in by_song.values():
        if len(members) >= 2:
            shuffled = list(members)
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            firsts.append(float(np.mean(shuffled[:half])))
            seconds.append(float(np.mean(shuffled[half:])))
    reliability = (
        float(spearmanr(firsts, seconds).statistic) if len(firsts) >= 8 else None
    )

    return {
        "checked": True,
        # Which pairs were called "same song".  Two p-values from two units are
        # two different claims, and a report that does not name its unit lets
        # the weaker one be quoted as the stronger.
        "same_song_unit": same_unit,
        "sequences": len(sequences),
        "songs": len(by_song),
        "same_song_pairs": len(same),
        "same_song_mean_diff": float(np.mean(same)),
        "diff_song_mean_diff": float(np.mean(diff)),
        "p_value_same_lt_diff": float(p_song),
        "genre_control": genre_control,
        "per_song_rate_split_half_spearman": reliability,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path, help="score a release's GT labels")
    source.add_argument("--plans", type=Path, help="score generated plans JSON")
    parser.add_argument("--split", default="train",
                        help="split for --data-root; train has the most "
                             "same-song pairs and no held-out role for GT structure")
    parser.add_argument("--reference-root", type=Path, default=None,
                        help="plans mode: release supplying GT per-song rates "
                             "for the generated-vs-GT correlation")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--max-p-value", type=float, default=0.01)
    parser.add_argument("--music-pairs", type=Path, default=None,
                        help="fingerprint_wild_music.py's verified pair list; on the "
                             "wild corpus this is what makes the same-song arm mean "
                             "'two performances of one track' rather than 'two cuts "
                             "of one take', which is AIST's unit and the paper's")
    parser.add_argument("--control-plans", type=Path, default=None,
                        help="a positive control in the same --plans format, "
                             "normally tools/make_beat_grid_control.py's output; "
                             "scored with the identical statistic so the arm can "
                             "be read against something known to carry only tempo")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    same_pairs = None
    if args.music_pairs is not None:
        from tools.eval_r_precision import load_verified_music_pairs

        same_pairs, _ = load_verified_music_pairs(args.music_pairs)

    if args.data_root:
        rates = sequence_rates_from_release(args.data_root, args.split)
        source_description = "{}#{}".format(args.data_root, args.split)
    else:
        rates = rates_from_plans(args.plans)
        source_description = str(args.plans)

    report = {
        "source": source_description,
        "statistic": "segments-per-second of full label timelines",
        "seed": args.seed,
        "structure": pair_statistics(rates, args.seed, same_pairs=same_pairs),
    }

    if args.control_plans is not None:
        control = pair_statistics(rates_from_plans(args.control_plans), args.seed,
                                  same_pairs=same_pairs)
        arm_p = report["structure"].get("p_value_same_lt_diff")
        control_p = control.get("p_value_same_lt_diff")
        report["positive_control"] = {
            "plans": str(args.control_plans),
            "structure": control,
            "explained_by_tempo": (
                None if arm_p is None or control_p is None else bool(control_p <= arm_p)
            ),
            "reading": ("the control carries tempo and nothing else; if it scores at "
                        "least as well as the arm, this gate does not distinguish "
                        "'planned the movement' from 'followed the beat'"),
        }
    else:
        report["positive_control"] = {
            "supplied": False,
            "reading": ("no positive control was scored.  A metronome with random "
                        "labels reaches p = 2.0e-12 on clean5b5 train, better than "
                        "the ground truth, so a pass here is not evidence of "
                        "music-conditioned choreography.  Build one with "
                        "tools/make_beat_grid_control.py and pass --control-plans"),
        }

    if args.plans and args.reference_root:
        reference = sequence_rates_from_release(args.reference_root, args.split)
        by_song = {}
        for _, (song, rate) in reference.items():
            by_song.setdefault(song, []).append(rate)
        gt_rate = {song: float(np.mean(values)) for song, values in by_song.items()}
        generated = {}
        for _, (song, rate) in rates.items():
            generated.setdefault(song, []).append(rate)
        shared = sorted(set(gt_rate) & set(generated))
        if len(shared) >= 5:
            correlation = spearmanr(
                [gt_rate[song] for song in shared],
                [float(np.mean(generated[song])) for song in shared],
            )
            report["generated_vs_gt_rate"] = {
                "songs": len(shared),
                "spearman": float(correlation.statistic),
                "p_value": float(correlation.pvalue),
            }
        # Ceiling for that correlation: per-song GT rates of the *training*
        # performances vs this split's performances.  Dancers do not fully
        # agree on a song's rate across performance groups (first measurement:
        # rho = -0.13 over 16 shared songs), and a music-only planner can at
        # best reproduce the training group's rates -- so read
        # generated_vs_gt_rate against this number, not against 1.0.
        if args.split != "train":
            train_ref = sequence_rates_from_release(args.reference_root, "train")
            train_song = {}
            for _, (song, rate) in train_ref.items():
                train_song.setdefault(song, []).append(rate)
            shared = sorted(set(train_song) & set(gt_rate))
            if len(shared) >= 5:
                ceiling = spearmanr(
                    [float(np.mean(train_song[song])) for song in shared],
                    [gt_rate[song] for song in shared],
                )
                report["gt_cross_split_rate_ceiling"] = {
                    "songs": len(shared),
                    "spearman": float(ceiling.statistic),
                    "p_value": float(ceiling.pvalue),
                }

    structure = report["structure"]
    failures = []
    if not structure.get("checked"):
        failures.append("structure statistics unavailable: {}".format(
            structure.get("reason")))
    elif structure["p_value_same_lt_diff"] > args.max_p_value:
        failures.append(
            "same-song structural similarity is not significant "
            "(p={:.3g} > {:.3g}): the segmentation rate does not respond to "
            "the song".format(structure["p_value_same_lt_diff"], args.max_p_value)
        )
    report["gate"] = {"enforced": args.gate, "passed": not failures,
                      "failures": failures}

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if failures:
        for failure in failures:
            print("GATE FAILED: {}".format(failure))
        if args.gate:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
