#!/usr/bin/env python3
"""Decide whether a captioner is good enough to drive M3's ``w/ LLM`` row.

Reading a handful of captions and finding them plausible does not answer the
question that matters.  These captions exist only to be clustered, so the
useful properties are measurable:

G1 -- **usable output**.  What fraction of replies parse into the schema, and
   how often does a field come back outside the vocabulary?  A captioner that
   is eloquent but off-schema half the time contributes noise.

G2 -- **consistency**.  Two segments that TMR already placed in the same
   prototype are, by construction, similar movements.  If the captioner is
   tracking movement rather than incidentals (clothing, background, camera),
   field agreement within a prototype must exceed agreement across prototypes.
   The gap is the signal; the ratio is reported as the lift.

G3 -- **discriminativeness**.  A captioner that answers "step, arms down,
   middle level" for everything would score perfectly on G2 and be useless.
   So the number of distinct captions and the share taken by the single most
   common caption are reported alongside.

G2 and G3 pull against each other on purpose: a captioner passes only by being
consistent *and* varied.  A permutation test says whether the G2 gap could have
arisen by chance.

``--compare`` adds the cross-check the plan calls for: two captioners run over
the same segments, scored on how often they say the same thing.  It is reported
against the agreement of *randomly paired* captions from the same two files,
because two models that both answer "step, arms down, middle level" most of the
time will agree often while agreeing about nothing.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.caption_segments_vlm import FIELDS  # noqa: E402


def load(path: pathlib.Path) -> List[Dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def field_agreement(a: Dict[str, str], b: Dict[str, str]) -> float:
    """Fraction of the schema's fields on which two captions agree."""
    return float(np.mean([a.get(f) == b.get(f) for f in FIELDS]))


def sample_pairs(rows: Sequence[Dict[str, object]], *, same: bool, count: int,
                 rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Pairs drawn within a prototype (same=True) or across prototypes."""
    by_prototype: Dict[int, List[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_prototype[int(row["prototype"])].append(index)
    eligible = [ids for ids in by_prototype.values() if len(ids) >= 2]
    pairs: List[Tuple[int, int]] = []
    if same:
        if not eligible:
            return pairs
        for _ in range(count):
            group = eligible[rng.integers(len(eligible))]
            i, j = rng.choice(len(group), size=2, replace=False)
            pairs.append((group[i], group[j]))
    else:
        prototypes = sorted(by_prototype)
        if len(prototypes) < 2:
            return pairs
        for _ in range(count):
            a, b = rng.choice(len(prototypes), size=2, replace=False)
            left = by_prototype[prototypes[a]]
            right = by_prototype[prototypes[b]]
            pairs.append((left[rng.integers(len(left))], right[rng.integers(len(right))]))
    return pairs


def permutation_p(within: np.ndarray, across: np.ndarray, *, rounds: int,
                  rng: np.random.Generator) -> float:
    """How often a random relabelling reproduces the observed gap."""
    observed = within.mean() - across.mean()
    pooled = np.concatenate([within, across])
    hits = 0
    for _ in range(rounds):
        rng.shuffle(pooled)
        if pooled[:len(within)].mean() - pooled[len(within):].mean() >= observed:
            hits += 1
    return (hits + 1) / (rounds + 1)


def cross_agreement(left: Sequence[Dict[str, object]], right: Sequence[Dict[str, object]],
                    *, rng: np.random.Generator) -> Dict[str, object]:
    """How often two captioners describe the same segment the same way.

    The chance baseline is the same statistic over randomly paired segments.
    Without it the number is unreadable: two captioners that both answer with
    the corpus's most common movement most of the time will agree constantly
    while carrying no information about the segment in front of them.
    """
    index = {(row["recording_id"], row["start"], row["end"]): row for row in right}
    shared = [(row, index[(row["recording_id"], row["start"], row["end"])])
              for row in left
              if (row["recording_id"], row["start"], row["end"]) in index]
    if not shared:
        return {"skipped": "the two files share no segment"}
    matched = np.asarray([field_agreement(dict(a["fields"]), dict(b["fields"]))
                          for a, b in shared])
    permuted = rng.permutation(len(shared))
    chance = np.asarray([field_agreement(dict(shared[i][0]["fields"]),
                                         dict(shared[j][1]["fields"]))
                         for i, j in enumerate(permuted)])
    same_sentence = float(np.mean([a["caption"] == b["caption"] for a, b in shared]))
    return {
        "shared_segments": len(shared),
        "field_agreement": round(float(matched.mean()), 4),
        "field_agreement_if_shuffled": round(float(chance.mean()), 4),
        "lift_over_chance": round(float(matched.mean() / max(1e-9, chance.mean())), 3),
        "identical_caption_rate": round(same_sentence, 4),
        "per_field": {field: round(float(np.mean([a["fields"].get(field) == b["fields"].get(field)
                                                  for a, b in shared])), 4)
                      for field in FIELDS},
    }


def probe(path: pathlib.Path, *, pairs: int, rounds: int, seed: int,
          compare: Optional[pathlib.Path] = None) -> Dict[str, object]:
    rows = load(path)
    if len(rows) < 4:
        raise SystemExit("error: {} has too few captions to probe".format(path))
    rng = np.random.default_rng(seed)

    fields = [dict(row["fields"]) for row in rows]
    unspecified = {
        field: round(float(np.mean([f.get(field) == "unspecified" for f in fields])), 4)
        for field in FIELDS
    }

    sentences = [str(row["caption"]) for row in rows]
    counter = collections.Counter(sentences)
    top_caption, top_count = counter.most_common(1)[0]

    within_pairs = sample_pairs(rows, same=True, count=pairs, rng=rng)
    across_pairs = sample_pairs(rows, same=False, count=pairs, rng=rng)
    within = np.asarray([field_agreement(fields[i], fields[j]) for i, j in within_pairs])
    across = np.asarray([field_agreement(fields[i], fields[j]) for i, j in across_pairs])

    result: Dict[str, object] = {
        "captions": len(rows),
        "models": sorted({str(row.get("model", "unknown")) for row in rows}),
        "G1_usable": {
            "unspecified_rate_per_field": unspecified,
            "mean_unspecified_rate": round(float(np.mean(list(unspecified.values()))), 4),
            "note": ("parse failures never reach this file; they are counted as "
                     "'unparsed' by the captioner itself"),
        },
        "G3_discriminative": {
            "distinct_captions": len(counter),
            "distinct_ratio": round(len(counter) / len(rows), 4),
            "largest_caption_share": round(top_count / len(rows), 4),
            "most_common_caption": top_caption,
        },
    }

    if len(within) and len(across):
        result["G2_consistency"] = {
            "within_prototype_agreement": round(float(within.mean()), 4),
            "across_prototype_agreement": round(float(across.mean()), 4),
            "lift": round(float(within.mean() / max(1e-9, across.mean())), 3),
            "permutation_p": permutation_p(within, across, rounds=rounds, rng=rng),
            "pairs_per_side": int(min(len(within), len(across))),
        }
    else:
        result["G2_consistency"] = {
            "skipped": "not enough prototypes with two captioned segments; "
                       "caption more segments before reading this gate"}

    if compare is not None:
        result["G4_cross_captioner"] = {
            "other": str(compare),
            **cross_agreement(rows, load(compare), rng=rng),
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=pathlib.Path, required=True)
    parser.add_argument("--compare", type=pathlib.Path, default=None,
                        help="second caption file over the same segments; reports how "
                             "often the two captioners agree, against a shuffled baseline")
    parser.add_argument("--pairs", type=int, default=4000)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = probe(args.captions, pairs=args.pairs, rounds=args.rounds, seed=args.seed,
                   compare=args.compare)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
