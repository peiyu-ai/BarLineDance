#!/usr/bin/env python3
"""How much do two people dancing the same song agree, frame by frame?

Why this exists
---------------
``probe_label_predictability`` asked whether music predicts the atomic label,
and gated on beating the **majority-class** baseline -- the share of the single
most common label, which on every release here is the transition token
(~0.20-0.23 of frames).  That gate had a floor and no ceiling, and CLAUDE.md 2.1
says a criterion may not pass judgement until it has both.

Measured on clean5b5 (2026-08-22), 627 fingerprint-verified cross-upload
same-track pairs aligned by the lag the fingerprint itself recorded:

    two different dancers, same song      all frames 0.1613   atomic-only 0.0481
    two dancers, different songs          all frames 0.0854   atomic-only 0.0037

**0.1613 is below the 0.2321 majority baseline the gate demanded.**  Two humans
performing the same track agree less often than a constant "always transition"
predictor scores, so the old gate could not be passed by the ground truth
itself, let alone by a model.  It was not a strict criterion; it was an
impossible one -- the mirror image of the never-firing gate CLAUDE.md opens
with, and just as useless.

Choreography is one-to-many: the same music admits many correct dances.  The
reachable ceiling for "predict which atomic movement happens now" is therefore
*another dancer's* answer, and that is what this module measures.  The
different-song arm is the floor, and it is not zero (0.0037 atomic-only, 0.0854
overall) because label frequencies are skewed -- a criterion that ignored it
would credit a model for the label prior.

What it does not measure
------------------------
Whether the vocabulary is *good*.  Two dancers might agree 5% of the time
because the vocabulary is too fine to be agreed on, or because dance is free.
This separates neither; it only says what number is reachable, so a probe's
score can be read as a fraction of it instead of against an impossible bar.

Same-account pairs are kept and counted separately in the report: one
choreographer repeating their own routine agrees with themselves more than two
strangers do, so the ceiling reported here is, if anything, generous.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]

# Two timelines that overlap for less than two seconds after alignment cannot
# say anything about agreement; the pair is dropped and counted, never scored.
MIN_OVERLAP_FRAMES = 60


def load_label_timelines(labels_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    """sequence_id -> the .npy holding its frame labels."""
    labels_dir = pathlib.Path(labels_dir)
    manifest = labels_dir / "labels.jsonl"
    if not manifest.exists():
        raise SystemExit("no labels.jsonl under {}".format(labels_dir))
    out: Dict[str, pathlib.Path] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            out[row["sequence_id"]] = labels_dir / row["labels_path"]
    return out


def _align(left: np.ndarray, right: np.ndarray, lag: int):
    """Overlapping frames of two timelines offset by ``lag`` frames."""
    if lag >= 0:
        a, b = left[lag:], right[: max(len(left) - lag, 0)]
    else:
        a, b = left[: max(len(right) + lag, 0)], right[-lag:]
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _agreement(a: np.ndarray, b: np.ndarray, label_map: Optional[np.ndarray]):
    if label_map is not None:
        a, b = label_map[a], label_map[b]
    both_atomic = (a != 0) & (b != 0)
    atomic = float((a[both_atomic] == b[both_atomic]).mean()) if both_atomic.sum() >= 30 else None
    return float((a == b).mean()), atomic, int(both_atomic.sum())


def measure(labels_dir, pairs_path, *, label_map=None, seed=20260822,
            control_pairs=600, min_overlap=MIN_OVERLAP_FRAMES) -> Dict[str, object]:
    """Same-song ceiling and different-song floor for a label space.

    ``label_map`` may arrive as a torch tensor: ``probe_label_predictability``
    loads it that way for the model, and indexing numpy labels with a torch
    tensor yields a torch bool that ``.mean()`` refuses -- caught by the first
    end-to-end run, not by the unit tests, which passed numpy.
    """
    if label_map is not None:
        label_map = np.asarray(
            label_map.detach().cpu() if hasattr(label_map, "detach") else label_map)
    timelines = load_label_timelines(labels_dir)
    cache: Dict[str, np.ndarray] = {}

    def load(seq):
        if seq not in cache:
            if len(cache) > 400:
                cache.clear()
            cache[seq] = np.load(timelines[seq])
        return cache[seq]

    pairs = []
    with pathlib.Path(pairs_path).open(encoding="utf-8") as handle:
        for line in handle:
            pairs.append(json.loads(line))

    same, same_atomic, dropped, cross_account = [], [], 0, 0
    for pair in pairs:
        left, right = pair.get("left"), pair.get("right")
        if left not in timelines or right not in timelines:
            continue
        a, b = _align(load(left), load(right), int(pair.get("lag_frames", 0)))
        if len(a) < min_overlap:
            dropped += 1
            continue
        overall, atomic, _ = _agreement(a, b, label_map)
        same.append(overall)
        if atomic is not None:
            same_atomic.append(atomic)
        if left.split(":")[1] != right.split(":")[1]:
            cross_account += 1

    rng = np.random.RandomState(seed)
    keys = sorted(timelines)
    diff, diff_atomic = [], []
    attempts = 0
    while len(diff) < control_pairs and attempts < control_pairs * 20:
        attempts += 1
        left, right = keys[rng.randint(len(keys))], keys[rng.randint(len(keys))]
        if left == right:
            continue
        # A random lag, because a same-song pair is aligned by a measured one:
        # scoring the control at lag 0 would give it a systematically different
        # amount of overlap than the arm it is the control for.
        a, b = _align(load(left), load(right), int(rng.randint(-90, 90)))
        if len(a) < min_overlap:
            continue
        overall, atomic, _ = _agreement(a, b, label_map)
        diff.append(overall)
        if atomic is not None:
            diff_atomic.append(atomic)

    def summarise(values):
        return float(np.mean(values)) if values else None

    if not same:
        raise SystemExit(
            "the pair list names no two sequences of this label set; the "
            "ceiling would be undefined and no fraction-of-ceiling may be read")

    return {
        "what": "frame agreement between two performances of one track",
        "ceiling_is": "same-song pairs: the reachable maximum for predicting "
                      "which atomic movement happens now",
        "floor_is": "different-song pairs: what the label prior alone buys",
        "same_song": {
            "pairs": len(same),
            "cross_upload_pairs": cross_account,
            "all_frames": summarise(same),
            "atomic_frames": summarise(same_atomic),
            "atomic_scored_pairs": len(same_atomic),
        },
        "different_song": {
            "pairs": len(diff),
            "all_frames": summarise(diff),
            "atomic_frames": summarise(diff_atomic),
        },
        "dropped_short_overlap": dropped,
        "min_overlap_frames": min_overlap,
        "labels": str(labels_dir),
        "pairs_path": str(pairs_path),
        "seed": seed,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=pathlib.Path, required=True,
                        help="directory holding labels.jsonl and labels/<hash>/labels.npy")
    parser.add_argument("--music-pairs", type=pathlib.Path, required=True,
                        help="fingerprint-verified cross-upload same-track pairs "
                             "(tools/fingerprint_wild_music.py), with lag_frames")
    parser.add_argument("--label-map", type=pathlib.Path, default=None,
                        help="coarse vocabulary map from build_coarse_vocabulary.py")
    parser.add_argument("--control-pairs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    label_map = None
    if args.label_map:
        payload = json.loads(args.label_map.read_text())
        label_map = np.asarray(payload["label_map"] if isinstance(payload, dict) else payload)

    report = measure(args.labels, args.music_pairs, label_map=label_map,
                     seed=args.seed, control_pairs=args.control_pairs)
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
