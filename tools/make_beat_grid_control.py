#!/usr/bin/env python3
"""A metronome with random labels, in the gate-v2 ``--plans`` format.

Why gate v2 needs this
----------------------
``probe_structure_conditioning`` asks whether same-song performances have more
similar *segments-per-second* than different-song pairs.  It has a floor (a
plan that ignores music scores nothing) and, until 2026-08-22, no ceiling and
no positive control -- the shape CLAUDE.md 2.1 says may not pass judgement.

Supplying one settles what the gate can and cannot see.  Cutting a plan every
four beats -- which is exactly M1's rule -- and filling each segment with a
**uniformly random class** produces a plan carrying zero information about
movement.  Scored on clean5b5 train (483 verified same-song pairs) it reads:

    ground truth                      p = 6.1e-03
    planner, stochastic + vote        p = 1.2e-04
    **metronome + random labels**     **p = 2.0e-12**

The control beats both.  So passing this gate says the plan's *rate* follows
the beat; it says nothing about which movement was chosen, and a planner that
had learned only tempo would pass it more convincingly than the ground truth
does.  Read the gate with this number beside it or not at all.

The beat channel is the last of the 35 music dimensions (envelope, 20 MFCC,
12 chroma, peak one-hot, beat one-hot), so the control needs no audio and no
model -- only the release the planner was trained on.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

BEAT_CHANNEL = 34
SLICE = re.compile(r"_slice(\d+)$")


def build(data_root: pathlib.Path, split: str, like: pathlib.Path, seed: int,
          beats_per_segment: int = 4):
    """One control plan per plan in ``like``, same sequence and same length."""
    data_root = pathlib.Path(data_root)
    names = json.loads((data_root / split / "names.json").read_text())
    music = np.load(data_root / split / "music.npy", mmap_mode="r")
    first = {}
    for index, name in enumerate(names):
        match = SLICE.search(name)
        if match and int(match.group(1)) == 0:
            first[SLICE.sub("", name)] = index

    payload = json.loads(pathlib.Path(like).read_text())
    plans = payload["plans"] if isinstance(payload, dict) else payload
    rng = np.random.RandomState(seed)
    classes = int(json.loads((data_root / "build.json").read_text())
                  ["window_policy"]["num_classes"])

    out, missing = [], 0
    for plan in plans:
        sequence = plan.get("sequence")
        index = first.get(sequence)
        if index is None:
            missing += 1
            continue
        frames = len(plan["labels"])
        beat = np.asarray(music[index, :, BEAT_CHANNEL]) > 0.5
        cuts = [int(f) for f in np.flatnonzero(beat)[::beats_per_segment] if 0 < f < frames]
        bounds = [0, *cuts, frames]
        labels = np.zeros(frames, dtype=int)
        for start, end in zip(bounds[:-1], bounds[1:]):
            # 1..num_classes-1: never the transition token, so the control is a
            # plan that is always dancing.  A control that also emitted
            # transitions would be testing two things at once.
            labels[start:end] = int(rng.randint(1, classes))
        out.append({"sequence": sequence, "song": plan.get("song"),
                    "labels": labels.tolist()})
    return {
        "plans": out,
        "generation_protocol": "CONTROL_beat_grid_random_labels",
        "headline_eligible": False,
        "headline_reason": "positive control for gate v2, not a generation result",
        "what_this_is": ("a cut every {} beats -- M1's own rule -- with uniformly "
                         "random classes; carries tempo and nothing else"
                         .format(beats_per_segment)),
        "source_release": str(data_root),
        "modelled_on": str(like),
        "sequences_without_a_first_slice": missing,
        "seed": seed,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--like", type=pathlib.Path, required=True,
                        help="plans JSON whose sequences and lengths the control copies")
    parser.add_argument("--beats-per-segment", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    report = build(args.data_root, args.split, args.like, args.seed,
                   args.beats_per_segment)
    if not report["plans"]:
        raise SystemExit("the control names no sequence of this release; a gate "
                         "read without a control is what this tool exists to prevent")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report), encoding="utf-8")
    print("wrote {} control plan(s) -> {}".format(len(report["plans"]), args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
