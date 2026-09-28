"""Census: how much of a generated clip is the SAME prototype pasted twice.

The retrieval side of this pipeline has a documented collapse mode, written
into ``IndexedAtomicMotionLibrary.retrieve``'s own docstring: the ``duration``
rule is a deterministic ``min`` whose result is CACHED on
``(label, target_length, ...)``, so two segments of the same class and the same
length receive *the identical tensor*.  ``--draft-recurrence-variety`` exists to
break that cache, and it is off in the shipped arm.

The 4-beat bar grid interacts with it by construction rather than by accident:
forcing every plan segment to the same number of beats makes ``target_length``
near-constant across a clip, so the cache key collides on essentially every
repeat of a label.

This tool reconstructs the retrieval -- same library, same rule, same exclusion
policy, same per-clip generator -- and reports, per clip:

  * how many plan segments there are, and how many DISTINCT prototypes served
    them (a prototype being a specific span of a specific source recording);
  * the share of segments whose prototype had already been used in that clip;
  * the largest number of segments served by any single prototype.

WHAT THIS DOES NOT ANSWER.  Ground truth choreography repeats on purpose -- a
chorus dances like a chorus -- so "the plan repeats a class" is not by itself a
defect, and this census deliberately does not gate on it.  What a real dancer
does NOT do is perform the repeat as the identical bytes, and that is what the
distinct-prototype count reads.  ``tools/measure_motion_repetition.py`` scores
the rendered result against ground truth on the same clips.
"""

import argparse
import collections
import json
import pathlib
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.atomic import labels_to_segments
from infer_atomic import IndexedAtomicMotionLibrary, _variety_rng


def reconstruct(library, labels, *, exclude, recurrence_variety, variety_rng):
    """Re-run the draft's retrieval and record WHICH prototype each segment got.

    Mirrors ``build_draft``'s loop: transition segments are skipped because they
    are never retrieved, and ``occurrence`` counts repeats of a label exactly as
    that loop counts them.
    """
    picks = []
    seen = {}
    for segment in labels_to_segments(torch.as_tensor(np.asarray(labels))):
        if segment.label == 0:
            continue
        occurrence = seen.get(int(segment.label), 0)
        seen[int(segment.label)] = occurrence + 1
        try:
            values = library.retrieve(
                segment.label, segment.length,
                exclude_retrieval_group_ids=exclude,
                occurrence=(occurrence if recurrence_variety else 0),
                variety_rng=variety_rng)
        except KeyError:
            picks.append((int(segment.label), None))
            continue
        # The tensor itself identifies the prototype: two segments served the
        # same bytes are the same prototype, whatever bookkeeping says.
        digest = hash(np.asarray(values, dtype=np.float32).tobytes())
        picks.append((int(segment.label), digest))
    return picks


def summarise_clip(picks):
    served = [digest for _, digest in picks if digest is not None]
    counts = collections.Counter(served)
    return {
        "segments": len(picks),
        "distinct_prototypes": len(counts),
        "repeat_share": (1.0 - len(counts) / len(served)) if served else None,
        "largest_prototype_run": max(counts.values()) if counts else 0,
        "distinct_labels": len({label for label, _ in picks}),
    }


def run(data_root, run_dir, clips, *, recurrence_variety, seed):
    manifest = json.loads((pathlib.Path(run_dir) / "manifest.json").read_text())
    sampling = manifest.get("sampling", {})
    library = IndexedAtomicMotionLibrary(
        data_root,
        retrieval_rule=sampling.get("retrieval_rule", "duration"),
        energy_floor_quantile=sampling.get("retrieval_energy_floor"))
    rows, per_clip = [], {}
    for clip in clips:
        path = pathlib.Path(run_dir) / (clip + ".pkl")
        if not path.exists():
            continue
        payload = pickle.load(open(path, "rb"))
        group = payload["prototype_retrieval"]["query_retrieval_group_id"]
        picks = reconstruct(
            library, payload["atomic_labels"], exclude=(group,),
            recurrence_variety=recurrence_variety,
            variety_rng=_variety_rng(seed, clip))
        summary = summarise_clip(picks)
        per_clip[clip] = summary
        rows.append(summary)
    if not rows:
        raise SystemExit(
            "error: none of the {} clips were found in {}.  Nothing was "
            "measured.".format(len(clips), run_dir))

    def median(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.median(values)) if values else None

    return {
        "run_dir": str(run_dir),
        "recurrence_variety": bool(recurrence_variety),
        "seed": seed,
        "clips": len(rows),
        "median_segments": median("segments"),
        "median_distinct_prototypes": median("distinct_prototypes"),
        "median_distinct_labels": median("distinct_labels"),
        "median_repeat_share": median("repeat_share"),
        "median_largest_prototype_run": median("largest_prototype_run"),
        "worst_clip": max(per_clip.items(),
                          key=lambda kv: kv[1]["largest_prototype_run"])[0],
        "per_clip": per_clip,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--recurrence-variety", action="store_true",
                        help="reconstruct as if --draft-recurrence-variety were on")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--out", type=pathlib.Path)
    arguments = parser.parse_args()

    clips = [line.strip() for line in arguments.clips.read_text().splitlines()
             if line.strip()]
    report = run(arguments.data_root, arguments.run_dir, clips,
                 recurrence_variety=arguments.recurrence_variety,
                 seed=arguments.seed)
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.out:
        arguments.out.write_text(text)
    print(json.dumps({k: v for k, v in report.items() if k != "per_clip"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
