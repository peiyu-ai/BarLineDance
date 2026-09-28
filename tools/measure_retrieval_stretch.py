"""What retrieval does to a prototype on its way into the plan's slot.

``IndexedAtomicMotionLibrary._values_at`` fills a plan slot with ONE global
linear resample of the chosen prototype::

    values = F.interpolate(values.T.unsqueeze(0), size=target_length,
                           mode="linear", align_corners=True).squeeze(0).T

Two consequences follow arithmetically, and this tool measures both rather than
assuming them:

  * A prototype stretched by a factor s has EVERY velocity divided by s and
    every acceleration divided by s squared.  Stretch a 1.4 s move into a 2.1 s
    slot and the dancer performs it at 2/3 speed -- which is what "the moves are
    small and mushy, nothing lands" describes.
  * A linear resample is uniform, so the move's internal accents -- the
    preparation, the strike, the recovery -- move in proportion.  An accent that
    sat on beat 2 of its own bar does not land on beat 2 of the query's bar
    unless the two bars happen to be the same length.

WHY THIS IS STRUCTURAL HERE RATHER THAN OCCASIONAL.  The corpus was cut on
4-beat bars and the plan is snapped to 4-beat bars, so both sides are "one bar".
But a bar is a TEMPO-dependent duration: four beats at 100 BPM is 2.4 s and at
140 BPM is 1.7 s.  Retrieval draws from every song in the library, so the
stretch factor is the ratio of two songs' tempi, and matching on ``min |duration
difference|`` picks the closest available length rather than a good one.

The energy-percentile column asks a different question: whether the duration
rule, by always taking the nearest length, systematically lands on a particular
kind of exemplar within a class.  A median percentile near 50 means it does not.
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

FPS = 30.0


def safe_candidates(library, label, exclude):
    """The pool ``retrieve`` would see, with the same fail-closed exclusion.

    Copied from ``retrieve`` rather than called into it because ``retrieve``
    returns the RESAMPLED tensor and this tool needs the native span length,
    which resampling has already destroyed by then.
    """
    candidates = library.index.get(int(label), ())
    if exclude:
        candidates = tuple(
            candidate for candidate in candidates
            if isinstance(candidate[3], str) and candidate[3]
            and candidate[3] not in exclude)
    return library._energy_floor(label, candidates,
                                 library.energy_floor_quantile)


def chosen_candidate(library, label, target_length, exclude):
    """Re-run the duration rule and return the candidate it picks, unresampled."""
    candidates = safe_candidates(library, label, frozenset(exclude))
    if not candidates:
        return None, ()
    chosen = min(candidates,
                 key=lambda item: abs((item[2] - item[1]) - target_length))
    return chosen, candidates


def energy_percentile(library, chosen, candidates):
    """Where the chosen prototype's energy sits inside its own class pool."""
    if len(candidates) < 2:
        return None
    energies = np.array([library._segment_energy(c) for c in candidates], float)
    mine = library._segment_energy(chosen)
    return float((energies < mine).mean() * 100.0)


def measure(library, run_dir, clips, seed=20260902):
    rows = []
    for clip in clips:
        path = pathlib.Path(run_dir) / (clip + ".pkl")
        if not path.exists():
            continue
        payload = pickle.load(open(path, "rb"))
        exclude = (payload["prototype_retrieval"]["query_retrieval_group_id"],)
        labels = torch.as_tensor(np.asarray(payload["atomic_labels"]))
        for segment in labels_to_segments(labels):
            if segment.label == 0:
                continue
            chosen, candidates = chosen_candidate(
                library, segment.label, segment.length, exclude)
            if chosen is None:
                continue
            native = int(chosen[2] - chosen[1])
            if native <= 0:
                continue
            rows.append({
                "clip": clip,
                "label": int(segment.label),
                "slot_frames": int(segment.length),
                "native_frames": native,
                "stretch": float(segment.length) / native,
                "pool": len(candidates),
                "energy_pct": energy_percentile(library, chosen, candidates),
            })
    return rows


def summarise(rows):
    stretch = np.array([r["stretch"] for r in rows], float)
    energy = np.array([r["energy_pct"] for r in rows if r["energy_pct"] is not None],
                      float)
    slowed = stretch > 1.0
    return {
        "segments": len(rows),
        "slot_seconds_median": round(float(np.median(
            [r["slot_frames"] for r in rows])) / FPS, 3),
        "native_seconds_median": round(float(np.median(
            [r["native_frames"] for r in rows])) / FPS, 3),
        "stretch_median": round(float(np.median(stretch)), 3),
        "stretch_p10": round(float(np.percentile(stretch, 10)), 3),
        "stretch_p90": round(float(np.percentile(stretch, 90)), 3),
        "share_slowed_down": round(float(slowed.mean()), 3),
        "share_beyond_1_4x": round(float((stretch > 1.4).mean()), 3),
        "share_below_0_7x": round(float((stretch < 0.7).mean()), 3),
        # A prototype stretched by s runs at 1/s of its own speed.
        "median_speed_factor": round(float(np.median(1.0 / stretch)), 3),
        "pool_median": int(np.median([r["pool"] for r in rows])),
        "energy_percentile_median": (round(float(np.median(energy)), 1)
                                     if energy.size else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    arguments = parser.parse_args()

    clips = [line.strip() for line in arguments.clips.read_text().splitlines()
             if line.strip()]
    library = IndexedAtomicMotionLibrary(arguments.data_root,
                                         retrieval_rule="duration")
    rows = measure(library, arguments.run_dir, clips)
    if not rows:
        raise SystemExit("error: no segments measured from {}".format(arguments.run_dir))
    report = {"run_dir": str(arguments.run_dir), **summarise(rows)}
    if arguments.out:
        arguments.out.write_text(json.dumps(
            {"summary": report, "rows": rows}, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
