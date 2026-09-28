#!/usr/bin/env python3
"""Is the prototype the same NUMBER OF BEATS as the slot it was played in?

THE GAP THIS MEASURES, named by the operator 2026-09-16: "检索 label to motion
从不检测拍数是吗,对卡节奏旋律挑选 motion 没有贡献?"  It does not.
``IndexedAtomicMotionLibrary._duration_band`` builds the candidate pool from
``abs(frames - target_length)`` alone, so a 2.4-second prototype that is six
beats of a 150 BPM song and one that is four beats of a 100 BPM song are equally
eligible for a four-beat slot -- and the winner is then stretched uniformly to
fill it, which is how a movement's own preparation and landing stop having
anything to do with THIS song (DEFECTS 1.6).

THE MEASUREMENT.  For every retrieved bar in an arm's ``prototype_retrieval.
retrieval_stretch.units_detail``:

    slot beats      = slot_frames   / the QUERY's median inter-beat interval
    prototype beats = native_frames / the SOURCE recording's own median interval
    mismatch        = |slot beats - prototype beats|

Both beat grids come from channel 34 of the two music arrays -- the query's from
``runs/txy_t_gt_eval/audio`` and the source's from the release's own
``music.npy``, which is frame-aligned with its motion.  NEITHER IS THE TARGET
CLIP'S MOTION, so this reads only what inference is allowed to read.

NO MOTION IS DECODED, deliberately.  The release is normalised to [-1, 1] and
three places in this repository used the [0, 1] inverse until 2026-09-16,
which silently doubles the skeleton; a column that never touches ``motion.npy``
cannot be wrong that way.  See tests/test_release_decode_matches_export.py.

WHAT GROUND TRUTH READS.  Ground truth does not retrieve, so it has no mismatch
by construction and cannot calibrate this column the way it calibrates
``settle`` or ``foot_skate``.  What it CAN calibrate is the alternative reading
of the same question -- whether a movement that is ``n`` beats long in its own
song stays ``n`` beats long -- which is 0 for ground truth trivially.  So the
line here is stated as what it is: an ARITHMETIC target of zero, with the
shipped arm's own distribution as the baseline to beat.  Measured on the twenty
eval clips' 200 bars before any filter existed: median 0.262 beats, p75 0.869,
p90 1.211, max 3.64, 37.5% of bars off by more than half a beat, and 40.0%
importing a prototype that is a different WHOLE NUMBER of beats than its slot.

READ IT BESIDE THE COUNTERS.  ``--draft-beat-span`` is a filter on an already
duration-tied pool; ``draft_beat_span_empty`` in the manifest says how often
nothing qualified and the whole tie was kept, which is the slot where the
tolerance bought nothing.  A tolerance whose ``empty`` rate is high is not
filtering, it is only pretending to.
"""
import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

BEAT_CHANNEL = 34


def beat_period(track, low=None, high=None):
    grid = np.flatnonzero(np.asarray(track)[:, BEAT_CHANNEL] > 0.5)
    if low is not None:
        inside = grid[(grid >= low) & (grid < high)]
        if len(inside) >= 2:
            return float(np.median(np.diff(inside)))
    return float(np.median(np.diff(grid))) if len(grid) >= 2 else float("nan")


def release_music(release):
    """``(split music array, names)`` for every split that has one."""
    out = {}
    for split in ("train", "val", "test"):
        path = pathlib.Path(release) / split
        if (path / "music.npy").is_file():
            out[split] = (np.load(str(path / "music.npy"), mmap_mode="r"),
                          json.loads((path / "names.json").read_text()))
    return out


def rows(arm, clips, audio_dir, release):
    music = release_music(release)
    out = []
    for clip in clips:
        path = pathlib.Path(arm) / (clip + ".pkl")
        query = pathlib.Path(audio_dir) / (clip + ".npy")
        if not (path.is_file() and query.is_file()):
            continue
        with open(path, "rb") as handle:
            record = pickle.load(handle)["prototype_retrieval"]
        units = record.get("retrieval_stretch", {}).get("units_detail") or []
        period = beat_period(np.load(str(query)))
        if period != period:
            continue
        for unit in units:
            index, start, end = unit["source"]
            for split, (array, names) in music.items():
                if index < len(names):
                    source = beat_period(np.asarray(array[index]), start, end)
                    break
            else:
                continue
            if source != source or source <= 0:
                continue
            out.append((clip,
                        unit["slot_frames"] / period,
                        unit["native_frames"] / source))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="runs/eval_clips_txy_t20.txt")
    ap.add_argument("--audio", default="runs/txy_t_gt_eval/audio")
    ap.add_argument("--release",
                    default="/cache/atomicdance-assets/scratch/txy_t/release_aligned_k8")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    args = ap.parse_args()

    clips = [c.strip() for c in pathlib.Path(args.clips).read_text().split() if c.strip()]
    print("{:<26}{:>7}{:>9}{:>8}{:>8}{:>8}{:>10}{:>10}".format(
        "arm", "bars", "median", "p75", "p90", "max", ">0.5 beat", "wrong n"))
    for spec in args.arm:
        name, _, directory = spec.partition("=")
        table = rows(directory, clips, args.audio, args.release)
        if not table:
            print("{:<26}{:>7}".format(name, 0))
            continue
        slot = np.array([r[1] for r in table])
        proto = np.array([r[2] for r in table])
        miss = np.abs(slot - proto)
        print("{:<26}{:>7}{:>9.3f}{:>8.3f}{:>8.3f}{:>8.2f}{:>9.1f}%{:>9.1f}%".format(
            name, len(table), float(np.median(miss)),
            float(np.percentile(miss, 75)), float(np.percentile(miss, 90)),
            float(miss.max()), float((miss > 0.5).mean() * 100),
            float((np.round(slot) != np.round(proto)).mean() * 100)))
        manifest = pathlib.Path(directory) / "manifest.json"
        if manifest.is_file():
            sampling = json.loads(manifest.read_text()).get("sampling", {})
            if sampling.get("draft_beat_span"):
                print("    tolerance {} beats; slots {}, narrowed {}, "
                      "nothing qualified {}, no query grid {}".format(
                          sampling.get("draft_beat_span"),
                          sampling.get("draft_beat_span_slots"),
                          sampling.get("draft_beat_span_applied"),
                          sampling.get("draft_beat_span_empty"),
                          sampling.get("draft_beat_span_blind")))
    print("\n'wrong n' = the prototype is a different WHOLE NUMBER of beats than "
          "its slot.\nGround truth has no reading here by construction; the "
          "target is 0 and the baseline is the shipped arm.")


if __name__ == "__main__":
    main()
