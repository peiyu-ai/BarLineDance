#!/usr/bin/env python3
"""Does having more people in the clip make the 3D worse, once the shot width is held?

Three reports already exist and none of them answers this alone:

* ``runs/wild_clip_population.json`` -- how many people are in each clip and how
  big the tracked subject is (``tools/audit_clip_population.py``);
* ``runs/wild_dancer_track_switches.json`` -- whether the track walked from one
  dancer onto another (``tools/audit_dancer_tracks.py``);
* ``runs/wild_v4_3d_quality.json`` -- the physical measures, beside AIST++ mocap
  (``tools/audit_wild_3d_quality.py``).

Joining them is the whole tool, and the join is where the care goes.

**The confound is shot width, and it is not small.**  Group footage is filmed
wide, so a multi-person clip is also a clip where the subject is a fraction of
the frame, and monocular reconstruction of a small subject is worse for reasons
that have nothing to do with the crowd.  Comparing solo against multi without
holding subject size is therefore a measurement of framing wearing a
measurement of population as a disguise.  Every comparison here is inside a
band of subject size.

**Switched clips are excluded, and are not the question.**  A track that walks
from dancer A to dancer B produces a motion that splices two people, which is
damage nobody disputes -- and stage B's worklist already drops them
(``excluded_dancer_switch``), leaving 0.22% in the released set.  The open
question is the clips that are *not* spliced, so those are what is compared.

**Read the reference carefully.**  AIST++ is right by construction, but foot
skate depends on what is being danced, so "wild is below mocap" is not a
quality verdict -- the two corpora are not dancing the same thing.  The
reference fixes the order of magnitude; the solo-against-multi contrast inside
one corpus is what carries the argument.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MEASURES = ("jitter_ratio", "skate_median", "root_speed_max",
            "penetration_fraction", "frozen_fraction", "lowest_toe_spread")


def recording_to_clip(recording_id: str) -> Optional[str]:
    parts = recording_id.split(":")
    return "{}__{}".format(parts[1], parts[2]) if len(parts) == 3 else None


def main(argv: Optional[List[str]] = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", type=pathlib.Path,
                        default=pathlib.Path("runs/wild_clip_population.json"))
    parser.add_argument("--switches", type=pathlib.Path,
                        default=pathlib.Path("runs/wild_dancer_track_switches.json"))
    parser.add_argument("--quality", type=pathlib.Path,
                        default=pathlib.Path("runs/wild_v4_3d_quality.json"))
    parser.add_argument("--bands", type=int, default=5)
    parser.add_argument("--min-per-cell", type=int, default=25,
                        help="a band with fewer than this on either side is "
                             "reported as too few rather than compared")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    population = {r["clip"]: r for r in
                  json.loads(args.population.read_text(encoding="utf-8"))["rows"]}
    switches = {r["clip"]: r for r in
                json.loads(args.switches.read_text(encoding="utf-8"))["records"]}
    quality = json.loads(args.quality.read_text(encoding="utf-8"))

    rows, dropped = [], {"no_population": 0, "no_subject_area": 0, "switched": 0}
    for record in quality["records"]:
        clip = recording_to_clip(record["recording_id"])
        if clip is None or clip not in population:
            dropped["no_population"] += 1
            continue
        entry = population[clip]
        if entry.get("subject_area_median") is None:
            dropped["no_subject_area"] += 1
            continue
        if switches.get(clip, {}).get("switches"):
            dropped["switched"] += 1
            continue
        rows.append({"clip": clip, "solo": entry["solo"],
                     "area": entry["subject_area_median"],
                     "account": entry.get("account"),
                     **{k: record.get(k) for k in MEASURES}})

    def median(values):
        clean = [v for v in values if v is not None and np.isfinite(v)]
        return round(float(np.median(clean)), 4) if clean else None

    reference = {k: median([r.get(k) for r in quality["reference_records"]])
                 for k in MEASURES}
    solo = [r for r in rows if r["solo"]]
    multi = [r for r in rows if not r["solo"]]

    areas = np.array([r["area"] for r in rows])
    edges = np.percentile(areas, np.linspace(0, 100, args.bands + 1))
    bands = []
    for index in range(args.bands):
        low, high = float(edges[index]), float(edges[index + 1])
        if index < args.bands - 1:
            members = [r for r in rows if low <= r["area"] < high]
        else:
            members = [r for r in rows if low <= r["area"] <= high]
        inside_solo = [r for r in members if r["solo"]]
        inside_multi = [r for r in members if not r["solo"]]
        band = {"area_low": round(low, 4), "area_high": round(high, 4),
                "solo_clips": len(inside_solo), "multi_clips": len(inside_multi)}
        if min(len(inside_solo), len(inside_multi)) < args.min_per_cell:
            band["comparable"] = False
        else:
            band["comparable"] = True
            for measure in MEASURES:
                s = median([r[measure] for r in inside_solo])
                m = median([r[measure] for r in inside_multi])
                band[measure] = {"solo": s, "multi": m,
                                 "multi_over_solo": (round(m / s, 3)
                                                     if s not in (None, 0) and m is not None
                                                     else None)}
        bands.append(band)

    result = {
        "generated_by": "tools/report_population_quality.py",
        "clips_compared": len(rows),
        "dropped": dropped,
        "solo_clips": len(solo),
        "multi_clips": len(multi),
        "reference_median": reference,
        "overall": {m: {"solo": median([r[m] for r in solo]),
                        "multi": median([r[m] for r in multi])} for m in MEASURES},
        "bands_by_subject_area": bands,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    print("compared {} clips ({} solo, {} multi); dropped {}".format(
        len(rows), len(solo), len(multi), dropped))
    print("\n{:<22} {:>10} {:>10} {:>11}".format("measure", "AIST++", "solo", "multi"))
    for measure in MEASURES:
        print("{:<22} {!s:>10} {!s:>10} {!s:>11}".format(
            measure, reference[measure], result["overall"][measure]["solo"],
            result["overall"][measure]["multi"]))
    print("\nheld at the same subject size:")
    head = "{:<16} {:>5} {:>6}".format("area band", "solo", "multi")
    for measure in ("jitter_ratio", "skate_median", "root_speed_max"):
        head += " | {:>19}".format(measure)
    print(head)
    for band in bands:
        line = "{:<16} {:>5} {:>6}".format(
            "{:.3f}-{:.3f}".format(band["area_low"], band["area_high"]),
            band["solo_clips"], band["multi_clips"])
        if not band["comparable"]:
            print(line + " | too few on one side")
            continue
        for measure in ("jitter_ratio", "skate_median", "root_speed_max"):
            cell = band[measure]
            line += " | {:>8} {:>8} {:>4.2f}x".format(
                cell["solo"], cell["multi"], cell["multi_over_solo"] or float("nan"))
        print(line)
    print("\nwrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
