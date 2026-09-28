#!/usr/bin/env python3
"""How many people are in each clip, how big the subject is, and how steady the shot.

Motion quality is decided before any model runs.  GVHMR is handed one box per
frame and reconstructs whoever is inside it, so a clip where the subject is a
sixteenth of the frame, or where three other dancers cross it, or where the
camera swings, produces a 3D motion that is technically valid and describes
something other than one person dancing.  Nothing in this repo measured that,
and the only signals that existed answer a different question:

* ``rival_count`` / ``rival_ratio`` count *tracks scoring at least half the
  winner*, where the score is persistence squared x root area x motion.  That
  is deliberately a "is there another plausible **dancer**" test, and it is
  computed **per upload** and copied into every clip of that upload.  A clip of
  a solo passage inside a group video carries the group's number.
* ``dancer_present_fraction`` asks whether the chosen dancer is on screen, not
  who else is.

``detections.npz`` already holds what is wanted, per clip and per frame:
``person_count`` (every box yolox_l returned for that frame) and ``dancer_box``
(the one that was tracked).  Reading them costs nothing and needs no GPU.

Three readings, and what each does and does not mean:

* **population** -- ``person_count`` per frame.  This counts people, not
  dancers: an audience, a queue in the background, and a studio's mirror
  reflections are all people to a detector.  A high count is therefore a reason
  to look, not a verdict, and this tool does not claim to separate a septet
  from a trio in front of a mirror.
* **subject scale** -- the tracked box's area as a fraction of the frame,
  over the frames where the dancer is actually present.  ``dancer_box`` is NaN
  where they are not, and a plain ``median`` over it returns NaN, which then
  sorts into the middle of a distribution and quietly moves every quantile.
  Every statistic here is a nan-aware one for that reason.
* **shot steadiness** -- how much the tracked box's centre and size move
  between consecutive frames, in units of the box's own size.  A dancer moving
  through a static frame and a static dancer in a moving frame are not
  separated by this; it is a measure of how much the crop has to travel, which
  is what costs the reconstruction.

Usage::

    audit_clip_population.py --output runs/wild_clip_population.json
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as futures
import json
import pathlib
import sys
from typing import Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

INGEST = pathlib.Path("/cache/atomicdance-assets/data/wild_ingest_v1")
BUNDLE_SOURCES = pathlib.Path("/cache/atomicdance-assets/data/wild3d/"
                              "wild_v4_raw_bundle/sources.jsonl")
ACCOUNTS = pathlib.Path("/cache/atomicdance-assets/data/wild3d/wild_v4_acct/"
                        "group_keys.json")
# A clip counts as solo when the detector sees at most one person in this share
# of its frames.  0.9 rather than 1.0 because a single frame with a passer-by
# should not reclassify a solo take, and not lower because at 0.75 a clip can
# spend a quarter of its length as a duet and still be called solo.
SOLO_SHARE = 0.9


def measure(stem: str) -> Optional[Dict[str, object]]:
    import numpy as np

    directory = INGEST / stem
    try:
        bundle = np.load(directory / "detections.npz")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    except Exception:                                         # noqa: BLE001
        return None
    counts = bundle["person_count"]
    boxes = bundle["dancer_box"]
    width, height = meta.get("video_w"), meta.get("video_h")
    if not len(counts) or not width or not height:
        return None

    present = np.isfinite(boxes).all(axis=1)
    row: Dict[str, object] = {
        "clip": stem,
        "frames": int(len(counts)),
        "people_median": float(np.median(counts)),
        "people_max": int(counts.max()),
        "share_at_most_one": float((counts <= 1).mean()),
        "share_two_or_more": float((counts >= 2).mean()),
        "share_three_or_more": float((counts >= 3).mean()),
        "dancer_present_share": float(present.mean()),
        "rival_count": int(bundle["rival_count"]),
        "rival_ratio": float(bundle["rival_ratio"]),
    }
    if present.sum() >= 2:
        kept = boxes[present]
        w = kept[:, 2] - kept[:, 0]
        h = kept[:, 3] - kept[:, 1]
        area = (w * h) / float(width * height)
        row["subject_area_median"] = float(np.median(area))
        # How far the crop travels per frame, in units of the subject's own
        # size, so a close-up and a wide shot of the same movement read alike.
        centres = np.stack([(kept[:, 0] + kept[:, 2]) / 2,
                            (kept[:, 1] + kept[:, 3]) / 2], axis=1)
        scale = np.maximum(np.sqrt(np.maximum(w * h, 1.0)), 1.0)
        step = np.linalg.norm(np.diff(centres, axis=0), axis=1) / scale[:-1]
        row["crop_travel_median"] = float(np.median(step))
        row["crop_travel_p95"] = float(np.percentile(step, 95))
        # A subject whose box touches the frame edge is partly outside it.
        edge = ((kept[:, 0] <= 1) | (kept[:, 1] <= 1) |
                (kept[:, 2] >= width - 1) | (kept[:, 3] >= height - 1))
        row["share_box_touches_edge"] = float(edge.mean())
    row["solo"] = row["share_at_most_one"] >= SOLO_SHARE
    return row


def quantiles(values, points=(0.1, 0.25, 0.5, 0.75, 0.9)):
    import numpy as np

    clean = [v for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return {"n": len(clean),
            **{"p{:g}".format(p * 100): round(float(np.percentile(clean, p * 100)), 4)
               for p in points}}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--released-only", action="store_true",
                        help="restrict to the clips the released bundle carries, "
                             "which is what training reads")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    stems = sorted(d.name for d in INGEST.glob("*__clip*") if d.is_dir())
    released = set()
    if BUNDLE_SOURCES.is_file():
        released = {json.loads(line)["legacy_source_name"]
                    for line in BUNDLE_SOURCES.open(encoding="utf-8") if line.strip()}
    if args.released_only:
        stems = [s for s in stems if s in released]
    if args.limit:
        stems = stems[: args.limit]
    accounts = json.loads(ACCOUNTS.read_text(encoding="utf-8")) if ACCOUNTS.is_file() else {}
    print("measuring {} clips".format(len(stems)), flush=True)

    rows = []
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(measure, stems), start=1):
            if row is not None:
                row["account"] = accounts.get(row["clip"].split("__clip")[0])
                row["released"] = row["clip"] in released
                rows.append(row)
            if index % 2000 == 0 or index == len(stems):
                print("[{}/{}]".format(index, len(stems)), flush=True)
    unreadable = len(stems) - len(rows)

    def summarise(group, name):
        if not group:
            return {"population": name, "clips": 0}
        solo = [r for r in group if r["solo"]]
        return {
            "population": name,
            "clips": len(group),
            "solo_clips": len(solo),
            "solo_share": round(len(solo) / len(group), 4),
            "majority_two_or_more": round(
                sum(1 for r in group if r["share_two_or_more"] >= 0.5) / len(group), 4),
            "majority_three_or_more": round(
                sum(1 for r in group if r["share_three_or_more"] >= 0.5) / len(group), 4),
            "people_median_histogram": dict(sorted(collections.Counter(
                int(r["people_median"]) for r in group).items())),
            "subject_area": quantiles([r.get("subject_area_median") for r in group]),
            "subject_area_solo": quantiles([r.get("subject_area_median") for r in solo]),
            "crop_travel_median": quantiles([r.get("crop_travel_median") for r in group]),
            "box_touches_edge": quantiles([r.get("share_box_touches_edge") for r in group]),
            "rival_count_zero_share": round(
                sum(1 for r in group if r["rival_count"] == 0) / len(group), 4),
        }

    result = {
        "generated_by": "tools/audit_clip_population.py",
        "solo_definition": "person_count <= 1 in at least {:.0%} of frames".format(SOLO_SHARE),
        "clips_measured": len(rows),
        "clips_unreadable": unreadable,
        "all": summarise(rows, "every clip in the ingest tree"),
        "released": summarise([r for r in rows if r["released"]],
                              "the released bundle, which training reads"),
        "by_account": {},
        "rows": rows,
    }
    by_account = collections.defaultdict(list)
    for row in rows:
        if row["released"]:
            by_account[row["account"]].append(row)
    for account, group in by_account.items():
        result["by_account"][account] = {
            "clips": len(group),
            "solo_share": round(sum(1 for r in group if r["solo"]) / len(group), 4),
            "subject_area_median": quantiles(
                [r.get("subject_area_median") for r in group], points=(0.5,)),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    for key in ("all", "released"):
        block = result[key]
        if not block["clips"]:
            continue
        print("\n--- {} (n={}) ---".format(block["population"], block["clips"]))
        print("  solo ({})              : {:.1%}".format(
            result["solo_definition"], block["solo_share"]))
        print("  two or more, most frames        : {:.1%}".format(block["majority_two_or_more"]))
        print("  three or more, most frames      : {:.1%}".format(block["majority_three_or_more"]))
        area = block["subject_area"]
        if area:
            print("  subject box / frame area        : median {:.3f}  p10 {:.3f}  p90 {:.3f}".format(
                area["p50"], area["p10"], area["p90"]))
        area = block["subject_area_solo"]
        if area:
            print("    on solo clips only            : median {:.3f}".format(area["p50"]))
        travel = block["crop_travel_median"]
        if travel:
            print("  crop travel per frame / box     : median {:.4f}  p90 {:.4f}".format(
                travel["p50"], travel["p90"]))
        edge = block["box_touches_edge"]
        if edge:
            print("  share of frames box touches edge: median {:.3f}  p90 {:.3f}".format(
                edge["p50"], edge["p90"]))
    if unreadable:
        print("\n{} clip(s) could not be measured".format(unreadable))
    print("\nwrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
