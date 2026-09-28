#!/usr/bin/env python3
"""Audit the 3D pose estimates the wild corpus was built from.

Everything downstream -- segmentation, the TMR vocabulary, the release -- reads
these arrays as if they were motion capture.  They are not: they are monocular
HMR output from TikTok video, and the failure modes are specific and physical.
This measures the ones that can be checked without ground truth, per sequence,
so a defect can be named and located rather than suspected.

What is checked, and why each one is the honest form of the question:

* **Floor height.**  The ground is *not* assumed to be ``z = 0``.  It is
  estimated per clip from the low percentile of toe height, because the
  quantity that matters is whether a clip is internally consistent, and
  measuring penetration against a plane the data never used would report a
  defect that is really a gauge offset.  The *spread* of the estimate across
  clips is reported separately: that is the gauge question, and it is a
  different question.
* **Body height.**  Head minus floor.  Forward kinematics here uses fixed SMPL
  offsets, so every dancer has the same skeleton and this number should be
  near-constant.  A clip that disagrees is one where the root translation, not
  the pose, has gone wrong.
* **Foot penetration.**  Frames whose lowest toe is below the estimated floor.
  Because that floor is itself the 2nd percentile of toe height, this measure
  has a **breakdown point of 2%**: it catches a foot punching through for a few
  frames, and by construction *cannot* catch a body that sits below the ground
  for most of the clip -- the estimate simply follows it down.  That case is
  not unmeasured, it is measured by different columns (``floor_z`` across
  clips, and ``lowest_toe_spread`` within one), and a test pins the blind spot
  so nobody later reads this number as covering it.
* **Foot skate.**  Horizontal speed of the *planted* foot only -- the one near
  the floor.  A swinging foot is supposed to move; charging it as skate would
  make fast dances look broken.
* **Jitter.**  Joint acceleration, reported both absolutely and divided by the
  clip's own median joint speed.  The ratio is the one that means something:
  a fast dance has large accelerations legitimately, and an absolute threshold
  would flag exactly the clips with the most content.
* **Root speed.**  Tracking failures show up as the pelvis teleporting.
* **Frozen frames.**  A stuck tracker repeats a pose; those frames are real
  rows in the training set that teach the model to hold still.

``--reference`` points at a mocap bundle (AIST++) so every number lands beside
one from data that is right by construction.  Without it the numbers are
readable but not judgeable, and this tool says so in the report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

TOES = (10, 11)
ANKLES = (7, 8)
HEAD = 15
PELVIS = 0
FPS = 30.0

# Floor is read at this percentile of per-frame minimum toe height: low enough
# to sit on the planted frames, high enough that a single penetrating frame
# cannot define the plane.
FLOOR_PERCENTILE = 2.0
PLANTED_MARGIN = 0.05      # m above the floor a toe may be and still count planted
PENETRATION_MARGIN = 0.03  # m below the floor before a frame counts as penetrating
FROZEN_MOTION = 1e-3       # m of largest joint displacement below which a frame is frozen


def sequence_metrics(joints: np.ndarray) -> Dict[str, float]:
    """Per-clip physical measures from [T, 24, 3] z-up joint positions."""
    frames = len(joints)
    toe_height = joints[:, TOES, 2].min(axis=1)
    floor = float(np.percentile(toe_height, FLOOR_PERCENTILE))

    velocity = np.diff(joints, axis=0) * FPS                     # m/s per joint
    speed = np.linalg.norm(velocity, axis=2)                     # [T-1, 24]
    acceleration = np.linalg.norm(np.diff(velocity, axis=0), axis=2) * FPS   # m/s^2
    median_speed = float(np.median(speed))

    planted = joints[:-1, TOES, 2] <= floor + PLANTED_MARGIN     # [T-1, 2]
    horizontal = np.linalg.norm(velocity[:, TOES, :2], axis=2)   # [T-1, 2]
    skate = horizontal[planted] if planted.any() else np.zeros(1)

    displacement = np.linalg.norm(np.diff(joints, axis=0), axis=2).max(axis=1)
    root_speed = np.linalg.norm(np.diff(joints[:, PELVIS], axis=0), axis=1) * FPS

    return {
        "frames": frames,
        "floor_z": round(floor, 4),
        "body_height": round(float(np.median(joints[:, HEAD, 2]) - floor), 4),
        "penetration_fraction": round(
            float((toe_height < floor - PENETRATION_MARGIN).mean()), 5),
        # p99 describes the bulk; on a clip where a foot punches through for
        # three frames out of six hundred the p99 is exactly zero, so the worst
        # case is reported next to it rather than hidden behind a quantile.
        "penetration_depth_p99": round(
            float(np.percentile(np.maximum(floor - toe_height, 0.0), 99)), 4),
        "penetration_depth_max": round(float(np.maximum(floor - toe_height, 0.0).max()), 4),
        # How far the lowest toe wanders above the floor over the clip.  A
        # dancer's supporting foot is on the ground nearly always, so on mocap
        # this is small and spikes only for jumps; on drifting monocular root
        # translation the whole body rides up and down and this opens up.
        "lowest_toe_spread": round(
            float(np.percentile(toe_height, 95) - np.percentile(toe_height, 5)), 4),
        "skate_median": round(float(np.median(skate)), 4),
        "skate_p95": round(float(np.percentile(skate, 95)), 4),
        "planted_fraction": round(float(planted.mean()), 4),
        "jitter_median": round(float(np.median(acceleration)), 3),
        "jitter_p95": round(float(np.percentile(acceleration, 95)), 3),
        # Acceleration in units of the clip's own speed: dimensionally 1/s, and
        # the only form comparable between a slow pose and a fast footwork clip.
        "jitter_ratio": round(float(np.median(acceleration) / max(median_speed, 1e-6)), 3),
        "median_joint_speed": round(median_speed, 4),
        "root_speed_p99": round(float(np.percentile(root_speed, 99)), 4),
        "root_speed_max": round(float(root_speed.max()), 4),
        "frozen_fraction": round(float((displacement < FROZEN_MOTION).mean()), 5),
    }


def _one(payload):
    bundle, row = payload
    from tools.convert_motion_to_guofeats import motion_151_to_joints

    try:
        motion = np.load(bundle / row["motion_path"])
        joints = motion_151_to_joints(motion)
    except Exception as error:                       # noqa: BLE001 - recorded, not raised
        return {"recording_id": row["recording_id"], "error": repr(error)}
    metrics = sequence_metrics(joints)
    metrics["recording_id"] = row["recording_id"]
    metrics["split"] = row.get("split")
    return metrics


def measure(bundle: pathlib.Path, *, limit: Optional[int], workers: int,
            manifest: str = "sequences.jsonl") -> List[Dict]:
    rows = [json.loads(line) for line in (bundle / manifest).open(encoding="utf-8")]
    if limit is not None:
        rows = rows[:limit]
    out: List[Dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(
                pool.map(_one, [(bundle, row) for row in rows], chunksize=8), start=1):
            out.append(result)
            if index % 200 == 0 or index == len(rows):
                print("[{}/{}] measured".format(index, len(rows)), flush=True)
    return out


def summarise(records: Sequence[Dict], name: str) -> Dict[str, object]:
    good = [r for r in records if "error" not in r]
    if not good:
        return {"name": name, "sequences": 0, "failed": len(records)}
    keys = ("floor_z", "body_height", "penetration_fraction", "penetration_depth_p99",
            "penetration_depth_max", "lowest_toe_spread", "skate_median", "skate_p95",
            "planted_fraction",
            "jitter_median",
            "jitter_ratio", "median_joint_speed", "root_speed_p99", "root_speed_max",
            "frozen_fraction")
    stats = {}
    for key in keys:
        values = np.asarray([r[key] for r in good], dtype=np.float64)
        stats[key] = {
            "median": round(float(np.median(values)), 4),
            "p05": round(float(np.percentile(values, 5)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "max": round(float(values.max()), 4),
        }
    floors = np.asarray([r["floor_z"] for r in good])
    heights = np.asarray([r["body_height"] for r in good])
    return {
        "name": name,
        "sequences": len(good),
        "failed": len(records) - len(good),
        "frames": int(sum(r["frames"] for r in good)),
        "per_metric": stats,
        # The gauge question, stated separately from the per-clip measures: a
        # corpus whose floor moves clip to clip is not one world, and the root
        # translation that goes into training carries that offset.
        "gauge": {
            "floor_z_spread_p95_minus_p05": round(
                float(np.percentile(floors, 95) - np.percentile(floors, 5)), 4),
            "floor_z_std": round(float(floors.std()), 4),
            "body_height_std": round(float(heights.std()), 4),
            "body_height_spread_p95_minus_p05": round(
                float(np.percentile(heights, 95) - np.percentile(heights, 5)), 4),
        },
    }


def build(*, bundle: pathlib.Path, output: pathlib.Path, limit: Optional[int],
          workers: int, reference: Optional[pathlib.Path],
          reference_limit: Optional[int]) -> Dict[str, object]:
    records = measure(bundle, limit=limit, workers=workers)
    report: Dict[str, object] = {
        "bundle": str(bundle.resolve()),
        "wild": summarise(records, "wild"),
        "records": sorted(records, key=lambda r: r.get("recording_id", "")),
    }
    if reference is not None:
        print("measuring reference {}".format(reference), flush=True)
        reference_records = measure(reference, limit=reference_limit, workers=workers)
        report["reference"] = summarise(reference_records, str(reference))
        report["reference_records"] = sorted(
            reference_records, key=lambda r: r.get("recording_id", ""))
    else:
        report["reference"] = {
            "absent": "no mocap bundle given; the numbers are readable but not "
                      "judgeable -- pass --reference to place them beside data "
                      "that is right by construction"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, default=None,
                        help="mocap bundle in the same 151-D layout, e.g. AIST++")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reference-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    report = build(bundle=args.bundle, output=args.output, limit=args.limit,
                   workers=args.workers, reference=args.reference,
                   reference_limit=args.reference_limit)
    for name in ("wild", "reference"):
        entry = report.get(name, {})
        if "per_metric" not in entry:
            continue
        print("\n== {} ({} sequences) ==".format(name, entry["sequences"]))
        for key, value in entry["per_metric"].items():
            print("  {:<26} median {:>9} p95 {:>9} max {:>9}".format(
                key, value["median"], value["p95"], value["max"]))
        print("  gauge: {}".format(json.dumps(entry["gauge"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
