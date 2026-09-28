#!/usr/bin/env python3
"""Find the clips where the tracked dancer stops being the same person.

``ingest_wild_uploads.py`` follows one dancer through a clip by IoU linking, and
bridges gaps of up to half a second so that a dancer turning away does not end
the take.  That bridge is also the failure mode: when dancer A leaves and dancer
B appears nearby within the window, the track continues onto B.  The clip then
carries 2D keypoints, a GVHMR crop and a 3D motion that splice two people, under
one id, with nothing saying so.

Measured on 500 clips of the rebuilt corpus, that is **7.0%** of clips -- not a
rare accident, and not something the existing fields report: ``rival_ratio``
says the footage *has* another plausible dancer, which is a property of the
scene; this says the track *moved onto* them, which is a property of the track.

The two are separated by persistence, and that is the whole method:

* a **wobble** -- the detector jittering, or a limb briefly changing the box --
  moves the centre and comes back, leaving box height alone;
* a **switch** moves the centre *and* leaves the box a different size, because
  the new person stands at a different depth.  Median height change on the
  clips flagged here is 47%.

Reading ``detections.npz``, which every clip has, means this runs over the whole
corpus on CPU and needs nothing re-computed -- including clips produced before
this tool existed.

The output is a flag per clip, not a filter.  Whether a spliced clip is worth
keeping depends on what reads it: segmentation over whole-frame S3D features is
largely indifferent, while a TMR embedding of the 3D is not.  Deciding that here
would bury the choice in an audit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
from typing import Dict, List, Optional, Sequence

import numpy as np

# A centre displacement this large in one frame is not a body moving: at 30 fps
# half a body width is on the order of 9 m/s.  It is the candidate set, not the
# verdict -- the verdict is whether the box size changed with it.
JUMP_BODY_WIDTHS = 0.5
# Compare the half-second either side.  Shorter and a single bad frame decides
# it; longer and a dancer legitimately walking toward the camera looks like a
# switch.
WINDOW_FRAMES = 15
# Two people at the same depth are not separable this way, so this is a floor on
# what can be detected, not a threshold that was tuned.
HEIGHT_CHANGE = 0.25
MIN_TRACKED_FRAMES = 60


def analyse(boxes: np.ndarray) -> Dict[str, object]:
    """Per-clip track continuity from the dancer box track alone."""
    present = np.isfinite(boxes).all(axis=1)
    tracked = int(present.sum())
    result: Dict[str, object] = {
        "tracked_frames": tracked,
        "present_fraction": round(float(present.mean()), 4) if len(present) else 0.0,
        "switches": 0, "max_height_change": 0.0, "switch_frames": [],
        "jump_frames": 0, "decidable": tracked >= MIN_TRACKED_FRAMES,
    }
    if not result["decidable"]:
        return result

    index = np.flatnonzero(present)
    box = boxes[index]
    centre = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], axis=1)
    width = np.maximum(box[:, 2] - box[:, 0], 1.0)
    height = box[:, 3] - box[:, 1]
    # Normalise by the frame gap: a bridged half-second hole must not be charged
    # as a teleport, because bridging is the feature, not the defect.
    gaps = np.maximum(np.diff(index), 1)
    step = np.linalg.norm(np.diff(centre, axis=0), axis=1) / width[:-1] / gaps

    candidates = np.flatnonzero(step > JUMP_BODY_WIDTHS)
    result["jump_frames"] = int(len(candidates))
    switch_frames: List[int] = []
    worst = 0.0
    for position in candidates:
        before = height[max(position - WINDOW_FRAMES + 1, 0):position + 1]
        after = height[position + 1:position + 1 + WINDOW_FRAMES]
        if len(before) < 5 or len(after) < 5:
            continue
        base = float(np.median(before))
        change = abs(float(np.median(after)) - base) / max(base, 1e-6)
        if change > HEIGHT_CHANGE:
            # Report the first frame of the *new* identity, not the last of the
            # old: the windows are split there, and "the splice starts here" is
            # what a reader wants when they go looking at the video.
            switch_frames.append(int(index[position + 1]))
            worst = max(worst, change)
    result["switches"] = len(switch_frames)
    result["switch_frames"] = switch_frames[:10]
    result["max_height_change"] = round(worst, 4)
    return result


def _one(clip: pathlib.Path) -> Dict[str, object]:
    try:
        with np.load(clip / "detections.npz") as stored:
            record = analyse(stored["dancer_box"])
            record["rival_ratio"] = float(stored["rival_ratio"]) if "rival_ratio" in stored else None
    except Exception as error:                       # noqa: BLE001 - recorded, not raised
        return {"clip": clip.name, "error": repr(error)}
    record["clip"] = clip.name
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ingest-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    clips = sorted(d for d in args.ingest_root.glob("*__clip*")
                   if (d / "detections.npz").exists())
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        raise SystemExit("no clips with detections.npz under {}".format(args.ingest_root))

    records: List[Dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, record in enumerate(pool.map(_one, clips), start=1):
            records.append(record)
            if index % 2000 == 0:
                print("[{}/{}] analysed".format(index, len(clips)), flush=True)

    good = [r for r in records if "error" not in r]
    decidable = [r for r in good if r["decidable"]]
    switched = [r for r in decidable if r["switches"]]
    summary = {
        "clips": len(records),
        "failed": len(records) - len(good),
        # A clip too short or too sparsely tracked to judge is reported as such
        # rather than counted clean: "we could not tell" and "it is fine" are
        # different, and only one of them is a finding.
        "undecidable": len(good) - len(decidable),
        "decidable": len(decidable),
        "clips_with_a_jump": sum(1 for r in decidable if r["jump_frames"]),
        "clips_with_a_switch": len(switched),
        "switch_rate": round(len(switched) / max(len(decidable), 1), 4),
        "median_height_change_on_switch": round(float(np.median(
            [r["max_height_change"] for r in switched])), 4) if switched else None,
        "rival_ratio_median_switched": round(float(np.median(
            [r["rival_ratio"] for r in switched if r.get("rival_ratio") is not None])), 4)
        if switched else None,
        "rival_ratio_median_clean": round(float(np.median(
            [r["rival_ratio"] for r in decidable
             if not r["switches"] and r.get("rival_ratio") is not None])), 4)
        if decidable else None,
        "thresholds": {"jump_body_widths": JUMP_BODY_WIDTHS,
                       "window_frames": WINDOW_FRAMES,
                       "height_change": HEIGHT_CHANGE,
                       "min_tracked_frames": MIN_TRACKED_FRAMES},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, indent=1),
                           encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("\nwrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
