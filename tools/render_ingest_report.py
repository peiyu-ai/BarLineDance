#!/usr/bin/env python3
"""Show what the ingestion decided, so the cut threshold can be judged by eye.

The shot-cut floor and ratio are free parameters that decide every clip boundary
in the corpus, and the numbers behind them -- a gap between 0.194 and 0.237 on
one axis, 4.7 and 11.4 on the other -- say where the tail starts but not whether
the frames either side of a firing are actually different shots.  Placing them
by quantile is not even possible: cuts are a ~1e-4 event, so the entire
calibration table sits inside the dancer-motion hump.  This renders both halves
of the check:

* **cut sheet** -- the frame before and the frame after every firing, side by
  side.  A correct threshold gives pairs that plainly differ; too low a
  threshold shows pairs that are the same shot with the dancer moved, and that
  is visible immediately.
* **timeline** -- dancer presence, cut firings and the resulting spans on one
  axis, so a span that ends for the wrong reason is locatable.

This is the same discipline the 3D audit used: a number that decides a corpus
gets a picture that can overturn it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

PANEL_WIDTH = 220


def _label(image, text, colour=(255, 255, 255)):
    import cv2

    cv2.putText(image, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(image, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return image


def cut_sheet(upload: pathlib.Path, cut_frames: Sequence[int], out_path: pathlib.Path,
              *, max_cuts: int = 8) -> int:
    import cv2

    capture = cv2.VideoCapture(str(upload))
    rows: List[np.ndarray] = []
    for frame_index in list(cut_frames)[:max_cuts]:
        pair = []
        for offset, tag in ((-1, "before"), (0, "after")):
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index + offset, 0))
            ok, frame = capture.read()
            if not ok:
                break
            scale = PANEL_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (PANEL_WIDTH, int(frame.shape[0] * scale)))
            pair.append(_label(frame, "{} f{}".format(tag, frame_index + offset)))
        if len(pair) == 2:
            height = max(p.shape[0] for p in pair)
            pair = [cv2.copyMakeBorder(p, 0, height - p.shape[0], 0, 4,
                                       cv2.BORDER_CONSTANT, value=(30, 30, 30))
                    for p in pair]
            rows.append(np.hstack(pair))
    capture.release()
    if not rows:
        return 0
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 6, 0, width - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(30, 30, 30)) for r in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
    return len(rows)


def timeline(present: np.ndarray, cuts: np.ndarray, spans: Sequence[Sequence[int]],
             out_path: pathlib.Path, *, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = len(present)
    figure, axis = plt.subplots(figsize=(12, 2.2))
    axis.fill_between(np.arange(frames), 0, present.astype(float), step="mid",
                      color="#4c8bf5", alpha=0.5, linewidth=0, label="dancer present")
    for index, frame_index in enumerate(np.flatnonzero(cuts)):
        axis.axvline(frame_index, color="#d9534f", linewidth=1.0,
                     label="shot cut" if index == 0 else None)
    for index, (begin, end, _why) in enumerate(spans):
        axis.plot([begin, end], [1.15, 1.15], linewidth=6, solid_capstyle="butt",
                  color="#2e8b57", label="kept span" if index == 0 else None)
    axis.set_xlim(0, max(frames, 1))
    axis.set_ylim(-0.05, 1.35)
    axis.set_yticks([])
    axis.set_xlabel("frame")
    axis.set_title(title, fontsize=9)
    axis.legend(loc="lower right", fontsize=7, ncol=3, framealpha=0.9)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=120)
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--uploads", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--shot-floor", type=float, default=None)
    parser.add_argument("--shot-ratio", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scan-cache", type=pathlib.Path, default=None,
                        help="reuse the ingest's detections; with it this "
                             "runs on CPU and costs nothing")
    args = parser.parse_args(argv)

    from tools.dwpose_video import DWPoseExtractor, link_tracks, score_track
    from tools.ingest_wild_uploads import (FPS, MAX_ABSENCE_FRAMES, MIN_FRAMES,
                                           SHOT_CUT_FLOOR, SHOT_CUT_RATIO,
                                           find_cuts, find_spans, scan_upload)

    floor = args.shot_floor if args.shot_floor is not None else SHOT_CUT_FLOOR
    ratio = args.shot_ratio if args.shot_ratio is not None else SHOT_CUT_RATIO
    # With a populated cache the detector is never called, so the check that
    # judges the boundaries costs nothing and can be re-run freely.
    extractor = None
    summary = []
    for upload in sorted(args.videos_dir.glob("*.mp4"))[:args.uploads]:
        if extractor is None and (args.scan_cache is None or not
                                  (args.scan_cache / (upload.stem + ".npz")).exists()):
            extractor = DWPoseExtractor(device=args.device)
        detections, cut_scores = scan_upload(extractor, upload, args.scan_cache)
        frames = len(detections)
        tracks = link_tracks(detections)
        ranked = sorted(tracks, key=lambda t: -score_track(t, frames=frames, frame_area=1.0))
        boxes = np.full((frames, 4), np.nan, dtype=np.float32)
        if ranked:
            boxes[ranked[0]["frames"]] = ranked[0]["boxes"]
        present = np.isfinite(boxes).all(axis=1)
        cuts = find_cuts(cut_scores, floor=floor, ratio=ratio)
        spans = find_spans(present, cuts, min_frames=MIN_FRAMES,
                           max_frames=int(round(args.max_seconds * FPS)),
                           max_absence=MAX_ABSENCE_FRAMES)
        drawn = cut_sheet(upload, np.flatnonzero(cuts),
                          args.out_dir / "{}_cuts.jpg".format(upload.stem))
        timeline(present, cuts, spans, args.out_dir / "{}_timeline.png".format(upload.stem),
                 title="{}  {} frames  {} cuts  {} spans  (floor {}, ratio {})".format(
                     upload.stem, frames, int(cuts.sum()), len(spans), floor, ratio))
        summary.append({"upload": upload.stem, "frames": frames,
                        "cuts": int(cuts.sum()), "cut_pairs_drawn": drawn,
                        "spans": [[int(a), int(b), why] for a, b, why in spans],
                        "present_fraction": round(float(present.mean()), 4)})
        print(json.dumps(summary[-1]), flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1),
                                               encoding="utf-8")
    print("\nwrote {}".format(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
