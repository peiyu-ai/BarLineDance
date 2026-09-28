#!/usr/bin/env python3
"""Measure how far a clip's camera actually rotates, to decide static-cam honestly.

Both visual-odometry backends fail on the same ~300 wild clips, and
instrumenting DPVO showed why: every patch inverse depth collapses to zero, so
the solver is pushing all points to infinity.  That is what bundle adjustment
does when there is no translational baseline, and it is also why SimpleVO's
essential-matrix solve returns None -- pure rotation is degenerate for both.

For such footage GVHMR's ``--static_cam`` asserts ``R_w2c = I`` for every
frame.  Whether that is a correct model or a fabrication is an empirical
question with a measurable answer: the error it introduces is exactly the
camera's true rotation away from frame 0.  So measure that, and let the number
decide.

Rotation is recoverable without any baseline -- that is the one thing pure
rotation makes *easy*.  Between two frames the background maps by a homography
``H = K R K^-1``, so ``R = K^-1 H K`` orthonormalized.  Two details keep it
honest:

* the dancer must not vote.  A person filling the frame produces a strong,
  consistent, and completely wrong "camera" motion.  GVHMR has already tracked
  and cached a per-frame person box, so features inside it are dropped; the
  homography is fit on background only.
* a homography needs enough background to fit.  Pairs with too few inliers are
  reported as unmeasured rather than silently contributing a bad rotation, and
  a clip with too few measured pairs is not declared static on faith.

Reports the maximum rotation from the first frame, which is the bound on the
error ``--static_cam`` would introduce.  Exit status is 0 when the clip is
within threshold, 2 when it is not, so a shell loop can branch on it.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

# GVHMR's own default intrinsics guess (hmr4d/utils/geo/hmr_cam.py): focal is
# the image diagonal.  Reproduced rather than imported so this tool runs from
# the repo root without the GVHMR checkout on sys.path.
def estimate_focal_length(width: int, height: int) -> float:
    return float((width ** 2 + height ** 2) ** 0.5)


DEFAULT_THRESHOLD_DEG = 3.0
MIN_INLIERS = 25
MIN_MEASURED_FRACTION = 0.5
BOX_MARGIN = 0.10


class MeasurementError(RuntimeError):
    pass


def _person_boxes(path: Optional[pathlib.Path], frames: int) -> Optional[np.ndarray]:
    if path is None or not path.is_file():
        return None
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    boxes = np.asarray(payload["bbx_xyxy"], dtype=np.float64)
    if len(boxes) < frames:
        # Tracking covers the video; a shorter array means a different video.
        raise MeasurementError(
            "person boxes cover {} frames, video has {}".format(len(boxes), frames))
    return boxes


def _mask_for(shape, box, margin=BOX_MARGIN):
    """255 where features may be used; 0 over the person."""
    height, width = shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    if box is None:
        return mask
    x0, y0, x1, y1 = box
    pad_x = (x1 - x0) * margin
    pad_y = (y1 - y0) * margin
    x0 = int(max(0, math.floor(x0 - pad_x)))
    y0 = int(max(0, math.floor(y0 - pad_y)))
    x1 = int(min(width, math.ceil(x1 + pad_x)))
    y1 = int(min(height, math.ceil(y1 + pad_y)))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 0
    return mask


def _rotation_from_homography(homography, intrinsics) -> Optional[np.ndarray]:
    """R = K^-1 H K, projected back onto SO(3); None if it is not a rotation."""
    candidate = np.linalg.inv(intrinsics) @ homography @ intrinsics
    if not np.isfinite(candidate).all():
        return None
    u, singular, vt = np.linalg.svd(candidate)
    if singular[-1] <= 1e-9 or singular[0] / singular[-1] > 5.0:
        # Far from a similarity: the pair is not explained by pure rotation.
        return None
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _geodesic_degrees(rotation) -> float:
    cosine = (np.trace(rotation) - 1.0) / 2.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def measure(
    video_path: pathlib.Path,
    *,
    person_boxes: Optional[pathlib.Path] = None,
    stride: int = 5,
    max_frames: Optional[int] = None,
    threshold_deg: float = DEFAULT_THRESHOLD_DEG,
) -> Dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise MeasurementError("cannot open {}".format(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes = _person_boxes(person_boxes, total)

    focal = estimate_focal_length(width, height)
    intrinsics = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )

    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher()

    cumulative = np.eye(3)
    previous = None
    angles: List[float] = []
    pair_angles: List[float] = []
    inlier_counts: List[int] = []
    attempted = measured = 0
    index = -1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if max_frames is not None and index >= max_frames:
            break
        if index % stride:
            continue
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = _mask_for(grey.shape, boxes[index] if boxes is not None else None)
        keypoints, descriptors = detector.detectAndCompute(grey, mask)
        current = (keypoints, descriptors)
        if previous is not None:
            attempted += 1
            rotation, inliers = _pair_rotation(previous, current, matcher, intrinsics)
            if rotation is not None:
                measured += 1
                inlier_counts.append(inliers)
                pair_angles.append(_geodesic_degrees(rotation))
                cumulative = rotation @ cumulative
                angles.append(_geodesic_degrees(cumulative))
        previous = current
    capture.release()

    measured_fraction = measured / attempted if attempted else 0.0
    enough = attempted >= 4 and measured_fraction >= MIN_MEASURED_FRACTION
    max_rotation = max(angles) if angles else float("nan")
    static = bool(enough and angles and max_rotation <= threshold_deg)

    return {
        "video": str(video_path),
        "frames": total,
        "resolution": [width, height],
        "stride": stride,
        "person_boxes_used": boxes is not None,
        "pairs_attempted": attempted,
        "pairs_measured": measured,
        "measured_fraction": round(measured_fraction, 4),
        "median_inliers": int(np.median(inlier_counts)) if inlier_counts else 0,
        "max_rotation_from_first_frame_deg": (
            round(max_rotation, 4) if angles else None
        ),
        "median_pair_rotation_deg": (
            round(float(np.median(pair_angles)), 4) if pair_angles else None
        ),
        "threshold_deg": threshold_deg,
        "static_camera": static,
        "verdict_reason": (
            "insufficient background to measure" if not enough
            else "within threshold" if static
            else "camera rotates beyond threshold"
        ),
        "method": (
            "background-only SIFT homography per sampled pair, R = K^-1 H K "
            "orthonormalized; person box dilated {:.0%} and excluded".format(BOX_MARGIN)
        ),
    }


def _pair_rotation(previous, current, matcher, intrinsics):
    (keypoints0, descriptors0) = previous
    (keypoints1, descriptors1) = current
    if descriptors0 is None or descriptors1 is None:
        return None, 0
    if len(descriptors0) < 2 or len(descriptors1) < 2:
        return None, 0
    matches = matcher.knnMatch(descriptors0, descriptors1, k=2)
    good = [m for m, n in (pair for pair in matches if len(pair) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    source = np.float32([keypoints0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target = np.float32([keypoints1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if homography is None or inlier_mask is None:
        return None, 0
    inliers = int(inlier_mask.sum())
    if inliers < MIN_INLIERS:
        return None, inliers
    rotation = _rotation_from_homography(homography, intrinsics)
    if rotation is None:
        return None, inliers
    return rotation, inliers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=pathlib.Path, required=True)
    parser.add_argument(
        "--person-boxes", type=pathlib.Path, default=None,
        help="GVHMR preprocess/bbx.pt; features inside the box are excluded")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--threshold-deg", type=float, default=DEFAULT_THRESHOLD_DEG)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = measure(
            args.video,
            person_boxes=args.person_boxes,
            stride=args.stride,
            max_frames=args.max_frames,
            threshold_deg=args.threshold_deg,
        )
    except MeasurementError as error:
        raise SystemExit("error: {}".format(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["static_camera"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
