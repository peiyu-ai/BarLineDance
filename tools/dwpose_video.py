#!/usr/bin/env python3
"""2D keypoints over a video: one dancer, tracked, with absence left visible.

The detector underneath is DWPose's, byte-identical to the one that produced the
current corpus (see ``third_party/DWPose/README.md``).  Two things about the
wrapper are this repo's, and both were forced by measurement rather than taste.

**Who the dancer is, is a track question, not a per-frame one.**  Lodge's wrapper
runs the pose model on every detected box and keeps whichever scores highest,
independently each frame.  Measured on this corpus that is 7.07 boxes per frame
(median 5) -- these uploads are full of bystanders -- so the choice is being made
~7 times a frame with no memory, and the winner is not even usually the biggest
person: the highest-scoring box is the largest one only 35.7% of the time, and
is outside the top three by area 24% of the time.  A confident pose on a small
clear bystander beats a partly-cropped foreground dancer.  So this links
detections into tracks and picks one track for the whole clip, on persistence,
size and motion.  The reference for "did it pick the dancer" is not an opinion:
GVHMR chose a person when it produced the 3D, and its choice is on disk in
``preprocess/bbx.pt`` for every clip in the corpus -- ``--validate-against``
scores this selection against it.

That also buys back the cost. Detection is 9.8 ms/frame and the pose model is
5.0 ms per *box*: running it on all ~10 boxes costs 53 ms/frame (18.9 fps), and
running it on the one tracked dancer costs 5 ms (67.6 fps end to end). Over the
corpus that is the difference between ~230 and ~34 GPU-hours.

**A frame with no dancer is written as a frame with no dancer.**  Lodge holds the
previous pose whenever fewer than three joints clear threshold.  Here that fill
is not neutral: ``preprocess_wild_3d`` gates on ``visible_joint_fraction`` and
``frozen_joint_pair_fraction``, so a filled frame passes the check it should fail
and trips a different one -- and boundary detection reads absence as the signal
for where a clip should end.  Absent frames are NaN with zero score.

Joints are OpenPose-18, matching the published corpus.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DWPOSE_ROOT = REPO_ROOT / "third_party" / "DWPose"
DEFAULT_WEIGHTS = DWPOSE_ROOT / "weights"
BODY_JOINTS = 18                 # OpenPose-18: DWPose's whole-body prefix
MIN_VISIBLE_JOINTS = 3           # below this a frame carries no usable pose
LINK_IOU = 0.3                   # box-to-track association threshold
MAX_GAP_FRAMES = 15              # a track survives half a second of occlusion


@dataclass
class VideoPose:
    keypoints: np.ndarray        # [T, 18, 2] pixels, NaN where the dancer is absent
    scores: np.ndarray           # [T, 18]
    dancer_box: np.ndarray       # [T, 4] xyxy pixels, NaN where absent
    person_count: np.ndarray     # [T] boxes the detector returned, dancer or not
    frame_indices: np.ndarray    # [T] index into the source video, for stride > 1
    width: int
    height: int
    fps: float
    track_score: float           # the winning track's score, for auditing
    # How close the runner-up came.  Much of this corpus is class and group
    # footage where several people dance the same choreography at once, and
    # there "the dancer" has no answer -- GVHMR picked one, this picks one, and
    # on half the clips checked they picked different people while both were
    # squarely on a dancer.  A clip where this is near 1.0 is not a clip where
    # selection went wrong; it is a clip where the question was ill-posed, and
    # that is a fact about the clip which downstream is entitled to know.
    rival_ratio: float
    rival_count: int             # tracks scoring at least half the winner


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0)
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-6)


def link_tracks(detections: Sequence[np.ndarray], *, iou_threshold: float = LINK_IOU,
                max_gap: int = MAX_GAP_FRAMES) -> List[dict]:
    """Greedy IoU linking of per-frame boxes into tracks.

    Greedy and IoU-only is deliberate.  A learned tracker would be another model
    to vendor, version and audit for a signal whose consumer is a threshold; and
    the failure mode that matters here -- two dancers swapping places -- is not
    one that a stronger appearance model fixes reliably either.  What the gap
    tolerance does buy is robustness to the common case: the dancer turning, or
    passing behind something, for a few frames.
    """
    tracks: List[dict] = []
    for index, boxes in enumerate(detections):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        live = [t for t in tracks if index - t["last_frame"] <= max_gap]
        taken = set()
        for track in sorted(live, key=lambda t: -len(t["frames"])):
            if not len(boxes):
                break
            scores = _iou(track["boxes"][-1], boxes)
            for candidate in np.argsort(-scores):
                if candidate in taken:
                    continue
                if scores[candidate] < iou_threshold:
                    break
                taken.add(int(candidate))
                track["frames"].append(index)
                track["boxes"].append(boxes[candidate])
                track["last_frame"] = index
                break
        for candidate in range(len(boxes)):
            if candidate not in taken:
                tracks.append({"frames": [index], "boxes": [boxes[candidate]],
                               "last_frame": index})
    for track in tracks:
        track["boxes"] = np.stack(track["boxes"])
        track["frames"] = np.asarray(track["frames"], dtype=np.int32)
    return tracks


def score_track(track: dict, *, frames: int, frame_area: float) -> float:
    """Persistence x size x motion -- what separates a dancer from a bystander.

    All three are needed and none alone works.  Persistence alone picks a
    stationary onlooker who is in every frame; size alone picks whoever walks
    closest to the camera; motion alone picks the camera's own jitter on a
    background figure.  Multiplying demands all three at once.

    The two exponents are not fitted, and each answers a failure the plain
    product has:

    * **coverage squared.**  With coverage linear, someone crossing close to the
      camera for eight frames of a hundred beats the dancer who is there the
      whole time, because their box is twenty times the area.  Being the subject
      of the video is nearly definitional here -- a track present a tenth of the
      time is far less than a tenth as likely to be who the video is of.
    * **square root of area.**  Area grows with the square of how close someone
      is to the camera, so raw area says a person twice as near is four times as
      much the dancer.  Its root is the box's linear scale, which is what "how
      big is this person on screen" actually means.
    """
    boxes = track["boxes"]
    coverage = len(track["frames"]) / max(frames, 1)
    area = float(np.median((boxes[:, 2] - boxes[:, 0]) *
                           (boxes[:, 3] - boxes[:, 1]))) / max(frame_area, 1.0)
    centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                        (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    scale = np.sqrt(max(area * frame_area, 1.0))
    # Motion normalised by the dancer's own size: a close-up dancer and a distant
    # one making the same movement should score the same.
    motion = (float(np.median(np.linalg.norm(np.diff(centres, axis=0), axis=1)))
              / scale) if len(centres) > 1 else 0.0
    # A track that never moves is a bystander, but a dancer filmed head-on moves
    # the box very little, so motion enters with a floor rather than as a factor
    # that can zero the product.
    return coverage ** 2 * np.sqrt(area) * (0.25 + motion)


class DWPoseExtractor:
    """DWPose over a video: detect every frame, track, pose the dancer only."""

    def __init__(self, weights_dir: Union[str, pathlib.Path, None] = None,
                 device: str = "cuda", score_threshold: float = 0.3):
        if str(DWPOSE_ROOT) not in sys.path:
            sys.path.insert(0, str(DWPOSE_ROOT))
        import onnxruntime as ort

        weights = pathlib.Path(weights_dir or DEFAULT_WEIGHTS)
        detector = weights / "yolox_l.onnx"
        pose = weights / "dw-ll_ucoco_384.onnx"
        for path in (detector, pose):
            if not path.exists():
                raise FileNotFoundError(
                    "{} is missing; run tools/setup_dwpose_env.sh".format(path))
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device.startswith("cuda") else ["CPUExecutionProvider"])
        if device.startswith("cuda") and "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "device={} but onnxruntime has no CUDAExecutionProvider (providers: {}); "
                "run under .venv_ortgpu -- see tools/setup_dwpose_env.sh".format(
                    device, ort.get_available_providers()))
        self.session_det = ort.InferenceSession(str(detector), providers=providers)
        self.session_pose = ort.InferenceSession(str(pose), providers=providers)
        self.score_threshold = score_threshold

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        from dwpose.onnxdet import inference_detector

        boxes = inference_detector(self.session_det, frame_bgr)
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    def pose(self, frame_bgr: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pose for one box, converted to OpenPose-18 the way DWPose does it."""
        from dwpose.onnxpose import inference_pose

        keypoints, scores = inference_pose(self.session_pose,
                                           box.reshape(1, 4), frame_bgr)
        # DWPose's whole-body -> OpenPose-18 remap: synthesise the neck from the
        # shoulders, then reorder.  Copied from ``dwpose/wholebody.py`` rather
        # than called, because that entry point insists on running detection and
        # pose together over every box, which is the cost this class avoids.
        info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        neck = np.mean(info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(info[:, 5, 2:4] > 0.3,
                                      info[:, 6, 2:4] > 0.3).astype(int)
        info = np.insert(info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        info[:, openpose_idx] = info[:, mmpose_idx]
        return (info[0, :BODY_JOINTS, :2].astype(np.float32),
                info[0, :BODY_JOINTS, 2].astype(np.float32))

    def video(self, video_path: Union[str, pathlib.Path], *, stride: int = 1,
              max_frames: Optional[int] = None,
              detections_only: bool = False) -> VideoPose:
        """Detect, track, then pose the dancer.

        ``detections_only`` skips the pose pass entirely, which is what boundary
        detection wants: it needs to know when the dancer is on screen, not what
        their elbows were doing, and detection alone runs 10x faster.
        """
        import cv2

        path = pathlib.Path(video_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError("cannot open video: {}".format(path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0

        # Frames are held so the pose pass does not decode the video twice; a
        # 64 s clip at 720p is ~2.6 GB, so long uploads are decoded in the
        # detection pass and re-read for pose instead.
        frames: List[np.ndarray] = []
        detections: List[np.ndarray] = []
        indices: List[int] = []
        keep_frames = not detections_only
        budget = 900                          # frames held in memory at most
        read_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if read_index % stride == 0:
                    detections.append(self.detect(frame))
                    indices.append(read_index)
                    if keep_frames and len(frames) < budget:
                        frames.append(frame)
                    elif keep_frames:
                        frames = []           # too long to hold; re-read below
                        keep_frames = False
                    if max_frames is not None and len(detections) >= max_frames:
                        break
                read_index += 1
        finally:
            capture.release()

        count = len(detections)
        tracks = link_tracks(detections)
        frame_area = float(width * height)
        ranked = sorted(tracks, key=lambda t: -score_track(t, frames=count, frame_area=frame_area))
        best = ranked[0] if ranked else None

        box_track = np.full((count, 4), np.nan, dtype=np.float32)
        track_score = rival_ratio = 0.0
        rival_count = 0
        if best is not None:
            box_track[best["frames"]] = best["boxes"]
            track_score = score_track(best, frames=count, frame_area=frame_area)
            scores_all = [score_track(t, frames=count, frame_area=frame_area) for t in ranked]
            rival_ratio = (scores_all[1] / track_score) if len(scores_all) > 1 and track_score else 0.0
            rival_count = sum(1 for s in scores_all[1:] if track_score and s >= 0.5 * track_score)

        keypoints = np.full((count, BODY_JOINTS, 2), np.nan, dtype=np.float32)
        scores = np.zeros((count, BODY_JOINTS), dtype=np.float32)
        if not detections_only and best is not None:
            source = (_held_frames(frames) if frames
                      else self._reread(path, indices))
            for position, frame in source:
                box = box_track[position]
                if not np.isfinite(box).all():
                    continue
                body, body_scores = self.pose(frame, box)
                visible = body_scores > self.score_threshold
                body[~visible] = np.nan
                if visible.sum() < MIN_VISIBLE_JOINTS:
                    body[:] = np.nan
                keypoints[position] = body
                scores[position] = body_scores

        return VideoPose(
            keypoints=keypoints, scores=scores, dancer_box=box_track,
            person_count=np.asarray([len(d) for d in detections], dtype=np.int32),
            frame_indices=np.asarray(indices, dtype=np.int32),
            width=width, height=height, fps=fps, track_score=round(track_score, 6),
            rival_ratio=round(float(rival_ratio), 4), rival_count=int(rival_count))

    @staticmethod
    def _reread(path: pathlib.Path, indices: Sequence[int]):
        import cv2

        wanted = {index: position for position, index in enumerate(indices)}
        capture = cv2.VideoCapture(str(path))
        read_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    return
                if read_index in wanted:
                    yield wanted[read_index], frame
                read_index += 1
        finally:
            capture.release()


def _held_frames(frames: Sequence[np.ndarray]):
    return list(enumerate(frames))


def save(pose: VideoPose, out_dir: pathlib.Path, *, normalize: bool = True) -> None:
    """Write what the corpus reads, plus the detection record behind it.

    ``keypoints.npy`` is normalised by frame size, which is what
    ``preprocess_wild_3d`` expects.  Unlike the published corpus it keeps NaN
    for absent joints instead of ``nan_to_num(..., nan=0.5)``; that substitution
    put missing joints at the centre of the frame, which made
    ``finite_joint_fraction`` identically 1.0 and unable to report anything.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    keypoints = pose.keypoints.astype(np.float32).copy()
    if normalize and len(keypoints):
        keypoints[..., 0] /= max(float(pose.width), 1.0)
        keypoints[..., 1] /= max(float(pose.height), 1.0)
    np.save(out_dir / "keypoints.npy", keypoints)
    np.save(out_dir / "scores.npy", pose.scores.astype(np.float32))
    np.savez(out_dir / "detections.npz", person_count=pose.person_count,
             dancer_box=pose.dancer_box, frame_indices=pose.frame_indices,
             track_score=np.float32(pose.track_score),
             rival_ratio=np.float32(pose.rival_ratio),
             rival_count=np.int32(pose.rival_count))


def save_gvhmr_bbx(pose: VideoPose, bbx_path: pathlib.Path,
                   base_enlarge: float = 1.2) -> None:
    """Hand the chosen dancer to GVHMR, so 2D and 3D describe the same person.

    GVHMR's ``demo.py`` runs its own YOLOv8 tracker and calls ``get_one_track``
    -- and on group footage its answer and ours are both defensible and often
    different, which would leave the corpus with a 2D record of one dancer and a
    3D record of another under one id.  It reads ``preprocess/bbx.pt`` when the
    file is already there, so writing it first makes the choice once, here,
    where it is recorded next to ``rival_ratio``.

    Gaps are filled by holding the last box.  That is a fill, and elsewhere in
    this file fills are refused -- but this one is a *crop window*, not a
    measurement: it decides which pixels the 3D model looks at, and GVHMR needs
    one for every frame.  The absence itself stays recorded, unfilled, in
    ``detections.npz``.
    """
    boxes = pose.dancer_box.copy()
    if not len(boxes):
        raise ValueError("no frames to write")
    missing = ~np.isfinite(boxes).all(axis=1)
    if missing.all():
        raise ValueError("no dancer track was found in this clip")
    index = np.where(~missing, np.arange(len(boxes)), 0)
    np.maximum.accumulate(index, out=index)
    boxes = boxes[index]
    # Frames before the first detection have nothing to hold; take the first.
    first = int(np.argmax(~missing))
    boxes[:first] = boxes[first]

    # bbx_xys is GVHMR's crop parameterisation, and its aspect-ratio and enlarge
    # conventions are theirs.  Call their function rather than reproduce it:
    # this repo has already shipped one descriptor that reimplemented somebody
    # else's convention and read an axis backwards.
    # hmr4d pulls in pytorch3d on import; this repo ships a shim for it, and the
    # ingestion process has no reason to have PYTHONPATH set the way the
    # extraction shard script does.
    for extra in (REPO_ROOT / "third_party" / "pytorch3d_compat",
                  REPO_ROOT / "third_party" / "GVHMR"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    import torch
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy

    bbx_xyxy = torch.from_numpy(boxes.astype(np.float32))
    bbx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bbx_xyxy": bbx_xyxy,
                "bbx_xys": get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=base_enlarge).float()},
               bbx_path)
