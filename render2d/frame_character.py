"""Crop the character still to the SAME framing the pose video uses.

WHY.  SteadyDancer is an image-to-video model driven by a pose picture: the
first frame is the character still and every later frame follows the skeleton.
If the still frames the person at one scale and the skeleton at another, the
model has to reconcile them, and it does so by zooming the whole scene over the
opening frames -- visible in the first run as a character that starts small in
the fairground and then fills the frame.  Measured: ``townfair.png`` holds the
figure at 58.7% of its height, ``project_pose_2d``'s camera puts ours at 75.8%.

The fix is to crop the STILL rather than shrink the pose.  The pose framing was
derived, not chosen -- ``project_pose_2d`` fits the body to (1 - 2*margin) of the
frame because a smaller pose is a weaker conditioning signal, the encoder seeing
fewer pixels of every limb -- so the still is the side that should move.

The person's box comes from the workflow's OWN yolo detector, not from my eye:
the same ``models/detection/yolov10m.onnx`` ``PoseAndFaceDetection`` loads.
"""
import argparse
import pathlib
import sys

import numpy as np
from PIL import Image
import os

# Root of the sibling checkouts (ComfyUI_Wan, Lodge, this repo); set E2E_ROOT.
E2E_ROOT = os.environ.get("E2E_ROOT", "/workspace/e2e")

PREPROCESS = pathlib.Path(
    E2E_ROOT + "/ComfyUI_Wan/custom_nodes/"
    "ComfyUI-WanAnimatePreprocess")
sys.path.insert(0, str(PREPROCESS))

# The pose camera's own numbers, measured over the fixed ten clips with
# render2d/project_pose_2d.project: head top, feet bottom, horizontal centre,
# as fractions of the frame.
POSE_TOP, POSE_BOTTOM, POSE_CENTRE_U = 0.149, 0.907, 0.493


_DETECTORS = {}


def _detectors(device):
    """Load yolo+ViTPose ONCE.  Each load costs minutes, and scoring a rendered
    video means detecting on hundreds of frames."""
    if device in _DETECTORS:
        return _DETECTORS[device]
    import importlib
    import types
    if "wanpre" not in sys.modules:
        for name, path in (("wanpre", PREPROCESS),
                           ("wanpre.models", PREPROCESS / "models"),
                           ("wanpre.pose_utils", PREPROCESS / "pose_utils")):
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    onnx = importlib.import_module("wanpre.models.onnx_models")
    utils = importlib.import_module("wanpre.pose_utils.pose2d_utils")
    root = pathlib.Path(E2E_ROOT + "/ComfyUI_Wan/models/detection")
    detector = onnx.Yolo(str(root / "yolov10m.onnx"), device)
    pose_model = onnx.ViTPose(str(root / "onnx" / "vitpose_h_wholebody_model.onnx"), device)
    detector.reinit()
    pose_model.reinit()
    _DETECTORS[device] = (detector, pose_model, utils)
    return _DETECTORS[device]


def character_pose(image, device="CUDAExecutionProvider"):
    """AAPose-20 keypoints of the person in an image, in pixels.

    The workflow's own ViTPose, run the way ``PoseAndFaceDetection`` runs it, so
    the reference's landmarks, the drawn skeleton's, and a scan of a RENDERED
    video are all the same 20 slots and can be compared one to one.
    """
    import cv2
    detector, pose_model, utils = _detectors(device)

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    bbox = detector(cv2.resize(rgb, (640, 640)).transpose(2, 0, 1)[None],
                    np.array([[height, width]]))[0][0]["bbox"]
    if bbox is None or len(bbox) < 4:
        raise SystemExit("no person detected")
    resolution = (256, 192)
    centre, scale = utils.bbox_from_detector(bbox, resolution, rescale=1.25)
    crop = utils.crop(rgb, centre, scale, resolution)[0]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalised = ((crop - mean) / std).transpose(2, 0, 1).astype(np.float32)
    keypoints = pose_model(normalised[None], np.array(centre)[None],
                           np.array(scale)[None])
    metas = utils.load_pose_metas_from_kp2ds_seq(keypoints, width=width, height=height)
    body = np.asarray(metas[0]["keypoints_body"], dtype=np.float64)
    body[:, 0] *= width
    body[:, 1] *= height
    return body


def person_box(image, device="CUDAExecutionProvider"):
    """(x0, y0, x1, y1) of the detected person, in pixels."""
    body = character_pose(image, device)
    seen = body[body[:, 2] > 0.3]
    if len(seen) == 0:
        raise SystemExit("no person detected")
    return [float(seen[:, 0].min()), float(seen[:, 1].min()),
            float(seen[:, 0].max()), float(seen[:, 1].max())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--figure", type=float, default=0.76,
                    help="how much of the frame height the person should fill")
    args = ap.parse_args()

    image = Image.open(args.image)
    body = character_pose(image)
    W, H = image.size
    # LANDMARKS, NOT THE DETECTOR'S BOX.  The box runs to the top of the hair
    # and its bottom to the shoes, and matching it is what made the earlier
    # crop 10-24% too big.  Neck-to-ankle is the span both this still and the
    # drawn skeleton have, so the crop is expressed in it -- and the full
    # figure's share of the frame follows from the same measured proportions
    # the face uses (docs section 68).
    neck = body[1, 1]
    ankle = 0.5 * (body[10, 1] + body[13, 1])
    top = float(min(body[[0, 14, 15, 16, 17], 1].min(), neck))
    bottom = float(max(body[[10, 13, 18, 19], 1].max(), ankle))
    figure = bottom - top
    crop_h = figure / args.figure
    crop_w = crop_h * args.width / args.height
    if crop_h > H or crop_w > W:
        raise SystemExit(
            "the person is {:.1%} of this image's height, so filling {:.0%} "
            "needs a {:.0f}x{:.0f} window and the image is only {}x{}".format(
                figure / H, args.figure, crop_w, crop_h, W, H))
    centre_x = 0.5 * (body[[2, 5, 8, 11], 0].min() + body[[2, 5, 8, 11], 0].max())
    left = min(max(centre_x - crop_w / 2, 0.0), W - crop_w)
    # Keep the same headroom the pose camera leaves, then slide if it runs out.
    up = min(max(top - 0.5 * (crop_h - figure), 0.0), H - crop_h)
    box = (int(round(left)), int(round(up)),
           int(round(left + crop_w)), int(round(up + crop_h)))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").crop(box).resize((args.width, args.height),
                                          Image.LANCZOS).save(out)
    got = (bottom - top) / crop_h
    print("{} -> {}  person {:.1%} of height -> {:.1%}".format(
        pathlib.Path(args.image).name, out, figure / H, got))
    if abs(got - args.figure) > 0.05:
        raise SystemExit("the crop landed at {:.1%}, not {:.0%}".format(got, args.figure))


if __name__ == "__main__":
    main()
