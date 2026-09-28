# DWPose (ONNX), vendored

`dwpose/` is IDEA-Research/DWPose's onnx branch — YOLOX person detection into
the `dw-ll_ucoco_384` whole-body pose model — copied from the Lodge repo's
`dld/data/dwpose/`, which is where this corpus's existing 2D keypoints came
from.  The four files are unmodified; keeping them byte-identical is what makes
"the same detector as the published corpus" a checkable claim rather than a
recollection.

What is **not** vendored is Lodge's `video_pose_extractor.py` wrapper.  Its
`extract_frame` holds the previous frame's keypoints whenever fewer than three
joints clear the score threshold:

```python
visible = sc > self.score_thresh
if visible.sum() < 3 and self._last_kp is not None:
    kp, sc = self._last_kp, self._last_sc
```

For Lodge that fill is harmless — it wants a dense pose track. For this repo it
destroys the signal two separate consumers depend on: `preprocess_wild_3d.py`
gates on `visible_joint_fraction` and on `frozen_joint_pair_fraction`, and the
fill makes an empty frame look like a visible, motionless dancer on both counts.
Boundary detection needs the same signal for the opposite reason: "nobody is on
screen here" is precisely where a clip should end.  `tools/dwpose_video.py` is
this repo's wrapper and records detections as they came.

## Weights

`weights/` holds `yolox_l.onnx` (217 MB) and `dw-ll_ucoco_384.onnx` (134 MB),
fetched by `tools/setup_dwpose_env.sh`.  They are not in git; like the rest of
`third_party/`, they live in the team OSS store (`tools/oss_assets.py`).

## Runtime

The ONNX sessions need `onnxruntime-gpu`; the interpreter this repo runs under
has CPU-only `onnxruntime` 1.17, and CPU inference over 76 hours of video is not
a real option. `tools/setup_dwpose_env.sh` builds `.venv_ortgpu` for exactly
this step, matching how Lodge isolates it.

## License

DWPose is Apache-2.0; the `dw-ll_ucoco_384` weights are trained on COCO-WholeBody
(CC BY 4.0) and UBody. Recorded here because the release side has to know.
