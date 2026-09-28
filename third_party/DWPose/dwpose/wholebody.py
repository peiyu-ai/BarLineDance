# Adapted from IDEA-Research/DWPose (onnx branch).
import numpy as np
import onnxruntime as ort

from .onnxdet import inference_detector
from .onnxpose import inference_pose


class WholebodyOnnx:
    """DWPose whole-body ONNX: YOLOX det + dw-ll_ucoco pose."""

    def __init__(self, det_path: str, pose_path: str, device: str = "cpu"):
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session_det = ort.InferenceSession(det_path, providers=providers)
        self.session_pose = ort.InferenceSession(pose_path, providers=providers)

    def __call__(self, ori_img_bgr: np.ndarray):
        """
        Args:
            ori_img_bgr: H×W×3 BGR uint8

        Returns:
            keypoints: [N, K, 2] pixel coordinates
            scores: [N, K]
        """
        det_result = inference_detector(self.session_det, ori_img_bgr)
        keypoints, scores = inference_pose(self.session_pose, det_result, ori_img_bgr)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3,
        ).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints_out = keypoints_info[..., :2]
        scores_out = keypoints_info[..., 2]
        return keypoints_out, scores_out
