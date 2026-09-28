"""Animate the character still by WARPING it with the pose -- no weights needed.

WHY THIS EXISTS BESIDE THE DIFFUSION BACKEND.  ``animate_2d.py`` drives Wan
Animate, which needs a checkpoint this machine cannot fetch: measured host by
host, pypi.org and files.pythonhosted.org answer 200 while huggingface.co,
cdn-lfs.huggingface.co, modelscope.cn and github's API all fail.  A stage that
can only ever dry-run is not a connected pipeline, so this backend closes the
loop with classical 2D puppet warping: it produces a real 2D cartoon video from
exactly the same two inputs, today, and the diffusion backend replaces it
unchanged once weights are staged.

HOW IT WORKS, and what it is honestly not.  The character still is posed once
(its own 2D skeleton is read from the rest pose the still was rendered from),
and every frame is a piecewise-affine warp of the still that carries the still's
joints onto the driving frame's joints.  That is a puppet: limbs move and
foreshorten, and the silhouette follows the dance.  It cannot invent what the
drawing does not contain -- a turn to profile has no profile to show, and
occluded limbs do not reappear -- which is precisely what a diffusion animator
adds.  The two are therefore reported separately and never averaged.
"""
import numpy as np

# Triangles over the COCO-18 skeleton, chosen so every limb is spanned and the
# torso is not a single quad (a quad hinges; two triangles do not).
TRIANGLES = [
    (1, 2, 5), (1, 2, 8), (1, 5, 11), (1, 8, 11), (2, 8, 11), (2, 5, 11),
    (2, 3, 8), (3, 4, 8), (5, 6, 11), (6, 7, 11),
    (8, 9, 11), (9, 10, 11), (11, 12, 8), (12, 13, 8),
    (0, 1, 2), (0, 1, 5), (0, 14, 16), (0, 15, 17), (0, 16, 17),
]


def _affine(src, dst):
    """The 2x3 affine taking triangle ``src`` onto ``dst``."""
    a = np.array([[src[0][0], src[0][1], 1, 0, 0, 0],
                  [0, 0, 0, src[0][0], src[0][1], 1],
                  [src[1][0], src[1][1], 1, 0, 0, 0],
                  [0, 0, 0, src[1][0], src[1][1], 1],
                  [src[2][0], src[2][1], 1, 0, 0, 0],
                  [0, 0, 0, src[2][0], src[2][1], 1]], dtype=np.float64)
    b = np.array([dst[0][0], dst[0][1], dst[1][0], dst[1][1],
                  dst[2][0], dst[2][1]], dtype=np.float64)
    try:
        solved = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None
    return solved.reshape(2, 3)


def warp_frame(image, source_points, target_points, background):
    """Piecewise-affine warp of ``image`` from source joints to target joints."""
    import cv2

    height, width = image.shape[:2]
    out = np.array(background, dtype=np.uint8)
    out = np.broadcast_to(out, (height, width, 3)).copy()
    for tri in TRIANGLES:
        src = source_points[list(tri)]
        dst = target_points[list(tri)]
        if not (np.isfinite(src).all() and np.isfinite(dst).all()):
            continue
        matrix = _affine(src, dst)
        if matrix is None:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(np.round(dst)), 255)
        if not mask.any():
            continue
        warped = cv2.warpAffine(image, matrix.astype(np.float32), (width, height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        out[mask > 0] = warped[mask > 0]
    return out


def animate(still, still_joints, pose_joints, background=(245, 245, 245)):
    """[H,W,3] still + its joints + [T,18,2] target joints -> list of frames."""
    still = np.asarray(still, dtype=np.uint8)
    frames = []
    for target in pose_joints:
        frames.append(warp_frame(still, still_joints, target, background))
    return frames
