"""Draw the generated dance as the pose video SteadyDancer was trained to read.

WHY THIS EXISTS BESIDE ``draw_pose_video.py``.  That one draws the ingest's own
COCO-18 skeleton, which is what a human checks the projection against.  The
SteadyDancer model does not read a skeleton, it reads a PICTURE of one: its
pose branch is ``pose images -> WanVideoEncode (VAE) -> pose_latents ->
WanVideoAddSteadyDancerEmbeds``.  So the drawing convention -- limb colours,
stick widths, which joints connect -- is part of the model's input distribution,
and a skeleton drawn some other way is out of distribution no matter how correct
the joint positions are.

THE DRAWING IS NOT REIMPLEMENTED HERE.  ``draw_aapose_by_meta_new`` and
``AAPoseMeta`` are imported from ``ComfyUI-WanAnimatePreprocess``, the same
functions the workflow's own ``DrawViTPose`` node calls, so the convention is
identical BY CONSTRUCTION rather than by my reading of it.  This is the whole
point: the alternative was to copy twenty colours and a limb table by eye.

AAPOSE IS COCO-18 PLUS TWO TOES.  Read off ``draw_aapose_new``'s own
``new_kep_list`` and ``limbSeq``: slots 0-17 are exactly the OpenPose COCO-18
order ``project_pose_2d`` already emits, slot 18 is LToe (drawn to LAnkle, slot
13) and slot 19 is RToe (drawn to RAnkle, slot 10).  SMPL has both -- joints 10
and 11, ``l_foot`` and ``r_foot`` -- so the two extra points are free.

HANDS AND FACE ARE LEFT EMPTY, at confidence 0, and that is a real limitation
worth stating rather than hiding: SMPL-24 has no finger joints, so the character
will be animated with no hand articulation.  The head still turns, because the
four COCO face points (eyes, ears) are synthesised from the head's own frame in
``project_pose_2d`` and AAPose draws the head from those.
"""
import argparse
import hashlib
import json
import math
import pathlib
import pickle
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from render2d.project_pose_2d import (  # noqa: E402
    SMPL, to_coco18, project, align_heading, head_frame, CAMERA_FACING)

import os

# Root of the sibling checkouts (ComfyUI_Wan, Lodge, this repo); set E2E_ROOT.
E2E_ROOT = os.environ.get("E2E_ROOT", "/workspace/e2e")

PREPROCESS = pathlib.Path(
    E2E_ROOT + "/ComfyUI_Wan/custom_nodes/"
    "ComfyUI-WanAnimatePreprocess")
sys.path.insert(0, str(PREPROCESS))
from pose_utils.pose2d_utils import AAPoseMeta            # noqa: E402
from pose_utils.human_visualization import draw_aapose_by_meta_new  # noqa: E402

SOURCE_FPS = 30.0
N_BODY = 20
N_HAND = 21
N_FACE = 69


# A point on the head is visible once its outward normal has come round far
# enough to face the lens.  The ramp runs from -0.35 to -0.05 so the switch sits
# at -0.20 -- a little past the silhouette, which is where a real detector loses
# a feature -- and takes a few frames rather than blinking.
# A 40-degree band centred on the silhouette (110 deg to 70 deg away from the
# lens), not the 7-degree one the first version used.  Narrow is worse twice
# over: on a fast turn it is less than a frame, and once the series is smoothed
# a narrow dip is averaged away entirely, so the face would stay lit through a
# back-facing moment -- the contradictory cue this exists to remove.
FACE_VISIBLE_LOW, FACE_VISIBLE_HIGH = -0.342, 0.342
# EARS GET THEIR OWN, MUCH MORE PERMISSIVE BAND.  An ear sits on the SIDE of the
# head, so its outward normal is already at the silhouette when the face is
# straight at the lens -- putting it on the nose's band scored a frontal ear at
# exactly 0.5 and the ears then vanished on two thirds of frames, which no
# detector does.  It is hidden only once the head has turned far enough to put
# it behind: this band runs 139 to 105 degrees.
EAR_VISIBLE_LOW, EAR_VISIBLE_HIGH = -0.75, -0.25
# ...and then over TIME.  The band above is about 7 degrees wide, which on a
# fast turn is less than one frame: measured on 7618203431723357818, the dance
# turns up to 24.6 deg/frame, so the face did not fade, it POPPED -- and the
# operator saw "转身突然加速闪烁,看着画面变化不丝滑".  A quarter of a second of
# smoothing makes the crossing take about eight frames however fast the body
# turns, which is what a fade has to be to read as one.
FACE_SMOOTH_SECONDS = 0.25


# WHAT THE REAL DETECTOR DRAWS, as a function of the TRUE yaw.  DWPose on 2,300
# corpus clips (781k frames) against GVHMR's camera-frame facing -- the pictures
# SteadyDancer was trained on are drawn from exactly this kind of output:
#
#     |yaw|     nose>=0.5  far ear>=0.5  L/R labelled as the truth
#     0-15        96%         97%            98.6%
#     45-60       96%         90%            97.9%
#     75-90       94%         80%            84.7%
#     105-120     89%         80%            86.3%
#     120-150     79%         87%            94.0%
#     150-180     55%         91%            90.7%
#
# So in training a MISSING EAR essentially never happens, at any angle, and a
# drawn nose is no evidence of a front view (half the full back views have one);
# what does separate front from back is the left/right colour placement, which
# the projection already gets right by construction.  The "geometric" bands
# above drop the far ear from |yaw| 30 deg -- 15% drawn at 30-45, 0.2% at 45-60,
# against the detector's 95% and 90% -- which hands the model a head shape it
# has only ever seen on someone turning away, and every phantom back view the
# 2026-09-21 census found starts in that band.
#
# The "detector" mode therefore draws both ears always, and keeps the nose and
# eyes until the body is past 120 deg, fading them out by 150 deg -- the one
# face cue that IS back-view evidence in the training data (absent on ~45% of
# real back views, on under 6% of anything facing the camera).
DETECTOR_FACE_FADE_START_DEG, DETECTOR_FACE_FADE_END_DEG = 120.0, 150.0


def face_confidence_detector(joints):
    """[T, 5] like ``face_confidence``, but matching the detector's statistics."""
    _across, _up, forward = head_frame(joints)
    toward = np.asarray(CAMERA_FACING, dtype=np.float64)[:2]
    facing = forward[:, :2] @ toward
    low = math.cos(math.radians(DETECTOR_FACE_FADE_END_DEG))
    high = math.cos(math.radians(DETECTOR_FACE_FADE_START_DEG))
    front = np.clip((facing - low) / (high - low), 0.0, 1.0)
    ears = np.ones_like(front)
    stacked = np.stack([front, front, front, ears, ears], axis=1)
    return _smooth_columns(stacked, FACE_SMOOTH_SECONDS)


# WHERE THE DETECTOR PUTS THE FIVE FACE POINTS, by |shoulder yaw|: medians over
# the same 781k corpus frames (all five points at conf >= 0.3), in units of the
# neck-to-mid-hip length, dx toward the side of the image the face points to,
# dy image-DOWN from the neck.  "N"/"F" = the eye/ear on the camera side / the
# other side.  Bin centres; 0 and 180 are symmetrised anchors (a face at exactly
# 0 or 180 has no "facing side", so the near/far roles must meet there or the
# points jump when the sign of a near-zero yaw flips).
#
# The shape that matters is past 75 deg, where our rigid face and the detector
# part ways: the detector's nose keeps sliding out to 0.27 L while ours stops at
# 0.13 L, its eyes pile up with the nose at the EDGE of the head outline, and its
# ear spread never falls below 0.13 L while ours collapses to 0.03 L -- so on a
# turned head we drew a face narrower and more "behind" than any training
# picture has.  Below 45 deg the two agree within 0.03 L.
FACE_TEMPLATE_YAW = np.array(
    [0.0, 7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5, 112.5, 127.5, 142.5, 157.5, 173.0, 180.0])
#                     nose            eyeN            eyeF            earN            earF
FACE_TEMPLATE = np.array([
    [[+0.000, -0.342], [-0.064, -0.396], [+0.064, -0.396], [-0.158, -0.336], [+0.158, -0.336]],
    [[+0.010, -0.342], [-0.055, -0.396], [+0.073, -0.396], [-0.154, -0.336], [+0.162, -0.335]],
    [[+0.049, -0.339], [-0.022, -0.392], [+0.102, -0.393], [-0.141, -0.331], [+0.170, -0.332]],
    [[+0.080, -0.330], [+0.006, -0.381], [+0.123, -0.385], [-0.130, -0.322], [+0.173, -0.327]],
    [[+0.106, -0.315], [+0.031, -0.365], [+0.136, -0.372], [-0.111, -0.307], [+0.162, -0.317]],
    [[+0.143, -0.297], [+0.072, -0.347], [+0.158, -0.354], [-0.080, -0.294], [+0.140, -0.306]],
    [[+0.184, -0.284], [+0.120, -0.333], [+0.183, -0.341], [-0.033, -0.282], [+0.100, -0.296]],
    [[+0.218, -0.281], [+0.158, -0.330], [+0.204, -0.337], [+0.019, -0.277], [+0.054, -0.289]],
    [[+0.251, -0.284], [+0.200, -0.333], [+0.228, -0.339], [+0.074, -0.282], [+0.006, -0.287]],
    [[+0.268, -0.289], [+0.225, -0.341], [+0.233, -0.343], [+0.131, -0.293], [-0.050, -0.293]],
    [[+0.268, -0.293], [+0.238, -0.346], [+0.213, -0.347], [+0.177, -0.306], [-0.090, -0.307]],
    [[+0.215, -0.303], [+0.205, -0.354], [+0.091, -0.355], [+0.178, -0.314], [-0.127, -0.318]],
    [[+0.076, -0.307], [+0.151, -0.357], [-0.063, -0.357], [+0.168, -0.317], [-0.151, -0.319]],
    [[+0.000, -0.307], [+0.107, -0.357], [-0.107, -0.357], [+0.160, -0.318], [-0.160, -0.318]],
])
# The detector's nose is drawn on 55% of full back views and its eyes on 58-79%;
# its 50% point for the nose is about 172 deg.  So: eyes and ears always, the
# nose fading out over the last 20 degrees.
CALIBRATED_NOSE_FADE_START_DEG, CALIBRATED_NOSE_FADE_END_DEG = 160.0, 180.0
# Real ear spread never falls below 0.13 L, not even at profile (corpus census).
EAR_SPREAD_FLOOR = 0.13


def shoulder_yaw(joints):
    """Signed yaw of the chest normal, degrees; 0 = facing the camera.

    Positive = the chest normal leans toward +x, which the projection puts on
    the image LEFT (image right is -x).
    """
    _across, _up, forward = head_frame(joints)
    toward = np.asarray(CAMERA_FACING, dtype=np.float64)
    side = np.cross(toward, [0.0, 0.0, 1.0])          # +x for a camera on +y
    return np.degrees(np.arctan2(forward @ side, forward @ toward))


def face_confidence_calibrated(joints):
    """[T, 5]: eyes and ears always drawn; the nose fades out past 160 deg."""
    yaw = np.abs(shoulder_yaw(joints))
    span = CALIBRATED_NOSE_FADE_END_DEG - CALIBRATED_NOSE_FADE_START_DEG
    nose = np.clip((CALIBRATED_NOSE_FADE_END_DEG - yaw) / span, 0.0, 1.0)
    ones = np.ones_like(nose)
    stacked = np.stack([nose, ones, ones, ones, ones], axis=1)
    return _smooth_columns(stacked, FACE_SMOOTH_SECONDS)


def _template_ear_crossing():
    """|yaw| where the template's near/far ear gap changes sign (about 102.6 deg)."""
    grid = np.linspace(0.0, 180.0, 3601)
    gap = (np.interp(grid, FACE_TEMPLATE_YAW, FACE_TEMPLATE[:, 4, 0])
           - np.interp(grid, FACE_TEMPLATE_YAW, FACE_TEMPLATE[:, 3, 0]))
    flips = np.nonzero(np.diff(np.sign(gap)) != 0)[0]
    return float(grid[flips[0]]) if len(flips) else 180.0


EAR_CROSSING_DEG = _template_ear_crossing()
# Within this many degrees of the crossing the floor tapers to zero, so the two
# ears pass through each other CONTINUOUSLY instead of swapping sides in one
# frame (review 2026-09-21: a hard floor jumped both ears 0.13 L at 102.6 deg,
# a per-frame flicker on any hold near profile).
EAR_FLOOR_TAPER_DEG = 12.0


def face_points_calibrated(uv, joints, width, height, world):
    """Replace the five face points of normalised ``uv`` with the detector's
    template, placed about each frame's projected neck.

    IN THE IMAGE, after the projection, because the template IS an image
    statistic (where a 2D detector draws the points), not an anatomy: no rigid
    3D face reproduces an ear spread that never falls below 0.13 L at profile.
    The size is the clip's MEDIAN torso, so the face does not breathe.

    ``world`` is the [T, 20, 3] AAPose world array ``uv`` was projected from;
    it supplies the pixels-per-metre for the head's own motion (below).
    """
    px = uv * np.array([width, height])
    neck = px[:, 1]
    hips = 0.5 * (px[:, 8] + px[:, 11])
    unit = float(np.median(np.linalg.norm(neck - hips, axis=1)))
    yaw = shoulder_yaw(joints)
    magnitude = np.abs(yaw)
    table = np.stack([np.stack([np.interp(magnitude, FACE_TEMPLATE_YAW, FACE_TEMPLATE[:, point, axis])
                                for axis in range(2)], axis=-1)
                      for point in range(5)], axis=1)            # [T, 5, 2]
    # ONE sign decides both the side the face points to and which eye/ear is
    # the near one.  Positive yaw puts the chest normal toward +x = image left,
    # and then the person's LEFT side is toward the camera (measured, not
    # assumed: (r - l).toward = -sin(yaw)).  Deriving the two from separate
    # tests broke the tie at exactly 0 differently and wrote a mirrored,
    # back-view labelling on a frontal frame (review 2026-09-21).
    turned_left = yaw > 0
    face_side = np.where(turned_left, -1.0, 1.0)
    right_near = ~turned_left
    offsets = table.copy()
    offsets[:, :, 0] *= face_side[:, None]
    placed = neck[:, None, :] + offsets * unit
    # THE EAR SPREAD HAS A FLOOR.  The template holds the median of each ear's
    # POSITION, and near profile the detector's ear order is a coin flip (it
    # agrees with the true order 46% of the time at 90-105 deg), so the two
    # medians converge -- to 0.035 L at 97.5 deg -- while on every real frame
    # the ears stay at least ~0.13 L apart.  A difference of medians is not the
    # median of differences; the floor restores the per-frame statistic,
    # keeping the template's order, and tapers to zero where that order flips.
    near_x, far_x = placed[:, 3, 0], placed[:, 4, 0]
    gap = far_x - near_x
    taper = np.clip(np.abs(magnitude - EAR_CROSSING_DEG) / EAR_FLOOR_TAPER_DEG, 0.0, 1.0)
    floor = EAR_SPREAD_FLOOR * unit * taper
    widen = np.where(np.abs(gap) < floor,
                     0.5 * (np.where(gap >= 0, floor, -floor) - gap), 0.0)
    placed[:, 3, 0] = near_x - widen
    placed[:, 4, 0] = far_x + widen
    nose, eye_near, eye_far, ear_near, ear_far = (placed[:, k] for k in range(5))
    r_eye = np.where(right_near[:, None], eye_near, eye_far)
    l_eye = np.where(right_near[:, None], eye_far, eye_near)
    r_ear = np.where(right_near[:, None], ear_near, ear_far)
    l_ear = np.where(right_near[:, None], ear_far, ear_near)
    # THE HEAD'S OWN MOTION, WITHOUT ITS FORWARD OFFSET.  The first version took
    # the rigid ear midpoint's deviation, which is the head joint's -- and
    # SMPL's head joint sits ~0.1 L IN FRONT of the shoulder midpoint, so on a
    # turned body that deviation carried fwd * sin(yaw) and pushed every turned
    # face a further ~0.1 L toward the side it faces, on top of a template that
    # already contains the real head's forward offset (review 2026-09-21: the
    # nose landed 0.30 L out at 90 deg against the template's 0.20).  So: the
    # head-to-neck vector in 3D with its component along the chest normal
    # removed, projected at the clip's own pixels-per-metre (exact for this
    # orthographic camera: one metre is W/span pixels on both axes).
    world = np.asarray(world, dtype=np.float64)
    ankles_w = 0.5 * (world[:, 10] + world[:, 13])
    ankles_px = 0.5 * (px[:, 10] + px[:, 13])
    metres = np.linalg.norm((world[:, 1] - ankles_w)[:, [0, 2]], axis=1)
    per_metre = float(np.median(np.linalg.norm(neck - ankles_px, axis=1) / np.maximum(metres, 1e-6)))
    _across, _up, forward = head_frame(joints)
    lean = np.asarray(joints)[:, SMPL["head"]] - world[:, 1]
    lean = lean - np.sum(lean * forward, axis=1, keepdims=True) * forward
    lean_px = np.stack([-lean[:, 0], -lean[:, 2]], axis=1) * per_metre
    nod = lean_px - np.median(lean_px, axis=0)
    out = px.copy()
    for slot, point in ((0, nose), (14, r_eye), (15, l_eye), (16, r_ear), (17, l_ear)):
        out[:, slot] = point + nod
    return out / np.array([width, height])


def face_confidence(joints):
    """[T, 5] confidence for nose, REye, LEye, REar, LEar, from the 3D.

    WHY IT IS NOT ALL ONES, which is what the first version emitted.  The
    operator, 2026-09-14: "转身的时候头部和身体出现 mismatch 错位".  Our four
    face points are SYNTHESISED in ``project_pose_2d.to_coco18`` from the
    shoulders' own frame and were drawn at full confidence on every frame, so a
    body that had turned away still carried a complete, front-facing face --
    a cue that contradicts the limbs, and the model resolved the contradiction
    by putting a forward-facing head on a turning body.

    A real ViTPose run does not do that: the nose and eyes drop out when the
    head turns away and the far ear drops out in profile, which is exactly the
    signal the model was trained to read.  Here that is computed from geometry
    rather than detected -- the head's own frame is already available -- so the
    nose and eyes follow the chest normal and each ear follows its own side.
    """
    across, _up, forward = head_frame(joints)
    toward = np.asarray(CAMERA_FACING, dtype=np.float64)[:2]

    def ramp(normal_xy, low, high):
        facing = normal_xy @ toward
        return np.clip((facing - low) / (high - low), 0.0, 1.0)

    front = ramp(forward[:, :2], FACE_VISIBLE_LOW, FACE_VISIBLE_HIGH)
    right_ear = ramp(-across[:, :2], EAR_VISIBLE_LOW, EAR_VISIBLE_HIGH)
    left_ear = ramp(across[:, :2], EAR_VISIBLE_LOW, EAR_VISIBLE_HIGH)
    stacked = np.stack([front, front, front, right_ear, left_ear], axis=1)
    return _smooth_columns(stacked, FACE_SMOOTH_SECONDS)


def _smooth_columns(values, seconds, fps=30.0):
    width = max(1, int(round(seconds * fps)) | 1)
    if width < 3 or len(values) < width:
        return values
    pad = width // 2
    kernel = np.ones(width) / width
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, column], kernel, mode="valid")[:len(values)]
                     for column in range(values.shape[1])], axis=1)


# OpenPose hand topology: slot 0 is the wrist, then five fingers of four
# joints each, in the order thumb, index, middle, ring, little.
HAND_FINGERS = 5
HAND_SEGMENTS = 4
# Where each finger points, as a fraction of a right angle away from the palm
# direction, and how long it is relative to the wrist-to-hand distance.  From
# the proportions of a hand, not tuned: the thumb leaves the palm at a wide
# angle and is short, the middle finger runs straight on and is longest.
FINGER_SPREAD = np.array([-0.55, -0.18, 0.0, 0.17, 0.33])
FINGER_LENGTH = np.array([0.75, 1.15, 1.25, 1.15, 0.92])


def synth_hands(joints, uv_scale):
    """[T, 21, 3] per hand, laid out from the wrist toward SMPL's hand joint.

    WHY INVENT THEM.  SMPL-24 has no fingers, so the first version passed hand
    keypoints of zero confidence and drew none -- and the operator's read of the
    result was that the hands melt.  That is what the model does when it is
    given nothing: ``WanVideoAddSteadyDancerEmbeds`` conditions on a PICTURE of
    a skeleton, and the pictures it was trained on have hands in them, so a
    wrist that simply stops is out of distribution and the hand is hallucinated
    fresh every frame.

    WHAT IS HONEST TO SYNTHESISE.  SMPL does carry ``l_hand``/``r_hand``, one
    joint past the wrist, so the hand's DIRECTION and SIZE are real; only the
    articulation is not.  A fixed open fan about that direction is therefore a
    truthful statement of what we know (where the hand is, which way it points,
    how big it is) and no statement at all about the fingers -- which is better
    than the model's alternative of inventing all four.  The fan does not
    animate, so a reviewer must not read finger motion into these videos.
    """
    left = joints[:, SMPL["l_hand"]] - joints[:, SMPL["l_wrist"]]
    right = joints[:, SMPL["r_hand"]] - joints[:, SMPL["r_wrist"]]
    hands = []
    for wrist_index, direction in ((SMPL["l_wrist"], left), (SMPL["r_wrist"], right)):
        wrist = joints[:, wrist_index]
        # IN THE IMAGE PLANE, and at a CONSTANT size.  The projection is
        # orthographic down +y, so a hand pointing at the lens has almost no
        # extent in x or z and a fan built on the raw 3D direction collapses to
        # a thin streak -- which reads as "no hand" and is exactly what the
        # model must not be told.  A real hand neither shrinks nor vanishes when
        # it turns, so the direction used is the visible part of the forearm's
        # own heading and the size is the clip's median wrist-to-hand distance.
        planar = np.stack([direction[:, 0], np.zeros(len(direction)),
                           direction[:, 2]], axis=1)
        norm = np.linalg.norm(planar, axis=1, keepdims=True)
        # Straight down when the hand points along the view axis: the forearm
        # says nothing about the in-plane heading there, and a hanging hand is
        # the least wrong guess.
        fallback = np.tile(np.array([0.0, 0.0, -1.0]), (len(direction), 1))
        unit = np.where(norm > 1e-4, planar / np.maximum(norm, 1e-9), fallback)
        side = np.stack([-unit[:, 2], np.zeros(len(unit)), unit[:, 0]], axis=1)
        size = float(np.median(np.linalg.norm(direction, axis=1)))
        points = np.zeros((len(joints), 1 + HAND_FINGERS * HAND_SEGMENTS, 3))
        points[:, 0] = wrist
        for finger in range(HAND_FINGERS):
            angle = FINGER_SPREAD[finger] * (math.pi / 2.0)
            heading = math.cos(angle) * unit + math.sin(angle) * side
            for segment in range(HAND_SEGMENTS):
                reach = FINGER_LENGTH[finger] * (segment + 1) / HAND_SEGMENTS
                slot = 1 + finger * HAND_SEGMENTS + segment
                points[:, slot] = wrist + heading * size * reach
        hands.append(points)
    return hands


# HANDS v2: the same in-plane, constant-size fan, but with a CONSISTENT thumb.
# ``synth_hands`` rotates both fans the same way, so in image space the left hand
# is drawn palm-to-camera and the right hand back-of-hand on EVERY frame,
# facing or turned away (audit 2026-09-21: 100% of frames of the ground truth
# and of fix7; real ViTPose hands agree with each other 64% of the time, ours
# 0%) -- one front-view hand and one back-view hand on every picture.
# Here the thumb goes where an anatomical thumb projects, chirality * (hand axis
# x palm normal), with the palm normal = -chest normal (back of the hand toward
# the camera when the body faces it: real hands show their back 62% of the
# time), so the two fans mirror each other and turn over WITH the body.  Shape
# from ViTPose-H on the fixed ten (241 confident hands): fingertip angles from
# the middle finger 11.7 / 6.0 / 0 / 5.7 / 12.6 deg (ours were 49.5 / 16.2 / 0 /
# 15.3 / 29.7), tip lengths 0.90 / 1.06 / 1.00 / 0.92 / 0.83 of the middle.
# Measured on the same frames: hands agree 0.00 -> 0.96, thumb 50 -> 12 deg.
HAND_V2_SPREAD = np.array([-11.7, -6.0, 0.0, 5.7, 12.6]) / 90.0
HAND_V2_LENGTH = 1.25 * 0.86 * np.array([0.90, 1.06, 1.00, 0.92, 0.83])
# An edge-on hand has no thumb side; it keeps its last one until the projected
# thumb clears this fraction of the fan's side vector.
HAND_V2_HYSTERESIS = 0.2


def synth_hands_v2(joints):
    """[T, 21, 3] per hand (left, right), like ``synth_hands`` but mirror-consistent."""
    _across, _up, forward = head_frame(joints)
    palm = -forward
    hands = []
    for wrist_name, hand_name, chirality in (("l_wrist", "l_hand", -1.0),
                                             ("r_wrist", "r_hand", +1.0)):
        wrist = joints[:, SMPL[wrist_name]]
        direction = joints[:, SMPL[hand_name]] - wrist
        planar = np.stack([direction[:, 0], np.zeros(len(direction)), direction[:, 2]], axis=1)
        norm = np.linalg.norm(planar, axis=1, keepdims=True)
        fallback = np.tile(np.array([0.0, 0.0, -1.0]), (len(direction), 1))
        unit = np.where(norm > 1e-4, planar / np.maximum(norm, 1e-9), fallback)
        side = np.stack([-unit[:, 2], np.zeros(len(unit)), unit[:, 0]], axis=1)
        axis = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        thumb = chirality * np.cross(axis, palm)
        score = np.sum(thumb * side, axis=1)        # side has no y, so this is the in-plane part
        sign = np.empty(len(score))
        current = 1.0 if score[0] >= 0 else -1.0
        for t, value in enumerate(score):
            if value > HAND_V2_HYSTERESIS:
                current = 1.0
            elif value < -HAND_V2_HYSTERESIS:
                current = -1.0
            sign[t] = current
        size = float(np.median(np.linalg.norm(direction, axis=1)))
        points = np.zeros((len(joints), 1 + HAND_FINGERS * HAND_SEGMENTS, 3))
        points[:, 0] = wrist
        for finger in range(HAND_FINGERS):
            angle = HAND_V2_SPREAD[finger] * (math.pi / 2.0)
            heading = math.cos(angle) * unit - math.sin(angle) * sign[:, None] * side
            for segment in range(HAND_SEGMENTS):
                points[:, 1 + finger * HAND_SEGMENTS + segment] = (
                    wrist + heading * size * HAND_V2_LENGTH[finger] * (segment + 1) / HAND_SEGMENTS)
        hands.append(points)
    return hands


# BODY RETARGET v2 (audit 2026-09-21, the fixed ten, ground truth through this
# pipeline paired frame by frame with the same clip's real DWPose, frontal
# frames only): half-widths and hip height in units of the neck-to-ankle length.
#                  real    ours    v1 (retarget_torso)   v2
#   shoulder half  0.109   0.140   0.101                 0.110
#   hip half       0.085   0.046   0.082                 0.085
#   hip height     0.403   0.439   0.439                 0.404
#   upper arm      0.163   0.173   0.198                 0.173
# v1 moved only the four torso KEYPOINTS, which stretched the drawn upper arm
# and left the hips at SMPL's femoral heads; v2 moves each arm chain with its
# shoulder and raises the hips along the spine to where COCO annotates them.
BODY_V2_SHOULDER_HALF = 0.1125
BODY_V2_HIP_HALF = 0.087
BODY_V2_HIP_RAISE = 0.036


def retarget_body(world):
    """[T, 20, 3] AAPose world points -> (retargeted, (left wrist shift, right wrist shift)).

    In 3D along each frame's own body axes, so a turned body still narrows by
    cos(yaw).  The neck and ankles do not move, so the character fit does not.
    """
    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)

    out = np.array(world, dtype=np.float64, copy=True)
    neck = 0.5 * (out[:, 2] + out[:, 5])
    ankles = 0.5 * (out[:, 10] + out[:, 13])
    height = float(np.median(np.linalg.norm((neck - ankles)[:, [0, 2]], axis=1)))
    across = unit(out[:, 5] - out[:, 2])
    new_right = neck - across * BODY_V2_SHOULDER_HALF * height
    new_left = neck + across * BODY_V2_SHOULDER_HALF * height
    shift_right, shift_left = new_right - out[:, 2], new_left - out[:, 5]
    out[:, [3, 4]] += shift_right[:, None]
    out[:, [6, 7]] += shift_left[:, None]
    out[:, 2], out[:, 5] = new_right, new_left
    middle = 0.5 * (out[:, 8] + out[:, 11])
    pelvis = unit(out[:, 11] - out[:, 8])
    spine = unit(neck - middle)
    middle = middle + spine * BODY_V2_HIP_RAISE * height
    out[:, 8] = middle - pelvis * BODY_V2_HIP_HALF * height
    out[:, 11] = middle + pelvis * BODY_V2_HIP_HALF * height
    return out, (shift_left, shift_right)


# YAW SOFT LIMIT -- an opt-in trade, OFF by default.  Every phantom back view
# the 2026-09-21 census found began with the body at 45-67 deg from the camera,
# the band where an orthographic skeleton reads equally well as a front
# three-quarter or its mirrored back three-quarter.  Compressing the RENDERED
# yaw in that band keeps the picture on the unambiguous side; past 90 deg the
# map expands again so a real turnaround still reaches 180 and crosses the
# profile zone faster than it did.  What it costs: turns look smaller in 2D (a
# 60 deg turn draws as ~40 deg at 0.5).  The operator has said turning is a
# beat-hitting device, so this is theirs to switch on, not a default.
YAW_SOFT_KNEE_DEG = 20.0


def soft_limit_yaw(joints, ratio):
    """Rotate each frame about the pelvis's vertical so |yaw| is compressed.

    identity below the knee; knee + (|y| - knee) * ratio up to 90 deg; then a
    straight line back to 180 at 180.  ``ratio`` 1 is the identity.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if ratio >= 1.0:
        return joints
    yaw = shoulder_yaw(joints)
    magnitude = np.abs(yaw)
    knee = YAW_SOFT_KNEE_DEG
    at_ninety = knee + (90.0 - knee) * ratio
    mapped = np.where(
        magnitude <= knee, magnitude,
        np.where(magnitude <= 90.0, knee + (magnitude - knee) * ratio,
                 at_ninety + (magnitude - 90.0) * (180.0 - at_ninety) / 90.0))
    # NOT smoothed.  A centred window started the correction before the turn,
    # rotating the figure the wrong way first and under-compressing fast turns
    # (review 2026-09-21); the map is continuous in yaw, so the unsmoothed
    # correction only carries the shoulders' own jitter times (1 - ratio).
    delta = np.radians(np.sign(yaw) * mapped - yaw)
    out = joints.copy()
    pivot = joints[:, SMPL["pelvis"], :2]
    for t in range(len(joints)):
        # shoulder_yaw is measured toward +x from +y; reducing it is a
        # rotation by +delta about z in that same sense.
        c, s = math.cos(-delta[t]), math.sin(-delta[t])
        xy = joints[t, :, :2] - pivot[t]
        out[t, :, 0] = pivot[t, 0] + c * xy[:, 0] - s * xy[:, 1]
        out[t, :, 1] = pivot[t, 1] + s * xy[:, 0] + c * xy[:, 1]
    return out


def fold_yaw(joints, cap):
    """--yaw-fold CAP: keep the RENDERED yaw inside the band SteadyDancer draws correctly.

    Measured 2026-09-24 on 27 short renders (7,390 frames, tools/score_2d_facing.py, categories of
    tools/pick_2d_facing_render.wrong_frames) against the driving body's shoulder yaw at the same instant:
    wrong 0.2% below 20 deg, 0.4% at 20-45, 2.9% at 45-67, 13% at 67-90, 57% at 90-120 and 84-88% past 120 --
    mostly "missed" (a face painted on a body whose back is to the camera).  Ground truth spends 11.9% of its
    frames past 67 deg and the 3D arms 3.2-6.2%, so turning away is dancing; drawing it is what fails, and no
    seed fixes it reliably.  So: identity up to CAP-25, a smooth rise to CAP at 90 deg (zero slope there), and a
    FOLD past 90 -- yaw 180-a is drawn as a -- so a back view becomes the matching front view and a turnaround
    becomes a twist out to CAP and back.  Continuous and periodic in yaw (h(180) = 0 = h(-180)), so no frame
    jumps.  The 3D panel keeps the real turn.  Off by default.
    """
    joints = np.asarray(joints, dtype=np.float64)
    if not cap or cap >= 90.0:
        return joints
    knee = max(0.0, float(cap) - 25.0)
    yaw = shoulder_yaw(joints)
    magnitude = np.abs(yaw)
    folded = np.where(magnitude <= 90.0, magnitude, 180.0 - magnitude)
    mapped = np.where(folded <= knee, folded,
                      knee + (float(cap) - knee) * np.sin((folded - knee) / (90.0 - knee) * np.pi / 2.0))
    delta = np.radians(np.sign(yaw) * mapped - yaw)
    out = joints.copy()
    pivot = joints[:, SMPL["pelvis"], :2]
    for t in range(len(joints)):
        c, s = math.cos(-delta[t]), math.sin(-delta[t])
        xy = joints[t, :, :2] - pivot[t]
        out[t, :, 0] = pivot[t, 0] + c * xy[:, 0] - s * xy[:, 1]
        out[t, :, 1] = pivot[t, 1] + s * xy[:, 0] + c * xy[:, 1]
    return out


def to_aapose20(joints, torso="smpl"):
    """[T, 24, 3] SMPL world joints -> [T, 20, 3] AAPose world points."""
    coco = to_coco18(joints)
    if torso == "detector":
        coco = retarget_torso(coco, joints)
    elif torso != "smpl":
        raise ValueError("torso must be 'smpl' or 'detector', got {!r}".format(torso))
    toes = np.stack([joints[:, SMPL["l_foot"]], joints[:, SMPL["r_foot"]]], axis=1)
    return np.concatenate([coco, toes], axis=1)


# Where the REAL detector puts shoulders and hips, as ratios the camera cannot
# move: DWPose ``keypoints.npy`` of the fixed ten clips, frames where the
# person's left shoulder is on the image right (facing the camera), per-clip
# medians then the median of the ten:
#     hip width / shoulder width     0.74-0.86, median 0.795   (ours 0.32)
#     shoulder width / neck-to-ankle 0.19-0.24, median 0.211   (ours 0.28)
# SMPL's hip joints are the femoral-head centres; COCO annotates the hips at the
# OUTER hip.  So our torso is a V the model never saw, and on width alone its
# pelvis reads as turned about 50 degrees on every frame (arccos(0.094/0.147)
# against the character's own hips).
DETECTOR_SHOULDER_OVER_NECK_ANKLE = 0.235
DETECTOR_HIP_OVER_SHOULDER = 0.72


def retarget_torso(coco, joints,
                   shoulder_ratio=DETECTOR_SHOULDER_OVER_NECK_ANKLE,
                   hip_ratio=DETECTOR_HIP_OVER_SHOULDER):
    """Move the COCO shoulder and hip points to the detector's convention.

    IN 3D, ABOUT EACH PAIR'S OWN MIDPOINT, with ONE factor per clip.  Scaling
    the 3D half-widths rather than the drawn pixels is what keeps the turn cue:
    a body turned by yaw still shows its shoulders and hips at cos(yaw) of their
    frontal width, because the widening happens before the projection.  The
    midpoints do not move, so the neck (the shoulder midpoint), the pelvis, the
    neck-to-ankle span the character fit uses, and every other joint are
    untouched -- only where the four torso KEYPOINTS sit changes, which is a
    labelling convention, not the skeleton.
    """
    out = np.array(coco, dtype=np.float64, copy=True)
    neck_z = 0.5 * (joints[:, SMPL["l_shoulder"], 2] + joints[:, SMPL["r_shoulder"], 2])
    ankle_z = 0.5 * (joints[:, SMPL["l_ankle"], 2] + joints[:, SMPL["r_ankle"], 2])
    neck_ankle = float(np.median(neck_z - ankle_z))
    for (right, left), target in (((2, 5), shoulder_ratio * neck_ankle),
                                  ((8, 11), hip_ratio * shoulder_ratio * neck_ankle)):
        width = float(np.median(np.linalg.norm(out[:, left] - out[:, right], axis=1)))
        factor = target / max(width, 1e-6)
        middle = 0.5 * (out[:, left] + out[:, right])
        out[:, left] = middle + factor * (out[:, left] - middle)
        out[:, right] = middle + factor * (out[:, right] - middle)
    return out


def metas_from_uv(uv, width, height, face=None, hands=None):
    """Normalised [T, 20, 2] -> the meta dicts ``AAPoseMeta`` consumes.

    ``from_humanapi_meta`` multiplies by (width, height), so what goes in here
    is normalised and what comes out is pixels -- the same path
    ``load_pose_metas_from_kp2ds_seq`` takes in the node.
    """
    metas = []
    for index, frame in enumerate(uv):
        confidence = np.ones((N_BODY, 1))
        if face is not None:
            # AAPose slots: 0 nose, 14 REye, 15 LEye, 16 REar, 17 LEar.
            for slot, value in zip((0, 14, 15, 16, 17), face[index]):
                confidence[slot, 0] = value
        body = np.concatenate([frame, confidence], axis=1)
        metas.append({
            "width": width, "height": height,
            "keypoints_body": body,
            # Confidence 0, so ``draw_aapose_new``'s threshold drops them.  A
            # zero-confidence point is skipped; a zero-POSITION point at
            # confidence 1 would be drawn in the frame's corner.
            "keypoints_left_hand": (
                np.concatenate([hands[0][index], np.ones((N_HAND, 1))], axis=1)
                if hands is not None else np.zeros((N_HAND, 3))),
            "keypoints_right_hand": (
                np.concatenate([hands[1][index], np.ones((N_HAND, 1))], axis=1)
                if hands is not None else np.zeros((N_HAND, 3))),
            "keypoints_face": np.zeros((N_FACE, 3)),
        })
    return metas


def fit_to_character(uv, reference, width, height, front=None):
    """Put the drawn skeleton on the reference person's OWN landmarks.

    SKELETON TO SKELETON, not box to box.  The first version matched the yolo
    box of the still to the bounding box of the drawn keypoints, and the two do
    not measure the same thing: the detector's box runs to the top of the HAIR
    while AAPose's highest keypoint is an eye or an ear, so the model had to
    build a skull above the target and the rendered body came out too big.
    Measured with the same detector on three clips and four frames each, the
    rendered figure was 1.10x to 1.24x the reference's height while frame 0
    (which IS the reference) read 1.03x.  The arithmetic agrees: the reference
    person is 479 px tall, a skull cap is about 10% of that, and
    (804-325+48)/832 = 63.3% -- the 63-64% that was measured.

    NECK TO ANKLE is the span used, because it is the one both sides have and
    neither a hairstyle nor a raised arm can move: the earlier box fit put the
    skeleton's top at whatever was highest that frame, which on a clip with
    overhead arms is a WRIST.

    The transform is still a similarity -- one scale for both axes -- and still
    built from medians over the whole clip, so the dancer rises and falls inside
    the frame instead of being pinned.
    """
    neck = uv[:, 1, :]
    ankles = 0.5 * (uv[:, 10, :] + uv[:, 13, :])
    # Median of each end, not the median of the difference: the two are not the
    # same number, and using different ones on the two sides makes "fit the pose
    # to its own landmarks" fail to be the identity -- which is the control this
    # has to pass.
    our_neck_v = float(np.nanmedian(neck[:, 1]))
    our_span = float(np.nanmedian(ankles[:, 1])) - our_neck_v
    our_centre_u = float(np.nanmedian(neck[:, 0]))

    ref_neck = reference[1]
    ref_ankles = 0.5 * (reference[10] + reference[13])
    ref_neck_v = ref_neck[1] / height
    ref_span = (ref_ankles[1] - ref_neck[1]) / height
    ref_centre_u = ref_neck[0] / width

    scale = ref_span / max(our_span, 1e-6)
    out = uv.copy()
    out[:, :, 1] = ref_neck_v + (uv[:, :, 1] - our_neck_v) * scale
    out[:, :, 0] = ref_centre_u + (uv[:, :, 0] - our_centre_u) * scale

    # THE HEAD IS FITTED SEPARATELY, because a cartoon is not built like a body.
    # One similarity cannot put the neck, the ankles AND the eyes where this
    # character has them: measured on townfair, the reference's eye-to-neck is
    # 0.1965 of its neck-to-ankle while ours is 0.1569, so a body-sized fit left
    # our face 13 px low on an 832-frame -- and the animator, told to put the
    # eyes there, TILTED THE HEAD DOWN to reach them.  That is the "头一直低着"
    # the operator reported twice.
    #
    # So the five face points get their own similarity onto the reference's own
    # face: the clip MEDIAN lands exactly on the character's eyes, nose and
    # ears, and every frame's deviation from that median is carried through at
    # the same scale, so the head still turns and nods.
    face = [0, 14, 15, 16, 17]
    theirs = np.stack([reference[i][:2] / np.array([width, height]) for i in face])
    mine = np.median(out[:, face, :], axis=0)
    their_width = float(np.linalg.norm(theirs[3] - theirs[4]))     # ear to ear
    my_width = float(np.linalg.norm(mine[3] - mine[4]))
    if front is not None:
        # --face-fit front (2026-09-23).  The distance between the two ears' clip MEDIANS collapses once the clip has
        # back-facing frames: the calibrated face template swaps left and right ear past ~103 deg of yaw, so the two
        # medians drift together and head_scale explodes (4.68 on 7650's hit arm, 14.1 on one val clip, against
        # 1.0-1.3 everywhere facing forward).  Every frame's face deviation -- body travel included -- is then
        # multiplied by it, the face leaves the neck, and SteadyDancer paints flying hair and bows the 3D never had
        # (DEFECTS §94).  Here the width is the median PER-FRAME ear gap over the frames that face the camera.
        gaps = np.linalg.norm(out[:, 16, :] - out[:, 17, :], axis=1)
        front = np.asarray(front, bool)[:len(gaps)]
        my_width = float(np.median(gaps[front])) if front.sum() >= 5 else float(np.median(gaps))
    head_scale = their_width / max(my_width, 1e-6)
    out[:, face, :] = (theirs.mean(axis=0)
                       + (out[:, face, :] - mine.mean(axis=0)) * head_scale)
    return out, {"scale": scale, "head_scale": head_scale,
                 "pose_neck_ankle": our_span,
                 "character_neck_ankle": ref_span,
                 "pose_neck_v": our_neck_v, "character_neck_v": ref_neck_v,
                 "pose_neck_u": our_centre_u, "character_neck_u": ref_centre_u}


# FITTED BONE BY BONE, NOT ONLY AS A WHOLE (operator 2026-09-22: "2d pose 进蒙皮
# 渲染时，要拉到和 image 首帧 figure 到同一尺寸，这样不会出现拉伸脖子，扭曲身体").
# ``fit_to_character`` matches ONE span (neck to ankle) and the face; every other
# bone kept the SMPL dancer's length.  Measured on the fixed ten against the
# character's own landmarks (units of each body's neck-to-ankle, frontal frames):
#
#     torso neck->mid-hip   character 0.329   ours 0.405   1.23x
#     forearm                         0.168        0.200   1.19x
#     hip width                       0.147        0.172   1.17x
#     thigh                           0.335        0.310   0.92x
#     (neck->nose, shoulders, upper arm, shin, ear width: 0.99-1.03x)
#
# The character is drawn with a short torso and long legs, so with the neck and
# ankles pinned our hips sat well below hers and the model had to stretch her to
# reach them.  Each bone is scaled to her length by one factor per clip, applied
# down the chain from the neck on every frame -- the way the official
# SteadyDancer pose alignment (MusePose's ``align_img``) retargets -- so a limb's
# direction and its foreshortening survive; only its full length changes.  Our
# "full length" is the 90th percentile of the drawn length over frames facing
# the camera: a limb is at its full in-plane length only when it lies in the
# image plane, and the character is measured standing straight, facing us.
BONE_FIT_PERCENTILE = 90
BONE_FIT_MAX_YAW_DEG = 30.0
BONE_FIT_CLAMP = (0.6, 1.6)
# THE NECK GETS A CEILING, NOT A FACTOR.  The drawn neck (shoulder midpoint to
# nose) swings 0.85-1.35x of the character's within a clip, because SMPL's
# shoulders ride up and down with the arms; the real detector swings as much on
# real dancers (cv 0.10-0.27), so the swing itself is in distribution.  What is
# not: the character's reference is RELAXED shoulders, and our relaxed,
# arms-down frames are exactly our longest necks -- 1.26x on the frames where the
# render shows a swan neck (703079 f124, 763751 f84, 741263 f190).  Shortening
# every neck instead would lower the median face, which is what once made the
# model tilt the head down (the "头一直低着" fix in fit_to_character).  So the
# median stays where fit_to_character put it and only the tail above the
# character's own neck is pulled back, the face moving as one rigid piece.
NECK_CEILING = 1.05          # of the character's neck->nose length; above detector noise
NECK_TAIL_KEEP = 0.25        # fraction of the excess above the ceiling kept


def fit_bones_to_character(uv, reference, width, height, facing_camera, hands=None):
    """Scale each bone of an already ``fit_to_character``-ed pose to the
    character's own length; returns (uv, hands, report).

    ``uv`` normalised [T, 20, 2]; ``reference`` the character's AAPose-20 in
    pixels; ``facing_camera`` [T] bool, the frames whose drawn lengths count as
    full lengths; ``hands`` optional (left, right) normalised [T, 21, 2] fans,
    which ride on the wrists.
    """
    size = np.array([width, height], dtype=np.float64)
    px = np.asarray(uv, dtype=np.float64) * size
    ref = np.asarray(reference, dtype=np.float64)[:, :2]
    use = np.asarray(facing_camera, dtype=bool)
    if use.sum() < 20:
        use = np.ones(len(px), dtype=bool)

    def full(lengths):
        return float(np.percentile(lengths[use], BONE_FIT_PERCENTILE))

    def seg(points, a, b):
        return np.linalg.norm(points[..., a, :] - points[..., b, :], axis=-1)

    mid = 0.5 * (px[:, 8] + px[:, 11])
    ref_mid = 0.5 * (ref[8] + ref[11])
    ours = {
        "shoulder_half": full(0.5 * (seg(px, 1, 2) + seg(px, 1, 5))),
        "upper_arm": full(0.5 * (seg(px, 2, 3) + seg(px, 5, 6))),
        "forearm": full(0.5 * (seg(px, 3, 4) + seg(px, 6, 7))),
        "torso": full(np.linalg.norm(px[:, 1] - mid, axis=1)),
        "hip_half": full(0.5 * (np.linalg.norm(px[:, 8] - mid, axis=1)
                                + np.linalg.norm(px[:, 11] - mid, axis=1))),
        "thigh": full(0.5 * (seg(px, 8, 9) + seg(px, 11, 12))),
        "shin": full(0.5 * (seg(px, 9, 10) + seg(px, 12, 13))),
    }
    theirs = {
        "shoulder_half": 0.5 * (seg(ref, 1, 2) + seg(ref, 1, 5)),
        "upper_arm": 0.5 * (seg(ref, 2, 3) + seg(ref, 5, 6)),
        "forearm": 0.5 * (seg(ref, 3, 4) + seg(ref, 6, 7)),
        "torso": float(np.linalg.norm(ref[1] - ref_mid)),
        "hip_half": 0.5 * (float(np.linalg.norm(ref[8] - ref_mid))
                           + float(np.linalg.norm(ref[11] - ref_mid))),
        "thigh": 0.5 * (seg(ref, 8, 9) + seg(ref, 11, 12)),
        "shin": 0.5 * (seg(ref, 9, 10) + seg(ref, 12, 13)),
    }
    factor = {k: float(np.clip(theirs[k] / max(ours[k], 1e-6), *BONE_FIT_CLAMP)) for k in ours}

    out = px.copy()
    neck = px[:, 1]
    shifts = {}
    # Arms: shoulder about the neck, then each segment about its new parent.
    for shoulder, elbow, wrist in ((2, 3, 4), (5, 6, 7)):
        out[:, shoulder] = neck + factor["shoulder_half"] * (px[:, shoulder] - neck)
        out[:, elbow] = out[:, shoulder] + factor["upper_arm"] * (px[:, elbow] - px[:, shoulder])
        out[:, wrist] = out[:, elbow] + factor["forearm"] * (px[:, wrist] - px[:, elbow])
        shifts[wrist] = out[:, wrist] - px[:, wrist]
    # Legs: the pelvis about the neck, hips about the pelvis, then down.
    new_mid = neck + factor["torso"] * (mid - neck)
    for hip, knee, ankle, toe in ((8, 9, 10, 19), (11, 12, 13, 18)):
        out[:, hip] = new_mid + factor["hip_half"] * (px[:, hip] - mid)
        out[:, knee] = out[:, hip] + factor["thigh"] * (px[:, knee] - px[:, hip])
        out[:, ankle] = out[:, knee] + factor["shin"] * (px[:, ankle] - px[:, knee])
        out[:, toe] = out[:, ankle] + (px[:, toe] - px[:, ankle])
    # The neck ceiling.  The excess is closed by RAISING THE SHOULDER GIRDLE
    # (neck point, shoulders, both arm chains and the hand fans) toward the
    # face, not by pulling the face down to the shoulders.  The long necks are
    # frames where SMPL's collars have rotated down; the first version moved
    # the face instead, and the model read "low shoulders, low face" as a
    # slouch -- the blind audit of 2026-09-22 flagged a slight chin tuck on
    # exactly the frames the ceiling touched (741263 f190, 765012 f66) while
    # the necks themselves were right.  Raising the girdle gives the model her
    # own relaxed shoulder-to-face relation and leaves the face where it was.
    ref_neck = float(np.linalg.norm(ref[1] - ref[0]))
    neck_vec = out[:, 0] - neck
    length = np.linalg.norm(neck_vec, axis=1)
    ceiling = NECK_CEILING * ref_neck
    kept = np.where(length > ceiling, ceiling + (length - ceiling) * NECK_TAIL_KEEP, length)
    lift = -neck_vec * ((kept - length) / np.maximum(length, 1e-6))[:, None]   # toward the face
    for slot in (1, 2, 3, 4, 5, 6, 7):
        out[:, slot] = out[:, slot] + lift
    for wrist in (4, 7):
        shifts[wrist] = shifts[wrist] + lift
    # STANDING ON HER FLOOR.  ``fit_to_character`` pins the MEDIAN NECK to hers
    # and scales the median neck-to-ankle span to hers; once every bone has her
    # length, a dancing body with bent knees is shorter than her straight stance,
    # so with the neck pinned the feet floated 20-30 px above the line she stands
    # on (measured 2026-09-22: median ankle 705-715 against her 735), and the
    # render compressed her legs to 0.93x to reach them.  So one vertical shift
    # puts the median of the LOWER ankle -- the supporting foot -- on her ankle
    # line; a bent-kneed frame then carries its head lower, as a dancer's does.
    support = np.maximum(out[:, 10, 1], out[:, 13, 1])
    floor_shift = float(max(ref[10, 1], ref[13, 1]) - np.median(support))
    out[:, :, 1] += floor_shift
    new_hands = None
    if hands is not None:
        # (left fan, right fan) hang off LWri (7) and RWri (4).
        new_hands = tuple(np.asarray(h, dtype=np.float64)
                          + ((shifts[w] + np.array([0.0, floor_shift])) / size)[:, None]
                          for h, w in zip(hands, (7, 4)))
    report = {"factors": factor, "ours_full_px": ours, "character_px": theirs,
              "neck_ceiling_px": ceiling,
              "neck_frames_pulled": float(np.mean(length > ceiling)),
              "neck_max_before": float(length.max() / ref_neck),
              "neck_max_after": float(kept.max() / ref_neck),
              "floor_shift_px": floor_shift}
    return out / size, new_hands, report


def character_landmarks(image_path, width, height):
    """The reference person's AAPose-20 keypoints, at the generation's size.

    CACHED BESIDE THE STILL.  Detecting them costs a ViTPose load (about three
    minutes here) and the answer is a property of the image, not of the clip --
    so a ten-clip batch was paying for it ten times over on one unchanging
    picture.  The cache is keyed on the image's bytes, so editing the still
    invalidates it.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from PIL import Image

    source = pathlib.Path(image_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = source.with_suffix(".landmarks-{}.json".format(digest))
    image = Image.open(image_path)
    if cache.is_file():
        body = np.asarray(json.loads(cache.read_text()), dtype=np.float64)
    else:
        from frame_character import character_pose
        body = character_pose(image)
        cache.write_text(json.dumps(np.asarray(body).tolist()))
    body = body.copy()
    body[:, 0] *= width / image.size[0]
    body[:, 1] *= height / image.size[1]
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--fps", type=float, default=30.0,
                    help="output rate.  The MOTION is resampled to it, not the "
                         "frame sequence -- see the note in main()")
    ap.add_argument("--no-align-heading", dest="align_heading",
                    action="store_false")
    ap.add_argument("--stick-width", type=int, default=-1, metavar="PX",
                    help="body line width; -1 is the vendor's auto rule, which "
                         "at 480x832 degenerates to ONE pixel. The pose picture "
                         "is VAE-encoded at 1/8 scale, so a 1 px limb is 0.125 "
                         "px there: measured, only 1.3%% of the frame is lit and "
                         "the brightest pixel after the downsample is 256/765. "
                         "Thick torso lines survive by overlapping; thin arms do "
                         "not, which is what the operator saw as arms that do "
                         "not follow")
    ap.add_argument("--no-hands", dest="hands", action="store_false",
                    help="leave the hand keypoints empty, which is what the "
                         "first version did and what makes the model melt them")
    ap.add_argument("--no-face-visibility", dest="face_visibility",
                    action="store_false",
                    help="draw every face keypoint on every frame, which tells "
                         "the animator the head faces the lens even when the "
                         "body has turned away")
    ap.add_argument("--face-model", choices=["geometric", "detector", "calibrated"],
                    default="geometric",
                    help="which face points are drawn at which yaw: the "
                         "geometric visibility bands, or the real detector's "
                         "own statistics (see face_confidence_detector)")
    ap.add_argument("--yaw-soft-limit", type=float, default=1.0, metavar="RATIO",
                    help="compress the rendered yaw between 20 and 90 deg by this "
                         "ratio (1 = off; see soft_limit_yaw -- turns look smaller)")
    ap.add_argument("--yaw-fold", type=float, default=0.0, metavar="DEG",
                    help="keep the rendered yaw within DEG of the camera: identity to DEG-25, smooth rise to DEG "
                         "at 90, and past 90 a back view is drawn as the matching front view (see fold_yaw; "
                         "0 = off)")
    ap.add_argument("--palette", choices=["standard", "swapped"], default="standard",
                    help="limb colours: the vendor's OpenPose palette, or the same "
                         "with red and blue exchanged -- which is what every pose "
                         "picture the SteadyDancer authors released carries "
                         "(their pose_align.py runs cv2.cvtColor(BGR2RGB) on a "
                         "picture drawn in RGB); see PALETTE below")
    ap.add_argument("--hands-model", choices=["v1", "v2"], default="v1",
                    help="v1: the original fan (left palm / right back on every "
                         "frame); v2: mirror-consistent thumbs, measured shape "
                         "(see synth_hands_v2)")
    ap.add_argument("--torso", choices=["smpl", "detector", "detector_v2"], default="smpl",
                    help="where the shoulder and hip KEYPOINTS sit: SMPL's own "
                         "joints, or moved to the real detector's convention "
                         "(see retarget_torso)")
    ap.add_argument("--face-fit", choices=["clip", "front"], default="clip",
                    help="how the face's own scale onto the character is measured: 'clip' = distance between the "
                         "ears' clip medians (every render before 2026-09-23); 'front' = median per-frame ear gap over "
                         "camera-facing frames, which does not collapse when the dancer turns away")
    ap.add_argument("--fit", choices=["similarity", "bones"], default="similarity",
                    help="with --match-character: one similarity (neck-to-ankle span "
                         "+ face), or additionally each bone to the character's own "
                         "length with a ceiling on the neck (fit_bones_to_character)")
    ap.add_argument("--match-character", default=None, metavar="IMAGE",
                    help="scale the drawn figure onto the person's box in this "
                         "still, so the pose and the character the animator "
                         "starts from are the same size")
    args = ap.parse_args()

    blob = pickle.load(open(args.motion, "rb"))
    joints = np.asarray(blob["full_pose"], dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1] != 24:
        raise SystemExit("expected [T, 24, 3] joints, got {}".format(joints.shape))

    # RESAMPLE THE MOTION, DO NOT LET THE LOADER DROP FRAMES.  The dance is at
    # 30 fps and the animator reads 16, and 30/16 = 1.875 is not an integer: a
    # nearest-frame pick therefore advances 1 source frame on some output frames
    # and 2 on others, which is a 100% instantaneous speed alternation that
    # switches 82 times in 331 frames -- a speed step four times a SECOND, for
    # the whole clip.  That is the "动画动作突然加速的帧太多了,不稳定流畅" the
    # operator reported, and it is neither the skinning nor the pose content:
    # it is the conversion between them.
    #
    # Measured on 7030793823240424742:clip000, fraction of output frames whose
    # speed jumps by more than 1.5x or drops below 0.67x:
    #     frame-dropped  39.6%   (speed ratio p10 0.52, p90 2.05)
    #     interpolated   20.4%   (speed ratio p10 0.67, p90 1.51)
    # The 20.4% that remains is the dance's own acceleration; the rest was ours.
    if abs(args.fps - SOURCE_FPS) > 1e-6:
        span = np.arange(0.0, len(joints) - 1, SOURCE_FPS / args.fps)
        lower = np.floor(span).astype(int)
        weight = (span - lower)[:, None, None]
        joints = joints[lower] * (1.0 - weight) + joints[lower + 1] * weight
        print("resampled {:g} -> {:g} fps by interpolation: {} frames".format(
            SOURCE_FPS, args.fps, len(joints)))

    # The same rotation ``render_avatar_video`` applies before it places a
    # camera.  Measured on the fixed ten, this clip's motion faces the camera
    # already (1.2 degrees off, 0.0% of frames back-to-camera), so it is close
    # to a no-op here -- but the 3D panel and the 2D animation have to be
    # looking at the same side of the dancer by construction, not by luck.
    if args.align_heading:
        joints, before, after = align_heading(joints)
        print("heading: median facing {:+.3f},{:+.3f} -> {:+.3f},{:+.3f}".format(
            before[0], before[1], after[0], after[1]))
    if args.yaw_soft_limit < 1.0:
        before = float(np.percentile(np.abs(shoulder_yaw(joints)), 95))
        joints = soft_limit_yaw(joints, args.yaw_soft_limit)
        print("yaw soft limit {:g}: p95 |yaw| {:.1f} -> {:.1f} deg".format(
            args.yaw_soft_limit, before, float(np.percentile(np.abs(shoulder_yaw(joints)), 95))))
    if args.yaw_fold:
        before = np.abs(shoulder_yaw(joints))
        joints = fold_yaw(joints, args.yaw_fold)
        print("yaw fold {:g}: frames past 67 deg {:.1%} -> {:.1%}, max |yaw| {:.1f} -> {:.1f}".format(
            args.yaw_fold, float((before > 67).mean()), float((np.abs(shoulder_yaw(joints)) > 67).mean()),
            float(before.max()), float(np.abs(shoulder_yaw(joints)).max())))
    # The drawn body's facing, one value per OUTPUT frame, for comfy_steadydancer --facing-yaw: the text prompt
    # of every context window that contains a turn away from the camera must not say "front view".
    np.save(str(args.out) + ".yaw.npy", shoulder_yaw(joints).astype(np.float32))
    world = to_aapose20(joints, torso="smpl" if args.torso == "detector_v2" else args.torso)
    wrist_shift = None
    if args.torso == "detector_v2":
        world, wrist_shift = retarget_body(world)
    hands = None
    if args.hands:
        hands = synth_hands_v2(joints) if args.hands_model == "v2" else synth_hands(joints, None)
        if wrist_shift is not None:
            # The fans hang off the wrists, which the retarget moved with the arm chain.
            hands = [hand + shift[:, None] for hand, shift in zip(hands, wrist_shift)]
    if hands is None:
        uv = project(world, args.width, args.height)
        hand_uv = None
    else:
        stacked = np.concatenate([world, hands[0], hands[1]], axis=1)
        projected = project(stacked, args.width, args.height, extent=world)
        uv = projected[:, :N_BODY]
        hand_uv = (projected[:, N_BODY:N_BODY + N_HAND],
                   projected[:, N_BODY + N_HAND:N_BODY + 2 * N_HAND])
    if args.face_model == "calibrated":
        uv = face_points_calibrated(uv, joints, args.width, args.height, world)
    fit = None
    if args.match_character:
        reference = character_landmarks(args.match_character, args.width, args.height)
        front = (np.abs(shoulder_yaw(joints)) < BONE_FIT_MAX_YAW_DEG) if args.face_fit == "front" else None
        uv, fit = fit_to_character(uv, reference, args.width, args.height, front=front)
        fit["face_fit"] = args.face_fit
        if not 0.7 <= fit["head_scale"] <= 1.5:
            # a gate that can fail: every head_scale above 1.5 in the 40 beat-round renders was a hair-flying render
            print("WARNING head_scale {:.2f} outside [0.7, 1.5]: the face will be drawn off the neck "
                  "(use --face-fit front)".format(fit["head_scale"]))
        if hand_uv is not None:
            hand_uv = tuple(
                np.stack([fit["character_neck_u"]
                          + (h[:, :, 0] - fit["pose_neck_u"]) * fit["scale"],
                          fit["character_neck_v"]
                          + (h[:, :, 1] - fit["pose_neck_v"]) * fit["scale"]],
                         axis=-1)
                for h in hand_uv)
        print("matched to {}: neck-to-ankle {:.1%} of frame -> {:.1%} "
              "(scale {:.3f})".format(pathlib.Path(args.match_character).name,
                                      fit["pose_neck_ankle"],
                                      fit["character_neck_ankle"], fit["scale"]))
        if args.fit == "bones":
            facing_camera = np.abs(shoulder_yaw(joints)) < BONE_FIT_MAX_YAW_DEG
            uv, hand_uv, bones = fit_bones_to_character(
                uv, reference, args.width, args.height, facing_camera, hand_uv)
            fit["bones"] = bones
            print("bone fit: " + ", ".join("{} {:.2f}".format(k, v)
                                           for k, v in bones["factors"].items())
                  + "; neck ceiling pulls {:.1%} of frames, max {:.2f} -> {:.2f}x; "
                    "floor shift {:+.1f} px".format(
                      bones["neck_frames_pulled"], bones["neck_max_before"],
                      bones["neck_max_after"], bones["floor_shift_px"]))
    face = None
    if args.face_visibility:
        face = {"detector": face_confidence_detector,
                "calibrated": face_confidence_calibrated,
                "geometric": face_confidence}[args.face_model](joints)
    if face is not None:
        drawn = float(np.mean(face > 0.5))
        print("face keypoints drawn on {:.1%} of (frame, point) pairs; "
              "nose/eyes {:.1%}, ears {:.1%}".format(
                  drawn, float(np.mean(face[:, 0] > 0.5)),
                  float(np.mean(face[:, 3:] > 0.5))))
    metas = metas_from_uv(uv, args.width, args.height, face, hand_uv)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".silent.mp4") if args.audio else out
    writer = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "{}x{}".format(args.width, args.height),
         "-framerate", "{:g}".format(args.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(raw)],
        stdin=subprocess.PIPE)
    inside = 0
    for meta in metas:
        canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        frame = draw_aapose_by_meta_new(
            canvas, AAPoseMeta.from_humanapi_meta(meta),
            draw_hand=args.hands, draw_head=True,
            body_stick_width=args.stick_width,
            hand_stick_width=(args.stick_width if args.stick_width > 0 else -1))
        if args.palette == "swapped":
            # PALETTE.  Exchanging R and B maps the right arm's red-orange-yellow
            # onto the hues the standard palette gives LEFT limbs, so under the
            # authors' swapped convention our standard-palette pictures say "left
            # limbs on the image left" -- a BACK view -- on every frame, while
            # the synthesised face says front.  Unproven which one training
            # used (their pose_extra.py and their own ComfyUI example draw the
            # standard palette); this flag exists to test it with one render.
            frame = np.ascontiguousarray(np.asarray(frame)[..., ::-1])
        writer.stdin.write(np.ascontiguousarray(frame).tobytes())
    writer.stdin.close()
    if writer.wait() != 0:
        raise SystemExit("ffmpeg failed writing {}".format(raw))

    if args.audio and pathlib.Path(args.audio).is_file():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                        "-i", args.audio, "-c:v", "copy", "-c:a", "aac",
                        "-shortest", str(out)], check=True)
        raw.unlink()

    inside = float(np.mean((uv >= 0.0) & (uv <= 1.0)))
    top = float(np.nanmedian(uv[:, :, 1].min(axis=1)))
    bottom = float(np.nanmedian(uv[:, :, 1].max(axis=1)))
    if fit is not None:
        pathlib.Path(str(out) + ".fit.json").write_text(json.dumps(fit, indent=2))
    # The keypoints AS DRAWN, in pixels, so the render can be scored against the
    # thing it was actually driven with rather than against a re-derivation.
    np.save(out.parent / "driven.npy",
            (uv * np.array([args.width, args.height])).astype(np.float32))
    print("{} frames {}x{} at {:g} fps -> {}".format(
        len(uv), args.width, args.height, args.fps, out))
    print("  figure spans {:.1%} of frame height; {:.1%} of joint coordinates "
          "inside the frame".format(bottom - top, inside))
    if inside < 0.95:
        raise SystemExit("more than 5% of joints fall outside the frame; the "
                         "pose would be cropped and the model sees a partial body")


if __name__ == "__main__":
    main()
