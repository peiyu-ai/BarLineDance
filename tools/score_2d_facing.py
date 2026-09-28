#!/usr/bin/env python3
"""Front or back: which way the 2D skin actually faces, head and body separately.

WHAT IT ANSWERS.  The operator's top complaint about the SteadyDancer skin is
front/back confusion: a face on a back-turned body, a back of the head on a
front-facing body, and head/body snapping between the two.  The pose we drive
with cannot say which way the figure faces except through its left/right limb
colours, so "did the render turn the wrong way" has to be read off the RENDER.
This reads it, per frame, and counts: back-view frames, head/body mismatch
frames, facing flips, flicker (a flip that comes back within 8 frames), and
frames where the render disagrees with the input pose.

BODY = the detector's left/right order, compared with nothing but the image.
ViTPose-H wholebody (the same ONNX the WanAnimate preprocess node runs) is run
on the rendered frame, cropped from the driven keypoints' box.  A figure seen
from the front has its right shoulder and right hip on the IMAGE LEFT; seen from
behind, on the image right.  So

    body_score = -((RSho.x - LSho.x) + (RHip.x - LHip.x)) / driven torso length

is positive for a front view, negative for a back view, near zero in profile.
This only works if the detector labels back views as back views instead of
mirroring them into a frontal reading.  It does, and that was checked before it
was allowed to judge anything: on 7618..818 f232-f244 (an unmistakable full
back view drawn from a near-frontal input) it reads -0.80..-0.99, while
f214-f222 read +0.73..+0.90.  Thresholds +-0.20 (BODY_FRONT / BODY_BACK).

HEAD = is there a face (ViTPose face landmarks spread out) and is the face
centre covered by hair.  Two cues, because each is blind where the other sees:
  * ``jaw_w`` -- distance between the two ends of ViTPose's jaw contour (face
    landmarks 0 and 16) / driven nose-neck length.  On the labelled frames:
    front 0.45-0.90 (median 0.71), profile 0.29-0.62 (0.39), back of the head
    0.01-0.44 (0.24).  On a back of the head the detector still hallucinates a
    face with confidence 0.4-0.96, so the CONFIDENCES are unusable; the
    GEOMETRY is not -- the hallucinated jaw collapses.  jaw_w >= 0.5 -> front.
  * ``dark_sum`` -- fraction of dark (HSV V < 110) pixels in a 0.35*head-length
    disk at the ViTPose face centre, plus the same at the driven face centre.
    Face skin reads ~0, back of the head ~1 per disk.  dark_sum >= 0.5 -> back;
    a collapsed jaw (jaw_w < 0.2) also -> back.  Otherwise profile.
  jaw_w separates front from the rest but NOT profile from back (they overlap
  0.29-0.44); dark_sum is what separates those.  dark_sum is a COLOUR cue and
  only valid for a character whose hair is darker than her face
  (townfair_fit.png: dark-brown bob).  Every run checks it -- on frames already
  called front by jaw_w alone, the disk above the face centre must be
  hair-dark and the face centre must not be -- and reports
  ``head_colour_calibration.ok = false`` (back calls fall back to jaw-only)
  instead of silently mislabelling a light-haired character.

VALIDATION (the deliverable -- CLAUDE.md 2.1).  153 frames hand-labelled by
looking at zoomed crops of the fix7 renders (output/sample_20260920_2d_fix7),
all five clips, stored below as HAND_LABELS.  Stratified: a systematic
every-32nd-frame sample per clip (unbiased for the common front case) plus
every window where the render looked turned (818 f164-f250, f286-f320; 7637
f56-f80, f216-f240; 7650 f24-f96; 7664 f0-f10, f180-f187; 7676 f176-f182).
Scored as three classes (front = front+3q_front, back = back+3q_back);
rows = my label, columns = tool, per-frame (unsmoothed) decisions:

  HEAD (152; 1 unsure)       front profile back
        front   87              85      2     0
        profile 33               2     25     6
        back    32               0      3    29      exact 91.4%
  All 13 misses are adjacent classes (profile vs 3q); zero front<->back.  The
  one clear miss is 7650 f62 (flying hair: back of head read as profile).
  BODY, blind labels (151; 2 unsure)
        front   92              92      0     0
        profile 20               9      9     2
        back    39               6      0    33      exact 88.7%
  The blind body labels were made from upper-body crops.  Every body
  disagreement was then re-examined on full-body crops (trouser front pockets
  vs seat and back pockets, which way the toes point).  Four of the six "back
  read as front" were MY error: 7637 f224-f230 is a 3q-FRONT body (front slant
  pockets, toes to the right) under a back-of-the-head -- the very head-back /
  body-front mismatch the operator describes.  818 f210 became unsure (back of
  the top, front of the trousers), 7650 f52 profile.  I re-examined agreements
  too (818 f186, f209, f211; 7650 f64): none changed.  Both versions are kept
  (``BODY<OLD`` in HAND_LABELS).  Re-examined:
        front   96              96      0     0
        profile 21              10      9     2
        back    33               0      0    33      exact 92.0%
  So: "back" is trustworthy (33/33 recall, 33/35 precision), front is
  trustworthy, PROFILE bodies read as front about half the time (a profile
  torso keeps the frontal left/right order) -- profile counts are a lower bound.
  After the 5-frame smoothing: head 93.4%, body 88.7% / 92.7%.
  Leave-one-clip-out (thresholds re-picked on four clips from a grid, scored on
  the fifth): head 91.4%, body 86.1% (blind) / 89.3% (re-examined).
  POSITIVE CONTROL 818: f212-f222 front, f232-f242 back, head and body.  Body
  11/11 front and 11/11 back; head 11/11 back and 10/11 front -- f212 reads
  back, and the zoomed crop agrees: f212's head is still mostly back of the
  hair with a ghost face at the edge (labelled 3q_back; front holds for the
  body only).

WHERE IT FAILS.  (1) Profile bodies read front about half the time (above).
(2) Hair in motion (7650 f62) can read profile.  (3) Validated on ONE
character, one arm (fix7); the head colour cue needs dark hair (self-checked).
(4) An arm across the face reads profile.  (5) The body call is one facing for
torso+hips; a render whose top is back-view and trousers front-view (818 f210,
7650 f28) gets whichever the detector weighs more.

COUNTS.  Labels are smoothed with a 5-frame mode filter (0.3 s at 16 fps).  The
side track maps front -> +1, back -> -1 and ignores profile (a profile between
two fronts is not a flip).  A flip is a change of side; a change whose new side
lasts <= 8 frames and then returns is one FLICKER event instead of two flips.
``mismatch`` = head side and body side both set and opposite (strict);
``face_visible_on_back_body`` also counts a PROFILE face on a back body (818
f174-f193: a face looking over the shoulder of a back-turned body, input
facing the camera).  ``disagree_with_input`` = render body side opposite the
input side, from the cues json (|sh_yaw| < 90 -> front) when given, else from
the driven pose's own left/right order (same +-0.20 rule).

SPEED.  ViTPose-H on GPU (capped at 3 GB, gpu_mem_limit) + head features: 14-35 s
per fix7 video (205-353 frames).  The weights are 2.5 GB of ONNX external data;
ONNX Runtime opening them on the NAS mmap-faults them in and took 245 s, so the
first run copies them sequentially to MODEL_CACHE on local /tmp (not NAS --
CLAUDE.md 1.2): measured 9 s with the file in page cache, ~3 min cold.  That is
a once-per-machine cost; ``--keypoints-cache`` skips ViTPose on re-runs.
``--device cpu`` works (keypoints within 0.13 px of the GPU's) but is NOT
within budget: 146 s for 32 frames on this box -- use the GPU.
``--validate`` prints the confusion on this clip's HAND_LABELS.

Usage:
    python3 tools/score_2d_facing.py --video OUT/<clip>.mp4 \\
        --driven OUT/work/<clip>/driven.npy [--cues cues/<clip>.json] \\
        [--out facing.json] [--keypoints-cache vitpose.npy] [--validate]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

# Root of the sibling checkouts (ComfyUI_Wan, Lodge, this repo); set E2E_ROOT.
E2E_ROOT = os.environ.get("E2E_ROOT", "/workspace/e2e")

PREPROCESS = pathlib.Path(
    E2E_ROOT + "/ComfyUI_Wan/custom_nodes/"
    "ComfyUI-WanAnimatePreprocess")
VITPOSE_ONNX = pathlib.Path(
    E2E_ROOT + "/ComfyUI_Wan/models/onnx/"
    "vitpose_h_wholebody_model.onnx")
VITPOSE_DATA = VITPOSE_ONNX.with_name("vitpose_h_wholebody_data.bin")
MODEL_CACHE = pathlib.Path(os.environ.get(
    "SCORE_2D_FACING_MODEL_CACHE", "/tmp/atomicdance_model_cache/vitpose_h_wholebody"))

# AAPose-20 order (render2d/aapose_video.py): the driven keypoints.
A_NOSE, A_NECK, A_RSHO, A_LSHO, A_RHIP, A_LHIP, A_REYE, A_LEYE = 0, 1, 2, 5, 8, 11, 14, 15
# COCO-wholebody-133 order: what ViTPose returns on the render.
W_NOSE, W_LEYE, W_REYE, W_LEAR, W_REAR = 0, 1, 2, 3, 4
W_LSHO, W_RSHO, W_LHIP, W_RHIP = 5, 6, 11, 12
W_FACE0 = 23                 # 68 face landmarks start here; jaw = face[0:17]

BODY_FRONT = 0.20            # body_score >= this -> front
BODY_BACK = -0.20            # body_score <= this -> back
BODY_MIN_CONF = 0.30         # shoulders+hips below this -> unknown
HEAD_JAW_FRONT = 0.50        # jaw_w >= this -> front
HEAD_JAW_COLLAPSED = 0.20    # jaw_w < this -> back (face contour collapsed)
HEAD_DARK_BACK = 0.50        # dark_sum >= this -> back
DARK_V = 110                 # HSV value below this = hair-dark
FACE_DISK = 0.35             # disk radius, in driven nose-neck lengths
SMOOTH_WINDOW = 5
FLICKER_MAX = 8

FRONT, PROFILE, BACK, UNKNOWN = "front", "profile", "back", "unknown"
SIDE = {FRONT: 1, BACK: -1, PROFILE: 0, UNKNOWN: 0}


# ----------------------------------------------------------------- hand labels
# The validation set, kept WITH the criterion it validated (CLAUDE.md 2).
# "frame:HEAD/BODY" per labelled frame of output/sample_20260920_2d_fix7/<clip>.mp4;
# F front, 3F three-quarter front, P profile, 3B three-quarter back, B back,
# U unsure (excluded).  "BODY<OLD" = the blind body label OLD was changed to BODY
# on re-examination of full-body crops (see header); --validate scores both.
# Labelled by looking at zoomed crops of the render only (skin panel), 2026-09-21.
HAND_LABELS = {
    "7618203431723357818__clip000": (
        "0:F/F 32:F/F 64:F/F 96:U/F 128:F/F 160:F/F 164:F/F 168:3F/3F 170:3F/3F "
        "172:3F/P 174:P/3B 176:P/3B 178:P/3B 180:P/3B 182:P/3B 184:P/3B 186:P/B "
        "188:P/B 190:P/3B 192:P/3B 194:3F/F 196:F/F 200:F/F 204:F/F 206:3B/B 207:B/B "
        "208:B/B 209:B/B 210:B/U<B 211:B/F 212:3B/F 213:3F/F 214:F/F 216:F/F 218:F/F "
        "220:F/F 222:3F/F 224:P/3F 226:P/P 228:P/P 230:3B/3B 232:B/B 234:B/B 236:B/B "
        "238:B/B 240:B/B 242:B/B 244:B/B 245:B/B 246:F/F 247:F/F 248:F/F 256:F/F "
        "286:F/F 288:3F/3F 290:P/P 292:P/P 294:3B/3B 296:3B/3B 298:P/3B 300:P/U "
        "302:F/F 310:F/F 320:F/F"),
    "7637514589861220209__clip000": (
        "0:F/F 32:3F/3F 56:3F/3F 60:3F/3F 64:P/P 66:P/P 68:P/P 70:P/P 72:P/P 74:3F/3F "
        "76:3F/3F 78:F/F 80:F/F 84:F/F 90:F/F 100:F/F 121:F/F 128:F/F 160:F/F 192:F/F "
        "216:P/P 220:P/P 222:P/P 224:3B/3F<3B 226:B/3F<3B 228:B/3F<3B 230:B/3F<3B "
        "232:P/P 234:3F/3F 236:3F/3F 238:F/3F 240:F/F 256:F/F 288:F/F"),
    "7650126416710192357__clip000": (
        "0:F/F 24:3F/3F 28:B/U 32:F/F 36:P/3F 40:P/P 44:P/P 48:P/P 52:3B/P<3B 56:B/B "
        "60:B/B 62:B/B 64:B/B 66:B/B 68:B/B 70:B/B 72:3B/P 74:P/P 76:P/P 78:3F/3F "
        "80:3F/3F 84:F/F 90:F/F 128:F/F 192:F/F 256:F/F 320:F/F"),
    "7664973324456613370__clip000": (
        "0:F/F 5:F/F 6:F/F 7:F/F 10:F/F 64:F/F 96:3F/3F 160:F/F 180:3F/3F 183:3F/3F "
        "185:P/3F 187:3F/3F 224:F/F 270:F/F"),
    "7676475940934935409__clip001": (
        "0:F/F 32:F/F 64:3F/3F 96:F/F 128:F/F 133:F/F 135:F/F 153:F/F 156:F/F "
        "176:3F/3F 178:3F/3F 180:3F/3F 182:3F/F 200:F/F"),
}

LABEL_CLASS = {"F": FRONT, "3F": FRONT, "P": PROFILE, "3B": BACK, "B": BACK, "U": None}


def parse_labels(blind: bool = False) -> Dict[str, List[tuple]]:
    """{clip: [(frame, head_class, body_class)]}, classes None where unsure.
    blind=True returns the body labels as first written, before re-examination."""
    out: Dict[str, List[tuple]] = {}
    for clip, text in HAND_LABELS.items():
        rows = []
        for tok in text.split():
            frame, hb = tok.split(":")
            head, body = hb.split("/")
            if "<" in body:
                now, before = body.split("<")
                body = before if blind else now
            rows.append((int(frame), LABEL_CLASS[head], LABEL_CLASS[body]))
        out[clip] = rows
    return out


def confusion(pairs: Sequence[tuple]) -> dict:
    """pairs of (label, prediction) -> 3x3 counts (+unknown column) and rates."""
    classes = [FRONT, PROFILE, BACK]
    table = {a: {b: 0 for b in classes + [UNKNOWN]} for a in classes}
    for lab, pred in pairs:
        if lab is None:
            continue
        table[lab][pred] += 1
    n = sum(sum(r.values()) for r in table.values())
    exact = sum(table[c][c] for c in classes)
    return {"table": table, "n": n, "exact": exact,
            "front_back_confusions": table[FRONT][BACK] + table[BACK][FRONT]}


# ----------------------------------------------------------------- decisions
def classify_body(score: float, conf: float = 1.0) -> str:
    """Render body facing from the detector's left/right order (see header)."""
    if not np.isfinite(score) or conf < BODY_MIN_CONF:
        return UNKNOWN
    if score >= BODY_FRONT:
        return FRONT
    if score <= BODY_BACK:
        return BACK
    return PROFILE


def classify_head(jaw_w: float, dark_sum: float, colour_ok: bool = True) -> str:
    """Render head facing: face spread (jaw_w) first, hair cover (dark_sum) second."""
    if not np.isfinite(jaw_w):
        return UNKNOWN
    if jaw_w >= HEAD_JAW_FRONT:
        return FRONT
    if jaw_w < HEAD_JAW_COLLAPSED:
        return BACK
    if colour_ok and np.isfinite(dark_sum) and dark_sum >= HEAD_DARK_BACK:
        return BACK
    return PROFILE


def smooth_labels(labels: Sequence[str], window: int = SMOOTH_WINDOW) -> List[str]:
    """Centred mode filter; a tie keeps the frame's own label."""
    half = window // 2
    out = []
    for i, own in enumerate(labels):
        seg = labels[max(0, i - half): i + half + 1]
        counts: Dict[str, int] = {}
        for lab in seg:
            counts[lab] = counts.get(lab, 0) + 1
        best = max(counts.values())
        winners = [lab for lab, c in counts.items() if c == best]
        out.append(own if own in winners else winners[0])
    return out


def side_runs(labels: Sequence[str]) -> List[dict]:
    """Runs of front/back side, merging across profile/unknown gaps of the same side.

    Each run: side (+1/-1), first, last (frame indices of its first and last
    frame carrying that side)."""
    runs: List[dict] = []
    for i, lab in enumerate(labels):
        s = SIDE[lab]
        if s == 0:
            continue
        if runs and runs[-1]["side"] == s:
            runs[-1]["last"] = i
        else:
            runs.append({"side": s, "first": i, "last": i})
    return runs


def count_flips(labels: Sequence[str], flicker_max: int = FLICKER_MAX) -> dict:
    """Flips = side changes; a change whose new side lasts <= flicker_max frames
    and then returns to the old side is one flicker event, not two flips."""
    runs = side_runs(labels)
    flips, flicker, events = 0, 0, []
    i = 1
    while i < len(runs):
        prev, cur = runs[i - 1], runs[i]
        span = cur["last"] - cur["first"] + 1
        if (span <= flicker_max and i + 1 < len(runs)
                and runs[i + 1]["side"] == prev["side"]):
            flicker += 1
            events.append({"type": "flicker", "first": cur["first"], "last": cur["last"],
                           "to": FRONT if cur["side"] > 0 else BACK})
            i += 2
            continue
        flips += 1
        events.append({"type": "flip", "at": cur["first"],
                       "to": FRONT if cur["side"] > 0 else BACK})
        i += 1
    changes = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    return {"flips": flips, "flicker": flicker, "label_changes": changes, "events": events}


def input_sides_from_cues(cues: dict, n: int) -> np.ndarray:
    yaw = np.asarray(cues["sh_yaw"], dtype=float)[:n]
    side = np.where(np.abs(yaw) < 90.0, 1, -1)
    return side


def body_scores(vp: np.ndarray, driven: np.ndarray) -> np.ndarray:
    """-(R-L shoulder x + R-L hip x) / driven torso length; + = front."""
    torso = np.linalg.norm(driven[:, A_NECK] - 0.5 * (driven[:, A_RHIP] + driven[:, A_LHIP]), axis=1)
    rl = (vp[:, W_RSHO, 0] - vp[:, W_LSHO, 0]) + (vp[:, W_RHIP, 0] - vp[:, W_LHIP, 0])
    return -rl / np.maximum(torso, 1e-6)


def input_sides_from_driven(driven: np.ndarray) -> np.ndarray:
    torso = np.linalg.norm(driven[:, A_NECK] - 0.5 * (driven[:, A_RHIP] + driven[:, A_LHIP]), axis=1)
    rl = (driven[:, A_RSHO, 0] - driven[:, A_LSHO, 0]) + (driven[:, A_RHIP, 0] - driven[:, A_LHIP, 0])
    s = -rl / np.maximum(torso, 1e-6)
    return np.where(s >= BODY_FRONT, 1, np.where(s <= BODY_BACK, -1, 0))


def summarize(head: Sequence[str], body: Sequence[str], input_side: np.ndarray) -> dict:
    hs = np.array([SIDE[x] for x in head])
    bs = np.array([SIDE[x] for x in body])
    n = min(len(hs), len(bs), len(input_side))
    hs, bs, ins = hs[:n], bs[:n], np.asarray(input_side)[:n]
    return {
        "frames": int(n),
        "body_back_frames": int((bs == -1).sum()),
        "head_back_frames": int((hs == -1).sum()),
        "body_profile_frames": int(sum(1 for x in body[:n] if x == PROFILE)),
        "head_profile_frames": int(sum(1 for x in head[:n] if x == PROFILE)),
        "mismatch_frames": int(((hs * bs) == -1).sum()),
        "mismatch_head_front_body_back": int(((hs == 1) & (bs == -1)).sum()),
        "mismatch_head_back_body_front": int(((hs == -1) & (bs == 1)).sum()),
        # softer: a face shown at all (front OR profile) on a back-view body --
        # 818 f174-f193 is a profile face looking over a back-turned body while
        # the input faces the camera -- and hair on a front/profile body.
        "face_visible_on_back_body": sum(1 for i in range(n)
                                         if head[i] in (FRONT, PROFILE) and body[i] == BACK),
        "back_of_head_on_nonback_body": sum(1 for i in range(n)
                                            if head[i] == BACK and body[i] in (FRONT, PROFILE)),
        "disagree_with_input": int(((bs * ins) == -1).sum()),
        "disagree_render_back_input_front": int(((bs == -1) & (ins == 1)).sum()),
        "disagree_render_front_input_back": int(((bs == 1) & (ins == -1)).sum()),
        "head_disagree_with_input": int(((hs * ins) == -1).sum()),
        "body_flips": count_flips(body[:n]),
        "head_flips": count_flips(head[:n]),
    }


# ----------------------------------------------------------------- features
def _dark_disk(value: np.ndarray, centre: np.ndarray, radius: float) -> float:
    h, w = value.shape
    x0, x1 = int(max(0, centre[0] - radius)), int(min(w, centre[0] + radius + 1))
    y0, y1 = int(max(0, centre[1] - radius)), int(min(h, centre[1] + radius + 1))
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= radius * radius
    return float((value[y0:y1, x0:x1][inside] < DARK_V).mean())


def head_features(value: np.ndarray, vp: np.ndarray, driven: np.ndarray) -> dict:
    """value: HSV V channel of the render; vp: [133,3] ViTPose; driven: [20,2]."""
    hl = float(np.linalg.norm(driven[A_NOSE] - driven[A_NECK]))
    face = vp[W_FACE0:W_FACE0 + 68]
    # jaw-contour END points (face[0], face[16]: the two jaw angles below the
    # ears).  Not max-min over the contour: in profile the contour wraps round
    # the chin and its x-extent stays wide (profile->front 7/33 vs 2/33).
    jaw_w = float(abs(face[16, 0] - face[0, 0]) / max(hl, 1e-6))
    c_vp = vp[[W_NOSE, W_LEYE, W_REYE], :2].mean(0)
    c_dr = driven[[A_NOSE, A_REYE, A_LEYE]].mean(0)
    d_vp = _dark_disk(value, c_vp, FACE_DISK * hl)
    d_dr = _dark_disk(value, c_dr, FACE_DISK * hl)
    above = _dark_disk(value, c_vp - np.array([0.0, 0.55 * hl]), 0.2 * hl)
    return {"jaw_w": jaw_w, "dark_sum": float(np.nansum([d_vp, d_dr])),
            "dark_face_vp": d_vp, "dark_face_driven": d_dr, "dark_above_face": above}


def colour_calibration(feats: List[dict]) -> dict:
    """Is this a dark-haired character?  On frames already called front by the
    jaw alone, the disk above the face must be hair-dark and the face not."""
    front = [f for f in feats if f["jaw_w"] >= HEAD_JAW_FRONT + 0.1]
    if len(front) < 5:
        return {"ok": False, "reason": "fewer than 5 confidently frontal frames", "n": len(front)}
    above = float(np.nanmedian([f["dark_above_face"] for f in front]))
    face = float(np.nanmedian([f["dark_face_vp"] for f in front]))
    ok = above >= 0.5 and face <= 0.2
    return {"ok": bool(ok), "n": len(front), "median_dark_above_face": above,
            "median_dark_face": face,
            "reason": "" if ok else "hair not darker than face on frontal frames"}


# ----------------------------------------------------------------- ViTPose
def _local_model() -> pathlib.Path:
    """Sequentially copy the ONNX + external data to local disk once.  Opening it
    straight off the NAS mmap-faults 2.5 GB and took 245 s; the copy takes ~9 s."""
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    for src in (VITPOSE_ONNX, VITPOSE_DATA):
        dst = MODEL_CACHE / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            tmp = dst.with_suffix(dst.suffix + ".part")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
    return MODEL_CACHE / VITPOSE_ONNX.name


def _pose2d_utils():
    path = PREPROCESS / "pose_utils" / "pose2d_utils.py"
    spec = importlib.util.spec_from_file_location("wan_pose2d_utils", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_vitpose(frames: List[np.ndarray], driven: np.ndarray, device: str = "cuda",
                batch: int = 16) -> np.ndarray:
    """[T,133,3] keypoints (x, y, conf) on the rendered frames, crop from the
    driven box, preprocessing as WanAnimatePreprocess PoseAndFaceDetection."""
    import cv2
    import onnxruntime as ort
    p2d = _pose2d_utils()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if device == "cuda":
        providers = [("CUDAExecutionProvider", {"gpu_mem_limit": 3 * 1024 ** 3,
                                                "arena_extend_strategy": "kSameAsRequested"}),
                     "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(_local_model()), so, providers=providers)
    name = sess.get_inputs()[0].name
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    out = []
    for b0 in range(0, len(frames), batch):
        xs, cs, ss = [], [], []
        for i in range(b0, min(len(frames), b0 + batch)):
            img = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            k = driven[i]
            bb = [k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()]
            pad = 0.1 * (bb[3] - bb[1])
            bb = [bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad]
            c, s = p2d.bbox_from_detector(bb, (256, 192), rescale=1.25)
            crop = p2d.crop(img, c, s, (256, 192))[0]
            xs.append(((crop - mean) / std).transpose(2, 0, 1).astype(np.float32))
            cs.append(np.asarray(c)); ss.append(np.asarray(s))
        heat = sess.run([], {name: np.stack(xs)})[0]
        pts, prob = p2d.keypoints_from_heatmaps(heatmaps=heat, center=np.stack(cs),
                                                scale=np.stack(ss) * 200, unbiased=True,
                                                use_udp=False)
        out.append(np.concatenate([pts, prob], 2))
    return np.concatenate(out, 0).astype(np.float32)


def read_frames(video: str, limit: Optional[int] = None) -> List[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(video)
    frames = []
    while limit is None or len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames


# ----------------------------------------------------------------- driver
def score_video(video: str, driven_path: str, cues_path: Optional[str] = None,
                device: str = "cuda", keypoints_cache: Optional[str] = None) -> dict:
    import cv2
    t0 = time.time()
    driven = np.load(driven_path).astype(np.float64)
    frames = read_frames(video, limit=len(driven))
    n = len(frames)
    if n == 0:
        raise SystemExit("no frames read from {}".format(video))
    driven = driven[:n]
    vp = None
    if keypoints_cache and os.path.exists(keypoints_cache):
        vp = np.load(keypoints_cache)
        if len(vp) < n:
            vp = None
    if vp is None:
        vp = run_vitpose(frames, driven, device=device)
        if keypoints_cache:
            np.save(keypoints_cache, vp)
    vp = vp[:n].astype(np.float64)

    bscore = body_scores(vp, driven)
    bconf = vp[:, [W_LSHO, W_RSHO, W_LHIP, W_RHIP], 2].min(1)
    feats = [head_features(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 2], vp[i], driven[i])
             for i, f in enumerate(frames)]
    calib = colour_calibration(feats)
    head_raw = [classify_head(f["jaw_w"], f["dark_sum"], calib["ok"]) for f in feats]
    body_raw = [classify_body(float(bscore[i]), float(bconf[i])) for i in range(n)]
    head = smooth_labels(head_raw)
    body = smooth_labels(body_raw)

    if cues_path:
        cues = json.load(open(cues_path))
        input_side = input_sides_from_cues(cues, n)
        input_source = "cues sh_yaw (|yaw|<90 -> front)"
    else:
        input_side = input_sides_from_driven(driven)
        input_source = "driven left/right order (+-{:.2f})".format(BODY_FRONT)
    summary = summarize(head, body, input_side)
    summary.update({"video": str(video), "input_side_source": input_source,
                    "head_colour_calibration": calib, "seconds": round(time.time() - t0, 1)})
    per_frame = [{"frame": i, "head": head[i], "body": body[i],
                  "head_raw": head_raw[i], "body_raw": body_raw[i],
                  "body_score": round(float(bscore[i]), 3), "body_conf": round(float(bconf[i]), 3),
                  "jaw_w": round(feats[i]["jaw_w"], 3), "dark_sum": round(feats[i]["dark_sum"], 3),
                  "input_side": int(input_side[i]) if i < len(input_side) else 0}
                 for i in range(n)]
    return {"summary": summary, "frames": per_frame}


def _track(labels: Sequence[str]) -> str:
    return "".join({FRONT: ".", PROFILE: "|", BACK: "#", UNKNOWN: "?"}[x] for x in labels)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--video", required=True, help="rendered 2D skin video")
    ap.add_argument("--driven", required=True, help="driven.npy [T,20,2] AAPose pixels")
    ap.add_argument("--cues", help="optional cues json with sh_yaw per pose frame")
    ap.add_argument("--out", help="write per-frame + summary json here")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--keypoints-cache", help="npy to reuse/save ViTPose keypoints")
    ap.add_argument("--validate", action="store_true",
                    help="also score this clip's HAND_LABELS frames (clip = video stem)")
    args = ap.parse_args(argv)
    res = score_video(args.video, args.driven, args.cues, args.device, args.keypoints_cache)
    s = res["summary"]
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
    body = [f["body"] for f in res["frames"]]
    head = [f["head"] for f in res["frames"]]
    print("{}  frames {}  ({} s)".format(pathlib.Path(args.video).name, s["frames"], s["seconds"]))
    print("  body back {}  head back {}  mismatch {} (face-on-back-body {}, back-of-head-on-front-body {})".format(
        s["body_back_frames"], s["head_back_frames"], s["mismatch_frames"],
        s["mismatch_head_front_body_back"], s["mismatch_head_back_body_front"]))
    print("  soft: face (front|profile) on back body {}, back of head on front|profile body {}".format(
        s["face_visible_on_back_body"], s["back_of_head_on_nonback_body"]))
    print("  body flips {} flicker {} | head flips {} flicker {}".format(
        s["body_flips"]["flips"], s["body_flips"]["flicker"],
        s["head_flips"]["flips"], s["head_flips"]["flicker"]))
    print("  disagree with input {} (render back / input front {}, render front / input back {}) [{}]".format(
        s["disagree_with_input"], s["disagree_render_back_input_front"],
        s["disagree_render_front_input_back"], s["input_side_source"]))
    cal = s["head_colour_calibration"]
    if not cal["ok"]:
        print("  WARNING head colour cue uncalibrated: {} -- head back calls are jaw-only".format(cal["reason"]))
    if args.validate:
        clip = pathlib.Path(args.video).stem
        labels = parse_labels(blind=False).get(clip)
        if not labels:
            print("  --validate: no hand labels for {}".format(clip))
        else:
            fr = res["frames"]
            for part, idx in (("head", 1), ("body", 2)):
                for key in ("_raw", ""):
                    pairs = [(row[idx], fr[row[0]][part + key]) for row in labels if row[0] < len(fr)]
                    c = confusion(pairs)
                    print("  validate {} {:6s}: exact {}/{}  front<->back {}  table {}".format(
                        part, key.strip("_") or "smooth", c["exact"], c["n"],
                        c["front_back_confusions"],
                        {a: [v[b] for b in (FRONT, PROFILE, BACK)] for a, v in c["table"].items()}))
    print("  legend . front  | profile  # back")
    for a in range(0, len(body), 80):
        print("  {:4d} B {}".format(a, _track(body[a:a + 80])))
        print("       H {}".format(_track(head[a:a + 80])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
