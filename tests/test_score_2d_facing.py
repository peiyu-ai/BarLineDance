"""tools/score_2d_facing -- the decision logic, on synthetic inputs.

The instrument's validation lives in the tool's header (153 hand-labelled frames
of the fix7 renders).  These tests pin the parts that turn per-frame features
into the counts the operator reads, where a silent change would move every
number without touching the detector: which side a score falls on, what the
mode filter keeps, and when a front->back->front excursion is one flicker
rather than two flips.  No weights, no video.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import score_2d_facing as sf  # noqa: E402

F, P, B, U = sf.FRONT, sf.PROFILE, sf.BACK, sf.UNKNOWN


# ------------------------------------------------------------------ body
def _driven_frontal(n=1):
    """AAPose-20 pixels of an upright figure facing the camera: the person's
    RIGHT shoulder/hip on the IMAGE LEFT."""
    d = np.zeros((n, 20, 2))
    d[:, sf.A_NOSE] = (240, 200)
    d[:, sf.A_NECK] = (240, 270)
    d[:, sf.A_RSHO] = (190, 275)
    d[:, sf.A_LSHO] = (290, 275)
    d[:, sf.A_RHIP] = (220, 460)
    d[:, sf.A_LHIP] = (260, 460)
    d[:, sf.A_REYE] = (230, 190)
    d[:, sf.A_LEYE] = (250, 190)
    return d


def _vp_from(rsho_x, lsho_x, rhip_x, lhip_x, conf=0.9):
    vp = np.zeros((1, 133, 3))
    vp[0, :, 2] = conf
    vp[0, sf.W_RSHO, 0], vp[0, sf.W_LSHO, 0] = rsho_x, lsho_x
    vp[0, sf.W_RHIP, 0], vp[0, sf.W_LHIP, 0] = rhip_x, lhip_x
    return vp


def test_body_score_sign_follows_the_detected_left_right_order():
    d = _driven_frontal()
    torso = np.linalg.norm(d[0, sf.A_NECK] - 0.5 * (d[0, sf.A_RHIP] + d[0, sf.A_LHIP]))
    front = sf.body_scores(_vp_from(190, 290, 220, 260), d)[0]
    back = sf.body_scores(_vp_from(290, 190, 260, 220), d)[0]   # detector says: seen from behind
    assert front == pytest.approx(140 / torso)
    assert back == pytest.approx(-140 / torso)
    assert sf.classify_body(front) == F and sf.classify_body(back) == B


@pytest.mark.parametrize("score,conf,expect", [
    (0.20, 0.9, F), (0.19, 0.9, P), (0.0, 0.9, P), (-0.19, 0.9, P), (-0.20, 0.9, B),
    (0.8, 0.29, U),            # shoulders/hips not found: say so, do not guess
    (float("nan"), 0.9, U),
])
def test_classify_body_thresholds(score, conf, expect):
    assert sf.classify_body(score, conf) == expect


def test_input_side_from_driven_uses_the_same_rule():
    d = np.concatenate([_driven_frontal(), _driven_frontal()], 0)
    d[1, [sf.A_RSHO, sf.A_LSHO]] = d[1, [sf.A_LSHO, sf.A_RSHO]]    # mirrored drawing
    d[1, [sf.A_RHIP, sf.A_LHIP]] = d[1, [sf.A_LHIP, sf.A_RHIP]]
    assert sf.input_sides_from_driven(d).tolist() == [1, -1]


def test_input_side_from_cues_splits_at_90_degrees():
    cues = {"sh_yaw": [0.0, 89.0, -89.0, 91.0, -175.0, 67.0]}
    assert sf.input_sides_from_cues(cues, 6).tolist() == [1, 1, 1, -1, -1, 1]


# ------------------------------------------------------------------ head
@pytest.mark.parametrize("jaw,dark,colour_ok,expect", [
    (0.71, 0.0, True, F),
    (0.55, 1.5, True, F),      # a spread face wins over dark pixels (bangs, a hand)
    (0.39, 0.2, True, P),
    (0.39, 0.5, True, B),      # face centre under hair
    (0.39, 1.8, False, P),     # colour cue disabled by the calibration check
    (0.10, 0.0, True, B),      # hallucinated jaw collapsed: back, colour or not
    (0.10, 0.0, False, B),
    (float("nan"), 1.0, True, U),
])
def test_classify_head(jaw, dark, colour_ok, expect):
    assert sf.classify_head(jaw, dark, colour_ok) == expect


def _head_image(face_visible):
    """V channel: bright background, a dark hair disk, and (if the face is
    visible) a bright face disk inside it where the face points sit."""
    v = np.full((400, 400), 240, np.uint8)
    yy, xx = np.mgrid[:400, :400]
    v[(xx - 200) ** 2 + (yy - 190) ** 2 < 70 ** 2] = 60
    if face_visible:
        v[(xx - 200) ** 2 + (yy - 215) ** 2 < 42 ** 2] = 235
    return v


def _head_points():
    driven = np.zeros((20, 2))
    driven[sf.A_NOSE] = (200, 220)
    driven[sf.A_NECK] = (200, 300)
    driven[sf.A_REYE] = (185, 205)
    driven[sf.A_LEYE] = (215, 205)
    vp = np.zeros((133, 3))
    vp[sf.W_NOSE, :2] = (200, 220)
    vp[sf.W_LEYE, :2] = (215, 205)
    vp[sf.W_REYE, :2] = (185, 205)
    vp[sf.W_FACE0 + 0, :2] = (160, 215)
    vp[sf.W_FACE0 + 16, :2] = (240, 215)
    return vp, driven


def test_head_features_read_hair_cover_and_jaw_width():
    vp, driven = _head_points()
    face = sf.head_features(_head_image(True), vp, driven)
    hair = sf.head_features(_head_image(False), vp, driven)
    assert face["jaw_w"] == pytest.approx(80 / 80)
    assert face["dark_sum"] < 0.1
    assert hair["dark_sum"] > 1.9
    assert face["dark_above_face"] >= 0.5      # the calibration probe lands on hair


def _feats(above, face, n=10, jaw=0.8):
    return [{"jaw_w": jaw, "dark_above_face": above, "dark_face_vp": face}] * n


def test_colour_calibration_can_fail():
    assert sf.colour_calibration(_feats(0.8, 0.05))["ok"]
    assert not sf.colour_calibration(_feats(0.0, 0.0))["ok"]      # light hair
    assert not sf.colour_calibration(_feats(0.8, 0.6))["ok"]      # dark face centre
    assert not sf.colour_calibration(_feats(0.8, 0.05, n=4))["ok"]  # too few frontal frames
    assert not sf.colour_calibration(_feats(0.8, 0.05, jaw=0.55))["ok"]  # not confidently frontal


# ------------------------------------------------------------------ smoothing and counts
def test_smooth_labels_drops_single_frame_blips_but_keeps_runs():
    assert sf.smooth_labels([F] * 5 + [B] + [F] * 5) == [F] * 11
    run = [F] * 5 + [B] * 3 + [F] * 5
    assert sf.smooth_labels(run) == run


def test_a_long_excursion_is_two_flips():
    c = sf.count_flips([F] * 10 + [B] * 20 + [F] * 10)
    assert (c["flips"], c["flicker"]) == (2, 0)


def test_a_short_excursion_that_returns_is_one_flicker():
    c = sf.count_flips([F] * 10 + [B] * 5 + [F] * 10)
    assert (c["flips"], c["flicker"]) == (0, 1)
    assert c["events"] == [{"type": "flicker", "first": 10, "last": 14, "to": B}]


@pytest.mark.parametrize("length,expect", [(8, (0, 1)), (9, (2, 0))])
def test_flicker_boundary_is_eight_frames(length, expect):
    c = sf.count_flips([F] * 10 + [B] * length + [F] * 10)
    assert (c["flips"], c["flicker"]) == expect


def test_profile_between_two_fronts_is_not_a_flip():
    c = sf.count_flips([F] * 10 + [P] * 12 + [F] * 10)
    assert (c["flips"], c["flicker"], c["label_changes"]) == (0, 0, 2)


def test_a_turn_through_profile_is_one_flip_and_profile_counts_in_the_span():
    c = sf.count_flips([F] * 10 + [P] * 4 + [B] * 30)
    assert (c["flips"], c["flicker"]) == (1, 0)
    # a run's span is first-to-last frame of that side, profile frames inside
    # included: B P B P B spans 5 frames, returns to front -> one flicker
    c = sf.count_flips([F] * 10 + [B, P, B, P, B] + [F] * 10)
    assert (c["flips"], c["flicker"]) == (0, 1)


def test_a_short_run_at_the_end_is_a_flip_not_a_flicker():
    c = sf.count_flips([F] * 10 + [B] * 3)
    assert (c["flips"], c["flicker"]) == (1, 0)


def test_summary_counts_mismatch_and_input_disagreement():
    head = [F, F, B, B, P, F]
    body = [F, B, F, B, B, U]
    inp = np.array([1, 1, 1, 1, 1, -1])
    s = sf.summarize(head, body, inp)
    assert s["mismatch_frames"] == 2
    assert s["mismatch_head_front_body_back"] == 1
    assert s["mismatch_head_back_body_front"] == 1
    assert s["face_visible_on_back_body"] == 2          # F/B and P/B
    assert s["back_of_head_on_nonback_body"] == 1       # B/F
    assert s["body_back_frames"] == 3
    assert s["disagree_with_input"] == 3                # input front, body back x3
    assert s["disagree_render_front_input_back"] == 0   # unknown body is not a disagreement


# ------------------------------------------------------------------ the validation set
def test_hand_labels_cover_what_the_header_claims():
    rows = sf.parse_labels()
    assert len(rows) == 5
    n = sum(len(r) for r in rows.values())
    assert n == 153
    heads = [h for r in rows.values() for _, h, _ in r]
    bodies = [b for r in rows.values() for _, _, b in r]
    for cls in (F, P, B):
        assert heads.count(cls) >= 20 and bodies.count(cls) >= 20
    # positive control: 818 f212-222 front body, f232-242 back head and body
    lab = {f: (h, b) for f, h, b in rows["7618203431723357818__clip000"]}
    for f in range(214, 223, 2):
        assert lab[f][1] == F
    for f in range(232, 243, 2):
        assert lab[f] == (B, B)


def test_blind_and_reexamined_labels_differ_only_where_documented():
    blind, now = sf.parse_labels(blind=True), sf.parse_labels()
    changed = sorted((c[:4], a[0]) for c in now for a, b in zip(now[c], blind[c]) if a != b)
    assert changed == [("7618", 210), ("7637", 224), ("7637", 226), ("7637", 228),
                       ("7637", 230), ("7650", 52)]


def test_confusion_excludes_unsure_and_counts_front_back_errors():
    c = sf.confusion([(F, F), (B, F), (None, B), (P, B), (B, B)])
    assert c["n"] == 4 and c["exact"] == 2 and c["front_back_confusions"] == 1
