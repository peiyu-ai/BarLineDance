"""render2d/aapose_video.fit_bones_to_character -- the drawn pose takes the character's proportions.

Operator 2026-09-22: "2d pose 进蒙皮渲染时，要拉到和 image 首帧 figure 到同一尺寸，这样
不会出现拉伸脖子，扭曲身体".  One similarity matched only the neck-to-ankle span;
the torso stayed 1.23x hers, the forearm 1.19x, the hips 1.17x, the thigh 0.92x.
These tests pin what the bone fit promises and nothing more: lengths become the
character's, directions and timing do not change, and the neck's long tail --
not its median -- is pulled back.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from render2d.aapose_video import (  # noqa: E402
    fit_bones_to_character, NECK_CEILING, NECK_TAIL_KEEP)

W, H = 480, 832


def character():
    """A frontal AAPose-20 figure in pixels (short torso, long legs, like townfair)."""
    p = np.zeros((20, 2))
    p[1] = (240, 300)                      # neck
    p[0] = (240, 250)                      # nose
    p[14], p[15], p[16], p[17] = (228, 244), (252, 244), (216, 250), (264, 250)
    p[2], p[5] = (206, 300), (274, 300)    # shoulders
    p[3], p[6] = (200, 360), (280, 360)    # elbows
    p[4], p[7] = (198, 410), (282, 410)    # wrists
    p[8], p[11] = (218, 400), (262, 400)   # hips
    p[9], p[12] = (216, 500), (264, 500)   # knees
    p[10], p[13] = (214, 600), (266, 600)  # ankles
    p[19], p[18] = (210, 620), (270, 620)  # toes
    return p


def dancer(frames=60):
    """The same figure with a longer torso and forearms, and a neck that grows
    on the last few frames (the face drawn higher above the shoulders), jittered
    so percentiles mean something."""
    rng = np.random.default_rng(0)
    base = character().copy()
    base[[8, 11, 9, 12, 10, 13, 18, 19], 1] += 60        # torso 60 px longer
    base[[4, 7], 1] += 15                                # forearms longer
    seq = np.repeat(base[None], frames, axis=0) + rng.normal(0, 0.3, (frames, 20, 2))
    seq[-5:, [0, 14, 15, 16, 17], 1] -= 25               # a swan neck on 5 frames
    return seq


def lengths(p):
    mid = 0.5 * (p[..., 8, :] + p[..., 11, :])
    return {"torso": np.linalg.norm(p[..., 1, :] - mid, axis=-1),
            "forearm": np.linalg.norm(p[..., 3, :] - p[..., 4, :], axis=-1),
            "thigh": np.linalg.norm(p[..., 8, :] - p[..., 9, :], axis=-1)}


def test_the_characters_own_pose_passes_through_unchanged():
    ref = character()
    seq = np.repeat(ref[None], 30, axis=0)
    out, _, report = fit_bones_to_character(seq / [W, H], ref, W, H, np.ones(30, bool))
    assert np.allclose(out * [W, H], seq, atol=1e-6)
    assert all(abs(f - 1.0) < 1e-9 for f in report["factors"].values())
    assert report["neck_frames_pulled"] == 0.0
    assert abs(report["floor_shift_px"]) < 1e-9


def test_a_bent_kneed_dancer_stands_on_her_floor():
    """With her bone lengths a bent-kneed pose is shorter than her stance; the
    supporting ankle must still land on her ankle line, not float above it."""
    ref = character()
    seq = np.repeat(ref[None], 30, axis=0).copy()
    # Bend both knees: ankles come up 40 px, knees forward (in the image, sideways).
    seq[:, [10, 13], 1] -= 40
    seq[:, [19, 18], 1] -= 40
    out, _, report = fit_bones_to_character(seq / [W, H], ref, W, H, np.ones(30, bool))
    out = out * [W, H]
    support = np.maximum(out[:, 10, 1], out[:, 13, 1])
    assert abs(np.median(support) - max(ref[10, 1], ref[13, 1])) < 1e-6
    assert report["floor_shift_px"] > 0


def test_bones_take_the_characters_lengths_and_keep_their_directions():
    ref = character()
    seq = dancer()
    out, _, report = fit_bones_to_character(seq / [W, H], ref, W, H, np.ones(len(seq), bool))
    out = out * [W, H]
    ref_len = lengths(ref)
    for bone in ("torso", "forearm", "thigh"):
        assert abs(np.median(lengths(out)[bone]) / ref_len[bone] - 1.0) < 0.02, bone
    assert report["factors"]["torso"] < 0.9 and report["factors"]["forearm"] < 0.9
    # Directions: the torso vector keeps its angle on every frame.
    mid_in = 0.5 * (seq[:, 8] + seq[:, 11]) - seq[:, 1]
    mid_out = 0.5 * (out[:, 8] + out[:, 11]) - out[:, 1]
    cos = np.sum(mid_in * mid_out, axis=1) / (np.linalg.norm(mid_in, axis=1) * np.linalg.norm(mid_out, axis=1))
    assert cos.min() > 0.9999


def test_the_neck_tail_is_pulled_back_and_the_face_moves_as_one_piece():
    ref = character()
    seq = dancer()
    out, _, report = fit_bones_to_character(seq / [W, H], ref, W, H, np.ones(len(seq), bool))
    out = out * [W, H]
    out[:, :, 1] -= report["floor_shift_px"]   # compare shapes, not placement
    ref_neck = np.linalg.norm(ref[1] - ref[0])
    before = np.linalg.norm(seq[:, 0] - seq[:, 1], axis=1) / ref_neck
    after = np.linalg.norm(out[:, 0] - out[:, 1], axis=1) / ref_neck
    # The ordinary frames are untouched ...
    assert np.allclose(after[:-5], before[:-5], atol=1e-6)
    # ... and the swan-neck frames are brought back to the ceiling plus a
    # quarter of their excess, not flattened.
    expected = NECK_CEILING + (before[-5:] - NECK_CEILING) * NECK_TAIL_KEEP
    assert np.allclose(after[-5:], expected, atol=1e-6)
    # The face keeps its shape: the eye-to-ear offsets do not change.
    for a, b in ((14, 16), (15, 17), (0, 14)):
        assert np.allclose(out[:, a] - out[:, b], seq[:, a] - seq[:, b], atol=1e-6)
    # And it is the SHOULDERS that move, not the face: pulling the face down to
    # dropped shoulders read as a chin tuck (blind audit 2026-09-22).  So on the
    # swan-neck frames the nose stays where it was relative to the ordinary
    # frames, and the shoulder line rises by the excess.
    nose_rise = seq[-5:, 0, 1] - out[-5:, 0, 1]
    assert np.allclose(nose_rise, np.median(seq[:-5, 0, 1] - out[:-5, 0, 1]), atol=1e-6)
    shoulder_rise = (seq[-5:, 2, 1] - out[-5:, 2, 1]) - np.median(seq[:-5, 2, 1] - out[:-5, 2, 1])
    assert (shoulder_rise > 5).all()


def test_the_hand_fans_ride_on_the_moved_wrists():
    ref = character()
    seq = dancer()
    left = np.repeat(seq[:, 7:8], 21, axis=1) + 5.0      # fans offset from the wrists
    right = np.repeat(seq[:, 4:5], 21, axis=1) + 5.0
    out, hands, _ = fit_bones_to_character(seq / [W, H], ref, W, H, np.ones(len(seq), bool),
                                           (left / [W, H], right / [W, H]))
    # (the floor shift moves wrists and fans together, so the offset is unchanged)
    out = out * [W, H]
    assert np.allclose(hands[0] * [W, H] - out[:, 7:8], 5.0, atol=1e-6)
    assert np.allclose(hands[1] * [W, H] - out[:, 4:5], 5.0, atol=1e-6)
