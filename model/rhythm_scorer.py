"""Score how well a motion excerpt HITS the music it is laid over -- the learned half of label-to-motion.

WHY (operator, 2026-09-23): "卡点舞上卡节奏是重中之重 …… label to motion 必须上有学习能力的模型来优化,
提升泛化性,别一支舞行,另一支舞不行,第一段卡点后面不卡,或者换个 motion 库就不卡了(现在就是)".
Retrieval so far chooses a unit by label, duration, beat COUNT and how well it joins its neighbour; nothing
asks whether this unit's accents land on THIS song's beats.  When the library changed (T1 -> T2, +38
sequences that "卡得轻") the beat-hitting fell with it (worklog 2026-09-22), because the hitting came from
whatever the library happened to contain.

WHAT IT LEARNS.  From the corpus's own (music, motion) pairs: the positive is a dancer's motion over the
music it was danced to; the negatives are (a) the SAME motion shifted by a fraction of a beat -- only the
music-motion alignment separates it from the positive -- and (b) other recordings' motion over that music.
Scoring a candidate therefore asks "does this motion move WITH this music", from the music and the
candidate alone: nothing about the target clip's motion is read (CLAUDE.md 1.6).

THE LEAK THAT SHAPES THE DATA.  The T-line segmentation slid bar lines onto the dancer's settle points
(worklog 2026-09-02), so a bar-aligned crop always starts on a motion settle and a shifted one does not:
a scorer trained on bar crops could name the shift from the motion alone.  Crops are therefore taken at
random offsets, and music and motion of a positive are resampled TOGETHER by a random tempo factor, so
neither the anchor nor the interpolation smoothing distinguishes a positive from a negative.  The
held-out check that this worked: with the music zeroed the shift accuracy must fall to chance.

FEATURES.  Motion enters only as per-frame KINETICS in the body's own frame (speeds of wrists, elbows,
ankles, knees, head, pelvis; the pelvis's vertical velocity; mean joint speed), divided by the crop's
median speed -- scale-free, and blind to pose, so a dancer's style or body cannot identify the song.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 48                 # frames the scorer sees (a 4-beat bar at ~110 BPM / 30 fps is ~65; one bar
                          # of the T grid is 44-55 frames): both music and motion are resampled to it
KIN_JOINTS = (20, 21, 18, 19, 7, 8, 4, 5, 15)   # wrists, elbows, ankles, knees, head
KIN_CHANNELS = len(KIN_JOINTS) + 3              # + pelvis world speed, pelvis vertical velocity, mean speed
MUSIC_CHANNELS = 35


def kinetic_frames(joints, plane=False):
    """[..., T, 24, 3] SMPL joints (z up) -> [..., T, KIN_CHANNELS] per-frame kinetics (metres/frame).

    Body frame = pelvis-relative with the yaw taken from the hip axis, so a turn of the whole body does
    not read as limb motion.  Frame 0 repeats frame 1 (a difference needs two frames)."""
    j = np.asarray(joints, dtype=np.float64)
    hips = j[..., 2, :2] - j[..., 1, :2]
    yaw = -np.arctan2(hips[..., 1], hips[..., 0])
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    off = j - j[..., :1, :]
    body = np.stack([c * off[..., 0] - s * off[..., 1], s * off[..., 0] + c * off[..., 1], off[..., 2]], -1)
    pelvis = np.diff(j[..., 0, :], axis=-2)                                # [..., T-1, 3]
    if plane:
        # WHAT A FRONT-VIEW RENDER CAN SHOW (2026-09-23): the body frame's forward axis (index 1 after the yaw
        # removal) is depth once the dancer faces the camera, so it is dropped; so is the pelvis's travel along
        # world y (the camera axis after face_camera).  Measured: 3D hits that live in depth do not survive to
        # the 2D skin (818: 3D on-beat/half-beat 0.71 for two arms, 0.86 vs 0.78 on the rendered character).
        body = body.copy(); body[..., 1] = 0.0
        pelvis = pelvis.copy(); pelvis[..., 1] = 0.0
    v = np.linalg.norm(np.diff(body, axis=-3), axis=-1)                    # [..., T-1, 24]
    out = np.concatenate([v[..., list(KIN_JOINTS)],
                          np.linalg.norm(pelvis, axis=-1)[..., None],
                          pelvis[..., 2:3],
                          v.mean(-1, keepdims=True)], axis=-1)
    first = out[..., :1, :]
    return np.concatenate([first, out], axis=-2).astype(np.float32)


def resample(x, length):
    """[B, T, C] -> [B, length, C], linear, align_corners (first and last frames kept)."""
    x = torch.as_tensor(x)
    if x.shape[1] == length:
        return x
    return F.interpolate(x.transpose(1, 2), size=length, mode="linear", align_corners=True).transpose(1, 2)


def normalise_kinetics(k):
    """Per crop: divide by the median of the mean-speed channel, so only the SHAPE of the motion's
    rhythm is seen, not its size.  [B, T, C] -> same."""
    k = torch.as_tensor(k)
    scale = k[..., -1].median(dim=1, keepdim=True).values.clamp_min(1e-3).unsqueeze(-1)
    out = k / scale
    out[..., -2] = k[..., -2] / scale[..., 0]            # signed vertical velocity keeps its sign
    return out.clamp(-10, 10)


class _Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x):
        return x + F.gelu(self.norm(self.conv(x)))


class RhythmScorer(nn.Module):
    """score(music [B, CROP, 35], kinetics [B, CROP, KIN]) -> [B].  The two streams are projected and
    stacked as CHANNELS OF THE SAME FRAME before any mixing, so every layer sees both at the same instant --
    alignment is what the architecture can express first."""

    def __init__(self, width=128, blocks=(1, 2, 4, 8, 1, 2, 4, 8), music_channels=None):
        super().__init__()
        # WHICH music channels the scorer may read.  All 35 let it recognise the SONG (20 MFCC + 12 chroma
        # are timbre and harmony): measured 2026-09-23, every variant reached 0.99 on held-in recordings
        # against shifted copies and ~chance on held-out ones -- it memorised pairings.  The rhythm
        # channels alone (onset envelope 0, onset peaks 33, beats 34) say when, not what.
        self.music_channels = list(range(MUSIC_CHANNELS)) if music_channels is None else list(music_channels)
        self.music_in = nn.Conv1d(len(self.music_channels), width // 2, 1)
        self.motion_in = nn.Conv1d(KIN_CHANNELS, width // 2, 1)
        self.blocks = nn.Sequential(*[_Block(width, d) for d in blocks])
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.register_buffer("music_mean", torch.zeros(MUSIC_CHANNELS))
        self.register_buffer("music_std", torch.ones(MUSIC_CHANNELS))

    def forward(self, music, kinetics):
        m = ((music - self.music_mean) / self.music_std)[..., self.music_channels]
        h = torch.cat([self.music_in(m.transpose(1, 2)), self.motion_in(kinetics.transpose(1, 2))], 1)
        h = self.blocks(h)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], -1)).squeeze(-1)


def load_scorer(path, device="cpu"):
    """(model, blob); ``blob.get("kinetics") == "plane"`` means kinetic_frames(..., plane=True) at inference."""
    blob = torch.load(str(path), map_location=device, weights_only=False)
    if blob.get("stage") != "rhythm_scorer":
        raise ValueError("{} is not a rhythm-scorer checkpoint".format(path))
    model = RhythmScorer(**blob.get("arch", {}))
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval(), blob
