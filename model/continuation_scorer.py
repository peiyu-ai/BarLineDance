"""Score whether a candidate unit is a natural NEXT BAR after the motion already placed -- learned continuity.

WHY (operator, 2026-09-23): "卡节奏卡音乐旋律一定要参照上一个motion的动作,不能有冲突感,比如突然加速,突然减速等,
要有上下动作本身配套衔接,不止是接缝", and label-to-motion must use a LEARNED model that generalises.  Retrieval
chooses a unit by label, duration, how well its first frames JOIN the previous unit's last frames (the join band)
and, since today, how well it hits the music (model/rhythm_scorer.py).  Nothing asks whether the unit CONTINUES
what the dancer was doing: the same energy level, the same body part leading, the same phrasing, travel and turn
in a compatible direction.  The feasibility study (runs/ext_20260923/contmodel_*.py) measured that such a thing
exists in real dance and is learnable from hand-made bar statistics: an 8-feature logistic regression separates
the true next bar from other uploads' bars at AUC 0.693 on held-out v5 recordings and 0.651 on the real T2
retrieval pool (same label, duration band).  This module is the learned version; tools/train_continuation_scorer.py
trains it and prints, on the same pairs, the logistic baseline it has to beat.

WHAT IT SEES.  Two spans of per-frame BODY-FRAME features (``continuation_frames``), both resampled at a fixed
number of samples per SLOT, so a bar's phrasing lines up whatever its tempo:
  * the CONTEXT: the last ``CONTEXT_BARS`` slot-lengths of motion before the bar line (at inference, the frames
    build_draft has already placed -- generated motion only; the target clip's motion is never read, CLAUDE.md 1.6);
  * the CANDIDATE: the unit as it will PLAY, i.e. its native range stretched over the slot, so its speeds are
    scaled by native/slot exactly as ``_values_at`` would play them ("突然加速/减速" includes a stretch).
Speeds are divided by the CONTEXT's median speed -- not per crop -- so the energy level of the candidate relative
to what came before survives (a per-crop normalisation would erase exactly the "sudden acceleration" it must see).

THE LEAK THAT SHAPES THE INPUT.  A real next bar is physically continuous with its context at the bar line: the
join cost alone separates it at AUC 0.996 on the T2 library.  A scorer that saw the seam would learn seam contact,
which retrieval already has (the join band) and which the completion model repairs anyway.  So ``SEAM_MASK``
frames on each side of the bar line are blanked (zeroed, with a visibility channel saying so): the scorer must
judge the MACRO continuity.  The trainer reports what an unmasked scorer reaches, so the size of the leak is a
measurement.

WHAT A CHECKPOINT ACTUALLY READS -- measured, and narrower than the design above (review of 2026-09-23; the
trainer's reach check, tools/train_continuation_scorer.py, prints all of it for any checkpoint).  The scorer is NOT a
phrase-level ("上下动作配套衔接, 不止是接缝") model.  It is a previous-bar model whose ranking is mostly the motion
trajectory just outside the seam mask plus dancer/style identity:
  * cont_v1_mask6_s0 (the r25 arms): its effective context is about the last 0.2-1.2 s before the bar line.  With only
    6 context frames visible beyond the mask (0.2 s) it already reaches 68% of its above-chance AUC on the T2 pool
    (0.732 of 0.846); 30 visible frames reach 94%.  The SECOND context bar adds +0.004 (T2, 130/231 recordings,
    P = 0.065), 1% of what it reads; replacing it with another upload's bar costs 0.004.  CONTEXT_BARS = 2 is
    therefore input the model ignores, not phrase context.
  * Of its 0.35 AUC above chance on T2, a context taken from the SAME recording but not adjacent still scores 0.694:
    about 0.19 is identity (same dancer / style), +0.156 is adjacency (183/206 recordings, P = 4e-32).
  * Widening the mask (12 / 16 frames, random 12-24, context-drop, 6 far negatives per example) moves the scorer
    toward identity, not toward the phrase: adjacency beyond identity falls to +0.07 (mask 12) and +0.05 (16), and the
    second bar still adds 1-4% of what the scorer reads in every recipe (cont_p*_far6_*.pt).
  * Trained to predict bar n from bar n-2 ALONE (previous bar hidden; cont_skip1_far6_s0.pt), the scorer gains
    nothing over its identity control: -0.000 on T2 (102/204, P = 1), -0.001 on T1, -0.020 on v5 val.  With these
    features the bar before last says nothing about the next bar beyond who is dancing, so no recipe here can make
    the second bar matter.  A phrase-level signal needs something these per-frame kinematics do not carry.
At inference the identity term cannot pick the previous unit's own dancer (no arm places two consecutive units from
one upload); it can only prefer a similar style.

FEATURES per frame (``FRAME_CHANNELS``), all in the body's own frame (pelvis-relative, yaw from the hip axis), so
where the dancer stands and which way they face never enter:
  0-11   model.rhythm_scorer.kinetic_frames: speeds of wrists, elbows, ankles, knees, head; pelvis speed, pelvis
         vertical velocity, mean joint speed (metres/frame)
  12-38  positions of the same nine joints relative to the pelvis (metres; the decoder's skeleton is fixed, so
         bone lengths carry no dancer identity)
  39     signed yaw rate of the hip axis (radians/frame)
  40-41  pelvis ground velocity in the body frame (metres/frame)
  42     pelvis height above the lower ankle (metres) -- floor-invariant: an absolute height would carry each
         recording's floor estimate, which names the recording
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rhythm_scorer import KIN_CHANNELS, KIN_JOINTS, kinetic_frames

FEATURE_VERSION = "continuation_frames_v1"
POS_JOINTS = KIN_JOINTS
CH_POS = tuple(range(KIN_CHANNELS, KIN_CHANNELS + 3 * len(POS_JOINTS)))
CH_YAW = CH_POS[-1] + 1
CH_GROUND = (CH_YAW + 1, CH_YAW + 2)
CH_HEIGHT = CH_YAW + 3
FRAME_CHANNELS = CH_HEIGHT + 1
MEAN_SPEED = KIN_CHANNELS - 1
# channels that are per-frame RATES: a unit played over more frames than it has moves proportionally slower
RATE_CHANNELS = tuple(range(KIN_CHANNELS)) + (CH_YAW,) + CH_GROUND
# rates divided by the context's median speed (the yaw rate is an angle, kept absolute)
SPEED_CHANNELS = tuple(range(KIN_CHANNELS)) + CH_GROUND
IN_CHANNELS = FRAME_CHANNELS + 2          # + visible flag + signed time to the bar line (in slot lengths)
VIS_CHANNEL = FRAME_CHANNELS
CANDIDATE_SAMPLES = 32
CONTEXT_BARS = 2
CONTEXT_SAMPLES = CANDIDATE_SAMPLES * CONTEXT_BARS
SEAM_MASK = 6                             # frames blanked on EACH side of the bar line


def continuation_frames(joints):
    """[..., T, 24, 3] SMPL joints (z up, 30 fps) -> [..., T, FRAME_CHANNELS] float32 (see the module docstring).

    Rates at frame 0 repeat frame 1 (kinetic_frames' convention)."""
    j = np.asarray(joints, dtype=np.float64)
    kin = kinetic_frames(j).astype(np.float64)
    hips = j[..., 2, :2] - j[..., 1, :2]
    heading = np.arctan2(hips[..., 1], hips[..., 0])
    c, s = np.cos(-heading)[..., None], np.sin(-heading)[..., None]
    off = j[..., list(POS_JOINTS), :] - j[..., :1, :]
    pos = np.stack([c * off[..., 0] - s * off[..., 1], s * off[..., 0] + c * off[..., 1], off[..., 2]], -1)
    yaw_rate = np.diff(np.unwrap(heading, axis=-1), axis=-1)[..., None]            # [..., T-1, 1]
    step = np.diff(j[..., 0, :2], axis=-2)                                           # [..., T-1, 2]
    c1, s1 = c[..., 1:, :], s[..., 1:, :]
    ground = np.concatenate([c1 * step[..., :1] - s1 * step[..., 1:2], s1 * step[..., :1] + c1 * step[..., 1:2]], -1)
    rates = np.concatenate([yaw_rate, ground], -1)
    rates = np.concatenate([rates[..., :1, :], rates], -2)
    height = (j[..., 0, 2] - np.minimum(j[..., 7, 2], j[..., 8, 2]))[..., None]
    out = np.concatenate([kin, pos.reshape(pos.shape[:-2] + (-1,)), rates, height], -1)
    return out.astype(np.float32)


def _gather(frames, x, lo_bound, hi_bound):
    """Linear interpolation of frames [F, C] at float positions x [M, S], each clamped to [lo, hi] of its row
    (``lo_bound``/``hi_bound`` [M] or [M, S])."""
    lo_bound = lo_bound.to(x.dtype)
    hi_bound = hi_bound.to(x.dtype)
    if lo_bound.dim() == 1:
        lo_bound = lo_bound[:, None]
    if hi_bound.dim() == 1:
        hi_bound = hi_bound[:, None]
    xc = torch.minimum(torch.maximum(x, lo_bound), hi_bound)
    lo = xc.floor()
    hi = torch.minimum(lo + 1, hi_bound.expand_as(lo))
    w = (xc - lo).unsqueeze(-1).to(frames.dtype)
    return frames[lo.long()] * (1 - w) + frames[hi.long()] * w


def _as(v, like, device):
    return torch.as_tensor(v, device=device).double().expand_as(like) if torch.as_tensor(v).dim() == 0 \
        else torch.as_tensor(v, device=device).double()


def sample_candidates(frames, starts, natives, slots, gap, samples=CANDIDATE_SAMPLES):
    """Candidate units AS THEY WILL PLAY.  ``frames`` [F, C] per-frame features of any number of recordings or
    library windows laid back to back; a unit is frames[start : start + native] played over ``slots`` frames
    (``native`` may be fractional: a rate-changed replay).  Returns (x [M, S, C] with rate channels scaled by
    (native-1)/(slot-1), visible [M, S] bool -- False within ``gap`` slot frames of the bar line --, time [M, S]
    in slot lengths after the bar line)."""
    device = frames.device
    starts = torch.as_tensor(starts, device=device).double()
    natives = torch.as_tensor(natives, device=device).double()
    slots = _as(slots, starts, device)
    gap = _as(gap, slots, device)
    k = torch.linspace(0.0, 1.0, samples, device=device, dtype=torch.float64)
    x = starts[:, None] + k[None] * (natives[:, None] - 1).clamp_min(0)
    out = _gather(frames, x, starts, (starts + natives - 1).ceil().clamp_min(starts))
    rate = ((natives - 1).clamp_min(1) / (slots - 1).clamp_min(1)).to(frames.dtype)
    out[..., list(RATE_CHANNELS)] = out[..., list(RATE_CHANNELS)] * rate[:, None, None]
    played = k[None] * (slots[:, None] - 1).clamp_min(0)
    visible = played >= gap[:, None] - 1e-9
    return out, visible, (played / slots[:, None].clamp_min(1)).to(frames.dtype)


def sample_context(frames, ends, slots, valid_from, gap, samples=CONTEXT_SAMPLES, rates=None, splice=None):
    """The ``CONTEXT_BARS`` slot-lengths of motion ending at the bar line ``ends`` (exclusive).

    ``rates`` [M] (default 1): native frames per slot frame -- the context PLAYED at a tempo, rate channels
    scaled accordingly (training's shared tempo factor; at inference the draft is already at slot rate, so 1).
    Slot positions p = -span .. -1 map to native ends-1 + (p+1)*rate.  Positions before ``valid_from`` (the
    recording's or the draft's first usable frame) and the last ``gap`` slot frames are invisible.
    ``splice`` = (at [M], other_end [M], other_from [M]) or None: native positions before ``at`` are read from
    ``other_end + (x - at)`` instead (another recording's motion ending at ITS bar line) -- the context then
    contains a cut, as the draft's context always does at inference.  ``at`` = -inf disables it per row.
    Returns (x [M, S, C], visible [M, S], time [M, S] in slot lengths before the bar line, negative)."""
    device = frames.device
    ends = torch.as_tensor(ends, device=device).double()
    slots = _as(slots, ends, device)
    valid_from = _as(valid_from, ends, device)
    gap = _as(gap, ends, device)
    rates = torch.ones_like(ends) if rates is None else _as(rates, ends, device)
    span = CONTEXT_BARS * slots
    k = torch.linspace(0.0, 1.0, samples, device=device, dtype=torch.float64)
    p = -span[:, None] + k[None] * (span[:, None] - 1)
    x = (ends[:, None] - 1) + (p + 1) * rates[:, None]
    visible = (x >= valid_from[:, None] - 1e-9) & (p <= -1 - gap[:, None] + 1e-9)
    if splice is None:
        out = _gather(frames, x, valid_from, ends - 1)
    else:
        at, other_end, other_from = (_as(v, ends, device) for v in splice)
        use = x < at[:, None]
        xo = other_end[:, None] + (x - at[:, None])
        own = _gather(frames, x, torch.maximum(valid_from, torch.where(torch.isfinite(at), at, valid_from)), ends - 1)
        oth = _gather(frames, torch.where(use, xo, other_end[:, None] - 1), other_from,
                      (other_end - 1).clamp_min(other_from))
        out = torch.where(use.unsqueeze(-1), oth, own)
        visible = torch.where(use, (xo >= other_from[:, None] - 1e-9) & (p <= -1 - gap[:, None] + 1e-9), visible)
    out = out.clone()
    out[..., list(RATE_CHANNELS)] = out[..., list(RATE_CHANNELS)] * rates[:, None, None].to(frames.dtype)
    return out, visible, (p / slots[:, None]).to(frames.dtype)


def context_scale(ctx, visible):
    """Median mean-joint speed over the context's visible samples, [B].  NaN if nothing is visible."""
    speed = ctx[..., MEAN_SPEED].masked_fill(~visible, float("nan"))
    return torch.nanmedian(speed, dim=-1).values


def assemble(ctx, ctx_vis, ctx_time, cand, cand_vis, cand_time, scale, in_mean, in_std):
    """-> (context input [B, Sc, IN], candidate input [B, K, Sk, IN]).  ``scale`` [B]: the context's median speed
    (``context_scale``), the SAME for the context and every candidate of an example."""
    scale = scale.clamp_min(1e-3).to(ctx.dtype)

    def one(x, vis, time, s):
        x = x.clone()
        x[..., list(SPEED_CHANNELS)] = x[..., list(SPEED_CHANNELS)] / s[..., None, None]
        x = ((x - in_mean) / in_std).clamp(-10, 10) * vis.unsqueeze(-1).to(x.dtype)
        return torch.cat([x, vis.unsqueeze(-1).to(x.dtype), time.unsqueeze(-1)], -1)

    c = one(ctx, ctx_vis, ctx_time, scale)
    k = cand.shape[1]
    b = one(cand.reshape(-1, *cand.shape[2:]), cand_vis.reshape(-1, cand_vis.shape[-1]),
            cand_time.reshape(-1, cand_time.shape[-1]), scale.repeat_interleave(k))
    return c, b.reshape(cand.shape[0], k, cand.shape[2], -1)


class _Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, width)

    def forward(self, x):
        return x + F.gelu(self.norm(self.conv(x)))


class ContinuationScorer(nn.Module):
    """score(context [B, Sc, IN], candidates [B, K, Sk, IN]) -> [B, K] logits.

    Two dilated-conv streams (context, candidate); every candidate sample then ATTENDS to the context's VISIBLE
    samples (what was the dancer doing that this moment should follow?), and a second conv stack reads the
    candidate together with what it attended to.  Candidates are scored independently of each other, so a pool of
    any size can be ranked, and the ranking of a pool does not depend on which other candidates are in it."""

    def __init__(self, width=96, blocks=(1, 2, 4, 8, 1, 2), post=(1, 2), heads=4, dropout=0.1):
        super().__init__()
        self.ctx_in = nn.Conv1d(IN_CHANNELS, width, 1)
        self.cand_in = nn.Conv1d(IN_CHANNELS, width, 1)
        self.ctx_blocks = nn.Sequential(*[_Block(width, d) for d in blocks])
        self.cand_blocks = nn.Sequential(*[_Block(width, d) for d in blocks])
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=dropout)
        self.attn_norm = nn.LayerNorm(width)
        self.mix = nn.Conv1d(2 * width, width, 1)
        self.post = nn.Sequential(*[_Block(width, d) for d in post])
        self.head = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))
        self.register_buffer("in_mean", torch.zeros(FRAME_CHANNELS))
        self.register_buffer("in_std", torch.ones(FRAME_CHANNELS))

    def encode_context(self, ctx):
        """-> (tokens [B, Sc, W], key padding mask [B, Sc] True = ignore, summary [B, W])."""
        tokens = self.ctx_blocks(self.ctx_in(ctx.transpose(1, 2))).transpose(1, 2)
        visible = ctx[..., VIS_CHANNEL] > 0.5
        # a context with nothing visible (the zeroed-context check) attends to everything rather than to nothing
        visible = torch.where(visible.any(1, keepdim=True), visible, torch.ones_like(visible))
        w = visible.unsqueeze(-1).to(tokens.dtype)
        summary = (tokens * w).sum(1) / w.sum(1).clamp_min(1)
        return tokens, ~visible, summary

    def score(self, encoded, cand):
        tokens, pad, summary = encoded
        b, k = cand.shape[:2]
        h = self.cand_blocks(self.cand_in(cand.reshape(b * k, *cand.shape[2:]).transpose(1, 2))).transpose(1, 2)
        keys = tokens.unsqueeze(1).expand(b, k, *tokens.shape[1:]).reshape(b * k, *tokens.shape[1:])
        mask = pad.unsqueeze(1).expand(b, k, pad.shape[1]).reshape(b * k, pad.shape[1])
        att, _ = self.attn(h, keys, keys, key_padding_mask=mask, need_weights=False)
        u = self.mix(torch.cat([h, self.attn_norm(att)], -1).transpose(1, 2))
        u = self.post(u)
        ctx_summary = summary.unsqueeze(1).expand(b, k, -1).reshape(b * k, -1)
        return self.head(torch.cat([u.mean(-1), u.amax(-1), ctx_summary], -1)).reshape(b, k)

    def forward(self, ctx, cand):
        return self.score(self.encode_context(ctx), cand)


def load_scorer(path, device="cpu", allow_research=False):
    """(model, blob).  ``blob["seam_mask"]`` is the gap the checkpoint was trained and selected at (frames each
    side of the bar line); inference must use the same.  A research checkpoint trained to skip the context's last
    bar (``--ctx-skip-bars``) is refused unless ``allow_research``: inference does not skip it."""
    blob = torch.load(str(path), map_location=device, weights_only=False)
    if blob.get("stage") != "continuation_scorer":
        raise ValueError("{} is not a continuation-scorer checkpoint".format(path))
    if int(blob.get("args", {}).get("ctx_skip_bars", 0) or 0) and not allow_research:
        raise ValueError("{} was trained with --ctx-skip-bars (research only); inference cannot use it".format(path))
    model = ContinuationScorer(**blob.get("arch", {}))
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval(), blob


@torch.no_grad()
def score_pool(model, context_frames, library_frames, starts, natives, slot, gap=SEAM_MASK, valid_from=0,
               device=None, scale_ref=1.0):
    """Inference entry point: score candidates from ``library_frames`` [F, C] (units at ``starts``/``natives``,
    played over ``slot`` frames) as continuations of ``context_frames`` [T, C] -- the frames already placed,
    ending at the bar line (only its last CONTEXT_BARS * slot frames are read; frames before ``valid_from`` are
    invisible).  ``scale_ref``: the checkpoint's speed scale for a context with nothing visible (training's
    fallback; ``blob["scale_ref"]``).  Returns a numpy array [K]."""
    device = device or next(model.parameters()).device
    ctx_frames = torch.as_tensor(np.asarray(context_frames, dtype=np.float32), device=device)
    lib = library_frames if torch.is_tensor(library_frames) else torch.as_tensor(
        np.asarray(library_frames, dtype=np.float32))
    lib = lib.to(device)
    t = len(ctx_frames)
    ctx, cvis, ctime = sample_context(ctx_frames, [t], [slot], [valid_from], gap)
    n = len(starts)
    cand, kvis, ktime = sample_candidates(lib, starts, natives, [slot] * n, gap)
    scale = context_scale(ctx, cvis)
    scale = torch.where(torch.isfinite(scale), scale, torch.full_like(scale, float(scale_ref)))
    c, k = assemble(ctx, cvis, ctime, cand[None], kvis[None], ktime[None], scale, model.in_mean, model.in_std)
    return model(c, k)[0].float().cpu().numpy()
