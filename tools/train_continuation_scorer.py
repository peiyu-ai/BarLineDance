#!/usr/bin/env python3
"""Train ``model.continuation_scorer`` on real consecutive bars of the v5 wild corpus, and run the checks that decide
whether it may be believed.  See the model's docstring for WHY and for the leak that shapes the input.

DATA (tools/continuation_scorer_data.py builds the caches).  v5 recordings re-joined from release_v3_timebase
windows, decoded to joints once, cut into 4-beat bars exactly as runs/ext_20260923/contmodel_build_v5.py cut them.
EXCLUDED from training (``--exclude``): the T eval clips, their uploads and every same-song recording
(runs/ext_20260923/rhythm_scorer_exclusions.json), the 孤单北半球 demo uploads (7609672652890196474,
7606328801027322865) and, with ``--exclude-libraries`` (default on), every upload of the T1 and T2 libraries --
so the real-pool checks below are on dancers the scorer never saw.

ONE EXAMPLE = a context and 16 candidates for the slot after it:
  * the POSITIVE: the recording's own next bar;
  * 8 RAND: other uploads' bars inside the duration band of the slot (what retrieval's pool is made of);
  * 3 HARD: the lowest seam-join cost among 32 more band bars (pose + 4 x velocity at the bar line, the
    contmodel_feats.pair ``join`` analogue) -- the candidates the join band keeps;
  * 2 RATE: the true next bar replayed at 0.8x / 1.25x ("突然加速/减速" with the right content);
  * 2 FAR: the same recording, >= 3 bars away (same dancer and song, wrong place).
A SHARED TEMPO FACTOR f ~ U[0.87, 1.15] plays the whole example (context and every candidate) over slot = f x L,
so the positive is not always the one candidate played at rate 1.  In ~30% of examples the context is SPLICED --
before the previous bar line it comes from another upload's bar -- because at inference the context is the draft,
which is cut at every bar line; in ~15% only the previous bar is visible (the draft's second slot).  InfoNCE over
the 16.  The seam mask (``--seam-mask``, 6 frames each side) blanks the bar line on both sides.

CHECKS (printed; stored in the checkpoint):
  v5 val        held-out v5 recordings, AUC of the true next bar against rand / far / rate negatives at slot = L --
                the protocol of runs/ext_20260923/contmodel_v5_stats.py -- and THE SAME ROWS scored by that script's
                8-feature logistic regression (its fitted weights, v5_stats_gap6_rel.json): baseline 0.693 there;
  T2 / T1 pool  the real retrieval pools: T library consecutive bars, true next bar against same-label other-upload
                bars in the duration band (rand) and the join-band 0.35 subset of them (join), plus same-recording far;
                logistic baseline on T2 0.651 / 0.658;
  ctx0          the same with the context zeroed (nothing visible, a fixed speed scale): must fall to ~chance, or the
                scorer ranks candidates by something other than what came before;
  paired        per recording, model AUC minus logistic AUC, sign test;
  adjacency     the same rows scored with the real context and with a same-recording NON-adjacent context (the
                identity control, ctxfar), per recording: what the scorer knows about the NEXT bar beyond who is
                dancing;
  reach         (added after the 2026-09-23 review found v1 reads mostly the frames just outside the mask) the same
                rows with less of the context visible: previous bar only, the earlier bar spliced from another
                upload (the draft's context always is), only 6 / 10 / 30 frames visible beyond the mask, and the
                last 10 / 30 frames beyond the mask hidden; per recording, full minus previous-bar-only (the second
                bar's gain, sign test) and the share of the full AUC reached with 6 visible frames;
  VERDICT       ``reach_verdict``: PHRASE-LEVEL only if the second bar's gain is significant on every T pool AND at
                least 10% of what the scorer reads; otherwise a PREVIOUS-BAR SCORER with its measured reach.
The checkpoint is chosen on v5 val rand AUC (``--select rand+far``: the mean with far); training stops when it has
not improved for ``--patience`` evals.

RECIPE OPTIONS for a wider-reaching scorer (all off by default = the v1 recipe, byte-for-byte the same sampling):
``--train-gap-max`` (a random mask per example), ``--ctx-drop-max`` (hide a further random tail of the context),
``--k-far`` (more same-recording negatives), ``--ctx-skip-bars 1`` (research: predict bar n from bar n-2 alone).
Measured 2026-09-23 (model/continuation_scorer.py's docstring): none of them makes the second bar matter.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
EXT = REPO / "runs" / "ext_20260923"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EXT))

from model.continuation_scorer import (CH_POS, CONTEXT_BARS, FEATURE_VERSION, FRAME_CHANNELS,  # noqa: E402
                                       ContinuationScorer, assemble, context_scale, sample_candidates,
                                       sample_context)
from tools.continuation_scorer_data import LIBS, load  # noqa: E402

DEMO_UPLOADS = ("7609672652890196474", "7606328801027322865")
TEMPO = (0.87, 1.15)
K_RAND, K_HARD, K_RATE, K_FAR = 8, 3, 2, 2
HARD_POOL = 32
RATES = (0.8, 1.25)
LR_JSON = "/cache/atomicdance-assets/runs/ext_20260923_scratch/contmodel/v5_stats_gap{}{}.json"


def sign_p(d):
    d = [x for x in d if x == x and abs(x) > 1e-12]
    n = len(d)
    w = sum(x > 0 for x in d)
    k = min(w, n - w)
    return w, n, (min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0)


def slack_of(length):
    return np.maximum(2.0, 0.15 * np.asarray(length, np.float64))


# ----------------------------------------------------------------------------------------------------- corpus
class Corpus:
    """Frames of a set of recordings on ``device`` plus flat arrays of their bars and contiguous bar pairs."""

    def __init__(self, frames, meta, device, keep=None):
        recs = [m for m in meta["recs"] if keep is None or keep(m)]
        self.recs = recs
        uploads = sorted({m["upload"] for m in recs})
        self.upload_id = {u: i for i, u in enumerate(uploads)}
        # keep only the kept recordings' frames, re-offset
        parts, offs, off = [], [], 0
        for m in recs:
            parts.append(frames[m["offset"]:m["offset"] + m["length"]])
            offs.append(off)
            off += m["length"]
        self.frames_np = np.concatenate(parts).astype(np.float32)
        self.frames = torch.from_numpy(self.frames_np).to(device)
        self.rec_start = np.array(offs, np.int64)
        self.rec_end = self.rec_start + np.array([m["length"] for m in recs], np.int64)
        self.rec_upload = np.array([self.upload_id[m["upload"]] for m in recs], np.int64)
        self.joints = None
        if "joints" in meta:
            self.joints = np.concatenate([meta["joints"][m["offset"]:m["offset"] + m["length"]] for m in recs])
        bs, be, br, bi, bl = [], [], [], [], []
        first = []
        for r, m in enumerate(recs):
            first.append(len(bs))
            for i, bar in enumerate(m["bars"]):
                bs.append(self.rec_start[r] + bar[0]); be.append(self.rec_start[r] + bar[1])
                br.append(r); bi.append(i); bl.append(bar[2] if len(bar) > 2 else -1)
        self.bar_start, self.bar_end = np.array(bs, np.int64), np.array(be, np.int64)
        self.bar_rec, self.bar_idx, self.bar_label = np.array(br, np.int64), np.array(bi, np.int64), np.array(bl)
        self.bar_len = self.bar_end - self.bar_start
        self.bar_upload = self.rec_upload[self.bar_rec]
        self.rec_first_bar = np.array(first + [len(bs)], np.int64)
        self.order = np.argsort(self.bar_len, kind="stable")
        self.sorted_len = self.bar_len[self.order]
        # contiguous pairs (bar g-1, bar g): g is the positive's global bar index
        g = np.arange(1, len(bs))
        ok = (self.bar_rec[g] == self.bar_rec[g - 1]) & (self.bar_start[g] == self.bar_end[g - 1])
        self.pairs = g[ok]

    def band(self, slot):
        s = slack_of(slot)
        lo = np.searchsorted(self.sorted_len, slot - s, "left")
        hi = np.searchsorted(self.sorted_len, slot + s, "right")
        return lo, hi


# ----------------------------------------------------------------------------------------------------- training
def join_cost(frames, ctx_end, ctx_rate, starts, natives, slots):
    """Seam join at the bar line in the body frame: mean joint distance of the candidate's first pose to the
    context's last, + 4 x the same for their per-slot-frame velocities.  ctx_* [B], candidates [B, M]."""
    pos = list(CH_POS)
    pc = frames[ctx_end - 1][:, pos].reshape(-1, 1, 9, 3)
    vc = (frames[ctx_end - 1][:, pos] - frames[ctx_end - 2][:, pos]).reshape(-1, 1, 9, 3) * ctx_rate[:, None, None, None]
    pk = frames[starts][..., pos].reshape(*starts.shape, 9, 3)
    rate = ((natives - 1).clamp_min(1) / (slots[:, None] - 1).clamp_min(1)).float()
    vk = (frames[starts + 1][..., pos] - frames[starts][..., pos]).reshape(*starts.shape, 9, 3) * rate[..., None, None]
    return (pk - pc).norm(dim=-1).mean(-1) + 4 * (vk - vc).norm(dim=-1).mean(-1)


def sample_batch(c, size, rng, gap, device, p_splice=0.3, p_partial=0.15, k_far=K_FAR, gap_max=None, ctx_drop=0):
    """-> dict of tensors for one training batch (candidate 0 is the positive).

    ``gap_max`` > ``gap``: every example draws its own seam mask from gap..gap_max (both sides, the same for the
    context and all its candidates).  ``ctx_drop`` > 0: the context additionally hides its last 0..ctx_drop visible
    frames (drawn per example).  Both off by default (the v1 recipe); they exist because v1 turned out to take most
    of its advantage from the few context frames just outside the mask (see the reach check)."""
    g = c.pairs[rng.integers(len(c.pairs), size=size)]
    rec = c.bar_rec[g]
    a0, b0, b1 = c.bar_start[g - 1], c.bar_start[g], c.bar_end[g]
    L = b1 - b0
    f = rng.uniform(*TEMPO, size=size)
    S = np.maximum(8, np.round(L * f)).astype(np.int64)
    r = (L - 1) / np.maximum(S - 1, 1)
    up = c.rec_upload[rec]
    lo, hi = c.band(S)
    n_draw = K_RAND + HARD_POOL + K_RATE + k_far + 8
    u = rng.random((size, n_draw))
    draw = c.order[np.minimum(lo[:, None] + np.floor(u * np.maximum(hi - lo, 1)[:, None]).astype(np.int64),
                              len(c.order) - 1)]
    bad = c.bar_upload[draw] == up[:, None]
    draw = np.take_along_axis(draw, np.argsort(bad, axis=1, kind="stable"), 1)     # other uploads first
    rand = draw[:, :K_RAND]
    pool = draw[:, K_RAND:K_RAND + HARD_POOL]
    spare = draw[:, K_RAND + HARD_POOL:]
    starts = [b0[:, None], c.bar_start[rand]]
    natives = [L[:, None].astype(np.float64), c.bar_len[rand].astype(np.float64)]
    # hard: lowest join among the pool (GPU)
    fr = c.frames
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    jc = join_cost(fr, t(b0), t(r).float(), t(c.bar_start[pool]), t(c.bar_len[pool]).double(), t(S).double())
    hard = pool[np.arange(size)[:, None], jc.topk(K_HARD, dim=1, largest=False).indices.cpu().numpy()]
    starts.append(c.bar_start[hard]); natives.append(c.bar_len[hard].astype(np.float64))
    # rate: the true next bar replayed faster / slower, when the recording has the frames
    rs, rn = np.zeros((size, K_RATE), np.int64), np.zeros((size, K_RATE))
    for k, rate in enumerate(RATES):
        n = rate * (L - 1) + 1
        fits = b0 + np.ceil(n) <= c.rec_end[rec]
        rs[:, k] = np.where(fits, b0, c.bar_start[spare[:, k]])
        rn[:, k] = np.where(fits, n, c.bar_len[spare[:, k]])
    starts.append(rs); natives.append(rn)
    # far: same recording, >= 3 bars away
    fs, fn = np.zeros((size, k_far), np.int64), np.zeros((size, k_far))
    first, last = c.rec_first_bar[rec], c.rec_first_bar[rec + 1]
    for k in range(k_far):
        pick = np.full(size, -1)
        for _try in range(6):
            j = first + np.floor(rng.random(size) * (last - first)).astype(np.int64)
            ok = (np.abs(j - g) >= 3) & (pick < 0)
            pick = np.where(ok, j, pick)
        fill = spare[:, K_RATE + k]
        fs[:, k] = np.where(pick >= 0, c.bar_start[np.maximum(pick, 0)], c.bar_start[fill])
        fn[:, k] = np.where(pick >= 0, c.bar_len[np.maximum(pick, 0)], c.bar_len[fill])
    starts.append(fs); natives.append(fn)
    starts = np.concatenate(starts, 1)
    natives = np.concatenate(natives, 1)
    k = starts.shape[1]
    # context: tempo-played, sometimes spliced, sometimes only the previous bar
    valid_from = c.rec_start[rec].astype(np.float64)
    roll = rng.random(size)
    partial = roll < p_partial
    valid_from = np.where(partial, a0, valid_from)
    splice = (roll >= p_partial) & (roll < p_partial + p_splice)
    other = spare[:, -1]
    at = np.where(splice, a0, -np.inf).astype(np.float64)
    other_end = c.bar_end[other].astype(np.float64)
    other_from = c.rec_start[c.bar_rec[other]].astype(np.float64)
    gaps = np.full(size, gap, np.float64)
    if gap_max is not None and gap_max > gap:
        gaps = rng.integers(gap, gap_max + 1, size=size).astype(np.float64)
    ctx_gaps = gaps + (rng.integers(0, ctx_drop + 1, size=size) if ctx_drop > 0 else 0) + CTX_SKIP_BARS * S
    ctx, cvis, ctime = sample_context(fr, t(b0), t(S), t(valid_from), t(ctx_gaps), rates=t(r),
                                      splice=(t(at), t(other_end), t(other_from)))
    cand, kvis, ktime = sample_candidates(fr, t(starts.reshape(-1)), t(natives.reshape(-1)),
                                          t(np.repeat(S, k)), t(np.repeat(gaps, k)))
    return {"ctx": ctx, "cvis": cvis, "ctime": ctime, "cand": cand.reshape(size, k, *cand.shape[1:]),
            "kvis": kvis.reshape(size, k, -1), "ktime": ktime.reshape(size, k, -1)}


def logits(model, b, zero_ctx=False, scale_ref=None):
    cvis = b["cvis"]
    scale = context_scale(b["ctx"], cvis)
    scale = torch.where(torch.isfinite(scale), scale, torch.full_like(scale, scale_ref or 1.0))
    if zero_ctx:
        cvis = torch.zeros_like(cvis)
        scale = torch.full_like(scale, scale_ref or 1.0)
    c, k = assemble(b["ctx"], cvis, b["ctime"], b["cand"], b["kvis"], b["ktime"], scale, model.in_mean, model.in_std)
    return model(c, k)


# ----------------------------------------------------------------------------------------------------- eval rows
def lr_model(gap, train=None):
    """The feasibility study's 8-feature logistic regression (relative-only).  At gap 6 its own fitted weights
    (v5_stats_gap6_rel.json); at any other gap the study stored no parameters, so it is refitted here exactly as
    contmodel_v5_stats.main does (30000 train pairs, <= 6 negatives per kind) on THIS trainer's training corpus."""
    path = pathlib.Path(LR_JSON.format(gap, "_rel"))
    if path.is_file():
        p = json.load(open(path))["lr_params"]
        source = str(path)
    else:
        import contmodel_v5_stats as cs
        cs.FEATS = [f for f in cs.FEATS if f not in ("lEA", "lEB")]
        data = {}
        for r, m in enumerate(train.recs):
            k = train.frames_np[train.rec_start[r]:train.rec_end[r], :12]
            first, last = train.rec_first_bar[r], train.rec_first_bar[r + 1]
            bars = [(int(a - train.rec_start[r]), int(b - train.rec_start[r]))
                    for a, b in zip(train.bar_start[first:last], train.bar_end[first:last])]
            data[m["rec"]] = {"kin": k, "bars": bars}
        rows = cs.build(data, np.random.default_rng(0), gap, max_neg=6, limit=30000)
        X, y = cs.matrix(rows)
        cs.fit_lr(X, y)
        p = cs.fit_lr.params
        source = "refitted at gap {} on {} train pairs".format(gap, len(rows))
    mu, sd, w = np.array(p["mu"]), np.array(p["sd"]), np.array(p["w"])
    feats = p["feats"]
    return lambda f: float(((np.array([f[k] for k in feats]) - mu) / sd) @ w + p["b"]), source


def v5_rows(c, rng, max_rand=20, max_far=20, limit=None):
    """contmodel_v5_stats.build's rows, as candidate lists: (g, [(kind, start, native)])."""
    rows = []
    pairs = c.pairs if limit is None else c.pairs[rng.permutation(len(c.pairs))[:limit]]
    for g in pairs:
        b0, b1 = c.bar_start[g], c.bar_end[g]
        L = b1 - b0
        rec = c.bar_rec[g]
        lo, hi = c.band(np.array([L]))
        cand = c.order[lo[0]:hi[0]]
        cands = []
        if len(cand):
            pick = rng.choice(cand, min(len(cand), 3 * max_rand), replace=False)
            pick = [p for p in pick if c.bar_upload[p] != c.rec_upload[rec]][:max_rand]
            cands += [("rand", c.bar_start[p], float(c.bar_len[p])) for p in pick]
        first, last = c.rec_first_bar[rec], c.rec_first_bar[rec + 1]
        far = [j for j in range(first, last) if abs(j - g) >= 3]
        if len(far) > max_far:
            far = list(rng.choice(far, max_far, replace=False))
        cands += [("far", c.bar_start[j], float(c.bar_len[j])) for j in far]
        for rate in RATES:
            if b0 + rate * (L - 1) < c.rec_end[rec] - 1:
                cands.append(("rate", b0, rate * (L - 1) + 1))
        rows.append({"g": int(g), "rec": int(rec), "slot": int(L), "cands": cands})
    return rows


def lib_rows(c, rng, max_neg=40, join_keep=0.35):
    """contmodel_v5_stats.t2_rows' rows: T library pairs, the real retrieval pool (same label, other upload,
    duration band; >= 5 or skipped; <= 40 sampled), its join-band subset, and same-recording far bars."""
    from contmodel_feats import describe, pair, resample
    rows = []
    for g in c.pairs:
        b0, b1 = c.bar_start[g], c.bar_end[g]
        L = b1 - b0
        rec = c.bar_rec[g]
        lab = c.bar_label[g]
        s = slack_of(L)
        pool = np.flatnonzero((c.bar_label == lab) & (c.bar_upload != c.rec_upload[rec])
                              & (np.abs(c.bar_len - L) <= s))
        if len(pool) < 5:
            continue
        if len(pool) > max_neg:
            pool = pool[rng.choice(len(pool), max_neg, replace=False)]
        a0, a1 = c.bar_start[g - 1], c.bar_end[g - 1]
        adesc = describe(c.joints[a0:a1])
        joins = [pair(adesc, describe(resample(c.joints[c.bar_start[p]:c.bar_end[p]], L)))["join"] for p in pool]
        keep = set(np.argsort(joins)[:max(2, int(round(join_keep * len(joins))))].tolist())
        cands = [("rand", c.bar_start[p], float(c.bar_len[p])) for p in pool]
        cands += [("join", c.bar_start[p], float(c.bar_len[p])) for i, p in enumerate(pool) if i in keep]
        first, last = c.rec_first_bar[rec], c.rec_first_bar[rec + 1]
        cands += [("far", c.bar_start[j], float(c.bar_len[j])) for j in range(first, last) if abs(j - g) >= 3]
        rows.append({"g": int(g), "rec": int(rec), "slot": int(L), "cands": cands})
    return rows


# --ctx-skip-bars 1 (research only; inference refuses such a checkpoint): the context additionally hides its whole
# LAST slot, in training and in every check, so the scorer must predict bar n from bar n-2 alone.  Its rand AUC minus
# its identity control is then what bar n-2 says about bar n beyond who is dancing -- the most any "second bar" can add.
CTX_SKIP_BARS = 0
REACH_VISIBLE = (6, 10, 30)     # context frames left visible beyond the mask (truncated context)
REACH_HIDE = (10, 30)           # context frames hidden beyond the mask (only the context's far part visible)


@torch.no_grad()
def model_scores(model, c, rows, gap, device, zero_ctx=False, scale_ref=None, batch=128, ends_override=None,
                 reach=None, rng=None):
    """-> per row (positive score, [candidate scores]) at slot = L, tempo 1, context = the recording's own frames
    before the positive -- or, with ``ends_override`` (one bar line per row), the recording's frames before THAT bar
    line (the identity control).

    ``reach`` changes only HOW MUCH of that context is visible (the reach check; candidates untouched):
      ("visible", V)  the context truncated to its last gap + V frames: only V frames beyond the mask are visible;
      ("hide", V)     the context's last gap + V frames hidden: only its FAR part is visible;
      ("prev",)       only the previous bar (the context starts at bar g-1's start);
      ("splice",)     before the previous bar, another upload's bar ending at its own bar line (``rng`` picks it) --
                      the real previous bar, a foreign second bar: what the draft's context always is."""
    model.eval()
    out = []
    t = lambda a: torch.as_tensor(np.asarray(a), device=device)  # noqa: E731
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        B = len(part)
        kmax = 1 + max(len(r["cands"]) for r in part)
        starts = np.zeros((B, kmax), np.int64)
        natives = np.zeros((B, kmax))
        slots = np.array([r["slot"] for r in part])
        for j, r in enumerate(part):
            g = r["g"]
            row_s = [c.bar_start[g]] + [x[1] for x in r["cands"]]
            row_n = [float(c.bar_len[g])] + [x[2] for x in r["cands"]]
            starts[j, :len(row_s)] = row_s; starts[j, len(row_s):] = row_s[0]
            natives[j, :len(row_n)] = row_n; natives[j, len(row_n):] = row_n[0]
        ends = (np.array([c.bar_start[r["g"]] for r in part]) if ends_override is None
                else np.asarray(ends_override[i:i + batch]))
        vf = np.array([c.rec_start[r["rec"]] for r in part], np.float64)
        ctx_gap, splice = np.full(B, float(gap)) + CTX_SKIP_BARS * slots, None
        if reach is not None and reach[0] == "visible":
            vf = np.maximum(vf, ends - gap - reach[1]).astype(np.float64)
        elif reach is not None and reach[0] == "hide":
            ctx_gap = ctx_gap + reach[1]
        elif reach is not None and reach[0] == "prev":
            vf = np.maximum(vf, [c.bar_start[r["g"] - 1] for r in part]).astype(np.float64)
        elif reach is not None and reach[0] == "splice":
            at, oe, of = [], [], []
            for r in part:
                while True:
                    o = int(rng.integers(len(c.bar_start)))
                    if c.bar_upload[o] != c.rec_upload[r["rec"]]:
                        break
                at.append(float(c.bar_start[r["g"] - 1]))
                oe.append(float(c.bar_end[o]))
                of.append(float(c.rec_start[c.bar_rec[o]]))
            splice = (t(at), t(oe), t(of))
        ctx, cvis, ctime = sample_context(c.frames, t(ends), t(slots), t(vf), t(ctx_gap), splice=splice)
        cand, kvis, ktime = sample_candidates(c.frames, t(starts.reshape(-1)), t(natives.reshape(-1)),
                                              t(np.repeat(slots, kmax)), gap)
        b = {"ctx": ctx, "cvis": cvis, "ctime": ctime, "cand": cand.reshape(B, kmax, *cand.shape[1:]),
             "kvis": kvis.reshape(B, kmax, -1), "ktime": ktime.reshape(B, kmax, -1)}
        s = logits(model, b, zero_ctx=zero_ctx, scale_ref=scale_ref).float().cpu().numpy()
        for j, r in enumerate(part):
            out.append((s[j, 0], s[j, 1:1 + len(r["cands"])]))
    model.train()
    return out


def far_context(c, rows, rng):
    """THE IDENTITY CONTROL: for each row, a bar line of the SAME recording whose two-bar context touches neither the
    positive nor its real context (it ends >= 3 bars before the positive, or starts >= 1 bar after it).  The
    positive then shares the context's dancer, song and reconstruction but is not what came next: an AUC that stays
    as high as with the real context would mean the scorer matches identity, not continuation.
    -> (rows that have such a bar line, their context ends)."""
    keep, ends = [], []
    for r in rows:
        g, rec = r["g"], r["rec"]
        first, last = c.rec_first_bar[rec], c.rec_first_bar[rec + 1]
        options = [j for j in range(first + 2, last) if j <= g - 3 or j >= g + 3]
        if options:
            keep.append(r)
            ends.append(c.bar_start[options[int(rng.integers(len(options)))]])
    return keep, np.array(ends, np.int64)


def lr_scores(c, rows, gap, score):
    """The logistic baseline on exactly the same rows (kinetics = continuation frames 0-11)."""
    from contmodel_v5_stats import kdesc, kin_resample, kpair
    out = []
    k_all = c.frames_np[:, :12]
    for r in rows:
        g = r["g"]
        a0, a1 = c.bar_start[g - 1], c.bar_end[g - 1]
        L = r["slot"]
        A = kdesc(k_all[a0:a1 - gap] if gap else k_all[a0:a1])

        def one(kind, start, native):
            if kind != "rate":
                seg = k_all[int(start):int(start) + int(round(native))]
                return score(kpair(A, kdesc(kin_resample(seg, L)[gap:])))
            rate = (native - 1) / (L - 1)
            x = start + rate * np.arange(L)
            lo = np.floor(x).astype(int)
            hi = np.minimum(lo + 1, len(k_all) - 1)
            w = (x - lo)[:, None]
            kk = (k_all[lo] * (1 - w) + k_all[hi] * w) * rate
            return score(kpair(A, kdesc(kk[gap:])))

        out.append((one("pos", c.bar_start[g], float(c.bar_len[g])),
                    np.array([one(kind, s, n) for kind, s, n in r["cands"]])))
    return out


def auc_table(c, rows, scores, kinds):
    """{kind: (AUC, recordings won, recordings, sign P, per-recording AUC dict)}."""
    res = {}
    for kind in kinds:
        wins, per = [], {}
        for r, (pos, neg) in zip(rows, scores):
            sel = np.array([x[0] == kind for x in r["cands"]], bool)
            if not sel.any():
                continue
            nv = neg[sel]
            w = float(np.mean(pos > nv) + 0.5 * np.mean(pos == nv))
            wins.append(w)
            per.setdefault(r["rec"], []).append(w)
        if wins:
            per = {k: float(np.mean(v)) for k, v in per.items()}
            k_, n_, p_ = sign_p([v - 0.5 for v in per.values()])
            res[kind] = (float(np.mean(wins)), k_, n_, p_, per)
    return res


def paired(model_tab, lr_tab):
    out = {}
    for kind in model_tab:
        if kind not in lr_tab:
            continue
        a, b = model_tab[kind][4], lr_tab[kind][4]
        d = [a[k] - b[k] for k in a if k in b]
        out[kind] = sign_p(d) + (float(np.mean(d)) if d else float("nan"),)
    return out


def fmt(tab):
    return "  ".join("{} {:.3f} ({}/{} P={:.2g})".format(k, v[0], v[1], v[2], v[3]) for k, v in tab.items())


def final_checks(model, gap, scale_ref, device, val, val_rows_all, libs, lib_rows_, lr_val, lr_lib):
    """Every check in the module docstring, on one checkpoint.  Printed and returned."""
    checks = {}
    rng = np.random.default_rng(4000)

    def report(label, c, rows, kinds, lr_tab):
        m = auc_table(c, rows, model_scores(model, c, rows, gap, device, scale_ref=scale_ref), kinds)
        m0 = auc_table(c, rows, model_scores(model, c, rows, gap, device, zero_ctx=True, scale_ref=scale_ref), kinds)
        frows, fends = far_context(c, rows, rng)
        ikinds = tuple(k for k in kinds if k != "far" and k != "rate")
        mf = auc_table(c, frows, model_scores(model, c, frows, gap, device, scale_ref=scale_ref, ends_override=fends),
                       ikinds)
        # adjacency beyond identity: the SAME rows with the real context minus with the non-adjacent one, per recording
        mr = auc_table(c, frows, model_scores(model, c, frows, gap, device, scale_ref=scale_ref), ikinds)
        adj = paired(mr, mf)
        pl = paired(m, lr_tab)
        print("{:22s} model    ".format("{} ({} pairs)".format(label, len(rows))) + fmt(m))
        print("{:22s} logistic ".format("") + fmt(lr_tab))
        print("{:22s} ctx0     ".format("") + fmt(m0))
        print("{:22s} ctxfar   ".format("") + fmt(mf) + "  ({} pairs: same-recording context, not adjacent)"
              .format(len(frows)))
        print("  paired model-logistic per recording: " + "  ".join(
            "{} {}/{} P={:.2g} mean {:+.3f}".format(k, *x) for k, x in pl.items()))
        print("  adjacency beyond identity (real context - same-recording non-adjacent context, same rows, per "
              "recording): " + "  ".join("{} {}/{} P={:.2g} mean {:+.3f}".format(k, *x) for k, x in adj.items()))
        return {"model": {k: x[:4] for k, x in m.items()}, "logistic": {k: x[:4] for k, x in lr_tab.items()},
                "ctx0": {k: x[:4] for k, x in m0.items()}, "ctxfar": {k: x[:4] for k, x in mf.items()},
                "paired": pl, "adjacency_beyond_identity": adj, "pairs": len(rows)}

    def reach_report(c, rows, kinds):
        """THE REACH CHECK: how far back the scorer actually reads.  Same rows, same candidates; only the visible part
        of the context changes.  Rows with two whole bars of the recording's own motion before the positive, so
        "full" really has a second bar to use."""
        rows = [r for r in rows if c.bar_start[r["g"]] - c.rec_start[r["rec"]] >= CONTEXT_BARS * r["slot"]]
        modes = [("full", None), ("prev", ("prev",)), ("splice", ("splice",))]
        modes += [("vis{}".format(v), ("visible", v)) for v in REACH_VISIBLE]
        modes += [("hide{}".format(v), ("hide", v)) for v in REACH_HIDE]
        tabs = {name: auc_table(c, rows, model_scores(model, c, rows, gap, device, scale_ref=scale_ref, reach=mode,
                                                      rng=np.random.default_rng(4100)), kinds)
                for name, mode in modes}
        out = {"pairs": len(rows), "auc": {name: {k: x[0] for k, x in tab.items()} for name, tab in tabs.items()}}
        for kind in kinds:
            print("  reach {:4s} ".format(kind) + "  ".join(
                "{} {:.3f}".format(name, tabs[name][kind][0]) for name, _m in modes if kind in tabs[name]))
        # the second context bar: full minus previous-bar-only, per recording
        second, near = {}, {}
        for kind in kinds:
            a, b, v6 = tabs["full"][kind][4], tabs["prev"][kind][4], tabs["vis6"][kind][4]
            d = [a[k] - b[k] for k in a if k in b]
            second[kind] = sign_p(d) + (float(np.mean(d)) if d else float("nan"),)
            full_above = tabs["full"][kind][0] - 0.5
            near[kind] = (tabs["vis6"][kind][0] - 0.5) / full_above if full_above > 1e-6 else float("nan")
        out["second_bar"] = second
        out["share_at_6_visible"] = near
        print("  second context bar (full - previous bar only, per recording): " + "  ".join(
            "{} {:+.4f} ({}/{} P={:.2g})".format(k, x[3], x[0], x[1], x[2]) for k, x in second.items()))
        print("  share of full's above-chance AUC reached with 6 visible context frames: " + "  ".join(
            "{} {:.2f}".format(k, x) for k, x in near.items()) + "  ({} pairs)".format(len(rows)))
        return out

    print("\n== checks (seam mask {}) ==".format(gap))
    checks["v5_val"] = report("v5 val", val, val_rows_all, ("rand", "far", "rate"), lr_val)
    if CTX_SKIP_BARS:
        print("  (context skips its last {} bar(s): no reach check)".format(CTX_SKIP_BARS))
    else:
        checks["v5_val"]["reach"] = reach_report(val, val_rows_all, ("rand", "far"))
    for name, c in libs.items():
        checks[name] = report("{} pool".format(name.upper()), c, lib_rows_[name], ("rand", "join", "far"), lr_lib[name])
        if not CTX_SKIP_BARS:
            checks[name]["reach"] = reach_report(c, lib_rows_[name], ("rand", "join", "far"))
    if not CTX_SKIP_BARS:
        checks["verdict"] = reach_verdict(checks, [n for n in libs])
        print("VERDICT: " + checks["verdict"]["text"])
    return checks


SECOND_BAR_SHARE = 0.10


def reach_verdict(checks, pools):
    """What the checkpoint may be CALLED.  The rule (the review of 2026-09-23, from the operator's "不止是接缝"): a
    scorer counts as reading the phrase only if the SECOND context bar measurably helps on the real pools -- full
    context beats previous-bar-only on the rand column, per recording, sign test P < 0.05 with a positive mean, on
    every T pool -- AND the gain is at least ``SECOND_BAR_SHARE`` of the scorer's own full-context AUC above chance.
    The share floor is MY OWN threshold (not from the operator or a paper): significance alone let a +0.005 gain
    (2% of what cont_p12r reads) pass, and a bar that moves 2% of a ranking does not make the ranking phrase-level.
    Otherwise it is a previous-bar scorer, and its reach is stated as the fewest visible context frames (beyond the
    mask) that recover 90% of its full-context AUC above chance on the T pools.  The sign test is able to see a small
    gain: on v5 val (~900 recordings) it resolves v1's +0.008 at P ~ 1e-24."""
    used, reach = True, 0
    for name in pools:
        r = checks[name]["reach"]
        w, n, pv, mean = r["second_bar"]["rand"]
        auc = r["auc"]
        above = auc["full"]["rand"] - 0.5
        used &= bool(pv < 0.05 and mean > 0 and above > 0 and mean >= SECOND_BAR_SHARE * above)
        need = next((v for v in REACH_VISIBLE if auc["vis{}".format(v)]["rand"] - 0.5 >= 0.9 * above), None)
        reach = max(reach, need if need is not None else 10 ** 6)
    gains = "; ".join("{} {:+.4f} = {:.0%} of its above-chance AUC, {}/{} P={:.2g}".format(
        nm, checks[nm]["reach"]["second_bar"]["rand"][3],
        checks[nm]["reach"]["second_bar"]["rand"][3] / max(1e-6, checks[nm]["reach"]["auc"]["full"]["rand"] - 0.5),
        *checks[nm]["reach"]["second_bar"]["rand"][:3]) for nm in pools)
    if used:
        text = "PHRASE-LEVEL: the second context bar helps on every T pool ({})".format(gains)
    else:
        text = ("PREVIOUS-BAR SCORER: the second context bar adds less than {:.0%} of what it reads, or nothing "
                "significant, on the T pools ({}); 90% of its above-chance AUC is reached with {} visible context "
                "frames beyond the mask".format(SECOND_BAR_SHARE, gains,
                                                reach if reach < 10 ** 6 else "more than {}".format(max(REACH_VISIBLE))))
    return {"second_bar_used": used, "frames_for_90pct": reach if reach < 10 ** 6 else None, "text": text}


# ----------------------------------------------------------------------------------------------------- main
def exclusions(args):
    ex = json.load(open(args.exclude))
    bad_up = set(ex["uploads"]) | set(DEMO_UPLOADS)
    bad_rec = set(ex["recordings"])
    lib_up = set()
    if args.exclude_libraries:
        for root, _seg in LIBS.values():
            for line in open(root / "windows.jsonl"):
                w = json.loads(line)
                lib_up.add((w.get("recording_id") or w["sequence_id"]).split(":")[1])
    return bad_up, bad_rec, lib_up


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seam-mask", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=10, help="evals without a val improvement before stopping")
    ap.add_argument("--p-splice", type=float, default=0.3)
    ap.add_argument("--p-partial", type=float, default=0.15)
    ap.add_argument("--k-far", type=int, default=K_FAR, help="same-recording far negatives per example")
    ap.add_argument("--train-gap-max", type=int, default=None,
                    help="draw each training example's seam mask from --seam-mask..this (checks and inference use "
                         "--seam-mask); default: fixed")
    ap.add_argument("--ctx-drop-max", type=int, default=0,
                    help="hide a further 0..N of the context's last visible frames per training example")
    ap.add_argument("--ctx-skip-bars", type=int, default=0, choices=(0, 1),
                    help="research only: hide the context's whole last slot (predict bar n from bar n-2)")
    ap.add_argument("--select", choices=("rand", "rand+far"), default="rand",
                    help="v5 val column the checkpoint is chosen on (rand+far: their mean)")
    ap.add_argument("--exclude", default=str(EXT / "rhythm_scorer_exclusions.json"))
    ap.add_argument("--no-exclude-libraries", dest="exclude_libraries", action="store_false")
    ap.add_argument("--val-limit", type=int, default=3000, help="v5 val pairs scored at every eval (all at the end)")
    ap.add_argument("--eval-only", action="store_true",
                    help="load --out and rerun the checks (written to <out>.checks.json); no training")
    args = ap.parse_args()
    global CTX_SKIP_BARS
    CTX_SKIP_BARS = args.ctx_skip_bars
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gap = args.seam_mask
    t0 = time.time()

    bad_up, bad_rec, lib_up = exclusions(args)
    frames, meta = load("v5_train")
    n_all = len(meta["recs"])
    keep_train = lambda m: (m["upload"] not in bad_up and m["rec"] not in bad_rec  # noqa: E731
                            and m["upload"] not in lib_up)
    dropped = {"eval/song/demo": sum(1 for m in meta["recs"] if m["upload"] in bad_up or m["rec"] in bad_rec),
               "T1/T2 library uploads": sum(1 for m in meta["recs"] if m["upload"] in lib_up
                                            and not (m["upload"] in bad_up or m["rec"] in bad_rec))}
    train = Corpus(frames, meta, device, keep_train)
    del frames
    vframes, vmeta = load("v5_val")
    # the logistic baseline's val protocol: exclusions, T library uploads NOT removed (as contmodel_v5_stats)
    val = Corpus(vframes, vmeta, device, lambda m: m["upload"] not in bad_up and m["rec"] not in bad_rec)
    print("train: {} of {} v5 train recordings ({} dropped: {}), {} bars, {} pairs; val: {} recordings, {} pairs".format(
        len(train.recs), n_all, n_all - len(train.recs), dropped, len(train.bar_start), len(train.pairs),
        len(val.recs), len(val.pairs)), flush=True)
    libs = {}
    for name in ("t2", "t1"):
        lf, lm = load(name)
        libs[name] = Corpus(lf, lm, device)
    lr_score, lr_path = lr_model(gap, train)
    erng = np.random.default_rng(1000)
    val_rows_all = v5_rows(val, erng)
    val_rows = [val_rows_all[i] for i in np.random.default_rng(7).permutation(len(val_rows_all))[:args.val_limit]]
    train_rows = v5_rows(train, np.random.default_rng(2000), limit=1500)
    lib_rows_ = {name: lib_rows(c, np.random.default_rng(3000)) for name, c in libs.items()}
    lr_val = auc_table(val, val_rows_all, lr_scores(val, val_rows_all, gap, lr_score), ("rand", "far", "rate"))
    lr_lib = {name: auc_table(c, lib_rows_[name], lr_scores(c, lib_rows_[name], gap, lr_score), ("rand", "join", "far"))
              for name, c in libs.items()}
    print("logistic baseline ({}), same rows:".format(lr_path))
    print("  v5 val  " + fmt(lr_val))
    for name in libs:
        print("  {} pool ".format(name.upper()) + fmt(lr_lib[name]) + "  ({} pairs)".format(len(lib_rows_[name])))
    print("rows built {:.0f}s".format(time.time() - t0), flush=True)
    if args.eval_only:
        from model.continuation_scorer import load_scorer
        model, blob = load_scorer(args.out, device, allow_research=True)
        if int(blob.get("args", {}).get("ctx_skip_bars", 0)) != CTX_SKIP_BARS:
            raise SystemExit("{} was trained with --ctx-skip-bars {}".format(
                args.out, blob.get("args", {}).get("ctx_skip_bars", 0)))
        if int(blob["seam_mask"]) != gap:
            raise SystemExit("{} was trained at seam mask {}; pass --seam-mask {}".format(
                args.out, blob["seam_mask"], blob["seam_mask"]))
        checks = final_checks(model, gap, blob["scale_ref"], device, val, val_rows_all, libs, lib_rows_, lr_val, lr_lib)
        json.dump(checks, open(args.out + ".checks.json", "w"), indent=1, default=float)
        return

    model = ContinuationScorer(width=args.width, dropout=args.dropout).to(device)
    recipe = {"k_far": args.k_far, "gap_max": args.train_gap_max, "ctx_drop": args.ctx_drop_max}
    select = (lambda v: v["rand"][0]) if args.select == "rand" else (lambda v: 0.5 * (v["rand"][0] + v["far"][0]))
    # input normalisation from training batches (speeds already divided by the context scale)
    with torch.no_grad():
        acc, scales = [], []
        for _ in range(8):
            b = sample_batch(train, 256, rng, gap, device, args.p_splice, args.p_partial, **recipe)
            s = context_scale(b["ctx"], b["cvis"])
            scales.append(s[torch.isfinite(s)])
            s = torch.where(torch.isfinite(s), s, torch.ones_like(s)).clamp_min(1e-3)
            x = b["cand"].clone()
            from model.continuation_scorer import SPEED_CHANNELS
            x[..., list(SPEED_CHANNELS)] /= s[:, None, None, None]
            acc.append(x[b["kvis"]].reshape(-1, FRAME_CHANNELS))
        acc = torch.cat(acc)
        model.in_mean.copy_(acc.mean(0))
        model.in_std.copy_(acc.std(0) + 1e-4)
        scale_ref = float(torch.cat(scales).median())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    history, best, since = [], None, 0
    for step in range(1, args.steps + 1):
        b = sample_batch(train, args.batch, rng, gap, device, args.p_splice, args.p_partial, **recipe)
        s = logits(model, b, scale_ref=scale_ref)
        loss = F.cross_entropy(s, torch.zeros(len(s), dtype=torch.long, device=device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            v = auc_table(val, val_rows, model_scores(model, val, val_rows, gap, device, scale_ref=scale_ref),
                          ("rand", "far", "rate"))
            tr = auc_table(train, train_rows, model_scores(model, train, train_rows, gap, device, scale_ref=scale_ref),
                           ("rand", "far", "rate"))
            t2 = auc_table(libs["t2"], lib_rows_["t2"], model_scores(model, libs["t2"], lib_rows_["t2"], gap, device,
                                                                     scale_ref=scale_ref), ("rand", "join"))
            ev = {"step": step, "loss": round(float(loss.detach()), 4), "seconds": round(time.time() - t0),
                  "val": {k: round(x[0], 4) for k, x in v.items()},
                  "train": {k: round(x[0], 4) for k, x in tr.items()},
                  "t2": {k: round(x[0], 4) for k, x in t2.items()}, "select": round(select(v), 4)}
            history.append(ev)
            print(json.dumps(ev), flush=True)
            if best is None or ev["select"] > best["select"]:
                best, since = ev, 0
                state = {k: x.detach().clone() for k, x in model.state_dict().items()}
            else:
                since += 1
                if since >= args.patience:
                    print("early stop: val {} AUC has not improved for {} evals (best step {}; train rand {} vs val {})"
                          .format(args.select, since, best["step"], best["train"]["rand"], best["val"]["rand"]),
                          flush=True)
                    break
    model.load_state_dict(state)
    checks = final_checks(model, gap, scale_ref, device, val, val_rows_all, libs, lib_rows_, lr_val, lr_lib)
    checks["best_step"] = best["step"]
    torch.save({"stage": "continuation_scorer", "feature": FEATURE_VERSION, "seam_mask": gap,
                "context_bars": CONTEXT_BARS, "arch": {"width": args.width, "dropout": args.dropout},
                "state_dict": model.state_dict(), "scale_ref": scale_ref, "args": vars(args), "checks": checks,
                "history": history, "logistic_params": lr_path,
                "exclusions": {"uploads": len(bad_up), "recordings": len(bad_rec), "library_uploads": len(lib_up),
                               "train_recordings": len(train.recs), "dropped": dropped}}, args.out)
    print("saved", args.out, "({:.0f}s)".format(time.time() - t0))


if __name__ == "__main__":
    main()
