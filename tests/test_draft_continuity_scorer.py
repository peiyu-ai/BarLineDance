"""--draft-continuity-scorer / -keep / -weight: the learned "is this the natural NEXT BAR" score in retrieval.

Pinned here without a checkpoint (the scorer is replaced by a stub that reads the candidate's start):
  * the keep filter narrows to the best-scoring ceil(keep * n) candidates, never fewer than two, and hands them
    back in their ORIGINAL order (the join band and the selector downstream see a subset, not a re-ranking);
  * it is a no-op with no placed context (the clip's first slot), with too few candidates, or with keep 0 -- and
    counts what it did;
  * the joint mode ranks by z(rhythm) + W * z(continuity) inside the rhythm keep fraction, and falls back to
    rhythm alone without a context;
  * the options are validated (a scorer needs a keep or a weight; keep and weight are exclusive; the weight
    needs the rhythm scorer);
  * the draft context is the PLACED draft only: it ends at the slot, stops at an unplaced frame, and is None when
    too little is placed (CLAUDE.md 1.6: generated motion, never the target clip's);
  * the model: features blind to where the dancer stands and faces, [B, K] scores, a candidate's score independent
    of the rest of its pool, and the seam mask really hides the frames it claims to hide.
"""
import ast
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic  # noqa: E402
from model import continuation_scorer as cs  # noqa: E402

Lib = infer_atomic.IndexedAtomicMotionLibrary
CONTEXT = (np.zeros((120, cs.FRAME_CHANNELS), np.float32), 0)


def _stub(keep=0.0, weight=0.0, rhythm_keep=0.0, score=lambda c: -abs(c[1] - 7), rhythm=lambda c: c[1]):
    lib = Lib.__new__(Lib)
    lib.continuity_keep = keep
    lib.continuity_weight = weight
    lib.rhythm_keep = rhythm_keep
    lib._continuity_scores = lambda candidates, context, length: np.array([score(c) for c in candidates], float)
    lib._rhythm_scores = lambda candidates, music: np.array([rhythm(c) for c in candidates], float)
    return lib


def test_keep_narrows_to_the_best_half_in_original_order():
    tied = [(0, s, s + 40, "g") for s in (20, 7, 0, 9, 6, 30, 8, 1)]
    kept = _stub(keep=0.5)._prefer_continuity(tied, CONTEXT, 40)
    # scores: 20->-13, 7->0, 0->-7, 9->-2, 6->-1, 30->-23, 8->-1, 1->-6: the best four are 7, 6, 8, 9
    assert kept == [(0, 7, 47, "g"), (0, 9, 49, "g"), (0, 6, 46, "g"), (0, 8, 48, "g")]


def test_keep_never_leaves_fewer_than_two():
    tied = [(0, s, s + 40, "g") for s in (7, 20, 30, 40)]
    assert len(_stub(keep=0.01)._prefer_continuity(tied, CONTEXT, 40)) == 2


def test_no_context_too_few_or_off_is_a_no_op_and_counted():
    tied = [(0, s, s + 40, "g") for s in (7, 20, 30, 40)]
    lib = _stub(keep=0.5)
    assert lib._prefer_continuity(tied, None, 40) == tied
    assert lib.continuity_no_context == 1 and not hasattr(lib, "continuity_slots")
    assert lib._prefer_continuity(tied[:2], CONTEXT, 40) == tied[:2]
    assert _stub(keep=0.0)._prefer_continuity(tied, CONTEXT, 40) == tied
    lib._prefer_continuity(tied, CONTEXT, 40)
    assert lib.continuity_slots == 1 and lib.continuity_applied == 1


def test_a_stub_library_without_the_options_is_untouched():
    """Libraries built with Lib.__new__ (other tests' stubs) carry no continuity attributes at all."""
    lib = Lib.__new__(Lib)
    tied = [(0, s, s + 40, "g") for s in (7, 20, 30, 40)]
    assert lib._prefer_continuity(tied, CONTEXT, 40) == tied


def test_joint_ranking_mixes_standardised_scores_inside_the_rhythm_keep():
    tied = [(0, s, s + 40, "g") for s in (0, 1, 2, 3, 4, 5, 6, 7)]
    music = np.zeros((48, 35), np.float32)
    # rhythm prefers late starts, continuity prefers early ones
    lib = _stub(weight=2.0, rhythm_keep=0.25, score=lambda c: -c[1], rhythm=lambda c: c[1])
    kept = lib._prefer_rhythm(tied, music, CONTEXT, 40)
    assert kept == [(0, 0, 40, "g"), (0, 1, 41, "g")]                 # W=2: continuity wins
    lib = _stub(weight=0.5, rhythm_keep=0.25, score=lambda c: -c[1], rhythm=lambda c: c[1])
    assert lib._prefer_rhythm(tied, music, CONTEXT, 40) == [(0, 6, 46, "g"), (0, 7, 47, "g")]   # rhythm wins
    assert lib.continuity_joint_slots == 1 and not hasattr(lib, "continuity_joint_changed")
    # without a placed context the joint mode is the rhythm filter alone, counted
    lib = _stub(weight=2.0, rhythm_keep=0.25, score=lambda c: -c[1], rhythm=lambda c: c[1])
    assert lib._prefer_rhythm(tied, music, None, 40) == [(0, 6, 46, "g"), (0, 7, 47, "g")]
    assert lib.continuity_no_context == 1


def test_options_are_validated():
    def make(**kw):
        Lib.__init__(Lib.__new__(Lib), data_root="/nonexistent", **kw)

    with pytest.raises(ValueError, match="needs --draft-continuity-keep or --draft-continuity-weight"):
        make(continuity_scorer="x.pt")
    with pytest.raises(ValueError, match="need --draft-continuity-scorer"):
        make(continuity_keep=0.5)
    with pytest.raises(ValueError, match="pass one"):
        make(continuity_scorer="x.pt", continuity_keep=0.5, continuity_weight=1.0)
    with pytest.raises(ValueError, match="needs --draft-rhythm-scorer and --draft-rhythm-keep"):
        make(continuity_scorer="x.pt", continuity_weight=1.0)
    with pytest.raises(ValueError, match="fraction of candidates kept"):
        make(continuity_scorer="x.pt", continuity_keep=1.5)


def test_the_learned_rule_applies_the_filter_after_the_rhythm_filter():
    source = pathlib.Path(infer_atomic.__file__).read_text()
    branch = source.split('elif self.retrieval_rule == "learned":', 1)[1].split("\n        elif ", 1)[0]
    rhythm = branch.index("self._prefer_rhythm(narrowed, slot_music, continuity_context, target_length)")
    cont = branch.index("self._prefer_continuity(narrowed, continuity_context, target_length)")
    band = branch.index("selector_join_band")
    assert rhythm < cont < band


def _fake_decode(values, normalizer_path):
    """values [T, D] -> joints [T, 24, 3] that remember which draft frames they came from."""
    v = np.asarray(values, dtype=np.float64)
    t = len(v)
    j = np.zeros((t, 24, 3))
    j[:, :, 0] = v[:, :1]                       # every joint's x = the frame's first value
    j[:, 1, 0] -= 0.1
    j[:, 2, 0] += 0.1                           # a hip axis, so the heading is defined
    return {"full_pose": j.astype(np.float32)}


def test_the_draft_context_is_the_placed_draft_before_the_slot(monkeypatch):
    monkeypatch.setattr(infer_atomic, "decode_motion", _fake_decode)
    lib = Lib.__new__(Lib)
    lib.normalizer_path = None
    draft = torch.arange(200, dtype=torch.float32)[:, None].repeat(1, 4)
    mask = torch.ones(200, 1)
    # nothing before the first slot, and too little after only a few frames
    assert lib._draft_context(draft, mask, 0, 40) is None
    assert lib._draft_context(draft, mask, 8, 40) is None
    frames, valid_from = lib._draft_context(draft, mask, 150, 40)
    assert valid_from == 0 and len(frames) == 2 * 40 + 1          # the last two slots, plus one frame for the rate
    mask[100] = 0                                                  # an unplaced frame ends the context
    frames, _ = lib._draft_context(draft, mask, 150, 40)
    assert len(frames) == 150 - 101
    draft[150:] = 1e6                                              # frames AT and after the slot are never read
    again, _ = lib._draft_context(draft, mask, 150, 40)
    assert np.array_equal(frames, again)


def _dancer(frames=60, seed=0):
    rng = np.random.default_rng(seed)
    j = np.cumsum(rng.normal(0, 0.01, (frames, 24, 3)), axis=0) + rng.normal(0, 0.3, (1, 24, 3))
    j[:, 1] = j[:, 0] + [-0.1, 0.0, 0.0]
    j[:, 2] = j[:, 0] + [0.1, 0.0, 0.0]
    j[:, 7, 2] = j[:, 0, 2] - 0.9
    j[:, 8, 2] = j[:, 0, 2] - 0.85
    return j


def test_frames_do_not_see_where_the_dancer_stands_or_faces():
    j = _dancer()
    base = cs.continuation_frames(j)
    assert base.shape == (60, cs.FRAME_CHANNELS)
    assert np.allclose(cs.continuation_frames(j + np.array([3.0, -2.0, 0.7])), base, atol=1e-5)
    a = np.deg2rad(70.0)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    assert np.allclose(cs.continuation_frames(j @ rot.T), base, atol=1e-5)


def _model():
    torch.manual_seed(0)
    return cs.ContinuationScorer(width=16, blocks=(1, 2), post=(1,), heads=2, dropout=0.0).eval()


def test_scores_have_pool_shape_and_do_not_depend_on_the_rest_of_the_pool():
    model = _model()
    lib = torch.from_numpy(cs.continuation_frames(_dancer(400, seed=1)))
    ctx = cs.continuation_frames(_dancer(120, seed=2))
    starts, natives = [0, 50, 100, 150, 200], [48, 52, 45, 50, 55]
    full = cs.score_pool(model, ctx, lib, starts, natives, 50)
    assert full.shape == (5,)
    alone = cs.score_pool(model, ctx, lib, starts[2:3], natives[2:3], 50)
    assert np.allclose(alone, full[2:3], atol=1e-5)
    reordered = cs.score_pool(model, ctx, lib, starts[::-1], natives[::-1], 50)
    assert np.allclose(reordered, full[::-1], atol=1e-5)


def test_the_seam_mask_hides_the_frames_it_claims_to_hide():
    model = _model()
    lib = cs.continuation_frames(_dancer(400, seed=1))
    ctx = cs.continuation_frames(_dancer(120, seed=2))
    gap = cs.SEAM_MASK
    base = cs.score_pool(model, ctx, torch.from_numpy(lib), [100], [50], 50, gap=gap)
    ctx2 = ctx.copy(); ctx2[-gap:] += 5.0                     # the context's last `gap` frames
    lib2 = lib.copy(); lib2[100:100 + gap] += 5.0             # the candidate's first `gap` frames
    assert np.allclose(cs.score_pool(model, ctx2, torch.from_numpy(lib), [100], [50], 50, gap=gap), base, atol=1e-5)
    assert np.allclose(cs.score_pool(model, ctx, torch.from_numpy(lib2), [100], [50], 50, gap=gap), base, atol=1e-5)
    # ...and with no mask the same change is seen
    unmasked = cs.score_pool(model, ctx, torch.from_numpy(lib), [100], [50], 50, gap=0)
    assert not np.allclose(cs.score_pool(model, ctx2, torch.from_numpy(lib), [100], [50], 50, gap=0), unmasked)


def test_candidates_play_at_the_slot_rate():
    frames = torch.ones(200, cs.FRAME_CHANNELS)
    x, vis, time = cs.sample_candidates(frames, [0, 0], [61, 31], [31, 31], 0)
    rate = x[:, 0, list(cs.RATE_CHANNELS)]
    assert torch.allclose(rate[0], torch.full_like(rate[0], 2.0))     # 61 native frames over a 31-frame slot
    assert torch.allclose(rate[1], torch.ones_like(rate[1]))
    still = [c for c in range(cs.FRAME_CHANNELS) if c not in cs.RATE_CHANNELS]
    assert torch.allclose(x[:, :, still], torch.ones_like(x[:, :, still]))   # positions are not rates
    assert vis.all() and float(time[0, 0]) == 0.0


def test_a_spliced_context_reads_the_other_recording_before_the_splice():
    frames = torch.zeros(300, cs.FRAME_CHANNELS)
    frames[:100, 12] = 1.0          # the other recording (frames 0-99)
    frames[100:, 12] = 2.0          # this recording
    x, vis, _ = cs.sample_context(frames, [250], [40], [100], 0, splice=([230.0], [100.0], [0.0]))
    val = x[0, :, 12]
    t = 250 - 1 + (-80 + torch.linspace(0, 1, cs.CONTEXT_SAMPLES, dtype=torch.float64) * 79 + 1)
    assert torch.all(val[t < 230] == 1.0) and torch.all(val[t >= 230] == 2.0)
    assert vis.all()


def test_a_context_the_mask_hides_entirely_uses_the_checkpoints_speed_scale():
    """Training falls back to the checkpoint's ``scale_ref`` when no context frame is visible; inference must too
    (it used 1.0, a speed scale no training example ever had)."""
    model = _model()
    lib = torch.from_numpy(cs.continuation_frames(_dancer(400, seed=1)))
    ctx = cs.continuation_frames(_dancer(10, seed=2))              # 10 frames, all inside a 12-frame mask
    a = cs.score_pool(model, ctx, lib, [0, 100], [50, 50], 50, gap=12, scale_ref=0.02)
    b = cs.score_pool(model, ctx, lib, [0, 100], [50, 50], 50, gap=12, scale_ref=0.05)
    assert not np.allclose(a, b)
    c_, cvis, ctime = cs.sample_context(torch.as_tensor(ctx), [10], [50], [0], 12)
    assert not cvis.any()
    cand, kvis, ktime = cs.sample_candidates(lib, [0, 100], [50, 50], [50, 50], 12)
    x, k = cs.assemble(c_, cvis, ctime, cand[None], kvis[None], ktime[None], torch.tensor([0.02]),
                       model.in_mean, model.in_std)
    assert np.allclose(model(x, k)[0].detach().numpy(), a, atol=1e-5)


def test_the_draft_context_needs_frames_beyond_the_scorers_mask(monkeypatch):
    monkeypatch.setattr(infer_atomic, "decode_motion", _fake_decode)
    lib = Lib.__new__(Lib)
    lib.normalizer_path = None
    draft = torch.arange(200, dtype=torch.float32)[:, None].repeat(1, 4)
    mask = torch.zeros(200, 1)
    mask[100:] = 1
    assert lib._draft_context(draft, mask, 112, 40) is not None          # gap 6: 12 placed frames suffice
    assert lib._draft_context(draft, mask, 112, 40, gap=12) is None      # gap 12: all 12 would be masked
    assert lib._draft_context(draft, mask, 118, 40, gap=12) is not None


def _reach_corpus():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
    import train_continuation_scorer as T
    frames = np.zeros((600, cs.FRAME_CHANNELS), np.float32)
    frames[:300, 12] = 1.0                                   # recording A (upload a)
    frames[300:, 12] = 2.0                                   # recording B (upload b)
    meta = {"recs": [{"rec": "wild_v5:a:clip000", "upload": "a", "offset": 0, "length": 300,
                      "bars": [(20, 60), (60, 100), (100, 140), (140, 180), (180, 220)]},
                     {"rec": "wild_v5:b:clip000", "upload": "b", "offset": 300, "length": 300,
                      "bars": [(20, 60), (60, 100), (100, 140), (140, 180)]}]}
    return T, T.Corpus(frames, meta, "cpu")


@pytest.mark.parametrize("reach", [None, ("visible", 6), ("visible", 30), ("hide", 10), ("prev",), ("splice",)])
def test_the_reach_modes_change_only_what_the_context_shows(monkeypatch, reach):
    """The reach check (train_continuation_scorer.final_checks) reads how far back the scorer looks by showing it
    less of the SAME context.  Each mode must hide exactly what it claims, and never touch the candidates."""
    T, c = _reach_corpus()
    seen = {}

    def spy(model, b, zero_ctx=False, scale_ref=None):
        seen.update(b)
        return torch.zeros(b["cand"].shape[:2])

    monkeypatch.setattr(T, "logits", spy)
    g = 3                                                    # recording A's bar (140, 180); context = bars 1-2
    row = {"g": g, "rec": 0, "slot": 40, "cands": [("rand", 300 + 100, 40.0)]}
    gap = 6
    T.model_scores(torch.nn.Identity(), c, [row], gap, "cpu", reach=reach, rng=np.random.default_rng(0))
    vis = seen["cvis"][0].numpy()
    pos = (seen["ctime"][0].numpy() * 40)                    # slot frames before the bar line (negative)
    val = seen["ctx"][0, :, 12].numpy()
    near = pos > -1 - gap - 1e-6
    assert not vis[near].any()                               # the seam mask always holds
    if reach is None:
        assert vis[~near].all()
    elif reach[0] == "visible":
        assert np.array_equal(vis, (pos >= -gap - reach[1] - 1e-6) & ~near)
    elif reach[0] == "hide":
        assert np.array_equal(vis, pos <= -1 - gap - reach[1] + 1e-6)
    elif reach[0] == "prev":
        assert np.array_equal(vis, (pos >= -40 - 1e-6) & ~near)
    else:                                                    # splice: the second bar is another upload's
        assert vis[~near].all()
        assert np.all(val[pos < -40 + 1e-6] == 2.0) and np.all(val[pos > -40 + 1.5] == 1.0)
    assert seen["kvis"].shape == (1, 2, cs.CANDIDATE_SAMPLES) and seen["kvis"][0, :, cs.CANDIDATE_SAMPLES // 2].all()


def test_the_reach_verdict_can_say_either_thing():
    """The verdict is a rule that can fail: a second-bar gain that is not significant on EVERY T pool makes the
    checkpoint a previous-bar scorer, and its reach is the fewest visible frames recovering 90% of the full AUC."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
    import train_continuation_scorer as T

    def pool(gain_p, mean, vis):
        auc = {"full": {"rand": 0.85}}
        auc.update({"vis{}".format(v): {"rand": a} for v, a in zip(T.REACH_VISIBLE, vis)})
        return {"reach": {"second_bar": {"rand": (130, 231, gain_p, mean)}, "auc": auc}}

    ok = {"t2": pool(0.001, 0.04, (0.70, 0.78, 0.83)), "t1": pool(0.01, 0.036, (0.70, 0.78, 0.83))}
    assert T.reach_verdict(ok, ["t2", "t1"])["second_bar_used"]
    # significant but a small share of what the scorer reads (0.01 of 0.35 above chance): not phrase-level
    small = {"t2": pool(0.001, 0.04, (0.70, 0.78, 0.83)), "t1": pool(0.001, 0.01, (0.70, 0.78, 0.83))}
    assert not T.reach_verdict(small, ["t2", "t1"])["second_bar_used"]
    one_fails = {"t2": pool(0.001, 0.02, (0.70, 0.78, 0.83)), "t1": pool(0.3, 0.01, (0.70, 0.78, 0.83))}
    v = T.reach_verdict(one_fails, ["t2", "t1"])
    assert not v["second_bar_used"] and v["frames_for_90pct"] == 30          # 0.83 - 0.5 >= 0.9 * 0.35
    near = {"t2": pool(0.5, -0.001, (0.70, 0.82, 0.84)), "t1": pool(0.5, 0.0, (0.70, 0.82, 0.84))}
    assert T.reach_verdict(near, ["t2", "t1"])["frames_for_90pct"] == 10     # 0.82 - 0.5 >= 0.315
    negative = {"t2": pool(0.001, -0.02, (0.6, 0.6, 0.6)), "t1": pool(0.001, 0.02, (0.6, 0.6, 0.6))}
    v = T.reach_verdict(negative, ["t2", "t1"])
    assert not v["second_bar_used"] and v["frames_for_90pct"] is None and "more than 30" in v["text"]
