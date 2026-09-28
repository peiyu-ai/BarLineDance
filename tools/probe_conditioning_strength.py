#!/usr/bin/env python3
"""What decides the completion model's output -- the conditioning, or the noise?

WHY.  On 2026-09-01 the operator asked why a GROUND-TRUTH plan still renders a
poor dance, and whether the fault is in label->motion.  A prior measurement,
recorded in model/atomic_completion.py's own comment, says the generated joint
rotations sit only 4-5% closer to the retrieved prototype than to a SHUFFLED
ordering of that prototype's own frames.  A second, measured today: re-running
one window with the same music and the same draft and only a different noise
draw moves the joints 0.35 m RMS, which is 12x the clip's own per-frame motion.

Those two say the same thing from two sides, and this makes it a decision:
hold everything fixed and change exactly ONE thing at a time, then compare how
far the output moves.

    D_noise   same music, same draft, different noise
    D_draft   same music, DIFFERENT draft (another clip's), same noise
    D_music   DIFFERENT music (another clip's), same draft, same noise

If D_draft and D_music are not clearly larger than D_noise, the conditioning is
not what the model is following, and no improvement to the plan -- including a
perfect ground-truth plan -- can reach the output.  That is a falsifiable
statement and it is the point of the probe.

CONTROL, so the scale is not free-floating: D_self, the same window re-run with
EVERYTHING identical including the noise seed, must read ~0.  A probe that
cannot show zero when nothing changed cannot be trusted when something does.
All distances are joint positions in metres via MotionGeometry, root-relative,
so a global translation does not inflate them.  No within-clip normalization
(docs/DANCE_QUALITY_DEFECTS.md 12.1).
"""
import argparse, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import infer_atomic as IA
from model.atomic_completion import MotionGeometry

p = argparse.ArgumentParser()
p.add_argument("--planner", required=True); p.add_argument("--completion", required=True)
p.add_argument("--data-root", required=True); p.add_argument("--audio-dir", required=True)
p.add_argument("--clips", required=True); p.add_argument("--limit", type=int, default=6)
p.add_argument("--guidance-weight", type=float, default=2.0)
p.add_argument("--draft-guidance-weight", type=float, default=None,
               help="amplify the plan; needs a --draft-drop-prob checkpoint. "
                    "1.0 reproduces the shipped two-term guidance exactly.")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
comp, cargs = IA._load_checkpoint(a.completion, "completion", dev)
plan, pargs = IA._load_checkpoint(a.planner, "planner", dev)
lib = IA.IndexedAtomicMotionLibrary(a.data_root)
geom = MotionGeometry(torch.load(str(pathlib.Path(a.data_root)/"normalizer.pt"), map_location="cpu")).to(dev).eval()
clips = [l.strip() for l in open(a.clips) if l.strip()][: a.limit]
W = cargs.seq_len

def prep(clip):
    music = IA._load_music(str(pathlib.Path(a.audio_dir)/(clip+".npy")), None, pargs.music_dim)
    IA.seed_everything(IA.sample_seed(20260816, clip))
    lab = IA.infer_plan(plan, music, pargs.seq_len, dev, deterministic=False,
                        temperature=1.0, plan_stride=15, plan_fusion="vote")[: len(music)]
    draft, mask = IA._source_safe_draft(lib, lab, cargs.motion_dim, clip, False)
    return music, draft, mask

def run(music, draft, mask, seed):
    torch.manual_seed(seed)
    m = IA._pad_frames(music[:W], W)[None].to(dev)
    d = IA._pad_frames(draft[:W], W)[None].to(dev)
    k = IA._pad_frames(mask[:W], W)[None].to(dev)
    with torch.no_grad():
        return comp.sample(m, d, k, guidance_weight=a.guidance_weight,
                           draft_guidance_weight=a.draft_guidance_weight)[0].cpu()

def J(x):
    with torch.no_grad():
        j = geom.joints(x[None].to(dev))[0].cpu().numpy()
    return j - j[:, :1]

def dist(x, y):
    return float(np.sqrt(((J(x) - J(y)) ** 2).sum(-1).mean()))

cache = {c: prep(c) for c in clips}
rows = []
for i, c in enumerate(clips):
    mu, dr, mk = cache[c]
    if len(mu) < W: continue
    other = clips[(i + 1) % len(clips)]
    mu2, dr2, mk2 = cache[other]
    if len(mu2) < W: continue
    base = run(mu, dr, mk, 4242)
    d_self  = dist(base, run(mu, dr, mk, 4242))
    d_noise = dist(base, run(mu, dr, mk, 918273))
    d_draft = dist(base, run(mu, dr2, mk2, 4242))
    d_music = dist(base, run(mu2, dr, mk, 4242))
    motion = float(np.linalg.norm(J(base)[1:] - J(base)[:-1], axis=-1).mean())
    rows.append((d_self, d_noise, d_draft, d_music, motion))
    print("{:<20} 同一切 {:.4f} | 换噪声 {:.4f} | 换草稿 {:.4f} | 换音乐 {:.4f} | 逐帧位移 {:.4f}"
          .format(c.split(":")[1][:18], d_self, d_noise, d_draft, d_music, motion))

r = np.array(rows)
print("\n{} clips  guidance={} draft_guidance={}  (米, 去根关节位置 RMS)"
      .format(len(rows), a.guidance_weight, a.draft_guidance_weight))
print("  D_self   完全不变        {:.4f}   <- 必须 ~0, 否则探针不可信".format(r[:,0].mean()))
print("  D_noise  只换噪声        {:.4f}".format(r[:,1].mean()))
print("  D_draft  只换草稿(计划)  {:.4f}   = D_noise 的 {:.2f}x".format(r[:,2].mean(), r[:,2].mean()/max(r[:,1].mean(),1e-9)))
print("  D_music  只换音乐        {:.4f}   = D_noise 的 {:.2f}x".format(r[:,3].mean(), r[:,3].mean()/max(r[:,1].mean(),1e-9)))
print("  动作自身逐帧位移         {:.4f}".format(r[:,4].mean()))
import json, os
if os.environ.get("PROBE_JSON"):
    json.dump({"guidance_weight": a.guidance_weight,
               "draft_guidance_weight": a.draft_guidance_weight,
               "checkpoint": a.completion, "clips": len(rows),
               "d_self": float(r[:,0].mean()), "d_noise": float(r[:,1].mean()),
               "d_draft": float(r[:,2].mean()), "d_music": float(r[:,3].mean()),
               "motion": float(r[:,4].mean()),
               "draft_over_noise": float(r[:,2].mean()/max(r[:,1].mean(),1e-9)),
               "music_over_noise": float(r[:,3].mean()/max(r[:,1].mean(),1e-9))},
              open(os.environ["PROBE_JSON"], "w"), indent=2)
    print(os.environ["PROBE_JSON"])
