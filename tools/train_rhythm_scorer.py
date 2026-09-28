#!/usr/bin/env python3
"""Train ``model.rhythm_scorer`` on a release's own (music, motion) pairs and run the checks that decide
whether it may be believed.  See the model's docstring for WHY and for the leak that shapes the data.

One training example = one music crop and 1 + K motion crops over it:
  * the POSITIVE: the same recording's motion over the same frames;
  * SHIFT negatives: the same recording's motion shifted by 4-12 frames (a quarter to three quarters of a
    beat) -- identical content, wrong alignment;
  * CONTENT negatives: other recordings' motion (a different retrieval group) at a random offset.
Music and the positive/shift motion are cropped at the SAME random offset; every crop of an example (music
and all 16 motion crops) is resampled by ONE random tempo factor in [0.87, 1.15] (the retrieval stretch
range) to CROP frames, so interpolation smoothing cannot mark the positive.  Nothing is anchored to a bar
line.

CHECKS on the held-out split (the release's val recordings), printed every eval and stored in the
checkpoint:
  acc16   the positive ranked first among all 16
  shift   the positive above all its shift negatives (chance 1/(S+1))
  content the positive above all content negatives (chance 1/(C+1))
  shift@music0  the same with the music replaced by its mean: MUST fall to chance, or the scorer found the
                shift in the motion alone (the bar-line leak, or another)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.rhythm_scorer import (CROP, KIN_CHANNELS, RhythmScorer, kinetic_frames,  # noqa: E402
                                 normalise_kinetics, resample)

SHIFTS = 4
CONTENT = 11
TEMPO = (0.87, 1.15)
SHIFT_RANGE = (4, 12)


def load_split(root, split, cache_dir, plane=False):
    """kinetics [N, 150, KIN], music [N, 150, 35], group index [N] for one release split."""
    cache = pathlib.Path(cache_dir) / "{}_{}_kinetics{}.npy".format(pathlib.Path(root).name, split,
                                                                "_plane" if plane else "")
    motion = np.load(root / split / "motion.npy", mmap_mode="r")
    music = np.asarray(np.load(root / split / "music.npy", mmap_mode="r"), dtype=np.float32)
    if cache.is_file():
        kin = np.load(cache)
    else:
        from infer_atomic import decode_motion
        kin = np.zeros(motion.shape[:2] + (KIN_CHANNELS,), dtype=np.float32)
        for first in range(0, len(motion), 64):
            block = torch.from_numpy(np.array(motion[first:first + 64], dtype=np.float32))
            joints = decode_motion(block.reshape(-1, block.shape[-1]), str(root / "normalizer.pt"))["full_pose"]
            joints = np.asarray(joints).reshape(len(block), motion.shape[1], 24, 3)
            kin[first:first + len(block)] = kinetic_frames(joints, plane=plane)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, kin)
    groups, sequences = {}, {}
    for line in open(root / "windows.jsonl"):
        w = json.loads(line)
        if w.get("split") == split:
            groups[int(w["array_index"])] = w["retrieval_group_id"]
            sequences[int(w["array_index"])] = w.get("sequence_id") or w.get("recording_id")
    names = sorted(set(groups.values()))
    lookup = {n: i for i, n in enumerate(names)}
    gid = np.array([lookup[groups[i]] for i in range(len(motion))])
    seq = np.array([sequences[i] for i in range(len(motion))], dtype=object)
    return kin, music, gid, names, seq


class Sampler:
    def __init__(self, kin, music, gid, rng, keep_groups=None, windows=None):
        self.kin, self.music, self.gid, self.rng = kin, music, gid, rng
        self.index = np.arange(len(kin)) if keep_groups is None else \
            np.flatnonzero(np.isin(gid, np.asarray(sorted(keep_groups))))
        if windows is not None:
            self.index = np.intersect1d(self.index, windows)
        self.frames = kin.shape[1]

    def batch(self, size):
        """[B, CROP, 35] music and [B, 1+S+C, CROP, KIN] normalised kinetics; candidate 0 is the positive.
        All 16 crops of an example share ONE tempo factor, so interpolation smoothing is identical across
        them and cannot mark the positive."""
        rng = self.rng
        music_out, kin_out = [], []
        for _ in range(size):
            w = self.index[rng.integers(len(self.index))]
            length = int(round(CROP * rng.uniform(*TEMPO)))
            lo = SHIFT_RANGE[1]
            t = int(rng.integers(lo, self.frames - length - lo + 1))
            crops = [self.kin[w, t:t + length]]
            for _k in range(SHIFTS):
                d = int(rng.integers(SHIFT_RANGE[0], SHIFT_RANGE[1] + 1)) * (1 if rng.random() < 0.5 else -1)
                crops.append(self.kin[w, t + d:t + d + length])
            for _k in range(CONTENT):
                while True:
                    w2 = self.index[rng.integers(len(self.index))]
                    if self.gid[w2] != self.gid[w]:
                        break
                t2 = int(rng.integers(0, self.frames - length + 1))
                crops.append(self.kin[w2, t2:t2 + length])
            kin_out.append(resample(np.stack(crops), CROP))
            music_out.append(resample(self.music[w, t:t + length][None].copy(), CROP)[0])
        music_t = torch.stack(music_out)
        kin_t = normalise_kinetics(torch.cat(kin_out))
        return music_t, kin_t.reshape(size, 1 + SHIFTS + CONTENT, CROP, -1)


def scores(model, music, kin):
    b, k = kin.shape[:2]
    rep = music[:, None].expand(b, k, *music.shape[1:]).reshape(b * k, *music.shape[1:])
    return model(rep, kin.reshape(b * k, *kin.shape[2:])).reshape(b, k)


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    out = {"acc16": [], "shift": [], "content": [], "shift@music0": [], "content@music0": []}
    for music, kin in batches:
        music, kin = music.to(device), kin.to(device)
        for tag, m in (("", music), ("@music0", model.music_mean.expand_as(music))):
            s = scores(model, m, kin)
            pos = s[:, :1]
            shift_ok = (pos > s[:, 1:1 + SHIFTS]).all(1).float()
            content_ok = (pos > s[:, 1 + SHIFTS:]).all(1).float()
            if tag == "":
                out["acc16"].append((s.argmax(1) == 0).float())
            out["shift" + tag].append(shift_ok)
            out["content" + tag].append(content_ok)
    model.train()
    return {k: float(torch.cat(v).mean()) for k, v in out.items()}


def main():
    global SHIFTS, CONTENT, SHIFT_RANGE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default="/cache/atomicdance-assets/runs/rhythm_scorer_cache")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--shifts", type=int, default=SHIFTS, help="shift negatives per example")
    ap.add_argument("--content", type=int, default=CONTENT, help="other-recording negatives per example")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--music-channels", default=None,
                    help="comma list of music channels the scorer may read; default all 35. "
                         "0,33,34 = onset envelope, onset peaks, beats (rhythm only)")
    ap.add_argument("--exclude", default=None, help="json {recordings, uploads} never to train on")
    ap.add_argument("--plane", action="store_true",
                    help="kinetics without the depth axis (what a front-view render shows)")
    ap.add_argument("--shift-min", type=int, default=SHIFT_RANGE[0],
                    help="smallest shift negative, frames (finer = sharper phase sensitivity)")
    ap.add_argument("--extra-eval-root", default=None,
                    help="a second release whose val split is also scored (e.g. the T line's)")
    ap.add_argument("--train-groups", default=None,
                    help="optional file of retrieval_group_ids to train on (library-swap test)")
    args = ap.parse_args()
    SHIFTS, CONTENT = args.shifts, args.content
    SHIFT_RANGE = (args.shift_min, SHIFT_RANGE[1])
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root = pathlib.Path(args.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kin, music, gid, names, seq = load_split(root, "train", args.cache_dir, args.plane)
    vkin, vmusic, vgid, _, _ = load_split(root, "val", args.cache_dir, args.plane)
    keep = None
    if args.train_groups:
        wanted = {l.strip() for l in open(args.train_groups) if l.strip()}
        keep = [names.index(n) for n in wanted if n in names]
        print("training on {} of {} recordings".format(len(keep), len(names)))
    windows = None
    if args.exclude:
        # the evaluation clips, every clip of their uploads, and every recording that shares their SONG
        # (music fingerprint pairs): the scorer must never have seen the test music or the test dance
        ex = json.load(open(args.exclude))
        bad_groups = {"wild_v5:" + u for u in ex["uploads"]}
        bad_seq = set(ex["recordings"])
        mask = np.array([names[g] not in bad_groups and s_ not in bad_seq for g, s_ in zip(gid, seq)])
        windows = np.flatnonzero(mask)
        print("exclusion: {} of {} training windows dropped".format(int((~mask).sum()), len(mask)))
    sampler = Sampler(kin, music, gid, rng, keep, windows=windows)
    # The held-out batches always use the standard mix (4 shift + 11 content), whatever this run trains on,
    # so every variant is read on the same questions.
    train_mix = (SHIFTS, CONTENT)
    SHIFTS, CONTENT = 4, 11
    vbatches = [Sampler(vkin, vmusic, vgid, np.random.default_rng(1000 + i)).batch(128) for i in range(12)]
    tbatches = [Sampler(kin, music, gid, np.random.default_rng(2000 + i), keep, windows).batch(128)
                for i in range(4)]
    xbatches = None
    if args.extra_eval_root:
        xk, xm, xg, _, _ = load_split(pathlib.Path(args.extra_eval_root), "val", args.cache_dir, args.plane)
        xbatches = [Sampler(xk, xm, xg, np.random.default_rng(3000 + i)).batch(128) for i in range(12)]
    SHIFTS, CONTENT = train_mix
    channels = [int(c) for c in args.music_channels.split(",")] if args.music_channels else None
    model = RhythmScorer(width=args.width, music_channels=channels).to(device)
    flat = music.reshape(-1, music.shape[-1])
    model.music_mean.copy_(torch.as_tensor(flat.mean(0)))
    model.music_std.copy_(torch.as_tensor(flat.std(0) + 1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    history, best = [], None
    start = time.time()
    for step in range(1, args.steps + 1):
        m, k = sampler.batch(args.batch)
        s = scores(model, m.to(device), k.to(device))
        loss = F.cross_entropy(s, torch.zeros(len(s), dtype=torch.long, device=device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            mix = (SHIFTS, CONTENT)
            SHIFTS, CONTENT = 4, 11
            ev = evaluate(model, vbatches, device)
            ev.update({"train_" + k: v for k, v in evaluate(model, tbatches, device).items()
                       if "@" not in k})
            if xbatches is not None:
                ev.update({"T_" + k: v for k, v in evaluate(model, xbatches, device).items()})
            SHIFTS, CONTENT = mix
            ev.update(step=step, loss=float(loss), seconds=round(time.time() - start))
            history.append(ev)
            print(json.dumps(ev), flush=True)
            if best is None or ev["shift"] + ev["content"] > best["shift"] + best["content"]:
                best = ev
                torch.save({"stage": "rhythm_scorer", "kinetics": "plane" if args.plane else "body3d",
                            "arch": {"width": args.width, "music_channels": channels},
                            "state_dict": model.state_dict(), "args": vars(args), "val": ev,
                            "history": history, "data_root": str(root),
                            "chance": {"acc16": 1 / (1 + SHIFTS + CONTENT), "shift": 1 / (1 + SHIFTS),
                                       "content": 1 / (1 + CONTENT)}}, args.out)
    print("best", json.dumps(best))


if __name__ == "__main__":
    main()
