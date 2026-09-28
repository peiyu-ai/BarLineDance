#!/usr/bin/env python3
"""Choose the planner's guidance weight on val, so test is not tuned on.

Classifier-free guidance gives the planner a knob with real leverage -- unlike
temperature, which is nearly cancelled by the reverse posterior's likelihood
term (T 1.0 -> 0.2 moved the transition share only 0.5263 -> 0.5178).  Measured
on the 2026-08-23 arms at epoch 36, raising w from 1.0 to 2.0 moved the mean
share 0.429 -> 0.197 against a ground truth of 0.322.

That leverage is exactly why the weight cannot be read off a test table.  It is
fitted here on val clips that share no recording with the test set.

Two criteria, reported together and both able to fail:

* ``share_mae`` -- mean over clips of |plan transition - that clip's own ground
  truth transition|.  Two-sided on purpose.  An earlier one-sided version of
  this gate counted only clips *above* a threshold, and the arm it crowned
  (62 "normal", 1 "failed") was emitting 0.077 transition against a ground
  truth of 0.322 -- further from the truth than the baseline it beat, in the
  mirror direction.
* ``segment_ratio`` -- planned atomic segments per clip over the ground truth's.
  Guidance buys a lower transition share partly by fragmenting: on the same
  arms the distinct-class count went 5.1 -> 8.5 -> 13.2 as w rose.  A weight
  that fixes the share by shattering the plan has not fixed anything, and only
  a second criterion can see it.

The chosen weight minimises ``share_mae`` among the weights whose
``segment_ratio`` stays within ``--max-segment-ratio``; if none qualify the
tool says so and returns the unconstrained minimiser marked as such, rather
than silently relaxing its own constraint.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.atomic import labels_to_segments, refine_plan  # noqa: E402
from infer_atomic import (_fuse_windows, _load_music, _pad_frames,  # noqa: E402
                          _window_starts)
from tools.eval_planner_checkpoint import load_planner  # noqa: E402


def plan_for(planner, args_, music, device, guidance, seed):
    from train_atomic import seed_everything

    window, stride = args_.seq_len, 15
    starts = _window_starts(len(music), window, stride)
    chunks, lengths = [], []
    for start in starts:
        chunk = music[start : start + window]
        lengths.append(len(chunk))
        chunks.append(_pad_frames(chunk, window))
    batch = torch.stack(chunks).to(device)
    padding = torch.arange(window, device=device)[None] >= torch.tensor(
        lengths, device=device)[:, None]
    seed_everything(seed)
    sampled = planner.sample(batch, padding_mask=padding, deterministic=False,
                             guidance_weight=guidance).cpu()
    fused = _fuse_windows(sampled.masked_fill(padding.cpu(), 0), starts, lengths,
                          len(music), window, "vote", tie_break="centre")
    return refine_plan(fused, 5, 6, transition_policy="protect")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--clip-list", type=pathlib.Path, required=True,
                        help="val clips; must share no recording with the test set")
    parser.add_argument("--audio-dir", type=pathlib.Path, required=True)
    parser.add_argument("--label-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--weights", type=float, nargs="+",
                        default=[1.0, 1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--max-segment-ratio", type=float, default=1.5,
                        help="reject a weight whose plan names this many times "
                             "the ground truth's atomic segments")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    truth = {}
    with args.label_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            truth[row["sequence_id"]] = row["labels_path"]
    root = args.label_manifest.parent
    clips = [c.strip() for c in args.clip_list.read_text(encoding="utf-8").splitlines()
             if c.strip()]

    device = torch.device(args.device)
    planner, train_args, _ = load_planner(args.checkpoint, device)
    guided = getattr(planner.model, "null_music", None) is not None
    if not guided and any(w != 1.0 for w in args.weights):
        raise SystemExit(
            "this planner has no null condition, so only weight 1.0 is defined; "
            "retrain with --planner-cond-drop-prob > 0")

    music_by_clip, gt = {}, {}
    for clip in clips:
        path = args.audio_dir / (clip + ".npy")
        if not path.is_file() or clip not in truth:
            continue
        music_by_clip[clip] = _load_music(path, None, train_args.music_dim)
        labels = torch.from_numpy(np.load(str(root / truth[clip])).astype(np.int64))
        gt[clip] = (float((labels == 0).float().mean()),
                    len([s for s in labels_to_segments(labels) if s.label]))
    if not music_by_clip:
        raise SystemExit("no val clip had both music and labels")

    report = {"checkpoint": str(args.checkpoint), "clips": len(music_by_clip),
              "seed": args.seed, "max_segment_ratio": args.max_segment_ratio,
              "by_weight": {}}
    print("{:>8} {:>10} {:>11} {:>14} {:>9}".format(
        "weight", "share", "share_mae", "segment_ratio", "eligible"))
    for weight in args.weights:
        errors, ratios, shares = [], [], []
        for clip, music in music_by_clip.items():
            plan = plan_for(planner, train_args, music, device, weight, args.seed)
            share = float((plan == 0).float().mean())
            segments = len([s for s in labels_to_segments(plan) if s.label])
            shares.append(share)
            errors.append(abs(share - gt[clip][0]))
            ratios.append(segments / max(gt[clip][1], 1))
        row = {"share": float(np.mean(shares)), "share_mae": float(np.mean(errors)),
               "segment_ratio": float(np.median(ratios))}
        row["eligible"] = row["segment_ratio"] <= args.max_segment_ratio
        report["by_weight"][str(weight)] = row
        print("{:>8} {:>10.4f} {:>11.4f} {:>14.2f} {:>9}".format(
            weight, row["share"], row["share_mae"], row["segment_ratio"],
            "yes" if row["eligible"] else "no"))

    eligible = {w: r for w, r in report["by_weight"].items() if r["eligible"]}
    pool = eligible or report["by_weight"]
    chosen = min(pool, key=lambda w: pool[w]["share_mae"])
    report["fitted_weight"] = float(chosen)
    report["constraint_satisfied"] = bool(eligible)
    print("\nfitted on val: guidance weight = {}{}".format(
        chosen, "" if eligible else "  (NO weight met the segment-ratio "
                                    "constraint; this is the unconstrained minimiser)"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
