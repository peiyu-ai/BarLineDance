#!/usr/bin/env python3
"""How much of the plan's transition share is the vote fusion's tie-break?

``tools/probe_plan_fusion.py`` (2026-08-18) established that
``--plan-fusion vote`` roughly quadrupled the transition share on the wild
w2_340 planner (9.6% no-fusion -> 38.5% vote, ground truth 13.5%) and named the
mechanism: at stride 15 about twenty windows cover each frame, each an
independent draw over K+1 classes, and transition holds a large enough share of
the mass to win most pluralities.  That explanation is right but incomplete, and
the missing half is mechanical rather than statistical:

``infer_atomic._fuse_windows(mode="vote")`` ends in ``counts.argmax(dim=1)``.
``torch.argmax`` returns the *first* maximal index, and transition is index 0,
so **every tie transition is part of goes to transition by construction** -- not
because it won the vote, but because it sorts first.  ``dataset.atomic
.majority_vote`` already treats this as something that must be decided rather
than inherited (it keeps the incumbent centre label on a tie, and says so); the
fusion path never got the same treatment.

Ties are common here because the counts are small: with ~7 windows covering a
frame and no window confident, the modal count is 2-6, so exact ties between the
top two classes are frequent rather than exotic.

Measured on clean5b5 ``planner_v1A_s15/planner_step135648.pt``, the 33 clips of
``runs/m6_clean5b5_stoch_vote/seed_20260816_g0``, seed 20260816:

    ground truth                                       0.2903
    raw per-window draws (no fusion)                   0.3865
    centre fusion                                      0.4217
    vote, transition must win outright                 0.4648
    vote, uniform tie-break among winners              0.4844
    vote, argmax tie-break  <- shipped until 2026-08-23 0.5106

So of the 22.0 points by which the shipped plan exceeds the ground truth,
9.6 are the planner, 7.8 are plurality voting, and **4.6 are the tie-break**:
9.0% of all planned frames are transition only because 0 sorts before every
other label.  7.28% of frames are ties at all, and 62.8% of those ties have
transition among the winners.

Acted on 2026-08-23: ``_fuse_windows`` now takes ``tie_break``, defaulting to
``"centre"`` -- among the tied classes, the one voted by the window holding the
frame most centrally.  That is the ``vote, uniform tie-break`` row's intent
without the randomness, so this probe's ``fair`` column is the estimate of where
the shipped default now sits.  ``--plan-vote-tie-break index`` reproduces the
old row.

This probe is a measurement, not a gate: it does not say which fusion is right
(``tools/probe_plan_fusion.py`` measured that vote's higher transition share
*improves* fid_k, which is its own problem).  It says how much of one arm's
behaviour is arithmetic that nobody chose.

Usage::

    python3 tools/probe_plan_vote_ties.py \\
        --checkpoint /dev/shm/.../planner_v1A_s15/planner_step135648.pt \\
        --audio-dir runs/wild_v4_acct_gt_eval/audio \\
        --clip-list runs/clean5b5_m6_clips.txt \\
        --label-manifest /dev/shm/.../ingroup_llm_v1/labels.jsonl \\
        --output runs/clean5b5_plan_vote_ties.json
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
from infer_atomic import (  # noqa: E402
    _load_music,
    _pad_frames,
    _window_starts,
    planner_wants_global_music,
    track_summary,
)
from tools.eval_planner_checkpoint import load_planner  # noqa: E402


def draw_windows(planner, music, window, stride, device, seed, temperature=1.0,
                 deterministic=False):
    """The per-window samples ``infer_plan`` fuses, before any fusion."""
    from train_atomic import seed_everything

    starts = _window_starts(len(music), window, stride)
    chunks, lengths = [], []
    for start in starts:
        chunk = music[start : start + window]
        lengths.append(len(chunk))
        chunks.append(_pad_frames(chunk, window))
    batch = torch.stack(chunks).to(device)
    padding = torch.arange(window, device=device)[None] >= torch.tensor(
        lengths, device=device)[:, None]
    extra = {}
    if planner_wants_global_music(planner):
        extra["global_music"] = track_summary(music).to(device)[None].expand(len(chunks), -1)
    seed_everything(seed)
    sampled = planner.sample(batch, padding_mask=padding, temperature=temperature,
                             deterministic=deterministic, **extra).cpu()
    return sampled, starts, lengths


def vote_counts(sampled, starts, lengths, total, classes):
    counts = torch.zeros(total, classes, dtype=torch.int32)
    for row, start, length in zip(sampled, starts, lengths):
        index = torch.arange(start, start + length)
        counts[index, row[:length].to(torch.long)] += 1
    return counts


def centre_fuse(sampled, starts, lengths, total):
    labels = torch.zeros(total, dtype=sampled.dtype)
    best = torch.full((total,), float("inf"))
    for row, start, length in zip(sampled, starts, lengths):
        index = torch.arange(start, start + length)
        centre = start + (length - 1) / 2.0
        distance = (index.to(torch.float32) - centre).abs()
        take = distance < best[index]
        labels[index[take]] = row[:length][take]
        best[index[take]] = distance[take]
    return labels


def tie_break_variants(counts, seed):
    """Four readings of the same counts, differing only where there is a tie."""
    top = counts.max(dim=1).values
    winners = counts == top[:, None]
    n_winners = winners.sum(dim=1)
    generator = torch.Generator().manual_seed(seed)
    jitter = torch.rand(counts.shape, generator=generator) + 1e-6

    argmax = counts.argmax(dim=1)                       # ships today: index order
    fair = (winners.float() * jitter).argmax(dim=1)     # uniform among winners
    no_zero = winners.clone()
    no_zero[:, 0] &= n_winners == 1                     # transition must win outright
    outright = torch.where(n_winners == 1, argmax,
                           (no_zero.float() * jitter).argmax(dim=1))
    return {
        "argmax": argmax, "fair": fair, "outright": outright,
        "_tied": n_winners > 1, "_zero_wins_tie": (n_winners > 1) & winners[:, 0],
        "_top_count": top,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--audio-dir", type=pathlib.Path, required=True)
    parser.add_argument("--clip-list", type=pathlib.Path, required=True,
                        help="one recording id per line, or a manifest.json with names")
    parser.add_argument("--label-manifest", type=pathlib.Path, required=True,
                        help="labels.jsonl giving each clip's ground-truth plan")
    parser.add_argument("--plan-stride", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--vote-window", type=int, default=5)
    parser.add_argument("--min-segment-length", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    if args.clip_list.suffix == ".json":
        names = json.loads(args.clip_list.read_text(encoding="utf-8"))
        names = names["names"] if isinstance(names, dict) else names
    else:
        names = [line.strip() for line in
                 args.clip_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_clips:
        names = names[: args.max_clips]

    label_root = args.label_manifest.parent
    truth_path = {}
    with args.label_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            truth_path[row["sequence_id"]] = row["labels_path"]
    missing = [n for n in names if n not in truth_path]
    if missing:
        raise SystemExit("error: no ground-truth plan for {}".format(missing[:3]))

    device = torch.device(args.device)
    planner, train_args, _ = load_planner(args.checkpoint, device)
    window = train_args.seq_len

    rows = []
    for name in names:
        music = _load_music(args.audio_dir / (name + ".npy"), None, train_args.music_dim)
        total = len(music)
        sampled, starts, lengths = draw_windows(
            planner, music, window, args.plan_stride, device, args.seed)
        counts = vote_counts(sampled, starts, lengths, total, planner.num_classes)
        variants = tie_break_variants(counts, args.seed)
        variants["centre"] = centre_fuse(sampled, starts, lengths, total)
        truth = np.load(str(label_root / truth_path[name]))

        row = {
            "name": name,
            "frames": int(total),
            "windows": len(starts),
            "gt_transition": float((truth == 0).mean()),
            "raw_window_transition": float((sampled == 0).float().mean()),
            "median_top_count": float(variants["_top_count"].float().median()),
            "tied_frames": float(variants["_tied"].float().mean()),
            "tied_frames_with_transition_among_winners":
                float(variants["_zero_wins_tie"].float().mean()),
        }
        for tag in ("centre", "outright", "fair", "argmax"):
            labels = variants[tag].to(torch.long)
            refined = refine_plan(labels, args.vote_window, args.min_segment_length)
            row[tag + "_transition"] = float((labels == 0).float().mean())
            row[tag + "_transition_after_refine"] = float((refined == 0).float().mean())
            row[tag + "_segments_after_refine"] = len(
                [s for s in labels_to_segments(refined) if s.label])
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in row.items()}))

    keys = [k for k in rows[0] if k != "name"]
    report = {
        "checkpoint": str(args.checkpoint),
        "plan_stride": args.plan_stride,
        "seed": args.seed,
        "clips": len(rows),
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
        "per_clip": rows,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(report["mean"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
