#!/usr/bin/env python3
"""What does overlapping the planner's windows cost the plan it produces?

``--plan-stride 15 --plan-fusion vote`` was pinned into the wild M6 driver as a
free win: it drops generated segment boundaries landing on the 150-frame window
seam from 18.92x the uniform expectation to 2.24x, and improves fid_k from
10.613 to 9.631 without retraining.  Both of those are boundary statistics.

Neither of them can see this: on the val split the planner's own ``sample`` emits
**23.8% transition** per 340-frame window, and by the time a draft is built on
the 24 held-out clips the plan is **48.7% transition**.  Transition is exactly
where the draft is empty, so if the fusion is what doubles it, the seam was
bought by emptying half the conditioning.

The mechanism this probe tests: at ``--plan-stride 15`` roughly twenty-two
windows cover each frame, each drawing independently over 4,160 classes.  One
class -- transition -- holds about a quarter of the mass while the other 4,159
split the rest, so it wins most plurality votes even where no window is
confident.  ``centre`` does not have that property by construction: it takes the
frame from whichever window holds it most centrally, so each frame is still one
draw from the model.

Reported against the ground-truth plan of the same frames, because "23.8%
transition" is only high or low relative to what the dancer actually did (13.5%).

**The numbers above were measured before 2026-08-23**, when ``vote`` still
resolved a tied count with ``counts.argmax`` and therefore handed every tie to
label 0.  ``tools/probe_plan_vote_ties.py`` priced that rule at 4.6 of the 22.0
points by which the plan's transition share exceeded the ground truth, so the
``vote`` rows here reproduce only with
``infer_plan(..., plan_vote_tie_break="index")``.  The mechanism this probe
names -- plurality over many independent draws favouring the one class that
holds a large share of the mass -- is the remaining, larger half.

Usage::

    python3 tools/probe_plan_fusion.py \\
        --checkpoint runs/planner_wild_v4_acct_w2_340/planner_step58968.pt \\
        --data-root /dev/shm/atomicdance-acct/release_w2_340 \\
        --audio-dir runs/wild_v4_acct_gt_eval/audio \\
        --recording-list runs/t_stride_sweep/clips24.txt --split test \\
        --output runs/t8_discovery/plan_fusion.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.atomic import labels_to_segments  # noqa: E402
from infer_atomic import _load_music, infer_plan  # noqa: E402
from tools.eval_planner_checkpoint import load_planner  # noqa: E402

CONFIGS = [
    ("stride_window_no_fusion", None, "centre"),
    ("stride15_vote", 15, "vote"),
    ("stride15_centre", 15, "centre"),
    ("stride75_vote", 75, "vote"),
]


def _describe(labels, truth=None):
    segments = [s for s in labels_to_segments(labels) if s.label]
    row = {
        "transition_share": float((labels == 0).float().mean()),
        "segments": len(segments),
        "median_segment_frames": float(np.median([s.length for s in segments]))
        if segments else 0.0,
        "distinct_classes": int(len({int(s.label) for s in segments})),
    }
    if truth is not None:
        width = min(len(labels), len(truth))
        a, b = labels[:width], truth[:width]
        both = (a > 0) & (b > 0)
        row["frames_both_name_a_class"] = int(both.sum())
        row["correct_on_those_frames"] = int((a[both] == b[both]).sum())
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--audio-dir", type=pathlib.Path, required=True)
    parser.add_argument("--recording-list", type=pathlib.Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    from train_atomic import seed_everything

    device = torch.device(args.device)
    planner, train_args, _ = load_planner(args.checkpoint, device)

    root = pathlib.Path(args.data_root)
    names = json.load(open(str(root / args.split / "names.json")))
    truth_labels = np.load(str(root / args.split / "labels.npy"), mmap_mode="r")
    first = {}
    for index, name in enumerate(names):
        recording = name.rsplit("_slice", 1)[0]
        first.setdefault(recording, index)

    wanted = [line.strip() for line in
              args.recording_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [name for name in wanted if name not in first]
    if missing:
        raise SystemExit("error: not in the {} split: {}".format(args.split, missing[:3]))

    per_config = {name: [] for name, _, _ in CONFIGS}
    truth_rows = []
    for recording in wanted:
        audio = args.audio_dir / (recording + ".npy")
        if not audio.is_file():
            raise SystemExit("error: no music for {} at {}".format(recording, audio))
        music = _load_music(audio, None, train_args.music_dim)
        truth = torch.from_numpy(np.array(truth_labels[first[recording]], dtype=np.int64))
        truth_rows.append(_describe(truth))
        for name, stride, fusion in CONFIGS:
            # Reseeded per configuration so the four are the same draw, not four
            # points on one random walk.
            seed_everything(args.seed)
            plan = infer_plan(planner, music, train_args.seq_len, device,
                              plan_stride=stride, plan_fusion=fusion)
            per_config[name].append(_describe(plan[: len(truth)], truth))

    def summarise(rows):
        out = {}
        for key in rows[0]:
            values = [r[key] for r in rows if r.get(key) is not None]
            out[key] = float(np.median(values))
        pooled_frames = sum(r.get("frames_both_name_a_class", 0) for r in rows)
        pooled_hits = sum(r.get("correct_on_those_frames", 0) for r in rows)
        if pooled_frames:
            out["pooled_accuracy_on_those_frames"] = pooled_hits / pooled_frames
            out["clips_with_at_least_one_correct"] = sum(
                1 for r in rows if r.get("correct_on_those_frames", 0) > 0)
        return out

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "recordings": len(wanted),
        "seed": args.seed,
        "ground_truth_plan": summarise(truth_rows),
        "configurations": {name: summarise(per_config[name]) for name, _, _ in CONFIGS},
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
