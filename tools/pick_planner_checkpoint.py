#!/usr/bin/env python3
"""Score every saved planner checkpoint on the whole validation split.

WHY THIS EXISTS.  ``train_atomic.py`` validates once, at the end, and its
``evaluate_planner`` reads a SINGLE batch (``next(iter(loader))``) -- fine as a
training heartbeat, too noisy to choose a checkpoint with.  On the T corpus the
choice matters: 5,329 training windows drive the train loss to 2e-04 by epoch
400, so "the last checkpoint" is the most overfit one, not the best one.

Reports sampled accuracy on non-transition frames, which is the quantity the
planner exists to get right, plus the transition share and segment count, so a
checkpoint that wins by emitting transition everywhere is visible rather than
merely high-scoring.
"""
from __future__ import annotations

import argparse, glob, json, os, pathlib, sys
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from train_atomic import (AtomicSequenceDataset, collate_atomic_sequences,  # noqa: E402
                          make_loader, move_batch, planner_model)


@torch.no_grad()
def score(model, loader, device, max_batches=None):
    model.eval()
    totals = {"valid": 0.0, "denoise_hit": 0.0, "sample_hit": 0.0,
              "nonzero": 0.0, "nonzero_hit": 0.0, "transition": 0.0,
              "segments": 0.0, "sequences": 0.0, "loss": 0.0, "batches": 0.0}
    for index, raw in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = move_batch(raw, device)
        summary = batch.get("global_music")
        out = model.training_step(batch["labels"], batch["music"],
                                  batch["padding_mask"], global_music=summary)
        valid = ~batch["padding_mask"]
        sample = model.sample(batch["music"], batch["padding_mask"],
                              deterministic=False, global_music=summary)
        hit = (sample == batch["labels"]) & valid
        nonzero = valid & (batch["labels"] != 0)
        counts = ((sample[:, 1:] != sample[:, :-1]) & valid[:, 1:]).sum(dim=1) + 1
        totals["valid"] += float(valid.sum())
        totals["denoise_hit"] += float(((out.logits.argmax(-1) == out.target_labels) & valid).sum())
        totals["sample_hit"] += float(hit.sum())
        totals["nonzero"] += float(nonzero.sum())
        totals["nonzero_hit"] += float((hit & nonzero).sum())
        totals["transition"] += float(((sample == 0) & valid).sum())
        totals["segments"] += float(counts.sum())
        totals["sequences"] += float(len(counts))
        totals["loss"] += float(out.loss)
        totals["batches"] += 1.0
    v = max(totals["valid"], 1.0)
    return {
        "loss": totals["loss"] / max(totals["batches"], 1.0),
        "denoising_accuracy": totals["denoise_hit"] / v,
        "sample_accuracy": totals["sample_hit"] / v,
        "sample_nonzero_accuracy": totals["nonzero_hit"] / max(totals["nonzero"], 1.0),
        "sample_transition_fraction": totals["transition"] / v,
        "sample_mean_segments": totals["segments"] / max(totals["sequences"], 1.0),
        "windows": int(totals["sequences"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    dataset = AtomicSequenceDataset(args.data_root, split=args.split)
    loader = make_loader(dataset, args.batch_size, 0, False, collate=collate_atomic_sequences)
    device = torch.device(args.device)

    rows = []
    paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")), key=os.path.getmtime)
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = planner_model(argparse.Namespace(**state["args"])).to(device)
        model.load_state_dict(state["model"])
        row = score(model, loader, device, args.max_batches)
        row["checkpoint"] = os.path.basename(path)
        row["epoch"] = state.get("epoch")
        row["train_loss"] = (state.get("metrics") or {}).get("train_loss")
        rows.append(row)
        print("{:36s} epoch {:>4}  val_loss {:.4f}  denoise {:.4f}  "
              "sample_nz {:.4f}  trans {:.4f}  seg {:.1f}".format(
                  row["checkpoint"], row["epoch"], row["loss"], row["denoising_accuracy"],
                  row["sample_nonzero_accuracy"], row["sample_transition_fraction"],
                  row["sample_mean_segments"]), flush=True)
        del model
        torch.cuda.empty_cache()

    best = max(rows, key=lambda r: r["sample_nonzero_accuracy"]) if rows else None
    report = {"schema_version": "atomicdance-planner-checkpoint-sweep-v1",
              "split": args.split, "data_root": args.data_root,
              "rows": rows, "best_by_sample_nonzero_accuracy": best,
              "how_to_read": (
                  "sample_nonzero_accuracy is the planner's job: agreeing with the "
                  "ground-truth label on frames that HAVE an atomic label. Read it "
                  "beside sample_transition_fraction -- a checkpoint that emits "
                  "transition everywhere scores 0 here rather than winning, which "
                  "is why both are reported.")}
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    if best:
        print("\nbest by sample_nonzero_accuracy: {} (epoch {}, {:.4f})".format(
            best["checkpoint"], best["epoch"], best["sample_nonzero_accuracy"]))
    print("wrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
