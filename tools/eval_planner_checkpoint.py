"""Evaluate a trained atomic planner over a *full* split, against baselines.

``train_atomic.evaluate_planner`` reports on a single batch (``next(iter(loader))``),
so with the released val split it scores 64 of 245 windows.  That is fine as a
training heartbeat and far too noisy to decide whether a planner is worth
keeping.  This tool sweeps the whole split and, more importantly, prints the
baselines the accuracy has to beat:

* **chance** -- ``1 / (num_classes + 1)``, i.e. uniform over the label space.
* **majority class** -- always predict the most frequent label.  On this data
  that is the transition token, which alone covers ~19-22% of frames.  A
  planner below this line has learned nothing usable, regardless of its loss.
* **GT segment count** -- real plans hold ~10-12 segments per 150-frame window.
  A sample with ~150 segments is per-frame noise that happens to have a decent
  frame accuracy; a sample with 1 is a collapsed constant plan.  Frame accuracy
  cannot distinguish those, so it is never sufficient on its own.

Sampling is stochastic, so ``--repeats`` reports the spread rather than a single
draw.

The report is tagged ``SELF_DRIVEN``: the planner sees only music and a padding
mask, never the target labels.  That is what makes these numbers quotable, in
contrast to the completion diagnostic, which builds its draft from ground-truth
labels and is tagged ``ORACLE_GROUND_TRUTH_PLAN``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_atomic import make_dataset, make_loader, move_batch, planner_model


def load_planner(checkpoint_path, device):
    """Rebuild the planner from the hyperparameters stored in its checkpoint."""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "planner":
        raise ValueError(
            "expected a planner checkpoint, got stage={!r}".format(checkpoint.get("stage"))
        )
    args = argparse.Namespace(**checkpoint["args"])
    model = planner_model(args)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), args, checkpoint


def split_baselines(labels, padding_mask, num_classes):
    """Reference points the planner must beat to be worth anything."""
    valid = ~padding_mask
    flat = labels[valid]
    counts = torch.bincount(flat, minlength=num_classes + 1).float()
    majority = float(counts.max() / counts.sum())
    segments = ((labels[:, 1:] != labels[:, :-1]) & valid[:, 1:]).sum(dim=1) + 1
    return {
        "chance_accuracy": 1.0 / (num_classes + 1),
        "majority_class_accuracy": majority,
        "majority_class_id": int(counts.argmax()),
        "gt_transition_fraction": float((flat == 0).float().mean()),
        "gt_mean_segments": float(segments.float().mean()),
        "gt_median_segments": float(segments.float().median()),
        "gt_classes_present": int((counts > 0).sum()),
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes, repeats):
    """Sweep the split; return per-repeat metrics plus the pooled label stats."""
    all_labels, all_masks = [], []
    per_repeat = []

    for repeat in range(repeats):
        correct = total = nonzero_correct = nonzero_total = 0
        transition = 0
        denoise_correct = denoise_total = 0
        segment_counts = []
        losses = []

        for batch in loader:
            batch = move_batch(batch, device)
            labels, music, padding = batch["labels"], batch["music"], batch["padding_mask"]
            valid = ~padding

            summary = batch.get("global_music")
            output = model.training_step(labels, music, padding, global_music=summary)
            losses.append(float(output.loss))
            denoise_correct += int(
                ((output.logits.argmax(dim=-1) == output.target_labels) & valid).sum()
            )
            denoise_total += int(valid.sum())

            sample = model.sample(music, padding, deterministic=False,
                                  global_music=summary)
            hit = (sample == labels) & valid
            correct += int(hit.sum())
            total += int(valid.sum())
            nonzero = valid & (labels != 0)
            nonzero_correct += int(hit[nonzero].sum())
            nonzero_total += int(nonzero.sum())
            transition += int(((sample == 0) & valid).sum())
            segment_counts.append(
                (((sample[:, 1:] != sample[:, :-1]) & valid[:, 1:]).sum(dim=1) + 1).cpu()
            )

            if repeat == 0:
                all_labels.append(labels.cpu())
                all_masks.append(padding.cpu())

        segments = torch.cat(segment_counts).float()
        per_repeat.append(
            {
                "loss": float(np.mean(losses)),
                "denoising_accuracy": denoise_correct / max(denoise_total, 1),
                "sample_accuracy": correct / max(total, 1),
                "sample_nonzero_accuracy": nonzero_correct / max(nonzero_total, 1),
                "sample_transition_fraction": transition / max(total, 1),
                "sample_mean_segments": float(segments.mean()),
                "sample_median_segments": float(segments.median()),
            }
        )

    labels = torch.cat(all_labels)
    masks = torch.cat(all_masks)
    return per_repeat, split_baselines(labels, masks, num_classes), int((~masks).sum())


def summarize(per_repeat):
    """Mean and spread across repeats; a single stochastic draw can mislead."""
    keys = per_repeat[0].keys()
    return {
        key: {
            "mean": float(np.mean([r[key] for r in per_repeat])),
            "std": float(np.std([r[key] for r in per_repeat])),
            "min": float(np.min([r[key] for r in per_repeat])),
            "max": float(np.max([r[key] for r in per_repeat])),
        }
        for key in keys
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="defaults to the data root recorded in the checkpoint")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate an evenly-spaced subsample of this many "
                             "windows.  Not a prefix: this release's rows are in "
                             "recording order, so the first N windows are the "
                             "first accounts alphabetically, and an accuracy "
                             "measured on them is measured on a handful of "
                             "dancers.  The stride and the count ride in the "
                             "report so a subsampled number is never mistaken "
                             "for a full-split one.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, train_args, checkpoint = load_planner(args.checkpoint, device)

    data_root = args.data_root or Path(train_args.data_root)
    # A checkpoint built with the whole-track head *requires* the summary --
    # the model refuses rather than zero-filling, because a zero vector is a
    # specific claim about a track and would silently score a different model
    # than the one that was trained.  Read from the checkpoint's own args so the
    # caller cannot get it wrong.
    dataset = make_dataset(str(data_root), args.split,
                           global_music=bool(getattr(train_args, "global_music", False)),
                           global_music_shuffle_seed=getattr(
                               train_args, "global_music_shuffle_seed", None))
    subsample = None
    if args.limit is not None and args.limit < len(dataset):
        from torch.utils.data import Subset

        stride = max(1, len(dataset) // args.limit)
        rows = list(range(0, len(dataset), stride))[: args.limit]
        subsample = {"requested": args.limit, "evaluated": len(rows),
                     "of_split": len(dataset), "stride": stride}
        dataset = Subset(dataset, rows)
    loader = make_loader(dataset, args.batch_size, args.workers, shuffle=False)

    per_repeat, baselines, frames = evaluate(
        model, loader, device, train_args.num_classes, args.repeats
    )
    summary = summarize(per_repeat)

    accuracy = summary["sample_accuracy"]["mean"]
    segments = summary["sample_mean_segments"]["mean"]
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_step": checkpoint.get("step"),
        "data_root": str(data_root),
        "split": args.split,
        "windows": len(dataset),
        # Present only when a subsample was taken, so its absence means the
        # whole split; a default of null would read the same either way.
        "subsample": subsample,
        "headline_eligible_split_coverage": subsample is None,
        "valid_frames": frames,
        "repeats": args.repeats,
        "seed": args.seed,
        # The planner is conditioned on music and a padding mask only.
        "generation_protocol": "SELF_DRIVEN",
        "headline_eligible": True,
        "metrics": summary,
        "baselines": baselines,
        "verdict": {
            "beats_chance": accuracy > baselines["chance_accuracy"],
            "beats_majority_class": accuracy > baselines["majority_class_accuracy"],
            "accuracy_over_majority": accuracy - baselines["majority_class_accuracy"],
            "segment_count_ratio": segments / max(baselines["gt_mean_segments"], 1e-9),
        },
        "dataset_provenance": checkpoint.get("dataset_provenance"),
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
