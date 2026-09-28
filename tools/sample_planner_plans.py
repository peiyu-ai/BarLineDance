"""Sample SELF_DRIVEN planner plans in the gate-v2 ``--plans`` format.

``probe_structure_conditioning.py --plans`` scores whether generated plans
reproduce the same-song structural coherence measured on ground truth.  It
needs ``{"plans": [{"song": .., "labels": [...]}, ...]}``; nothing in the
repo emitted that, so the planner half of gate v2 has never actually run.

Sampling protocol, chosen to mirror the GT statistic:

* one window per source *sequence* (the first slice), never multiple
  overlapping slices -- the GT gate stitches slices back into sequences, and
  window overlap would count every boundary up to ten times;
* same-song pairs therefore come only from *distinct* performances of a song,
  exactly like the GT Mann-Whitney;
* the planner sees music and a padding mask only (SELF_DRIVEN), so these
  plans are quotable alongside the GT numbers.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_planner_checkpoint import load_planner
from train_atomic import make_dataset

_SLICE = re.compile(r"_slice(\d+)$")

# The fourth tool that needed "which dances share a song", and the fourth
# regex it would have grown.  ``_(m[A-Za-z]{2}\d+)_`` matches nothing on the
# wild corpus, and the loop below *skips* a sequence whose song will not parse
# -- so a wild release would sample plans and publish an empty ``plans`` list
# without any step failing.  One shared parser, so four tools cannot disagree
# about which dances share a song.
from tools.eval_r_precision import music_key  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="defaults to the data root recorded in the checkpoint")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, train_args, checkpoint = load_planner(args.checkpoint, device)

    data_root = args.data_root or Path(train_args.data_root)
    # Same reason as in eval_planner_checkpoint: a global-music checkpoint
    # refuses to run without its summary, and the flags come from the
    # checkpoint rather than from whoever is sampling.
    dataset = make_dataset(str(data_root), args.split,
                           global_music=bool(getattr(train_args, "global_music", False)),
                           global_music_shuffle_seed=getattr(
                               train_args, "global_music_shuffle_seed", None))

    # First slice of every source sequence, in name order.
    chosen = {}
    for index in range(len(dataset)):
        name = dataset.names[index]
        sequence = _SLICE.sub("", name)
        slice_id = int(_SLICE.search(name).group(1))
        if sequence not in chosen or slice_id < chosen[sequence][1]:
            chosen[sequence] = (index, slice_id)
    picks = sorted((sequence, index) for sequence, (index, _) in chosen.items())

    plans, unnamed = [], []
    with torch.no_grad():
        for offset in range(0, len(picks), args.batch_size):
            batch = picks[offset:offset + args.batch_size]
            music = torch.stack(
                [dataset[index]["music"] for _, index in batch]
            ).to(device)
            padding = torch.zeros(
                music.shape[:2], dtype=torch.bool, device=device
            )
            summary = None
            if getattr(train_args, "global_music", False):
                summary = torch.stack(
                    [dataset[index]["global_music"] for _, index in batch]).to(device)
            sample = model.sample(music, padding, deterministic=False,
                                  global_music=summary)
            for row, (sequence, _) in zip(sample.cpu(), batch):
                song = music_key(sequence)
                if song is None:
                    unnamed.append(sequence)
                    continue
                plans.append({
                    "song": song,
                    "sequence": sequence,
                    "labels": row.tolist(),
                })

    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_step": checkpoint.get("step"),
        "data_root": str(data_root),
        "split": args.split,
        "seed": args.seed,
        "generation_protocol": "SELF_DRIVEN",
        # Sequences whose song would not parse are skipped, and skipping
        # silently is how a wild release publishes an empty plan list with
        # every step reporting success.  Counted here, and refused below.
        "sequences_with_no_song": len(unnamed),
        "plans": plans,
    }
    # The refusal the comment above promises.  An empty or near-empty plan list
    # is what a corpus this parser does not understand produces, and every step
    # downstream would report success on it: gate v2 would read zero plans and
    # say "too few same-song pairs", which reads as a small corpus rather than
    # as a tool that understood none of it.
    if unnamed:
        print("{} of {} sequence(s) name no song, e.g. {}".format(
            len(unnamed), len(picks), unnamed[0]), file=sys.stderr)
    if not plans:
        print("error: no sequence named a song, so there are no plans to write; "
              "the song parser does not understand this corpus", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("wrote {} plans ({} sequences, split={}) to {}".format(
        len(plans), len(picks), args.split, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
