"""Measure how much of the atomic label sequence music can explain at all.

Before blaming a planner for weak accuracy, it is worth knowing the ceiling.
The released vocabulary (``kinematic_atomic_100_v2``) was discovered from
*motion* descriptors -- heading-canonicalised kinematics, DCT-compressed --
with music never entering the clustering.  Nothing guarantees that the
resulting cluster identity is a function of the audio.  If it is not, then no
planner, however well trained, can beat the label marginal, and the right fix
is the label space rather than the generative model.

This probe removes diffusion from the question.  It trains the same music
encoder + Transformer trunk the planner uses, but as a plain frame-wise
classifier with cross-entropy.  That is a strictly easier problem than
diffusion sampling, so its accuracy upper-bounds what the planner can reach on
this label space.

Read the output as:

* accuracy near the **majority-class** baseline -> music does not determine the
  labels; the vocabulary is the problem.
* accuracy well above it -> the labels are learnable and any gap the planner
  shows is the planner's own.

Two controls separate signal from bookkeeping:

* ``--shuffle-music`` pairs each window with another window's music.  A probe
  that scores the same either way is reading the label prior, not the audio.
* the ``nonzero`` accuracies exclude the transition token, which alone accounts
  for roughly a fifth of all frames and can carry an otherwise empty model.

The majority-class gate is withdrawn (2026-08-22)
-------------------------------------------------
**Old rule**: fail unless held-out accuracy beats the majority-class baseline,
"a vocabulary below this is not worth planning on".  It failed on AIST
(0.1430 vs 0.1794) and on clean5b5 (0.1311 vs 0.2321), and both were reported
as "the labels are not predictable from music".

**What overturns it**: ``tools/same_song_agreement.py`` measured what the
ground truth itself scores.  Two people dancing the *same track*, aligned by
the fingerprint's own lag, agree on **0.1613** of frames (627 pairs, clean5b5)
-- **below the 0.2321 majority baseline the gate demanded.**  Choreography is
one-to-many, so a constant "always transition" predictor outscores the truth,
and the old rule was not strict but unreachable: the mirror of the never-firing
gate CLAUDE.md opens with.

**New rule**: the reachable ceiling is another dancer's answer and the floor is
what different-song pairs score, so this tool now reports the probe as a
*fraction of that range* and gates on things that can actually be failed --
beating the different-song floor, and losing measurably when the music is
mispaired.  ``--min-accuracy-over-majority`` still exists but is off unless
passed, so the older AIST reports remain reproducible from their own command
lines.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# sys.path[0] is tools/ when this runs as a script, not the repo root; the
# repo-package import below otherwise rides the ambient PYTHONPATH's stray
# trailing colon (which puts CWD on the path) -- a coincidence, not a contract.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.atomic_planner import AtomicPlannerTransformer  # noqa: E402
from train_atomic import make_dataset, make_loader, move_batch


class MusicOnlyClassifier(nn.Module):
    """Planner trunk with the noisy-label input pinned to a constant token.

    Reusing ``AtomicPlannerTransformer`` keeps the capacity and conditioning
    path identical to the planner, so the comparison is about the objective and
    the label space rather than about architecture.
    """

    def __init__(self, num_atomic_classes, music_dim, **kwargs):
        super().__init__()
        self.trunk = AtomicPlannerTransformer(
            num_atomic_classes=num_atomic_classes, music_dim=music_dim, **kwargs
        )

    def forward(self, music, padding_mask):
        batch, frames = music.shape[:2]
        # A constant label token and a constant timestep leave the music as the
        # only input that varies, so the trunk cannot copy from a noisy label.
        blank = torch.zeros((batch, frames), dtype=torch.long, device=music.device)
        timesteps = torch.zeros((batch,), dtype=torch.long, device=music.device)
        return self.trunk(blank, music, timesteps, padding_mask)


def load_label_map(path):
    """Return a LongTensor mapping original label -> coarse label."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload["label_map"] if isinstance(payload, dict) else payload
    mapping = torch.tensor(values, dtype=torch.long)
    if int(mapping[0]) != 0:
        raise ValueError("label 0 is the transition token and must map to 0")
    return mapping


def apply_label_map(labels, mapping):
    """Remap a label tensor, refusing out-of-range labels rather than clamping."""
    if mapping is None:
        return labels
    if int(labels.max()) >= len(mapping):
        raise ValueError(
            "label {} exceeds the label map length {}".format(int(labels.max()), len(mapping))
        )
    return mapping.to(labels.device)[labels]


def accuracy_stats(logits, labels, padding_mask):
    valid = ~padding_mask
    predicted = logits.argmax(dim=-1)
    hit = (predicted == labels) & valid
    nonzero = valid & (labels != 0)
    return {
        "correct": int(hit.sum()),
        "total": int(valid.sum()),
        "nonzero_correct": int(hit[nonzero].sum()),
        "nonzero_total": int(nonzero.sum()),
    }


_SONG_IN_NAME = re.compile(r"_(m[A-Za-z]{2}\d+)_")


def _song_of(name):
    """The AIST backing track in a window name, or None outside AIST."""
    match = _SONG_IN_NAME.search(name)
    return match.group(1) if match else None


def donor_permutation(names):
    """For each window, the index of a window whose music is a *different song*.

    ``music.roll(1, dims=0)`` used to serve as the shuffled-music control, on
    the reasoning that every window then keeps a real music tensor that is not
    its own.  It does not do that here.  The eval loader is built with
    ``shuffle=False``, and in this release's window order the neighbour of a
    window is another recording *of the same backing track* 99.5% of the time
    (measured 2026-08-13 on both the song-disjoint and the old test splits).
    Music features are a property of the track, so the "mispaired" music was
    the same song -- and ``--min-real-minus-shuffled``, which demands that
    mispairing COST accuracy, could not fail no matter what the model did.
    That is a gate that reads like a check, which this repo treats as worse
    than no check at all.

    Windows are grouped by song and each group is given another group's music,
    by rotating the song order.  Returns ``(index array, statistics)`` so the
    control can be audited from the report rather than trusted.
    """
    songs = [_song_of(name) for name in names]
    known = sum(1 for song in songs if song is not None)
    # Outside AIST no name carries a song id, and grouping by a single None key
    # would leave the identity permutation -- every window handed back its own
    # music while the report claimed 100% mispairing, because a None donor for a
    # None window counted as "different".  Fall back to the recording, which is
    # the coarsest unit these names do carry, and say which unit was used.
    unit = "backing track"
    if known == 0:
        keyed = [name.rsplit("__slice", 1)[0] for name in names]
        unit = "recording (no backing-track id in these names)"
    else:
        keyed = songs
    groups = {}
    for index, key in enumerate(keyed):
        groups.setdefault(key, []).append(index)
    keys = sorted(groups, key=lambda k: (k is None, k))
    donor = np.arange(len(names))
    if len(keys) > 1:
        for position, key in enumerate(keys):
            source = groups[keys[(position + 1) % len(keys)]]
            for offset, index in enumerate(groups[key]):
                donor[index] = source[offset % len(source)]
    # Counted on the unit actually used, and an unknown key is never credited as
    # different: the whole point is to know what fraction really got mispaired.
    achieved = sum(
        1 for i, j in enumerate(donor)
        if keyed[i] is not None and keyed[j] is not None and keyed[i] != keyed[j]
    )
    return donor, {
        # Named for the unit actually grouped on, so a recording-level fallback
        # cannot be read as a backing-track-level control.
        "groups_in_split": sum(1 for k in keys if k is not None),
        "songs_in_split": known and sum(1 for k in keys if k is not None) or 0,
        "windows": len(names),
        "donor_is_a_different_group": achieved / max(len(names), 1),
        "donor_is_a_different_song": (achieved / max(len(names), 1)) if known else 0.0,
        "names_without_a_song_id": len(names) - known,
        "grouping_unit": unit,
        "method": "windows grouped by {}, each group given the next group's music".format(unit),
    }


class DonorMusicDataset(torch.utils.data.Dataset):
    """The eval split with each window's music replaced by its donor's."""

    def __init__(self, base, donor):
        self.base = base
        self.donor = donor

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        item["music"] = self.base[int(self.donor[index])]["music"]
        return item


def run_split(model, loader, device, shuffle_music, label_map=None):
    model.eval()
    totals = {"correct": 0, "total": 0, "nonzero_correct": 0, "nonzero_total": 0}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            music = batch["music"]
            if shuffle_music:
                # Kept only for the legacy path; the honest control is a donor
                # dataset built by donor_permutation(), because rolling within a
                # batch hands back the same song.  See that docstring.
                music = music.roll(1, dims=0)
            logits = model(music, batch["padding_mask"])
            labels = apply_label_map(batch["labels"], label_map)
            for key, value in accuracy_stats(logits, labels, batch["padding_mask"]).items():
                totals[key] += value
    return {
        "accuracy": totals["correct"] / max(totals["total"], 1),
        "nonzero_accuracy": totals["nonzero_correct"] / max(totals["nonzero_total"], 1),
        "frames": totals["total"],
    }


def label_baselines(loader, num_classes, label_map=None):
    counts = torch.zeros(num_classes + 1)
    for batch in loader:
        valid = ~batch["padding_mask"]
        labels = apply_label_map(batch["labels"], label_map)
        counts += torch.bincount(labels[valid], minlength=num_classes + 1).float()
    return {
        "chance_accuracy": 1.0 / (num_classes + 1),
        "majority_class_accuracy": float(counts.max() / counts.sum()),
        "majority_class_id": int(counts.argmax()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--music-dim", type=int, default=35)
    parser.add_argument("--seq-len", type=int, default=150)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--eval-split",
        default="val",
        choices=["val", "test"],
        help="split to score; prefer a song-disjoint one, since sharing backing "
             "tracks with train inflates every music-conditioned number",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="JSON from build_coarse_vocabulary.py remapping original labels to a "
             "coarser vocabulary; index i holds the coarse label for label i",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="exit non-zero unless the label space clears the thresholds below",
    )
    parser.add_argument(
        "--same-song-labels",
        type=Path,
        default=None,
        help="label directory (labels.jsonl + labels/<hash>/labels.npy) used to "
             "measure the same-song ceiling; without it no fraction-of-ceiling "
             "is reported and the floor check cannot run",
    )
    parser.add_argument(
        "--music-pairs",
        type=Path,
        default=None,
        help="fingerprint-verified cross-upload same-track pairs with "
             "lag_frames; required with --same-song-labels",
    )
    parser.add_argument(
        "--min-ceiling-fraction",
        type=float,
        default=0.0,
        help="required share of the same-song ceiling, measured on atomic "
             "frames only.  0 reports without judging: no defensible threshold "
             "has been established yet, and inventing one here is what the "
             "withdrawn majority rule did",
    )
    parser.add_argument(
        "--min-accuracy-over-majority",
        type=float,
        default=None,
        help="WITHDRAWN as a default (see module docstring): the majority "
             "baseline sits above the same-song ceiling, so the ground truth "
             "cannot pass it either.  Still honoured when passed explicitly, "
             "so pre-2026-08-22 reports reproduce from their command lines",
    )
    parser.add_argument(
        "--min-real-minus-shuffled",
        type=float,
        default=0.02,
        help="required accuracy drop when each window is paired with another "
             "window's music; near zero means the model ignores the audio",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    label_map = load_label_map(args.label_map) if args.label_map else None
    if label_map is not None:
        # The coarse vocabulary sets the output width; label 0 stays transition.
        args.num_classes = int(label_map.max())
        print("label map: {} original -> {} coarse classes".format(
            len(label_map) - 1, args.num_classes), flush=True)

    train_loader = make_loader(
        make_dataset(str(args.data_root), "train"), args.batch_size, args.workers, True
    )
    val_loader = make_loader(
        make_dataset(str(args.data_root), args.eval_split), args.batch_size, args.workers, False
    )

    model = MusicOnlyClassifier(
        num_atomic_classes=args.num_classes,
        music_dim=args.music_dim,
        latent_dim=args.latent_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            logits = model(batch["music"], batch["padding_mask"])
            valid = ~batch["padding_mask"]
            targets = apply_label_map(batch["labels"], label_map)
            per_token = F.cross_entropy(
                logits.transpose(1, 2), targets, reduction="none"
            )
            loss = (per_token * valid).sum() / valid.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss))
        if epoch % 10 == 0 or epoch == args.epochs:
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train": run_split(model, train_loader, device, False, label_map),
                "val": run_split(model, val_loader, device, False, label_map),
            }
            history.append(record)
            print(
                "epoch={epoch} loss={train_loss:.4f} "
                "train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
                "val_nonzero={val_nz:.4f}".format(
                    epoch=record["epoch"],
                    train_loss=record["train_loss"],
                    train_acc=record["train"]["accuracy"],
                    val_acc=record["val"]["accuracy"],
                    val_nz=record["val"]["nonzero_accuracy"],
                ),
                flush=True,
            )

    baselines = label_baselines(val_loader, args.num_classes, label_map)

    eval_dataset = make_dataset(str(args.data_root), args.eval_split)
    names = getattr(eval_dataset, "names", None)
    if names:
        donor, donor_stats = donor_permutation(names)
        donor_loader = make_loader(
            DonorMusicDataset(eval_dataset, donor), args.batch_size, args.workers, False
        )
        shuffled = run_split(model, donor_loader, device, False, label_map)
    else:
        donor_stats = {"method": "roll within batch (no window names to group by song)"}
        shuffled = run_split(model, val_loader, device, True, label_map)
    print("mispaired-music control: {}".format(json.dumps(donor_stats)), flush=True)
    final = history[-1]

    ceiling = None
    if args.same_song_labels is not None:
        if args.music_pairs is None:
            raise SystemExit("--same-song-labels needs --music-pairs; the ceiling "
                             "is defined by verified same-track pairs, not by the "
                             "label directory alone")
        from tools.same_song_agreement import measure as measure_ceiling
        ceiling = measure_ceiling(args.same_song_labels, args.music_pairs,
                                  label_map=label_map, seed=args.seed)
        print("same-song ceiling: atomic {:.4f} / floor {:.4f} ({} pairs)".format(
            ceiling["same_song"]["atomic_frames"] or float("nan"),
            ceiling["different_song"]["atomic_frames"] or float("nan"),
            ceiling["same_song"]["pairs"]), flush=True)

    report = {
        "probe": "supervised music -> atomic label upper bound",
        "note": (
            "Cross-entropy classifier sharing the planner trunk. Strictly easier "
            "than diffusion sampling, so this upper-bounds planner accuracy on "
            "this label space."
        ),
        "data_root": str(args.data_root),
        "epochs": args.epochs,
        "seed": args.seed,
        "generation_protocol": "SELF_DRIVEN",
        "headline_eligible": False,
        "headline_reason": "diagnostic probe, not a generation result",
        "history": history,
        "final": final,
        "eval_split": args.eval_split,
        "label_map": str(args.label_map) if args.label_map else None,
        "num_classes": args.num_classes,
        "val_shuffled_music": shuffled,
        "mispaired_music_control": donor_stats,
        "baselines": baselines,
        "same_song_ceiling": ceiling,
        "thresholds": {
            "min_accuracy_over_majority": args.min_accuracy_over_majority,
            "min_real_minus_shuffled": args.min_real_minus_shuffled,
            "min_ceiling_fraction": args.min_ceiling_fraction,
        },
        "verdict": {
            "val_beats_majority": final["val"]["accuracy"] > baselines["majority_class_accuracy"],
            "val_over_majority": final["val"]["accuracy"] - baselines["majority_class_accuracy"],
            "real_minus_shuffled": final["val"]["accuracy"] - shuffled["accuracy"],
            "nonzero_real_minus_shuffled": (
                final["val"]["nonzero_accuracy"] - shuffled["nonzero_accuracy"]
            ),
        },
    }

    # Read on atomic frames, because the ceiling is only defined there: two
    # dancers both resting is agreement about nothing, and the transition token
    # is what made the withdrawn majority rule unreachable in the first place.
    if ceiling is not None:
        top = ceiling["same_song"]["atomic_frames"]
        bottom = ceiling["different_song"]["atomic_frames"]
        probe_nonzero = final["val"]["nonzero_accuracy"]
        span = (top - bottom) if (top is not None and bottom is not None) else None
        report["verdict"].update({
            "ceiling_atomic": top,
            "floor_atomic": bottom,
            "fraction_of_ceiling": (probe_nonzero / top) if top else None,
            "beats_different_song_floor": (probe_nonzero > bottom) if bottom is not None else None,
            "music_effect_share_of_span": (
                (probe_nonzero - shuffled["nonzero_accuracy"]) / span
                if span else None
            ),
        })

    verdict = report["verdict"]
    failures = []
    if args.min_accuracy_over_majority is not None:
        # Only when the caller asks for it: the default is withdrawn because the
        # majority baseline sits above the same-song ceiling (module docstring).
        if verdict["val_over_majority"] < args.min_accuracy_over_majority:
            failures.append(
                "held-out accuracy is {:.4f} over the majority baseline, below the "
                "required {:.4f} (note: this baseline is above the same-song "
                "ceiling on every corpus measured so far)".format(
                    verdict["val_over_majority"], args.min_accuracy_over_majority
                )
            )
    if verdict.get("beats_different_song_floor") is False:
        failures.append(
            "held-out atomic accuracy {:.4f} does not beat the different-song "
            "floor {:.4f}: the probe carries no more than the label prior".format(
                final["val"]["nonzero_accuracy"], verdict["floor_atomic"]
            )
        )
    fraction = verdict.get("fraction_of_ceiling")
    if fraction is not None and fraction < args.min_ceiling_fraction:
        failures.append(
            "held-out atomic accuracy reaches {:.1%} of the same-song ceiling "
            "({:.4f} of {:.4f}), below the required {:.1%}".format(
                fraction, final["val"]["nonzero_accuracy"], verdict["ceiling_atomic"],
                args.min_ceiling_fraction
            )
        )
    if verdict["real_minus_shuffled"] < args.min_real_minus_shuffled:
        failures.append(
            "pairing each window with another window's music costs only {:.4f}, "
            "below the required {:.4f}: the model is reading the label prior, "
            "not the audio".format(
                verdict["real_minus_shuffled"], args.min_real_minus_shuffled
            )
        )
    report["gate"] = {
        "enforced": args.gate,
        "passed": not failures,
        "failures": failures,
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    if failures:
        for failure in failures:
            print("GATE FAILED: {}".format(failure))
        if args.gate:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
