#!/usr/bin/env python3
"""Can the 35-D music features predict anything at all on an unheard song?

``tools/probe_label_predictability.py`` reports that the atomic label sequence
is not predictable from music, and on 2026-08-13 it reported that for four
different vocabularies on a song-disjoint split.  That result has two readings
and they call for opposite work:

1. **the label space is music-orthogonal** -- discovery clustered motion and the
   resulting classes simply do not track the audio.  Then the fix is the
   vocabulary, which is what the finetune plan assumes.
2. **the music channel is dead** -- the 35-D features, or their alignment to the
   motion timeline, carry nothing that survives a change of song.  Then every
   number above is about a broken input and the vocabulary is not implicated at
   all.

Nothing measured so far separates them, because every probe run so far has used
the atomic labels as its target.  This one changes the target and holds the
input fixed: same windows, same music tensor, same planner trunk, but predicting
the **dance genre** -- which AIST records rather than infers, which the music was
chosen for, and which a listener identifies in a bar or two.

Read it as a floor on the input, not as a result about dance:

* genre accuracy well above the majority genre on **held-out songs** -> the
  features generalise across songs, so reading (1) stands and the vocabulary is
  the thing to change;
* genre accuracy at the majority baseline -> reading (2), and the atomic-label
  conclusions are conclusions about a dead channel.

The mispaired-music control is the same one the label probe uses: every window
is given a different song's music (``donor_permutation``).  Here it should
*cost* accuracy -- if it does not, the model is reading something other than the
audio and even this floor is uninformative.

Usage::

    python3 tools/probe_music_informativeness.py \\
        --data-root data/atomic_aistpp/aist_songsplit_rg2_v2_release_v1 \\
        --eval-split test --output data/wild3d/reports/aist_songsplit_music_probe.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.probe_label_predictability import (  # noqa: E402
    DonorMusicDataset,
    MusicOnlyClassifier,
    donor_permutation,
)
from train_atomic import make_dataset, make_loader, move_batch  # noqa: E402

GENRE = re.compile(r"^g([A-Z]{2})_")


def genre_of(name):
    """The AIST genre code in a window name, or None."""
    stem = name.rsplit("/", 1)[-1]
    match = GENRE.match(stem)
    return match.group(1) if match else None


def genre_targets(names):
    """(target index per window, ordered genre list); refuses an unlabelled corpus."""
    genres = [genre_of(name) for name in names]
    missing = sum(1 for value in genres if value is None)
    if missing:
        raise SystemExit(
            "error: {} of {} window names carry no genre; this probe is AIST-only".format(
                missing, len(names)))
    ordered = sorted(set(genres))
    index = {genre: position for position, genre in enumerate(ordered)}
    return np.asarray([index[value] for value in genres], dtype=np.int64), ordered


class SequenceGenreClassifier(nn.Module):
    """The planner's music trunk, mean-pooled to one label per window.

    Same encoder and conditioning path as the label probe, so a difference
    between the two is about the target rather than about the architecture.
    """

    def __init__(self, num_genres, music_dim, **kwargs):
        super().__init__()
        self.backbone = MusicOnlyClassifier(num_genres, music_dim, **kwargs)
        self.head = nn.Linear(num_genres + 1, num_genres)

    def forward(self, music, padding_mask):
        frame_logits = self.backbone(music, padding_mask)
        valid = (~padding_mask).unsqueeze(-1).float()
        pooled = (frame_logits * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


SONG_IN_NAME = re.compile(r"_(m[A-Za-z]{2}\d+)_")


def evaluate(model, loader, targets, device):
    """Window accuracy, plus the predictions, so the honest unit can be formed."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch["music"], batch["padding_mask"])
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
    predicted = np.concatenate(predictions) if predictions else np.zeros(0, dtype=np.int64)
    return float((predicted == targets).mean()), len(predicted), predicted


def song_level(names, targets, predicted):
    """Accuracy over songs, which is the unit this test actually samples.

    Windows of one song are not independent evidence about that song's genre:
    they share the audio.  With one test song per genre the split holds ten
    independent items however many windows it slices them into, so a window
    accuracy computed over 3,534 rows reads as far more evidence than it is.
    This repo has the same warning on record for the AIST test split.
    """
    by_song = collections.defaultdict(list)
    truth = {}
    for name, target, guess in zip(names, targets, predicted):
        match = SONG_IN_NAME.search(name)
        song = match.group(1) if match else name
        by_song[song].append(int(guess))
        truth[song] = int(target)
    correct = 0
    for song, guesses in by_song.items():
        vote = collections.Counter(guesses).most_common(1)[0][0]
        correct += int(vote == truth[song])
    return {
        "songs": len(by_song),
        "correct": correct,
        "accuracy": correct / max(len(by_song), 1),
        "unit": "one vote per song (majority over its windows)",
    }


def train_once(train_subset, train_targets, eval_subset, eval_targets, eval_names,
               genres, args, device):
    """One fit and its held-out song-level score."""
    model = SequenceGenreClassifier(
        len(genres), args.music_dim, latent_dim=args.latent_dim,
        num_layers=args.layers, num_heads=args.heads,
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    order = np.arange(len(train_subset))
    generator = np.random.default_rng(args.seed)
    for _ in range(args.epochs):
        model.train()
        generator.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            picks = order[start:start + args.batch_size]
            loader = make_loader(
                torch.utils.data.Subset(train_subset, picks.tolist()), len(picks), 0, False)
            batch = move_batch(next(iter(loader)), device)
            expected = torch.from_numpy(train_targets[picks]).to(device)
            loss = loss_fn(model(batch["music"], batch["padding_mask"]), expected)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    loader = make_loader(eval_subset, args.batch_size, args.workers, False)
    window_accuracy, _, predicted = evaluate(model, loader, eval_targets, device)
    return window_accuracy, song_level(eval_names, eval_targets, predicted)


def cross_validate(args, device):
    """Hold out songs in folds, so the unit of evidence is all 60 songs.

    The fixed split cannot answer this question: one test song per genre is ten
    independent items, and ten items need 4/10 correct before chance is ruled
    out.  Pooling the splits and folding over songs raises the unit to every
    song in the corpus, which is the difference between a floor that is measured
    and one that is merely printed.  Nothing leaks: a song is wholly inside one
    fold, which is the same rule the release's own split uses.

    No mispaired-music control runs here, and that is a decision rather than an
    omission.  The label probe needs one because its model could score by reading
    the label prior instead of the audio.  This model's only input is the music
    tensor -- MusicOnlyClassifier pins the label token to a constant and the
    timestep to zero -- and every window is the same 150 frames, so there is no
    second channel to read.  Above chance on a song it never heard, it used the
    audio.
    """
    from math import comb

    parts, names, targets = [], [], []
    genres = None
    for split in ("train", "val", "test"):
        try:
            dataset = make_dataset(str(args.data_root), split)
        except Exception:
            continue
        part_targets, part_genres = genre_targets(dataset.names)
        genres = genres or part_genres
        if part_genres != genres:
            raise SystemExit("error: {} carries a different genre set".format(split))
        parts.append(dataset)
        names.extend(dataset.names)
        targets.append(part_targets)
    pooled = torch.utils.data.ConcatDataset(parts)
    targets = np.concatenate(targets)

    songs = [SONG_IN_NAME.search(name).group(1) if SONG_IN_NAME.search(name) else name
             for name in names]
    ordered_songs = sorted(set(songs))
    folds = args.cross_validate_songs
    assignment = {song: index % folds for index, song in enumerate(ordered_songs)}

    total_correct = total_songs = 0
    window_scores = []
    per_fold = []
    for fold in range(folds):
        held = [i for i, song in enumerate(songs) if assignment[song] == fold]
        rest = [i for i, song in enumerate(songs) if assignment[song] != fold]
        window_accuracy, song_score = train_once(
            torch.utils.data.Subset(pooled, rest), targets[rest],
            torch.utils.data.Subset(pooled, held), targets[held],
            [names[i] for i in held], genres, args, device)
        total_correct += song_score["correct"]
        total_songs += song_score["songs"]
        window_scores.append(window_accuracy)
        per_fold.append({"fold": fold, "songs": song_score["songs"],
                         "correct": song_score["correct"],
                         "window_accuracy": window_accuracy})
        print("fold {}/{}: {}/{} songs, window_acc={:.4f}".format(
            fold + 1, folds, song_score["correct"], song_score["songs"],
            window_accuracy), flush=True)

    chance = 1.0 / len(genres)
    p_value = sum(comb(total_songs, i) * chance ** i * (1 - chance) ** (total_songs - i)
                  for i in range(total_correct, total_songs + 1))
    threshold = next(
        (i for i in range(total_songs + 1)
         if sum(comb(total_songs, j) * chance ** j * (1 - chance) ** (total_songs - j)
                for j in range(i, total_songs + 1)) < 0.05),
        None)
    report = {
        "probe": "music -> dance genre, held out by song, pooled over every split",
        "data_root": str(args.data_root),
        "folds": folds,
        "genres": genres,
        "songs_scored": total_songs,
        "songs_correct": total_correct,
        "song_accuracy": total_correct / max(total_songs, 1),
        "chance": chance,
        "binomial_p_at_chance": p_value,
        "songs_needed_for_p_below_0.05": threshold,
        "mean_window_accuracy": float(np.mean(window_scores)),
        "per_fold": per_fold,
        "reading": ("song_accuracy well above chance with a small p -> the 35-D features "
                    "carry cross-song structure, so an unpredictable label space is a fact "
                    "about the labels; at chance -> the music channel is the problem and "
                    "every atomic-label conclusion is about a dead input"),
        "headline_eligible": False,
        "headline_reason": "diagnostic probe, not a generation result",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        path = pathlib.Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--eval-split", default="test", choices=("val", "test"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--music-dim", type=int, default=35)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--cross-validate-songs", type=int, default=0,
        help="pool every split and hold out songs in this many folds. The fixed "
             "split has one test song per genre, so it holds ten independent items "
             "and needs 4/10 correct to clear p<0.05 -- it cannot resolve a modest "
             "effect at all. Folding over all 60 songs is what makes the floor "
             "measurable rather than merely reported.")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.cross_validate_songs > 1:
        return cross_validate(args, device)

    train_set = make_dataset(str(args.data_root), "train")
    eval_set = make_dataset(str(args.data_root), args.eval_split)
    train_targets, genres = genre_targets(train_set.names)
    eval_targets, eval_genres = genre_targets(eval_set.names)
    if eval_genres != genres:
        raise SystemExit("error: train and {} carry different genre sets".format(args.eval_split))

    # Loaders are unshuffled so the target array lines up positionally; the
    # training pass shuffles indices itself.
    train_loader = make_loader(train_set, args.batch_size, args.workers, False)
    eval_loader = make_loader(eval_set, args.batch_size, args.workers, False)

    model = SequenceGenreClassifier(
        len(genres), args.music_dim, latent_dim=args.latent_dim,
        num_layers=args.layers, num_heads=args.heads,
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    order = np.arange(len(train_set))
    generator = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator.shuffle(order)
        losses = []
        for start in range(0, len(order), args.batch_size):
            picks = order[start:start + args.batch_size]
            batch = move_batch(
                make_loader(torch.utils.data.Subset(train_set, picks.tolist()),
                            len(picks), 0, False).__iter__().__next__(), device)
            expected = torch.from_numpy(train_targets[picks]).to(device)
            logits = model(batch["music"], batch["padding_mask"])
            loss = loss_fn(logits, expected)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
        if epoch % 5 == 0 or epoch == args.epochs:
            accuracy, _, _ = evaluate(model, eval_loader, eval_targets, device)
            print("epoch={} loss={:.4f} {}_acc={:.4f}".format(
                epoch, float(np.mean(losses)), args.eval_split, accuracy), flush=True)

    accuracy, frames, predicted = evaluate(model, eval_loader, eval_targets, device)
    songs = song_level(eval_set.names, eval_targets, predicted)
    donor, donor_stats = donor_permutation(eval_set.names)
    donor_loader = make_loader(
        DonorMusicDataset(eval_set, donor), args.batch_size, args.workers, False)
    shuffled, _, shuffled_predicted = evaluate(model, donor_loader, eval_targets, device)
    shuffled_songs = song_level(eval_set.names, eval_targets, shuffled_predicted)

    # Binomial tail at chance, on songs.  With ten songs the test simply cannot
    # resolve a modest effect, and saying so is the point of reporting it.
    from math import comb
    n, k, p0 = songs["songs"], songs["correct"], 1.0 / len(genres)
    p_value = sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1))
    smallest_significant = next(
        (i for i in range(n + 1)
         if sum(comb(n, j) * p0 ** j * (1 - p0) ** (n - j) for j in range(i, n + 1)) < 0.05),
        None)

    counts = collections.Counter(eval_targets.tolist())
    majority = max(counts.values()) / max(len(eval_targets), 1)
    report = {
        "probe": "music -> dance genre, a floor on whether the audio carries anything",
        "data_root": str(args.data_root),
        "eval_split": args.eval_split,
        "genres": genres,
        "windows": frames,
        "epochs": args.epochs,
        "accuracy": accuracy,
        "song_level": songs,
        "song_level_with_another_songs_music": shuffled_songs,
        "song_level_binomial_p_at_chance": p_value,
        "songs_needed_correct_for_p_below_0.05": smallest_significant,
        "accuracy_with_another_songs_music": shuffled,
        "real_minus_shuffled": accuracy - shuffled,
        "majority_genre_baseline": majority,
        "chance": 1.0 / len(genres),
        "accuracy_over_majority": accuracy - majority,
        "mispaired_music_control": donor_stats,
        "reading": (
            "above majority on held-out songs -> the 35-D features generalise across "
            "songs, so a label space that is unpredictable from them is a fact about "
            "the labels; at majority -> the music channel is the problem and the "
            "atomic-label conclusions are about a dead input. Read song_level, not "
            "the window accuracy: windows of one song share its audio, so the split "
            "holds as many independent items as it holds songs. Check "
            "songs_needed_correct_for_p_below_0.05 before concluding anything -- a "
            "test that cannot reach significance has not measured this."),
        "headline_eligible": False,
        "headline_reason": "diagnostic probe, not a generation result",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        path = pathlib.Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
