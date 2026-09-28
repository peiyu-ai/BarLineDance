"""Can the music features linearly predict a sequence's GT segment rate?

Gate v2 on the kinematic x0 planner found that same-song structural coherence
generalizes across splits, but generated per-song rates track GT only on
train (Spearman 0.51) and not on val (-0.07).  Before reading that as a
planner defect, two controls are required, and this probe runs both:

* **GT cross-split ceiling** -- per-song mean GT rate of train performances
  vs val performances of the *same songs*.  If the dancers themselves do not
  agree on a song's rate across performance groups, no music-conditioned
  model can correlate with val GT, and the planner's val number is at
  ceiling rather than broken.  (First measurement: rho = -0.13 over the 16
  shared songs -- the ceiling is zero.)
* **Feature-mode contrast** -- ridge from music features to rate in two
  pooling modes.  ``full_stitch`` pools the whole stitched sequence: it
  reaches val song rho ~0.6, but that leaks sequence length and late-song
  content, which the planner never sees.  ``first_window`` pools only the
  slice-0 window, exactly the planner's input: it collapses to
  non-significant, proving the leak and closing the case.

Rates come from ``probe_structure_conditioning.sequence_rates_from_release``
(stitched full sequences, so window overlap cannot multiply boundaries).
Evaluation is train -> val at sequence level plus per-song means, with a
Spearman rank check alongside R^2 so a monotone-but-miscalibrated probe
still shows up.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.probe_structure_conditioning import sequence_rates_from_release

_SLICE = re.compile(r"_slice(\d+)$")


def stitched_music_means(release, split):
    """Time-pooled music features of every stitched full sequence."""
    music = np.load(Path(release) / split / "music.npy", mmap_mode="r")
    names = json.loads(
        (Path(release) / split / "names.json").read_text(encoding="utf-8"))
    chunks = {}
    for index, window in enumerate(names):
        base = window.split("/")[-1]
        sequence = _SLICE.sub("", base)
        start = int(_SLICE.search(base).group(1)) * 15
        chunks.setdefault(sequence, {})[start] = index
    output = {}
    for sequence, rows in chunks.items():
        window_len = music.shape[1]
        length = max(rows) + window_len
        timeline = np.zeros((length, music.shape[2]), dtype=np.float64)
        for start, index in sorted(rows.items()):
            timeline[start:start + window_len] = music[index]
        output[sequence] = timeline.mean(axis=0)
    return output


def first_window_means(release, split):
    """Time-pooled music of only the slice-0 window: the planner's actual input."""
    music = np.load(Path(release) / split / "music.npy", mmap_mode="r")
    names = json.loads(
        (Path(release) / split / "names.json").read_text(encoding="utf-8"))
    output = {}
    for index, window in enumerate(names):
        base = window.split("/")[-1]
        sequence = _SLICE.sub("", base)
        slice_id = int(_SLICE.search(base).group(1))
        if sequence not in output or slice_id < output[sequence][1]:
            output[sequence] = (np.asarray(music[index]).mean(axis=0), slice_id)
    return {sequence: mean for sequence, (mean, _) in output.items()}


FEATURE_MODES = {
    "full_stitch": stitched_music_means,
    "first_window": first_window_means,
}


def collect(release, split, mode):
    rates = sequence_rates_from_release(release, split)
    features = FEATURE_MODES[mode](release, split)
    sequences = sorted(set(rates) & set(features))
    x = np.stack([features[s] for s in sequences])
    y = np.array([rates[s][1] for s in sequences])
    songs = [rates[s][0] for s in sequences]
    return x, y, songs, sequences


def ridge_fit(x, y, alpha):
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-8
    xn = (x - mean) / std
    xn = np.hstack([xn, np.ones((len(xn), 1))])
    eye = np.eye(xn.shape[1])
    eye[-1, -1] = 0.0  # never shrink the intercept
    weights = np.linalg.solve(xn.T @ xn + alpha * eye, xn.T @ y)
    return mean, std, weights


def ridge_predict(x, fit):
    mean, std, weights = fit
    xn = (x - mean) / std
    return np.hstack([xn, np.ones((len(xn), 1))]) @ weights


def r_squared(y_true, y_pred):
    residual = float(((y_true - y_pred) ** 2).sum())
    total = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - residual / total


def song_means(values, songs):
    by_song = {}
    for value, song in zip(values, songs):
        by_song.setdefault(song, []).append(value)
    return {song: float(np.mean(vals)) for song, vals in by_song.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/atomic_aistpp/aist_kinematic_release_v1"))
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    modes = {}
    for mode in FEATURE_MODES:
        x_train, y_train, songs_train, _ = collect(args.data_root, "train", mode)
        x_eval, y_eval, songs_eval, _ = collect(args.data_root, args.eval_split, mode)
        results = []
        for alpha in args.alphas:
            fit = ridge_fit(x_train, y_train, alpha)
            pred_train = ridge_predict(x_train, fit)
            pred_eval = ridge_predict(x_eval, fit)
            gt_song = song_means(y_eval, songs_eval)
            pred_song = song_means(pred_eval, songs_eval)
            keys = sorted(gt_song)
            song_rho = spearmanr([gt_song[k] for k in keys],
                                 [pred_song[k] for k in keys])
            seq_rho = spearmanr(y_eval, pred_eval)
            results.append({
                "alpha": alpha,
                "train_r2": r_squared(y_train, pred_train),
                "eval_r2": r_squared(y_eval, pred_eval),
                "eval_spearman_seq": float(seq_rho.statistic),
                "eval_spearman_seq_p": float(seq_rho.pvalue),
                "eval_spearman_song": float(song_rho.statistic),
                "eval_spearman_song_p": float(song_rho.pvalue),
            })
        modes[mode] = results

    # GT cross-split ceiling: do the two performance groups even agree on a
    # per-song rate?  Any model correlation above this is unreachable.
    train_song = song_means(y_train, songs_train)
    eval_song = song_means(y_eval, songs_eval)
    shared = sorted(set(train_song) & set(eval_song))
    ceiling = spearmanr([train_song[s] for s in shared],
                        [eval_song[s] for s in shared])

    report = {
        "data_root": str(args.data_root),
        "eval_split": args.eval_split,
        "train_sequences": len(y_train),
        "eval_sequences": len(y_eval),
        "train_songs": len(set(songs_train)),
        "eval_songs": len(set(songs_eval)),
        "songs_shared_train_eval": shared,
        "gt_rate_train_mean": float(y_train.mean()),
        "gt_rate_train_std": float(y_train.std()),
        "gt_rate_eval_mean": float(y_eval.mean()),
        "gt_rate_eval_std": float(y_eval.std()),
        "gt_cross_split_ceiling": {
            "songs": len(shared),
            "spearman": float(ceiling.statistic),
            "p_value": float(ceiling.pvalue),
        },
        "probe": "ridge: time-pooled 35-D music -> segments-per-second",
        "results": modes,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
