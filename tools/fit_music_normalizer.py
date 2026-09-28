#!/usr/bin/env python3
"""Per-channel mean/std for the 35-D music feature vector, fit on the train split.

WHY THIS EXISTS, IN THE UNITS THAT MATTER.

Both stages read the music through a single bare ``nn.Linear`` -- the planner at
``model/atomic_planner.py`` (``music_projection``, added per frame to the label
embedding) and the completion decoder through ``model/model.py``'s
``cond_projection``.  Nothing scales the input first.  The 35 channels are
EDGE's librosa stack (``data/audio_extraction/baseline_features.py``):

    0        onset strength envelope
    1-20     20 MFCC
    21-32    12 chroma_cens
    33       onset-peak one-hot
    34       beat one-hot   (librosa.beat.beat_track)

and their scales differ by three orders of magnitude.  Measured 2026-08-30 on a
release array, per-channel standard deviation:

    MFCC c1  63.9      MFCC c2  47.8      MFCC c3  32.8
    chroma   0.02-0.05
    onset peak (33)    0.35
    beat (34)          0.23

A linear layer's response to a channel is proportional to that channel's scale
at initialisation, so **the beat channel enters at roughly 0.05% of the input
energy and the twelve chroma channels are effectively dead**.  That is a plain
mechanism for "the model does not dance to the beat" which needs no appeal to
capacity or to the objective: the beat is not in the input in any usable
magnitude.

WHAT THIS DOES NOT CLAIM.  Normalising makes the beat channel *available*; it
does not make anything *use* it.  The criterion for whether it did is the
period-lock reading of ``tools/probe_music_beat_alignment.py`` at n>=400 (at
n=100 the ground truth itself reads null -- see docs/DANCE_QUALITY_DEFECTS.md
section 6), plus R-precision, whose ground-truth ceiling of 0.39 against a
generated 0.02 is the positive control.  Report both or neither.

CONTRACT.  The statistics are fit on the **train split only** and frozen to a
file whose path and hash go into the training checkpoint's args, exactly as
``normalizer.pt`` does for motion.  Consumers apply it only when the checkpoint
says so, so every checkpoint trained before this file keeps reading raw
features and nothing changes underneath it.

A constant channel (std 0) is left alone rather than "fixed" with an epsilon --
same rule as ``tools/fit_motion_normalizer.py``: a channel that never varies
carries no information and scaling its noise up is inventing some.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch

MUSIC_DIM = 35
BEAT_CHANNEL = 34
ONSET_PEAK_CHANNEL = 33


def fit(path, block=4096):
    """Streaming mean/std over [N, T, 35] without loading 5.9 GB at once."""
    array = np.load(path, mmap_mode="r")
    if array.ndim != 3 or array.shape[-1] != MUSIC_DIM:
        raise SystemExit("{} is {}, expected [N, T, {}]".format(path, array.shape, MUSIC_DIM))
    count = 0
    total = np.zeros(MUSIC_DIM, np.float64)
    total_squared = np.zeros(MUSIC_DIM, np.float64)
    for start in range(0, len(array), block):
        chunk = np.asarray(array[start:start + block], np.float64).reshape(-1, MUSIC_DIM)
        count += len(chunk)
        total += chunk.sum(0)
        total_squared += (chunk * chunk).sum(0)
    mean = total / count
    variance = np.maximum(total_squared / count - mean * mean, 0.0)
    return mean, np.sqrt(variance), count


def energy_share(std):
    """Each channel's share of the summed input variance -- the number the
    docstring above quotes, and the one the positive-control test asserts on."""
    variance = np.asarray(std, np.float64) ** 2
    total = variance.sum()
    return variance / total if total > 0 else variance


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-root", required=True,
                        help="release directory; fit uses <root>/train/music.npy only")
    parser.add_argument("--output", required=True, help="destination .pt")
    parser.add_argument("--report", help="destination .json (defaults beside --output)")
    args = parser.parse_args()

    root = pathlib.Path(args.release_root)
    source = root / "train" / "music.npy"
    if not source.is_file():
        raise SystemExit("no train/music.npy under {}".format(root))
    mean, std, frames = fit(source)

    before = energy_share(std)
    # After z-scoring, every non-constant channel contributes equally.
    after = np.where(std > 0, 1.0 / max(int((std > 0).sum()), 1), 0.0)
    payload = {"mean": torch.tensor(mean, dtype=torch.float32),
               "std": torch.tensor(std, dtype=torch.float32)}
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    report = {
        "release_root": str(root.resolve()),
        "source": str(source.resolve()),
        "frames": int(frames),
        "music_dim": MUSIC_DIM,
        "constant_channels": [int(i) for i in np.flatnonzero(std <= 0)],
        "variance_share_before": {str(i): float(before[i]) for i in range(MUSIC_DIM)},
        "beat_channel": {
            "index": BEAT_CHANNEL,
            "std": float(std[BEAT_CHANNEL]),
            "variance_share_before": float(before[BEAT_CHANNEL]),
            "variance_share_after": float(after[BEAT_CHANNEL]),
        },
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    report_path = pathlib.Path(args.report) if args.report else out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("{} frames -> {}".format(frames, out))
    print("beat channel {}: std {:.4f}, variance share {:.5f}% -> {:.3f}%".format(
        BEAT_CHANNEL, std[BEAT_CHANNEL], 100 * before[BEAT_CHANNEL], 100 * after[BEAT_CHANNEL]))
    loudest = int(np.argmax(std))
    print("loudest channel {}: std {:.2f}, variance share {:.2f}%".format(
        loudest, std[loudest], 100 * before[loudest]))
    print(report_path)


if __name__ == "__main__":
    main()
