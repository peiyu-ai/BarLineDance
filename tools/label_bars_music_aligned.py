#!/usr/bin/env python3
"""Fit once, then assign: the shipped music-aligned bar labels (``music_aligned_cca_k8_v1``).

WHAT THIS IS.  The shipped planner (``runs/planner_aligned_k8``) and retrieval
library (``scratch/txy_t/release_aligned_k8``) do not use the TMR vocabulary as
their label: they use an 8-class space made by fitting, on TRAIN bars only, a
StandardScaler + CCA(4) between each bar's sub-bar music and an 11-number motion
description, then k-means(8) on the motion-side canonical variates.  Its reason
is recorded where it was built: the 21-class TMR label reads 0.1136 against a
0.1580 majority floor under GroupKFold over recordings -- worse than a constant
-- while this one reads 0.1790 against 0.1383.

WHY THIS FILE EXISTS.  The space was produced by a script in another session's
``/tmp`` scratchpad (preserved at ``runs/txy_t2_20260922/preserved_from_516f17be``)
that *refits* whatever train bars it is given and never saved the fit.  Pointed
at a corpus with more clips it would silently redefine all eight classes under
the same ``label_space_id``, and every artifact bound to k8 -- the planner's
output ids, the library's painted labels, the selector -- would then be wrong
with nothing to say so.  So the fit and the assignment are separated:

* ``fit`` refits on the original inputs exactly and **refuses to save unless the
  three published ``labels.npy`` arrays reproduce byte-for-byte** (``--check``).
  That reproduction is the only proof the saved model is the shipped one.
* ``assign`` labels any bar release with the saved model.  A bar's label
  depends only on its motion features -- ``cca.transform(x, y)[1]`` is computed
  from ``y`` alone -- and those features are read on motion in the units of the
  normalizer the space was fitted in (``data/wild3d/txy_t_normalizer``, sha
  76d785ec...), so ``assign`` normalizes raw motion with *that* normalizer, in
  the same float32 arithmetic as ``tools/apply_motion_normalizer.py``, rather
  than reading a newer corpus's normalized arrays.

The feature code below is the scratch script's, verbatim, including the quirk
that "joints" are the first three rot6d components of each joint.  Changing it
would change the space.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import shutil
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LABEL_SPACE_ID = "music_aligned_cca_k8_v1"
SUB = 8
LIMB = {"arms": (16, 17, 18, 19, 20, 21, 22, 23),
        "legs": (1, 2, 4, 5, 7, 8, 10, 11),
        "torso": (3, 6, 9, 12, 15)}


def subbar(x, bins=SUB):
    e = np.linspace(0, len(x), bins + 1).astype(int)
    return np.stack([x[a:b].mean(0) if b > a else x[min(a, len(x) - 1)]
                     for a, b in zip(e, e[1:])])


def motion_features(span, joints):
    """A compact, interpretable description of what the body did in this bar."""
    rot = span[:, 7:]
    d = np.abs(np.diff(rot, axis=0))
    rel = joints - joints[:, :1, :]
    step = np.linalg.norm(np.diff(rel, axis=0), axis=2)
    total = max(step.sum(), 1e-9)
    feats = [
        d.mean(),                                   # rotation energy
        d.std(),                                    # how uneven within the bar
        np.linalg.norm(span[-1, 4:6] - span[0, 4:6]),   # travel
        span[:, 6].std(),                           # vertical movement
        span[:, :4].mean(),                         # contact share
        np.abs(np.diff(span[:, :4], axis=0)).mean(),    # contact switching
    ]
    for name in ("arms", "legs", "torso"):
        feats.append(step[:, list(LIMB[name])].sum() / total)
    smooth = np.convolve(step.mean(1), np.ones(3) / 3.0, mode="same")
    feats.append(float(np.argmin(smooth)) / max(len(smooth) - 1, 1))
    feats.append(float(np.argmax(smooth)) / max(len(smooth) - 1, 1))
    return np.array(feats, dtype=np.float64)


def window_features(motion: np.ndarray, music: np.ndarray, spans) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Per-bar (motion features, music features, usable) for one planner row.

    A row with any bar under 10 frames, or running past its motion or music, is
    zero-filled whole -- the scratch script's rule, kept because the published
    labels were made with it.
    """
    feats, tunes = [], []
    for s, e in spans:
        if e > len(motion) or e > len(music) or e - s < 10:
            return np.zeros((len(spans), 11)), np.zeros((len(spans), 16)), False
        joints = motion[s:e, 7:].reshape(e - s, 24, 6)[:, :, :3]
        feats.append(motion_features(motion[s:e], joints))
        sb = subbar(music[s:e])
        tunes.append(np.concatenate([sb[:, 0], sb[:, 33]]))
    return np.stack(feats), np.stack(tunes), True


def normalize_like_the_fit(raw: np.ndarray, normalizer_pt: pathlib.Path) -> np.ndarray:
    """``tools/apply_motion_normalizer._normalize_sequence``, same float32 order."""
    import torch

    state = torch.load(str(normalizer_pt), map_location="cpu", weights_only=False)
    data_min = state["data_min"].numpy().astype(np.float32).reshape(-1)
    data_max = state["data_max"].numpy().astype(np.float32).reshape(-1)
    data_range = data_max - data_min
    safe_range = np.where(data_range == np.float32(0.0), np.float32(1.0), data_range).astype(np.float32)
    working = np.asarray(raw, dtype=np.float32)
    return np.float32(2.0) * (working - data_min) / safe_range - np.float32(1.0)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return sha256_bytes(pathlib.Path(path).read_bytes())


def read_windows(bars_path: pathlib.Path) -> List[dict]:
    return [json.loads(l) for l in open(bars_path, encoding="utf-8") if l.strip()]


def split_label_arrays(windows: Sequence[dict], assign) -> Dict[str, np.ndarray]:
    out = {}
    for split in ("train", "val", "test"):
        rows = sorted((w for w in windows if w["split"] == split), key=lambda w: w["index"])
        if not rows:
            continue
        labels = np.zeros((len(rows), rows[0]["motion"].shape[0]), dtype=np.int64)
        for position, w in enumerate(rows):
            labels[position] = assign(w["motion"].astype(np.float64), w["music"].astype(np.float64))
        out[split] = labels
    return out


def featurize(windows_raw: Sequence[dict], load_seq) -> List[dict]:
    cache, out = {}, []
    for w in windows_raw:
        seq = w["sequence_id"]
        if seq not in cache:
            loaded = load_seq(seq)
            if loaded is None:
                continue
            cache[seq] = loaded
        motion, music = cache[seq]
        feats, tunes, ok = window_features(motion, music, w["bar_frame_spans"])
        out.append({"split": w["split"], "index": int(w["array_index"]), "motion": feats,
                    "music": tunes, "usable": ok, "sequence_id": seq})
    return out


def command_fit(args) -> int:
    from sklearn import __version__ as sklearn_version
    from sklearn.cluster import KMeans
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import StandardScaler

    meta = {}
    for line in open(args.sequences, encoding="utf-8"):
        r = json.loads(line)
        meta[r["sequence_id"]] = (r["motion_path"], r["music_path"])
    load = lambda seq: ((np.load(meta[seq][0]).astype(np.float64), np.load(meta[seq][1]).astype(np.float64))
                        if seq in meta else None)
    windows = featurize(read_windows(args.bars), load)
    train = [w for w in windows if w["split"] == "train" and w["usable"]]
    M = np.concatenate([w["motion"] for w in train])
    X = np.concatenate([w["music"] for w in train])
    ms, xs = StandardScaler().fit(M), StandardScaler().fit(X)
    cca = CCA(n_components=4, max_iter=1000).fit(xs.transform(X), ms.transform(M))
    km = KMeans(n_clusters=args.clusters, n_init=10, random_state=0).fit(
        cca.transform(xs.transform(X), ms.transform(M))[1])
    model = {"motion_scaler": ms, "music_scaler": xs, "cca": cca, "kmeans": km}
    assign = make_assign(model)
    labels = split_label_arrays(windows, assign)
    reproduced = {}
    if args.check:
        for split, array in labels.items():
            published = np.load(pathlib.Path(args.check) / split / "labels.npy")
            reproduced[split] = bool(array.dtype == published.dtype and array.shape == published.shape
                                     and np.array_equal(array, published))
        print("reproduces {}: {}".format(args.check, reproduced))
        if not all(reproduced.values()):
            print("refusing to save: the refit is not the published space", file=sys.stderr)
            return 1
    payload = pickle.dumps(model)
    out = pathlib.Path(args.model_out)
    if out.exists():
        raise SystemExit("{} exists; a fitted space is written once".format(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    meta_out = {
        "label_space_id": LABEL_SPACE_ID, "clusters": args.clusters, "sklearn": sklearn_version,
        "fit_inputs": {"bars": str(args.bars), "bars_sha256": sha256_file(args.bars),
                       "sequences": str(args.sequences), "sequences_sha256": sha256_file(args.sequences)},
        "fit_train_bars": int(len(M)),
        "reproduces_published_labels": reproduced or None,
        "model_sha256": sha256_bytes(payload),
        "motion_units": "the normalizer the fit's sequences were normalized with; assign must use the same",
    }
    out.with_suffix(".json").write_text(json.dumps(meta_out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta_out, indent=1))
    return 0


def make_assign(model: Mapping) -> callable:
    ms, xs, cca, km = model["motion_scaler"], model["music_scaler"], model["cca"], model["kmeans"]
    return lambda feats, tune: km.predict(cca.transform(xs.transform(tune), ms.transform(feats))[1])


def command_assign(args) -> int:
    model_bytes = pathlib.Path(args.model).read_bytes()
    model = pickle.loads(model_bytes)
    rows = {}
    for line in open(args.sequences, encoding="utf-8"):
        r = json.loads(line)
        rows[r["sequence_id"]] = r
    root = pathlib.Path(args.sequences).resolve().parent

    def load(seq):
        r = rows.get(seq)
        if r is None:
            return None
        raw = np.load(root / r["motion_path"])
        motion = normalize_like_the_fit(raw, pathlib.Path(args.normalizer)).astype(np.float64)
        music = np.load(root / r["music_path"]).astype(np.float64)
        return motion, music

    raw_windows = read_windows(args.bars)
    windows = featurize(raw_windows, load)
    missing = sorted({w["sequence_id"] for w in raw_windows} - {w["sequence_id"] for w in windows})
    if missing:
        raise SystemExit("{} bar-release sequences are not in {}: {}".format(len(missing), args.sequences, missing[:3]))
    labels = split_label_arrays(windows, make_assign(model))
    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit("{} exists; releases publish into a new directory".format(out))
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    src = pathlib.Path(args.src_bar_release)
    for split, array in labels.items():
        dst = staging / split
        dst.mkdir(parents=True)
        for name in ("music.npy", "names.json", "motion.npy", "retrieval_groups.json"):
            if (src / split / name).is_file():
                shutil.copy2(src / split / name, dst / name)
        existing = np.load(src / split / "labels.npy", mmap_mode="r") if (src / split / "labels.npy").is_file() else None
        if existing is not None and existing.shape != array.shape:
            raise SystemExit("{} {}: {} rows x {} bars in the source, {} here".format(
                src, split, existing.shape[0], existing.shape[1], array.shape))
        np.save(dst / "labels.npy", array)
    shutil.copy2(args.bars, staging / "bars.jsonl")
    hist = {split: np.bincount(a.reshape(-1), minlength=int(model["kmeans"].n_clusters)).tolist()
            for split, a in labels.items()}
    unusable = sum(1 for w in windows if not w["usable"])
    (staging / "aligned_labels_build.json").write_text(json.dumps({
        "derived_from": str(src), "label_space_id": LABEL_SPACE_ID,
        "labels": "ASSIGNED by a frozen model; nothing was refit",
        "model": str(args.model), "model_sha256": sha256_bytes(model_bytes),
        "motion_normalized_with": str(args.normalizer), "normalizer_sha256": sha256_file(args.normalizer),
        "motion_sequences": str(args.sequences), "bars": str(args.bars),
        "rows_zero_filled_as_unusable": unusable, "label_histogram": hist,
        "tool": "tools/label_bars_music_aligned.py assign",
    }, indent=2) + "\n", encoding="utf-8")
    staging.rename(out)
    print("assigned", {s: a.shape for s, a in labels.items()}, "hist", hist, "unusable rows", unusable, "->", out)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--bars", required=True, type=pathlib.Path)
    fit.add_argument("--sequences", required=True, type=pathlib.Path,
                     help="sequences_normalized.jsonl the space was fitted on (absolute motion/music paths)")
    fit.add_argument("--clusters", type=int, default=8)
    fit.add_argument("--check", type=pathlib.Path, default=None,
                     help="published bar release whose labels.npy the refit must reproduce")
    fit.add_argument("--model-out", required=True, type=pathlib.Path)
    fit.set_defaults(func=command_fit)
    asg = sub.add_parser("assign")
    asg.add_argument("--model", required=True, type=pathlib.Path)
    asg.add_argument("--bars", required=True, type=pathlib.Path, help="bars.jsonl of the bar release to label")
    asg.add_argument("--src-bar-release", required=True, type=pathlib.Path,
                     help="bar release whose music/names/motion/retrieval_groups are copied")
    asg.add_argument("--sequences", required=True, type=pathlib.Path,
                     help="RAW sequences.jsonl (motion_path = motion_151_raw) of the bundle the bars index")
    asg.add_argument("--normalizer", required=True, type=pathlib.Path,
                     help="normalizer.pt the space was FITTED in (not the new corpus's)")
    asg.add_argument("--out", required=True, type=pathlib.Path)
    asg.set_defaults(func=command_assign)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
