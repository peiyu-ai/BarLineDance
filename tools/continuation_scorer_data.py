#!/usr/bin/env python3
"""Per-recording continuation frames (model.continuation_scorer.continuation_frames) and 4-beat bar grids for the
continuation scorer: the v5 wild corpus (training / held-out val) and the T1/T2 libraries (the real retrieval
pools the scorer is checked on).  Everything lands under /cache/atomicdance-assets/runs/continuation_scorer_cache
(CLAUDE.md 1.2: nothing large on /workspace).

v5 (release_v3_timebase).  Windows are 150 frames at stride 15 and overlapping windows hold IDENTICAL frames
(checked on the val split: max |difference| 0.0, root included), so a recording is re-joined by placing every
window at its ``start_frame`` -- the same stitching runs/ext_20260923/contmodel_build_v5.py did for kinetics -- and
then decoded ONCE to SMPL joints with the release's own normalizer (infer_atomic.decode_motion).  Bars follow
contmodel_build_v5.bar_grid exactly: the music's beat channel (34), the 4-beat phase with the largest mean onset
(channel 0) at its downbeats, bars of 18-120 frames.  Music is read only to cut bars of the TRAINING recordings.

T libraries.  The train split re-joined by ``start_frame``, decoded, and cut by the T segmentation
(runs/txy_t{,2}_seg_beat4/segmentation.json) into bars labelled by their majority k8 label -- the same material as
runs/ext_20260923/contmodel_build_t2lib.py (joints are kept: the logistic baseline's join-band subset needs them).

usage: python3 tools/continuation_scorer_data.py {v5-train|v5-val|t1|t2|all} [--workers N]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import pickle
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CACHE = pathlib.Path("/cache/atomicdance-assets/runs/continuation_scorer_cache")
V5_ROOT = pathlib.Path("/cache/atomicdance-assets/scratch/v5rekey/release_v3_timebase")
LIBS = {
    "t1": (pathlib.Path("/cache/atomicdance-assets/scratch/txy_t/release_aligned_k8"),
           pathlib.Path("/cache/atomicdance-assets/runs/txy_t_seg_beat4/segmentation.json")),
    "t2": (pathlib.Path("/cache/atomicdance-assets/scratch/txy_t2/release_aligned_k8_20260922"),
           pathlib.Path("/cache/atomicdance-assets/runs/txy_t2_seg_beat4/segmentation.json")),
}


def bar_grid(beats, onset):
    """runs/ext_20260923/contmodel_build_v5.bar_grid, verbatim."""
    if len(beats) < 9:
        return []
    best = max(range(4), key=lambda ph: onset[beats[ph::4]].mean())
    down = beats[best::4]
    return [(int(a), int(b)) for a, b in zip(down[:-1], down[1:]) if 18 <= b - a <= 120]


def read_windows(root, splits):
    """{recording: [(start_frame, array_index), ...]}, {recording: split}, {recording: retrieval_group_id}."""
    recs, split_of, group = collections.defaultdict(list), {}, {}
    with open(root / "windows.jsonl") as fh:
        for line in fh:
            w = json.loads(line)
            if w["split"] not in splits:
                continue
            r = w.get("recording_id") or w["sequence_id"]
            recs[r].append((int(w["start_frame"]), int(w["array_index"]), int(w["end_frame_exclusive"])))
            split_of[r] = w["split"]
            group[r] = w["retrieval_group_id"]
    return recs, split_of, group


def _stitch(arr, windows):
    windows = sorted(windows)
    total = max(s + (e - s) for s, _a, e in windows)
    out = np.zeros((total,) + arr.shape[2:], dtype=np.float32)
    for s, a, e in windows:
        out[s:e] = arr[a][: e - s]
    return out


_W = {}


def _init(root, split, normalizer):
    import torch
    torch.set_num_threads(1)
    _W["motion"] = np.load(root / split / "motion.npy", mmap_mode="r")
    music = root / split / "music.npy"
    _W["music"] = np.load(music, mmap_mode="r") if music.is_file() else None
    _W["labels"] = np.load(root / split / "labels.npy", mmap_mode="r")
    _W["normalizer"] = str(normalizer)


def _decode(motion):
    import torch
    from infer_atomic import decode_motion
    return np.asarray(decode_motion(torch.from_numpy(motion), _W["normalizer"])["full_pose"], np.float32)


def _v5_job(item):
    from model.continuation_scorer import continuation_frames
    rec, windows = item
    joints = _decode(_stitch(_W["motion"], windows))
    frames = continuation_frames(joints)
    # music: only the onset envelope (0) and beat track (34), stitched the same way, to cut the bar grid
    m = np.zeros((len(frames), 2), np.float32)
    for s, a, e in sorted(windows):
        m[s:e] = np.asarray(_W["music"][a][: e - s][:, [0, 34]])
    beats = np.flatnonzero(m[:, 1] > 0.5)
    return rec, frames, bar_grid(beats, m[:, 0]), None


def _lib_job(item):
    from model.continuation_scorer import continuation_frames
    rec, windows = item
    joints = _decode(_stitch(_W["motion"], windows))
    ws = sorted(windows)
    lab = np.full(len(joints), -1, np.int64)
    for s, a, e in ws:
        lab[s:e] = _W["labels"][a][: e - s]
    return rec, continuation_frames(joints), joints, lab


def build_v5(split, workers):
    from multiprocessing import Pool
    t0 = time.time()
    recs, _split_of, group = read_windows(V5_ROOT, {split})
    items = sorted(recs.items())
    print("v5 {}: {} recordings, {} windows ({:.0f}s to index)".format(
        split, len(items), sum(len(v) for v in recs.values()), time.time() - t0), flush=True)
    frames, meta, offset = [], [], 0
    with Pool(workers, initializer=_init, initargs=(V5_ROOT, split, V5_ROOT / "normalizer.pt")) as pool:
        for n, (rec, f, bars, _m) in enumerate(pool.imap(_v5_job, items, chunksize=4)):
            frames.append(f)
            meta.append({"rec": rec, "group": group[rec], "upload": rec.split(":")[1], "offset": offset,
                         "length": len(f), "bars": bars})
            offset += len(f)
            if n % 1000 == 0:
                print("  {} / {}  {:.0f}s".format(n, len(items), time.time() - t0), flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "v5_{}_frames.npy".format(split), np.concatenate(frames).astype(np.float32))
    with open(CACHE / "v5_{}_meta.pkl".format(split), "wb") as fh:
        pickle.dump({"recs": meta, "root": str(V5_ROOT), "feature": _feature_version()}, fh, protocol=4)
    nb = sum(len(m["bars"]) for m in meta)
    npair = sum(sum(1 for x, y in zip(m["bars"][:-1], m["bars"][1:]) if x[1] == y[0]) for m in meta)
    print("v5 {} done: {} recordings, {} frames, {} bars, {} contiguous pairs, {:.0f}s".format(
        split, len(meta), offset, nb, npair, time.time() - t0), flush=True)


def build_lib(name, workers):
    from multiprocessing import Pool
    root, seg_path = LIBS[name]
    t0 = time.time()
    recs, _s, group = read_windows(root, {"train"})
    seg = {r["sequence"]: r for r in json.load(open(seg_path))["records"]}
    items = sorted(recs.items())
    frames, joints_all, meta, offset = [], [], [], 0
    with Pool(workers, initializer=_init, initargs=(root, "train", root / "normalizer.pt")) as pool:
        for rec, f, joints, lab in pool.imap(_lib_job, items, chunksize=2):
            stem = rec.split(":", 1)[1].replace(":", "__")
            bars = []
            if stem in seg:
                for sgm in seg[stem]["segments"]:
                    a, b = int(sgm["start"]), int(sgm["end"])
                    if b <= len(f):
                        vals, cnt = np.unique(lab[a:b], return_counts=True)
                        bars.append((a, b, int(vals[np.argmax(cnt)]), float(cnt.max() / max(1, b - a))))
            frames.append(f)
            joints_all.append(joints)
            meta.append({"rec": rec, "group": group[rec], "upload": rec.split(":")[1], "offset": offset,
                         "length": len(f), "bars": bars})
            offset += len(f)
    np.save(CACHE / "{}_frames.npy".format(name), np.concatenate(frames).astype(np.float32))
    np.save(CACHE / "{}_joints.npy".format(name), np.concatenate(joints_all).astype(np.float32))
    with open(CACHE / "{}_meta.pkl".format(name), "wb") as fh:
        pickle.dump({"recs": meta, "root": str(root), "segmentation": str(seg_path),
                     "feature": _feature_version()}, fh, protocol=4)
    nb = sum(len(m["bars"]) for m in meta)
    npair = sum(sum(1 for x, y in zip(m["bars"][:-1], m["bars"][1:]) if x[1] == y[0]) for m in meta)
    print("{} done: {} recordings, {} frames, {} bars, {} consecutive pairs, {:.0f}s".format(
        name, len(meta), offset, nb, npair, time.time() - t0), flush=True)


def _feature_version():
    from model.continuation_scorer import FEATURE_VERSION
    return FEATURE_VERSION


def load(name):
    """(frames [F, C] float32, meta dict) for v5_train / v5_val / t1 / t2 (t1/t2 also carry meta["joints"])."""
    stem = name
    frames = np.load(CACHE / "{}_frames.npy".format(stem))
    meta = pickle.load(open(CACHE / "{}_meta.pkl".format(stem), "rb"))
    joints = CACHE / "{}_joints.npy".format(stem)
    if joints.is_file():
        meta["joints"] = np.load(joints, mmap_mode="r")
    return frames, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["v5-train", "v5-val", "t1", "t2", "all"])
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    todo = ["t2", "t1", "v5-val", "v5-train"] if args.what == "all" else [args.what]
    for what in todo:
        if what.startswith("v5-"):
            build_v5(what[3:], args.workers)
        else:
            build_lib(what, min(args.workers, 32))


if __name__ == "__main__":
    main()
