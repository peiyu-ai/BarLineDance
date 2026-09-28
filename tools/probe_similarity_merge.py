#!/usr/bin/env python3
"""Does merging adjacent segments *by embedding similarity* beat merging them at random?

WHY THIS EXISTS.  The operator's proposal: "how many movements to merge should
not be a hand-picked constant, it should follow the similarity between adjacent
motions".  Reference [16] (PSVL) does NOT do this -- it populates every
consecutive combination and samples uniformly, and its own Table 3 (bottom)
shows the two similarity-flavoured scoring functions it tried (compactness,
diversity) LOSING to uniform random sampling.  So this file measures the
proposal directly rather than transplanting anything.

WHAT IT BUILDS.  Base arm = Alg.1 output (runs/wild_v4_seg_r0visual).  Each
base segment is embedded with the pipeline's own segment encoder (TMR mu, the
same path tools/cluster_atomics_tmr.py:encode_segments takes).  For a clip with
segments s_0..s_{n-1}, d_i = cosine distance(mu(s_i), mu(s_{i+1})) is attached
to the interior boundary between them.  A threshold is a *quantile q of the
pooled d over the whole corpus*, so nothing is hand-picked: q is swept.

Three arms per q, all removing the SAME number of interior boundaries per clip:

* ``sim{q}``  -- remove the q-fraction with the SMALLEST d.  The proposal.
* ``rnd{q}``  -- remove a uniformly random subset of the same size.  This is
  the control that separates "similarity-driven" from "merely coarser"; the
  plain length sweep already showed coarser alone does nothing.
* ``anti{q}`` -- remove the q-fraction with the LARGEST d.  The direction
  check.  If similarity-driven merging helps, this must hurt; if sim and anti
  read the same, d carries no information about where a cut belongs.

GUARD.  A clip is never merged below 2 segments (the settle probe needs an
interior cut, and a shrinking clip set across arms would not be one corpus).
When a q would remove every interior boundary of a clip, the boundary with the
largest d is retained.  The count of clips where that fires is reported.

SINGLE PASS, STATED.  Distances are computed once on the base segments and not
recomputed after a merge.  An agglomerative variant would re-encode the merged
span each round; that is a different algorithm and is not what is measured here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.convert_motion_to_guofeats import GuofeatsError, motion_151_to_guofeats  # noqa: E402
from tools.probe_segmentation_boundaries import canonical_key  # noqa: E402


def load_bundle_rows(bundle: pathlib.Path) -> dict:
    rows = {}
    with open(bundle / "sequences.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[canonical_key(row.get("recording_id") or row["sequence_id"])] = row
    return rows


def encode(bundle, clips_path, seg_path, cache_path, device, batch_size, min_frames=18):
    """{key: (boundaries, mu[n,256], halves_mu[n,2,256])} for every listed clip."""
    if cache_path.is_file():
        blob = np.load(cache_path, allow_pickle=True)
        return {k: v for k, v in blob["payload"].item().items()}
    rows = load_bundle_rows(bundle)
    seg = json.loads(pathlib.Path(seg_path).read_text(encoding="utf-8"))
    arm = {canonical_key(r["sequence"]): r for r in seg["records"]}
    wanted = [canonical_key(l.strip()) for l in
              pathlib.Path(clips_path).read_text(encoding="utf-8").splitlines() if l.strip()]

    from tools.tmr_runtime import TMREncoder
    encoder = TMREncoder(device=device, text=False)

    payload, features, owners = {}, [], []
    skipped = {"no_row": 0, "no_arm": 0, "stale": 0, "convert": 0, "too_short_clip": 0}
    for key in wanted:
        row, record = rows.get(key), arm.get(key)
        if row is None:
            skipped["no_row"] += 1
            continue
        if record is None:
            skipped["no_arm"] += 1
            continue
        bounds = [int(b) for b in record["boundaries"]]
        if len(bounds) < 3 or bounds[-1] != int(row["frame_count"]):
            skipped["stale"] += 1
            continue
        try:
            motion = np.load(bundle / row["assets"]["motion_151_raw"])
            conv = motion_151_to_guofeats(motion)
        except (GuofeatsError, IndexError, ValueError):
            skipped["convert"] += 1
            continue
        guo, index = conv["features"], conv["source_frame_index"]
        spans, ok = list(zip(bounds[:-1], bounds[1:])), True
        slices = []
        for start, end in spans:
            lo = int(np.searchsorted(index, start, side="left"))
            hi = int(np.searchsorted(index, end, side="right"))
            if hi - lo < 4:
                ok = False
                break
            slices.append((lo, hi))
        if not ok:
            skipped["too_short_clip"] += 1
            continue
        payload[key] = {"boundaries": bounds, "n": len(spans)}
        for lo, hi in slices:
            features.append(guo[lo:hi])
            owners.append((key, "full"))
            mid = lo + (hi - lo) // 2
            # halves: the SAME movement cut in two.  Positive control for
            # "a pair that genuinely should be merged".
            a, b = guo[lo:max(mid, lo + 2)], guo[min(mid, hi - 2):hi]
            features.append(a); owners.append((key, "h0"))
            features.append(b); owners.append((key, "h1"))
    embeddings = encoder.encode_motion(features, batch_size=batch_size)
    cursor = {}
    for pos, (key, kind) in enumerate(owners):
        cursor.setdefault(key, {"full": [], "h0": [], "h1": []})[kind].append(pos)
    for key, buckets in cursor.items():
        payload[key]["mu"] = embeddings[buckets["full"]]
        payload[key]["h0"] = embeddings[buckets["h0"]]
        payload[key]["h1"] = embeddings[buckets["h1"]]
    payload["_skipped"] = skipped
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, payload=np.asarray(payload, dtype=object))
    return payload


def cosine_distance(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return 1.0 - (a * b).sum(-1)


def merged_boundaries(bounds, distances, drop_mask):
    """Interior boundary i (between segment i and i+1) is dropped where mask is True."""
    keep = [bounds[0]]
    for i, b in enumerate(bounds[1:-1]):
        if not drop_mask[i]:
            keep.append(int(b))
    keep.append(bounds[-1])
    return keep


def write_arm(path, per_clip):
    records = []
    for key, bounds in sorted(per_clip.items()):
        records.append({
            "sequence": key.replace(":", "__"),
            "boundaries": [int(b) for b in bounds],
            "segments": [{"start": int(a), "end": int(b), "frames": int(b - a)}
                         for a, b in zip(bounds[:-1], bounds[1:])],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=pathlib.Path, required=True)
    p.add_argument("--clips", type=pathlib.Path, required=True)
    p.add_argument("--arm", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--cache", type=pathlib.Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--quantiles", default="0.2,0.4,0.6,0.8")
    p.add_argument("--seed", type=int, default=20260820)
    args = p.parse_args()

    payload = encode(args.bundle, args.clips, args.arm, args.cache,
                     args.device, args.batch_size)
    skipped = payload.pop("_skipped", {})
    # TMR's vendored guofeats transform emits a non-finite row on rare inputs
    # (one clip in this corpus, one row of 280).  A single NaN poisons every
    # pooled quantile, so the clip is dropped -- not patched -- and counted,
    # because every arm has to be scored on the same clips.
    nonfinite = [k for k, row in payload.items()
                 if not (np.isfinite(row["mu"]).all() and np.isfinite(row["h0"]).all()
                         and np.isfinite(row["h1"]).all())]
    for key in nonfinite:
        payload.pop(key)
    print("clips encoded {} | skipped {} | dropped for non-finite embedding {}".format(
        len(payload), skipped, len(nonfinite)))

    rng = np.random.default_rng(args.seed)
    dists, halves, randpairs = {}, [], []
    for key, row in payload.items():
        mu = row["mu"]
        dists[key] = cosine_distance(mu[:-1], mu[1:])
        halves.append(cosine_distance(row["h0"], row["h1"]))
        n = len(mu)
        if n >= 4:
            pairs = rng.integers(0, n, (4 * n, 2))
            pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > 1]
            if len(pairs):
                randpairs.append(cosine_distance(mu[pairs[:, 0]], mu[pairs[:, 1]]))
    pooled = np.concatenate([d for d in dists.values() if len(d)])
    halves = np.concatenate(halves)
    randpairs = np.concatenate(randpairs)
    print("interior boundaries {} | adjacent d: median {:.4f} mean {:.4f} "
          "p05 {:.4f} p95 {:.4f}".format(len(pooled), np.median(pooled), pooled.mean(),
                                         np.percentile(pooled, 5), np.percentile(pooled, 95)))
    print("halves-of-one-segment d (should-merge control): n {} median {:.4f}".format(
        len(halves), np.median(halves)))
    print("non-adjacent same-clip d (should-not-merge control): n {} median {:.4f}".format(
        len(randpairs), np.median(randpairs)))

    stats = {"adjacent": pooled.tolist()[:0], "n_adjacent": int(len(pooled)),
             "adjacent_quantiles": {str(q): float(np.quantile(pooled, q))
                                    for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)},
             "halves_median": float(np.median(halves)),
             "nonadjacent_median": float(np.median(randpairs)),
             "arms": {}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "distances.npz", adjacent=pooled,
                        halves=halves, nonadjacent=randpairs)

    for q in [float(x) for x in args.quantiles.split(",")]:
        tau = float(np.quantile(pooled, q))
        arms = {"sim": {}, "rnd": {}, "anti": {}}
        guarded = 0
        total_dropped = 0
        for key, row in payload.items():
            bounds, d = row["boundaries"], dists[key]
            m = len(d)                                   # interior boundaries
            want = int((d < tau).sum())
            if want >= m:                                # guard: keep >= 2 segments
                want = m - 1
                guarded += 1
            total_dropped += want
            order = np.argsort(d)
            sim = np.zeros(m, dtype=bool); sim[order[:want]] = True
            anti = np.zeros(m, dtype=bool); anti[order[m - want:]] = True if want else False
            rnd = np.zeros(m, dtype=bool)
            if want:
                rnd[rng.choice(m, size=want, replace=False)] = True
            arms["sim"][key] = merged_boundaries(bounds, d, sim)
            arms["anti"][key] = merged_boundaries(bounds, d, anti)
            arms["rnd"][key] = merged_boundaries(bounds, d, rnd)
        tag = "q{:02d}".format(int(round(q * 100)))
        for kind, per_clip in arms.items():
            name = "{}{}".format(kind, tag)
            write_arm(args.out_dir / "{}.json".format(name), per_clip)
            lengths = np.concatenate([np.diff(b) for b in per_clip.values()])
            stats["arms"][name] = {
                "threshold": tau, "quantile": q,
                "segments": int(sum(len(b) - 1 for b in per_clip.values())),
                "median_seconds": float(np.median(lengths) / 30.0),
                "mean_seconds": float(lengths.mean() / 30.0),
                "clips_hitting_guard": guarded,
                "boundaries_dropped": total_dropped,
            }
        print("q={:.2f} tau={:.4f}  dropped {}/{} interior boundaries, "
              "guard fired on {} clips, median {:.2f}s".format(
                  q, tau, total_dropped, len(pooled), guarded,
                  stats["arms"]["sim" + tag]["median_seconds"]))
    (args.out_dir / "merge_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("arms -> {}".format(args.out_dir))


if __name__ == "__main__":
    main()
