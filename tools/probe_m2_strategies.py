#!/usr/bin/env python3
"""Sweep clustering strategies for M2, scored on held-out uploads.

The question
------------
``tools/probe_m2_headroom.py`` established two things about clean5b5: K is not
the bottleneck (TMR-fitted median ratio moves only 0.033 from K=20 to K=375),
and the representation is (pose-fitted reaches 0.855 at K=53 against TMR's
0.930).  The operator's call was to stop worrying about downstream metrics for
now and find a strategy that actually makes segments cluster.

So this sweeps representations and algorithms against the coherence ratio.

The trap, stated plainly
------------------------
**This tool optimises a criterion this repository invented.**  Picking the
winner by best ratio is exactly the move CLAUDE.md section 2.1 warns about: the
2026-08-19 ``boundary contrast`` episode selected a fusion weight against a
metric that turned out to reward the defect it was supposed to remove.  Two
guards, neither of which makes the result safe, only interpretable:

1. **Everything is scored out of sample.**  K-Means is fitted on half the
   uploads and the other half is assigned by nearest centroid; the ratio is
   computed on the held-out half only.  Split by upload, not by segment, because
   consecutive segments of one dancer are near-duplicates.

2. **A second criterion nothing here optimises: account lift.**  How much a
   partition concentrates the choreographer account, against a permutation null.
   High is *bad* -- an account is a person, a room and a camera.  A strategy that
   improves the ratio by discovering identity rather than movement will show it
   here, and that is the one failure mode the ratio cannot see by construction.

Read the two together.  A strategy that improves the ratio **and** holds or
lowers account lift is doing something real.  One that improves the ratio while
raising lift has found the dancer, not the dance.

A note on ``de-account`` preprocessing
--------------------------------------
Subtracting each upload's own mean vector is included because accounts are known
to be enriched (lift 1.274).  It is *not* obviously fair: it uses upload identity
at fit time, and the ratio already excludes same-upload pairs, so the two
interact in a way this sweep does not disentangle.  It is reported and flagged,
not recommended on its ratio alone.

On the reject option
--------------------
K-Means must assign every point, which is how a junk drawer forms: prototype 8
holds 42 of the corpus's most peripheral segments because they had to go
somewhere.  ``reject`` drops the farthest fraction from every cluster after
fitting and reports the coverage it kept, because a strategy that clusters 60%
of the corpus beautifully is a different product from one that clusters all of
it -- and the ratio alone would call the first one better.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, Optional

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.probe_prototype_coherence import table  # noqa: E402


# ---------------------------------------------------------------- criteria

def summarise_ratio(rows: dict) -> dict:
    raw = np.array([r["ratio"] for r in rows.values()], dtype=np.float64)
    readable = np.isfinite(raw)
    ratios = raw[readable]
    return {
        "groups": int(len(raw)),
        "groups_readable": int(readable.sum()),
        "median_ratio": float(np.median(ratios)),
        "q25_ratio": float(np.percentile(ratios, 25)),
        "n_below_085": int((ratios < 0.85).sum()),
        "n_at_or_above_1": int((ratios >= 1.0).sum()),
        "frac_at_or_above_1": float((ratios >= 1.0).mean()),
    }


def account_lift(labels: np.ndarray, accounts: np.ndarray, uploads: np.ndarray,
                 seed: int, draws: int = 200) -> dict:
    """Purity of the account inside a group, against an entity-level null.

    The permutation reassigns accounts **between uploads**, not between
    segments: an upload belongs wholly to one account, and a segment-level
    shuffle would be beaten by any partition that merely keeps one upload
    together.  Group sizes, within-upload cohesion and the account's marginal
    distribution all stay at their observed values.
    """
    def purity(acct: np.ndarray) -> float:
        total = 0
        for group in np.unique(labels):
            members = acct[labels == group]
            if len(members):
                total += collections.Counter(members.tolist()).most_common(1)[0][1]
        return total / len(labels)

    observed = purity(accounts)
    upload_of = {u: i for i, u in enumerate(np.unique(uploads))}
    upload_index = np.array([upload_of[u] for u in uploads])
    upload_account = np.empty(len(upload_of), dtype=accounts.dtype)
    for u, i in upload_of.items():
        upload_account[i] = accounts[uploads == u][0]

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(draws):
        null.append(purity(rng.permutation(upload_account)[upload_index]))
    null = np.array(null)
    return {"observed_purity": float(observed),
            "null_purity_mean": float(null.mean()),
            "null_purity_sd": float(null.std(ddof=1)),
            "lift": float(observed / null.mean())}


# ------------------------------------------------------------ representations

def zscore(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd < 1e-9] = 1.0
    return (x - mu) / sd


def deaccount(x: np.ndarray, uploads: np.ndarray) -> np.ndarray:
    """Subtract each upload's own mean, removing a per-dancer offset."""
    out = x.copy()
    for u in np.unique(uploads):
        mask = uploads == u
        out[mask] -= out[mask].mean(0)
    return out


def build_space(name: str, tmr: np.ndarray, pose: np.ndarray, dyn: np.ndarray,
                uploads: np.ndarray, train: np.ndarray) -> Optional[np.ndarray]:
    """All spaces are standardised against the TRAIN half only."""
    if name == "tmr":
        return tmr
    if name == "tmr_l2":
        norm = np.linalg.norm(tmr, axis=1, keepdims=True)
        return tmr / np.maximum(norm, 1e-9)
    if name == "tmr_z":
        return zscore(tmr, tmr[train])
    if name == "tmr_whiten":
        from sklearn.decomposition import PCA
        model = PCA(n_components=min(128, tmr.shape[1]), whiten=True,
                    random_state=0).fit(tmr[train])
        return model.transform(tmr)
    if name == "pose":
        return pose
    if name == "pose_z":
        return zscore(pose, pose[train])
    if name == "pose_dyn":
        return np.hstack([zscore(pose, pose[train]), zscore(dyn, dyn[train])])
    if name.startswith("fuse") and not name.endswith("_deacct"):
        # The exact-match de-account branch below sits *after* this prefix test,
        # so without the suffix guard "fuse_0.5_deacct" is captured here and
        # silently returns the plain fusion -- a second arm reporting the first
        # arm's numbers under a different name.
        weight = float(name.split("_")[1])
        a = zscore(tmr, tmr[train])
        b = zscore(pose, pose[train])
        # Scale so each block contributes comparably before weighting: without
        # this the 264-dim pose block would outweigh the 256-dim TMR block only
        # by dimension count, which is not a choice anyone made.
        a = a / np.sqrt(a.shape[1])
        b = b / np.sqrt(b.shape[1])
        return np.hstack([(1.0 - weight) * a, weight * b])
    if name == "tmr_deacct":
        return deaccount(zscore(tmr, tmr[train]), uploads)
    if name == "pose_deacct":
        return deaccount(zscore(pose, pose[train]), uploads)
    if name.startswith("fuse") and name.endswith("_deacct"):
        base = build_space(name[: -len("_deacct")], tmr, pose, dyn, uploads, train)
        return deaccount(base, uploads)
    raise ValueError("unknown space {!r}".format(name))


# ---------------------------------------------------------------- algorithms

def fit_predict(algorithm: str, space: np.ndarray, train: np.ndarray,
                score: np.ndarray, k: int, seed: int):
    """Returns (labels_on_score, distance_to_own_centroid_on_score)."""
    if algorithm in ("kmeans", "reject"):
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(space[train])
        labels = model.predict(space[score])
        centres = model.cluster_centers_[labels]
    elif algorithm == "spherical":
        from sklearn.cluster import KMeans
        unit = space / np.maximum(np.linalg.norm(space, axis=1, keepdims=True), 1e-9)
        model = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(unit[train])
        labels = model.predict(unit[score])
        centres = model.cluster_centers_[labels]
        return labels, np.linalg.norm(unit[score] - centres, axis=1)
    elif algorithm == "gmm":
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=k, covariance_type="diag",
                                random_state=seed, max_iter=100).fit(space[train])
        labels = model.predict(space[score])
        centres = model.means_[labels]
    elif algorithm == "ward":
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import NearestCentroid
        fitted = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
            space[train])
        nc = NearestCentroid().fit(space[train], fitted)
        labels = nc.predict(space[score])
        index = {c: i for i, c in enumerate(nc.classes_)}
        centres = nc.centroids_[[index[l] for l in labels]]
    else:
        raise ValueError(algorithm)
    return labels, np.linalg.norm(space[score] - centres, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=pathlib.Path,
                        default=pathlib.Path("/dev/shm/atomicdance-m2-review/"
                                             "segment_features_spans.npz"))
    parser.add_argument("--embeddings", type=pathlib.Path,
                        default=pathlib.Path("/dev/shm/atomicdance-m2-review/"
                                             "clean5b5_tmr_embeddings.npz"))
    parser.add_argument("--group-keys", type=pathlib.Path,
                        default=REPO / "runs/wild_v4_group_keys.json")
    parser.add_argument("-k", type=int, default=53)
    parser.add_argument("--reject-frac", type=float, default=0.25,
                        help="fraction dropped per cluster by the reject arm")
    parser.add_argument("--output", type=pathlib.Path,
                        default=REPO / "runs/clean5b5_m2_strategies.json")
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=False)
    pose = data["vector"].astype(np.float64)
    uploads = data["upload"]
    dyn = np.column_stack([
        (data["end"] - data["start"]) / 30.0,
        data["max_step"], data["head_below"],
        data["segment_index"] / np.maximum(data["segments_in_clip"], 1),
    ]).astype(np.float64)

    blob = np.load(args.embeddings, allow_pickle=True)
    index = {(str(r), int(s), int(e)): i for i, (r, s, e) in enumerate(
        zip(blob["recordings"], blob["starts"], blob["ends"]))}
    rows = [index.get((str(r), int(s), int(e))) for r, s, e in
            zip(data["recording"], data["start"], data["end"])]
    if any(r is None for r in rows):
        raise SystemExit("{} segments have no embedding; rebuild the feature dump "
                         "with --spans-from".format(sum(1 for r in rows if r is None)))
    tmr = np.asarray(blob["embeddings"], dtype=np.float64)[np.array(rows)]

    keys = json.loads(args.group_keys.read_text(encoding="utf-8"))

    def account_of(recording: str) -> str:
        """`wild_v4:<upload>:clipNNN` -> the account that uploaded it.

        The map is keyed by the bare upload id, so every intermediate shape has
        to be tried rather than assumed: a lookup that missed would return "?"
        for every row and the account check -- the only criterion here that is
        not being optimised -- would silently become a constant.
        """
        text = str(recording)
        parts = text.split(":")
        for candidate in (parts[1] if len(parts) > 2 else None,
                          text.rsplit(":", 1)[0], text, parts[0]):
            if candidate is not None and candidate in keys:
                return keys[candidate]
        return "?"

    accounts = np.array([account_of(r) for r in data["recording"]])
    unknown = int((accounts == "?").sum())
    if unknown:
        raise SystemExit("{} of {} segments have no account in {}".format(
            unknown, len(accounts), args.group_keys))
    print("{} segments, tmr {}, pose {}, {} accounts".format(
        len(pose), tmr.shape, pose.shape, len(set(accounts.tolist()))), flush=True)

    rng = np.random.default_rng(args.seed)
    unique = np.unique(uploads)
    rng.shuffle(unique)
    train_uploads = set(unique[:len(unique) // 2].tolist())
    is_train = np.array([u in train_uploads for u in uploads])
    train, score = np.flatnonzero(is_train), np.flatnonzero(~is_train)
    su, sp, sa = uploads[score], pose[score], accounts[score]
    print("{} train / {} held out\n".format(len(train), len(score)), flush=True)

    baseline = {
        "m2_actual": (data["label"], None),
    }
    results = {}

    # M2's own labels, and a size-preserving permutation of them.
    for name, (labels, _) in baseline.items():
        r = summarise_ratio(table(sp, labels[score], su, args.seed))
        r["account"] = account_lift(labels[score], sa, su, args.seed)
        r["coverage"] = 1.0
        results[name] = r
    perm = rng.permutation(len(score))
    r = summarise_ratio(table(sp, data["label"][score][perm], su, args.seed))
    r["account"] = account_lift(data["label"][score][perm], sa, su, args.seed)
    r["coverage"] = 1.0
    results["shuffled"] = r

    spaces = ["tmr", "tmr_l2", "tmr_z", "tmr_whiten", "pose", "pose_z", "pose_dyn",
              "fuse_0.25", "fuse_0.5", "fuse_0.75",
              "tmr_deacct", "pose_deacct", "fuse_0.5_deacct"]
    header = "{:<24}{:>9}{:>8}{:>8}{:>9}{:>9}"
    print(header.format("strategy", "median", "<.85", ">=1.0", "acct_lift", "cover"),
          flush=True)
    print(header.format("m2_actual",
                        "{:.4f}".format(results["m2_actual"]["median_ratio"]),
                        results["m2_actual"]["n_below_085"],
                        results["m2_actual"]["n_at_or_above_1"],
                        "{:.3f}".format(results["m2_actual"]["account"]["lift"]),
                        "1.00"), flush=True)
    print(header.format("shuffled",
                        "{:.4f}".format(results["shuffled"]["median_ratio"]),
                        results["shuffled"]["n_below_085"],
                        results["shuffled"]["n_at_or_above_1"],
                        "{:.3f}".format(results["shuffled"]["account"]["lift"]),
                        "1.00"), flush=True)

    for space_name in spaces:
        space = build_space(space_name, tmr, pose, dyn, uploads, train)
        if space is None:
            continue
        for algorithm in ("kmeans",):
            labels, distance = fit_predict(algorithm, space, train, score,
                                           args.k, args.seed)
            key = "{}|{}".format(space_name, algorithm)
            r = summarise_ratio(table(sp, labels, su, args.seed))
            r["account"] = account_lift(labels, sa, su, args.seed)
            r["coverage"] = 1.0
            r["space"], r["algorithm"] = space_name, algorithm
            results[key] = r
            print(header.format(key, "{:.4f}".format(r["median_ratio"]),
                                r["n_below_085"], r["n_at_or_above_1"],
                                "{:.3f}".format(r["account"]["lift"]), "1.00"),
                  flush=True)

    # Algorithms, on the two spaces worth comparing on.
    for space_name in ("tmr", "fuse_0.5"):
        space = build_space(space_name, tmr, pose, dyn, uploads, train)
        for algorithm in ("spherical", "gmm", "ward"):
            try:
                labels, _ = fit_predict(algorithm, space, train, score,
                                        args.k, args.seed)
            except Exception as error:                        # noqa: BLE001
                print("{}|{} failed: {}".format(space_name, algorithm, error),
                      flush=True)
                continue
            key = "{}|{}".format(space_name, algorithm)
            r = summarise_ratio(table(sp, labels, su, args.seed))
            r["account"] = account_lift(labels, sa, su, args.seed)
            r["coverage"] = 1.0
            r["space"], r["algorithm"] = space_name, algorithm
            results[key] = r
            print(header.format(key, "{:.4f}".format(r["median_ratio"]),
                                r["n_below_085"], r["n_at_or_above_1"],
                                "{:.3f}".format(r["account"]["lift"]), "1.00"),
                  flush=True)

    # Reject option: drop the farthest members of every cluster.
    for space_name in ("tmr", "fuse_0.5", "pose_z"):
        space = build_space(space_name, tmr, pose, dyn, uploads, train)
        labels, distance = fit_predict("kmeans", space, train, score, args.k, args.seed)
        keep = np.ones(len(labels), dtype=bool)
        for group in np.unique(labels):
            mask = np.flatnonzero(labels == group)
            if len(mask) < 4:
                continue
            cut = np.quantile(distance[mask], 1.0 - args.reject_frac)
            keep[mask[distance[mask] > cut]] = False
        key = "{}|reject{:.0f}".format(space_name, args.reject_frac * 100)
        r = summarise_ratio(table(sp[keep], labels[keep], su[keep], args.seed))
        r["account"] = account_lift(labels[keep], sa[keep], su[keep], args.seed)
        r["coverage"] = float(keep.mean())
        r["space"], r["algorithm"] = space_name, "kmeans+reject"
        results[key] = r
        print(header.format(key, "{:.4f}".format(r["median_ratio"]),
                            r["n_below_085"], r["n_at_or_above_1"],
                            "{:.3f}".format(r["account"]["lift"]),
                            "{:.2f}".format(r["coverage"])), flush=True)

    # ---------------------------------------------------------------- cross-space
    # The circularity this answers: the primary ratio is measured in pose space,
    # so a pose-fitted clustering is being graded in the space it was fitted in
    # (held-out uploads or not).  Scoring every arm in BOTH spaces breaks the
    # symmetry: if pose-fitted groups are tight in pose *and* respectable in TMR
    # while TMR-fitted groups are tight only in TMR, that asymmetry is not
    # something either space's home advantage can produce.
    print("\ncross-space: fit in one, score the ratio in the other", flush=True)
    cross = {}
    eval_spaces = {"pose": pose, "tmr": tmr}
    print("{:<20}{:>14}{:>14}".format("fit space", "score:pose", "score:tmr"),
          flush=True)
    for fit_name in ("tmr", "pose", "fuse_0.5"):
        space = build_space(fit_name, tmr, pose, dyn, uploads, train)
        labels, _ = fit_predict("kmeans", space, train, score, args.k, args.seed)
        row = {}
        for eval_name, eval_vectors in eval_spaces.items():
            row[eval_name] = summarise_ratio(
                table(eval_vectors[score], labels, su, args.seed))["median_ratio"]
        cross[fit_name] = row
        print("{:<20}{:>14.4f}{:>14.4f}".format(
            fit_name, row["pose"], row["tmr"]), flush=True)
    perm2 = rng.permutation(len(score))
    row = {name: summarise_ratio(table(vec[score], data["label"][score][perm2],
                                       su, args.seed))["median_ratio"]
           for name, vec in eval_spaces.items()}
    cross["shuffled"] = row
    print("{:<20}{:>14.4f}{:>14.4f}".format("shuffled", row["pose"], row["tmr"]),
          flush=True)

    report = {
        "k": args.k,
        "cross_space": cross,
        "cross_space_note": "fit in the row's space, score the median coherence "
                            "ratio in the column's. The primary table grades pose "
                            "fits in pose space; this asks whether the advantage "
                            "survives being scored somewhere else.",
        "protocol": "fit on half the uploads, assign held-out half by nearest "
                    "centroid, score on the held-out half only",
        "primary": "median within/between ratio in signature-pose space, "
                   "cross-upload pairs only. lower is better.",
        "secondary": "account lift against an upload-level permutation null. "
                     "HIGH IS BAD. Nothing in this sweep optimises it, which is "
                     "why it is the check on the primary.",
        "warning": "this sweep selects on a criterion this repository invented. "
                   "A winner here is a hypothesis, not a result; it has to be "
                   "re-measured on whatever downstream metric the vocabulary is "
                   "actually for.",
        "segments_scored": int(len(score)),
        "reject_frac": args.reject_frac,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nwrote {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
