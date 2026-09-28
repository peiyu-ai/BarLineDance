"""Does the dance come back to a movement where the MUSIC comes back?

WHAT THIS IS NOT.  "Given this bar's music, name its class" was measured and
refuted: +0.005 over a train-fold majority floor, mean speed R^2 -0.31,
unevenness -0.51, settle phase -0.17, while the SAME features name which of 80
recordings a bar came from at 59x chance.  At bar resolution the absolute
pairing is not learnable, and no planner architecture fixes that.

WHAT THIS IS.  A relational question the refutation above does not answer:
whatever the classes are, do two bars that SOUND ALIKE carry the SAME one?  A
choreography can be unpredictable bar by bar and still repeat its chorus, so
this can have an answer where the first question has none -- and it does.

    statistic = mean(music cosine | same class) - mean(... | different class)
                over bar pairs at least --min-gap bars apart

THE NULL IS A CIRCULAR SHIFT, not a permutation.  Neighbouring bars both sound
alike and carry the same class, so shuffling bars freely manufactures a large
positive out of autocorrelation alone; a circular shift keeps both
autocorrelations and the whole distribution of |i-j|, and destroys only the
pairing between the two tracks.  Reported per clip against its own null, then
pooled -- one clip's cosines are not comparable with another's.

POSITIVE CONTROL, and it is not optional (CLAUDE.md section 2.1 rule 3): the
ground-truth row must read positive before any generated row may be read as
negative.  On the ten fixed T clips it does, by a wide margin:

    ground truth       obs +0.2595   null +0.1174   dz +3.17   10/10 clips
    aligned planner    obs +0.0750   null -0.0389   dz +0.94    4/5
    shipped 21-class   obs -0.1367   null +0.0101   dz -2.45    0/2

The descriptor is ``infer_atomic.bar_music_descriptors`` -- imported, not
re-implemented, so this measurement and the ``--plan-music-repeat`` mechanism
it judges cannot drift apart.  Channels 0/33/34 (tempo, onset, beat) are
excluded there because they repeat every bar by construction.
"""
import argparse
import collections
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from infer_atomic import bar_grid_bounds, bar_music_descriptors  # noqa: E402


def detect_stride(first, second, tolerance=1e-5):
    """Frames between the starts of two consecutive release windows.

    THE WINDOWS OVERLAP, and assuming otherwise cost a whole conclusion.  This
    release stores 150-frame windows at a stride of 15, so 135 of every 150
    frames are shared with the next window.  Concatenating the windows of one
    clip therefore repeats each frame about ten times -- and a repeated track
    manufactures exactly the statistic this file measures, because bars ten
    apart are then literally the same music AND the same class.  Measured that
    way ground truth read +0.2595 against a +0.1174 null, dz +3.17, 10 clips of
    10, which was duplication and not choreography.  It also inflated the clip
    from 619 frames to 4800 while its own audio is 619, and that discrepancy is
    what exposed it.

    Returns the stride, or ``None`` when no shift lines the two windows up --
    in which case the caller must not stitch them.
    """
    length = len(first)
    for stride in range(1, length):
        if np.abs(first[stride:] - second[:length - stride]).max() < tolerance:
            return stride
    return None


def stitch(windows):
    """The clip's own contiguous track, each frame appearing exactly once."""
    if len(windows) == 1:
        return windows[0]
    stride = detect_stride(windows[0], windows[1])
    if stride is None:
        raise ValueError(
            "consecutive windows of this clip do not overlap at any shift, so "
            "they cannot be stitched into one track; the release layout is not "
            "what this tool assumes")
    track = [windows[0]]
    for window in windows[1:]:
        track.append(window[-stride:])
    return np.concatenate(track)


def bar_majority(labels, bounds):
    """One class per bar: the class holding the most FRAMES in it.

    Frames and not segments, because a bar that flickers through three classes
    should be named by the one it was actually in.
    """
    return np.asarray([
        np.bincount(np.asarray(labels[start:end], dtype=np.int64)).argmax()
        for start, end in zip(bounds[:-1], bounds[1:])
    ])


def contrast(similarity, classes, min_gap):
    """``None`` when either side of the split is too small to be a mean."""
    i, j = np.triu_indices(len(classes), k=min_gap)
    if not len(i):
        return None
    same = classes[i] == classes[j]
    if same.sum() < 3 or (~same).sum() < 3:
        return None
    values = similarity[i, j]
    return float(values[same].mean() - values[~same].mean())


def probe(descriptors, classes, min_gap=2, permutations=2000, seed=0):
    similarity = descriptors @ descriptors.T
    observed = contrast(similarity, classes, min_gap)
    if observed is None:
        return None
    n = len(classes)
    if n <= 2 * min_gap + 1:
        return None
    rng = np.random.default_rng(seed)
    null = [
        c for c in (
            contrast(similarity, np.roll(classes, int(k)), min_gap)
            for k in rng.integers(min_gap, n - min_gap, size=permutations)
        ) if c is not None
    ]
    if len(null) < 50:
        return None
    null = np.asarray(null)
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "dz": float((observed - null.mean()) / max(null.std(), 1e-9)),
        "p": float((null >= observed).mean()),
        "bars": int(n),
    }


def pooled(rows):
    if not rows:
        return None
    observed = np.array([r["observed"] for r in rows])
    null = np.array([r["null_mean"] for r in rows])
    return {
        "clips": len(rows),
        "observed": float(observed.mean()),
        "null": float(null.mean()),
        "dz": float(np.mean([r["dz"] for r in rows])),
        "clips_above_own_null": int((observed > null).sum()),
        "median_p": float(np.median([r["p"] for r in rows])),
    }


def _line(name, stats):
    if stats is None:
        return "{:24s}  (no clip carried enough bars to measure)".format(name)
    return ("{:24s} obs {observed:+.4f}  null {null:+.4f}  dz {dz:+.2f}  "
            "{clips_above_own_null}/{clips} clips above own null  "
            "median p {median_p:.3f}").format(name, **stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True,
                    help="release split holding the ground-truth labels, music "
                         "and names (e.g. .../release_aligned_k8/test)")
    ap.add_argument("--clips", required=True,
                    help="one clip name per line; the fixed ten are "
                         "runs/vis_clips_t10.txt")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=DIR",
                    help="a generated arm's directory of <clip>.pkl; repeatable")
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--min-gap", type=int, default=2)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--json", help="write the full per-clip table here")
    args = ap.parse_args()

    release = pathlib.Path(args.release)
    labels = np.load(release / "labels.npy")
    music = np.load(release / "music.npy")
    names = json.load(open(release / "names.json"))
    slices = collections.defaultdict(list)
    for index, name in enumerate(names):
        base, ordinal = name.rsplit("_slice", 1)
        slices[base].append((int(ordinal), index))

    clips = [c.strip() for c in open(args.clips) if c.strip()]
    grids, truth = {}, {}
    for clip in clips:
        if clip not in slices:
            continue
        order = [i for _, i in sorted(slices[clip])]
        clip_music = stitch([music[i] for i in order])
        clip_labels = stitch([labels[i][:, None] for i in order])[:, 0]
        bounds, _ = bar_grid_bounds(clip_music, args.beats_per_bar,
                                    length=len(clip_labels))
        if bounds is None or len(bounds) - 1 < 2 * args.min_gap + 2:
            continue
        grids[clip] = (np.asarray(bounds), bar_music_descriptors(clip_music, bounds).numpy())
        truth[clip] = clip_labels

    report = {"clips": len(grids), "rows": {}}
    if not grids:
        raise SystemExit("no clip in --clips carried a bar grid in this release")

    def run(name, tracks):
        rows = []
        for clip, (descriptors, classes) in tracks.items():
            r = probe(descriptors, classes, args.min_gap, args.permutations)
            if r:
                r["clip"] = clip
                rows.append(r)
        stats = pooled(rows)
        report["rows"][name] = {"pooled": stats, "per_clip": rows}
        print(_line(name, stats))

    run("ground truth", {
        c: (d, bar_majority(truth[c], b)) for c, (b, d) in grids.items()})

    for spec in args.arm:
        # rpartition, not partition: an arm name is allowed to contain "=" --
        # "aligned t=1.0=DIR" silently became name "aligned t", directory
        # "1.0=DIR", every pkl missed, and the row printed "no clip carried
        # enough bars" as though the arm had been measured and found wanting.
        name, _, directory = spec.rpartition("=")
        tracks = {}
        for clip, (bounds, descriptors) in grids.items():
            path = pathlib.Path(directory) / (clip + ".pkl")
            if not path.exists():
                continue
            with open(path, "rb") as handle:
                track = np.asarray(pickle.load(handle)["atomic_labels"]).reshape(-1)
            # A generated track is shorter than the recording, so keep only the
            # bars it fully covers -- and cut the SAME bars out of the
            # descriptors, or the arm and the ground-truth row would be read on
            # two different grids and the comparison would be meaningless.
            kept = int(np.searchsorted(bounds, len(track), side="right")) - 1
            if kept - 1 < 2 * args.min_gap + 2:
                continue
            tracks[clip] = (descriptors[:kept - 1],
                            bar_majority(track.astype(np.int64), bounds[:kept]))
        run(name, tracks)

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=2))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
