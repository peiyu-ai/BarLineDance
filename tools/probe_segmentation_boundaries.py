#!/usr/bin/env python3
"""Does a segmentation cut where the 3D motion turns?  With the controls.

M1 is scored here against the modality *the rest of the pipeline consumes*.
The shipped M1 cuts on S3D video features; M2's clustering, the retrieval
library, the draft and the completion model all consume the 151-D motion.  The
two are known to be weakly synchronised (r=0.253, WILD_ATOMIC_PIPELINE_PLAN
section 7.6) and nothing had ever measured where that leaves the cuts.

Two statistics, neither of which means anything alone:

* **within-segment variance** of the rotation block.  Any segmentation into
  shorter pieces lowers this by being shorter, so it is reported *relative to a
  control that keeps each recording's own segment lengths and only moves the
  cuts*.  Without that control the comparison is with nothing.
* **boundary contrast**: the frame-to-frame change at the cut divided by the
  median change inside the two segments it separates.  A cut placed where the
  motion turns reads above 1.

A third statistic answers the other half of the question -- not "is the cut in
the right place" but "did one movement survive as one segment".  A cut placed
in the middle of a movement leaves two neighbours that are near-duplicates of
each other, so **adjacent segments are compared against non-adjacent ones from
the same clip**: the median distance between consecutive segment means divided
by the median distance between random segment pairs.  Near 1 means each cut
separates material as different as any two segments in that clip; well below 1
means movements are being cut into pieces.

It is confounded with how finely you cut -- cut everything twice as often and
neighbours look more alike -- which is exactly why it is reported beside the
same-lengths-moved control rather than alone.  The control has each arm's own
segment lengths by construction, so the comparison is against a segmentation
that is equally fine and placed anywhere.

The same two statistics run on **either modality** (``--signal motion`` or
``--signal visual``), and running both is the point.  A motion ruler cannot
credit the visual half of a fused segmentation -- on it, a motion-only arm ties
a fused one by construction -- so "should the visual half stay" is a question
this file answers by asking the mirror question in S3D feature space, not a
question to defer to whatever clusters on top.  A fused arm has to read above
its controls **on both rulers**; a single-modality arm should fall back to its
controls on the other one.  The visual ruler measures in cosine geometry
because that is what ``segment_visual_atomics`` compares S3D frames in; the
motion ruler stays on the raw rotation block it was defined on, so every
reading already published against it stays comparable.

And one positive control, because a ruler that cannot see a boundary says
nothing about M1 when it reads 1.0: ``peaks`` cuts at the largest values of the
very change signal being measured, same number of cuts as the arm, at least
``--min-length`` frames apart.  **It is this ruler's upper bound, not a
proposal** -- it is circular by construction, and a segmentation built that way
would cut mid-phrase every time the dancer accelerates.

Refusals rather than skips, because a probe that quietly scores nothing is the
failure mode this repository keeps paying for:

* a segmentation whose sequence names do not meet the bundle's is refused, not
  scored on the empty intersection;
* ``--limit`` samples at an even stride, never a prefix.  Both release row
  order and ``eval_planner_checkpoint --limit`` have already been bitten by
  prefixes standing in for samples.

Usage::

    python3 tools/probe_segmentation_boundaries.py \\
        --bundle /cache/atomicdance-assets/data/wild3d/wild_v4_raw_bundle \\
        --arm shipped=runs/wild_v4_seg/segmentation.json \\
        --arm fused=runs/c1_fused_seg/segmentation.json \\
        --limit 2000 --output runs/c1_judge/boundaries.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

FPS = 30.0


def canonical_key(name: str) -> str:
    """One key for the several spellings of a clip that this corpus uses.

    The S3D feature stem is ``<upload>__clipNNN``, the bundle's recording id is
    ``wild_v4:<upload>:clipNNN``, and AIST uses the sequence name unchanged.
    Comparing the raw strings would silently intersect to nothing.
    """
    key = name.strip()
    for prefix in ("wild_v4:", "wild_v3:", "wild_v2:"):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key.replace("__", ":").replace("/", ":")


def load_feature_index(features_dir: pathlib.Path) -> dict:
    """{canonical key: path} over an ``extract_visual_features`` output dir."""
    if not features_dir.is_dir():
        raise SystemExit("no feature directory at {}".format(features_dir))
    return {canonical_key(p.stem): p for p in features_dir.glob("*.npz")}


def load_bundle_index(bundle: pathlib.Path) -> dict:
    manifest = bundle / "sequences.jsonl"
    if not manifest.exists():
        raise SystemExit("no sequences.jsonl under {}".format(bundle))
    index = {}
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rel = row["assets"]["motion_151_raw"]
            index[canonical_key(row.get("recording_id") or row["sequence_id"])] = bundle / rel
    return index


def load_arm(path: pathlib.Path) -> dict:
    """{canonical key: [boundaries]} from a segment_visual_atomics report."""
    report = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    out = {}
    for record in report["records"]:
        out[canonical_key(record["sequence"])] = [int(b) for b in record["boundaries"]]
    return out


def load_series(path: pathlib.Path, signal: str) -> np.ndarray:
    """The per-frame descriptor the two statistics are computed on."""
    if signal == "motion":
        return np.load(path).astype(np.float32)
    with np.load(path, allow_pickle=False) as bundle:
        features = bundle["features"].astype(np.float32)
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)


def visual_change(features: np.ndarray) -> np.ndarray:
    """1 - cosine between consecutive S3D frames; rows are already unit norm."""
    return 1.0 - (features[:-1] * features[1:]).sum(-1)


def variance_block(series: np.ndarray, signal: str) -> np.ndarray:
    return series[:, 7:] if signal == "motion" else series


def change_signal(motion: np.ndarray) -> np.ndarray:
    """Per-frame-step motion change: summed |delta rot6d| averaged over joints.

    A change in the representation the network regresses, not a geodesic angle.
    It is monotone in rotation change over one frame step, which is all the two
    statistics need, and it is stated as a proxy rather than reported as an
    angle.
    """
    rot = motion[:, 7:].reshape(len(motion), 24, 6)
    return np.abs(np.diff(rot, axis=0)).sum(-1).mean(-1)


def neighbour_ratio(block: np.ndarray, spans, rng) -> float:
    """median distance between adjacent segment means / between random pairs.

    Both halves are taken inside one clip, so the clip's own scale cancels and
    a corpus median over clips is meaningful.
    """
    means = []
    for start, end in spans:
        end = min(int(end), len(block))
        if end - int(start) >= 5:
            means.append(block[int(start):end].mean(0))
    if len(means) < 4:
        return float("nan")
    means = np.stack(means)
    adjacent = np.linalg.norm(np.diff(means, axis=0), axis=1)
    pairs = rng.integers(0, len(means), (min(200, len(means) * 4), 2))
    pairs = pairs[np.abs(pairs[:, 0] - pairs[:, 1]) > 1]
    if len(pairs) < 5:
        return float("nan")
    random_pairs = np.linalg.norm(means[pairs[:, 0]] - means[pairs[:, 1]], axis=1)
    denominator = float(np.median(random_pairs))
    if denominator <= 0:
        return float("nan")
    return float(np.median(adjacent) / denominator)


def score(block: np.ndarray, change: np.ndarray, spans) -> tuple:
    within, contrast = [], []
    for start, end in spans:
        end = min(int(end), len(change))
        start = int(start)
        if end - start < 5:
            continue
        within.append(float(block[start:end].var(0).sum()))
        inner = float(np.median(change[start:end]))
        if inner > 0 and start > 1:
            contrast.append(float(change[start - 1] / inner))
    return within, contrast


def spans_of(boundaries):
    return list(zip(boundaries[:-1], boundaries[1:]))


def shuffled_spans(spans, total, rng):
    """Same lengths, different places -- the only honest denominator.

    Two randomisations, and the second is not decoration: shuffling the lengths
    alone is a **no-op whenever the lengths are equal**, because laying them
    back down from the same start reproduces the original cuts exactly.  A
    segmentation with a fixed segment length would then be compared against
    itself and score 1.000 -- an identity dressed as a control.  So the cut
    positions are also rotated within the covered span, which preserves every
    length (one segment is split at the wrap) and destroys the alignment with
    the motion.  ``tests/test_probe_segmentation_boundaries.py`` builds the
    equal-length case and fails on the no-op version.
    """
    low, high = int(spans[0][0]), min(int(spans[-1][1]), total)
    span = high - low
    if span <= 0:
        return []
    lengths = [int(b - a) for a, b in spans]
    rng.shuffle(lengths)
    offsets = np.cumsum(lengths)[:-1]
    delta = int(rng.integers(0, span))
    cuts = sorted({int((offset + delta) % span) for offset in offsets} - {0})
    edges = [0] + cuts + [span]
    return [(low + a, low + b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def peak_spans(change, spans, min_length):
    """The ruler's upper bound: cut at the largest changes.  Not a proposal."""
    low, high = int(spans[0][0]), min(int(spans[-1][1]), len(change))
    if high - low < 3 * min_length:
        return []
    wanted = len(spans) - 1
    cuts = []
    for candidate in np.argsort(change[low:high])[::-1] + low:
        if len(cuts) >= wanted:
            break
        # +1 because a span starting at f is scored on change[f-1] -- the step
        # *into* it.  Cutting at the peak index itself puts the peak one frame
        # before the boundary, which made this control under-report itself
        # (caught by test_the_positive_control_is_computed_and_beats_random on
        # a fixture whose turns are known exactly).
        if all(abs(int(candidate) + 1 - c) >= min_length for c in cuts):
            cuts.append(int(candidate) + 1)
    edges = [low] + sorted(cuts) + [high]
    return list(zip(edges[:-1], edges[1:]))


def paired_sign_test(a, b, rng, rounds=20000):
    """Per-clip paired sign test: how often does arm b beat arm a on this clip?

    Pooling every segment from every clip and taking one median hides the fact
    that the clips are the sampling unit, not the segments -- a handful of long
    clips can carry a pooled median.  The repository has already been bitten by
    a difference that looked consistent across two seeds and came back p = 0.206
    under a per-clip paired test.
    """
    diff = np.array([y - x for x, y in zip(a, b) if x == x and y == y])
    if len(diff) < 10:
        return None
    wins = int((diff > 0).sum())
    n = int((diff != 0).sum())
    # exact-ish two-sided sign test by simulation, so no scipy dependency
    draws = rng.binomial(n, 0.5, rounds)
    p = float((np.abs(draws - n / 2) >= abs(wins - n / 2)).mean())
    return {"clips": n, "wins": wins, "median_delta": float(np.median(diff)),
            "p_two_sided": p}


def probe(bundle, arms, limit, seed, min_length, signal="motion", features_dir=None,
          clips=None):
    index = (load_bundle_index(bundle) if signal == "motion"
             else load_feature_index(features_dir))
    rng = np.random.default_rng(seed)
    results = {}
    shared = None
    for name, arm in arms.items():
        keys = sorted(set(arm) & set(index))
        if not keys:
            raise SystemExit(
                "arm {!r}: none of its {} sequence names meet the bundle's {} "
                "(example arm key {!r}, example bundle key {!r})".format(
                    name, len(arm), len(index),
                    next(iter(sorted(arm)), None), next(iter(sorted(index)), None)))
        shared = keys if shared is None else [k for k in shared if k in keys]
    if not shared:
        raise SystemExit("the arms share no sequence; they cannot be compared")
    if clips is not None:
        # Restrict to a named corpus BEFORE sampling.  Without this an arm that
        # covers the whole 13,783-clip wild_v4 corpus and one that covers only
        # the 2,103-clip clean5 list are scored on different footage, and the
        # difference reads as a difference between the arms.  Refuse rather
        # than score the empty intersection, same as the arm/bundle check.
        wanted = {canonical_key(line.strip()) for line
                  in pathlib.Path(clips).read_text(encoding="utf-8").splitlines()
                  if line.strip()}
        shared = [k for k in shared if k in wanted]
        if not shared:
            raise SystemExit(
                "--clips {} shares no sequence with the arms".format(clips))
    if limit and limit < len(shared):
        stride = len(shared) / limit          # even stride, never a prefix
        shared = [shared[int(i * stride)] for i in range(limit)]

    per_clip = {name: [] for name in arms}
    blank = lambda: {"within": [], "contrast": [], "lengths": [], "neighbour": []}
    per_arm = {name: blank() for name in arms}
    per_arm["_random"] = blank()
    per_arm["_peaks"] = blank()
    control_arm = next(iter(arms))
    for key in shared:
        series = load_series(index[key], signal)
        change = change_signal(series) if signal == "motion" else visual_change(series)
        block = variance_block(series, signal)
        for name, arm in arms.items():
            spans = spans_of(arm[key])
            if len(spans) < 2:
                # Still record a hole: the paired test zips these lists by
                # position, so an arm that silently skipped a clip would shift
                # every later clip against a different clip's reading.
                per_clip[name].append(float("nan"))
                continue
            w, c = score(block, change, spans)
            per_arm[name]["within"] += w
            per_arm[name]["contrast"] += c
            per_arm[name]["lengths"] += [int(b - a) for a, b in spans]
            per_clip[name].append(float(np.median(c)) if c else float("nan"))
            per_arm[name]["neighbour"].append(neighbour_ratio(block, spans, rng))
            if name == control_arm:
                control = shuffled_spans(spans, len(series), rng)
                w, c = score(block, change, control)
                per_arm["_random"]["within"] += w
                per_arm["_random"]["contrast"] += c
                per_arm["_random"]["neighbour"].append(neighbour_ratio(block, control, rng))
                peaks = peak_spans(change, spans, min_length)
                if peaks:
                    w, c = score(block, change, peaks)
                    per_arm["_peaks"]["within"] += w
                    per_arm["_peaks"]["contrast"] += c
                    per_arm["_peaks"]["neighbour"].append(neighbour_ratio(block, peaks, rng))

    base = float(np.median(per_arm["_random"]["within"]))
    out = {"recordings": len(shared), "control_built_from": control_arm,
           "clips": str(clips) if clips else None,
           "signal": signal, "min_length_frames": min_length, "seed": seed, "arms": {}}
    for name, data in per_arm.items():
        if not data["within"]:
            continue
        row = {
            "segments": len(data["within"]),
            "within_segment_variance": float(np.median(data["within"])),
            "within_vs_random": float(np.median(data["within"]) / base),
            "boundary_contrast": float(np.median(data["contrast"])) if data["contrast"] else None,
        }
        finite = [v for v in data["neighbour"] if v == v]
        row["adjacent_over_random"] = float(np.median(finite)) if finite else None
        if data["lengths"]:
            lens = np.array(data["lengths"])
            row["segment_seconds"] = {
                "median": float(np.median(lens) / FPS),
                "p10": float(np.percentile(lens, 10) / FPS),
                "p90": float(np.percentile(lens, 90) / FPS),
            }
            row["inside_1_to_2p5_s"] = float(((lens >= FPS) & (lens <= 2.5 * FPS)).mean())
        out["arms"][name] = row

    reference = control_arm
    out["paired_vs_" + reference] = {}
    for name in arms:
        if name == reference:
            continue
        result = paired_sign_test(per_clip[reference], per_clip[name],
                                  np.random.default_rng(seed))
        if result:
            out["paired_vs_" + reference][name] = result
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal", choices=("motion", "visual"), default="motion",
                        help="which modality's ruler to run; see the module docstring "
                             "for why a fused arm has to pass both")
    parser.add_argument("--features-dir", type=pathlib.Path,
                        help="S3D features, required by --signal visual")
    parser.add_argument("--bundle", type=pathlib.Path,
                        help="raw bundle root: sequences.jsonl + sequences/<sha>/motion_151_raw.npy")
    parser.add_argument("--arm", action="append", default=[], required=True,
                        help="name=path/to/segmentation.json; repeat for each arm")
    parser.add_argument("--clips", type=pathlib.Path,
                        help="restrict scoring to this clip list (one name per line) "
                             "before --limit samples; without it every arm is scored "
                             "on whatever it happens to cover")
    parser.add_argument("--limit", type=int, default=2000,
                        help="recordings to score, sampled at an even stride (not a prefix)")
    parser.add_argument("--min-length", type=int, default=18,
                        help="minimum frames between the positive control's cuts")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            raise SystemExit("--arm wants name=path, got {!r}".format(spec))
        name, path = spec.split("=", 1)
        arms[name] = load_arm(pathlib.Path(path))
    if args.signal == "motion" and args.bundle is None:
        raise SystemExit("--signal motion needs --bundle")
    if args.signal == "visual" and args.features_dir is None:
        raise SystemExit("--signal visual needs --features-dir")
    report = probe(args.bundle, arms, args.limit, args.seed, args.min_length,
                   signal=args.signal, features_dir=args.features_dir,
                   clips=args.clips)

    print("recordings {} | {} ruler | random control built from arm {!r}".format(
        report["recordings"], report["signal"], report["control_built_from"]))
    print("%-16s %9s %12s %10s %10s %10s %9s" % (
        "arm", "segments", "within-var", "vs random", "contrast", "adj/rand", "median s"))
    for name, row in report["arms"].items():
        print("%-16s %9d %12.4f %11.1f%% %10s %10s %9s" % (
            name, row["segments"], row["within_segment_variance"],
            100 * row["within_vs_random"],
            "%.3f" % row["boundary_contrast"] if row["boundary_contrast"] else "--",
            "%.3f" % row["adjacent_over_random"] if row["adjacent_over_random"] else "--",
            "%.2f" % row["segment_seconds"]["median"] if "segment_seconds" in row else "--"))
    paired = report.get("paired_vs_" + report["control_built_from"], {})
    if paired:
        print("\nper-clip paired sign test on boundary contrast, vs {!r}:".format(
            report["control_built_from"]))
        for name, row in paired.items():
            print("  %-14s %d/%d clips better, median delta %+.3f, p = %.4f" % (
                name, row["wins"], row["clips"], row["median_delta"], row["p_two_sided"]))
    print("\n_random is the same lengths moved; _peaks is this ruler's upper bound, "
          "not a proposal (it cuts at the peaks of the signal being measured).")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("report -> {}".format(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
