#!/usr/bin/env python3
"""Judge a discovered atomic vocabulary against what the paper reports.

Running all three discovery stages produces a label bundle.  Whether that
bundle is any *good* is a separate question, and "the tool exited 0" does not
answer it.  This audit answers it in two registers.

**Shape** -- does the vocabulary look like the paper's?  Fig. 4a gives the
segment duration histogram, Fig. 4b the samples per prototype, Fig. 4c the
samples per sub-prototype, and the text gives 7.3 sub-prototypes per prototype
at 31.8 samples each.  Those are reported side by side with ours, with the
paper's counts rescaled to our corpus size, because the wild corpus is not
AIST++ and an absolute count comparison would be theatre.

**Coherence** -- did re-clustering actually find something?  A vocabulary can
have perfect shape statistics and be noise: split each prototype into seven
arbitrary pieces and the histogram still matches.  So the audit measures
whether members of a sub-prototype are closer to each other, in the keyframe
pose-and-dynamics space, than to the other members of their own prototype.
The comparison is *within the prototype* on purpose -- against the whole corpus
any grouping looks coherent, because prototypes were already separated by M2,
and that would credit re-clustering with M2's work.  A permutation test says
whether the gap could be chance.

With ``--captions`` it also reports semantic purity: how much more members of a
sub-prototype agree on the caption schema than members of the same prototype in
different sub-prototypes.  For the ``w/ LLM`` row this is close to a tautology
-- the LLM grouped on those captions -- so it is reported as a check that the
grouping was *applied*, not as evidence that it was right.

Every criterion carries an explicit threshold and a pass/fail, so the verdict
is one field rather than a paragraph to interpret.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.cluster_atomics_tmr import build_row_index, resolve_row  # noqa: E402
from tools.convert_motion_to_guofeats import motion_151_to_joints  # noqa: E402
from tools.motion_beats import descriptor_vector  # noqa: E402
from tools.recluster_atomics_ingroup import segments_of  # noqa: E402

FPS = 30.0

# Fig. 4a/4b/4c, read off the published bars.  Kept as counts rather than
# fractions so the source stays checkable against the figure.
PAPER_DURATION = {"<0.7": 1366, "0.7-0.9": 7712, "0.9-1.1": 8054,
                  "1.1-1.3": 2095, ">1.3": 3967}
PAPER_PROTOTYPE = {"<200": 10, "200-250": 31, "250-300": 32,
                   "300-350": 18, ">350": 9}
PAPER_SUBPROTOTYPE = {"<20": 109, "20-35": 431, "35-50": 99, ">50": 91}
PAPER_SUBS_PER_PROTOTYPE = 7.3
PAPER_SAMPLES_PER_SUB = 31.8


class AuditError(RuntimeError):
    pass


def bucket_durations(seconds: Sequence[float]) -> Dict[str, int]:
    edges = [(0.0, 0.7, "<0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.1, "0.9-1.1"),
             (1.1, 1.3, "1.1-1.3"), (1.3, np.inf, ">1.3")]
    counts = {name: 0 for _, _, name in edges}
    for value in seconds:
        for low, high, name in edges:
            if low <= value < high:
                counts[name] += 1
                break
    return counts


def bucket_sizes(sizes: Sequence[int], edges: Sequence[Tuple[float, float, str]]) -> Dict[str, int]:
    counts = {name: 0 for _, _, name in edges}
    for value in sizes:
        for low, high, name in edges:
            if low <= value < high:
                counts[name] += 1
                break
    return counts


def rescale(reference: Dict[str, int], total: int) -> Dict[str, float]:
    """The paper's histogram scaled to our population size."""
    denominator = sum(reference.values()) or 1
    return {name: round(count * total / denominator, 1) for name, count in reference.items()}


def decode_ids(producer: pathlib.Path) -> Dict[int, Tuple[int, int]]:
    """compact label id -> (prototype, sub-prototype).

    The published bundle carries compact ids, because a padded id space would
    hand a third of the D3PM vocabulary to tokens that never occur.  The
    original layout is kept in ``producer.npz`` precisely so an audit can get
    the two components back.
    """
    data = np.load(producer)
    width = int(data["width"][0])
    used = data["used_raw_ids"]
    return {index + 1: (int((raw - 1) // width) + 1, int((raw - 1) % width))
            for index, raw in enumerate(used)}


def tmr_embeddings(labels_dir: pathlib.Path, bundle: pathlib.Path, *, device: str,
                   batch_size: int = 64) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
    """Embed every accepted segment in TMR's motion-text space.

    The keyframe descriptor is not a neutral place to judge coherence: the
    ``w/o LLM`` variant runs k-means *in it*, so it passes that gate by
    construction while a caption-driven grouping is judged on how well it
    happens to align with someone else's objective.  TMR's space is the one
    neither variant optimises -- M2 used it for the 100-way split, and neither
    M3 variant touches it again -- so it can compare the two fairly.
    """
    from tools.convert_motion_to_guofeats import motion_151_to_guofeats
    from tools.tmr_runtime import TMREncoder

    rows = {json.loads(line)["recording_id"]: json.loads(line)
            for line in (bundle / "sequences.jsonl").open(encoding="utf-8")}
    index = build_row_index(rows)
    encoder = TMREncoder(device=device, text=False)
    owners: List[Tuple[str, int, int]] = []
    windows: List[np.ndarray] = []
    for line in (labels_dir / "labels.jsonl").open(encoding="utf-8"):
        entry = json.loads(line)
        row = resolve_row(index, entry["recording_id"]) or rows.get(entry["recording_id"])
        if row is None:
            continue
        labels = np.load(labels_dir / entry["labels_path"])
        payload = motion_151_to_guofeats(np.load(bundle / row["motion_path"]))
        guofeats, source = payload["features"], payload["source_frame_index"]
        lookup = {int(value): position for position, value in enumerate(source)}
        for start, end, label in segments_of(labels):
            if label <= 0 or end - start < 4:
                continue
            # Guo features are resampled to 20 fps, so a motion-frame span has
            # to be mapped through the index the converter returns rather than
            # sliced directly -- slicing raw would take the wrong moments.
            lo = min((lookup[k] for k in range(start, end) if k in lookup), default=None)
            hi = max((lookup[k] for k in range(start, end) if k in lookup), default=None)
            if lo is None or hi is None or hi - lo < 2:
                continue
            owners.append((entry["recording_id"], start, end))
            windows.append(guofeats[lo:hi + 1])
    if not windows:
        raise AuditError("no segment could be converted for TMR encoding")
    return np.asarray(encoder.encode_motion(windows, batch_size=batch_size),
                      dtype=np.float64), owners


def gather(labels_dir: pathlib.Path, bundle: pathlib.Path) -> Dict[str, object]:
    """Every accepted segment with its label, duration and keyframe descriptor."""
    rows = {json.loads(line)["recording_id"]: json.loads(line)
            for line in (bundle / "sequences.jsonl").open(encoding="utf-8")}
    index = build_row_index(rows)
    owners: List[Tuple[str, int, int]] = []
    ids: List[int] = []
    durations: List[float] = []
    descriptors: List[np.ndarray] = []
    for line in (labels_dir / "labels.jsonl").open(encoding="utf-8"):
        entry = json.loads(line)
        row = resolve_row(index, entry["recording_id"]) or rows.get(entry["recording_id"])
        if row is None:
            continue
        labels = np.load(labels_dir / entry["labels_path"])
        joints = motion_151_to_joints(np.load(bundle / row["motion_path"]))
        for start, end, label in segments_of(labels):
            if label <= 0 or end - start < 4:
                continue
            owners.append((entry["recording_id"], start, end))
            ids.append(int(label))
            durations.append((end - start) / FPS)
            descriptors.append(descriptor_vector(joints[start:end]))
    if not owners:
        raise AuditError("no accepted segments in {}".format(labels_dir))
    return {"owners": owners, "ids": np.asarray(ids), "durations": np.asarray(durations),
            "features": np.stack(descriptors).astype(np.float64)}


def coherence(features: np.ndarray, ids: np.ndarray, prototypes: np.ndarray, *,
              pairs: int, rounds: int, rng: np.random.Generator) -> Dict[str, object]:
    """Are sub-prototype members closer to each other than to their prototype?

    Distances are standardised first so the six dynamics numbers cannot be
    swamped by hundreds of pose coordinates.  Both samples are drawn *inside a
    prototype*: the alternative -- comparing against the whole corpus -- would
    measure M2's clustering and attribute it to M3.
    """
    scaled = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    by_prototype: Dict[int, List[int]] = collections.defaultdict(list)
    for position, prototype in enumerate(prototypes):
        by_prototype[int(prototype)].append(position)

    within: List[float] = []
    across: List[float] = []
    usable = [group for group in by_prototype.values()
              if len(set(ids[group].tolist())) >= 2]
    if not usable:
        return {"skipped": "no prototype holds two sub-prototypes"}
    for _ in range(pairs):
        group = usable[rng.integers(len(usable))]
        members = np.asarray(group)
        labels = ids[members]
        choice = int(labels[rng.integers(len(labels))])
        same = members[labels == choice]
        other = members[labels != choice]
        if len(same) < 2 or len(other) < 1:
            continue
        i, j = rng.choice(len(same), size=2, replace=False)
        within.append(float(np.linalg.norm(scaled[same[i]] - scaled[same[j]])))
        k = int(rng.integers(len(other)))
        across.append(float(np.linalg.norm(scaled[same[i]] - scaled[other[k]])))
    if len(within) < 10:
        return {"skipped": "too few comparable pairs"}

    within_array, across_array = np.asarray(within), np.asarray(across)
    observed = across_array.mean() - within_array.mean()
    pooled = np.concatenate([within_array, across_array])
    hits = 0
    for _ in range(rounds):
        rng.shuffle(pooled)
        if pooled[len(within_array):].mean() - pooled[:len(within_array)].mean() >= observed:
            hits += 1
    return {
        "within_subprototype_distance": round(float(within_array.mean()), 4),
        "across_subprototype_distance": round(float(across_array.mean()), 4),
        "separation": round(float(observed), 4),
        "ratio": round(float(across_array.mean() / max(1e-9, within_array.mean())), 4),
        "permutation_p": (hits + 1) / (rounds + 1),
        "pairs": len(within_array),
        "reference": "pairs drawn inside a prototype, so this is M3's own contribution",
    }


def caption_purity(captions: Dict[Tuple[str, int, int], Dict[str, str]],
                   owners: Sequence[Tuple[str, int, int]], ids: np.ndarray,
                   prototypes: np.ndarray, *, pairs: int,
                   rng: np.random.Generator) -> Dict[str, object]:
    """Caption-field agreement inside a sub-prototype vs elsewhere in its prototype."""
    from tools.caption_segments_vlm import FIELDS

    fields = [captions.get(owner) for owner in owners]
    by_prototype: Dict[int, List[int]] = collections.defaultdict(list)
    for position, prototype in enumerate(prototypes):
        if fields[position]:
            by_prototype[int(prototype)].append(position)
    usable = [group for group in by_prototype.values()
              if len(set(ids[group].tolist())) >= 2]
    if not usable:
        return {"skipped": "no captioned prototype holds two sub-prototypes"}

    def agree(a, b):
        return float(np.mean([a.get(f) == b.get(f) for f in FIELDS]))

    within, across = [], []
    for _ in range(pairs):
        members = np.asarray(usable[rng.integers(len(usable))])
        labels = ids[members]
        choice = int(labels[rng.integers(len(labels))])
        same, other = members[labels == choice], members[labels != choice]
        if len(same) < 2 or len(other) < 1:
            continue
        i, j = rng.choice(len(same), size=2, replace=False)
        within.append(agree(fields[same[i]], fields[same[j]]))
        across.append(agree(fields[same[i]], fields[other[int(rng.integers(len(other)))]]))
    if len(within) < 10:
        return {"skipped": "too few captioned pairs"}
    return {
        "within_subprototype_agreement": round(float(np.mean(within)), 4),
        "across_subprototype_agreement": round(float(np.mean(across)), 4),
        "lift": round(float(np.mean(within) / max(1e-9, np.mean(across))), 4),
        "pairs": len(within),
        "note": "on the w/ LLM row the grouping was made from these captions, so this "
                "checks that the grouping was applied, not that it was correct",
    }


def audit(*, labels_dir: pathlib.Path, bundle: pathlib.Path,
          captions_path: Optional[pathlib.Path] = None, pairs: int = 4000,
          rounds: int = 2000, seed: int = 20260810,
          tmr_device: Optional[str] = None) -> Dict[str, object]:
    collected = gather(labels_dir, bundle)
    ids = collected["ids"]
    mapping = decode_ids(labels_dir / "producer.npz")
    missing = sorted(set(ids.tolist()) - set(mapping))
    if missing:
        raise AuditError("labels carry ids {} that producer.npz does not decode".format(
            missing[:5]))
    prototypes = np.asarray([mapping[int(value)][0] for value in ids])

    sizes = collections.Counter(ids.tolist())
    subs_per_prototype = collections.Counter(
        prototype for prototype, _ in (mapping[value] for value in sorted(set(ids.tolist()))))
    prototype_sizes = collections.Counter(prototypes.tolist())

    rng = np.random.default_rng(seed)
    report: Dict[str, object] = {
        "labels": str(labels_dir.resolve()),
        "segments": int(len(ids)),
        "prototypes": len(prototype_sizes),
        "subprototypes": len(sizes),
        "mean_subprototypes_per_prototype": round(
            float(np.mean(list(subs_per_prototype.values()))), 2),
        "mean_samples_per_subprototype": round(float(np.mean(list(sizes.values()))), 2),
        "median_samples_per_subprototype": float(np.median(list(sizes.values()))),
        "paper": {"subprototypes_per_prototype": PAPER_SUBS_PER_PROTOTYPE,
                  "samples_per_subprototype": PAPER_SAMPLES_PER_SUB,
                  "subprototypes_total": 730},
        "duration_seconds": {
            "median": round(float(np.median(collected["durations"])), 3),
            "mean": round(float(np.mean(collected["durations"])), 3),
            "ours": bucket_durations(collected["durations"]),
            "paper_rescaled": rescale(PAPER_DURATION, len(ids)),
            "note": "these are the *labelled runs* read back from frame labels, so "
                    "adjacent segments sharing a label have merged; compare Fig. 4a "
                    "against the segmenter's own output, not against this",
        },
        "samples_per_prototype": {
            "ours": bucket_sizes(list(prototype_sizes.values()),
                                 [(0, 200, "<200"), (200, 250, "200-250"),
                                  (250, 300, "250-300"), (300, 350, "300-350"),
                                  (350, np.inf, ">350")]),
            "paper_rescaled": rescale(PAPER_PROTOTYPE, len(prototype_sizes)),
            "note": "the paper's buckets are absolute counts on AIST++; ours are a "
                    "different corpus size, so read the shape, not the bars",
        },
        "samples_per_subprototype": {
            "ours": bucket_sizes(list(sizes.values()),
                                 [(0, 20, "<20"), (20, 35, "20-35"),
                                  (35, 50, "35-50"), (50, np.inf, ">50")]),
            "paper_rescaled": rescale(PAPER_SUBPROTOTYPE, len(sizes)),
        },
        "coherence_keyframe": {
            **coherence(collected["features"], ids, prototypes,
                        pairs=pairs, rounds=rounds, rng=rng),
            "caveat": "the w/o-LLM variant runs k-means in this very space, so it "
                      "passes here by construction; read coherence_tmr instead",
        },
    }

    if tmr_device is not None:
        vectors, owners = tmr_embeddings(labels_dir, bundle, device=tmr_device)
        position = {owner: index for index, owner in enumerate(collected["owners"])}
        keep = np.asarray([position[owner] for owner in owners if owner in position])
        rows = np.asarray([index for index, owner in enumerate(owners)
                           if owner in position])
        report["coherence_tmr"] = {
            **coherence(vectors[rows], ids[keep], prototypes[keep],
                        pairs=pairs, rounds=rounds, rng=rng),
            "space": "TMR motion encoder (mu), the space neither M3 variant optimises",
            "segments": int(len(rows)),
        }

    if captions_path is not None:
        rows = {}
        with captions_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.setdefault((row["recording_id"], int(row["start"]), int(row["end"])),
                                dict(row.get("fields") or {}))
        report["caption_purity"] = caption_purity(
            rows, collected["owners"], ids, prototypes, pairs=pairs, rng=rng)

    report["verdict"] = verdict(report)
    return report


def verdict(report: Dict[str, object]) -> Dict[str, object]:
    """One pass/fail per criterion, with the threshold stated next to it.

    The shape bands are wide on purpose: the wild corpus is a fifth of AIST++'s
    length and has no genre pre-split, so demanding 7.3 exactly would be
    demanding a coincidence.  The coherence gate is the one that cannot be
    passed by construction, and it is the one that decides.
    """
    checks = []
    subs = float(report["mean_subprototypes_per_prototype"])
    checks.append({"criterion": "sub-prototypes per prototype within 4-11 (paper 7.3)",
                   "value": subs, "pass": 4.0 <= subs <= 11.0})
    samples = float(report["mean_samples_per_subprototype"])
    checks.append({"criterion": "samples per sub-prototype within 20-45 (paper 31.8)",
                   "value": samples, "pass": 20.0 <= samples <= 45.0})
    # TMR space when it was computed, keyframe space otherwise: the gate has to
    # name which space it judged, because they do not agree and one of them
    # flatters the k-means variant.
    coherent = report.get("coherence_tmr") or report.get("coherence_keyframe") or {}
    if "ratio" in coherent:
        checks.append({
            "criterion": "sub-prototype members closer to each other than to the rest "
                         "of their prototype (ratio > 1.02, p < 0.01)",
            "value": coherent["ratio"],
            "pass": bool(coherent["ratio"] > 1.02 and coherent["permutation_p"] < 0.01)})
    else:
        checks.append({"criterion": "coherence measurable", "value": None, "pass": False})
    tiny = report["samples_per_subprototype"]["ours"].get("<20", 0)
    share = tiny / max(1, int(report["subprototypes"]))
    checks.append({
        "criterion": "sub-prototypes with under 20 samples stay under 40% "
                     "(paper: 109/730 = 15%)",
        "value": round(share, 4), "pass": share < 0.40})
    return {"checks": checks, "pass": all(check["pass"] for check in checks)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=pathlib.Path, required=True)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--captions", type=pathlib.Path, default=None)
    parser.add_argument("--pairs", type=int, default=4000)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--tmr-device", default=None,
                        help="device for TMR encoding; enables the neutral-space "
                             "coherence gate (e.g. cuda:0)")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit(labels_dir=args.labels, bundle=args.bundle,
                       captions_path=args.captions, pairs=args.pairs,
                       rounds=args.rounds, seed=args.seed, tmr_device=args.tmr_device)
    except (AuditError, FileNotFoundError) as error:
        raise SystemExit("error: {}".format(error))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["verdict"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
