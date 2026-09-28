#!/usr/bin/env python3
"""One table, many arms, and the columns whose absence chose the wrong arm.

WHY THIS EXISTS BESIDE ``tools/score_generation_diagnostics.py``.  That tool is
a deep single-run diagnostic (follow rate against a shuffle control, jerk at
plan boundaries and blend steps) and it rebuilds the retrieval library to do it.
This one is the cheap wide table used to *choose between arms*, and it exists
because on 2026-08-30 the table used to choose between arms carried three
columns -- segments/s, limb lag-0, arm span -- and on those three the winning
arm was ``--completion-sample-steps 5``, which buys arm span by **cutting the
motion energy from 0.571 to 0.366 m/s against a ground truth of 0.804**.  The
reviewer saw a duller dance and the table said it had improved.  A missing
column is not a neutral omission; it is a criterion that cannot fail.

THE COLUMNS, and what each is allowed to say.

``energy``   mean root-relative joint speed, m/s, and its per-clip ratio to the
             same clip's ground truth.  This is the column that was missing.  It
             is deliberately the plainest possible amplitude measure: FID cannot
             see amplitude at all (``eval/metrics.py`` standardises per
             dimension, so halving every clip leaves ``fid_k`` at 0.0), so
             amplitude needs an instrument that is not a distance.

``span``     mean wrist-to-wrist distance, root-relative.  "One arm held open"
             was the operator's own description of the defect; this is it in
             metres.  Ground truth 0.663 m **on the v5 line**; on the T line's
             20 eval clips it is **0.628 m** (measured 2026-09-12), so the
             shipped T arm's 0.607 is 0.021 m short, not 0.056.

             EVERY GROUND-TRUTH CONSTANT IN THIS FILE IS CORPUS-SPECIFIC and
             two of them have now been read across lines and inverted a
             conclusion: this one, and ``seg/s`` below.  Name the corpus when
             quoting one, and measure it on the line you are working on.

``root``     max horizontal displacement of the root from its own mean, and its
             ratio to ground truth.  D3: generated 0.256 m against 1.206 m, with
             path length 1.29-1.35x -- i.e. jittering in place, not travelling.

``lag0``     share of the six limb pairs whose speed-envelope cross-correlation
             peaks at lag 0.  **Whole clip against whole clip only.**  The
             statistic is strongly length-dependent -- the same ground-truth
             clips read 0.239 at 40-frame chunks, 0.321 at 90, 0.628 whole --
             so a comparison across different lengths measures the lengths.  A
             conclusion was published from exactly that mistake on 2026-08-30
             ("the library prototypes read 0.000, so the vocabulary is
             coordinated and splicing destroys it"); the prototypes are 40
             frames long.

``jitter``   share of the root-relative velocity spectrum between 5 and 15 Hz.
             **Read it beside ``energy`` or not at all.**  Raising the
             completion's guidance weight raises energy and raises this with it:
             measured 2026-08-30, guidance 2.0 -> 3.5 -> 5.0 -> 8.0 gives jitter
             0.087, 0.095, 0.135, 0.176 against a ground truth of 0.096, so
             everything past 3.5 is buying "more motion" with shake.  A
             music-normalized checkpoint read energy 0.966 of ground truth --
             the best number in the whole table -- at jitter 0.169, and that
             number means nothing without this column beside it.

WHICH CLIPS ARE IN THE TABLE.  Clips whose ``prototype_retrieval.
safe_draft_condition_fraction`` is 0 never went through retrieval -- the
fail-closed path in ``infer_atomic._source_safe_draft`` handed the completion an
all-zero draft -- so their row is the completion on music alone, a different
experiment.  They are dropped from every arm AND from ground truth, and the two
header lines above the table name them.  Two of the twenty T-line eval clips are
like this and every table published before 2026-09-04 pooled them in.  Pass
``--include-unsourced`` to reproduce those older numbers.

``twist``    torso twist rate, degrees per second: the unwrapped angle between
             the shoulder line and the hip line, mean absolute first difference.
             This is the operator's "缺乏有韵律的身体扭动" in one number --
             ground truth 49.3, the shipped baseline 29.0.  A body that only
             raises and lowers its arms reads high on energy and flat here.

``wristsync``  Pearson correlation of the two wrists' root-relative heights.
             The operator's "双手同步抬起" directly: ground truth +0.473, and
             the defective arm read +0.656.  Distinct from ``lag0``: lag0 asks
             whether limb SPEED envelopes peak together, this asks whether the
             hands occupy mirrored heights -- a body can fail either alone.

``skate``    mean horizontal foot speed (m/s) over the frames where that foot
             is at ground level (lowest decile of its own height) -- a planted
             foot that translates is the physics violation the operator named:
             "脚步只能是踩踏正常移动遵循物理规律".  Ground truth is the
             calibration (wild reconstructions themselves skate a little); the
             defective direction is exceeding it.

``e_corr``   Pearson correlation across clips between the arm's per-clip energy
             and the ground truth's.  THE beat-hitting number: the completion
             outputs a near-generic energy per song (0.210 against the ground
             truth's self-correlation of 1), so energetic songs get the
             flattest dance and "hits the beat" fails exactly there.  A
             per-clip ratio column cannot see this; only the correlation can.

``seg/s``    atomic segments per second from the clip's own label track.  Ground
             truth 0.917; the shipped bar grid produced 0.469 by construction.
             **THOSE TWO NUMBERS ARE THE v5 LINE AND DO NOT TRANSFER.**  v5 is
             cut by S3D at a 0.967 s median; the T line is cut on a four-beat
             music grid.  Measured 2026-09-12 on the 20 T eval clips from
             ``data/wild3d/txy_t_labels``, ground truth changes label 0.445
             times per second with a median run of 1.90 s, and the T arms read
             0.526 -- ABOVE ground truth, not below.  Read on the T line,
             0.917 says "raise the rate" and the corpus says the opposite.

``R``        resultant of the music-beat phase at motion-beat events, with the
             repository's own gap-preserving shuffle as the null
             (``tools/probe_music_beat_alignment``).  **Printed, never gated at
             this sample size**: at n=100 the ground truth itself wins only
             51/99 (P=0.84) while its validated reading is 224/393 (P=0.0064).
             A column with no power is reported so it cannot be quietly
             re-invented, not so it can decide anything.

THE GATES are ratios to ground truth on the same clips, paired per clip, and
they are chosen to be failable by the arms that exist: the shipped baseline
fails ``root``, the sample-steps arm fails ``energy``, and the previous
generation's artifacts (``runs/vis_now_v4`` on the clean5 corpus) pass
``energy`` at 1.09.  ``--replay`` runs exactly those three as the positive
control for the table itself.
"""
import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools import arm_sourcing
from tools.motion_beats import find_motion_beats
from tools.probe_music_beat_alignment import beat_phase, resultant, shuffled_events

FPS = 30.0
BEAT_CHANNEL = 34
LIMBS = {"LA": [16, 18, 20, 22], "RA": [17, 19, 21, 23],
         "LL": [1, 4, 7, 10], "RL": [2, 5, 8, 11]}
PAIRS = [("LA", "RA"), ("LL", "RL"), ("LA", "RL"), ("RA", "LL"), ("LA", "LL"), ("RA", "RL")]


def root_relative(joints):
    return joints - joints[:, :1, :]


def energy(joints):
    return float(np.linalg.norm(np.diff(root_relative(joints), axis=0), axis=2).mean() * FPS)


def arm_span(joints):
    relative = root_relative(joints)
    return float(np.linalg.norm(relative[:, 20] - relative[:, 21], axis=1).mean())


def root_range(joints):
    ground = joints[:, 0, :2]
    return float(np.linalg.norm(ground - ground.mean(0), axis=1).max())


def jitter_share(joints, low=5.0, high=15.0):
    """Fraction of the velocity spectrum in the shake band.

    Ground truth is the target, not zero: a generation at 0.0 would be a
    different defect (over-smoothed).  THE TARGET IS CORPUS-SPECIFIC, like
    ``span`` and ``seg/s`` above and for the same reason: **0.096 is the v5
    line**; on the T line's 20 eval clips ground truth reads **0.0509**
    (measured 2026-09-13).  Quoting the v5 number on the T line makes an arm at
    0.0996 look like it is on target when it is nearly twice the dancer.  Root-
    relative so a wandering root cannot dominate the spectrum, Hann-windowed and
    mean-removed so the window edges do not leak into the band being read.
    """
    relative = root_relative(joints)
    velocity = np.diff(relative, axis=0).reshape(len(relative) - 1, -1)
    if len(velocity) < 16:
        return float("nan")
    window = np.hanning(len(velocity))[:, None]
    spectrum = np.fft.rfft((velocity - velocity.mean(0)) * window, axis=0)
    frequency = np.fft.rfftfreq(len(velocity), 1.0 / FPS)
    power = (np.abs(spectrum) ** 2).sum(1)
    total = power[1:].sum()
    return float(power[(frequency >= low) & (frequency < high)].sum() / max(total, 1e-30))


def torso_twist_rate(joints):
    """Degrees per second of shoulder-line-vs-hip-line rotation."""
    relative = root_relative(joints)
    shoulders = relative[:, 17] - relative[:, 16]
    hips = relative[:, 2] - relative[:, 1]
    angle = np.unwrap(np.arctan2(shoulders[:, 1], shoulders[:, 0])
                      - np.arctan2(hips[:, 1], hips[:, 0]))
    return float(np.degrees(np.abs(np.diff(angle)).mean() * FPS))


def wrist_height_correlation(joints):
    relative = root_relative(joints)
    left, right = relative[:, 20, 2], relative[:, 21, 2]
    if left.std() < 1e-9 or right.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


FOOT_JOINTS = (7, 8, 10, 11)


def foot_skate(joints):
    """Horizontal foot speed while the foot is at its own ground level."""
    joints = np.asarray(joints, float)
    speeds = []
    for foot in FOOT_JOINTS:
        height = joints[:, foot, 2]
        grounded = height <= np.quantile(height, 0.10) + 0.02
        if grounded[:-1].sum() < 5:
            continue
        horizontal = np.linalg.norm(np.diff(joints[:, foot, :2], axis=0), axis=1) * FPS
        speeds.append(float(horizontal[grounded[:-1]].mean()))
    return float(np.mean(speeds)) if speeds else float("nan")


def _envelopes(joints):
    speeds = np.linalg.norm(np.diff(root_relative(joints), axis=0), axis=2) * FPS
    return {name: speeds[:, index].mean(1) for name, index in LIMBS.items()}


def _peak_lag(a, b, max_lag=15):
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    best, best_lag, n = -2.0, 0, len(a)
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b[:n - lag]) if lag >= 0 else (a[:n + lag], b[-lag:])
        if len(x) < 20:
            continue
        value = float((x * y).mean())
        if value > best:
            best, best_lag = value, lag
    return best_lag


def lag_zero_share(joints):
    """See the header: whole clip against whole clip, never across lengths."""
    envelope = _envelopes(joints)
    return float(np.mean([_peak_lag(envelope[a], envelope[b]) == 0 for a, b in PAIRS]))


def segments_per_second(labels, frames):
    labels = np.asarray(labels)
    runs = 1 + int(np.count_nonzero(np.diff(labels)))
    return runs / (frames / FPS)


def beat_resultant(joints, music, rng, draws=40):
    frames = len(joints)
    beats = np.flatnonzero(np.asarray(music)[:frames, BEAT_CHANNEL] > 0.5)
    if len(beats) < 6:
        return None
    # Default ``max_beats``.  Copying ``max_beats=10**6`` from the segmentation
    # probes flattens R for every row alike -- ground truth read 0.1128 against
    # a null of 0.1104, i.e. the positive control failed and the column could
    # judge nothing.  More events means more phases means a lower resultant by
    # construction.
    events = np.sort(np.asarray(find_motion_beats(joints), int))
    events = events[(events >= 0) & (events < frames)]
    if len(events) < 3:
        return None
    value = resultant(beat_phase(events, beats))
    null = np.nanmean([resultant(beat_phase(shuffled_events(events, 0, frames, rng), beats))
                       for _ in range(draws)])
    return value, null


def load(path):
    payload = pickle.load(open(path, "rb"))
    return (np.asarray(payload["full_pose"], float),
            np.asarray(payload["atomic_labels"]) if "atomic_labels" in payload else None)


def score(arm_dir, clips, ground_truth, audio_dir, labels_root=None, seed=20260830):
    rng = np.random.default_rng(seed)
    rows = {"energy": [], "span": [], "root": [], "lag0": [], "seg": [], "jitter": [], "twist": [], "wristsync": [], "skate": [],
            "energy_ratio": [], "root_ratio": [], "R": [], "R_null": []}
    for clip in clips:
        path = arm_dir / (clip + ".pkl")
        if not path.is_file():
            continue
        joints, labels = load(path)
        rows["energy"].append(energy(joints))
        rows["span"].append(arm_span(joints))
        rows["root"].append(root_range(joints))
        rows["lag0"].append(lag_zero_share(joints))
        rows["jitter"].append(jitter_share(joints))
        rows["twist"].append(torso_twist_rate(joints))
        rows["skate"].append(foot_skate(joints))
        rows["wristsync"].append(wrist_height_correlation(joints))
        if labels is None and labels_root is not None:
            labels = labels_root.get(clip)
        if labels is not None:
            rows["seg"].append(segments_per_second(np.asarray(labels)[:len(joints)], len(joints)))
        reference = ground_truth.get(clip)
        if reference is not None:
            rows.setdefault("gt_energy", []).append(energy(reference))
            rows.setdefault("own_energy", []).append(rows["energy"][-1])
            # (energy correlation is computed in summarise from the raw per-clip pairs)
    # Paired per clip, not a ratio of medians: clips differ enormously
            # in how much their own dancer moves.
            rows["energy_ratio"].append(rows["energy"][-1] / max(energy(reference), 1e-9))
            rows["root_ratio"].append(rows["root"][-1] / max(root_range(reference), 1e-9))
        music_path = audio_dir / (clip + ".npy") if audio_dir else None
        if music_path is not None and music_path.is_file():
            got = beat_resultant(joints, np.load(music_path), rng)
            if got is not None:
                rows["R"].append(got[0])
                rows["R_null"].append(got[1])
    return rows


def summarise(name, rows):
    resultants = np.array(rows["R"], float)
    nulls = np.array(rows["R_null"], float)
    valid = np.isfinite(resultants) & np.isfinite(nulls)
    wins = int((resultants[valid] > nulls[valid]).sum())
    total = int(valid.sum())
    return {
        "arm": name,
        "clips": len(rows["energy"]),
        "energy_ms": float(np.mean(rows["energy"])) if rows["energy"] else float("nan"),
        "energy_vs_gt": float(np.median(rows["energy_ratio"])) if rows["energy_ratio"] else float("nan"),
        "span_m": float(np.mean(rows["span"])) if rows["span"] else float("nan"),
        "root_m": float(np.median(rows["root"])) if rows["root"] else float("nan"),
        "root_vs_gt": float(np.median(rows["root_ratio"])) if rows["root_ratio"] else float("nan"),
        "lag0": float(np.mean(rows["lag0"])) if rows["lag0"] else float("nan"),
        "jitter": float(np.nanmean(rows["jitter"])) if rows["jitter"] else float("nan"),
        "twist": float(np.nanmean(rows["twist"])) if rows["twist"] else float("nan"),
        "skate": float(np.nanmean(rows["skate"])) if rows["skate"] else float("nan"),
        "wristsync": float(np.nanmean(rows["wristsync"])) if rows["wristsync"] else float("nan"),
        "energy_corr": (float(np.corrcoef(rows["gt_energy"], rows["own_energy"])[0, 1])
                        if len(rows.get("gt_energy", [])) > 3 else float("nan")),
        "seg_per_s": float(np.mean(rows["seg"])) if rows["seg"] else float("nan"),
        "beat_R": float(np.median(resultants[valid])) if total else float("nan"),
        "beat_R_null": float(np.median(nulls[valid])) if total else float("nan"),
        "beat_wins": "{}/{} P={:.3g}".format(wins, total, binomtest(wins, total, 0.5).pvalue)
                     if total else "-",
    }


HEADER = ("{:<30} {:>5} {:>8} {:>8} {:>7} {:>7} {:>8} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {:>7}  {}"
          .format("arm", "n", "energy", "vs GT", "span", "root", "vs GT",
                  "lag0", "jitter", "seg/s", "twist", "wsync", "skate", "e_corr", "R wins"))
ROW = ("{arm:<30} {clips:>5} {energy_ms:>8.3f} {energy_vs_gt:>8.3f} {span_m:>7.3f} "
       "{root_m:>7.2f} {root_vs_gt:>8.3f} {lag0:>7.3f} {jitter:>7.3f} {seg_per_s:>7.3f} {twist:>6.1f} {wristsync:>6.3f} {skate:>6.3f} {energy_corr:>7.3f} "
       "{beat_wins}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--labels-root", default=None,
                        help="M3 label tree, for rows whose pickles carry no labels "
                             "(the ground truth's do not)")
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--json", default=None)
    parser.add_argument("--gate-energy", type=float, default=None,
                        help="fail when energy_vs_gt falls below this")
    parser.add_argument("--gate-root", type=float, default=None)
    parser.add_argument("--gate-jitter", type=float, default=None,
                        help="fail when the 5-15 Hz share EXCEEDS this; pair it with "
                             "--gate-energy or an arm can pass by shaking")
    arm_sourcing.add_arguments(parser)
    args = parser.parse_args()

    requested = [line.strip() for line in open(args.clips) if line.strip()]
    # THE SOURCING GATE.  Clips whose draft was never retrieved are the
    # completion running on music alone, not this arm; pooling them mixes two
    # experiments.  Excluded from EVERY arm and from ground truth, so the table
    # still compares like with like.  See tools/arm_sourcing.py.
    arm_specs = [(spec.partition("=")[0], spec.partition("=")[2]) for spec in args.arm]
    clips, sourcing = arm_sourcing.select_clips(
        requested, arm_specs, threshold=args.sourcing_threshold,
        include_unsourced=args.include_unsourced)
    for line in arm_sourcing.format_header(sourcing):
        print(line)
    gt_dir = pathlib.Path(args.ground_truth_dir)
    ground_truth = {}
    for clip in clips:
        path = gt_dir / (clip + ".pkl")
        if path.is_file():
            ground_truth[clip] = load(path)[0]
    labels_root = None
    if args.labels_root:
        root = pathlib.Path(args.labels_root)
        labels_root = {}
        for line in open(root / "labels.jsonl"):
            record = json.loads(line)
            if record["sequence_id"] in clips:
                labels_root[record["sequence_id"]] = np.load(root / record["labels_path"])
    audio_dir = pathlib.Path(args.audio_dir) if args.audio_dir else None

    print(HEADER)
    reports, failed = [], []
    for name, directory in arm_specs:
        rows = score(pathlib.Path(directory), clips, ground_truth, audio_dir, labels_root)
        report = summarise(name, rows)
        reports.append(report)
        print(ROW.format(**report))
        if args.gate_energy is not None and not report["energy_vs_gt"] >= args.gate_energy:
            failed.append("{}: energy_vs_gt {:.3f} < {:.3f}".format(
                name, report["energy_vs_gt"], args.gate_energy))
        if args.gate_root is not None and not report["root_vs_gt"] >= args.gate_root:
            failed.append("{}: root_vs_gt {:.3f} < {:.3f}".format(
                name, report["root_vs_gt"], args.gate_root))
        if args.gate_jitter is not None and not report["jitter"] <= args.gate_jitter:
            failed.append("{}: jitter {:.3f} > {:.3f}".format(
                name, report["jitter"], args.gate_jitter))
    if args.json:
        # A dict, not the bare list this used to write: the sourcing header
        # travels WITH the numbers, so a filtered table cannot be read as an
        # unfiltered one.  Readers of the old shape take payload["arms"].
        payload = {"sourcing": sourcing, "arms": reports}
        pathlib.Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for line in failed:
        print("FAIL " + line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
