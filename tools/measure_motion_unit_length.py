#!/usr/bin/env python3
"""How long is one motion unit *in this corpus*, rather than in the paper's.

The paper's segments run about a second on AIST++, and that number has been
carried into the wild corpus as if it were a law.  It is not: a second is a
property of AIST++'s tempo and genre mix.  Two corpora of different dances have
no reason to share a segment duration, so a target expressed in **seconds**
cannot transfer.  What can transfer is a *structural* ratio, and this file
measures the quantity that ratio is built from.

The grain, and why it is the paper's own construction
-----------------------------------------------------
A **motion beat** is a local minimum of mean joint speed -- where a movement
momentarily settles.  It is not invented here: the paper defines it and uses it
in M3, "selected keyframes identified as motion beats (local minima of
segment-wise joint velocities)", and ``tools/motion_beats.find_motion_beats``
is this repository's implementation, reused unmodified.

**Beats are not boundaries.**  The paper's own example -- a kick is "the
preparatory weight shift, leg extension, and recovery" -- contains a beat in
the middle of it: the leg at full extension is a speed minimum.  So a complete
motion process spans *several* beats, and the segment target is

    target_seconds = k * inter_beat_interval

where ``k`` is beats per motion unit.  ``k`` is what is estimated on AIST++,
where the paper's own duration distribution is published, and then carried to
the wild corpus, whose ``inter_beat_interval`` this tool measures directly.
Carrying ``k`` rather than seconds is the same discipline
``report_paper_alignment.py`` already applies to cluster populations: rates and
shapes transfer between corpora of different size, absolute counts do not.

The knob is swept, not chosen
-----------------------------
``find_motion_beats`` takes ``min_separation`` (frames) and ``prominence``, and
both place a floor under the interval this tool reports.  A single setting would
make the answer a restatement of the setting, so ``--sweep`` runs a grid and the
report carries every cell.  A conclusion that survives only one cell is not a
conclusion.

The instrument is shown to have power before any null result is read
--------------------------------------------------------------------
If two corpora come out the same, that is only informative once the measurement
has been shown capable of telling them apart.  ``--scale-control`` resamples a
subsample in time by known factors and checks the measured interval moves by
those factors.  An instrument that reads the same interval on a corpus played at
half speed is measuring its own smoothing window, not the dance.

Usage::

    measure_motion_unit_length.py --bundle data/atomic_aistpp/aist_full_performance_v1 \\
        --sample 400 --scale-control --output runs/unit_length_aist.json
    measure_motion_unit_length.py --bundle /cache/.../wild_v4_raw_bundle \\
        --clips runs/clean5/clips.txt --sample 400 --output runs/unit_length_clean5.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FPS = 30.0
# Fig. 4a costed two ways, because the paper's three published figures are
# mutually inconsistent (see report_paper_alignment.py).  Both are carried.
PAPER_FIG4A = {"<0.7": 1366, "0.7-0.9": 7712, "0.9-1.1": 8054, "1.1-1.3": 2095, ">1.3": 3967}
PAPER_FLOORS = {"<0.7": 0.0, "0.7-0.9": 0.7, "0.9-1.1": 0.9, "1.1-1.3": 1.1, ">1.3": 1.3}
PAPER_MIDS = {"<0.7": 0.35, "0.7-0.9": 0.8, "0.9-1.1": 1.0, "1.1-1.3": 1.2, ">1.3": 1.5}


class UnitLengthError(RuntimeError):
    """A measurement that would be misleading is refused rather than reported."""


def paper_segment_seconds() -> Dict[str, float]:
    total = sum(PAPER_FIG4A.values())
    return {
        "segments": total,
        "mean_at_bucket_floors": sum(PAPER_FLOORS[k] * v for k, v in PAPER_FIG4A.items()) / total,
        "mean_at_bucket_midpoints": sum(PAPER_MIDS[k] * v for k, v in PAPER_FIG4A.items()) / total,
        "text_hours_over_segments": 5.2 * 3600 / total,
        "note": "the three disagree; report_paper_alignment.py:21-32 has the arithmetic",
    }


def bundle_rows(bundle: pathlib.Path) -> List[dict]:
    manifest = bundle / "sequences.jsonl"
    if not manifest.is_file():
        raise UnitLengthError("no sequences.jsonl under {}".format(bundle))
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def stem_of(row: dict) -> str:
    """``wild_v4:<upload>:clipNNN`` -> ``<upload>__clipNNN``; AIST ids pass through."""
    name = row.get("recording_id") or row.get("sequence_id") or ""
    parts = name.split(":")
    if len(parts) == 3:
        return "{}__{}".format(parts[1], parts[2])
    return name


def beats_of(joints: np.ndarray, *, min_separation: int, prominence: float) -> List[int]:
    from tools.motion_beats import find_motion_beats

    return find_motion_beats(joints, min_separation=min_separation,
                             max_beats=len(joints), prominence=prominence)


def intervals_for(joints: np.ndarray, *, min_separation: int,
                  prominence: float, estimator: str = "beats") -> np.ndarray:
    if estimator == "autocorr":
        period = autocorr_period(joints)
        return np.zeros(0) if period is None else np.array([period])
    beats = beats_of(joints, min_separation=min_separation, prominence=prominence)
    if len(beats) < 2:
        return np.zeros(0)
    return np.diff(np.asarray(beats, dtype=np.float64)) / FPS


def autocorr_period(joints: np.ndarray, *, min_lag: int = 4,
                    max_lag: int = 90) -> Optional[float]:
    """Settle-to-settle interval as the first prominent peak of the speed autocorrelation.

    Counting local minima is the definition, but on this corpus it is not the
    measurement: monocular reconstruction adds frame-scale jitter to the speed
    trace, every wobble of which is a local minimum, and ``find_motion_beats``
    answers with a fixed 3-frame smoothing window whose *duration* does not move
    when the tempo does.  Measured on 2026-08-20, that leaves the beat estimator
    reading only 1.22x on wild motion played at 2x duration where 2.00x is the
    truth -- and 1.56x on AIST++, so its error is corpus-dependent and the two
    corpora cannot be compared through it.

    Autocorrelation is the same quantity read a way that jitter cannot dominate:
    additive frame-scale noise is broadband and lands almost entirely at lag 0,
    while the settle rhythm is what repeats.  It is validated by the same scale
    control, not asserted.
    """
    speed = np.asarray(joint_speed_of(joints), dtype=np.float64)
    if len(speed) < 2 * min_lag + 4:
        return None
    speed = speed - speed.mean()
    norm = float((speed * speed).sum())
    if norm <= 0:
        return None
    top = min(max_lag, len(speed) - min_lag - 1)
    if top <= min_lag:
        return None
    lags = np.arange(min_lag, top + 1)
    values = np.array([float((speed[:-lag] * speed[lag:]).sum()) / norm for lag in lags])
    if len(values) < 3:
        return None
    # First interior local maximum: the *first* repetition, not the strongest,
    # because a phrase that repeats twice puts a taller peak at twice the period.
    interior = np.where((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1
    interior = [i for i in interior if values[i] > 0.0]
    if not interior:
        return None
    return float(lags[interior[0]]) / FPS


def joint_speed_of(joints: np.ndarray) -> np.ndarray:
    from tools.motion_beats import joint_speed

    return joint_speed(joints)


def resample(joints: np.ndarray, factor: float) -> np.ndarray:
    """Play the motion at ``factor`` x duration, by linear interpolation in time.

    Speeding up (``factor`` < 1) decimates, and decimating without a low-pass
    first aliases the fast content down into the band the estimator reads -- so
    the control would be testing this function rather than the estimator.  That
    is not hypothetical: measured on 2026-08-20, both estimators read ~1.5x
    error on the 0.5x arm and <=1.15x on the 2.0x arm, and the asymmetry
    disappeared once this filter was added.  Width ``1/factor`` is the Nyquist
    window for the target rate.
    """
    if factor < 1.0:
        width = int(round(1.0 / factor))
        if width > 1:
            kernel = np.ones(width) / width
            padded = np.pad(joints, ((width, width), (0, 0), (0, 0)), mode="edge")
            smoothed = np.empty_like(padded)
            for joint in range(padded.shape[1]):
                for axis in range(padded.shape[2]):
                    smoothed[:, joint, axis] = np.convolve(
                        padded[:, joint, axis], kernel, mode="same")
            joints = smoothed[width:-width]
    length = max(3, int(round(len(joints) * factor)))
    source = np.linspace(0.0, len(joints) - 1.0, length)
    low = np.floor(source).astype(int)
    high = np.clip(low + 1, 0, len(joints) - 1)
    weight = (source - low)[:, None, None]
    return joints[low] * (1.0 - weight) + joints[high] * weight


def summarise(values: Sequence[float]) -> Optional[Dict[str, float]]:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not len(array):
        return None
    return {"n": int(len(array)),
            "median": round(float(np.median(array)), 4),
            "mean": round(float(array.mean()), 4),
            "p25": round(float(np.percentile(array, 25)), 4),
            "p75": round(float(np.percentile(array, 75)), 4)}


def measure(bundle: pathlib.Path, *, clips: Optional[set], sample: int,
            grid: Sequence[tuple], scale_control: bool,
            estimator: str = "beats") -> Dict[str, object]:
    from tools.convert_motion_to_guofeats import motion_151_to_joints

    rows = bundle_rows(bundle)
    if clips is not None:
        rows = [row for row in rows if stem_of(row) in clips]
        if not rows:
            raise UnitLengthError("no bundle row matches the clip manifest")
    rows.sort(key=stem_of)
    if sample and sample < len(rows):
        # Even stride, never a prefix: release row order is not random and this
        # repo has already been bitten by a prefix standing in for a sample.
        step = len(rows) / float(sample)
        rows = [rows[int(index * step)] for index in range(sample)]

    joints_cache: List[np.ndarray] = []
    for row in rows:
        motion = np.load(bundle / row["motion_path"])
        joints_cache.append(motion_151_to_joints(motion))

    cells = []
    for min_separation, prominence in grid:
        per_clip_median, pooled = [], []
        beats_per_second = []
        for joints in joints_cache:
            gaps = intervals_for(joints, min_separation=min_separation,
                                 prominence=prominence, estimator=estimator)
            if not len(gaps):
                continue
            per_clip_median.append(float(np.median(gaps)))
            pooled.extend(gaps.tolist())
            beats_per_second.append((len(gaps) + 1) / (len(joints) / FPS))
        cells.append({
            "min_separation_frames": min_separation,
            "prominence": prominence,
            "clips_with_at_least_two_beats": len(per_clip_median),
            "interval_seconds_pooled": summarise(pooled),
            "interval_seconds_per_clip_median": summarise(per_clip_median),
            "beats_per_second": summarise(beats_per_second),
        })

    control = None
    if scale_control:
        # Gate 3 of CLAUDE.md 2.1: before a "the two corpora agree" reading can
        # mean anything, the instrument has to move when the tempo does.
        base = grid[0]
        control = {"factors": {}, "reference": {"min_separation_frames": base[0],
                                                "prominence": base[1]}}
        subset = joints_cache[: min(80, len(joints_cache))]
        for factor in (0.5, 1.0, 2.0):
            medians = []
            for joints in subset:
                gaps = intervals_for(resample(joints, factor),
                                     min_separation=base[0], prominence=base[1],
                                     estimator=estimator)
                if len(gaps):
                    medians.append(float(np.median(gaps)))
            control["factors"][str(factor)] = summarise(medians)
        one = control["factors"]["1.0"]
        for factor in ("0.5", "2.0"):
            cell = control["factors"][factor]
            if cell and one:
                control.setdefault("observed_over_expected", {})[factor] = round(
                    (cell["median"] / one["median"]) / float(factor), 4)
        control["reading"] = ("observed_over_expected near 1.0 means the measured "
                              "interval tracks tempo; near 1/factor means it is "
                              "pinned by min_separation or the smoothing window "
                              "and cannot see tempo at all")

    return {
        "bundle": str(bundle),
        "clips_measured": len(rows),
        "clips_available": len(bundle_rows(bundle)) if clips is None else None,
        "grid": cells,
        "scale_control": control,
        "paper_segment_seconds": paper_segment_seconds(),
        "what_this_is_not": ("a segment length.  A complete motion process spans "
                             "several beats -- the paper's own kick contains one at "
                             "full extension -- so the segment target is k x this "
                             "interval, with k estimated where the paper's duration "
                             "is published"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=pathlib.Path, required=True)
    parser.add_argument("--clips", type=pathlib.Path, default=None)
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--min-separation", type=int, nargs="+", default=[3, 5, 8, 12])
    parser.add_argument("--prominence", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--estimator", choices=("beats", "autocorr"), default="beats",
                        help="beats = count local speed minima (the definition); "
                             "autocorr = first peak of the speed autocorrelation "
                             "(jitter-robust). Both must pass --scale-control "
                             "before either is read.")
    parser.add_argument("--scale-control", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)

    clips = None
    if args.clips:
        clips = {line.strip() for line in args.clips.read_text(encoding="utf-8").splitlines()
                 if line.strip()}
    grid = [(sep, prom) for sep in args.min_separation for prom in args.prominence]
    result = measure(args.bundle, clips=clips, sample=args.sample, grid=grid,
                     scale_control=args.scale_control, estimator=args.estimator)
    result["estimator"] = args.estimator

    print("{}  {} clips".format(args.bundle, result["clips_measured"]))
    print("{:>8} {:>10} {:>8} {:>10} {:>10} {:>10}".format(
        "min_sep", "prominence", "clips", "median_s", "p25", "p75"))
    for cell in result["grid"]:
        stat = cell["interval_seconds_per_clip_median"] or {}
        print("{:>8} {:>10} {:>8} {:>10} {:>10} {:>10}".format(
            cell["min_separation_frames"], cell["prominence"],
            cell["clips_with_at_least_two_beats"],
            stat.get("median", "-"), stat.get("p25", "-"), stat.get("p75", "-")))
    if result["scale_control"]:
        print("scale control (observed/expected, 1.0 = tracks tempo):",
              json.dumps(result["scale_control"].get("observed_over_expected", {})))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
