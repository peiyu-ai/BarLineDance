#!/usr/bin/env python3
"""One verdict per arm, with every criterion that has already been missed once.

On 2026-08-18 five conclusions about this pipeline were overturned inside a
single day.  They have one cause, not five: **the criteria were incomplete at
the moment the conclusion was written**, in four distinct ways, each of which
had happened at least once before.  This tool exists so that a reader of an
arm's result cannot be handed a number without the things that decide whether
it means anything.

The four failures, and what this tool does about each:

* **One feature family.** ``fid_k`` (kinetic, 72-D velocity/energy) and
  ``fid_m`` (manual, 32-D pose geometry) can move in opposite directions on the
  same generations.  The ``random`` retrieval arm improved fid_k on both seeds
  and lost fid_m in three of four cells, because what it actually changed was
  draft *speed* -- ``_values_at`` resamples 97.3% of random picks against 10.2%
  of duration picks -- and kinetic features are velocity features.  So both
  families are always computed and a disagreement in sign is a *verdict*
  (``SPLIT_FAMILIES``), not a footnote.

* **No paired test.** The same ``random`` arm's headline cell is p = 0.206
  under a per-clip paired exchangeability permutation.  A difference that looks
  consistent across two seeds can still be inside the estimator's own noise,
  because the two seeds are not two independent samples of the same size.

* **The noise band was never measured beside the effect.** The only honest
  denominator is the same configuration re-run under a different seed.  On this
  protocol it is 2.7% on the leak-free set and 6.2% on the full one -- and the
  SELF medoid effect is *smaller on the full set than the seed alone*.  The
  band is therefore measured from a supplied seed pair rather than hardcoded,
  and an arm inside it is ``UNDETERMINED``, never "no change".

* **fid decided alone.** ``eval/metrics.py`` standardises each distribution by
  its own per-dimension statistics before FID, so fid is exactly invariant to
  any per-dimension affine change of the features, and -- measured here, not
  argued -- invariant to *permuting the generated rows*, which destroys every
  generated-to-ground-truth pairing.  It cannot see roughness, amplitude, or
  whether the clip was danced to its own song.  The ``dn005`` arm moves every
  visible axis toward ground truth while fid_k regresses 64-77%.  So roughness
  rides in the verdict and an arm whose two axes disagree is reported as
  ``AXES_DISAGREE`` rather than as a regression full stop.

**This gate can fail, and its acceptance test is that it fails in known
places.**  ``--replay`` scores the twelve arms of 2026-08-18 and asserts the
three rulings that were established by hand: ``random`` must not come out
IMPROVED, SELF ``medoid`` must come out UNDETERMINED, and ``gapfill``/``root_xy``
must land inside the noise band.  A change to the verdict logic that loses any
of those has broken the tool, not discovered something.

FID is computed here rather than shelled out to ``eval/evaluate.py`` because the
permutation needs thousands of evaluations.  ``tr(sqrt(A·B))`` is taken from the
eigenvalues of ``A·B`` (real and non-negative for PSD A, B) instead of
``scipy.linalg.sqrtm``, and the agreement with ``eval.metrics.calc_fid`` is
checked on the real matrices at startup -- if it ever drifts, the tool refuses
rather than reporting the faster number.

Usage::

    python3 tools/judge_arm.py \\
        --arm runs/t18_draft/random_self/features \\
        --baseline runs/t14_oracle/features_self \\
        --ground-truth runs/t14_oracle/gt_features340 \\
        --clean-arm runs/t18_draft/random_self/clean \\
        --clean-baseline runs/t14_oracle/clean_self \\
        --clean-ground-truth runs/t14_oracle/clean_gt \\
        --noise-pair runs/t14_oracle/features_self:runs/t18_draft/s2_duration/features \\
        --clean-noise-pair runs/t14_oracle/clean_self:runs/t18_draft/s2_duration/clean \\
        --roughness-arm runs/t20_roughness/random_self_s1.json \\
        --roughness-baseline runs/t20_roughness/self_duration_s1.json \\
        --output runs/t23_judge/random_self.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# The two families eval/evaluate.py reports.  Both, always: the whole point.
FAMILIES = ("kinetic", "manual")
FAMILY_DIR = {"kinetic": "kinetic_features", "manual": "manual_features"}
FAMILY_KEY = {"kinetic": "fid_k", "manual": "fid_m"}

# The roughness axes that fid is invariant to.  Each is reported as a ratio to
# ground truth, so 1.0 is the target and the direction of "better" is |x-1|.
ROUGHNESS_AXES = (
    ("jerk", ("jerk", "median")),
    ("joint_speed", ("speed", "median_of_all")),
    ("ankle_speed", ("speed", "ankle")),
    ("across_joint_cv", ("speed", "across_joint_cv")),
    ("wrist_over_pelvis", ("speed", "wrist_over_pelvis")),
    ("floor_path", ("amplitude", "floor_path_m_per_s")),
    ("reach_span", ("amplitude", "reach_span_shoulders")),
    ("turn_rate", ("amplitude", "turn_rate_deg_s")),
)

DEFAULT_PERMUTATIONS = 2000
# Below this the observed delta is not distinguishable from re-drawing the same
# arm, so no verdict is issued.  Measured from --noise-pair when one is given.
FALLBACK_NOISE_FRACTION = 0.027


class JudgeError(RuntimeError):
    """A criterion could not be evaluated, which is never the same as passing."""


# --------------------------------------------------------------------------- #
# feature loading


def clip_of(stem: str) -> str:
    """Strip the ``_s<seed>`` suffix inference appends, so arms align by clip.

    Two arms of one protocol carry the same clips under different seeds, and the
    paired test needs them matched.  ``rfind`` rather than ``split`` because clip
    names contain underscores of their own.
    """
    marker = stem.rfind("_s")
    if marker > 0 and stem[marker + 2:].isdigit():
        return stem[:marker]
    return stem


def load_family(directory: pathlib.Path, family: str) -> Tuple[List[str], np.ndarray]:
    root = pathlib.Path(directory) / FAMILY_DIR[family]
    if not root.is_dir():
        raise JudgeError("missing {} features: {}".format(family, root))
    paths = sorted(root.glob("*.npy"))
    if len(paths) < 2:
        raise JudgeError("need at least two samples in {}".format(root))
    names = [clip_of(p.stem) for p in paths]
    if len(set(names)) != len(names):
        raise JudgeError("duplicate clip keys in {} -- clip_of collided".format(root))
    matrix = np.stack([np.load(p).reshape(-1) for p in paths]).astype(np.float64)
    return names, matrix


def align(a_names: Sequence[str], a: np.ndarray,
          b_names: Sequence[str], b: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Restrict two arms to the clips they share, in one order.

    A paired test on unpaired rows is not a paired test; refusing here is
    cheaper than discovering it in the p-value.
    """
    shared = sorted(set(a_names) & set(b_names))
    if len(shared) < 2:
        raise JudgeError("arms share fewer than two clips")
    ai = {n: i for i, n in enumerate(a_names)}
    bi = {n: i for i, n in enumerate(b_names)}
    return shared, a[[ai[n] for n in shared]], b[[bi[n] for n in shared]]


# --------------------------------------------------------------------------- #
# FID


def standardize(features: np.ndarray) -> np.ndarray:
    """eval/metrics.normalize_separately, reproduced so the permutation can run.

    Each distribution by its own per-dimension mean and std.  This is the step
    that makes fid blind to per-dimension affine change; it is reproduced rather
    than worked around because the number has to stay the published one.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    return (features - mean) / std


def _trace_sqrt_product(sigma_a: np.ndarray, sigma_b: np.ndarray) -> float:
    """tr(sqrt(A·B)) from eigenvalues instead of a matrix square root.

    A·B is similar to a symmetric PSD matrix when A and B are PSD, so its
    eigenvalues are real and non-negative up to numerical error; clipping at
    zero is the standard treatment and is the same thing scipy's sqrtm does
    with its imaginary-part tolerance.
    """
    values = np.linalg.eigvals(sigma_a @ sigma_b)
    if np.abs(values.imag).max() > 1e-4 * max(1.0, np.abs(values.real).max()):
        raise JudgeError("covariance product has a large imaginary spectrum")
    return float(np.sqrt(np.clip(values.real, 0.0, None)).sum())


def fid(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    pred = standardize(prediction)
    truth = standardize(ground_truth)
    difference = pred.mean(axis=0) - truth.mean(axis=0)
    sigma_p = np.atleast_2d(np.cov(pred, rowvar=False))
    sigma_t = np.atleast_2d(np.cov(truth, rowvar=False))
    return float(
        difference.dot(difference)
        + np.trace(sigma_p)
        + np.trace(sigma_t)
        - 2.0 * _trace_sqrt_product(sigma_p, sigma_t)
    )


def check_against_reference(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Refuse to use the fast path if it has drifted from the published one."""
    from eval.metrics import calc_fid, normalize_separately

    reference = calc_fid(normalize_separately(prediction), normalize_separately(ground_truth))
    mine = fid(prediction, ground_truth)
    if abs(mine - reference) > 1e-6 * max(1.0, abs(reference)):
        raise JudgeError(
            "fast FID disagrees with eval.metrics.calc_fid: {} vs {}".format(mine, reference)
        )
    return reference


# --------------------------------------------------------------------------- #
# paired permutation


def paired_permutation(arm: np.ndarray, baseline: np.ndarray, truth: np.ndarray,
                       *, permutations: int, seed: int) -> Dict[str, float]:
    """Null: the arm's flag changes nothing, so per clip the two rows may swap.

    This is the exchangeability the arms actually have -- same clip, same seed,
    same everything but the flag -- and it is stronger than a bootstrap over
    clips because it holds the clip set fixed.  One-sided in the observed
    direction; the sign convention is baseline - arm, so positive means the arm
    improved (lower fid).
    """
    observed = fid(baseline, truth) - fid(arm, truth)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        swap = rng.random(len(arm)) < 0.5
        a = np.where(swap[:, None], baseline, arm)
        b = np.where(swap[:, None], arm, baseline)
        null[index] = fid(b, truth) - fid(a, truth)
    tail = float((null >= observed).mean()) if observed >= 0 else float((null <= observed).mean())
    return {
        "observed_delta": observed,
        "null_sd": float(null.std(ddof=1)),
        "p_one_sided": tail,
        "permutations": permutations,
    }


# --------------------------------------------------------------------------- #
# roughness


def _dig(payload: dict, path: Sequence[str]):
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node if isinstance(node, (int, float)) else None


def roughness_ratios(report: dict) -> Dict[str, Optional[float]]:
    """Each axis as generated/ground-truth, from one score_generation_diagnostics run.

    A ratio and not a raw value because the ground truth of a 24-clip prefix and
    of the 300-clip protocol differ by 1.41x on both speed and jerk, and every
    cost gate calibrated on the former was wrong by that factor.
    """
    out: Dict[str, Optional[float]] = {}
    for name, path in ROUGHNESS_AXES:
        generated = _dig(report.get("summary", {}), path)
        truth = _dig(report.get("ground_truth", {}), path)
        out[name] = (generated / truth) if (generated is not None and truth) else None
    return out


def roughness_movement(arm: Dict[str, Optional[float]],
                       baseline: Dict[str, Optional[float]]) -> Dict[str, object]:
    """How many axes moved toward 1.0, which is where ground truth sits."""
    toward, away, unchanged = [], [], []
    for name in arm:
        a, b = arm.get(name), baseline.get(name)
        if a is None or b is None:
            continue
        gain = abs(b - 1.0) - abs(a - 1.0)
        if abs(gain) < 1e-9:
            unchanged.append(name)
        elif gain > 0:
            toward.append(name)
        else:
            away.append(name)
    return {"toward_ground_truth": sorted(toward),
            "away_from_ground_truth": sorted(away),
            "unchanged": sorted(unchanged),
            "net": len(toward) - len(away)}


# --------------------------------------------------------------------------- #
# the verdict


def verdict_for_set(deltas: Dict[str, Dict[str, float]], band: Dict[str, float],
                    baseline_fid: Dict[str, float], alpha: float) -> Dict[str, object]:
    """Noise band first, then family agreement, then the p-value.

    The order is the correction of a real mistake.  A first version tested
    family agreement first, and reported SPLIT_FAMILIES for the `gapfill
    interpolate` arm whose two deltas were +0.45% and -0.89% -- both inside
    their own seed-noise bands, i.e. two re-draws of nothing pointing opposite
    ways.  It also swallowed the `no-plan` arm's -41% kinetic regression
    because manual had drifted +4% the other way.  Two numbers can only
    disagree once each of them is a number: a family whose delta is inside its
    band carries no sign to disagree with.

    Sign convention is baseline - arm, so positive means the arm improved.
    """
    reasons: List[str] = []
    fractions = {
        f: abs(deltas[f]["observed_delta"]) / abs(baseline_fid[f]) if baseline_fid[f] else 0.0
        for f in FAMILIES
    }
    outside = [f for f in FAMILIES if fractions[f] >= band[f]]

    if not outside:
        reasons.append(
            "both families inside their seed-noise bands: kinetic {:.2%} of {:.2%}, "
            "manual {:.2%} of {:.2%}".format(
                fractions["kinetic"], band["kinetic"], fractions["manual"], band["manual"])
        )
        return {"verdict": "UNDETERMINED", "reasons": reasons}

    signs = {f: np.sign(deltas[f]["observed_delta"]) for f in outside}
    if len(outside) == 2 and signs["kinetic"] != signs["manual"]:
        reasons.append(
            "both families are outside their bands and disagree in sign: "
            "kinetic {:+.4f} ({:.2%}), manual {:+.4f} ({:.2%})".format(
                deltas["kinetic"]["observed_delta"], fractions["kinetic"],
                deltas["manual"]["observed_delta"], fractions["manual"])
        )
        return {"verdict": "SPLIT_FAMILIES", "reasons": reasons}

    # Kinetic decides when it has something to say, because it is the family the
    # headline is quoted in; manual decides alone only when kinetic did not move.
    decisive = "kinetic" if "kinetic" in outside else "manual"
    quiet = "manual" if decisive == "kinetic" else "kinetic"

    if deltas[decisive]["p_one_sided"] > alpha:
        reasons.append(
            "{} delta {:.2%} clears its {:.2%} band but paired permutation "
            "p = {:.3f} > {:.3f}".format(
                decisive, fractions[decisive], band[decisive],
                deltas[decisive]["p_one_sided"], alpha)
        )
        return {"verdict": "UNDETERMINED", "reasons": reasons}

    reasons.append(
        "{} decides: {:+.4f} ({:.2%} of baseline, band {:.2%}), p = {:.3f}".format(
            decisive, deltas[decisive]["observed_delta"], fractions[decisive],
            band[decisive], deltas[decisive]["p_one_sided"])
    )
    if np.sign(deltas[quiet]["observed_delta"]) != np.sign(deltas[decisive]["observed_delta"]):
        reasons.append(
            "{} drifted the other way but stayed inside its band "
            "({:+.4f}, {:.2%} of {:.2%})".format(
                quiet, deltas[quiet]["observed_delta"], fractions[quiet], band[quiet])
        )
    return {"verdict": "IMPROVED" if deltas[decisive]["observed_delta"] > 0 else "REGRESSED",
            "reasons": reasons}


def combine(sets: Dict[str, Dict[str, object]], roughness: Optional[Dict[str, object]]) -> Dict[str, object]:
    """One verdict from the sets that were supplied, plus the axis fid cannot see.

    The leak-free set decides when it exists -- it is the one a headline may
    quote -- but a disagreement with the full set is recorded, because that is
    exactly what happened to the SELF medoid arm and reading only one of them
    produced a ruling that does not hold.
    """
    reasons: List[str] = []
    primary = "clean" if "clean" in sets else "all"
    verdict = str(sets[primary]["verdict"])
    reasons.extend("[{}] {}".format(primary, r) for r in sets[primary]["reasons"])

    other = "all" if primary == "clean" else None
    if other and other in sets:
        reasons.extend("[{}] {}".format(other, r) for r in sets[other]["reasons"])
        if sets[other]["verdict"] != verdict:
            reasons.append(
                "sets disagree: {} on the leak-free set, {} on the full one".format(
                    verdict, sets[other]["verdict"])
            )
            verdict = "UNDETERMINED"

    if roughness and verdict == "REGRESSED" and roughness["net"] > 0:
        reasons.append(
            "but {} roughness axes moved toward ground truth and {} away".format(
                len(roughness["toward_ground_truth"]), len(roughness["away_from_ground_truth"]))
        )
        verdict = "AXES_DISAGREE"
    return {"verdict": verdict, "reasons": reasons}


# --------------------------------------------------------------------------- #
# driver


def score_set(arm_dir, baseline_dir, truth_dir, *, permutations, seed, verify) -> Dict[str, object]:
    result: Dict[str, Dict[str, float]] = {}
    baseline_fid: Dict[str, float] = {}
    n_shared = None
    for family in FAMILIES:
        a_names, a = load_family(pathlib.Path(arm_dir), family)
        b_names, b = load_family(pathlib.Path(baseline_dir), family)
        _, truth = load_family(pathlib.Path(truth_dir), family)
        shared, a, b = align(a_names, a, b_names, b)
        n_shared = len(shared)
        if verify:
            check_against_reference(a, truth)
        baseline_fid[family] = fid(b, truth)
        stats = paired_permutation(a, b, truth, permutations=permutations, seed=seed)
        stats["arm_fid"] = fid(a, truth)
        stats["baseline_fid"] = baseline_fid[family]
        result[family] = stats
    return {"clips": n_shared,
            "fid": {FAMILY_KEY[f]: result[f]["arm_fid"] for f in FAMILIES},
            "baseline_fid": {FAMILY_KEY[f]: baseline_fid[f] for f in FAMILIES},
            "families": result}


def measure_band(pair: str, truth_dir) -> Dict[str, float]:
    """The seed-noise band per family, from two runs of one configuration.

    Per family because the two are different estimators on different feature
    counts (72-D velocity vs 32-D geometry) and there is no reason their
    re-draw spread should match -- measured on this protocol it does not.
    Hardcoding either would make the denominator survive a protocol change that
    invalidates it, which is the shape of every stale constant this repo has
    already paid for.
    """
    left, right = pair.split(":", 1)
    band: Dict[str, float] = {}
    for family in FAMILIES:
        l_names, left_m = load_family(pathlib.Path(left), family)
        r_names, right_m = load_family(pathlib.Path(right), family)
        _, truth = load_family(pathlib.Path(truth_dir), family)
        _, left_m, right_m = align(l_names, left_m, r_names, right_m)
        a, b = fid(left_m, truth), fid(right_m, truth)
        band[family] = abs(a - b) / max(abs(a), abs(b))
    return band


# --------------------------------------------------------------------------- #
# acceptance: the arms of 2026-08-18, and the rulings established by hand

# The baseline for each arm is the SELF+duration arm at the *same* seed, which
# is what every one of these was actually run against.
REPLAY_BASELINE = {
    20260816: ("runs/t14_oracle/features_self", "runs/t14_oracle/clean_self",
               "runs/t20_roughness/self_duration_s1.json"),
    20260817: ("runs/t18_draft/s2_duration/features", "runs/t18_draft/s2_duration/clean",
               "runs/t20_roughness/self_duration_s2.json"),
}
REPLAY_TRUTH = ("runs/t14_oracle/gt_features340", "runs/t14_oracle/clean_gt")
REPLAY_ARMS = [
    ("noplan_s1", 20260816, "runs/t16_noplan/features", "runs/t16_noplan/clean",
     "runs/t20_roughness/noplan_s1.json"),
    ("medoid_s1", 20260816, "runs/t17_medoid/features_medoid", "runs/t17_medoid/clean_medoid",
     "runs/t20_roughness/medoid_s1.json"),
    ("random_s1", 20260816, "runs/t18_draft/random_self/features",
     "runs/t18_draft/random_self/clean", "runs/t20_roughness/random_self_s1.json"),
    ("gapfill_interp_s1", 20260816, "runs/t18_draft/gapfill_interp/features",
     "runs/t18_draft/gapfill_interp/clean", "runs/t20_roughness/gapfill_interp_s1.json"),
    ("root_xy_s1", 20260816, "runs/t18_draft/root_xy/features",
     "runs/t18_draft/root_xy/clean", "runs/t20_roughness/root_xy_s1.json"),
    ("dn005_s1", 20260816, "runs/t18_draft/dn005_s1/features",
     "runs/t18_draft/dn005_s1/clean", "runs/t20_roughness/dn005_s1.json"),
    ("random_s2", 20260817, "runs/t18_draft/s2_random/features",
     "runs/t18_draft/s2_random/clean", "runs/t20_roughness/random_self_s2.json"),
    ("noplan_s2", 20260817, "runs/t18_draft/s2_noplan/features",
     "runs/t18_draft/s2_noplan/clean", "runs/t20_roughness/noplan_s2.json"),
    ("dn005_s2", 20260817, "runs/t18_draft/dn005_s2/features",
     "runs/t18_draft/dn005_s2/clean", "runs/t20_roughness/dn005_s2.json"),
]
# What this gate must say, established by hand on 2026-08-18.  Mostly
# *forbidden* outcomes: a rule change that starts calling `random` an
# improvement has lost the criterion rather than found something.  "No
# conclusion" is a set of two, because which way an arm fails to be a result is
# informative but the ruling under test is only that it is not one.
NO_CONCLUSION = ("UNDETERMINED", "SPLIT_FAMILIES")
REPLAY_EXPECTATIONS = {
    "random_s1": {"not": ("IMPROVED",)},
    "random_s2": {"not": ("IMPROVED",)},
    "medoid_s1": {"is": NO_CONCLUSION},
    "gapfill_interp_s1": {"is": NO_CONCLUSION},
    "root_xy_s1": {"is": NO_CONCLUSION},
    "noplan_s1": {"is": ("REGRESSED", "AXES_DISAGREE")},
    "noplan_s2": {"is": ("REGRESSED", "AXES_DISAGREE")},
}
NOISE_PAIR = "runs/t14_oracle/features_self:runs/t18_draft/s2_duration/features"
CLEAN_NOISE_PAIR = "runs/t14_oracle/clean_self:runs/t18_draft/s2_duration/clean"


def replay(permutations: int, seed: int, alpha: float, output: Optional[str]) -> int:
    """Score the arms of 2026-08-18 and refuse if a settled ruling has moved.

    The arms live under runs/, which is not in git, so this is an acceptance
    command rather than a unit test -- and it REFUSES when a directory is
    missing instead of scoring whatever it can find, because a partial replay
    that exits zero is exactly the shape of gate this repo keeps paying for.
    """
    parser = build_parser()
    reports, failures = {}, []
    for name, seed_id, arm, clean_arm, roughness in REPLAY_ARMS:
        base_f, base_c, base_r = REPLAY_BASELINE[seed_id]
        for path in (arm, clean_arm, roughness, base_f, base_c, base_r, *REPLAY_TRUTH):
            if not pathlib.Path(path).exists():
                raise JudgeError("replay needs {} and it is missing".format(path))
        options = parser.parse_args([
            "--arm", arm, "--baseline", base_f, "--ground-truth", REPLAY_TRUTH[0],
            "--clean-arm", clean_arm, "--clean-baseline", base_c,
            "--clean-ground-truth", REPLAY_TRUTH[1],
            "--noise-pair", NOISE_PAIR, "--clean-noise-pair", CLEAN_NOISE_PAIR,
            "--roughness-arm", roughness, "--roughness-baseline", base_r,
            "--permutations", str(permutations), "--seed", str(seed),
            "--alpha", str(alpha), "--name", name,
        ])
        report = run(options)
        reports[name] = report
        verdict = report["verdict"]
        rule = REPLAY_EXPECTATIONS.get(name)
        status = "         "
        if rule:
            if "not" in rule and verdict in rule["not"]:
                failures.append("{}: got {}, which this gate must never say".format(name, verdict))
                status = "BROKEN   "
            elif "is" in rule and verdict not in rule["is"]:
                failures.append("{}: got {}, expected one of {}".format(
                    name, verdict, "/".join(rule["is"])))
                status = "BROKEN   "
            else:
                status = "as ruled "
        print("{:20} {:16} {}  {}".format(name, verdict, status, report["reasons"][0]))
    if output:
        path = pathlib.Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports, indent=2, sort_keys=True))
    if failures:
        print("\nACCEPTANCE FAILED:", file=sys.stderr)
        for line in failures:
            print("  " + line, file=sys.stderr)
        return 1
    print("\nacceptance: {} arms scored, every settled ruling reproduced".format(len(REPLAY_ARMS)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--arm")
    p.add_argument("--baseline")
    p.add_argument("--ground-truth")
    p.add_argument("--replay", action="store_true",
                   help="score the 2026-08-18 arms and check the settled rulings")
    p.add_argument("--clean-arm")
    p.add_argument("--clean-baseline")
    p.add_argument("--clean-ground-truth")
    p.add_argument("--noise-pair", help="A:B, two feature dirs of one config at two seeds")
    p.add_argument("--clean-noise-pair")
    p.add_argument("--roughness-arm")
    p.add_argument("--roughness-baseline")
    p.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    p.add_argument("--seed", type=int, default=20260818)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--name", default=None)
    p.add_argument("--output")
    p.add_argument("--no-verify-fid", action="store_true",
                   help="skip the agreement check against eval.metrics.calc_fid")
    return p


def run(options) -> Dict[str, object]:
    sets: Dict[str, Dict[str, object]] = {}
    measured: Dict[str, Dict[str, object]] = {}
    bands: Dict[str, Dict[str, float]] = {}

    measured["all"] = score_set(options.arm, options.baseline, options.ground_truth,
                                permutations=options.permutations, seed=options.seed,
                                verify=not options.no_verify_fid)
    bands["all"] = (measure_band(options.noise_pair, options.ground_truth)
                    if options.noise_pair else dict.fromkeys(FAMILIES, FALLBACK_NOISE_FRACTION))

    if options.clean_arm and options.clean_baseline and options.clean_ground_truth:
        measured["clean"] = score_set(options.clean_arm, options.clean_baseline,
                                      options.clean_ground_truth,
                                      permutations=options.permutations, seed=options.seed,
                                      verify=not options.no_verify_fid)
        bands["clean"] = (measure_band(options.clean_noise_pair, options.clean_ground_truth)
                          if options.clean_noise_pair
                          else dict.fromkeys(FAMILIES, FALLBACK_NOISE_FRACTION))

    for key, block in measured.items():
        by_family = {f: block["baseline_fid"][FAMILY_KEY[f]] for f in FAMILIES}
        sets[key] = verdict_for_set(block["families"], bands[key], by_family, options.alpha)

    movement = None
    if options.roughness_arm and options.roughness_baseline:
        arm_r = roughness_ratios(json.loads(pathlib.Path(options.roughness_arm).read_text()))
        base_r = roughness_ratios(json.loads(pathlib.Path(options.roughness_baseline).read_text()))
        movement = roughness_movement(arm_r, base_r)
        movement["arm_ratios"] = arm_r
        movement["baseline_ratios"] = base_r

    combined = combine(sets, movement)
    return {
        "name": options.name or pathlib.Path(options.arm).parent.name,
        "verdict": combined["verdict"],
        "reasons": combined["reasons"],
        "seed_noise_band": bands,
        "sets": {k: {**measured[k], "verdict": sets[k]["verdict"], "reasons": sets[k]["reasons"]}
                 for k in measured},
        "roughness": movement,
        "alpha": options.alpha,
        "permutation_seed": options.seed,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = build_parser().parse_args(argv)
    try:
        if options.replay:
            return replay(options.permutations, options.seed, options.alpha, options.output)
        if not (options.arm and options.baseline and options.ground_truth):
            raise JudgeError("--arm, --baseline and --ground-truth are required without --replay")
        report = run(options)
    except JudgeError as error:
        print("REFUSED: {}".format(error), file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True)
    if options.output:
        path = pathlib.Path(options.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
