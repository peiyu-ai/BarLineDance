#!/usr/bin/env python3
"""Is the checkpoint pick a property of the checkpoint, or of one noise draw?

The planner is stochastic by default (``deterministic_planner: false`` in the
shipping manifest), so every column in ``tools/exp_plannerckpt_select.py`` is
read off ONE draw of 20 clips.  A criterion whose winner changes when the seed
changes has not selected a checkpoint, it has selected a draw -- and the
repository has already paid for a metric read off a single draw
(``docs/T_SERIES_PLAN.md`` section T4: ``sample_nonzero_accuracy`` picked epoch
250 and the reading was judged noise).

Two things are reported, and they answer different questions:

  gate/rank stability   the same selection report re-run under N seeds, joined
                        by checkpoint: does the gate verdict flip, does the
                        js_train ordering flip, does the WINNER change.  This is
                        the one that decides whether the pick may be quoted.
  reseed_disagree       per-frame disagreement between the plans one checkpoint
                        produced for the same clip under two seeds, computed
                        from the saved ``.npz`` plans rather than by resampling.
                        This is the scale the music-swap control is read
                        against: a swap disagreement below the reseed
                        disagreement means changing the music moves the plan
                        LESS than changing the noise does.

  python3 tools/exp_plannerckpt_stability.py \\
      --report seed20260902=runs/opt_plannerckpt/selection.json \\
      --report seed20260904=runs/opt_plannerckpt/selection_s20260904.json \\
      --plans /cache/atomicdance-tmp/opt/plannerckpt/plans \\
      --out runs/opt_plannerckpt/stability.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def load_report(path):
    report = json.loads(pathlib.Path(path).read_text())
    if report.get("refused"):
        raise SystemExit("error: {} is a REFUSAL, not a ranking: {}".format(
            path, report.get("reason")))
    rows = {row["checkpoint"]: row for row in report["rows"]}
    verdicts = {v["checkpoint"]: v for v in report["decision"]["verdicts"]}
    return {"rows": rows, "verdicts": verdicts,
            "winner": report["decision"]["winner"], "seed": report.get("seed")}


def rank_of(rows, verdicts, objective="js_train"):
    """Rank among GATE SURVIVORS only.  A failing checkpoint has no rank.

    Ranking the failures too would invite reading "it came 4th" as a near miss,
    when the gate is the whole point of the rule.
    """
    survivors = [name for name, v in verdicts.items() if v["passes_gate"]]
    ordered = sorted(survivors, key=lambda n: rows[n][objective])
    return {name: index + 1 for index, name in enumerate(ordered)}


def compare(reports, objective="js_train"):
    names = sorted(set.intersection(*[set(r["rows"]) for r in reports.values()]))
    ranks = {label: rank_of(r["rows"], r["verdicts"], objective)
             for label, r in reports.items()}
    table = []
    for name in names:
        row = {"checkpoint": name}
        for label, report in reports.items():
            row["{}_gate".format(label)] = report["verdicts"][name]["passes_gate"]
            row["{}_{}".format(label, objective)] = report["rows"][name][objective]
            row["{}_rank".format(label)] = ranks[label].get(name)
        gates = {row["{}_gate".format(label)] for label in reports}
        values = [row["{}_{}".format(label, objective)] for label in reports]
        row["gate_stable"] = len(gates) == 1
        row["{}_spread".format(objective)] = float(max(values) - min(values))
        row["rank_values"] = [row["{}_rank".format(label)] for label in reports]
        row["rank_stable"] = len(set(row["rank_values"])) == 1
        table.append(row)
    winners = {label: r["winner"] for label, r in reports.items()}
    return {
        "objective": objective,
        "seeds": {label: r["seed"] for label, r in reports.items()},
        "winners": winners,
        "winner_stable": len(set(winners.values())) == 1,
        "gate_flips": [r["checkpoint"] for r in table if not r["gate_stable"]],
        "rank_flips": [r["checkpoint"] for r in table if not r["rank_stable"]],
        "rows": table,
    }


def plan_files(directory):
    """{checkpoint: {seed: path}} from ``<name>.pt.seed<N>.npz`` filenames."""
    out = {}
    for path in sorted(pathlib.Path(directory).glob("*.npz")):
        match = re.match(r"^(?P<ckpt>.+)\.seed(?P<seed>\d+)\.npz$", path.name)
        if not match:
            continue
        out.setdefault(match.group("ckpt"), {})[int(match.group("seed"))] = path
    return out


def reseed_disagreement(paths_by_seed):
    """Mean per-frame disagreement between two seeds' plans, over all clips."""
    pairs = []
    for left, right in itertools.combinations(sorted(paths_by_seed), 2):
        a, b = np.load(paths_by_seed[left]), np.load(paths_by_seed[right])
        shared = sorted(set(a.files) & set(b.files))
        if not shared:
            continue
        per_clip = []
        for clip in shared:
            x, y = a[clip].ravel(), b[clip].ravel()
            n = min(len(x), len(y))
            per_clip.append(1.0 - float((x[:n] == y[:n]).mean()))
        pairs.append({"seeds": [left, right], "clips": len(shared),
                      "reseed_disagree": float(np.mean(per_clip)),
                      "per_clip_min": float(np.min(per_clip)),
                      "per_clip_max": float(np.max(per_clip))})
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="append", required=True,
                        metavar="LABEL=PATH")
    parser.add_argument("--plans", default=None,
                        help="directory of saved per-seed plan .npz files")
    parser.add_argument("--objective", default="js_train")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    reports = {}
    for entry in args.report:
        label, _, path = entry.partition("=")
        reports[label] = load_report(path)
    if len(reports) < 2:
        raise SystemExit("error: stability needs at least two reports; got one.  "
                         "A single draw cannot show that a pick is stable.")

    result = compare(reports, args.objective)
    if args.plans:
        result["reseed"] = {ckpt: reseed_disagreement(seeds)
                            for ckpt, seeds in plan_files(args.plans).items()
                            if len(seeds) > 1}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")

    labels = list(reports)
    header = "%-28s %s" % ("checkpoint", "  ".join(
        "%-8s" % label[:8] for label in labels))
    print(header)
    print("-" * len(header))
    for row in result["rows"]:
        cells = []
        for label in labels:
            cells.append("%.4f%s%s" % (
                row["{}_{}".format(label, args.objective)],
                "P" if row["{}_gate".format(label)] else "f",
                "#{}".format(row["{}_rank".format(label)])
                if row["{}_rank".format(label)] else "  "))
        print("%-28s %s" % (row["checkpoint"], "  ".join(cells)))
    print()
    print("winners per seed: {}".format(result["winners"]))
    print("winner stable: {}   gate flips: {}   rank flips: {}".format(
        result["winner_stable"], result["gate_flips"] or "none",
        result["rank_flips"] or "none"))
    print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
