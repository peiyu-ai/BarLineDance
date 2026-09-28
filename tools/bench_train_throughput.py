#!/usr/bin/env python3
"""Price a training configuration before booking cards for it.

This repository has twice paid for a stage whose only symptom was that it was
slow, and once for 26 card-hours spent on the wrong segmentation.  Both are the
same failure: a configuration was committed to without anyone measuring what it
costs per step.  This tool runs the *real* ``train_atomic.py`` loop -- the same
release validation, the same prototype library, the same loader -- for a bounded
number of steps per configuration, and reports the steady-state rate each one
reaches together with what a full run would therefore cost.

It measures throughput and nothing else.  A configuration that is fast and wrong
looks excellent here, so the numbers only answer "how long", never "which arm".

Two costs are reported separately because they behave differently:

* ``startup_s`` -- release hash validation plus, for the completion stage, the
  prototype library build.  Paid once per launch, and *once* even for a
  multi-GPU run, because the ranks are forked after the library exists.
* ``steady_step_per_s`` -- the rate after the first step, so the loader's worker
  fork is not billed to it.

Example -- what the wild 340-frame completion stage costs on one card, and on
four with the single-GPU recipe preserved::

    python3 tools/bench_train_throughput.py \\
        --data-root /dev/shm/atomicdance-acct/release_w2_340 \\
        --stage completion --num-classes 4159 --seq-len 340 --steps 60 \\
        --arm one_card:gpus=0,batch=64 \\
        --arm four_cards_same_recipe:gpus=0,2,3,4,global_batch=64 \\
        --arm four_cards_bf16:gpus=0,2,3,4,global_batch=64,amp=bf16
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
THROUGHPUT_LINE = re.compile(r"^throughput=(\{.*\})\s*$", re.MULTILINE)


def parse_arm(specification):
    """``name:key=value,key=value`` -> (name, {key: value})."""
    if ":" not in specification:
        raise ValueError("arm {!r} must be name:key=value[,key=value...]".format(specification))
    name, _, body = specification.partition(":")
    settings = {}
    for piece in body.split(","):
        piece = piece.strip()
        if not piece:
            continue
        key, _, value = piece.partition("=")
        key = key.strip()
        if not value:
            raise ValueError("arm {!r} setting {!r} has no value".format(name, key))
        settings[key] = value.strip()
    unknown = set(settings) - {"gpus", "batch", "global_batch", "amp", "workers", "tf32"}
    if unknown:
        raise ValueError("arm {!r} has unknown settings: {}".format(name, sorted(unknown)))
    if "batch" in settings and "global_batch" in settings:
        raise ValueError(
            "arm {!r} declares both batch and global_batch; they mean different "
            "recipes and train_atomic.py refuses the pair".format(name)
        )
    if "gpus" in settings:
        # ``0;2;3`` as well as ``0,2,3``: the comma is already the arm separator.
        settings["gpus"] = settings["gpus"].replace(";", ",")
    return name, settings


def build_command(cli, settings, output_dir):
    command = [
        sys.executable, str(REPOSITORY_ROOT / "train_atomic.py"),
        "--stage", cli.stage,
        "--data-root", cli.data_root,
        "--output-dir", str(output_dir),
        "--num-classes", str(cli.num_classes),
        "--seq-len", str(cli.seq_len),
        "--seed", str(cli.seed),
        "--epochs", str(cli.epochs),
        "--max-steps", str(cli.steps),
        "--workers", settings.get("workers", str(cli.workers)),
    ]
    if "global_batch" in settings:
        command += ["--global-batch-size", settings["global_batch"]]
    else:
        command += ["--batch-size", settings.get("batch", str(cli.batch_size))]
    if settings.get("gpus"):
        command += ["--gpus", settings["gpus"]]
    if settings.get("amp", "off") != "off":
        command += ["--amp", settings["amp"]]
    if settings.get("tf32", "0") not in ("0", "false", "False"):
        command += ["--tf32"]
    return command


def run_arm(cli, name, settings, output_root):
    output_dir = output_root / name
    command = build_command(cli, settings, output_dir)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(REPOSITORY_ROOT), capture_output=True, text=True)
    wall = time.perf_counter() - started
    record = {"arm": name, "settings": settings, "wall_seconds": round(wall, 1),
              "command": " ".join(command)}
    if completed.returncode != 0:
        # A configuration that cannot run is a result, not an omission: report it
        # in the table rather than dropping the row.
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-4:]
        record["failed"] = True
        record["reason"] = " | ".join(tail)
        return record
    match = None
    for match in THROUGHPUT_LINE.finditer(completed.stdout):
        pass
    if match is None:
        record["failed"] = True
        record["reason"] = "no throughput line; the run produced no timed steps"
        return record
    throughput = json.loads(match.group(1))
    record.update(throughput)
    record["failed"] = False
    record["startup_s"] = round(wall - throughput["wall_seconds"], 1)
    if cli.full_run_steps and throughput.get("steady_step_per_s"):
        record["full_run_hours_at_{}_steps".format(cli.full_run_steps)] = round(
            cli.full_run_steps / throughput["steady_step_per_s"] / 3600.0, 2
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--stage", choices=("planner", "completion"), required=True)
    parser.add_argument("--arm", action="append", required=True,
                        help="name:gpus=0;2,global_batch=64,amp=bf16 (repeatable)")
    parser.add_argument("--steps", type=int, default=60, help="timed steps per arm")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64, help="per-rank default")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=4159)
    parser.add_argument("--seq-len", type=int, default=340)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output-root", required=True,
                        help="scratch directory for the bounded runs' checkpoints")
    parser.add_argument("--full-run-steps", type=int, default=None,
                        help="if given, extrapolate each arm to a run of this many optimizer steps")
    parser.add_argument("--out", default="")
    cli = parser.parse_args()

    arms = [parse_arm(specification) for specification in cli.arm]
    names = [name for name, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit("arm names must be unique; got {}".format(names))
    output_root = Path(cli.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for name, settings in arms:
        print("== {} ({})".format(name, settings), flush=True)
        record = run_arm(cli, name, settings, output_root)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    baseline = next((r for r in records if not r["failed"]), None)
    for record in records:
        if not record["failed"] and baseline and baseline.get("steady_step_per_s"):
            record["speedup_vs_first_arm"] = round(
                record["steady_step_per_s"] / baseline["steady_step_per_s"], 3)

    print("\n{:<28} {:>10} {:>12} {:>14} {:>10}".format(
        "arm", "step/s", "windows/s", "startup_s", "speedup"))
    for record in records:
        if record["failed"]:
            print("{:<28} {:>10} {:>12} {:>14} {:>10}  {}".format(
                record["arm"], "-", "-", "-", "-", record["reason"][:70]))
            continue
        print("{:<28} {:>10} {:>12} {:>14} {:>10}".format(
            record["arm"],
            record.get("steady_step_per_s", "-"),
            record.get("global_windows_per_s", "-"),
            record.get("startup_s", "-"),
            record.get("speedup_vs_first_arm", "-")))

    if cli.out:
        Path(cli.out).write_text(json.dumps(
            {"stage": cli.stage, "data_root": cli.data_root, "steps": cli.steps,
             "arms": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
