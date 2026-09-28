#!/usr/bin/env python3
"""Pick the planner checkpoint by what the operator SEES, not by val loss.

WHY THIS EXISTS.  The shipped planner (epoch 6) was chosen by validation loss.
``docs/DANCE_QUALITY_DEFECTS.md`` section 18.2 records what that bought: epoch 6
is the most collapsed checkpoint in the whole sweep -- raw-draw top class 30.3%
against a training marginal of 7.9% -- and its plans hold one atomic movement
for 20.4 s (p90 15.4 s) where ground truth holds 3.4 s (p90 4.4 s).  Epoch 12,
which fixes both columns at zero training cost, was found by an operator looking
at rendered strips.  Val loss is mode-seeking on a categorical target: guessing
the mode lowers it, so it cannot see "is the distribution right" and it cannot
see "how long is one pose held".  It is the ``boundary contrast`` shape from
CLAUDE.md section 2.1 -- able to fail, pointed the wrong way.

This tool builds the criterion that WOULD have found epoch 12, applies it to
every checkpoint, and reports the ranking.  It ships nothing and trains nothing.

WHAT IT MEASURES, per checkpoint, on the 20 held-out clips:

  js_train / js_gt   Jensen-Shannon divergence (base 2, so bounded in [0, 1])
                     between the pooled class marginal of the FINAL plan --
                     the labels that actually reach retrieval, not the raw draw
                     -- and (a) the training marginal, (b) the ground-truth test
                     marginal.  ``js_train`` is the objective; ``js_gt`` is
                     reported beside it and is deliberately NOT the objective,
                     because the 20 test clips are the thing being judged and
                     selecting on their own marginal is selecting on the answer.
  plan shape         filler share, segments, longest-hold median/p90/max, from
                     ``tools.score_plan_shape`` -- the same function, and the
                     same ground-truth reference read through the label
                     manifest, never a directory glob (CLAUDE.md section 1.1).
  novelty            1 - mean pairwise cosine similarity of the 20 per-clip
                     label histograms.  Ground truth's own reading is printed
                     beside it, because "more novel" is not automatically
                     better: a planner drawing uniformly at random would win it.
  val columns        val loss, denoising accuracy and sample_nonzero_accuracy,
                     carried over from ``runs/txy_t_planner_sweep_short.json``
                     so the old criterion and the new one sit in one table.

THE CONTROL COLUMN, and why it is not optional.  Every column above is a
statement about a DISTRIBUTION, and a planner that ignored music entirely and
sampled from the training marginal would score perfectly on all of them.  So
two controls are reported beside them:

  music_swap_disagree   the same clip planned twice from the SAME per-clip noise
                        seed, once with its own music and once with another
                        clip's music cut to the same length.  Same noise, same
                        window count, only the conditioning differs -- so a
                        planner that ignores music returns byte-identical plans
                        and this column reads exactly 0.000.  It cannot read 0
                        by accident.
  reseed_disagree       the same clip, its OWN music, a different noise seed.
                        This is the scale: how much of the plan is noise rather
                        than conditioning.  ``music_swap_disagree`` is only
                        meaningful read against it.
  gt_frame_agree        per-frame agreement between the plan and this clip's
                        ground-truth labels, printed beside
                        ``gt_frame_agree_wrongclip`` -- the SAME plan scored
                        against a DIFFERENT clip's ground truth.  Without that
                        second number the first one has no zero point, and
                        CLAUDE.md section 2.1 rule 3 forbids reporting a null
                        from an instrument whose power was never shown.

THE RULE.  Two stages, gate first:

  1. every plan-shape column must sit inside a band around ground truth
     (``--band-*`` below; the defaults are stated in ``default_gates``);
  2. among the survivors, minimise ``js_train``.

The bands are INVENTED HERE and are labelled as such (CLAUDE.md section 2.1
rule 1).  They are not free parameters, though: two checkpoints already carry an
operator verdict, so the band is required to reproduce it -- epoch 6, which the
operator rejected, must FAIL, and epoch 12, which the operator chose by eye,
must PASS.  ``tests/test_exp_plannerckpt_select.py`` pins both directions.

The rule REFUSES rather than guesses: a checkpoint missing any gate column or
the objective makes ``select`` raise, and the CLI exits non-zero naming the
checkpoint and the column.  A criterion that silently skips a checkpoint it
could not measure is the "gate that never fires" this repository has paid for
before (``tools/check_disk_headroom.py`` docstring).

INSTRUMENT.  Plans are produced by calling ``infer_atomic.infer_plan`` itself
with the shipping flags and the shipping per-clip seed ``sample_seed(seed,
name)`` -- not a reimplementation of it, and not the once-per-run seeding that
``tools/probe_plan_collapse.py`` uses, which draws different noise than a real
run does.  ``--verify-against`` re-derives an existing arm's plans and demands
they match frame for frame before any column is read off them.

  python3 tools/exp_plannerckpt_select.py \\
      --checkpoint-dir runs/planner_txy_t_short_s20260902 \\
      --clips runs/eval_clips_txy_t20.txt \\
      --audio-dir runs/txy_t_gt_eval/audio \\
      --labels-root data/wild3d/txy_t_labels \\
      --train-labels /cache/atomicdance-assets/scratch/txy_t/release_v3/train \\
      --val-sweep runs/txy_t_planner_sweep_short.json \\
      --verify-against runs/txy_t_m6_ep12:planner_epoch12_step1008.pt \\
      --out runs/opt_plannerckpt/selection.json
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import infer_atomic  # noqa: E402
from infer_atomic import (_load_checkpoint, _load_music, infer_plan,  # noqa: E402
                          resolve_device, sample_seed, seed_everything)
from tools.score_plan_shape import ground_truth_labels  # noqa: E402
from tools.score_plan_shape import shape_of as plan_shape_of  # noqa: E402
from tools.score_plan_shape import summarise as summarise_shape  # noqa: E402

NUM_CLASSES = 20
FPS = 30.0

# Copied from runs/txy_t_m6_ep12/manifest.json["sampling"].  Named here rather
# than defaulted in the signature so a reader can diff this block against the
# manifest without reading argparse.
SHIPPING_PLAN_FLAGS = {
    "deterministic": False,
    "temperature": 1.0,
    "vote_window": 5,
    "min_segment_length": 6,
    "plan_stride": 15,
    "plan_fusion": "vote",
    "plan_vote_tie_break": "centre",
    "plan_transition_policy": "protect",
    "plan_merge_order": "shortest",
    "planner_guidance_weight": 1.0,
    "planner_transition_logit_bias": 0.0,
    "plan_bar_grid": True,
    "plan_bar_beats": 4,
}


# --------------------------------------------------------------------------
# columns
# --------------------------------------------------------------------------
def class_histogram(labels, num_classes=NUM_CLASSES):
    """Counts over classes 1..num_classes.  Label 0 (filler) is excluded.

    Filler is excluded because it is already its own column: leaving it in
    would let a plan that is 90% filler look like it matched the marginal.
    """
    values = np.asarray(labels).ravel()
    counts = np.zeros(num_classes, dtype=np.float64)
    for value in values[(values >= 1) & (values <= num_classes)]:
        counts[int(value) - 1] += 1.0
    return counts


def _normalise(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("cannot normalise an all-zero histogram")
    return counts / total


def jensen_shannon(p_counts, q_counts):
    """JS divergence in BITS, so identical inputs give 0 and disjoint give 1.

    Base 2 is chosen so the column has a fixed upper bound and "how far along
    the way to disjoint" is readable without a second reference number.
    """
    p = _normalise(p_counts)
    q = _normalise(q_counts)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cross_clip_novelty(histograms):
    """1 - mean pairwise cosine similarity of per-clip label histograms.

    High means the 20 plans use different movements from each other.  It is NOT
    a quality column on its own -- a uniform random planner maximises it -- so
    ground truth's own reading is always reported next to it.
    """
    rows = [np.asarray(h, dtype=np.float64) for h in histograms]
    rows = [r for r in rows if r.sum() > 0]
    if len(rows) < 2:
        return None
    unit = [r / np.linalg.norm(r) for r in rows]
    sims = [float(unit[i] @ unit[j])
            for i in range(len(unit)) for j in range(i + 1, len(unit))]
    return 1.0 - float(np.mean(sims))


def frame_agreement(a, b):
    """Fraction of frames on which two label sequences agree, over min length."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    n = min(len(a), len(b))
    if n == 0:
        return None
    return float((a[:n] == b[:n]).mean())


# --------------------------------------------------------------------------
# plan sampling -- the shipping call, the shipping seed
# --------------------------------------------------------------------------
def load_planner(checkpoint, device):
    planner, planner_args = _load_checkpoint(checkpoint, "planner", device)
    return planner, planner_args


@torch.no_grad()
def plan_one(planner, planner_args, music, device, clip_seed, flags=None):
    """One clip's final plan, through ``infer_atomic.infer_plan`` itself.

    ``available_labels`` is left None on purpose: the shipping ep12 arm records
    ``frames_rewritten_to_transition: 0``, so the library-availability rewrite
    was a no-op there, and passing a library here would make this tool depend on
    a retrieval index that has nothing to do with checkpoint choice.  The
    ``--verify-against`` check is what proves the omission changed nothing.
    """
    seed_everything(clip_seed)
    settings = dict(SHIPPING_PLAN_FLAGS)
    settings.update(flags or {})
    return infer_plan(planner, music, planner_args.seq_len, device, **settings).cpu().numpy()


def clip_music(audio_dir, clip, music_dim):
    return _load_music(pathlib.Path(audio_dir) / (clip + ".npy"), None, music_dim)


def fit_length(music, length):
    """Another clip's music, made exactly ``length`` frames long.

    Truncation where possible; wrap-around only when the donor is shorter, which
    is why the donor is chosen as the LONGEST other clip -- 19 of 20 clips then
    only truncate.  The count of wrapped frames is reported per clip so a reader
    can see how much of the control input is a repeat.
    """
    music = torch.as_tensor(music)
    if len(music) >= length:
        return music[:length], 0
    repeats = int(np.ceil(length / len(music)))
    tiled = torch.cat([music] * repeats, dim=0)[:length]
    return tiled, int(length - len(music))


def swap_partners(clips, lengths):
    """Each clip's music donor: the LONGEST other clip.  Deterministic."""
    order = sorted(clips, key=lambda c: (-lengths[c], c))
    longest, second = order[0], order[1]
    return {c: (second if c == longest else longest) for c in clips}


# --------------------------------------------------------------------------
# one checkpoint
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_checkpoint(checkpoint, clips, audio_dir, device, *, seed,
                        reseed, gt_labels, train_counts, gt_counts,
                        hold_threshold=4.0, controls="full", flags=None,
                        keep_plans=False):
    """One checkpoint's row.

    ``controls`` costs sampling passes over the 20 clips and is therefore
    selectable: ``none`` is one pass (the plan itself), ``swap`` is two (adds
    the music-swap control), ``full`` is three (adds the reseed control).  The
    reseed control can also be recovered afterwards from saved plans of two
    seeds, which is how the seed-stability run pays for it.
    """
    want_swap = controls in ("swap", "full")
    want_reseed = controls == "full"
    planner, planner_args = load_planner(checkpoint, device)
    musics = {c: clip_music(audio_dir, c, planner_args.music_dim) for c in clips}
    lengths = {c: len(musics[c]) for c in clips}
    donors = swap_partners(clips, lengths)

    plans, hists, shapes = {}, [], []
    wrapped_frames = 0
    swap_rows, reseed_rows, gt_rows, gt_wrong_rows = [], [], [], []
    rotation = {c: clips[(index + 1) % len(clips)] for index, c in enumerate(clips)}

    for clip in clips:
        clip_seed = sample_seed(seed, clip)
        labels = plan_one(planner, planner_args, musics[clip], device, clip_seed, flags)
        plans[clip] = labels
        hists.append(class_histogram(labels))
        shapes.append(plan_shape_of(labels))

        if want_swap:
            donor_music, wrapped = fit_length(musics[donors[clip]], lengths[clip])
            wrapped_frames += wrapped
            swapped = plan_one(planner, planner_args, donor_music, device, clip_seed, flags)
            swap_rows.append(1.0 - frame_agreement(labels, swapped))
        if want_reseed:
            other_seed = sample_seed(reseed, clip)
            again = plan_one(planner, planner_args, musics[clip], device, other_seed, flags)
            reseed_rows.append(1.0 - frame_agreement(labels, again))

        if clip in gt_labels:
            gt_rows.append(frame_agreement(labels, gt_labels[clip]))
            partner = rotation[clip]
            if partner in gt_labels:
                gt_wrong_rows.append(frame_agreement(labels, gt_labels[partner]))

    pooled = np.sum(hists, axis=0)
    shape = summarise_shape([s for s in shapes if s], hold_threshold)
    used = int((pooled > 0).sum())
    share = np.sort(_normalise(pooled))[::-1]

    row = {
        "checkpoint": pathlib.Path(checkpoint).name,
        "clips": len(clips),
        "js_train": jensen_shannon(pooled, train_counts),
        "js_gt": jensen_shannon(pooled, gt_counts),
        "classes_used": used,
        "top_share": float(share[0]),
        "top3_share": float(share[:3].sum()),
        "danced_frames": int(pooled.sum()),
        "novelty": cross_clip_novelty(hists),
        "class_counts": {str(i + 1): int(v) for i, v in enumerate(pooled)},
    }
    row.update({key: shape[key] for key in
                ("filler_share", "segments", "median_segment_s",
                 "longest_hold_median_s", "longest_hold_p90_s",
                 "longest_hold_max_s", "clips_over_threshold")})
    if swap_rows:
        row["music_swap_disagree"] = float(np.mean(swap_rows))
        row["swap_wrapped_frames"] = wrapped_frames
        row["swap_donors"] = donors
    if reseed_rows:
        row["reseed_disagree"] = float(np.mean(reseed_rows))
    if gt_rows:
        row["gt_frame_agree"] = float(np.mean(gt_rows))
    if gt_wrong_rows:
        row["gt_frame_agree_wrongclip"] = float(np.mean(gt_wrong_rows))

    del planner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, (plans if keep_plans else None)


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------
def default_gates(gt_shape):
    """The band around ground truth.  INVENTED HERE -- see the module docstring.

    Each entry is (column, low, high, why).  ``None`` is "no bound on that side".
    """
    return [
        ("filler_share", gt_shape["filler_share"] - 0.10,
         gt_shape["filler_share"] + 0.10,
         "the operator's 'no breathing room': ground truth spends 30.4% of "
         "frames on no named movement, the ep6 arm 0.4%"),
        ("longest_hold_median_s", 0.5 * gt_shape["longest_hold_median_s"],
         1.5 * gt_shape["longest_hold_median_s"],
         "the operator's 招单一 (one move held far too long), typical clip"),
        ("longest_hold_p90_s", None, 2.0 * gt_shape["longest_hold_p90_s"],
         "招单一 in the tail: ep6 reads 15.4 s against ground truth 4.4 s"),
        ("longest_hold_max_s", None, 2.0 * gt_shape["longest_hold_max_s"],
         "招单一 at its worst clip: ep6 reads 20.4 s against 9.2 s"),
        ("segments", 0.5 * gt_shape["segments"], 2.0 * gt_shape["segments"],
         "guards the opposite failure -- a plan chopped into half-second "
         "pieces would win the hold columns and lose the dance"),
    ]


class Refusal(RuntimeError):
    """A checkpoint could not be judged.  Never downgraded to a skip."""


def select(rows, gates, objective="js_train"):
    """Gate on plan shape, then minimise ``objective``.  Refuse on a hole.

    Raises ``Refusal`` naming the checkpoint and the column when any row lacks a
    gate column or the objective: a checkpoint that could not be measured must
    not be quietly dropped from the ranking, because dropping it looks exactly
    like losing.
    """
    verdicts = []
    for row in rows:
        name = row.get("checkpoint", "<unnamed>")
        for column, _, _, _ in gates:
            if row.get(column) is None:
                raise Refusal(
                    "refusing to rank {!r}: gate column {!r} is missing.  A "
                    "checkpoint that was not measured cannot be ranked."
                    .format(name, column))
        if row.get(objective) is None:
            raise Refusal(
                "refusing to rank {!r}: objective {!r} is missing."
                .format(name, objective))
        failures = []
        for column, low, high, why in gates:
            value = float(row[column])
            if low is not None and value < low:
                failures.append({"column": column, "value": value,
                                 "bound": "low", "limit": low, "why": why})
            if high is not None and value > high:
                failures.append({"column": column, "value": value,
                                 "bound": "high", "limit": high, "why": why})
        verdicts.append({"checkpoint": name, "passes_gate": not failures,
                         "gate_failures": failures,
                         objective: float(row[objective])})
    survivors = [v for v in verdicts if v["passes_gate"]]
    winner = min(survivors, key=lambda v: v[objective]) if survivors else None
    return {
        "objective": objective,
        "gates": [{"column": c, "low": low, "high": high, "why": why}
                  for c, low, high, why in gates],
        "verdicts": verdicts,
        "survivors": [v["checkpoint"] for v in survivors],
        "winner": winner["checkpoint"] if winner else None,
        "winner_objective": winner[objective] if winner else None,
        "refused": False,
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
COLUMNS = [
    ("checkpoint", "%-28s", "%-28s"),
    ("js_train", "%9s", "%9.4f"),
    ("js_gt", "%8s", "%8.4f"),
    ("filler_share", "%8s", "%8.4f"),
    ("segments", "%6s", "%6.1f"),
    ("longest_hold_median_s", "%8s", "%8.2f"),
    ("longest_hold_p90_s", "%8s", "%8.2f"),
    ("longest_hold_max_s", "%8s", "%8.2f"),
    ("classes_used", "%6s", "%6d"),
    ("top_share", "%8s", "%8.4f"),
    ("novelty", "%8s", "%8.4f"),
    ("music_swap_disagree", "%8s", "%8.4f"),
    ("reseed_disagree", "%8s", "%8.4f"),
    ("gt_frame_agree", "%8s", "%8.4f"),
    ("gt_frame_agree_wrongclip", "%9s", "%9.4f"),
    ("val_loss", "%8s", "%8.4f"),
    ("sample_nonzero_accuracy", "%8s", "%8.4f"),
]
SHORT = {"longest_hold_median_s": "hold_med", "longest_hold_p90_s": "hold_p90",
         "longest_hold_max_s": "hold_max", "filler_share": "filler",
         "classes_used": "nclass", "music_swap_disagree": "swapdis",
         "reseed_disagree": "seeddis", "gt_frame_agree": "gt_agr",
         "gt_frame_agree_wrongclip": "gt_wrong",
         "sample_nonzero_accuracy": "samp_nz", "segments": "segs"}


def render_table(rows, gate_lookup=None):
    header = " ".join(fmt % SHORT.get(name, name) for name, fmt, _ in COLUMNS)
    lines = [header + "  gate", "-" * (len(header) + 6)]
    for row in rows:
        cells = []
        for name, head_fmt, cell_fmt in COLUMNS:
            value = row.get(name)
            cells.append(head_fmt % "-" if value is None else cell_fmt % value)
        mark = ""
        if gate_lookup is not None:
            mark = "PASS" if gate_lookup.get(row["checkpoint"]) else "fail"
        lines.append(" ".join(cells) + "  " + mark)
    return "\n".join(lines)


def markdown_table(rows, gate_lookup, winner):
    names = [name for name, _, _ in COLUMNS]
    head = "| " + " | ".join([SHORT.get(n, n) for n in names] + ["gate"]) + " |"
    rule = "|" + "|".join(["---"] * (len(names) + 1)) + "|"
    out = [head, rule]
    for row in rows:
        cells = []
        for name in names:
            value = row.get(name)
            if value is None:
                cells.append("-")
            elif isinstance(value, str):
                cells.append("**{}**".format(value)
                             if value == winner else value)
            elif isinstance(value, int):
                cells.append(str(value))
            else:
                cells.append("{:.4f}".format(value))
        cells.append("PASS" if gate_lookup.get(row["checkpoint"]) else "fail")
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def train_marginal(train_root, num_classes=NUM_CLASSES):
    """Class counts over the training split's VALID frames.

    Read from the release arrays rather than recomputed from the label
    manifest: this is the distribution the planner was actually trained on,
    window for window.
    """
    root = pathlib.Path(train_root)
    labels = np.load(root / "labels.npy")
    mask_path = root / "label_valid_mask.npy"
    values = labels[np.load(mask_path)] if mask_path.exists() else labels.ravel()
    return class_histogram(values, num_classes)


def gt_marginal(gt_labels, num_classes=NUM_CLASSES):
    counts = np.zeros(num_classes, dtype=np.float64)
    for labels in gt_labels.values():
        counts += class_histogram(labels, num_classes)
    return counts


def val_sweep_columns(path):
    if not path:
        return {}
    report = json.loads(pathlib.Path(path).read_text())
    return {row["checkpoint"]: {"val_loss": row.get("loss"),
                                "denoising_accuracy": row.get("denoising_accuracy"),
                                "sample_nonzero_accuracy": row.get("sample_nonzero_accuracy"),
                                "epoch": row.get("epoch")}
            for row in report.get("rows", [])}


def verify_against(arm_dir, plans):
    """Frame-for-frame check of re-derived plans against a saved arm's plans."""
    directory = pathlib.Path(arm_dir)
    checked, mismatched = 0, []
    for clip, labels in plans.items():
        path = directory / (clip + ".pkl")
        if not path.exists():
            mismatched.append({"clip": clip, "reason": "no saved plan"})
            continue
        with path.open("rb") as handle:
            saved = np.asarray(pickle.load(handle)["atomic_labels"]).ravel()
        checked += 1
        mine = np.asarray(labels).ravel()
        if len(saved) != len(mine) or not bool((saved == mine).all()):
            differing = (int((saved[:min(len(saved), len(mine))]
                              != mine[:min(len(saved), len(mine))]).sum())
                         if len(saved) and len(mine) else None)
            mismatched.append({"clip": clip, "saved_frames": int(len(saved)),
                               "mine_frames": int(len(mine)),
                               "differing_frames": differing})
    return {"arm": str(arm_dir), "clips_checked": checked,
            "mismatched": mismatched, "exact": not mismatched}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint", action="append", default=None,
                        help="basename; repeatable.  Default: every *.pt in the dir")
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--labels-root", required=True,
                        help="the label release holding labels.jsonl (read as a "
                             "manifest, never globbed)")
    parser.add_argument("--train-labels", required=True,
                        help="release split dir holding labels.npy for TRAIN")
    parser.add_argument("--val-sweep", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--reseed", type=int, default=20260903,
                        help="second seed used only for the reseed control")
    parser.add_argument("--hold-threshold", type=float, default=4.0)
    parser.add_argument("--controls", default="full",
                        choices=("none", "swap", "full"),
                        help="none: 1 sampling pass per checkpoint.  swap: 2 "
                             "(adds the music-swap control).  full: 3 (adds the "
                             "reseed control).")
    parser.add_argument("--save-plans", type=pathlib.Path, default=None,
                        help="directory for one .npz of per-clip plans per "
                             "checkpoint, so the controls and the seed-stability "
                             "comparison are reproducible without resampling")
    parser.add_argument("--verify-against", default=None, metavar="DIR:CKPT",
                        help="re-derive this checkpoint's plans and demand they "
                             "match the saved arm frame for frame")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    torch.set_num_threads(2)
    clips = [line.strip() for line in args.clips.read_text().splitlines() if line.strip()]
    device = resolve_device(args.device)

    gt_labels = ground_truth_labels(args.labels_root, clips)
    missing = [c for c in clips if c not in gt_labels]
    if missing:
        raise SystemExit("error: {} clips have no ground-truth labels under {}: {}"
                         .format(len(missing), args.labels_root, missing[:3]))
    gt_shape = summarise_shape([plan_shape_of(gt_labels[c]) for c in clips],
                               args.hold_threshold)
    gt_hists = [class_histogram(gt_labels[c]) for c in clips]
    gt_shape["novelty"] = cross_clip_novelty(gt_hists)
    train_counts = train_marginal(args.train_labels)
    gt_counts = gt_marginal(gt_labels)

    directory = pathlib.Path(args.checkpoint_dir)
    names = args.checkpoint or sorted(p.name for p in directory.glob("*.pt"))
    val_columns = val_sweep_columns(args.val_sweep)

    verify_dir, verify_ckpt = (args.verify_against.rsplit(":", 1)
                               if args.verify_against else (None, None))

    if args.save_plans:
        args.save_plans.mkdir(parents=True, exist_ok=True)

    rows, verification = [], None
    for name in names:
        row, plans = evaluate_checkpoint(
            directory / name, clips, args.audio_dir, device, seed=args.seed,
            reseed=args.reseed, gt_labels=gt_labels, train_counts=train_counts,
            gt_counts=gt_counts, hold_threshold=args.hold_threshold,
            controls=args.controls, keep_plans=True)
        row.update(val_columns.get(name, {}))
        rows.append(row)
        if args.save_plans:
            np.savez_compressed(args.save_plans / "{}.seed{}.npz".format(name, args.seed),
                                **{c: np.asarray(v) for c, v in plans.items()})
        if name == verify_ckpt:
            verification = verify_against(verify_dir, plans)
        print(render_table([row]).splitlines()[-1], flush=True)

    if verify_ckpt and verification is None:
        raise SystemExit("error: --verify-against named {!r}, which was not "
                         "among the checkpoints evaluated".format(verify_ckpt))
    if verification is not None and not verification["exact"]:
        raise SystemExit(
            "error: the instrument does not reproduce {}: {} of {} clips differ.  "
            "Refusing to report columns read off a pipeline that is not the "
            "shipping one.  {}".format(verify_dir, len(verification["mismatched"]),
                                       verification["clips_checked"],
                                       verification["mismatched"][:2]))

    gates = default_gates(gt_shape)
    try:
        decision = select(rows, gates)
    except Refusal as refusal:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"schema_version": "atomicdance-plannerckpt-selection-v1",
             "refused": True, "reason": str(refusal), "rows": rows,
             "ground_truth": gt_shape}, indent=1, sort_keys=True) + "\n")
        print("REFUSED: {}".format(refusal), file=sys.stderr)
        return 2

    lookup = {v["checkpoint"]: v["passes_gate"] for v in decision["verdicts"]}
    report = {
        "schema_version": "atomicdance-plannerckpt-selection-v1",
        "checkpoint_dir": str(directory),
        "clips": clips,
        "seed": args.seed,
        "reseed": args.reseed,
        "controls": args.controls,
        "plan_flags": SHIPPING_PLAN_FLAGS,
        "ground_truth": gt_shape,
        "train_marginal": {str(i + 1): int(v) for i, v in enumerate(train_counts)},
        "gt_marginal": {str(i + 1): int(v) for i, v in enumerate(gt_counts)},
        "rows": rows,
        "decision": decision,
        "instrument_verification": verification,
        "how_to_read": (
            "The gate is the operator's four defects made numeric; js_train is the "
            "tie-break among survivors.  Read music_swap_disagree beside js_train: "
            "the gate and js columns are statements about a distribution and a "
            "planner that ignored music could satisfy them, and swapdis reads "
            "exactly 0.000 for such a planner because the noise draw is held "
            "fixed.  gt_frame_agree is meaningless without gt_frame_agree_wrongclip "
            "beside it, which is the same plan scored against another clip's "
            "ground truth."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    ordered = sorted(rows, key=lambda r: (not lookup[r["checkpoint"]], r["js_train"]))
    print()
    print(render_table(ordered, lookup))
    print()
    print("ground truth: filler {:.4f}  segs {:.1f}  hold med/p90/max "
          "{:.2f}/{:.2f}/{:.2f}  novelty {:.4f}".format(
              gt_shape["filler_share"], gt_shape["segments"],
              gt_shape["longest_hold_median_s"], gt_shape["longest_hold_p90_s"],
              gt_shape["longest_hold_max_s"], gt_shape["novelty"]))
    print("survivors: {}".format(", ".join(decision["survivors"]) or "NONE"))
    print("winner: {} (js_train {:.4f})".format(
        decision["winner"], decision["winner_objective"])
        if decision["winner"] else "winner: NONE -- every checkpoint failed the gate")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_table(ordered, lookup, decision["winner"]) + "\n")
    print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
