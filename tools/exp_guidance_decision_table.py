#!/usr/bin/env python3
"""Assemble one arm x column decision table from the sweep's per-tool JSONs.

MEASURES NOTHING ITSELF.  Every number here is read out of a JSON another tool
wrote, and the key it was read from is recorded in ``provenance`` so a reader
can go back to the file rather than trust this table.  It exists because the
guidance sweep is decided on SIX tools at once and CLAUDE.md 2.1 rule 4 asks
that disagreeing criteria be seen side by side rather than one at a time.

Arm names must match across the input JSONs; ``ground truth`` and ``DRAFT`` are
carried as reference rows, and the controls each tool emits are passed through
into ``controls`` instead of being dropped.
"""

import argparse
import json
import pathlib

COLUMNS = [
    ("energy_vs_gt", "energy/GT", "%.4f"),
    ("jitter", "jitter", "%.4f"),
    ("twist", "twist deg/s", "%.2f"),
    ("skate", "skate", "%.4f"),
    ("root_vs_gt", "root/GT", "%.4f"),
    ("still_window_share", "still-window share", "%.3f"),
    ("sustained_hold_share", "sust. hold share", "%.5f"),
    ("sustained_longest_hold_s", "longest hold s", "%.4f"),
    ("p90_over_p10", "p90/p10", "%.3f"),
    ("settle", "settle", "%+.4f"),
    ("within_clip_novelty", "within-clip nov.", "%.4f"),
    ("cross_clip_novelty", "cross-clip nov.", "%.4f"),
    ("seam_jerk_ratio", "seam jerk", "%.3f"),
]


def _load(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def build(root):
    root = pathlib.Path(root)
    arm_table = _load(root / "arm_table.json") or []
    stillness = _load(root / "stillness.json") or {}
    settle = _load(root / "settle.json") or []
    repetition = _load(root / "repetition.json") or {}
    cross = _load(root / "cross_clip.json") or {}
    seam = _load(root / "seam.json") or {}

    rows, prov = {}, {}

    def put(arm, col, value, source):
        rows.setdefault(arm, {})[col] = value
        prov.setdefault(col, source)

    for r in arm_table:
        arm = r["arm"]
        for k in ("energy_vs_gt", "jitter", "twist", "skate", "root_vs_gt",
                  "energy_ms", "lag0", "wristsync", "seg_per_s"):
            if k in r:
                put(arm, k, r[k], "arm_table.json[].%s" % k)

    for arm, r in (stillness.get("arms") or {}).items():
        put(arm, "still_window_share", r.get("window_share_pooled"),
            "stillness.json arms[].window_share_pooled")
        put(arm, "sustained_hold_share", r.get("clip_sustained_hold_share_median"),
            "stillness.json arms[].clip_sustained_hold_share_median")
        put(arm, "sustained_longest_hold_s", r.get("clip_sustained_longest_hold_s_median"),
            "stillness.json arms[].clip_sustained_longest_hold_s_median")
        put(arm, "p90_over_p10", r.get("clip_p90_over_p10_median"),
            "stillness.json arms[].clip_p90_over_p10_median")
    gt_still = stillness.get("ground_truth")
    if gt_still:
        for col, key in (("still_window_share", "window_share_pooled"),
                         ("sustained_hold_share", "clip_sustained_hold_share_median"),
                         ("sustained_longest_hold_s", "clip_sustained_longest_hold_s_median"),
                         ("p90_over_p10", "clip_p90_over_p10_median")):
            put("ground truth", col, gt_still.get(key),
                "stillness.json ground_truth.%s" % key)

    for r in settle:
        if r["arm"].strip().startswith("control:"):
            continue
        put(r["arm"], "settle", r.get("settle"), "settle.json[].settle")
        put(r["arm"], "settle_p", r.get("p"), "settle.json[].p")

    for arm, r in (repetition.get("arms") or {}).items():
        put(arm, "within_clip_novelty", r.get("novelty"), "repetition.json arms[].novelty")
    if repetition.get("ground_truth"):
        put("ground truth", "within_clip_novelty",
            repetition["ground_truth"].get("novelty"), "repetition.json ground_truth.novelty")

    for arm, r in (cross.get("arms") or {}).items():
        put(arm, "cross_clip_novelty", r.get("novelty_median"),
            "cross_clip.json arms[].novelty_median")
        put(arm, "cross_clip_novelty_shuffled_control", r.get("novelty_median_shuffled_owner"),
            "cross_clip.json arms[].novelty_median_shuffled_owner")

    for arm, r in (seam.get("arms") or {}).items():
        put(arm, "seam_jerk_ratio", r.get("jerk_ratio"), "seam.json arms[].jerk_ratio")
        put(arm, "seam_jerk_shuffled_null", r.get("jerk_ratio_shuffled_null"),
            "seam.json arms[].jerk_ratio_shuffled_null")

    controls = {
        "stillness": stillness.get("controls"),
        "stillness_gt_in_22_1_band": stillness.get("ground_truth_in_22_1_band"),
        # Which clips and windows the stillness criterion REFUSED, per arm and
        # by name (tools/stillness_criterion.py).  Carried here because the
        # ``still_window_share`` column below is a share over the MEASURABLE
        # windows only, and a reader who cannot see the denominator cannot
        # compare two arms that dropped different amounts -- the published
        # draft column 0.351 was a share over a denominator holding 199 windows
        # the criterion could not measure.
        "stillness_measurability": stillness.get("measurability"),
        "settle_gt_controls": [r for r in settle if r["arm"].strip().startswith("control:")],
        "cross_clip_shuffled_owner": {
            a: r.get("novelty_median_shuffled_owner")
            for a, r in (cross.get("arms") or {}).items()},
        "seam_shuffled_null": {
            a: r.get("jerk_ratio_shuffled_null")
            for a, r in (seam.get("arms") or {}).items()},
    }
    return {"arms": rows, "provenance": prov, "controls": controls}


ORDER = ["ground truth", "DRAFT", "DRAFT ep12", "g1.0", "g1.5",
         "g2.0 shipping", "OUTPUT ep12 shipping"]


def markdown(table):
    arms = sorted(table["arms"], key=lambda a: (ORDER.index(a) if a in ORDER else 99, a))
    head = "| arm | " + " | ".join(h for _, h, _ in COLUMNS) + " |"
    sep = "|---" * (len(COLUMNS) + 1) + "|"
    lines = [head, sep]
    for arm in arms:
        cells = []
        for key, _, fmt in COLUMNS:
            v = table["arms"][arm].get(key)
            cells.append("n/a" if v is None else (fmt % v))
        lines.append("| %s | %s |" % (arm, " | ".join(cells)))
    measurability = (table.get("controls") or {}).get("stillness_measurability") or {}
    notes = []
    for who, block in measurability.items():
        if isinstance(block, dict) and (block.get("clips_dropped")
                                        or block.get("windows_dropped")):
            notes.append("%s: stillness measured on %s clip(s) / %s window(s); "
                         "REFUSED %s clip(s) and %s window(s) as unmeasurable (%s)"
                         % (who, block.get("clips_measured"),
                            block.get("windows_measured"),
                            block.get("clips_dropped"), block.get("windows_dropped"),
                            ", ".join(sorted(block.get("dropped_clips") or {})) or "-"))
    if notes:
        lines.append("")
        lines.append("Stillness denominators are not equal across arms:")
        lines.extend("- " + n for n in notes)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="runs/opt_guidance")
    ap.add_argument("--out", default="runs/opt_guidance/decision_table.json")
    ap.add_argument("--markdown", default="runs/opt_guidance/decision_table.md")
    args = ap.parse_args()
    table = build(args.root)
    pathlib.Path(args.out).write_text(json.dumps(table, indent=1, sort_keys=True))
    md = markdown(table)
    pathlib.Path(args.markdown).write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
