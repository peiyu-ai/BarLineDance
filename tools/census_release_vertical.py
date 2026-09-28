"""Per-window, per-frame HOP and DEEP-SQUAT flags for every window of a release.

WHY IT HAS TO EXIST.  2026-09-17, twenty eval clips: fix5 is airborne on 3.45%
of frames against ground truth's 0.57% (higher on 15 of 17 clips that differ),
and its draft already reads 3.46% -- completion passes the hops through, so the
choice is made at retrieval.  The operator saw one on 7676475940934935409:clip001
(a one-knee hop at f300).  ``--draft-hop-guard`` and ``--draft-music-energy``
read the bits this writes.

THE DETECTORS, taken unchanged from the measurement that found the excess
(scratchpad music_energy/common.py, 2026-09-17), so the property selected on
and the property measured are one property:

  * HOP (bit 1): the lowest foot joint (SMPL 7, 8, 10, 11) more than 0.15 m above
    the running 5th percentile of the lowest foot over a centred 2 s window.
    Validated on the one hop seen on video (fires: feet 0.39 m up) and three
    negatives; a 0.05 m margin read reconstruction noise (ground truth 28%
    "airborne").  The 0.15 m and 2 s were chosen AFTER seeing that hop, so the
    validation is not independent of them.
  * DEEP SQUAT (bit 2): pelvis more than 0.15 m below the running 80th
    percentile of pelvis height over a centred 3 s window -- the local standing
    height, so a monocular reconstruction's slow depth drift moves with it.  It
    measures "pelvis low", not "lunge".

DECODED PER SEQUENCE, NOT PER WINDOW.  The release materialises 150-frame
windows at stride 15; the running references need context on both sides, and a
window cut in the middle of a crouch would otherwise call the crouch its own
standing height.  Each sequence is stitched back from its windows (overlaps are
checked to agree), decoded once through ``tools.render_dance_video._decode_raw_151``
with the release's [-1, 1] affine, flagged, and sliced back into its windows.

POSITIVE CONTROL: ``--reference-clips`` reads the same detectors off the staged
ground-truth pickles; the measurement read hop 0.0057 and squat 0.0395 on the
twenty eval clips.
"""
import argparse
import json
import pathlib
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.render_dance_video import _decode_raw_151     # noqa: E402

FPS = 30.0
FEET = (7, 8, 10, 11)
HOP_RISE = 0.15
HOP_WINDOW = int(2.0 * FPS)
SQUAT_DROP = 0.15
SQUAT_WINDOW = int(3.0 * FPS)
HOP_FLAG = 1
SQUAT_FLAG = 2


def rolling_percentile(x, window, q):
    """Centred running percentile, truncated at the ends (no padding)."""
    x = np.asarray(x, dtype=np.float64)
    half = window // 2
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = np.percentile(x[max(0, i - half):min(len(x), i + half + 1)], q)
    return out


def vertical_series(joints):
    """joints [T, 24, 3], z up, metres -> (uint8 flags [T], local floor [T] metres).

    The LOCAL FLOOR is the hop detector's own reference -- the running 5th
    percentile of the lowest foot over a centred 2 s window -- and it is written
    out because it is also what ``--draft-unit-floor`` levels each retrieved unit
    by.  The recording floor (``recording_floors.json``) is ONE number per
    upload, the 5th percentile over the whole recording, and a monocular
    reconstruction drifts in depth: measured 2026-09-17 on fix5's drafts, whole
    units sit 15-20 cm above their neighbours (7637514589861220209:clip000,
    lowest foot 0.566 and 0.581 m against 0.36-0.42 for the rest), and those
    floating bars are most of what the hop detector reads in the draft -- only
    33 of its 375 airborne frames come from a library unit that really hops.
    """
    joints = np.asarray(joints, dtype=np.float64)
    low = joints[:, FEET, 2].min(axis=1)
    floor = rolling_percentile(low, HOP_WINDOW, 5)
    hop = (low - floor) > HOP_RISE
    pelvis = joints[:, 0, 2]
    squat = (pelvis - rolling_percentile(pelvis, SQUAT_WINDOW, 80)) < -SQUAT_DROP
    return (hop * HOP_FLAG + squat * SQUAT_FLAG).astype(np.uint8), floor


def vertical_flags(joints):
    """joints [T, 24, 3], z up, metres -> uint8 [T] with HOP_FLAG | SQUAT_FLAG."""
    return vertical_series(joints)[0]


def release_affine(release):
    # [-1, 1], not [0, 1]: see tools/census_release_yaw_steps.py, which paid for it.
    normalizer = torch.load(release / "normalizer.pt", map_location="cpu",
                            weights_only=False)
    low = normalizer["data_min"].numpy().astype(np.float64)
    high = normalizer["data_max"].numpy().astype(np.float64)
    span = high - low
    span[span == 0] = 1.0
    return span / 2.0, low + span / 2.0


def split_flags(release, split, rows, affine):
    motion = np.load(release / split / "motion.npy", mmap_mode="r")
    names = json.loads((release / split / "names.json").read_text())
    by_sequence = {}
    for row in rows:
        if row["split"] == split:
            by_sequence.setdefault(row["sequence_id"], []).append(row)
    scale, offset = affine
    out = {}
    floors = {}
    started = time.time()
    for count, (sequence, windows) in enumerate(sorted(by_sequence.items())):
        windows.sort(key=lambda r: r["start_frame"])
        length = max(r["end_frame_exclusive"] for r in windows)
        stitched = np.full((length, motion.shape[2]), np.nan)
        for row in windows:
            a, b = row["start_frame"], row["end_frame_exclusive"]
            values = np.asarray(motion[row["array_index"]], dtype=np.float64)[:b - a]
            seen = ~np.isnan(stitched[a:b, 0])
            if seen.any() and not np.allclose(stitched[a:b][seen], values[seen], atol=1e-5):
                raise SystemExit("{}: overlapping windows disagree at {}".format(sequence, a))
            stitched[a:b] = values
        if np.isnan(stitched[:, 0]).any():
            raise SystemExit("{}: windows leave holes; refusing to guess".format(sequence))
        joints, _ = _decode_raw_151(stitched * scale + offset)
        flags, floor = vertical_series(joints)
        for row in windows:
            a, b = row["start_frame"], row["end_frame_exclusive"]
            window = np.zeros(motion.shape[1], dtype=np.uint8)
            window[:b - a] = flags[a:b]
            out[names[row["array_index"]]] = window
            local = np.full(motion.shape[1], np.nan, dtype=np.float32)
            local[:b - a] = floor[a:b]
            floors[names[row["array_index"]]] = local
        if count and count % 50 == 0:
            print("  {} {}/{} sequences, {:.0f}s".format(
                split, count, len(by_sequence), time.time() - started), flush=True)
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit("{}: {} windows not covered by windows.jsonl, first {}".format(
            split, len(missing), missing[:3]))
    return out, floors


def reference_rates(reference_dir, clips):
    hop, squat = [], []
    for clip in clips:
        with open(pathlib.Path(reference_dir) / (clip + ".pkl"), "rb") as handle:
            pose = np.asarray(pickle.load(handle)["full_pose"], dtype=np.float64)
        flags = vertical_flags(pose.reshape(len(pose), -1, 3))
        hop.append(float((flags & HOP_FLAG).astype(bool).mean()))
        squat.append(float((flags & SQUAT_FLAG).astype(bool).mean()))
    return float(np.mean(hop)), float(np.mean(squat))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--split", action="append", default=None, help="default: train")
    ap.add_argument("--out", required=True, metavar="NPZ")
    ap.add_argument("--floor-out", default=None, metavar="NPZ",
                    help="also write the per-frame LOCAL FLOOR (metres, raw) for "
                         "--draft-unit-floor")
    ap.add_argument("--reference-dir", default=None,
                    help="staged ground-truth pickles for the positive control")
    ap.add_argument("--reference-clips", default="runs/eval_clips_txy_t20.txt")
    args = ap.parse_args()

    if args.reference_dir:
        clips = [c for c in pathlib.Path(args.reference_clips).read_text().split() if c]
        hop, squat = reference_rates(args.reference_dir, clips)
        print("reference ({} clips): hop {:.4f} of frames, deep squat {:.4f} "
              "(the measurement read 0.0057 / 0.0395)".format(len(clips), hop, squat))

    release = pathlib.Path(args.release)
    rows = [json.loads(line) for line in open(release / "windows.jsonl")]
    affine = release_affine(release)
    flags = {}
    floors = {}
    for split in args.split or ["train"]:
        split_out, split_floors = split_flags(release, split, rows, affine)
        flags.update(split_out)
        floors.update(split_floors)
    if args.floor_out:
        floor_out = pathlib.Path(args.floor_out)
        floor_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(floor_out, **floors)
        # PROVENANCE, checked by infer_atomic before --draft-unit-floor is used:
        # the levelled and unlevelled T releases share every window name, so a
        # name check cannot tell whether these metres already had the floor
        # taken off.
        build = {}
        if (release / "build.json").is_file():
            build = json.loads((release / "build.json").read_text())
        floor_out.with_suffix(".json").write_text(json.dumps({
            "release": str(release),
            "floor_levelled": bool(build.get("floor_levelled", False)),
            "floor_reference_m": build.get("floor_reference_m"),
            "label_space_id": build.get("label_space_id"),
            "splits": args.split or ["train"],
            "rule": "running p{} of the lowest of SMPL joints {} over {} frames, "
                    "metres, raw decode".format(5, list(FEET), HOP_WINDOW),
            "windows": len(floors),
        }, indent=1))
        print("per-frame local floor -> {}".format(floor_out))
    stacked = np.stack(list(flags.values()))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **flags)
    summary = {
        "rule": "bit 1 hop: lowest foot > {} m over its running p5 ({} frames); "
                "bit 2 deep squat: pelvis > {} m under its running p80 ({} frames)".format(
                    HOP_RISE, HOP_WINDOW, SQUAT_DROP, SQUAT_WINDOW),
        "windows": len(flags),
        "hop_frame_share": float((stacked & HOP_FLAG).astype(bool).mean()),
        "squat_frame_share": float((stacked & SQUAT_FLAG).astype(bool).mean()),
        "windows_with_hop": int((stacked & HOP_FLAG).any(axis=1).sum()),
        "windows_with_squat": int((stacked & SQUAT_FLAG).any(axis=1).sum()),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("per-frame flags -> {}".format(out))


if __name__ == "__main__":
    main()
