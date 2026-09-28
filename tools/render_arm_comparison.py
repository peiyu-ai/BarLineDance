#!/usr/bin/env python3
"""Put two trained arms and the ground truth side by side, as pose filmstrips.

A video convinces a person; a filmstrip lets them compare.  ``render_dance_video``
already produces the former and is the right tool for "is this dance any good".
It is the wrong tool for "is arm A better than arm B", because two videos cannot
be looked at simultaneously and the reader ends up comparing their memory of one
against their view of the other.

Three decisions here, each of which changes what the picture is able to show:

* **World coordinates, never the normalised pose.**  ``render_prototype_cards``
  draws canonical pose -- translation removed, divided by shoulder width -- which
  is right for reading *what shape the body is in* and exactly wrong here: it
  divides out the vertical drift and the travel across the floor, which are two
  of the things the arms can differ on.  This module reuses the joint positions
  as the generator emitted them, the same array ``render_dance_video`` draws.

* **One axis box per clip, shared by every arm.**  Fitting the box to each arm
  separately rescales them independently, so an arm that drifts a metre gets
  zoomed out until it looks as tidy as one that does not.  The box here is the
  union over the arms being compared, so a bigger excursion *looks* bigger.

* **Deterministic clip selection, printed with the output.**  Picking the clips
  that flatter a conclusion is the easiest way to make a filmstrip lie.  The
  default takes the first ``--count`` clip ids in sorted order out of the
  intersection of all arms, and the manifest records the full candidate count so
  a reader can see what fraction they are looking at.

Output is inline SVG (one ``<svg>`` per clip per arm) plus a JSON manifest.
Strokes use ``currentColor`` so the same file is legible on light and dark
backgrounds -- this repo has already shipped a PNG that baked axis ink to near
black and was unreadable for half its readers.

    python3 tools/render_arm_comparison.py \
        --gt-dir runs/wild_v4_acct_gt_eval/motion \
        --arm '150 frames:runs/m6_wild_v4_acct/motion' \
        --arm '340 frames:runs/m6_wild_v4_acct_w2_340/motion' \
        --count 4 --output runs/arm_comparison
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np

# SMPL kinematic tree, matching vis.py / tools/render_dance_video.py.
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    20, 21,
]
# The seed suffix infer_atomic's driver appends when it flattens per-seed output
# into one directory.  Stripping it is how a generated file is matched back to
# the clip -- and to the ground truth, which carries no suffix.
SEED_SUFFIX = re.compile(r"_s\d+$")


def clip_id(stem):
    return SEED_SUFFIX.sub("", stem)


def load_pose(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    pose = np.asarray(payload["full_pose"], dtype=np.float32)
    if pose.ndim != 3 or pose.shape[1:] != (24, 3):
        raise SystemExit("{}: expected full_pose [T,24,3], got {}".format(path, pose.shape))
    return pose


def index_arm(directory, one_per_clip=True):
    """Map clip id -> path.  With several seeds present, take the lowest seed.

    Choosing the lowest rather than a random one keeps the figure reproducible
    from the command line alone, and keeps every arm on the *same* seed index so
    the comparison is not partly a comparison of draws.
    """
    directory = Path(directory)
    chosen = {}
    for path in sorted(directory.glob("*.pkl")):
        key = clip_id(path.stem)
        if not one_per_clip or key not in chosen:
            chosen[key] = path
    return chosen


def project(pose, elev_deg, azim_deg):
    """Orthographic projection of [T,24,3] to [T,24,2], z up.

    Orthographic rather than perspective on purpose: a perspective camera makes
    a dancer who travels toward it grow, which would read as a change in the
    motion rather than in the position.
    """
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    # Right-handed: rotate about z by -azim, then tilt the view by elev.
    ca, sa = math.cos(azim), math.sin(azim)
    ce, se = math.cos(elev), math.sin(elev)
    right = np.array([-sa, ca, 0.0], dtype=np.float32)
    up = np.array([-ca * se, -sa * se, ce], dtype=np.float32)
    flat = pose.reshape(-1, 3)
    out = np.stack([flat @ right, flat @ up], axis=-1)
    return out.reshape(pose.shape[0], 24, 2)


def sample_frames(length, count):
    """Evenly spaced frame indices including both ends."""
    if length <= 0:
        return []
    if count >= length:
        return list(range(length))
    return [int(round(i * (length - 1) / (count - 1))) for i in range(count)]


def svg_for(pose2d, frames, box, cell, gap, stroke):
    """One filmstrip row: `len(frames)` skeletons laid left to right."""
    (x0, x1), (y0, y1) = box
    span_x = max(x1 - x0, 1e-6)
    span_y = max(y1 - y0, 1e-6)
    # One scale for both axes -- an anisotropic fit would stretch a crouch into a
    # normal stance, which is a change to the pose, not to the framing.
    scale = min(cell / span_x, cell / span_y)
    pad_x = (cell - span_x * scale) / 2
    pad_y = (cell - span_y * scale) / 2

    width = len(frames) * cell + max(len(frames) - 1, 0) * gap
    parts = [
        '<svg viewBox="0 0 {:.1f} {:.1f}" width="100%" '
        'preserveAspectRatio="xMidYMid meet" role="img" '
        'xmlns="http://www.w3.org/2000/svg">'.format(width, cell)
    ]
    for slot, frame in enumerate(frames):
        ox = slot * (cell + gap) + pad_x
        joints = pose2d[frame]
        segments = []
        for joint, parent in enumerate(SMPL_PARENTS):
            if parent < 0:
                continue
            ax = ox + (joints[joint, 0] - x0) * scale
            ay = cell - pad_y - (joints[joint, 1] - y0) * scale
            bx = ox + (joints[parent, 0] - x0) * scale
            by = cell - pad_y - (joints[parent, 1] - y0) * scale
            segments.append("M{:.1f} {:.1f}L{:.1f} {:.1f}".format(ax, ay, bx, by))
        parts.append(
            '<path d="{}" fill="none" stroke="{}" stroke-width="{:.2f}" '
            'stroke-linecap="round" stroke-linejoin="round"/>'.format(
                "".join(segments), stroke, max(cell / 90.0, 0.6)
            )
        )
        # Floor line at the box's own z=min, so a body that leaves the ground
        # reads as leaving the ground rather than as a different framing.
        floor_y = cell - pad_y
        parts.append(
            '<path d="M{:.1f} {:.1f}L{:.1f} {:.1f}" stroke="{}" stroke-width="0.5" '
            'stroke-dasharray="2 3" opacity="0.35" fill="none"/>'.format(
                ox, floor_y, ox + span_x * scale, floor_y, stroke
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-dir", required=True,
                        help="ground-truth motion directory (one .pkl per clip)")
    parser.add_argument("--arm", action="append", default=[], metavar="NAME:DIR",
                        help="a generated-motion directory, labelled; repeatable")
    parser.add_argument("--clips", default="",
                        help="optional file of clip ids, one per line, overriding --count")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--frames", type=int, default=8, help="poses per filmstrip")
    parser.add_argument("--view-elev", type=float, default=12.0)
    parser.add_argument("--view-azim", type=float, default=30.0)
    parser.add_argument("--cell", type=float, default=100.0)
    parser.add_argument("--gap", type=float, default=6.0)
    parser.add_argument("--stroke", default="currentColor")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    arms = []
    for spec in args.arm:
        if ":" not in spec:
            raise SystemExit("--arm expects NAME:DIR, got {!r}".format(spec))
        name, _, directory = spec.partition(":")
        arms.append((name.strip(), Path(directory.strip())))
    if not arms:
        raise SystemExit("at least one --arm is required")

    gt = index_arm(args.gt_dir)
    indexed = [(name, index_arm(directory)) for name, directory in arms]

    shared = set(gt)
    for _, table in indexed:
        shared &= set(table)
    candidates = sorted(shared)
    if not candidates:
        raise SystemExit("no clip id is present in the ground truth and every arm")

    if args.clips:
        wanted = [line.strip() for line in Path(args.clips).read_text().splitlines() if line.strip()]
        missing = [c for c in wanted if c not in shared]
        if missing:
            raise SystemExit("{} requested clip(s) absent from some arm, first: {}".format(
                len(missing), missing[0]))
        selected = wanted[: args.count] if args.count else wanted
    else:
        selected = candidates[: args.count]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for cid in selected:
        series = [("ground truth", load_pose(gt[cid]))]
        for name, table in indexed:
            series.append((name, load_pose(table[cid])))

        # One box for the clip, union over the arms.  See the module docstring:
        # per-arm boxes would rescale a drifting arm until it looked tidy.
        projected = [(name, project(pose, args.view_elev, args.view_azim)) for name, pose in series]
        stacked = np.concatenate([p.reshape(-1, 2) for _, p in projected], axis=0)
        mins, maxs = stacked.min(axis=0), stacked.max(axis=0)
        centre = (mins + maxs) / 2
        radius = float((maxs - mins).max()) / 2 * 1.06 + 0.05
        box = ((centre[0] - radius, centre[0] + radius),
               (centre[1] - radius, centre[1] + radius))

        entry = {"clip": cid, "arms": []}
        for (name, pose2d), (_, pose) in zip(projected, series):
            frames = sample_frames(len(pose2d), args.frames)
            svg = svg_for(pose2d, frames, box, args.cell, args.gap, args.stroke)
            path = output / "{}__{}.svg".format(
                cid.replace(":", "_"), re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_"))
            path.write_text(svg, encoding="utf-8")
            root_z = pose[:, 0, 2]
            lowest_toe = np.minimum(pose[:, 10, 2], pose[:, 11, 2])
            entry["arms"].append({
                "name": name,
                "svg": str(path),
                "frames": int(len(pose)),
                "seconds": round(len(pose) / 30.0, 2),
                # Reported next to the picture so the reader is not asked to
                # eyeball drift: the same lowest-toe wander audit_wild_3d_quality
                # thresholds against mocap's p99 of 0.5303 m.
                "lowest_toe_wander_m": round(float(lowest_toe.max() - lowest_toe.min()), 4),
                "root_height_range_m": round(float(root_z.max() - root_z.min()), 4),
            })
        rows.append(entry)

    manifest = {
        "what_it_is": "pose filmstrips in world coordinates, one row per arm, "
                      "one shared axis box per clip",
        "selection": "first {} of {} clip ids present in the ground truth and every arm, "
                     "sorted".format(len(selected), len(candidates)),
        "candidates_total": len(candidates),
        "view": {"elev": args.view_elev, "azim": args.view_azim, "projection": "orthographic"},
        "frames_per_strip": args.frames,
        "not_normalised": "joint positions as emitted; translation and floor height are kept, "
                          "so vertical drift and travel are visible rather than divided out",
        "clips": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
