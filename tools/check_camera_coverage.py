#!/usr/bin/env python3
"""Does every row actually appear in the shot the renderer builds for it?

WHY THIS IS A CRITERION AND NOT AN EYEBALL CHECK.  ``render_avatar_video``
frames the shot on the body and, until 2026-08-30, aimed it at the reference
row's root alone, with the reason written into the code: an arm that wanders
should be seen to wander.  That is right while the wandering arm is still on
screen, and measured over 100 clips it was not -- the generated root travels
0.33 m against the ground truth's 1.21 m, so the peak root-to-root separation
is a median 1.198 m and a p90 of 2.005 m against a half-frame of roughly
1.25 m.  The reviewer then reads "the model barely moves" as "the model is not
in the picture", which is a different defect with a different fix.

WHAT IT MEASURES.  Each row's pelvis is pushed through the *actual* camera
matrix ``_camera_pose`` returns for that frame, and the fraction of frames
where it falls outside the frustum -- or outside ``--margin`` of the frame
half-width -- is reported per row.  Nothing here re-derives the camera: it
imports the renderer's own function, so a change to the framing rule cannot
pass this gate without also changing the picture.

PROVENANCE, and what it is NOT.  Invented here (CLAUDE.md 2.1 gate 1): no part
of the paper says where a camera goes.  It is a gate on the *instrument*, not
on the dance -- a row can be perfectly framed and still be a bad dance -- so it
must never be quoted as a quality reading.  Its positive control lives in
``tests/test_camera_coverage.py``: a synthetic two-row scene whose roots are
2.0 m apart, which the old reference-following camera must fail and the
centroid camera must pass.
"""
import argparse
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.render_avatar_video import FOV, _camera_pose, _z_up_to_y_up, stage, floor_of


def visible_fraction(joints_per_row, floor=None, margin=0.90, view="front"):
    """Share of frames where each row's pelvis sits inside the frame.

    ``margin`` 1.0 is the frame edge; the default 0.90 asks for the dancer to be
    inside 90% of the half-width, because a pelvis exactly on the edge means the
    limbs are already out.
    """
    roots = [np.asarray(j)[:, 0, :] for j in joints_per_row]
    span = min(len(r) for r in roots)
    roots = np.stack([r[:span] for r in roots])
    if floor is None:
        floor = floor_of(joints_per_row[0])
    spec = stage(joints_per_row, floor)
    spec["spread"] = float(np.linalg.norm(
        roots[:, None, :, :2] - roots[None, :, :, :2], axis=-1).max()) if len(roots) > 1 else 0.0
    target = roots.mean(0)[:, :2]
    inside = np.zeros(len(roots))
    for frame in range(span):
        pose = _camera_pose(spec, look_at=target[frame], view=view)
        world_to_camera = np.linalg.inv(pose)
        for row in range(len(roots)):
            point = _z_up_to_y_up() @ np.append(roots[row, frame], 1.0)
            camera = world_to_camera @ point
            depth = -camera[2]
            if depth <= 1e-6:
                continue
            half = depth * np.tan(FOV / 2)
            if abs(camera[0]) <= margin * half and abs(camera[1]) <= margin * half:
                inside[row] += 1
    return inside / span


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion", action="append", required=True, metavar="TITLE:PATH")
    parser.add_argument("--margin", type=float, default=0.90)
    parser.add_argument("--gate", type=float, default=0.95,
                        help="minimum visible fraction per row")
    args = parser.parse_args()

    titles, rows = [], []
    for spec in args.motion:
        title, path = spec.split(":", 1)
        titles.append(title)
        rows.append(np.asarray(pickle.load(open(path, "rb"))["full_pose"], float))
    fractions = visible_fraction(rows, margin=args.margin)
    worst = 1.0
    for title, fraction in zip(titles, fractions):
        print("{:<34} visible {:.3f}".format(title, fraction))
        worst = min(worst, float(fraction))
    print("gate {:.2f}: {}".format(args.gate, "PASS" if worst >= args.gate else "FAIL"))
    raise SystemExit(0 if worst >= args.gate else 1)


if __name__ == "__main__":
    main()
