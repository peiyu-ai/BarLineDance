"""Give every eval clip a directory the renderer can SKIN, not just plot.

``runs/txy_t_gt_eval/motion/*.pkl`` holds joint positions only
(``tools/export_wild_eval_motion.py`` writes ``full_pose`` and no SMPL
parameters), so ground truth cannot be skinned from the file the FID set was
built from.  ``render_avatar_video`` therefore accepts a directory holding
``atomic_motion_151.npy`` and decodes it through the repo's own decoder.

``data/wild3d/ingest_v1_converted/`` carries such a directory for most clips but
NOT for all of them: 4 of the fixed ten T-line clips are missing from it, and
dropping those four would silently change the ruler -- CLAUDE.md 1.5 rule 6
fixes the ten precisely so that a round cannot be compared against a different
set.  The T line keeps its own copy of the same 151-D array as
``data/wild3d/txy_t_performance/sequences/<hash>/motion_151_raw.npy``; this
stages it under the name the renderer expects.

THE GATE.  A staged clip is only accepted when the joints decoded from the
151-D match the eval pickle's own ``full_pose`` -- the array every score in this
repository was computed from.  Without it a mis-keyed hash would stage ANOTHER
DANCE as ground truth and the comparison would look perfectly normal.
"""
import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

PERFORMANCE = pathlib.Path("data/wild3d/txy_t_performance")
CONVERTED = pathlib.Path("data/wild3d/ingest_v1_converted")


def sequence_index():
    index = {}
    for line in open(PERFORMANCE / "sequences.jsonl"):
        row = json.loads(line)
        key = row.get("sequence_id") or row.get("recording_id")
        if key and row.get("motion_path"):
            index[key] = PERFORMANCE / row["motion_path"]
    return index


def legacy_name(clip):
    _, recording, part = clip.split(":")
    return "{}__{}".format(recording, part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--eval-dir", default="runs/txy_t_gt_eval/motion")
    ap.add_argument("--out", default="runs/txy_t_gt_render")
    ap.add_argument("--tolerance", type=float, default=1e-3,
                    help="metres; max joint disagreement against the eval pickle")
    args = ap.parse_args()

    from tools.render_dance_video import _decode_raw_151

    index = sequence_index()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    staged = reused = 0
    for clip in (line.strip() for line in open(args.clips)):
        if not clip:
            continue
        name = legacy_name(clip)
        target = out / name
        converted = CONVERTED / name / "atomic_motion_151.npy"
        if converted.is_file():
            if not target.exists():
                target.symlink_to((CONVERTED / name).resolve(), target_is_directory=True)
            reused += 1
            continue
        source = index.get(clip)
        if source is None or not source.is_file():
            raise SystemExit("no 151-D source for {}; it is in neither "
                             "ingest_v1_converted nor the T performance tree"
                             .format(clip))
        raw = np.load(source)
        joints, _ = _decode_raw_151(raw)
        reference = pickle.load(open(pathlib.Path(args.eval_dir) / (clip + ".pkl"), "rb"))
        truth = np.asarray(reference["full_pose"], dtype=np.float64)
        frames = min(len(truth), len(joints))
        gap = float(np.abs(np.asarray(joints[:frames], np.float64) - truth[:frames]).max())
        if gap > args.tolerance:
            raise SystemExit(
                "{}: the staged 151-D decodes to joints {:.4f} m from the eval "
                "pickle's own full_pose, over {} frames. That is a different "
                "take or a mis-keyed hash, and staging it would put ANOTHER "
                "DANCE on screen as ground truth.".format(clip, gap, frames))
        target.mkdir(exist_ok=True)
        np.save(target / "atomic_motion_151.npy", raw)
        print("staged {}  ({} frames, joints agree to {:.2e} m)".format(clip, frames, gap))
        staged += 1
    print("{} staged from the T performance tree, {} reused from "
          "ingest_v1_converted -> {}".format(staged, reused, out))


if __name__ == "__main__":
    main()
