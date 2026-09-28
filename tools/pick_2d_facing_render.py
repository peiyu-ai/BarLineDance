#!/usr/bin/env python3
"""Per clip, keep the 2D render whose facing is least wrong.

WHY SELECTION AND NOT ONLY A FIX.  The SteadyDancer front/back confusion is a
property of the SAMPLE, not only of the input (2026-09-21, clip 818): the same
pose video at the same seed reproduces its four phantom back views frame for
frame, while seed 43 -- with nothing else changed -- renders none of them, and
the pose-picture changes that were tried (detector-like torso, ears always
drawn, palette swap, reference-pose pairing) moved the count no more than a
seed does.  Since ``tools/score_2d_facing.py`` reads the rendered facing
reliably (validated on 153 hand-labelled frames: body back-view recall 33/33,
zero front<->back confusions for the head), rendering more than one sample and
keeping the cleanest is a fix whose effect does not depend on a theory of why
the model flips.

WHAT "WRONG" MEANS, per frame, from the scorer's smoothed labels and the input
pose's own side (``input_side`` +1 facing the camera, -1 away):

  * phantom      body back       while the input faces the camera
  * missed       body front      while the input is turned away
  * hair_on_front head back      while the input faces the camera and the body
                                 is not back (the operator's "头背身正")
  * face_on_back head front      on a back body (the operator's "头正身背")

``wrong`` is the UNION, so a frame counts once however many ways it is wrong.
Ties are broken by the number of flicker events (a flip that reverts within 8
frames), then by body+head flips.  Profile frames are never counted wrong: the
scorer reads profile bodies as front about half the time, so a profile-based
rule would be judging the scorer's noise.

    python3 tools/pick_2d_facing_render.py ARM_DIR [ARM_DIR ...] --out PICK.json

Each ARM_DIR holds ``<clip>.mp4`` and ``facing/<clip>.json`` (the scorer's
``--out``).  Clips missing from an arm are skipped for that arm, never scored as
clean.

GUARDS (review 2026-09-21 -- each one is a way the cleanest-looking render is
the wrong one):
  * the score must name THIS mp4 and be newer than it (a re-render leaves the
    old score behind);
  * the render's frame count must match its own pose video (``work/<clip>/
    driven.npy``) within the sampler's 0-4 frame rounding -- a render of
    another clip's pose (the staging race produced one: 329 frames for a
    349-frame pose) or a truncated one is refused, not ranked;
  * frames where the detector lost the body (``unknown``) count as wrong, so a
    render cannot win by being unreadable;
  * with ``--cues-dir`` the input side is recomputed for every arm from the
    same cues, with a profile band (75-105 deg) left unjudged -- the scorer's
    own cues path has no band, so a back reading at 88 deg counted as a
    phantom.  Without it, arms scored from different input sources are refused.
"""
import argparse
import json
import pathlib
import sys


PROFILE_BAND_DEG = (75.0, 105.0)
LENGTH_SLACK_FRAMES = 4


def sides_from_cues(cues_path, n):
    """+1 facing the camera, -1 away, 0 inside the profile band (not judged)."""
    yaw = [abs(v) for v in json.loads(pathlib.Path(cues_path).read_text())["sh_yaw"]][:n]
    low, high = PROFILE_BAND_DEG
    return [1 if v < low else (-1 if v > high else 0) for v in yaw]


def wrong_frames(frames, sides=None):
    """Per-category and union counts of wrong frames for one scored render.

    ``sides`` overrides each row's ``input_side``; a side of 0 is not judged.
    """
    counts = {"phantom": 0, "missed": 0, "hair_on_front": 0, "face_on_back": 0,
              "unknown": 0, "wrong": 0}
    for index, row in enumerate(frames):
        head, body = row["head"], row["body"]
        side = row.get("input_side", 0) if sides is None else (
            sides[index] if index < len(sides) else 0)
        if side == 0:
            continue
        flags = {
            "phantom": body == "back" and side == 1,
            "missed": body == "front" and side == -1,
            "hair_on_front": head == "back" and side == 1 and body != "back",
            "face_on_back": head == "front" and body == "back",
            "unknown": body == "unknown",
        }
        for key, value in flags.items():
            counts[key] += int(value)
        counts["wrong"] += int(any(flags.values()))
    return counts


def score(path, sides=None):
    data = json.loads(pathlib.Path(path).read_text())
    summary = data["summary"]
    counts = wrong_frames(data["frames"], sides)
    counts["frames"] = summary["frames"]
    counts["flicker"] = summary["body_flips"]["flicker"] + summary["head_flips"]["flicker"]
    counts["flips"] = summary["body_flips"]["flips"] + summary["head_flips"]["flips"]
    counts["source"] = summary.get("input_side_source", "")
    counts["video"] = summary.get("video", "")
    return counts


def refuse_reason(arm, clip, scored, counts):
    """Why this scored render must not be ranked, or None."""
    video = pathlib.Path(arm, clip + ".mp4")
    if counts["video"] and pathlib.Path(counts["video"]).resolve() != video.resolve():
        return "score is for {}".format(counts["video"])
    if scored.stat().st_mtime < video.stat().st_mtime:
        return "score is older than the render"
    driven = pathlib.Path(arm, "work", clip, "driven.npy")
    if driven.is_file():
        import numpy as np
        pose_frames = len(np.load(driven))
        if not 0 <= pose_frames - counts["frames"] <= LENGTH_SLACK_FRAMES:
            return "render has {} frames, its pose {}".format(counts["frames"], pose_frames)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("arms", nargs="+", help="arm directories")
    ap.add_argument("--out", default=None, help="write the per-clip choice here")
    ap.add_argument("--cues-dir", default=None,
                    help="<clip>.json with sh_yaw per frame: recompute the input side "
                         "for every arm from the same cues, with a profile band")
    args = ap.parse_args(argv)

    table = {}
    for arm in args.arms:
        for scored in sorted(pathlib.Path(arm, "facing").glob("*__clip*.json")):
            clip = scored.stem
            video = pathlib.Path(arm, clip + ".mp4")
            if not video.is_file():
                continue
            sides = None
            if args.cues_dir:
                cues = pathlib.Path(args.cues_dir, clip + ".json")
                if not cues.is_file():
                    print("SKIP {} {}: no cues".format(pathlib.Path(arm).name, clip))
                    continue
                sides = sides_from_cues(cues, 100000)
            counts = score(scored, sides)
            reason = refuse_reason(arm, clip, scored, counts)
            if reason:
                print("REFUSED {} {}: {}".format(pathlib.Path(arm).name, clip, reason))
                continue
            table.setdefault(clip, {})[arm] = counts
    if not table:
        sys.exit("no scored renders found under {}".format(args.arms))
    if not args.cues_dir:
        for clip, per_arm in table.items():
            sources = {c["source"] for c in per_arm.values()}
            if len(sources) > 1:
                sys.exit("{}: arms were scored from different input sources {}; "
                         "pass --cues-dir to put them on one scale".format(clip, sorted(sources)))

    picks = {}
    print("{:<30}".format("clip") + "".join("{:>18}".format(pathlib.Path(a).name[:17]) for a in args.arms) + "   pick")
    for clip, per_arm in sorted(table.items()):
        best = min(per_arm, key=lambda a: (per_arm[a]["wrong"], per_arm[a]["flicker"], per_arm[a]["flips"]))
        picks[clip] = {"arm": best, "video": str(pathlib.Path(best, clip + ".mp4")),
                       "scores": per_arm}
        cells = "".join("{:>18}".format(
            "{}w/{}f".format(per_arm[a]["wrong"], per_arm[a]["flicker"]) if a in per_arm else "-")
            for a in args.arms)
        print("{:<30}{}   {}".format(clip, cells, pathlib.Path(best).name))
    for arm in args.arms:
        present = [c for c in table if arm in table[c]]
        total = sum(table[c][arm]["wrong"] for c in present)
        bad = sum(table[c][arm]["wrong"] > 0 for c in present)
        print("{:<30} {} clips, {} wrong frames, {} clips with any".format(
            pathlib.Path(arm).name, len(present), total, bad))
    total = sum(p["scores"][p["arm"]]["wrong"] for p in picks.values())
    bad = sum(p["scores"][p["arm"]]["wrong"] > 0 for p in picks.values())
    print("{:<30} {} clips, {} wrong frames, {} clips with any".format("PICKED", len(picks), total, bad))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(picks, indent=2))


if __name__ == "__main__":
    main()
