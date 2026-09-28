"""Run one shard of wild-video S3D extraction.

The parallel launcher originally fed this logic to ``python -`` through a
heredoc, but the trailing ``< /dev/null`` (needed so nohup fully detaches)
overrode the heredoc and python read an empty script -- three shards exited
silently with empty logs.  A real script file cannot lose its own source.

Shards partition by sha256 of the video stem so membership is stable across
runs and independent of listing order.  Each shard symlinks its subset into a
private directory and drives ``tools/extract_visual_features.py`` over it;
that extractor skips outputs whose recorded video hash matches, so shards are
resumable and safe to relaunch.
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--video-dir", type=Path,
                        default=Path("data/wild_videos_pilot"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/wild_visual_s3d"))
    args = parser.parse_args()

    videos = sorted(args.video_dir.glob("*.mp4"))
    mine = [
        v for v in videos
        if int(hashlib.sha256(v.stem.encode()).hexdigest(), 16)
        % args.num_shards == args.shard
    ]
    print("shard {}/{}: {} of {} videos".format(
        args.shard, args.num_shards, len(mine), len(videos)), flush=True)

    # Name the shard directory after its source.  A fixed name accumulates the
    # union of every --video-dir it was ever run with, and the extractor globs
    # that directory, so a later run silently re-walks an earlier run's clips.
    shard_dir = args.video_dir.parent / "{}_shard{}".format(args.video_dir.name, args.shard)
    shard_dir.mkdir(parents=True, exist_ok=True)
    for video in mine:
        dst = shard_dir / video.name
        if not dst.exists():
            dst.symlink_to(video.resolve())

    sys.argv = [
        "extract_visual_features",
        "--video-dir", str(shard_dir),
        "--output-dir", str(args.output_dir),
        "--video-per-motion", "1",
    ]
    import tools.extract_visual_features as extractor

    return extractor.main()


if __name__ == "__main__":
    raise SystemExit(main())
