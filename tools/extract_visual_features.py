"""Per-frame video embeddings for visual atomic discovery (paper Alg. 1, step 1).

The paper's segmentation cuts motion where per-frame *visual* features change
their self-similarity pattern, using an I3D encoder over the paired dance
video.  The released code ships none of that, and the kinematic replacement --
built without any visual input -- measures as music-orthogonal at every
granularity (worklog.md).  This tool supplies the missing visual signal.

Encoder: torchvision's Kinetics-400 **S3D**.  The paper names I3D but pins no
variant or checkpoint; S3D is its direct successor on the same training corpus
and, decisively, is installable in this environment today.  The encoder name
is recorded in every output header so features from different encoders can
never be mixed silently.

Alignment contract: one embedding per **30 fps motion frame**.  AIST videos are
~60 fps while the released 151-D motion is 30 fps, so motion frame ``t`` maps to
video frame ``2 t`` (the same 2:1 the AIST++ preprocessing uses).  Each
embedding pools a 16-frame (~0.27 s) video window centred on its motion frame,
with edges clamped -- so features exist for every motion frame rather than
leaving the boundaries undefined.

Outputs one ``<video_stem>.npz`` per video:

    features   [T30, 1024]  float16  (fp16 halves disk with ~1e-3 error, far
                                      below cluster-distance scales)
    meta       json string: encoder, video hash, fps, frame counts, window size

Existing outputs whose recorded video hash matches are skipped, so the tool is
resumable and safe to re-run after adding videos.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ENCODER_NAME = "torchvision/s3d KINETICS400_V1"
WINDOW = 16          # video frames per embedding window (~0.27 s at 60 fps)
# Video frames per 30 fps motion frame.  AIST footage is ~60 fps (ratio 2);
# the wild TikTok clips are already 30 fps (ratio 1).  Guessing this from the
# container fps would silently misalign features on variable-fps files, so it
# is an explicit argument recorded in every output.


def file_sha256(path, limit=1 << 20):
    """Hash the first megabyte -- enough to detect a swapped/truncated file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def load_encoder(device):
    from torchvision.models.video import S3D_Weights, s3d

    weights = S3D_Weights.KINETICS400_V1
    model = s3d(weights=weights).eval().to(device)
    return model, weights.transforms()


DECODE_SHORT_SIDE = 256  # matches the transform's resize target


@torch.no_grad()
def encode_video(path, model, transform, device, batch_size, video_per_motion):
    """Decode once at reduced resolution, then window from the buffer.

    Adjacent motion frames share 14 of their 16 video frames, so fetching each
    window independently decodes every frame about eight times -- and at the
    source's full 1080p even though the transform immediately resizes to a
    256 short side.  Decoding the whole clip once, already at that short side,
    turns the decoder from the bottleneck into a startup cost.
    """
    import decord

    probe = decord.VideoReader(str(path), num_threads=2)
    total = len(probe)
    fps = float(probe.get_avg_fps())
    height, width = probe[0].shape[:2]
    del probe
    if width <= height:
        out_w = DECODE_SHORT_SIDE
        out_h = int(round(height * DECODE_SHORT_SIDE / width / 2) * 2)
    else:
        out_h = DECODE_SHORT_SIDE
        out_w = int(round(width * DECODE_SHORT_SIDE / height / 2) * 2)
    reader = decord.VideoReader(str(path), num_threads=2, width=out_w, height=out_h)

    motion_frames = total // video_per_motion
    if motion_frames < 1:
        raise ValueError("{} has {} frames, too short".format(path, total))

    buffer = torch.from_numpy(
        reader.get_batch(list(range(total))).asnumpy()
    )  # [T, H, W, C] uint8, ~125 MB for a 10 s clip

    # The transform -- resize to 256, centre-crop 224, normalise -- runs where
    # the model runs.  Profiled on one 500-frame clip: decode 0.6 s, transform
    # on CPU 123 s, S3D forward 5.8 s.  The preprocessing was 95% of the cost
    # and the GPU sat idle through it; moving it across is 107x on the transform
    # alone.  The buffer is already short-side 256 (decord resized during
    # decode), so this is ~175 MB of uint8 per clip on the card.
    #
    # It is the same computation, checked rather than assumed: features from the
    # two paths agree to a minimum cosine similarity of 0.99999982 and a mean
    # relative difference of 6.3e-4 -- float ordering, not a different result.
    buffer = buffer.to(device, non_blocking=True)

    # Window index plan: centre a WINDOW-frame span on video frame 2t,
    # clamping at the edges so every motion frame gets a well-formed window.
    half = WINDOW // 2
    starts = np.clip(
        np.arange(motion_frames) * video_per_motion - half, 0, max(total - WINDOW, 0)
    )

    features = []
    for begin in range(0, motion_frames, batch_size):
        block = starts[begin : begin + batch_size]
        clips = torch.stack(
            [
                transform(buffer[s : s + WINDOW].permute(0, 3, 1, 2))
                for s in block
            ]
        )
        pooled = model.features(clips).mean(dim=(2, 3, 4))
        features.append(pooled.half().cpu())
    return torch.cat(features).numpy(), total, fps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard", type=int, default=0,
                        help="this worker's index; the corpus is split by "
                             "position so shards are disjoint and their union "
                             "is the whole directory")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N videos (smoke run)")
    parser.add_argument("--video-per-motion", type=int, default=2,
                        help="video frames per 30 fps motion frame: 2 for ~60 fps "
                             "AIST footage, 1 for 30 fps wild clips")
    args = parser.parse_args()

    videos = sorted(args.video_dir.glob("*.mp4"))
    if args.limit is not None:
        videos = videos[: args.limit]
    if args.num_shards > 1:
        videos = videos[args.shard:: args.num_shards]
    if not videos:
        raise SystemExit("no .mp4 under {}".format(args.video_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, transform = load_encoder(device)

    done = failed = skipped = 0
    for index, video in enumerate(videos, start=1):
        target = args.output_dir / "{}.npz".format(video.stem)
        source_hash = file_sha256(video)

        if target.is_file():
            try:
                with np.load(target, allow_pickle=False) as existing:
                    meta = json.loads(str(existing["meta"]))
                if meta.get("video_sha256_1mb") == source_hash:
                    skipped += 1
                    continue
            except Exception:
                pass  # unreadable/partial output: recompute below

        try:
            features, total_frames, fps = encode_video(
                video, model, transform, device, args.batch_size,
                args.video_per_motion,
            )
        except Exception as error:  # noqa: BLE001 - keep the sweep going
            print("FAIL {}: {}: {}".format(video.name, type(error).__name__, error), flush=True)
            failed += 1
            continue

        meta = {
            "encoder": ENCODER_NAME,
            "video": video.name,
            "video_sha256_1mb": source_hash,
            "video_frames": total_frames,
            "video_fps": fps,
            "motion_fps": 30.0,
            "video_frames_per_motion_frame": args.video_per_motion,
            "window_video_frames": WINDOW,
            "decode_short_side": DECODE_SHORT_SIDE,
            "feature_dim": int(features.shape[1]),
            "motion_frames": int(features.shape[0]),
        }
        # Write via a temp name so an interrupted run never leaves a readable
        # but truncated .npz behind.  The temp name must itself end in .npz,
        # because np.savez appends that suffix to anything else.
        staging = target.with_name(target.stem + ".tmp.npz")
        np.savez_compressed(staging, features=features, meta=json.dumps(meta))
        staging.rename(target)
        done += 1
        if index % 20 == 0 or index == len(videos):
            print("[{}/{}] done={} skipped={} failed={}".format(
                index, len(videos), done, skipped, failed), flush=True)

    print("finished: done={} skipped={} failed={}".format(done, skipped, failed))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
