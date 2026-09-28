"""Run GVHMR on one video and save world-frame SMPL-X predictions, no rendering.

This is the extraction driver for the wild-3D corpus.  GVHMR's own
``tools/demo/demo.py`` renders after predicting, and rendering is deliberately
unavailable here: the pytorch3d compat shim stubs the mesh types so a fake
renderer can never pass for a real one.  This driver reuses GVHMR's *own*
preprocess and predict path -- detection, ViTPose, image features, SimpleVO,
then the network -- and stops after ``torch.save``.

Run it from the GVHMR checkout with the shim first on PYTHONPATH:

    cd third_party/GVHMR
    PYTHONPATH=../pytorch3d_compat:. python ../../tools/run_gvhmr_extract.py \
        --video <clip.mp4> --output-root <dir>

``--use-dpvo`` swaps SimpleVO for DPVO and needs two more entries on the path,
``../torch_scatter_compat`` and ``third-party/DPVO``.  It exists because
SimpleVO is brittle rather than merely imprecise: it samples every 8th frame
and dies on the *first* pair where SIFT finds no descriptors or the solver
returns no pose, so one untextured frame discards a whole clip.  That is 45% of
this corpus.  DPVO tracks the same clips at roughly 45 frames/s.

GVHMR's ``tools/`` is a namespace package (no ``__init__.py``), and a regular
top-level ``tools`` package anywhere on ``sys.path`` -- this repo's own, or
the one the installed nvfuser egg ships in dist-packages -- shadows it at any
path position, because namespace packages only materialize when the *entire*
path scan finds no regular package.  ``tools.demo.demo`` is therefore loaded
from its file path below, bypassing package resolution entirely; its own
imports are hmr4d-only, so nothing else needs the GVHMR ``tools`` name.

Outputs land in ``<output-root>/<video_stem>/``:

    hmr4d_results.pt   GVHMR's native prediction dict:
                       smpl_params_global {body_pose, betas, global_orient,
                       transl}, smpl_params_incam, K_fullimg, net_outputs
    extract_meta.json  provenance: video path+hash, frame count, checkpoint
                       hash, VO mode -- enough to audit any sample later

The world frame is GVHMR's gravity-aligned y-up ('ay'); conversion to the
151-D z-up representation is a separate, already-tested step and is *not* done
here, so a failure in conversion can never corrupt raw HMR outputs.
"""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch

# GVHMR predates torch 2.6's weights_only=True default and calls bare
# torch.load on its own artifacts: preprocess outputs it just wrote (bbx,
# vitpose, slam trajectories) and the pinned release checkpoint.  All are
# trusted inputs of this driver, so restore the old default rather than
# patching every call site in the vendored checkout.  Explicit weights_only
# arguments are left untouched.
_torch_load = torch.load


def _load_trusting(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _load_trusting

# GVHMR's SLAM reader is a separate process feeding frames over a
# multiprocessing Queue, and every item carries a torch tensor (the
# intrinsics).  Under torch's default ``file_descriptor`` strategy the tensor's
# storage travels as a file descriptor that the consumer fetches over the
# *producer's* resource_sharer unix socket -- so when the reader runs out of
# frames and exits, the socket is unlinked and every tensor still queued
# becomes unfetchable.  The consumer then raises FileNotFoundError near the end
# of the clip, stops draining, and the producer's feeder thread blocks on a
# full pipe: both sides deadlock in their atexit join and the worker is gone
# for good.  Measured on this corpus: 15 clips, each one permanently wedging a
# shard.  ``file_system`` passes storages as named shm segments instead, which
# outlive the producer, so the exit race has nothing to lose.
#
# Module level is not incidental.  DPVO calls set_start_method('spawn', True)
# at import, so the reader does not inherit this process's setting; a spawned
# child re-imports sys.argv[0] as ``__mp_main__``, which makes module scope
# here the one place that reaches both sides.
torch.multiprocessing.set_sharing_strategy("file_system")


def sha256_head(path, limit=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


SIMPLE_VO = "SimpleVO(sift)"
DPVO = "DPVO"
NO_VO = "none"


def _vo_name(args):
    if args.static_cam:
        return NO_VO
    return DPVO if args.use_dpvo else SIMPLE_VO


def _reject_nonfinite_camera_track(cfg):
    """Fail loudly on a diverged camera track instead of predicting from NaN.

    DPVO exits 0 whether or not it converged, and a diverged run writes NaN
    into the trajectory.  ``compute_cam_angvel`` then propagates that through
    the whole prediction: the network still returns tensors, ``torch.save``
    still succeeds, and the corruption only surfaces later when scipy cannot
    take an SVD of a NaN rotation matrix.  Measured on this corpus, that was
    every one of the 368 conversion failures in the first DPVO sweep.

    Deleting the trajectory matters as much as raising: it is cached, so a
    retry would otherwise reuse the NaN rather than re-track.
    """
    slam_path = Path(cfg.paths.slam)
    if cfg.static_cam or not slam_path.is_file():
        return
    import numpy as np

    trajectory = np.asarray(torch.load(slam_path, map_location="cpu"), dtype=np.float64)
    if np.isfinite(trajectory).all():
        return
    first_bad = int(np.flatnonzero(~np.isfinite(trajectory).reshape(len(trajectory), -1).all(axis=1))[0])
    slam_path.unlink()
    raise SystemExit(
        "visual odometry diverged: non-finite camera track from frame {} of {} "
        "(cached trajectory removed)".format(first_bad, len(trajectory))
    )


def _make_dpvo_inference_only():
    """Run GVHMR's DPVO wrapper under ``no_grad``, which it forgets to do.

    Upstream DPVO's own demo decorates its whole loop with ``@torch.no_grad``.
    GVHMR's ``SLAMModel.track`` does not, and nothing below it does either, so
    every tracked frame keeps its autograd graph alive: measured at ~75 MiB per
    frame, growing linearly until a 414-frame clip exhausts a 72 GiB card
    around frame 350.  With this patch the same clip holds flat at 0.17 GiB.

    Patched here rather than in the vendored checkout so the fix travels with
    this repo instead of living in an untracked third-party tree.
    """
    from hmr4d.utils.preproc.slam import SLAMModel

    if getattr(SLAMModel.track, "_inference_only", False):
        return
    original = SLAMModel.track

    def track(self):
        with torch.no_grad():
            return original(self)

    track._inference_only = True
    SLAMModel.track = track


class _ModelHolder:
    """Instantiate GVHMR once and reuse it across a batch.

    Per-clip process startup dominated the sweep: 64 s of wall clock per clip
    against a median 17 s of actual preprocessing, so 74% was interpreter
    start, hydra compose, model instantiation and checkpoint load -- paid again
    for every one of 6041 clips.
    """

    def __init__(self, cfg_factory):
        self._cfg_factory = cfg_factory
        self._model = None

    def predict(self, cfg, data):
        if self._model is None:
            import hydra
            from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL

            model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            self._model = model.eval().cuda()
        return self._model.predict(data, static_cam=cfg.static_cam)


def extract_one(video, args, tools, model_holder):
    """Extract one clip.  Returns the meta dict; raises on any refusal."""
    get_video_lwh, load_data_dict, run_preprocess, compose, register_store_gvhmr = tools

    length, width, height = get_video_lwh(video)
    print("[input] {} ({} frames, {}x{})".format(video, length, width, height), flush=True)

    overrides = [
        "video_name={}".format(video.stem),
        "static_cam={}".format(args.static_cam),
        "verbose=False",
        "use_dpvo={}".format(args.use_dpvo),
        "output_root={}".format(args.output_root),
    ]
    if args.f_mm is not None:
        overrides.append("f_mm={}".format(args.f_mm))
    cfg = compose(config_name="demo", overrides=overrides)

    paths = cfg.paths
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # GVHMR's demo copies the video into its own output layout.  A symlink
    # satisfies every reader in that path and does not duplicate the corpus:
    # the copies were on track for ~25 GB.
    staged_video = Path(cfg.video_path)
    if not staged_video.exists():
        staged_video.symlink_to(video.resolve())

    started = time.time()
    # DPVO seeds patch depths with an unseeded torch.rand_like, so divergence is
    # partly a function of that draw rather than of the footage alone: clips
    # that blow up on one attempt track cleanly on the next.  Retrying is not a
    # fabrication -- it is the same estimator on the same frames -- and it is
    # cheap, because the guard deletes only the trajectory, leaving detection,
    # ViTPose and image features cached, so a retry re-runs the tracker alone.
    attempts = max(1, int(args.vo_attempts))
    for attempt in range(1, attempts + 1):
        run_preprocess(cfg)
        try:
            _reject_nonfinite_camera_track(cfg)
            break
        except SystemExit as divergence:
            if attempt == attempts:
                raise
            print("[vo] attempt {}/{} diverged ({}); retrying".format(
                attempt, attempts, divergence), flush=True)
    attempts_used = attempt
    data = load_data_dict(cfg)
    preprocess_seconds = time.time() - started

    result_path = Path(paths.hmr4d_results)
    if result_path.exists():
        print("[predict] exists, skipping: {}".format(result_path), flush=True)
    else:
        started = time.time()
        pred = model_holder.predict(cfg, data)
        predict_seconds = time.time() - started
        # Detach before saving so the file holds tensors, not graph references.
        from hmr4d.utils.net_utils import detach_to_cpu

        pred = detach_to_cpu(pred)
        # Second net: a finite camera track is necessary, not sufficient.  A
        # result file that exists is treated downstream as a result that is
        # usable, so it must never hold NaN.
        nonfinite = sorted(
            key for key, value in pred["smpl_params_global"].items()
            if not bool(torch.isfinite(value).all())
        )
        if nonfinite:
            raise SystemExit(
                "prediction is non-finite in {}; refusing to write {}".format(
                    ", ".join(nonfinite), result_path))
        torch.save(pred, result_path)
        print("[predict] {:.1f}s -> {}".format(predict_seconds, result_path), flush=True)

    global_params = torch.load(result_path, map_location="cpu")["smpl_params_global"]
    meta = {
        "video": str(video),
        "video_sha256_1mb": sha256_head(video),
        "video_frames": int(length),
        "video_wh": [int(width), int(height)],
        "backend": "GVHMR",
        "checkpoint": str(cfg.ckpt_path),
        "checkpoint_sha256_1mb": sha256_head(cfg.ckpt_path),
        "static_cam": bool(args.static_cam),
        "visual_odometry": _vo_name(args),
        "world_convention": "gravity-aligned y-up ('ay')",
        "preprocess_seconds": round(preprocess_seconds, 1),
        "visual_odometry_attempts_allowed": int(args.vo_attempts),
        # How many the tracker actually needed.  Final success is a censored
        # measure of divergence propensity -- a clip rescued on attempt 3 and
        # one that tracked first time both land in the corpus looking alike --
        # and comparing footage (or clip lengths) on how hard the camera was to
        # track requires the uncensored number.
        "visual_odometry_attempts_used": int(attempts_used),
        "smpl_params_global_shapes": {
            key: list(value.shape) for key, value in global_params.items()
        },
    }
    if args.use_dpvo:
        # The tracker is a second learned model with its own weights; auditing a
        # world trajectory later means knowing which ones produced it.
        vo_ckpt = Path("inputs/checkpoints/dpvo/dpvo.pth")
        meta["visual_odometry_checkpoint"] = str(vo_ckpt.resolve())
        meta["visual_odometry_checkpoint_sha256_1mb"] = sha256_head(vo_ckpt)
        meta["visual_odometry_inference_only_patch"] = (
            "SLAMModel.track wrapped in torch.no_grad by tools/run_gvhmr_extract.py"
        )
    if args.static_cam and args.static_cam_evidence is not None:
        # static_cam asserts R_w2c = I for every frame; the error that assertion
        # introduces is bounded by the camera's measured rotation, so the
        # measurement travels with the extraction that relied on it.
        meta["static_cam_justification"] = json.loads(
            Path(args.static_cam_evidence).read_text(encoding="utf-8"))
    meta_path = Path(cfg.output_dir) / "extract_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("[meta] -> {}".format(meta_path), flush=True)
    for key, value in meta["smpl_params_global_shapes"].items():
        print("    smpl_params_global.{}: {}".format(key, value))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--video-list", type=Path, default=None,
                        help="file of video paths, one per line; the model is "
                             "loaded once and reused for all of them")
    parser.add_argument("--static-cam-evidence", type=Path, default=None,
                        help="measure_camera_rotation.py JSON to record with a "
                             "--static-cam extraction")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--static-cam", action="store_true",
                        help="skip visual odometry (fixed-camera footage only)")
    parser.add_argument("--use-dpvo", action="store_true",
                        help="track the camera with DPVO instead of SimpleVO")
    parser.add_argument("--vo-attempts", type=int, default=3,
                        help="re-track when the camera track diverges; DPVO's "
                             "depth init is unseeded, so a retry is a fresh draw")
    parser.add_argument("--f-mm", type=int, default=None,
                        help="full-frame focal length hint, per GVHMR's demo")
    args = parser.parse_args()

    if (args.video is None) == (args.video_list is None):
        raise SystemExit("give exactly one of --video or --video-list")
    if args.static_cam and args.use_dpvo:
        raise SystemExit("--static-cam and --use-dpvo are mutually exclusive")

    if args.video is not None:
        videos = [args.video]
    else:
        videos = [Path(line.strip()) for line in
                  args.video_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [v for v in videos if not v.is_file()]
    if missing:
        raise SystemExit("video not found: {}".format(missing[0]))

    # Imports resolve against the GVHMR checkout (plus the pytorch3d shim);
    # importing late keeps --help fast and error messages clean.
    from hydra import compose, initialize_config_module
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.utils.video_io_utils import get_video_lwh

    import importlib.util

    demo_path = Path.cwd() / "tools" / "demo" / "demo.py"
    if not demo_path.is_file():
        raise SystemExit(
            "run from the GVHMR checkout: {} not found".format(demo_path))
    spec = importlib.util.spec_from_file_location("gvhmr_demo", demo_path)
    gvhmr_demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gvhmr_demo)

    if args.use_dpvo:
        # GVHMR's slam.py wraps the DPVO imports in a bare ``except: pass``, so
        # a missing dependency leaves ``cfg``/``DPVO``/``Timer`` unbound and the
        # only symptom is ``NameError: name 'cfg' is not defined`` raised fifty
        # lines later, once per clip, naming neither the import nor the package.
        # One missing package (pypose) cost a full diagnostic round-trip that
        # way.  Import the same three names here, where the ImportError is
        # allowed to say what it is, before any GPU time is spent.
        try:
            from dpvo.config import cfg as _dpvo_cfg  # noqa: F401
            from dpvo.dpvo import DPVO as _DPVO       # noqa: F401
            from dpvo.utils import Timer as _Timer    # noqa: F401
        except ImportError as error:
            raise SystemExit(
                "--use-dpvo but the DPVO import chain is broken: {}\n"
                "  GVHMR swallows this and fails later as \"name 'cfg' is not "
                "defined\".\n  Fix with tools/setup_dpvo_env.sh, or drop "
                "--use-dpvo to fall back to SimpleVO.".format(error))
        _make_dpvo_inference_only()

    failures = 0
    # One hydra context and one model for the whole batch; compose per clip,
    # since every clip needs its own video_name and output paths.
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        tools = (get_video_lwh, gvhmr_demo.load_data_dict, gvhmr_demo.run_preprocess,
                 compose, register_store_gvhmr)
        holder = _ModelHolder(compose)
        for video in videos:
            try:
                extract_one(video, args, tools, holder)
            except BaseException as error:  # one bad clip must not end the batch
                failures += 1
                # The message alone is not a diagnosis: "name 'cfg' is not
                # defined" names neither the file nor the branch it came from,
                # and the batch loop is the only place the traceback exists.
                # Printing it costs a few lines per failed clip and saves a
                # reproduction round-trip on every one of them.
                print("EXTRACT_FAIL {} {}".format(video.stem, error), flush=True)
                traceback.print_exc()
                if isinstance(error, KeyboardInterrupt):
                    raise
            finally:
                # A batch holds one CUDA context for hours; return each clip's
                # peak to the driver rather than letting high-water marks from
                # one long clip squeeze the next.
                torch.cuda.empty_cache()
    if len(videos) > 1:
        print("BATCH_DONE total={} failed={}".format(len(videos), failures), flush=True)
    return 1 if failures and len(videos) == 1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
