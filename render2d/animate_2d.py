"""Character still + driving pose video -> a 2D cartoon dance, with the music.

THE MODEL.  ``diffusers`` 0.40 ships ``WanAnimatePipeline`` directly, so the
ComfyUI custom nodes this is usually run through are not needed: the same Wan
Animate weights are driven from Python here, which keeps the whole 2D stage
inside this repository and inside one process.

THE WEIGHTS ARE NOT IN THIS ENVIRONMENT, and that is checked rather than
discovered halfway through a ten-clip run.  Host by host: ``pypi.org`` and
``files.pythonhosted.org`` answer 200, while ``huggingface.co``,
``cdn-lfs.huggingface.co``, ``modelscope.cn`` and github's API all fail, so
nothing can be fetched at run time.  ``--model`` therefore takes a LOCAL
directory and the script refuses early, naming what it wanted, if that
directory is not a usable Wan Animate checkpoint.  Everything upstream of the
sampler -- the pose projection, the pose video, the character still, the audio
mux -- runs without it, which is why those stages are separate scripts with
their own outputs.

FRAME BUDGET.  Wan Animate samples a fixed number of frames per call; a 20 s
clip at 30 fps is 600, far past one window.  Long clips are therefore rendered
in overlapping windows and cross-faded, with the LAST frames of one window
seeding the next so the character does not change identity mid-clip.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

FPS = 30.0
# A diffusers Wan Animate checkpoint is a directory, not a file; these are the
# parts that must be present before the sampler is worth loading.
REQUIRED = ("model_index.json",)

# THE WEIGHTS ARE ON THIS PROJECT'S OSS PREFIX, all of them.
#
# A correction worth keeping, because the wrong version of it was acted on:
# a first pass listed ``models/`` through ``head -30``, saw only the Qwen2.5-VL
# files the truncation left, and concluded the transformer backbone and the
# text encoder "were never uploaded".  They were.  Listing the same prefix with
# ``-d`` shows a complete ComfyUI Wan stack, and ``models/WAN22/`` holds
# Wan2.2-Animate-14B itself:
#
#   models/WAN22/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors  backbone
#   models/WAN22/Wan2_2_VAE_bf16.safetensors                              VAE
#   models/WAN22/WanAnimate_relight_lora_fp16.safetensors                 relight
#   models/WAN22/lightx2v_..._cfg_step_distill_rank64_bf16.safetensors    few-step
#   models/umt5_xxl_fp8_e4m3fn_scaled.safetensors                         text
#   models/Wan21_I2V_SteadyDancer_fp16.safetensors                        SteadyDancer
#
# The lesson is the cheap one: never conclude "absent" from a truncated listing.
OSS_PREFIX = os.environ.get("OSS_ASSET_PREFIX", "oss://example-bucket/example/prefix/")
OSS_WAN22 = OSS_PREFIX + "models/WAN22/"


# The staged weights are ComfyUI single files, not a diffusers directory, so
# the pipeline is assembled component by component with ``from_single_file``
# rather than ``from_pretrained``.  Names as they sit on OSS.
SINGLE_FILES = {
    "transformer": "Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors",
    "vae": "Wan2_2_VAE_bf16.safetensors",
    "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
}


def build_pipeline(directory, dtype):
    """Assemble WanAnimatePipeline from the ComfyUI single files.

    THE TEXT SIDE IS OPTIONAL HERE, and that is the point.  Wan uses UMT5, whose
    vocabulary is 256k; the only tokenizer available offline on this machine is
    the T5 one ComfyUI bundles, at 32k -- checked, not assumed, and the two are
    not interchangeable.  ``WanAnimatePipeline.__call__`` accepts
    ``prompt_embeds``/``negative_prompt_embeds`` directly, so the pipeline runs
    with a constant (zero) text conditioning and the animation is driven by the
    character image and the pose video, which is what this demo actually wants.
    Staging the UMT5 tokenizer next to its encoder on OSS is what turns the
    text prompt back on; nothing else changes.
    """
    import torch
    from diffusers import (AutoencoderKLWan, WanAnimatePipeline,
                           WanAnimateTransformer3DModel)

    present = {name: directory / filename for name, filename in SINGLE_FILES.items()}
    required = ("transformer", "vae")
    absent = [str(present[name]) for name in required if not present[name].is_file()]
    if absent:
        raise SystemExit(
            "missing component(s):\n  {}\nAll are staged at {}"
            .format("\n  ".join(absent), OSS_WAN22))

    # The converted, SHARDED transformer if it is there, else the raw single
    # file.  ``from_single_file`` needs a config.json from the hub, which this
    # machine cannot reach, so the normal path is the conversion:
    #   render2d/convert_comfy_wan.py --source <comfy .safetensors> --out <dir>
    converted = directory / "transformer"
    if (converted / "diffusion_pytorch_model.safetensors.index.json").is_file():
        # CONSTRUCTED ON ``meta``.  Instantiating 14B parameters on the CPU
        # allocates about 28 GB before a single weight is read, and the cgroup
        # limit here is 48 GB -- the process was OOM-killed with an empty log
        # while the GPU still showed 3 MiB, i.e. it never got as far as loading.
        # On ``meta`` the modules exist with no storage, and ``assign=True``
        # below hands each shard's tensors straight in.
        with torch.device("meta"):
            transformer = WanAnimateTransformer3DModel(added_kv_proj_dim=5120)
        transformer = _load_sharded(transformer, converted, dtype)
    else:
        transformer = WanAnimateTransformer3DModel.from_single_file(
            str(present["transformer"]), torch_dtype=dtype)
    # The VAE goes the same way as the transformer: ComfyUI names converted by
    # render2d/convert_comfy_vae.py, config pinned by shape (see that file), and
    # the model built on ``meta`` so nothing large lands in the cgroup's 48 GB.
    converted_vae = directory / "vae_diffusers.safetensors"
    if converted_vae.is_file():
        from safetensors.torch import load_file
        from convert_comfy_vae import CONFIG as VAE_CONFIG

        # Built directly, NOT on meta: the VAE is about 500 MB, well inside the
        # cgroup limit, and a meta build leaves its buffers (latents_mean/std,
        # which are not in the state dict) with no storage -- the later move to
        # cuda then fails with "Cannot copy out of meta tensor".
        vae = AutoencoderKLWan(**VAE_CONFIG)
        state = load_file(str(converted_vae))
        result = vae.load_state_dict(state, strict=False, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise SystemExit(
                "converted VAE does not match: {} missing, {} unexpected"
                .format(len(result.missing_keys), len(result.unexpected_keys)))
    else:
        vae = AutoencoderKLWan.from_single_file(
            str(present["vae"]), torch_dtype=torch.float32)
    scheduler = _default_scheduler()
    image_encoder, image_processor = _build_clip(directory, dtype)
    pipe = WanAnimatePipeline(
        tokenizer=None, text_encoder=None, vae=vae, scheduler=scheduler,
        image_processor=image_processor, image_encoder=image_encoder,
        transformer=transformer)
    return pipe


def _build_clip(directory, dtype):
    """CLIP-H vision tower, constructed locally and filled from the staged file.

    ``clip_vision_h.safetensors`` on OSS is already in HuggingFace's
    ``vision_model.*`` naming -- 521 tensors, width 1280, 32 layers, patch 14 --
    so unlike the transformer and the VAE it needs no renaming, only a config,
    which is the published CLIP-H one written out here rather than downloaded.
    """
    import torch
    from safetensors.torch import load_file
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
    from transformers import CLIPImageProcessor

    config = CLIPVisionConfig(
        hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
        num_attention_heads=16, image_size=224, patch_size=14,
        projection_dim=1024, hidden_act="gelu")
    model = CLIPVisionModelWithProjection(config)
    state = load_file(str(directory / "clip_vision_h.safetensors"))
    result = model.load_state_dict(state, strict=False)
    # The projection head is not in the file and is not used: ``encode_image``
    # reads ``hidden_states[-2]``, not the projected embedding.  Anything else
    # missing would mean the config is wrong, so it is reported.
    unexpected_missing = [k for k in result.missing_keys
                          if not k.startswith("visual_projection")]
    if unexpected_missing:
        raise SystemExit("CLIP vision mismatch: {} unexpected missing (first {})"
                         .format(len(unexpected_missing), unexpected_missing[0]))
    return model.to(dtype), CLIPImageProcessor(
        crop_size=224, size={"shortest_edge": 224})


def _load_sharded(model, directory, dtype):
    """Load the converted shards into an already-constructed model.

    Reported, not silent: a missing or unexpected key here means the converter's
    mapping drifted from the model's, and loading with ``strict=False`` and no
    message is how a half-loaded transformer ends up producing noise that looks
    like a bad prompt.
    """
    import torch
    import json

    from safetensors.torch import load_file

    # SHARD BY SHARD, STRAIGHT ONTO THE GPU.  Accumulating all 33 GB in a dict
    # first was OOM-killed by the cgroup (limit 48 GB, and the load needs a
    # temporary copy on top) -- the same CONSTRAINT_MEMCG kill the converter hit,
    # with the same empty log.  The card has 46 GB, so each shard is read, moved
    # to cuda and released before the next one.
    index = json.loads(
        (directory / "diffusion_pytorch_model.safetensors.index.json").read_text())
    missing, unexpected = [], []
    placed = set()
    for shard in sorted(set(index["weight_map"].values())):
        part = load_file(str(directory / shard), device="cuda")
        result = model.load_state_dict(part, strict=False, assign=True)
        placed.update(part)
        unexpected.extend(result.unexpected_keys)
        del part
    missing = [k for k in model.state_dict() if k not in placed]
    # Buffers are not in the shards, so ``assign=True`` leaves them on meta and
    # the pipeline's later ``.to("cuda")`` dies with "Cannot copy out of meta
    # tensor".  They are tiny (rope frequency tables), so they are rebuilt on
    # the device directly.
    for name, buffer in list(model.named_buffers()):
        if buffer.is_meta:
            owner = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            leaf = name.rsplit(".", 1)[-1]
            setattr(owner, leaf, torch.zeros(buffer.shape, dtype=buffer.dtype,
                                             device="cuda"))
    if missing or unexpected:
        raise SystemExit(
            "the converted weights do not match the model: {} missing, {} "
            "unexpected (first: {} / {}). Re-run convert_comfy_wan.py -- it "
            "prints the same two lists."
            .format(len(missing), len(unexpected),
                    missing[0] if missing else "-",
                    unexpected[0] if unexpected else "-"))
    return model.to(dtype)


def _default_scheduler():
    """Wan's own flow-matching schedule, constructed rather than downloaded."""
    from diffusers import UniPCMultistepScheduler

    return UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True,
        num_train_timesteps=1000, flow_shift=5.0)


def zero_text_embeds(batch, dtype, device, length=226, width=4096):
    """Constant text conditioning, since the UMT5 tokenizer is not available.

    Zeros rather than a random vector: a random one is a real prompt drawn from
    nowhere, and two runs would then differ for a reason that has nothing to do
    with the dance.
    """
    import torch

    return torch.zeros((batch, length, width), dtype=dtype, device=device)


def check_model(path):
    directory = pathlib.Path(path)
    if not directory.is_dir():
        raise SystemExit(
            "no Wan Animate checkpoint at {}.\n"
            "This machine cannot download one: pypi answers but huggingface.co, "
            "cdn-lfs.huggingface.co, modelscope.cn and github's API are all "
            "unreachable from here. Stage the weights onto local disk (or OSS, "
            "then `tools/oss_assets.py pull --cache`) and pass --model.\n"
            "Everything upstream of the sampler runs without it: "
            "project_pose_2d.py, draw_pose_video.py and make_character_image.py "
            "each write their own output.".format(directory))
    # A ComfyUI layout is a directory of single files; a diffusers layout has
    # model_index.json.  Either is accepted, and which one was found is printed,
    # because loading the wrong one fails deep inside the loader with a message
    # about tensor names rather than about layout.
    # Only the components this pipeline actually loads.  The UMT5 encoder is in
    # SINGLE_FILES for documentation, but it is deliberately NOT required: its
    # tokenizer is not available offline, so the run passes ``prompt_embeds``
    # instead.  Requiring it here made a complete checkpoint look incomplete.
    needed = ("transformer", "vae")
    if all((directory / SINGLE_FILES[name]).is_file() for name in needed):
        return directory
    missing = [name for name in REQUIRED if not (directory / name).is_file()]
    if missing:
        raise SystemExit(
            "{} is not a diffusers checkpoint; missing {}.\n"
            "Wan2.2-Animate-14B IS staged at {} (backbone, VAE, relight LoRA, "
            "few-step LoRA) with the UMT5 text encoder one level up; pull those "
            "and point --model at the directory."
            .format(directory, ", ".join(missing), OSS_WAN22))
    return directory


def load_pose_frames(pose_video, width=None, height=None):
    """Decode the driving pose video into RGB frames, through ffmpeg.

    NOT ``imageio``/``pyav``: on this machine that call returns nothing and
    exits 0 while the renders are using the cores, which is the worst possible
    failure -- a stage that reports success having produced no frames.  ffmpeg
    is already a hard dependency of every other stage here and its frame count
    can be checked against ffprobe, so the decode either matches or fails loudly.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,width,height", "-of", "csv=p=0",
         str(pose_video)], capture_output=True, text=True, check=True).stdout.strip()
    w, h, count = (int(float(v)) for v in probe.split(",")[:3])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(pose_video), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8)
    expected = count * h * w * 3
    if frames.size != expected:
        raise SystemExit(
            "decoded {} bytes from {} but ffprobe counted {} frames of {}x{} "
            "({} bytes); the decode and the container disagree"
            .format(frames.size, pose_video, count, w, h, expected))
    return list(frames.reshape(count, h, w, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", required=True, help="still .png")
    ap.add_argument("--pose-video", required=True, help="from draw_pose_video.py")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="local Wan Animate checkpoint")
    ap.add_argument("--window", type=int, default=77,
                    help="frames per sampler call; Wan's own window is 81 and "
                         "77 leaves room for the 4-frame cross-fade")
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--prompt", default="a 2D anime character dancing, clean "
                                        "flat cel shading, plain background")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the inputs and print the plan without loading "
                         "the model -- the stage's own smoke test")
    args = ap.parse_args()

    from PIL import Image

    character = Image.open(args.character).convert("RGB")
    pose = load_pose_frames(args.pose_video, character.width, character.height)
    windows = max(1, int(np.ceil((len(pose) - args.overlap)
                                 / max(1, args.window - args.overlap))))
    print("character {}x{} | pose {} frames ({:.1f}s) | {} window(s) of {}"
          .format(character.width, character.height, len(pose),
                  len(pose) / FPS, windows, args.window))
    if args.dry_run:
        print("dry run: inputs are consistent; the sampler was not loaded")
        return

    directory = check_model(args.model)
    import torch

    pipe = build_pipeline(directory, torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()

    # The pipeline's OWN segmenting, not a hand-rolled sliding window: it
    # carries conditioning frames from one segment into the next
    # (``prev_segment_conditioning_frames``), which is what keeps the character
    # from changing identity mid-clip.  A window written here would have to
    # re-derive that and would get it subtly wrong.
    embeds = zero_text_embeds(1, torch.bfloat16, pipe._execution_device)
    # ``face_video`` is REQUIRED by this pipeline even though nothing here
    # drives the face: the demo animates a body from a pose video and has no
    # face reference.  It is therefore the character's OWN face, held still --
    # a constant, which leaves the expression as the still image has it rather
    # than inventing motion the dance does not contain.  Sized to the
    # transformer's declared ``motion_encoder_size``.
    # The pipeline indexes ``pose_video[0].size``, i.e. it wants PIL images, not
    # the numpy frames the ffmpeg decode produces.
    pose = [Image.fromarray(frame) for frame in pose]

    face_size = int(getattr(pipe.transformer.config, "motion_encoder_size", 512))
    face_still = character.resize((face_size, face_size))
    face = [face_still] * len(pose)

    result = pipe(
        image=character,
        pose_video=pose,
        face_video=face,
        prompt_embeds=embeds,
        negative_prompt_embeds=embeds,
        height=character.height, width=character.width,
        segment_frame_length=args.window,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        output_type="np",
    )
    produced = [(frame * 255).astype(np.uint8) for frame in result.frames[0]]

    from diffusers.utils import export_to_video

    silent = pathlib.Path(args.out).with_suffix(".silent.mp4")
    export_to_video(produced, str(silent), fps=FPS)
    if args.audio and pathlib.Path(args.audio).is_file():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent),
                        "-i", args.audio, "-c:v", "copy", "-c:a", "aac",
                        "-shortest", args.out], check=True)
        silent.unlink()
    else:
        silent.rename(args.out)
    print("{} frames -> {}".format(len(produced), args.out))


if __name__ == "__main__":
    main()
