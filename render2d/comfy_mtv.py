"""Drive MTV-Crafter with OUR 3D joints: the depth the 2D path cannot carry.

WHY A THIRD DRIVER.  ``comfy_steadydancer.py`` conditions on a picture of a flat
skeleton.  Measured on the fixed ten clips, an orthographic projection keeps
only 80% of the wrists' motion and 77% of the body's -- everything along the
view axis is gone, and no amount of line width or framing brings it back.
MTV-Crafter conditions on the joint COORDINATES: its tokenizer's first layer is
``encoder.conv_in.weight`` [128, 3, 3, 3], three channels of raw XYZ with no
rendering step, and in its own training statistics the depth axis has the
largest spread of the three (x 243.2 mm, y 335.3 mm, z 618.0 mm).

VALIDATED BEFORE THE FIRST RENDER, because everything downstream is conditioned
on what survives the codebook: round-tripping the ten clips through the VQ-VAE
alone gives a median per-joint error of 72.1 mm for our generated motion against
51.9 mm for real captured ground truth -- the same order, and both below the
~100 mm this repository has measured as the threshold of visibility.  Depth
survives too (40.9 mm against 27.0 mm lateral).  ``tools/score_mtv_roundtrip.py``.

WHAT THIS REWIRES IN KIJAI'S EXAMPLE.  The example is marked WIP and was built
around a different machine, so nothing in it is taken on trust:

  * the whole NLF branch is removed.  It ESTIMATES 3D joints from a video; we
    already have them, and routing ours through it would mean drawing them to
    pixels and guessing them back -- losing the depth this path exists to keep,
    and pulling a 493 MB model to do it.  ``AtomicDanceLoadJoints3D`` supplies
    the ``NLFPRED`` payload directly.
  * every model name is remapped: the example asks for Windows paths
    (``wanvideo\\...``), a VAE without the ``WanVideo_`` prefix that the real
    object has, a ``umt5_xxl_fp16`` we do not hold, and a MAGREF base we do not
    hold.  The base becomes the Wan2.1 I2V-720p fp8 already on disk -- the
    adapter is 120 tensors of ``motion_attn``/``norm4`` on blocks 0,4,...,36
    with no backbone weights, so it merges onto it.
  * the preview branches (pose drawing, side-by-side combines) are dropped; a
    graph is rejected if any OUTPUT node cannot be validated.

THE BASE, SETTLED 2026-09-17.  The first renders put the adapter on the stock
``Wan2_1-I2V-14B-720p`` base: it loaded without a word and the video stood
still.  The base's own tensor list was the only thing that could tell: 0 of its
1784 tensors are ``motion_attn``/``norm4``, so the adapter's weights were merged
into a backbone with no motion cross-attention to run them.  The MTV-finetuned
base (``Wan2_1-I2V-14B-MTV-Crafter_fp8_e4m3fn_scaled_KJ``) has that path, and
``assert_motion_path`` now refuses any base that does not -- "it loaded" is not
evidence that it performs.

"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from render2d.comfy_animate import (  # noqa: E402
    SERVER, COMFY_ROOT, load_schema, api_format, fix_model_names, post, wait, probe)

WORKFLOW = (COMFY_ROOT / "custom_nodes/ComfyUI-WanVideoWrapper/example_workflows"
            / "wanvideo_2_1_14B_MTV_Crafter_example_WIP.json")

NODE_VQVAE = 146        # LoadVQVAE
NODE_ENCODE = 147       # MTVCrafterEncodePoses
NODE_IMAGE = 159        # LoadImage -- the character still
NODE_RESIZE = 160       # ImageResizeKJv2 on the character
NODE_EMBEDS = 157       # WanVideoImageToVideoEncode
NODE_MODEL = 197        # WanVideoModelLoader
NODE_EXTRA = 198        # WanVideoExtraModelSelect
NODE_SAMPLER = 154      # WanVideoSampler
NODE_DECODE = 161       # WanVideoDecode
NODE_SAVE = 162         # VHS_VideoCombine -- repointed at the decode
NODE_MOTION = 167       # WanVideoAddMTVMotion
NODE_TEXT = 163         # WanVideoTextEncodeCached
NODE_LORA = 156         # WanVideoLoraSelect
NODE_VAE = 158          # WanVideoVAELoader
NODE_BLOCKSWAP = 164    # WanVideoSetBlockSwap -- the model before the LoRAs
NODE_WIDTH, NODE_HEIGHT = 176, 177      # INTConstant
NODE_JOINTS = 900       # the node we add

# Upstream MTV-Crafter infers in 512-area buckets (384x672 at this aspect).
WIDTH, HEIGHT = (int(v) for v in os.environ.get("MTV_SIZE", "480x832").split("x"))
# The frame rate the joints were resampled to (mtv_motion.py --fps); the video is written at it.
MODEL_FPS = float(os.environ.get("MTV_FPS", "16"))
# Upstream feeds the reference latent without noise.
NOISE_AUG = float(os.environ.get("MTV_NOISE_AUG", "0.0"))
BASE_MODEL = os.environ.get(
    "MTV_BASE", "WanVideo/Wan2_1-I2V-14B-MTV-Crafter_fp8_e4m3fn_scaled_KJ.safetensors")
# "none" loads the base alone -- the finetuned base already carries the motion path.
ADAPTER = os.environ.get("MTV_ADAPTER", "none")
MOTION_KEYS = ("motion_attn", "norm4")
VQVAE = "WanVideo_MTV_Crafter_4DMoT_VQVAE_fp32.safetensors"
TEXT_ENCODER = "umt5-xxl-enc-bf16.safetensors"
LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
VAE = "wan_2.1_vae.safetensors"
POSITIVE = ("a girl dancing, facing the camera, full body in frame, front view, "
            "steady camera, clean background")
# Upstream MTV-Crafter's negative prompt.  It only does anything when the guidance
# pass is pointed at it (MTV_NEG_GUIDANCE=1): at cfg 1 a cfg-distilled LoRA never
# evaluates the negative branch, so "bad quality video" was decoration.
NEGATIVE = "bad hands, extra limbs, fused fingers, blurry, low quality"
STEPS = int(os.environ.get("MTV_STEPS", "6"))
CFG = float(os.environ.get("MTV_CFG", "1.0"))
MOTION_STRENGTH = float(os.environ.get("MTV_STRENGTH", "1.2"))
SEED = int(os.environ.get("MTV_SEED", "0"))
# "FRAMES,OVERLAP" in pixel frames, e.g. "81,16"; empty = one window over the clip.
CONTEXT = os.environ.get("MTV_CONTEXT", "")
NODE_CONTEXT = 901      # the WanVideoContextOptions node we add when CONTEXT is set
NODE_CLIP_ENCODE = 171  # WanVideoClipVisionEncode, image_1 <- the resized character
# The finetune rewrote cross_attn.v_img in every block, so it was trained WITH the
# image embedding; the example leaves the encoder's output unconnected.
USE_CLIP = os.environ.get("MTV_CLIP", "1") != "0"
# Guidance on the motion axis (needs render2d/patches/wanvideowrapper_mtv.patch);
# 1.0 = off, and every value above it costs one extra transformer pass per step.
MOTION_CFG = float(os.environ.get("MTV_MOTION_CFG", "1.0"))
# The guidance pass can point away from the negative prompt as well as away from
# no-motion -- one pass, both axes.
NEG_GUIDANCE = os.environ.get("MTV_NEG_GUIDANCE", "0") != "0"
# Upstream overwrites each later context window's first frame with the reference
# image's conditioning; with MTV motion that frame is mid-dance.  "0" leaves the
# window's own conditioning alone.
WINDOW_REF = os.environ.get("MTV_WINDOW_REF", "on")   # on | soft | off
ROPE_PATCH = COMFY_ROOT / "custom_nodes/ComfyUI-WanVideoWrapper/nodes_sampler.py"


def rope_patched():
    """Is render2d/patches/wanvideowrapper_mtv_freqs.patch applied?  Unpatched, the
    motion attention's rotation silently becomes a cos(theta) scaling."""
    text = ROPE_PATCH.read_text()
    return ("mtv_freqs = mtv_freqs.to(device)\n" in text
            and "mtv_freqs.to(device, dtype)" not in text
            and '"motion_cfg"' in text and '"window_reference"' in text)


def tensor_names(path):
    """The tensor names a safetensors file declares, read from its header only."""
    import struct
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return [k for k in header if k != "__metadata__"]


def assert_motion_path(base, adapter):
    """Refuse a model whose backbone has nowhere to put the motion tokens.

    The adapter only ADDS weights; the base decides whether any layer reads them.
    Counted on the base alone, because an adapter merged into a base without the
    path is exactly the silent static-video case this gate exists for.
    """
    root = COMFY_ROOT / "models" / "diffusion_models"
    names = tensor_names(root / base)
    count = sum(any(k in n for k in MOTION_KEYS) for n in names)
    if count == 0:
        raise SystemExit(
            "{} has 0 of its {} tensors on the motion path ({}); the adapter would "
            "load and do nothing -- refusing".format(base, len(names), "/".join(MOTION_KEYS)))
    adapter_count = 0
    if adapter and count:
        # sd.update keeps the base's motion *.scale_weight, so a bf16 adapter merged on
        # top is multiplied by 0.02-0.05 and the motion branch goes almost silent.
        raise SystemExit("{} already carries {} motion tensors; merging an adapter on top "
                         "would be rescaled by the base's fp8 scales -- use MTV_ADAPTER=none"
                         .format(base, count))
    if adapter:
        adapter_count = sum(any(k in n for k in MOTION_KEYS)
                            for n in tensor_names(root / adapter))
    return {"base_tensors": len(names), "base_motion_tensors": count,
            "adapter_motion_tensors": adapter_count}


def rewire(graph, joints_path):
    """Feed our own joints in and drop everything that existed to estimate them."""
    by_id = {n["id"]: n for n in graph["nodes"]}
    links = {l[0]: l for l in graph["links"]}
    next_link = max(links, default=0) + 1

    graph["nodes"].append({
        "id": NODE_JOINTS, "type": "AtomicDanceLoadJoints3D",
        "widgets_values": [str(joints_path)],
        "inputs": [], "outputs": [{"name": "pose_results", "type": "NLFPRED"},
                                  {"name": "frames", "type": "INT"}],
        "mode": 0,
    })

    def repoint(node_id, slot_name, source_id, source_slot, kind):
        nonlocal next_link
        node = by_id[node_id]
        for slot in node.get("inputs", []) or []:
            if slot["name"] != slot_name:
                continue
            graph["links"] = [l for l in graph["links"] if l[0] != slot.get("link")]
            graph["links"].append([next_link, source_id, source_slot, node_id, 0, kind])
            slot["link"] = next_link
            next_link += 1
            return
        raise SystemExit("node {} has no input {!r}".format(node_id, slot_name))

    repoint(NODE_ENCODE, "poses", NODE_JOINTS, 0, "NLFPRED")
    repoint(NODE_EMBEDS, "num_frames", NODE_JOINTS, 1, "INT")
    # The character's own size, not a size derived from a driving video.
    repoint(NODE_RESIZE, "width", NODE_WIDTH, 0, "INT")
    repoint(NODE_RESIZE, "height", NODE_HEIGHT, 0, "INT")
    repoint(NODE_SAVE, "images", NODE_DECODE, 0, "IMAGE")

    drop = {"NLFPredict", "DownloadAndLoadNLFModel", "LoadNLFModel", "DrawNLFPoses",
            "VHS_LoadVideo", "ImageConcatMulti", "VHS_SplitImages",
            "GetLatentRangeFromBatch", "GetImageRangeFromBatch"}
    graph["nodes"] = [n for n in graph["nodes"]
                      if n["type"] not in drop
                      and not (n["type"] == "VHS_VideoCombine" and n["id"] != NODE_SAVE)
                      and not (n["type"] == "ImageResizeKJv2" and n["id"] != NODE_RESIZE)]
    alive = {n["id"] for n in graph["nodes"]}
    graph["links"] = [l for l in graph["links"] if l[1] in alive and l[3] in alive]
    live = {l[0] for l in graph["links"]}
    for node in graph["nodes"]:
        for slot in node.get("inputs", []) or []:
            if slot.get("link") is not None and slot["link"] not in live:
                slot["link"] = None
    return graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", required=True, help="a file under ComfyUI input/")
    ap.add_argument("--joints", required=True, help="[T,24,3] camera-frame .npy")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0, help="first model frame (16 fps) to render")
    ap.add_argument("--max-frames", type=int, default=0, help="render at most this many model frames")
    ap.add_argument("--reverse", action="store_true",
                    help="play the joints backwards -- the control that separates following from generic motion")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import numpy as np
    joints = np.load(args.joints)
    if args.start or args.max_frames or args.reverse:
        stop = args.start + args.max_frames if args.max_frames else len(joints)
        joints = joints[args.start:stop]
        if args.reverse:
            joints = joints[::-1].copy()
        sliced = pathlib.Path(args.out).with_suffix(".joints_{}_{}{}.npy".format(
            args.start, stop, "_rev" if args.reverse else ""))
        sliced.parent.mkdir(parents=True, exist_ok=True)
        np.save(sliced, joints)
        args.joints = str(sliced)
    adapter = None if ADAPTER.lower() == "none" else ADAPTER
    motion_path = assert_motion_path(BASE_MODEL, adapter)
    print("motion path: base {base_motion_tensors}/{base_tensors} tensors, "
          "adapter {adapter_motion_tensors}".format(**motion_path))
    print("joints: {} frames at {} fps ({:.1f} s)".format(
        len(joints), MODEL_FPS, len(joints) / MODEL_FPS))

    load_schema()
    graph = rewire(json.loads(WORKFLOW.read_text()),
                   pathlib.Path(args.joints).resolve())
    prompt = api_format(graph)

    prompt[str(NODE_JOINTS)]["inputs"]["path"] = str(pathlib.Path(args.joints).resolve())
    prompt[str(NODE_IMAGE)]["inputs"]["image"] = pathlib.Path(args.character).name
    prompt[str(NODE_VQVAE)]["inputs"]["model_name"] = VQVAE
    # THE WAN2.1 VAE, PINNED.  The example asks for a Windows path that does not
    # resolve here, and the name-repair fallback then matched it to
    # ``Wan2_2_VAE_bf16`` on a shared prefix -- the Wan2.2 VAE, which has 48
    # latent channels against Wan2.1's 16, about to be handed to a Wan2.1 I2V
    # base.  A repair that picks the nearest name is right for a typo and wrong
    # across a model generation, so this one is stated outright.
    prompt[str(NODE_VAE)]["inputs"]["model_name"] = VAE
    prompt[str(NODE_MODEL)]["inputs"]["model"] = BASE_MODEL
    prompt[str(NODE_MODEL)]["inputs"]["base_precision"] = "bf16"
    prompt[str(NODE_MODEL)]["inputs"]["quantization"] = "disabled"
    prompt[str(NODE_MODEL)]["inputs"]["load_device"] = "main_device"
    if adapter:
        prompt[str(NODE_EXTRA)]["inputs"]["extra_model"] = adapter
    else:
        del prompt[str(NODE_EXTRA)]
        prompt[str(NODE_MODEL)]["inputs"].pop("extra_model", None)
    prompt[str(NODE_TEXT)]["inputs"]["model_name"] = TEXT_ENCODER
    prompt[str(NODE_TEXT)]["inputs"]["positive_prompt"] = POSITIVE
    prompt[str(NODE_TEXT)]["inputs"]["negative_prompt"] = NEGATIVE
    if os.environ.get("MTV_NO_LORA"):
        # The lightx2v distill LoRA applies 1261 weight patches to the very
        # blocks the motion adapter writes into.  Dropping it is the second
        # cheapest test of "the adapter is being drowned out" -- costing more
        # steps, since the distill is what makes 6 steps enough.
        for node_id, node in list(prompt.items()):
            if node["class_type"] in ("WanVideoLoraSelect", "WanVideoSetLoRAs"):
                del prompt[node_id]
        for node in prompt.values():
            for name, value in list(node["inputs"].items()):
                if isinstance(value, list) and len(value) == 2 and value[0] not in prompt:
                    del node["inputs"][name]
        prompt[str(NODE_SAMPLER)]["inputs"]["model"] = [str(NODE_BLOCKSWAP), 0]
    else:
        prompt[str(NODE_LORA)]["inputs"]["lora"] = LORA
    prompt[str(NODE_WIDTH)]["inputs"]["value"] = WIDTH
    prompt[str(NODE_HEIGHT)]["inputs"]["value"] = HEIGHT
    embeds = prompt[str(NODE_EMBEDS)]["inputs"]
    embeds["width"], embeds["height"] = WIDTH, HEIGHT
    embeds["noise_aug_strength"] = NOISE_AUG
    if USE_CLIP:
        embeds["clip_embeds"] = [str(NODE_CLIP_ENCODE), 0]
    sampler = prompt[str(NODE_SAMPLER)]["inputs"]
    sampler["steps"], sampler["cfg"] = STEPS, CFG
    sampler["seed"] = SEED
    if CONTEXT:
        frames, overlap = (int(v) for v in CONTEXT.split(","))
        prompt[str(NODE_CONTEXT)] = {"class_type": "WanVideoContextOptions", "inputs": {
            "context_schedule": "uniform_standard", "context_frames": frames,
            "context_stride": 4, "context_overlap": overlap, "freenoise": True,
            "verbose": False, "fuse_method": "linear"}}
        sampler["context_options"] = [str(NODE_CONTEXT), 0]
    prompt[str(NODE_MOTION)]["inputs"]["strength"] = MOTION_STRENGTH
    if MOTION_CFG > 1.0:
        prompt[str(NODE_MOTION)]["inputs"]["motion_cfg"] = MOTION_CFG
        prompt[str(NODE_MOTION)]["inputs"]["negative_guidance"] = NEG_GUIDANCE
    prompt[str(NODE_MOTION)]["inputs"]["window_reference"] = WINDOW_REF
    save = prompt[str(NODE_SAVE)]["inputs"]
    save["frame_rate"] = MODEL_FPS
    save["loop_count"] = 0
    save["save_output"] = True
    save["filename_prefix"] = "atomicdance_mtv"

    for node in prompt.values():
        for name, value in list(node["inputs"].items()):
            if isinstance(value, list) and len(value) == 2 and value[0] not in prompt:
                del node["inputs"][name]
    prompt = fix_model_names(prompt)
    # fix_model_names repairs by nearest name; the three that decide what runs are
    # re-checked here so a "repair" can never swap the base or the adapter.
    if prompt[str(NODE_MODEL)]["inputs"]["model"] != BASE_MODEL:
        raise SystemExit("base renamed to {!r}".format(prompt[str(NODE_MODEL)]["inputs"]["model"]))
    if adapter and prompt[str(NODE_EXTRA)]["inputs"]["extra_model"] != adapter:
        raise SystemExit("adapter renamed to {!r}".format(prompt[str(NODE_EXTRA)]["inputs"]["extra_model"]))
    record = {
        "joints": str(pathlib.Path(args.joints).resolve()),
        "joints_sha1": hashlib.sha1(np.load(args.joints).tobytes()).hexdigest(),
        "frames": int(len(joints)), "start": args.start, "reverse": args.reverse, "character": args.character,
        "base": BASE_MODEL, "adapter": adapter, "motion_path": motion_path,
        "lora": None if os.environ.get("MTV_NO_LORA") else LORA,
        "steps": STEPS, "cfg": CFG, "strength": MOTION_STRENGTH, "motion_cfg": MOTION_CFG, "neg_guidance": NEG_GUIDANCE,
        "window_reference": WINDOW_REF, "negative": NEGATIVE, "seed": SEED,
        "context": CONTEXT or None, "clip": USE_CLIP, "rope_patch": rope_patched(), "width": WIDTH, "height": HEIGHT, "fps": MODEL_FPS, "noise_aug": NOISE_AUG,
        "prompt": POSITIVE,
    }

    if not rope_patched() and not os.environ.get("MTV_ALLOW_UNPATCHED_ROPE"):
        raise SystemExit("ComfyUI-WanVideoWrapper is missing render2d/patches/"
                         "wanvideowrapper_mtv.patch; the motion RoPE would be cos-only")

    if args.dry_run:
        print(json.dumps(prompt, indent=1))
        return

    result = post(prompt)
    print("queued", result.get("prompt_id"))
    entry = wait(result["prompt_id"], timeout=14400)
    produced = []
    for node_output in entry.get("outputs", {}).values():
        for item in node_output.get("gifs", []) + node_output.get("videos", []):
            produced.append(pathlib.Path(item["fullpath"]) if item.get("fullpath")
                            else COMFY_ROOT / item.get("type", "output")
                            / item.get("subfolder", "") / item["filename"])
    produced = [p for p in produced if p.is_file()]
    if not produced:
        raise SystemExit("the workflow completed but wrote no video; outputs were "
                         + json.dumps(entry.get("outputs", {}))[:500])
    newest = max(produced, key=lambda p: p.stat().st_mtime)
    target = pathlib.Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.audio and pathlib.Path(args.audio).is_file():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(newest),
                        "-i", args.audio, "-c:v", "copy",
                        "-c:a", "libmp3lame", "-b:a", "128k",
                        "-shortest", str(target)], check=True)
    else:
        target.write_bytes(newest.read_bytes())
    info = probe(target)
    record.update({"video": str(target), "source": str(newest),
                   "video_frames": info["frames"], "video_fps": info["fps"]})
    target.with_suffix(".json").write_text(json.dumps(record, indent=1))
    print("{} -> {} ({} frames at {:g} fps = {:.1f} s)".format(
        newest.name, target, info["frames"], info["fps"],
        info["frames"] / info["fps"]))


if __name__ == "__main__":
    main()
