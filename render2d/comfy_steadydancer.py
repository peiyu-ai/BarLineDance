"""Drive the repository's Wan2.1 SteadyDancer workflow with OUR pose video.

WHICH MODEL, AND WHY NOT THE OTHER ONE.  ``../ComfyUI_Wan`` ships two families.
Wan2.2-Animate takes a driving VIDEO and re-enacts it; Wan2.1 SteadyDancer takes
a POSE SEQUENCE and drives a character with it, which is what we have.  The
distinction is in the weights, not in taste -- compared tensor by tensor,
``Wan21_SteadyDancer_fp8_e4m3fn_scaled_KJ`` carries 1834 tensors against the
base ``Wan2_1-I2V-14B-720p``'s 1784, and the extra 50 are exactly the pose
conditioning path: ``condition_embedding_align.*`` (cross-attention, proj_p,
proj_r), ``condition_embedding_spatial.*`` (conv_p, conv_q, fc_scale),
``condition_embedding_temporal.*``, plus ``patch_embedding_fuse`` and
``patch_embedding_ref_c``.  ``WanVideoAddSteadyDancerEmbeds`` puts the pose
latent into ``sdancer_embeds``, and ONLY those 50 tensors read it -- so the base
I2V model would silently ignore the pose entirely.

WHERE OUR POSE GOES IN.  The workflow's pose branch is

    VHS_LoadVideo -> PoseAndFaceDetection (yolo+ViTPose onnx) -> DrawViTPose
                  -> ImageResizeKJv2 -> "poses" -> WanVideoEncode (VAE) -> LATENT

so the model is conditioned on a PICTURE of a skeleton, and the detector exists
only to produce that picture from real footage.  ``aapose_video.py`` already
draws it with the same vendor function ``DrawViTPose`` calls, so the two
detection nodes are removed and the loaded frames feed the resize directly.
This also drops the only part of the graph that needed the ONNX models.

THE FRAME RATE.  Wan2.1 is a 16 fps model and AtomicDance is 30 fps.  Both ends
are set to 16: ``force_rate`` resamples the 30 fps pose video on load, and the
combiner writes at 16, so the DURATION is preserved (a 20.6 s dance stays
20.6 s) and the temporal spacing the model sees is the one it was trained on.
Leaving the loader at 30 and the writer at 16 is the trap -- that is what turns
a 20.6 s dance into a 38.7 s one.

MEMORY.  The earlier Wan2.2-Animate attempt was OOM-killed by the container's
48 GB cgroup limit (``/sys/fs/cgroup/memory/memory.limit_in_bytes`` =
51539607552), because the native ComfyUI loader dequantises fp8 to bf16 in CPU
RAM.  This path does not go there: ``WanVideoModelLoader`` with
``quantization='disabled'`` AUTOSELECTS by the weights, and these carry a
``scaled_fp8`` tensor, so they stay fp8 (16.4 GB); ``load_device='main_device'``
streams them straight onto the L20's 46 GB.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# The UI-graph -> API-format conversion is generic and was expensive to get
# right (Get/Set flattening, widget order from the server's own schema, the
# seed's control-mode entry); it is shared rather than copied.
from render2d.comfy_animate import (              # noqa: E402
    SERVER, COMFY_ROOT, load_schema, api_format, fix_model_names, post, wait, probe)

WORKFLOW = COMFY_ROOT / ("user/default/workflows/"
                         "wanvideo_2_1_14B_SteadyDancer_pose_control_example_01.json")

NODE_VIDEO = 75      # VHS_LoadVideo      -- our AAPose pose video
NODE_IMAGE = 76      # LoadImage          -- the character still
NODE_SIZE = 91       # GetImageSizeAndCount -- the loaded pose frames
NODE_POSE_RESIZE = 77  # ImageResizeKJv2  -- pose branch, fed by DrawViTPose
NODE_DECODE = 28     # WanVideoDecode     -- the animation itself
NODE_SAVE = 83       # VHS_VideoCombine   -- repointed at the decode output
NODE_PREVIEW = 117   # VHS_VideoCombine   -- a second copy of the pose video
NODE_SAMPLER = 119   # WanVideoSamplerSettings
NODE_SCHEDULER = 122  # WanVideoScheduler
NODE_BLOCKSWAP = 39  # WanVideoBlockSwap
NODE_MODEL_LOADER = 22  # WanVideoModelLoader -- attention_mode
NODE_COMPILE = 35    # WanVideoTorchCompileSettings
NODE_DECODE_NODE = 28  # WanVideoDecode

# CFG 1.0, NOT the graph's 5, and this is a correctness fix rather than a
# preference.  The saved graph leaves ``pose_latents_negative`` unconnected, so
# ``WanVideoAddSteadyDancerEmbeds`` stores ``cond_neg = None``; the model then
# does ``sdancer_input["cond_neg"]`` on the unconditional pass and dies with
# ``'NoneType' object has no attribute 'unsqueeze'`` (model.py:2511).  At cfg
# 1.0 the sampler skips that pass entirely (nodes_sampler.py:1500,
# ``math.isclose(cfg_scale, 1.0)``), which is also what a cfg-step-DISTILL LoRA
# wants -- the graph loads ``lightx2v_I2V_14B_480p_cfg_step_distill`` -- and it
# halves the compute.
CFG = 1.0
# The distill LoRA's own step count, and the value the graph's sampler node
# already carries; its scheduler node disagrees at 30, so both are set here
# rather than left to whichever one wins.
STEPS = 4
# The L20 has 46 GB and the fp8 weights are 16.4 GB, so nothing needs to live in
# CPU RAM.  The graph ships blocks_to_swap 35 of 40 (a 24 GB-card setting) and
# that streams most of the model over PCIe on every step -- the first step of
# the failed run took 68 s.
BLOCKS_TO_SWAP = 0
NODE_TEXT = 92       # WanVideoTextEncodeCached

# THE POSITIVE PROMPT IS EMPTY IN THE SAVED GRAPH, and that is not a neutral
# setting for a text-conditioned model: with nothing said, the only thing
# telling Wan which way the body faces is a stick figure, and a stick figure is
# front/back ambiguous by construction (orthographic joints carry no depth).
# The operator caught the consequence in the animation -- the character turned
# her back on a clip whose motion never turns away (measured with the
# renderer's own body_facing: median facing 1.2 degrees off the camera, 0.0% of
# frames back-to-camera).  The graph's NEGATIVE prompt already names the
# failure it expects ("倒着走"), so the positive side is where the camera
# relationship belongs.
POSITIVE_PROMPT = ("a girl dancing, facing the camera, full body in frame, "
                   "front view, steady camera, clean background")

# FACING-AWARE TEXT (--facing-yaw).  The front-view positive above and the back-view negative the drivers add
# were both aimed at PHANTOM back views, and both are one sentence for the whole clip -- so during a real turn
# they tell the model "front view" while the stick figure shows a back.  Measured 2026-09-24 (27 renders, 7,390
# frames, DEFECTS 96.6): wrong 0.2-0.4% below 45 deg from the camera, 57-88% past 90 deg, almost all a face
# painted on a turned-away body.  With --facing-yaw every context window gets its own prompt (the wrapper's
# '|' prompt travel, one prompt per latent frame, so a window uses the prompt of its last latent): the front
# prompt where the window's body stays within TURN_YAW of the camera, TURN_PROMPT -- which names no facing --
# where it turns further.  The yaw is our own generated motion's, never a target's.
TURN_PROMPT = "a girl dancing and turning around, full body in frame, steady camera, clean background"
TURN_YAW = 67.0

# The pose branch's own strengths, raised from the graph's 1.0: the same
# ambiguity is what these control, and the pose is the one input we are certain
# about.
POSE_STRENGTH_SPATIAL = 1.0
POSE_STRENGTH_TEMPORAL = 1.0
NODE_SDANCER = 71    # WanVideoAddSteadyDancerEmbeds
NODE_CONTEXT = 87    # WanVideoContextOptions
NODE_POSE_ENCODE = 72  # WanVideoEncode   -- the pose frames -> pose latents
NODE_POSE_CLIP = 82    # WanVideoClipVisionEncode -- CLIP of the first pose frame

MODEL_FPS = 16       # Wan2.1's native rate


def reference_pose_image(character, width, height):
    """The character still's OWN pose, drawn exactly as the pose frames are.

    The landmarks are the workflow's ViTPose on the still (cached beside it by
    ``aapose_video.character_landmarks``), i.e. what the detector-driven
    pipeline SteadyDancer was trained on would have drawn for this image.  Same
    vendor drawing function and stick width as ``run_2d_steadydancer.sh`` passes
    to ``aapose_video.py``.  Staged under a content name, like the pose video.
    """
    import numpy as np
    from PIL import Image
    from render2d.aapose_video import (character_landmarks, AAPoseMeta,
                                       draw_aapose_by_meta_new, metas_from_uv)
    landmarks = character_landmarks(str(COMFY_ROOT / "input" / character), width, height)
    uv = (landmarks[:, :2] / np.array([width, height]))[None]
    meta = metas_from_uv(uv, width, height)[0]
    meta["keypoints_body"][:, 2] = landmarks[:, 2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    frame = draw_aapose_by_meta_new(canvas, AAPoseMeta.from_humanapi_meta(meta),
                                    draw_hand=False, draw_head=True,
                                    body_stick_width=5, hand_stick_width=5)
    image = Image.fromarray(np.ascontiguousarray(frame))
    staged = COMFY_ROOT / "input" / "refpose_{}.png".format(
        hashlib.sha1(np.ascontiguousarray(frame).tobytes()).hexdigest()[:16])
    if not staged.is_file():
        image.save(staged)
    return staged


def use_our_pose(graph):
    """Remove the detector and feed the loaded pose frames to the pose branch."""
    by_id = {n["id"]: n for n in graph["nodes"]}
    links = {l[0]: l for l in graph["links"]}
    next_link = max(links, default=0) + 1

    def repoint(node_id, slot_name, source_id, source_slot, kind):
        nonlocal next_link
        node = by_id[node_id]
        for slot in node.get("inputs", []) or []:
            if slot["name"] != slot_name:
                continue
            graph["links"] = [l for l in graph["links"] if l[0] != slot.get("link")]
            graph["links"].append([next_link, source_id, source_slot,
                                   node_id, 0, kind])
            slot["link"] = next_link
            next_link += 1
            return
        raise SystemExit("node {} has no input {!r}".format(node_id, slot_name))

    # The pose picture is ours, so the resize reads the loaded frames.
    repoint(NODE_POSE_RESIZE, "image", NODE_SIZE, 0, "IMAGE")
    # The saved video is the ANIMATION, not the workflow's side-by-side strip
    # (result | character | poses): the pose video already exists on its own.
    repoint(NODE_SAVE, "images", NODE_DECODE, 0, "IMAGE")

    # Both detector nodes and the ONNX loader that serves them.  Nothing else
    # consumes them once the resize is repointed.
    drop = {"PoseAndFaceDetection", "DrawViTPose", "OnnxDetectionModelLoader"}
    graph["nodes"] = [n for n in graph["nodes"]
                      if n["type"] not in drop and n["id"] != NODE_PREVIEW]
    alive = {n["id"] for n in graph["nodes"]}
    graph["links"] = [l for l in graph["links"] if l[1] in alive and l[3] in alive]
    for node in graph["nodes"]:
        for slot in node.get("inputs", []) or []:
            if slot.get("link") is not None and slot["link"] not in {
                    l[0] for l in graph["links"]}:
                slot["link"] = None
    return graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", required=True, help="a file under ComfyUI input/")
    ap.add_argument("--pose-video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--frames", type=int, default=0, help="0 = the whole clip")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the prompt without queueing it")
    # The knobs below exist so a front/back experiment can change ONE thing
    # and keep the rest byte-identical; every value used is written to the
    # sidecar .json next to the video, so an arm is identified by what it ran
    # with, not by its directory name (the manifest-omits-the-flag lesson).
    ap.add_argument("--attention", choices=["sageattn", "sdpa"],
                    default=os.environ.get("COMFY_ATTENTION") or None,
                    help="WanVideoModelLoader attention_mode; default = the graph's own (sageattn). "
                         "sdpa where SageAttention's Triton kernels cannot compile (sm_120 GPUs, "
                         "2026-09-23); exact rather than quantised attention, so pixels differ from "
                         "a sageattn render and arms are compared only at the same setting")
    ap.add_argument("--no-compile", action="store_true",
                    default=bool(os.environ.get("COMFY_NO_COMPILE")),
                    help="drop the graph's WanVideoTorchCompileSettings (node 35): its Inductor compile "
                         "subprocess crashes on sm_120 GPUs (2026-09-23).  Compilation changes speed, "
                         "not the sampled video")
    ap.add_argument("--seed", type=int, default=42,
                    help="sampler seed; 42 is the graph's own PrimitiveNode value")
    ap.add_argument("--prompt", default=None,
                    help="positive prompt; default POSITIVE_PROMPT")
    ap.add_argument("--context-schedule", default=None,
                    choices=["uniform_standard", "uniform_looped", "static_standard"])
    ap.add_argument("--context-frames", type=int, default=None)
    ap.add_argument("--context-overlap", type=int, default=None)
    ap.add_argument("--context-stride", type=int, default=None,
                    help="pixel frames; 4 = one latent, i.e. no dilated windows")
    ap.add_argument("--ref-pose", choices=["window_first", "clip_first", "character"],
                    default="window_first",
                    help="which pose SteadyDancer pairs with the reference image in "
                         "every context window (needs the wrapper patch "
                         "patches/wanvideowrapper_steadydancer_ref.patch): the "
                         "window's own first frame (upstream), the clip's frame 0, "
                         "or the character still's OWN pose -- see REFERENCE PAIRING")
    ap.add_argument("--window-reference", choices=["on", "off"], default="on",
                    help="'on' (upstream) overwrites each later window's first frame "
                         "with the reference image; 'off' feeds it to the reference "
                         "tokens only")
    ap.add_argument("--pose-clip", choices=["off", "on"], default="off",
                    help="deliver CLIP of the pose frame paired with the reference "
                         "to the model (upstream stores it under a key the model "
                         "never reads; needs the wrapper patch)")
    ap.add_argument("--pose-latent-strength", type=float, default=None,
                    help="WanVideoEncode latent_strength for the pose frames (and "
                         "the reference pose); the graph has 0.8, the official "
                         "SteadyDancer code 1.0")
    ap.add_argument("--cfg-step0", type=float, default=None,
                    help="classifier-free guidance on the FIRST (highest-noise) step "
                         "only, the step that fixes global layout and facing; the "
                         "other steps stay at cfg 1 as the distill LoRA expects. "
                         "The negative pose is the positive pose itself, so the "
                         "guidance is text-only (see --negative-extra)")
    ap.add_argument("--negative-extra", default="",
                    help="text appended to the graph's negative prompt; inert unless "
                         "some step runs at cfg != 1")
    ap.add_argument("--size", default=None, metavar="WxH",
                    help="render size (both the character still and the pose frames are resized to it, and the "
                         "image-to-video encode follows the pose frames); default the graph's 480x832.  Give the pose "
                         "video and the character still at this size too (aapose_video --width/--height, and e.g. "
                         "townfair_fit720.png) so neither is upscaled")
    ap.add_argument("--facing-yaw", default=None,
                    help="per pose-frame shoulder yaw (deg) of the driving body (aapose_video writes "
                         "<pose>.yaw.npy); context windows that turn past --turn-yaw get --turn-prompt")
    ap.add_argument("--turn-yaw", type=float, default=TURN_YAW)
    ap.add_argument("--turn-prompt", default=TURN_PROMPT)
    ap.add_argument("--pose-strength-spatial", type=float, default=POSE_STRENGTH_SPATIAL)
    ap.add_argument("--pose-strength-temporal", type=float, default=POSE_STRENGTH_TEMPORAL)
    args = ap.parse_args()

    info = probe(args.pose_video)
    width, height, fps, frames = (info["width"], info["height"],
                                  info["fps"], info["frames"])
    seconds = frames / fps
    print("pose video: {}x{} {} frames at {:g} fps ({:.1f} s)".format(
        width, height, frames, fps, seconds))

    # STAGED UNDER ITS OWN CONTENT HASH, not under its basename.  Every clip's
    # pose video is called ``aapose.mp4``, and the loader reads the file when
    # the prompt EXECUTES, not when it is queued -- so with two drivers queued
    # at once the second copy overwrote the first before it ran.  Measured
    # 2026-09-21: the torso arm for 818 came back 329 frames long against its
    # 349-frame pose, because it had rendered clip 424's pose, staged by
    # the batch queued behind it.  A content name also means a CHANGED pose
    # can never be served from a cached load of the same filename.
    pose_bytes = pathlib.Path(args.pose_video).read_bytes()
    pose_sha1 = hashlib.sha1(pose_bytes).hexdigest()
    staged = COMFY_ROOT / "input" / "aapose_{}.mp4".format(pose_sha1[:16])
    # A name is not proof of content: a copy interrupted half-way would keep
    # its name forever (review 2026-09-21), so an existing file is re-hashed.
    if not staged.is_file() or hashlib.sha1(staged.read_bytes()).hexdigest() != pose_sha1:
        partial = staged.with_suffix(".part")
        partial.write_bytes(pose_bytes)
        partial.replace(staged)

    load_schema()
    needs_patch = (args.ref_pose != "window_first" or args.window_reference != "on"
                   or args.pose_clip != "off")
    if needs_patch:
        # Unknown inputs are silently dropped by the server, so an unpatched
        # wrapper would render the upstream behaviour under the new arm's name.
        with urllib.request.urlopen(SERVER + "/object_info/WanVideoAddSteadyDancerEmbeds",
                                    timeout=60) as response:
            node = json.loads(response.read())["WanVideoAddSteadyDancerEmbeds"]
        declared = set(node["input"].get("optional", {}))
        missing = {"ref_pose", "ref_pose_latent", "window_reference", "pose_clip"} - declared
        if missing:
            raise SystemExit("the server's WanVideoAddSteadyDancerEmbeds lacks {} -- "
                             "apply render2d/patches/wanvideowrapper_local.patch and "
                             "restart ComfyUI".format(sorted(missing)))
    if args.pose_clip == "on" and args.ref_pose == "window_first":
        # Node 82 encodes frame 0 of the pose video, while ref_c under
        # window_first is each WINDOW's first pose: the two would describe
        # different poses in every window after the first.
        raise SystemExit("--pose-clip on needs --ref-pose clip_first or character")
    graph = use_our_pose(json.loads(WORKFLOW.read_text()))
    prompt = api_format(graph)

    video = prompt[str(NODE_VIDEO)]["inputs"]
    video["video"] = staged.name
    # The pose video is ALREADY at the model's rate, because the motion was
    # resampled to it rather than the frames dropped.  Asking the loader to
    # force a rate it already has is a no-op; asking it to force a DIFFERENT one
    # would reintroduce the drop.
    video["force_rate"] = int(round(fps)) if abs(fps - MODEL_FPS) > 0.5 else 0
    video["custom_width"] = width
    video["custom_height"] = height
    video["frame_load_cap"] = int(args.frames)
    video["skip_first_frames"] = 0
    video["select_every_nth"] = 1
    prompt[str(NODE_IMAGE)]["inputs"]["image"] = pathlib.Path(args.character).name
    save = prompt[str(NODE_SAVE)]["inputs"]
    save["frame_rate"] = MODEL_FPS
    save["save_output"] = True
    # The saved graph carries loop_count 22, which would repeat the finished
    # dance twenty-two times in the file.
    save["loop_count"] = 0
    sampler = prompt[str(NODE_SAMPLER)]["inputs"]
    sampler["cfg"] = CFG
    sampler["steps"] = STEPS
    prompt[str(NODE_SCHEDULER)]["inputs"]["steps"] = STEPS
    if args.attention:
        prompt[str(NODE_MODEL_LOADER)]["inputs"]["attention_mode"] = args.attention
    if args.no_compile:
        prompt[str(NODE_MODEL_LOADER)]["inputs"].pop("compile_args", None)
        prompt.pop(str(NODE_COMPILE), None)
    prompt[str(NODE_BLOCKSWAP)]["inputs"]["blocks_to_swap"] = BLOCKS_TO_SWAP
    # VAE TILING, because the decode is what the container's 48 GB cgroup
    # cannot take: a full clip is ~330 frames and decoding them in one piece
    # killed the server outright (memory.failcnt was already in the thousands).
    # ``auto_render.py`` in the same ComfyUI turns exactly this on, with the
    # comment "控制 VAE Tiling 防爆内存"; the tile sizes are the graph's own.
    prompt[str(NODE_DECODE_NODE)]["inputs"]["enable_vae_tiling"] = True
    prompt[str(NODE_TEXT)]["inputs"]["positive_prompt"] = (
        args.prompt if args.prompt is not None else POSITIVE_PROMPT)
    if args.size:
        size_w, size_h = (int(v) for v in args.size.lower().split("x"))
        for node in ("68", "77"):     # ImageResizeKJv2: the character still, and the pose frames
            prompt[node]["inputs"]["width"] = size_w
            prompt[node]["inputs"]["height"] = size_h
    facing_windows = None
    if args.facing_yaw:
        front = prompt[str(NODE_TEXT)]["inputs"]["positive_prompt"]
        yaw = np.abs(np.load(args.facing_yaw).astype(np.float64))
        used = frames if not args.frames else min(frames, args.frames)
        if abs(len(yaw) - frames) > 2:
            raise SystemExit("--facing-yaw has {} frames, the pose video {}".format(len(yaw), frames))
        latents = (used - 1) // 4 + 1
        context_frames = args.context_frames or prompt[str(NODE_CONTEXT)]["inputs"].get("context_frames", 81)
        window = (int(context_frames) - 1) // 4 + 1
        per_latent = []
        for k in range(latents):
            lo, hi = max(0, 4 * (k - window + 1)), min(len(yaw), 4 * k + 4)
            per_latent.append(bool(hi > lo and yaw[lo:hi].max() > args.turn_yaw))
        prompt[str(NODE_TEXT)]["inputs"]["positive_prompt"] = "|".join(
            args.turn_prompt if turn else front for turn in per_latent)
        facing_windows = {"latents": latents, "window_latents": window, "turn_yaw": args.turn_yaw,
                          "turn_latents": int(sum(per_latent)), "turn_prompt": args.turn_prompt,
                          "front_prompt": front}
        print("facing-aware prompts: {}/{} windows turn past {:g} deg".format(
            sum(per_latent), latents, args.turn_yaw))
    sdancer = prompt[str(NODE_SDANCER)]["inputs"]
    sdancer["pose_strength_spatial"] = args.pose_strength_spatial
    sdancer["pose_strength_temporal"] = args.pose_strength_temporal
    # The seed reaches the sampler as a flattened PrimitiveNode constant, so it
    # is an ordinary input here.
    sampler["seed"] = int(args.seed)
    context = prompt[str(NODE_CONTEXT)]["inputs"]
    for key, value in (("context_schedule", args.context_schedule),
                       ("context_frames", args.context_frames),
                       ("context_overlap", args.context_overlap),
                       ("context_stride", args.context_stride)):
        if value is not None:
            context[key] = value
    # one prefix per server when several share this ComfyUI tree, so their output counters cannot collide
    save["filename_prefix"] = os.environ.get("COMFY_OUTPUT_PREFIX", "atomicdance_2d")
    # REFERENCE PAIRING.  SteadyDancer conditions on a reference PAIR: the
    # reference image (``ref_x``) and a pose latent (``ref_c``).  It was trained
    # with first-frame preservation -- frame 0 IS the reference -- so the pair is
    # "this appearance is this pose".  Under context windows the wrapper takes
    # ``ref_c`` from each WINDOW's first frame, so a window that starts mid-turn
    # is told the front-facing still belongs to a turned skeleton, and reads the
    # rest of its window relative to that.  Measured on 818 (2026-09-21): the
    # windows covering the phantom back view f228-f244 pair the still with poses
    # at +51 and +62 deg while the dance there is at -30..-46, i.e. 80-110 deg
    # "away from how the reference looks".
    if args.ref_pose != "window_first":
        sdancer["ref_pose"] = "clip_first" if args.ref_pose == "clip_first" else "window_first"
    if args.ref_pose == "character":
        still = reference_pose_image(args.character, width, height)
        prompt["900"] = {"class_type": "LoadImage", "inputs": {"image": still.name}}
        encode = json.loads(json.dumps(prompt[str(NODE_POSE_ENCODE)]["inputs"]))
        encode["image"] = ["900", 0]
        prompt["901"] = {"class_type": "WanVideoEncode", "inputs": encode}
        sdancer["ref_pose_latent"] = ["901", 0]
    if args.window_reference != "on":
        sdancer["window_reference"] = args.window_reference
    if args.pose_clip != "off":
        sdancer["pose_clip"] = args.pose_clip
        if args.ref_pose == "character":
            # The CLIP feature must describe the same pose ref_c does.
            prompt[str(NODE_POSE_CLIP)]["inputs"]["image_1"] = ["900", 0]
    if args.cfg_step0 is not None:
        # FIRST-STEP GUIDANCE.  At cfg 1 the sampler never runs the negative
        # branch, so the graph's negative prompt (which already names 倒着走)
        # has never done anything.  One guided step at the highest noise, where
        # the facing is decided, with the SAME pose on both branches: the only
        # difference between them is the text.  The SteadyDancer node needs a
        # negative pose latent or the uncond pass dies on None.unsqueeze.
        prompt["910"] = {"class_type": "CreateCFGScheduleFloatList", "inputs": {
            "steps": STEPS, "cfg_scale_start": args.cfg_step0,
            "cfg_scale_end": args.cfg_step0, "interpolation": "linear",
            "start_percent": 0.0, "end_percent": 0.0}}
        sampler["cfg"] = ["910", 0]
        sdancer["pose_latents_negative"] = [str(NODE_POSE_ENCODE), 0]
    if args.negative_extra:
        text = prompt[str(NODE_TEXT)]["inputs"]
        text["negative_prompt"] = (text.get("negative_prompt", "") + "，" + args.negative_extra).strip("，")
    if facing_windows is not None or args.prompt is not None:
        # T5 RESIDENT (any text other than the default, whose embeddings are already in the disk cache).  WanVideoTextEncodeCached loads umt5-xxl (11 GB, from the NAS) INSIDE the node on every
        # disk-cache miss, and its cache key is the whole positive string -- so per-window prompts, a new '|' string
        # for every clip, reloaded it every time: 20-30 min per job with eight servers loading at once (2026-09-25).
        # The wrapper's separate loader node has fixed inputs, so ComfyUI's execution cache keeps the loaded encoder
        # between jobs and only the encode itself runs; on the 72 GB cards it stays on the GPU next to the model.
        text = prompt[str(NODE_TEXT)]["inputs"]
        # the same weights copied to local CPFS (/cache/atomicdance-assets/models, linked into ComfyUI as
        # umt5-xxl-enc-bf16.cache.safetensors): the loader mmaps parameter by parameter, which from the NAS ran
        # at 5-9 s a tensor with eight servers reading at once
        model_name = {"umt5-xxl-enc-bf16.safetensors": "umt5-xxl-enc-bf16.cache.safetensors"}.get(
            text["model_name"], text["model_name"])
        prompt["9201"] = {"class_type": "LoadWanVideoT5TextEncoder",
                          "inputs": {"model_name": model_name, "precision": text["precision"],
                                     "load_device": "main_device", "quantization": text.get("quantization", "disabled")}}
        prompt[str(NODE_TEXT)] = {"class_type": "WanVideoTextEncode",
                                  "inputs": {"positive_prompt": text["positive_prompt"],
                                             "negative_prompt": text["negative_prompt"], "t5": ["9201", 0],
                                             "force_offload": False, "use_disk_cache": False, "device": "gpu"}}
    if args.pose_latent_strength is not None:
        for node in (str(NODE_POSE_ENCODE), "901"):
            if node in prompt:
                prompt[node]["inputs"]["latent_strength"] = args.pose_latent_strength
    settings = {
        "seed": sampler["seed"], "cfg": sampler["cfg"], "steps": sampler["steps"],
        "positive_prompt": (prompt[str(NODE_TEXT)]["inputs"]["positive_prompt"] if facing_windows is None
                            else "(per window, see facing_windows)"),
        "facing_windows": facing_windows,
        "pose_strength_spatial": sdancer["pose_strength_spatial"],
        "pose_strength_temporal": sdancer["pose_strength_temporal"],
        "ref_pose": args.ref_pose,
        "pose_clip": args.pose_clip,
        "cfg_step0": args.cfg_step0,
        "attention": args.attention or "graph default (sageattn)",
        "torch_compile": not args.no_compile,
        "negative_prompt": prompt[str(NODE_TEXT)]["inputs"].get("negative_prompt"),
        "pose_latent_strength": prompt[str(NODE_POSE_ENCODE)]["inputs"]["latent_strength"],
        "window_reference": args.window_reference,
        "context": {k: context.get(k) for k in (
            "context_schedule", "context_frames", "context_stride",
            "context_overlap", "freenoise", "fuse_method")},
        "character": args.character,
        "pose_video": str(args.pose_video),
        "pose_video_sha1": pose_sha1,
    }

    # A reference to a node that was removed validates as KeyError '<id>', not
    # as "this optional input is absent", so dangling ones are dropped here.
    for node in prompt.values():
        for name, value in list(node["inputs"].items()):
            if isinstance(value, list) and len(value) == 2 and value[0] not in prompt:
                del node["inputs"][name]
    prompt = fix_model_names(prompt)

    if args.dry_run:
        print(json.dumps(prompt, indent=1))
        print(json.dumps(settings, indent=1))
        return

    result = post(prompt)
    print("queued", result.get("prompt_id"))
    entry = wait(result["prompt_id"], timeout=14400)

    produced = []
    for node_output in entry.get("outputs", {}).values():
        for item in node_output.get("gifs", []) + node_output.get("videos", []):
            # ``fullpath`` when the node reports it: the type ("output" /
            # "temp") names a DIRECTORY under the ComfyUI root that
            # ``subfolder`` does not include, so joining root+subfolder+filename
            # misses it and the run reads as "completed but wrote no video".
            if item.get("fullpath"):
                produced.append(pathlib.Path(item["fullpath"]))
            else:
                produced.append(COMFY_ROOT / item.get("type", "output")
                                / item.get("subfolder", "") / item["filename"])
    produced = [p for p in produced if p.is_file()]
    if not produced:
        raise SystemExit("the workflow completed but wrote no video; outputs were "
                         + json.dumps(entry.get("outputs", {}))[:500])
    newest = max(produced, key=lambda p: p.stat().st_mtime)

    final = pathlib.Path(args.out)
    final.parent.mkdir(parents=True, exist_ok=True)
    # Written to .part and moved into place only after the duration gate: a
    # render that fails the gate used to stay at --out, and the batch driver
    # then skipped that clip as "already rendered" (review 2026-09-21).
    target = final.with_name(final.stem + ".part" + final.suffix)
    if args.audio and pathlib.Path(args.audio).is_file():
        # MP3, NOT AAC.  ``tools/render_sample_strip.py`` states the reason in
        # its own header -- "the reviewer's IDE plays mp3 in an mp4 container
        # and will not play AAC" -- and this driver ignored it, so the first
        # ten 2D videos carried a correct, correctly levelled audio track that
        # the operator could not hear.  The 3D strips and these must sound the
        # same way or they cannot be judged the same way.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(newest),
                        "-i", args.audio, "-c:v", "copy",
                        "-c:a", "libmp3lame", "-b:a", "128k",
                        "-shortest", str(target)], check=True)
    else:
        target.write_bytes(newest.read_bytes())

    result_info = probe(target)
    got, got_fps = result_info["frames"], result_info["fps"]
    expected = (min(args.frames, frames) / fps) if args.frames > 0 else seconds
    print("{} -> {} ({} frames at {:g} fps = {:.1f} s against the pose video's "
          "{:.1f} s)".format(newest.name, final, got, got_fps, got / got_fps,
                             expected))
    if abs(got / got_fps - expected) > max(0.5, expected * 0.05):
        target.unlink()
        raise SystemExit("the animation's duration differs from the dance's by "
                         "more than 5%; it will not line up with its music")
    target.replace(final)
    pathlib.Path(str(final) + ".json").write_text(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
