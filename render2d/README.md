# render2d — 2D cartoon animation from the generated dance

Turns a generated 3D dance into a **2D cartoon video**: one still character
image plus a driving pose video, animated by a video diffusion model
(ComfyUI + Wan Animate).

    generated motion (24x3 world joints, 30 fps)
        -> project_pose_2d.py      : 18-joint OpenPose COCO, normalised [0,1]
        -> draw_pose_video.py      : the driving pose video the sampler reads
        -> comfy/wan_animate.json  : character image + pose video -> frames
        -> mux with the clip's own audio.wav

## Why this pose format and not another

The ingest already stores DWPose output for every clip as
`keypoints.npy` `[T, 18, 2]` normalised to `[0, 1]`, with `scores.npy`
`[T, 18]`, in OpenPose **COCO-18** order:

    0 nose  1 neck  2 Rsho 3 Relb 4 Rwri  5 Lsho 6 Lelb 7 Lwri
    8 Rhip  9 Rkne 10 Rank 11 Lhip 12 Lkne 13 Lank
    14 Reye 15 Leye 16 Rear 17 Lear

That is the format the ComfyUI pose nodes consume, so the projection targets it
rather than inventing one: a generated clip and a real clip then drive the
animator through exactly the same path, which is what makes the real video a
usable control for "is the 2D stage working, or is the dance wrong".

## The ten clips

`runs/vis_clips_t10.txt` -- the same fixed ten every other round renders, so a
2D result can be put beside the 3D one without changing the ruler
(CLAUDE.md 1.5 rule 7).

## Running it

    render2d/run_2d_pipeline.sh <arm-dir> <out-dir> --model /cache/wan22

Stages 1-3 (pose projection, pose video, character still) need no weights.
Stage 4 needs the Wan2.2-Animate weights, which ARE staged on this project's
OSS prefix under `models/WAN22/` -- backbone, VAE, relight LoRA, few-step LoRA,
with the UMT5 encoder one level up in `models/`.

### One character, ten clips

The character still is rendered once and reused for all ten, because the demo
is "this character dancing these ten dances".  A different drawing per clip
would make the ten incomparable for the same reason the fixed ten clips exist.

### Why the weights need converting first

The staged files are in **ComfyUI layout** (`blocks.N.cross_attn.k`, fp8 with
per-tensor `scale_weight`), and `from_single_file` fetches a `config.json` from
the hub to interpret them -- which fails here, the network reaching pypi but
not huggingface.co.  The configuration is not actually unknown: read off the
tensor shapes it is 40 layers, 40 heads x 128 = 5120, `in_channels` 36, patch
(1, 2, 2), every one of which is already the default of
`WanAnimateTransformer3DModel`.  So:

    python3 render2d/convert_comfy_wan.py \
        --source /cache/wan22/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors \
        --out    /cache/wan22/transformer

writes diffusers-named shards and prints any key it could not place.  Two
mappings in it are not obvious and were derived from the shapes:

* `added_kv_proj_dim=1280` must be passed to the model, or its image
  cross-attention slots do not exist and the checkpoint's 200 `add_k_proj` /
  `add_v_proj` tensors read as "extra";
* the face adapter stores k and v FUSED in one `linear1_kv` of width 10240,
  which is split into the two (5120, 5120) halves the model asks for.

### Memory

The conversion is streamed and sharded because the limit is the **cgroup's**,
not the machine's: `free -g` reports 740 GB available here while
`/sys/fs/cgroup/memory.max` is 48 GB, and the first two attempts were
OOM-killed with an empty log and a normal-looking exit.  Same shape as the disk
rule in CLAUDE.md 1.2 -- `df` is not the disk criterion, `free` is not the
memory criterion.

## The MTV-Crafter path: 3D joints straight into the video model

    fix-arm pickle (full_pose, world z-up, 30 fps)
        -> mtv_motion.py   : align heading, follow-cam, body size, camera frame, 16 fps
        -> comfy_mtv.py    : ComfyUI + WanVideoWrapper, MTV-finetuned Wan2.1 I2V base
        -> score_mtv_follow: does the video do THIS dance?  (tools/)
        -> run_mtv.sh      : the fixed ten, plus the sample strip with the MTV panel beside it

    render2d/run_mtv.sh [arm-dir] [out-dir] [strip-dir] [character.png]

Needs: `Wan2_1-I2V-14B-MTV-Crafter_fp8_e4m3fn_scaled_KJ.safetensors` (operator's copy on
the OSS `models/` prefix, CRC64 4427192660244681590) symlinked into ComfyUI's
`models/diffusion_models/WanVideo/`, the 4DMoT VQ-VAE, and the wrapper patch in
`patches/` applied.  `comfy_mtv.py` refuses to run without the patch, refuses a base
with no motion tensors, and refuses the separate adapter on top of the MTV base.

Every one of these was silent -- loaded, sampled, decoded, muxed -- and is now a gate:

| defect | what the video showed | fix |
|---|---|---|
| base with no `motion_attn` path | character stands still | `assert_motion_path` |
| root 0.5-0.9 m off centre (z +4.65) | character walks out, frames become background wall | follow-cam (1 s high-pass of the horizontal root) + body size to the training mean |
| camera on the wrong side (dancer's back shown) | barely moves | `x,y,z -> -x,-z,-y`; `assert_faces_camera` |
| wrapper casts the complex motion RoPE to a real dtype (cos only) | barely moves | `patches/wanvideowrapper_mtv_freqs.patch` |
| bf16 adapter merged onto the fp8 MTV base keeps the base's `scale_weight` | motion branch x0.02-0.05 | `MTV_ADAPTER=none`, combination refused |
| 81-frame windows (training is 49; the key layout depends on token count) | moves, but not this dance | `MTV_CONTEXT=49,24` |

Knobs (environment): `MTV_CONTEXT`, `MTV_STRENGTH` (2.0 under overlapping windows),
`MTV_CLIP`, `MTV_NOISE_AUG`, `MTV_SIZE`, `MTV_FPS`, `MTV_SEED`, `MTV_NO_LORA`/`MTV_STEPS`/`MTV_CFG`.
Each render writes a sidecar `.json` with every setting, the joints' SHA-1 and whether the
patch was applied.

**Judging a render**: `tools/score_mtv_follow.py` fits the detected 2D body to the
projected joints (one similarity for the whole clip, and one per frame) and prints the
same fit against the pose shifted by half the clip.  On 818 the 3D skin render -- which
follows by construction -- reads shape 0.035 against 0.110; a render that does not follow
reads about its own null.  `found` counts only frames whose keypoints are confident:
the detector "found a person" in every frame of a video that had collapsed into a wall.

## The SteadyDancer path: pose picture conventions, front/back, proportions (2026-09-21/22)

    render2d/run_2d_steadydancer.sh [arm-dir] [out-dir] [character.png] [clips]
    # AAPOSE_EXTRA -> aapose_video.py, SD_EXTRA -> comfy_steadydancer.py,
    # SD_NEGATIVE_EXTRA -> one quoted --negative-extra (free text is not word-split)

The recommended settings, and what each one answers (details and measurements in
`docs/DANCE_QUALITY_DEFECTS.md` §89):

| flag | what it fixes |
|---|---|
| `--face-model calibrated` | the far ear vanished from 30° (the detector keeps it at every angle); eyes always drawn, nose fades only past 160°; face shape by yaw from 781k DWPose frames |
| `--torso detector_v2` | SMPL hips (femoral heads, 0.32 of shoulder width) moved to where COCO annotates them (0.76); arm chains ride with the shoulders |
| `--hands-model v2` | the old fans drew the left hand palm-to-camera and the right back-of-hand on every frame |
| `--fit bones` | each bone scaled to the character's own length (torso was 1.23x hers, forearm 1.19x, hips 1.17x); a ceiling on the neck's long tail; supporting ankle on her floor line |
| `--cfg-step0 2.0` + `SD_NEGATIVE_EXTRA="背影，背面，后脑勺，back view, from behind, back of head"` | one guided step at the highest noise, same pose on both branches, text-only: away from back views |

What was measured and ruled out: the model barely reads limb colours (`--palette swapped`
renders nearly identically); pairing each window's reference with the character's own pose
(`--ref-pose character`) changes pixels but not the facing; `--window-reference off` loses the
character's identity after the first window. These need the wrapper patch
`patches/wanvideowrapper_local.patch` (one diff against upstream HEAD; the driver refuses to
run a patched option on an unpatched server).

**Front/back flips are a property of the sample.** The same seed reproduces them frame for
frame; another seed can remove them all. So an arm is judged over the ten clips at more than
one seed, and a finished render set is best-of-N:

    python3 tools/score_2d_facing.py --video OUT/<clip>.mp4 --driven OUT/work/<clip>/driven.npy \
        --cues <clip cues json> --out OUT/facing/<clip>.json --keypoints-cache OUT/facing/<clip>.vp.npy
    python3 tools/pick_2d_facing_render.py ARM_A ARM_B ... --cues-dir <cues dir> --out pick.json

The pose videos are staged in ComfyUI's `input/` under their content hash (every clip's file
is called `aapose.mp4`, and the loader reads it at execution time, so two queued drivers used
to render each other's pose).
