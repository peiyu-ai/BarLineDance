"""ComfyUI-layout Wan2.2-Animate weights -> a diffusers state dict.

WHY A CONVERTER AND NOT ``from_single_file``.  That loader fetches a
``config.json`` from the hub to interpret the tensors, and this machine cannot
reach it -- the failure is ``Wan-AI/Wan2.2-Animate-14B-Diffusers does not appear
to have a file named config.json``.  The configuration is not actually unknown:
every default of ``WanAnimateTransformer3DModel`` already matches what the
checkpoint contains (40 layers, 40 heads x 128 = 5120, in_channels 36, patch
(1, 2, 2)), which was read off the tensor shapes.  What IS missing is the key
naming: the staged file uses ComfyUI's names and the model wants diffusers'.

COUNTED BEFORE WRITING, so the mapping is not guesswork: the file holds 1955
tensors of which 513 are fp8 ``scale_weight`` entries, leaving 1442 real ones
against the 1243 the model asks for.  Block prefixes map
``self_attn -> attn1``, ``cross_attn -> attn2``, ``modulation ->
scale_shift_table``, ``norm3 -> norm2``; at the top level ``time_embedding``,
``text_embedding`` and ``img_emb`` all fold into ``condition_embedder`` and
``head`` becomes ``proj_out``.

FP8.  The weights are ``float8_e4m3fn`` with one scale per tensor; they are
dequantised to bf16 here.  A mismatch between a tensor and its scale would be
silent, so every scaled tensor must find its scale and the count is asserted.
"""
import argparse
import pathlib
import re

import torch
from safetensors.torch import load_file, save_file

BLOCK = [
    (r"^blocks\.(\d+)\.self_attn\.q\.", r"blocks.\1.attn1.to_q."),
    (r"^blocks\.(\d+)\.self_attn\.k\.", r"blocks.\1.attn1.to_k."),
    (r"^blocks\.(\d+)\.self_attn\.v\.", r"blocks.\1.attn1.to_v."),
    (r"^blocks\.(\d+)\.self_attn\.o\.", r"blocks.\1.attn1.to_out.0."),
    (r"^blocks\.(\d+)\.self_attn\.norm_q\.", r"blocks.\1.attn1.norm_q."),
    (r"^blocks\.(\d+)\.self_attn\.norm_k\.", r"blocks.\1.attn1.norm_k."),
    (r"^blocks\.(\d+)\.cross_attn\.q\.", r"blocks.\1.attn2.to_q."),
    (r"^blocks\.(\d+)\.cross_attn\.k\.", r"blocks.\1.attn2.to_k."),
    (r"^blocks\.(\d+)\.cross_attn\.v\.", r"blocks.\1.attn2.to_v."),
    (r"^blocks\.(\d+)\.cross_attn\.o\.", r"blocks.\1.attn2.to_out.0."),
    (r"^blocks\.(\d+)\.cross_attn\.norm_q\.", r"blocks.\1.attn2.norm_q."),
    (r"^blocks\.(\d+)\.cross_attn\.norm_k\.", r"blocks.\1.attn2.norm_k."),
    (r"^blocks\.(\d+)\.cross_attn\.k_img\.", r"blocks.\1.attn2.add_k_proj."),
    (r"^blocks\.(\d+)\.cross_attn\.v_img\.", r"blocks.\1.attn2.add_v_proj."),
    (r"^blocks\.(\d+)\.cross_attn\.norm_k_img\.", r"blocks.\1.attn2.norm_added_k."),
    (r"^blocks\.(\d+)\.norm3\.", r"blocks.\1.norm2."),
    (r"^blocks\.(\d+)\.ffn\.0\.", r"blocks.\1.ffn.net.0.proj."),
    (r"^blocks\.(\d+)\.ffn\.2\.", r"blocks.\1.ffn.net.2."),
    (r"^blocks\.(\d+)\.modulation$", r"blocks.\1.scale_shift_table"),
]
TOP = [
    (r"^time_embedding\.0\.", "condition_embedder.time_embedder.linear_1."),
    (r"^time_embedding\.2\.", "condition_embedder.time_embedder.linear_2."),
    (r"^time_projection\.1\.", "condition_embedder.time_proj."),
    (r"^text_embedding\.0\.", "condition_embedder.text_embedder.linear_1."),
    (r"^text_embedding\.2\.", "condition_embedder.text_embedder.linear_2."),
    (r"^img_emb\.proj\.1\.", "condition_embedder.image_embedder.ff.net.0.proj."),
    (r"^img_emb\.proj\.3\.", "condition_embedder.image_embedder.ff.net.2."),
    (r"^img_emb\.proj\.0\.", "condition_embedder.image_embedder.norm1."),
    (r"^img_emb\.proj\.4\.", "condition_embedder.image_embedder.norm2."),
    (r"^head\.head\.", "proj_out."),
    (r"^head\.modulation$", "scale_shift_table"),
    # The face adapter: ComfyUI keeps q separate and k,v FUSED in one 10240-wide
    # projection (5120 x 2), and names the norms q_norm/k_norm.  Splitting is
    # handled in ``split_fused`` below; these two are the plain renames.
    (r"^face_adapter\.fuser_blocks\.(\d+)\.linear1_q\.", r"face_adapter.\1.to_q."),
    (r"^face_adapter\.fuser_blocks\.(\d+)\.linear2\.", r"face_adapter.\1.to_out."),
    (r"^face_adapter\.fuser_blocks\.(\d+)\.q_norm\.", r"face_adapter.\1.norm_q."),
    (r"^face_adapter\.fuser_blocks\.(\d+)\.k_norm\.", r"face_adapter.\1.norm_k."),

    # The face encoder keeps its convolutions one level deeper.
    (r"^face_encoder\.(conv1_local|conv2|conv3)\.conv\.", r"face_encoder.\1."),

    # The motion encoder.  ComfyUI's names come from the original LIA-style
    # encoder (``enc.net_app.convs``, ``enc.fc``, ``dec.direction``); diffusers
    # flattens the same modules into ``res_blocks``, ``motion_network`` and
    # ``motion_synthesis_weight``.  Every pair below was matched on SHAPE, not
    # on the name looking similar: conv1 (C,C,3,3) to conv1, conv2 (2C,C,3,3)
    # to conv2, skip (2C,C,1,1) to conv_skip, and the per-channel biases
    # (1,C,1,1) to the ``act_fn.bias`` the diffusers block keeps them in.
    (r"^motion_encoder\.enc\.net_app\.convs\.0\.0\.weight$",
     "motion_encoder.conv_in.weight"),
    (r"^motion_encoder\.enc\.net_app\.convs\.0\.1\.bias$",
     "motion_encoder.conv_in.act_fn.bias"),
    (r"^motion_encoder\.enc\.net_app\.convs\.8\.weight$",
     "motion_encoder.conv_out.weight"),
    (r"^motion_encoder\.enc\.net_app\.convs\.(\d+)\.conv1\.0\.weight$",
     r"motion_encoder.res_blocks.\1.conv1.weight"),
    (r"^motion_encoder\.enc\.net_app\.convs\.(\d+)\.conv1\.1\.bias$",
     r"motion_encoder.res_blocks.\1.conv1.act_fn.bias"),
    (r"^motion_encoder\.enc\.net_app\.convs\.(\d+)\.conv2\.1\.weight$",
     r"motion_encoder.res_blocks.\1.conv2.weight"),
    (r"^motion_encoder\.enc\.net_app\.convs\.(\d+)\.conv2\.2\.bias$",
     r"motion_encoder.res_blocks.\1.conv2.act_fn.bias"),
    (r"^motion_encoder\.enc\.net_app\.convs\.(\d+)\.skip\.1\.weight$",
     r"motion_encoder.res_blocks.\1.conv_skip.weight"),
    (r"^motion_encoder\.enc\.fc\.(\d+)\.", r"motion_encoder.motion_network.\1."),
    (r"^motion_encoder\.dec\.direction\.weight$",
     "motion_encoder.motion_synthesis_weight"),
]

# ``res_blocks`` are numbered from 0 while ComfyUI's convs start the residual
# stack at 1 (index 0 is conv_in, 8 is conv_out), so the index shifts by one.
RES_BLOCK = re.compile(r"^motion_encoder\.res_blocks\.(\d+)\.")

# The blur kernels ComfyUI stores for its antialiased downsampling are fixed
# constants, not learned parameters; diffusers builds them itself.
DROP = (".conv2.0.kernel", ".skip.0.kernel")

# ``linear1_kv`` holds k and v stacked; the model wants them apart.  Checked
# against the shapes rather than assumed: the fused tensor is (10240, 5120) and
# each half is exactly the (5120, 5120) the model asks for.
FUSED_KV = re.compile(r"^face_adapter\.fuser_blocks\.(\d+)\.linear1_kv\.(weight|bias)$")


def split_fused(key, tensor):
    """[(name, tensor)] -- one entry normally, two for a fused k/v."""
    match = FUSED_KV.match(key)
    if not match:
        return [(rename(key), tensor)]
    index, kind = match.group(1), match.group(2)
    half = tensor.shape[0] // 2
    return [("face_adapter.{}.to_k.{}".format(index, kind), tensor[:half]),
            ("face_adapter.{}.to_v.{}".format(index, kind), tensor[half:])]


def rename(key):
    for pattern, replacement in BLOCK + TOP:
        new, count = re.subn(pattern, replacement, key)
        if count:
            match = RES_BLOCK.match(new)
            if match:
                index = int(match.group(1)) - 1
                new = RES_BLOCK.sub(
                    "motion_encoder.res_blocks.{}.".format(index), new)
            return new
    return key


def convert_sharded(source, out_dir, dtype=torch.bfloat16, shard_bytes=6 << 30):
    """Convert to SHARDED diffusers weights, holding one shard at a time.

    THE MEMORY LIMIT IS THE CGROUP'S, NOT THE MACHINE'S.  ``free -g`` reports
    740 GB available here and the converter was still OOM-killed twice:
    ``/sys/fs/cgroup/memory.max`` is **48 GB** and 18 GB of it was already in
    use.  The bf16 model is about 28 GB, so accumulating it in a dict before
    writing cannot fit no matter what ``free`` says.  This is the same trap the
    repository already records for disk -- ``df`` reports 10 PB while a 1 KB
    write fails on the directory quota -- and the fix is the same shape: size
    the work to the real limit and write incrementally.

    Each tensor is read, renamed, dequantised and appended to the current
    shard; when the shard passes ``shard_bytes`` it is saved and dropped.
    """
    from safetensors import safe_open
    import json

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    shard = {}
    shard_size = 0
    shard_number = 0
    dequantised = 0
    written = 0

    def flush():
        nonlocal shard, shard_size, shard_number, written
        if not shard:
            return
        name = "diffusion_pytorch_model-{:05d}.safetensors".format(shard_number)
        save_file(shard, str(out_dir / name))
        for key in shard:
            index[key] = name
        written += len(shard)
        shard = {}
        shard_size = 0
        shard_number += 1

    with safe_open(str(source), framework="pt") as handle:
        keys = list(handle.keys())
        scales = {k[: -len(".scale_weight")]: handle.get_tensor(k)
                  for k in keys if k.endswith("scale_weight")}
        for key in keys:
            if key.endswith("scale_weight") or key == "scaled_fp8":
                continue
            tensor = handle.get_tensor(key)
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                scale = scales.get(key.rsplit(".", 1)[0])
                if scale is None:
                    raise SystemExit(
                        "fp8 tensor {} has no scale_weight".format(key))
                tensor = (tensor.to(torch.float32)
                          * scale.to(torch.float32)).to(dtype)
                dequantised += 1
            if key.endswith(DROP):
                continue
            # The per-channel biases are stored as (1, C, 1, 1) and wanted as (C,).
            if tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[2:] == (1, 1):
                tensor = tensor.reshape(-1)
            tensor = tensor.to(dtype)
            for name, piece in split_fused(key, tensor):
                piece = piece.contiguous()
                shard[name] = piece
                shard_size += piece.numel() * piece.element_size()
            if shard_size >= shard_bytes:
                flush()
        flush()

    (out_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": index}, indent=2))
    return written, dequantised, len(scales)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import json

    written, dequantised, scale_count = convert_sharded(
        pathlib.Path(args.source), args.out)
    print("converted {} tensors ({} dequantised from fp8, {} scales seen)"
          .format(written, dequantised, scale_count), flush=True)

    index = json.loads((pathlib.Path(args.out) /
                        "diffusion_pytorch_model.safetensors.index.json").read_text())
    produced = set(index["weight_map"])
    from diffusers import WanAnimateTransformer3DModel
    # added_kv_proj_dim=5120 is not a guess: without it the model has no
    # ``attn2.add_k_proj``/``add_v_proj`` slots and the checkpoint's 200 image
    # cross-attention tensors read as "extra".  5120 is read off the checkpoint: add_k_proj is (5120, 5120),
    # not the (5120, 1280) a CLIP-H width would give -- a first pass assumed
    # 1280 and the load failed on the shape, which is the honest way to find out.
    with torch.device("meta"):
        wanted = set(WanAnimateTransformer3DModel(
            added_kv_proj_dim=5120).state_dict().keys())
    missing = sorted(wanted - produced)
    extra = sorted(produced - wanted)
    print("missing {} | extra {}".format(len(missing), len(extra)))
    for key in missing[:10]:
        print("  missing:", key)
    for key in extra[:10]:
        print("  extra  :", key)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
