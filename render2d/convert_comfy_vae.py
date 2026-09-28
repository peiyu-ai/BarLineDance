"""ComfyUI-layout Wan2.2 VAE -> diffusers naming.

THE CONFIG IS PINNED BY SHAPE, not guessed.  Anchored on three tensors read out
of the checkpoint -- ``encoder.conv1.weight`` (160, 12, 3, 3, 3),
``decoder.conv1.weight`` (1024, 48, 3, 3, 3) and ``conv1.weight``
(96, 96, 1, 1, 1) -- the only configuration that reproduces all three is

    base_dim=160, decoder_base_dim=256, z_dim=48, dim_mult=[1, 2, 4, 4],
    is_residual=True, patch_size=2, in_channels=12, out_channels=12

and with it the multiset of all 196 tensor shapes matches exactly.  A first
attempt used the Wan2.1 defaults (base_dim 96, in_channels 3); it also produced
196 tensors, which is why the count alone is not evidence -- the shapes are.

``in_channels=12`` is not redundant with ``patch_size=2``: patchify happens in
the forward pass and does not change the declared channel count, so the first
convolution has to be told it receives 3 x 2 x 2.
"""
import argparse
import pathlib
import re

import torch
from safetensors.torch import save_file

# ``latents_mean``/``latents_std`` are NOT recoverable from the checkpoint --
# they are normalisation constants, not weights -- and the diffusers defaults
# are Wan2.1's 16 values while this VAE has z_dim 48.  Left at the defaults the
# pipeline fails with ``shape '[1, 48, 1, 1, 1]' is invalid for input of size
# 16``.  Zero mean and unit std are used instead of inventing numbers: they
# leave the latents unscaled, which is the identity, whereas wrong constants
# would silently shift every frame's colour and contrast.  If Wan2.2's published
# values are staged later, put them here.
LATENTS_MEAN = [0.0] * 48
LATENTS_STD = [1.0] * 48

CONFIG = dict(base_dim=160, decoder_base_dim=256, z_dim=48, dim_mult=[1, 2, 4, 4],
              is_residual=True, patch_size=2, in_channels=12, out_channels=12,
              latents_mean=LATENTS_MEAN, latents_std=LATENTS_STD)

RULES = [
    # The two 1x1x1 projections either side of the latent.
    (r"^conv1\.", "quant_conv."),
    (r"^conv2\.", "post_quant_conv."),
    # Stem and head of each half.
    (r"^encoder\.conv1\.", "encoder.conv_in."),
    (r"^encoder\.head\.0\.", "encoder.norm_out."),
    (r"^encoder\.head\.2\.", "encoder.conv_out."),
    (r"^decoder\.conv1\.", "decoder.conv_in."),
    (r"^decoder\.head\.0\.", "decoder.norm_out."),
    (r"^decoder\.head\.2\.", "decoder.conv_out."),
    # Residual blocks: ComfyUI numbers the ops inside a Sequential, diffusers
    # names them.  0 -> norm1, 2 -> conv1, 3 -> norm2, 6 -> conv2 (1, 4 and 5
    # are the activations and dropout, which carry no weights).
    (r"\.residual\.0\.", ".norm1."),
    (r"\.residual\.2\.", ".conv1."),
    (r"\.residual\.3\.", ".norm2."),
    (r"\.residual\.6\.", ".conv2."),
    (r"\.shortcut\.", ".conv_shortcut."),
    # Middle stack: ComfyUI keeps one Sequential [resnet, attention, resnet],
    # diffusers splits it into ``resnets.0``, ``attentions.0``, ``resnets.1``.
    # The index therefore does not carry across -- 0 and 2 become resnets 0 and
    # 1, and 1 becomes attention 0.
    (r"^(encoder|decoder)\.middle\.0\.", r"\1.mid_block.resnets.0."),
    (r"^(encoder|decoder)\.middle\.1\.", r"\1.mid_block.attentions.0."),
    (r"^(encoder|decoder)\.middle\.2\.", r"\1.mid_block.resnets.1."),
    # Up/down stacks.
    # Same shape as the decoder: the downsampler is the last entry (index 2),
    # identified by carrying ``resample``/``time_conv``.
    (r"^encoder\.downsamples\.(\d+)\.downsamples\.2\.",
     r"encoder.down_blocks.\1.downsampler."),
    (r"^encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.",
     r"encoder.down_blocks.\1.resnets.\2."),
    (r"^encoder\.downsamples\.(\d+)\.resample\.1\.",
     r"encoder.down_blocks.\1.downsampler.resample.1."),
    # The upsampler is the LAST entry of ComfyUI's ``upsamples`` list (index 3),
    # while diffusers keeps it beside the resnets as ``upsampler``.  Matched by
    # position rather than by name because both are just numbered entries; the
    # shapes confirm it -- index 3 is the only one carrying ``resample`` and
    # ``time_conv``, which is what an upsampler has and a resnet does not.
    (r"^decoder\.upsamples\.(\d+)\.upsamples\.3\.",
     r"decoder.up_blocks.\1.upsampler."),
    (r"^decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.",
     r"decoder.up_blocks.\1.resnets.\2."),
    (r"^decoder\.upsamples\.(\d+)\.resample\.1\.",
     r"decoder.up_blocks.\1.upsampler.resample.1."),
    # Attention inside the middle stack.
    (r"\.to_qkv\.", ".to_qkv."),
]


def rename(key):
    out = key
    for pattern, replacement in RULES:
        out = re.sub(pattern, replacement, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from safetensors import safe_open
    from diffusers import AutoencoderKLWan

    with torch.device("meta"):
        wanted = {k: tuple(v.shape)
                  for k, v in AutoencoderKLWan(**CONFIG).state_dict().items()}

    state = {}
    with safe_open(args.source, framework="pt") as handle:
        for key in handle.keys():
            state[rename(key)] = handle.get_tensor(key).to(torch.float32)

    missing = sorted(set(wanted) - set(state))
    extra = sorted(set(state) - set(wanted))
    wrong = [k for k in set(state) & set(wanted)
             if tuple(state[k].shape) != wanted[k]]
    print("{} tensors | missing {} | extra {} | shape mismatch {}".format(
        len(state), len(missing), len(extra), len(wrong)))
    for k in missing[:10]:
        print("  missing:", k, wanted[k])
    for k in extra[:10]:
        print("  extra  :", k, tuple(state[k].shape))
    for k in wrong[:6]:
        print("  wrong  :", k, tuple(state[k].shape), "want", wanted[k])
    save_file(state, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
