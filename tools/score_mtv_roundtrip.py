"""Can MTV-Crafter's tokenizer reproduce OUR motion at all?

THE POINT OF DOING THIS FIRST.  Everything downstream of the tokenizer -- the
adapter, the sampler, the render -- is conditioned on whatever comes out of its
codebook.  If our dance does not survive the round trip, no amount of sampler
tuning matters and the 17.69 GB fallback base would not help either.  It costs
one forward pass and no GPU sampler run, so it goes before the first render
rather than after the first disappointment.

WHAT THE READING MEANS.  The error is per joint in MILLIMETRES, which is the
unit the operator's own judgements are in (this repository already scales
effects that way: 112 cm is a different dance, 10 cm is invisible).  The
reference is the clip's own body: a 1.7 m dancer, so a 20 mm median error is
about 1% of standing height and far below anything a viewer resolves, while a
100 mm error on a wrist is a visibly different arm.

CONTROLS, so the number can be read:
  * ground truth through the same round trip -- if OUR motion reconstructs worse
    than real captured motion does, the gap is ours, not the tokenizer's;
  * the per-joint breakdown -- wrists and ankles move fastest and are where a
    temporal codebook loses the most, so a flat average would hide it;
  * the DEPTH axis reported separately, because depth is the whole reason for
    taking this path and an encoder that collapsed it would still look fine on
    a total-distance average.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
import os

# Root of the sibling checkouts (ComfyUI_Wan, Lodge, this repo); set E2E_ROOT.
E2E_ROOT = os.environ.get("E2E_ROOT", "/workspace/e2e")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
WRAPPER = pathlib.Path(E2E_ROOT + "/ComfyUI_Wan/custom_nodes/"
                       "ComfyUI-WanVideoWrapper")

JOINT_NAMES = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2",
               "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot", "neck",
               "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
               "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand"]


def load_vqvae(path, device):
    """Built exactly as ``MTV/nodes.py:LoadVQVAE`` builds it -- the same class,
    the same widths -- so a shape that loads here loads there."""
    import importlib
    import types
    if "wanwrap" not in sys.modules:
        for name, sub in (("wanwrap", ""), ("wanwrap.MTV", "MTV"),
                          ("wanwrap.MTV.motion4d", "MTV/motion4d")):
            module = types.ModuleType(name)
            module.__path__ = [str(WRAPPER / sub)] if sub else [str(WRAPPER)]
            sys.modules[name] = module
    # The classes live in ``motion4d/vqvae.py``; the package __init__ re-exports
    # them for the node, but importing the module directly is what makes this
    # runnable outside ComfyUI.
    m4d = importlib.import_module("wanwrap.MTV.motion4d.vqvae")
    from safetensors.torch import load_file

    encoder = m4d.Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072,
                          downsample_time=[2, 2], downsample_joint=[1, 1])
    quant = m4d.VectorQuantizer(nb_code=8192, code_dim=3072)
    decoder = m4d.Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3,
                          upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0],
                          joint_upsample_rate=[1.0, 1.0])
    model = m4d.SMPL_VQVAE(encoder, decoder, quant).to(device)
    model.load_state_dict(load_file(str(path)), strict=True)
    model.eval()
    return model


def round_trip(model, camera, mean, std, device):
    normalised = torch.tensor((camera - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        out = model(normalised.unsqueeze(0).to(device))
    recon = out[0][0].to(dtype=torch.float32).cpu().numpy() * std + mean
    return recon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqvae", default="/cache/mtvcrafter/"
                    "WanVideo_MTV_Crafter_4DMoT_VQVAE_fp32.safetensors")
    ap.add_argument("--clips", default="runs/vis_clips_t10.txt")
    ap.add_argument("--arm", default="/cache/atomicdance-assets/runs/t_beat/floorslew")
    ap.add_argument("--truth", default="runs/txy_t_gt_eval/motion")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from render2d.mtv_motion import prepare, load_stats

    device = torch.device(args.device)
    model = load_vqvae(args.vqvae, device)
    mean, std = load_stats()

    names = [line.strip() for line in open(args.clips) if line.strip()]
    rows = {"generated": [], "ground truth": []}
    per_joint = {"generated": [], "ground truth": []}
    per_axis = {"generated": [], "ground truth": []}
    for clip in names:
        for tag, root in (("generated", pathlib.Path(args.arm)),
                          ("ground truth", pathlib.Path(args.truth))):
            path = root / (clip + ".pkl")
            if not path.is_file():
                continue
            camera, _, _ = prepare(path)
            recon = round_trip(model, camera, mean, std, device)
            span = min(len(camera), len(recon))
            error = np.linalg.norm(camera[:span] - recon[:span], axis=-1)
            rows[tag].append(float(np.median(error)))
            per_joint[tag].append(np.median(error, axis=0))
            per_axis[tag].append(np.median(np.abs(camera[:span] - recon[:span]),
                                           axis=(0, 1)))

    print("VQ-VAE round trip, per-joint error in MILLIMETRES ({} clips)"
          .format(len(rows["generated"])))
    for tag in ("ground truth", "generated"):
        if not rows[tag]:
            print("  {:14s} no clip found".format(tag))
            continue
        axis = np.median(np.stack(per_axis[tag]), axis=0)
        print("  {:14s} median {:6.1f} mm   worst clip {:6.1f} mm   "
              "per axis x {:.1f} / y {:.1f} / depth {:.1f}".format(
                  tag, float(np.median(rows[tag])), float(np.max(rows[tag])), *axis))
    if per_joint["generated"]:
        joint = np.median(np.stack(per_joint["generated"]), axis=0)
        order = np.argsort(-joint)[:6]
        print("  worst joints (generated): " + ", ".join(
            "{} {:.0f}mm".format(JOINT_NAMES[i], joint[i]) for i in order))


if __name__ == "__main__":
    main()
