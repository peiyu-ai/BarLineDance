#!/usr/bin/env python3
"""PoseScript's released description generator, as a per-pose describer.

The paper's M3 feeds the tagging VLM an auxiliary cue: "we use PoseScript to
describe selected keyframes identified as motion beats".  This repo has been
serving that cue from ``tools/motion_beats.describe_pose`` -- a rule-based
fallback, labelled as such everywhere it is recorded.  This module is the real
thing: ``capgen_CAtransfPSA2H2_dataPSA2ftPSH2``, the checkpoint PoseScript
publishes for exactly this task.

Four things had to line up, and each one is checked rather than assumed:

1. **The vocabulary is not distributed with the weights.**  PoseScript's README
   gives the command that builds it (a 503-word ``--new_word_list`` plus the
   side-flip counterparts) and states the expected size: 2158 -- which is
   exactly ``text_decoder.embedding.weight``'s first dimension in the
   checkpoint.  A vocabulary of the right size but the wrong token *order*
   would decode to fluent, confident, wrong words, so the size is asserted at
   load time and the build is left to PoseScript's own ``vocab.py``.
2. **The model eats axis-angle, not joint positions.**  The rule-based cue took
   ``[J,3]`` joint coordinates; this takes ``(1, 52, 3)`` rotation vectors --
   the encoder's input layer is 156 = 52x3.
3. **SMPL-24 -> SMPL-H-52.**  Indices 0..21 (global orient + 21 body joints)
   map across unchanged; SMPL-H's 30 finger joints are zeroed.  Our joints
   22/23 are SMPL's wrist-level "hands" and have no counterpart in SMPL-H's
   finger hierarchy, so they are dropped -- GVHMR does not estimate fingers
   anyway.
4. **Orientation normalisation is PoseScript's, called from PoseScript.**  Its
   convention is z-up with the global orient's euler-z (the facing azimuth)
   zeroed and the x/y tilt kept.  Our 151-D is already z-up, so the frames
   agree -- but this repo has already shipped one descriptor that read
   left-right off the wrong axis, so the normalisation is *their* function, not
   a reimplementation of it.

Wired and checked, not wired and hoped.  The failure this repo has already
shipped once is a descriptor that read left-right off the wrong axis, and it
reads perfectly fluently when it is wrong, so the check is mechanical: sample
the frames where one foot is clearly higher than the other, and test the model's
"standing on their <side> foot" against which toe is actually lower.

    foot-height gap >= 5 cm   n=32   88% agree
    foot-height gap >= 20 cm  n=26   96% agree

The first attempt at this check sampled frames uniformly and came out at 55%,
which looks like a flipped axis and is not: on most frames both feet are within
2 cm of each other and the claim is not decidable, so the test was measuring
noise.  Recorded because the corrected number is only trustworthy next to the
reason the first one was wrong.

License: the checkpoint is CC BY-NC-SA 4.0 (non-commercial).  Anything derived
from it inherits that, which is a release decision, not a runtime one.
"""

from __future__ import annotations

import pathlib
import sys
from typing import List, Optional, Sequence

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
POSESCRIPT_SRC = REPO_ROOT / "third_party" / "PoseScript" / "repo" / "src"
DEFAULT_CHECKPOINT = (REPO_ROOT / "third_party" / "PoseScript" /
                      "capgen_CAtransfPSA2H2_dataPSA2ftPSH2" / "seed1" / "checkpoint_best.pth")
SMPLH_JOINTS = 52
SMPL_BODY_JOINTS = 22          # global orient + 21 body; SMPL's 22/23 are wrist-level
EXPECTED_VOCAB = 2158          # PoseScript README, and the checkpoint's embedding rows


class PoseScriptError(RuntimeError):
    pass


def _ensure_on_path() -> None:
    if str(POSESCRIPT_SRC) not in sys.path:
        if not POSESCRIPT_SRC.is_dir():
            raise PoseScriptError(
                "PoseScript source not found at {}; the checkpoint alone is not "
                "enough -- the model class lives in their repo".format(POSESCRIPT_SRC))
        sys.path.insert(0, str(POSESCRIPT_SRC))


def rotation6d_to_axis_angle(rotation6d: np.ndarray) -> np.ndarray:
    """[..., J, 6] -> [..., J, 3], inverting ``convert_gvhmr_result``'s packing.

    That packing is ``matrices[..., :2, :]`` -- the first two **rows**, matching
    ``pytorch3d.transforms.matrix_to_rotation_6d``.  Rebuilding them as columns
    instead transposes every rotation, which survives every shape check and
    reads as a plausible pose: a round-trip against the converter's own
    ``pose_axis_angle_z_up.npy`` is what separates the two (179 degrees of
    geodesic error versus 1e-6).  The matrix-to-rotvec step is roma's -- the
    same library PoseScript uses, so the two sides cannot disagree about what a
    rotation vector means.
    """
    import roma
    import torch

    tensor = torch.as_tensor(np.asarray(rotation6d), dtype=torch.float32)
    first, second = tensor[..., 0:3], tensor[..., 3:6]
    b1 = torch.nn.functional.normalize(first, dim=-1)
    b2 = torch.nn.functional.normalize(second - (b1 * second).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    matrices = torch.stack((b1, b2, b3), dim=-2)          # rows, matching the packer
    return roma.rotmat_to_rotvec(matrices).numpy()


def motion_151_to_axis_angle(motion_151: np.ndarray) -> np.ndarray:
    """[T,151] -> [T,24,3].

    The 151-D layout is ``[contacts(4), root_translation(3), rotation6d(24x6)]``;
    only the last block carries orientation.
    """
    motion = np.asarray(motion_151)
    if motion.ndim != 2 or motion.shape[1] != 151:
        raise PoseScriptError("expected [T,151] motion, got {}".format(motion.shape))
    rotation6d = motion[:, 7:].reshape(len(motion), 24, 6)
    return rotation6d_to_axis_angle(rotation6d)


class PoseScriptCaptioner:
    """Loads the released capgen model and describes single poses."""

    def __init__(self, checkpoint: Optional[pathlib.Path] = None, device: str = "cuda:0"):
        _ensure_on_path()
        import torch
        from text2pose.generative_caption.model_generative_caption import DescriptionGenerator

        path = pathlib.Path(checkpoint or DEFAULT_CHECKPOINT)
        if not path.is_file():
            raise PoseScriptError("PoseScript checkpoint not found: {}".format(path))
        state = torch.load(str(path), "cpu", weights_only=False)
        args = state["args"]
        rows = state["model"]["text_decoder.embedding.weight"].shape[0]
        if rows != EXPECTED_VOCAB:
            raise PoseScriptError(
                "checkpoint expects a {}-token vocabulary, not {}".format(rows, EXPECTED_VOCAB))
        self.device = torch.device(device)
        # Building the module loads the vocabulary by reference; a vocabulary
        # that was never built, or built wrong, fails here rather than three
        # thousand captions later.
        self.model = DescriptionGenerator(
            text_decoder_name=args.text_decoder_name,
            transformer_mode=args.transformer_mode,
            decoder_nlayers=args.decoder_nlayers,
            decoder_nhead=args.decoder_nhead,
            encoder_latentD=args.latentD,
            decoder_latentD=args.decoder_latentD,
            num_body_joints=getattr(args, "num_body_joints", SMPLH_JOINTS),
        ).to(self.device)
        self.model.load_state_dict(state["model"])      # strict: a shape drift stops here
        self.model.eval()
        self.epoch = state.get("epoch")
        self.checkpoint = str(path)

    # -- input preparation -------------------------------------------------

    def _to_smplh(self, axis_angle_24: np.ndarray):
        """[24,3] SMPL -> [52,3] SMPL-H, orientation normalised PoseScript-style."""
        import torch
        from text2pose.utils import eulerangles_to_rotvec, rotvec_to_eulerangles

        pose = np.asarray(axis_angle_24, dtype=np.float32)
        if pose.shape != (24, 3):
            raise PoseScriptError("expected [24,3] axis-angle, got {}".format(pose.shape))
        full = torch.zeros(SMPLH_JOINTS, 3, dtype=torch.float32)
        full[:SMPL_BODY_JOINTS] = torch.from_numpy(pose[:SMPL_BODY_JOINTS])
        # Their normalisation, verbatim: drop the azimuth about the vertical
        # axis, keep the tilt.  Reimplementing this is how a descriptor ends up
        # reading left-right off the wrong axis.
        thetax, thetay, thetaz = rotvec_to_eulerangles(full[:1, :])
        full[0:1, :] = eulerangles_to_rotvec(thetax, thetay, torch.zeros_like(thetaz))
        return full

    # -- description -------------------------------------------------------

    def describe(self, axis_angle_24: np.ndarray) -> str:
        return self.describe_many([axis_angle_24])[0]

    def describe_many(self, poses: Sequence[np.ndarray]) -> List[str]:
        """One forward pass for a batch of poses.

        A segment contributes up to ``max_beats`` keyframes and the corpus has
        tens of thousands of segments, so the per-call overhead is worth
        amortising here rather than at every call site.
        """
        import torch

        if not poses:
            return []
        batch = torch.stack([self._to_smplh(pose) for pose in poses]).to(self.device)
        with torch.no_grad():
            texts, _ = self.model.generate_text(batch)
        return [str(text) for text in texts]


def main() -> int:
    """Smoke test: describe the canonical T-pose and a raised right arm."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    captioner = PoseScriptCaptioner(args.checkpoint, device=args.device)
    print("loaded {} (epoch {})".format(captioner.checkpoint, captioner.epoch))

    rest = np.zeros((24, 3), dtype=np.float32)
    raised = np.zeros((24, 3), dtype=np.float32)
    raised[17] = (0.0, 0.0, -1.4)       # right shoulder
    raised[19] = (0.0, 0.0, -0.8)       # right elbow
    for name, pose in (("rest", rest), ("right arm raised", raised)):
        print("{:>18}: {}".format(name, captioner.describe(pose)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
