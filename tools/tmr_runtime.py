#!/usr/bin/env python3
"""Load TMR's released encoders and embed motion or text in their joint space.

The paper's M2 clusters segments in TMR's motion-text embedding space.  TMR
publishes weights but no importable package, so this reconstructs exactly the
inference path its own code takes, using its vendored modules rather than a
reimplementation:

* motion: Guo 263-D -> ``Normalizer`` ``(x - mean) / (std + eps)`` ->
  ``ACTORStyleEncoder`` -> **mu**, the first of the two VAE tokens.  TMR's
  ``encode`` defaults to ``sample_mean=True``, so retrieval uses mu and never
  the reparameterised sample -- taking the sample instead would add noise to
  every cluster assignment.
* text: DistilBERT token embeddings (768-D, *not* pooled) -> the same encoder
  architecture with its own weights -> mu.

Both land in the same 256-D space, which is what makes the T1 gate possible:
probe sentences must retrieve the segments they describe.  Nothing else in this
repo can check that the bridge into TMR is wired correctly -- shapes flowing is
not evidence.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TMR_ROOT = REPO_ROOT / "third_party" / "TMR"
DEFAULT_MODEL = "tmr_humanml3d_guoh3dfeats"
LATENT_DIM = 256
GUOFEATS_DIM = 263
TEXT_MODEL = "distilbert-base-uncased"


class TMRUnavailable(RuntimeError):
    pass


def _load_module(name: str, path: pathlib.Path):
    """Load by file path: TMR's modules import as ``src.model``, which we lack."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TMRUnavailable("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TMREncoder:
    def __init__(self, root: pathlib.Path = TMR_ROOT, model: str = DEFAULT_MODEL,
                 device: str = "cpu", text: bool = True):
        self.root = pathlib.Path(root)
        self.device = torch.device(device)
        weights = self.root / "models" / model / "last_weights"
        if not (weights / "motion_encoder.pt").is_file():
            raise TMRUnavailable(
                "no TMR weights at {}; run tools/setup_tmr_env.sh".format(weights))

        config = json.loads((self.root / "models" / model / "config.json").read_text())
        motion_cfg = config["model"]["motion_encoder"]
        text_cfg = config["model"]["text_encoder"]

        actor = _load_module("tmr_actor", self.root / "model_ref" / "actor.py")
        self.motion_encoder = self._build(actor, motion_cfg, weights / "motion_encoder.pt")
        self.text_encoder = None
        self._text_to_emb = None
        if text:
            self.text_encoder = self._build(actor, text_cfg, weights / "text_encoder.pt")

        stats = self.root / "stats" / "humanml3d" / "guoh3dfeats"
        self.mean = torch.load(stats / "mean.pt", map_location="cpu", weights_only=False).float()
        self.std = torch.load(stats / "std.pt", map_location="cpu", weights_only=False).float()
        self.eps = 1e-12
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

    def _build(self, actor, cfg: Dict, weights_path: pathlib.Path):
        encoder = actor.ACTORStyleEncoder(
            nfeats=cfg["nfeats"], vae=cfg["vae"], latent_dim=cfg["latent_dim"],
            ff_size=cfg["ff_size"], num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"], dropout=cfg["dropout"],
            activation=cfg["activation"],
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        missing, unexpected = encoder.load_state_dict(state, strict=True), None
        del missing, unexpected
        return encoder.eval().to(self.device)

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / (self.std + self.eps)

    @torch.no_grad()
    def encode_motion(self, segments: Sequence[np.ndarray], batch_size: int = 64
                      ) -> np.ndarray:
        """[N] variable-length Guo 263-D arrays -> [N, 256] mu embeddings."""
        if not segments:
            return np.zeros((0, LATENT_DIM), dtype=np.float32)
        out: List[np.ndarray] = []
        for start in range(0, len(segments), batch_size):
            chunk = segments[start : start + batch_size]
            lengths = [len(np.asarray(s)) for s in chunk]
            if min(lengths) < 1:
                raise TMRUnavailable("empty motion segment in batch at {}".format(start))
            longest = max(lengths)
            padded = torch.zeros(len(chunk), longest, GUOFEATS_DIM)
            mask = torch.zeros(len(chunk), longest, dtype=torch.bool)
            for i, segment in enumerate(chunk):
                array = torch.from_numpy(np.asarray(segment, dtype=np.float32))
                padded[i, : len(array)] = array
                mask[i, : len(array)] = True
            padded = self.normalize(padded.to(self.device))
            # Padding is normalized too, then masked out; the encoder never
            # attends to it, so its value cannot reach the embedding.
            mu = self.motion_encoder({"x": padded, "mask": mask.to(self.device)})[:, 0]
            out.append(mu.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        if self.text_encoder is None:
            raise TMRUnavailable("built without the text encoder")
        if self._text_to_emb is None:
            text_module = _load_module("tmr_text_encoder",
                                       self.root / "model_ref" / "text_encoder.py")
            self._text_to_emb = text_module.TextToEmb(
                TEXT_MODEL, mean_pooling=False, device=str(self.device))
        embedded = self._text_to_emb(list(texts))
        tokens = embedded["x"].to(self.device)
        lengths = embedded["length"]
        mask = torch.arange(tokens.shape[1], device=self.device)[None] < lengths.to(
            self.device)[:, None]
        mu = self.text_encoder({"x": tokens, "mask": mask})[:, 0]
        return mu.cpu().numpy().astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T


def available(root: pathlib.Path = TMR_ROOT, model: str = DEFAULT_MODEL) -> bool:
    return (root / "models" / model / "last_weights" / "motion_encoder.pt").is_file()


if __name__ == "__main__":  # tiny self-check
    encoder = TMREncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    motion = [np.random.default_rng(0).normal(size=(40, GUOFEATS_DIM)).astype(np.float32)]
    print("motion embedding:", encoder.encode_motion(motion).shape)
    print("text embedding:", encoder.encode_text(["a person kicks"]).shape)
