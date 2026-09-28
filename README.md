# BarLineDance

**Music-driven dance generation at the level of musical bars: learn moves from in-the-wild dance videos, then assemble a new dance along a song's bar lines.**

Given a song, BarLineDance produces a 3D dance (an SMPL joint sequence), which can then be rendered as a 2D skeleton and as an animated cartoon character.
Every move comes from a real dancer. In-the-wild dance videos are reconstructed in 3D and cut into bar-length movement units.
At generation time a planner picks a movement class for each bar of the song. The best-fitting real clips are then retrieved from the movement library, spliced together, and smoothed at the seams.

---

## How it works

```
In-the-wild dance videos
  │  tools/ingest_wild_uploads.py     cut clips at shot changes / dancer exits; track the dancer with DWPose
  │  tools/run_gvhmr_extract.py       monocular 3D reconstruction with GVHMR -> world-frame SMPL-X
  │  tools/convert_gvhmr_result.py    -> 151-D motion (4 contacts + 3 root translation + 24 x rot6d, z-up)
  ▼
Cut on the music's beat grid into 4-beat units (one bar) -> cluster into a movement vocabulary -> build a release
  │  tools/build_bar_planner_release.py   one token = one bar
  ▼
Generation (infer_atomic.py)
  1. planner          listens to the whole song and assigns a movement class to every bar (model/atomic_planner.py)
  2. label -> motion  retrieves real clips of that class from the library, time-stretches them to the bar, and scores
                      the candidates: learned selector (model/retrieval_selector.py), rhythm scorer (model/rhythm_scorer.py),
                      same-dancer continuation across bars (model/continuation_scorer.py), step-to-beat lock, energy following, ...
  3. completion       a diffusion model redraws only the neighbourhood of each bar seam (model/atomic_completion.py)
  ▼
3D dance (.pkl)
  │  tools/render_sample_strip.py     source video + skeleton + VRM skinned avatar + plan timeline, side by side
  │  render2d/run_2d_steadydancer.sh  3D -> AAPose 2D pose video -> Wan2.1 SteadyDancer cartoon character video
  ▼
Videos to watch
```

Key points:

- **The unit is the bar.** Movement boundaries fall on the song's bar lines, not on fixed-length frame windows.
- **Moves are retrieved, not invented.** The content of every bar comes from a recording of a real dancer in the library.
  The diffusion model only handles the transitions near the seams.
- **Generation sees only the music.** Any timing alignment may read the music only. Reading the target clip's ground-truth
  motion counts as cheating (see `CLAUDE.md` §1.6).

## Status

- The default generation config registered in the repo is **T2** (`configs/t_line/current.argv` -> `t2.argv`).
  How the versions differ is described in [`configs/t_line/README.md`](configs/t_line/README.md).
- Later iterations (F11 / J7 / K13, ...) add `infer_atomic.py` flags on top of T2's data and models.
  Each arm's definition, measurements and the operator's verdict are recorded in
  [`docs/DANCE_QUALITY_DEFECTS.md`](docs/DANCE_QUALITY_DEFECTS.md) and [`worklog.md`](worklog.md).
  Their argv files live under `runs/` and are not checked in.
- The main open problem: **the timing of the moves does not follow the song.** Beats often land in the middle of a move
  instead of at its end point. The symptoms, the measured structural facts and the approaches that are off-limits are
  written up in [`CLAUDE.md`](CLAUDE.md) §1.6.

## Repository layout

| Path | Contents |
|---|---|
| `infer_atomic.py` | Generation entry point: planner -> retrieval and splicing -> completion -> 3D output. It has many flags; pass them via `@configs/...argv` |
| `train_atomic.py` | Training entry point, `--stage planner` or `--stage completion` |
| `model/` | Planner, completion diffusion model, retrieval selector, rhythm and continuation scorers |
| `dataset/` | Dataset loading, bar tokens (`bar_tokens.py`), movement discovery (`atomic_discovery.py`), representation conversion |
| `configs/t_line/` | Named generation configs (argv) and render environments (env); `current.*` points at the default |
| `tools/` | In-the-wild data pipeline (ingest, GVHMR, conversion, audits), release building, rendering, OSS asset management |
| `render2d/` | 3D -> 2D cartoon video (AAPose + SteadyDancer / Wan Animate, via ComfyUI) |
| `eval/` | FID / diversity / Beat Alignment Score evaluation on AIST++ |
| `tests/` | Unit tests (about 240 files) |
| `third_party/` | DWPose, VRM character model, pytorch3d / torch_scatter compatibility shims, etc. |
| `SMPL-to-FBX/` | Export SMPL motion to FBX |
| `docs/` | Design notes and diagnostic write-ups (see below) |
| `worklog.md` | Dated work log (what was done / evidence / conclusion); newest entries at the end |
| `CLAUDE.md` | Working rules for the repo: storage discipline, how metrics must be validated, how to report |

## Environment

Linux, Python 3.7, PyTorch 1.12.1 + CUDA 11.6. A GPU with at least 16 GB of memory is recommended for training and inference.

```bash
conda create -n barline python=3.7 -y
conda activate barline
pip install -r requirements.txt
pip install git+https://github.com/rodrigo-castellon/jukemirlib.git@a91d87fcae0dd89085752421e794ea7e1b300735
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.1
```

Additional requirements per stage:

| Stage | Requirement |
|---|---|
| 3D reconstruction | GVHMR checked out at `third_party/GVHMR` (not included), optionally DPVO; model weights obtained separately |
| SMPL | SMPL model files, downloaded from the [SMPL website](https://smpl.is.tue.mpg.de/) under its license |
| 2D rendering | ComfyUI + Wan2.2-Animate / SteadyDancer weights, see [`render2d/README.md`](render2d/README.md) |
| Tests | `pytest` |

**The repository holds code only.** Datasets, preprocessing outputs, model weights, video and audio are not in git
(see `.gitignore`). In this project's environment they are stored on OSS with a local read cache under `/cache`, and the paths
in `configs/t_line/*.argv` point there. To run elsewhere, substitute your own data and checkpoints.

## Usage

Run all commands from the repository root.

### 1. In-the-wild video -> 3D

```bash
# Cut clips by content (shot changes, dancer leaving the frame) rather than by fixed duration.
# --max-seconds has no default; measure it with tools/scan_clip_length.py first.
python3 tools/ingest_wild_uploads.py --videos-dir <upload dir> --out-root <clip output dir> --max-seconds <N>

# Monocular 3D reconstruction (inside the GVHMR checkout, with the compat shim first on PYTHONPATH)
cd third_party/GVHMR
PYTHONPATH=../pytorch3d_compat:. python ../../tools/run_gvhmr_extract.py \
    --video <clip.mp4> --output-root <dir>
cd -

# Convert to the 151-D representation
python3 tools/convert_gvhmr_result.py \
    --result <dir>/<stem>/hmr4d_results.pt \
    --extract-meta <dir>/<stem>/extract_meta.json \
    --output-dir <converted output dir>
```

For batch runs use `tools/run_gvhmr_ingest_shard*.{sh,py}`. Details and pitfalls are in
[`docs/WILD_3D_PREPROCESSING.md`](docs/WILD_3D_PREPROCESSING.md) and
[`docs/WILD_ATOMIC_PIPELINE_PLAN.md`](docs/WILD_ATOMIC_PIPELINE_PLAN.md).

### 2. Build a training release

The rules for cutting, labelling and splitting are in [`docs/TRAINING_DATA_RELEASE.md`](docs/TRAINING_DATA_RELEASE.md)
and [`docs/SOURCE_MANIFEST_CONTRACT.md`](docs/SOURCE_MANIFEST_CONTRACT.md).
The planner's "one token = one bar" release is built with the tool below (defaults: 4-bar windows, 4 beats per bar, 21 classes):

```bash
python3 tools/build_bar_planner_release.py --help
```

Note: the motion stored in a bar release is an all-zero placeholder. **It may only be used to train the planner.**
The completion model needs a frame-level release.

### 3. Training

```bash
python train_atomic.py --stage planner    --data-root <release> --output-dir runs/<planner name> --device cuda
python train_atomic.py --stage completion --data-root <release> --output-dir runs/<completion name> --device cuda
```

Use `--resume CHECKPOINT` to continue training and `--max-steps 10` for a short debugging run. The seeds, step counts and
selection rationale used for T2 are documented in the comments of `configs/t_line/t2.argv`.

### 4. Generation

```bash
python3 infer_atomic.py @configs/t_line/current.argv \
    --audio-dir <dir holding <name>.npy music features for each song> \
    --sequence-list <list of names to generate> \
    --output-dir <a new directory>
```

- Arguments after the `@file` override the same arguments inside it, so a single flag can be tried without copying the whole config.
- `infer_atomic.py` refuses to pair a planner / completion checkpoint trained on a different release with `--data-root`.
  The output manifest records the full argv and the sha256 of every `@` file, so a run can be reproduced.
- Switching the default version means editing the single line in `configs/t_line/current.argv`.

### 5. Rendering and review

First load the environment variables that pair with the generation config:

```bash
set -a; . configs/t_line/current.env; set +a
```

**Side-by-side review strip** (source video + raw skeleton + VRM skinned avatar + plan timeline), always on the same fixed ten clips:

```bash
python3 tools/render_sample_strip.py \
    --clips runs/vis_clips_t10.txt \
    --arm <arm name>=<generation output dir> \
    --out output/sample_YYYYMMDD_<keyword>/
```

`--arm` can be repeated for a side-by-side comparison. `runs/vis_clips_t10.txt` lives under `runs/` and is not checked in.

**2D cartoon character video** (source video | 3D skeleton | 2D skeleton | skinned character, four panels):

```bash
render2d/run_2d_steadydancer.sh <generation output dir> <output dir>
```

To judge whether a generation is good, watch the videos first, then use metrics to locate and cross-check issues.
The reasoning and the concrete procedure are in `CLAUDE.md` §1.5.

### 6. Evaluation and tests

Evaluation on AIST++ (FID, diversity, BAS): `python -m eval.evaluate --help`.
The metrics used on in-the-wild data (beat lock, step-to-beat lock, cross-song diversity, ...) are defined, together with
their controls, in `docs/DANCE_QUALITY_DEFECTS.md`.

```bash
python -m pytest tests/
```

## Documentation

The documents under `docs/`, `worklog.md` and `CLAUDE.md` are written in Chinese.

| Document | Contents |
|---|---|
| [`docs/DANCE_QUALITY_DEFECTS.md`](docs/DANCE_QUALITY_DEFECTS.md) | Dance-quality defects diagnosed one by one, and every round of experimental arms (the longest and most current) |
| [`docs/T_SERIES_PLAN.md`](docs/T_SERIES_PLAN.md) | T line: 4-beat segmentation, 20-class vocabulary, training plan |
| [`docs/MUSIC_CONDITIONING_ALIGNMENT.md`](docs/MUSIC_CONDITIONING_ALIGNMENT.md) | Music conditioning and alignment |
| [`docs/VOCABULARY_DIAGNOSIS.md`](docs/VOCABULARY_DIAGNOSIS.md) | Movement vocabulary diagnosis |
| [`docs/ROUGHNESS_DIAGNOSIS.md`](docs/ROUGHNESS_DIAGNOSIS.md) | Motion roughness / jitter diagnosis |
| [`docs/WILD_3D_PREPROCESSING.md`](docs/WILD_3D_PREPROCESSING.md) | 3D preprocessing of in-the-wild videos |
| [`docs/WILD_ATOMIC_PIPELINE_PLAN.md`](docs/WILD_ATOMIC_PIPELINE_PLAN.md) · [`docs/WILD_CLEAN5_PIPELINE_PLAN.md`](docs/WILD_CLEAN5_PIPELINE_PLAN.md) | In-the-wild data pipeline plans |
| [`docs/TRAINING_DATA_RELEASE.md`](docs/TRAINING_DATA_RELEASE.md) · [`docs/SOURCE_MANIFEST_CONTRACT.md`](docs/SOURCE_MANIFEST_CONTRACT.md) | Training-data release chain and manifest contract |
| [`docs/UPSTREAM_RELEASE_AUDIT.md`](docs/UPSTREAM_RELEASE_AUDIT.md) | Audit of the baseline data release (split overlap, inconsistent labels) |
| [`docs/FINETUNE_PLAN.md`](docs/FINETUNE_PLAN.md) · [`docs/LODGE_LESSONS.md`](docs/LODGE_LESSONS.md) | Plan and lessons carried over from Lodge |

## Acknowledgements and license

3D reconstruction uses [GVHMR](https://github.com/zju3dv/GVHMR), 2D rendering uses Wan Animate / SteadyDancer,
pose detection uses DWPose, and the body model is [SMPL](https://smpl.is.tue.mpg.de/).

The code is released under the [LICENSE](LICENSE) (MIT, inherited from [EDGE](https://github.com/Stanford-TML/EDGE)).
SMPL, GVHMR, all model weights, and the music and video material remain subject to their own licenses and are not
licensed by this repository.
