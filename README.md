# BarLineDance

Music-driven dance generation, composed bar by bar from real movement.
Project page with generated samples, measurements and their caveats: <https://peiyu-ai.github.io/barline-page/>.

BarLineDance is a fork of [AtomicDance](https://github.com/oceanflowlab/AtomicDance), the code release of
*Music-to-Dance Generation via Atomic Movements* (Cai et al., ECCV 2026), and keeps its MIT license.
It is not affiliated with that paper's authors.  The upstream README is kept as
[UPSTREAM_README.md](UPSTREAM_README.md); first-person statements, results, checkpoint links and
environment versions there are the upstream authors' (except the section marked as added by this fork) and
have not been re-validated for this fork.

## What this fork changes

The upstream two-stage design stays: a planner chooses a sequence of movement labels, retrieval fills
the plan with real movement from a library, and a completion model smooths the seams.  This fork:

* adapts the pipeline to dance videos from the web, reconstructed to 3D (world-grounded human mesh
  recovery), instead of AIST++'s multi-view studio recordings;
* lays the dance on the song's own beat grid: one movement label per four-beat bar, and each bar is
  filled with a retrieved bar of real dance, re-timed to fit;
* lets a source dancer continue: at a bar line the previous bar's source clip may carry on into its own
  next bar (up to four in a row) whatever the planned label is.  In the project page's whole songs and
  continuations, 54-65% of bars are continued and 47-59% show a movement the planner did not choose;
* chooses among retrieval candidates with rules that read only the query song and the library
  (never the target dance), for example how footfalls land against this song's beats and how busy the
  movement is where the song is loud or quiet;
* generates whole songs, continuing a dance past the end of its source clip;
* adds a 2D rendering path (`render2d/`) that drives a rendered character with the generated 3D motion.

The version shown on the project page is **K13**.  Its generation config is
[`configs/t_line/k13.argv`](configs/t_line/k13.argv), which `tests/test_t_line_config.py` pins to the argv
recorded by the page's whole-song runs.  The page's continuations and its 50-clip measurements used the same
flags plus `--plan-from` (an earlier run's plan from the same planner, so compared versions share one plan).

## What is and is not in this repository

This is research code published so that the method on the project page can be read.  It is not a
packaged release, and it does not run end to end from a fresh clone.

Not included:

* the motion library and training data (reconstructed from third-party dance videos, not redistributed),
  the songs, the trained checkpoints and the rendered media;
* the working notes that many comments cite (`CLAUDE.md`, `worklog.md`, `docs/`, `runs/`, `output/`,
  scratch scripts); a comment that points at one of them records where a number came from, and the
  number cannot be checked from this repository alone;
* body-model assets under restrictive licenses (SMPL and SMPL-X models, the body-part segmentation
  files, the SMPLitex texture), and the VRM sample avatar (VRM Public License 1.0), left out only for
  size; the notes under `third_party/` say where each comes from.

Local storage paths are written as `/path/to/...`, and the source-video ids are replaced with
placeholders of the same shape (`7000000000000000NNN`); comments abbreviate a placeholder as `uNNN`.

## Layout

| path | what |
|---|---|
| `infer_atomic.py` | generation: planner, retrieval draft, completion, and the retrieval rules |
| `train_atomic.py`, `model/`, `dataset/` | planner and completion models, the retrieval selector and the two scorers, their training and data loading |
| `configs/t_line/` | named generation configs (`k13.argv` is the project page's version) |
| `tools/` | data preparation, reconstruction wrappers, training of the selector and scorers, evaluation rulers, experiment drivers, 3D rendering |
| `render2d/` | 2D character rendering driven by the generated motion (the page uses its SteadyDancer path) |
| `data/`, `eval/`, `vis.py` | from upstream: the 35-D music features (`data/audio_extraction/`), AIST++ splits, evaluation |
| `third_party/` | vendored DWPose, import shims, notes on the assets that are not included |
| `tests/` | unit tests; many need the local data and skip without it, and the 3D-render tests need `pyrender` |

Generation, from the repository root, once the library, checkpoints and sidecar files that the config
names exist:

```bash
python3 infer_atomic.py @configs/t_line/k13.argv \
    --audio-dir <dir of <name>.npy music features or <name>.wav> \
    --sequence-list <file with one name per line> --output-dir <new dir>
```

The music features are 35-D at 30 fps (`data/audio_extraction/baseline_features.py`).

Tests are run one file per process, for example `python3 -m pytest -q tests/test_t_line_config.py`.
The fork was developed on Linux with Python 3.12 and CUDA 12; `requirements.txt` is upstream's and
has not been updated.

## License

MIT, see [LICENSE](LICENSE) (inherited through AtomicDance from EDGE).  Exceptions, each under its own terms:

* code adapted from PyTorch3D, torch_scatter, rotary-embedding-torch, scikit-learn and fairmotion:
  the files, their origins and the licence texts are listed in
  [third_party/licenses/README.md](third_party/licenses/README.md), and each file names its source;
* `third_party/DWPose/dwpose/`: adapted from IDEA-Research/DWPose, Apache-2.0
  ([third_party/DWPose/LICENSE](third_party/DWPose/LICENSE));
* `render2d/patches/*.patch`: changes to kijai/ComfyUI-WanVideoWrapper, which is Apache-2.0;
* `third_party/smpl_models/uv/smpl_uv.obj`: the SMPL UV template, CC BY 4.0 as stated in its header.

Songs remain the property of their rights holders.  `tools/song_stores.py` and `tools/retrieve_full_song.py`
identify and fetch songs through third-party services; they are here so the continuation pipeline can be
read, need your own accounts, and must be used only within those services' terms and copyright law.

## Citation

BarLineDance has no paper yet.  If you use the upstream method, please cite:

```bibtex
@inproceedings{cai2026atomicdance,
  title={Music-to-Dance Generation via Atomic Movements},
  author={Cai, Xinhao and Sun, Yixuan and Zheng, Minghang and Chen, Qingchao and
          Jin, Xin and Zhu, Song-chun and Liu, Yang},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
