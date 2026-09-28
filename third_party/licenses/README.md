# Licences of third-party code adapted in this repository

| file(s) here | adapted from | licence text |
|---|---|---|
| `dataset/rotation_ops.py` (quaternion, 6-D and matrix/quaternion routines) | PyTorch3D, `pytorch3d/transforms/rotation_conversions.py` (Meta Platforms) | `pytorch3d_LICENSE` (BSD) |
| `third_party/torch_scatter_compat/torch_scatter/__init__.py` (`broadcast`, `scatter_sum`) | torch_scatter (Matthias Fey) | `torch_scatter_LICENSE` (MIT) |
| `model/rotary_embedding_torch.py` | lucidrains/rotary-embedding-torch (Phil Wang) | `rotary-embedding-torch_LICENSE` (MIT) |
| `dataset/scaler.py` | scikit-learn's `MinMaxScaler` and `_handle_zeros_in_scale`, ported to torch | `scikit-learn_COPYING` (BSD-3-Clause) |
| `eval/utils/helpers.py`, `eval/utils/kinetic.py`, `eval/utils/manual.py` | fairmotion (Facebook), modified by Ruilong Li | `fairmotion_LICENSE` (BSD) |

The texts were copied from each project's repository (scikit-learn's from its installed distribution) on 2026-09-28.
