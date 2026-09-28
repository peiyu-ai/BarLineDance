# T line generation configs

One place that names what "the current T-line model" is. Before 2026-09-22 the shipped
argv (arm `fix7`) existed only as a scratch file, and the library and the two checkpoints
were three independent flags that nothing compared.

| file | what |
|---|---|
| `current.argv` / `current.env` | **the default**: a one-line pointer (`@configs/t_line/t2.argv`, `. configs/t_line/t2.env`). Switching versions = editing this line. |
| `t2.argv` / `t2.env` | T2 (2026-09-22): library `scratch/txy_t2/release_aligned_k8_20260922` (+24 clips from 17 new 汤汤汤小圆 uploads), retrained planner (seed 20260901, chosen on val beat-hit) and completion, regenerated sidecars, re-decoded audio overlay. Shipped 3D arm for the fixed 10: `runs/t_beat/t2_ship_vis10`; render: `output/sample_20260922_t2_shipped/` |
| `t1_fix7.argv` / `t1_fix7.env` | T1 = fix7 (2026-09-17), kept as a named fallback; reproduces `fix7a_20` byte-for-byte |
| `t2_fix8.argv` | T2 data/models with the **fix8 recipe** (DEFECTS §90, `--seam-transition after --draft-only`, no stagger/inpaint). Not the default — fix8 has no operator verdict yet. Adopting it = pointing `current.argv` here, which keeps the new library and models. |
| `t2_fix8_turns.argv` | `t2_fix8.argv` with `--face-camera 0` (DEFECTS §91): keeps face_camera's rigid mean-facing rotation, drops the slow pull that aborted sustained turns ("转身一半"). Completed turns test/val 7/12 → 11/18 (GT 12/24), off-camera share back to GT's level; plan and units identical to fix8. Not the default — no operator verdict yet. |
| `t2_fix8_cont.argv` | `t2_fix8_turns` + **source continuation** (DEFECTS §92.2): at a bar line the unit's own dancer continues into their next bar (any label, ≤4 bars in a row, lookahead) — 73–76% of bar lines stop being a cut; reversals right after the bar line 0.71/0.72 → 0.58/0.63 (GT 0.52/0.50). Not the default. |
| `t2_fix8_full.argv` | `t2_fix8_cont` + **prefer-full (stops) 0.25 + hold-by-music** (DEFECTS §92.3): picks the quarter of candidates whose moves land most extended — elbow at the stops +7°/+10°, reach +0.04/+0.06 vs cont; energy 1.14/1.07× GT (up to 1.22 on another plan). **Rejected by the operator (2026-09-23)**: 舒展 does not mean picking the biggest moves, and its render hit the beat worse than T1's. Kept only so §92.3 can be reproduced. |
| `t2_fix8_beat.argv` | `t2_fix8_cont` + **downbeat stop on continued bar lines + the learned rhythm scorer** (`model/rhythm_scorer.py`, keep the best-aligned quarter of each slot's candidates; DEFECTS §93). Beat gain test/val 0.457/0.408 vs T2 ship 0.187/0.112 and T1 0.216/0.220; both halves of every clip hit (100%/100%); the same gain on the T1 library (0.41–0.46), i.e. it no longer depends on which library it is given. Not the default — awaiting the operator's verdict on `output/sample_20260923_beat/`. |

Two axes, kept separate: the **data/models version** (T1 → T2: which library and checkpoints) and the
**recipe** (fix7 → fix8: which inference flags). `current` picks one of each.

Generate (from the repo root; per-invocation flags go after the `@file` and override it):

```
python3 infer_atomic.py @configs/t_line/current.argv \
    --audio-dir <dir of <name>.npy music features> --sequence-list <names> --output-dir <new dir>
```

Render / score with the matching environment: `set -a; . configs/t_line/current.env; set +a`.

Guards (tests/test_t_line_config.py, and infer_atomic itself):

* `current` resolves to T2 in every generation flag; no T1 path appears in it.
* T1 and T2 differ only in data root, the two checkpoints, four sidecars, `--floor-anchor` and `--ingest-root`.
* infer_atomic refuses a planner/completion trained on a different release than `--data-root` was
  derived from (`build.json` `derived_from`); the manifest records `release_binding`, plus
  `generation_config` (argv + sha256 of each `@` file), `audio_dir` and `ingest_root`.
  A deliberate mix (library ablation under fixed models) needs `--allow-cross-release-checkpoints`.

How T2 was built and evaluated: `runs/txy_t2_20260922/` and worklog 2026-09-22.
