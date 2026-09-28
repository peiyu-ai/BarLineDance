# Wild 3D → BarLineDance 训练数据发布链

本文件定义从 wild 原视频到 `train_atomic.py` 可读训练包的唯一发布顺序。它将
“已经有 2D pose cache”“WHAM 命令跑过”和“可用于训练”明确分开；任何跳过其中
一个 gate 的目录都只能作调试输入，不能作为实验数据根目录。

## 当前已发布的 AIST 训练基线（不是论文复现或模型结果）

本 checkout 已发布一个可由 `train_atomic.py` 读取的、source-safe 的 AIST 3D
kinematic pseudo-label baseline：

- `aist_raw_performance_v1` 以 performance（去除 legacy `_chNN`）为 split 与
  retrieval group 单位，保留 838 / 114 / 40 条 train / val / test sequence，分别来自
  218 / 26 / 20 个 performance；旧窗口 labels 从未读取；
- `aist_performance_normalizer_v1` 只以 838 条 train sequence、354,195 帧拟合 151-D
  min/max；`aist_performance_normalized_v2` 用同一冻结 artifact 处理全部 992 条
  sequence；
- `aist_kinematic_labels_v2` 在完整 3D timeline 上仅以 train 拟合 segmenter、descriptor
  statistics、K-Means 和 acceptance threshold，发布 100 类、13,552 个 train event；它是
  可审计的运动学 baseline，**不是**论文 I3D/TMR/LLM semantic discovery 的复现；
- `aist_kinematic_release_v1` 只物化 label mask 完整有效的 150-frame / stride-15 窗口：
  4,460 / 245 / 128 个 train / val / test 窗口。它带有逐 split、hash 固定的
  `retrieval_groups.json`，全体 performance group 两两不相交。

该 release 的 train/test 重叠标签一致性为 1.0，train atomic-frame 的 source-safe
retrieval coverage 为 1.0，并已通过 planner / completion 的单步 CPU 训练 smoke。后两者
只验证读取与闭环数据契约：
completion 使用 target plan 的路径仍标记 `ORACLE_GROUND_TRUTH_PLAN`，全部不构成模型
指标或 headline result。此 baseline 不含任何 wild 3D 样本，也不能据此声称论文复现或
wild fine-tune 效果。

## 当前已发布的证据

| 产物 | 事实 | 训练资格 |
| --- | --- | --- |
| `data/atomic_aistpp/source_manifest_v3` | 992 条 AIST legacy camera/source timeline（264 个 performance）、18,105 个重建窗口；不读取上游窗口 label | 仅供 train-only discovery/labeling 输入 |
| `data/atomic_aistpp/aist_raw_performance_v1` | 992 条 raw z-up / body-only 151-D sequence；performance-held train/val/test 为 838/114/40 | 仅供 train-only fit/discovery 输入 |
| `data/atomic_aistpp/aist_performance_normalizer_v1` + `aist_performance_normalized_v2` | normalizer 仅以 838 条 train sequence 拟合；全部 split 使用同一 151-D artifact | 可作为 label / materializer 输入 |
| `data/atomic_aistpp/aist_kinematic_labels_v2` | 完整 timeline 3D kinematic pseudo-label；100 类、13,552/1,522/396 train/val/test events | 可作为 baseline 标签输入，非 semantic 论文复现 |
| `data/atomic_aistpp/aist_kinematic_release_v1` | 4,460/245/128 个全 valid 150-frame windows；30 FPS、normalizer 与 retrieval group sidecar 已 hash 固定 | 可用于 source-safe train/val/test baseline；不是 wild release 或模型成绩 |
| `data/wild3d/source_manifests/tiktok_new_b2gpu/exact_content_audit_full_v1` | 2,846/2,846 条原始 recording 完成 SHA-256；未发现跨 split 的**字节完全相同**文件 | 不放行训练 |
| `data/wild3d/source_manifests/tiktok_new_b2gpu/exact_content_merged_v1` | 2,846 source / 6,041 sequence 已写入 content hash；原 split 未改 | 全部 `accepted_for_training=false` |

最后一个 bundle 明确保持
`split_status=provisional_pending_near_duplicate_content_qc`。精确 byte hash 不能发现
重编码、裁剪、trim、镜像、感知近重复或语义重复，因此
`duplicate_content_qc_complete=false` 是正确状态，而不是待补写的字段。

## 不可跳过的依赖图

```text
冻结 recording split
  → exact-byte 内容审计并入新 manifest
  → near-duplicate QC + world-HMR/3D/human QC
  → 明确接受完整 raw 151-D sequence
  → 仅 train source 拟合 normalizer
  → 用同一冻结 normalizer 物化 train/val/test 模型输入
  → train-only recognizer/discovery 产生完整 sequence 标签与 mask
  → 仅将全 valid 的 150-frame view 物化为 indexed 训练 bundle
  → source-safe retrieval audit、训练与 held-out 评测
```

每一箭头均产生一个新的、不可覆盖的版本目录；不就地重写 raw motion、source manifest
或旧的结果。`recording_id` 是 split 与内容审计的根单位；`retrieval_group_id` 是
prototype exclusion 的单位（AIST 中为同一 performance 的所有 camera view）。二者都不能
用 clip、track 或 window 名代替。

## 0. 已验证的 AIST 构建链与未来重建

当前目录名带版本且不可覆盖；如需重建，必须选择新的输出目录，而不是改写下列已发布
artifact。下面是一条与 `aist_kinematic_release_v1` 相同的构建顺序（示例使用新的 `*_v2`
名称）：

```bash
python tools/rebase_atomic_aist_source.py \
  --source-bundle data/atomic_aistpp/source_manifest_v3 \
  --upstream-normalizer data/atomic_aistpp/normalizer.pt \
  --output-dir data/atomic_aistpp/aist_raw_performance_v2

python tools/fit_motion_normalizer.py \
  --sequence-manifest data/atomic_aistpp/aist_raw_performance_v2/sequences.jsonl \
  --source-manifest data/atomic_aistpp/aist_raw_performance_v2/sources.jsonl \
  --output-dir data/atomic_aistpp/aist_performance_normalizer_v2

python tools/apply_motion_normalizer.py \
  --sequence-manifest data/atomic_aistpp/aist_raw_performance_v2/sequences.jsonl \
  --source-manifest data/atomic_aistpp/aist_raw_performance_v2/sources.jsonl \
  --normalizer-bundle data/atomic_aistpp/aist_performance_normalizer_v2 \
  --output-dir data/atomic_aistpp/aist_performance_normalized_v3

python tools/discover_kinematic_atomics.py \
  --sources data/atomic_aistpp/aist_raw_performance_v2/sources.jsonl \
  --sequences data/atomic_aistpp/aist_performance_normalized_v3/sequences_normalized.jsonl \
  --output-dir data/atomic_aistpp/aist_kinematic_labels_v3 \
  --device auto

python tools/materialize_atomic_windows.py \
  --sources data/atomic_aistpp/aist_raw_performance_v2/sources.jsonl \
  --sequences data/atomic_aistpp/aist_performance_normalized_v3/sequences_normalized.jsonl \
  --labels data/atomic_aistpp/aist_kinematic_labels_v3/labels.jsonl \
  --output-dir data/atomic_aistpp/aist_kinematic_release_v2 \
  --window-length 150 --window-stride 15 --min-label-valid-fraction 1.0

python tools/audit_atomic_dataset.py \
  --data-root data/atomic_aistpp/aist_kinematic_release_v2 \
  --eval-source-list '' --window-stride 15 \
  --min-overlap-label-agreement 1.0 --min-safe-retrieval-fraction 0.99 \
  --output data/wild3d/reports/aist_kinematic_release_v2_audit.json
```

这个链路不读取上游 `labels.npy`，不把 camera 拼入 151-D model input，也不让 val/test
参与 normalizer、vocabulary 或 acceptance threshold 的拟合。上述 audit 与
`train_atomic.py` 的 release contract 都必须通过，才可以把新 release 用于训练。

## 1. 固化精确内容审计

当前 b2gpu bundle 已运行过下列命令，输出是
`exact_content_merged_v1`。以后对新的 staging version 必须再次运行，且输出目录必须是新版本：

```bash
python tools/merge_wild_content_audit.py \
  --sources data/wild3d/source_manifests/tiktok_new_b2gpu/sources.jsonl \
  --sequences data/wild3d/source_manifests/tiktok_new_b2gpu/sequences.jsonl \
  --audit-report data/wild3d/source_manifests/tiktok_new_b2gpu/exact_content_audit_full_v1/report.json \
  --audit-records data/wild3d/source_manifests/tiktok_new_b2gpu/exact_content_audit_full_v1/duplicate_content_audit.jsonl \
  --output-dir data/wild3d/source_manifests/tiktok_new_b2gpu/exact_content_merged_v2
```

该工具对 report、输入 `sources.jsonl` 和 audit JSONL 都做 SHA-256 绑定，要求每个
source 恰有一条成功 hash 记录，并拒绝精确重复跨 split。它只传播已经观察到的精确
duplicate group，绝不重划 split、更不把样本升为训练可用。

## 2. 完成 3D、近重复与人工 QC 后冻结 raw release

WHAM 转换、reconcile 与音频对齐见 [WILD_3D_PREPROCESSING.md](WILD_3D_PREPROCESSING.md)。
只有满足下列条件的 sequence 才能进入这个 release：

1. `hmr_status=candidate`，strict `frame_ids` 连续，world-HMR 151D 是
   `AtomicDance_151D / z_up_world_body_only / raw / camera_in_model_input=false`；
2. source 级近重复 QC 已完成，所有发现的 duplicate group 同 split；
3. 3D/global render、身份稳定、镜头切换、reprojection 和人工 QC 已通过；
4. source 与 sequence 都显式 `qc.accepted_for_training=true`。未通过者留在
   pending/quarantine，不能用默认值或补零绕过。

相机 `camera.npz` 保留作审计副产物；其未注册 DPVO gauge 绝不能拼进 151D motion。

## 3. 训练集拟合、全 split 应用同一 normalizer

对一个已经冻结且接受的 raw release，先拟合。该命令只读取
`split=train && qc.accepted_for_training=true` 的 raw motion；val/test 即使有数组也不参与
min/max：

```bash
python tools/fit_motion_normalizer.py \
  --sequence-manifest <RAW_RELEASE>/sequences_raw_accepted.jsonl \
  --source-manifest <RAW_RELEASE>/sources.jsonl \
  --output-dir <RAW_RELEASE>/normalizer_train_v1
```

然后使用该 artifact 原样处理 release 中所有 split，而不是在 val/test 重拟合：

```bash
python tools/apply_motion_normalizer.py \
  --sequence-manifest <RAW_RELEASE>/sequences_raw_accepted.jsonl \
  --source-manifest <RAW_RELEASE>/sources.jsonl \
  --normalizer-bundle <RAW_RELEASE>/normalizer_train_v1 \
  --output-dir <RAW_RELEASE>/motion_normalized_v1
```

变换固定为 `2 * (raw - data_min) / safe_range - 1`；constant dimension 的
`safe_range=1`，不 clip、不修复 NaN。输出同时保存 raw motion、camera、audio 和新的
`motion_151_normalized`，并把 normalizer artifact SHA-256 与 `fit_split=train` 写回
sequence provenance。fit report 同时绑定 source **和完整 sequence manifest** 的 SHA-256；
两者任一字节变化（包括新增/替换 raw sequence）都必须重新 fit，不能给新清单套旧统计。
训练代码不会在 dataloader 中自动 normalize，所以 indexed
训练包必须只使用该冻结的 normalized motion。

## 4. 全 sequence 赋标签，再物化窗口

train-only recognizer / discovery 的输出必须生成独立 `labels.jsonl`，每条 accepted label
绑定：当前 `sources.jsonl` hash、输入 normalized motion hash、151D/坐标/normalizer
contract、producer artifact hash，以及 `label_valid_mask`。`0` 仅表示已知 transition；
未知帧必须是 `-1` 加 false mask。

`aist_kinematic_labels_v2` 已走通这条完整 sequence → mask → window bridge，但它只是
运动学 pseudo-label baseline。wild 数据在通过 3D / duplicate / human QC 后，仍需由
train-only recognizer 或冻结的 label producer 生成同样完整、带 uncertainty 的 label
timeline，不能把无标签 pose 直接送入 `train_atomic.py`。随后用下列 bridge 生成 legacy
上游 array 格式：

```bash
python tools/materialize_atomic_windows.py \
  --sources <RAW_RELEASE>/sources.jsonl \
  --sequences <RAW_RELEASE>/motion_normalized_v1/sequences_normalized.jsonl \
  --labels <LABEL_RELEASE>/labels.jsonl \
  --output-dir data/atomic_wild/<RELEASE_VERSION>
```

materializer 会校验 source/sequence/label 外键、group split、label 的 train-only source
manifest hash、frame alignment、motion/music hash 与表示契约。它仅复制连续时间线的
精确 range view；当前 D3PM 数据格式不支持无效 label loss，因此任何 partial mask 或
`-1` 都会写入 `quarantine.jsonl`，不会变成类 0。训练窗名仍解析为其完整
`retrieval_group_id` sidecar，使 completion retrieval 可排除同一 performance / 同源
prototype。

输出 `build.json`、`windows.jsonl`、`quarantine.jsonl` 和 split arrays 都应与 checkpoint /
evaluation report 一起固定。只有 `build.json` 显示 train-only normalizer、完整 label
provenance、非零 train/val/test windows，且 source-safe retrieval coverage 通过门槛时，才可把
该目录传给 `train_atomic.py`。训练期只可根据 val 选择设置；test 留给冻结 checkpoint 的最终
held-out 报告。

相关的身份与标签语义见 [SOURCE_MANIFEST_CONTRACT.md](SOURCE_MANIFEST_CONTRACT.md)。
