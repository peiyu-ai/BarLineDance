# Source-level 数据 manifest 契约

这份契约是上游发布包 + wild 微调数据的唯一身份、切分和标签语义规范。它解决
上游包把 `*_sliceN` 当样本、但没有可靠完整 source 标签的问题；在实现新的
dataset builder 前也可作为 review checklist。任何不满足本文件的数据只能用于
code smoke，不能用于 held-out 训练或评测。

## 三层身份

| ID | 表示什么 | 不变量 |
| --- | --- | --- |
| `recording_id` | 一条原始舞蹈录制（同一视频/音频/人物表演的所有裁剪、转码和导出） | split、内容审计和可回溯身份的根单位；不可由 window 序号替代 |
| `retrieval_group_id` | 不能互相作为 prototype 的录制集合 | 必须完全落在一个 split；completion 的 exclusion 单位。wild 默认等于 `recording_id`，AIST 则是去掉 `_chNN` 的同一 performance，覆盖所有相机视角 |
| `sequence_id` | 一个 `recording_id` 内、单人、连续且无镜头切换/帧缺口的 3D 运动时间线 | 只属于一个 recording；先完成 3D QC、分段和赋标签，后切窗 |
| `window_id` | 从一个 sequence 导出的固定训练窗 | 必须记录 sequence-relative `[start_frame, end_frame)`；不得跨 sequence 或依赖 dataloader index |

`sequence_id` 和 `window_id` 不能被误当成 completion exclusion key。对于 AIST，去掉
`_sliceN` **只**能恢复一个 recording 名，仍必须用显式 `retrieval_group_id` 排除同一
performance 的所有 `_chNN` 相机视角。对 wild 数据，同一个原视频的多个
segment/track 都继承同一个 `recording_id`，通常也继承同一个 `retrieval_group_id`。

当前 `source_id_from_name()` 仅是 legacy package 的兼容解析，不能识别重命名、重编码或
一个 performance 的多个相机。新数据的 indexed dataset/inference item 必须显式携带
与 array 对齐的 `retrieval_group_id` sidecar；一旦 retrieval 请求 group exclusion，
缺失或未知 provenance 必须 fail-closed（零 draft / quarantine），不得把它当作
“不同 source”。外部推理名称没有精确 sidecar mapping 时也必须 zero draft；禁止在运行时
剥离 `_chNN` 猜 group，后续若需要 alias 必须单独冻结映射和 hash。

## 规范化 manifest

manifest 采用 UTF-8 JSONL，字段名固定、ID 全局唯一，所有派生文件都引用这些 ID
而不是隐式文件名。建议在一个数据版本下保存以下四张表。

### `sources.jsonl`

每个 `recording_id` 一行，至少含：

```json
{
  "schema_version": "source-manifest-v1",
  "recording_id": "aistpp/gBR_sBM_cAll_d04_mBR1_ch03",
  "retrieval_group_id": "aistpp/gBR_sBM_cAll_d04_mBR1",
  "duplicate_content_group_id": null,
  "split": "train",
  "source_kind": "aistpp|wild",
  "raw_uri": "...",
  "content_sha256": "...",
  "fps": 30,
  "audio_id": "...",
  "dancer_id": "...",
  "provenance": {"dataset_version": "...", "ingested_at": "..."}
}
```

`raw_uri` 可以是受控的相对路径或资产 ID；若隐私策略不允许保存账号/人名，应保存
稳定的脱敏 ID。`content_sha256`、原视频 hash、音频 hash 或可比较的感知 hash 至少
要有一种，以发现重复上传、转码和裁剪后的跨 split 泄漏。

### `sequences.jsonl`

每个无间断的单人运动时间线一行，至少含 `sequence_id`、`recording_id`、
`person_track_id`、`source_start_frame`、`source_end_frame_exclusive`、`frame_count`、
`fps`、`is_contiguous`、`motion_path`、`coordinate_system`、`preprocess_version` 和
`qc_status`。`frame_ids` 必须严格递增；有 gap、镜头切换或 ID switch 的部分拆成
新的 sequence 或 quarantine，不能用 padding 抹平。

### `labels.jsonl`

标签以**完整 sequence**为对象，而不是对每个 150 帧 window 独立预测。每行至少含
`sequence_id`、`labels_path`、`label_valid_mask_path`、`confidence_path`、
`entropy_path`、`label_space_id`、`producer_version`、`fit_split` 和输入 motion hash。
若使用 soft label，也须记录 `probabilities_path`、类别顺序和温度/校准版本。

### `windows.jsonl`

每个训练窗一行，至少含 `window_id`、`sequence_id`、冗余的 `recording_id`（用于
join 校验）、`split`、`start_frame`、`end_frame_exclusive`、`length`、`motion_path`、
`music_path`、`label_space_id`、`label_valid_fraction` 与 builder version。第一版固定
`length=150`、30 FPS；所有 array 必须与该精确 frame range 对齐。window 只引用完整
sequence 标签的同一段 view，不能存一份重新推断的重叠标签。

## Source-held-out 切分

1. 在拟合 normalizer、atomic discovery、cluster center、recognizer、pseudo-label
   阈值或 prototype 前，创建并冻结 `sources.jsonl` 的 split。一个 `recording_id`、
   `retrieval_group_id` 或已知 duplicate group 的所有 sequence/window 必须属于同一 split。
2. train、validation、test 的 `recording_id` 与 `retrieval_group_id` 交集必须为零；内容/音频/感知 hash
   命中的派生录制也必须同侧，无法判定则 quarantine。所有 split 指标按 recording
   聚合或 bootstrap，不能因滑窗数量而放大长视频权重。
3. normalizer、原子词表/cluster、prototype library、recognizer、pseudo-label
   calibration 和任何 feature fine-tuning 只能 `fit_split: train`。validation/test
   只能调用冻结的产物，绝不参与 fit。
4. completion retrieval 对每个 query 排除相同 `retrieval_group_id`、相同 sequence 和
   时间相交 candidate；validation/test library 只来自 train。无安全 candidate 时
   保持 zero-mask/跳过，不得退回同源 GT。当前训练还须通过配置的
   `safe_draft_condition_fraction` 门槛（默认 `>= 0.99`）。
5. song、dancer、账号等元数据是额外泛化轴，应单独报告 unseen-song / unseen-dancer
   / unseen-account 结果；它们不能替代 recording-level 隔离。

`split_manifest_sha256` 和 sources/sequences/windows/labels 的版本或 hash 必须进入
每次训练、checkpoint 与评测 report。之后不得仅通过重排 dataloader 或重写
`crossmodal_test.txt` 改变 held-out 定义。

## 标签与 mask 语义

原子类别 `0..100` 中的 `0` 是一个**已知的 transition**，因此只有
`label=0 && label_valid_mask=true` 才表示 transition。未知、低置信或尚未人工审核
的 pseudo label 必须使用无效 mask（规范存储可用 `label=-1`）并保留 confidence /
entropy；绝不能把它改写为 `0` 来凑齐训练 shape。

训练 loss、class coverage 和 duration 统计都必须应用 `label_valid_mask`。不允许的
frame 应被 masked 或 quarantine，而非作为 transition 参与监督。pseudo label 的
producer 必须是 train-only 的、冻结版本；它不得读取 target future motion、测试
source、目标歌曲 GT 或 oracle plan。

对同一 `sequence_id` 的两个重叠 windows，所有同时 `valid` 的 shared frame 必须有
相同 label、mask、label-space 和 producer version。任一冲突说明标签不是在完整
source 上生成：隔离该 sequence 并重新 discovery/赋标，不能用 majority vote 把
冲突静默覆盖。

## 构建顺序与 release gate

```text
冻结 recording split
  → exact-byte 内容审计；近重复/3D/human QC 后明确接受连续 raw 3D sequence
  → 仅以 accepted train source 拟合 raw normalizer / vocabulary / recognizer
  → 用同一个冻结 normalizer 物化 train/val/test 的模型输入（绝不重拟合）
  → 对完整 normalized sequence 产生 labels + mask + uncertainty
  → 切 150-frame windows，并仅作 labels 的 range view
  → source-safe retrieval audit → 训练 / held-out 评测
```

builder 的最低自动检查为：ID 唯一性与外键、split 继承、frame range/连续性、motion/
music/label shape 和有限值、label 范围与 mask 语义、重叠 window 一致性、跨 split
content 重复、train-only 产物 provenance、以及 source-safe draft coverage。任何一项
失败均生成 quarantine/report，不自动修正输入。

## 对当前 release 的迁移规则

官方 `atomic_aistpp` 可用于解析连续 motion/music 和 151D 格式，但其 package 中
窗口 labels 不能被升级为 canonical source labels：审计已证明共享 motion/music
帧存在冲突标签。迁移时先按受审计的 `recording_id` 重建完整 sequence，丢弃这些
window-local labels 作为训练真值，再以 train-only 流程重做 discovery/labeling。wild
candidate 也必须先拥有 recording → sequence → window 的 manifest 链，才可进入
pseudo-label 阶段。

相关证据与执行阶段见 [UPSTREAM_RELEASE_AUDIT.md](UPSTREAM_RELEASE_AUDIT.md)、
[FINETUNE_PLAN.md](FINETUNE_PLAN.md) 和
[WILD_3D_PREPROCESSING.md](WILD_3D_PREPROCESSING.md)。可执行的 immutable release
顺序和当前 wild 资产状态见 [TRAINING_DATA_RELEASE.md](TRAINING_DATA_RELEASE.md)。
