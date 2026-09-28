# Lodge 开发复盘：新 BarLineDance 路线的输入约束

本文不是对 Lodge 的否定。它把问题空间、野外数据工程和大量失败模式暴露得很充分；新项目应继承这些可复用资产，而不应把已有的选招/粗细两阶段链继续当作主架构。

证据均来自同级仓库 `../Lodge`，尤其是 `docs/interaction/dev_story.md`、`docs/interaction/current-plan-v2.md`、`docs/interaction/_iter432_gt_dependency_audit.md` 与野外数据脚本。本文固定于引入上游方法时的状态，后续实验以本仓库的记录为准。

## 一句话结论

Lodge 值得带走的是「2D pose 数据链 + 清洗/QC + 真实推理评测纪律」；不值得带走的是「按歌曲规则选招的动作库、coarse-to-fine 主链、以及把人、相机、人物尺度混在 image-space 坐标里学习」。

新主线必须先获得可审计的 3D body/world motion 与单独相机轨迹，再把上游方法的 atomic planning 与 completion 训练在这个表示上。

## 应继承的资产

### 1. 野外 2D pose 数据工程

已验证可复用的流程为：

```text
原始视频
  → 16 秒、30 FPS 切片
  → DWPose / 置信度
  → keypoints、scores、audio、meta
  → 全 clip 清洗（raw / clean / clean2 并存）
  → 节奏特征和质量审计
  → 数据集合并与完整歌曲可视化
```

关键实现：

- `../Lodge/scripts/preprocess_wild_videos.py`
- `../Lodge/dld/data/video_pose_extractor.py`
- `../Lodge/scripts/clean_gt_pose.py`
- `../Lodge/scripts/audit_gt_pose_quality.py`
- `../Lodge/docs/interaction/wild_data_update_pipeline.md`

新项目不复制或覆盖这些原始资产。每个 3D 样本必须继续能追溯到：`pose2d_raw`、`pose2d_clean`、`confidence`、原视频、音频、preprocess 版本和 quality report。

### 2. 清洗是数据质量收益，不是可选后处理

Lodge 记录显示，原始野外 keypoint 中的插值坡道、闪跳和末端抖动会被模型直接学成 swimming / 漂移。对整段 clip 做 confidence-gated Hampel + 自适应 One-Euro 清洗，比仅对 loss 降低低置信度权重可靠；后者会错误地丢掉静止、遮挡和收束动作。

因此：

- 清洗在完整 clip 上进行，不能按训练窗口各自处理；
- 清洗后的几何可作为 target，但 confidence/visibility 不能丢；
- raw observation 永远保留，禁止 in-place 覆盖；
- 3D lifting 前不做过度平滑，长遮挡和身份跳变应 quarantine，而不是伪造连续值。

### 3. 评测与实验纪律

Lodge 最有价值的另一部分是负面证据：短窗口、单 scalar metric 与 teacher-forced 修补都可能看起来很好，却不能证明可部署生成能力。新项目从第一天起固定以下标签：

| 标签 | 含义 | 是否可进入主结果 |
| --- | --- | --- |
| `SELF_DRIVEN` | 推理时没有目标 future-GT、目标动作、目标相机或逐歌规则 | 可以 |
| `ORACLE` | 使用 GT 只为定位上限 / 做消融 | 不可以 |
| `CHEAT` | GT 进入初始化、条件、选择、缩放、门控或后处理 | 不可以 |
| `META_TUNED` | 用测试歌曲 GT 选 recipe、随机种子或超参 | 不可以 |

每个实验都要保存命令、配置、数据版本、checkpoint、seed，并输出完整歌曲、原速、带音频视频。`Tang10` 已被大量历史决策使用，只能作为开发集，不再称为无偏 test set。

## 不应继承的架构性问题

### 1. coarse-to-fine 不能自动提供结构

如果 coarse 只表达平滑 root / energy envelope，关键 pose、动作过程与 phrase 边界仍全部压给 fine；层级只是在转移 conditional-mean collapse。Lodge 的 GT-coarse oracle 可改善局部指标，但 predicted-coarse closed loop 仍显著退化，且曾出现训练使用 refined hop、推理输入 planner hop 的条件分布不一致。

上游方法的两阶段划分不同：第一阶段是可读的 **atomic movement plan**，第二阶段是明确以 prototype / transition 为条件的 **motion completion**。它仍需做 GT-plan oracle 与 predicted-plan closed-loop 的逐层报告，但不应退化成抽稀关键帧再由 fine 猜全部动作。

### 2. 动作库应该是数据工具，不是最终决策器

Lodge 证明 1--2.5 秒完整动作单元比单帧 token 更有舞蹈语义，这一点保留；但动作库选招在后期逐渐变成规则检索拼接。当前审计中 learned-alone 的独立选招比例为 0%，且不同歌曲使用历史方案 mosaic，无法说明统一泛化模型。

新系统里的 atomic library 只承担：

- 原子动作发现、去重和覆盖率统计；
- duration-aware prototype retrieval；
- 邻近样本诊断 / 非学习基线；
- completion 的条件，而非逐歌规则选择器。

### 3. 2D image coordinates 把相机与人混在一起

`x / width, y / height` 同时编码了镜头 pan/zoom、透视深度、人物大小与真实 root travel。Lodge 后续的 rigid bone、free log-length 与 2.5D 补偿正说明这种表示是结构性瓶颈，而不是加几个 loss 可以解决的问题。

新语义必须显式拆开：

```text
observation:   raw/clean 2D pose + visibility/confidence
camera:        intrinsics + camera-to-world pose (or explicitly unregistered VO gauge) + registration / scale caveat
body:          SMPL local joint rotations（或 root-relative 3D joints）
root_motion:   world / canonical root translation 与 global orientation
quality:       reprojection、contact、bone、identity、cut、uncertainty
```

`tools/preprocess_wild_3d.py` 是这一约束的第一个可运行实现：它把 WHAM 的全局 3D 输出和 DPVO 相机轨迹分别保存，拒绝只含 camera-space pose 的结果。

## 新项目的硬性门槛

1. 按 source video / song / dancer / account 分组切分，禁止窗口随机切分。
2. Atomic label、normalizer、训练样本只能从 train source 生成；验证/测试不得泄漏进入 prototype library 或 cluster center。
3. 先做 3D 与相机 QC，再进入 atomic discovery；低质 clip 不强行插值进训练集。
4. 所有 full inference 必须在删除 future-GT 字段后仍可运行。
5. 同时报告 pose realism、动作幅度、jerk/flash、root/contact、music normal/zero/shuffle/wrong-song、diversity 和结构一致性；BAS 仅是 sanity check。
6. 先完成无 wild fine-tune 的上游复现，再逐步引入 wild；任何提升必须有 AIST-only、mixed-data、source-held-out 三个对照。

## 最小迁移清单

```text
从 Lodge 带走：视频/2D pose cache、cleaner、QC 思路、音频及完整歌曲评测。
在 BarLineDance 重建：3D SMPL/world/camera corpus、原子发现/对齐、planner、completion。
绝不带走：测试歌 GT、逐歌 recipe、GT 选动作、GT scale/floor/anchor、teacher-forced headline。
```
