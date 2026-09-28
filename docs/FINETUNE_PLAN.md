# 上游方法 + 野外 3D 数据开发计划

> **2026-08-08 重大修订。** 三项实测结果推翻了本文原有的阶段排序，详见
> [../worklog.md](../worklog.md)：
>
> 1. **`kinematic_atomic_100_v2` 标签空间与音乐正交**，在 K=5/10/25/50/100
>    **每一档**都不可由音乐预测（held-out 上「配错歌」的代价 ≤ +0.0116）。
>    因此 P2 里「以冻结的 AIST kinematic producer 为 recognizer 起点」这条**不再成立**，
>    不能作为 planner 的标签来源。
> 2. **paper 的 discovery 需要配对视频**（Alg.1 的 I3D 视觉自相似分割），而本机
>    **AIST 原始视频原本一个都没有**（`Lodge/aist_plusplus` 只有 keypoints2d + wav）。
>    这正是本地 kinematic 发现缺失视觉输入的根本原因。现已用
>    `tools/fetch_aist_videos.py` 按 release 引用的 564 条序列补齐（c01 机位，约 8 GB）。
> 3. **wild 语料自带视频**（2849 原视频 / 7060 clip），因此它不只是「额外训练数据」，
>    而是与 AIST 视频并列的 discovery 试验场。
>
> 由此，**P2 的核心工作从「沿用 kinematic 伪标签」改为「重建带视觉/语义输入的
> discovery」**，并且任何新词表在进入训练前必须通过发布 gate：
>
> ```bash
> python tools/audit_atomic_dataset.py --data-root <release> --require-song-disjoint-splits
> ```
>
> 另有两个已确认的数据缺陷需在重建 split 时一并修复：val 与 train **共享 100% 的歌**；
> test 仅 128 窗口且分布偏移（触地率 0.77 vs train 0.45，root 速度只有一半）。

> **2026-08-14 更正：`probe_label_predictability --gate` 从准入 gate 降为诊断。**
> 上面第 1 条的观测（标签空间与音乐正交）没有变，**它作为训练准入判据的资格没了**。
> 08-08 本文自己就写过「帧准确率 > majority 是过强的错误度量」，08-13 却把它重新
> 装成硬闸门，连着否掉六个词表并阻止了 planner 训练。08-14 把被它否掉的两个词表
> 各训了一个 600-epoch planner，`gate v2 --plans` 在 val 上：
>
> | planner | 类数 | split | p |
> | --- | ---: | --- | ---: |
> | songsplit 630 | 630 | song-disjoint | **8.16e-05** PASS |
> | songsplit 1286 | 1,286 | song-disjoint | **4.65e-03** PASS |
>
> 帧级标签不可由音乐预测、与 planner 采样的结构随歌而动，**这两件事同时成立** ——
> paper 的 planner 是结构采样器不是预测器，所以帧准确率判否不能否掉它。
>
> 准入判据因此只剩 release 审计（上面那条）。`probe_label_predictability` 继续跑、
> 继续入档，但**它的判否不再阻止训练**；能阻止训练的是
> `probe_structure_conditioning.py --plans ... --gate`，而它需要先有 checkpoint。
> **没有便宜的前置判据 —— 这是这条路线的真实成本,不是可以绕开的。**
>
> 同一天的第二条：1,286 类那版虽然通过，但 **86% 的输出帧是 transition**、每 150 帧
> 窗口只切 2.94 段（630 类是 9.88 段）。**通过不等于可用**，词表过细时 planner 会退回
> transition。可带走的是 630 类那版。

## 目标与范围

以官方 [上游论文](https://arxiv.org/html/2607.13978)(Cai et al.)与当前官方代码 `oceanflowlab/AtomicDance` 为基线，建立可复现的 music-to-dance 微调路线。目标不是复活 Lodge 的选招器，而是验证：在 camera/body 解耦的野外 3D 语料上，原子动作计划是否能带来更好的结构、节奏和可控性。

本仓库拉取的基线提交：`4fdaea56cf69b57f59cfc88c6691612ab120c5f8`（`main`，2026-07-24）。该提交和所有本项目本地改动必须在实验 manifest 中固定。

## 论文方法与实际代码的边界

论文提出：

```text
完整音乐 → D3PM atomic plan（类型、起止、时长）
        → duration-nearest prototype draft + mask/noise
        → transition-aware continuous diffusion completion
        → 连续的 3D SMPL dance
```

- 原子发现：visual-similarity temporal segmentation → TMR embedding K-Means → 近中心高置信样本 → genre 内 LLM-assisted re-clustering；论文中有 100 个 base prototype。
- planner：frame-level `0..100`（0 是 transition）的离散扩散，随后 majority vote + minimum-duration merge。
- completion：按预测片段时长检索 prototype，对原子片段施加受控 noise，并生成 transition。
- 表示：4 contact + 3 root translation + 24×6D rotation = **151 维**。代码在 `infer_atomic.py:291-301` 与 `train_atomic.py:328` 明确这一点。

不过当前开源版本并不是论文的完整 data-discovery release：

| 项目 | 当前状态 | 对微调的含义 |
| --- | --- | --- |
| Atomic 训练集 | 官方 `atomic_aistpp.zip`，含已对齐 motion/music/labels/normalizer | 可以作为表示与 label-space 标尺 |
| I3D visual feature | 未提供可运行入口 | 新 wild discovery 不能假设已有实现 |
| TMR encoder / checkpoint | 未提供 | 无法直接复现论文的 segment clustering |
| Gemini 标注与 LLM re-cluster | 未提供 prompt、输入输出和 mapping | 不能声称复现完整 semantic discovery |
| 预训练 checkpoints | 本地不存在；README 指向单独 Drive folder | 需取得后核验 hash / stage / train split |
| planner "full music" | 代码的 `seq_len=150`，长音频在 `infer_atomic.py` 中分块 | 这是待修复的公开代码/论文差异，不能忽略 |
| completion prototype retrieval | 上游从整个 train set 建库并允许样本取回自身/重叠 slice；本 checkout 已在 train/inference 排除 query retrieval/performance group（同一表演的全部 camera view）、对未知 provenance fail-closed | 原始 0.5 s stride 的 5 s slice 会造成近重复 GT draft；仍须冻结 performance-group-held-out split，并将 GT-plan diagnostic 和 predicted-plan closed loop 分开报告 |
| 默认训练/评测 split | 包内 train 的 952 个 source 与 `data/splits/crossmodal_test.txt` 的 147 个评测 source 重合 100 个；包内 test 只重合 4 个 | README 默认流程不是 source-held-out benchmark，不能引用其结果作为无泄漏基线 |
| overlapping slice labels | `slice0/1` 的重叠 motion/music 可逐元素相同，但 labels 可仅 51.1% 一致；全 train 的相邻 slice 也存在大量 transition/class 冲突 | 必须在完整 source 上发现/赋原子标签，再切 150 帧 window，并把 overlap consistency 设为入训硬门槛 |

因此，首版微调不把“重新跑论文 discovery”假装为已解决，而采用分阶段、可验证的 pseudo-label 路线。当前已发布的 AIST 3D kinematic labels 是这条路线的
可运行 baseline，不是对论文 I3D/TMR/LLM semantic discovery 的替代或复现。

训练数据的 `recording_id → sequence_id → window_id` 身份、source-held-out split
和 unknown-label mask 的强制语义定义在
[SOURCE_MANIFEST_CONTRACT.md](SOURCE_MANIFEST_CONTRACT.md)。这是 P0/P2 的数据
接口，不是可选的命名约定。

上游 archive 的可复现 hash、shape 与两项数据泄漏证据见
[UPSTREAM_RELEASE_AUDIT.md](UPSTREAM_RELEASE_AUDIT.md)。这不是可选的旁路
审计：P0 未通过前，任何 release 的数值都只能称为 code smoke。

本机已经用 `tools/build_source_manifest.py` 将上游窗口重建为不可变的
`data/atomic_aistpp/source_manifest_v3`：992 条 legacy camera/source timeline（覆盖 264 个
performance）、992 条连续 sequence、18,105 个 window，且每个 motion/music window 与发布包
逐元素一致。该工具故意不读取
`labels.npy`，所以 bundle 的 label 数为 0；它只能作为 train-only full-sequence
discovery/labeling 的输入，绝不能被升级为干净的监督集。若需重建，必须指定一个新的
输出目录（builder 拒绝覆盖已冻结版本）。

在此输入之上，首个 source-safe AIST baseline 已发布为
`data/atomic_aistpp/aist_kinematic_release_v1`：先按 performance 切为
838 / 114 / 40 条 train / val / test sequence，再只以 train 拟合 151-D normalizer 和
kinematic discovery；最终只物化 4,460 / 245 / 128 个 mask 完整的 150-frame 窗口。
它的 train/test 重叠标签一致性与 train atomic-frame source-safe retrieval coverage 都是 1.0，且三 split 的
performance retrieval group 两两不相交。这个事实证明了数据链和防泄漏约束可以工作；
它**不**是模型质量、论文复现或 wild 数据结论。完整 artifact、hash 绑定和复现命令见
[TRAINING_DATA_RELEASE.md](TRAINING_DATA_RELEASE.md)。

## 数据与坐标契约

### 上游 151 维表示的可训练样本

```text
motion_151[T,151]      = [contacts(4), root_translation_z_up(3), rot6d(24,6)]
music[T,35]            = 与 motion 同帧、30 FPS
atomic_labels[T]       = 0 transition，1..100 atomic class
sample metadata        = source/video/song/dancer/split/preprocess version
```

上游 EDGE/AIST conversion 将 y-up 旋转 +90° about X 后放入 z-up 表示；野外转换必须完全一致，否则 normalizer、foot contact 和 root trajectory 的统计会失真。

### 相机与人体必须单独留存

Wild 3D 训练输入可以使用 `motion_151`，但训练语料还必须保存下列不可丢失副产物：

```text
pose2d_raw / pose2d_clean / scores / valid mask
WHAM pose_world/trans_world（或同等 world-HMR 输出）
camera.npz（原始 DPVO c2w、未注册 visual-odometry gauge；translation up-to-scale）
atomic_motion_151.npy（z-up、未归一化）
quality.json / metadata.json
```

这让模型只学习人的动作，而相机可用于审计、数据筛选和后续有显式 registration 的 video renderer；禁止把 camera translation 拼到 root feature 上，也不能把未注册 DPVO 直接当作人体 world 的外参。

## 实施阶段与提升门槛

### P0：上游基线可复现（先于任何 wild 训练）

截至本版本，P0 的数据安全子门槛已经落地：上游 package 审计、label-free source
timeline 重建、performance-held split、train-only normalizer、完整 timeline kinematic
labels、固定 30 FPS 的 indexed release，以及 retrieval-group exclusion 均已通过。
`train_atomic.py` 的 planner / completion 各跑过单步 CPU smoke；completion 仍是
`ORACLE_GROUND_TRUTH_PLAN` diagnostic，不能被引用为生成成绩。剩余的 P0 是在兼容环境
中核验官方 checkpoint / 官方评测资产，并以 predicted plan 得到真实 closed loop。

1. 下载官方 `data/atomic_aistpp`，校验 archive hash、`motion/music/labels/names/normalizer` shape、name alignment、label range 和 normalizer；运行 `tools/audit_atomic_dataset.py`，审计 train/val/test source 与评测清单的交集、以及相邻 window 的 label consistency。
2. 建立适配本机的运行环境。官方 Python 3.7 + Torch 1.12 + CUDA 11.6 不直接适用于本机 Python 3.12 / Blackwell；锁定一个独立 Python 3.10/3.11 环境，先跑 unit/inference smoke，再解决 PyTorch3D/SMPL 依赖。
3. 按 [source manifest contract](SOURCE_MANIFEST_CONTRACT.md) 的 `recording_id` 为单位重建 train/val/test；任何 train source 与最终 evaluation source 的交集必须是 0。先在完整 `sequence_id` 上发现/赋标签（unknown 绝不伪装为 transition），再按固定 `window_id` 切片；相邻 shared frame 的 atomic labels/mask 必须一致，否则样本隔离并回溯 discovery。
4. 在开始 completion 实验前，修正 prototype retrieval：同一 performance 的全部 source/camera view、同一原始 sequence、以及时间重叠 slice 必须从候选库排除；validation/test 只使用 train performance-group 的 prototype library。**本 checkout 已实现 train/inference retrieval-group exclusion、unknown-provenance fail-closed、无安全候选 zero-mask，以及按 exclusion 区分的 retrieval cache；默认要求每个训练 batch 的 non-transition atomic frames 中 safe draft coverage ≥99%。**经验证的 release 在训练期只读 `val`，绝不以 `test` 选模型；completion 的 validation diagnostic 因使用 target GT labels 构造 plan，只能记为 `ORACLE_COMPLETION_DIAGNOSTIC`。无 `build.json` 的上游包只保留显式标记的 test fallback code smoke，不能作为 headline。真正 closed-loop 要用冻结 planner 的 predicted plan。SELF_DRIVEN inference 的时长仅来自音频，评测 motion 目录只能提供 name roster，不能读取 target motion 内容；外部 AIST 名称若不精确匹配发布的 sample name，必须 zero-mask，后续只能通过显式、哈希固定的 source-to-group alias 映射启用 prototype retrieval，禁止运行时猜测 `_chNN`。
5. 获得官方 planner/completion checkpoint 后，记录 URL、SHA256、训练参数和 checkpoint stage；先复现一个官方 eval 片段，不马上微调。
6. 先使用 `--max-steps` 做两阶段训练 smoke；训练/推理 config 与数据版本写入 manifest。

**Gate P0**：planner 与 completion 都能读取官方数据、输出 shape 正确、没有 normalizer/坐标系漂移；train/eval source overlap 为 0、相邻 window labels 一致、completion retrieval 没有 self/near-duplicate draft。每份结果必须带 `generation_protocol` 与 `headline_eligible`；`ORACLE_GROUND_TRUTH_PLAN`/`ORACLE_COMPLETION_DIAGNOSTIC` 只能作为上限。没有 checkpoint 或 evaluation asset 时只报告“代码 smoke”，不报告论文复现；不使用 release 默认 README 评测数字作为 held-out 结果。

### P1：野外 3D 语料（当前已启动）

1. 从 Lodge cache 生成可恢复 manifest，先筛单人、完整身体、足够长、可见度高的 clip。
2. 用 world-grounded SMPL HMR（首选 WHAM）得到 `pose_world`、`trans_world` 和完整视频 DPVO trajectory。禁止 `--estimate_local_only` 进入训练语料；DPVO 与人体 world 默认视为未注册的两套 gauge。
3. 用 `tools/preprocess_wild_3d.py convert-wham` 转为 151D，并保留按 `frame_ids` 对齐的原始 DPVO camera / source 2D observation / fresh-global-run provenance。
4. 按 source 做 human review 和自动 QC：reprojection、身份稳定、镜头切换、骨长、root speed、foot contact、极端尺度、重投影误差。
5. 只把通过 QC 的数据标为 `accepted`；当前 converter 输出统一是 `candidate`，不能直接混入训练。

**Gate P1**：至少一组真实野外 clip 从 2D cache → WHAM → 151D → validate 端到端通过，且 camera asset 与 source/provenance 完整；人工查看原视频、2D overlay、3D/global render 后再扩大规模。

### P2：Atomic label 对齐，而不是硬塞未标注 wild 数据

官方 planner 和 completion 都需要 frame-aligned atomic labels；野外 3D 数据只有姿态时不能直接调用 `train_atomic.py`。

已完成的第一步是 AIST-only kinematic baseline：`discover_kinematic_atomics.py` 在完整
151-D sequence 上做 heading-canonicalized 运动学分段、train-only descriptor/K-Means
fit 和 acceptance threshold，产出 `kinematic_atomic_100_v2`。它保留真实转身、不读取或
拼接 camera，unknown frame 是 `-1` + false mask，随后才以 full-valid windows 物化。
这给 P2 提供可训练的防泄漏起点，但仍缺少论文的视觉/语义发现，因此下列顺序保持必要：

1. 将这个 AIST-only producer 与 checkpoint 固定为 recognizer / segment embedder 的训练输入；任何新 vocabulary、center 或 threshold 仍只在 **AIST source-disjoint train split** 拟合，先完整 source 后切窗。
2. 对通过 P1 的 wild sample 先做 event segmentation，再由冻结 recognizer 给 soft label、entropy、nearest prototype distance；低置信 segment 设为 invalid-mask / quarantine，不能伪装为 transition 或强行随机赋类。
3. 使用 source-held-out validation 检查 pseudo label：class coverage、duration distribution、retrieval distance、人工样本一致性，并与 AIST kinematic baseline 对照。
4. Completion 微调先使用 high-confidence pseudo labels，且 prototype library 只来自 train retrieval group。Planner 在 label quality 通过后再训练/微调；GT plan 只允许 oracle diagnostic。
5. 等 feature extractor、TMR / re-clustering 可重现后，再评估“新 wild atomic vocabulary”以及与官方 100 类的 cross-vocabulary alignment；这应是单独实验，不和第一版混在一起。

**Gate P2**：任何 pseudo-label 脚本都不得读测试 source、target future motion 或目标歌曲 GT；必须输出 label uncertainty，且类别/时长分布不能明显偏离 AIST baseline。

### P3：closed-loop 微调与消融

最小实验矩阵：

| 实验 | 训练数据 | plan 来源 | 目的 |
| --- | --- | --- | --- |
| A | 官方 AIST | GT plan | 验证 completion / normalizer 基线 |
| B | 官方 AIST | predicted plan | 建立真实 closed-loop 基线 |
| C | AIST + accepted wild | GT/pseudo plan | 判断数据域增益与噪声损失 |
| D | AIST + accepted wild | predicted plan | 唯一可作为最终系统候选的设置 |
| E | C/D 的 oracle plan | oracle only | 量化 planner 上限，不进入 headline |

先完成 B 再做 C/D。禁止依据单首歌或最好的 seed 选择结果；每组固定 source-level split、同一预算、多 seed。

### P4：评测与发布门槛

- 运动：FID/Div、root path、bone stability、jerk/flash、foot contact、完整歌曲 seam。
- 音乐：normal / zero / shuffled / wrong-song 对照；BAS 仅 sanity check。
- 结构：atomic duration/class coverage、重复/变化、论文 R-precision 的可重现实现。
- 泛化：未见 source、未见 dancer、未见 song 分开报告。
- 人评：盲测成对视频，原速且带音频，不使用 GT 选出的 sample。

只有 D 在上述门槛与完整的 SELF_DRIVEN 审计都成立时，才可以声明 wild fine-tune 改善了系统。

## 立即执行顺序

```text
1. 固定已发布的 AIST kinematic release，先在 source-held val 上做 predicted-plan
   planner / completion 实验；现有单步 smoke 不视为指标。
2. 运行 wild cache inventory、冻结 staging manifest 与 exact-byte audit（本机 b2gpu
   inventory / staging / audit 已完成），维持所有条目 `accepted_for_training=false`。
3. 在已授权、已验证的 WHAM + SMPL 环境中，对一个真实 clip 跑完整 world-HMR；当前机器
   不具备该依赖，不能用 2D/DPVO 伪造 151-D 真值。
4. 转成 151D、validate、审查相机/人体分离资产；随后完成 near-duplicate 与人工 3D QC。
5. 以冻结的 AIST producer 为 recognizer 起点，为通过 P1 的 wild sequence 生成带
   uncertainty 的 pseudo labels，再经全 sequence → window release gate。
6. 只在 high-confidence、source-safe 的 mixed release 上进入 completion fine-tune，最后
   用 predicted plan 做 held-out closed-loop 评测。
```

可运行的 P1 命令在 [WILD_3D_PREPROCESSING.md](WILD_3D_PREPROCESSING.md)。
