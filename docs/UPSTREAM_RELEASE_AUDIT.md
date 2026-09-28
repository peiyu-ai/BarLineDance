# 上游发布包审计（本地基线，不是训练结论）

审计对象：官方仓库 `oceanflowlab/AtomicDance` 的提交
`4fdaea56cf69b57f59cfc88c6691612ab120c5f8`，以及下载的
`data/atomic_aistpp.zip`。

本地 archive SHA-256：

```text
9cdd6fc0b824402bb4dcc109951c7a767dca50723bd775b3ca585c11c05852c8
```

`unzip -tq` 通过；本审计只读数组，不会改写官方数据。完整机器可读报告在
`data/wild3d/reports/atomic_aistpp_audit.json`（生成文件，未纳入 git）。

## 可用的表示基线

| split | windows | source videos | motion | music | labels |
| --- | ---: | ---: | --- | --- | --- |
| train | 17,733 | 952 | `[N,150,151]` float32 | `[N,150,35]` | `[N,150]` uint8 |
| test | 372 | 40 | `[N,150,151]` float32 | `[N,150,35]` | `[N,150]` uint8 |

labels 范围为 `0..100`（`0` 是 transition）；`normalizer.pt` 存在且为
151 维。因此这个包可以用来验证上游的 151D 表示、array layout 和
代码 shape，但不能直接被当成干净的 source-held-out 微调/评测集。

## 阻断性发现

### 1. README 默认评测 source 已进入 train

官方 README 的评测命令使用 `data/splits/crossmodal_test.txt`（147 个完整
source）。把包中名称的 `_sliceN` 去掉后，train 的 952 个 source 与该 list
重合 **100** 个，test 只重合 4 个。也就是说，按“发布包 train + README
crossmodal evaluation”跑出的结果包含训练 source，不能报告为 held-out 成绩。

### 2. 重叠 window 的 atomic label 并非同一完整 source 标签的切片

上游窗口长度为 150 帧、相邻 `_sliceN` 的 stride 为 15 帧，故应共享 135 帧。
审计比较了每个相邻 pair 的共享 labels：

| split | adjacent pairs | label agreement | 一边 transition、另一边 atomic |
| --- | ---: | ---: | ---: |
| train | 16,781 | 67.93% | 28.29% |
| test | 332 | 84.89% | 13.24% |

例如 `gBR_sBM_cAll_d04_mBR1_ch03_slice0/1` 的共享 motion/music 可逐元素
对齐，但 label agreement 只有 51.11%。这不符合“先在完整视频发现原子，后切
window”的标签契约，也会污染 planner 和 prototype retrieval 的监督。

### 3. completion prototype 可取回目标或近重复 GT

`train_atomic.py` 用全 train 建 motion library，`dataset/atomic.py` 的
`retrieve()` 仅按 duration 选 prototype；它不排除 query 自身、同一 source
或重叠时间窗。并且 completion 的当前评估路径使用 `train_loader`。在 0.5 秒
stride 的强重叠窗口中，这会把 completion 变成从近 GT draft 去噪，不能作为
闭环生成证明。

本 checkout 已对这两处做了最小的安全修正：prototype 保留显式
`retrieval_group_id` provenance，训练和 inference 都按完整 query performance group
排除 prototype（AIST 中同一 performance 的全部 `_chNN` 相机都被排除，比仅排除
相邻 window 更严格），并把 exclusion 放入 inference cache key；未知 provenance
在要求 exclusion 时 fail-closed。无安全候选时该段保持 zero-mask，而不会退回同源 GT。
训练默认要求 non-transition
atomic frames 中 safe draft coverage ≥99%。`test_loader` 上的 completion 检查仍用
target GT plan，因此结果明确标为 `ORACLE_COMPLETION_DIAGNOSTIC`，不属于 self-driven
指标。上述只修复 retrieval 泄漏；README advertised evaluation list 与 train 的 source
overlap、以及窗口标签漂移仍是训练前硬门槛。

## 新项目的最低数据门槛

1. 去 `_sliceN` 后的 recording（同时结合 song/dancer/account）是可回溯切分单位；
   `retrieval_group_id` 是 prototype exclusion 单位。AIST 必须再把同一 performance 的
   `_chNN` 合并成一个 group；两种 ID 在 train/val/test 间的交集都必须为 0。
2. discovery、cluster center、normalizer、pseudo label、prototype library
   均只能从 train source 建立。
3. 完整 source 完成 3D atomic segmentation / label 后才允许切 150 帧；共享帧
   label agreement 必须满足阈值（默认至少 99%，否则 quarantine）。
4. retrieval 必须排除同 retrieval/performance group、相交时间、query 自身；validation/test 的
   target 永不进入 library。
5. `--plan-source gt` 或任何目标 motion 参与的路径只能标记 `ORACLE`，不进入
   SELF_DRIVEN headline。

运行审计：

```bash
python tools/audit_atomic_dataset.py \
  --data-root data/atomic_aistpp \
  --eval-source-list data/splits/crossmodal_test.txt \
  --output data/wild3d/reports/atomic_aistpp_audit.json
```

当前 release 的非零退出码是预期结果：它在阻止我们把上游数据问题带进新架构，
而不是下载或解析失败。

报告还包含 `source_safe_retrieval_audit`：它检查在排除 query retrieval/performance group 后，
每个 atomic frame 是否仍有另一个 group 的 prototype。新数据版本必须满足该 coverage
门槛，不能为了填满 draft 而关闭 source exclusion。
