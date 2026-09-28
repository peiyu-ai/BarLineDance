# clean5:在高独舞账号语料上跑通整条 pipeline

**这份计划的目的是降低开发、验证和分析的复杂度,不是产出可发布的 benchmark。**
两者的判据不同,混在一起会让第一阶段永远不结束:「跑通」的退出条件是缺陷类别清零,
「达到预期」的退出条件是分数,而分数需要的语料规模和 split 口径这一阶段拿不到。
本文把两者显式分开(§6),并在每处写清哪些数字**不可引用**。

参照:上游论文 §3.1–3.3、Alg. 1、Tab. 1–3;`docs/WILD_ATOMIC_PIPELINE_PLAN.md`
是模块流的总表,本文只覆盖 clean5 这一轮。

---

> **本节的数字是 Phase 0 之前的,已被 2026-08-20 的重算取代。现行语料:**
>
> | | Phase 0 前 | **现在** |
> | --- | ---: | ---: |
> | clips | 2,103 | **1,999** |
> | 段(beat4h) | 15,115 | **14,231** |
> | 素材时长 | 10.10 h | **9.52 h** |
> | 段内时长 | 8.42 h | **7.90 h**(83.0%) |
> | K=100 时段/类 | 151 | **142.3**(论文 268.57) |
>
> 账号构成:Annala呀 455 / 每周都有周二 552 / 汤汤汤小圆 238 / SHERRY 369 / Peng 385。
> **Annala 掉得最多(551 → 455)**:419 条陈旧里 363 条是它的,102 条 VO 失败里 96 条是它的。
> 权威产物是 `runs/clean5/corpus.json` 与 `runs/clean5/clips.txt`,不是本节的表。
> **M2 的 K 要按 14,231 重取:14,231 ÷ 268.57 = 53.0 → K=53**(原计划按 15,115 算的是 56)。

## 1. 语料定义:阈值,不是名单

**选中规则是账号的实测独舞比例 > 0.55,账号集合由它推出。**
手敲的账号名单没有出处:普查重跑之后它无法被重新推导,也没有任何东西发现它过期了。
入口是 `tools/select_clean_accounts_corpus.py --min-solo-share 0.55`,产物:

```
runs/clean5/clips.txt    2,103 个 clip stem,下游按它消费
runs/clean5/corpus.json  账号构成、排除项、来源、分割与特征的一致性检查
```

必须显式传 `--segmentation`,指向一份**建在当前特征上**的分割(§2.3):

```
python3 tools/select_clean_accounts_corpus.py --min-solo-share 0.55 \
    --segmentation runs/wild_v4_seg_r0visual/segmentation.json \
    --output-dir runs/clean5
```

"独舞"的定义来自 `tools/audit_clip_population.py`:逐帧读 `detections.npz` 的
`person_count`,**至少 90% 的帧里画面上不超过 1 个人**才算 solo。0.9 而不是 1.0,
因为一个路人穿过一帧不该让一条独舞变成群舞;不更低,因为 0.75 时一条 clip
可以有四分之一时长是双人还被叫做独舞。

| 账号 | uploads | clips | 独舞率 | 段(visual) | 段(fused050) | 时长 | 帧占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Annala呀 | 506 | 551 | 0.904 | 9,125 | 8,701 | 2.64 h | 26.1% |
| 每周都有周二 | 432 | 555 | 0.710 | 9,068 | 8,747 | 2.59 h | 25.7% |
| O-DOG编舞师-SHERRY🌙 | 183 | 370 | 0.630 | 6,658 | 6,494 | 1.91 h | 18.9% |
| O-DOG编舞师-Peng | 275 | 385 | 0.584 | 6,609 | 6,431 | 1.83 h | 18.1% |
| 汤汤汤小圆 | 171 | 242 | 0.703 | 4,105 | 4,003 | 1.14 h | 11.2% |
| **合计** | **1,567** | **2,103** | **0.754** | **35,565** | **34,376** | **10.10 h** | |

> **更正(2026-08-20)**:本表第一版报 34,855 段 / 9.91 h,那是从**已发布的分割**里读的,
> 而它对「每周都有周二」的 498 条 clip 索引着已经不存在的特征(§2.3)。真实数是
> **10.10 h**,少报了 20,141 帧,全部来自那一个账号。工具现在有一道能失败的闸挡这件事,
> 负对照(指向已发布分割 → 退出 1 并点名 498 条)和阳性对照(指向修正臂 → 0 条不一致、
> 退出 0)都验过。

三个尺度参照:

* **10.10 h > AIST++ 的 5.2 h。** 这一轮不是"小语料",规模上高于论文自己的语料。
* **加权独舞率 0.754,全语料是 0.302。** `tools/report_population_quality.py` 在
  控制主体尺寸之后量到:多人相对独舞,滑步 1.07×–1.47×(随主体变大而增长)、
  根速度 1.04×–1.15×,而**抖动完全没有效应**(0.99×–1.03×,不单调)。
  所以这个筛选买到的是滑步和根速度,不是抖动 —— 野外比 AIST++ 动捕高约 9% 的抖动
  换语料换不掉。
* **K=100 时 355.6 段/prototype(visual)/ 343.8(fused050),论文是 268.57。**
  要精确对上论文密度需要 K≈132 / K≈128。**但 K 不要现在冻死** —— 段数是它唯一的输入,
  而其中一部分还要随 §5 Phase 0 的重算而变。
* **M6c MultiModality 在这里可定义。** 按 `eval_multimodality.py:177-187` 的真实构造
  (连通分量**自身必须是完全图**、且跨 ≥2 upload;不是在分量里找团)实测:
  **181 组 / 418 clip**,对照已发布 test split 的 **144 组 / 320 clip**(该对照用同一函数
  复现,逐个吻合)。绝对数 +25.7% 组 / +30.6% clip;**但每 clip 密度略低于 test split**,
  所以能说"组更多",不能说"音乐结构更密"。

**阈值 0.55 不是一条干净的分界线,这一点要写明。** 卡在里面的最后一个账号(Peng 0.5840,
Wilson 95% CI [0.534, 0.632])与卡在外面的第一个(兔子_、🍉 0.5391,[0.503, 0.575])
**统计上无法区分**。可辩护的说法是"实测差 0.045",不是"它们分得开"。
如果第 6 个账号的加入会改变某个结论,那个结论就不能只靠这个阈值支撑。

**K 不用压。** 单账号方案要把 K 从 100 压到 15 才对得上论文密度,那意味着 M2 没法和论文
Tab. 2 直接比;这一轮 K≈130 就落在论文的操作点上。

**排除项(报告而不是静默应用)**:2 条 fps 重切孤儿。孤儿是重切后不再被产出的名字,
而当前凭据能覆盖不能删,所以它的对象永远留在 OSS、任何目录列举都会把它带回来。
其中 10 条(全语料)就在已发布 bundle 里 —— **"读发布集合"不构成保护**,必须按名字排除。

---

## 2. 准备:三层各欠什么(全部实测)

### 2.1 ingest:不欠

`tools/ingest_wild_uploads.py` 的 `manifest_rows()` 是"这次运行产出了什么"的权威陈述
(按 `ingested_at` 取每个上传最新的一行;逐行累加会把重跑过的上传算两遍)。对这 5 个账号:

| | |
| --- | --- |
| uploads(按 group_keys 归属) | 1,859 |
| **未 ingest** | **0** |
| ingest 后产出 0 条 clip | 73 |
| 切失败 | **0** |
| 产出 clip | 2,618 |
| 其中已进发布 bundle | 2,103 |
| **产出但从未发布** | **515** |

那 73 个产 0 clip 的上传**不是缺口**:内容切分的判据(无舞者的段落、镜头切换后不足
`FRAMES_PER_CLUSTER × MIN_CLUSTERS = 340` 帧)在正常工作。end_reason 分布
`length_split` / `dancer_absent` / `shot_cut` / `upload_end` 也是这条判据的痕迹。

**515 条"产出但未发布"是一个待定项**,不是这一轮的阻塞。它约等于 +20% 语料,
但它没进发布 bundle 一定有原因(音频筛选、3D QC、舞者轨迹切换),
**在弄清那个原因之前不要把它们加回来** —— 那正是 §2.3 那类缺陷的来路。

### 2.2 3D / S3D:欠 419 条,0 条缺失

闸是 `tools/audit_clip_freshness.py`,它问的是**"这份产物是不是用 clip 当前的字节建的"**,
不是"文件在不在"。后者对这个语料永远回答"在":已发布的语料曾经带着一代特征,
对象存在、长度正确、和下游全都接得上,而它们来自更老的一版切分。

```
python3 tools/audit_clip_freshness.py --clips runs/clean5/clips.txt \
    --output runs/clean5/freshness.json --workers 16
→ stage 3d   stale=419  missing=0
  stage s3d  stale=419  missing=0
```

这 419 条来自 2026-08-19 的 fps 重切(非整数帧率上传的音画偏移与速度错误)。
**重切后的 clip.mp4 还没有 push 到 OSS**:本地 cache 里有新字节,OSS 里是已发布的旧字节。
抽样验证过这条不对称 —— 陈旧的 8/8 条本地有 `clip.mp4`,新鲜的 8/8 条本地没有。

### 2.3 一条现有闸看不见的陈旧:M1 的切分 vs 它索引的 S3D

**这是本轮新发现的,而且它比 fps 那条更隐蔽。**

`segment_visual_atomics.py:321` 把 `motion_frames` 记为**运行当时的 S3D 特征长度**。
拿它和磁盘上现在的特征长度对:

| | |
| --- | --- |
| 全语料 `runs/wild_v4_seg/segmentation.json` 与当前 S3D 长度一致 | **9,854 / 13,783** |
| 不一致 | **3,929(28.5%)** |
| clean5 内不一致 | **498 / 2,103,全部是「每周都有周二」** |
| 其中同时被 `audit_clip_freshness` 判为陈旧的 | **0** |

**最后一行是这条的要害**:这 498 条在 freshness 闸下全部读作 `fresh`,因为那把闸问的是
"S3D 是不是用 clip 当前字节建的"(是),它不问"分割是不是用当前 S3D 建的"。
两把闸之间有一条没人守的缝。

**证据是内容,不是时间戳。** 3,929 这个数与 worklog 记的 R0 帧数判据
(3,929 = 28.5%;按内容哈希判 3,991;差的 62 条是"旧切分刚好等长")**逐个吻合**,
所以已发布的 M1 就是 R0 之前的那一份。时间戳只是旁证(分割 08-19 07:19;
不一致那批的 `.npz` 抽 400 条全部落在 08-19 08 时,一致那批 397/400 落在 07 时),
**不要把 mtime 当判据** —— 缓存回拉会重置它,一次核查里就有人据此得出相反读数。

**修法不是重算,是换产物。** R0 之后的全量重切分已经存在,只是从没被合并、也从没发布:

```
/cache/atomicdance-assets/scratch/c1/full/arms/{visual,fused050,motiononly}_{0..63}.json
参数与已发布一致:fpc=34 / L_min=18 / index_weight=4.0 / seed=20260808
特征目录:scratch/c1/full/feat/  (13,783 个符号链接,指向修正后的 wild_visual_s3d)
```

三条臂**与当前 S3D 帧数 13,783/13,783 全部一致**,已发布那份是 9,854/13,783。

### 2.4 完整的陈旧地图:三层两两对照

对 clean5 的 2,103 条做三方比较(已发布分割 / 修正臂 / 当前 S3D / 当前 ingest 长度):

| | 已发布分割 == S3D | 修正臂 == S3D | S3D == 当前 ingest 长度 |
| --- | ---: | ---: | ---: |
| **fresh(1,684 条)** | 1,186 一致 / **498 不一致** | **1,684 / 0** | **1,684 / 0** |
| **stale(419 条)** | 419 / 0 | 419 / 0 | 162 一致 / **257 不一致** |

读法:

* **fresh 那 498 条**(全部是「每周都有周二」)只在已发布分割上错;换成修正臂就干净了。
* **stale 那 419 条**的分割和 S3D **彼此一致**,因为两者都是从**旧字节**建的 ——
  两个读数都干净,而两个都是要被作废的东西。其中 **257 条重切后长度也变了**,
  另外 162 条长度没变但内容变了(就是 R0 记过的"旧切分刚好等长"那一类)。
* 合计 **917 / 2,103(43.6%)** 在这两条轴之一上是陈旧的;端到端干净的只有 1,186 条。

**这直接决定 Phase 0 的顺序**:stage B/F 把那 419 条按新字节重算之后,
它们的 S3D 会变,**修正臂对它们也随即过期** —— 所以 **M1 必须排在 B/F 之后**,
而不是像本文第一版写的那样"直接用已合并的修正臂、不重算"。
M1 全量重跑很便宜(上次三条臂 × 64 shard 约 3 分钟)。

---

## 3. M1 已换成音乐拍网格 —— 我们的 M1 不再是论文的 M1

**决定(2026-08-20,操作者):每 4 拍一刀。** 产物 `runs/wild_v4_seg_beat4h/`,
工具 `tools/segment_on_music_beats.py`。完整来龙去脉见 worklog 2026-08-20 那节,
这里只留下判决所依据的东西。

### 3.1 为什么不是"选一条 Alg.1 的臂"

本文上一版在三条 Alg.1 臂(visual / fused050 / motiononly)之间挑。**那个选择本身是错的**,
因为选它们用的两把尺子是同一把:

* `boundary contrast` 的最优解是切在**速度峰值**上,而论文要切在**落定**处(§2.1 已记);
* `adjacent_over_random` **同样单调于切点处的速度**(六条臂上 0.734→0.861),
  把论文规定的落定切法排在随机对照**之下**,在边界已知的夹具上给**正确分割 0.000**、
  给故意切在动作中间的 **+0.03**。

**所以"两把尺子都指向 fused w=0.5"不是旁证,是同一个读数数了两遍。** 那条臂在唯一
有论文出处的判据上排最后。**本文此前推荐 fused w=0.5 的段落作废。**

### 3.2 论文这边没有可继承的判据

Fig.4a 是唯一的分割数字,而它**对切点位置完全盲**:段长在录像内重排,
**97.19% 的切点换位置、直方图变化为 0**。`fpc=34 / L_min=18` 正是拟合它得到的,
而 Alg.1 把 `N` 和 `L_min` 列为 Input、**论文没给过值**。
> **`L_min = 0.6 s` 没有论文依据。**

方法出处 [16](PSVL)**有**真值(人工标注 moment + 随机基线),上游论文继承了机制、
丢掉了评估;[16] 还多一步"合并相邻 atomic event",但它产出的是**重叠候选池不是划分**,
不能直接搬。更根上:[16] 的前提"特征在事件边界处剧变"**在舞蹈上是反的**。

### 3.3 D 是什么,以及它的不变量

音乐拍网格来自每条 clip 的 35-D 音乐特征**通道 34**(`librosa.beat.beat_track`),逐帧对齐。
它规整:间隔变异系数中位 0.037,**500 条全部 < 0.10**,中位 32 拍/clip。

| | |
| --- | ---: |
| clips | **2,101 / 2,103**(2 条按名字排除,账目闭合) |
| 段数 | **15,115**,7.2 段/clip |
| 段长中位 | **1.933 s**(1.200–3.833 s) |
| **结构违规** | **0 / 15,115** —— `validate_grid_records()`,并由不 import 该工具的独立检查器复核 |
| 语料 | 10.10 h → **8.42 h** |

**闸的关键负对照**:删掉一个网格点 → `span_not_k_beats == 1` 而 `boundary_off_grid` 仍是 0。
**一个只问"每刀是否在拍上"的闸会放过这个缺陷。**

### 3.4 代价,以及没救的那部分

丢掉 1.69 h = **前奏 0.38 + 尾巴 0.56 = 0.94 h(整拍规则留不住)**
+ **头尾的完整拍 0.73 h(可救,只能做成 1–3 拍的段)**。
**操作者定:先不救**,以保住"每段恰好 4 拍"。

### 3.5 必须一起入档的三件

1. **D 完全不看视频,所以 M1 不再是论文的 M1。** 与 S3D≠I3D 同级:
   任何"复现论文"的说法不能再套在 M1 上,Tab.2 的行不能再当 M1 的参照。
2. **下拍没有检测。** `--phase energy` 是"下拍更响"的猜测,4 拍档中位领先 16.2%,
   **6.0% 的 clip 领先不到 2%**。而且 `beats[0]` 按构造就响(librosa 锚在强 onset 上),
   去掉这一帧,offset 0 的胜出率从 41.8% 掉到 30.6% —— **相位规则部分在量节拍器。**
3. **舞者不在音乐的相位上。** 落定点落在拍上 19.4%(随机 ~20%,p=0.42),
   但锁在拍的**周期**上(R 0.1293 vs 错周期 0.1110,p=0.0087)。
   我据此提议按舞者相位平移网格,**留出验证失败(平移显著更差 p=0.047),已收回。**

### 3.6 一个会波及旧读数的工具 bug

`probe_segmentation_boundaries.py:295` 与 `probe_settle_alignment.py:293` 的
`_random` / `_peaks` 控制组**只按传入的第一条臂构造**。交换参数顺序,
D 的 `within_vs_random` 从 102.9% 变 145.9%,r0visual 从 98.2% 变 69.3%。
> **此前任何多臂调用里,非第一条臂的这几个数是拿别人的段长当分母算的。**
> 现在所有读数**每臂单跑**。

---

## 4. 重算 419 条:哪些 stage 能按清单驱动,哪些不能

**清单是 `runs/clean5/freshness.json` 本身**,不要手抄 stem。`tools/redo_manifest.py`
是三个读取方共用的那一个 reader(接受纯 stem 列表、freshness 的 `stale.<stage>`、
census 的 `clips`;文件不描述所问的 stage 时抛异常而不是返回空)。
实测 `load(freshness,'3d')` 与 `load(...,'s3d')` 各 419 条,**两个集合完全相同**。

| stage | 能否按清单驱动 | 实测状态 |
| --- | --- | --- |
| **B** 3D(GVHMR) | ✅ `--redo` / `REDO=` | 冻结模拟得到**恰好 419 条,0 多余、0 孤儿** |
| **F** S3D | ✅ `--redo` / `REDO=` | `REDO=... bash tools/run_stage_f_oss_fleet.sh` 写法就是对的 |
| **C** inventory/staging | ❌ | **失败模式与 worklog 的预测相反**,见下 |
| **D** audio35 | ❌ | 无闸、全量重算:正确但是 33× 的浪费(419/13,783 = 3.0%) |
| **E** bundle/normalizer | ❌ **且不该加** | 组归属检查与 normalizer 拟合是全语料统计量,子集版本是另一个量 |
| **G/M1** 分割 | ❌ | 无闸无清单;shard part 在 OSS 里会累积(删不掉) |

### 4.1 stage B 的写法和 worklog 记的不一样

`--redo` 只在 `--freeze` 时生效(故意的,免得某个 shard 悄悄忽略它),而
`tools/run_stage_b_oss_fleet.sh` **从不显式传 REDO** —— 它靠环境继承到达 freeze,
而同一个继承也到达它随后拉起的 14 个 shard,那些 shard **没有 `--freeze`**,于是
撞上 `run_gvhmr_ingest_shard_oss.py:300-306` 的守卫并**退出 1**。

所以 stage B 要分成两条命令,第二条**必须把 REDO 和 FREEZE 都清掉**:

```
REDO=runs/clean5/freshness.json python3 tools/run_gvhmr_ingest_shard_oss.py --freeze --num-shards 14
bash tools/run_stage_b_oss_fleet.sh          # REDO / FREEZE 都不设
```

不需要 `RETRY_FAILED`:`forced` 是从 skip 集合里**减掉**的,所以一个失败标记拦不住被点名的 clip。

**一个要盯住的坑**:419 条里 **111 条同时带着 `.extract_failed` 标记和 `quality.json`**。
而重抽失败时 `extract_one` 在两次 `publish_dir` 之前就返回,**什么都不发布** ——
于是 2026-08-19 那份陈旧 3D 原地留下,对下游每一道存在性闸都和成功无法区分。
这 111 条要单独核对退出状态。

### 4.2 stage C 的问题比"跳过"严重:它列 OSS 前缀

`run_wild_stage_c_oss.py:186` 是 `contents = clip_contents(INGEST_ROOT)` —— 它从
**OSS 前缀列举**推出自己的 clip 集合,而不是从清单。这正是 CLAUDE.md §1.1 明令禁止的那条。

实测那次列举回 **17,790 条 clip**,其中:

* **775 条是 fps 重切孤儿** —— 永远删不掉,一列就回来;
* **1,075 条是 2026-08-12 冻结的 worklist 里从来没有的名字**(worklist 16,715 条,是前缀的真子集)。

它**没有跳过闸**,所以重跑确实会刷新那 419 条 —— 代价是把 1,850 个不该在的名字一起带进来。
**最小修法**:给 `inventory` 子命令加 `--clips <manifest>`,把第 189 行换成清单的 stem,
并对清单里前缀提供不了的名字**报错而不是静默丢弃**(照 stage F 的形状写)。
**在这条修好之前不要跑 stage C。**

### 4.3 一条只能"做"不能"检查"的边界

`audit_clip_freshness.py` 只覆盖 7 种派生产物里的 **2 种**(`redo_manifest.STAGES = ('3d','s3d')`),
因为只有 GVHMR 和 S3D 抽取器记了 `video_sha256_1mb`。**B 和 F 重算之后,
inventory / staging / reconcile / audio35 / bundle / normalized / segmentation
没有任何一把闸能失败** —— C/D/E/M1 的重跑只能**做完**,不能事后验。
这一条要写进执行记录,否则"跑过了"和"跑对了"分不开。

---

## 4b. split:标准形态在这个规模上建不出来,而且我先前算错了谁进 test

**更正。** 我先前手算的结论是"放宽 share 之后 Annala 进 test、每周都有周二 进 val、
train ≈ 3.0 h"。拿真工具跑(导入 `assign_accounts` 并对 161,376 组参数穷举)之后:

* **卡住的不是 share,是 `--max-eval-account-fraction 0.15`。** 5 个账号里 4 个超过 15%
  帧占比(Annala 0.2611、每周 0.2566、SHERRY 0.1887、Peng 0.1813),只剩汤汤汤小圆
  0.1124 可进 eval,而 val+test 要 20%。真实报错是
  `accounts eligible for eval hold 11.2% of the corpus but val+test want 20.0%`。
  **先调 share 永远没用,上限必须先升到 ≥ 0.19。**
* **`O-DOG编舞师-Peng` 的哈希是第 0 位**(`25c03c1f` < Annala `36838a91` < 每周 `3f019550`
  < SHERRY `45dd0ed7` < 汤汤汤小圆 `4f759e12`),所以它在 **127,952 组成功配置里 100% 落在 test**。
  我先前那句"Annala 进 test"来自一个**不含 Peng 的**四账号模拟,是错的。
* **保持默认 `--min-eval-accounts 2`,train 最多 2.64 h**(只剩 Annala,且它因 26.1% 超限被强制进 train)。
  eval 占 73.9%。**那不是一个能训练的切分。**
* **train 的上界是 6.36 h**,要把 `--min-eval-accounts` 降到 1 —— 也就是允许一个 eval split
  就是一个编舞师,正是第 4 条不变量存在的目的。最优配置:
  `--min-eval-accounts 1 --share-tolerance 0.01 --test-share 0.18 --val-share 0.18
  --max-eval-account-fraction 0.19` → train {Annala, 每周都有周二, 汤汤汤小圆} 1,348 clip / 6.36 h,
  val {SHERRY} 1.91 h,test {Peng} 1.83 h。

### 4b.1 顺带查出工具本身的两个洞(与本轮无关,但要修)

* **五条不变量没有一条检查 train 非空。** 实测
  `--test-share 0.44 --val-share 0.45 --max-eval-account-fraction 0.27 --share-tolerance 0.11`
  **退出 0**,发布了一份 `train 0 条 / val 1,167 / test 936` 的 bundle,
  还盖着 `split_status: account_disjoint_source_safe`。
  127,952 组成功配置里 **2,812 组 train 为空**。修法:在
  `assign_account_disjoint_split.py:532` 的不变量块里加 train 非空 / 最小 train 份额。
* **两条不变量在这个语料上永远不会触发**:`duplicate_content_group_id` 在 13,783 条已发布行上
  **全部为 null**;`retrieval_group_id` 按构造不可能跨 split(split 是按账号赋的,每个 upload
  只映射到一个账号)。**不要把"没有 duplicate_content_group_id 跨 split"当证据引用** ——
  在那个字段被填上之前它是空的。这正是 CLAUDE.md §2 说的"永远不会触发的闸读起来像检查过了"。

### 4b.2 结论与建议

两条路,选一条并**写进 bundle 的 `split_status`**:

**(A) upload 级切分,`split_status` 明写不是 source-safe。** train ≈ 8 h。
挡不住的是同一编舞师跨 upload 重拍同一套 routine —— account-disjoint 存在的全部理由。
这条泄漏的**下界**可测(`runs/wild_v4_music_groups_pairs.jsonl` 里落在这 5 个账号内部的已验证同曲对),
而指纹在已知真对上的召回只有 0.564,所以真实重叠比它多。**评测数字只做趋势,不可引用。**

**(B) 上面那条 6.36 h 的 account-disjoint 配置。** 代价写明:test 是**一个编舞师**(Peng,1.83 h),
所以每一个 headline 数字描述的都是"泛化到一个上传者"。

建议 **(A) 做主线,(B) 跑一次当冒烟** —— 同一份语料只换切分参数,
换来的是 account-disjoint 那条代码路径在这一阶段就被跑过、调过,而不是 Phase 2 第一次遇到。

## 5. 阶段与退出条件

### Phase 0 —— 已完成(2026-08-20)。执行记录与两处顺序更正

**结果**:语料 **2,103 → 1,999 条**,验收闸 `stale=0 missing=0`(`runs/clean5/freshness_final.json`),
分割 `runs/clean5b5_seg/segmentation.json` = **14,231 段 / 中位 1.933 s / 结构违规 0**,
选择器一致性闸**不带 `--allow-stale-segmentation` 自己过**(`0 clip(s) disagree`;
同一把闸对旧分割读 201)。完整来龙去脉见 worklog 2026-08-20 续那节。

**顺序更正一:D/E 必须排在 M1 之前。** 下面第 5 步"M1 全量重跑"排在 C/D/E 之前,
那是 Alg.1 的 M1(读 S3D)。**新 M1 读的是 bundle 里的 `music_35`**
(`segment_on_music_beats.py:500-505`),由 stage D 从 `audio.wav` 算
(`run_wild_stage_d_oss.py:70-71`)—— 正是 fps 缺陷取错位置的那个文件。

**顺序更正二:多一步 push,而且它排在最前。** 本文原来没有这一步。
`asset_io.read_bytes` 本地优先,所以不 push 也能算出正确的 3D —— 但 OSS 上仍是旧字节,
**"读到哪一版"取决于跑它的机器有没有本地缓存**。`tools/push_recut_clips.py`,419 条 / 7.86 GB。

**实际执行的顺序**:push → B → F → 验收闸 → 配额闸 → C(修好之后)→ D → E → **M1** → 重选语料。

**没能补回的 102 条**:stage B 上 `visual odometry diverged`(有归因的失败,24.3%,
对照全语料基线 22.0%)。操作者定:从 clean5 排除,计入 `stale_derivatives` 而非 `orphan`。
**没有给它们重算 S3D** —— 只刷新 S3D 会造出"两个派生物来自视频的不同剪辑"的半干净态。

**分片数是语料的隐式常量**:inventory 8 / reconcile 8 / audio35 16 **必须沿用**。
旧 part 名字在 OSS 里删不掉,数目一变 `merge` 会看到两代 part 并因重复 id 拒绝整批。

**下面是原始计划,保留以便对照。**

### Phase 0(原文)—— 补齐,顺序不能换

0. **给 stage C 的 `inventory` 加 `--clips <manifest>`**(§4.2)。在这条修好之前
   **不要跑 stage C**:它会从 OSS 前缀列举里带回 775 个永远删不掉的孤儿
   和 1,075 个 worklist 从来没有的名字。
1. **stage B**,两条命令(§4.1),然后单独核对那 111 条带 `.extract_failed` 的退出状态。
2. **stage F**:`REDO=runs/clean5/freshness.json bash tools/run_stage_f_oss_fleet.sh`。
3. **重跑 `audit_clip_freshness`** 作为 B/F 的验收闸,期望 `stale=0 missing=0`。
   这是链条上**最后一处能自动失败的地方**(§4.3)。
4. **E 之前**先 `python3 tools/check_disk_headroom.py --probe-gb 25` ——
   它是这一串里唯一有真实本地占用的(峰值约 20 GB 在 /dev/shm)。
5. **M1 全量重跑**(§2.4:B/F 之后那 419 条的 S3D 变了,修正臂对它们已过期)。
   `NUM_SHARDS >= 8` —— 上次是 8,而 OSS 删不掉旧 part,shard 数变少会让 merge
   看到重复序列。跑完用 `select_clean_accounts_corpus.py` 的一致性闸自检
   (期望 `segmentation vs features: 0 clip(s) disagree`)。
6. **stage C(修好之后)→ D → E**,全量重跑。
7. **重跑 `select_clean_accounts_corpus.py`**,让 §1 的数字与新产物一致。

### Phase 1 —— 跑通。退出条件是缺陷类别清零,不是分数

1. **M1 —— 已完成,但换了方法(§3)。** 不再是 Alg.1,是每 4 拍的音乐网格
   (`runs/wild_v4_seg_beat4h/`,2,101 clip / 15,115 段 / 中位 1.933 s / 结构违规 0)。
   **`total_variation_from_fig4a` 不再是这一阶段的判据** —— D 的 TV 是 0.752,
   而那个统计量对切点位置是盲的(§3.2)。M1 的判据现在只有两条:
   `validate_grid_records()` 的结构不变量(能失败,实跑 0 违规),以及**操作者看分镜图**。
2. **M2**:K 按 D 的段数重取 —— 15,115 ÷ 268.57 = **56.3 → K=56**
   (沿用 K=100 会得 151 段/类)。跑法:
   `run_wild_stage_g_m2_oss.py --tag clean5b4 --bundle-tag wild_v4 run --classes 56`。
   报接受率与类离散度;
   **必须带一个已知答案为好的阳性样例**,不能只有"坏输入上会失败"。
3. **M3**:组内相干性闸(中立 TMR 空间);抽检 caption 描述的是不是这个舞者实际做的动作。
4. **M4b**:段数下降幅度、同歌 < 异歌的方向。
5. **M5**:transition loss 收敛;检索排除同 `retrieval_group_id` 生效。
6. **M6**:**能出数即可,数值标注"不可引用"**;MultiModality 一栏按 §4 的口径判断是否可定义。

**外加一条**:Phase 1 的每个判据,在写下结论前先拿全语料跑一次同样的判据做对照。
不是为了看分数,是为了看**这把尺子在两个规模上是不是同向** —— §2.1 的"两条判据不同向时
先怀疑尺子",在这里的形态是"同一条判据在两个语料上不同向时先怀疑尺子"。
M1/M3a 的全量产物都已存在,这基本不花钱。

### Phase 2 —— 才谈质量

扩到能建标准 account-disjoint split 的规模。两个候选口径规模几乎相同,
到时候按 Phase 1 的结果再定,本阶段不预设。

---

## 6. 已知缺口(不因这一轮而改变,也不被这一轮解决)

* **M3 的 genre 预切分没有来源。** `runs/genre_gate_video.json`:VLM 舞种标注在 AIST++
  GT 上 80 条判对 9 条(accuracy 0.1125,随机 0.10),**没过闸**。论文 M3 先按舞种切每个
  prototype,野外这一步目前空着,与语料选择无关。
* **stage C 从 OSS 前缀列举推导 clip 集合**(§4.2)。这比 worklog 预测的"存在即跳过"更糟:
  它根本没有跳过闸(重跑确实会刷新那 419 条),但会把 775 个孤儿和 1,075 个
  worklist 之外的名字一起带进来。**已移入 Phase 0 第 0 步。**
* **stage C 的 staging 切分在语料增长下不幂等**:`preprocess_wild_3d.py:892-905` 的
  `_stable_source_splits` 按 sha256 排序后**按精确条数切**,录像总数一变,
  边界就移动,靠近边界的录像会翻面。
* **`assign_account_disjoint_split.py` 的不变量不检查 train 非空**,而且其中两条
  在这个语料上永远不会触发(§4b.1)。
* **`probe_segmentation_boundaries.py` / `probe_settle_alignment.py` 的控制组只按
  第一条臂构造**(§3.6)。已知影响:任何多臂调用里非第一条臂的 `within_vs_random` /
  `_random` / `_peaks` 无效。**工具本身还没修**,现在靠"每臂单跑"绕开。
* **仍然没有任何人工标注的边界。** 本轮所有分割判据都是自造的,D 是靠眼睛选的。
  §3 的每一条读数都不能替代真值;补一份 GT 是让它们能被判决的唯一办法。
* **下拍没有检测**,`--phase energy` 是猜测,6.0% 的 clip 上等于抛硬币(§3.5)。
* **重切后的 clip.mp4 还没 push。** 一旦 push,已发布的那份就被覆盖,
  `output/recut_contact_sheet.png` 那张新旧对照图就再也做不出来了。
* **134 个上传同时挂在两个账号目录下**(全语料,涉及的 19 个账号全是 O-DOG):
  **134 个字节数全部相同,ETag 相同的是 100 个**;其余 34 个至少有一份是分片上传
  (ETag 带 `-N` 后缀),**那 34 个的 ETag 无从比较**,不是内容不同。
  (本文第一版写"能读到 ETag 的 105 个字节相同",那是只解析了一部分行的结果。)
  它让"账号 = 独立编舞师"这条前提对那批上传不成立;**触及本语料 4 条 clip**。
* **515 条"产出但从未发布"的 clip**(§2.1b)是待定项,且**捡它之前必须先把 fps 重切
  推到那 315 条所在的上传**,否则按实测 27.5% 的名字废弃率约 87 条产物会变成孤儿。
* **`data/wild3d` 下有两棵 converted 树**:`ingest_v1_converted`(15,294)和
  `converted`(5,907,是 `run_gvhmr_wild_shard.sh:28` 的 `CONVERTED_ROOT`,
  也在 `push_assets_to_oss.sh` 的推送清单里)。**只数一棵会漏** ——
  在 §2.1b 那 515 条上,只数 `ingest_v1_converted` 得 151,两棵并起来是 214。
  任何"这条 clip 有没有 3D"的判断都要同时看两棵。
* **`audit_wild_3d_quality.py` 不是准入闸。** `runs/wild_v4_3d_quality.json` 全仓库只被
  `report_population_quality.py` 读,是报告不是判据。不要把它的结论当成"这条 clip 被 QC 拒了"。
