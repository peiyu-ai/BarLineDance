# 对齐论文模块流的开发实验计划(野外 3D 语料)

**本文第一版是按"仓库里现有工具恰好能做什么"组织的,那是错的。** 这一版按论文
§3 的模块流重排,每个模块先写清**论文规定了什么**,再写**仓库现状**,再写**要做
什么**与**判据**。差距一律显式列出,不靠"baseline 也算跑通"这种说法糊过去。

参照:上游论文 PDF §3.1–3.3、Alg. 1、Tab. 1–3(此前叫 `paper.pdf`,同一份文件,
sha256 `81aa7151…`)。

---

## 0. 论文的模块流(照抄,不改写)

```
数据: AIST++ 1408 seq / 5.2h / 10 genres / 30 subjects / 配对视频 + 3D 动作
  │
  ├─ M1 Segmentation (Alg.1)
  │     I3D 逐帧视觉特征 → T×T 余弦自相似矩阵 A
  │     c_t = A[t,:] ,附加 t/T → N-means → 标签跳变处切分
  │     迭代合并 < L_min 的段(并入较短邻居)
  │
  ├─ M2 Clustering
  │     每段用 TMR motion encoder 编码(motion-text 联合空间)
  │     全数据集 K-Means → 100 个 prototype(平均 268.57 段/类)
  │     只保留靠近簇心的段,丢弃模糊边缘点
  │
  ├─ M3 In-group Re-clustering
  │     Gemini-2.5-Pro 描述每个视频段: signature pose + movement dynamics
  │     PoseScript 标注 motion beats(段内关节速度局部极小)关键帧,作 VLM 辅助线索
  │     每个 prototype 先按 genre 预切分
  │     summarizing LLM 迭代: 取互相相似的 caption 作 sub-prototype → 蒸馏成语义标签
  │     直到未分组段数低于阈值。平均 7.3 sub-prototype/类,31.8 样本/个
  │     ⇒ 最终词表 ≈ 100 × 7.3 ≈ 730 类
  │
  ├─ M4 Atomic Movement Planning
  │     D3PM over 帧标签 y_i ∈ {0..K},0 = transition
  │     预训练 music encoder,条件是**整首音乐**而非片段
  │     Transformer 反向过程
  │     M4b 后处理: 滑窗多数投票 + 最短时长合并(短段并入语义最兼容的邻居)
  │
  ├─ M5 Dance Completion
  │     按 label 从 pre-clustered 数据库**检索**时长最接近的段 → 缩放到目标时长 → 填入
  │     对填入的 primitive 施加 **masked noise**(多样性 + 与邻接的连贯)
  │     DDPM f_θ(m_t, t, x_music, M0, w) 预测 m0
  │     L = L_diff + λ·L_trans,L_trans = Σ_boundaries ‖M_b− − M_b+‖₁
  │
  └─ M6 Metrics: FID_k, FID_g, Div_k, Div_g, BAS, R-precision, MultiModality
```

---

## 1. 模块对齐审计(这是本文的核心)

| 模块 | 论文规定 | 仓库现状 | 对齐 |
| --- | --- | --- | --- |
| **M1** 分割 | I3D 视觉特征 + Alg.1 | `tools/segment_visual_atomics.py`,**逐行照 Alg.1 步骤 2–7**,但用 **S3D** 代替 I3D。N 与 L_min 论文未给,现由 `tools/calibrate_segmentation.py` 对着 Fig.4a 标定(fpc=34 / L_min=18):**5282 序列 / 78,227 段 / 均值 1.022s**,与 Fig.4a 的总变差 **0.420 → 0.162** | ⚠️ 编码器替换(已记录) |
| **M2** 聚类 | TMR motion encoder + K-Means + 只留近簇心 | **已对齐**:`tools/cluster_atomics_tmr.py` 用官方 TMR encoder;T1 校准门通过(文本语义追踪关节速度 p=5.2e-37,近邻同 genre lift 2.06) | ✅ |
| **M3** 组内再聚类 | Gemini-2.5-Pro + PoseScript + genre 预切 + LLM 蒸馏 | 两个 LLM 角色都已就位:`tools/caption_segments_vlm.py`(打标 VLM,Qwen3-VL 代 Gemini,人物裁剪 + PoseScript 关键帧线索)+ `tools/summarize_subprototypes_llm.py`(论文的 summarizing LLM)+ `tools/recluster_atomics_ingroup.py --subprototypes`。野外 v2:w/ LLM **543 子类**(带语义标签),w/o LLM **857 子类、8.57/类、32.06 样本**。**中立(TMR)空间下只有 w/o LLM 通过组内相干性闸(1.0398 vs 1.0046)** | ⚠️ genre 预切分缺来源;w/ LLM 的收益要 FID/R 才能兑现 |
| **M4** planner | D3PM,整首音乐条件 | `model/atomic_planner.py`(`max_seq_len=18000`,确为整首)+ `train_atomic.py --stage planner` | ✅ |
| **M4b** planner 后处理 | 滑窗多数投票 + 最短时长合并 | **已实现** `tools/postprocess_atomic_plan.py`;AIST A/B:段数 727→549(−24.5%),同歌<异歌 p 5.35e-05→**1.14e-08**,genre 控制仍不显著 | ✅ |
| **M5** completion | 时长匹配检索 + 缩放 + masked noise + DDPM + transition loss | `infer_atomic.py` 的 `AtomicRetrieval`(duration-aware)+ `--draft-noise-ratio` + `model/atomic_completion.py::transition_loss` | ✅ |
| **M6** FID_k/FID_g/Div/BAS | 四项 | `eval/metrics.py`:`calc_fid` / `calc_diversity` / `calculate_BAS` | ✅ |
| **M6b** R-precision | 论文自提,且是它相对基线优势最大的一栏(26.6 vs Lodge 18.2) | **已实现** `tools/eval_r_precision.py`。**不是** HumanML3D 的文本-动作 R,是论文自定义的结构一致性:音乐最相似的一对,其舞蹈是否互为 top-3。AIST GT 标定:pool 20 → 54.0,pool 40 → **34.4±10.0**,pool 128 → 7.0,论文 GT 42.1 落在 pool ≈ 25–35 | ⚠️ 论文未给 pool size,只能同 pool 比 |
| **M6c** MultiModality | 同一音乐 5 次采样的方差 | **已实现** `tools/eval_multimodality.py`。单独一个 MM 值没有意义(忽略音乐的模型把它最大化),所以与 Div 同距离同标准化并报 **MM/Div**,判据是**保持组大小的置换检验 p 值**。AIST GT 标定:test **0.4213**(零假设 0.996,p=0.005);val 按音乐 **0.4985** vs 按舞种 0.6169——音乐比舞种紧 19%。**~~野外语料上这一栏无法定义~~**(2026-08-16 更正见下) | ✅ |

### 更正(2026-08-16):M6c 在野外可以定义,靠的是指纹的**完全子图**

原文写「每 clip 自带音轨,一首曲子只有一支舞」。前半句对,后半句错 ——
错在**当时没有办法看出两条 clip 用的是同一首曲子**,不在于那种情况不存在。
`tools/fingerprint_wild_music.py` 之后证明了 14,569 对同曲关系。

但它自己产的 `track_of`(连通分量)**不能当分组键**,工具自己写了
`track_of_is_usable_as_a_music_key: false`:单链传递会让 `a--b--c` 把 `a` 和 `c` 归成一组,
而那一对从未过阈值 —— 本语料上最大的分量因此吞掉 2,335 条 clip(17%)。

可用的是分量里的**完全子图**:组内每一对都过了标定的操作点,任何一步都不假设传递性;
再要求**跨至少两个 upload**,否则又退回「同一个 take 的几段切片」。
野外 test split 上剩下 **144 组 / 320 条 clip**(121 个对、15 个三元组、7 个四元组、
1 个五元组)—— 不同编舞者跳同一首曲子,正是 AIST 的口径。

代价写明而不是藏起来:**这是按证据的性质选出的子集,不是语料的样本** ——
它测的是「指纹能连上的那些 clip」,而指纹在已知真对上的召回是 0.564。
入口是 `eval_multimodality.py --music-pairs --bundle`。

**另有一条:我此前在野外语料上跑的是 `discover_kinematic_atomics.py`,它连 M1/M2
的视觉替代版都不是**——它用 kinematic 特征(根轨迹/接触/rot6d/关节角速度)做
Alg.1 形状的分割,用 DCT 段嵌入代替 TMR。它的 docstring 第一句就声明自己是
baseline 而非论文复现。**把它汇报成"Atomic Movement Discovery 跑完"是我的错误。**

## 2. 缺口的代价(论文自己的数字)

Tab. 2(cluster base = 100):

| ReCluster | FID_k↓ | FID_g↓ | R↑ |
| --- | --- | --- | --- |
| ✗ | 32.68 | 12.09 | 21.3 |
| w/o LLM | 30.11 | 11.27 | 23.3 |
| **w/ LLM** | **25.26** | **9.03** | **26.6** |

Tab. 3:planner 后处理 25.26 → **24.02**(FID_k),R 26.6 → 27.5。

所以:**M3 缺失 ≈ FID_k 退 29%、R 退 20%**;M4b 缺失 ≈ 再退 1.24。论文 headline
25.26 就是 "w/ LLM" 那一行。**不做 M3,就不可能对上 headline**,这一点必须写在
任何结论前面。

## 3. M3 标注成本核算(Gemini,2026-08 定价)

定价依据(检索核实,非记忆):2.5 Pro **$1.25/M 输入、$10/M 输出**(≤200k 上下文);
2.5 Flash **$0.30/M 输入、$2.50/M 输出**;**Batch API 一律 −50%**;视频输入约
**258 token/秒**(默认分辨率;low-res 约 66 token/秒)。

### 标注量(以实测语料为基)

| 语料 | 时长 | 段数(按论文 0.81 s/段) |
| --- | --- | --- |
| AIST(复现论文 M3 作对照) | 5.2 h = 18,720 s | ≈ 23k(论文 Fig.4a 实和 23,194) |
| 野外·当前快照 | 38,353 s | ≈ 47k |
| 野外·收敛预估(~5,100 可训 clip) | ≈ 77,000 s | ≈ **95k** |

### 每段 token 账(caption 步)

1.5 s 视频片段 ≈ 390 tok + 指令与 PoseScript 辅助 ≈ 500 tok → **~900 in / ~200 out**。

### 成本表

| 步骤 | 量 | Pro 交互 | Pro Batch | Flash Batch |
| --- | --- | --- | --- | --- |
| caption 野外全量 | 95k 段 | $297 | **$149** | $37 |
| caption AIST 对照 | 23k 段 | $72 | $36 | $9 |
| LLM 蒸馏(纯文本,迭代 ~2.5 遍) | ~48M in / 5M out | $110 | $55 | $14 |
| 野外 genre 标注(见下) | 5.1k clip × ~4.5k tok | $34 | $17 | $4 |
| **合计** | | **≈ $510** | **≈ $260** | **≈ $64** |

**结论:全量 M3-b 的钱不是障碍。** 推荐组合:caption 用 **Pro + Batch**(质量敏感,
$185),蒸馏用 **Flash + Batch**($14),合计 **≈ $220**;全 Flash 方案 $64 但
caption 质量风险自担。**免费层不可行**:Pro 免费档 ~100 请求/天,95k 段要跑约
2.6 年——免费 key 只用于 prompt 调试与抽检(~50 段)。

### 免费替代

| 方案 | 成本 | 质量/代价 |
| --- | --- | --- |
| 本地开源 VLM(Qwen2.5-VL-7B/72B,闲置 72GB 卡) | $0,95k 段 ≈ 13–130 GPU-h | 是一次**替换**,须像 S3D≠I3D 一样入档;细粒度舞蹈描述质量待抽检对比 |
| M3-a(无 LLM 组内再聚类) | $0,纯本地 | 对应 Tab.2 的 30.11 行,拿不到 25.26 |
| PoseScript 关键帧描述 | $0(naver/posescript 开源,规则+模型本地跑) | 论文本来就是辅助线索,无论走哪档都免费 |

**野外 genre**:TikTok 无舞种标签,论文 M3 的 genre 预切分在野外需要来源。用 VLM
对**每 clip**(非每段)打 genre 标签即可($4–34),入档为"VLM 估计 genre",不冒充 GT。

## 4. 执行计划(reframed)

```
W0  阶段A收敛监控(进行中)────────────────┐
W1  T1: TMR 集成(无外部依赖,立即开始)     │ 并行代码任务(无依赖):
W2  M1: 野外视觉分割(S3D+Alg.1,工具已有)   │  · M4b 后处理 + AIST A/B
W3  M2: TMR 嵌入聚类 → 100 prototype        │  · M6b R-precision(GT 校准≈42.1)
W4  M3-b: 免费key调prompt → Pro Batch caption│  · M6c MultiModality
      + PoseScript 辅助 + Flash 蒸馏         │
      (key 未到位则先 M3-a 顶上)            │
W5  M4: planner 训练(整首音乐条件)          │
W6  M5: completion 训练                      │
W7  M6 全套指标 + gate v2(判决点)+ 画廊    │
```

### T1 — TMR 集成(新,关闭 M2 缺口)

已就位:`third_party/TMR/models/tmr_humanml3d_guoh3dfeats/last_weights/`(md5 校验
`7b6d8814f9c1ca972f62852ebb6c7a6f`)+ `stats/humanml3d/guoh3dfeats/{mean,std}.pt`
(263-D,已加载验证)。待做:

1. `tools/setup_tmr_env.sh`:固化下载(gdown id + md5)与 stats 取回;
2. `tools/convert_motion_to_guofeats.py`:151-D → FK 关节(复用已钉死的
   `SMPLSkeleton`)→ 取 22 关节(丢 22/23 手关节,本来就是 identity)→ z-up→y-up
   → 30→20 fps 重采样 → 按 `prepare/compute_guoh3dfeats.py` 逐行实现 Guo 263-D;
3. **判据 T1**:(i) AIST 段编码全有限;(ii) 最近邻检索同 genre 富集显著高于随机;
   (iii) 用 text_encoder 编 "a person kicks" 等 10 条探针文本,对应动作段的相似度
   显著高于打散基线——TMR 是 motion-text 联合空间,这一条能真正验证空间没接错。

### M1/M2 判据(不变,补一条)

段时长分布对 Fig.4a;100 类平均样本数按规模折算对 268.57;S3D≠I3D 的替换必须
写进产物 metadata。**M2 切到 TMR 后,与 S3D 段均值版本做一次对照聚类**,报告
类平衡与轮廓系数的变化——这就是"编码器替换值多少"的直接测量。

### M3-b 执行细则(已落地,走本地 Qwen 而非 Gemini)

Gemini 在本网络不可达,两个 LLM 角色都由本地 Qwen 承担,**替换写进每一条 caption
行和每一条标签行**,与 S3D 代 I3D 同样处理。整条链一条命令:

```
TAG=wild_v2 FRAMES_PER_CLUSTER=34 MIN_LENGTH=18 bash tools/run_atomic_discovery.sh
```

1. **打标 VLM**:`tools/caption_segments_vlm.py`。输出**闭词表 JSON**(6 个轴)+
   自由 `summary`;贪心解码。闭词表不是偷懒——caption 存在的唯一目的是被聚类,
   同一个动作两次必须回同样的词,自由散文恰好在模型有余地的地方摇摆;
2. **PoseScript 辅助线索**:`--posescript` 把 motion beat 关键帧的规则版 posecode
   写进 prompt,并明说"与画面冲突时以画面为准"——它来自同一批帧的 3D 估计,不该
   反过来覆盖画面;
3. **选模型用闸不用感觉**:`tools/probe_caption_quality.py` 的 G1/G2/G3 + `--compare`
   交叉复标。实测 7B 把 17.5% 的段压成同一句,30B 只有 6.1%;
4. **summarizing LLM**:`tools/summarize_subprototypes_llm.py`,严格按论文的循环
   ——(i) 挑出互相相似的一组 caption 作 sub-prototype,(ii) 蒸馏成语义标签,
   重复到未分组段低于阈值。残余按字段一致度就近挂靠,并**单独计数**,不把 LLM
   没分的段算成 LLM 分的;
5. **写回**:`tools/recluster_atomics_ingroup.py --captions ... --subprototypes ...`
   ⇒ Tab.2 的 **w/ LLM** 行;不给 `--subprototypes` 时是别的东西,report 会说清楚;
6. **验收**:`tools/audit_atomic_vocabulary.py`(对 Fig.4a/4b/4c + 组内相干性置换
   检验)与 `tools/build_atomic_gallery.py`(每个子类一张接触表,行=不同 upload 的
   成员段,列=段内采样帧)。**形态达标但相干性不达标 = 随机切分**,所以相干性那条
   才是判决项。

## 5. 必须写在任何结论前面的话

1. **野外语料没有配对 GT 舞蹈**,FID 只能与"野外 GT"分布比,不得与 AIST 数字混排;
2. **S3D ≠ I3D**;M2 修复后 TMR 一致,但分割编码器仍是替换,必须标注;
3. **未做 M3-b 之前不得声称复现 25.26**;做完也要标注 caption 模型与论文版本差异
   (Gemini 2.5 Pro 是论文用的,若用 Flash/本地 VLM 须写明);
4. 帧准确率不进任何主结论(本仓既有结论);
5. Gemini 上传的是**用户提供的 TikTok 片段**,发送即对外发布,标注前确认合规;
6. **R 只在同一 pool size 下可比**。实测同一批 AIST GT clip:pool 20 是 54.0,
   pool 128 是 7.0——同一份数据、同一个实现,差 7.7 倍。lift over chance 也不是
   不变量(它反向变化)。论文没给 pool,所以任何与 26.6 / 42.1 的对比都必须把
   自己假设的 pool 印在旁边,不得只报一个 R;
7. **词表规模(K=100、子类 ≈ 段数/32)是抄论文抄来的,不是本语料测出来的。**
   论文 Tab.2 自己扫过 base num:75 → FID_k 33.24,100 → **32.68**,125 → 34.57
   ——多不等于好,过多会 over-fragment。要在野外改 K,必须像论文那样用 FID/R 判,
   不能用"野外更丰富所以类该更多"这种先验。

---

# 6. 本轮执行计划(2026-08-11):扩产语料上的完整 discovery

## 6.0 为什么有这一轮

一次资产盘点发现语料根本不缺:**6,041 条 clip 早就有 2D + 分段**,其中 3,373 条
从未过 3D,而 2,434 条的已发布词表是**音频过滤**的结果,不是素材短缺。补齐之后
discovery 的有效语料从 **2,434 → ≈5,270**,所以整条链要在新语料上重跑一遍。

补齐的账目(已收口):

| | 数量 |
| --- | --- |
| 原 pending | 3,373 |
| 其中 OSS 上早有 raw、只是从未转换 | 3,111 → **转换成功 3,108**(3 条缺 `extract_meta.json`) |
| 其中连 raw 都没有、需重抽 | 262 → **成功 131 / 失败 131** |
| **最终转换总量** | **5,907** |
| 彻底拿不回来 | 134(131 条 VO 发散 + 3 条缺元数据) |

失败全部是 `visual odometry diverged: non-finite camera track`,是内容层面的真失败
(这 262 条正是当初唯一没产出结果的残渣),不是环境问题。

## 6.1 依赖图(这一节是本文的重点)

```
[3D 补齐 ✔]──┐
             ├─→ reconcile v4 ──→ audio_35_v4 ──→ bundle v2 ──→ fit norm ──→ apply norm ──┐
[视频链接 ✔]─┘        (闸:hmr候选)   (闸:bundle准入)                                      │
                                                                                          │
[S3D 补齐]────────────────────────────────────────────────────────────────────────────────┤
                                                                                          ↓
[PoseScript 建表+接线]──┐                                                            M1 分割
[genre prompt 验于AIST]─┤                                                                 ↓
                        └────────────────────────────────────────────────→ M2 TMR 聚类
                                        (必须在 M3a 开跑前定稿)                           ↓
                                                                                    M3a caption
                                                                                          ↓
                                                                                    M3b 摘要 LLM
                                                                                          ↓
                                                                            M3c 组内再聚类(--genre-split)
                                                                                          ↓
                                                                                   audit + gallery
```

**三条容易被忽略的因果边:**

1. **audio 是 bundle 的准入闸,不是可选项。** `wild_performance_v1/report.json` 写着
   `selection: audio_feature_status == candidate`。新 clip 的 audio 状态全是 `None`
   ——不是提取失败,是**音频特征按已转换 3D 的 `frame_ids` 逐帧对齐提取**,当初它们
   没转换所以标 `not_attempted`。不重跑音频阶段,直接建 bundle 出来还是 2,434 条,
   **新 3D 对 discovery 完全不可见**。
2. **PoseScript 与 genre 必须在 M3a 之前定稿。** caption 是 4–5 小时的贵阶段,cue 换
   了要重跑,genre 字段与 caption 同一遍出才近乎免费。
3. **必须换 TAG(`wild_v3`)。** `run_atomic_discovery.sh` 每一步都是"输出已存在就
   跳过",沿用 `wild_v2` 会把 2,434 条语料时代的 segmentation 与 labels 直接复用,
   产出一个名不副实的结果。

## 6.2 本轮踩到的四个坑(都已修,记下来免得重犯)

1. **符号链接把相对路径打穿。** `third_party/GVHMR` 变成指向缓存的符号链接后,
   `cd third_party/GVHMR && python ../../tools/...` 落到缓存的父目录,那里没有
   `tools/`,每条 clip 都在解释器那一步失败。改用 `$REPO` 绝对路径。
2. **reconcile 会把易失路径烙进账本。** `reconcile_wild_hmr_sequences` 对
   `converted_root` 调 `.resolve()`,而 `data/wild3d/converted` 现在是指向 `/cache`
   的符号链接,于是 manifest 里写的是缓存绝对路径(v3 记的是仓库路径)。缓存被清
   即失效,**产出后必须归一化回仓库路径**。
3. **`PERSON_BOXES` 默认值只覆盖一半语料。** `run_atomic_discovery.sh` 默认
   `data/wild3d/gvhmr_raw`,那是缓存符号链接、只有 3,373 条;全量 6,041 条在
   `data/wild3d/gvhmr_boxes`(当初抽 bbx 时留的),布局一致。用默认值会静默丢掉
   原有 2,667 条的人物裁剪框。
4. **等待循环自匹配。** `until [ "$(pgrep -fc run_gvhmr_wild_shard.sh)" = 0 ]` 会匹配
   到等待器自己的命令行,永远等不到 0。

## 6.3 PoseScript:从规则版换成发布模型

权重已下载(`capgen_CAtransfPSA2H2_dataPSA2ftPSH2/seed1/checkpoint_best.pth`),但
**代码从未接上**——`caption_segments_vlm.py::posescript_cue` 至今调规则版
`describe_pose`。接线要点:

1. **词表发布方不带,要自己建。** README 给了确切命令(含几百词的 `--new_word_list`
   与 `--make_compatible_to_side_flip`),并写明**期望大小 2158**——正好等于
   checkpoint 里 `text_decoder.embedding.weight (2158, 512)`。这个数字是校验位:
   建出来不是 2158 就是建错了。词表错会让模型吐出**通顺但错误**的词,是读几条
   caption 发现不了的静默损坏。
2. **输入是轴角不是关节坐标。** 规则版吃 `[J,3]` 关节位置,真模型吃 `(1,52,3)` 轴角
   (编码器输入 156 = 52×3)。轴角从 151-D 的 `[contacts(4), root(3), rot6d(24×6)]`
   反解。
3. **关节映射**:我们 24 关节 SMPL 的 `[0..21]`(global + 21 body)对到 SMPL-H 的
   `[0..21]`,手指 30 关节置零;我们的 22/23(手腕级)与 SMPL-H 的手指层级不对应,
   丢弃——GVHMR 本来也不估计手指。
4. **朝向归一化直接调用他们自己的函数**,不重新实现。PoseScript 的约定是 z-up、
   把 global orient 的 euler-z 置零(抹掉朝向角)、保留 x/y 倾斜。我们的数据本来
   就是 z-up,两边同框——但本仓已经栽过一次"左右读错轴",所以这一步不自己写。
5. **许可**:CC BY-NC-SA 4.0(非商用),当前分支是 `p0-source-safe-release`,产出物
   带该模型的痕迹需要在 release 侧有个明确决定。

## 6.4 genre 预切分:能力对齐的正确做法

论文 M3 明确 "we first pre-split each atomic movement prototype P_j by dance genre"。
`recluster_atomics_ingroup.py` **已实现 `--genre-split`**,`genre_of()` 认 AIST 的
`^(g[A-Z]{2})_` 命名。**缺的是野外的 genre 来源,不是能力。**

做法(与 §3 的预案一致,补上验证):

1. 给 VLM 的闭词表 schema 加 `genre` 字段,**按 clip 而非按段**打标;
2. **拿 AIST 当参照系验证**:AIST 有 10 类 genre 真值写在序列名里。同一套 prompt
   跑在 AIST clip 上,出混淆矩阵与准确率。**这是"能不能信"的闸**,不是读几条觉得像。
   这与用动捕当参照系审计野外 3D 是同一套做法——没有参照系,VLM 的 genre 会被当成
   事实入档;
3. 旁证:worklog 记过 **genre 从音乐 35-D 特征可判**(但同段也警告 val 有泄漏,那个
   1.0 不能当泛化)。可用音频判的 genre 与 VLM 判的做一致性交叉检验;
4. **入档为 "VLM 估计 genre",不冒充 GT**;未过闸则维持 `genre_presplit: false`,
   如实记录缺失——编一个比记录缺失更糟。

## 6.5 待办与判据

| # | 事项 | 依赖 | 判据 |
| --- | --- | --- | --- |
| 1 | 3D 补齐收口 | — | ✔ 5,907 转换,失败原因已归类 |
| 2 | 660 条视频链接 | 1 | ✔ 无断链 |
| 3 | Qwen 权重回拉 | — | ✔ 7B + 30B 在缓存 |
| 4 | S3D 补 635 条 | 2 | 全部 converted clip 有 `.npz` |
| 5 | reconcile v4 | 1 | candidate 数 = 5,907;路径归一化回仓库路径 |
| 6 | audio_35_v4 | 5 | 新 clip 由 `not_attempted` 转为 candidate/quarantine |
| 7 | bundle v2 | 6 | 序列数 ≫ 2,434 |
| 8 | fit/apply normalizer v2 | 7 | 仅用 train split 拟合 |
| 9 | PoseScript 接线 | — | 词表 len == 2158;抽检 cue 与画面一致 |
| 10 | genre prompt | — | AIST 混淆矩阵达标才启用 `--genre-split` |
| 11 | discovery `wild_v3` | 4,8,9,10 | `audit_atomic_vocabulary.py` 四条判据 + 组内相干性置换检验 |

**相干性那条才是判决项**——形态达标但相干性不达标 = 随机切分。

---

# 7. 前端重建(2026-08-11 下午):切分不再由 ffmpeg 的时钟决定

## 7.0 为什么 §6 的计划不够

§6 假定语料的 clip 边界是给定的,只重跑下游。**这个假定是错的。** 现行 clip 来自
Lodge 的 `scripts/preprocess_wild_videos.py`,里面两个常数决定了下游的一切:

```python
SEG = 16          # seconds per clip (match historical 16s cache)
MIN_FRAMES = 180  # past60 + future120
```

`SEG=16` 的理由写在注释里:对齐历史缓存。切法是 `ffmpeg -f segment` 定长盲切,
不看镜头、不看有没有人在跳。`MIN_FRAMES=180` 是 Lodge 自己 motion-continuation
的窗口(60 帧过去 + 120 帧未来),对本仓没有任何意义。

代价是实测的,不是推测的:

- **78,227 段里 10,564 段(13.5%)的边界是 ffmpeg 的时钟**(每条 clip 的首尾段);
- clip 越短,Alg.1 分得越粗(簇数 = T/34)。6–7 s clip 平均段长 **1.171 s**,
  16.7 s 满长 **1.022 s**,单调——论文目标是 0.81 s;
- 原始上传中位 24.9 s,**83% 会被 16 s 规则切开**,于是同一个上传的前后半段被用
  不同粒度切分,差异的唯一来源是 ffmpeg 的时钟。

所以这一轮把切分收回本仓:`tools/ingest_wild_uploads.py`。

## 7.1 三个常数,各有各的来源

| 常数 | Lodge | 本仓 | 来源 |
| --- | --- | --- | --- |
| 切在哪 | 定长盲切 | 镜头切换 + 无舞者间隙 | 同一遍检测顺带得到 |
| 下限 | 180 帧(`past60+future120`) | **340 帧** | Alg.1 簇数 = T/34,要 10 个簇才谈得上自相似结构 |
| 上限 | 16 s | **待定** | `tools/scan_clip_length.py` 的输出;工具**故意不带默认值** |

上限是唯一推不出来的:往长了走段密度还没饱和(0.988 段/秒仍在上升),往短了走
VO 漂移少。**这两条谁赢要测**,所以 `--max-seconds` 没有默认值,没跑扫描就建不出语料。

## 7.2 切镜阈值:分位数根本够不着

`--calibrate-cuts` 报的分位数(p50 0.022 / p90 0.077 / **p99 0.125**)全部落在
"舞者在动"那个包里。40 条上传 ~51k 帧中 **510 帧超过 0.10,其中只有 8 帧是真切镜**
——切镜是 1e-4 量级的事件,任何分位数都会淹没在运动里。我最初写的 0.115 会在超过
1% 的帧上触发,25 秒的视频切出上百刀。

真假在**两个轴上同时**分开:

| | 真切镜(8 帧) | 最接近的假阳性 |
| --- | --- | --- |
| 绝对差值 | 0.237 – 0.754 | ≤ 0.194 |
| 与本片中位数之比 | 11.4 – 80.3 | ≤ 4.7 |

所以两条都要:只用绝对值会在**有人从镜头前扫过**时触发(图上确认:连续 3 帧,是遮挡
不是切镜);只用比值会在**中位差为 0 的静态视频**上炸成 2723×(图上确认:前后两帧
几乎一模一样)。取 `FLOOR=0.22, RATIO=8.0`,落在两个间隙里,偏保守一侧——
**漏切和多切代价不对称**:漏掉的切镜会在 Alg.1 的自相似矩阵里留下不连续,它本来
就会在那里切;多切一刀会把一个好的 take 劈成两半,两半都可能低于 340 帧而整条丢掉。

## 7.3 多舞者:这批语料的一条从未入档的性质

抽 60 条已发布 clip 实测:

| | |
| --- | --- |
| 平均每帧检出人数 | **7.52** |
| 存在"对手" track(分数 ≥ 主舞者 50%)的 clip | **70%** |
| 对手分数 ≥ 80% 的 clip | **50%** |

**大量是舞蹈班/群舞素材,"哪个是舞者"没有唯一答案。** 拿 GVHMR 自己选的人
(`preprocess/bbx.pt`,6,041 条全有)做参照:12 条里 6 条 IoU 0.88–0.96,6 条完全
选了不同的人——而画出来看,**两个框都稳稳站在真舞者身上**。这不是选错,是问题本身
没定义。

两条后果:

1. **现有语料的 2D 与 3D 可能描述不同的人。** 修法:ingestion 选定后把框写进
   `preprocess/bbx.pt`,GVHMR 读它(`demo.py` 在文件存在时不跑自己的 tracker),
   两者由构造一致,而且这个任意选择只做一次、记录一次;
2. **整帧 S3D 驱动 M1,单人 3D 驱动 M2。** 这只有在同框的人不同步时才是缺陷。
   实测(n=33,主/次 track 的运动信号相关):**median r=0.253,时间反转对照
   -0.006,Wilcoxon p=1.2e-9**——同步是真的,但很弱。所以整帧 S3D 不是在测无关
   对象,也不是干净的单人信号。`rival_ratio` 逐 clip 入档,下游可分层。

## 7.4 主舞者判据,以及它买回来的算力

`score = coverage² × √area × (0.25 + motion)`,两个指数都不是拟合的:

- **coverage 平方**:coverage 线性时,一个只出现 8/100 帧但贴着镜头的路人会盖过
  全程在场的舞者(测试逮到的)。"是不是这条视频的主角"在这里近乎定义性质;
- **面积开方**:面积随距离平方增长,原始面积等于说"近一倍就是舞者程度的四倍";
  开方后是框的线性尺度,那才是"这个人在画面上多大"的意思。

代价侧:检测 9.8 ms/帧,pose **每个框** 5.0 ms。跑全部 ~10 个框是 53 ms/帧
(18.9 fps),只跑选定的舞者是 5 ms(端到端 **67.6 fps**)。全语料 2D 从
**~230 卡时降到 ~34 卡时**,6.9×。

## 7.5 这一轮的待办与判据

| # | 事项 | 依赖 | 判据 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | S3D 补齐 | — | 全部 converted clip 有 `.npz` | ✔ 6 分片全绿,0 失败 |
| 2 | v5 manifest 路径归一化 | — | 0 行含 `/cache` | ✔ 6,041 行 |
| 3 | ingestion 迁入本仓 | — | 帧对齐精确;16 个测试 | ✔ `cut_clip` 中段切分 f0≡源 f100 |
| 4 | 切镜阈值标定 | 3 | 图能裁决,不是只有分位数 | ✔ 见 §7.2 |
| 5 | **长度扫描(P1 闸口)** | 3 | VO 发散率 + 垂直漂移 vs 长度,对动捕 p99 | ✔ 144/144,判决 **24 s**(§7.7) |
| 6 | 边界判定验证渲染 | 4,5 | cut sheet + timeline 能裁决 | ✔ 读缓存,不花 GPU |
| 7 | 原片全量到齐 | — | OSS 全部唯一上传在本地 | ✔ 10,793 条 / 87 GB / 0 失败 |
| 8 | PoseScript 接线闸 | — | 词表 len == 2158;左右机械核验 | ✔ 2158;可判定帧上 96% |
| 9 | 全量内容重切(10,793 上传) | 5,6 | 每条 clip 有 2D/audio/bbx/meta,账本无缺口 | ⏳ 12 分片 / 6 卡 |
| 10 | A/B 配对对照(盲切 vs 内容切,同 200 上传) | 9 | 段长分布 + 假边界率,**挡着第 11 条** | ✔ 逐 clip 段长散布 **−61%**、假边界 −24%、留存 90.8% |
| 11 | GVHMR 3D 重建(注入 bbx) | 9,10 | 2D 与 3D 同一人(构造保证) | ⏳ 4,7xx / 17,790;**吞吐缺陷已修**(§7.12),358 → 1,032 条/h |
| 12 | S3D + reconcile + audio + bundle + normalizer | 11 | candidate 数 = 转换成功数 | 待 11 |
| 13 | discovery M1–M3(PoseScript + genre 预切) | 12 | `audit_atomic_vocabulary.py` 四条 + 相干性置换检验 | 待 12 |
| 14 | genre 标注器(旁路) | AIST 视频 | AIST 混淆矩阵达标才启用 `--genre-split` | ✘ **不通过**:10 类均衡样本准确率 0.083(chance 0.10),16 帧无改善 → 本轮 `genre_presplit: false`,如实记缺失 |

**整条链一条命令**:`TAG=wild_v4 MAX_SECONDS=24 bash tools/run_wild_rebuild.sh`
(阶段 A–G,每步跳过已存在的产物,`FROM=`/`ONLY=` 可从中间接上)。剩余算力实测
约 **70 小时 / 6 卡**,单次会话覆盖不了,所以驱动必须可断点续跑而不是一串手敲命令。

**第 7 条替换了 §6 的"旧语料 M2 对照"。** 理由:对照的目的是回答"内容切分有没有让
词表变好",而 2,434 条的旧语料对 ~15,000 条的新语料本来就不可比(本仓已经测到 R
随 pool size 变 7.7 倍)。**同一批上传两种切法**才是配对比较,而且便宜得多。

## 7.6 必须写在结论前面的话(追加两条)

8. **这批语料多数是群舞素材,单舞者是少数。** 论文与 AIST++ 的前提是单人,本语料
   70% 不满足。选人这件事已由构造统一(bbx 注入),但"整帧视觉特征 vs 单人 3D"的
   弱同步(r=0.253)是真实存在的噪声源,不得省略不报;
9. **长度扫描的样本偏长尾。** 要让同一段素材切成 8–64 s,上传本身必须 ≥65 s,而
   语料上传时长 p90 才 49 s。所以扫描测的是"估计器对长度怎么反应",不是"语料平均
   难度"——任何由它定出的上限都要带这句。

## 7.7 长度闸口的判决:**`--max-seconds 24`**

24 条上传 × 6 个长度 × 完整 3D 链路。两族度量:`full` 是整段(下游实际消费的),
`head` 只看**每一行都相同的前 8 秒**——它把"帧多了漂移累积"和"长序列连共享部分
都估计得更差"分开。

| 长度 | n | 失败率 | full 脚踝漂移 | **head 脚踝漂移** | 超动捕 p99 |
| --- | --- | --- | --- | --- | --- |
| 8s | 24 | 0.0% | 0.0882 | 0.0882 | 4% |
| 16s | 23 | 0.0% | 0.1175 | 0.0885 | 13% |
| 24s | 22 | 8.3% | 0.1663 | 0.0876 | 18% |
| 32s | 21 | 12.5% | 0.2436 | 0.0793 | 19% |
| 48s | 19 | 16.7% | 0.3340 | 0.0936 | 37% |
| 64s | 16 | 33.3% | **0.5120** | **0.1297** | 50% |

**head 那一列是判决的关键**:同一段前 8 秒,无论它属于 8s 还是 48s 的 clip,估计
质量都一样(0.0798–0.0885,无趋势)。所以 `full` 的恶化是曝光量,不是估计器退化
——**直到 64s,连共享的头部都退化 47%,而 full 的 0.512 已经贴上动捕 p99 的 0.5303**
(典型的 64s clip = 最差 1% 的动捕)。

按**实际 clip 长度分布**加权(上限只对长 take 生效,多数 clip 短于上限;权重取每条
clip 自己的时长,因为丢一条 30s 比丢一条 12s 损失大):

| 上限 | clip 数 | 均长 | E[失败] | 可用留存 | E[超 p99] | 全语料卡时 |
| --- | --- | --- | --- | --- | --- | --- |
| 16s | 1402 | 14.0 | 0.0% | 89.0% | 11.0% | 211 |
| 20s | 1255 | 16.1 | 1.3% | **90.3%** | 12.9% | 208 |
| **24s** | **1145** | **17.8** | **2.9%** | **89.3%** | **14.2%** | **203** |
| 32s | 948 | 21.5 | 6.2% | 86.3% | 16.3% | 191 |
| 48s | 761 | 26.8 | 9.9% | 82.8% | 20.5% | 183 |
| 64s | 703 | 29.0 | 12.1% | 80.9% | 24.3% | 183 |

(705 条已缓存上传的切分重放 × 闸口的最终曲线;更早一版基于 124/144 的部分结果,
数字略差而结论相同,此处只保留最终值以免两套数字并存。)

**成本不是变量**(全程只差 13%,因为均长不随上限成比例增长)。留存在 16–24s 是一个
平台,之后下滑;质量单调变差。取 **24 s**:留存仍在平台上(86.6%,距最大值 1pp)
的**最长**上限,而更长的 clip 正是粒度靠近 0.81 s 的方向。

**绝对水平不可引用,趋势才可以。** 要让同一段素材切成 8–64 s,上传必须 ≥65 s,而
语料上传 p90 才 49 s——样本落在长尾,难度偏高。同一指标在已发布语料上只有 5.7%
越界,不是这里的 14.2%。**跨长度的比较是配对的(同一素材),所以趋势有效;绝对
数字不得与语料统计混排。**

### 顺带被 dry run 逮到的一个 bug

`find_spans` 原本对超长跨度做等分。在 16 s 上限下,一个 500 帧的 take 等分成两段
250 帧,**两段都低于 340 帧的下限,于是整条丢掉**——278 条缓存上传里 112 条(40%)
一条 clip 都产不出。改成:等分只在每段都过下限时使用,否则按整段截取、丢掉不足
下限的尾巴(最多损失 339 帧,而不是全部)。修复后同一批上传的留存从 68.6% 升到
89.6%。**这个错误在建语料之前被抓住,靠的是检测缓存能在 CPU 上重放切分决策。**

## 7.8 重建后的规模,以及它把成本推到哪一步

按已切的 2,106 条上传实测(1.63 clips/上传,均长 17.5 s)外推:

| | wild_v2 | 重建后(投影) | 倍数 |
| --- | --- | --- | --- |
| clip | 5,282 | **17,624** | 3.3× |
| 视频秒 | 79,922 | 309,135 | 3.9× |
| 段 | 78,227 | ~302,953 | 3.9× |
| **要打标的段** | 27,475 | **~106,403** | **3.9×** |

打标量按 wild_v2 实测比例(27,475 / 78,227 = **35.1%**)推——不是全部段都打:M2 的
接受闸和 `--min-frames` 会先滤一遍。

**成本因此压在 M3a 上**,而不是 3D:

| 阶段 | 6 卡墙钟(实测或投影) |
| --- | --- |
| 内容切分 | ~8 h(实测速率) |
| GVHMR 3D | ~35 h(2.66 GPU-s/视频秒实测) |
| C–E staging/audio/bundle/normalizer | ~2 h |
| F S3D | ~3.3 h(GPU transform 后实测 48 条/分/3 卡) |
| M1 分割 | ~1.6 h(8 分片;单进程是 13 h) |
| M2 TMR 聚类 | ~4 h |
| **M3a 打标 VLM** | **~19 h** |
| M3b/M3c + audit + gallery | ~3 h |
| **合计** | **≈ 76 h** |

M1 的 8 分片是这一轮加的:单进程在 ~18k clip 上是 13 小时的一步,而 `--shard` /
`--merge-glob` 本来就在工具里。**分片与单进程等价性已验**:40 条特征、503 段、
零条边界不同。线程上限不是可选项——numpy 每进程吃满所有核,8 个不加限制的分片会
把 256 核的负载推到 643,比单进程还慢。

### 7.9 一个必须在 M3c 之前决定的事:词表会跟着语料四倍膨胀

`recluster_atomics_ingroup.py --target-size 32` 把每个 prototype 内的子类数定成
「该类段数 / 32」,K=100 固定。所以**子类总数是语料段数的一次函数**:

| | 段数 | 子类 |
| --- | --- | --- |
| 论文 | 23,194 | 730(100 × 7.3) |
| wild_v2 | 78,227(打标 27,475) | 857 |
| 重建后(投影) | ~302,953(打标 ~106,403) | **~3,300** |

**这个四倍膨胀没有任何测量支持——它只是语料变大了。** 而论文 Tab. 2 自己扫过
base num:75 → FID_k 33.24,100 → **32.68**,125 → 34.57。**多不等于好,过细会
over-fragment。**

所以 M3c 之前要在三条里选一条,并且用 FID/R 判而不是先验:

1. **保持 `--target-size 32`**,接受 ~3,300 子类。要论证的是"野外语料的动作多样性
   确实是 AIST 的 4.5 倍",而这条无人测过;
2. **按语料规模缩放 target-size**(段数/857 ≈ 每子类 124 段),把子类数钉在 wild_v2
   的量级,变量只剩语料质量;
3. **像论文那样扫 K 与 target-size**,用 FID_k / R-precision 定。这是唯一有判据的
   做法,代价是 M3c 之后要多跑几轮评测。

**默认走第 2 条**(变量最少,可与 wild_v2 直接对比),第 3 条作为随后的扫描。无论
走哪条,**子类数必须与 wild_v2 并排报告,并注明它是被什么决定的**——否则"词表更大"
会被读成"发现了更多动作"。

## 7.10 genre 预切分:本轮不做,并说清为什么

`tools/label_clip_genre_vlm.py` 对 AIST 十舞种打标(真值在文件名里):

| 配置 | n | 准确率 | chance | 答 "other" |
| --- | --- | --- | --- | --- |
| 8 帧,10 类均衡 | 36 | **0.083** | 0.10 | 64% |
| 16 帧 vs 8 帧(同一批,配对) | 16 | 0.062 / 0.062 | 0.10 | 62% / 75% |

**在 chance 上或以下,加倍帧数毫无改善。** 所以 `--genre-split` 本轮不启用,产物记
`genre_presplit: false`。

不是保守:pre-split 是把 prototype 按舞种切开,**在 chance 水平的标签上切等于把同一
个动作随机撒进十个组**,比不切更糟。论文 Tab. 2 的收益来自"w/ LLM"那一行,不是来自
genre 这一步单独的贡献,所以缺它可以如实标注而不必伪造。

**限定**:论文用 Gemini-2.5-Pro **吃视频**,这里是本地 VLM **吃采样静帧**。否定的是
这个配置。要重开这一条,方向是给模型时序而不是更多静帧——16 帧不比 8 帧强,说明
瓶颈不在帧数。

### 更正(2026-08-12):`--genre-split` 在 wild_v4 是**开着**的,但切的不是 genre

本节此前写着"本轮不启用 `--genre-split`",而 `run_wild_rebuild.sh` 的 G 阶段确实传了
这个 flag。**两者不矛盾,是本节写漏了一层**:被否定的是"让 VLM **预测**舞种",不是
"按某个观测字段预切分"。实际用的 key 是 `GROUP_KEYS` 指向的
`runs/wild_v4_group_keys.json`——**上传者/编舞账号**,从 OSS 的路径元数据里读出来的,
10,793 条上传对应 **25 个账号**。

`recluster_atomics_ingroup.py::genre_of` 把理由写清楚了,值得抄在这里:AIST 的 genre
是**数据集字段,论文是读它不是预测它**;TikTok 没有这个字段,VLM 预测又在 chance 上
(0.083 / 0.113),所以换一个**同样是观测而非推断**的字段。一个编舞者的风格比"hip-hop"
更具体,所以这个分组**可能比 genre 更紧**。

**但它是否真的按语义分组是另一个问题,答案在 `audit_atomic_vocabulary.py` 的相干性
置换检验里,不能假定。** 另有一条要并排报告:账号分布**极不均衡**——最大的账号
`番茄Cherry` 一个人占 2,409 / 10,793(22%),所以"预切分"在它身上几乎等于没切。

产物里应记 `presplit_key: choreographer_account`,而不是 `genre_presplit: true/false`
——后者会把一个观测字段读成舞种。

## 7.11 语料的第三条性质:6.27% 的 clip 中途换了舞者

`tools/audit_dancer_tracks.py`,全语料 6,924 条(0 不可判 / 0 失败):

| | |
| --- | --- |
| 有跳变候选 | 1,242(17.9%) |
| **确认换人** | **434(6.27%)** |
| 换人时框高中位变化 | 45.6% |
| `rival_ratio` 中位:换人 clip | **0.6415** |
| `rival_ratio` 中位:干净 clip | **0.2422** |

成因是特性的另一面:IoU 链接桥接半秒间隙,好让舞者转身不结束 take;A 离开、B 在附近
出现时,track 就接到 B。**两个独立算出来的量互相印证**——换过人的 clip,对手 track
强度是干净 clip 的 2.6 倍,方向与理论一致。

**输出是逐 clip 标记,不是过滤器。** 一条拼接 clip 的害处取决于谁读它:整帧 S3D 的
分割基本不在乎(它本来就在看所有人),单人 3D 的 TMR 嵌入很在乎(它会把两个人的
动作混成一个嵌入)。

**这个决定已经做了,而且比本文原先写的位置更早**(2026-08-12 更正)。本文此前写着
"M2 之前要决定",而 `run_gvhmr_ingest_shard.sh` 的 worklist 构建**在 3D 入口就把
`switches > 0` 的 clip 排除了**——阶段 B 的实际分母因此是 **16,715** 而不是盘上的
17,790 条 clip(排除 1,075 条,6.0%)。理由写在脚本里:这些 clip 的姿态拼了两个人,
在它们上面跑 GVHMR 是浪费算力。

代价与之前分析的一致,只是提前发生:少 6.0% 的语料,换掉"嵌入描述的不是一个人"。
**产物里必须并排报告这个分母**,否则 16,715 会被读成语料规模,而它是筛过的。

### 三条性质放在一起看

| 性质 | 数值 | 影响谁 |
| --- | --- | --- |
| 群舞占多数 | 70% 的 clip 有 ≥50% 强度的对手 track | "哪个是舞者"没有唯一答案;已用 bbx 注入统一 |
| 同框者弱同步 | r=0.253(反转对照 -0.006,p=1.2e-9) | 整帧 S3D 不是在测无关对象,但也不干净 |
| 中途换人 | 6.27% | 那些 clip 的 2D/3D 拼了两个人 |

三条都是同一个根源:**这批语料的单人假设不成立**。论文与 AIST++ 的前提是一个舞者
一段动捕;本语料多数不是。这不使它不可用,但任何与论文数字的对比都必须带着这三行。

## 7.12 阶段 B 的吞吐缺陷:worker 在悄悄死光,而所有监控都说它活着

**症状只有"慢"**:按每卡实测速率,6 卡 21 worker 该在 1,000 条/h 量级,实测 358。
两个互相矛盾的现象把方向指出来——**CPU 负载 43/256,6 张卡里 4 张 0%**,两头都没跑满;
而单条 17 秒的 clip 已经跑了 5 小时 32 分。

按 wall / CPU 时间分开数:**21 个 worker 有 15 个卡死**(状态 S,CPU 占用 0.8–6.2%,
wall 1.2–5.5 h),`py-spy` 打出来 15 个**同一个栈、零产物**。

根因在 `hmr4d/utils/preproc/slam.py`:读帧器是独立 `Process`,逐帧经 `Queue` 传一个
**torch tensor**(intrinsics)。torch 默认 `file_descriptor` 策略把 storage 当文件
描述符发,消费者要回**生产者的** `resource_sharer` socket 去取。`video_stream` 末尾
用 `while not queue.empty()` 等消费者跟上,而 `queue.empty()` 是众所周知不可靠的判据
——它提前返回 True,读帧器函数返回、进入 atexit(resource_sharer 先被关掉),此时
消费者再取剩下的帧就报 `[Errno 2] No such file or directory`。**所以它永远炸在进度条
97% 附近。** 消费者抛异常后不再排空 → 生产者 feeder 线程堵在满管道上 → 双方各自
atexit join 对方,永远。

失败分类(全量日志):**VO 真发散 290 · 这个 bug 15 · tensor 尺寸不匹配 1**。
15 条,和卡死的 worker 数一样——**每触发一次就永久吃掉一个 worker**。

修法:`tools/run_gvhmr_extract.py` **模块级**一行
`torch.multiprocessing.set_sharing_strategy("file_system")`。storage 走命名 shm,
生命周期不绑在生产者的 socket 上。模块级不是随手放的:DPVO 在 `dpvo/dpvo.py:13`
import 时就 `set_start_method('spawn', True)`,读帧器因此**不继承**父进程的策略,而
spawn 的子进程会把 `sys.argv[0]` 当 `__mp_main__` 重新 import——模块作用域是唯一同时
够到两边的位置。`tests/test_run_gvhmr_extract.py` 钉住这一条,并把两个方向都测:
`file_system` 能在生产者退出中途取回,`file_descriptor` 抛 `FileNotFoundError`。

效果:GPU 0–6 从"4 张 0%"变成全部 68–99%,load 43 → 214,吞吐 **358 → 1,032 条/h**,
卡死 worker 0。剩余时长从 39–61 h 降到 **~13 h**。

**补丁不需要重启分片**(每条 clip 都是新进程),但 `run_gvhmr_ingest_shard.sh`
**不能在运行中改**——bash 按字节偏移边跑边读脚本,21 个副本正在跑时改它会让它们跳进
错位的位置。所以下面三条留到阶段 B 收工后一并做:

1. **`.extract_failed` 要记原因。** 现在是 `touch` 出来的空文件,只记"失败了"。原因只
   在每片 6.7 MB 的日志里,而日志会被下一次驱动覆盖。阶段 A 的规矩(失败入账本 + 异常
   类型与消息)没传到阶段 B;
2. **给 extract 加 `timeout`**,任何单条 clip 都不该能吃掉一个 worker;
3. **补一次 `ONLY=B` 扫描**:被这个 bug 打掉的 16 条已记进
   `runs/stage_b_wedged_clips.json` 并清掉了 `.extract_failed` 标记,但当前这一轮的
   worklist 已经走过它们,不补扫就永远不会重跑。290 条真发散的也值得同扫一次——
   DPVO 的 patch 深度是无种子的 `torch.rand_like`,重试不是确定性失败。

**存活监控这次是失效的**:它测"进程在不在",而一个 5 小时不动的 worker 和一个正常
worker 在 `ps` 里长得一样。判据应改成"最近有没有新产物"。**而吞吐本身就该是监控项**
——这个 bug 唯一的外部症状就是慢,没有任何东西会为"慢"报警。

## 7.13 工作区配额撞满,而它的第一个症状是"文件写出来是 0 字节"

2026-08-12 中午,`/workspace/<user>`(NAS,配额约 1 TB)写满。表现是
**创建成功、写入失败**:

```
runs/ 1KB           FAIL [Errno 122] Disk quota exceeded
logs/ 追加 1MB      FAIL
仓库根 1KB          FAIL
data/wild3d/ingest_v1_gvhmr_raw/<clip>/ 1MB   OK   <- 阶段 B 因此毫发无伤
/cache, /tmp        OK
```

**它是静默的**:`pathlib.write_text` 先 `open(w)` 建出 0 字节文件再写,写失败时文件
已经在那里了。所以 22 份 alignment 报告、两份日志全部是 0 字节而没有任何报错——
我是在前台重跑一次、不重定向 stderr,才看见 `Errno 122`。**凡是"产物存在但是空的",
先怀疑配额,不要怀疑工具。**

**阶段 B 没有受影响**(实测:12 小时全部 2–3 MB 正常产物,0 个 0 字节),但
**C–G 会死**:`fan_out` 的 `bash -c "$cmd" > logs/xxx.log` 一旦重定向失败,shard 就
非零退出,而驱动对非零退出的处理是 `return 1` 中止全链——这是 §7.5 第 6 条那次教训
的正确行为,但它会让整条链在一个与算法无关的原因上停住。

处置:**搬而不删**,沿用本仓 `data/atomic_aistpp -> /cache` 的既有模式。

| 目录 | 大小 | 处置 |
| --- | --- | --- |
| `runs/` | 54.25 GB / 16,079 文件 | → `/cache/atomicdance-assets/runs`,软链 |
| `data/aist_videos/` | 22 GB / 1,368 文件 | → `/cache/...`,软链(当天刚验证全量在 OSS) |
| `data/wild_videos_20260811/` | 87 GB | → `/cache/...`,软链(切分已完成,不再被读) |
| `logs/gvhmr_all_shard*.log` | 93 MB | 删(Aug 9 的旧 run,结论已入 worklog) |

共腾出 **231 GB**(`df` 1126 → 895 GB)。搬迁脚本逐文件核对大小后才删源目录,三次
搬迁的 mismatched 均为 0。

**没搬 `data/wild_ingest_v1`(409 GB)**:21 个阶段 B 分片正在读它。

**余量还要盯**:阶段 B 跑完 16,715 条会把 `ingest_v1_gvhmr_raw` 从 27 GB 推到约
77 GB,C–F 还要再写 bundle 与 S3D。现在的 129 GB 余量够,但**这条链没有磁盘监控**
——和吞吐一样,它属于"只有在坏掉之后才被发现"的那一类。
