# 音乐条件化：与论文对齐的 TODO

> 起因：M6 的每个数字都带着一句限定 ——「**整首歌条件化没实现**（150 帧分块）」。
> 这份文档把那句限定拆成可以逐条关掉的工作，每条带自己的判据和它的代价。
> 2026-08-15 建立。

## 1. 论文怎么做的（从上游论文原文读出）

Fig. 2 把两级的音乐输入画成不同的东西：

* **(1) Atomic Movements Planning** ← `Full Music Input` → **Full-music-awared Atomic Movement Planner**
* **(2) Dance Completion** ← `Periodic Music Input`

§3.2：

> ...a sequence of atomic movements—each with a specific category, temporal
> position, and duration **conditioned on full music instead of clips**—thereby
> bridging musical phrasing with choreographic structure.

机制（同节）：

> We first use a pretrained music encoder to encode the music sequence M into
> music features `c_music = Enc(M)`. The reverse process
> `p_θ(y_{t−1} | y_t, c_music, t)` is parameterized by a Transformer-based
> diffusion model conditioned on the music features.
> During inference, we start from a uniformly random sequence `y_T` and
> iteratively sample ... until t = 0. The resulting sequence
> `Ŷ = [ŷ_1, …, ŷ_F]` represents the planned atomic movements.

§3.1 定义 `F = L × R_F`，L 是整首音乐的长度。

**所以 planner 的输入和输出都是整首**：一次去噪出整首歌的标签序列。理由不是工程
偏好 —— planner 要负责的正是「哪个动作出现在歌的哪个位置」，而乐句结构（主歌 /
副歌 / drop）只在整首的尺度上存在。completion 那一级不需要它，论文给的是 periodic 输入。

## 2. 我们差在哪（三处，一处比一处深）

| # | 差异 | 出处 |
| --- | --- | --- |
| a | 模型的位置编码只有 150 | `model/atomic_planner.py:35` 默认 `max_seq_len=18000`，但 `train_atomic.py:405` 传的是 `max_seq_len=args.seq_len`，而 `--seq-len` 默认 150 |
| b | 音乐条件逐帧相加，作用域 = 窗口 | `atomic_planner.py:74` `x = x + music_projection(music_features)`，第 71 行强制与标签序列同长 |
| c | 推理按 150 帧**不重叠**切块，各块独立 | `infer_atomic.py:333` `for start in range(0, len(music), window_size)` |

**`WILD_ATOMIC_PIPELINE_PLAN.md` 第 59 行把 M4 标成「`max_seq_len=18000`，确为整首 ✅」
——这句读的是默认值，而真正的调用方把它覆盖了。** 与 `MIN_FREE_MIB`、`TARGET=12946`
是同一个形状：一个基于常数的断言，而实际调用者换掉了那个常数。

## 3. 切块伪影是实测存在的，不是推论

生成的 40 条序列（`runs/m6_songsplit630/motion`，630 类 planner）与真值在同一口径下：

| | 段边界数 | 落在 0 mod 150 | 均匀期望 | 倍数 | z |
| --- | ---: | ---: | ---: | ---: | ---: |
| **生成** | 408 | **84** | 2.7 | **30.9×** | **+49.4** |
| 真值（1,363 条录像）| 11,881 | 60 | 79.2 | 0.76× | −2.2 |

**每 5 个生成段边界里有 1 个是块边界。** 真值在同一统计下略低于随机，所以这不是分割器
的周期性，是我们切出来的。它同时说明「630 类 planner 每窗 9.88 段」这个数被伪影抬高了。

## 4. 语料能支持到哪一步（实测）

| | n | 中位 | p90 | 最长 |
| --- | ---: | ---: | ---: | ---: |
| AIST 录像（动作）| 1,363 | 285 帧 / 9.5 s | 915 / 30.5 s | 1,425 / 47.5 s |
| AIST 音乐 | 60 | **1,152 帧 / 38.4 s** | — | 1,890 / 63.0 s |
| wild_v4 序列 | 13,783 | **516 帧 / 17.2 s** | 664 / 22.1 s | 720 / 24.0 s |

两条要点：

1. **AIST 的音乐比它的动作录像长 4 倍。** 所以「整首音乐条件」在 AIST 上是可做的：
   条件是 1,152 帧的音轨，而计划的是这条录像覆盖的那一段。M6 已经在生成音乐长度的
   舞蹈（mBR2 = 1,189 帧）。
2. **两个语料都没有真正的「整首歌」**：最长 63 s。所以我们能做到的上限是
   「整条录像 / 整个音轨」，不是流行歌的 3–4 分钟。这一条要跟着任何结果一起报。

## 5. 三条工作

### W1 —— 推理改重叠窗 + 逐帧投票（不重训）

* 前提判据：§3 那张表。**已测，成立。**
* 改法：`infer_atomic.plan_labels` 从不重叠改成重叠 + 逐帧投票。completion 那边已有
  `_window_starts` / `_blend_weights`，planner 侧只有 `refine_plan`（M4b 的多数投票）。
  标签是离散的，所以融合走投票或 logit 平均，不走加权和。
* 代价：不重训。重跑 M6 约 10 分钟。
* 判据：0 mod 150 的超出量应当**掉回真值的量级**（0.76×，z≈0）；同时报 FID_k 与每窗段数。
  **两个方向都能失败** —— 如果超出量掉了而 FID 没动，说明伪影不影响头号指标，那也是结论。
* 限制：条件作用域**仍然是 5 秒**。W1 治的是边界，不是条件化。

### W3 —— 全曲条件向量（重训 planner，窗口不变）

**状态（2026-08-16）：已落地并在训。** `train_atomic.py --global-music` /
`--global-music-shuffle-seed`。摘要按 `windows.jsonl` 的逐窗帧范围**拼接**——不是对窗口
取平均，因为窗口重叠十层，平均它们会把中段加权成首尾的十倍，那是另一条轨的摘要。
拼接也让这条路径**不需要任何 stride 常数**（一个这里假设、那里改掉的 stride，本仓已吃过四次）。

**零假设的第一版是错的，而抓住它的是训练 loss，不是任何指标。** 第一版按*序列*错位排列；
诚实摘要在一首歌内基本共享（AIST val：228 条序列只有 30 个不同摘要），所以按序列打乱等于
给同一首歌的各条序列**各发一个不同的向量**——那是比被打乱对象**更细**的条件信号，
一个更好的序列 id。实测 epoch 100：诚实 0.4591 / 按序列打乱 **0.1691**（好拟合近三倍）。
**一个比它所否定的条件更容易拟合的零假设不是零假设，是第二个更强的条件。**
改成按**歌**错位排列（每首歌的序列接收另一首歌对应序列的摘要，组内按 rank 配对），
于是"一首歌的序列看到几个不同向量"这个粒度被保住：诚实 102 个不同摘要 / 打乱 100 个。
修正后 epoch 100 是诚实 0.4591 / 打乱 0.4649——差 1.3%，这才是零假设该有的样子。

* 改法：`AtomicPlannerTransformer` 增加一路整轨池化嵌入，广播到每一帧，与现有的逐帧
  音乐特征相加。窗口仍是 150，**样本数不变**（14,409），所以不碰 A 线那条样本饥饿。
* 代价：一次 planner 训练（约 2.5 h）+ M6。completion 不动。
* 判据（都能失败）：
  - `MM/Div`：当前 **0.098**，真值 0.745。全曲条件若带来「同一首歌可以有不同编排」，这个数应当升。
  - R-precision **排除同曲**：当前 lift **1.19**（无信号）。若音乐相似度开始对应动作相似度，应当升。
  - gate v2 `--plans`：不得掉出通过。
* **必须的对照**：全曲向量对一首歌是常数，所以它可以退化成「歌的 id」。split 已经是
  song-disjoint，所以在**没听过的歌**上的增益是真的；另外加一条零假设 —— 把全曲向量
  在歌之间**打乱**重测，若增益不变，那它测的不是音乐。

### W2 —— 长序列训练（开发在 AIST，测量在 wild）

* 改法：`--seq-len` 从 150 提到整条序列；`max_seq_len` 跟着走（默认 18000 本来就够）。
* 为什么测量放在 wild：

  | | 训练样本 | 序列中位 | 相对 150 窗 |
  | --- | ---: | ---: | ---: |
  | AIST | 911 条录像 | 285 帧 | 1.9× |
  | wild_v4 | ~9,600 条序列 | 516 帧 | 3.4× |

  **wild 在两个轴上都更好**：样本多 10 倍、序列长 1.8 倍。AIST 上把 14,409 个窗口
  换成 911 个样本，正是 A 线判否的那个机制（每类样本掉到 11）。
* 代价：注意力 O(n²)，720 帧对 150 帧是 23 倍；显存与步时都要重新标定。
* 依赖：**wild 的 release**（M3c → materialize → audit），现在还没有。
* 判据：与 W3 相同，外加一条 —— **它必须打得过 W3**，否则贵的那个没有理由。

## 6. 顺序与资源

```
现在（CPU，M3b 占着 7 张卡）   W1 的判据（已测完）→ W1 代码 → W3 代码
卡一空出来                     W3 训练 + M6 + 打乱对照
wild release 就绪之后           W2 在 wild 上训练 + M6，与 W3 并排
```

W1 与 W3 都不依赖 B 线，可以在 M3b/M3c 跑着的时候写完。W2 依赖 B 线，所以它的开发
（代码 + AIST 上的收敛性冒烟）先做，测量等语料。
