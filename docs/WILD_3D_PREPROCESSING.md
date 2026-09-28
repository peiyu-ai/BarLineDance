# Wild 3D pose 预处理：相机与舞蹈动作解耦

## 设计决定

选择 [WHAM](https://github.com/yohanshin/WHAM) 作为第一版 backend adapter，因为其官方 custom-video 输出恰好包含：

```text
pose_world[T,72]  # 24 个 SMPL axis-angle，world global root
trans_world[T,3] # 人的 world root trajectory
slam_results[V,7]# 完整输入视频的 DPVO c2w，[tx,ty,tz,qx,qy,qz,qw]
```

这比仅输出 root-relative 3D joint 的 lifter 更适合 BarLineDance：它保留完整 24-joint SMPL 表示，并能把相机轨迹从人的 root motion 中剥离。WHAM 的代码为 MIT；其 SMPL body model、权重和数据仍有各自许可，使用前必须由数据/模型资产持有人确认授权。不要把这一步绕过成 `--estimate_local_only`，那会重新丢掉 world/camera 解耦。

必须明确一个边界：WHAM 只把 DPVO 的**角速度**用于人体推理，未把 DPVO 的 SE(3)/尺度注册到 `trans_world`。因此 `camera.npz` 保存的是原始、未注册的 visual-odometry gauge（平移 up-to-scale），仅用于审计/reprojection；它不是人体 z-up world，也绝不能被拼进 151D 训练特征。只有完成独立 gravity + SE(3)/Sim(3) registration 后，才能宣称两者共享坐标系。

`tools/preprocess_wild_3d.py` 不下载权重、不上传视频、也不修改 Lodge cache。它只做可审计的 inventory、WHAM queue、坐标/表示转换和验证。

## 不接受的“无 SMPL”捷径

当前没有一个本地可运行、同时满足 `24 × rotation6d + world root translation +
camera/body separation` 的替代 backend。已有的 DPVO checkpoint 只能估计相机，不能
恢复 SMPL 人体姿态或 world root；2D-to-3D skeleton lifter 也无法可靠观测单目尺度和
全局平移，更不能直接产出上游方法所需的 24-joint rotation 表示。它们最多可继续
作为 2D QC/诊断，绝不能被硬转为 151D 训练“真值”。

[GVHMR](https://github.com/zju3dv/GVHMR)、[TRAM](https://github.com/yufu-wang/tram)
和 [WHAC](https://github.com/MotrixLab/WHAC) 是以后可单独评估的 world-HMR 后备方案，
但其官方安装同样要求注册/授权的 SMPL（或 SMPL-X）body assets 和额外 HMR/checkpoint；
它们不构成绕过当前授权资产的路径。因此本版本保持 WHAM 为唯一可发布 backend，并在
preflight 未通过时 fail-closed。

本地已固定 WHAM code checkout 为 `third_party/WHAM` 的
`2b54f7797391c94876848b905ed875b154c4a295`（含其 DPVO/ViTPose submodule）；
该目录被 git 忽略，因为它及其模型是运行时第三方依赖。先做只读 preflight：

```bash
git clone --recursive https://github.com/yohanshin/WHAM.git third_party/WHAM
git -C third_party/WHAM checkout 2b54f7797391c94876848b905ed875b154c4a295
git -C third_party/WHAM submodule update --init --recursive
```

```bash
python tools/preprocess_wild_3d.py preflight-wham \
  --wham-root third_party/WHAM \
  --python <WHAM_ENV_PYTHON> \
  --output data/wild3d/reports/wham_preflight.json
```

预期在未安装资产时报告缺少 SMPL/辅助 body assets 和 WHAM/ViTPose/DPVO
checkpoints。不要自动运行 WHAM 的 `fetch_demo_data.sh`：它要求持有人提供
SMPLify/SMPL 注册凭据。由有授权的环境安装后再次 preflight，`ready: true`
才可执行队列。

当前机器只有 Python 3.12 / PyTorch 2.8 的 Blackwell (`sm_120`) 环境；WHAM 上游
说明的 Python 3.9 / PyTorch 1.11 / CUDA 11.3 组合不能被假定为兼容。应由资产持有人
提供已验证、含 SMPL 与全部 checkpoint 的 WHAM Python，并把它的绝对路径传给下列
命令（或通过 `WHAM_PYTHON` 覆盖生成脚本的默认值）。

## 当前可启动范围

首批只处理可从 2D cache 回溯到真实视频的 b2gpu 子集，避免把失联
source 的 cache 误排入 HMR 队列：

```text
/workspace/<user>/e2e/Lodge/data/in_the_wild/tiktok_oss_dl_b2gpu
  2,849 个原视频
    → /workspace/<user>/e2e/Lodge/data/in_the_wild/tiktok_new_segments_b2gpu
      7,058 个约 16 秒切片
        → /workspace/<user>/e2e/Lodge/data/in_the_wild/tiktok_new_b2gpu
          6,052 个完整 2D pose cache
```

其它 wild cache 仍可做 2D inventory，但目前 `meta.source` 指向不可用的
segment 根，不能安全运行 WHAM。第一条已验证可作为 smoke 的记录是
`6933985164326489359__clip001`：480 帧/30 FPS，720×1280，2D 可见关节比例
0.9848、median score 0.8252、冻结比例 0.00046。其 manifest 已可生成；它
是 HMR 输入候选，不代表已经获得或通过 3D 结果。

## 输出格式

每个成功转换的 clip 保存为独立目录：

```text
wild3d/<clip_id>/
  atomic_motion_151.npy       [T,151]  未归一化，上游论文/EDGE z-up 表示
  pose_axis_angle_z_up.npy    [T,24,3]
  rotation6d_z_up.npy         [T,24,6]
  body_rotation6d_local.npy   [T,23,6]
  root_rotation6d_world.npy   [T,6]
  root_translation_z_up.npy   [T,3]
  contacts.npy                [T,4]   L ankle, R ankle, L toe, R toe
  contact_valid_mask.npy      [T,4]   frame-gap 边界处为 false；不可作监督
  frame_ids.npy               [T]
  camera.npz                  必有；原始 DPVO c2w（未注册 gauge，仅审计）
  quality.json
  metadata.json
```

主表示严格为：

```text
atomic_motion_151 = [4 contacts, 3 root_translation_z_up, 24 × rotation6d]
```

WHAM 人体是 y-up；EDGE/上游代码的 AIST++ 预处理是 z-up。转换固定做 `+90°` about X，即 `(x,y,z) → (x,-z,y)`，并对全局 root orientation 左乘同一变换。body local rotation、**未注册 DPVO camera trajectory**、原始 2D observations 均独立保存。DPVO 不做 y-up/z-up 转换，不能被折叠到人体 world 或 image-space pose 文件里。

## 第一次运行：先 inventory，再跑一个世界坐标 smoke

以下示例的 `<LODGE_WILD_CACHE>` 应指向 Lodge 的**已有** cache root（每个子目录含 `meta.json` 和 `keypoints*.npy`）。inventory 是幂等、只读 Lodge 数据的：

```bash
cd <仓库根目录>

python tools/preprocess_wild_3d.py inventory \
  --cache-root <LODGE_WILD_CACHE> \
  --recursive \
  --output data/wild3d/manifests/wild_inventory.jsonl
```

本机首批可直接运行的 inventory 命令为：

```bash
python tools/preprocess_wild_3d.py inventory \
  --cache-root /workspace/<user>/e2e/Lodge/data/in_the_wild/tiktok_new_b2gpu \
  --output data/wild3d/manifests/tiktok_new_b2gpu.jsonl
```

它会生成 JSONL 与 `.summary.json`；只有 `ready_for_wham` 进入下一步。`quarantine` 不是删除，而是需要人工定位的质量问题。

在 HMR 前也必须先冻结“原视频 → clip sequence”的身份关系，不能让
`__clipNNN` 变成 split 或 prototype exclusion 的单位。可将 inventory 物化为
pre-HMR staging manifest：

```bash
python tools/preprocess_wild_3d.py build-wild-staging-manifest \
  --inventory data/wild3d/manifests/tiktok_new_b2gpu.jsonl \
  --output-dir data/wild3d/source_manifests/tiktok_new_b2gpu \
  --corpus tiktok \
  --split-seed 20260805
```

它写出 `sources.jsonl`（`recording_id` 是 split 身份，wild 默认令
`retrieval_group_id=recording_id`；若发现跨录制近重复/同表演，必须合并为更粗的 group）、
`sequences.jsonl`（每个 clip）和 `split_candidates_v1.json`。本机当前产物覆盖
2,846 个录制、6,041 条 ready sequence，按原视频而非 clip 进行 80/10/10 的
确定性候选划分。该 split 明确标为 provisional：未完成近重复内容审计、3D QC 和
标签前，任何条目都不能进入训练。完整字段语义见
[SOURCE_MANIFEST_CONTRACT.md](SOURCE_MANIFEST_CONTRACT.md)。

已对这份 staging 的 2,846 个 raw recording 完成一次全量**精确字节** SHA-256
审计，并发布了
`data/wild3d/source_manifests/tiktok_new_b2gpu/exact_content_merged_v1`：未观察到
跨 split 的 byte-identical 文件，6,041 条 sequence 已继承 source content hash。该结果
不是 near-duplicate QC；merge 后全部仍为
`provisional_pending_near_duplicate_content_qc` 和
`accepted_for_training=false`。精确审计的重跑命令、后续 normalizer / labels / window
发布顺序见 [TRAINING_DATA_RELEASE.md](TRAINING_DATA_RELEASE.md)。

在取得并安装 WHAM、DPVO、SMPL 和官方模型权重的独立环境后，先生成不执行的队列：

```bash
python tools/preprocess_wild_3d.py queue-wham \
  --manifest data/wild3d/manifests/wild_inventory.jsonl \
  --wham-root <WHAM_CHECKOUT> \
  --result-root data/wild3d/wham_raw \
  --output data/wild3d/run_wham_candidates.sh \
  --video-root <RAW_OR_SEGMENTED_VIDEO_ROOT> \
  --python <WHAM_ENV_PYTHON>
```

审阅生成脚本，先执行一行或一个小 clip：

```bash
WHAM_PYTHON=<WHAM_ENV_PYTHON> \
  bash data/wild3d/run_wham_candidates.sh
```

本 checkout 还已生成通过 2D QC 的 `6933985164326489359__clip001` 单条 smoke
脚本：在已授权、已验证的 WHAM 环境中可直接执行：

```bash
WHAM_PYTHON=<WHAM_ENV_PYTHON> CUDA_VISIBLE_DEVICES=0 \
  bash data/wild3d/run_wham_smoke.sh
```

队列会在 WHAM checkout 内启动 `demo.py`（其配置使用相对路径），并检查实际
的 `<result>/<clip>/<video>/wham_output.pkl`。每次新任务会在同一目录写入
`wild3d_wham_global_run.json`：它记录 fresh cache、未使用 `--estimate_local_only`
以及预期输出路径。已有结果**只有同时带这个 marker 才会跳过**；缺 marker 的
历史结果会硬失败，避免把旧 local-only 产物误收进语料。若 partial output 已有
`tracking_results.pth` 或 `slam_results.pth`，队列也会拒绝自动复用或删除，须先人工
审计/清理。它还会在任务开始前显式 import DPVO/SLAM 与 WHAM 的 detector/
feature extractor；若该 world runtime 未正确安装，脚本会失败而不是让 WHAM
静默退化为 camera-local 结果。
所有生成的 smoke、单队列和 shard launcher 都先以**同一个** `WHAM_PYTHON` 做完整
asset + world-runtime preflight；`ready != true` 时会在创建 marker、输出目录或调用
`demo.py` 前退出。这样缺 checkpoint 时不会让 WHAM 随机初始化并留下伪结果。
`--max-tasks 1` 可用来先只生成一个 smoke 队列。不要向命令加入
`--estimate_local_only`。

全量 b2gpu 任务可按 GPU 分片，而不需要修改原始 manifest。以下例子生成七个
独立、可恢复的 shard；实际 GPU 编号由启动时的 `GPU_IDS` 显式指定，避免占用
正在被其他作业使用的卡：

```bash
python tools/preprocess_wild_3d.py queue-wham-shards \
  --manifest data/wild3d/manifests/tiktok_new_b2gpu.jsonl \
  --wham-root third_party/WHAM \
  --result-root data/wild3d/wham_raw \
  --output-dir data/wild3d/queues/tiktok_new_b2gpu \
  --shards 7 \
  --video-root /workspace/<user>/e2e/Lodge/data/in_the_wild/tiktok_new_segments_b2gpu \
  --python <WHAM_ENV_PYTHON>

WHAM_PYTHON=<WHAM_ENV_PYTHON> GPU_IDS=0,1,2,3,4,5,6 \
  bash data/wild3d/queues/tiktok_new_b2gpu/launch_wham_shards.sh
```

launcher 与每个 shard 都在任何任务前做完整 preflight；只有带对应 fresh-global
marker 的非空 `wham_output.pkl` 会跳过，因此中断后可使用同一 launcher 恢复，
但不会误复用无 provenance 的旧结果。

WHAM 自己把输出写在 `<result-root>/<clip_id>/<video_stem>/wham_output.pkl`；同目录的 `slam_results.pth` 也必须保留。随后转换：

```bash
python tools/preprocess_wild_3d.py convert-wham \
  --wham-output <.../wham_output.pkl> \
  --slam <.../slam_results.pth> \
  --run-provenance <.../wild3d_wham_global_run.json> \
  --source-cache <LODGE_WILD_CACHE>/<clip_id> \
  --source-video <SEGMENTED_VIDEO> \
  --output-dir data/wild3d/converted/<clip_id>

python tools/preprocess_wild_3d.py validate \
  --output-dir data/wild3d/converted/<clip_id>

python tools/preprocess_wild_3d.py reconcile-wild-hmr \
  --staging-sequences data/wild3d/source_manifests/tiktok_new_b2gpu/exact_content_merged_v1/sequences.jsonl \
  --converted-root data/wild3d/converted \
  --output data/wild3d/source_manifests/tiktok_new_b2gpu/hmr_reconciled_v1/sequences.jsonl
```

这里故意以 `exact_content_merged_v1/sequences.jsonl` 为 staging 输入，让 HMR 输出继承
已审计的 content/group 身份；不要回到旧的未合并 `sequences.jsonl`。输出目录必须是新的
版本，不能回写 exact-audit bundle。

`--slam` 与 `--run-provenance` 都是必填项；若 WHAM 找到多个 person track，`convert-wham` 也会拒绝猜测，必须显式加 `--person-id`。转换会用 track 的严格递增 `frame_ids` 从完整视频 `slam_results` 取相机行；没有覆盖这些 index 的 cropped 相机文件会被拒绝。frame ID 有缺口的产物可保留供排查，但 `validate` 会拒绝其进入 contact/motion 训练。这正是防止 2D pipeline 中帧间最高分人物切换被带进 3D corpus 的保护。

最后一条 `reconcile-wild-hmr` 不会把“WHAM 命令结束”当作入训许可：它将每个
`sequence_id` 标为 `pending`、`quarantine` 或经过严格 validate 的 `candidate`，并写入
151D/camera/frame-ID 资产路径。`candidate` 仍需 duplicate-content、人工/3D QC 与
atomic label gate；因此相机或失败 HMR 不会通过 manifest 的缺省值泄漏到训练数据。

只有在上述 post-HMR manifest 中出现已验证的 `hmr_status: candidate` 后，才物化模型
所需的音乐特征。不要复用 Lodge cache 中的 39-D `music_rhythm*.npy`：上游代码的
released baseline 是 30 FPS、35-D 特征。下面命令对每条 candidate 从同一 source cache
的 `audio.wav` 调用该 baseline extractor，并以转换结果的 `frame_ids.npy` 精确取行：

```bash
python tools/extract_wild_music_features.py \
  --input-sequences data/wild3d/source_manifests/tiktok_new_b2gpu/hmr_reconciled_v1/sequences.jsonl \
  --output-dir data/wild3d/source_manifests/tiktok_new_b2gpu/audio_35_v1
```

输出 bundle 是新的不可覆盖目录，其中 `sequences_audio.jsonl` 保留全部 input row，
`music_35/<sha256(sequence_id)>.npy` 是成功 candidate 的 `[T,35]` 特征，且 manifest
写入 source/audio/artifact SHA-256、extractor 版本和 frame-selection provenance。它不
padding、插值或静音回退；`audio.wav` 缺失、特征行不够、frame ID 不连续或 source FPS
不是精确 30 时，该 candidate 会变为 `audio_feature_status: quarantine` 并保留原因码。
HMR 尚未完成的 `pending` row 原样透传。即使音频成功，`accepted_for_training` 仍为 false；
它只通过了音频对齐门，尚未通过后续 QC/labels/frozen split。

本机已有一份 merge 前 staging 派生的 `audio_35_v1`：其 6,041 条输入均为 HMR `pending`，
故摘要为 `candidate: 0`、`not_attempted: 6041`、`quarantine: 0`。它只是 pending 透传的
历史审计产物，不能与新的 content-merged HMR release 混用；待有转换后的 candidate 时应以上述
content-merged HMR manifest 发布一个**新的**版本目录，绝不覆盖旧 bundle。

## 质量门控

自动阶段只给出 `candidate`，不自动宣称训练可用。每批至少审查：

1. 原视频与 2D detector / tracker overlay：单人、无 ID switch、无镜头切换跨段。
2. camera trajectory：无显著 NaN、突跳或强烈 zoom/VO failure；它是未注册、up-to-scale 的 VO gauge，不能作为人体尺度监督，也不能直接与人体 root 作差。
3. 3D global render：检查脚、root travel、朝向的内在合理性；相机/人体共同坐标的 reprojection 要等完成显式 registration 后再计算。
4. 数值：骨长、rotation/root speed、contact rate、frame alignment 与音频帧数。
5. reprojection：以保存的 2D raw/clean、camera intrinsics/extrinsics 和 SMPL 重投影计算误差；这是将 candidate 提升为 `accepted` 的硬门槛，第一版 adapter 不假造该数字。

任何 clip 失败时只移动 manifest 状态到 `quarantine/reject`，不覆盖其 raw 2D cache 或 3D 原始输出。

## 进入 BarLineDance 训练之前

转换成功只有 3D corpus 的开始。官方 planner/completion 还要求 `atomic_labels[T]`；不能把无标签 wild motion 直接喂给 `train_atomic.py`。必须先根据 [FINETUNE_PLAN.md](FINETUNE_PLAN.md) 中的 AIST-train-only recognizer/segment alignment 建立高置信 pseudo labels，再建立 train-only prototype library 和 normalizer。

这条顺序保证：相机位置不会成为动作特征，wild 的噪声不会被误写成 atomic label，且最后评估不会依赖目标歌曲的 GT。
