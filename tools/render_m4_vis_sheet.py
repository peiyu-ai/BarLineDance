#!/usr/bin/env python3
"""Build the M4 inspection sheet: generated pose video beside its ground truth.

Why this page has the shape it has
----------------------------------
A video wall on its own would be read wrong here, in three specific ways that
this page is built to prevent:

* **"The planner's output" is not one thing.**  ``infer_atomic`` fuses the
  per-window samples (``--plan-stride`` / ``--plan-fusion``) and then merges
  short segments before completion ever sees the plan.  On this checkpoint the
  repo's pinned ``vote`` setting collapses the plan to 67% transition, while
  ``centre`` lands near the ground-truth segment rate -- from the *same* model
  and the same seed.  Both are shown, with the plan statistics beside them,
  because quoting either alone would be quoting a fusion setting.
* **The oracle arm is the only thing that separates blame.**  Feeding the true
  label sequence through the same completion checkpoint says how much of the
  loss is the planner's and how much is retrieval + completion's.
* **Two caveats make every number here an upper bound**: this corpus's test
  split shares choreographers *and* backing tracks with train, and no audio is
  muxed, so music-motion sync cannot be judged from this page at all.
"""
import json
import pathlib
import pickle
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "output" / "clean5b5_m4_vis"
LABELS = pathlib.Path("/dev/shm/atomicdance-m3a-segmentation/ingroup_llm_v1")
ARMS = {
    "vote": REPO / "runs/vis_clean5b5_planner",
    "centre": REPO / "runs/vis_clean5b5_centre",
    "no-stride": REPO / "runs/vis_clean5b5_nostride",
}


def short(clip):
    return clip.replace("wild_v4:", "").replace(":clip", "_c")


def seg_stats(labels):
    labels = np.asarray(labels)
    n = 1 + int(np.count_nonzero(np.diff(labels)))
    return {"frames": int(labels.size), "segments": n,
            "frames_per_segment": float(labels.size / n),
            "transition_share": float((labels == 0).mean()),
            "distinct_classes": int(len({int(v) for v in labels} - {0}))}


def motion_energy(path):
    """Root-relative joint speed and pose spread.

    Root-relative because global translation belongs to the retrieval draft,
    not to the plan: a clip that travels would otherwise read as more "motion"
    than one that dances in place.  Neither number is a quality score -- a
    flailing clip scores high on both -- so both are only read as a *ratio
    against this clip's own ground truth*.
    """
    with open(path, "rb") as handle:
        pose = np.asarray(pickle.load(handle)["full_pose"], dtype=float)
    rel = pose - pose[:, :1, :]
    return (float(np.abs(np.diff(rel, axis=0)).mean()),
            float(rel.reshape(len(rel), -1).std(axis=0).mean()))


def gt_label_paths():
    out = {}
    for line in (LABELS / "labels.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        out[row["sequence_id"]] = row["labels_path"]
    return out


def main():
    clips = [c.strip() for c in (REPO / "runs/clean5b5_vis6.txt").read_text().split() if c.strip()]
    triple = [c.strip() for c in (REPO / "runs/clean5b5_vis3.txt").read_text().split() if c.strip()]
    gtp = gt_label_paths()

    def video(name, caption):
        if not (OUT / name).exists():
            return '<div class="cell missing">缺 {}</div>'.format(name)
        return ('<div class="cell"><video controls loop muted playsinline preload="metadata" '
                'src="{}"></video><div class="cap">{}</div></div>').format(name, caption)

    html = ['<meta charset="utf-8"><title>M4 生成抽检</title>', """
<style>
 body{font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
      margin:0 auto;max-width:1220px;padding:24px;color:#111;background:#fafafa}
 h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:34px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}
 .note{background:#fff;border:1px solid #e3e3e3;border-left:3px solid #b00;padding:12px 14px;margin:12px 0;border-radius:3px}
 .note.good{border-left-color:#0a7}
 .note b{color:#b00} .note.good b{color:#076}
 .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
 .cell{background:#fff;border:1px solid #e3e3e3;border-radius:3px;padding:6px}
 .cell.missing{color:#999;padding:24px;text-align:center}
 video{width:100%;display:block;background:#000;border-radius:2px}
 .cap{font-size:12px;color:#555;padding:5px 2px 2px}
 table{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin-top:6px}
 th,td{border:1px solid #e3e3e3;padding:6px 8px;text-align:right} th{background:#f2f2f2;text-align:left}
 td.l,th.l{text-align:left} tr.gt td{background:#f6faf7}
 .bad{color:#b00;font-weight:600}
 code{background:#f0f0f0;padding:1px 4px;border-radius:2px;font-size:12px}
 p.sub{color:#666;margin:4px 0 10px}
</style>"""]
    html.append("<h1>M4 生成抽检 — clean5b5 / s15 planner</h1>")
    html.append("<p class='sub'>planner <code>planner_v1A_s15/planner_step135648.pt</code>"
                "(stride-15 release,40,187 训练窗,216 epoch / 135,648 步)→ completion "
                "<code>completion_wild_v4_acct/completion_step134496.pt</code>,seed 20260822,"
                "6 条 test clip。</p>")
    html.append("""
<div class="note">
<b>看之前必须知道的三件事:</b>
<ol style="margin:6px 0 0 18px;padding:0">
<li><b>没有声音</b>(这批 clip 手上只有 35 维音乐特征,没有 wav),所以
    <b>“踩不踩得上拍”在这一页判不了</b>,只能判姿态与连贯性。</li>
<li><b>这个 test 不是干净的留出集</b>:5 个账号全部同时出现在 train/val/test;
    184 条 test 录像里 <b>38 条(20.7%)</b>与 train 有指纹确认的同曲(指纹还是下界)。</li>
<li><b>词表的天花板已经量过,没过闸</b>:同一套音乐编码器接一个普通逐帧分类器
    (比扩散严格更容易,所以是任何 planner 的上界)——train 0.9931 / <b>test 0.1311</b>,
    而“永远输出 transition”的 baseline 是 <b>0.2321</b>;把窗口配上别的歌只掉 0.0065。
    <b>music → 标签在这个词表上没学到</b>,下面看到的就是这句话的画面版。</li>
</ol></div>""")

    html.append("<h2>1. 整条 clip:两种融合设置的 planner vs 真值</h2>")
    html.append("<p class='sub'>左 = 仓库钉死的 <code>--plan-stride 15 --plan-fusion vote</code>;"
                "中 = 同一个模型同一个 seed,只把融合换成 <code>centre</code>;右 = 真值动作。"
                "<b>两个左边的差别不是模型的差别,是融合设置的差别</b> —— 见第 2 节。</p>")
    html.append('<div class="grid3">')
    for clip in clips:
        s = short(clip)
        html.append(video(s + "_planner.mp4", "planner · vote 融合 · " + s))
        html.append(video(s + "_centre.mp4", "planner · centre 融合 · " + s))
        html.append(video(s + "_gt.mp4", "真值 · " + s))
    html.append('</div>')

    html.append("<h2>2. 计划本身长什么样:融合设置把同一个模型读成两种东西</h2>")
    html.append("<p class='sub'>闭环里 <code>infer_atomic</code> 先融合重叠窗口、再合并过短段,"
                "然后 completion 才看到这份计划。所以下表是<b>后处理之后</b>的计划,"
                "不是 planner 的原始逐窗采样(那个是每 150 帧窗 10.7 段 / 13.5 帧每段)。</p>")
    html.append("<table><tr><th class='l'>arm</th><th>段数</th><th>帧/段</th>"
                "<th>transition 占比</th><th>用到的类数</th><th class='l'>n</th></tr>")
    agg = {}
    for arm, root in ARMS.items():
        vals = [seg_stats(pickle.load(open(root / (c + ".pkl"), "rb"))["atomic_labels"]) for c in clips]
        agg[arm] = vals
    agg["ground truth"] = [seg_stats(np.load(LABELS / gtp[c])) for c in clips]
    for arm in ("vote", "centre", "no-stride", "ground truth"):
        v = agg[arm]
        tr = float(np.mean([x["transition_share"] for x in v]))
        bad = ' class="bad"' if arm != "ground truth" and tr > 0.5 else ""
        html.append("<tr{}><td class='l'>{}</td><td>{:.1f}</td><td>{:.1f}</td><td{}>{:.0%}</td>"
                    "<td>{:.1f}</td><td class='l'>{}</td></tr>".format(
                        ' class="gt"' if arm == "ground truth" else "",
                        arm if arm != "ground truth" else "真值",
                        float(np.mean([x["segments"] for x in v])),
                        float(np.mean([x["frames_per_segment"] for x in v])),
                        bad, tr, float(np.mean([x["distinct_classes"] for x in v])), len(v)))
    html.append("</table>")
    html.append("""<div class="note"><b>这张表里最该看的一列是 transition。</b>
真值有 34% 的帧不属于任何原子动作;planner 的计划是 <b>54%–67%</b>。
<code>vote</code> 在 10 个重叠窗上做多数投票,而当逐窗采样本身接近先验时,
<b>投票投出来的就是先验 —— 也就是 transition</b>(两条 clip 直接塌成 1 个类 / 98% transition)。
所以视频里“动作发软、大部分时间在过渡”不是渲染问题,是计划里真的没有动作。</div>""")

    html.append("<h2>3. 同一个 5 秒窗口的三联:planner 计划 / 真值计划 / 真值动作</h2>")
    html.append("<p class='sub'>中间一列是 <b>ORACLE</b>:把<b>真实标签序列</b>喂给<b>同一个 completion</b>。"
                "它与右边的差 = 检索+completion 这一级的损失;它与左边的差 = planner 这一级的损失。"
                "<b>这是把“谁的锅”分开的唯一一组对照。</b></p>")
    html.append('<div class="grid3">')
    for clip in triple:
        s = short(clip)
        html.append(video(s + "_planner150.mp4", "planner 计划(5s) · " + s))
        html.append(video(s + "_oracle150.mp4", "ORACLE 真值计划(5s) · " + s))
        html.append(video(s + "_gt150.mp4", "真值动作(5s) · " + s))
    html.append('</div>')

    html.append("<h2>4. 动作能量:生成的动作比真值“少动”多少</h2>")
    html.append("<p class='sub'>去掉整体位移后的关节速度与姿态离散度。两者都不是质量分(乱挥也会高),"
                "只按<b>与本条 clip 自己的真值之比</b>读。</p>")
    html.append("<table><tr><th class='l'>口径</th><th>关节速度</th><th>占真值</th>"
                "<th>姿态离散</th><th>占真值</th><th class='l'>n</th></tr>")
    full = {"planner · vote": [motion_energy(ARMS["vote"] / (c + ".pkl")) for c in clips],
            "planner · centre": [motion_energy(ARMS["centre"] / (c + ".pkl")) for c in clips],
            "真值": [motion_energy(REPO / "runs/wild_v4_acct_gt_eval/motion" / (c + ".pkl")) for c in clips]}
    tri = {"planner 计划(5s)": [motion_energy(REPO / "runs/vis_clean5b5_planner150" / (c + ".pkl")) for c in triple],
           "ORACLE 真值计划(5s)": [motion_energy(REPO / "runs/vis_clean5b5_oracle150" / (c + ".pkl")) for c in triple],
           "真值动作(5s)": [motion_energy(REPO / "runs/vis_clean5b5_gt150" / (c + ".pkl")) for c in triple]}

    def block(groups, reference):
        ref = np.array(groups[reference])
        for name, vals in groups.items():
            v = np.array(vals)
            html.append("<tr{}><td class='l'>{}</td><td>{:.4f}</td><td>{:.0%}</td>"
                        "<td>{:.4f}</td><td>{:.0%}</td><td class='l'>{}</td></tr>".format(
                            ' class="gt"' if name == reference else "", name,
                            v[:, 0].mean(), v[:, 0].mean() / ref[:, 0].mean(),
                            v[:, 1].mean(), v[:, 1].mean() / ref[:, 1].mean(), len(v)))
    block(full, "真值")
    html.append("<tr><td colspan='6' style='background:#f7f7f7;height:6px;padding:0'></td></tr>")
    block(tri, "真值动作(5s)")
    html.append("</table>")
    html.append("""<div class="note good"><b>这是本页最重要的一行:</b>
把<b>真实标签序列</b>喂给<b>同一个 completion</b>(ORACLE),动作能量回到真值的 <b>81%</b>;
换成 planner 自己产的计划,只剩 <b>38%</b>,三条 clip 上排序一致(planner &lt; ORACLE &lt; 真值)。
<b>所以“动作发软、姿态趋于站立”的责任在 planner / 词表这一级,不在 completion 与检索。</b></div>""")

    html.append("<h2>5. 每条 clip 的计划数字(vote 融合,与视频一一对应)</h2>")
    html.append("<table><tr><th class='l'>clip</th><th class='l'>arm</th><th>帧</th><th>段</th>"
                "<th>帧/段</th><th>transition</th><th>类数</th></tr>")
    for i, clip in enumerate(clips):
        s = short(clip)
        for arm, key in (("planner · vote", "vote"), ("planner · centre", "centre"), ("真值", "ground truth")):
            st = agg[key][i]
            html.append("<tr{}><td class='l'>{}</td><td class='l'>{}</td><td>{}</td><td>{}</td>"
                        "<td>{:.1f}</td><td{}>{:.0%}</td><td>{}</td></tr>".format(
                            ' class="gt"' if key == "ground truth" else "", s, arm,
                            st["frames"], st["segments"], st["frames_per_segment"],
                            ' class="bad"' if key != "ground truth" and st["transition_share"] > 0.7 else "",
                            st["transition_share"], st["distinct_classes"]))
    html.append("</table>")

    target = OUT / "index.html"
    target.write_text("\n".join(html), encoding="utf-8")
    print("wrote", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
