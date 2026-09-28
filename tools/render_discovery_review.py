#!/usr/bin/env python3
"""One page for a person to check discovery on: captions and prototypes, with frames.

Every other page in ``output/`` asks whether a *cut* is in the right place.  This
one asks the two questions after it, which no gate in this repo can answer:

* **does the caption describe what the dancer actually does in that span**, and
* **do the segments inside one M2 prototype look like the same movement**.

Both are eyeball questions by construction.  M2's acceptance rate and cluster
dispersion say how tight the embedding is, not whether the thing it is tight
about is a movement; M3b's coverage floor counts captions, not their truth.

**Which generation this shows, and why it cannot show the other one.**  The
captions and prototypes here are ``clean5b4`` -- built on the pre-Phase-0 corpus
(2,101 clips / 15,115 segments, K=56).  Phase 0 replaced the corpus on
2026-08-20 and that generation is superseded.  Worse for this page: the 419
clips the fps re-cut touched had their old bytes **overwritten in the store**
when the corrected cut was pushed, so their old span boundaries no longer index
any video that exists.  Rendering them would pair one generation's frames with
another's boundaries -- the exact failure that produced an identical-looking
contact sheet on 2026-08-19.

So every example is drawn from the **1,682 clips the re-cut did not touch**, and
for those the two segmentations were measured to agree on **1,682 of 1,682**
clips, boundary for boundary.  What you see is therefore a caption and a
prototype that survive into the current corpus unchanged.  (On the re-cut clips
the two grids agree on only 197 of 317, and even those describe different
pixels -- which is why the re-run must exclude them **by name** rather than
trust the span key to differ.)

**Sampling is stratified and seeded, and both are printed.**  Drawing from the
whole corpus draws in proportion to clip count, so the largest account fills the
page.  Picking the example that looks best is the failure mode this guards
against, so the seed is written into the page and changing it is how you check
that the impression survives a different draw.

**Frames come from the object store, never from disk.**  ``asset_io.read_bytes``
resolves local-first and the ingest tree is parked under ``/cache``; a page that
read local-first would show whatever generation this pod happens to hold.

Usage::

    python3 tools/render_discovery_review.py \\
        --tag clean5b4 --segmentation runs/wild_v4_seg_beat4h/segmentation.json \\
        --exclude runs/clean5/freshness.json \\
        --caption-examples 8 --prototypes 6 --members 5 \\
        --output output/clean5_discovery_review.html
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import os
import pathlib
import random
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import asset_io, redo_manifest                                   # noqa: E402
from tools.render_segmentation_example import as_data_uri, decode, strip    # noqa: E402
from tools.run_wild_stage_g_m3a_oss import m3a_keys                         # noqa: E402
from tools.run_wild_stage_c_oss import repo_key                             # noqa: E402

OSSUTIL = "/opt/data-infra/ossutil64"


def fetch_clip(stem: str) -> Optional[pathlib.Path]:
    """The clip as the object store holds it -- what the boundaries describe."""
    handle, name = tempfile.mkstemp(suffix=".mp4")
    os.close(handle)
    target = pathlib.Path(name)
    url = "oss://" + asset_io.remote_path(
        "data/wild_ingest_v1/{}/clip.mp4".format(stem))
    argv = [OSSUTIL, "cp", url, str(target), "-f"]
    config = REPO / "ossutilconfig"
    if config.is_file():
        argv += ["-c", str(config)]
    done = subprocess.run(argv, capture_output=True, text=True)
    if done.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        return None
    return target


def clip_stem(recording_id: str) -> str:
    _, upload, clip = recording_id.split(":")
    return "{}__{}".format(upload, clip)


def load_captions(tag: str, parts_suffix: str = "") -> List[Dict]:
    """Every published caption row for a generation, from its parts.

    From the parts rather than ``captions.jsonl`` because ``merge`` has not been
    run for this generation -- and because the parts are what ``merge`` would
    read, so a page built on them cannot disagree with the release about what
    was captioned.
    """
    keys = m3a_keys(tag, parts_suffix)
    rows: List[Dict] = []
    for name in sorted(asset_io.list_prefix(keys["parts"])):
        if not name.endswith(".jsonl"):
            continue
        rows.extend(asset_io.read_jsonl(repo_key(keys["parts"], name)))
    return rows


def render_strip(video: pathlib.Path, spans, labels) -> Optional[str]:
    frames = decode(video)
    canvas = strip(frames, spans)
    if canvas is None:
        return None
    return as_data_uri(canvas)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="clean5b4")
    parser.add_argument("--parts-suffix", default="")
    parser.add_argument("--segmentation", required=True,
                        help="the grid the captions were keyed to, not the newest one")
    parser.add_argument("--group-keys", default="runs/wild_v4_group_keys.json")
    parser.add_argument("--exclude", default=None,
                        help="a freshness audit; its clips are dropped because "
                             "their old bytes no longer exist in the store")
    parser.add_argument("--caption-examples", type=int, default=8)
    parser.add_argument("--prototypes", type=int, default=6)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--phase0", default="runs/clean5/corpus.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    grid = json.loads(pathlib.Path(args.segmentation).read_text(encoding="utf-8"))
    known = {record["sequence"] for record in grid["records"]}
    accounts = json.loads(pathlib.Path(args.group_keys).read_text(encoding="utf-8"))

    dropped: set = set()
    if args.exclude:
        for stage in redo_manifest.STAGES:
            dropped |= set(redo_manifest.load(args.exclude, stage))

    rows = load_captions(args.tag, args.parts_suffix)
    eligible = [row for row in rows
                if clip_stem(row["recording_id"]) in known
                and clip_stem(row["recording_id"]) not in dropped]
    print("captions {} -> eligible {} ({} dropped as re-cut, {} not in this grid)".format(
        len(rows), len(eligible),
        sum(1 for r in rows if clip_stem(r["recording_id"]) in dropped),
        sum(1 for r in rows if clip_stem(r["recording_id"]) not in known)), flush=True)

    def account_of(row):
        return accounts.get(row["recording_id"].split(":")[1], "?")

    # -- caption examples: one per account, then round-robin ------------------
    by_account: Dict[str, List[Dict]] = collections.defaultdict(list)
    for row in eligible:
        by_account[account_of(row)].append(row)
    for pool in by_account.values():
        rng.shuffle(pool)
    picks: List[Dict] = []
    names = sorted(by_account)
    while len(picks) < args.caption_examples and any(by_account[n] for n in names):
        for name in names:
            if by_account[name] and len(picks) < args.caption_examples:
                picks.append(by_account[name].pop())

    # -- prototype examples: prototypes with members in many distinct clips ---
    by_prototype: Dict[int, List[Dict]] = collections.defaultdict(list)
    for row in eligible:
        by_prototype[int(row["prototype"])].append(row)
    ranked = sorted(by_prototype,
                    key=lambda p: -len({r["recording_id"] for r in by_prototype[p]}))
    chosen = ranked[: args.prototypes * 3]
    rng.shuffle(chosen)
    chosen = sorted(chosen[: args.prototypes])

    # -- one fetch+decode per clip, shared by every span it appears in --------
    wanted: Dict[str, List] = collections.defaultdict(list)
    for row in picks:
        wanted[clip_stem(row["recording_id"])].append(row)
    members: Dict[int, List[Dict]] = {}
    for prototype in chosen:
        pool = by_prototype[prototype]
        seen, taken = set(), []
        rng.shuffle(pool)
        for row in pool:
            if row["recording_id"] in seen:
                continue
            seen.add(row["recording_id"])
            taken.append(row)
            if len(taken) >= args.members:
                break
        members[prototype] = taken
        for row in taken:
            wanted[clip_stem(row["recording_id"])].append(row)

    print("fetching {} clip(s) for {} caption example(s) and {} prototype(s)".format(
        len(wanted), len(picks), len(chosen)), flush=True)
    images: Dict[tuple, str] = {}
    for index, (stem, rows_here) in enumerate(sorted(wanted.items()), 1):
        video = fetch_clip(stem)
        if video is None:
            print("  no video for {}".format(stem), flush=True)
            continue
        try:
            frames = decode(video)
            for row in rows_here:
                canvas = strip(frames, [(int(row["start"]), int(row["end"]))])
                if canvas is not None:
                    images[(row["recording_id"], row["start"], row["end"])] = \
                        as_data_uri(canvas)
        finally:
            video.unlink(missing_ok=True)
        if index % 5 == 0:
            print("  {}/{} clips".format(index, len(wanted)), flush=True)

    phase0 = {}
    if args.phase0 and pathlib.Path(args.phase0).is_file():
        phase0 = json.loads(pathlib.Path(args.phase0).read_text(encoding="utf-8"))

    page = build_page(args, picks, chosen, members, images, eligible, phase0,
                      account_of, len(rows), len(dropped))
    target = pathlib.Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page, encoding="utf-8")
    print("wrote {} ({:.1f} MB)".format(target, target.stat().st_size / 1e6))
    return 0


def caption_vocabulary(rows) -> Dict[str, object]:
    """How many different things the captioner actually said.

    Counted rather than eyeballed because the failure this looks for is
    invisible in a sample of eight: a template with a few slots produces
    captions that each read fine and that collectively carry very little.  The
    verb slot is parsed with the same pattern the captioner writes, so a
    caption that does not match it is reported as ``(other)`` rather than
    silently dropped.
    """
    import re

    captions = [row.get("caption", "") for row in rows]
    summaries = [row.get("summary", "") for row in rows]
    pattern = re.compile(r"a person performs an? ([a-z\- ]+?) with")
    verbs = collections.Counter()
    for text in captions:
        found = pattern.match(text)
        verbs[found.group(1) if found else "(other)"] += 1
    total = max(1, len(captions))
    return {
        "rows": len(captions),
        "distinct_captions": len(set(captions)),
        "caption_share": len(set(captions)) / total,
        "distinct_summaries": len(set(summaries)),
        "summary_share": len(set(summaries)) / total,
        "distinct_verbs": len(verbs),
        "verbs": [(verb, count / total) for verb, count in verbs.most_common(10)],
        "pose_share": verbs.get("pose", 0) / total,
    }


def _esc(value) -> str:
    return html.escape(str(value))


def build_page(args, picks, chosen, members, images, eligible, phase0,
               account_of, total_rows, dropped_clips) -> str:
    out: List[str] = []
    add = out.append
    add("<title>clean5 发现层复核</title>")
    add(STYLE)
    add("<div class=wrap>")
    add("<div class=eyebrow>{} · seed {}</div>".format(_esc(args.tag), _esc(args.seed)))
    add("<h1>clean5 发现层复核</h1>")
    add("<p class=lede>两个问题,本仓没有任何一道闸能回答:<b>caption 描述的是不是这个"
        "舞者实际做的动作</b>,以及<b>同一个 prototype 里的段看起来是不是同一个动作</b>。"
        "M2 的接受率与类离散度说的是嵌入有多紧,不是它紧在一个动作上;M3b 的覆盖率闸数的"
        "是 caption 的条数,不是它们的真假。所以这一页是给眼睛用的。</p>")

    add("<div class=note><b>先读这条:这里展示的是已作废的那一代。</b> "
        "caption 与 prototype 都来自 <code>{}</code>,建在 Phase 0 之前的语料上"
        "(2,101 clip / 15,115 段,K=56)。而且例子<b>只从 fps 重切没有碰过的 1,682 条 "
        "clip</b> 里抽:被重切那批的旧字节在推送修正剪辑时已被覆盖,它们的旧切点索引的"
        "视频不再存在,渲染出来会是&ldquo;一代的画面配另一代的切点&rdquo;—— 2026-08-19 那张两行完全"
        "相同的分镜图就是这么来的。在没被重切的那批上,新旧两份拍网格实测<b>逐条相同"
        "(1,682 / 1,682)</b>,所以你看到的东西原样活在当前语料里。"
        "抽样按账号分层,随机种子 <code>{}</code> —— 换个种子重跑,是检验这个印象站不站得住"
        "的办法。</div>".format(_esc(args.tag), _esc(args.seed)))

    if phase0:
        totals = phase0.get("totals", {})
        add("<h2><span class=n>1</span>Phase 0 交付了什么</h2><div class=rule></div>")
        add("<div class=scroll><table><tr><th>量</th><th>之前</th><th>现在</th>"
            "<th>这个数从哪来</th></tr>")
        for label, before, now, source in [
            ("clip", "2,103", "{:,}".format(totals.get("clips", 0)),
             "runs/clean5/corpus.json"),
            ("段", "15,115", "{:,}".format(totals.get("segments", 0)),
             "runs/clean5b5_seg/segmentation.json"),
            ("素材时长", "10.10 h", "{} h".format(totals.get("hours")),
             "motion_frames 之和 ÷ 30"),
            ("段内时长", "8.42 h", "{} h".format(totals.get("segmented_hours")),
             "拍网格不覆盖前奏与尾巴"),
            ("K=100 时段/类", "151", str(totals.get("segments_per_prototype_at_k100")),
             "论文 268.57"),
        ]:
            add("<tr><td>{}</td><td class=num>{}</td><td class='num hi'>{}</td>"
                "<td class=src>{}</td></tr>".format(
                    _esc(label), _esc(before), _esc(now), _esc(source)))
        add("</table></div>")
        add("<div class=gate><b>三道闸的读数,以及它们能失败这件事。</b> "
            "freshness <code>stale=0 missing=0</code>(1,999 / 1,999)—— 同一把闸在 stage B "
            "之前读 419、之后读 102。 "
            "segmentation-vs-features <code>0 clip(s) disagree</code>,<b>不带</b> "
            "<code>--allow-stale-segmentation</code> —— 它几分钟前对旧分割读 201。 "
            "语料清单 <code>1,059 + 16 == 1,075</code>,residue 0。</div>")
        by_account = phase0.get("by_account", {})
        if by_account:
            add("<div class=scroll><table><tr><th>账号</th><th>独舞率</th><th>clip</th>"
                "<th>段</th><th>小时</th></tr>")
            for name, row in sorted(by_account.items(),
                                    key=lambda kv: -kv[1]["clips"]):
                add("<tr><td>{}</td><td class=num>{}</td><td class=num>{}</td>"
                    "<td class=num>{}</td><td class=num>{}</td></tr>".format(
                        _esc(name), row["solo_share"], row["clips"],
                        row["segments"], row["hours"]))
            add("</table></div>")

    add("<h2><span class=n>2</span>Caption 例子</h2><div class=rule></div>")
    add("<p class=lede>一行一个段:在段内均匀取八帧,然后是 VLM 写下的东西。"
        "已发布的 {:,} 条 caption 里 {:,} 条可用,{} 条 clip 因被重切而排除。</p>".format(
            total_rows, len(eligible), dropped_clips))
    for row in picks:
        add(caption_card(row, images, account_of))

    vocab = caption_vocabulary(eligible)
    add("<h2><span class=n>3</span>同一个问题,放到语料尺度上</h2><div class=rule></div>")
    add("<p class=lede>八个例子就是八个例子。下面是把同一批 caption 数了一遍 —— "
        "模板化这种缺陷在八条抽样里是看不见的,每一条读起来都没问题,而它们合起来"
        "承载不了多少东西。</p>")
    add("<div class=scroll><table><tr><th>量</th><th>值</th><th>读法</th></tr>")
    add("<tr><td>caption 条数</td><td class=num>{:,}</td><td class=src>"
        "可用的,即排除重切 clip 之后</td></tr>".format(vocab["rows"]))
    add("<tr><td>不同的 caption 字符串</td><td class='num hi'>{:,}({:.1%})</td>"
        "<td class=src>VLM 在一个很窄的模板里写</td></tr>".format(
            vocab["distinct_captions"], vocab["caption_share"]))
    add("<tr><td>不同的 summary</td><td class=num>{:,}({:.1%})</td>"
        "<td class=src>自由文本那一半宽一些</td></tr>".format(
            vocab["distinct_summaries"], vocab["summary_share"]))
    add("<tr><td>动词槽的取值数</td><td class='num hi'>{}</td><td class=src>"
        "&ldquo;a person performs a <i>X</i> with &hellip;&rdquo; 里的 X</td></tr>".format(
            vocab["distinct_verbs"]))
    add("</table></div>")
    add("<div class=scroll><table><tr><th>动词</th><th>占全部 caption 的比例</th></tr>")
    for verb, share in vocab["verbs"]:
        width = max(1, round(share * 100 * 4))
        add("<tr><td class=mono>{}</td><td><span class=bar style='width:{}px'>"
            "</span><span class=num> {:.1%}</span></td></tr>".format(
                _esc(verb), width, share))
    add("</table></div>")
    add("<div class=gate><b>这个数是什么,不是什么。</b> "
        "论文的 atomic movement 是&ldquo;a complete motion process&hellip; a kick, which "
        "comprises the preparatory weight shift, leg extension, and recovery&rdquo;。"
        "<b>{:.0%} 的 caption 说舞者做的是一个 <i>pose</i></b> —— 一个静态构型,不是一个"
        "动作过程。<b>但这是关于 captioner 词表的陈述,不是&ldquo;段切错了&rdquo;的证明</b>:"
        "1.9 秒的四拍段确实可能只装得下一个保持的姿势。哪一种,数数是判不出来的,"
        "只有上面那些帧条能判 —— 这就是两者并排放在这一页的原因。</div>".format(
            vocab["pose_share"]))

    add("<h2><span class=n>4</span>Prototype 例子</h2><div class=rule></div>")
    add("<p class=lede>每一块是一个 M2 prototype(这一代 K=56)。成员<b>特意</b>取自"
        "<b>不同的 clip</b>:同一条 clip 的两个段像,是因为一堆与聚类无关的理由。"
        "要问的是一块里的各行是不是同一个动作。</p>")
    for prototype in chosen:
        rows_here = members[prototype]
        add("<div class=proto><h3>prototype {} <span class=dim>—— 展示 {} 个成员段,"
            "来自 {} 条 clip</span></h3>".format(
                prototype, len(rows_here),
                len({r["recording_id"] for r in rows_here})))
        for row in rows_here:
            add(caption_card(row, images, account_of, compact=True))
        add("</div>")

    add("<h2><span class=n>5</span>这一页上没有的东西</h2><div class=rule></div>")
    add("<ul class=missing>"
        "<li><b>M2 与 M3a 还没有在当前语料上重跑。</b>上面的一切都是已作废的那一代。</li>"
        "<li><b>M3b / M3c 本轮从没跑过</b> —— 还没有子类,也没有 LLM 蒸馏出来的词表。</li>"
        "<li><b>没有任何人工标注的边界。</b>本轮每一把分割尺子都是自造的,拍网格是靠眼睛"
        "选的。补一份 GT 是让这些判据能被判决的唯一办法。</li>"
        "<li><b>caption 行里没有视频哈希</b>,所以事后没有东西能分辨一条 caption 描述的是"
        "哪一版剪辑。这就是重跑必须<b>按名字</b>排除重切 clip、而不能指望 span key 对不上"
        "的原因 —— 实测那批里有 197 / 317 条切点碰巧完全相同。</li>"
        "</ul>")
    add("</div>")
    return "\n".join(out)


def caption_card(row, images, account_of, compact=False) -> str:
    key = (row["recording_id"], row["start"], row["end"])
    image = images.get(key)
    stem = clip_stem(row["recording_id"])
    seconds = (int(row["end"]) - int(row["start"])) / 30.0
    parts = ["<div class='card{}'>".format(" compact" if compact else "")]
    parts.append("<div class=meta><code>{}</code> · frames {}&ndash;{} · "
                 "{:.2f}s · prototype <b>{}</b> · {} · <span class=dim>{}</span>"
                 "</div>".format(
                     _esc(stem), row["start"], row["end"], seconds,
                     row.get("prototype"), _esc(account_of(row)),
                     _esc(row.get("split", ""))))
    if image:
        parts.append("<div class=framebox><img src='{}' alt='段内均匀取样的八帧'>"
                     "</div>".format(image))
    else:
        parts.append("<div class=missingimg>no frames &mdash; the store did not "
                     "serve this clip</div>")
    if row.get("summary"):
        parts.append("<div class=summary>{}</div>".format(_esc(row["summary"])))
    parts.append("<div class=caption>{}</div>".format(_esc(row.get("caption", ""))))
    if row.get("posescript_cue"):
        parts.append("<details><summary>PoseScript cue given to the VLM</summary>"
                     "<div class=cue>{}</div></details>".format(
                         _esc(row["posescript_cue"])))
    parts.append("</div>")
    return "".join(parts)


STYLE = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Public+Sans:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Palette taken from the repository's own contact sheets: strip() draws the
   cut marker between segment rows in (220,90,90), so that red is the page's
   accent and the chrome ties to the images instead of competing with them.
   Neutrals are biased cool so the dark video strips sit on them without a
   colour cast. Amber is reserved for one meaning: superseded. */
:root{
  --paper:#f4f6f8; --card:#ffffff; --sunk:#eceff3;
  --ink:#14161b; --muted:#5b6270; --faint:#8b93a1;
  --line:#dde2e9; --cut:#c0403f; --cut-soft:#f3e2e1; --amber:#96660f;
  --amber-soft:#f7edda;
  --serif:"Newsreader",ui-serif,Georgia,"Songti SC","Noto Serif CJK SC",serif;
  --sans:"Public Sans",-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",
    "Noto Sans CJK SC","Microsoft YaHei",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#101318; --card:#181c23; --sunk:#13171d;
    --ink:#e7eaf0; --muted:#98a1b0; --faint:#6e7787;
    --line:#262c36; --cut:#e0736f; --cut-soft:#2b1e1f; --amber:#d3a54a;
    --amber-soft:#2a2318;
  }
}
:root[data-theme="dark"]{
  --paper:#101318; --card:#181c23; --sunk:#13171d;
  --ink:#e7eaf0; --muted:#98a1b0; --faint:#6e7787;
  --line:#262c36; --cut:#e0736f; --cut-soft:#2b1e1f; --amber:#d3a54a;
  --amber-soft:#2a2318;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;
  font:400 15.5px/1.65 var(--sans);
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:3.5rem 1.5rem 6rem;
  display:flex;flex-direction:column;gap:0}
.eyebrow{font:500 11.5px/1 var(--mono);letter-spacing:.14em;
  text-transform:uppercase;color:var(--cut)}
h1{font:600 2.35rem/1.15 var(--serif);letter-spacing:-.015em;
  margin:.5rem 0 .6rem;text-wrap:balance}
h2{font:600 1.4rem/1.3 var(--serif);margin:3.5rem 0 .2rem;text-wrap:balance}
h2 .n{font:500 .8em/1 var(--mono);color:var(--cut);margin-right:.55rem}
h3{font:600 1.02rem/1.3 var(--sans);margin:0 0 .75rem;color:var(--ink)}
p{max-width:72ch;margin:.7rem 0}
.lede{color:var(--muted);max-width:70ch}
code,.mono{font:400 13px/1.5 var(--mono)}
code{background:var(--sunk);padding:.1em .38em;border-radius:3px}
.rule{height:1px;background:var(--line);margin:.55rem 0 1.4rem}
/* callouts: one meaning each */
.note,.gate{border-radius:5px;padding:.95rem 1.15rem;margin:1.3rem 0;
  max-width:76ch;font-size:14.5px}
.note{background:var(--amber-soft);border-left:3px solid var(--amber)}
.gate{background:var(--cut-soft);border-left:3px solid var(--cut)}
.note b,.gate b{font-weight:700}
/* tables */
.scroll{overflow-x:auto;margin:1.2rem 0}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:min(100%,520px)}
th{font:500 11px/1.4 var(--mono);letter-spacing:.09em;text-transform:uppercase;
  color:var(--faint);text-align:left;padding:.35rem .7rem;
  border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:.45rem .7rem;border-bottom:1px solid var(--line);vertical-align:top}
td.num{text-align:right;font-family:var(--mono);font-size:13px;
  font-variant-numeric:tabular-nums;white-space:nowrap}
td.hi{font-weight:700;color:var(--cut)}
td.src{color:var(--muted);font-size:13px}
tr:last-child td{border-bottom:none}
.bar{display:inline-block;height:9px;border-radius:1px;vertical-align:middle;
  background:var(--cut);opacity:.8;margin-right:.5rem}
/* cards */
.card{background:var(--card);border:1px solid var(--line);border-radius:7px;
  padding:1rem 1.15rem;margin:1.1rem 0;
  display:flex;flex-direction:column;gap:.55rem}
.card.compact{margin:.8rem 0;padding:.85rem .95rem}
.framebox{overflow-x:auto;background:#101014;border-radius:4px;
  padding:0;line-height:0}
.card img{display:block;height:auto;border-radius:4px;max-width:none}
.meta{font:400 12.5px/1.5 var(--mono);color:var(--muted);
  display:flex;flex-wrap:wrap;gap:.15rem .8rem;align-items:baseline}
.meta b{color:var(--cut);font-weight:500}
.summary{font:600 1.02rem/1.45 var(--sans)}
.caption{color:var(--muted);max-width:74ch}
.cue{color:var(--faint);font:400 12.5px/1.55 var(--mono);white-space:pre-wrap;
  margin-top:.45rem;max-width:74ch}
.missingimg{color:var(--faint);font-style:italic;padding:.8rem 0}
.proto{border:1px solid var(--line);border-left:3px solid var(--cut);
  border-radius:0 7px 7px 0;padding:1.1rem 1.2rem;margin:1.6rem 0;
  background:var(--sunk)}
.proto .card{background:var(--card)}
.dim{color:var(--faint);font-weight:400}
ul.missing{max-width:76ch;padding-left:1.1rem}
ul.missing li{margin:.5rem 0}
details summary{cursor:pointer;color:var(--faint);font:400 12.5px/1.5 var(--mono)}
details summary:focus-visible,a:focus-visible{outline:2px solid var(--cut);
  outline-offset:2px}
@media (prefers-reduced-motion:no-preference){
  .card{transition:border-color .15s ease}
}
</style>
"""


if __name__ == "__main__":
    raise SystemExit(main())
