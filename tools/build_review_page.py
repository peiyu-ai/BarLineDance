#!/usr/bin/env python3
"""Fold a review set rendered by ``tools/render_review_set.sh`` into one HTML.

Why base64 rather than links: ``tools/build_single_file_gallery.py`` records the
reason at length -- a page and its video tree travel separately, and the
reviewer who opens the copied HTML gets neither picture nor sound with no way to
tell which went missing.  Same rule here.

Why the numbers are ON the page: a review page that only shows video invites the
reader to judge by eye, and the eye is what disagreed with the beat ruler on
2026-08-30 (it was right, and the ruler was replaced).  Both belong side by
side, with the measurement's own null next to it wherever it has one.
"""
import argparse
import base64
import html
import pathlib
import subprocess


def audio_ok(path):
    """A muxed video can carry a well-formed silent stream.  Measure, don't trust."""
    out = subprocess.run(["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].split("dB")[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--title", default="Review")
    ap.add_argument("--table", help="markdown-ish table file rendered above the videos")
    ap.add_argument("--max-mb", type=float, default=40.0,
                    help="above this total, videos are referenced by relative path "
                         "instead of embedded. Embedding is the default because a "
                         "page and its video tree travel separately and the reviewer "
                         "then gets neither picture nor sound with no way to tell "
                         "which went missing; but a 70 MB data: URI page does not "
                         "open either, so the budget is explicit rather than a "
                         "surprise. When it trips, the page says so at the top.")
    args = ap.parse_args()

    videos = sorted(pathlib.Path(args.videos).glob("*.mp4"))
    if not videos:
        raise SystemExit("no videos in {}".format(args.videos))

    total = sum(v.stat().st_size for v in videos)
    embed = total <= args.max_mb * 1e6
    out_dir = pathlib.Path(args.output).resolve().parent
    blocks = []
    for v in videos:
        level = audio_ok(v)
        if level is None or level < -60.0:
            raise SystemExit("{} carries no audio (mean_volume {})".format(v, level))
        if embed:
            source = "data:video/mp4;base64," + base64.b64encode(v.read_bytes()).decode()
        else:
            try:
                source = str(v.resolve().relative_to(out_dir))
            except ValueError:
                source = str(v.resolve())
        blocks.append(
            '<figure><figcaption>{name} <span class="q">audio {lvl:.1f} dB</span></figcaption>'
            '<video controls preload="metadata" src="{src}"></video></figure>'
            .format(name=html.escape(v.stem), lvl=level, src=html.escape(source)))
    if not embed:
        blocks.insert(0, '<p class="warn">{:.0f} MB of video, over the {:.0f} MB embed '
                         'budget: the players below reference files in <code>{}</code> '
                         'beside this page. Move the page and they go dark.</p>'
                         .format(total / 1e6, args.max_mb,
                                 html.escape(pathlib.Path(args.videos).name)))

    table = ""
    if args.table:
        rows = [r for r in pathlib.Path(args.table).read_text(encoding="utf-8").splitlines() if r.strip()]
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if "|" in r
                 and set(r.replace("|", "").replace("-", "").strip()) != set()]
        if cells:
            head = "".join("<th>{}</th>".format(html.escape(c)) for c in cells[0])
            body = "".join("<tr>{}</tr>".format(
                "".join("<td>{}</td>".format(html.escape(c)) for c in row)) for row in cells[1:])
            table = "<table><thead><tr>{}</tr></thead><tbody>{}</tbody></table>".format(head, body)

    page = """<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>
:root{{color-scheme:light dark;--bg:#12110f;--fg:#eae6df;--dim:#8d867c;--rule:#2c2a26;--ok:#7fbf7f}}
body{{margin:0;padding:2.2rem 1.6rem 4rem;background:var(--bg);color:var(--fg);
 font:15px/1.6 ui-sans-serif,system-ui,"Helvetica Neue",sans-serif;max-width:1180px;margin-inline:auto}}
h1{{font-size:1.35rem;font-weight:600;letter-spacing:-.01em;margin:0 0 .3rem}}
p.sub{{color:var(--dim);margin:0 0 2rem}}
table{{border-collapse:collapse;width:100%;margin:0 0 2.4rem;font-variant-numeric:tabular-nums}}
th,td{{text-align:right;padding:.42rem .7rem;border-bottom:1px solid var(--rule)}}
th:first-child,td:first-child{{text-align:left}}
thead th{{color:var(--dim);font-weight:500;font-size:.85rem;text-transform:uppercase;letter-spacing:.05em}}
tbody tr:first-child td{{color:var(--ok)}}
figure{{margin:0 0 2.6rem}}
figcaption{{color:var(--dim);font-size:.85rem;margin-bottom:.45rem;font-variant-numeric:tabular-nums}}
.q{{color:var(--rule);margin-left:.6rem}}
p.warn{{color:#d9a441;border-left:2px solid #d9a441;padding-left:.8rem;margin:0 0 2rem}}
video{{width:100%;display:block;background:#000;border:1px solid var(--rule)}}
</style>
<h1>{title}</h1><p class="sub">{n} clip(s), rows share one stage and one floor.</p>
{table}{blocks}""".format(title=html.escape(args.title), n=len(videos),
                          table=table, blocks="\n".join(blocks))
    out = pathlib.Path(args.output)
    out.write_text(page, encoding="utf-8")
    print("{} ({:.1f} MB, {} clip(s))".format(out, out.stat().st_size / 1e6, len(videos)))


if __name__ == "__main__":
    main()
