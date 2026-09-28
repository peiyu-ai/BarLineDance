"""Where in the plan pipeline does the class distribution collapse?

MEASURED FACT this exists to explain (20 held-out clips, arm ``txy_t_m6_seam``):
the ground-truth training marginal over the 20 atomic classes is nearly flat --
top class 7.6% of danced frames, top-3 21.8% -- while the shipped plans put
**42.0%** of danced frames in one class and 63.5% in three, and never use 7 of
the 20 classes at all.  Per clip the two agree (median 4 distinct classes each),
so the collapse is ACROSS clips: the same few moves in every song.

Between the model's draw and the plan that reaches retrieval there are four
stages, each of which can flatten the tail:

  1. ``planner.sample``          -- the raw draw, one label per frame per window
  2. ``_fuse_windows``           -- overlapping windows reduced to one label
  3. ``snap_plan_to_bar_grid``   -- boundaries forced onto bar lines, each bar
                                    taking the majority label inside it
  4. ``refine_plan``             -- majority smoothing over ``vote_window`` and
                                    merging of segments below the minimum length

Stages 2, 3 and 4 are all majority operations, and a majority is mode-seeking by
construction: a class that is the plurality in one window out of seven cannot
survive one.  ``_fuse_windows``'s own docstring says so -- ``centre`` "keeps the
sampling distribution the model was trained to produce" while ``vote`` "is an
ensemble over the draws, which is a different estimator".  The shipped arm uses
``vote``.

This probe reports the marginal after each stage so the blame lands on a stage
rather than on a guess.  It changes nothing and trains nothing.
"""

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import infer_atomic
from infer_atomic import (_fuse_windows, _load_checkpoint, _load_music,
                          refine_plan, resolve_device, seed_everything,
                          snap_plan_to_bar_grid)


def shape_of(labels, classes=None):
    """The marginal over danced (non-transition) frames."""
    values = np.asarray(labels).ravel()
    values = values[values != 0]
    counts = collections.Counter(int(v) for v in values)
    total = sum(counts.values())
    if not total:
        return {"classes_used": 0, "top_share": None, "top3_share": None,
                "frames": 0}
    share = sorted((c / total for c in counts.values()), reverse=True)
    return {
        "classes_used": len(counts),
        "top_share": round(share[0], 4),
        "top3_share": round(float(sum(share[:3])), 4),
        "frames": total,
        "counts": {str(k): v for k, v in sorted(counts.items())},
    }


def accumulate(store, stage, labels):
    store.setdefault(stage, collections.Counter())
    values = np.asarray(labels).ravel()
    for v in values[values != 0]:
        store[stage][int(v)] += 1


@torch.no_grad()
def stage_plans(planner, music, window_size, device, *, temperature,
                guidance_weight, plan_stride, fusion, tie_break,
                bar_beats, vote_window, min_segment_length,
                transition_policy, merge_order, deterministic=False):
    """Reproduce ``infer_plan`` and return the labels after each stage.

    Mirrors the body of ``infer_atomic.infer_plan`` rather than reimplementing
    it: the same window starts, the same fusion call, the same snap and the same
    refine.  If that function changes, this probe is measuring the old pipeline
    and its numbers stop meaning anything -- ``tests/test_probe_plan_collapse.py``
    pins the agreement by running both on the same input.
    """
    music = torch.as_tensor(music)
    stride = int(plan_stride or window_size)
    starts = (list(range(0, len(music), window_size)) if stride == window_size
              else infer_atomic._window_starts(len(music), window_size, stride))
    chunks, lengths = [], []
    for start in starts:
        chunk = music[start:start + window_size]
        lengths.append(len(chunk))
        chunks.append(infer_atomic._pad_frames(chunk, window_size))
    batch = torch.stack(chunks).to(device)
    padding = (torch.arange(window_size, device=device)[None]
               >= torch.tensor(lengths, device=device)[:, None])
    extra = {}
    if infer_atomic.planner_wants_global_music(planner):
        extra["global_music"] = infer_atomic.track_summary(music).to(device)[None].expand(
            len(chunks), -1)
    sampled = planner.sample(batch, padding_mask=padding, temperature=temperature,
                             deterministic=deterministic,
                             guidance_weight=guidance_weight, **extra).cpu()

    stages = {}
    # 1. the raw draw, every window's own frames, nothing reduced
    stages["1_raw_draw"] = torch.cat([row[:length]
                                      for row, length in zip(sampled, lengths)])
    # 2. overlapping windows reduced to one label per frame
    stages["2_fused"] = (torch.cat([row[:length] for row, length in zip(sampled, lengths)])
                         if stride == window_size
                         else _fuse_windows(sampled, starts, lengths, len(music),
                                            window_size, fusion, tie_break=tie_break))
    # 3. boundaries forced onto the bar grid
    snapped, _ = snap_plan_to_bar_grid(stages["2_fused"].clone(), music,
                                       beats_per_segment=bar_beats)
    stages["3_bar_snapped"] = snapped
    # 4. majority smoothing and short-segment merging
    stages["4_refined"] = refine_plan(snapped.clone(), vote_window,
                                      min_segment_length,
                                      transition_policy=transition_policy,
                                      merge_order=merge_order)
    return stages


def run(planner_checkpoint, clips, audio_dir, *, device="cuda", seed=20260902,
        temperature=1.0, guidance_weight=1.0, plan_stride=15, fusion="vote",
        tie_break="centre", bar_beats=4, vote_window=5, min_segment_length=6,
        transition_policy="protect", merge_order="shortest"):
    seed_everything(seed)
    device = resolve_device(device)
    planner, planner_args = _load_checkpoint(planner_checkpoint, "planner", device)
    pooled, measured = {}, 0
    for clip in clips:
        path = pathlib.Path(audio_dir) / (clip + ".npy")
        if not path.exists():
            continue
        music = _load_music(path, None, planner_args.music_dim)
        stages = stage_plans(planner, music, planner_args.seq_len, device,
                             temperature=temperature, guidance_weight=guidance_weight,
                             plan_stride=plan_stride, fusion=fusion,
                             tie_break=tie_break, bar_beats=bar_beats,
                             vote_window=vote_window,
                             min_segment_length=min_segment_length,
                             transition_policy=transition_policy,
                             merge_order=merge_order)
        for stage, labels in stages.items():
            accumulate(pooled, stage, labels.numpy())
        measured += 1
    if not measured:
        raise SystemExit(
            "error: none of the {} clips had music under {}.  Nothing was "
            "measured.".format(len(clips), audio_dir))
    return {
        "planner_checkpoint": str(planner_checkpoint),
        "clips": measured,
        "settings": {"fusion": fusion, "tie_break": tie_break,
                     "plan_stride": plan_stride, "temperature": temperature,
                     "guidance_weight": guidance_weight,
                     "vote_window": vote_window,
                     "min_segment_length": min_segment_length,
                     "bar_beats": bar_beats, "merge_order": merge_order},
        "stages": {stage: shape_of(
            np.repeat(list(counter.keys()), list(counter.values())))
            for stage, counter in sorted(pooled.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", required=True)
    parser.add_argument("--clips", required=True, type=pathlib.Path)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance-weight", type=float, default=1.0)
    parser.add_argument("--plan-stride", type=int, default=15)
    parser.add_argument("--fusion", default="vote")
    parser.add_argument("--vote-window", type=int, default=5)
    parser.add_argument("--min-segment-length", type=int, default=6)
    parser.add_argument("--out", type=pathlib.Path)
    arguments = parser.parse_args()

    clips = [line.strip() for line in arguments.clips.read_text().splitlines()
             if line.strip()]
    report = run(arguments.planner, clips, arguments.audio_dir,
                 device=arguments.device, seed=arguments.seed,
                 temperature=arguments.temperature,
                 guidance_weight=arguments.guidance_weight,
                 plan_stride=arguments.plan_stride, fusion=arguments.fusion,
                 vote_window=arguments.vote_window,
                 min_segment_length=arguments.min_segment_length)
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.out:
        arguments.out.write_text(text)
    slim = {s: {k: v for k, v in d.items() if k != "counts"}
            for s, d in report["stages"].items()}
    print(json.dumps({"settings": report["settings"], "clips": report["clips"],
                      "stages": slim}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
