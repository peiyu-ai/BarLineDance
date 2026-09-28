#!/usr/bin/env python3
"""One clip's plan as a time strip: which spans are movement and which are filler.

A stacked pose video shows *that* a generation is thin; it cannot show *why*,
because the thing that decides it -- which atomic movement the planner asked
for at each moment -- is invisible in a skeleton.  This is the strip
``tools/render_m4_plan_sheet.py`` draws for one arm, cut down to what a
side-by-side gallery needs and widened to every row of the stack, on the same
time axis as the music.

Filler is named, not inferred.  Label 0 is the transition class: the token the
planner emits when it is naming no atomic movement, and the reason a clip can
carry many segment boundaries while looking under-danced.  It is drawn grey and
labelled, so "the plan spends a third of this clip on filler" is a bar you can
see rather than a number in a table.

Two readings sit beside each row because they disagree and the disagreement is
the point:

``segments``      how often the plan changes its mind -- a plan-level count.
``accents/s``     peaks of whole-body deceleration in the rendered motion --
                  what a viewer actually reads as hitting a beat.

Measured 2026-08-29 over 120 held-out clips, these two order the arms
*oppositely*: the ungridded arm carries 1.413 boundaries/s against the ground
truth's 0.877 (60% more) while producing 1.789 visible accents/s against its
2.208 (19% fewer).  More boundaries, less dancing -- which is what filler is,
and why a boundary count on its own cannot be the criterion.
"""

from __future__ import annotations

import io
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FPS = 30.0
TRANSITION = 0
# SMPL kinematic tree, same list as tools/render_dance_video.py.
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
           18, 19, 20, 21]
FILLER_COLOUR = "#c6ccd4"
# Qualitative, colour-blind-safe, and deliberately not a gradient: neighbouring
# classes have no ordinal relationship, so a sequential map would invent one.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def runs_of(labels):
    """``[(start, end, label)]`` for maximal equal-label runs."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = [0, *edges.tolist(), len(labels)]
    return [(bounds[i], bounds[i + 1], int(labels[bounds[i]]))
            for i in range(len(bounds) - 1)]


def accents(joints):
    """Frames where whole-body speed changes sharply -- a visible accent.

    Deliberately kinematic rather than label-based.  A boundary between two
    similar prototypes changes the plan and nothing on screen; this is what a
    viewer reads as the dancer hitting something.
    """
    joints = np.asarray(joints, float)
    if len(joints) < 6:
        return np.array([], int)
    speed = np.linalg.norm(np.diff(joints, axis=0), axis=2).mean(1)
    change = np.abs(np.diff(speed))
    # A percentile alone has no floor, so a body that is not moving has its own
    # numerical noise promoted to the 90th percentile and every frame comes back
    # an accent -- the unit test caught exactly that on a static skeleton.  The
    # second term ties the threshold to the clip's own motion scale, so "no
    # accents" stays reachable.  On real dancing the percentile dominates: over
    # 120 held-out clips the medians are unchanged to three decimals.
    threshold = max(np.percentile(change, 90), 0.02 * np.median(speed))
    if threshold <= 0:
        return np.array([], int)
    peaks = [i for i in range(1, len(change) - 1)
             if change[i] >= threshold and change[i] >= change[i - 1]
             and change[i] >= change[i + 1]]
    return np.array(peaks, int) + 1


def filler_fraction(labels):
    labels = np.asarray(labels)
    return float((labels == TRANSITION).mean()) if len(labels) else 0.0


def ground_relative(pose):
    """Remove the horizontal root only; keep height.

    Taken from ``tools/render_m4_plan_sheet.py``, which records why: subtracting
    the whole root centres every frame on the pelvis and therefore *deletes
    vertical motion*, so a jump and a stand render identically.  Height is the
    movement's own and has to stay -- on this corpus it is the channel the
    floating defect lives in.
    """
    pose = np.asarray(pose, float)
    out = pose.copy()
    out[..., 0] -= pose[..., :1, 0]
    out[..., 1] -= pose[..., :1, 1]
    return out


def draw_skeleton(ax, pose, colour, floor=None):
    """Front view (x lateral, z up) -- the projection a dance reader expects.

    ``floor`` draws this clip's own ground line, the 5th percentile of its
    lowest joint.  Without it a body that never comes down looks the same as one
    that lands, which is the whole complaint the sheet exists to answer.
    """
    x, z = pose[:, 0], pose[:, 2]
    for joint, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        ax.plot([x[joint], x[parent]], [z[joint], z[parent]],
                color=colour, linewidth=1.15)
    ax.scatter(x, z, s=1.6, color="#c92a2a", zorder=3)
    if floor is not None:
        ax.axhline(floor, color="#adb5bd", linewidth=0.8, linestyle=(0, (4, 3)),
                   zorder=0)
    ax.set_axis_off()
    ax.set_aspect("equal")


def render_sheet(clip, rows, music=None, tags=None, thumbs=10, width=13.0):
    """Plan strip plus a row of poses per arm, on one time axis.

    The video and this sheet answer different questions and neither replaces the
    other: the video shows the dance, the sheet shows *what the planner asked
    for* beside it -- and unlike the video it renders in an IDE preview, which
    will not play ``<video>`` (recorded in ``tools/render_m4_plan_sheet.py``).
    """
    lanes = len(rows)
    music_lanes = 1 if music is not None else 0
    heights = ([0.55] if music is not None else []) + [0.5, 1.15] * lanes
    figure = plt.figure(figsize=(width, 0.62 * sum(heights) + 0.6))
    grid = figure.add_gridspec(len(heights), thumbs, height_ratios=heights,
                               hspace=0.22, wspace=0.02)
    frames = max(len(labels) for _, labels, _ in rows)
    seconds = frames / FPS
    beats = np.array([], float)

    if music is not None:
        music = np.asarray(music)
        span = min(frames, len(music))
        ax = figure.add_subplot(grid[0, :])
        ax.fill_between(np.arange(span) / FPS, music[:span, 0],
                        color="#b9c0c9", linewidth=0)
        beats = np.flatnonzero(music[:span, 34] > 0.5) / FPS
        for beat in beats:
            ax.axvline(beat, color="#7b8794", linewidth=0.5, alpha=0.75)
        ax.set_xlim(0, seconds)
        ax.set_ylabel("music", rotation=0, ha="right", va="center", fontsize=8.5)
        ax.set_yticks([]); ax.set_xticks([])
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    for lane, (name, labels, joints) in enumerate(rows):
        ax = figure.add_subplot(grid[music_lanes + 2 * lane, :])
        _draw_lane(ax, name, labels, joints, beats, seconds, tags)
        if joints is None:
            continue
        poses = ground_relative(joints)
        floor = float(np.percentile(poses[..., 2].min(axis=1), 5))
        times = np.linspace(0, len(poses) - 1, thumbs).astype(int)
        low = min(poses[..., 2].min(), floor) - 0.08
        high = poses[..., 2].max() + 0.08
        for column, frame in enumerate(times):
            cell = figure.add_subplot(grid[music_lanes + 2 * lane + 1, column])
            draw_skeleton(cell, poses[frame], "#3d5a80", floor=floor)
            cell.set_ylim(low, high)

    figure.suptitle(clip, fontsize=9, x=0.012, ha="left", y=0.998, color="#57626f")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=118, bbox_inches="tight",
                   facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def _draw_lane(ax, name, labels, joints, beats, seconds, tags):
    for beat in beats:
        ax.axvline(beat, color="#dfe3e8", linewidth=0.5, zorder=0)
    for start, end, label in runs_of(labels):
        colour = FILLER_COLOUR if label == TRANSITION else PALETTE[label % len(PALETTE)]
        ax.axvspan(start / FPS, end / FPS, facecolor=colour, zorder=1,
                   linewidth=0.6, edgecolor="white")
        width_s = (end - start) / FPS
        if width_s < 1.15:
            continue
        if label == TRANSITION:
            text = "filler"
        elif tags is not None and label in tags:
            text = tags[label]
            if len(text) > int(width_s * 8):
                text = text[:max(3, int(width_s * 8) - 1)] + "\u2026"
        else:
            text = str(label)
        ax.text((start + end) / 2 / FPS, 0.5, text, ha="center", va="center",
                fontsize=7.4, zorder=2,
                color="#4a5560" if label == TRANSITION else "white")
    peaks = accents(joints) if joints is not None else np.array([], int)
    for frame in peaks:
        ax.plot(frame / FPS, 1.06, marker="v", markersize=3.0,
                color="#12161c", clip_on=False, zorder=3)
    beat_hits = 0
    if len(beats) and len(peaks):
        beat_frames = beats * FPS
        beat_hits = sum(1 for p in peaks if np.min(np.abs(beat_frames - p)) <= 2)
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8.5)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_ylim(0, 1); ax.set_xlim(0, seconds)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.text(1.005, 0.5,
            "{} seg \u00b7 {:.0f}% filler \u00b7 {:.2f} accents/s ({:.2f} on beat)".format(
                len(runs_of(labels)), 100 * filler_fraction(labels),
                len(peaks) / max(seconds, 1e-9), beat_hits / max(seconds, 1e-9)),
            transform=ax.transAxes, ha="left", va="center", fontsize=7.2,
            color="#57626f")


def render(clip, rows, music=None, tags=None, width=13.0):
    """PNG bytes for one clip.

    ``rows`` is ``[(label, labels, joints)]`` top to bottom -- ground truth
    first, by the same convention the stacked video uses.
    """
    lanes = len(rows)
    height = 0.62 * lanes + (1.05 if music is not None else 0.35)
    figure, axes = plt.subplots(
        lanes + (1 if music is not None else 0), 1,
        figsize=(width, height), sharex=True,
        gridspec_kw={"height_ratios": ([0.7] if music is not None else []) + [1] * lanes,
                     "hspace": 0.18})
    axes = np.atleast_1d(axes)
    frames = max(len(labels) for _, labels, _ in rows)
    seconds = frames / FPS
    beats = np.array([], float)

    index = 0
    if music is not None:
        music = np.asarray(music)
        span = min(frames, len(music))
        time = np.arange(span) / FPS
        ax = axes[0]
        ax.fill_between(time, music[:span, 0], color="#b9c0c9", linewidth=0)
        beats = np.flatnonzero(music[:span, 34] > 0.5) / FPS
        for beat in beats:
            ax.axvline(beat, color="#7b8794", linewidth=0.5, alpha=0.75)
        ax.set_ylabel("music", rotation=0, ha="right", va="center", fontsize=8.5)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        index = 1

    for lane, (name, labels, joints) in enumerate(rows):
        ax = axes[index + lane]
        for beat in beats:
            ax.axvline(beat, color="#dfe3e8", linewidth=0.5, zorder=0)
        for start, end, label in runs_of(labels):
            colour = FILLER_COLOUR if label == TRANSITION else PALETTE[label % len(PALETTE)]
            ax.axvspan(start / FPS, end / FPS, facecolor=colour, zorder=1,
                       linewidth=0.6, edgecolor="white")
            width_s = (end - start) / FPS
            if width_s < 1.15:
                continue
            if label == TRANSITION:
                text = "filler"
            elif tags is not None and label in tags:
                text = tags[label]
                if len(text) > int(width_s * 8):
                    text = text[:max(3, int(width_s * 8) - 1)] + "…"
            else:
                text = str(label)
            ax.text((start + end) / 2 / FPS, 0.5, text, ha="center", va="center",
                    fontsize=7.6, zorder=2,
                    color="#4a5560" if label == TRANSITION else "white")
        if joints is not None:
            for frame in accents(joints):
                ax.plot(frame / FPS, 1.06, marker="v", markersize=3.2,
                        color="#12161c", clip_on=False, zorder=3)
        beat_hits = 0
        peaks = accents(joints) if joints is not None else np.array([], int)
        if len(beats) and len(peaks):
            beat_frames = beats * FPS
            beat_hits = sum(1 for p in peaks if np.min(np.abs(beat_frames - p)) <= 2)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8.5)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, seconds)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.text(1.005, 0.5,
                "{} seg · {:.0f}% filler · {:.2f} accents/s ({:.2f} on beat)".format(
                    len(runs_of(labels)), 100 * filler_fraction(labels),
                    len(peaks) / max(seconds, 1e-9), beat_hits / max(seconds, 1e-9)),
                transform=ax.transAxes, ha="left", va="center", fontsize=7.4,
                color="#57626f")

    axes[-1].set_xlabel("seconds", fontsize=8.5)
    axes[-1].tick_params(labelsize=8)
    figure.suptitle(clip, fontsize=9, x=0.012, ha="left", y=0.995, color="#57626f")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=132, bbox_inches="tight",
                   facecolor="white")
    plt.close(figure)
    return buffer.getvalue()
