"""Result figures for the BarLine Dance paper.

Every number here is copied from the repository's own records; the source
(file:section) is given next to each block so a reader can check it.
D = docs/DANCE_QUALITY_DEFECTS.md, W = worklog.md.
Run from paper/:  python3 figures/make_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "pdf.fonttype": 42,
})
GT = "#4d4d4d"
BASE = "#9fb4c7"
OURS = "#2b6cb0"
ACC = "#d9822b"
NULL = "#cfcfcf"


def bars(ax, labels, values, colors, fmt="{:.3f}", ypad=0.004):
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.68)
    for xi, v in zip(x, values):
        ax.text(xi, v + (ypad if v >= 0 else -ypad * 3), fmt.format(v),
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=0.6)


# ---------------------------------------------------------------- Fig. step lock
# D §99.1-99.2 (test-20 + val-30, A0 plan): GT 0.155 vs off-song grid 0.016;
# F11 0.094; A0 -0.006 (project-page evidence, W 2026-09-26); J2 0.232; J7 0.224.
fig, ax = plt.subplots(figsize=(3.4, 2.2))
labels = ["GT", "GT on\nother song", "A0", "F11", "J2", "J7"]
vals = [0.155, 0.016, -0.006, 0.094, 0.232, 0.224]
cols = [GT, NULL, BASE, BASE, OURS, OURS]
bars(ax, labels, vals, cols)
ax.set_ylabel("foot-landing beat lock")
ax.set_ylim(-0.03, 0.27)
ax.set_title("Steps land on this song's beat")
fig.tight_layout()
fig.savefig("figures/step_lock.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. gaming
# Whole-song continuation, 9 songs, 3D (D §97.10-97.11): hit and dipburst.
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.1))
arms = ["A0", "E11", "F11", "G17"]
hit = [0.610, 0.613, 0.622, 0.651]
dip = [0.028, 0.046, 0.067, 0.159]
c = [BASE, BASE, OURS, ACC]
bars(axes[0], arms, hit, c, ypad=0.001)
axes[0].set_ylim(0.58, 0.665)
axes[0].axhline(0.58, color="black", lw=0.6)
axes[0].set_title("per-beat arrival (hit) ↑")
bars(axes[1], arms, dip, c)
axes[1].set_ylim(0, 0.185)
axes[1].set_title("stop-then-burst (dipburst) ↓")
fig.tight_layout()
fig.savefig("figures/metric_gaming.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. energy
# K series, 8 full songs, measured on ViTPose of the rendered 720p video
# (D §100.4, W 2026-09-26/27).
fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.1))
arms = ["F11", "J7", "K13"]
c = [BASE, BASE, OURS]
bars(axes[0], arms, [-0.006, -0.021, 0.333], c)
axes[0].set_ylim(-0.06, 0.39)
axes[0].set_title("loudness following (ρ)")
# F11 has no video-level loud/quiet ratio on record, so it is left out.
bars(axes[1], ["J7", "K13"], [1.001, 1.192], [BASE, OURS], fmt="{:.2f}", ypad=0.005)
axes[1].set_ylim(0.9, 1.24)
axes[1].axhline(0.9, color="black", lw=0.6)
axes[1].set_title("speed ratio loud / quiet")
bars(axes[2], arms, [0.023, 0.028, 0.044], c, ypad=0.001)
axes[2].set_ylim(0, 0.052)
axes[2].set_title("on/off-beat contrast")
fig.tight_layout()
fig.savefig("figures/energy_follow.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. turns/seams
# D §91-92 (test / val): completed turns and reversal right after a bar line.
fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.1))
w = 0.26
x = np.arange(2)
axes[0].bar(x - w, [7, 12], w, color=BASE, label="fix8")
axes[0].bar(x, [11, 18], w, color=OURS, label="+ keep turns")
axes[0].bar(x + w, [12, 24], w, color=GT, label="GT")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["test", "val"])
axes[0].set_ylim(0, 31)
axes[0].set_title("completed turns (count)")
axes[0].legend(fontsize=6.2, loc="upper center", ncol=3, handlelength=1, columnspacing=0.6)
axes[1].bar(x - w, [0.71, 0.72], w, color=BASE, label="retrieval cut")
axes[1].bar(x, [0.58, 0.63], w, color=OURS, label="+ source continuation")
axes[1].bar(x + w, [0.52, 0.50], w, color=GT, label="GT")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["test", "val"])
axes[1].set_ylim(0, 1.15)
axes[1].set_title("reversal right after bar line ↓")
axes[1].legend(fontsize=6.2, loc="upper center", ncol=3, handlelength=1, columnspacing=0.6)
fig.tight_layout()
fig.savefig("figures/turns_seams.pdf")
plt.close(fig)
print("ok")
