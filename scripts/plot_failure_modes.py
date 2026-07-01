"""
Scatter plot of failure rate vs success rate for all (train, eval) pairs.
Visualises whether "cannot survive" and "survives but does not perform"
form distinct clusters or are arbitrary.
"""

import os
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text

BASE = os.path.join(os.path.dirname(__file__), "..", "results_seeded_combined")

success_df = pd.read_csv(os.path.join(BASE, "combined_success_rate_mean.csv"), index_col=0)
failure_df = pd.read_csv(os.path.join(BASE, "combined_failure_rate_mean.csv"), index_col=0)

EVAL_TASKS  = [c for c in success_df.columns if c != "mean_over_eval_tasks"]
success_df  = success_df[EVAL_TASKS]
failure_df  = failure_df[EVAL_TASKS]
TRAIN_TASKS = list(success_df.index)

SHORT = {
    "A1_forward": "A1", "A2_omni": "A2",
    "B1_rough":   "B1", "B2_stairs": "B2",
    "C2_gap":     "C2", "MTL": "MTL",
}

COLORS = {
    "A1_forward": "#4e79a7",
    "A2_omni":    "#f28e2b",
    "B1_rough":   "#59a14f",
    "B2_stairs":  "#e15759",
    "C2_gap":     "#b07aa1",
    "MTL":        "#17becf",
}

FAIL_THRESH    = 0.25
SUCCESS_THRESH = 0.45

fig, ax = plt.subplots(figsize=(11, 8))

# quadrant shading
shade = dict(alpha=0.07, zorder=0)
ax.fill_betweenx([SUCCESS_THRESH, 1.02], 0,             FAIL_THRESH, color="green",  **shade)
ax.fill_betweenx([-0.02, SUCCESS_THRESH], 0,            FAIL_THRESH, color="orange", **shade)
ax.fill_betweenx([-0.02, SUCCESS_THRESH], FAIL_THRESH,  1.02,        color="red",    **shade)
ax.fill_betweenx([SUCCESS_THRESH, 1.02],  FAIL_THRESH,  1.02,        color="grey",   **shade)

ax.axvline(FAIL_THRESH,    color="gray", lw=0.9, ls="--", alpha=0.6)
ax.axhline(SUCCESS_THRESH, color="gray", lw=0.9, ls="--", alpha=0.6)

ax.text(0.12, 0.72, "Good transfer",            ha="center", fontsize=9.5, color="#2ca02c",   style="italic")
ax.text(0.12, 0.15, "Survives,\ndoes not perform", ha="center", fontsize=9.5, color="darkorange", style="italic")
ax.text(0.63, 0.15, "Cannot survive",            ha="center", fontsize=9.5, color="#d62728",  style="italic")

# scatter + collect texts
texts = []

for train in TRAIN_TASKS:
    for ev in EVAL_TASKS:
        s     = success_df.loc[train, ev]
        f     = failure_df.loc[train, ev]
        color = COLORS[train]
        lbl   = f"{SHORT[train]}->{SHORT[ev]}"
        is_diag = (train == ev)

        if is_diag:
            ax.scatter(f, s, color=color, s=200, marker="*", zorder=6,
                       edgecolors="k", linewidths=0.6)
        else:
            marker = "D" if train == "MTL" else "o"
            ax.scatter(f, s, color=color, s=60, marker=marker, zorder=5,
                       edgecolors="k", linewidths=0.4, alpha=0.9)

        # label every point; adjustText will sort the collisions
        txt = ax.text(f, s, lbl, fontsize=7.5,
                      color=color if is_diag else "black",
                      fontweight="bold" if is_diag else "normal",
                      alpha=0.9)
        texts.append(txt)

# push labels apart; arrowprops draws a thin line from label to point
adjust_text(
    texts,
    ax=ax,
    expand=(1.3, 1.6),
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
    force_points=(0.4, 0.6),
    force_text=(0.4, 0.6),
)

# legend
handles = [mpatches.Patch(color=COLORS[t], label=SHORT[t]) for t in TRAIN_TASKS]
handles += [
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
               markersize=12, markeredgecolor="k", label="Specialist on own task"),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["MTL"],
               markersize=8,  markeredgecolor="k", label="MTL policy"),
]
ax.legend(handles=handles, loc="upper right", fontsize=8,
          title="Train task", title_fontsize=8.5, framealpha=0.9)

ax.set_xlabel("Mean failure rate", fontsize=12)
ax.set_ylabel("Mean success rate", fontsize=12)
ax.set_title(
    "Failure Mode Scatter: Success vs Failure Rate\n"
    "(all specialist + MTL cross-evaluation pairs, combined seeds)",
    fontsize=11,
)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.08)
ax.grid(True, alpha=0.25)

plt.tight_layout()
out = os.path.join(BASE, "failure_mode_scatter.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
