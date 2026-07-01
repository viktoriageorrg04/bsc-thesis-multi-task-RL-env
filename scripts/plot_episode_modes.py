"""
Per-episode failure mode breakdown as stacked bar chart.
For each (train, eval) pair shows fraction of episodes that:
  - succeeded
  - survived but did not perform (time_out, no success)
  - could not survive (failure termination)
"""

import os, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

RESULTS_DIRS = [
    "results_seeded_1024",
    "results_seeded_4096",
]

SHORT = {
    "A1_forward": "A1", "A2_omni": "A2",
    "B1_rough": "B1", "B2_stairs": "B2",
    "C2_gap": "C2", "MTL": "MTL",
}

TRAIN_ORDER = ["A1_forward", "A2_omni", "B1_rough", "B2_stairs", "C2_gap", "MTL"]
EVAL_ORDER  = ["A1_forward", "A2_omni", "B1_rough", "B2_stairs", "C2_gap"]

COLORS_TRAIN = {
    "A1_forward": "#4e79a7", "A2_omni": "#f28e2b",
    "B1_rough":   "#59a14f", "B2_stairs": "#e15759",
    "C2_gap":     "#b07aa1", "MTL": "#17becf",
}

MODE_COLORS = {
    "success":              "#2ca02c",
    "survives_no_perform":  "#ff7f0e",
    "cannot_survive":       "#d62728",
}

# ── collect episodes ──────────────────────────────────────────────────────────
# data[train_task][eval_task] = list of episode dicts
data = defaultdict(lambda: defaultdict(list))

for results_dir in RESULTS_DIRS:
    for run_dir in glob.glob(os.path.join(results_dir, "*")):
        folder = os.path.basename(run_dir)
        # folder name like "A1_forward_s0" → train_task = "A1_forward"
        parts = folder.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].startswith("s"):
            continue
        train_task = parts[0]
        if train_task not in SHORT:
            continue
        for ev_task in EVAL_ORDER:
            json_path = os.path.join(run_dir, f"{ev_task}.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path) as f:
                d = json.load(f)
            data[train_task][ev_task].extend(d.get("episodes", []))

# MTL runs are nested: results_seeded_4096/MTL_s*/*/MTL_unified/{eval}.json
for json_path in glob.glob("results_seeded_4096/MTL_s*/*/*/*.json"):
    ev_task = os.path.splitext(os.path.basename(json_path))[0]
    if ev_task not in EVAL_ORDER:
        continue
    with open(json_path) as f:
        d = json.load(f)
    data["MTL"][ev_task].extend(d.get("episodes", []))

# ── compute fractions ─────────────────────────────────────────────────────────
def classify(ep):
    if ep["termination_reason"] == "failure":
        return "cannot_survive"
    elif ep["episode_success"]:
        return "success"
    else:
        return "survives_no_perform"

fracs = {}  # (train, eval) -> {mode: fraction}
for train in TRAIN_ORDER:
    for ev in EVAL_ORDER:
        eps = data[train][ev]
        if not eps:
            fracs[(train, ev)] = None
            continue
        counts = {"success": 0, "survives_no_perform": 0, "cannot_survive": 0}
        for ep in eps:
            counts[classify(ep)] += 1
        n = len(eps)
        fracs[(train, ev)] = {k: v / n for k, v in counts.items()}

# ── plot: one panel per eval task ────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=True)
fig.suptitle(
    "Per-episode failure mode breakdown by eval task\n"
    "(stacked: green=success, orange=survives/no-perform, red=cannot-survive)",
    fontsize=11,
)

modes = ["success", "survives_no_perform", "cannot_survive"]

for ax, ev in zip(axes, EVAL_ORDER):
    x = np.arange(len(TRAIN_ORDER))
    bottoms = np.zeros(len(TRAIN_ORDER))

    for mode in modes:
        heights = []
        for train in TRAIN_ORDER:
            fr = fracs.get((train, ev))
            heights.append(fr[mode] if fr else 0.0)
        heights = np.array(heights)
        ax.bar(x, heights, bottom=bottoms,
               color=MODE_COLORS[mode], width=0.65, alpha=0.88,
               edgecolor="white", linewidth=0.4)
        bottoms += heights

    ax.set_title(f"Eval: {SHORT[ev]}", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[t] for t in TRAIN_ORDER], fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Fraction of episodes", fontsize=10)

legend_handles = [
    mpatches.Patch(color=MODE_COLORS["success"],             label="Success"),
    mpatches.Patch(color=MODE_COLORS["survives_no_perform"], label="Survives, no task completion"),
    mpatches.Patch(color=MODE_COLORS["cannot_survive"],      label="Cannot survive"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = os.path.join("results_seeded_combined", "episode_mode_breakdown.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
