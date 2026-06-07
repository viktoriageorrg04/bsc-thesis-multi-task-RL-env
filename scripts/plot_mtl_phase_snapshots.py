#!/usr/bin/env python3
"""
Figure B: MTL policy stability — per-task success rate at each training phase endpoint.

Produces a line chart showing how each task's success rate evolved across the four
training phases (P0 -> P1 step-up -> P1 B2-ramp -> P2 all-recover), mean ± std across
3 seeds.  Intended to illustrate retention and interference dynamics, not final
cross-task performance (see cross-eval matrix for that).

Usage:
    python scripts/plot_mtl_phase_snapshots.py
    python scripts/plot_mtl_phase_snapshots.py --out figures/fig_b_mtl_phases.pdf
    python scripts/plot_mtl_phase_snapshots.py --out figures/fig_b.png --dpi 300
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# task config 

TASKS = {
    "A1_forward": {"label": "A1 (Forward Walk)", "color": "#2196F3", "ls": "-"},
    "A2_omni":    {"label": "A2 (Omni Walk)",    "color": "#4CAF50", "ls": "-"},
    "B1_rough":   {"label": "B1 (Rough Walk)",   "color": "#FF9800", "ls": "-"},
    "B2_stairs":  {"label": "B2 (Stair Climb)",  "color": "#F44336", "ls": "-"},
    "C2_gap":     {"label": "C2 (Gap Crossing)", "color": "#9C27B0", "ls": "-"},
}

# phase config 
# all 4 phases, n=3 seeds each.

PHASES = [
    {
        "label": "P0\nBalanced",
        "sublabel": "broad gait init",
        "dirs": {
            "s0": "results_mtl_supporting/MTL_s0/results_mtl_s0_ckpt1450_1024epsa/MTL_unified/summary.json",
            "s1": "results_mtl_supporting/MTL_s1/results_mtl_s1_ckpt1450_1024epsa/MTL_unified/summary.json",
            "s2": "results_mtl_supporting/MTL_s2/results_mtl_s2_ckpt1499_1024epsa/MTL_unified/summary.json",
        },
    },
    {
        "label": "P1\nB2 Step-up",
        "sublabel": "B2 targeted (retain)",
        "dirs": {
            "s0": "results_mtl_supporting/MTL_s0/results_mtl_s0_ckpt1950_1024epsa/MTL_unified/summary.json",
            "s1": "results_mtl_supporting/MTL_s1/results_mtl_s1_ckpt1949_1024epsa/MTL_unified/summary.json",
            "s2": "results_mtl_supporting/MTL_s2/results_mtl_s2_ckpt1998_1024epsa/MTL_unified/summary.json",
        },
    },
    {
        "label": "P1\nB2 Ramp",
        "sublabel": "B2 commands ramped",
        "dirs": {
            "s0": "results_mtl_supporting/MTL_s0/results_mtl_s0_ckpt2048_1024epsa/MTL_unified/summary.json",
            "s1": "results_mtl_supporting/MTL_s1/results_mtl_s1_ckpt2048_1024epsa/MTL_unified/summary.json",
            "s2": "results_mtl_supporting/MTL_s2/results_mtl_s2_ckpt2097_1024epsa/MTL_unified/summary.json",
        },
    },
    {
        "label": "P2\nAll-Recover",
        "sublabel": "balanced recovery",
        "dirs": {
            "s0": "results_seeded_4096/MTL_s0/results_mtl_s0_ckpt2247_1024epsa/MTL_unified/summary.json",
            "s1": "results_seeded_4096/MTL_s1/results_mtl_s1_ckpt2247_1024epsa/MTL_unified/summary.json",
            "s2": "results_seeded_4096/MTL_s2/results_mtl_s2_ckpt2296_1024epsa/MTL_unified/summary.json",
        },
    },
]

# helpers

def load_phase_rates(root: Path, phase: dict) -> dict[str, list[float]]:
    per_task: dict[str, list[float]] = {tk: [] for tk in TASKS}
    for _, rel_path in phase["dirs"].items():
        if rel_path is None:
            continue
        p = root / rel_path
        if not p.exists():
            print(f"[WARN] Missing: {p}")
            continue
        data = json.loads(p.read_text())
        for tk in TASKS:
            if tk in data:
                per_task[tk].append(data[tk]["success_rate"])
    return per_task


# main figure

def make_figure(root: Path, out: Path, dpi: int = 150) -> None:
    n_phases = len(PHASES)
    x = np.arange(n_phases)

    phase_data = [load_phase_rates(root, ph) for ph in PHASES]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    for tk, cfg in TASKS.items():
        means, stds = [], []
        for rates in phase_data:
            vals = np.array(rates[tk])
            if len(vals) == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=0))

        means = np.array(means) * 100
        stds  = np.array(stds)  * 100

        valid = ~np.isnan(means)
        ax.plot(x[valid], means[valid],
                color=cfg["color"], linewidth=2.0, linestyle=cfg["ls"],
                marker="o", markersize=6, label=cfg["label"], zorder=3)
        ax.fill_between(x[valid],
                        (means - stds)[valid], (means + stds)[valid],
                        color=cfg["color"], alpha=0.13, linewidth=0, zorder=2)

    # vertical dashed separator between P1 and P2
    ax.axvline(x=2.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
    ax.text(2.55, 103, "P2 starts", fontsize=7, color="grey", va="top")

    # 70% target line
    ax.axhline(y=70, color="black", linewidth=0.7, linestyle=":", alpha=0.4, zorder=1)
    ax.text(n_phases - 0.05, 71, "70% target", fontsize=7, color="grey",
            ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([ph["label"] for ph in PHASES], fontsize=9,
                       multialignment="center")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_ylim(0, 107)
    ax.set_ylabel("Success Rate", fontsize=10)
    ax.set_xlabel("Training Phase", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(loc="lower right", fontsize=8, frameon=False,
              ncol=1, handlelength=1.8)

    fig.suptitle(
        "MTL Policy: Per-Task Success Rate Across Training Phases  (mean ± std, n=3 seeds each)",
        fontsize=11, fontweight="bold",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[INFO] Saved: {out}")
    plt.close(fig)


# entry point

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".",
                        help="Project root directory (default: current directory)")
    parser.add_argument("--out", default="figures/fig_b_mtl_phases.pdf",
                        help="Output path (pdf or png)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    make_figure(Path(args.root), Path(args.out), dpi=args.dpi)


if __name__ == "__main__":
    main()
