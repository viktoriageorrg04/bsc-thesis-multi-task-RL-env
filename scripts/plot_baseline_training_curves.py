#!/usr/bin/env python3
"""
Figure A: Single-task baseline training curves.

Produces a 2x3 grid of subplots — one per task — showing the smoothed mean
episode reward over training iterations with ±1 std shading across 3 seeds.

Usage:
    python scripts/plot_baseline_training_curves.py
    python scripts/plot_baseline_training_curves.py --out figures/fig_a_baseline_training.pdf
    python scripts/plot_baseline_training_curves.py --out figures/fig_a.png --dpi 300
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# task config

TASKS = [
    {
        "label": "A1 — Forward Walk",
        "log_dir": "logs/rsl_rl/unitree_go2_a1_legacy_1024_seeds",
        "color": "#2196F3",
    },
    {
        "label": "A2 — Omni Walk",
        "log_dir": "logs/rsl_rl/unitree_go2_a2_baseline_1024_seeds",
        "color": "#4CAF50",
    },
    {
        "label": "B1 — Rough Walk",
        "log_dir": "logs/rsl_rl/unitree_go2_b1_baseline_1024_seeds",
        "color": "#FF9800",
    },
    {
        "label": "B2 — Stair Climb",
        "log_dir": "logs/rsl_rl/unitree_go2_b2_local_apr15_recheck_1024_seeds",
        "color": "#F44336",
    },
    {
        "label": "C2 — Gap Crossing",
        "log_dir": "logs/rsl_rl/unitree_go2_c2_from_a2_refined_1024_seeds",
        "color": "#9C27B0",
    },
]

# helpers

def load_seed_curves(log_dir: Path) -> list[np.ndarray]:
    """Return list of smoothed reward arrays, one per seed run found."""
    csv_paths = sorted(log_dir.glob("*/analysis/Train_mean_reward.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No Train_mean_reward.csv found under {log_dir}")
    curves = []
    for p in csv_paths:
        df = pd.read_csv(p)
        curves.append(df["smoothed"].to_numpy(dtype=float))
    return curves


def align_stack(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip all curves to the shortest length, return (steps, mean, std)."""
    min_len = min(len(c) for c in curves)
    mat = np.stack([c[:min_len] for c in curves])          # (n_seeds, T)
    return np.arange(min_len), mat.mean(axis=0), mat.std(axis=0)


# main figs

def make_figure(root: Path, out: Path, dpi: int = 150) -> None:
    fig, axes = plt.subplots(
        2, 3,
        figsize=(13, 6),
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for i, task in enumerate(TASKS):
        ax = axes_flat[i]
        log_dir = root / task["log_dir"]

        try:
            curves = load_seed_curves(log_dir)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            ax.text(0.5, 0.5, "data not found", ha="center", va="center",
                    transform=ax.transAxes, color="red")
            ax.set_title(task["label"], fontsize=10, fontweight="bold")
            continue

        steps, mean, std = align_stack(curves)
        n_seeds = len(curves)
        color = task["color"]

        ax.plot(steps, mean, color=color, linewidth=1.6)
        ax.fill_between(steps, mean - std, mean + std,
                        color=color, alpha=0.20, linewidth=0)

        # individual seed traces (thin, transparent) — optional visual aid
        for c in curves:
            ax.plot(np.arange(len(c[:len(steps)])), c[:len(steps)],
                    color=color, linewidth=0.4, alpha=0.35)

        ax.set_title(task["label"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Mean Episode Reward", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        # seed count annotation
        ax.annotate(f"n={n_seeds} seeds", xy=(0.98, 0.04),
                    xycoords="axes fraction", ha="right", fontsize=7,
                    color="grey")

    # 6th cell: combined legend
    ax_leg = axes_flat[5]
    ax_leg.axis("off")
    legend_elements = []
    for task in TASKS:
        legend_elements.append(
            plt.Line2D([0], [0], color=task["color"], linewidth=2,
                       label=task["label"])
        )
    # add shading entry once
    from matplotlib.patches import Patch
    legend_elements.append(
        Patch(facecolor="grey", alpha=0.3, label="±1 std (seeds)")
    )
    ax_leg.legend(handles=legend_elements, loc="center", fontsize=9,
                  frameon=False, title="Task", title_fontsize=9)

    fig.suptitle(
        "Single-Task Baseline Training Dynamics  (smoothed reward, 3 seeds)",
        fontsize=12, fontweight="bold",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[INFO] Saved: {out}")
    plt.close(fig)


# entry pt

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--out", default="figures/fig_a_baseline_training.pdf",
        help="Output path (pdf or png)",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    make_figure(Path(args.root), Path(args.out), dpi=args.dpi)


if __name__ == "__main__":
    main()
