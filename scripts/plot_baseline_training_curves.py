#!/usr/bin/env python3
"""
Figure A: specialist and unified MTL training curves.

Produces a 2x3 grid of subplots: five single-task specialists and the unified
MTL policy. Each subplot shows the smoothed mean episode reward over training
iterations with +/-1 std shading across 3 seeds.

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
from matplotlib.patches import Patch


TASKS = [
    {
        "label": "A1 - Forward Walk",
        "log_dir": "logs/rsl_rl/unitree_go2_a1_legacy_1024_seeds",
        "color": "#2196F3",
    },
    {
        "label": "A2 - Omni Walk",
        "log_dir": "logs/rsl_rl/unitree_go2_a2_baseline_1024_seeds",
        "color": "#4CAF50",
    },
    {
        "label": "B1 - Rough Walk",
        "log_dir": "logs/rsl_rl/unitree_go2_b1_baseline_1024_seeds",
        "color": "#FF9800",
    },
    {
        "label": "B2 - Stair Climb",
        "log_dir": "logs/rsl_rl/unitree_go2_b2_local_apr15_recheck_1024_seeds",
        "color": "#F44336",
    },
    {
        "label": "C2 - Gap Crossing",
        "log_dir": "logs/rsl_rl/unitree_go2_c2_from_a2_refined_1024_seeds",
        "color": "#9C27B0",
    },
    {
        "label": "MTL - Unified Policy",
        "log_dir": "logs/rsl_rl/unitree_go2_mtl_conditioned_minruns",
        "color": "#212121",
        "mtl": True,
    },
]

MTL_PHASE_BOUNDARIES = [
    (0, "P0\nbalanced", 0.95),
    (1500, "P1\nstep-up", 0.95),
    (2000, "P1\nramp", 0.82),
    (2100, "P2\nrecover", 0.95),
]


def load_seed_curves(log_dir: Path) -> list[np.ndarray]:
    """Return one smoothed reward array per seed run."""
    csv_paths = sorted(log_dir.glob("*/analysis/Train_mean_reward.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No Train_mean_reward.csv found under {log_dir}")

    curves = []
    for path in csv_paths:
        df = pd.read_csv(path)
        curves.append(df["smoothed"].to_numpy(dtype=float))
    return curves


def load_mtl_seed_curves(log_dir: Path) -> list[np.ndarray]:
    """Return one concatenated smoothed reward curve per retained MTL seed path."""
    curves = []
    for seed_dir in sorted(log_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue

        phase_csvs = sorted(
            path for path in seed_dir.glob("*/analysis/Train_mean_reward.csv")
            if "retired" not in path.parts
        )
        if not phase_csvs:
            continue

        phase_curves = []
        for path in phase_csvs:
            df = pd.read_csv(path)
            phase_curves.append(df["smoothed"].to_numpy(dtype=float))
        curves.append(np.concatenate(phase_curves))

    if not curves:
        raise FileNotFoundError(f"No MTL Train_mean_reward.csv found under {log_dir}")
    return curves


def align_stack(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip curves to the shortest length and return steps, mean, and std."""
    min_len = min(len(curve) for curve in curves)
    mat = np.stack([curve[:min_len] for curve in curves])
    return np.arange(min_len), mat.mean(axis=0), mat.std(axis=0)


def make_figure(root: Path, out: Path, dpi: int = 150) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.20, wspace=0.24, hspace=0.42)
    axes_flat = axes.flatten()

    for i, task in enumerate(TASKS):
        ax = axes_flat[i]
        log_dir = root / task["log_dir"]

        try:
            curves = load_mtl_seed_curves(log_dir) if task.get("mtl") else load_seed_curves(log_dir)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            ax.text(
                0.5,
                0.5,
                "data not found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="red",
            )
            ax.set_title(task["label"], fontsize=10, fontweight="bold")
            continue

        steps, mean, std = align_stack(curves)
        n_seeds = len(curves)
        color = task["color"]

        ax.plot(steps, mean, color=color, linewidth=1.6)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.20, linewidth=0)

        for curve in curves:
            ax.plot(
                np.arange(len(curve[: len(steps)])),
                curve[: len(steps)],
                color=color,
                linewidth=0.4,
                alpha=0.35,
            )

        ax.set_title(task["label"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Mean Episode Reward", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate(
            f"n={n_seeds} seeds",
            xy=(0.98, 0.04),
            xycoords="axes fraction",
            ha="right",
            fontsize=7,
            color="grey",
        )

        if task.get("mtl"):
            ax.set_xlim(left=-60, right=max(steps) + 40)
            for x_pos, label, y_frac in MTL_PHASE_BOUNDARIES:
                ax.axvline(
                    x=x_pos,
                    color="#D62728",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.90,
                    zorder=5,
                )
                ax.text(
                    x_pos + 18,
                    y_frac,
                    label,
                    transform=ax.get_xaxis_transform(),
                    va="top",
                    ha="left",
                    fontsize=7.5,
                    color="#D62728",
                    fontweight="bold",
                    linespacing=0.9,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.85,
                    },
                    zorder=6,
                )

    legend_elements = [
        plt.Line2D([0], [0], color=task["color"], linewidth=2, label=task["label"])
        for task in TASKS
    ]
    legend_elements.append(Patch(facecolor="grey", alpha=0.3, label="+/-1 std (seeds)"))
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        fontsize=8,
        frameon=False,
        title="Policy/task",
        title_fontsize=8,
    )

    fig.suptitle(
        "Training Dynamics of Specialist and Unified MTL Policies (smoothed reward, 3 seeds)",
        fontsize=12,
        fontweight="bold",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[INFO] Saved: {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--out", default="figures/fig_a_baseline_training.pdf", help="Output path")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    make_figure(Path(args.root), Path(args.out), dpi=args.dpi)


if __name__ == "__main__":
    main()
