"""Plot MTL learning curves across multiple phase runs with sampling annotations.

Example:
  C:/Users/pavel/anaconda3/envs/env_isaaclab/python.exe scripts/plot_mtl_phases.py \
    --experiment_dir logs/rsl_rl/unitree_go2_mtl_schedule \
    --runs 2026-04-16_10-00-00_p0_rough 2026-04-16_11-00-00_p1_stairs 2026-04-16_12-00-00_p2_gap
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def _read_scalar_series(log_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not glob(os.path.join(log_dir, "events.out.tfevents.*")):
        return None
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    points = ea.Scalars(tag)
    if not points:
        return None
    steps = np.array([p.step for p in points], dtype=np.int64)
    values = np.array([p.value for p in points], dtype=np.float64)
    return steps, values


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _resolve_run_path(experiment_dir: str, run: str) -> str:
    if os.path.isdir(run):
        return os.path.abspath(run)
    return os.path.abspath(os.path.join(experiment_dir, run))


def _load_sampling_profile(run_path: str) -> dict | None:
    profile_path = os.path.join(run_path, "params", "sampling_profile.json")
    if not os.path.isfile(profile_path):
        return None
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _phase_label(run_name: str, sampling_profile: dict | None) -> str:
    if sampling_profile is None:
        return run_name
    strategy = sampling_profile.get("strategy", "unknown")
    if strategy == "focus":
        focus_terrain = sampling_profile.get("focus_terrain", "unknown")
        focus_prob = float(sampling_profile.get("focus_prob", 0.0))
        alt_prob = max(0.0, 1.0 - focus_prob)
        return f"{run_name}\nfocus={focus_terrain} {focus_prob:.2f}, alt={alt_prob:.2f}"
    if strategy == "uniform":
        return f"{run_name}\nuniform (0.25 each)"
    return f"{run_name}\n{strategy}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MTL phase curves with sampling-profile annotations.")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="logs/rsl_rl/unitree_go2_mtl_unified",
        help="Directory containing phase run folders.",
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Run folder names (or absolute paths) in sequence.")
    parser.add_argument(
        "--tags",
        nargs="+",
        default=[
            "Train/mean_reward",
            "Train/mean_episode_length",
            "Loss/value_function",
            "Loss/surrogate",
            "Loss/entropy",
        ],
        help="TensorBoard tags to plot as stacked subplots.",
    )
    parser.add_argument("--smoothing", type=int, default=25, help="Moving-average window.")
    parser.add_argument("--title", type=str, default="MTL Learning Curves with Sampling Phases")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: <experiment_dir>/analysis).")
    parser.add_argument("--hide_raw", action="store_true", default=False, help="Hide raw curves and plot smoothed only.")
    args = parser.parse_args()

    run_infos = []
    primary_tag = args.tags[0]
    for run in args.runs:
        run_path = _resolve_run_path(args.experiment_dir, run)
        if not os.path.isdir(run_path):
            raise FileNotFoundError(f"Run folder not found: {run_path}")
        series = _read_scalar_series(run_path, primary_tag)
        if series is None:
            raise ValueError(f"Primary tag '{primary_tag}' missing for run: {run_path}")
        steps, _ = series
        if steps.size == 0:
            raise ValueError(f"Primary tag '{primary_tag}' is empty for run: {run_path}")
        run_name = os.path.basename(run_path.rstrip("\\/"))
        sampling_profile = _load_sampling_profile(run_path)
        run_infos.append(
            {
                "name": run_name,
                "path": run_path,
                "len": int(steps.max()) + 1,
                "profile": sampling_profile,
            }
        )

    start = 0
    for info in run_infos:
        info["start"] = start
        info["end"] = start + info["len"] - 1
        info["label"] = _phase_label(info["name"], info["profile"])
        start = info["end"] + 1

    output_dir = args.output_dir or os.path.join(args.experiment_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plotting: {exc}") from exc

    fig, axes = plt.subplots(len(args.tags), 1, figsize=(13, 3.8 * len(args.tags)), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for tag_idx, tag in enumerate(args.tags):
        ax = axes[tag_idx, 0]
        for run_idx, info in enumerate(run_infos):
            series = _read_scalar_series(info["path"], tag)
            color = cmap(run_idx % 10)
            ax.axvspan(info["start"], info["end"], color=color, alpha=0.08)
            if series is None:
                ax.text(
                    (info["start"] + info["end"]) / 2.0,
                    0.5,
                    "tag missing",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )
                continue
            steps, values = series
            x = steps + info["start"]
            smoothed = _moving_average(values, args.smoothing)
            if not args.hide_raw:
                ax.plot(x, values, linewidth=0.9, alpha=0.25, color=color)
            ax.plot(x, smoothed, linewidth=2.0, color=color, label=info["name"] if tag_idx == 0 else None)

        for info in run_infos[:-1]:
            ax.axvline(info["end"] + 0.5, linestyle="--", linewidth=1.0, alpha=0.6, color="black")

        for info in run_infos:
            ax.text(
                (info["start"] + info["end"]) / 2.0,
                1.02,
                info["label"],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(tag)
        ax.set_xlabel("global iteration (concatenated phases)")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.25)
        if tag_idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png_path = os.path.join(output_dir, "learning_curves_mtl_phases.png")
    fig.savefig(png_path, dpi=170)
    plt.close(fig)

    csv_path = os.path.join(output_dir, "mtl_phase_ranges.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("run,start_step,end_step,label,path\n")
        for info in run_infos:
            label = info["label"].replace("\n", " | ")
            f.write(f"{info['name']},{info['start']},{info['end']},\"{label}\",\"{info['path']}\"\n")

    print(f"[INFO] Saved plot: {png_path}")
    print(f"[INFO] Saved phase ranges: {csv_path}")


if __name__ == "__main__":
    main()
