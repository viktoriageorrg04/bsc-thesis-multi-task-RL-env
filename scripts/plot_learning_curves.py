"""Export learning curves from an RSL-RL TensorBoard run directory.

Example:
  isaaclab -p scripts/plot_learning_curves.py \
    --log_dir C:/Users/pavel/OneDrive/Desktop/IsaacLab/logs/rsl_rl/unitree_go2_rough/2026-03-27_11-41-52
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def _latest_events_file(log_dir: str) -> str | None:
    candidates = glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _read_scalar_series(log_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    event_file = _latest_events_file(log_dir)
    if event_file is None:
        return None

    ea = event_accumulator.EventAccumulator(event_file)
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


def _safe_tag_name(tag: str) -> str:
    return tag.replace("/", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TensorBoard learning curves to CSV and PNG.")
    parser.add_argument("--log_dir", type=str, required=True, help="Run directory containing events.out.tfevents.*")
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
        help="TensorBoard scalar tags to export/plot.",
    )
    parser.add_argument("--smoothing", type=int, default=25, help="Moving-average window (>=1).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory (default: <log_dir>/analysis).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.log_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    available = []
    for tag in args.tags:
        series = _read_scalar_series(args.log_dir, tag)
        if series is None:
            print(f"[WARN] Tag not found: {tag}")
            continue
        steps, values = series
        smoothed = _moving_average(values, args.smoothing)
        available.append((tag, steps, values, smoothed))

        csv_path = os.path.join(output_dir, f"{_safe_tag_name(tag)}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,value,smoothed\n")
            for step, value, smooth in zip(steps, values, smoothed, strict=False):
                f.write(f"{int(step)},{float(value):.10g},{float(smooth):.10g}\n")

    if not available:
        print("[ERROR] None of the requested tags were found.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable. CSV export complete, PNG skipped: {exc}")
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(11, 3.5 * len(available)), squeeze=False)
    for idx, (tag, steps, values, smoothed) in enumerate(available):
        ax = axes[idx, 0]
        ax.plot(steps, values, linewidth=1.0, alpha=0.3, label="raw")
        ax.plot(steps, smoothed, linewidth=2.0, label=f"ma({max(1, args.smoothing)})")
        ax.set_title(tag)
        ax.set_xlabel("iteration")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    png_path = os.path.join(output_dir, "learning_curves.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Export complete: {output_dir}")


if __name__ == "__main__":
    main()
