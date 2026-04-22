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

    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0},
    )
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


def _list_checkpoint_iters(log_dir: str) -> list[int]:
    ckpts = glob(os.path.join(log_dir, "model_*.pt"))
    iters: list[int] = []
    for path in ckpts:
        base = os.path.basename(path)
        try:
            iters.append(int(base.replace("model_", "").replace(".pt", "")))
        except ValueError:
            continue
    return sorted(set(iters))


def _best_checkpoint_by_reward(log_dir: str, steps: np.ndarray, values: np.ndarray) -> tuple[int, float] | None:
    ckpt_iters = _list_checkpoint_iters(log_dir)
    if not ckpt_iters or steps.size == 0 or values.size == 0:
        return None

    best_iter = None
    best_reward = None
    for ckpt_it in ckpt_iters:
        idx = np.searchsorted(steps, ckpt_it, side="right") - 1
        if idx < 0:
            continue
        reward = float(values[idx])
        if best_reward is None or reward > best_reward:
            best_reward = reward
            best_iter = ckpt_it

    if best_iter is None or best_reward is None:
        return None
    return best_iter, best_reward


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
        if tag == "Train/mean_reward":
            best = _best_checkpoint_by_reward(args.log_dir, steps, values)
            if best is not None:
                best_iter, best_reward = best
                ax.axvline(best_iter, color="tab:red", linestyle="--", linewidth=1.6, label="best ckpt")
                ax.scatter([best_iter], [best_reward], color="tab:red", s=25, zorder=4)
                ax.annotate(
                    f"best: model_{best_iter}.pt ({best_reward:.2f})",
                    xy=(best_iter, best_reward),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color="tab:red",
                )
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
