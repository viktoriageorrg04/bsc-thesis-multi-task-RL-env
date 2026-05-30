"""Aggregate seeded cross-evaluation results into mean/std matrices.

Input layout is produced by scripts/eval_completed_seeded.cmd:
  results_seeded_1024/A1_forward_s0/summary.json
  results_seeded_1024/A1_forward_s1/summary.json
  ...

Usage:
  python scripts/aggregate_seeded_eval.py --results_root results_seeded_1024 --metric success_rate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


TASK_ORDER = ["A1_forward", "A2_omni", "B1_rough", "B2_stairs", "C2_gap"]
SEED_RE = re.compile(r"^(?P<task>.+)_s(?P<seed>\d+)$")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate seeded eval summaries.")
    parser.add_argument("--results_root", type=str, default="results_seeded_1024")
    parser.add_argument("--metric", type=str, default="success_rate")
    parser.add_argument("--out_prefix", type=str, default=None)
    parser.add_argument("--no_heatmap", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        raise FileNotFoundError(root)

    values: dict[str, dict[str, list[float]]] = {
        train: {eval_task: [] for eval_task in TASK_ORDER} for train in TASK_ORDER
    }
    seed_rows: dict[str, dict[str, float | None]] = {}

    for summary_path in sorted(root.glob("*/summary.json")):
        match = SEED_RE.match(summary_path.parent.name)
        if not match:
            continue
        train = match.group("task")
        if train not in values:
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        seed_row: dict[str, float | None] = {}
        for eval_task in TASK_ORDER:
            metric_value = None
            if isinstance(summary.get(eval_task), dict):
                metric_value = summary[eval_task].get(args.metric)
            if metric_value is not None:
                metric_float = float(metric_value)
                values[train][eval_task].append(metric_float)
                seed_row[eval_task] = metric_float
            else:
                seed_row[eval_task] = None
        seed_rows[summary_path.parent.name] = seed_row

    means = {
        train: {eval_task: _mean(values[train][eval_task]) for eval_task in TASK_ORDER}
        for train in TASK_ORDER
    }
    stds = {
        train: {eval_task: _std(values[train][eval_task]) for eval_task in TASK_ORDER}
        for train in TASK_ORDER
    }
    counts = {
        train: {eval_task: len(values[train][eval_task]) for eval_task in TASK_ORDER}
        for train in TASK_ORDER
    }

    out_prefix = args.out_prefix or f"seeded_{args.metric}"
    mean_csv = root / f"{out_prefix}_mean.csv"
    std_csv = root / f"{out_prefix}_std.csv"
    count_csv = root / f"{out_prefix}_n.csv"
    seed_csv = root / f"{out_prefix}_per_seed.csv"
    json_path = root / f"{out_prefix}.json"

    for path, matrix in [(mean_csv, means), (std_csv, stds)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["train_task", *TASK_ORDER])
            for train in TASK_ORDER:
                writer.writerow([train, *[_fmt(matrix[train][eval_task]) for eval_task in TASK_ORDER]])

    with count_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["train_task", *TASK_ORDER])
        for train in TASK_ORDER:
            writer.writerow([train, *[counts[train][eval_task] for eval_task in TASK_ORDER]])

    with seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["train_seed", *TASK_ORDER])
        for train_seed in sorted(seed_rows):
            writer.writerow([train_seed, *[_fmt(seed_rows[train_seed][eval_task]) for eval_task in TASK_ORDER]])

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metric": args.metric,
                "task_order": TASK_ORDER,
                "mean": means,
                "std": stds,
                "n": counts,
                "per_seed": seed_rows,
            },
            f,
            indent=2,
        )

    if not args.no_heatmap:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[WARN] matplotlib unavailable; skipped heatmap export ({exc}).")
        else:
            heatmap_path = root / f"{out_prefix}_mean_heatmap.png"
            heatmap_values = [
                [float("nan") if means[train][eval_task] is None else float(means[train][eval_task]) for eval_task in TASK_ORDER]
                for train in TASK_ORDER
            ]
            fig, ax = plt.subplots(figsize=(1.0 + 1.05 * len(TASK_ORDER), 1.1 + 0.65 * len(TASK_ORDER)))
            im = ax.imshow(heatmap_values, aspect="auto", cmap="YlGnBu", interpolation="nearest")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"mean {args.metric}")
            ax.set_xticks(range(len(TASK_ORDER)))
            ax.set_xticklabels(TASK_ORDER, rotation=35, ha="right")
            ax.set_yticks(range(len(TASK_ORDER)))
            ax.set_yticklabels(TASK_ORDER)
            ax.set_xlabel("Eval task")
            ax.set_ylabel("Train task")
            ax.set_title(f"Seeded cross-eval mean ({args.metric})")
            for i, train in enumerate(TASK_ORDER):
                for j, eval_task in enumerate(TASK_ORDER):
                    value = means[train][eval_task]
                    label = "" if value is None else f"{value:.3f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")
            fig.tight_layout(pad=0.6)
            fig.savefig(heatmap_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
            plt.close(fig)
            print(f"[DONE] Wrote: {heatmap_path}")

    print(f"[DONE] Wrote: {mean_csv}")
    print(f"[DONE] Wrote: {std_csv}")
    print(f"[DONE] Wrote: {count_csv}")
    print(f"[DONE] Wrote: {seed_csv}")
    print(f"[DONE] Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
