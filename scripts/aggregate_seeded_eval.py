"""Aggregate seeded cross-evaluation results into mean/std matrices.

Input layout is produced by scripts/eval_completed_seeded.cmd:
  results_seeded_1024/A1_forward_s0/summary.json
  results_seeded_1024/A1_forward_s1/summary.json
  ...

It also supports nested MTL summaries, for example:
  results_seeded_4096/MTL_s0/results_mtl_s0_ckpt2247_1024epsa/MTL_unified/summary.json
  results_seeded_4096/MTL_s1/results_mtl_s1_ckpt2247_1024epsa/MTL_unified/summary.json

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
DEFAULT_TRAIN_ORDER = [*TASK_ORDER, "MTL"]
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


def _annotation_color(im, value: float) -> str:
    """Choose readable text color from the rendered cell shade."""
    r, g, b, _ = im.cmap(im.norm(value))
    # Relative luminance approximation for sRGB.
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.45 else "#222222"


def _train_seed_from_summary(summary_path: Path, root: Path) -> tuple[str, str] | None:
    """Resolve the aggregate row and per-seed label for a summary path."""
    rel_parts = summary_path.relative_to(root).parts
    if len(rel_parts) < 2:
        return None

    # std seeded layout
    first_dir = rel_parts[0]
    match = SEED_RE.match(first_dir)
    if match:
        return match.group("task"), first_dir

    # nested conditioned-MTL layout
    for part in rel_parts[:-1]:
        match = SEED_RE.match(part)
        if match:
            return match.group("task"), part

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate seeded eval summaries.")
    parser.add_argument("--results_root", type=str, default="results_seeded_1024")
    parser.add_argument("--metric", type=str, default="success_rate")
    parser.add_argument("--out_prefix", type=str, default=None)
    parser.add_argument("--no_heatmap", action="store_true")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan recursively for summary.json files (default: true).",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        raise FileNotFoundError(root)

    values: dict[str, dict[str, list[float]]] = {}
    seed_rows: dict[str, dict[str, float | None]] = {}

    summary_paths = root.rglob("summary.json") if args.recursive else root.glob("*/summary.json")
    for summary_path in sorted(summary_paths):
        resolved = _train_seed_from_summary(summary_path, root)
        if resolved is None:
            continue
        train, train_seed = resolved
        values.setdefault(train, {eval_task: [] for eval_task in TASK_ORDER})
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
        seed_rows[train_seed] = seed_row

    train_order = [row for row in DEFAULT_TRAIN_ORDER if row in values]
    train_order += sorted(row for row in values if row not in train_order)

    means = {
        train: {eval_task: _mean(values[train][eval_task]) for eval_task in TASK_ORDER}
        for train in train_order
    }
    stds = {
        train: {eval_task: _std(values[train][eval_task]) for eval_task in TASK_ORDER}
        for train in train_order
    }
    counts = {
        train: {eval_task: len(values[train][eval_task]) for eval_task in TASK_ORDER}
        for train in train_order
    }
    row_means = {
        train: _mean([v for v in means[train].values() if v is not None])
        for train in train_order
    }
    row_stds = {
        train: _std([v for v in means[train].values() if v is not None])
        for train in train_order
    }
    seed_mean_rows = {
        train_seed: _mean([v for v in row.values() if v is not None])
        for train_seed, row in seed_rows.items()
    }

    out_prefix = args.out_prefix or f"seeded_{args.metric}"
    mean_csv = root / f"{out_prefix}_mean.csv"
    std_csv = root / f"{out_prefix}_std.csv"
    count_csv = root / f"{out_prefix}_n.csv"
    seed_csv = root / f"{out_prefix}_per_seed.csv"
    row_mean_csv = root / f"{out_prefix}_row_mean.csv"
    json_path = root / f"{out_prefix}.json"

    for path, matrix, row_aggregate in [(mean_csv, means, row_means), (std_csv, stds, row_stds)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["train_task", *TASK_ORDER, "mean_over_eval_tasks"])
            for train in train_order:
                writer.writerow(
                    [train, *[_fmt(matrix[train][eval_task]) for eval_task in TASK_ORDER], _fmt(row_aggregate[train])]
                )

    with count_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["train_task", *TASK_ORDER])
        for train in train_order:
            writer.writerow([train, *[counts[train][eval_task] for eval_task in TASK_ORDER]])

    with seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["train_seed", *TASK_ORDER, "mean_over_eval_tasks"])
        for train_seed in sorted(seed_rows):
            writer.writerow(
                [
                    train_seed,
                    *[_fmt(seed_rows[train_seed][eval_task]) for eval_task in TASK_ORDER],
                    _fmt(seed_mean_rows[train_seed]),
                ]
            )

    with row_mean_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["train_task", "mean_over_eval_tasks", "std_over_eval_tasks"])
        for train in train_order:
            writer.writerow([train, _fmt(row_means[train]), _fmt(row_stds[train])])

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metric": args.metric,
                "task_order": TASK_ORDER,
                "train_order": train_order,
                "mean": means,
                "std": stds,
                "n": counts,
                "row_mean": row_means,
                "row_std": row_stds,
                "per_seed": seed_rows,
                "per_seed_mean": seed_mean_rows,
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
                for train in train_order
            ]
            fig, ax = plt.subplots(figsize=(1.0 + 1.05 * len(TASK_ORDER), 1.1 + 0.65 * len(train_order)))
            im = ax.imshow(heatmap_values, aspect="auto", cmap="YlGnBu", interpolation="nearest")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"mean {args.metric}")
            ax.set_xticks(range(len(TASK_ORDER)))
            ax.set_xticklabels(TASK_ORDER, rotation=35, ha="right")
            ax.set_yticks(range(len(train_order)))
            ax.set_yticklabels(train_order)
            ax.set_xlabel("Eval task")
            ax.set_ylabel("Train task")
            ax.set_title(f"Seeded cross-eval mean ({args.metric})")
            for i, train in enumerate(train_order):
                for j, eval_task in enumerate(TASK_ORDER):
                    value = means[train][eval_task]
                    label = "" if value is None else f"{value:.3f}"
                    color = "#222222" if value is None else _annotation_color(im, value)
                    ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
            fig.tight_layout(pad=0.6)
            fig.savefig(heatmap_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
            plt.close(fig)
            print(f"[DONE] Wrote: {heatmap_path}")

    print(f"[DONE] Wrote: {mean_csv}")
    print(f"[DONE] Wrote: {std_csv}")
    print(f"[DONE] Wrote: {count_csv}")
    print(f"[DONE] Wrote: {seed_csv}")
    print(f"[DONE] Wrote: {row_mean_csv}")
    print(f"[DONE] Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
