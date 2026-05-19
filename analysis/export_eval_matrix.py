"""Export cross-evaluation summaries into matrix files.

Usage:
  python analysis/export_eval_matrix.py --results_root results --metric success_rate

Expected input:
  results/<train_short>/summary.json
where each summary.json is a dict keyed by eval short task names.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASK_ORDER = ["A1_forward", "A2_omni", "B1_rough", "B2_stairs", "C2_gap"]


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _load_summary(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"summary file is not a dict: {path}")
    return data


def _sort_train_rows(rows: list[str]) -> list[str]:
    known = [r for r in TASK_ORDER if r in rows]
    unknown = sorted(r for r in rows if r not in TASK_ORDER)
    return known + unknown


def main() -> int:
    parser = argparse.ArgumentParser(description="Export cross-eval matrix from summary.json files.")
    parser.add_argument("--results_root", type=str, default="results", help="Root results directory.")
    parser.add_argument("--metric", type=str, default="success_rate", help="Metric field to export.")
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help="Output file prefix (default: matrix_<metric>) under results_root.",
    )
    parser.add_argument("--no_heatmap", action="store_true", help="Skip heatmap image export.")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Train-task result folder names to exclude from the exported matrix.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"results_root does not exist: {results_root}")

    summaries: dict[str, dict[str, dict]] = {}
    excluded = set(args.exclude)
    for summary_path in sorted(results_root.glob("*/summary.json")):
        train_short = summary_path.parent.name
        if train_short in excluded:
            continue
        summaries[train_short] = _load_summary(summary_path)

    if not summaries:
        raise RuntimeError(f"No summary files found under: {results_root}")

    train_rows = _sort_train_rows(list(summaries.keys()))
    eval_cols = TASK_ORDER.copy()

    matrix: dict[str, dict[str, object]] = {}
    for train in train_rows:
        row: dict[str, object] = {}
        per_eval = summaries[train]
        for eval_task in eval_cols:
            metric_val = None
            if eval_task in per_eval and isinstance(per_eval[eval_task], dict):
                metric_val = per_eval[eval_task].get(args.metric)
            row[eval_task] = metric_val
        matrix[train] = row

    out_prefix = args.out_prefix or f"matrix_{args.metric}"
    csv_path = results_root / f"{out_prefix}.csv"
    md_path = results_root / f"{out_prefix}.md"
    json_path = results_root / f"{out_prefix}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["train_task", *eval_cols])
        for train in train_rows:
            writer.writerow([train, *[_format_value(matrix[train][c]) for c in eval_cols]])

    with md_path.open("w", encoding="utf-8") as f:
        header = "| train_task | " + " | ".join(eval_cols) + " |\n"
        sep = "|" + "---|" * (len(eval_cols) + 1) + "\n"
        f.write(header)
        f.write(sep)
        for train in train_rows:
            vals = " | ".join(_format_value(matrix[train][c]) for c in eval_cols)
            f.write(f"| {train} | {vals} |\n")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metric": args.metric,
                "eval_columns": eval_cols,
                "train_rows": train_rows,
                "matrix": matrix,
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
            heatmap_path = results_root / f"{out_prefix}_heatmap.png"
            values: list[list[float]] = []
            for train in train_rows:
                row_vals: list[float] = []
                for eval_task in eval_cols:
                    val = matrix[train][eval_task]
                    row_vals.append(float("nan") if val is None else float(val))
                values.append(row_vals)

            fig, ax = plt.subplots(figsize=(1.0 + 1.05 * len(eval_cols), 1.1 + 0.65 * len(train_rows)))
            im = ax.imshow(values, aspect="auto", cmap="YlGnBu", interpolation="nearest")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(args.metric)

            ax.set_xticks(range(len(eval_cols)))
            ax.set_xticklabels(eval_cols, rotation=35, ha="right")
            ax.set_yticks(range(len(train_rows)))
            ax.set_yticklabels(train_rows)
            ax.set_xlabel("Eval task")
            ax.set_ylabel("Train task")
            ax.set_title(f"Cross-eval heatmap ({args.metric})")

            for i, train in enumerate(train_rows):
                for j, eval_task in enumerate(eval_cols):
                    v = matrix[train][eval_task]
                    label = "" if v is None else f"{float(v):.3f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")

            fig.tight_layout(pad=0.6)
            fig.savefig(heatmap_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
            plt.close(fig)
            print(f"[DONE] Wrote: {heatmap_path}")

    print(f"[DONE] Wrote: {csv_path}")
    print(f"[DONE] Wrote: {md_path}")
    print(f"[DONE] Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
