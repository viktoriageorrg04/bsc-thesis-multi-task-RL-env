"""Generate the minimum result package for the thesis results chapter.

The script reads cross-evaluation summaries from ``results/<policy>/summary.json``
and creates:

- success-rate heatmap
- specialist-vs-MTL retention drop chart
- family-level average table
- appendix matrices for failure rate and mean alive time

Example:
    python analysis/generate_result_package.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = ["A1_forward", "A2_omni", "B1_rough", "B2_stairs", "C2_gap"]
SPECIALIST_ROWS = ["A1_forward", "A2_omni", "B1_rough", "B2_stairs", "C2_gap"]
FAMILIES = {
    "Family A": ["A1_forward", "A2_omni"],
    "Family B": ["B1_rough", "B2_stairs"],
    "Family C": ["C2_gap"],
}


def _load_summaries(results_root: Path) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for path in sorted(results_root.glob("*/summary.json")):
        with path.open("r", encoding="utf-8") as f:
            summaries[path.parent.name] = json.load(f)
    if not summaries:
        raise FileNotFoundError(f"No summary.json files found under {results_root}")
    return summaries


def _ordered_policies(summaries: dict[str, dict], mtl_row: str) -> list[str]:
    rows = [row for row in SPECIALIST_ROWS if row in summaries]
    if mtl_row in summaries:
        rows.append(mtl_row)
    else:
        raise KeyError(f"MTL row '{mtl_row}' not found. Available: {sorted(summaries)}")
    return rows


def _metric_matrix(summaries: dict[str, dict], policies: list[str], metric: str) -> np.ndarray:
    matrix = np.full((len(policies), len(TASKS)), np.nan, dtype=np.float64)
    for row_idx, policy in enumerate(policies):
        summary = summaries[policy]
        for col_idx, task in enumerate(TASKS):
            if task in summary and metric in summary[task]:
                matrix[row_idx, col_idx] = float(summary[task][metric])
    return matrix


def _write_matrix_csv(path: Path, policies: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["train_task", *TASKS])
        for policy, row in zip(policies, matrix):
            writer.writerow([policy, *[f"{value:.6f}" if np.isfinite(value) else "" for value in row]])


def _write_matrix_md(path: Path, policies: list[str], matrix: np.ndarray, decimals: int = 3) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| train_task | " + " | ".join(TASKS) + " |\n")
        f.write("|---" * (len(TASKS) + 1) + "|\n")
        for policy, row in zip(policies, matrix):
            values = [f"{value:.{decimals}f}" if np.isfinite(value) else "" for value in row]
            f.write(f"| {policy} | " + " | ".join(values) + " |\n")


def _plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".3f",
) -> None:
    fig_width = max(6.8, 1.05 * len(col_labels) + 2.4)
    fig_height = max(4.8, 0.62 * len(row_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_xlabel("Eval task")
    ax.set_ylabel("Train task")
    ax.set_title(title)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, format(value, fmt), ha="center", va="center", color="black", fontsize=8)

    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _specialist_scores(success_matrix: np.ndarray, policies: list[str]) -> np.ndarray:
    values = []
    for task in TASKS:
        row_idx = policies.index(task)
        col_idx = TASKS.index(task)
        values.append(success_matrix[row_idx, col_idx])
    return np.asarray(values, dtype=np.float64)


def _plot_retention_drop(path: Path, specialist: np.ndarray, mtl: np.ndarray, threshold: float) -> None:
    drops = specialist - mtl
    x = np.arange(len(TASKS))

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    bars = ax.bar(x, drops, color="#4c78a8", edgecolor="black", linewidth=0.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(1.0 - threshold, color="#d62728", linestyle="--", linewidth=1.0, label=f"{threshold:.0%} target margin")
    ax.set_xticks(x, TASKS, rotation=30, ha="right")
    ax.set_ylabel("Specialist success - MTL success")
    ax.set_title("Retention Drop of Unified Policy Relative to Specialists")
    ax.set_ylim(min(-0.05, float(np.nanmin(drops)) - 0.05), max(0.35, float(np.nanmax(drops)) + 0.08))
    ax.legend(frameon=False)

    for bar, drop, spec, mtl_value in zip(bars, drops, specialist, mtl):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{drop:.3f}\n({spec:.2f}->{mtl_value:.2f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_family_table(path_csv: Path, path_md: Path, specialist: np.ndarray, mtl: np.ndarray) -> None:
    rows = []
    for family, tasks in FAMILIES.items():
        indices = [TASKS.index(task) for task in tasks]
        spec_avg = float(np.nanmean(specialist[indices]))
        mtl_avg = float(np.nanmean(mtl[indices]))
        rows.append(
            {
                "family": family,
                "tasks": ", ".join(tasks),
                "specialist_success": spec_avg,
                "mtl_success": mtl_avg,
                "retention_drop": spec_avg - mtl_avg,
            }
        )

    with path_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with path_md.open("w", encoding="utf-8") as f:
        f.write("| family | tasks | specialist_success | mtl_success | retention_drop |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['family']} | {row['tasks']} | "
                f"{row['specialist_success']:.3f} | {row['mtl_success']:.3f} | {row['retention_drop']:.3f} |\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/figures/result_package"))
    parser.add_argument("--mtl_row", type=str, default="MTL_unified_final_s33")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    summaries = _load_summaries(args.results_root)
    policies = _ordered_policies(summaries, args.mtl_row)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    success = _metric_matrix(summaries, policies, "success_rate")
    failure = _metric_matrix(summaries, policies, "failure_rate")
    alive = _metric_matrix(summaries, policies, "mean_alive_time_s")

    _write_matrix_csv(args.output_dir / "success_rate_matrix.csv", policies, success)
    _write_matrix_md(args.output_dir / "success_rate_matrix.md", policies, success)
    _plot_heatmap(
        args.output_dir / "success_rate_heatmap.png",
        success,
        policies,
        TASKS,
        "Cross-Eval Heatmap (Success Rate)",
        "success_rate",
        vmin=0.0,
        vmax=1.0,
    )

    specialist = _specialist_scores(success, policies)
    mtl = success[policies.index(args.mtl_row), :]
    _plot_retention_drop(args.output_dir / "retention_drop.png", specialist, mtl, args.threshold)
    _write_family_table(args.output_dir / "family_averages.csv", args.output_dir / "family_averages.md", specialist, mtl)

    for metric_name, matrix, vmax, title in [
        ("failure_rate", failure, 1.0, "Cross-Eval Heatmap (Failure Rate)"),
        ("mean_alive_time_s", alive, None, "Cross-Eval Heatmap (Mean Alive Time)"),
    ]:
        _write_matrix_csv(args.output_dir / f"{metric_name}_matrix.csv", policies, matrix)
        _write_matrix_md(args.output_dir / f"{metric_name}_matrix.md", policies, matrix)
        _plot_heatmap(
            args.output_dir / f"{metric_name}_heatmap.png",
            matrix,
            policies,
            TASKS,
            title,
            metric_name,
            vmin=0.0,
            vmax=vmax,
        )

    phase_plot = args.results_root / "figures" / "mtl_phases" / "learning_curves_mtl_phases.png"
    print(f"[OK] Wrote result package to: {args.output_dir}")
    if phase_plot.exists():
        print(f"[OK] Phase/checkpoint curve already available at: {phase_plot}")
    else:
        print("[WARN] Phase/checkpoint curve not found. Generate it with scripts/plot_mtl_phases.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
