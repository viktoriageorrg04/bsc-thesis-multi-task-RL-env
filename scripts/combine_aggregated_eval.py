"""Combine specialist and MTL aggregate evaluation matrices.

Example:
  python scripts/combine_aggregated_eval.py \
    --baseline_root results_seeded_1024 \
    --mtl_root results_seeded_4096 \
    --metric success_rate \
    --out_root results_seeded_combined
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASK_ORDER = ["A1_forward", "A2_omni", "B1_rough", "B2_stairs", "C2_gap"]
ROW_ORDER = [*TASK_ORDER, "MTL"]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _sort_rows(rows: list[dict[str, str]], key: str) -> list[dict[str, str]]:
    order = {name: idx for idx, name in enumerate(ROW_ORDER)}
    return sorted(rows, key=lambda row: (order.get(row.get(key, ""), len(order)), row.get(key, "")))


def _as_float(value: str) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["train_task", *TASK_ORDER, "mean_over_eval_tasks"]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "---|" * len(fields) + "\n")
        for row in rows:
            f.write("| " + " | ".join(row.get(field, "") for field in fields) + " |\n")


def _write_heatmap(path: Path, rows: list[dict[str, str]], metric: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipped heatmap export ({exc}).")
        return

    values = []
    labels = []
    for row in rows:
        labels.append(row["train_task"])
        values.append([
            float("nan") if _as_float(row.get(task, "")) is None else float(row[task])
            for task in TASK_ORDER
        ])

    fig, ax = plt.subplots(figsize=(1.0 + 1.05 * len(TASK_ORDER), 1.1 + 0.65 * len(rows)))
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"mean {metric}")
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels(TASK_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Eval task")
    ax.set_ylabel("Train task")
    ax.set_title(f"Specialist + MTL mean ({metric})")
    for i, row in enumerate(rows):
        for j, task in enumerate(TASK_ORDER):
            value = _as_float(row.get(task, ""))
            label = "" if value is None else f"{value:.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="black")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[DONE] Wrote: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine specialist and MTL aggregate matrices.")
    parser.add_argument("--baseline_root", type=str, default="results_seeded_1024")
    parser.add_argument("--mtl_root", type=str, default="results_seeded_4096")
    parser.add_argument("--metric", type=str, default="success_rate")
    parser.add_argument("--out_root", type=str, default="results_seeded_combined")
    parser.add_argument("--no_heatmap", action="store_true")
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root)
    mtl_root = Path(args.mtl_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    prefix = f"seeded_{args.metric}"
    fieldnames = ["train_task", *TASK_ORDER, "mean_over_eval_tasks"]

    baseline_mean = _read_rows(baseline_root / f"{prefix}_mean.csv")
    mtl_mean = _read_rows(mtl_root / f"{prefix}_mean.csv")
    mean_rows = _sort_rows(baseline_mean + mtl_mean, "train_task")

    baseline_std = _read_rows(baseline_root / f"{prefix}_std.csv")
    mtl_std = _read_rows(mtl_root / f"{prefix}_std.csv")
    std_rows = _sort_rows(baseline_std + mtl_std, "train_task")

    baseline_seed = _read_rows(baseline_root / f"{prefix}_per_seed.csv")
    mtl_seed = _read_rows(mtl_root / f"{prefix}_per_seed.csv")
    seed_rows = sorted(baseline_seed + mtl_seed, key=lambda row: row.get("train_seed", ""))
    seed_fields = ["train_seed", *TASK_ORDER, "mean_over_eval_tasks"]

    _write_rows(out_root / f"combined_{args.metric}_mean.csv", fieldnames, mean_rows)
    _write_rows(out_root / f"combined_{args.metric}_std.csv", fieldnames, std_rows)
    _write_rows(out_root / f"combined_{args.metric}_per_seed.csv", seed_fields, seed_rows)
    _write_markdown(out_root / f"combined_{args.metric}_mean.md", mean_rows)

    with (out_root / f"combined_{args.metric}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metric": args.metric,
                "task_order": TASK_ORDER,
                "row_order": [row["train_task"] for row in mean_rows],
                "mean": mean_rows,
                "std": std_rows,
                "per_seed": seed_rows,
            },
            f,
            indent=2,
        )

    if not args.no_heatmap:
        _write_heatmap(out_root / f"combined_{args.metric}_mean_heatmap.png", mean_rows, args.metric)

    print(f"[DONE] Wrote: {out_root / f'combined_{args.metric}_mean.csv'}")
    print(f"[DONE] Wrote: {out_root / f'combined_{args.metric}_std.csv'}")
    print(f"[DONE] Wrote: {out_root / f'combined_{args.metric}_per_seed.csv'}")
    print(f"[DONE] Wrote: {out_root / f'combined_{args.metric}_mean.md'}")
    print(f"[DONE] Wrote: {out_root / f'combined_{args.metric}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
