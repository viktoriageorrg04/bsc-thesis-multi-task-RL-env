# Multi-Task RL Locomotion Benchmark

BSc Thesis - Data Science and AI  
Author: Viktoria Georgieva

This repository contains a benchmark for multi-task reinforcement learning on
Unitree Go2 locomotion tasks in Isaac Lab. It trains specialist policies and
unified multi-task PPO policies with `rsl_rl`, then cross-evaluates every policy
on the same five-task benchmark matrix.

## Current Scope

The benchmark covers five tasks across three terrain families:

| ID | Short name | Task | Terrain | Commands |
|----|------------|------|---------|----------|
| A1 | `A1_forward` | Forward walk | Flat plane | forward-only |
| A2 | `A2_omni` | Omni walk | Flat plane | omnidirectional |
| B1 | `B1_rough` | Rough walk | Random rough | omnidirectional |
| B2 | `B2_stairs` | Stair climb | Inverted pyramid stairs | forward-only |
| C2 | `C2_gap` | Gap crossing | Mesh gaps, 0-20 cm | forward-biased |

All benchmark tasks use the same Unitree Go2 robot (12-DoF joint position
action space, and a 235-dimensional policy obs space).

## Main Environment IDs

| Purpose | Gym ID |
|---------|--------|
| A1 specialist | `MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0` |
| A2 specialist | `MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0` |
| B1 specialist | `MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0` |
| B2 specialist | `MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0` |
| C2 specialist | `MTL-Custom-Gap-Unitree-Go2-C2-v0` |
| Unified MTL | `MTL-Unified-Unitree-Go2-AllTerrains-v0` |
<!-- | Task-conditioned MTL | `MTL-Conditioned-Unitree-Go2-AllTerrains-v0` | -->

## Repository Layout

```text
envs/
  families/
    flat_velocity/       # A1, A2
    rough_velocity/      # B1, B2
    agility_terrain/     # C2
    multi_task/          # unified and task-conditioned MTL envs
  rewards/
  success.py             # per-task evaluation success thresholds

scripts/
  train.py                       # generic single-task training entrypoint
  mtl_train.py                   # unified/scheduled MTL training entrypoint
  mtl_train_conditioned.py       # conditioned MTL wrapper defaults
  evaluate.py                    # evaluate one checkpoint on one/all tasks
  cross_eval.sh                  # bash cross-eval launcher
  cross_eval_conditioned.cmd     # Windows conditioned-MTL cross-eval launcher
  export_eval_matrix.py          # matrix CSV/JSON/Markdown/heatmap export
  aggregate_seeded_eval.py       # aggregate seeded evals into mean/std matrices
  combine_aggregated_eval.py     # combine specialist and MTL aggregate outputs
  plot_learning_curves.py
  plot_mtl_phases.py

docs/
  task_family_plan.md
  eval_protocol.md
  mtl_phase_playbook.md
```

## Setup

Isaac Lab and Isaac Sim are expected to be installed separately. From the
Python environment used by Isaac Lab, install this repo in editable mode:

```bash
python -m pip install -e .
```

For development utilities:

```bash
python -m pip install -e ".[dev]"
```

Most training and evaluation commands should be launched through
`isaaclab.bat -p` on Windows. If you use `scripts/cross_eval.sh`, set
`ISAACLAB_BAT` when the default local path does not match your machine.

```bash
export ISAACLAB_BAT=/c/Users/<user>/path/to/IsaacLab/isaaclab.bat
```

## Training

Train a specialist baseline with `scripts/train.py`:

```powershell
& "C:\path\to\IsaacLab\isaaclab.bat" -p scripts/train.py `
  --task MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0 `
  --headless --num_envs 1024 --max_iterations 1500 `
  --experiment_name unitree_go2_b1_baseline
```

Train the unified MTL policy with terrain sampling control:

```powershell
& "C:\path\to\IsaacLab\isaaclab.bat" -p scripts/mtl_train.py `
  --task MTL-Unified-Unitree-Go2-AllTerrains-v0 `
  --headless --num_envs 4096 --max_iterations 400 `
  --sampling_strategy focus --focus_terrain rough --focus_prob 0.75 `
  --pretrained_checkpoint logs/rsl_rl/unitree_go2_rough/<run>/model_1499.pt `
  --experiment_name unitree_go2_mtl_schedule --run_name p0_rough_anchor
```

<!-- Train the task-conditioned MTL variant:

```powershell
& "C:\path\to\IsaacLab\isaaclab.bat" -p scripts/mtl_train_conditioned.py `
  --headless --num_envs 512 --max_iterations 1500 `
  --sampling_strategy uniform --seed 0 -->
```

Training logs and checkpoints are written under:

```text
logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
```

## Evaluation

Evaluate one checkpoint on all five benchmark tasks:

```bash
bash scripts/cross_eval.sh \
  MTL-Unified-Unitree-Go2-AllTerrains-v0 \
  logs/rsl_rl/<experiment>/<run>/model_<iter>.pt \
  64 256
```

Evaluate a task-conditioned checkpoint from Windows:

```bat
scripts\cross_eval_conditioned.cmd ^
  logs\rsl_rl\<experiment>\<run>\model_<iter>.pt ^
  256 64 results
```

The evaluator writes one row per trained policy:

```text
results/<train_short>/summary.json
results/<train_short>/<eval_task>.json
```

Export a consolidated metric matrix:

```bash
python scripts/export_eval_matrix.py --results_root results --metric success_rate
```

For seeded experiments, aggregate and combine results:

```bash
python scripts/aggregate_seeded_eval.py --results_root results_seeded_1024 --metric success_rate
python scripts/combine_aggregated_eval.py \
  --baseline_root results_seeded_1024 \
  --mtl_root results_seeded_4096 \
  --metric success_rate \
  --out_root results_seeded_combined
```

## Current Training Practice

Recent MTL work uses phased `focus` or `custom` sampling via `scripts/mtl_train.py`
with `--phase_profile` for per-phase command/reward overrides:

- Phase 0: rough anchor: prioritize `rough`, retain B1
- Phase 1: stairs rehearsal: prioritize `stairs`, improve B2
- Phase 2: gap rehearsal: prioritize `gap`, improve C2
- Phase 3: rough recovery: re-anchor `rough` after transfer phases

Checkpoint selection should be based on cross-evaluation metrics.

## Documentation

- [Task inventory and environment IDs](docs/task_family_plan.md)
- [Evaluation protocol](docs/eval_protocol.md)
<!-- - [MTL phase playbook and retention thresholds](docs/mtl_phase_playbook.md) -->

## Research Questions

- RQ1: Can a unified multi-task policy match specialist policies on each terrain?
- RQ2: Does training across terrain families produce positive cross-family transfer?
