# Evaluation Protocol Specification

This document defines how evaluation is run and how matrix outputs are represented.

## Scope

- Benchmarked tasks:
  - `A1_forward`
  - `A2_omni`
  - `B1_rough`
  - `B2_stairs`
  - `C2_gap`
- Each trained policy is evaluated on all 5 tasks (cross-eval row).

## Standard Command

Run cross-eval for one trained checkpoint:

```bash
bash scripts/cross_eval.sh <TRAIN_TASK_GYM_ID> <CHECKPOINT_PATH> [NUM_ENVS] [NUM_EPISODES]
```

Example:

```bash
bash scripts/cross_eval.sh \
  MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0 \
  logs/rsl_rl/unitree_go2_rough/2026-04-09_01-42-49/model_1440.pt \
  64 256
```

## Output Artifacts

For a policy trained on `A1_forward`:

- Combined row summary:
  - `results/A1_forward/summary.json`
- Per-task detailed files:
  - `results/A1_forward/A1_forward.json`
  - `results/A1_forward/A2_omni.json`
  - `results/A1_forward/B1_rough.json`
  - `results/A1_forward/B2_stairs.json`
  - `results/A1_forward/C2_gap.json`

`summary.json` is the canonical row-level matrix representation for that trained policy.

## Matrix Representation

### Row-Level Matrix (Single Trained Policy)

`results/<train_short>/summary.json` maps:

- key = eval task short name
- value = metric bundle (`success_rate`, `failure_rate`, `mean_alive_time_s`, etc.)

This is a 1x5 matrix row for the given trained policy.

### Full Matrix (Multiple Trained Policies)

When you run cross-eval for multiple trained policies, each policy adds one row directory:

- `results/A1_forward/summary.json`
- `results/A2_omni/summary.json`
- `results/B1_rough/summary.json`
- `results/B2_stairs/summary.json`
- `results/C2_gap/summary.json`

Export a consolidated matrix:

```bash
python scripts/export_eval_matrix.py --results_root results --metric success_rate
```

Generated files:

- `results/matrix_success_rate.csv`
- `results/matrix_success_rate.md`
- `results/matrix_success_rate.json`
- `results/matrix_success_rate_heatmap.png` (if `matplotlib` is available)

## Required Reported Metrics

At minimum:

- `success_rate`
- `failure_rate`
- `mean_alive_time_s`
- `mean_success_step_ratio`
- `mean_lin_vel_error`
- `mean_ang_vel_error`
- `num_episodes`

## Phase-Based Model Selection (Current Practice)

For phased MTL training, evaluate checkpoints during each phase and select by task-specific targets instead of using only the final checkpoint.

Selection priorities:

1. Phase 0 and Phase 3 (rough retention focus)
- Primary: `B1_rough.success_rate`
- Gate: keep `B1_rough.success_rate >= 0.933`
- Hard warning: repeated evals below `0.903` or `B1_rough.failure_rate > 0.10`

2. Phase 1 (stairs rehearsal)
- Primary: improve `B2_stairs.success_rate` while keeping B1 above retention gate

3. Phase 2 (gap rehearsal)
- Primary: improve `C2_gap.success_rate` while keeping B1 above retention gate

Practical cadence:

- Run cross-eval every 50 training iterations (or every checkpoint save interval).
- Use two consecutive healthy evals before phase handoff.

## Notes

- Cross-eval is executed per task in separate processes (safer with IsaacLab than in-process task switching).
- `summary.json` is merged incrementally during `cross_eval.sh` runs.
- Mean training reward is useful for trend monitoring, but phase handoff decisions should be based on cross-eval metrics.
