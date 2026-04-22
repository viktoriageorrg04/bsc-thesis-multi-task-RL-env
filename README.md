# Multi-Task RL Locomotion Benchmark

BSc Thesis - Data Science and AI  
Author: Viktoria Georgieva

Multi-task reinforcement learning for quadruped locomotion on the Unitree Go2,
built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) with rsl_rl PPO.

## Current Development (April 2026)

- Single-task baselines for A1, A2, B1, B2, C2 are implemented and cross-evaluated.
- Unified MTL environment `MTL-Unified-Unitree-Go2-AllTerrains-v0` is active via `scripts/mtl_train.py`.
- Current finding: pure `uniform` terrain sampling often hurts specialist retention.
- Preferred strategy: phased `focus` sampling with rehearsal, plus explicit B1 retention gates.

See:

- [Task inventory and env IDs](docs/task_family_plan.md)
- [Evaluation protocol](docs/eval_protocol.md)
- [MTL phase playbook and retention thresholds](docs/mtl_phase_playbook.md)

## Tasks

5 single-task baselines across 3 terrain families:

| ID | Task | Terrain | Commands |
|----|------|---------|----------|
| A1 | Forward walk | Flat | forward-only |
| A2 | Omni walk | Flat | omnidirectional |
| B1 | Rough walk | Random rough | omnidirectional |
| B2 | Stair climb | Inverted pyramid stairs (eval task) | forward-only |
| C2 | Gap crossing | Mesh gaps (0-20 cm) | forward-biased |

All tasks share a 235-dim observation space (proprioception + height scan) and
a 12-DoF joint position action space, so one policy can be used for all tasks.

## Project Layout

```text
envs/
  families/
    flat_velocity/      # A1, A2
    rough_velocity/     # B1, B2
    agility_terrain/    # C2
    multi_task/         # unified MTL env
  rewards/

scripts/
  train.py              # single-task + generic train entrypoint
  mtl_train.py          # unified MTL training entrypoint
  evaluate.py           # per-policy cross-eval collection
  cross_eval.sh         # launcher for full 5-task cross-eval
  export_eval_matrix.py # matrix + heatmap export
  plot_mtl_phases.py    # concatenated phase learning curves

docs/
  task_family_plan.md
  eval_protocol.md
  mtl_phase_playbook.md
```

## Quick Start

```bash
# Train unified MTL with rough-focused sampling (recommended phase-0 style)
C:\...\isaaclab.bat -p scripts/mtl_train.py \
  --task MTL-Unified-Unitree-Go2-AllTerrains-v0 \
  --headless --num_envs 512 \
  --sampling_strategy focus --focus_terrain rough --focus_prob 0.75 \
  --pretrained_checkpoint logs/rsl_rl/unitree_go2_rough/<run>/model_1499.pt
```

```bash
# Cross-evaluate one checkpoint on all 5 benchmark tasks
bash scripts/cross_eval.sh \
  MTL-Unified-Unitree-Go2-AllTerrains-v0 \
  logs/rsl_rl/<experiment>/<run>/model_<iter>.pt \
  64 256
```

## Research Questions

- RQ1: Does a unified multi-task policy match specialist single-task policies on each terrain?
- RQ2: Does training across terrain families produce positive cross-family transfer?
