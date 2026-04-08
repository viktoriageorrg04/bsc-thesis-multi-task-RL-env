# Multi-Task RL Locomotion Benchmark

BSc Thesis — Data Science and AI
**Author:** Viktoria Georgieva

Multi-task reinforcement learning for quadruped locomotion on the **Unitree Go2**,
built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) with rsl_rl PPO.

---

## Tasks

5 single-task baselines across 3 terrain families:

| ID | Task | Terrain | Commands | Status |
|----|------|---------|----------|--------|
| A1 | Forward walk | Flat | forward-only `[0.5, 1.0]` m/s
| A2 | Omni walk | Flat | omnidirectional `±1.0` m/s
| B1 | Rough walk | Random rough | omnidirectional `±1.0` m/s
| B2 | Stair climb | Pyramid stairs | forward-biased `[0.3, 0.8]` m/s
| C2 | Gap crossing | Mesh gaps `0–20 cm` | forward-biased `[0.4, 0.8]` m/s

All tasks share a 235-dim observation space (proprioception + height scan) and
a 12-DoF joint position action space — single policy network is feasible.

---

## Project layout

```
envs/
  families/
    flat_velocity/      # A1, A2 — flat terrain configs
    rough_velocity/     # B1, B2 — rough terrain configs
    agility_terrain/    # C2 — gap crossing config + safe base height wrapper
    multi_task/         # unified multi-task env (in progress)
  rewards/              # custom reward terms (stand_terms, safe_base_height)

scripts/
  train.py              # training entry point (PPO via rsl_rl)
  visualize.py          # headless video recording + live playback
  evaluate.py           # evaluation / metrics collection
  plot_learning_curves.py

benchmark/              # multi-task benchmark layer
  interface.py          # standard Gym wrapper
  task_sampler.py       # multi-task episode sampling
  metrics.py            # success rate, transfer metrics

analysis/               # post-hoc analysis scripts
  compare.py            # single-task vs multi-task
  transfer.py           # cross-family transfer
  reward_sensitivity.py

docs/
  task_family_plan.md   # task definitions, reward spec, success condition
  eval_protocol.md      # evaluation protocol spec

videos/                 # recorded evaluation clips (per task)
logs/                   # rsl_rl training checkpoints (gitignored)
```

---

## Quick start

```bash
# activate Isaac Lab conda env
conda activate env_isaaclab

# train a single task (example: C2 gap crossing)
C:\...\isaaclab.bat -p scripts/train.py \
  --task MTL-Custom-Gap-Unitree-Go2-C2-v0 \
  --headless --num_envs 1024 --max_iterations 1500 \
  --plot_learning_curves

# record a video of a trained checkpoint
C:\...\isaaclab.bat -p scripts/visualize.py \
  --task MTL-Custom-Gap-Unitree-Go2-C2-Play-v0 \
  --checkpoint logs/rsl_rl/unitree_go2_rough/<run>/model_1500.pt \
  --video --video_length 500 --headless --enable_cameras \
  --follow --deterministic_eval
```

---

## Research questions

- **RQ1:** Does a unified multi-task policy match specialist single-task policies on each terrain?
- **RQ2:** Does training across terrain families produce positive cross-family transfer?
