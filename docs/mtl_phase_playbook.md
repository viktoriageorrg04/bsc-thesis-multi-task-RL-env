# MTL Phase Playbook

This document describes the current training strategy for improving multi-task capability while preserving B1 rough-walk performance.

## Why Phase Training

Uniform sampling (`0.25/0.25/0.25/0.25`) is simple but often causes specialist skill interference. In recent runs, B1 remained strong while B2/C2 degraded, and final checkpoints sometimes regressed after earlier peaks.

## Current Recommended Schedule

Use `scripts/mtl_train.py` with `sampling_strategy=focus` and checkpoint handoff between phases.

1. Phase 0 (`p0_rough_anchor`)
- Goal: retain B1 while adapting to unified terrain sampler.
- Suggested sampling: `focus_terrain=rough`, `focus_prob=0.75` to `0.80`.
- Suggested length: 300-450 iterations.

2. Phase 1 (`p1_stairs_rehearsal`)
- Goal: add stair skill without collapsing rough gait.
- Suggested sampling: `focus_terrain=stairs`, `focus_prob=0.60`.
- Suggested length: 200-300 iterations.

3. Phase 2 (`p2_gap_rehearsal`)
- Goal: add gap recovery and step planning.
- Suggested sampling: `focus_terrain=gap`, `focus_prob=0.60`.
- Suggested length: 200-300 iterations.

4. Phase 3 (`p3_rough_recover`)
- Goal: reconsolidate B1 performance after transfer phases.
- Suggested sampling: `focus_terrain=rough`, `focus_prob=0.80`.
- Suggested length: 150-250 iterations.

## Phase-0 Health Criteria (B1 Retention)

Baseline anchor (from current `results/B1_rough/summary.json`):
- `B1_success_rate = 0.9535`
- `B1_failure_rate = 0.0465`

Expected behavior in phase 0:
- Small early dip is normal (roughly 3-5 percentage points in success rate).
- Recovery should happen before phase-end if training is healthy.

Retention gates:
- Healthy end-of-phase target: `B1_success_rate >= 0.933` (baseline minus 2 points).
- Preferred target: `B1_success_rate >= 0.953` (baseline match).
- Red flag: repeated evals below `0.903` (baseline minus 5 points).
- Red flag: `B1_failure_rate > 0.10`.

## Checkpoint Selection Rule

Do not select the last checkpoint by default.

Use cross-eval to choose the best checkpoint per phase:
- Primary metric in phase 0 and phase 3: `B1_rough.success_rate`
- Secondary tie-breakers: lower `B1_rough.failure_rate`, then higher `C2_gap` or `B2_stairs` depending on phase goal.

Practical cadence:
- Evaluate every 50 iterations.
- Promote checkpoint to next phase only after two stable evals near the target.

## Reading Learning Curves

`Train/mean_reward` around `27-28` can be fine in unified runs.

Interpretation rules:
- Absolute reward value alone is not sufficient.
- Watch for peak-then-decline behavior across many checkpoints.
- Trust cross-eval metrics over reward when deciding phase handoff.

## Example Commands

```bash
# Phase 0
C:\...\isaaclab.bat -p scripts/mtl_train.py \
  --task MTL-Unified-Unitree-Go2-AllTerrains-v0 \
  --headless --num_envs 512 --max_iterations 400 \
  --pretrained_checkpoint logs/rsl_rl/unitree_go2_rough/<run>/model_1499.pt \
  --sampling_strategy focus --focus_terrain rough --focus_prob 0.75 \
  --noise_std_type scalar --init_noise_std 1.0 \
  --learning_rate 1e-4 --schedule adaptive --entropy_coef 0.01 \
  --experiment_name unitree_go2_mtl_schedule_v3 --run_name p0_rough_anchor
```

```bash
# Cross-eval current phase checkpoint
bash scripts/cross_eval.sh \
  MTL-Unified-Unitree-Go2-AllTerrains-v0 \
  logs/rsl_rl/unitree_go2_mtl_schedule_v3/<run>/model_<iter>.pt \
  64 256
```
