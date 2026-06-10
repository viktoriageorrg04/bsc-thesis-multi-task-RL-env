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

## Available `--phase_profile` Values

The `--phase_profile` flag (on `scripts/mtl_train.py`) applies per-terrain command
and reward overrides on top of the terrain sampling proportions set by
`--sampling_strategy`. Available profiles:

| Profile | Description |
|---------|-------------|
| `default` / `p0_rough` | No overrides; unified env defaults |
| `p0_gait` | Level-0 gait bootstrap with easier forward-biased commands |
| `p1_b2_easy` | Easy corrected-stairs bridge after `p0_gait` |
| `p1_b2_stepup` | Reduced step-height B2 curriculum |
| `p1_b2_stepup_retain` | Reduced-height B2 with broad flat/rough rehearsal |
| `p1_b2_ramp` | Stair-heavy transition bridge from easy B2 to benchmark B2 |
| `p1_mixed` | Intermediate command/terrain curriculum after `p0_gait` |
| `p1_omni` | Bridge from `p1_mixed` toward full omni/heading eval commands |
| `p1_stairs` | Stair-biased commands + reward tuning for B2 rehearsal stability |
| `p2_b2safe` | Balanced recovery while retaining a small B2 progress signal |

## Example Commands

```bash
# Phase 0 — rough anchor (focus sampling + no profile overrides)
C:\...\isaaclab.bat -p scripts/mtl_train.py \
  --task MTL-Unified-Unitree-Go2-AllTerrains-v0 \
  --headless --num_envs 1024 --max_iterations 1500 \
  --pretrained_checkpoint logs/rsl_rl/unitree_go2_rough/<run>/model_1499.pt \
  --sampling_strategy custom --custom_terrain_probs "0.35 0.35 0.15 0.15" \
  --phase_profile p2_b2safe \
  --learning_rate 3e-4 --entropy_coef 0.001 \
  --experiment_name unitree_go2_mtl_schedule --run_name p0_balanced_b2safe
```

```bash
# Cross-eval current phase checkpoint
bash scripts/cross_eval.sh \
  MTL-Unified-Unitree-Go2-AllTerrains-v0 \
  logs/rsl_rl/unitree_go2_mtl_schedule/<run>/model_<iter>.pt \
  64 256
```
