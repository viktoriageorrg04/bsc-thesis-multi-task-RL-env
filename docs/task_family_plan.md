# Task Inventory and Family Plan

Robot: Unitree Go2 (fixed across all tasks, shared 12-DoF action space)
Framework: Isaac Lab + rsl_rl PPO

## Task Families

### Family A - Flat velocity tracking

| ID | Task | Terrain | Command Profile |
|----|------|---------|-----------------|
| A1 | Forward walk | Flat plane | forward-only |
| A2 | Omni walk | Flat plane | omnidirectional |

### Family B - Rough velocity tracking

| ID | Task | Terrain | Command Profile |
|----|------|---------|-----------------|
| B1 | Rough walk | `random_rough` | omnidirectional |
| B2 | Stair climb | `pyramid_stairs_inv` (eval) | forward-only |

### Family C - Agility terrain

| ID | Task | Terrain | Command Profile |
|----|------|---------|-----------------|
| C2 | Gap crossing | `gap` (`0.0-0.20 m`) | forward-biased |

## Gym Registrations

| Gym ID | Env Config Class |
|--------|------------------|
| `MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0` | `Go2A1ForwardWalkEnvCfg` |
| `MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0` | `Go2A2OmniWalkEnvCfg` |
| `MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0` | `Go2B1RoughWalkEnvCfg` |
| `MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0` | `Go2B2StairClimbEnvCfg` |
| `MTL-Custom-Gap-Unitree-Go2-C2-v0` | `Go2C2GapCrossingEnvCfg` |
| `MTL-Custom-Gap-Unitree-Go2-C2-Flat-v0` | `Go2C2FlatPretrainEnvCfg` |
| `MTL-Unified-Unitree-Go2-AllTerrains-v0` | `Go2MultiTaskEnvCfg` |
| `*-Play-v0` variants | `*_PLAY` classes |

## Shared Interface

Action space:
- 12 joint position targets (3 joints x 4 legs)

Observation space (235-dim):

| Term | Dims |
|------|------|
| base linear velocity | 3 |
| base angular velocity | 3 |
| projected gravity | 3 |
| velocity commands | 3 |
| joint positions (relative) | 12 |
| joint velocities (relative) | 12 |
| previous actions | 12 |
| height scan | 187 |

All tasks share identical observation and action dimensions.

## Reward Design

Single-task and unified MTL configs are based on B1-proven shaping, with task-specific overrides where needed.

Core terms used across tasks:

| Term | Typical Weight | Notes |
|------|----------------|-------|
| `track_lin_vel_xy_exp` | +2.0 | primary tracking signal |
| `track_ang_vel_z_exp` | +1.0 | yaw tracking |
| `feet_air_time` | 0.016 to 0.5 | task-dependent stepping incentive |
| `undesired_contacts` | -1.0 to -0.45 | thigh/calf contact penalty |
| `flat_orientation_l2` | -5.0 to -2.0 | posture control |
| `base_height` | task-dependent | terrain-relative target |
| `dof_torques_l2` | task-dependent | regularization |

Notes:
- C2 and unified MTL use a safe base-height wrapper to handle invalid height-scanner rays on gaps.
- B2 has task-specific command and reward tuning different from B1/unified defaults.

## Curriculum Notes

There are two separate curriculum controls in the codebase:

1. Terrain-level curriculum manager (`self.curriculum.terrain_levels`)
- `None` means difficulty is not adaptively advanced by the curriculum manager.

2. Terrain generator curriculum (`terrain_generator.curriculum`)
- `False` disables generator-side curriculum behavior.

Locking to easiest terrain requires both a disabled progression path and `max_init_terrain_level = 0`.

## Success Conditions

Evaluation uses per-task thresholds from `envs/success.py`.

General pattern:
- Per-step success: alive + tracking errors under task threshold.
- Episode success: success-step ratio above task minimum and no failure termination.

See `docs/eval_protocol.md` for reporting schema and matrix export rules.
