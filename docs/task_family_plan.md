# Task Inventory & Family Plan

> **Robot:** Unitree Go2 (fixed across all tasks — shared 12-DoF obs/action space)
> **Framework:** Isaac Lab · rsl_rl PPO

---

## Task Families

### Family A — Flat-ground velocity tracking

| ID | Task | Terrain | Command profile |
|----|------|---------|----------------|
| A1 | Forward walk | Flat | `lin_vel_x ∈ [0.5, 1.0]`, `lin_vel_y ≈ 0`, `ang_vel_z ≈ 0` |
| A2 | Omni walk | Flat | `lin_vel_x ∈ [-1, 1]`, `lin_vel_y ∈ [-1, 1]`, `heading ∈ [-π, π]` |

### Family B — Rough-terrain velocity tracking

| ID | Task | Terrain | Command profile |
|----|------|---------|----------------|
| B1 | Rough walk | `random_rough` only | same as A2 (omnidirectional) |
| B2 | Stair climb | `pyramid_stairs` only | `lin_vel_x ∈ [0.3, 0.8]`, low lateral/yaw |

### Family C — Agility terrain (custom)

| ID | Task | Terrain | Command profile |
|----|------|---------|----------------|
| C2 | Gap crossing | `MeshGapTerrainCfg` only | `lin_vel_x ∈ [0.4, 0.8]`, low lateral/yaw |

> C1 (stepping stones) was dropped — `HfSteppingStonesTerrainCfg` has no
> reference training implementation in Isaac Lab and proved untrainable.

---

## Gym Registrations

| Gym ID | Env config class | Notes |
|--------|-----------------|-------|
| `MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0` | `Go2A1ForwardWalkEnvCfg` | |
| `MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0` | `Go2A2OmniWalkEnvCfg` | |
| `MTL-Velocity-Rough-Unitree-Go2-B1-v0` | `Go2B1RoughWalkEnvCfg` | |
| `MTL-Velocity-Rough-Unitree-Go2-B2-v0` | `Go2B2StairClimbEnvCfg` | |
| `MTL-Custom-Gap-Unitree-Go2-C2-v0` | `Go2C2GapCrossingEnvCfg` | single-run from scratch |
| `MTL-Custom-Gap-Unitree-Go2-C2-Flat-v0` | `Go2C2FlatPretrainEnvCfg` | Phase 1 pretrain |
| `*-Play-v0` variants | `*_PLAY` classes | 50 envs, no noise/pushes |

---

## Shared Interface

**Action space:** joint position targets, `R^12` (3 joints × 4 legs)

**Observation space (235-dim):**

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

All tasks share identical obs/action dimensions — single policy is feasible.

---

## Reward Structure (all tasks)

All single-task configs use the same B1-proven reward shaping:

| Term | Weight | Notes |
|------|--------|-------|
| `track_lin_vel_xy_exp` | +2.0 | main learning signal |
| `track_ang_vel_z_exp` | +1.0 | |
| `feet_air_time` | +0.25 | restored from Go2 default 0.01 |
| `undesired_contacts` | -1.0 | thigh + calf bodies |
| `flat_orientation_l2` | -5.0 | prevents body tilt |
| `base_height_l2` | -10.0 | terrain-relative via height_scanner |
| `dof_torques_l2` | -1e-5 | restored from Go2 default -2e-4 |

C2 uses `_safe_base_height_l2` (custom wrapper) instead of `core_mdp.base_height_l2`
to filter inf/nan ray-cast hits before averaging.

---

## Success Condition (evaluation)

Per-step success: agent alive **and** tracking errors below threshold:
- `‖v_xy − v_xy_cmd‖₂ ≤ 0.25 m/s`
- `|ω_z − ω_z_cmd| ≤ 0.50 rad/s`

Episode success: success-step ratio ≥ 0.80, no fall termination before timeout.
