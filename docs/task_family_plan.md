# Task Inventory & Family Plan

> **Goal:** minimal-but-sufficient set of locomotion tasks, organized into
> families, to support a multi-task RL study on Isaac Lab.

---

## 1. What Isaac Lab gives us out of the box

Isaac Lab ships **one locomotion task type**: **velocity tracking**
(`LocomotionVelocityRoughEnvCfg`).  Every registered Gym env is a
`(robot × terrain)` combination of that single task type.

### 1a. Registered Gym environments (training variants only, `-Play` excluded)

| Robot | Flat | Rough |
|-------|:----:|:-----:|
| Anymal-B | ✅ | ✅ |
| Anymal-C | ✅ | ✅ |
| Anymal-D | ✅ | ✅ |
| Unitree A1 | ✅ | ✅ |
| Unitree Go1 | ✅ | ✅ |
| Unitree Go2 | ✅ | ✅ |
| Unitree G1 (humanoid) | ✅ | ✅ |
| Unitree H1 (humanoid) | ✅ | ✅ |
| Cassie | ✅ | ✅ |
| Digit | ✅ | ✅ |
| Spot | ✅ | — |

**22 training envs**, but they all do the same thing:
track a sampled (lin_vel_x, lin_vel_y, ang_vel_z) command.

### 1b. Terrain sub-types already defined in the `ROUGH_TERRAINS_CFG`

Used by every `*RoughEnvCfg`:

| Key in config | Generator class | What it looks like |
|---|---|---|
| `pyramid_stairs` | `MeshPyramidStairsTerrainCfg` | ascending/descending stair pyramids |
| `pyramid_stairs_inv` | `MeshInvertedPyramidStairsTerrainCfg` | descending-first stair pyramids |
| `boxes` | `MeshRandomGridTerrainCfg` | random box grid (rubble) |
| `random_rough` | `HfRandomUniformTerrainCfg` | heightfield noise |
| `hf_pyramid_slope` | `HfPyramidSlopedTerrainCfg` | sloped ramps up |
| `hf_pyramid_slope_inv` | `HfInvertedPyramidSlopedTerrainCfg` | sloped ramps down |

Additional **mesh** terrain types available but unused in default config:
`MeshRailsTerrainCfg`, `MeshPitTerrainCfg`, `MeshBoxTerrainCfg`,
`MeshGapTerrainCfg`, `MeshFloatingRingTerrainCfg`, `MeshStarTerrainCfg`,
`MeshRepeatedPyramidsTerrainCfg`, `MeshRepeatedBoxesTerrainCfg`,
`MeshRepeatedCylindersTerrainCfg`.

Additional **heightfield** types available but unused:
`HfPyramidStairsTerrainCfg`, `HfInvertedPyramidStairsTerrainCfg`,
`HfDiscreteObstaclesTerrainCfg`, `HfWaveTerrainCfg`,
`HfSteppingStonesTerrainCfg`.

### 1c. Command interface

`UniformVelocityCommandCfg` with knobs:
- `lin_vel_x`, `lin_vel_y`, `ang_vel_z` ranges
- `heading_command` (bool) — heading-based vs direct angular vel
- `rel_standing_envs` — fraction of envs that receive zero command

### 1d. What Isaac Lab does NOT give us

- Only velocity-tracking tasks — no stand, turn-in-place, or stop-and-balance tasks.
- No explicit "task family" abstraction — terrain is a curriculum detail,
  not a registered task axis.
- No multi-task sampling, task-ID conditioning, or success metrics.
- No built-in task that isolates a single terrain type (e.g. stairs-only).

---

## 2. Key design decision: **fix the robot**

Changing the robot changes obs/action dimensions and dynamics, which breaks
the "shared interface" requirement of multi-task RL.

**Recommendation -> pick one quadruped robot** (e.g. **Unitree Go2** or
**Anymal-D** — both well-supported, 12 DoF, quadruped).  All tasks below
assume one fixed morphology.

> *Rationale:* the study is about multi-task generalization **across task
> conditions**, not across robot morphologies. Fixing the robot lets us use a
> single policy network with a shared obs/action space, which is the standard
> multi-task RL setup.

---

## 3. Proposed task families

### Design principles
1. **Each task = a specific (command profile × terrain) pair** that can be
   independently registered as a Gym env.
2. **Families group tasks by the dominant skill** they require, not by a
   single hyperparameter knob.
3. **Minimal set:** 3 families × 2 tasks = **6 tasks** (expandable to 3 × 3
   = 9 if time allows).

---

### Family A — Flat-ground velocity tracking (baseline locomotion skills)

| Task | Terrain | Command profile | What it tests |
|------|---------|----------------|---------------|
| A1: **Forward walk** | flat | lin_vel_x ∈ [0.5, 1.0], lin_vel_y ≈ 0, ang_vel_z ≈ 0 | basic forward gait |
| A2: **Omnidirectional walk** | flat | lin_vel_x ∈ [-1, 1], lin_vel_y ∈ [-1, 1], heading ∈ [-π, π] | lateral & turning gait |

**Reuse:** directly derived from the existing `*FlatEnvCfg` — only the
command ranges change.

*Optional A3:* **Stand still** — command ≈ 0, reward for low joint velocity
and upright posture. (Needs a small reward tweak; no terrain work.)

---

### Family B — Rough-terrain velocity tracking (balance & foot-placement skills)

| Task | Terrain | Command profile | What it tests |
|------|---------|----------------|---------------|
| B1: **Rough walk** | `random_rough` (HfRandomUniform, default params) | same omni-cmd as A2 | proprioceptive balance |
| B2: **Stair climb** | `pyramid_stairs` only (isolate from mix) | forward-biased cmd: lin_vel_x ∈ [0.3, 0.8] | swing-foot timing, step clearance |

**Reuse:** derived from existing `*RoughEnvCfg`, but with the
`TerrainGeneratorCfg.sub_terrains` dict **trimmed to a single entry** so
each task uses exactly one terrain type.

*Optional B3:* **Slope traverse** — `hf_pyramid_slope` only, forward
command. Tests incline adaptation.

---

### Family C — Challenging / stress-test terrains (custom contribution)

| Task | Terrain | Command profile | What it tests |
|------|---------|----------------|---------------|
| C1: **Stepping stones** | `HfSteppingStonesTerrainCfg` (available but unused in defaults) | forward cmd, moderate speed | precise foot placement |
| C2: **Gap crossing** | `MeshGapTerrainCfg` (available but unused) | forward cmd, moderate speed | stride planning, balance recovery |

**What's needed:** write a new env config per task that plugs the chosen
terrain type into the same `LocomotionVelocityRoughEnvCfg` base.  This is a
config-only change (instantiate the same base class with a different
`sub_terrains` dict).  No new reward terms, obs, or actions required.

*Optional C3:* **Low-friction flat** — flat terrain with
`dynamic_friction` set to e.g. 0.2 via the existing `EventTerm`
(`randomize_rigid_body_material`). Tests robustness / sim2real readiness.

---

## 4. What needs to be built (ordered by effort)

| # | Item | Effort | What exists | What to add |
|---|------|--------|-------------|-------------|
| 1 | **Task configs for A1, A2** | Low | `FlatEnvCfg` exists | Override command ranges only |
| 2 | **Task configs for B1, B2** | Low | `RoughEnvCfg` exists | Override `sub_terrains` to single type + command ranges |
| 3 | **Task configs for C1, C2** | Low–Med | Terrain generators exist, env base exists | New env cfg combining base class + single terrain entry; may need param tuning (gap width, stone size) |
| 4 | **Gym registration** | Low | Pattern exists (see `__init__.py` files) | Register 6 new env IDs |
| 5 | **Success metric** | Med | Not built-in | Define per-family success: e.g. "velocity error < ε for ≥ T seconds while alive" |
| 6 | **Multi-task sampler** | Med | Not built-in | Wrapper that samples task at episode start; optionally appends task-ID to obs |
| 7 | **Eval harness** | Med | Not built-in | Runs trained policy on each task independently; logs success rates |

Items 1-4 are config work.  Items 5-7 are the benchmark-layer work (your
main contribution).

---

## 5. Summary for the meeting

```
                    Family A              Family B              Family C
                  (flat ground)        (rough terrain)       (stress-test)
                ┌──────────────┐    ┌──────────────────┐   ┌────────────────┐
  Tasks         │ A1: fwd walk │    │ B1: rough walk   │   │ C1: stepping   │
  (required)    │ A2: omni walk│    │ B2: stair climb  │   │     stones     │
                └──────────────┘    └──────────────────┘   │ C2: gap cross  │
                                                           └────────────────┘
                ┌──────────────┐    ┌──────────────────┐   ┌────────────────┐
  Tasks         │ A3: stand    │    │ B3: slope walk   │   │ C3: low-fric   │
  (optional)    │     still    │    │                  │   │     flat       │
                └──────────────┘    └──────────────────┘   └────────────────┘

  Robot: fixed (Unitree Go2 or Anymal-D)
  Interface: shared obs (proprioception + height scan) / shared 12-DoF action
  Algo: any (PPO, SAC, …) via standard Gym API
  Eval: success-based (vel tracking error threshold)
```

**6 tasks (+ 3 optional) is minimal for multi-task RL:**
- 2 tasks per family → within-family transfer
- 3 families → cross-family transfer
- All share one obs/action space → single policy feasible
- Task-ID vs No-Task-ID ablation possible
- Scaling test: train on 2 → 4 → 6 tasks, measure degradation

**No new physics, reward terms, or observation types needed.** Everything
is assembled from existing Isaac Lab building blocks; the novelty is in the
benchmark organization and evaluation protocol.

## 6. Minimal Benchmark Spec (v0.1, Phase 1 Freeze)

### 6.1 Fixed robot and scope
- Fixed robot: **Unitree Go2**.
- Benchmark baseline scope: **6 tasks** across 3 families:
  - Family A: A1 Forward Walk, A2 Omni Walk
  - Family B: B1 Rough Walk, B2 Stair Climb
  - Family C: C1 Stepping Stones, C2 Gap Crossing
- Out of scope for this milestone: optional A3/B3/C3 and multi-robot variants.

### 6.2 Shared interface (all tasks)

#### Action space
- Control mode: joint position action (Isaac Lab `JointPositionActionCfg`).
- Controlled joints: all Go2 leg joints (12 total: 3 per leg x 4 legs).
- Action vector: `a_t in R^12`, normalized in policy space, then scaled by env config.

#### Observation space
- Core proprioception terms:
  - base linear velocity
  - base angular velocity
  - projected gravity
  - commanded base velocity
  - relative joint positions
  - relative joint velocities
  - previous action
- Terrain term:
  - `height_scan` for rough/custom tasks.
- Shared shape policy:
  - when a task has no scanner (flat), use a zero-filled `height_scan` placeholder in the benchmark wrapper so all tasks expose the same observation dimension.

### 6.3 Task definitions (Phase 1 defaults)

| Task | Terrain | Command ranges |
|---|---|---|
| A1 Forward Walk | plane | `lin_vel_x in [0.5, 1.0]`, `lin_vel_y = 0`, `ang_vel_z = 0` |
| A2 Omni Walk | plane | `lin_vel_x in [-1.0, 1.0]`, `lin_vel_y in [-1.0, 1.0]`, heading/turn full-range |
| B1 Rough Walk | `random_rough` only | same as A2 |
| B2 Stair Climb | `pyramid_stairs` only | forward-biased: `lin_vel_x in [0.3, 0.8]`, low lateral/yaw |
| C1 Stepping Stones | `HfSteppingStonesTerrainCfg` only | forward moderate speed, low lateral/yaw |
| C2 Gap Crossing | `MeshGapTerrainCfg` only | forward moderate speed, low lateral/yaw |

Notes:
- B/C tasks isolate a single `sub_terrain` key each (no mixed terrain set).
- C1/C2 are config-level extensions on top of the same locomotion base env.

### 6.4 Success condition (benchmark-level, not reward-level)
Define per-step success signal:
- agent is alive (no terminal fall/contact),
- planar tracking error below threshold: `||v_xy - v_xy_cmd||_2 <= eps_lin`,
- yaw-rate tracking error below threshold: `|w_z - w_z_cmd| <= eps_ang`.

Phase 1 default thresholds:
- `eps_lin = 0.25 m/s`
- `eps_ang = 0.50 rad/s`

Episode success:
- success-step ratio >= `0.80` over the episode
- and no failure termination before timeout.

### 6.5 Termination criteria
Use shared termination logic:
- timeout at episode horizon (`20 s` default),
- illegal base contact / fall,
- optional terrain out-of-bounds (enabled for generated terrains).

### 6.6 Required logged stats per episode
- `episode_success` (bool)
- `success_step_ratio`
- `mean_lin_vel_error_xy`
- `mean_ang_vel_error_z`
- `alive_time_s`
- `termination_reason`
- `task_id`, `family_id`
