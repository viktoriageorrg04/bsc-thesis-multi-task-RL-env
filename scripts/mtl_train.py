"""Unified multi-task training driver with stability defaults.

This script is intentionally separate from scripts/train.py so single-task
baseline training behavior remains unchanged.

Usage:
    isaaclab.bat -p scripts/mtl_train.py --headless --num_envs 1024 --max_iterations 1500
    isaaclab.bat -p scripts/mtl_train.py --headless --sampling_strategy focus --focus_terrain stairs --focus_prob 0.7
"""

from __future__ import annotations

import argparse
import json
import importlib.metadata as metadata
import os
import random
import sys
import time
from dataclasses import MISSING
from datetime import datetime
from glob import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher

import numpy as np


_FOCUS_TERRAIN_TO_SUBTERRAIN = {
    "flat": "flat",
    "rough": "random_rough",
    "stairs": "pyramid_stairs_inv",
    "gap": "gap",
}


def _is_missing(value) -> bool:
    return isinstance(value, type(MISSING))


def _has_non_missing_attr(obj, attr_name: str) -> bool:
    return hasattr(obj, attr_name) and not _is_missing(getattr(obj, attr_name))


def _update_agent_cfg_from_cli(agent_cfg: RslRlBaseRunnerCfg, args) -> RslRlBaseRunnerCfg:
    if args.seed is not None:
        agent_cfg.seed = random.randint(0, 10000) if args.seed == -1 else args.seed
    if args.resume:
        agent_cfg.resume = True
    if args.load_run is not None:
        agent_cfg.load_run = args.load_run
    if args.checkpoint is not None:
        agent_cfg.load_checkpoint = args.checkpoint
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name
    if args.logger is not None:
        agent_cfg.logger = args.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args.log_project_name:
        agent_cfg.wandb_project = args.log_project_name
        agent_cfg.neptune_project = args.log_project_name
    return agent_cfg


def _build_terrain_sampling_proportions(
    strategy: str,
    focus_terrain: str | None,
    focus_prob: float,
    custom_terrain_probs: list[float] | None = None,
) -> dict[str, float]:
    """Create per-subterrain sampling probabilities.

    Strategies:
      - uniform: 25/25/25/25
      - focus: one terrain gets `focus_prob`, remainder split over others
      - custom: explicit flat rough stairs gap probabilities
    """
    if strategy == "uniform":
        return {key: 0.25 for key in _FOCUS_TERRAIN_TO_SUBTERRAIN.values()}

    if strategy == "custom":
        if custom_terrain_probs is None or len(custom_terrain_probs) != 4:
            raise ValueError("--custom_terrain_probs requires exactly four values: flat rough stairs gap.")
        probs = [float(v) for v in custom_terrain_probs]
        if any(v < 0.0 for v in probs) or sum(probs) <= 0.0:
            raise ValueError("--custom_terrain_probs must be non-negative and sum to a positive value.")
        total = sum(probs)
        terrain_keys = [_FOCUS_TERRAIN_TO_SUBTERRAIN[name] for name in ("flat", "rough", "stairs", "gap")]
        return {key: prob / total for key, prob in zip(terrain_keys, probs, strict=True)}

    if strategy != "focus":
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    if focus_terrain is None:
        raise ValueError("--focus_terrain is required when --sampling_strategy focus")

    focus_key = _FOCUS_TERRAIN_TO_SUBTERRAIN[focus_terrain]
    focus_prob = float(focus_prob)
    if not (0.25 <= focus_prob <= 0.97):
        raise ValueError("--focus_prob must be in [0.25, 0.97] for stable rehearsal sampling.")

    other_keys = [k for k in _FOCUS_TERRAIN_TO_SUBTERRAIN.values() if k != focus_key]
    rehearse_prob_each = (1.0 - focus_prob) / len(other_keys)
    proportions = {k: rehearse_prob_each for k in other_keys}
    proportions[focus_key] = focus_prob
    return proportions


def _apply_terrain_sampling_profile(env_cfg: ManagerBasedRLEnvCfg, args) -> dict[str, float]:
    """Apply terrain sampling profile directly to the unified MTL terrain generator."""
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    terrain_gen = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_gen is None or not hasattr(terrain_gen, "sub_terrains"):
        raise ValueError("Expected a terrain generator with sub_terrains for unified MTL training.")

    proportions = _build_terrain_sampling_proportions(
        strategy=args.sampling_strategy,
        focus_terrain=args.focus_terrain,
        focus_prob=args.focus_prob,
        custom_terrain_probs=args.custom_terrain_probs,
    )

    missing = [k for k in proportions if k not in terrain_gen.sub_terrains]
    if missing:
        raise ValueError(
            f"Terrain generator is missing expected sub-terrains: {missing}. "
            "Use this script with MTL-Unified-Unitree-Go2-AllTerrains-v0."
        )

    for key, prob in proportions.items():
        terrain_gen.sub_terrains[key].proportion = float(prob)

    pretty = ", ".join(f"{k}={proportions[k]:.3f}" for k in sorted(proportions))
    print(f"[INFO] Terrain sampling profile ({args.sampling_strategy}): {pretty}")
    return proportions


def _apply_phase_profile_overrides(env_cfg: ManagerBasedRLEnvCfg, phase_profile: str) -> dict[str, object]:
    """Apply optional phase-specific command/reward overrides.

    Profiles:
      - default / p0_rough: keep unified env defaults (backward-compatible)
      - p0_gait: level-0 gait bootstrap with easier forward-biased commands
      - p1_b2_easy: easy corrected-stairs bridge after p0_gait
      - p1_b2_stepup: reduced-height B2 contact/step-up curriculum
      - p1_b2_stepup_retain: reduced-height B2 bridge with broad flat/rough rehearsal
      - p1_b2_ramp: stair-heavy transition bridge from easy B2 to benchmark B2
      - p1_mixed: intermediate command/terrain curriculum after p0_gait
      - p1_omni: bridge from p1_mixed toward full omni/heading eval commands
      - p2_b2safe: balanced recovery while retaining a small B2 progress signal
      - p1_stairs: stair-biased commands + reward tuning to improve B2 rehearsal stability
    """
    profile = phase_profile or "default"
    if profile in {"default", "p0_rough"}:
        print(f"[INFO] Phase profile: {profile} (no additional cmd/reward overrides).")
        return {"profile": profile, "overrides_applied": False}

    if profile == "p0_gait":
        setattr(env_cfg, "mtl_command_profile", "p0_gait")
        terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
        if terrain is not None:
            terrain.max_init_terrain_level = 0
        curriculum = getattr(env_cfg, "curriculum", None)
        if curriculum is not None and hasattr(curriculum, "terrain_levels"):
            curriculum.terrain_levels = None

        cmd = env_cfg.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0

        print("[INFO] Phase profile: p0_gait (level-0 command bootstrap applied).")
        return {
            "profile": profile,
            "overrides_applied": True,
            "mtl_command_profile": "p0_gait",
            "max_init_terrain_level": 0,
            "terrain_curriculum_enabled": False,
            "commands": {
                "flat_rough": {"lin_vel_x": [0.2, 0.6], "lin_vel_y": [-0.15, 0.15], "ang_vel_z": [-0.2, 0.2]},
                "stairs": {"lin_vel_x": [0.35, 0.7], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]},
                "gap": {"lin_vel_x": [0.35, 0.7], "lin_vel_y": [-0.05, 0.05], "ang_vel_z": [-0.1, 0.1]},
            },
        }

    if profile == "p1_b2_easy":
        setattr(env_cfg, "mtl_command_profile", "p1_b2_easy")
        terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
        if terrain is not None:
            terrain.max_init_terrain_level = 0

        cmd = env_cfg.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0

        reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
        if reset_base is not None and isinstance(getattr(reset_base, "params", None), dict):
            pose_range = reset_base.params.get("pose_range")
            if isinstance(pose_range, dict):
                pose_range["yaw"] = (-0.2, 0.2)

        print("[INFO] Phase profile: p1_b2_easy (easy corrected-stairs bridge applied).")
        return {
            "profile": profile,
            "overrides_applied": True,
            "mtl_command_profile": "p1_b2_easy",
            "max_init_terrain_level": 0,
            "terrain_curriculum_enabled": True,
            "reset_yaw": [-0.2, 0.2],
            "commands": {
                "flat_rough": {"lin_vel_x": [0.15, 0.55], "lin_vel_y": [-0.2, 0.2], "ang_vel_z": [-0.25, 0.25]},
                "stairs": {"lin_vel_x": [0.2, 0.45], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]},
                "gap": {"lin_vel_x": [0.3, 0.6], "lin_vel_y": [-0.05, 0.05], "ang_vel_z": [-0.1, 0.1]},
            },
        }

    if profile == "p1_b2_stepup":
        setattr(env_cfg, "mtl_command_profile", "p1_b2_stepup")
        setattr(env_cfg, "mtl_stair_forward_progress_weight", 0.75)
        terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
        terrain_gen = getattr(terrain, "terrain_generator", None)
        if terrain is not None:
            terrain.max_init_terrain_level = 0
        if terrain_gen is not None and hasattr(terrain_gen, "sub_terrains"):
            stairs = terrain_gen.sub_terrains.get("pyramid_stairs_inv")
            if stairs is not None and hasattr(stairs, "step_height_range"):
                stairs.step_height_range = (0.02, 0.08)

        cmd = env_cfg.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0

        reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
        if reset_base is not None and isinstance(getattr(reset_base, "params", None), dict):
            pose_range = reset_base.params.get("pose_range")
            if isinstance(pose_range, dict):
                pose_range["x"] = (-0.25, 0.5)
                pose_range["y"] = (-0.25, 0.25)
                pose_range["yaw"] = (-0.1, 0.1)

        # B2 is currently stalling at the stair face.  Temporarily make foot
        # lifting and forward progress more valuable only in this named phase.
        rw = getattr(env_cfg, "rewards", None)
        feet_air = getattr(rw, "feet_air_time", None) if rw is not None else None
        if feet_air is not None and isinstance(getattr(feet_air, "params", None), dict):
            values = list(feet_air.params.get("values", (0.25, 0.25, 0.02, 0.5)))
            if len(values) == 4:
                values[2] = 0.06
                feet_air.params["values"] = tuple(values)

        print("[INFO] Phase profile: p1_b2_stepup (reduced-height B2 step-up curriculum applied).")
        return {
            "profile": profile,
            "overrides_applied": True,
            "mtl_command_profile": "p1_b2_stepup",
            "mtl_stair_forward_progress_weight": 0.75,
            "max_init_terrain_level": 0,
            "terrain_curriculum_enabled": True,
            "stair_step_height_range": [0.02, 0.08],
            "reset_pose": {"x": [-0.25, 0.5], "y": [-0.25, 0.25], "yaw": [-0.1, 0.1]},
            "commands": {
                "flat_rough": {"lin_vel_x": [0.15, 0.5], "lin_vel_y": [-0.15, 0.15], "ang_vel_z": [-0.2, 0.2]},
                "stairs": {"lin_vel_x": [0.25, 0.55], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]},
                "gap": {"lin_vel_x": [0.3, 0.55], "lin_vel_y": [-0.05, 0.05], "ang_vel_z": [-0.1, 0.1]},
            },
            "rewards": {
                "stairs_feet_air_time": 0.06,
            },
        }

    if profile == "p1_b2_stepup_retain":
        setattr(env_cfg, "mtl_command_profile", "p1_b2_stepup_retain")
        setattr(env_cfg, "mtl_stair_forward_progress_weight", 0.45)
        terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
        terrain_gen = getattr(terrain, "terrain_generator", None)
        if terrain is not None:
            terrain.max_init_terrain_level = 0
        if terrain_gen is not None and hasattr(terrain_gen, "sub_terrains"):
            stairs = terrain_gen.sub_terrains.get("pyramid_stairs_inv")
            if stairs is not None and hasattr(stairs, "step_height_range"):
                stairs.step_height_range = (0.02, 0.08)

        cmd = env_cfg.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0

        reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
        if reset_base is not None and isinstance(getattr(reset_base, "params", None), dict):
            pose_range = reset_base.params.get("pose_range")
            if isinstance(pose_range, dict):
                pose_range["x"] = (-0.25, 0.5)
                pose_range["y"] = (-0.25, 0.25)
                pose_range["yaw"] = (-0.15, 0.15)

        rw = getattr(env_cfg, "rewards", None)
        feet_air = getattr(rw, "feet_air_time", None) if rw is not None else None
        if feet_air is not None and isinstance(getattr(feet_air, "params", None), dict):
            values = list(feet_air.params.get("values", (0.25, 0.25, 0.02, 0.5)))
            if len(values) == 4:
                values[2] = 0.04
                feet_air.params["values"] = tuple(values)

        print("[INFO] Phase profile: p1_b2_stepup_retain (easy B2 bridge with broad rehearsal applied).")
        return {
            "profile": profile,
            "overrides_applied": True,
            "mtl_command_profile": "p1_b2_stepup_retain",
            "mtl_stair_forward_progress_weight": 0.45,
            "max_init_terrain_level": 0,
            "terrain_curriculum_enabled": True,
            "stair_step_height_range": [0.02, 0.08],
            "reset_pose": {"x": [-0.25, 0.5], "y": [-0.25, 0.25], "yaw": [-0.15, 0.15]},
            "commands": {
                "flat_rough": {"lin_vel_x": [-0.6, 0.9], "lin_vel_y": [-0.7, 0.7], "ang_vel_z": [-0.7, 0.7]},
                "stairs": {"lin_vel_x": [0.25, 0.6], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]},
                "gap": {"lin_vel_x": [0.35, 0.75], "lin_vel_y": [-0.1, 0.1], "ang_vel_z": [-0.2, 0.2]},
            },
            "rewards": {
                "stairs_feet_air_time": 0.04,
            },
        }

    if profile == "p1_b2_ramp":
        setattr(env_cfg, "mtl_command_profile", "p1_b2_ramp")
        setattr(env_cfg, "mtl_stair_forward_progress_weight", 0.50)
        terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
        if terrain is not None:
            terrain.max_init_terrain_level = 1

        cmd = env_cfg.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0

        reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
        if reset_base is not None and isinstance(getattr(reset_base, "params", None), dict):
            pose_range = reset_base.params.get("pose_range")
            if isinstance(pose_range, dict):
                pose_range["yaw"] = (-0.15, 0.15)

        print("[INFO] Phase profile: p1_b2_ramp (B2 transition/climb bridge applied).")
        return {
            "profile": profile,
            "overrides_applied": True,
            "mtl_command_profile": "p1_b2_ramp",
            "mtl_stair_forward_progress_weight": 0.50,
            "max_init_terrain_level": 1,
            "terrain_curriculum_enabled": True,
            "reset_yaw": [-0.15, 0.15],
            "commands": {
                "flat_rough": {"lin_vel_x": [0.15, 0.55], "lin_vel_y": [-0.2, 0.2], "ang_vel_z": [-0.25, 0.25]},
                "stairs": {"lin_vel_x": [0.35, 0.75], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]},
                "gap": {"lin_vel_x": [0.3, 0.6], "lin_vel_y": [-0.05, 0.05], "ang_vel_z": [-0.1, 0.1]},
            },
        }

    if profile == "p1_mixed":
        setattr(env_cfg, "mtl_command_profile", "p1_mixed")
        terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
        if terrain is not None:
            terrain.max_init_terrain_level = 2

        cmd = env_cfg.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0

        # Match the single-task B2/C2 assumption that forward commands align with
        # terrain direction. Random yaw on stairs/gaps injects non-stationary
        # sideways/downhill variants into a phase intended to bridge toward full MTL.
        reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
        if reset_base is not None and isinstance(getattr(reset_base, "params", None), dict):
            pose_range = reset_base.params.get("pose_range")
            if isinstance(pose_range, dict):
                pose_range["yaw"] = (-0.2, 0.2)

        print("[INFO] Phase profile: p1_mixed (intermediate command/terrain curriculum applied).")
        return {
            "profile": profile,
            "overrides_applied": True,
            "mtl_command_profile": "p1_mixed",
            "max_init_terrain_level": 2,
            "terrain_curriculum_enabled": True,
            "reset_yaw": [-0.2, 0.2],
            "commands": {
                "flat_rough": {"lin_vel_x": [-0.2, 0.7], "lin_vel_y": [-0.35, 0.35], "ang_vel_z": [-0.35, 0.35]},
                "stairs": {"lin_vel_x": [0.4, 0.8], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]},
                "gap": {"lin_vel_x": [0.4, 0.8], "lin_vel_y": [-0.08, 0.08], "ang_vel_z": [-0.15, 0.15]},
            },
        }

    if profile == "p1_omni":
        setattr(env_cfg, "mtl_command_profile", "p1_omni")
        terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
        if terrain is not None:
            terrain.max_init_terrain_level = 3

        cmd = env_cfg.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0

        reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
        if reset_base is not None and isinstance(getattr(reset_base, "params", None), dict):
            pose_range = reset_base.params.get("pose_range")
            if isinstance(pose_range, dict):
                pose_range["yaw"] = (-0.2, 0.2)

        print("[INFO] Phase profile: p1_omni (wide omni bridge curriculum applied).")
        return {
            "profile": profile,
            "overrides_applied": True,
            "mtl_command_profile": "p1_omni",
            "max_init_terrain_level": 3,
            "terrain_curriculum_enabled": True,
            "reset_yaw": [-0.2, 0.2],
            "commands": {
                "flat_rough": {"lin_vel_x": [-0.6, 0.9], "lin_vel_y": [-0.7, 0.7], "ang_vel_z": [-0.7, 0.7]},
                "stairs": {"lin_vel_x": [0.4, 0.85], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]},
                "gap": {"lin_vel_x": [0.4, 0.8], "lin_vel_y": [-0.1, 0.1], "ang_vel_z": [-0.2, 0.2]},
            },
        }

    if profile == "p2_b2safe":
        setattr(env_cfg, "mtl_command_profile", "p2_b2safe")
        setattr(env_cfg, "mtl_stair_forward_progress_weight", 0.20)
        terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
        if terrain is not None:
            terrain.max_init_terrain_level = 2

        cmd = env_cfg.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0

        reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
        if reset_base is not None and isinstance(getattr(reset_base, "params", None), dict):
            pose_range = reset_base.params.get("pose_range")
            if isinstance(pose_range, dict):
                pose_range["yaw"] = (-0.2, 0.2)

        print("[INFO] Phase profile: p2_b2safe (balanced recovery with B2 retention applied).")
        return {
            "profile": profile,
            "overrides_applied": True,
            "mtl_command_profile": "p2_b2safe",
            "mtl_stair_forward_progress_weight": 0.20,
            "max_init_terrain_level": 2,
            "terrain_curriculum_enabled": True,
            "reset_yaw": [-0.2, 0.2],
            "commands": {
                "flat_rough": {"lin_vel_x": [-0.6, 0.9], "lin_vel_y": [-0.7, 0.7], "ang_vel_z": [-0.7, 0.7]},
                "stairs": {"lin_vel_x": [0.35, 0.75], "lin_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0]},
                "gap": {"lin_vel_x": [0.35, 0.75], "lin_vel_y": [-0.1, 0.1], "ang_vel_z": [-0.2, 0.2]},
            },
        }

    if profile != "p1_stairs":
        raise ValueError(f"Unknown phase profile: {profile}")

    # Command profile: forward-biased stair climbing
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.0
    cmd.ranges.lin_vel_x = (0.4, 0.9)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cmd.ranges.heading = (0.0, 0.0)

    # Keep reset heading close to forward, when available.
    reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
    if reset_base is not None and isinstance(getattr(reset_base, "params", None), dict):
        pose_range = reset_base.params.get("pose_range")
        if isinstance(pose_range, dict):
            pose_range["yaw"] = (-0.2, 0.2)

    # Reward profile: B2-inspired tuning while staying in unified MTL task.
    rw = env_cfg.rewards
    rw.flat_orientation_l2.weight = -2.0
    rw.base_height.weight = -3.8
    if isinstance(getattr(rw.base_height, "params", None), dict):
        rw.base_height.params["target_height"] = 0.43

    if hasattr(rw, "dof_pos_limits") and rw.dof_pos_limits is not None:
        rw.dof_pos_limits.weight = -1.0
    if hasattr(rw, "feet_air_time") and rw.feet_air_time is not None:
        rw.feet_air_time.weight = 0.016
        if isinstance(getattr(rw.feet_air_time, "params", None), dict):
            rw.feet_air_time.params["threshold"] = 0.18
    if hasattr(rw, "dof_torques_l2") and rw.dof_torques_l2 is not None:
        rw.dof_torques_l2.weight = -3.0e-5
    if hasattr(rw, "dof_acc_l2") and rw.dof_acc_l2 is not None:
        rw.dof_acc_l2.weight = -5.0e-7
    if hasattr(rw, "action_rate_l2") and rw.action_rate_l2 is not None:
        rw.action_rate_l2.weight = -0.0022
    if hasattr(rw, "undesired_contacts") and rw.undesired_contacts is not None:
        rw.undesired_contacts.weight = -0.45

    print("[INFO] Phase profile: p1_stairs (stair-biased cmd/reward overrides applied).")
    return {
        "profile": profile,
        "overrides_applied": True,
        "commands": {
            "heading_command": bool(cmd.heading_command),
            "lin_vel_x": [float(cmd.ranges.lin_vel_x[0]), float(cmd.ranges.lin_vel_x[1])],
            "lin_vel_y": [float(cmd.ranges.lin_vel_y[0]), float(cmd.ranges.lin_vel_y[1])],
            "ang_vel_z": [float(cmd.ranges.ang_vel_z[0]), float(cmd.ranges.ang_vel_z[1])],
        },
        "rewards": {
            "flat_orientation_l2": float(rw.flat_orientation_l2.weight),
            "base_height": float(rw.base_height.weight),
            "feet_air_time": float(getattr(rw.feet_air_time, "weight", 0.0)),
            "dof_torques_l2": float(getattr(rw.dof_torques_l2, "weight", 0.0)),
            "undesired_contacts": float(getattr(rw.undesired_contacts, "weight", 0.0)),
        },
    }


def _read_scalar_series(log_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not glob(os.path.join(log_dir, "events.out.tfevents.*")):
        return None

    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    if tag not in scalar_tags:
        return None

    points = ea.Scalars(tag)
    if not points:
        return None

    steps = np.array([p.step for p in points], dtype=np.int64)
    values = np.array([p.value for p in points], dtype=np.float64)
    return steps, values


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _safe_tag_name(tag: str) -> str:
    return tag.replace("/", "_").replace(" ", "_")


def _list_checkpoint_iters(log_dir: str) -> list[int]:
    ckpts = glob(os.path.join(log_dir, "model_*.pt"))
    iters: list[int] = []
    for path in ckpts:
        base = os.path.basename(path)
        try:
            iters.append(int(base.replace("model_", "").replace(".pt", "")))
        except ValueError:
            continue
    return sorted(set(iters))


def _best_checkpoint_by_reward(log_dir: str, steps: np.ndarray, values: np.ndarray) -> tuple[int, float] | None:
    ckpt_iters = _list_checkpoint_iters(log_dir)
    if not ckpt_iters or steps.size == 0 or values.size == 0:
        return None

    best_iter = None
    best_reward = None
    for ckpt_it in ckpt_iters:
        idx = np.searchsorted(steps, ckpt_it, side="right") - 1
        if idx < 0:
            continue
        reward = float(values[idx])
        if best_reward is None or reward > best_reward:
            best_reward = reward
            best_iter = ckpt_it

    if best_iter is None or best_reward is None:
        return None
    return best_iter, best_reward


def _export_learning_curves(log_dir: str, tags: list[str], smoothing: int) -> None:
    output_dir = os.path.join(log_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    available = []
    for tag in tags:
        series = _read_scalar_series(log_dir, tag)
        if series is None:
            print(f"[WARN] Plot tag not found in logs: {tag}")
            continue
        available.append((tag, series[0], series[1]))

        csv_path = os.path.join(output_dir, f"{_safe_tag_name(tag)}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,value,smoothed\n")
            smoothed = _moving_average(series[1], smoothing)
            for step, value, smooth in zip(series[0], series[1], smoothed, strict=False):
                f.write(f"{int(step)},{float(value):.10g},{float(smooth):.10g}\n")

    if not available:
        print("[WARN] No requested tags were available; skipped curve plot export.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; CSVs exported but no PNG plot was created ({exc}).")
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(11, 3.5 * len(available)), squeeze=False)
    for idx, (tag, steps, values) in enumerate(available):
        ax = axes[idx, 0]
        smoothed = _moving_average(values, smoothing)
        ax.plot(steps, values, linewidth=1.0, alpha=0.3, label="raw")
        ax.plot(steps, smoothed, linewidth=2.0, label=f"ma({max(1, smoothing)})")
        if tag == "Train/mean_reward":
            best = _best_checkpoint_by_reward(log_dir, steps, values)
            if best is not None:
                best_iter, best_reward = best
                ax.axvline(best_iter, color="tab:red", linestyle="--", linewidth=1.6, label="best ckpt")
                ax.scatter([best_iter], [best_reward], color="tab:red", s=25, zorder=4)
                ax.annotate(
                    f"best: model_{best_iter}.pt ({best_reward:.2f})",
                    xy=(best_iter, best_reward),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color="tab:red",
                )
        ax.set_title(tag)
        ax.set_xlabel("iteration")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    png_path = os.path.join(output_dir, "learning_curves.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Learning curves exported to: {output_dir}")


def _apply_mtl_stability_overrides(agent_cfg: RslRlBaseRunnerCfg, args) -> None:
    """Apply robust policy/distribution overrides for unified MTL runs only."""
    if _has_non_missing_attr(agent_cfg, "policy"):
        policy = agent_cfg.policy
        if hasattr(policy, "noise_std_type"):
            policy.noise_std_type = args.noise_std_type
        if hasattr(policy, "init_noise_std"):
            policy.init_noise_std = args.init_noise_std

    if _has_non_missing_attr(agent_cfg, "algorithm"):
        algo = agent_cfg.algorithm
        if args.learning_rate is not None and hasattr(algo, "learning_rate"):
            algo.learning_rate = float(args.learning_rate)
        if args.entropy_coef is not None and hasattr(algo, "entropy_coef"):
            algo.entropy_coef = float(args.entropy_coef)
        if args.desired_kl is not None and hasattr(algo, "desired_kl"):
            algo.desired_kl = float(args.desired_kl)
        if args.max_grad_norm is not None and hasattr(algo, "max_grad_norm"):
            algo.max_grad_norm = float(args.max_grad_norm)
        if args.schedule is not None and hasattr(algo, "schedule"):
            algo.schedule = str(args.schedule)


def _apply_garage_mtppo_cli_profile(args) -> dict[str, object]:
    """Apply Garage-MTPPO-inspired settings before env/agent config is finalized."""
    profile = args.mtl_profile or "default"
    if profile != "garage":
        print("[INFO] MTL profile: default.")
        return {"profile": "default", "overrides_applied": False}

    previous_sampling = {
        "sampling_strategy": args.sampling_strategy,
        "focus_terrain": args.focus_terrain,
        "focus_prob": float(args.focus_prob),
    }

    if args.garage_force_uniform:
        args.sampling_strategy = "uniform"
        args.focus_terrain = None
    args.init_noise_std = 0.5
    if args.entropy_coef is None:
        args.entropy_coef = 0.0

    print(
        "[INFO] MTL profile: garage "
        "(Garage-style optimizer/normalization; terrain exposure may be overridden for ablations, "
        f"mini-batch advantage normalization, entropy_coef={args.entropy_coef})."
    )
    if args.garage_force_uniform and (
        previous_sampling["sampling_strategy"] != "uniform" or previous_sampling["focus_terrain"] is not None
    ):
        print(
            "[INFO] Garage MTPPO profile overrides terrain sampling to uniform "
            f"(was {previous_sampling})."
        )
    elif not args.garage_force_uniform:
        print(
            "[INFO] Garage MTPPO profile keeps requested terrain sampling for ablation: "
            f"strategy={args.sampling_strategy}, focus_terrain={args.focus_terrain}, focus_prob={args.focus_prob}."
        )

    return {
        "profile": "garage",
        "overrides_applied": True,
        "previous_sampling": previous_sampling,
        "forced_sampling_strategy": "uniform" if args.garage_force_uniform else None,
        "init_noise_std": float(args.init_noise_std),
        "entropy_coef": float(args.entropy_coef) if args.entropy_coef is not None else None,
        "schedule": args.schedule,
        "desired_kl": float(args.desired_kl),
    }


def _set_model_obs_normalization(agent_cfg: RslRlBaseRunnerCfg, enabled: bool) -> None:
    """Set observation normalization on both pre- and post-deprecation RSL-RL cfg schemas."""
    has_new_model_cfg = _has_non_missing_attr(agent_cfg, "actor") and _has_non_missing_attr(agent_cfg, "critic")
    if hasattr(agent_cfg, "empirical_normalization") and not has_new_model_cfg:
        agent_cfg.empirical_normalization = bool(enabled)

    if _has_non_missing_attr(agent_cfg, "policy"):
        policy = agent_cfg.policy
        if hasattr(policy, "actor_obs_normalization"):
            policy.actor_obs_normalization = bool(enabled)
        if hasattr(policy, "critic_obs_normalization"):
            policy.critic_obs_normalization = bool(enabled)

    if _has_non_missing_attr(agent_cfg, "actor") and hasattr(agent_cfg.actor, "obs_normalization"):
        agent_cfg.actor.obs_normalization = bool(enabled)
    if _has_non_missing_attr(agent_cfg, "critic") and hasattr(agent_cfg.critic, "obs_normalization"):
        agent_cfg.critic.obs_normalization = bool(enabled)


def _apply_garage_mtppo_agent_profile(agent_cfg: RslRlBaseRunnerCfg, args) -> None:
    """Mirror the Garage MTPPO assumptions that map cleanly onto RSL-RL."""
    if (args.mtl_profile or "default") != "garage":
        return

    if hasattr(agent_cfg, "obs_groups"):
        agent_cfg.obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    _set_model_obs_normalization(agent_cfg, enabled=True)

    if _has_non_missing_attr(agent_cfg, "algorithm"):
        algo = agent_cfg.algorithm
        if hasattr(algo, "normalize_advantage_per_mini_batch"):
            algo.normalize_advantage_per_mini_batch = True


def _ensure_log_std_distribution(agent_cfg: RslRlBaseRunnerCfg, args) -> None:
    """After deprecation handling, enforce actor distribution config explicitly."""
    if not _has_non_missing_attr(agent_cfg, "actor"):
        return

    actor = agent_cfg.actor
    dist_cfg = getattr(actor, "distribution_cfg", None)
    if dist_cfg is None:
        if RslRlMLPModelCfg is None:
            print("[WARN] RslRlMLPModelCfg unavailable; leaving actor distribution config unchanged.")
            return
        actor.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=float(args.init_noise_std),
            std_type=str(args.noise_std_type),
        )
        return

    if hasattr(dist_cfg, "init_std"):
        dist_cfg.init_std = float(args.init_noise_std)
    if hasattr(dist_cfg, "std_type"):
        dist_cfg.std_type = str(args.noise_std_type)


def _enable_distribution_safety_patch() -> None:
    """Patch rsl_rl Gaussian distribution update paths with finite/std clamps."""
    import torch
    from torch.distributions import Normal
    try:
        from rsl_rl.modules.distribution import GaussianDistribution, HeteroscedasticGaussianDistribution
    except ImportError:
        print("[WARN] rsl_rl.modules.distribution unavailable; skipping distribution safety patch.")
        return

    if getattr(GaussianDistribution, "_bsc_safe_patch_applied", False):
        return

    def _sanitize_std(std: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(std, nan=1.0, posinf=5.0, neginf=1.0e-6).clamp(min=1.0e-6, max=5.0)

    def _sanitize_mean(mean: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(mean, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

    def _safe_gaussian_update(self, mlp_output: torch.Tensor) -> None:
        mean = _sanitize_mean(mlp_output)
        if self.std_type == "scalar":
            self.std_param.data.nan_to_num_(nan=1.0, posinf=5.0, neginf=1.0e-6)
            self.std_param.data.clamp_(min=1.0e-6, max=5.0)
            std = self.std_param.expand_as(mean)
        elif self.std_type == "log":
            self.log_std_param.data.nan_to_num_(nan=0.0, posinf=2.0, neginf=-20.0)
            self.log_std_param.data.clamp_(min=-20.0, max=2.0)
            std = torch.exp(self.log_std_param).expand_as(mean)
        else:
            raise ValueError(f"Unknown std_type: {self.std_type}")
        self._distribution = Normal(mean, _sanitize_std(std))

    def _safe_hetero_update(self, mlp_output: torch.Tensor) -> None:
        if self.std_type == "scalar":
            mean, std = torch.unbind(mlp_output, dim=-2)
            std = _sanitize_std(std)
        elif self.std_type == "log":
            mean, log_std = torch.unbind(mlp_output, dim=-2)
            log_std = torch.nan_to_num(log_std, nan=0.0, posinf=2.0, neginf=-20.0).clamp(min=-20.0, max=2.0)
            std = _sanitize_std(torch.exp(log_std))
        else:
            raise ValueError(f"Unknown std_type: {self.std_type}")
        mean = _sanitize_mean(mean)
        self._distribution = Normal(mean, std)

    GaussianDistribution.update = _safe_gaussian_update
    HeteroscedasticGaussianDistribution.update = _safe_hetero_update
    GaussianDistribution._bsc_safe_patch_applied = True

    print("[INFO] Applied runtime distribution safety patch for rsl_rl Gaussian policies.")


def _load_runner_checkpoint_compat(runner: OnPolicyRunner, checkpoint_path: str) -> None:
    """Load checkpoint with compatibility fallbacks across local/ALICE RSL-RL schemas."""
    try:
        runner.load(checkpoint_path)
        return
    except KeyError as exc:
        missing_key = str(exc).strip("'\"")
        if missing_key != "model_state_dict":
            raise

        loaded = torch.load(checkpoint_path, weights_only=False, map_location=runner.device)
        actor_state = loaded.get("actor_state_dict")
        critic_state = loaded.get("critic_state_dict")
        if actor_state is None or critic_state is None:
            raise

        if hasattr(runner.alg, "actor") and hasattr(runner.alg, "critic"):
            missing_actor, unexpected_actor = runner.alg.actor.load_state_dict(actor_state, strict=False)
            missing_critic, unexpected_critic = runner.alg.critic.load_state_dict(critic_state, strict=False)
            if loaded.get("iter") is not None:
                runner.current_learning_iteration = loaded["iter"]

            print("[WARN] Loaded split actor/critic checkpoint into split RSL-RL runner.")
            if missing_actor or unexpected_actor or missing_critic or unexpected_critic:
                print(f"[WARN] actor missing={missing_actor}, unexpected={unexpected_actor}")
                print(f"[WARN] critic missing={missing_critic}, unexpected={unexpected_critic}")
            return

        if hasattr(runner.alg, "policy"):
            monolithic_state = {}
            for key, value in actor_state.items():
                if key.startswith("mlp."):
                    monolithic_state[f"actor.{key.removeprefix('mlp.')}"] = value
                elif key == "distribution.std_param":
                    monolithic_state["std"] = value
                elif key == "distribution.log_std_param":
                    monolithic_state["log_std"] = value
                elif key.startswith("obs_normalizer."):
                    monolithic_state[key] = value
                    monolithic_state[f"actor_{key}"] = value
                    monolithic_state[f"actor.{key}"] = value

            for key, value in critic_state.items():
                if key.startswith("mlp."):
                    monolithic_state[f"critic.{key.removeprefix('mlp.')}"] = value
                elif key.startswith("obs_normalizer."):
                    monolithic_state[f"critic_{key}"] = value
                    monolithic_state[f"critic.{key}"] = value

            load_result = runner.alg.policy.load_state_dict(monolithic_state, strict=False)
            if loaded.get("iter") is not None:
                runner.current_learning_iteration = loaded["iter"]

            print("[WARN] Loaded split actor/critic checkpoint into monolithic RSL-RL policy.")
            missing_policy = getattr(load_result, "missing_keys", [])
            unexpected_policy = getattr(load_result, "unexpected_keys", [])
            if isinstance(load_result, tuple) and len(load_result) == 2:
                missing_policy, unexpected_policy = load_result
            if missing_policy or unexpected_policy:
                print(f"[WARN] policy missing={missing_policy}, unexpected={unexpected_policy}")
            return

        raise RuntimeError(
            "Checkpoint has split actor/critic state dicts, but this RSL-RL runner "
            "does not expose compatible actor/critic or policy attributes."
        )
    except RuntimeError as exc:
        message = str(exc)
        std_schema_mismatch = (
            ("distribution.std_param" in message and "distribution.log_std_param" in message)
            or ("distribution.log_std_param" in message and "distribution.std_param" in message)
        )
        if not std_schema_mismatch:
            raise

        print(
            "[WARN] Checkpoint uses different distribution std schema "
            "(std_param vs log_std_param). Retrying non-strict load."
        )
        runner.load(checkpoint_path, strict=False)


parser = argparse.ArgumentParser(description="Train unified MTL policy with robust defaults.")
parser.add_argument(
    "--task",
    type=str,
    default="MTL-Unified-Unitree-Go2-AllTerrains-v0",
    help="Gym ID (defaults to unified all-terrains task).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel envs.")
parser.add_argument("--seed", type=int, default=None, help="Random seed (-1 for random).")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max PPO iterations.")
parser.add_argument(
    "--sampling_strategy",
    type=str,
    default="uniform",
    choices=("uniform", "focus", "custom"),
    help=(
        "Task sampling strategy for terrain mixture. "
        "'focus' = one terrain + rehearsal on others, 'custom' = explicit flat rough stairs gap probabilities."
    ),
)
parser.add_argument(
    "--focus_terrain",
    type=str,
    default=None,
    choices=tuple(_FOCUS_TERRAIN_TO_SUBTERRAIN.keys()),
    help="Terrain to prioritize when sampling_strategy=focus.",
)
parser.add_argument(
    "--focus_prob",
    type=float,
    default=0.70,
    help="Sampling probability for focused terrain (rest is split across other terrains).",
)
parser.add_argument(
    "--custom_terrain_probs",
    nargs=4,
    type=float,
    default=None,
    metavar=("FLAT", "ROUGH", "STAIRS", "GAP"),
    help="Custom terrain sampling probabilities for --sampling_strategy custom; values are normalized.",
)
parser.add_argument(
    "--phase_profile",
    type=str,
    default="default",
    choices=(
        "default",
        "p0_rough",
        "p0_gait",
        "p1_b2_easy",
        "p1_b2_stepup",
        "p1_b2_stepup_retain",
        "p1_b2_ramp",
        "p1_mixed",
        "p1_omni",
        "p2_b2safe",
        "p1_stairs",
    ),
    help=(
        "Optional phase-specific command/reward profile for unified MTL task. "
        "'default' keeps existing behavior, 'p0_rough' is an alias of default, "
        "'p0_gait' applies level-0 gait bootstrap commands, "
        "'p1_b2_easy' applies an easy corrected-stairs bridge, "
        "'p1_b2_stepup' applies reduced-height B2 step-up curriculum, "
        "'p1_b2_stepup_retain' applies reduced-height B2 with broad flat/rough rehearsal, "
        "'p1_b2_ramp' applies a B2 transition/climb bridge, "
        "'p1_mixed' applies intermediate command/terrain curriculum, "
        "'p1_omni' widens p1_mixed toward full omni benchmark commands, "
        "'p2_b2safe' applies balanced recovery with B2 retention, "
        "'p1_stairs' applies stair-biased command/reward overrides."
    ),
)
parser.add_argument(
    "--mtl_profile",
    type=str,
    default="default",
    choices=("default", "garage"),
    help=(
        "Optional MTL PPO recipe. 'garage' adapts Garage MTPPO assumptions: uniform task exposure, "
        "one-hot conditioned observations from the task env, obs normalization, centered mini-batch "
        "advantages, and no entropy bonus."
    ),
)
parser.add_argument(
    "--garage_force_uniform",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="When using --mtl_profile garage, force uniform task exposure unless disabled for ablations.",
)
parser.add_argument(
    "--task_diagnostics",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Log per-task reward, tracking, termination, terrain, and command diagnostics during MTL training.",
)
parser.add_argument(
    "--plot_learning_curves",
    action="store_true",
    default=False,
    help="Export learning-curve CSV/PNG from TensorBoard scalars after training.",
)
parser.add_argument("--plot_smoothing", type=int, default=25, help="Moving-average window for plotted curves.")
parser.add_argument(
    "--plot_tags",
    nargs="+",
    default=[
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Loss/value_function",
        "Loss/surrogate",
        "Loss/entropy",
    ],
    help="TensorBoard scalar tags to export/plot.",
)

parser.add_argument(
    "--noise_std_type",
    type=str,
    default="log",
    choices=("scalar", "log"),
    help="Actor noise std parameterization. Default: log.",
)
parser.add_argument("--init_noise_std", type=float, default=0.5, help="Initial actor noise std.")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="PPO learning rate.")
parser.add_argument("--entropy_coef", type=float, default=None, help="PPO entropy coefficient.")
parser.add_argument("--desired_kl", type=float, default=0.01, help="PPO desired KL (adaptive schedule).")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="PPO grad norm clip.")
parser.add_argument(
    "--schedule",
    type=str,
    default="adaptive",
    choices=("adaptive", "fixed"),
    help="PPO LR schedule.",
)
parser.add_argument(
    "--distribution_safety_patch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply runtime safety patch to rsl_rl Gaussian distributions (sanitize NaN/invalid std).",
)

parser.add_argument("--experiment_name", type=str, default="unitree_go2_mtl_unified", help="Log experiment name.")
parser.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
parser.add_argument("--pretrained_checkpoint", type=str, default=None, help="Weights to initialize before training.")
parser.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
parser.add_argument("--log_project_name", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from tensorboard.backend.event_processing import event_accumulator

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
)
try:
    from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
except ImportError:
    RslRlMLPModelCfg = None
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    try:
        from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
    except ImportError:
        def handle_deprecated_rsl_rl_cfg(agent_cfg, _installed_version):
            return agent_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import envs.registry


_TASK_DIAGNOSTIC_ORDER = ("flat", "rough", "stairs", "gap")
_TASK_DIAGNOSTIC_SUBTERRAINS = ("flat", "random_rough", "pyramid_stairs_inv", "gap")


def _diagnostic_task_ids(raw_env) -> torch.Tensor:
    terrain = raw_env.scene.terrain
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is None:
        return torch.zeros(raw_env.num_envs, dtype=torch.long, device=raw_env.device)

    if getattr(terrain, "terrain_origins", None) is not None:
        num_cols = terrain.terrain_origins.shape[1]
    else:
        num_cols = 20

    terrain_gen = getattr(getattr(getattr(raw_env, "cfg", None), "scene", None), "terrain", None)
    terrain_gen = getattr(terrain_gen, "terrain_generator", None)
    sub_terrains = getattr(terrain_gen, "sub_terrains", None)
    if sub_terrains is None:
        return torch.zeros(raw_env.num_envs, dtype=torch.long, device=raw_env.device)

    proportions = torch.tensor(
        [float(sub_terrains[name].proportion) for name in _TASK_DIAGNOSTIC_SUBTERRAINS],
        dtype=torch.float32,
        device=terrain_types.device,
    )
    boundaries = torch.cumsum(proportions / proportions.sum() * num_cols, dim=0)[:-1]
    return torch.bucketize(terrain_types.to(torch.float32), boundaries).clamp(max=len(_TASK_DIAGNOSTIC_ORDER) - 1)


class MtlTaskDiagnosticsWrapper(gym.Wrapper):
    """Inject per-task MTL diagnostics into RSL-RL's extras['log'] stream."""

    def step(self, action):
        obs, reward, terminated, truncated, extras = self.env.step(action)
        raw_env = self.unwrapped
        extras = dict(extras)
        log = dict(extras.get("log", {}))

        task_ids = _diagnostic_task_ids(raw_env)
        rewards = reward.reshape(-1)
        terminated_f = terminated.reshape(-1).float()
        truncated_f = truncated.reshape(-1).float()

        robot = raw_env.scene["robot"]
        command = raw_env.command_manager.get_command("base_velocity")
        lin_err_xy = torch.linalg.vector_norm(command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
        yaw_err = torch.abs(command[:, 2] - robot.data.root_ang_vel_b[:, 2])
        terrain_levels = getattr(raw_env.scene.terrain, "terrain_levels", None)

        for task_index, task_name in enumerate(_TASK_DIAGNOSTIC_ORDER):
            mask = task_ids == task_index
            prefix = f"Task/{task_name}"
            log[f"{prefix}/env_frac"] = mask.float().mean().detach()
            if not torch.any(mask):
                continue

            log[f"{prefix}/reward"] = rewards[mask].mean().detach()
            log[f"{prefix}/error_vel_xy"] = lin_err_xy[mask].mean().detach()
            log[f"{prefix}/error_vel_yaw"] = yaw_err[mask].mean().detach()
            log[f"{prefix}/base_contact"] = terminated_f[mask].mean().detach()
            log[f"{prefix}/time_out"] = truncated_f[mask].mean().detach()
            log[f"{prefix}/cmd_lin_x"] = command[mask, 0].mean().detach()
            log[f"{prefix}/cmd_lin_y_abs"] = command[mask, 1].abs().mean().detach()
            log[f"{prefix}/cmd_yaw_abs"] = command[mask, 2].abs().mean().detach()
            if terrain_levels is not None:
                log[f"{prefix}/terrain_level"] = terrain_levels[mask].float().mean().detach()

        if "episode" in extras:
            episode = dict(extras["episode"])
            episode.update(log)
            extras["episode"] = episode
        else:
            extras["log"] = log
        return obs, reward, terminated, truncated, extras


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if not (
        "MTL-Unified-Unitree-Go2-AllTerrains" in args_cli.task
        or "MTL-Conditioned-Unitree-Go2-AllTerrains" in args_cli.task
    ):
        raise ValueError(
            f"scripts/mtl_train.py is intended for unified MTL tasks only. Received: {args_cli.task}"
        )

    agent_cfg = _update_agent_cfg_from_cli(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    mtl_profile_info = _apply_garage_mtppo_cli_profile(args_cli)
    sampling_proportions = _apply_terrain_sampling_profile(env_cfg, args_cli)
    phase_profile_info = _apply_phase_profile_overrides(env_cfg, args_cli.phase_profile)

    _apply_garage_mtppo_agent_profile(agent_cfg, args_cli)
    _apply_mtl_stability_overrides(agent_cfg, args_cli)
    _apply_garage_mtppo_agent_profile(agent_cfg, args_cli)

    installed_rsl_rl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    _apply_garage_mtppo_agent_profile(agent_cfg, args_cli)
    _ensure_log_std_distribution(agent_cfg, args_cli)
    _apply_garage_mtppo_agent_profile(agent_cfg, args_cli)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_root_path = os.path.join(project_root, "logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging root directory: {log_root_path}")
    print(f"[INFO] Saving this run to: {log_dir}")

    if _has_non_missing_attr(agent_cfg, "algorithm"):
        algo = agent_cfg.algorithm
        print(
            "[INFO] Effective PPO overrides: "
            f"lr={getattr(algo, 'learning_rate', 'n/a')}, "
            f"entropy_coef={getattr(algo, 'entropy_coef', 'n/a')}, "
            f"desired_kl={getattr(algo, 'desired_kl', 'n/a')}, "
            f"max_grad_norm={getattr(algo, 'max_grad_norm', 'n/a')}, "
            f"schedule={getattr(algo, 'schedule', 'n/a')}, "
            f"normalize_advantage_per_mini_batch="
            f"{getattr(algo, 'normalize_advantage_per_mini_batch', 'n/a')}"
        )

    actor = getattr(agent_cfg, "actor", None)
    critic = getattr(agent_cfg, "critic", None)
    actor_dist = getattr(actor, "distribution_cfg", None) if actor is not None else None
    print(
        "[INFO] Effective actor distribution: "
        f"class={getattr(actor_dist, 'class_name', 'None')}, "
        f"std_type={getattr(actor_dist, 'std_type', 'None')}, "
        f"init_std={getattr(actor_dist, 'init_std', 'None')}"
    )
    print(
        "[INFO] Effective observation normalization: "
        f"actor={getattr(actor, 'obs_normalization', 'n/a')}, "
        f"critic={getattr(critic, 'obs_normalization', 'n/a')}, "
        f"obs_groups={getattr(agent_cfg, 'obs_groups', 'n/a')}"
    )

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if args_cli.task_diagnostics:
        env = MtlTaskDiagnosticsWrapper(env)
        print("[INFO] Enabled per-task MTL diagnostics under TensorBoard tag prefix Task/<task>/...")

    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    start_time = time.time()

    if args_cli.distribution_safety_patch:
        _enable_distribution_safety_patch()

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        _load_runner_checkpoint_compat(runner, resume_path)
    elif args_cli.pretrained_checkpoint is not None:
        print(f"[INFO] Loading pretrained weights from: {args_cli.pretrained_checkpoint}")
        _load_runner_checkpoint_compat(runner, args_cli.pretrained_checkpoint)
        print("[INFO] Pretrained weights loaded. Training starts from iteration 0.")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    sampling_profile = {
        "strategy": args_cli.sampling_strategy,
        "focus_terrain": args_cli.focus_terrain,
        "focus_prob": float(args_cli.focus_prob),
        "proportions": {k: float(v) for k, v in sampling_proportions.items()},
        "phase_profile": args_cli.phase_profile,
        "phase_overrides": phase_profile_info,
        "mtl_profile": mtl_profile_info,
        "task_diagnostics": bool(args_cli.task_diagnostics),
    }
    with open(os.path.join(log_dir, "params", "sampling_profile.json"), "w", encoding="utf-8") as f:
        json.dump(sampling_profile, f, indent=2)

    runner.learn(num_learning_iterations=int(agent_cfg.max_iterations), init_at_random_ep_len=True)
    if args_cli.plot_learning_curves:
        _export_learning_curves(
            log_dir=log_dir,
            tags=args_cli.plot_tags,
            smoothing=max(1, args_cli.plot_smoothing),
        )

    elapsed = time.time() - start_time
    print(
        f"[INFO] MTL training completed in {elapsed:.1f}s ({elapsed / 3600:.2f}h), "
        f"iterations={agent_cfg.max_iterations}"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
