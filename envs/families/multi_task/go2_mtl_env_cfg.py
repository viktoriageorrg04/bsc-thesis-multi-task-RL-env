"""Unified multi-task env config: 4 terrain types in one TerrainGeneratorCfg.

  - flat: fast omni-dir walking (fam A)
  - random_rough: moderate omni-dir walking (fam B1)
  - pyramid_stairs_inv: forward-biased stair climbing (fam B2)
  - gap: moderate forward gap crossing (fam C2)

The shared reward function uses the B1-proven set (track_vel + feet_air_time +
orientation + height + torques + contacts).  Any performance difference vs.
single-task baselines is attributable to the multi-task training regime.
"""

import copy
from functools import lru_cache
import math
import torch

import isaaclab.envs.mdp as core_mdp
import isaaclab.terrains as terrain_gen
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)
from envs.families.agility_terrain.go2_fam_c_env_cfg import Go2C2GapCrossingEnvCfg
from envs.families.flat_velocity.go2_fam_a_env_cfg import Go2A2OmniWalkEnvCfg
from envs.families.rough_velocity.go2_fam_b_env_cfg import Go2B1RoughWalkEnvCfg, Go2B2StairClimbEnvCfg

_TASK_ORDER = ("flat", "random_rough", "pyramid_stairs_inv", "gap")
_TASK_CFG_TYPES = (Go2A2OmniWalkEnvCfg, Go2B1RoughWalkEnvCfg, Go2B2StairClimbEnvCfg, Go2C2GapCrossingEnvCfg)

# terrain gen

_MTL_TERRAIN_GEN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # flat ground (fam A baseline)
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.0, 0.0),
            noise_step=0.01,
            border_width=0.25,
        ),
        # random rough (fam B1)
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        # pyramid stairs (fam B2)
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.04, 0.16),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # gap crossing (fam C2)
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.25,
            gap_width_range=(0.0, 0.20),
            platform_width=2.0,
        ),
    },
)


class TaskConditionedVelocityCommand(UniformVelocityCommand):
    """Velocity command sampler whose ranges are selected from the env task id."""

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        task_ids = _mtl_task_index(self._env)[env_ids]
        command_profile = getattr(getattr(self._env, "cfg", None), "mtl_command_profile", "default")
        command_ranges = _task_command_ranges(str(command_profile))

        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False

        for task_index, ranges in enumerate(command_ranges):
            task_env_ids = env_ids[task_ids == task_index]
            if task_env_ids.numel() == 0:
                continue

            r = torch.empty(task_env_ids.numel(), device=self.device)
            self.vel_command_b[task_env_ids, 0] = r.uniform_(*ranges["lin_x"])
            self.vel_command_b[task_env_ids, 1] = r.uniform_(*ranges["lin_y"])
            self.vel_command_b[task_env_ids, 2] = r.uniform_(*ranges["yaw"])

            if ranges["heading"]:
                self.heading_target[task_env_ids] = r.uniform_(*ranges["heading_range"])
                self.is_heading_env[task_env_ids] = True

            standing_prob = float(ranges["standing"])
            if standing_prob > 0.0:
                self.is_standing_env[task_env_ids] = r.uniform_(0.0, 1.0) <= standing_prob


# reward fixes

def _safe_base_height_l2(
    env,
    target_height: float | torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """base_height_l2 with finite-safe ray handling for mixed terrains (incl. gaps)."""
    asset = env.scene[asset_cfg.name]
    if not torch.is_tensor(target_height):
        target_height = torch.full((env.num_envs,), float(target_height), dtype=torch.float32, device=env.device)
    else:
        target_height = target_height.to(device=env.device, dtype=torch.float32)

    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        ray_z = sensor.data.ray_hits_w[..., 2]
        valid = torch.isfinite(ray_z)
        ray_z_clean = torch.where(valid, ray_z, torch.zeros_like(ray_z))
        valid_count = valid.float().sum(dim=1).clamp(min=1.0)
        ground_z = ray_z_clean.sum(dim=1) / valid_count
        all_invalid = ~valid.any(dim=1)
        ground_z[all_invalid] = asset.data.root_pos_w[all_invalid, 2] - target_height[all_invalid]
        adjusted_target = target_height + ground_z
    else:
        adjusted_target = target_height
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target)


def _mtl_task_index(env) -> torch.Tensor:
    """Return terrain/task bucket id for each env: flat, rough, stairs, gap."""
    terrain = env.scene.terrain
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_types is None:
        return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    if getattr(terrain, "terrain_origins", None) is not None:
        num_cols = terrain.terrain_origins.shape[1]
    else:
        num_cols = _MTL_TERRAIN_GEN.num_cols

    terrain_gen = getattr(getattr(getattr(env, "cfg", None), "scene", None), "terrain", None)
    terrain_gen = getattr(terrain_gen, "terrain_generator", None)
    sub_terrains = getattr(terrain_gen, "sub_terrains", None) or _MTL_TERRAIN_GEN.sub_terrains
    proportions = torch.tensor(
        [sub_terrains[name].proportion for name in _TASK_ORDER],
        dtype=torch.float32,
        device=terrain_types.device,
    )
    boundaries = torch.cumsum(proportions / proportions.sum() * num_cols, dim=0)[:-1]
    return torch.bucketize(terrain_types.to(torch.float32), boundaries).clamp(max=len(_TASK_ORDER) - 1)


def _mtl_task_one_hot(env) -> torch.Tensor:
    """One-hot task id observation aligned with the terrain generator columns."""
    task_ids = _mtl_task_index(env)
    return torch.nn.functional.one_hot(task_ids, num_classes=len(_TASK_ORDER)).to(torch.float32)


def _mtl_task_weights(env, values: tuple[float, float, float, float]) -> torch.Tensor:
    value_tensor = torch.tensor(values, dtype=torch.float32, device=env.device)
    return value_tensor[_mtl_task_index(env)]


@lru_cache(maxsize=1)
def _single_task_reference_cfgs() -> tuple[UnitreeGo2RoughEnvCfg, ...]:
    """Reference task configs used as the source of conditioned MTL commands/rewards."""
    return tuple(cfg_type() for cfg_type in _TASK_CFG_TYPES)


def _reward_term(cfg: UnitreeGo2RoughEnvCfg, name: str):
    return getattr(cfg.rewards, name, None)


def _reward_weights(name: str) -> tuple[float, float, float, float]:
    return tuple(float(getattr(_reward_term(cfg, name), "weight", 0.0) or 0.0) for cfg in _single_task_reference_cfgs())


def _reward_param(name: str, param: str, default: float) -> tuple[float, float, float, float]:
    values = []
    for cfg in _single_task_reference_cfgs():
        term = _reward_term(cfg, name)
        params = getattr(term, "params", None) or {}
        values.append(float(params.get(param, default)))
    return tuple(values)


def _apply_command_profile(ranges: tuple[dict[str, object], ...], profile: str) -> tuple[dict[str, object], ...]:
    if profile not in {"p0_gait", "p1_b2_easy", "p1_b2_stepup", "p1_b2_ramp", "p1_mixed", "p1_omni", "p2_b2safe"}:
        return ranges

    profiled = [dict(r) for r in ranges]
    if profile == "p1_b2_easy":
        # an easy forward bridge before p1_mixed (corrected inverted stairs)
        for task_index in (0, 1):
            profiled[task_index].update(
                {
                    "lin_x": (0.15, 0.55),
                    "lin_y": (-0.2, 0.2),
                    "yaw": (-0.25, 0.25),
                    "heading_range": (0.0, 0.0),
                    "heading": False,
                    "standing": 0.0,
                }
            )
        profiled[2].update({"lin_x": (0.2, 0.45), "lin_y": (0.0, 0.0), "yaw": (0.0, 0.0), "heading": False, "standing": 0.0})
        profiled[3].update({"lin_x": (0.3, 0.6), "lin_y": (-0.05, 0.05), "yaw": (-0.1, 0.1), "heading": False, "standing": 0.0})
        return tuple(profiled)

    if profile == "p1_b2_stepup":
        # first successful contact/step-up bridge: slower than benchmark B2,
        # but faster than p1_b2_easy and aimed at entering stairs
        for task_index in (0, 1):
            profiled[task_index].update(
                {
                    "lin_x": (0.15, 0.5),
                    "lin_y": (-0.15, 0.15),
                    "yaw": (-0.2, 0.2),
                    "heading_range": (0.0, 0.0),
                    "heading": False,
                    "standing": 0.0,
                }
            )
        profiled[2].update({"lin_x": (0.25, 0.55), "lin_y": (0.0, 0.0), "yaw": (0.0, 0.0), "heading": False, "standing": 0.0})
        profiled[3].update({"lin_x": (0.3, 0.55), "lin_y": (-0.05, 0.05), "yaw": (-0.1, 0.1), "heading": False, "standing": 0.0})
        return tuple(profiled)

    if profile == "p1_b2_ramp":
        # targeted bridge from p0/easy stairs into the benchmark speed cmd
        for task_index in (0, 1):
            profiled[task_index].update(
                {
                    "lin_x": (0.15, 0.55),
                    "lin_y": (-0.2, 0.2),
                    "yaw": (-0.25, 0.25),
                    "heading_range": (0.0, 0.0),
                    "heading": False,
                    "standing": 0.0,
                }
            )
        profiled[2].update({"lin_x": (0.40, 0.90), "lin_y": (0.0, 0.0), "yaw": (0.0, 0.0), "heading": False, "standing": 0.0})
        profiled[3].update({"lin_x": (0.3, 0.6), "lin_y": (-0.05, 0.05), "yaw": (-0.1, 0.1), "heading": False, "standing": 0.0})
        return tuple(profiled)

    if profile == "p1_omni":
        # bridge from p1_mixed toward the A2/B1 full-omni benchmark
        for task_index in (0, 1):
            profiled[task_index].update(
                {
                    "lin_x": (-0.6, 0.9),
                    "lin_y": (-0.7, 0.7),
                    "yaw": (-0.7, 0.7),
                    "heading_range": (0.0, 0.0),
                    "heading": False,
                    "standing": 0.0,
                }
            )
        profiled[2].update({"lin_x": (0.4, 0.85), "lin_y": (0.0, 0.0), "yaw": (0.0, 0.0), "heading": False, "standing": 0.0})
        profiled[3].update({"lin_x": (0.4, 0.8), "lin_y": (-0.1, 0.1), "yaw": (-0.2, 0.2), "heading": False, "standing": 0.0})
        return tuple(profiled)

    if profile == "p2_b2safe":
        # balanced recovery after B2 has been learned
        for task_index in (0, 1):
            profiled[task_index].update(
                {
                    "lin_x": (-0.6, 0.9),
                    "lin_y": (-0.7, 0.7),
                    "yaw": (-0.7, 0.7),
                    "heading_range": (0.0, 0.0),
                    "heading": False,
                    "standing": 0.0,
                }
            )
        profiled[2].update({"lin_x": (0.40, 0.90), "lin_y": (0.0, 0.0), "yaw": (0.0, 0.0), "heading": False, "standing": 0.0})
        profiled[3].update({"lin_x": (0.35, 0.75), "lin_y": (-0.1, 0.1), "yaw": (-0.2, 0.2), "heading": False, "standing": 0.0})
        return tuple(profiled)

    if profile == "p1_mixed":
        # intermediate cmd curriculum
        for task_index in (0, 1):
            profiled[task_index].update(
                {
                    "lin_x": (-0.2, 0.7),
                    "lin_y": (-0.35, 0.35),
                    "yaw": (-0.35, 0.35),
                    "heading_range": (0.0, 0.0),
                    "heading": False,
                    "standing": 0.0,
                }
            )
        profiled[2].update({"lin_x": (0.4, 0.8), "lin_y": (0.0, 0.0), "yaw": (0.0, 0.0), "heading": False, "standing": 0.0})
        profiled[3].update({"lin_x": (0.4, 0.8), "lin_y": (-0.08, 0.08), "yaw": (-0.15, 0.15), "heading": False, "standing": 0.0})
        return tuple(profiled)

    # bootstrap gait learning before asking the policy to solve full omni cmds
    for task_index in (0, 1):
        profiled[task_index].update(
            {
                "lin_x": (0.2, 0.6),
                "lin_y": (-0.15, 0.15),
                "yaw": (-0.2, 0.2),
                "heading_range": (0.0, 0.0),
                "heading": False,
                "standing": 0.0,
            }
        )
    profiled[2].update({"lin_x": (0.35, 0.7), "lin_y": (0.0, 0.0), "yaw": (0.0, 0.0), "heading": False, "standing": 0.0})
    profiled[3].update({"lin_x": (0.35, 0.7), "lin_y": (-0.05, 0.05), "yaw": (-0.1, 0.1), "heading": False, "standing": 0.0})
    return tuple(profiled)


@lru_cache(maxsize=8)
def _task_command_ranges(profile: str = "default") -> tuple[dict[str, object], ...]:
    ranges = []
    for cfg in _single_task_reference_cfgs():
        cmd = cfg.commands.base_velocity
        ranges.append(
            {
                "lin_x": tuple(float(v) for v in cmd.ranges.lin_vel_x),
                "lin_y": tuple(float(v) for v in cmd.ranges.lin_vel_y),
                "yaw": tuple(float(v) for v in cmd.ranges.ang_vel_z),
                "heading_range": tuple(float(v) for v in cmd.ranges.heading),
                "heading": bool(cmd.heading_command),
                "standing": float(cmd.rel_standing_envs),
            }
        )
    return _apply_command_profile(tuple(ranges), profile)


def _task_weighted_track_lin_vel_xy_exp(
    env, values: tuple[float, float, float, float], stds: tuple[float, float, float, float], command_name: str
) -> torch.Tensor:
    task_ids = _mtl_task_index(env)
    out = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for task_index, std in enumerate(stds):
        mask = task_ids == task_index
        if torch.any(mask):
            out[mask] = core_mdp.track_lin_vel_xy_exp(env, command_name=command_name, std=std)[mask]
    return out * _mtl_task_weights(env, values)


def _task_weighted_track_ang_vel_z_exp(
    env, values: tuple[float, float, float, float], stds: tuple[float, float, float, float], command_name: str
) -> torch.Tensor:
    task_ids = _mtl_task_index(env)
    out = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for task_index, std in enumerate(stds):
        mask = task_ids == task_index
        if torch.any(mask):
            out[mask] = core_mdp.track_ang_vel_z_exp(env, command_name=command_name, std=std)[mask]
    return out * _mtl_task_weights(env, values)


def _task_weighted_undesired_contacts(
    env, values: tuple[float, float, float, float], thresholds: tuple[float, float, float, float], sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    task_ids = _mtl_task_index(env)
    out = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for task_index, threshold in enumerate(thresholds):
        mask = task_ids == task_index
        if torch.any(mask):
            contacts = core_mdp.undesired_contacts(env, sensor_cfg=sensor_cfg, threshold=threshold).to(dtype=out.dtype)
            out[mask] = contacts[mask]
    return out * _mtl_task_weights(env, values)


def _task_weighted_flat_orientation_l2(env, values: tuple[float, float, float, float]) -> torch.Tensor:
    return core_mdp.flat_orientation_l2(env) * _mtl_task_weights(env, values)


def _task_weighted_joint_torques_l2(env, values: tuple[float, float, float, float]) -> torch.Tensor:
    return core_mdp.joint_torques_l2(env) * _mtl_task_weights(env, values)


def _task_weighted_action_rate_l2(env, values: tuple[float, float, float, float]) -> torch.Tensor:
    return core_mdp.action_rate_l2(env) * _mtl_task_weights(env, values)


def _task_weighted_joint_acc_l2(env, values: tuple[float, float, float, float]) -> torch.Tensor:
    return core_mdp.joint_acc_l2(env) * _mtl_task_weights(env, values)


def _task_weighted_joint_pos_limits(env, values: tuple[float, float, float, float]) -> torch.Tensor:
    return core_mdp.joint_pos_limits(env) * _mtl_task_weights(env, values)


def _task_weighted_feet_air_time(
    env,
    values: tuple[float, float, float, float],
    thresholds: tuple[float, float, float, float],
    sensor_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    task_ids = _mtl_task_index(env)
    out = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for task_index, threshold in enumerate(thresholds):
        mask = task_ids == task_index
        if torch.any(mask):
            out[mask] = locomotion_mdp.feet_air_time(
                env, sensor_cfg=sensor_cfg, command_name=command_name, threshold=threshold
            )[mask]
    return out * _mtl_task_weights(env, values)


def _task_weighted_base_height_l2(
    env,
    weights: tuple[float, float, float, float],
    targets: tuple[float, float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    task_ids = _mtl_task_index(env)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=env.device)
    per_env_target = target_tensor[task_ids]
    return _safe_base_height_l2(env, per_env_target, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg) * _mtl_task_weights(
        env, weights
    )


def _stair_forward_progress(
    env,
    max_reward: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Small B2-only incentive to keep moving forward onto the stair face."""
    weight = float(getattr(getattr(env, "cfg", None), "mtl_stair_forward_progress_weight", 0.0))
    if weight <= 0.0:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    asset = env.scene[asset_cfg.name]
    task_mask = (_mtl_task_index(env) == 2).to(dtype=torch.float32)
    forward_vel = torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=float(max_reward))
    return weight * task_mask * forward_vel


def _apply_mtl_rewards(cfg: UnitreeGo2RoughEnvCfg) -> None:
    """B1-proven reward set shared across all terrains in the unified env."""
    rw = cfg.rewards

    # velocity tracking (stronger than Go2 defaults)
    rw.track_lin_vel_xy_exp.weight = 2.0
    rw.track_ang_vel_z_exp.weight = 1.0

    # stepping incentive (Go2 default is 0.01; kills stepping)
    rw.feet_air_time.weight = 0.25

    # penalize thigh + calf ground contacts
    rw.undesired_contacts = RewTerm(
        func=core_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )

    # upright posture
    rw.flat_orientation_l2 = RewTerm(func=core_mdp.flat_orientation_l2, weight=-5.0)

    # terrain-relative base height
    rw.base_height = RewTerm(
        func=_safe_base_height_l2,
        weight=-10.0,
        params={
            "target_height": 0.38,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    # lighter torque penalty (Go2 default -2e-4 is 20x too heavy)
    rw.dof_torques_l2.weight = -1.0e-5


def _apply_task_conditioned_rewards(cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Task-conditioned reward wrappers sourced from the single-task configs.

    Isaac Lab reward weights are normally global, so this variant sets each
    wrapped term's global weight to 1.0 and applies the task-specific signed
    weight inside the reward function based on the env's terrain column.  The
    source of truth for weights and reward params is the corresponding
    single-task config file.
    """
    rw = cfg.rewards
    track_lin = _reward_term(_single_task_reference_cfgs()[0], "track_lin_vel_xy_exp")
    track_ang = _reward_term(_single_task_reference_cfgs()[0], "track_ang_vel_z_exp")

    rw.track_lin_vel_xy_exp = RewTerm(
        func=_task_weighted_track_lin_vel_xy_exp,
        weight=1.0,
        params={
            "values": _reward_weights("track_lin_vel_xy_exp"),
            "stds": _reward_param(
                "track_lin_vel_xy_exp", "std", float((track_lin.params or {}).get("std", math.sqrt(0.25)))
            ),
            "command_name": "base_velocity",
        },
    )
    rw.track_ang_vel_z_exp = RewTerm(
        func=_task_weighted_track_ang_vel_z_exp,
        weight=1.0,
        params={
            "values": _reward_weights("track_ang_vel_z_exp"),
            "stds": _reward_param(
                "track_ang_vel_z_exp", "std", float((track_ang.params or {}).get("std", math.sqrt(0.25)))
            ),
            "command_name": "base_velocity",
        },
    )
    rw.feet_air_time = RewTerm(
        func=_task_weighted_feet_air_time,
        weight=1.0,
        params={
            "values": _reward_weights("feet_air_time"),
            "thresholds": _reward_param("feet_air_time", "threshold", 0.5),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
        },
    )
    rw.undesired_contacts = RewTerm(
        func=_task_weighted_undesired_contacts,
        weight=1.0,
        params={
            "values": _reward_weights("undesired_contacts"),
            "thresholds": _reward_param("undesired_contacts", "threshold", 1.0),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
        },
    )
    rw.flat_orientation_l2 = RewTerm(
        func=_task_weighted_flat_orientation_l2,
        weight=1.0,
        params={
            "values": _reward_weights("flat_orientation_l2"),
        },
    )
    rw.base_height = RewTerm(
        func=_task_weighted_base_height_l2,
        weight=1.0,
        params={
            "weights": _reward_weights("base_height"),
            "targets": _reward_param("base_height", "target_height", 0.38),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    rw.dof_torques_l2 = RewTerm(
        func=_task_weighted_joint_torques_l2,
        weight=1.0,
        params={"values": _reward_weights("dof_torques_l2")},
    )
    rw.action_rate_l2 = RewTerm(
        func=_task_weighted_action_rate_l2,
        weight=1.0,
        params={"values": _reward_weights("action_rate_l2")},
    )
    rw.dof_acc_l2 = RewTerm(
        func=_task_weighted_joint_acc_l2,
        weight=1.0,
        params={"values": _reward_weights("dof_acc_l2")},
    )
    rw.dof_pos_limits = RewTerm(
        func=_task_weighted_joint_pos_limits,
        weight=1.0,
        params={"values": _reward_weights("dof_pos_limits")},
    )
    rw.stair_forward_progress = RewTerm(
        func=_stair_forward_progress,
        weight=1.0,
        params={
            "max_reward": 0.8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


def _apply_task_conditioned_commands(cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Use task-specific command ranges while preserving the 3D velocity command observation."""
    cmd = cfg.commands.base_velocity
    cmd.class_type = TaskConditionedVelocityCommand
    cmd.heading_command = True
    cmd.rel_heading_envs = 1.0
    cmd.rel_standing_envs = 0.0

    # fallback and heading-control clip limits
    cmd.ranges.lin_vel_x = (-1.0, 1.0)
    cmd.ranges.lin_vel_y = (-1.0, 1.0)
    cmd.ranges.ang_vel_z = (-1.0, 1.0)
    cmd.ranges.heading = (-math.pi, math.pi)


# env configs

@configclass
class Go2MultiTaskEnvCfg(UnitreeGo2RoughEnvCfg):
    """Unified multi-task env: 4 terrain types x shared cmd profile.

    Inherits from UnitreeGo2RoughEnvCfg to get:
      - Go2-scaled actuators, height scanner, contact sensors
      - terrain curriculum

    The cmd distribution is the union of all single-task profiles:
      lin_vel_x in [-1.0, 1.0], lin_vel_y in [-1.0, 1.0], heading in [-pi, pi]
    """

    def __post_init__(self):
        super().__post_init__()
        self.mtl_stair_forward_progress_weight = 0.0

        self.scene.terrain.terrain_generator = copy.deepcopy(_MTL_TERRAIN_GEN)

        # B1-proven reward set
        _apply_mtl_rewards(self)

        # cmds (union of all single-task profiles)
        cmd = self.commands.base_velocity
        cmd.heading_command = True
        cmd.rel_heading_envs = 1.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (-1.0, 1.0)
        cmd.ranges.lin_vel_y = (-1.0, 1.0)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        cmd.ranges.heading = (-math.pi, math.pi)


@configclass
class Go2MultiTaskEnvCfg_PLAY(Go2MultiTaskEnvCfg):
    """Lightweight play/vis variant for the unified multi-task env."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 6
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class Go2MultiTaskConditionedEnvCfg(Go2MultiTaskEnvCfg):
    """Unified MTL env with task ID observations and task-weighted rewards."""

    def __post_init__(self):
        super().__post_init__()
        _apply_task_conditioned_rewards(self)
        _apply_task_conditioned_commands(self)
        self.observations.policy.task_id = ObsTerm(func=_mtl_task_one_hot)

        # forward-only stair/gap tasks
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.2, 0.2)

        self.scene.terrain.max_init_terrain_level = 1


@configclass
class Go2MultiTaskConditionedEnvCfg_PLAY(Go2MultiTaskConditionedEnvCfg):
    """Lightweight play/vis variant for conditioned MTL."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 6
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
