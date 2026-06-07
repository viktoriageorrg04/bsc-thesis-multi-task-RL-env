"""Fam B (rough velocity) task configs for Unitree Go2.

These configs are overrides of Isaac Lab's built-in Go2 rough-velocity task.
The goal is to isolate rough-terrain skills by using one terrain type per task:
B1 = random_rough only, B2 = pyramid_stairs only.
"""

import copy
import math

import isaaclab.envs.mdp as core_mdp
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)


def _keep_single_sub_terrain(cfg: UnitreeGo2RoughEnvCfg, terrain_key: str) -> None:
    """trim terrain gen to one named sub-terrain
    -> Isaac Lab rough env is a mix of terrain types but we need:
     one task = one cmd profile x one terrain
     (A1, A2, B1, B2 are all single-task benchmarks so this is fine)
    """
    generator = cfg.scene.terrain.terrain_generator
    if generator is None:
        return
    if terrain_key not in generator.sub_terrains:
        raise KeyError(f"Unknown sub-terrain key: {terrain_key}")

    terrain_generator = copy.deepcopy(generator)
    terrain_generator.sub_terrains = {terrain_key: terrain_generator.sub_terrains[terrain_key]}
    cfg.scene.terrain.terrain_generator = terrain_generator


def _b2_forward_progress(
    env,
    max_reward: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense B2 incentive for actually moving into the stair face."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_b[:, 0], min=0.0, max=float(max_reward))


def _b2_forward_speed_shortfall(
    env,
    min_speed: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty helper for B2 policies that survive while barely moving."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(float(min_speed) - asset.data.root_lin_vel_b[:, 0], min=0.0, max=float(min_speed))


def _fix_rough_reward_shaping(cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Override Go2 rough-env reward defaults that prevent locomotion learning.
      - feet_air_time weight 0.01 (12.5x below base default)
      - undesired_contacts disabled entirely
      - dof_torques_l2 penalty 20x base default
      - flat_orientation_l2 weight 0.0 (no posture signal)
    These conspire so that "collapse and do nothing" is locally optimal.
    """
    rw = cfg.rewards

    # stepping reward
    rw.feet_air_time.weight = 0.25

    # velocity tracking
    rw.track_lin_vel_xy_exp.weight = 2.0
    rw.track_ang_vel_z_exp.weight = 1.0

    # penalize thigh and calf ground contacts
    rw.undesired_contacts = RewTerm(
        func=core_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )

    # upright posture incentive
    rw.flat_orientation_l2 = RewTerm(func=core_mdp.flat_orientation_l2, weight=-5.0)

    # always-on height incentive; uses height_scanner for terrain-relative height
    rw.base_height = RewTerm(
        func=core_mdp.base_height_l2,
        weight=-10.0,
        params={
            "target_height": 0.38,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    rw.dof_torques_l2.weight = -1.0e-5


@configclass
class Go2B1RoughWalkEnvCfg(UnitreeGo2RoughEnvCfg):
    """fam B / task B1: rough walk on random_rough terrain only"""

    def __post_init__(self):
        super().__post_init__()
        _keep_single_sub_terrain(self, "random_rough")

        # B1 uses the same omni command profile as A2
        cmd = self.commands.base_velocity
        cmd.heading_command = True
        cmd.rel_heading_envs = 1.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (-1.0, 1.0)
        cmd.ranges.lin_vel_y = (-1.0, 1.0)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        cmd.ranges.heading = (-math.pi, math.pi)

        _fix_rough_reward_shaping(self)

@configclass
class Go2B2StairClimbEnvCfg(UnitreeGo2RoughEnvCfg):
    """fam B / task B2: stair climb on stairs only (climb-focused)."""

    def __post_init__(self):
        super().__post_init__()

        # use inverted stairs only (forward ~= uphill from spawn)
        gen = copy.deepcopy(self.scene.terrain.terrain_generator)
        gen.sub_terrains = {
            "pyramid_stairs_inv": gen.sub_terrains["pyramid_stairs_inv"],
        }
        gen.sub_terrains["pyramid_stairs_inv"].proportion = 1.0
        gen.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.04, 0.16)
        self.scene.terrain.terrain_generator = gen

        # easier curriculum start, matching the successful Apr15 scratch run
        self.scene.terrain.max_init_terrain_level = 1

        # straight-forward only
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.4, 0.9)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)

        # keep reset heading close to forward
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.2, 0.2)

        rw = self.rewards
        rw.flat_orientation_l2.weight = -2.0
        rw.track_lin_vel_xy_exp.weight = 1.5
        rw.track_lin_vel_xy_exp.params["std"] = 0.5
        rw.track_ang_vel_z_exp.weight = 0.75
        rw.track_ang_vel_z_exp.params["std"] = 0.5

        rw.base_height = RewTerm(
            func=core_mdp.base_height_l2,
            weight=-3.5,
            # weight=-4.2,
            params={
                "target_height": 0.42,
                # "target_height": 0.41,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )

        # discourage extreme joint angles
        rw.dof_pos_limits.weight = -2.0

        # rw.feet_air_time.weight = 0.02
        rw.feet_air_time.weight = 0.04
        # rw.feet_air_time.params["threshold"] = 0.25
        rw.feet_air_time.params["threshold"] = 0.35
        rw.dof_torques_l2.weight = -2.0e-5
        rw.dof_acc_l2.weight = -2.5e-7
        rw.action_rate_l2.weight = -0.003

        rw.undesired_contacts = RewTerm(
            func=core_mdp.undesired_contacts,
            weight=-0.7,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
                "threshold": 1.0,
            },
        )


@configclass
class Go2B1RoughWalkEnvCfg_PLAY(Go2B1RoughWalkEnvCfg):
    """lightweight play/visualization variant for B1"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 5  # adjust at your liking
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class Go2B2StairClimbEnvCfg_PLAY(Go2B2StairClimbEnvCfg):
    """lightweight play/visualization variant for B2"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # self.scene.terrain.max_init_terrain_level = 6  # adjust at your liking
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
