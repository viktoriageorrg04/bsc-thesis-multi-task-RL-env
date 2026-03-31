"""Fam B (rough velocity) task configs for Unitree Go2.

These configs are overrides of Isaac Lab's built-in Go2 rough-velocity task.
The goal is to isolate rough-terrain skills by using one terrain type per task:
B1 = random_rough only, B2 = pyramid_stairs only.
"""

import copy
import math

import isaaclab.envs.mdp as core_mdp
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
    """fam B / task B2: stair climb on pyramid_stairs terrain only"""

    def __post_init__(self):
        super().__post_init__()
        _keep_single_sub_terrain(self, "pyramid_stairs")

        # scale down stair height for Go2 (upstream doesn't do this unlike
        # boxes/random_rough) — 23 cm steps are over half the robot's height
        stairs = self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"]
        stairs.step_height_range = (0.04, 0.16)

        # B2 is forward-biased with low lateral/yaw commands
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.3, 0.8)
        cmd.ranges.lin_vel_y = (-0.15, 0.15)
        cmd.ranges.ang_vel_z = (-0.3, 0.3)
        cmd.ranges.heading = (0.0, 0.0)

        _fix_rough_reward_shaping(self)

        rw = self.rewards
        # climbing stairs requires vertical velocity
        rw.lin_vel_z_l2.weight = -0.5
        # the robot naturally pitches when ascending
        rw.flat_orientation_l2.weight = -1.5
        # lower weight to avoid the policy fighting unstable height readings
        rw.base_height.weight = -5.0


@configclass
class Go2B1RoughWalkEnvCfg_PLAY(Go2B1RoughWalkEnvCfg):
    """lightweight play/visualization variant for B1"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
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
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
