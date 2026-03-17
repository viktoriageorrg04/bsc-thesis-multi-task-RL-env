"""Fam B (rough velocity) task configs for Unitree Go2.

These configs are overrides of Isaac Lab's built-in Go2 rough-velocity task.
The goal is to isolate rough-terrain skills by using one terrain type per task:
B1 = random_rough only, B2 = pyramid_stairs only.
"""

import copy
import math

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

    # copy first so task-specific mutations do not leak across configs
    terrain_generator = copy.deepcopy(generator)
    terrain_generator.sub_terrains = {terrain_key: terrain_generator.sub_terrains[terrain_key]}
    cfg.scene.terrain.terrain_generator = terrain_generator


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


@configclass
class Go2B2StairClimbEnvCfg(UnitreeGo2RoughEnvCfg):
    """fam B / task B2: stair climb on pyramid_stairs terrain only"""

    def __post_init__(self):
        super().__post_init__()
        _keep_single_sub_terrain(self, "pyramid_stairs")

        # B2 is forward-biased with low lateral/yaw commands
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.3, 0.8)
        cmd.ranges.lin_vel_y = (-0.15, 0.15)  # a small lateral drift so that climbing remains mostly forward
        cmd.ranges.ang_vel_z = (-0.3, 0.3)  # minor yaw corrections for balance/foot placement
        cmd.ranges.heading = (0.0, 0.0)


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
