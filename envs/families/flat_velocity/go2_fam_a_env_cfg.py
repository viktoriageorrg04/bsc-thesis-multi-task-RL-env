"""Fam A (flat velocity) task configs for Unitree Go2.

Uses UnitreeGo2RoughEnvCfg as the base (not FlatEnvCfg) so that the
height scanner is active and obs dim = 235, matching B/C/MTL envs.
A flat-only terrain generator means the scanner simply returns zeros.

A1 = forward-only walking, A2 = omni-directional walking.
"""

import copy
import math

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

# flat terrain gen; ensures height scanner exists (235D obs)
# but the terrain itself is flat, so scan readings are ~0
_FLAT_TERRAIN_GEN = TerrainGeneratorCfg(
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
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.0, 0.0),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)


@configclass
class Go2A1ForwardWalkEnvCfg(UnitreeGo2RoughEnvCfg):
    """fam A / task A1: forward walk on flat terrain"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = copy.deepcopy(_FLAT_TERRAIN_GEN)

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.5, 1.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)


@configclass
class Go2A2OmniWalkEnvCfg(UnitreeGo2RoughEnvCfg):
    """fam A / task A2: omni-directional walk on flat terrain"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = copy.deepcopy(_FLAT_TERRAIN_GEN)

        cmd = self.commands.base_velocity
        cmd.heading_command = True
        cmd.rel_heading_envs = 1.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (-1.0, 1.0)
        cmd.ranges.lin_vel_y = (-1.0, 1.0)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        cmd.ranges.heading = (-math.pi, math.pi)


@configclass
class Go2A1ForwardWalkEnvCfg_PLAY(Go2A1ForwardWalkEnvCfg):
    """lightweight play/vis variant for A1."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class Go2A2OmniWalkEnvCfg_PLAY(Go2A2OmniWalkEnvCfg):
    """lightweight play/vis variant for A2."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
