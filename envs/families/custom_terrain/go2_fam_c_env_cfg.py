"""Fam C (custom terrain) task configs for Unitree Go2.

These are the custom stress-test tasks (not present in the Isaac Lab default
suite).

C1 = stepping stones (HfSteppingStonesTerrainCfg)
     tests precise foot placement; deep holes punish any missed step
C2 = gap crossing (MeshGapTerrainCfg)
     tests stride planning and balance recovery across discrete gaps

Both are derived from UnitreeGo2RoughEnvCfg so they inherit the Go2-scaled
rewards, actuator config, height scanner, and termination logic unchanged.
The only changes are the terrain gen and the cmd distribution.
"""

import copy

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)


# 10x20 grid of 8x8 m subterrains; each is filled with rnd placed stones at varying heights and spacings
# surrounded by 10 m deep holes so that the robot must step on the stones to survive
_STEPPING_STONES_GEN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    # border_width=20.0,
    border_width=20.0,  # this very much depends on how many parallel envs we have
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,  # heightfield grid res; one height sample per 5 cm 
    vertical_scale=0.005,  # heightfield height unit; 5mm precision
    # no slope correction; stepping stones are discrete obstacles
    slope_threshold=None,
    sub_terrains={
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=1.0,
            stone_height_max=0.08,  # 8 cm max step; ~1 Go2 shin length
            stone_width_range=(0.25, 0.45),  # ~1-2 foot-widths; requires care
            stone_distance_range=(0.05, 0.12),  # gaps that force deliberate stepping
            holes_depth=-10.0,  # deep punishing fall; terminates episode
            platform_width=1.5,  # spawn/reset platform
        ),
    },
)

# same but subterrains are flat platforms with a gap cut around them
_GAP_TERRAIN_GEN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=1.0,
            # narrow gaps test clearance,
            # wider ones test recovery and momentum management
            gap_width_range=(0.08, 0.25),
            platform_width=1.5,  # spawn/reset platform
        ),
    },
)


# C1: Stepping Stones

@configclass
class Go2C1SteppingStonesEnvCfg(UnitreeGo2RoughEnvCfg):
    """Fam C / task C1: stepping stones; tests precise foot placement.

    failure modes:
      - foot slips between stones -> base contact; episode terminates
      - too fast -> over-shoots stone; falls in gap
      - too slow / stops -> zero-cmd penalised by tracking reward
    tuning ranges (adjust after first training run):
      lin_vel_x [0.3, 0.6]; forward speed
      lin_vel_y [-0.1, 0.1]; minimal lateral; stones are forward-oriented
      ang_vel_z [-0.2, 0.2]; slight yaw correction
    """

    def __post_init__(self):
        super().__post_init__()

        # replace terrain generator entirely
        self.scene.terrain.terrain_generator = copy.deepcopy(_STEPPING_STONES_GEN)

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.3, 0.6)  # slow bcs stones require deliberate steps
        cmd.ranges.lin_vel_y = (-0.1, 0.1)  # minimal lateral drift
        cmd.ranges.ang_vel_z = (-0.2, 0.2)  # minor yaw corrections
        cmd.ranges.heading = (0.0, 0.0)


@configclass
class Go2C1SteppingStonesEnvCfg_PLAY(Go2C1SteppingStonesEnvCfg):
    """lightweight play/vis variant for C1."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# C2: Gap Crossing

@configclass
class Go2C2GapCrossingEnvCfg(UnitreeGo2RoughEnvCfg):
    """Fam C / task C2: gap crossing — tests stride planning and recovery.

    failure modes:
      - stride too short -> falls into gap; episode terminates
      - too slow -> stalls at edge; velocity commands penalised
      - excess yaw at gap -> lands sideways; tips over
    initial tuning ranges (adjust after first training run):
      lin_vel_x  [0.4, 0.8]; moderate speed; momentum assists gap crossing
      lin_vel_y  [-0.1, 0.1]; small lateral; gaps are forward-oriented
      ang_vel_z  [-0.2, 0.2]; small yaw; straight approach is safer
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = copy.deepcopy(_GAP_TERRAIN_GEN)

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.4, 0.8)
        cmd.ranges.lin_vel_y = (-0.1, 0.1)
        cmd.ranges.ang_vel_z = (-0.2, 0.2)
        cmd.ranges.heading = (0.0, 0.0)


@configclass
class Go2C2GapCrossingEnvCfg_PLAY(Go2C2GapCrossingEnvCfg):
    """lightweight play/vis variant for C2."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
