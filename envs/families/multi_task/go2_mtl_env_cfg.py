"""unified multi-task env config: all 5 terrain types in one TerrainGeneratorCfg

  - flat: fast omni-dir walking
  - random_rough: moderate omni-dir walking
  - pyramid_stairs: forward-biased stair climbing
  - stepping_stones: slow, precise forward stepping
  - gap: moderate forward gap crossing

the shared reward function is identical to single-task envs (aka velocity
tracking + regularization), so any performance difference vs. baselines
is attributable to the multi-task training regime
"""

import copy
import math

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

# terrain gen: 5 sub-terrains, equal proportion
# - flat is included so the multi-task policy also sees the baseline terrain
# - proportions are equal (0.2) so each terrain type gets the same share of envs
# - difficulty scaling is handled per-row by Isaac Lab's curriculum sys

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
            proportion=0.2,
            noise_range=(0.0, 0.0),  # zero noise = flat
            noise_step=0.01,
            border_width=0.25,
        ),
        # random rough (fam B1)
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.01, 0.06),  # Go2-scaled (same as UnitreeGo2RoughEnvCfg)
            noise_step=0.01,
            border_width=0.25,
        ),
        # pyramid stairs (fam B2)
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # stepping stones (fam C1)
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.08,
            stone_width_range=(0.25, 0.45),
            stone_distance_range=(0.05, 0.12),
            holes_depth=-10.0,
            platform_width=1.5,
        ),
        # gap crossing (fam C2)
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.2,
            gap_width_range=(0.08, 0.25),
            platform_width=1.5,
        ),
    },
)


@configclass
class Go2MultiTaskEnvCfg(UnitreeGo2RoughEnvCfg):
    """unified multi-task env: 5 terrain types × shared cmd profile

    inerits from ``UnitreeGo2RoughEnvCfg`` to get:
      - Go2-scaled actuators, rewards, height scanner, contact sensors
      - terrain curriculum

    the cmd distribution is the union of all single-task profiles:
      lin_vel_x in [-1.0, 1.0]
      lin_vel_y in [-1.0, 1.0]
      heading   in [-pi, pi]
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = copy.deepcopy(_MTL_TERRAIN_GEN)

        # cmds (union of all task profiles)
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
    """lightweight play/vis variant for the unified multi-task env"""

    def __post_init__(self):
        super().__post_init__()
        # small scene for interactive vis
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spread robots randomly across terrain types instead of curriculum
        self.scene.terrain.max_init_terrain_level = None
        # fewer terrain patches to save GPU memory during vis
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # deterministic
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
