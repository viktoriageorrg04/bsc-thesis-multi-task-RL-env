"""Fam C (agility terrain) task configs for Unitree Go2.

These are the custom stress-test tasks (not present in the Isaac Lab default
suite).  Isaac Lab's default ROUGH_TERRAINS_CFG does NOT include stepping
stones — so C1/C2 are truly custom and need a two-phase training approach.

C1 = stepping stones (HfSteppingStonesTerrainCfg)
     tests precise foot placement; gaps punish any missed step
C2 = gap crossing (MeshGapTerrainCfg)
     tests stride planning and balance recovery across discrete gaps

Two-phase training (needed because stepping stones have no reference impl):
  Phase 1  →  C1-Flat / C2-Flat:  learn walking on gently-rough terrain
              (same cmd profile + rewards, just easier terrain)
  Phase 2  →  C1 / C2:  finetune on the target terrain (--pretrained_checkpoint)

Both are derived from UnitreeGo2RoughEnvCfg so they inherit the Go2-scaled
rewards, actuator config, height scanner, and termination logic unchanged.
"""

import copy

import isaaclab.envs.mdp as core_mdp
import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)


# Phase 1 terrain: gently-rough ground (flat at difficulty 0, up to 3 cm noise
# at difficulty 9).  Same terrain type used for Go2 rough default, just
# scaled down so the robot learns basic locomotion without surprises.
_FLAT_PRETRAIN_GEN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        "gentle_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.0, 0.03),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)

# Stepping-stone specifics (thesis-friendly difficulty):
#   stone_height_max = 0.01  → ±1 cm bumps (NOT difficulty-scaled).
#   stone_width_range = (0.30, 0.50) → wide stones so the robot can walk.
#   stone_distance_range = (0.0, 0.10) → max 10 cm gap at hardest difficulty.
#   holes_depth = -0.2 m → foot can slip in without the body crashing.
#   platform_width = 2.0 m → safe spawn zone for ±0.5 m reset offset.
_STEPPING_STONES_GEN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=None,
    sub_terrains={
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=1.0,
            stone_height_max=0.01,
            stone_width_range=(0.30, 0.50),
            stone_distance_range=(0.0, 0.10),
            holes_depth=-0.2,
            platform_width=2.0,
        ),
    },
)

# C2 terrain: 100 % gap crossing
_GAP_TERRAIN_GEN = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    sub_terrains={
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=1.0,
            gap_width_range=(0.0, 0.20),
            platform_width=2.0,
        ),
    },
)


def _apply_fam_c_rewards(cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Common reward rebalancing for all Fam C tasks.

    The Go2 rough defaults cripple learning on hard single-terrain tasks:
      - feet_air_time  = 0.01   (12.5x below base → no stepping incentive)
      - undesired_contacts = None (disabled → flipping costs nothing)
      - dof_torques_l2 = -2e-4  (20x above base → afraid to move)
      - flat_orientation_l2 = 0  (disabled → can tilt freely)
    We restore base-like values + moderate posture incentives.
    """
    rw = cfg.rewards

    # Velocity tracking: walking >> standing still
    rw.track_lin_vel_xy_exp.weight = 2.0
    rw.track_ang_vel_z_exp.weight = 1.0

    # Stepping incentive: Go2 default 0.01 gives zero foot-lift signal
    rw.feet_air_time.weight = 0.25

    # Penalize thigh/calf ground contact — directly punishes flip/crawl
    rw.undesired_contacts = RewTerm(
        func=core_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )

    # Posture: moderate weights to avoid PPO surrogate-loss explosion
    rw.flat_orientation_l2 = RewTerm(func=core_mdp.flat_orientation_l2, weight=-2.0)
    rw.base_height = RewTerm(
        func=core_mdp.base_height_l2,
        weight=-3.0,
        params={
            "target_height": 0.37,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    # Lighten torque penalty: Go2 default -2e-4 is 20x base
    rw.dof_torques_l2.weight = -1.0e-5


# C1: Stepping Stones

# Phase 1: flat pretrain — learn walking with C1 command profile

@configclass
class Go2C1FlatPretrainEnvCfg(UnitreeGo2RoughEnvCfg):
    """Phase 1 pretrain for C1: learn walking on gentle-rough terrain.

    Same command profile and rewards as C1 stepping stones, but on flat
    terrain so the robot learns basic forward locomotion without stepping
    stone complications.  Use the resulting checkpoint as
    --pretrained_checkpoint for the C1 stepping stones task.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = copy.deepcopy(_FLAT_PRETRAIN_GEN)
        self.scene.terrain.max_init_terrain_level = 0

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.3, 0.6)
        cmd.ranges.lin_vel_y = (-0.1, 0.1)
        cmd.ranges.ang_vel_z = (-0.2, 0.2)
        cmd.ranges.heading = (0.0, 0.0)

        _apply_fam_c_rewards(self)


@configclass
class Go2C1FlatPretrainEnvCfg_PLAY(Go2C1FlatPretrainEnvCfg):
    """lightweight play/vis variant for C1-Flat."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# Phase 2: stepping stones (finetune from Phase 1 checkpoint)

@configclass
class Go2C1SteppingStonesEnvCfg(UnitreeGo2RoughEnvCfg):
    """Fam C / task C1: stepping stones — tests precise foot placement.

    100% stepping stones.  At difficulty 0 the stones are wide (0.50m) with
    zero gaps — essentially flat.  Curriculum scales gap width up and stone
    width down.

    Train with --pretrained_checkpoint pointing to a Phase 1 (C1-Flat)
    checkpoint so the robot starts with walking skills instead of from scratch.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = copy.deepcopy(_STEPPING_STONES_GEN)
        self.scene.terrain.max_init_terrain_level = 0
        self.sim.physx.gpu_collision_stack_size = 2**29

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.3, 0.6)
        cmd.ranges.lin_vel_y = (-0.1, 0.1)
        cmd.ranges.ang_vel_z = (-0.2, 0.2)
        cmd.ranges.heading = (0.0, 0.0)

        _apply_fam_c_rewards(self)


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

# Phase 1: flat pretrain — learn walking with C2 command profile

@configclass
class Go2C2FlatPretrainEnvCfg(UnitreeGo2RoughEnvCfg):
    """Phase 1 pretrain for C2: learn walking on gentle-rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = copy.deepcopy(_FLAT_PRETRAIN_GEN)
        self.scene.terrain.max_init_terrain_level = 0

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.4, 0.8)
        cmd.ranges.lin_vel_y = (-0.1, 0.1)
        cmd.ranges.ang_vel_z = (-0.2, 0.2)
        cmd.ranges.heading = (0.0, 0.0)

        _apply_fam_c_rewards(self)


@configclass
class Go2C2FlatPretrainEnvCfg_PLAY(Go2C2FlatPretrainEnvCfg):
    """lightweight play/vis variant for C2-Flat."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# Phase 2: gap crossing (finetune from Phase 1 checkpoint)

@configclass
class Go2C2GapCrossingEnvCfg(UnitreeGo2RoughEnvCfg):
    """Fam C / task C2: gap crossing — tests stride planning and recovery.

    100% gap terrain.  At difficulty 0 gaps are zero-width — flat ground.
    Curriculum scales gap width up.

    Train with --pretrained_checkpoint pointing to a Phase 1 (C2-Flat)
    checkpoint so the robot starts with walking skills instead of from scratch.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = copy.deepcopy(_GAP_TERRAIN_GEN)
        self.scene.terrain.max_init_terrain_level = 0

        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.4, 0.8)
        cmd.ranges.lin_vel_y = (-0.1, 0.1)
        cmd.ranges.ang_vel_z = (-0.2, 0.2)
        cmd.ranges.heading = (0.0, 0.0)

        _apply_fam_c_rewards(self)


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
