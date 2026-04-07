"""Fam C (agility terrain) task config for Unitree Go2.

C2 = gap crossing (MeshGapTerrainCfg); tests stride planning and balance recovery

Isaac Lab's default ROUGH_TERRAINS_CFG does NOT include gap crossing, so C2
needs a two-phase training approach:
Phase 1 -> C2-Flat:  learn walking on gently-rough terrain; (same cmd profile + rewards, just easier terrain)
Phase 2 -> C2:  finetune on gap terrain (--pretrained_checkpoint)
"""

import copy

import torch

import isaaclab.envs.mdp as core_mdp
import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)


def _safe_base_height_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """base_height_l2 with inf/nan clamping for broken ray-cast hits."""
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        ray_z = sensor.data.ray_hits_w[..., 2]
        # filter out inf/nan rays before averaging
        valid = torch.isfinite(ray_z)
        ray_z_clean = torch.where(valid, ray_z, torch.zeros_like(ray_z))
        valid_count = valid.float().sum(dim=1).clamp(min=1.0)
        ground_z = ray_z_clean.sum(dim=1) / valid_count
        # if all rays miss for an env, give zero penalty
        all_invalid = ~valid.any(dim=1)
        ground_z[all_invalid] = asset.data.root_pos_w[all_invalid, 2] - target_height
        adjusted_target = target_height + ground_z
    else:
        adjusted_target = target_height
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target)


def _same_end_pair_penalty(
    env,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize bounding gait: both front or both rear feet airborne together.

    Go2 foot order (.*FOOT regex): FL=0, FR=1, RL=2, RR=3.
    Returns 0 (trot/walk) to 2 (all feet off ground).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    in_air = (air_time > 0).float()
    front_pair = in_air[:, 0] * in_air[:, 1]  # FL + FR both airborne
    rear_pair = in_air[:, 2] * in_air[:, 3]  # RL + RR both airborne
    return front_pair + rear_pair


# Phase 1 terrain: gently-rough ground
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

# C2 terrain: 100 % gap crossing
# gap_width_range = (0.0, 0.20) -> zero gap at difficulty 0, 20 cm at max
# platform_width = 2.0 m -> spawn zone
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
    rw = cfg.rewards

    # velocity tracking
    rw.track_lin_vel_xy_exp.weight = 2.0
    rw.track_ang_vel_z_exp.weight = 1.0

    # stepping incentive
    rw.feet_air_time.weight = 0.25

    # penalize thigh/calf ground contact (directly punishes flip/crawl)
    rw.undesired_contacts = RewTerm(
        func=core_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )

    # posture: strong orientation to prevent body tilt (matches B1)
    rw.flat_orientation_l2 = RewTerm(func=core_mdp.flat_orientation_l2, weight=-5.0)
    rw.base_height = RewTerm(
        func=_safe_base_height_l2,
        weight=-10.0,
        params={
            "target_height": 0.37,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    # torque penalty
    rw.dof_torques_l2.weight = -1.0e-5



# Phase 1: flat pretrain; learn walking with C2 cmd profile

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
    """Fam C / task C2: gap crossing; tests stride planning and recovery.

    100% gap terrain. At difficulty 0 gaps are zero-width — flat ground.
    Curriculum scales gap width up to 20 cm at max difficulty.

    Train with --pretrained_checkpoint pointing to a Phase 1 (C2-Flat)
    ckpt so the robot starts with walking skills instead of from scratch.
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
