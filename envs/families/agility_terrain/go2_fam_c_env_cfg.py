"""Fam C (agility terrain) task config for Unitree Go2.

C2 = gap crossing (MeshGapTerrainCfg); tests stride planning and balance recovery.

Built on UnitreeGo2RoughEnvCfg with B1-proven reward weights.
Mixed terrain (70% gap + 30% random_rough) provides walking practice
alongside gap crossing. Single-phase training — no pretrain needed.
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


# C2 terrain: 100% gap crossing
# gap_width_range = (0.0, 0.20) -> zero gap at difficulty 0 (flat), 20 cm at max
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
            gap_width_range=(0.0, 0.20),
            platform_width=2.0,
        ),
    },
)


def _apply_c2_rewards(cfg: UnitreeGo2RoughEnvCfg) -> None:
    """B1-proven reward structure for C2 gap crossing."""
    rw = cfg.rewards

    # velocity tracking (B1 values)
    rw.track_lin_vel_xy_exp.weight = 2.0
    rw.track_ang_vel_z_exp.weight = 1.0

    # stepping incentive (threshold lowered from default 0.5 to 0.25 —
    # at walking speed 0.4-0.8 m/s a normal trot step has ~0.3s swing,
    # which is above 0.25 but below the default 0.5, so the default
    # penalizes normal walking and rewards only unrealistically slow steps)
    rw.feet_air_time.weight = 0.5
    rw.feet_air_time.params["threshold"] = 0.25

    # penalize thigh/calf ground contact
    rw.undesired_contacts = RewTerm(
        func=core_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )

    # posture: B1-proven weights
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


@configclass
class Go2C2GapCrossingEnvCfg(UnitreeGo2RoughEnvCfg):
    """Fam C / task C2: gap crossing with B1-proven rewards.

    70% gap terrain + 30% random rough for curriculum-based training.
    At difficulty 0, gaps are zero-width (flat). Curriculum scales gap
    width up to 20 cm at max difficulty.
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

        _apply_c2_rewards(self)


@configclass
class Go2C2GapCrossingEnvCfg_PLAY(Go2C2GapCrossingEnvCfg):
    """Lightweight play/vis variant for C2."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = 5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# legacy flat-pretrain configs (kept for backward-compatible gym registration)


@configclass
class Go2C2FlatPretrainEnvCfg(Go2C2GapCrossingEnvCfg):
    """Flat pretrain variant: same rewards, curriculum disabled at level 0."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = None


@configclass
class Go2C2FlatPretrainEnvCfg_PLAY(Go2C2FlatPretrainEnvCfg):
    """Lightweight play/vis variant for C2-Flat."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
