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
import math
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

# ── terrain generator ──────────────────────────────────────────────────────────
# 4 sub-terrains, equal 25% proportion.  Difficulty ranges match single-task
# baselines so the multi-task policy faces equivalent challenges.

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
        # random rough (fam B1) — matches Go2 rough env scaling
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        # pyramid stairs (fam B2) — matches Go2-scaled step heights
        "pyramid_stairs_inv": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.04, 0.16),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # gap crossing (fam C2) — matches single-task gap config
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.25,
            gap_width_range=(0.0, 0.20),
            platform_width=2.0,
        ),
    },
)


# reward fixes

def _safe_base_height_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """base_height_l2 with finite-safe ray handling for mixed terrains (incl. gaps)."""
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        ray_z = sensor.data.ray_hits_w[..., 2]
        valid = torch.isfinite(ray_z)
        ray_z_clean = torch.where(valid, ray_z, torch.zeros_like(ray_z))
        valid_count = valid.float().sum(dim=1).clamp(min=1.0)
        ground_z = ray_z_clean.sum(dim=1) / valid_count
        all_invalid = ~valid.any(dim=1)
        ground_z[all_invalid] = asset.data.root_pos_w[all_invalid, 2] - target_height
        adjusted_target = target_height + ground_z
    else:
        adjusted_target = target_height
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target)


def _apply_mtl_rewards(cfg: UnitreeGo2RoughEnvCfg) -> None:
    """B1-proven reward set shared across all terrains in the unified env."""
    rw = cfg.rewards

    # velocity tracking (stronger than Go2 defaults)
    rw.track_lin_vel_xy_exp.weight = 2.0
    rw.track_ang_vel_z_exp.weight = 1.0

    # stepping incentive (Go2 default is 0.01 — kills stepping)
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

        self.scene.terrain.terrain_generator = copy.deepcopy(_MTL_TERRAIN_GEN)

        # B1-proven reward set
        _apply_mtl_rewards(self)

        # commands (union of all single-task profiles)
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
