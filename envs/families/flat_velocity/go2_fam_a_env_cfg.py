"""Fam A (flat velocity) task configs for Unitree Go2.

Aligned to Isaac Lab's official Go2 flat setup:
- flat plane terrain
- no height scanner / no height_scan observation term
- flat reward tuning
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from envs.rewards import stand_base_height_l2


@configclass
class Go2A1ForwardWalkEnvCfg(UnitreeGo2FlatEnvCfg):
    """Fam A / task A1: forward walk on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        # A1 specialization: forward-only command profile
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        # increase zero-command exposure so policy learns to stand still
        cmd.rel_standing_envs = 0.15
        # keep forward-walk focus, but ease the minimum speed for earlier stable walking
        cmd.ranges.lin_vel_x = (0.3, 1.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)

        # stand quality terms; active only when command is near zero
        self.rewards.stand_still = RewTerm(
            func=locomotion_mdp.stand_still_joint_deviation_l1,
            weight=-0.2,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stand_base_height = RewTerm(
            func=stand_base_height_l2,
            weight=-5.0,
            params={
                "command_name": "base_velocity",
                "target_height": 0.34,
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class Go2A2OmniWalkEnvCfg(UnitreeGo2FlatEnvCfg):
    """Fam A / task A2: omni-directional walk on flat terrain."""

    def __post_init__(self):
        super().__post_init__()
        # Keep Isaac Lab default flat omni command profile.
        cmd = self.commands.base_velocity
        cmd.heading_command = True
        cmd.rel_heading_envs = 1.0
        cmd.rel_standing_envs = 0.10
        cmd.ranges.lin_vel_x = (-1.0, 1.0)
        cmd.ranges.lin_vel_y = (-1.0, 1.0)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        cmd.ranges.heading = (-math.pi, math.pi)


@configclass
class Go2A1ForwardWalkEnvCfg_PLAY(Go2A1ForwardWalkEnvCfg):
    """Lightweight play/vis variant for A1."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class Go2A2OmniWalkEnvCfg_PLAY(Go2A2OmniWalkEnvCfg):
    """Lightweight play/vis variant for A2."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
