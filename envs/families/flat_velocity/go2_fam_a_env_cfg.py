"""Fam A (flat velocity) task configs for Unitree Go2.

Built on UnitreeGo2RoughEnvCfg (not FlatEnvCfg) so that all 5 tasks share
the same observation space (including height_scan).  The terrain is still a
flat plane — the height scanner simply returns uniform values.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as core_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)
from envs.rewards import stand_base_height_l2


def _apply_flat_reward_fixes(cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Fix Go2 rough-env reward defaults for flat-terrain walking.

    The rough env ships with values that prevent clean locomotion learning:
      - feet_air_time  0.01  (no stepping incentive)
      - flat_orientation  0.0 (no posture signal)
      - dof_torques_l2  -2e-4 (20x too heavy)
    """
    rw = cfg.rewards
    rw.feet_air_time.weight = 0.25
    rw.flat_orientation_l2.weight = -2.5
    rw.dof_torques_l2.weight = -1.0e-5


@configclass
class Go2A1ForwardWalkEnvCfg(UnitreeGo2RoughEnvCfg):
    """Fam A / task A1: forward walk on flat terrain.

    Extends the rough env (keeping the height scanner) but uses a flat plane
    so the observation space matches B1/B2/C2 for cross-evaluation.
    """

    def __post_init__(self):
        super().__post_init__()

        # flat terrain — height scanner stays for obs compatibility
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no curriculum on flat terrain
        self.curriculum.terrain_levels = None

        # fix rough-env reward defaults
        _apply_flat_reward_fixes(self)

        # forward-only commands
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.20
        cmd.ranges.lin_vel_x = (0.35, 1.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)

        # stand quality terms; active only when command is near zero
        self.rewards.stand_still = RewTerm(
            func=locomotion_mdp.stand_still_joint_deviation_l1,
            weight=-0.15,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.stand_base_height = RewTerm(
            func=stand_base_height_l2,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "target_height": 0.38,
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


@configclass
class Go2A2OmniWalkEnvCfg(UnitreeGo2RoughEnvCfg):
    """Fam A / task A2: omni-directional walk on flat terrain.

    Same rough-base approach as A1 for obs space compatibility.
    """

    def __post_init__(self):
        super().__post_init__()

        # flat terrain — height scanner stays for obs compatibility
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no curriculum on flat terrain
        self.curriculum.terrain_levels = None

        # fix rough-env reward defaults
        _apply_flat_reward_fixes(self)

        # omni-directional commands
        cmd = self.commands.base_velocity
        cmd.heading_command = True
        cmd.rel_heading_envs = 1.0
        cmd.rel_standing_envs = 0.05
        cmd.ranges.lin_vel_x = (-1.0, 1.0)
        cmd.ranges.lin_vel_y = (-1.0, 1.0)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        cmd.ranges.heading = (-math.pi, math.pi)

        # always-on height incentive — stronger than A1 to prevent ground-dragging
        self.rewards.base_height = RewTerm(
            func=core_mdp.base_height_l2,
            weight=-4.0,
            params={
                "target_height": 0.40,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )


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
