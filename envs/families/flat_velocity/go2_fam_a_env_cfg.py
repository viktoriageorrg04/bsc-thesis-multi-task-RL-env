"""Fam A (flat velocity) task configs for Unitree Go2.

These configs are overrides of Isaac Lab's built-in Go2 flat-velocity task.
The goal here is to define benchmark tasks by changing only command distributions (aka rnd ranges used to gen motion cmds):
A1 = forward-only walking, A2 = omni-directional walking.
"""

import math

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)


@configclass
class Go2A1ForwardWalkEnvCfg(UnitreeGo2FlatEnvCfg):
    """fam A / task A1: forward walk on flat terrain
    - keep terrain, rewards, and obs identical to the Go2 flat baseline
    - restrict cmds to positive forward speed only
    """

    def __post_init__(self):
        super().__post_init__()

        cmd = self.commands.base_velocity
        # A1 is not a heading-tracking task! so, we command explicit yaw-rate = 0 instead
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        # desable "standing" so that every env receives a moving cmd
        cmd.rel_standing_envs = 0.0

        # x in [0.5, 1.0], y = 0, yaw-rate = 0; forward-only distr
        cmd.ranges.lin_vel_x = (0.5, 1.0)  #  always positive forward vel
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        # keep heading range fixed since heading-cmd mode is disabled
        cmd.ranges.heading = (0.0, 0.0)


@configclass
class Go2A2OmniWalkEnvCfg(UnitreeGo2FlatEnvCfg):
    """fam A / task A2: omni-directional walk on flat terrain
    - keep the same flat Go2 base task
    - enable broad x/y translation and turning to test lateral/turning gait
    """

    def __post_init__(self):
        super().__post_init__()

        cmd = self.commands.base_velocity
        # A2 uses heading-command mode for full turning behavior
        cmd.heading_command = True
        cmd.rel_heading_envs = 1.0
        cmd.rel_standing_envs = 0.0

        # x in [-1, 1], y in [-1, 1], heading in [-pi, pi]; omni-dir distr
        cmd.ranges.lin_vel_x = (-1.0, 1.0)  # forward/backward
        cmd.ranges.lin_vel_y = (-1.0, 1.0)  # sideways in y
        cmd.ranges.ang_vel_z = (-1.0, 1.0)  # yaw rotation rate cmd
        cmd.ranges.heading = (-math.pi, math.pi)


# those are just for debug/demo; non-PLAY varinats of the configs above will be used for training and eval

class Go2A1ForwardWalkEnvCfg_PLAY(Go2A1ForwardWalkEnvCfg):
    """lightweight play/vis variant for A1;
    uses fewer envs and disables stochastic perturbations
    """

    def __post_init__(self):
        super().__post_init__()
        # small scene for interactive play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # make behavior deterministic/clean for viewing
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


class Go2A2OmniWalkEnvCfg_PLAY(Go2A2OmniWalkEnvCfg):
    """lightweight play/vis variant for A2;
    uses fewer envs and disables stochastic perturbations.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
