"""Fam B - velocity tracking on rough / irregular terrain (Isaac Lab built-in)."""

# same rationale as fam A

import gymnasium as gym

_RSL_RL_GO2_ROUGH_CFG = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:"
    "UnitreeGo2RoughPPORunnerCfg"
)

gym.register(
    id="MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_b_env_cfg:Go2B1RoughWalkEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_b_env_cfg:Go2B1RoughWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_b_env_cfg:Go2B2StairClimbEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_b_env_cfg:Go2B2StairClimbEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)
