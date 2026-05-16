"""
Fam A — velocity tracking on flat terrain.

Uses the rough PPO runner config so the observation space (including
height_scan) matches B1/B2/C2 for cross-evaluation.
"""

import gymnasium as gym

_RSL_RL_GO2_ROUGH_CFG = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:"
    "UnitreeGo2RoughPPORunnerCfg"
)

gym.register(
    id="MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_a_env_cfg:Go2A1ForwardWalkEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Velocity-Flat-Unitree-Go2-A1-Forward-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_a_env_cfg:Go2A1ForwardWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Velocity-Flat-Unitree-Go2-A1-Forward-Legacy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_a_env_cfg:Go2A1ForwardWalkLegacyEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Velocity-Flat-Unitree-Go2-A1-Forward-Legacy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_a_env_cfg:Go2A1ForwardWalkLegacyEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_a_env_cfg:Go2A2OmniWalkEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Velocity-Flat-Unitree-Go2-A2-Omni-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_a_env_cfg:Go2A2OmniWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)
