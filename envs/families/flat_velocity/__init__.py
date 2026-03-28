"""
Fam A — velocity tracking on flat terrain.
"""

import gymnasium as gym

_RSL_RL_GO2_FLAT_CFG = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:"
    "UnitreeGo2FlatPPORunnerCfg"
)

gym.register(
    id="MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_a_env_cfg:Go2A1ForwardWalkEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_FLAT_CFG,
    },
)

gym.register(
    id="MTL-Velocity-Flat-Unitree-Go2-A1-Forward-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_a_env_cfg:Go2A1ForwardWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_FLAT_CFG,
    },
)

gym.register(
    id="MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_a_env_cfg:Go2A2OmniWalkEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_FLAT_CFG,
    },
)

gym.register(
    id="MTL-Velocity-Flat-Unitree-Go2-A2-Omni-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_a_env_cfg:Go2A2OmniWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_FLAT_CFG,
    },
)
