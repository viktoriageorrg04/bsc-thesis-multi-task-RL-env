"""Fam C — agility terrains: gap crossing"""

import gymnasium as gym

_RSL_RL_FAM_C_CFG = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:"
    "UnitreeGo2RoughPPORunnerCfg"
)

# C2: Gap Crossing

gym.register(
    id="MTL-Custom-Gap-Unitree-Go2-C2-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_c_env_cfg:Go2C2FlatPretrainEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_FAM_C_CFG,
    },
)

gym.register(
    id="MTL-Custom-Gap-Unitree-Go2-C2-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_c_env_cfg:Go2C2FlatPretrainEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_FAM_C_CFG,
    },
)

gym.register(
    id="MTL-Custom-Gap-Unitree-Go2-C2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_c_env_cfg:Go2C2GapCrossingEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_FAM_C_CFG,
    },
)

gym.register(
    id="MTL-Custom-Gap-Unitree-Go2-C2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_c_env_cfg:Go2C2GapCrossingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_FAM_C_CFG,
    },
)
