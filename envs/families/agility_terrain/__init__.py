"""Fam C — agility terrains: gap crossing"""

import gymnasium as gym

# custom runner cfg that switches noise_std_type to "log" so that the policy
# std = exp(log_std) is always positive — prevents PPO std collapse crash
_RSL_RL_FAM_C_CFG = f"{__name__}.rsl_rl_ppo_cfg:Go2FamCPPORunnerCfg"

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
