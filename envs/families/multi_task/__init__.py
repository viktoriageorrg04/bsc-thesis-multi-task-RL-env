"""unified multi-task env; gym registration"""

import gymnasium as gym

_RSL_RL_GO2_ROUGH_CFG = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:"
    "UnitreeGo2RoughPPORunnerCfg"
)

gym.register(
    id="MTL-Unified-Unitree-Go2-AllTerrains-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_mtl_env_cfg:Go2MultiTaskEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Unified-Unitree-Go2-AllTerrains-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_mtl_env_cfg:Go2MultiTaskEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Conditioned-Unitree-Go2-AllTerrains-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_mtl_env_cfg:Go2MultiTaskConditionedEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Conditioned-Unitree-Go2-AllTerrains-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_mtl_env_cfg:Go2MultiTaskConditionedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)
