"""Fam C — custom stress-test terrains: stepping stones + gap crossing"""

import gymnasium as gym

# reuse the rough PPO runner cfg; same obs/action space as fam B
_RSL_RL_GO2_ROUGH_CFG = (
    "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:"
    "UnitreeGo2RoughPPORunnerCfg"
)

gym.register(
    id="MTL-Custom-SteppingStones-Unitree-Go2-C1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_c_env_cfg:Go2C1SteppingStonesEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Custom-SteppingStones-Unitree-Go2-C1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_c_env_cfg:Go2C1SteppingStonesEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Custom-Gap-Unitree-Go2-C2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_c_env_cfg:Go2C2GapCrossingEnvCfg",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)

gym.register(
    id="MTL-Custom-Gap-Unitree-Go2-C2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_fam_c_env_cfg:Go2C2GapCrossingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _RSL_RL_GO2_ROUGH_CFG,
    },
)
