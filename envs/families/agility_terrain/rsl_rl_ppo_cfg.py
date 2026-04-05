"""RSL-RL PPO config for Fam C tasks (stepping stones, gap crossing).

Inherits Go2 rough defaults unchanged.  The std-collapse crash that motivated
earlier tweaks (entropy_coef, num_learning_epochs, noise_std_type) is now
handled by a monkey-patch in train.py that clamps std_param >= 1e-6.
"""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2RoughPPORunnerCfg,
)


@configclass
class Go2FamCPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    pass
