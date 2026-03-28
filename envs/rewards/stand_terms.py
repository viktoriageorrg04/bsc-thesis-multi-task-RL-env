"""Custom reward helpers for command-conditioned standing behavior."""

from __future__ import annotations

import torch

import isaaclab.envs.mdp as core_mdp
from isaaclab.managers import SceneEntityCfg


def stand_base_height_l2(
    env,
    command_name: str,
    target_height: float,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base-height error only when commanded velocity is near zero."""
    base_height_err = core_mdp.base_height_l2(
        env=env,
        target_height=target_height,
        asset_cfg=asset_cfg,
        sensor_cfg=None,
    )
    command = env.command_manager.get_command(command_name)
    stand_mask = torch.norm(command[:, :2], dim=1) < command_threshold
    return base_height_err * stand_mask
