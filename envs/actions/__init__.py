"""Shared action schema for all benchmark tasks."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActionSchema:
    # (default) Isaac Lab locomotion velocity tasks use joint-pos ctrl
    control_mode: str = "joint_position"
    # Go2 has 12 actuated leg joints (3 per leg x 4)
    action_dim: int = 12
    # matches the task configs using joint_names=[".*"]
    joint_name_pattern: str = ".*"


DEFAULT_ACTION_SCHEMA = ActionSchema()


def validate_action_shape(action: torch.Tensor, action_dim: int = DEFAULT_ACTION_SCHEMA.action_dim) -> None:
    """Raise an error if the action's last dimension is not action_dim."""
    if action.shape[-1] != action_dim:
        raise ValueError(f"Expected action last dim {action_dim}, got {action.shape[-1]}")