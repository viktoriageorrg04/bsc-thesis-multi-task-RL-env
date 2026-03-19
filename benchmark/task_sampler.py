"""Multi-task sampling; Task ID vs No Task ID conditions.

TaskSampler picks a task gym-ID for each episode from a weighted
distribution.  
TaskIDConfig controls whether the scalar task-ID is appended to 
the policy obs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


# config

@dataclass(frozen=True)
class TaskIDConfig:
    """Controls the Task-ID obs augmentation ablation.

    append_task_id=True -> scalar task index concatenated to obs (aka Task-ID mode)
    append_task_id=False -> obs unchanged (aka No-Task-ID mode)
    """

    append_task_id: bool = False


# convenience presets
TASK_ID_ON = TaskIDConfig(append_task_id=True)
TASK_ID_OFF = TaskIDConfig(append_task_id=False)


# sampler

# all 6 training task IDs
ALL_TRAIN_TASK_IDS: tuple[str, ...] = (
    "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0",
    "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0",
    "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0",
    "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0",
    "MTL-Custom-SteppingStones-Unitree-Go2-C1-v0",
    "MTL-Custom-Gap-Unitree-Go2-C2-v0",
)


@dataclass
class TaskSampler:
    """Weighted categorical sampler over registered task gym-IDs.
    
    task_ids : sequence of gym-ID strs
        the tasks to sample from; order determines the integer task index
    weights : sequence of floats or None
        sampling probs (will be normalised); ``None`` -> uniform
    seed : int or None
        RNG seed for reproducibility. ``None`` -> non-deterministic
    """

    task_ids: tuple[str, ...] = ALL_TRAIN_TASK_IDS
    weights: tuple[float, ...] | None = None
    seed: int | None = None

    # internal state
    _probs: torch.Tensor = field(init=False, repr=False)
    _generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.task_ids) == 0:
            raise ValueError("task_ids must not be empty")

        # normalise weights -> probs
        if self.weights is None:
            raw = torch.ones(len(self.task_ids))
        else:
            if len(self.weights) != len(self.task_ids):
                raise ValueError(
                    f"len(weights)={len(self.weights)} != "
                    f"len(task_ids)={len(self.task_ids)}"
                )
            raw = torch.tensor(self.weights, dtype=torch.float64)
            if (raw < 0).any():
                raise ValueError("weights must be non-negative")
            if raw.sum() == 0:
                raise ValueError("weights must not all be zero")

        self._probs = (raw / raw.sum()).float()

        self._generator = torch.Generator()
        if self.seed is not None:
            self._generator.manual_seed(self.seed)

    # public API

    @property
    def num_tasks(self) -> int:
        return len(self.task_ids)

    @property
    def probabilities(self) -> torch.Tensor:
        """nomr sampling probabilities (read-only copy)"""
        return self._probs.clone()

    def sample(self) -> tuple[int, str]:
        """draw one task; returns ``(task_index, gym_id)``"""
        idx = int(torch.multinomial(self._probs, 1, generator=self._generator).item())
        return idx, self.task_ids[idx]

    def sample_batch(self, n: int) -> list[tuple[int, str]]:
        """draw *n* tasks independently; returns list of ``(task_index, gym_id)``"""
        indices = torch.multinomial(
            self._probs, n, replacement=True, generator=self._generator
        )
        return [(int(i), self.task_ids[i]) for i in indices]

    def task_index(self, gym_id: str) -> int:
        """return the integer index for a gym-ID, or raise ValueError"""
        return self.task_ids.index(gym_id)