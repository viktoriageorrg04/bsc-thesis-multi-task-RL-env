"""Minimal benchmark interface v0.1 wrapper (obs/action/success/termination)."""

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import torch

from envs.actions import DEFAULT_ACTION_SCHEMA, validate_action_shape
from envs.observations import DEFAULT_OBSERVATION_SCHEMA
from envs.success import (
    DEFAULT_SUCCESS_CONFIG,
    EpisodeStatsTracker,
    compute_step_success_from_errors,
    compute_tracking_errors,
    get_success_config,
)


@dataclass(frozen=True)
class BenchmarkInterfaceConfig:
    """config for the benchmark wrapper"""

    # optional task ID conditioning; if True append scalar task_id to policy obs
    append_task_id: bool = False
    # cmd + robot names are from Isaac Lab locomotion velocity env cfg
    command_name: str = "base_velocity"
    robot_name: str = "robot"
    # expected horizon from LocomotionVelocityRoughEnvCfg; set by Isaac Lab as a default val
    episode_length_s: float = 20.0
    # whether to include per-step reward breakdown in info
    log_reward_breakdown: bool = True


def _extract_reward_breakdown(env) -> dict[str, torch.Tensor]:
    """extract per-term rewards from Isaac Lab's RewardManager"""
    breakdown = {}
    if not hasattr(env, "reward_manager"):
        return breakdown

    rm = env.reward_manager
    term_names = rm.active_terms

    # _step_reward is shape (num_envs, num_terms); holds rewards/dt for curr step
    if hasattr(rm, "_step_reward"):
        for idx, name in enumerate(term_names):
            breakdown[f"reward/{name}"] = rm._step_reward[:, idx].clone()

    return breakdown


class MinimalBenchmarkWrapper(gym.Wrapper):
    """wraps Isaac Lab envs with explicit benchmark v0.1 interface behavior

    key features for multi-task RL eval:
    - per-task success thresholds via get_success_config()
    - reward term breakdown logging (from RewardManager)
    - per-episode success/failure metrics
    - optional task ID in obs
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        task_id: int,
        family_id: int,
        task_gym_id: str,
        cfg: BenchmarkInterfaceConfig = BenchmarkInterfaceConfig(),
    ):
        super().__init__(env)
        self.task_id = int(task_id)
        self.family_id = int(family_id)
        self.task_gym_id = str(task_gym_id)
        self.cfg = cfg

        self.action_schema = DEFAULT_ACTION_SCHEMA
        self.obs_schema = DEFAULT_OBSERVATION_SCHEMA

        # per-task success config
        self.success_cfg = get_success_config(self.task_gym_id)

        num_envs = int(getattr(env, "num_envs", 1))
        step_dt = float(getattr(env, "step_dt", 0.02))
        device = str(getattr(env, "device", "cpu"))
        self._tracker = EpisodeStatsTracker(
            num_envs=num_envs, step_dt=step_dt, cfg=self.success_cfg, device=device
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._normalize_obs(obs)
        return obs, info

    def step(self, action):
        # action schema enforcement
        if isinstance(action, torch.Tensor):
            validate_action_shape(action, self.action_schema.action_dim)

        obs, reward, terminated, truncated, info = self.env.step(action)

        terminated_t = torch.as_tensor(terminated, device=getattr(self.env, "device", "cpu")).bool()
        truncated_t = torch.as_tensor(truncated, device=getattr(self.env, "device", "cpu")).bool()

        # extract reward breakdown before computing success (dbg)
        if self.cfg.log_reward_breakdown:
            reward_breakdown = _extract_reward_breakdown(self.env)
            info = dict(info)
            info.update(reward_breakdown)

        # success formula
        lin_err_xy, ang_err_z = compute_tracking_errors(
            self.env,
            command_name=self.cfg.command_name,
            robot_name=self.cfg.robot_name,
        )
        step_success = compute_step_success_from_errors(
            lin_err_xy=lin_err_xy,
            ang_err_z=ang_err_z,
            terminated=terminated_t,
            cfg=self.success_cfg,
        )

        episode_rows = self._tracker.update(
            step_success=step_success,
            lin_err_xy=lin_err_xy,
            ang_err_z=ang_err_z,
            terminated=terminated_t,
            truncated=truncated_t,
            task_id=self.task_id,
            family_id=self.family_id,
        )

        obs = self._normalize_obs(obs)

        # append episode metrics when sub-envs finish
        if episode_rows:
            info = dict(info) if not isinstance(info, dict) else info
            info["benchmark/episodes"] = episode_rows

        return obs, reward, terminated, truncated, info

    def _normalize_obs(self, obs: Any):
        # Isaac Lab returns a dict of obs groups
        if isinstance(obs, dict) and "policy" in obs and isinstance(obs["policy"], torch.Tensor):
            policy_obs = obs["policy"]

            # assert expected policy obs dim
            if policy_obs.shape[-1] != self.obs_schema.policy_dim:
                raise ValueError(
                    f"Expected policy obs dim {self.obs_schema.policy_dim}, "
                    f"got {policy_obs.shape[-1]}. All task configs must enable the height scanner."
                )

            if self.cfg.append_task_id:
                task_col = torch.full(
                    (policy_obs.shape[0], 1),
                    float(self.task_id),
                    device=policy_obs.device,
                    dtype=policy_obs.dtype,
                )
                policy_obs = torch.cat((policy_obs, task_col), dim=-1)

            obs = dict(obs)
            obs["policy"] = policy_obs
        return obs
    