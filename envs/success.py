"""Unified success / termination logic shared across all task families.

Termination rules (inherited from Isaac Lab's LocomotionVelocityRoughEnvCfg):
  - Failure (terminated=True): base link contacts the ground
        Source: TerminationsCfg.base_contact (DoneTerm with illegal_contact)
        Sensor: ContactSensorCfg on "{ENV_REGEX_NS}/Robot/.*", body_names="base"
  - Time-out (truncated=True): episode reaches the horizon (episode_length_s = 20.0 s)
        Source: TerminationsCfg.time_out (DoneTerm with time_out, time_out=True)

These run inside Isaac Lab's env stepping logic!
=> the benchmark layer only reads the resulting signals.
"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SuccessConfig:
    # arbitrary default thresholds
    eps_lin: float = 0.25   # m/s
    eps_ang: float = 0.50   # rad/s
    min_success_ratio: float = 0.80


# a reusable default to use across wrappers/metrics/checks
DEFAULT_SUCCESS_CONFIG = SuccessConfig()


def compute_tracking_errors(
    env,
    *,
    command_name: str = "base_velocity",
    robot_name: str = "robot",
) -> tuple[torch.Tensor, torch.Tensor]:
    """return per-env tracking errors used by the success metric

    lin_err_xy = ||v_xy - v_xy_cmd||_2
    ang_err_z  = |w_z - w_z_cmd|
    """
    robot = env.scene[robot_name]
    cmd = env.command_manager.get_command(command_name)

    lin_err_xy = torch.linalg.vector_norm(cmd[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
    ang_err_z = torch.abs(cmd[:, 2] - robot.data.root_ang_vel_b[:, 2])
    return lin_err_xy, ang_err_z


def compute_step_success_from_errors(
    lin_err_xy: torch.Tensor,
    ang_err_z: torch.Tensor,
    terminated: torch.Tensor,
    cfg: SuccessConfig = DEFAULT_SUCCESS_CONFIG,
) -> torch.Tensor:
    """per-step success: alive and below linear/yaw thresholds"""
    alive = ~terminated.bool()
    return alive & (lin_err_xy <= cfg.eps_lin) & (ang_err_z <= cfg.eps_ang)


class EpisodeStatsTracker:
    """tracks episode-level success stats for vectorized Isaac Lab envs"""

    def __init__(self, num_envs: int, step_dt: float, cfg: SuccessConfig = DEFAULT_SUCCESS_CONFIG, device: str = "cpu"):
        self.num_envs = num_envs
        self.step_dt = float(step_dt)
        self.cfg = cfg
        self.device = torch.device(device)

        self.success_steps = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.total_steps = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.lin_err_sum = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.ang_err_sum = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.had_failure = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

    # normalizes ids for logging; always returns int scalars for one specific id
    def _id_value(self, value: int | torch.Tensor, env_id: int) -> int:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return int(value.item())
            return int(value[env_id].item())
        return int(value)

    def update(
        self,
        *,
        step_success: torch.Tensor,
        lin_err_xy: torch.Tensor,
        ang_err_z: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        task_id: int | torch.Tensor,
        family_id: int | torch.Tensor,
    ) -> list[dict[str, Any]]:
        """update buffers and emit completed episode stats for done envs"""
        terminated = terminated.bool()
        truncated = truncated.bool()
        done = terminated | truncated

        self.success_steps += step_success.float()
        self.total_steps += 1.0
        self.lin_err_sum += lin_err_xy.float()
        self.ang_err_sum += ang_err_z.float()
        self.had_failure |= terminated

        episode_rows: list[dict[str, Any]] = []
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)

        # finalize per-env episode stats for done envs
        for idx in done_ids.tolist():
            steps = max(float(self.total_steps[idx].item()), 1.0)
            success_ratio = float(self.success_steps[idx].item() / steps)
            mean_lin_err = float(self.lin_err_sum[idx].item() / steps)
            mean_ang_err = float(self.ang_err_sum[idx].item() / steps)
            failure = bool(self.had_failure[idx].item())  # check for failure termination

            episode_success = (success_ratio >= self.cfg.min_success_ratio) and (not failure)
            term_reason = "failure" if failure else "time_out"

            # metrics dict for this episode
            episode_rows.append(
                {
                    "episode_success": episode_success,
                    "success_step_ratio": success_ratio,
                    "mean_lin_vel_error_xy": mean_lin_err,
                    "mean_ang_vel_error_z": mean_ang_err,
                    "alive_time_s": steps * self.step_dt,
                    "termination_reason": term_reason,
                    "task_id": self._id_value(task_id, idx),
                    "family_id": self._id_value(family_id, idx),
                }
            )

        if done_ids.numel() > 0:
            self.success_steps[done_ids] = 0.0
            self.total_steps[done_ids] = 0.0
            self.lin_err_sum[done_ids] = 0.0
            self.ang_err_sum[done_ids] = 0.0
            self.had_failure[done_ids] = False

        return episode_rows
