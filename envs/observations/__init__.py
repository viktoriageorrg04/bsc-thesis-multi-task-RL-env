"""Shared obs schema utilities for benchmark tasks."""

from dataclasses import dataclass, field


def _grid_ray_count(size: tuple[float, float] = (1.6, 1.0), resolution: float = 0.1) -> int:
    nx = int(round(size[0] / resolution)) + 1  # 17
    ny = int(round(size[1] / resolution)) + 1  # 11
    return nx * ny  # 187


@dataclass(frozen=True)
class ObservationSchema:
    proprio_dim: int = 48
    height_scan_dim: int = _grid_ray_count()
    terms: tuple[str, ...] = field(
        default=(
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
            "height_scan",
        )
    )

    @property
    def policy_dim(self) -> int:
        return self.proprio_dim + self.height_scan_dim


# a reusable default to use across wrappers/metrics/checks
DEFAULT_OBSERVATION_SCHEMA = ObservationSchema()
