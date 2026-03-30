"""Tests for environment registration and basic config specs."""

import math
import atexit
import os
import sys
import ctypes
from pathlib import Path

_DLL_DIR_HANDLES = []
_TORCH_PRIME_ERROR = None


def _prime_torch_runtime_for_windows() -> None:
    """Preload torch runtime DLLs before gymnasium/numpy load conda's OpenMP."""
    global _TORCH_PRIME_ERROR
    if os.name != "nt":
        return
    try:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        conda_bin = Path(sys.prefix) / "Library" / "bin"
        if conda_bin.exists():
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(conda_bin)))

        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.exists():
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(torch_lib)))
            ctypes.CDLL(str(torch_lib / "fbgemm.dll"))
    except Exception as exc:
        _TORCH_PRIME_ERROR = exc


# must run before importing gymnasium (which pulls in numpy/MKL)
_prime_torch_runtime_for_windows()

import gymnasium as gym
import pytest

# h5py must be imported before SimulationApp
try:
    import h5py
except Exception:
    pass

from isaacsim import SimulationApp

# Isaac Sim runtime must be initialized before importing isaaclab/isaaclab_tasks modules
_SIM_APP = SimulationApp({"headless": True})
atexit.register(_SIM_APP.close)

# trigger gym.register() side-effects
import envs


# gym registration

EXPECTED_IDS = [
    "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0",
    "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-Play-v0",
    "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0",
    "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-Play-v0",
    "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0",
    "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-Play-v0",
    "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0",
    "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-Play-v0",
    "MTL-Custom-SteppingStones-Unitree-Go2-C1-v0",
    "MTL-Custom-SteppingStones-Unitree-Go2-C1-Play-v0",
    "MTL-Custom-Gap-Unitree-Go2-C2-v0",
    "MTL-Custom-Gap-Unitree-Go2-C2-Play-v0",
]


@pytest.mark.parametrize("env_id", EXPECTED_IDS)
def test_gym_id_registered(env_id: str):
    assert env_id in gym.registry, f"{env_id} not found in gym.registry"


def _load_fam_a_cfgs():
    if _TORCH_PRIME_ERROR is not None:
        pytest.skip(f"Skipping fam A cfg tests due to torch runtime preload failure: {_TORCH_PRIME_ERROR}")
    try:
        from envs.families.flat_velocity.go2_fam_a_env_cfg import (
            Go2A1ForwardWalkEnvCfg,
            Go2A2OmniWalkEnvCfg,
        )
    except Exception as exc:  # pragma: no cover - runtime/env dependent
        pytest.skip(f"Skipping fam A cfg tests due to Isaac runtime import failure: {exc}")
    return Go2A1ForwardWalkEnvCfg, Go2A2OmniWalkEnvCfg


def _load_fam_b_cfgs():
    if _TORCH_PRIME_ERROR is not None:
        pytest.skip(f"Skipping fam B cfg tests due to torch runtime preload failure: {_TORCH_PRIME_ERROR}")
    try:
        from envs.families.rough_velocity.go2_fam_b_env_cfg import (
            Go2B1RoughWalkEnvCfg,
            Go2B2StairClimbEnvCfg,
        )
    except Exception as exc:  # pragma: no cover - runtime/env dependent
        pytest.skip(f"Skipping fam B cfg tests due to Isaac runtime import failure: {exc}")
    return Go2B1RoughWalkEnvCfg, Go2B2StairClimbEnvCfg


# fam A config specs


def test_a1_command_ranges():
    Go2A1ForwardWalkEnvCfg, _ = _load_fam_a_cfgs()
    cfg = Go2A1ForwardWalkEnvCfg()
    cmd = cfg.commands.base_velocity
    assert cmd.ranges.lin_vel_x == (0.5, 1.0)
    assert cmd.ranges.lin_vel_y == (0.0, 0.0)
    assert cmd.ranges.ang_vel_z == (0.0, 0.0)
    assert cmd.heading_command is False
    assert cmd.rel_standing_envs == 0.0


def test_a2_command_ranges():
    _, Go2A2OmniWalkEnvCfg = _load_fam_a_cfgs()
    cfg = Go2A2OmniWalkEnvCfg()
    cmd = cfg.commands.base_velocity
    assert cmd.ranges.lin_vel_x == (-1.0, 1.0)
    assert cmd.ranges.lin_vel_y == (-1.0, 1.0)
    assert cmd.ranges.ang_vel_z == (-1.0, 1.0)
    assert cmd.ranges.heading == (-math.pi, math.pi)
    assert cmd.heading_command is True
    assert cmd.rel_standing_envs == 0.0


def test_a1_height_scanner_disabled_for_flat():
    Go2A1ForwardWalkEnvCfg, _ = _load_fam_a_cfgs()
    cfg = Go2A1ForwardWalkEnvCfg()
    assert cfg.scene.height_scanner is None
    assert cfg.observations.policy.height_scan is None


def test_a2_height_scanner_disabled_for_flat():
    _, Go2A2OmniWalkEnvCfg = _load_fam_a_cfgs()
    cfg = Go2A2OmniWalkEnvCfg()
    assert cfg.scene.height_scanner is None
    assert cfg.observations.policy.height_scan is None


def test_a1_flat_terrain():
    Go2A1ForwardWalkEnvCfg, _ = _load_fam_a_cfgs()
    cfg = Go2A1ForwardWalkEnvCfg()
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None


# fam B config specs


def test_b1_single_terrain():
    Go2B1RoughWalkEnvCfg, _ = _load_fam_b_cfgs()
    cfg = Go2B1RoughWalkEnvCfg()
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None
    assert list(gen.sub_terrains.keys()) == ["random_rough"]


def test_b2_single_terrain():
    _, Go2B2StairClimbEnvCfg = _load_fam_b_cfgs()
    cfg = Go2B2StairClimbEnvCfg()
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None
    assert list(gen.sub_terrains.keys()) == ["pyramid_stairs"]


def test_b1_command_ranges():
    Go2B1RoughWalkEnvCfg, _ = _load_fam_b_cfgs()
    cfg = Go2B1RoughWalkEnvCfg()
    cmd = cfg.commands.base_velocity
    assert cmd.ranges.lin_vel_x == (-1.0, 1.0)
    assert cmd.ranges.lin_vel_y == (-1.0, 1.0)
    assert cmd.heading_command is True
    assert cmd.rel_standing_envs == 0.0


def test_b2_command_ranges():
    _, Go2B2StairClimbEnvCfg = _load_fam_b_cfgs()
    cfg = Go2B2StairClimbEnvCfg()
    cmd = cfg.commands.base_velocity
    assert cmd.ranges.lin_vel_x == (0.3, 0.8)
    assert cmd.ranges.lin_vel_y == (-0.15, 0.15)
    assert cmd.ranges.ang_vel_z == (-0.3, 0.3)
    assert cmd.heading_command is False
    assert cmd.rel_standing_envs == 0.0


def test_b1_does_not_mutate_b2():
    """configs use deepcopy so B1 terrain trimming must not affect B2."""
    Go2B1RoughWalkEnvCfg, Go2B2StairClimbEnvCfg = _load_fam_b_cfgs()
    cfg_b1 = Go2B1RoughWalkEnvCfg()
    cfg_b2 = Go2B2StairClimbEnvCfg()
    assert "random_rough" not in cfg_b2.scene.terrain.terrain_generator.sub_terrains
    assert "pyramid_stairs" not in cfg_b1.scene.terrain.terrain_generator.sub_terrains


# fam C helpers + config specs


def _load_fam_c_cfgs():
    if _TORCH_PRIME_ERROR is not None:
        pytest.skip(f"Skipping fam C cfg tests due to torch runtime preload failure: {_TORCH_PRIME_ERROR}")
    try:
        from envs.families.agility_terrain.go2_fam_c_env_cfg import (
            Go2C1SteppingStonesEnvCfg,
            Go2C2GapCrossingEnvCfg,
        )
    except Exception as exc:  # pragma: no cover - runtime/env dependent
        pytest.skip(f"Skipping fam C cfg tests due to Isaac runtime import failure: {exc}")
    return Go2C1SteppingStonesEnvCfg, Go2C2GapCrossingEnvCfg


def test_c1_single_terrain():
    Go2C1SteppingStonesEnvCfg, _ = _load_fam_c_cfgs()
    cfg = Go2C1SteppingStonesEnvCfg()
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None
    assert list(gen.sub_terrains.keys()) == ["stepping_stones"]


def test_c2_single_terrain():
    _, Go2C2GapCrossingEnvCfg = _load_fam_c_cfgs()
    cfg = Go2C2GapCrossingEnvCfg()
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None
    assert list(gen.sub_terrains.keys()) == ["gap"]


def test_c1_command_ranges():
    Go2C1SteppingStonesEnvCfg, _ = _load_fam_c_cfgs()
    cfg = Go2C1SteppingStonesEnvCfg()
    cmd = cfg.commands.base_velocity
    assert cmd.ranges.lin_vel_x == (0.3, 0.6)
    assert cmd.ranges.lin_vel_y == (-0.1, 0.1)
    assert cmd.ranges.ang_vel_z == (-0.2, 0.2)
    assert cmd.heading_command is False
    assert cmd.rel_standing_envs == 0.0


def test_c2_command_ranges():
    _, Go2C2GapCrossingEnvCfg = _load_fam_c_cfgs()
    cfg = Go2C2GapCrossingEnvCfg()
    cmd = cfg.commands.base_velocity
    assert cmd.ranges.lin_vel_x == (0.4, 0.8)
    assert cmd.ranges.lin_vel_y == (-0.1, 0.1)
    assert cmd.ranges.ang_vel_z == (-0.2, 0.2)
    assert cmd.heading_command is False
    assert cmd.rel_standing_envs == 0.0


def test_c1_does_not_mutate_c2():
    """C1 and C2 terrain generators are independent deep copies."""
    Go2C1SteppingStonesEnvCfg, Go2C2GapCrossingEnvCfg = _load_fam_c_cfgs()
    cfg_c1 = Go2C1SteppingStonesEnvCfg()
    cfg_c2 = Go2C2GapCrossingEnvCfg()
    assert "gap" not in cfg_c1.scene.terrain.terrain_generator.sub_terrains
    assert "stepping_stones" not in cfg_c2.scene.terrain.terrain_generator.sub_terrains

def test_c1_height_scanner_enabled():
    """C1 inherits rough-env height scanner; must stay enabled."""
    Go2C1SteppingStonesEnvCfg, _ = _load_fam_c_cfgs()
    cfg = Go2C1SteppingStonesEnvCfg()
    assert cfg.scene.height_scanner is not None
    assert cfg.observations.policy.height_scan is not None


def test_c2_height_scanner_enabled():
    """C2 inherits rough-env height scanner; must stay enabled."""
    _, Go2C2GapCrossingEnvCfg = _load_fam_c_cfgs()
    cfg = Go2C2GapCrossingEnvCfg()
    assert cfg.scene.height_scanner is not None
    assert cfg.observations.policy.height_scan is not None
    