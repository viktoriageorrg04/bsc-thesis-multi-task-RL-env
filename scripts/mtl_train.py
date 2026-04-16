"""Unified multi-task training driver with stability defaults.

This script is intentionally separate from scripts/train.py so single-task
baseline training behavior remains unchanged.

Usage:
    isaaclab.bat -p scripts/mtl_train.py --headless --num_envs 1024 --max_iterations 1500
    isaaclab.bat -p scripts/mtl_train.py --headless --sampling_strategy focus --focus_terrain stairs --focus_prob 0.7
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import random
import sys
import time
from dataclasses import MISSING
from datetime import datetime
from glob import glob

from isaaclab.app import AppLauncher

import numpy as np


_FOCUS_TERRAIN_TO_SUBTERRAIN = {
    "flat": "flat",
    "rough": "random_rough",
    "stairs": "pyramid_stairs",
    "gap": "gap",
}


def _is_missing(value) -> bool:
    return isinstance(value, type(MISSING))


def _has_non_missing_attr(obj, attr_name: str) -> bool:
    return hasattr(obj, attr_name) and not _is_missing(getattr(obj, attr_name))


def _update_agent_cfg_from_cli(agent_cfg: RslRlBaseRunnerCfg, args) -> RslRlBaseRunnerCfg:
    if args.seed is not None:
        agent_cfg.seed = random.randint(0, 10000) if args.seed == -1 else args.seed
    if args.resume:
        agent_cfg.resume = True
    if args.load_run is not None:
        agent_cfg.load_run = args.load_run
    if args.checkpoint is not None:
        agent_cfg.load_checkpoint = args.checkpoint
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name
    if args.logger is not None:
        agent_cfg.logger = args.logger
    if agent_cfg.logger in {"wandb", "neptune"} and args.log_project_name:
        agent_cfg.wandb_project = args.log_project_name
        agent_cfg.neptune_project = args.log_project_name
    return agent_cfg


def _build_terrain_sampling_proportions(
    strategy: str,
    focus_terrain: str | None,
    focus_prob: float,
) -> dict[str, float]:
    """Create per-subterrain sampling probabilities.

    Strategies:
      - uniform: 25/25/25/25
      - focus: one terrain gets `focus_prob`, remainder split over others
    """
    if strategy == "uniform":
        return {key: 0.25 for key in _FOCUS_TERRAIN_TO_SUBTERRAIN.values()}

    if strategy != "focus":
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    if focus_terrain is None:
        raise ValueError("--focus_terrain is required when --sampling_strategy focus")

    focus_key = _FOCUS_TERRAIN_TO_SUBTERRAIN[focus_terrain]
    focus_prob = float(focus_prob)
    if not (0.25 <= focus_prob <= 0.97):
        raise ValueError("--focus_prob must be in [0.25, 0.97] for stable rehearsal sampling.")

    other_keys = [k for k in _FOCUS_TERRAIN_TO_SUBTERRAIN.values() if k != focus_key]
    rehearse_prob_each = (1.0 - focus_prob) / len(other_keys)
    proportions = {k: rehearse_prob_each for k in other_keys}
    proportions[focus_key] = focus_prob
    return proportions


def _apply_terrain_sampling_profile(env_cfg: ManagerBasedRLEnvCfg, args) -> None:
    """Apply terrain sampling profile directly to the unified MTL terrain generator."""
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    terrain_gen = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_gen is None or not hasattr(terrain_gen, "sub_terrains"):
        raise ValueError("Expected a terrain generator with sub_terrains for unified MTL training.")

    proportions = _build_terrain_sampling_proportions(
        strategy=args.sampling_strategy,
        focus_terrain=args.focus_terrain,
        focus_prob=args.focus_prob,
    )

    missing = [k for k in proportions if k not in terrain_gen.sub_terrains]
    if missing:
        raise ValueError(
            f"Terrain generator is missing expected sub-terrains: {missing}. "
            "Use this script with MTL-Unified-Unitree-Go2-AllTerrains-v0."
        )

    for key, prob in proportions.items():
        terrain_gen.sub_terrains[key].proportion = float(prob)

    pretty = ", ".join(f"{k}={proportions[k]:.3f}" for k in sorted(proportions))
    print(f"[INFO] Terrain sampling profile ({args.sampling_strategy}): {pretty}")


def _read_scalar_series(log_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not glob(os.path.join(log_dir, "events.out.tfevents.*")):
        return None

    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])
    if tag not in scalar_tags:
        return None

    points = ea.Scalars(tag)
    if not points:
        return None

    steps = np.array([p.step for p in points], dtype=np.int64)
    values = np.array([p.value for p in points], dtype=np.float64)
    return steps, values


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _safe_tag_name(tag: str) -> str:
    return tag.replace("/", "_").replace(" ", "_")


def _export_learning_curves(log_dir: str, tags: list[str], smoothing: int) -> None:
    output_dir = os.path.join(log_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    available = []
    for tag in tags:
        series = _read_scalar_series(log_dir, tag)
        if series is None:
            print(f"[WARN] Plot tag not found in logs: {tag}")
            continue
        available.append((tag, series[0], series[1]))

        csv_path = os.path.join(output_dir, f"{_safe_tag_name(tag)}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,value,smoothed\n")
            smoothed = _moving_average(series[1], smoothing)
            for step, value, smooth in zip(series[0], series[1], smoothed, strict=False):
                f.write(f"{int(step)},{float(value):.10g},{float(smooth):.10g}\n")

    if not available:
        print("[WARN] No requested tags were available; skipped curve plot export.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; CSVs exported but no PNG plot was created ({exc}).")
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(11, 3.5 * len(available)), squeeze=False)
    for idx, (tag, steps, values) in enumerate(available):
        ax = axes[idx, 0]
        smoothed = _moving_average(values, smoothing)
        ax.plot(steps, values, linewidth=1.0, alpha=0.3, label="raw")
        ax.plot(steps, smoothed, linewidth=2.0, label=f"ma({max(1, smoothing)})")
        ax.set_title(tag)
        ax.set_xlabel("iteration")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    png_path = os.path.join(output_dir, "learning_curves.png")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Learning curves exported to: {output_dir}")


def _apply_mtl_stability_overrides(agent_cfg: RslRlBaseRunnerCfg, args) -> None:
    """Apply robust policy/distribution overrides for unified MTL runs only."""
    if _has_non_missing_attr(agent_cfg, "policy"):
        policy = agent_cfg.policy
        if hasattr(policy, "noise_std_type"):
            policy.noise_std_type = args.noise_std_type
        if hasattr(policy, "init_noise_std"):
            policy.init_noise_std = args.init_noise_std

    if _has_non_missing_attr(agent_cfg, "algorithm"):
        algo = agent_cfg.algorithm
        if args.learning_rate is not None and hasattr(algo, "learning_rate"):
            algo.learning_rate = float(args.learning_rate)
        if args.entropy_coef is not None and hasattr(algo, "entropy_coef"):
            algo.entropy_coef = float(args.entropy_coef)
        if args.desired_kl is not None and hasattr(algo, "desired_kl"):
            algo.desired_kl = float(args.desired_kl)
        if args.max_grad_norm is not None and hasattr(algo, "max_grad_norm"):
            algo.max_grad_norm = float(args.max_grad_norm)
        if args.schedule is not None and hasattr(algo, "schedule"):
            algo.schedule = str(args.schedule)


def _ensure_log_std_distribution(agent_cfg: RslRlBaseRunnerCfg, args) -> None:
    """After deprecation handling, enforce actor distribution config explicitly."""
    if not _has_non_missing_attr(agent_cfg, "actor"):
        return

    actor = agent_cfg.actor
    dist_cfg = getattr(actor, "distribution_cfg", None)
    if dist_cfg is None:
        actor.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=float(args.init_noise_std),
            std_type=str(args.noise_std_type),
        )
        return

    if hasattr(dist_cfg, "init_std"):
        dist_cfg.init_std = float(args.init_noise_std)
    if hasattr(dist_cfg, "std_type"):
        dist_cfg.std_type = str(args.noise_std_type)


def _enable_distribution_safety_patch() -> None:
    """Patch rsl_rl Gaussian distribution update paths with finite/std clamps."""
    import torch
    from torch.distributions import Normal
    from rsl_rl.modules.distribution import GaussianDistribution, HeteroscedasticGaussianDistribution

    if getattr(GaussianDistribution, "_bsc_safe_patch_applied", False):
        return

    def _sanitize_std(std: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(std, nan=1.0, posinf=5.0, neginf=1.0e-6).clamp(min=1.0e-6, max=5.0)

    def _sanitize_mean(mean: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(mean, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

    def _safe_gaussian_update(self, mlp_output: torch.Tensor) -> None:
        mean = _sanitize_mean(mlp_output)
        if self.std_type == "scalar":
            self.std_param.data.nan_to_num_(nan=1.0, posinf=5.0, neginf=1.0e-6)
            self.std_param.data.clamp_(min=1.0e-6, max=5.0)
            std = self.std_param.expand_as(mean)
        elif self.std_type == "log":
            self.log_std_param.data.nan_to_num_(nan=0.0, posinf=2.0, neginf=-20.0)
            self.log_std_param.data.clamp_(min=-20.0, max=2.0)
            std = torch.exp(self.log_std_param).expand_as(mean)
        else:
            raise ValueError(f"Unknown std_type: {self.std_type}")
        self._distribution = Normal(mean, _sanitize_std(std))

    def _safe_hetero_update(self, mlp_output: torch.Tensor) -> None:
        if self.std_type == "scalar":
            mean, std = torch.unbind(mlp_output, dim=-2)
            std = _sanitize_std(std)
        elif self.std_type == "log":
            mean, log_std = torch.unbind(mlp_output, dim=-2)
            log_std = torch.nan_to_num(log_std, nan=0.0, posinf=2.0, neginf=-20.0).clamp(min=-20.0, max=2.0)
            std = _sanitize_std(torch.exp(log_std))
        else:
            raise ValueError(f"Unknown std_type: {self.std_type}")
        mean = _sanitize_mean(mean)
        self._distribution = Normal(mean, std)

    GaussianDistribution.update = _safe_gaussian_update
    HeteroscedasticGaussianDistribution.update = _safe_hetero_update
    GaussianDistribution._bsc_safe_patch_applied = True

    print("[INFO] Applied runtime distribution safety patch for rsl_rl Gaussian policies.")


parser = argparse.ArgumentParser(description="Train unified MTL policy with robust defaults.")
parser.add_argument(
    "--task",
    type=str,
    default="MTL-Unified-Unitree-Go2-AllTerrains-v0",
    help="Gym ID (defaults to unified all-terrains task).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel envs.")
parser.add_argument("--seed", type=int, default=None, help="Random seed (-1 for random).")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max PPO iterations.")
parser.add_argument(
    "--sampling_strategy",
    type=str,
    default="uniform",
    choices=("uniform", "focus"),
    help="Task sampling strategy for terrain mixture. 'focus' = one terrain + rehearsal on others.",
)
parser.add_argument(
    "--focus_terrain",
    type=str,
    default=None,
    choices=tuple(_FOCUS_TERRAIN_TO_SUBTERRAIN.keys()),
    help="Terrain to prioritize when sampling_strategy=focus.",
)
parser.add_argument(
    "--focus_prob",
    type=float,
    default=0.70,
    help="Sampling probability for focused terrain (rest is split across other terrains).",
)
parser.add_argument(
    "--plot_learning_curves",
    action="store_true",
    default=False,
    help="Export learning-curve CSV/PNG from TensorBoard scalars after training.",
)
parser.add_argument("--plot_smoothing", type=int, default=25, help="Moving-average window for plotted curves.")
parser.add_argument(
    "--plot_tags",
    nargs="+",
    default=[
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Loss/value_function",
        "Loss/surrogate",
        "Loss/entropy",
    ],
    help="TensorBoard scalar tags to export/plot.",
)

parser.add_argument(
    "--noise_std_type",
    type=str,
    default="log",
    choices=("scalar", "log"),
    help="Actor noise std parameterization. Default: log.",
)
parser.add_argument("--init_noise_std", type=float, default=0.7, help="Initial actor noise std.")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="PPO learning rate.")
parser.add_argument("--entropy_coef", type=float, default=0.005, help="PPO entropy coefficient.")
parser.add_argument("--desired_kl", type=float, default=0.01, help="PPO desired KL (adaptive schedule).")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="PPO grad norm clip.")
parser.add_argument(
    "--schedule",
    type=str,
    default="fixed",
    choices=("adaptive", "fixed"),
    help="PPO LR schedule.",
)
parser.add_argument(
    "--distribution_safety_patch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply runtime safety patch to rsl_rl Gaussian distributions (sanitize NaN/invalid std).",
)

parser.add_argument("--experiment_name", type=str, default="unitree_go2_mtl_unified", help="Log experiment name.")
parser.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
parser.add_argument("--pretrained_checkpoint", type=str, default=None, help="Weights to initialize before training.")
parser.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
parser.add_argument("--log_project_name", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from tensorboard.backend.event_processing import event_accumulator

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlMLPModelCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import envs.registry


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if "MTL-Unified-Unitree-Go2-AllTerrains" not in args_cli.task:
        raise ValueError(
            f"scripts/mtl_train.py is intended for unified MTL tasks only. Received: {args_cli.task}"
        )

    agent_cfg = _update_agent_cfg_from_cli(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    _apply_terrain_sampling_profile(env_cfg, args_cli)

    _apply_mtl_stability_overrides(agent_cfg, args_cli)

    installed_rsl_rl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    _ensure_log_std_distribution(agent_cfg, args_cli)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_root_path = os.path.join(project_root, "logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging root directory: {log_root_path}")
    print(f"[INFO] Saving this run to: {log_dir}")

    if _has_non_missing_attr(agent_cfg, "algorithm"):
        algo = agent_cfg.algorithm
        print(
            "[INFO] Effective PPO overrides: "
            f"lr={getattr(algo, 'learning_rate', 'n/a')}, "
            f"entropy_coef={getattr(algo, 'entropy_coef', 'n/a')}, "
            f"desired_kl={getattr(algo, 'desired_kl', 'n/a')}, "
            f"max_grad_norm={getattr(algo, 'max_grad_norm', 'n/a')}, "
            f"schedule={getattr(algo, 'schedule', 'n/a')}"
        )

    actor = getattr(agent_cfg, "actor", None)
    actor_dist = getattr(actor, "distribution_cfg", None) if actor is not None else None
    print(
        "[INFO] Effective actor distribution: "
        f"class={getattr(actor_dist, 'class_name', 'None')}, "
        f"std_type={getattr(actor_dist, 'std_type', 'None')}, "
        f"init_std={getattr(actor_dist, 'init_std', 'None')}"
    )

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    start_time = time.time()

    if args_cli.distribution_safety_patch:
        _enable_distribution_safety_patch()

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
    elif args_cli.pretrained_checkpoint is not None:
        print(f"[INFO] Loading pretrained weights from: {args_cli.pretrained_checkpoint}")
        runner.load(args_cli.pretrained_checkpoint)
        print("[INFO] Pretrained weights loaded. Training starts from iteration 0.")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=int(agent_cfg.max_iterations), init_at_random_ep_len=True)
    if args_cli.plot_learning_curves:
        _export_learning_curves(
            log_dir=log_dir,
            tags=args_cli.plot_tags,
            smoothing=max(1, args_cli.plot_smoothing),
        )

    elapsed = time.time() - start_time
    print(
        f"[INFO] MTL training completed in {elapsed:.1f}s ({elapsed / 3600:.2f}h), "
        f"iterations={agent_cfg.max_iterations}"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
