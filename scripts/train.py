"""Single-task and multi-task training driver.

  # single-task baseline (RQ1 baseline, one per task)
  isaaclab -p <repo>/scripts/train.py --task MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0

  # multi-task (RQ1 treatment — one policy, all terrains)
  isaaclab -p <repo>/scripts/train.py --task MTL-Unified-Unitree-Go2-AllTerrains-v0

  # override num_envs for lighter GPU
  isaaclab -p <repo>/scripts/train.py --task ... --num_envs 2048

  # resume from ckpt
  isaaclab -p <repo>/scripts/train.py --task ... --resume --load_run <run_dir>

Logs are written to: logs/rsl_rl/<experiment_name>/<timestamp>/
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL (single-task or multi-task).")

# env
parser.add_argument("--task", type=str, required=True, help="Gym ID of the task to train on.")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of parallel envs.")
parser.add_argument("--seed", type=int, default=None, help="Random seed (-1 for random).")

# training
parser.add_argument("--max_iterations", type=int, default=None, help="Override max PPO iterations.")
parser.add_argument(
    "--early_stop_patience",
    type=int,
    default=0,
    help="Number of metric checks without improvement before stopping (0 disables early stopping).",
)
parser.add_argument(
    "--early_stop_metric",
    type=str,
    default="Train/mean_reward",
    help="TensorBoard scalar tag to monitor for early stopping.",
)
parser.add_argument(
    "--early_stop_mode",
    type=str,
    default="max",
    choices={"max", "min"},
    help="Whether larger or smaller metric values are better.",
)
parser.add_argument(
    "--early_stop_min_delta",
    type=float,
    default=0.0,
    help="Minimum absolute metric improvement to reset patience.",
)
parser.add_argument(
    "--early_stop_check_interval",
    type=int,
    default=50,
    help="Number of PPO iterations between early-stop checks.",
)
parser.add_argument(
    "--early_stop_warmup",
    type=int,
    default=200,
    help="Minimum PPO iterations before early-stop checks are applied.",
)
parser.add_argument(
    "--plot_learning_curves",
    action="store_true",
    default=False,
    help="Export learning-curve plots and CSV files from TensorBoard scalars after training.",
)
parser.add_argument(
    "--plot_smoothing",
    type=int,
    default=25,
    help="Moving-average window used for plotted curves (>=1).",
)
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

# video
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (steps).")

# rsl-rl specifics
parser.add_argument("--experiment_name", type=str, default=None, help="Override experiment folder name.")
parser.add_argument("--run_name", type=str, default=None, help="Run name suffix for the log directory.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume from a checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
parser.add_argument(
    "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger backend."
)
parser.add_argument("--log_project_name", type=str, default=None, help="Project name for wandb/neptune.")

# AppLauncher (headless, device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import logging
import os
import random
import time
from datetime import datetime
from glob import glob

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tensorboard.backend.event_processing import event_accumulator

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import importlib.metadata as metadata

import isaaclab_tasks
import envs.registry
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _update_agent_cfg(agent_cfg: RslRlBaseRunnerCfg, args) -> RslRlBaseRunnerCfg:
    """Apply CLI overrides to the RSL-RL runner config."""
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


def _latest_events_file(log_dir: str) -> str | None:
    """Return newest TensorBoard events file in log_dir."""
    candidates = glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _read_scalar_series(log_dir: str, tag: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load scalar series for one TensorBoard tag."""
    event_file = _latest_events_file(log_dir)
    if event_file is None:
        return None

    ea = event_accumulator.EventAccumulator(event_file)
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


def _metric_improved(candidate: float, best: float | None, mode: str, min_delta: float) -> bool:
    """Check if candidate metric is better than best by at least min_delta."""
    if best is None:
        return True
    if mode == "max":
        return candidate > best + min_delta
    return candidate < best - min_delta


def _train_with_early_stopping(runner: OnPolicyRunner, max_iterations: int, log_dir: str) -> int:
    """Train in chunks and stop early when monitored metric plateaus."""
    enabled = args_cli.early_stop_patience > 0
    if not enabled:
        runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
        return max_iterations

    check_interval = max(1, args_cli.early_stop_check_interval)
    warmup_iters = max(0, args_cli.early_stop_warmup)
    patience = max(1, args_cli.early_stop_patience)
    min_delta = max(0.0, args_cli.early_stop_min_delta)
    metric_tag = args_cli.early_stop_metric
    mode = args_cli.early_stop_mode

    best_metric = None
    checks_without_improvement = 0
    trained_iterations = 0

    print(
        f"[INFO] Early stopping enabled: metric='{metric_tag}', mode='{mode}', "
        f"patience={patience}, min_delta={min_delta}, check_interval={check_interval}, warmup={warmup_iters}"
    )

    while trained_iterations < max_iterations:
        chunk = min(check_interval, max_iterations - trained_iterations)
        runner.learn(
            num_learning_iterations=chunk,
            init_at_random_ep_len=(trained_iterations == 0),
        )
        trained_iterations += chunk

        if trained_iterations < warmup_iters:
            continue

        series = _read_scalar_series(log_dir, metric_tag)
        if series is None:
            print(
                f"[WARN] Early-stop metric '{metric_tag}' not found yet in TensorBoard logs. "
                "Skipping this check."
            )
            continue

        step, value = series[0][-1], float(series[1][-1])
        improved = _metric_improved(value, best_metric, mode, min_delta)
        if improved:
            best_metric = value
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1

        print(
            f"[EARLY-STOP] iter={trained_iterations}, tb_step={step}, metric={value:.6f}, "
            f"best={best_metric:.6f}, bad_checks={checks_without_improvement}/{patience}"
        )

        if checks_without_improvement >= patience:
            print(f"[INFO] Early stopping triggered at iteration {trained_iterations}.")
            break

    return trained_iterations


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average with edge-preserving behavior for short arrays."""
    window = max(1, int(window))
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _safe_tag_name(tag: str) -> str:
    """Filesystem-safe filename component from TensorBoard tag."""
    return tag.replace("/", "_").replace(" ", "_")


def _export_learning_curves(log_dir: str, tags: list[str], smoothing: int) -> None:
    """Export curve CSV files and a combined PNG plot for requested tags."""
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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL PPO."""
    # CLI overrides
    agent_cfg = _update_agent_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    installed_rsl_rl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # logging directory (always inside this project repo, independent of launch cwd)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_root_path = os.path.join(project_root, "logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging root directory: {log_root_path}")
    print(f"[INFO] Saving this run to: {log_dir}")

    env_cfg.log_dir = log_dir

    # create env
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # resume ckpt path
    resume_path = None
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    start_time = time.time()

    # wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create PPO runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # save configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # train (optionally with early stopping)
    trained_iterations = _train_with_early_stopping(
        runner=runner,
        max_iterations=int(agent_cfg.max_iterations),
        log_dir=log_dir,
    )

    if args_cli.plot_learning_curves:
        _export_learning_curves(
            log_dir=log_dir,
            tags=args_cli.plot_tags,
            smoothing=max(1, args_cli.plot_smoothing),
        )

    elapsed = time.time() - start_time
    print(
        f"[INFO] Training completed in {elapsed:.1f}s ({elapsed / 3600:.2f}h), "
        f"trained_iterations={trained_iterations}/{agent_cfg.max_iterations}"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# # single-task baselines
# isaaclab -p scripts/train.py --task MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0 --headless
# isaaclab -p scripts/train.py --task MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0 --headless
# isaaclab -p scripts/train.py --task MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0 --headless
# isaaclab -p scripts/train.py --task MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0 --headless
# isaaclab -p scripts/train.py --task MTL-Custom-SteppingStones-Unitree-Go2-C1-v0 --headless
# isaaclab -p scripts/train.py --task MTL-Custom-Gap-Unitree-Go2-C2-v0 --headless

# # multi-task
# isaaclab -p scripts/train.py --task MTL-Unified-Unitree-Go2-AllTerrains-v0 --headless

# --num_envs 2048  # override parallel envs (default 4096)
# --max_iterations 1500  # override training length
# --seed 42  # reproducibility
# --experiment_name foo  # custom log folder name


# RQ1: "Does multi-task RL produce a more capable policy than single-task?"

# train 6 single-task policies + 1 multi-task policy
# eval each on all 6 tasks (42 data points)
# compare: MTL row vs. diagonal of single-task matrix


# RQ2: "How does performance transfer across task fams?"

# the 6×6 cross-evaluation matrix shows this
# single-task A1 evaluated on B2 (stairs); does flat training transfer?
# MTL evaluated on all 6; does joint training improve off-diagonal scores?
# group by fam (A, B, C); within-family vs. cross-family transfer
