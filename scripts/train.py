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

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

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

    # logging directory
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

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

    # train
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    elapsed = time.time() - start_time
    print(f"[INFO] Training completed in {elapsed:.1f}s ({elapsed / 3600:.2f}h)")

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
