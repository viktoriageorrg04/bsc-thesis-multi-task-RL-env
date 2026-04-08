"""Cross-evaluate a trained checkpoint on all 5 single-task envs.

  # eval multi-task policy on all tasks
  isaaclab -p scripts/evaluate.py \
    --task MTL-Unified-Unitree-Go2-AllTerrains-v0 \
    --checkpoint logs/rsl_rl/unitree_go2_rough/<run>/model_1500.pt \
    --headless --num_envs 64

  # eval single-task policy (e.g. A1) on all tasks
  isaaclab -p scripts/evaluate.py \
    --task MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0 \
    --checkpoint logs/rsl_rl/unitree_go2_rough/<run>/model_1500.pt \
    --headless --num_envs 64

Output: results/<experiment_name>/<eval_task>.json per eval task
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cross-evaluate a trained policy on all benchmark tasks.")
parser.add_argument("--task", type=str, required=True, help="Gym ID the policy was TRAINED on.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=64, help="Envs per eval task.")
parser.add_argument("--num_episodes", type=int, default=256, help="Min episodes to collect per task.")
parser.add_argument("--output_dir", type=str, default="results", help="Root dir for JSON outputs.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import importlib.metadata as metadata

import isaaclab_tasks
import envs.registry
from isaaclab_tasks.utils.hydra import hydra_task_config

from envs.success import (
    EpisodeStatsTracker,
    TASK_SUCCESS_CONFIGS,
    compute_tracking_errors,
    compute_step_success_from_errors,
)

# the 5 single-task eval envs (Play variants use fewer envs + no curriculum)
EVAL_TASK_IDS = (
    "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0",
    "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0",
    "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0",
    "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0",
    "MTL-Custom-Gap-Unitree-Go2-C2-v0",
)

# family id lookup
TASK_FAMILY = {
    "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0": 0,
    "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0": 0,
    "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0": 1,
    "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0": 1,
    "MTL-Custom-Gap-Unitree-Go2-C2-v0": 2,
}

# short names for output files
TASK_SHORT = {
    "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0": "A1_forward",
    "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0": "A2_omni",
    "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0": "B1_rough",
    "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0": "B2_stairs",
    "MTL-Custom-Gap-Unitree-Go2-C2-v0": "C2_gap",
}


def _load_policy(checkpoint_path: str, env, agent_cfg):
    """load trained policy from ckpt"""
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    return policy


def _evaluate_on_task(policy, eval_task_id: str, num_envs: int, num_episodes: int, device: str):
    """run policy on a single eval task, collect episode stats"""
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(eval_task_id, device=device, num_envs=num_envs, use_fabric=True)
    env = gym.make(eval_task_id, cfg=env_cfg)

    success_cfg = TASK_SUCCESS_CONFIGS.get(eval_task_id)
    tracker = EpisodeStatsTracker(
        num_envs=num_envs,
        step_dt=env.unwrapped.step_dt,
        cfg=success_cfg,
        device=device,
    )

    # wrap for rsl-rl obs format
    wrapped_env = RslRlVecEnvWrapper(env)
    obs = wrapped_env.get_observations()

    all_episodes = []
    step_count = 0

    with torch.inference_mode():
        while len(all_episodes) < num_episodes:
            actions = policy(obs)
            obs, _, dones, _ = wrapped_env.step(actions)

            # compute success metrics on the unwrapped env
            raw_env = env.unwrapped
            lin_err, ang_err = compute_tracking_errors(raw_env)
            terminated = raw_env.termination_manager.terminated
            truncated = raw_env.termination_manager.time_outs

            step_success = compute_step_success_from_errors(
                lin_err, ang_err, terminated, success_cfg
            )

            rows = tracker.update(
                step_success=step_success,
                lin_err_xy=lin_err,
                ang_err_z=ang_err,
                terminated=terminated,
                truncated=truncated,
                task_id=EVAL_TASK_IDS.index(eval_task_id),
                family_id=TASK_FAMILY[eval_task_id],
            )
            all_episodes.extend(rows)
            step_count += 1

            # (don't run forever...)
            if step_count > 50_000:
                break

    env.close()
    return all_episodes


def _summarize(episodes: list[dict]) -> dict:
    """aggregate episode-level stats"""
    n = len(episodes)
    if n == 0:
        return {"num_episodes": 0, "success_rate": 0.0}

    successes = sum(1 for e in episodes if e["episode_success"])
    return {
        "num_episodes": n,
        "success_rate": successes / n,
        "mean_success_step_ratio": sum(e["success_step_ratio"] for e in episodes) / n,
        "mean_lin_vel_error": sum(e["mean_lin_vel_error_xy"] for e in episodes) / n,
        "mean_ang_vel_error": sum(e["mean_ang_vel_error_z"] for e in episodes) / n,
        "mean_alive_time_s": sum(e["alive_time_s"] for e in episodes) / n,
        "failure_rate": sum(1 for e in episodes if e["termination_reason"] == "failure") / n,
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """cross-eval trained policy on all 5 benchmark tasks"""

    # create a dummy env from the training task to load the policy
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device or "cuda:0"

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)

    policy = _load_policy(args_cli.checkpoint, wrapped, agent_cfg)
    env.close()

    train_short = TASK_SHORT.get(args_cli.task, args_cli.task.replace("-", "_"))
    if "Unified" in args_cli.task or "AllTerrains" in args_cli.task:
        train_short = "MTL_unified"
    out_dir = os.path.join(args_cli.output_dir, train_short)
    os.makedirs(out_dir, exist_ok=True)

    # cross-eval on all 5 tasks
    summary_all = {}
    for eval_task in EVAL_TASK_IDS:
        short = TASK_SHORT[eval_task]
        print(f"\n{'='*60}")
        print(f"[EVAL] Policy: {train_short} | Task: {short}")
        print(f"{'='*60}")

        episodes = _evaluate_on_task(
            policy, eval_task, args_cli.num_envs, args_cli.num_episodes,
            device=args_cli.device or "cuda:0",
        )
        summary = _summarize(episodes)
        summary["train_task"] = args_cli.task
        summary["eval_task"] = eval_task
        summary_all[short] = summary

        # write per-task JSON
        out_path = os.path.join(out_dir, f"{short}.json")
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "episodes": episodes}, f, indent=2)
        print(f"  success_rate: {summary['success_rate']:.2%}")
        print(f"  mean_alive_time: {summary['mean_alive_time_s']:.1f}s")
        print(f"  saved: {out_path}")

    # write combined summary
    combined_path = os.path.join(out_dir, "summary.json")
    with open(combined_path, "w") as f:
        json.dump(summary_all, f, indent=2)
    print(f"\n[DONE] Combined summary: {combined_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
