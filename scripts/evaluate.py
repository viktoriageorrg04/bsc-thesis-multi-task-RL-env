"""Cross-evaluate a trained checkpoint on all 5 single-task envs.

  # eval single-task policy (e.g. A1) on all tasks
  isaaclab -p scripts/evaluate.py \
    --task MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0 \
    --checkpoint logs/rsl_rl/unitree_go2_rough/<run>/model_1500.pt \
    --headless --num_envs 64 --num_episodes 256

  # eval on a single task only (no cross-eval)
  isaaclab -p scripts/evaluate.py \
    --task MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0 \
    --checkpoint logs/rsl_rl/unitree_go2_rough/<run>/model_1500.pt \
    --headless --num_envs 64 --eval_task A1_forward

Output:
  - results/<train_short>/<eval_task>.json per eval task
  - results/<train_short>/summary.json (row-level matrix for this trained policy)
"""

import argparse
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Cross-evaluate a trained policy on all benchmark tasks.")
parser.add_argument("--task", type=str, required=True, help="Gym ID the policy was TRAINED on.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=64, help="Envs per eval task.")
parser.add_argument("--num_episodes", type=int, default=256, help="Min episodes to collect per task.")
parser.add_argument("--output_dir", type=str, default="results", help="Root dir for JSON outputs.")
parser.add_argument(
    "--report_train_task",
    type=str,
    default=None,
    help="Optional: original task the policy was trained on (for reporting/output grouping).",
)
parser.add_argument("--eval_task", type=str, default=None,
                    help="If set, evaluate on this single task only (short name e.g. A1_forward). "
                         "Otherwise cross-eval on all 5.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os
import time

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

# the 5 single-task eval envs
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

# reverse: short name → gym ID
SHORT_TO_GYM = {v: k for k, v in TASK_SHORT.items()}


def _run_eval_on_env(policy, env, eval_task_id, num_episodes, device):
    """Run evaluation loop on an already-created env. Returns list of episode dicts."""
    success_cfg = TASK_SUCCESS_CONFIGS.get(eval_task_id)
    tracker = EpisodeStatsTracker(
        num_envs=env.unwrapped.num_envs,
        step_dt=env.unwrapped.step_dt,
        cfg=success_cfg,
        device=device,
    )

    wrapped_env = RslRlVecEnvWrapper(env)
    obs = wrapped_env.get_observations()

    all_episodes = []
    step_count = 0
    t_eval = time.time()

    with torch.inference_mode():
        while len(all_episodes) < num_episodes:
            actions = policy(obs)
            obs, _, dones, _ = wrapped_env.step(actions)

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

            if step_count % 100 == 0:
                elapsed = time.time() - t_eval
                print(f"  Step {step_count} | Episodes: {len(all_episodes)}/{num_episodes} | {elapsed:.0f}s")

            if step_count > 50_000:
                print(f"  [WARN] Hit step limit (50k). Collected {len(all_episodes)} episodes.")
                break

    elapsed = time.time() - t_eval
    print(f"  Done: {len(all_episodes)} episodes in {elapsed:.1f}s ({step_count} steps)")
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
    """Evaluate trained policy. Uses a single env (no close+reopen) to avoid Isaac Sim hangs."""

    device = args_cli.device or "cuda:0"
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    # --- create the one and only env, load policy, and eval on training task ---
    print(f"[INFO] Creating environment for: {args_cli.task}")
    t0 = time.time()
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    print(f"[INFO] Env created in {time.time() - t0:.1f}s")

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    train_obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    print(f"[INFO] Policy obs dim: {train_obs_dim}")
    print(f"[INFO] Checkpoint: {args_cli.checkpoint}")

    reported_train_task = args_cli.report_train_task or args_cli.task
    train_short = TASK_SHORT.get(reported_train_task, reported_train_task.replace("-", "_"))
    if "Unified" in reported_train_task or "AllTerrains" in reported_train_task:
        train_short = "MTL_unified"
    out_dir = os.path.join(args_cli.output_dir, train_short)
    os.makedirs(out_dir, exist_ok=True)

    # determine which tasks to evaluate on
    if args_cli.eval_task:
        eval_gym_id = SHORT_TO_GYM.get(args_cli.eval_task)
        if eval_gym_id is None:
            print(f"[ERROR] Unknown eval_task '{args_cli.eval_task}'. Valid: {list(SHORT_TO_GYM.keys())}")
            env.close()
            return
        eval_tasks = [eval_gym_id]
    else:
        eval_tasks = list(EVAL_TASK_IDS)

    summary_all = {}
    t_total = time.time()

    for task_idx, eval_task in enumerate(eval_tasks, 1):
        short = TASK_SHORT[eval_task]
        print(f"[EVAL {task_idx}/{len(eval_tasks)}] Policy: {train_short} | Task: {short}")

        if eval_task == args_cli.task:
            # reuse the already-loaded env (no close+reopen needed)
            print(f"  Reusing loaded environment (same as training task)")
            episodes = _run_eval_on_env(policy, env, eval_task, args_cli.num_episodes, device)
        else:
            # for cross-eval on a different task, we need a new env
            # NOTE: Isaac Sim may hang here if the first env was closed.
            # Workaround: run with --eval_task for each task separately.
            print(f"  [WARN] Cross-task eval requires new env. If it hangs, run separately:")
            print(f"         --task {eval_task} --eval_task {short}")
            try:
                from isaaclab_tasks.utils import parse_env_cfg
                eval_cfg = parse_env_cfg(eval_task, device=device, num_envs=args_cli.num_envs, use_fabric=True)
                eval_env = gym.make(eval_task, cfg=eval_cfg)
                eval_obs_dim = eval_env.unwrapped.observation_manager.group_obs_dim["policy"][0]
                print(f"  Eval obs dim: {eval_obs_dim} (policy expects: {train_obs_dim})")
                if eval_obs_dim != train_obs_dim:
                    print(f"  [SKIP] obs dim mismatch!")
                    eval_env.close()
                    summary_all[short] = {
                        "num_episodes": 0, "success_rate": None,
                        "skipped": True, "reason": "obs_dim_mismatch",
                    }
                    continue
                episodes = _run_eval_on_env(policy, eval_env, eval_task, args_cli.num_episodes, device)
                eval_env.close()
            except Exception as e:
                print(f"  [ERROR] Failed to create eval env: {e}")
                summary_all[short] = {
                    "num_episodes": 0, "success_rate": None,
                    "skipped": True, "reason": str(e),
                }
                continue

        summary = _summarize(episodes)
        summary["train_task"] = reported_train_task
        summary["eval_task"] = eval_task
        summary_all[short] = summary

        out_path = os.path.join(out_dir, f"{short}.json")
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "episodes": episodes}, f, indent=2)
        print(f"  success_rate: {summary['success_rate']:.2%}")
        print(f"  mean_alive_time: {summary['mean_alive_time_s']:.1f}s")
        print(f"  failure_rate: {summary['failure_rate']:.2%}")
        print(f"  saved: {out_path}")

    # write combined summary.
    # When running single-task evals repeatedly (e.g. via cross_eval.sh with --eval_task),
    # keep previously-written task summaries and only update the current task key.
    combined_path = os.path.join(out_dir, "summary.json")
    summary_to_write = summary_all
    if args_cli.eval_task and os.path.exists(combined_path):
        try:
            with open(combined_path, "r") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                existing.update(summary_all)
                summary_to_write = existing
        except Exception:
            # Fall back to writing only current summary if existing file is unreadable.
            summary_to_write = summary_all

    with open(combined_path, "w") as f:
        json.dump(summary_to_write, f, indent=2)
    print(f"\n[DONE] Total eval time: {time.time() - t_total:.0f}s")
    print(f"[DONE] Combined summary: {combined_path}")


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        # Isaac Sim shutdown can occasionally hang in headless batch runs on Windows.
        # In this mode we prefer deterministic process termination after outputs are written.
        if not args_cli.headless:
            try:
                simulation_app.close()
            except Exception:
                pass

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
