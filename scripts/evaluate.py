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
import os
import sys
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
try:
    from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
except ImportError:
    RslRlMLPModelCfg = None
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    try:
        from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
    except ImportError:
        def handle_deprecated_rsl_rl_cfg(agent_cfg, _installed_version):
            return agent_cfg

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


def _has_checkpoint_key(checkpoint_path: str, state_name: str, key_substr: str) -> bool:
    loaded = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state = loaded.get(state_name, {})
    return any(key_substr in key for key in state)


def _align_agent_cfg_to_checkpoint(agent_cfg: RslRlBaseRunnerCfg, checkpoint_path: str) -> None:
    """Make the eval runner schema match the saved checkpoint before loading weights."""
    loaded = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    actor_state = loaded.get("actor_state_dict", {})
    critic_state = loaded.get("critic_state_dict", {})
    model_state = loaded.get("model_state_dict", {})

    actor_has_obs_norm = any("obs_normalizer." in key for key in actor_state) or any(
        key.startswith("actor_obs_normalizer.") for key in model_state
    )
    critic_has_obs_norm = any("obs_normalizer." in key for key in critic_state) or any(
        key.startswith("critic_obs_normalizer.") for key in model_state
    )
    actor_uses_log_std = any("distribution.log_std_param" in key for key in actor_state) or "log_std" in model_state

    if actor_has_obs_norm or critic_has_obs_norm:
        if hasattr(agent_cfg, "policy") and agent_cfg.policy is not None:
            if hasattr(agent_cfg.policy, "actor_obs_normalization"):
                agent_cfg.policy.actor_obs_normalization = actor_has_obs_norm
            if hasattr(agent_cfg.policy, "critic_obs_normalization"):
                agent_cfg.policy.critic_obs_normalization = critic_has_obs_norm
        if hasattr(agent_cfg, "actor") and agent_cfg.actor is not None and hasattr(agent_cfg.actor, "obs_normalization"):
            agent_cfg.actor.obs_normalization = actor_has_obs_norm
        if hasattr(agent_cfg, "critic") and agent_cfg.critic is not None and hasattr(agent_cfg.critic, "obs_normalization"):
            agent_cfg.critic.obs_normalization = critic_has_obs_norm
        print(
            "[INFO] Eval cfg aligned to checkpoint observation normalization: "
            f"actor={actor_has_obs_norm}, critic={critic_has_obs_norm}"
        )

    if actor_uses_log_std:
        if hasattr(agent_cfg, "policy") and agent_cfg.policy is not None and hasattr(agent_cfg.policy, "noise_std_type"):
            agent_cfg.policy.noise_std_type = "log"
        if (
            hasattr(agent_cfg, "actor")
            and agent_cfg.actor is not None
            and RslRlMLPModelCfg is not None
            and hasattr(agent_cfg.actor, "distribution_cfg")
        ):
            dist_cfg = getattr(agent_cfg.actor, "distribution_cfg", None)
            init_std = getattr(dist_cfg, "init_std", 1.0)
            agent_cfg.actor.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
                init_std=float(init_std),
                std_type="log",
            )
        print("[INFO] Eval cfg aligned to checkpoint actor distribution std_type=log.")


def _load_runner_checkpoint_compat(runner: OnPolicyRunner, checkpoint_path: str) -> None:
    """Load checkpoint with compatibility fallback for std/log-std key rename."""
    try:
        runner.load(checkpoint_path)
        return
    except KeyError as exc:
        if str(exc).strip("'\"") != "actor_state_dict":
            raise

        loaded = torch.load(checkpoint_path, weights_only=False, map_location=runner.device)
        model_state = loaded.get("model_state_dict")
        if model_state is None:
            raise

        actor_state = {}
        critic_state = {}
        for key, value in model_state.items():
            if key == "std":
                actor_state["distribution.std_param"] = value
            elif key == "log_std":
                actor_state["distribution.log_std_param"] = value
            elif key.startswith("actor."):
                actor_state[f"mlp.{key.removeprefix('actor.')}"] = value
            elif key.startswith("actor_obs_normalizer."):
                actor_state[f"obs_normalizer.{key.removeprefix('actor_obs_normalizer.')}"] = value
            elif key.startswith("critic."):
                critic_state[f"mlp.{key.removeprefix('critic.')}"] = value
            elif key.startswith("critic_obs_normalizer."):
                critic_state[f"obs_normalizer.{key.removeprefix('critic_obs_normalizer.')}"] = value

        missing_actor, unexpected_actor = runner.alg.actor.load_state_dict(actor_state, strict=False)
        missing_critic, unexpected_critic = runner.alg.critic.load_state_dict(critic_state, strict=False)
        if loaded.get("iter") is not None:
            runner.current_learning_iteration = loaded["iter"]

        print(
            "[WARN] Loaded newer monolithic ActorCritic checkpoint into local "
            "split actor/critic RSL-RL model for inference."
        )
        if missing_actor or unexpected_actor or missing_critic or unexpected_critic:
            print(f"[WARN] actor missing={missing_actor}, unexpected={unexpected_actor}")
            print(f"[WARN] critic missing={missing_critic}, unexpected={unexpected_critic}")
        return
    except RuntimeError as exc:
        message = str(exc)
        std_schema_mismatch = (
            "distribution.std_param" in message and "distribution.log_std_param" in message
        )
        if not std_schema_mismatch:
            raise

        print(
            "[WARN] Checkpoint uses different distribution std schema "
            "(std_param vs log_std_param). Retrying non-strict load for inference."
        )
        runner.load(checkpoint_path, strict=False)


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

CONDITIONED_TASK_INDEX = {
    "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0": 0,
    "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0": 0,
    "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0": 1,
    "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0": 2,
    "MTL-Custom-Gap-Unitree-Go2-C2-v0": 3,
}


def _constant_conditioned_task_id(env, task_index: int) -> torch.Tensor:
    task_id = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
    task_id[:, task_index] = 1.0
    return task_id


def _maybe_add_conditioned_task_id(env_cfg: ManagerBasedRLEnvCfg, eval_task_id: str, reported_train_task: str) -> None:
    if "MTL-Conditioned-Unitree-Go2-AllTerrains" not in reported_train_task:
        return
    if eval_task_id not in CONDITIONED_TASK_INDEX:
        return
    task_index = CONDITIONED_TASK_INDEX[eval_task_id]
    env_cfg.observations.policy.task_id = ObsTerm(
        func=_constant_conditioned_task_id,
        params={"task_index": task_index},
    )


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
    reported_train_task = args_cli.report_train_task or args_cli.task
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device
    _maybe_add_conditioned_task_id(env_cfg, args_cli.task, reported_train_task)

    # create the one and only env, load policy, and eval on training task
    print(f"[INFO] Creating environment for: {args_cli.task}")
    t0 = time.time()
    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    print(f"[INFO] Env created in {time.time() - t0:.1f}s")

    installed_version = metadata.version("rsl-rl-lib")
    _align_agent_cfg_to_checkpoint(agent_cfg, args_cli.checkpoint)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    _align_agent_cfg_to_checkpoint(agent_cfg, args_cli.checkpoint)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    _load_runner_checkpoint_compat(runner, args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    train_obs_dim = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    print(f"[INFO] Policy obs dim: {train_obs_dim}")
    print(f"[INFO] Checkpoint: {args_cli.checkpoint}")

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
            # reuse the already-loaded env
            print(f"  Reusing loaded environment (same as training task)")
            episodes = _run_eval_on_env(policy, env, eval_task, args_cli.num_episodes, device)
        else:
            # for cross-eval on a different task, we need a new env
            # NOTE: Isaac Sim may hang here if the first env was closed
            # workaround: run with --eval_task for each task separately
            print(f"  [WARN] Cross-task eval requires new env. If it hangs, run separately:")
            print(f"         --task {eval_task} --eval_task {short}")
            try:
                from isaaclab_tasks.utils import parse_env_cfg
                eval_cfg = parse_env_cfg(eval_task, device=device, num_envs=args_cli.num_envs, use_fabric=True)
                _maybe_add_conditioned_task_id(eval_cfg, eval_task, reported_train_task)
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

    # write combined summary
    # when running single-task evals repeatedly (e.g. via cross_eval.sh with --eval_task),
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
            # fall back to writing only current summary if existing file is unreadable
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
        if not args_cli.headless:
            try:
                simulation_app.close()
            except Exception:
                pass

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
