"""visualize a benchmark env – random actions or a trained checkpoint"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize benchmark environments.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Gym ID of the task to visualize.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained model .pt file. If omitted, uses random actions.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video clip.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_folder",
    type=str,
    default=None,
    help="Optional video output directory override. Default: <project>/videos/<task>.",
)
parser.add_argument("--follow", action="store_true", default=False, help="Camera tracks the robot (good for terrain envs).")
parser.add_argument("--eye", type=float, nargs=3, default=None, help="Camera eye position (x y z), relative when --follow.")
parser.add_argument("--lookat", type=float, nargs=3, default=None, help="Camera lookat target (x y z).")
parser.add_argument(
    "--deterministic_eval",
    action="store_true",
    default=False,
    help="Disable observation corruption and random push/disturbance events for cleaner checkpoint playback.",
)
parser.add_argument(
    "--conditioned_eval_task",
    type=str,
    default=None,
    choices=("A1_forward", "A2_omni", "B1_rough", "B2_stairs", "C2_gap"),
    help="Inject the 4-way MTL task-id observation for conditioned checkpoints.",
)
parser.add_argument(
    "--conditioned_task_index",
    type=int,
    default=None,
    choices=(0, 1, 2, 3),
    help="Inject a raw 4-way MTL task-id index: flat=0, rough=1, stairs=2, gap=3.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# rewrite sys.argv so Hydra only sees its own args
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from importlib import metadata

import envs

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    try:
        from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
    except ImportError:
        def handle_deprecated_rsl_rl_cfg(agent_cfg, _installed_version):
            return agent_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
try:
    from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
except ImportError:
    RslRlMLPModelCfg = None


CONDITIONED_TASK_INDEX = {
    "A1_forward": 0,
    "A2_omni": 0,
    "B1_rough": 1,
    "B2_stairs": 2,
    "C2_gap": 3,
}


def _constant_conditioned_task_id(env, task_index: int) -> torch.Tensor:
    task_id = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
    task_id[:, task_index] = 1.0
    return task_id


def _maybe_add_conditioned_task_id(env_cfg: ManagerBasedRLEnvCfg) -> None:
    task_index = args_cli.conditioned_task_index
    if args_cli.conditioned_eval_task is not None:
        task_index = CONDITIONED_TASK_INDEX[args_cli.conditioned_eval_task]
    if task_index is None:
        return

    env_cfg.observations.policy.task_id = ObsTerm(
        func=_constant_conditioned_task_id,
        params={"task_index": task_index},
    )
    print(f"[INFO] Injected conditioned MTL task_id index={task_index}.")


def _has_checkpoint_key(checkpoint_path: str, state_name: str, key_substr: str) -> bool:
    loaded = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state = loaded.get(state_name, {})
    return any(key_substr in key for key in state)


def _align_agent_cfg_to_checkpoint(agent_cfg: RslRlBaseRunnerCfg, checkpoint_path: str) -> None:
    """Make the playback runner schema match the saved checkpoint before loading weights."""
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
            "[INFO] Playback cfg aligned to checkpoint observation normalization: "
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
        print("[INFO] Playback cfg aligned to checkpoint actor distribution std_type=log.")


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
                # Newer RSL-RL ActorCritic checkpoints store the policy std at
                # top-level; older split actor/critic models store it in the
                # actor distribution module.
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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    _maybe_add_conditioned_task_id(env_cfg)

    if args_cli.deterministic_eval:
        env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg, "events"):
            if hasattr(env_cfg.events, "base_external_force_torque"):
                env_cfg.events.base_external_force_torque = None
            if hasattr(env_cfg.events, "push_robot"):
                env_cfg.events.push_robot = None
        print("[INFO] Deterministic eval enabled (corruption/disturbance disabled).")
    elif args_cli.checkpoint and not args_cli.task.endswith("-Play-v0"):
        print(
            "[WARN] You are visualizing a non-Play task; training-time corruption/randomization may be active. "
            "Use --deterministic_eval or a *-Play-v0 task for cleaner playback."
        )

    # camera override: follow the robot for terrain envs
    if args_cli.follow:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = tuple(args_cli.eye) if args_cli.eye else (3.0, 3.0, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    else:
        if args_cli.eye:
            env_cfg.viewer.eye = tuple(args_cli.eye)
        if args_cli.lookat:
            env_cfg.viewer.lookat = tuple(args_cli.lookat)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording BEFORE rsl-rl wrapper so steps are captured
    if args_cli.video:
        if args_cli.video_folder:
            video_dir = os.path.abspath(args_cli.video_folder)
        else:
            video_dir = os.path.join(os.path.dirname(__file__), "..", "videos", args_cli.task)
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # load trained policy if checkpoint provided
    policy = None
    if args_cli.checkpoint:
        wrapped = RslRlVecEnvWrapper(env)
        installed_version = metadata.version("rsl-rl-lib")
        _align_agent_cfg_to_checkpoint(agent_cfg, args_cli.checkpoint)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        _align_agent_cfg_to_checkpoint(agent_cfg, args_cli.checkpoint)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        _load_runner_checkpoint_compat(runner, args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print(f"[INFO] Loaded policy from: {args_cli.checkpoint}")
        obs = wrapped.get_observations()
    else:
        print("[INFO] No checkpoint – using random actions")
        env.reset()

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    def _get_actions():
        nonlocal obs
        if policy:
            return policy(obs)
        else:
            return 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1

    def _step(actions):
        nonlocal obs
        if policy:
            obs, _, _, _ = wrapped.step(actions)
        else:
            env.step(actions)

    if args_cli.video:
        with torch.inference_mode():
            for _ in range(args_cli.video_length):
                actions = _get_actions()
                _step(actions)
        print("[INFO] Video recording complete.")
    else:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = _get_actions()
                _step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
