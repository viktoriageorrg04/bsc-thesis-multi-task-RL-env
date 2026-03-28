"""visualize a benchmark env – random actions or a trained checkpoint"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize benchmark environments.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Gym ID of the task to visualize.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained model .pt file. If omitted, uses random actions.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video clip.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video (in steps).")
parser.add_argument("--follow", action="store_true", default=False, help="Camera tracks the robot (good for terrain envs).")
parser.add_argument("--eye", type=float, nargs=3, default=None, help="Camera eye position (x y z), relative when --follow.")
parser.add_argument("--lookat", type=float, nargs=3, default=None, help="Camera lookat target (x y z).")
parser.add_argument(
    "--deterministic_eval",
    action="store_true",
    default=False,
    help="Disable observation corruption and random push/disturbance events for cleaner checkpoint playback.",
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
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs

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
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
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
