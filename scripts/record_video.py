"""Record demo videos for trained single-task baselines.

Automatically finds the best checkpoint (highest mean reward) for a task
and launches visualize.py with the correct Play variant and camera settings.

Usage (single task):
    isaaclab.bat -p scripts/record_video.py --task B2
    isaaclab.bat -p scripts/record_video.py --task B2 --num_envs 3 --video_length 800

Record all 5 tasks in sequence:
    isaaclab.bat -p scripts/record_video.py --all
"""

import argparse
import glob
import os
import sys

_PROJ = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

_ROUGH_ROOT = os.path.join(_PROJ, "logs", "rsl_rl", "unitree_go2_rough")

# All 5 tasks now share unitree_go2_rough (A1/A2 rebuilt on RoughPPORunnerCfg).
# B1/B2/C2 and A1 use explicit log_dir (runs are complete).
# A2: uses log_root + discriminator once retrained.
#   A1 uniquely logs Episode_Reward/stand_still.
#   A2 has neither stand_still nor undesired_contacts (unlike B1/B2/C2).
TASKS = {
    # Use follow-camera for flat tasks too; static world camera can drift off-subject
    # and produce sky/blue-only clips in some runs.
    "A1": {"play_id": "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-Play-v0", "log_dir":  os.path.join(_ROUGH_ROOT, "2026-04-09_01-42-49"), "follow": True, "eye": [1, -5, 3], "lookat": [1, 0, 0]},
    "A2": {"play_id": "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-Play-v0", "log_dir": os.path.join(_ROUGH_ROOT, "2026-04-09_11-10-20"), "follow": True, "eye": [0, -5, 3], "lookat": [0, 0, 0]},
    "B1": {"play_id": "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-Play-v0", "log_dir":  os.path.join(_ROUGH_ROOT, "2026-03-31_17-54-23"), "follow": True},
    "B2": {"play_id": "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-Play-v0", "log_dir":  os.path.join(_ROUGH_ROOT, "2026-04-01_23-15-56"), "follow": True},
    "C2": {"play_id": "MTL-Custom-Gap-Unitree-Go2-C2-Play-v0", "log_dir":  os.path.join(_ROUGH_ROOT, "2026-04-08_19-51-57"), "follow": True},
}


def _latest_run(
    log_root: str,
    discriminator_tag: str | None,
    forbidden_tags: list[str] | None = None,
) -> str:
    """Return the most recently modified run directory under log_root that:
      - contains at least one model_*.pt checkpoint
      - has (or lacks) the discriminator_tag in its TF scalar events
      - does not contain any of forbidden_tags
    Used for A2 which shares unitree_go2_rough with B1/B2/C2.
      A1: discriminator_tag='Episode_Reward/stand_still'  (must HAVE it)
      A2: discriminator_tag=None, forbidden_tags=['Episode_Reward/undesired_contacts']
          (must NOT have stand_still, must NOT have undesired_contacts)
    """
    from tensorboard.backend.event_processing import event_accumulator as ea_mod

    dirs = [
        d for d in glob.glob(os.path.join(log_root, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "model_*.pt"))
    ]
    if not dirs:
        raise FileNotFoundError(f"No training runs with checkpoints found in {log_root}")
    dirs.sort(key=os.path.getmtime, reverse=True)

    for d in dirs:
        ea = ea_mod.EventAccumulator(d, size_guidance={ea_mod.SCALARS: 1})
        ea.Reload()
        tags = set(ea.Tags().get("scalars", []))
        if forbidden_tags and any(ft in tags for ft in forbidden_tags):
            continue
        if discriminator_tag is not None:
            if discriminator_tag in tags:
                return d
        else:
            if "Episode_Reward/stand_still" not in tags:
                return d

    raise FileNotFoundError(
        f"No matching run found in {log_root} for discriminator_tag={discriminator_tag!r}, "
        f"forbidden_tags={forbidden_tags!r}"
    )


def _find_best_checkpoint(log_dir: str) -> tuple[str, float, int]:
    """Return (checkpoint_path, best_reward, best_iteration) using TF events."""
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(
        log_dir, size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    rewards = ea.Scalars("Train/mean_reward")
    best = max(rewards, key=lambda p: p.value)

    # match to closest saved checkpoint
    pts = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No model_*.pt files found in {log_dir}")
    ckpt_nums = {}
    for p in pts:
        num = int(os.path.basename(p).replace("model_", "").replace(".pt", ""))
        ckpt_nums[num] = p
    closest_iter = min(ckpt_nums.keys(), key=lambda n: abs(n - best.step))
    return ckpt_nums[closest_iter], best.value, closest_iter


def main():
    parser = argparse.ArgumentParser(description="Record demo videos for trained baselines.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=str, choices=list(TASKS.keys()), help="Short task name (A1, A2, B1, B2, C2).")
    group.add_argument("--all", action="store_true", help="Record videos for all 5 tasks.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default: 1).")
    parser.add_argument("--video_length", type=int, default=500, help="Video length in steps (default: 500).")
    args = parser.parse_args()

    tasks_to_record = list(TASKS.keys()) if args.all else [args.task]

    # find isaaclab.bat
    isaaclab_bat = os.path.normpath(
        os.path.join(_PROJ, "..", "OneDrive", "Desktop", "IsaacLab", "isaaclab.bat")
    )
    if not os.path.isfile(isaaclab_bat):
        isaaclab_bat = os.environ.get("ISAACLAB_PATH", "isaaclab.bat")

    vis_script = os.path.join(_PROJ, "scripts", "visualize.py")

    for task_key in tasks_to_record:
        cfg = TASKS[task_key]
        print(f"  Task {task_key}: {cfg['play_id']}")

        # resolve log directory
        if "log_dir" in cfg:
            log_dir = cfg["log_dir"]
        else:
            log_dir = _latest_run(cfg["log_root"], cfg.get("discriminator_tag"), cfg.get("forbidden_tags"))
        print(f"  Run dir: {log_dir}")

        ckpt_path, reward, ckpt_iter = _find_best_checkpoint(log_dir)
        print(f"  Best reward: {reward:.2f} at iter {ckpt_iter}")
        print(f"  Checkpoint:  {ckpt_path}")

        cmd = [
            isaaclab_bat, "-p", vis_script,
            "--task", cfg["play_id"],
            "--checkpoint", ckpt_path,
            "--headless", "--enable_cameras",
            "--video", "--video_length", str(args.video_length),
            "--deterministic_eval",
            "--num_envs", str(args.num_envs),
        ]
        if cfg.get("follow"):
            cmd.append("--follow")
        if "eye" in cfg:
            cmd.extend(["--eye"] + [str(v) for v in cfg["eye"]])
        if "lookat" in cfg:
            cmd.extend(["--lookat"] + [str(v) for v in cfg["lookat"]])

        print(f"  Command: {' '.join(cmd)}\n")

        import subprocess
        result = subprocess.run(cmd, cwd=_PROJ)
        if result.returncode != 0:
            print(f"  [WARN] Task {task_key} exited with code {result.returncode}")


if __name__ == "__main__":
    main()
