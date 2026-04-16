"""Record and compose a 2x2 terrain comparison video for one ckpt.

This helper records four Play tasks (one per terrain family) with the same policy
checkpoint, trims warmup frames to remove startup jitter, and exports a split-screen
MP4 for qualitative comparison.

Default 4-way layout:
    top-left:  A1 flat
    top-right: B1 rough
    bottom-left: B2 stairs
    bottom-right: C2 gap

Usage:
    isaaclab.bat -p scripts/visualize_split4.py \
      --checkpoint logs/rsl_rl/unitree_go2_mtl_unified/<run>/model_1500.pt \
      --video_length 500 --warmup_steps 120

If you already recorded clips and only want to re-compose:
    isaaclab.bat -p scripts/visualize_split4.py --checkpoint <ckpt> --reuse_recordings
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import time
from dataclasses import dataclass

import imageio.v2 as iio
import numpy as np


@dataclass(frozen=True)
class TileTask:
    key: str
    label: str
    play_id: str
    follow: bool = True
    eye: tuple[float, float, float] | None = None
    lookat: tuple[float, float, float] | None = None


TASKS = {
    "A1": TileTask(
        key="A1",
        label="A1 flat",
        play_id="MTL-Velocity-Flat-Unitree-Go2-A1-Forward-Play-v0",
        eye=(1.0, -5.0, 3.0),
        lookat=(1.0, 0.0, 0.0),
    ),
    "B1": TileTask(
        key="B1",
        label="B1 rough",
        play_id="MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-Play-v0",
    ),
    "B2": TileTask(
        key="B2",
        label="B2 stairs",
        play_id="MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-Play-v0",
    ),
    "C2": TileTask(
        key="C2",
        label="C2 gap",
        play_id="MTL-Custom-Gap-Unitree-Go2-C2-Play-v0",
    ),
}

DEFAULT_ORDER = ("A1", "B1", "B2", "C2")


def _project_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_isaaclab_bat(project_root: str) -> str:
    candidate = os.path.normpath(
        os.path.join(project_root, "..", "OneDrive", "Desktop", "IsaacLab", "isaaclab.bat")
    )
    if os.path.isfile(candidate):
        return candidate

    env_path = os.environ.get("ISAACLAB_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    return "isaaclab.bat"


def _latest_video_for_task(project_root: str, play_id: str) -> str:
    video_dir = os.path.join(project_root, "videos", play_id)
    files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not files:
        raise FileNotFoundError(f"No mp4 videos found under: {video_dir}")
    return max(files, key=os.path.getmtime)


def _record_task_clip(
    project_root: str,
    isaaclab_bat: str,
    task: TileTask,
    checkpoint: str,
    video_length: int,
    warmup_steps: int,
    num_envs: int,
    device: str | None,
) -> str:
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    vis_script = os.path.join(project_root, "scripts", "visualize.py")
    total_steps = int(video_length) + int(warmup_steps)
    start_ts = time.time()

    before_latest = None
    before_mtime = -1.0
    try:
        before_latest = _latest_video_for_task(project_root, task.play_id)
        before_mtime = os.path.getmtime(before_latest)
    except FileNotFoundError:
        pass

    cmd = [
        isaaclab_bat,
        "-p",
        vis_script,
        "--task",
        task.play_id,
        "--checkpoint",
        checkpoint,
        "--headless",
        "--enable_cameras",
        "--video",
        "--video_length",
        str(total_steps),
        "--deterministic_eval",
        "--num_envs",
        str(num_envs),
    ]

    if task.follow:
        cmd.append("--follow")
    if task.eye is not None:
        cmd.extend(["--eye", str(task.eye[0]), str(task.eye[1]), str(task.eye[2])])
    if task.lookat is not None:
        cmd.extend(["--lookat", str(task.lookat[0]), str(task.lookat[1]), str(task.lookat[2])])
    if device:
        cmd.extend(["--device", device])

    print(f"[INFO] Recording {task.label}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        raise RuntimeError(f"Recording failed for {task.key} with exit code {result.returncode}.")

    latest = _latest_video_for_task(project_root, task.play_id)
    latest_mtime = os.path.getmtime(latest)

    produced_fresh_artifact = False
    if before_latest is None:
        produced_fresh_artifact = latest_mtime >= (start_ts - 2.0)
    else:
        produced_fresh_artifact = (latest != before_latest) or (latest_mtime > before_mtime + 1e-6)

    if not produced_fresh_artifact:
        raise RuntimeError(
            f"{task.key}: no fresh video artifact was written. "
            "IsaacLab can return exit code 0 even when visualize failed; check console logs above."
        )

    print(f"[INFO] {task.label} clip: {latest}")
    return latest


def _resize_frame(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame

    # Lightweight nearest-neighbor resize without extra dependencies.
    y_idx = np.linspace(0, frame.shape[0] - 1, target_h).astype(np.int32)
    x_idx = np.linspace(0, frame.shape[1] - 1, target_w).astype(np.int32)
    return frame[y_idx][:, x_idx]


def _iter_trimmed_frames(path: str, start_frame: int):
    reader = iio.get_reader(path)
    index = 0
    try:
        for frame in reader:
            if index >= start_frame:
                yield frame
            index += 1
    finally:
        reader.close()


def _tile_four(frames: list[np.ndarray]) -> np.ndarray:
    top = np.concatenate([frames[0], frames[1]], axis=1)
    bottom = np.concatenate([frames[2], frames[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _compose_split_video(
    output_path: str,
    clips: list[str],
    warmup_steps: int,
    video_length: int,
    fps: int,
) -> None:
    iters = [iter(_iter_trimmed_frames(path, warmup_steps)) for path in clips]

    first_frames = []
    for it in iters:
        try:
            first_frames.append(next(it))
        except StopIteration as exc:
            raise RuntimeError(
                "One clip has no frames after warmup. Reduce --warmup_steps or re-record."
            ) from exc

    target_h, target_w = first_frames[0].shape[0], first_frames[0].shape[1]
    writer = iio.get_writer(output_path, fps=fps, codec="libx264", quality=8)

    try:
        current = [
            _resize_frame(frame, target_h=target_h, target_w=target_w) for frame in first_frames
        ]
        writer.append_data(_tile_four(current))

        written = 1
        while written < video_length:
            frames = []
            for it in iters:
                try:
                    frames.append(next(it))
                except StopIteration:
                    print(
                        "[WARN] A source clip ended early; stopping at shortest clip length "
                        f"({written} frames written)."
                    )
                    return

            frames = [_resize_frame(f, target_h=target_h, target_w=target_w) for f in frames]
            writer.append_data(_tile_four(frames))
            written += 1
    finally:
        writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a 2x2 split-screen terrain comparison video.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to policy checkpoint (.pt).")
    parser.add_argument(
        "--tasks",
        nargs=4,
        default=list(DEFAULT_ORDER),
        choices=list(TASKS.keys()),
        help="Exactly four short task keys in tile order (top-left, top-right, bottom-left, bottom-right).",
    )
    parser.add_argument("--video_length", type=int, default=500, help="Final output length in frames.")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=120,
        help="Frames to skip from each source clip before composition (reduces startup jitter).",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Env count per recording run.")
    parser.add_argument("--fps", type=int, default=30, help="Output fps.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output mp4 path. Default: videos/split4_<ckpt_name>.mp4",
    )
    parser.add_argument(
        "--reuse_recordings",
        action="store_true",
        default=False,
        help="Skip re-recording and use latest existing mp4 for each task.",
    )
    parser.add_argument("--device", type=str, default=None, help="Optional device override (e.g. cuda:0).")
    args = parser.parse_args()

    project_root = _project_root()
    isaaclab_bat = _resolve_isaaclab_bat(project_root)

    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    if args.output is None:
        out_name = f"split4_{ckpt_name}.mp4"
        output_path = os.path.join(project_root, "videos", out_name)
    else:
        output_path = os.path.normpath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ordered_tasks = [TASKS[k] for k in args.tasks]
    clips: list[str] = []

    for task in ordered_tasks:
        if args.reuse_recordings:
            clip = _latest_video_for_task(project_root, task.play_id)
            print(f"[INFO] Reusing {task.label} clip: {clip}")
        else:
            clip = _record_task_clip(
                project_root=project_root,
                isaaclab_bat=isaaclab_bat,
                task=task,
                checkpoint=args.checkpoint,
                video_length=args.video_length,
                warmup_steps=args.warmup_steps,
                num_envs=args.num_envs,
                device=args.device,
            )
        clips.append(clip)

    print("[INFO] Composing split-screen video...")
    _compose_split_video(
        output_path=output_path,
        clips=clips,
        warmup_steps=args.warmup_steps,
        video_length=args.video_length,
        fps=args.fps,
    )

    print(f"[DONE] Split video saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
