"""Create a thesis-ready contact sheet of benchmark terrain previews.

The script records short clips from the lightweight Play environments, extracts
one clean frame per task, labels each tile, writes a single PNG figure, and can
also save clean per-task screenshots without labels.

Default layout shows the distinct terrain families:
    A1 flat | B1 rough
    B2 stairs | C2 gap

Example:
    isaaclab.bat -p scripts/preview_benchmark_terrains.py --device cuda:0

With a trained policy checkpoint:
    isaaclab.bat -p scripts/preview_benchmark_terrains.py \
      --checkpoint logs/rsl_rl/<experiment>/<run>/model_1500.pt --device cuda:0

If recordings already exist and only the PNG should be rebuilt:
    python scripts/preview_benchmark_terrains.py --reuse_recordings
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import subprocess
import time
from dataclasses import dataclass

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class PreviewTask:
    key: str
    label: str
    play_id: str
    eye: tuple[float, float, float]


TASKS = {
    "A1": PreviewTask(
        key="A1",
        label="A1 flat",
        play_id="MTL-Velocity-Flat-Unitree-Go2-A1-Forward-Play-v0",
        eye=(2.4, -4.8, 1.9),
    ),
    "A2": PreviewTask(
        key="A2",
        label="A2 omni-flat",
        play_id="MTL-Velocity-Flat-Unitree-Go2-A2-Omni-Play-v0",
        eye=(2.4, -4.8, 1.9),
    ),
    "B1": PreviewTask(
        key="B1",
        label="B1 rough",
        play_id="MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-Play-v0",
        eye=(2.6, -5.2, 2.0),
    ),
    "B2": PreviewTask(
        key="B2",
        label="B2 stairs",
        play_id="MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-Play-v0",
        eye=(2.8, -5.4, 2.1),
    ),
    "C2": PreviewTask(
        key="C2",
        label="C2 gap",
        play_id="MTL-Custom-Gap-Unitree-Go2-C2-Play-v0",
        eye=(2.8, -5.6, 2.2),
    ),
}

DEFAULT_TERRAIN_ORDER = ("A1", "B1", "B2", "C2")


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


def _latest_video_for_task(video_base_dir: str, play_id: str) -> str:
    video_dir = os.path.join(video_base_dir, play_id)
    files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not files:
        raise FileNotFoundError(f"No mp4 videos found under: {video_dir}")
    return max(files, key=os.path.getmtime)


def _is_probably_blank_frame(frame: np.ndarray) -> bool:
    if frame.size == 0:
        return True
    return float(np.mean(frame)) < 3.0 and float(np.percentile(frame, 99)) < 8.0


def _resize_frame(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame
    y_idx = np.linspace(0, frame.shape[0] - 1, target_h).astype(np.int32)
    x_idx = np.linspace(0, frame.shape[1] - 1, target_w).astype(np.int32)
    return frame[y_idx][:, x_idx]


def _record_task_clip(
    project_root: str,
    isaaclab_bat: str,
    task: PreviewTask,
    checkpoint: str | None,
    video_length: int,
    num_envs: int,
    device: str | None,
    recordings_base_dir: str,
) -> str:
    if checkpoint is not None and not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    vis_script = os.path.join(project_root, "scripts", "visualize.py")
    start_ts = time.time()
    before_latest = None
    before_mtime = -1.0
    try:
        before_latest = _latest_video_for_task(recordings_base_dir, task.play_id)
        before_mtime = os.path.getmtime(before_latest)
    except FileNotFoundError:
        pass

    cmd = [
        isaaclab_bat,
        "-p",
        vis_script,
        "--task",
        task.play_id,
        "--headless",
        "--enable_cameras",
        "--video",
        "--video_length",
        str(video_length),
        "--deterministic_eval",
        "--num_envs",
        str(num_envs),
        "--follow",
        "--eye",
        str(task.eye[0]),
        str(task.eye[1]),
        str(task.eye[2]),
        "--video_folder",
        os.path.join(recordings_base_dir, task.play_id),
    ]
    if checkpoint is not None:
        cmd.extend(["--checkpoint", checkpoint])
    if device:
        cmd.extend(["--device", device])

    print(f"[INFO] Recording {task.label}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        raise RuntimeError(f"Recording failed for {task.key} with exit code {result.returncode}.")

    latest = _latest_video_for_task(recordings_base_dir, task.play_id)
    latest_mtime = os.path.getmtime(latest)
    if before_latest is None:
        fresh = latest_mtime >= (start_ts - 2.0)
    else:
        fresh = (latest != before_latest) or (latest_mtime > before_mtime + 1e-6)
    if not fresh:
        raise RuntimeError(f"{task.key}: no fresh video artifact was written.")

    return latest


def _extract_frame(path: str, frame_index: int) -> np.ndarray:
    reader = iio.get_reader(path)
    selected = None
    try:
        for idx, frame in enumerate(reader):
            if idx < frame_index:
                continue
            if _is_probably_blank_frame(frame):
                continue
            selected = frame
            break
    finally:
        reader.close()

    if selected is None:
        raise RuntimeError(f"No usable non-blank frame found in: {path}")
    return selected


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", max(18, image.width // 32))
    except OSError:
        font = ImageFont.load_default()

    padding = max(8, image.width // 80)
    bbox = draw.textbbox((0, 0), label, font=font)
    label_w = bbox[2] - bbox[0]
    label_h = bbox[3] - bbox[1]
    box = (
        padding,
        padding,
        padding * 3 + label_w,
        padding * 3 + label_h,
    )
    draw.rectangle(box, fill=(0, 0, 0, 150))
    draw.text((padding * 2, padding * 2), label, fill=(255, 255, 255, 255), font=font)

    return np.asarray(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))


def _compose_contact_sheet(
    output_path: str,
    tasks: list[PreviewTask],
    clips: list[str],
    frame_index: int,
    columns: int,
    tile_width: int,
    tile_height: int,
    individual_dir: str | None,
) -> None:
    frames = []
    for task, clip in zip(tasks, clips, strict=True):
        frame = _extract_frame(clip, frame_index)
        frame = _resize_frame(frame, target_h=tile_height, target_w=tile_width)
        if individual_dir is not None:
            os.makedirs(individual_dir, exist_ok=True)
            individual_path = os.path.join(individual_dir, f"{task.key}_{task.label.replace(' ', '_')}.png")
            iio.imwrite(individual_path, frame)
            print(f"[INFO] Saved screenshot: {individual_path}")
        frames.append(_label_frame(frame, task.label))

    rows = math.ceil(len(frames) / columns)
    pad = np.full((tile_height, tile_width, 3), 245, dtype=np.uint8)
    while len(frames) < rows * columns:
        frames.append(pad.copy())

    row_images = []
    for row in range(rows):
        row_frames = frames[row * columns : (row + 1) * columns]
        row_images.append(np.concatenate(row_frames, axis=1))

    sheet = np.concatenate(row_images, axis=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    iio.imwrite(output_path, sheet)
    print(f"[DONE] Terrain preview saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a benchmark terrain preview PNG.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TERRAIN_ORDER),
        choices=list(TASKS.keys()),
        help="Task keys to include. Default: distinct terrains A1 B1 B2 C2.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional policy checkpoint.")
    parser.add_argument("--video_length", type=int, default=90, help="Frames recorded per task.")
    parser.add_argument("--frame_index", type=int, default=45, help="Frame extracted from each recording.")
    parser.add_argument("--num_envs", type=int, default=1, help="Env count per recording run.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda:0.")
    parser.add_argument("--columns", type=int, default=2, help="Number of columns in the contact sheet.")
    parser.add_argument("--tile_width", type=int, default=640, help="Output width of each tile.")
    parser.add_argument("--tile_height", type=int, default=360, help="Output height of each tile.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Default: videos/benchmark_terrain_preview.png",
    )
    parser.add_argument(
        "--recordings_root",
        type=str,
        default="videos/benchmark_terrain_preview_recordings",
        help="Directory where temporary per-task recordings are stored.",
    )
    parser.add_argument(
        "--individual_dir",
        type=str,
        default=None,
        help=(
            "Directory for unlabeled per-task screenshots. "
            "Default: <output_stem>_individuals next to the contact sheet."
        ),
    )
    parser.add_argument(
        "--no_individuals",
        action="store_true",
        default=False,
        help="Only write the contact sheet; do not save unlabeled per-task screenshots.",
    )
    parser.add_argument(
        "--reuse_recordings",
        action="store_true",
        default=False,
        help="Skip recording and use latest existing clips under recordings_root.",
    )
    args = parser.parse_args()

    project_root = _project_root()
    isaaclab_bat = _resolve_isaaclab_bat(project_root)

    recordings_base_dir = args.recordings_root
    if not os.path.isabs(recordings_base_dir):
        recordings_base_dir = os.path.join(project_root, recordings_base_dir)
    os.makedirs(recordings_base_dir, exist_ok=True)

    output_path = args.output or os.path.join(project_root, "videos", "benchmark_terrain_preview.png")
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)

    individual_dir = None
    if not args.no_individuals:
        if args.individual_dir is None:
            stem = os.path.splitext(os.path.basename(output_path))[0]
            individual_dir = os.path.join(os.path.dirname(output_path), f"{stem}_individuals")
        else:
            individual_dir = args.individual_dir
            if not os.path.isabs(individual_dir):
                individual_dir = os.path.join(project_root, individual_dir)

    ordered_tasks = [TASKS[key] for key in args.tasks]
    clips = []
    for task in ordered_tasks:
        if args.reuse_recordings:
            clip = _latest_video_for_task(recordings_base_dir, task.play_id)
            print(f"[INFO] Reusing {task.label} clip: {clip}")
        else:
            clip = _record_task_clip(
                project_root=project_root,
                isaaclab_bat=isaaclab_bat,
                task=task,
                checkpoint=args.checkpoint,
                video_length=args.video_length,
                num_envs=args.num_envs,
                device=args.device,
                recordings_base_dir=recordings_base_dir,
            )
        clips.append(clip)

    _compose_contact_sheet(
        output_path=output_path,
        tasks=ordered_tasks,
        clips=clips,
        frame_index=args.frame_index,
        columns=max(1, args.columns),
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        individual_dir=individual_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
