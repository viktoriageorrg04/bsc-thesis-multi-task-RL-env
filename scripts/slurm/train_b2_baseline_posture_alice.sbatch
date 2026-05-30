#!/bin/bash

# Reproduce the B2 stairs baseline with the posture reward annealing used by the
# original successful B2 run (2026-04-15_14-38-36).
#
# Usage:
#   NUM_ENVS=1024 MAX_ITERATIONS=2200 sbatch --array=0-2 --partition=gpu-mig-40g scripts/slurm/train_b2_baseline_posture_alice.sbatch
#
# Single-seed reproduction of the original seed:
#   NUM_ENVS=1024 MAX_ITERATIONS=2200 SEED=42 sbatch --array=0-0 --partition=gpu-mig-40g scripts/slurm/train_b2_baseline_posture_alice.sbatch

#SBATCH --job-name=go2_b2_posture
#SBATCH --partition=gpu-a100-80g,gpu-mig-40g
#SBATCH --time=1-0:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

module load cuDNN/8.9.7.29-CUDA-12.3.2

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/bsc-thesis-multi-task-RL-env}"
CONTAINER="${CONTAINER:-$HOME/containers/isaac-lab_2.3.2.sif}"
CONTAINER_PROJECT_ROOT="/workspace/bsc-thesis-multi-task-RL-env"
CONTAINER_ISAACLAB_ROOT="/workspace/isaaclab"
ISAACSIM_CACHE_ROOT="${ISAACSIM_CACHE_ROOT:-$HOME/isaacsim_cache}"

TASK_ID="MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0"
SEEDS=(0 1 2)
SEED="${SEED:-${SEEDS[$SLURM_ARRAY_TASK_ID]}}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2200}"

cd "$PROJECT_ROOT"
mkdir -p slurm_logs
mkdir -p "$ISAACSIM_CACHE_ROOT/kit/cache"
mkdir -p "$ISAACSIM_CACHE_ROOT/kit/data"
mkdir -p "$ISAACSIM_CACHE_ROOT/kit/logs"

echo "[INFO] Host: $(hostname)"
echo "[INFO] Task: ${TASK_ID}"
echo "[INFO] Seed: ${SEED}"
echo "[INFO] Num envs: ${NUM_ENVS}"
echo "[INFO] Max iterations: ${MAX_ITERATIONS}"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Container: ${CONTAINER}"

apptainer exec --nv --writable-tmpfs \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  --bind "$ISAACSIM_CACHE_ROOT/kit/cache:/isaac-sim/kit/cache" \
  --bind "$ISAACSIM_CACHE_ROOT/kit/data:/isaac-sim/kit/data" \
  --bind "$ISAACSIM_CACHE_ROOT/kit/logs:/isaac-sim/kit/logs" \
  --bind "$PROJECT_ROOT:$CONTAINER_PROJECT_ROOT" \
  "$CONTAINER" \
  bash -lc "cd '$CONTAINER_PROJECT_ROOT' && '$CONTAINER_ISAACLAB_ROOT/isaaclab.sh' -p scripts/train.py \
    --task '$TASK_ID' \
    --headless \
    --seed '$SEED' \
    --num_envs '$NUM_ENVS' \
    --max_iterations '$MAX_ITERATIONS' \
    --no-stand_reward_anneal \
    --posture_reward_anneal \
    --posture_anneal_end_frac 0.35 \
    --posture_height_weight_early -10.0 \
    --posture_height_weight_late -3.5 \
    --posture_orientation_weight_early -5.0 \
    --posture_orientation_weight_late -2.0 \
    --distribution_safety_patch \
    --plot_learning_curves \
    --plot_smoothing '${PLOT_SMOOTHING:-25}' \
    --experiment_name unitree_go2_b2_posture_apr15_refined_1024_seeds \
    --run_name B2_stairs_posture_apr15_refined_1024_s${SEED}"
