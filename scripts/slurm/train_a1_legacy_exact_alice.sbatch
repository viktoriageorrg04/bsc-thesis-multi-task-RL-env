#!/bin/bash

# Rerun the April A1 single-task baseline config on ALICE across seeds.
#
# This uses the dedicated Legacy A1 Gym ID, which restores the old always-on
# base_height reward term:
#   base_height weight = -5.0, target_height = 0.38
#
# Usage:
#   NUM_ENVS=2048 MAX_ITERATIONS=3000 sbatch scripts/slurm/train_a1_legacy_exact_alice.sbatch
#
# Smoke test:
#   NUM_ENVS=64 MAX_ITERATIONS=2 sbatch scripts/slurm/train_a1_legacy_exact_alice.sbatch

#SBATCH --job-name=go2_a1_legacy
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

TASK_ID="MTL-Velocity-Flat-Unitree-Go2-A1-Forward-Legacy-v0"
SEEDS=(0 1 2)
SEED="${SEED:-${SEEDS[$SLURM_ARRAY_TASK_ID]}}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"

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
    --no-posture_reward_anneal \
    --plot_learning_curves \
    --plot_smoothing '${PLOT_SMOOTHING:-25}' \
    --experiment_name unitree_go2_a1_legacy_2048_seeds \
    --run_name A1_forward_legacy_2048_s${SEED}"
