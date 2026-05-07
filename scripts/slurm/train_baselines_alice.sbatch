#!/bin/bash

# Train single-task Unitree Go2 baselines on ALICE as a task x seed array.
#
# Usage:
#   sbatch scripts/slurm/train_baselines_alice.sbatch
#
# Quick smoke test:
#   NUM_ENVS=64 MAX_ITERATIONS=2 sbatch --array=0-0 scripts/slurm/train_baselines_alice.sbatch
#
# Full run:
#   NUM_ENVS=4096 MAX_ITERATIONS=1500 sbatch scripts/slurm/train_baselines_alice.sbatch

#SBATCH --job-name=go2_baselines
#SBATCH --partition=gpu-a100-80g,gpu-mig-40g
#SBATCH --time=1-0:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --array=0-14
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

# ----- Experiment grid: 5 tasks x 3 seeds = 15 array jobs -----
TASK_IDS=(
  "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0"
  "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0"
  "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0"
  "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0"
  "MTL-Custom-Gap-Unitree-Go2-C2-v0"
)

TASK_SHORT=(
  "A1_forward"
  "A2_omni"
  "B1_rough"
  "B2_stairs"
  "C2_gap"
)

SEEDS=(0 1 2)
NUM_SEEDS=${#SEEDS[@]}

TASK_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

TASK_ID=${TASK_IDS[$TASK_INDEX]}
SHORT=${TASK_SHORT[$TASK_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

# ----- ALICE / container setup -----
module load cuDNN/8.9.7.29-CUDA-12.3.2

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/bsc-thesis-multi-task-RL-env}"
CONTAINER="${CONTAINER:-$HOME/containers/isaac-lab_2.3.2.sif}"
CONTAINER_PROJECT_ROOT="/workspace/bsc-thesis-multi-task-RL-env"
CONTAINER_ISAACLAB_ROOT="/workspace/isaaclab"
ISAACSIM_CACHE_ROOT="${ISAACSIM_CACHE_ROOT:-$HOME/isaacsim_cache}"

cd "$PROJECT_ROOT"
mkdir -p slurm_logs
mkdir -p "$ISAACSIM_CACHE_ROOT/kit/cache"
mkdir -p "$ISAACSIM_CACHE_ROOT/kit/data"
mkdir -p "$ISAACSIM_CACHE_ROOT/kit/logs"

echo "[INFO] Host: $(hostname)"
echo "[INFO] Task: ${TASK_ID}"
echo "[INFO] Seed: ${SEED}"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Container: ${CONTAINER}"
echo "[INFO] Num envs: ${NUM_ENVS:-4096}"
echo "[INFO] Max iterations: ${MAX_ITERATIONS:-1500}"

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
    --num_envs '${NUM_ENVS:-4096}' \
    --max_iterations '${MAX_ITERATIONS:-1500}' \
    --experiment_name unitree_go2_baselines_seeds \
    --run_name '${SHORT}_s${SEED}'"
