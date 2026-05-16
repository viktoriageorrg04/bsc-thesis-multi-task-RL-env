#!/bin/bash

# Train unified multi-task Unitree Go2 policies on ALICE with equal terrain sampling.
#
# Smoke test:
#   NUM_ENVS=64 MAX_ITERATIONS=2 sbatch --array=0-0 scripts/slurm/train_mtl_equal_alice.sbatch
#
# Full run:
#   NUM_ENVS=4096 MAX_ITERATIONS=1500 sbatch scripts/slurm/train_mtl_equal_alice.sbatch

#SBATCH --job-name=go2_mtl_eq
#SBATCH --partition=gpu-a100-80g,gpu-mig-40g
#SBATCH --time=1-0:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

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
echo "[INFO] Task: MTL-Unified-Unitree-Go2-AllTerrains-v0"
echo "[INFO] Seed: ${SEED}"
echo "[INFO] Sampling: uniform/equal"
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
  bash -lc "cd '$CONTAINER_PROJECT_ROOT' && '$CONTAINER_ISAACLAB_ROOT/isaaclab.sh' -p scripts/mtl_train.py \
    --task MTL-Unified-Unitree-Go2-AllTerrains-v0 \
    --headless \
    --seed '$SEED' \
    --num_envs '${NUM_ENVS:-4096}' \
    --max_iterations '${MAX_ITERATIONS:-1500}' \
    --sampling_strategy uniform \
    --phase_profile default \
    --distribution_safety_patch \
    --plot_learning_curves \
    --plot_smoothing '${PLOT_SMOOTHING:-25}' \
    --experiment_name unitree_go2_mtl_equal_seeds \
    --run_name 'equal_s${SEED}'"
