#!/bin/bash

# Train task-conditioned multi-task Unitree Go2 policies on ALICE.
#
# Local smoke-test equivalent:
#   C:\Users\pavel\OneDrive\Desktop\IsaacLab\isaaclab.bat -p scripts\mtl_train_conditioned.py --headless --num_envs 64 --seed 0 --max_iterations 5 --run_name smoke_conditioned --plot_learning_curves
#
# ALICE smoke test:
#   NUM_ENVS=64 MAX_ITERATIONS=5 sbatch --array=0-0 scripts/slurm/train_mtl_conditioned_alice.sbatch
#
# Full run:
#   NUM_ENVS=4096 MAX_ITERATIONS=3000 sbatch --array=0-2 scripts/slurm/train_mtl_conditioned_alice.sbatch
#
# Phase run with pretrained checkpoint:
#   NUM_ENVS=4096 MAX_ITERATIONS=400 PHASE_PROFILE=p2_b2safe \
#   SAMPLING_STRATEGY=custom CUSTOM_TERRAIN_PROBS="0.25 0.25 0.25 0.25" \
#   PRETRAINED_CHECKPOINT=/workspace/bsc-thesis-multi-task-RL-env/logs/.../model_2196.pt \
#   RUN_NAME_PREFIX=p2_b2safe_from_balanced sbatch --array=0-0 scripts/slurm/train_mtl_conditioned_alice.sbatch

#SBATCH --job-name=go2_mtl_cond
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
echo "[INFO] Task: MTL-Conditioned-Unitree-Go2-AllTerrains-v0"
echo "[INFO] Seed: ${SEED}"
echo "[INFO] Sampling: uniform/equal"
echo "[INFO] Num envs: ${NUM_ENVS:-4096}"
echo "[INFO] Max iterations: ${MAX_ITERATIONS:-3000}"
echo "[INFO] Phase profile: ${PHASE_PROFILE:-default}"
echo "[INFO] Sampling strategy: ${SAMPLING_STRATEGY:-uniform}"
echo "[INFO] Custom terrain probs: ${CUSTOM_TERRAIN_PROBS:-}"
echo "[INFO] Pretrained checkpoint: ${PRETRAINED_CHECKPOINT:-}"

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--pretrained_checkpoint "$PRETRAINED_CHECKPOINT")
fi
if [[ -n "${CUSTOM_TERRAIN_PROBS:-}" ]]; then
  # shellcheck disable=SC2206
  TERRAIN_PROBS=(${CUSTOM_TERRAIN_PROBS})
  EXTRA_ARGS+=(--custom_terrain_probs "${TERRAIN_PROBS[@]}")
fi
if [[ "${MTL_PROFILE:-garage}" != "default" ]]; then
  EXTRA_ARGS+=(--mtl_profile "${MTL_PROFILE:-garage}")
fi
if [[ "${NO_GARAGE_FORCE_UNIFORM:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--no-garage_force_uniform)
fi
if [[ -n "${LEARNING_RATE:-}" ]]; then
  EXTRA_ARGS+=(--learning_rate "$LEARNING_RATE")
fi
if [[ -n "${ENTROPY_COEF:-}" ]]; then
  EXTRA_ARGS+=(--entropy_coef "$ENTROPY_COEF")
fi
if [[ -n "${SCHEDULE:-fixed}" ]]; then
  EXTRA_ARGS+=(--schedule "${SCHEDULE:-fixed}")
fi

printf -v EXTRA_ARGS_QUOTED " %q" "${EXTRA_ARGS[@]}"

apptainer exec --nv --writable-tmpfs \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  --bind "$ISAACSIM_CACHE_ROOT/kit/cache:/isaac-sim/kit/cache" \
  --bind "$ISAACSIM_CACHE_ROOT/kit/data:/isaac-sim/kit/data" \
  --bind "$ISAACSIM_CACHE_ROOT/kit/logs:/isaac-sim/kit/logs" \
  --bind "$PROJECT_ROOT:$CONTAINER_PROJECT_ROOT" \
  "$CONTAINER" \
  bash -lc "cd '$CONTAINER_PROJECT_ROOT' && '$CONTAINER_ISAACLAB_ROOT/isaaclab.sh' -p scripts/mtl_train_conditioned.py \
    --headless \
    --seed '$SEED' \
    --num_envs '${NUM_ENVS:-4096}' \
    --max_iterations '${MAX_ITERATIONS:-3000}' \
    --experiment_name '${EXPERIMENT_NAME:-unitree_go2_mtl_conditioned_postfix}' \
    --sampling_strategy '${SAMPLING_STRATEGY:-uniform}' \
    --phase_profile '${PHASE_PROFILE:-default}' \
    --distribution_safety_patch \
    --plot_learning_curves \
    --plot_smoothing '${PLOT_SMOOTHING:-25}' \
    --run_name '${RUN_NAME_PREFIX:-conditioned}_s${SEED}' \
    $EXTRA_ARGS_QUOTED"
