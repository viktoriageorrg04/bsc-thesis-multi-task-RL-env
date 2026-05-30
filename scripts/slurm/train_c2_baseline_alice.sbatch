#!/bin/bash

# Rerun the C2 gap-crossing baseline on ALICE across seeds.
#
# Usage:
#   NUM_ENVS=1024 MAX_ITERATIONS=1500 sbatch scripts/slurm/train_c2_baseline_alice.sbatch
#
# Optional gait initialization:
#   PRETRAINED_CHECKPOINT_PATTERN='/workspace/bsc-thesis-multi-task-RL-env/logs/rsl_rl/unitree_go2_a2_baseline_2048_seeds/*_A2_omni_2048_s{seed}/model_1499.pt' \
#     NUM_ENVS=1024 MAX_ITERATIONS=1500 sbatch --array=0-2 scripts/slurm/train_c2_baseline_alice.sbatch
#
# Smoke test:
#   NUM_ENVS=64 MAX_ITERATIONS=2 sbatch scripts/slurm/train_c2_baseline_alice.sbatch

#SBATCH --job-name=go2_c2_base
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

TASK_ID="MTL-Custom-Gap-Unitree-Go2-C2-v0"
SEEDS=(0 1 2)
SEED="${SEED:-${SEEDS[$SLURM_ARRAY_TASK_ID]}}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-unitree_go2_c2_from_a2_1024_seeds}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-C2_gap_from_a2_1024}"

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--pretrained_checkpoint "$PRETRAINED_CHECKPOINT")
elif [[ -n "${PRETRAINED_CHECKPOINT_PATTERN:-}" ]]; then
  CHECKPOINT_PATTERN="${PRETRAINED_CHECKPOINT_PATTERN//\{seed\}/$SEED}"
  shopt -s nullglob
  MATCHES=($CHECKPOINT_PATTERN)
  shopt -u nullglob
  if (( ${#MATCHES[@]} != 1 )); then
    echo "[ERROR] Expected exactly one checkpoint for pattern: ${CHECKPOINT_PATTERN}" >&2
    printf '[ERROR] Matches: %s\n' "${MATCHES[@]:-<none>}" >&2
    exit 2
  fi
  EXTRA_ARGS+=(--pretrained_checkpoint "${MATCHES[0]}")
fi

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
echo "[INFO] Experiment name: ${EXPERIMENT_NAME}"
echo "[INFO] Run prefix: ${RUN_NAME_PREFIX}"
echo "[INFO] Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Container: ${CONTAINER}"

apptainer exec --nv --writable-tmpfs \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  --env PRETRAINED_ARGS="${EXTRA_ARGS[*]:-}" \
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
    --distribution_safety_patch \
    \$PRETRAINED_ARGS \
    --plot_learning_curves \
    --plot_smoothing '${PLOT_SMOOTHING:-25}' \
    --experiment_name '$EXPERIMENT_NAME' \
    --run_name '${RUN_NAME_PREFIX}_s${SEED}'"
