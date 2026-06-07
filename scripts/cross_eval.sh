#!/usr/bin/env bash
# Cross-evaluate a ckpt on all 5 tasks using separate processes.
# Usage: bash scripts/cross_eval.sh <TRAIN_TASK_GYM_ID> <CHECKPOINT_PATH> [NUM_ENVS] [NUM_EPISODES]
# Optional env vars:
#   TASK_TIMEOUT_SEC  : per-task wall-clock timeout (default: 900, set 0 to disable)
#   MAX_RETRIES       : retries per failed task (default: 1)
#   RETRY_DELAY_SEC   : sleep between retries (default: 5)
#
# Example:
#   bash scripts/cross_eval.sh \
#     MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0 \
#     logs/rsl_rl/unitree_go2_rough/2026-04-09_01-42-49/model_1440.pt \
#     64 256

set -u

TRAIN_TASK="${1:?Usage: cross_eval.sh <TRAIN_TASK> <CHECKPOINT> [NUM_ENVS] [NUM_EPISODES]}"
CHECKPOINT="${2:?Provide checkpoint path}"
NUM_ENVS="${3:-64}"
NUM_EPISODES="${4:-256}"
TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-900}"
MAX_RETRIES="${MAX_RETRIES:-1}"
RETRY_DELAY_SEC="${RETRY_DELAY_SEC:-5}"

# you can override this from the shell
#   ISAACLAB_BAT=/c/Users/<user>/.../IsaacLab/isaaclab.bat bash scripts/cross_eval.sh ...
if [[ -z "${ISAACLAB_BAT:-}" ]]; then
    if [[ -f "/c/Users/pavel/OneDrive/Desktop/IsaacLab/isaaclab.bat" ]]; then
        ISAACLAB_BAT="/c/Users/pavel/OneDrive/Desktop/IsaacLab/isaaclab.bat"
    elif [[ -f "/mnt/c/Users/pavel/OneDrive/Desktop/IsaacLab/isaaclab.bat" ]]; then
        ISAACLAB_BAT="/mnt/c/Users/pavel/OneDrive/Desktop/IsaacLab/isaaclab.bat"
    else
        ISAACLAB_BAT="/c/Users/pavel/OneDrive/Desktop/IsaacLab/isaaclab.bat"
    fi
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

to_windows_path() {
    local path_in="$1"
    if command -v wslpath >/dev/null 2>&1; then
        wslpath -w "${path_in}"
    elif command -v cygpath >/dev/null 2>&1; then
        cygpath -w "${path_in}"
    else
        printf '%s\n' "${path_in}"
    fi
}

from_windows_path() {
    local path_in="$1"
    if command -v wslpath >/dev/null 2>&1; then
        wslpath -u "${path_in}"
    elif command -v cygpath >/dev/null 2>&1; then
        cygpath -u "${path_in}"
    else
        printf '%s\n' "${path_in}"
    fi
}

file_mtime_epoch() {
    local p="$1"
    if [[ ! -f "${p}" ]]; then
        echo 0
        return
    fi
    if stat -c %Y "${p}" >/dev/null 2>&1; then
        stat -c %Y "${p}"
        return
    fi
    if stat -f %m "${p}" >/dev/null 2>&1; then
        stat -f %m "${p}"
        return
    fi
    echo 0
}

EVAL_TASKS=(
    "A1_forward"
    "A2_omni"
    "B1_rough"
    "B2_stairs"
    "C2_gap"
)

case "${TRAIN_TASK}" in
    "MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0") TRAIN_SHORT="A1_forward" ;;
    "MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0") TRAIN_SHORT="A2_omni" ;;
    "MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0") TRAIN_SHORT="B1_rough" ;;
    "MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0") TRAIN_SHORT="B2_stairs" ;;
    "MTL-Custom-Gap-Unitree-Go2-C2-v0") TRAIN_SHORT="C2_gap" ;;
    *Unified*|*AllTerrains*) TRAIN_SHORT="MTL_unified" ;;
    *) TRAIN_SHORT="${TRAIN_TASK//-/_}" ;;
esac
OUT_DIR="results/${TRAIN_SHORT}"

echo "Cross-evaluation: ${TRAIN_TASK}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Envs: ${NUM_ENVS} | Episodes: ${NUM_EPISODES}"
echo "Per-task timeout: ${TASK_TIMEOUT_SEC}s | Retries: ${MAX_RETRIES}"

if [[ ! "${TRAIN_TASK}" =~ -v[0-9]+$ ]]; then
    echo "Error: TRAIN_TASK must be a full Gym ID (e.g. ...-v0), got: ${TRAIN_TASK}" >&2
    exit 1
fi

if [[ ! -f "${ISAACLAB_BAT}" ]]; then
    echo "Error: isaaclab launcher not found: ${ISAACLAB_BAT}" >&2
    echo "Set ISAACLAB_BAT to your isaaclab.bat path (Git Bash style), e.g." >&2
    echo "  ISAACLAB_BAT=/c/Users/<user>/.../IsaacLab/isaaclab.bat" >&2
    exit 1
fi

if ! command -v cmd.exe >/dev/null 2>&1; then
    echo "Error: cmd.exe not found. Cannot execute Windows .bat from this shell." >&2
    exit 1
fi

ISAACLAB_BAT_WIN="$(to_windows_path "${ISAACLAB_BAT}")"
EVAL_SCRIPT_WIN="$(to_windows_path "${SCRIPT_DIR}/evaluate.py")"

if [[ "${CHECKPOINT}" =~ ^[A-Za-z]:[^\\/].* ]]; then
    echo "Error: checkpoint path looks malformed: ${CHECKPOINT}" >&2
    echo "In bash, backslashes are escape chars. Use one of these forms:" >&2
    echo "  1) /c/Users/.../model_1499.pt" >&2
    echo "  2) 'C:\\Users\\...\\model_1499.pt'" >&2
    exit 1
fi

if [[ -f "${CHECKPOINT}" ]]; then
    CHECKPOINT_ABS="$(cd "$(dirname "${CHECKPOINT}")" && pwd)/$(basename "${CHECKPOINT}")"
elif [[ "${CHECKPOINT}" =~ ^[A-Za-z]:\\.*$ ]]; then
    CHECKPOINT_UNIX="$(from_windows_path "${CHECKPOINT}")"
    if [[ -f "${CHECKPOINT_UNIX}" ]]; then
        CHECKPOINT_ABS="$(cd "$(dirname "${CHECKPOINT_UNIX}")" && pwd)/$(basename "${CHECKPOINT_UNIX}")"
    else
        echo "Error: checkpoint not found: ${CHECKPOINT}" >&2
        exit 1
    fi
elif [[ "${CHECKPOINT}" == /* ]] && [[ -f "${CHECKPOINT}" ]]; then
    CHECKPOINT_ABS="$(cd "$(dirname "${CHECKPOINT}")" && pwd)/$(basename "${CHECKPOINT}")"
else
    echo "Error: checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi
CHECKPOINT_WIN="$(to_windows_path "${CHECKPOINT_ABS}")"

FAILED_TASKS=()
LAST_RUN_TIMED_OUT=0

run_eval_task_once() {
    local task="$1"
    local expected_json="${OUT_DIR}/${task}.json"
    local before_mtime
    local after_mtime
    local rc=0
    LAST_RUN_TIMED_OUT=0
    local eval_gym_id=""

    case "${task}" in
        "A1_forward") eval_gym_id="MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0" ;;
        "A2_omni") eval_gym_id="MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0" ;;
        "B1_rough") eval_gym_id="MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0" ;;
        "B2_stairs") eval_gym_id="MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0" ;;
        "C2_gap") eval_gym_id="MTL-Custom-Gap-Unitree-Go2-C2-v0" ;;
        *)
            echo ">>> Unknown eval task short name: ${task}" >&2
            return 2
            ;;
    esac

    before_mtime="$(file_mtime_epoch "${expected_json}")"

    if [[ "${TASK_TIMEOUT_SEC}" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
        timeout --foreground --preserve-status "${TASK_TIMEOUT_SEC}" \
            cmd.exe /c call "${ISAACLAB_BAT_WIN}" \
                -p "${EVAL_SCRIPT_WIN}" \
                --task "${eval_gym_id}" \
                --checkpoint "${CHECKPOINT_WIN}" \
                --report_train_task "${TRAIN_TASK}" \
                --headless \
                --num_envs "${NUM_ENVS}" \
                --num_episodes "${NUM_EPISODES}" \
                --eval_task "${task}"
        rc=$?
        if [[ "${rc}" -eq 124 ]]; then
            LAST_RUN_TIMED_OUT=1
        fi
    else
        cmd.exe /c call "${ISAACLAB_BAT_WIN}" \
            -p "${EVAL_SCRIPT_WIN}" \
            --task "${eval_gym_id}" \
            --checkpoint "${CHECKPOINT_WIN}" \
            --report_train_task "${TRAIN_TASK}" \
            --headless \
            --num_envs "${NUM_ENVS}" \
            --num_episodes "${NUM_EPISODES}" \
            --eval_task "${task}"
        rc=$?
    fi

    after_mtime="$(file_mtime_epoch "${expected_json}")"
    artifact_updated=0
    if [[ -s "${expected_json}" ]] && [[ "${after_mtime}" -gt "${before_mtime}" ]]; then
        artifact_updated=1
    fi

    # some launcher setups return non-zero despite writing the expected artifact for this run
    if [[ "${rc}" -ne 0 ]] && [[ "${artifact_updated}" -eq 1 ]]; then
        echo ">>> ${task}: command returned ${rc}, but found ${expected_json}. Treating as success." >&2
        rc=0
    fi

    # guard against false success (launcher returns zero even when eval failed early)
    if [[ "${rc}" -eq 0 ]] && [[ "${artifact_updated}" -eq 0 ]]; then
        echo ">>> ${task}: no fresh output artifact written (${expected_json}). Marking as failure." >&2
        rc=3
    fi

    return "${rc}"
}

for task in "${EVAL_TASKS[@]}"; do
    echo ""
    echo ">>> Evaluating on: ${task} ..."
    attempt=0
    task_ok=0
    while [[ "${attempt}" -le "${MAX_RETRIES}" ]]; do
        attempt=$((attempt + 1))
        echo ">>> ${task}: attempt ${attempt}/$((MAX_RETRIES + 1))"
        if run_eval_task_once "${task}"; then
            task_ok=1
            break
        fi
        rc=$?
        if [[ "${LAST_RUN_TIMED_OUT}" -eq 1 ]]; then
            echo ">>> ${task}: timed out after ${TASK_TIMEOUT_SEC}s (rc=${rc})." >&2
        else
            echo ">>> ${task}: failed with exit code ${rc}." >&2
        fi
        if [[ "${attempt}" -le "${MAX_RETRIES}" ]]; then
            echo ">>> ${task}: retrying in ${RETRY_DELAY_SEC}s..." >&2
            sleep "${RETRY_DELAY_SEC}"
        fi
    done

    if [[ "${task_ok}" -eq 1 ]]; then
        echo ">>> ${task} complete."
    else
        FAILED_TASKS+=("${task}")
        echo ">>> ${task} failed after $((MAX_RETRIES + 1)) attempts. Continuing to next task." >&2
    fi
done

echo ""
if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo "Cross-evaluation finished with task launch errors:"
    printf ' - %s\n' "${FAILED_TASKS[@]}"
    echo "Check per-task JSON files under results/ for successfully completed tasks."
    exit 1
else
    echo "All evaluations complete!"
    echo "Results in: results/"
    echo "Row summary: ${OUT_DIR}/summary.json"
    echo "Export matrix (+heatmap): python scripts/export_eval_matrix.py --results_root results --metric success_rate"
fi
