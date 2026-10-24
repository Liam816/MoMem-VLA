#!/usr/bin/env bash
# usage: bash scripts/eval_mikasa.sh [config.yaml] [--flag value ...]
#   default config: configs/eval/mikasa.yaml
# Protocol knobs live in the YAML; trailing flags override it, e.g.
#   CHECKPOINT=checkpoints/simplememvla_mikasa GPUS=0,1,2,3 bash scripts/eval_mikasa.sh --save_videos
# Env vars only describe the run: CHECKPOINT, GPUS (one worker per GPU; NUM_GPUS defaults to their
# count), RUN_TAG / RUN_DIR / LOG_FILE, JOINT_DELTA_LIMIT (-> MIKASA_JOINT_DELTA_LIMIT), PYBIN.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$REPO_ROOT/scripts/legacy_env.sh"
check_legacy_env scripts/eval_mikasa.sh \
  N_EPISODES=--n_episodes \
  START_SEED=--start_seed \
  EXECUTE_HORIZON=--execute_horizon \
  DECISION_PATH=--decision_path \
  NUM_DENOISING_STEPS=--num_denoising_steps \
  EVAL_TEMPERATURE=--eval_temperature \
  MAX_SUBTASK_TOKENS=--max_subtask_tokens \
  COMPUTE_DTYPE=--compute_dtype \
  ATTN_IMPLEMENTATION=--attn_implementation \
  CONTROL_MODE=--control_mode \
  TASKS=--tasks \
  SAVE_VIDEOS=--save_videos

CONFIG="$REPO_ROOT/configs/eval/mikasa.yaml"
if [[ "${1:-}" == *.yaml || "${1:-}" == *.yml ]]; then CONFIG="$(realpath "$1")"; shift; fi
CHECKPOINT="${CHECKPOINT:-}"
JOINT_DELTA_LIMIT="${JOINT_DELTA_LIMIT:-}"
if [ -n "$JOINT_DELTA_LIMIT" ]; then
  export MIKASA_JOINT_DELTA_LIMIT="$JOINT_DELTA_LIMIT"
fi
GPUS="${GPUS:-${GPU:-0}}"
IFS=, read -r -a _gpus <<< "$GPUS"
NUM_GPUS="${NUM_GPUS:-${#_gpus[@]}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/logs/mikasa_sim/$RUN_TAG}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/eval.log}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -z "${TRITON_CACHE_DIR:-}" ]; then
  TRITON_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/simplememvla/triton"
fi
if [ -z "${TORCHINDUCTOR_CACHE_DIR:-}" ]; then
  TORCHINDUCTOR_CACHE_DIR="$(dirname "$TRITON_CACHE_DIR")/torchinductor"
fi
export TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

unset __EGL_VENDOR_LIBRARY_DIRS
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
PYBIN="${PYBIN:-python}"

mkdir -p "$RUN_DIR"
echo "[eval_mikasa_sim] config : $CONFIG"
echo "[eval_mikasa_sim] run dir: $RUN_DIR"
echo "[eval_mikasa_sim] log    : $LOG_FILE"

set +e
"$PYBIN" -m mikasa_sim.eval_success \
  --config "$CONFIG" \
  ${CHECKPOINT:+--pretrained_checkpoint "$CHECKPOINT"} \
  --num_gpus "$NUM_GPUS" \
  --output_dir "$RUN_DIR" \
  --log_file "$LOG_FILE" \
  "$@" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
echo "[eval_mikasa_sim] done (exit $status). Log: $LOG_FILE  Results: $RUN_DIR/summary.json"
exit "$status"
