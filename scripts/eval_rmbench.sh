#!/usr/bin/env bash
# usage: bash scripts/eval_rmbench.sh [config.yaml] [--flag value ...]
#   default config: configs/eval/rmbench.yaml
# Protocol knobs live in the YAML; trailing flags override it, e.g.
#   CHECKPOINT=checkpoints/simplememvla_rmbench bash scripts/eval_rmbench.sh \
#     --tasks observe_and_pickup --seeds_per_task 2 --group_size 2
# Env vars only describe the run: CHECKPOINT, GPUS (one worker per GPU; NUM_GPUS defaults to their
# count), RUN_TAG / RUN_DIR / LOG_FILE, VIDEO_DIR (empty disables videos), PYBIN.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$REPO_ROOT/scripts/legacy_env.sh"
check_legacy_env scripts/eval_rmbench.sh \
  TASK_CONFIG=--task_config \
  INSTRUCTION_TYPE=--instruction_type \
  SEEDS_PER_TASK=--seeds_per_task \
  GROUP_SIZE=--group_size \
  EXECUTE_HORIZON=--execute_horizon \
  CAPTURE_STRIDE=--capture_stride \
  PIPELINED=--pipelined \
  DENSE_SUBSTEPS=--dense_substeps \
  NUM_DENOISING_STEPS=--num_denoising_steps \
  EVAL_TEMPERATURE=--eval_temperature \
  MAX_SUBTASK_TOKENS=--max_subtask_tokens \
  COMPUTE_DTYPE=--compute_dtype \
  ATTN_IMPLEMENTATION=--attn_implementation \
  TASKS=--tasks \
  EXPERT_CHECK="--expert_check / --no_expert_check" \
  VIDEO_MAX_PER_TASK=--video_max_per_task

CONFIG="$REPO_ROOT/configs/eval/rmbench.yaml"
if [[ "${1:-}" == *.yaml || "${1:-}" == *.yml ]]; then CONFIG="$(realpath "$1")"; shift; fi
CHECKPOINT="${CHECKPOINT:-}"
GPUS="${GPUS:-${GPU:-0}}"
IFS=, read -r -a _gpus <<< "$GPUS"
NUM_GPUS="${NUM_GPUS:-${#_gpus[@]}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/logs/rmbench_sim/$RUN_TAG}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/eval.log}"
VIDEO_DIR="${VIDEO_DIR-$RUN_DIR/videos}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUTF8="${PYTHONUTF8:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

unset __EGL_VENDOR_LIBRARY_DIRS
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
PYBIN="${PYBIN:-python}"

mkdir -p "$RUN_DIR"
[ -n "$VIDEO_DIR" ] && mkdir -p "$VIDEO_DIR"
echo "[eval_rmbench_sim] config : $CONFIG"
echo "[eval_rmbench_sim] run dir: $RUN_DIR"
echo "[eval_rmbench_sim] log    : $LOG_FILE"
echo "[eval_rmbench_sim] videos : ${VIDEO_DIR:-<disabled>}"

set +e
"$PYBIN" -m rmbench_sim.eval_success \
  --config "$CONFIG" \
  ${CHECKPOINT:+--pretrained_checkpoint "$CHECKPOINT"} \
  --num_gpus "$NUM_GPUS" \
  --log_file "$LOG_FILE" \
  ${VIDEO_DIR:+--video_dir "$VIDEO_DIR"} \
  "$@" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
echo "[eval_rmbench_sim] done (exit $status). Log: $LOG_FILE  Results: $RUN_DIR/results.json"
exit "$status"
