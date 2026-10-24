#!/usr/bin/env bash
# usage: bash scripts/eval_libero.sh [config.yaml] [--flag value ...]
#   default config: configs/eval/libero.yaml
# Protocol knobs live in the YAML; trailing flags override it, e.g.
#   CHECKPOINT=checkpoints/simplememvla_libero bash scripts/eval_libero.sh --task_suites libero_10
# Env vars only describe the run: CHECKPOINT, GPUS (one worker per GPU; NUM_GPUS defaults to their
# count), RUN_TAG / RUN_DIR / LOG_FILE, VIDEO_DIR (empty disables videos), MUJOCO_GL, PYBIN.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="$REPO_ROOT/configs/eval/libero.yaml"
if [[ "${1:-}" == *.yaml || "${1:-}" == *.yml ]]; then CONFIG="$(realpath "$1")"; shift; fi
CHECKPOINT="${CHECKPOINT:-}"
GPUS="${GPUS:-${GPU:-0}}"
IFS=, read -r -a _gpus <<< "$GPUS"
NUM_GPUS="${NUM_GPUS:-${#_gpus[@]}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/logs/libero_sim/$RUN_TAG}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/eval.log}"
VIDEO_DIR="${VIDEO_DIR-$RUN_DIR/videos}"

PYBIN="${PYBIN:-python}"
if ! "$PYBIN" -c "import torch, transformers, robosuite, libero" >/dev/null 2>&1; then
  echo "PYBIN=$PYBIN cannot import torch/transformers/robosuite/libero." >&2
  echo "Activate the LIBERO eval env first, or pass PYBIN=/path/to/env/bin/python" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset __EGL_VENDOR_LIBRARY_DIRS
if [[ "$NUM_GPUS" == "1" && "$GPUS" != *,* && "$MUJOCO_GL" == "egl" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-$GPUS}"
fi

mkdir -p "$RUN_DIR"
echo "[eval_libero_sim] config : $CONFIG"
echo "[eval_libero_sim] run dir: $RUN_DIR"

cd "$REPO_ROOT"
set +e
"$PYBIN" -m libero_sim.eval_success \
  --config "$CONFIG" \
  ${CHECKPOINT:+--pretrained_checkpoint "$CHECKPOINT"} \
  --num_gpus "$NUM_GPUS" \
  --log_file "$LOG_FILE" \
  ${VIDEO_DIR:+--video_dir "$VIDEO_DIR"} \
  "$@" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
exit "$status"
