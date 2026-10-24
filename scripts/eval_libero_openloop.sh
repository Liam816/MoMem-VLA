#!/usr/bin/env bash
# usage: bash scripts/eval_libero_openloop.sh [config.yaml] [--flag value ...]
#   default config: configs/eval/openloop/libero.yaml; CHECKPOINT=<ckpt> overrides its checkpoint.
# Open-loop (dataset replay) action L1 + sub-task accuracy; trailing flags override the YAML.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="$REPO_ROOT/configs/eval/openloop/libero.yaml"
if [[ "${1:-}" == *.yaml || "${1:-}" == *.yml ]]; then CONFIG="$(realpath "$1")"; shift; fi
CHECKPOINT="${CHECKPOINT:-}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYBIN="${PYBIN:-python}"
if ! "$PYBIN" -c "import torch, transformers" >/dev/null 2>&1; then
  echo "PYBIN=$PYBIN cannot import torch/transformers; activate the eval env or pass PYBIN=..." >&2
  exit 1
fi

"$PYBIN" -m libero_sim.openloop_eval \
  --config "$CONFIG" \
  ${CHECKPOINT:+--pretrained_checkpoint "$CHECKPOINT"} \
  "$@"
