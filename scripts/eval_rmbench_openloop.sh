#!/usr/bin/env bash
# usage: bash scripts/eval_rmbench_openloop.sh [config.yaml] [--flag value ...]
#   default config: configs/eval/openloop/rmbench.yaml; CHECKPOINT=<ckpt> overrides its checkpoint.
# Open-loop (dataset replay) action L1 + sub-task accuracy; trailing flags override the YAML.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="$REPO_ROOT/configs/eval/openloop/rmbench.yaml"
if [[ "${1:-}" == *.yaml || "${1:-}" == *.yml ]]; then CONFIG="$(realpath "$1")"; shift; fi
CHECKPOINT="${CHECKPOINT:-}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYBIN="${PYBIN:-python}"

"$PYBIN" -m rmbench_sim.openloop_eval \
  --config "$CONFIG" \
  ${CHECKPOINT:+--pretrained_checkpoint "$CHECKPOINT"} \
  "$@"
