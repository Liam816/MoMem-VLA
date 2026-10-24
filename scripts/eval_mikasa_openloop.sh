#!/usr/bin/env bash
# usage: bash scripts/eval_mikasa_openloop.sh [config.yaml] [--flag value ...]
#   default config: configs/eval/openloop/mikasa.yaml; CHECKPOINT=<ckpt> overrides its checkpoint.
# Open-loop (dataset replay) action L1 + sub-task accuracy; trailing flags override the YAML.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/legacy_env.sh"
check_legacy_env scripts/eval_mikasa_openloop.sh \
  ROOT_DIR=--root \
  REPO_ID=--repo_id \
  NUM_EPISODES=--num_episodes \
  NUM_DENOISING_STEPS=--num_denoising_steps \
  TEMPERATURE=--temperature \
  NUM_OPEN_LOOP_STEPS=--num_open_loop_steps \
  MAX_SUBTASK_TOKENS=--max_subtask_tokens \
  ATTN_IMPLEMENTATION=--attn_implementation

CONFIG="$REPO_ROOT/configs/eval/openloop/mikasa.yaml"
if [[ "${1:-}" == *.yaml || "${1:-}" == *.yml ]]; then CONFIG="$(realpath "$1")"; shift; fi
CHECKPOINT="${CHECKPOINT:-}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYBIN="${PYBIN:-python}"

"$PYBIN" -m mikasa_sim.openloop_eval \
  --config "$CONFIG" \
  ${CHECKPOINT:+--pretrained_checkpoint "$CHECKPOINT"} \
  "$@"
