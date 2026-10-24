#!/usr/bin/env bash
# usage: bash scripts/train.sh <benchmark | path/to/config.yaml> [--flag value ...]
#   benchmark = rmbench | robomme | mikasa | robomemarena | libero (-> configs/train/<benchmark>.yaml)
# Hyperparameters live in the YAML; trailing flags override it, e.g.
#   bash scripts/train.sh libero --per_device_train_batch_size 4 --gradient_accumulation_steps 2
# Env vars only describe the launch: CUDA_VISIBLE_DEVICES, GPUS_PER_NODE, NNODES, NODE_RANK,
# MASTER_ADDR, MASTER_PORT, WANDB_*. DRY_RUN=1 prints the command instead of running it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/legacy_env.sh"
check_legacy_env scripts/train.sh "${TRAIN_LEGACY_ENV[@]}"

CONFIG="${1:-}"
if [[ -z "$CONFIG" || "$CONFIG" == --* ]]; then
  echo "usage: bash scripts/train.sh <benchmark|config.yaml> [--flag value ...]" >&2
  exit 1
fi
shift
if [[ "$CONFIG" != *.yaml && "$CONFIG" != *.yml ]]; then
  CONFIG="$REPO_ROOT/configs/train/${CONFIG}.yaml"
elif [[ -f "$CONFIG" ]]; then
  CONFIG="$(realpath "$CONFIG")"
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}"
export WANDB_PROJECT="${WANDB_PROJECT:-simplememvla}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

IFS=, read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
GPUS_PER_NODE="${GPUS_PER_NODE:-${#_gpus[@]}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6001}"

CMD=(torchrun
  --nproc_per_node "$GPUS_PER_NODE"
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  train.py
  --config "$CONFIG"
  "$@"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES WANDB_MODE=$WANDB_MODE WANDB_PROJECT=$WANDB_PROJECT PYTHONPATH=$PYTHONPATH"
  echo "cd $REPO_ROOT"
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

"${CMD[@]}"
