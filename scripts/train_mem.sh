#!/usr/bin/env bash
# Single-frame + history vision-token cache experiments (frozen VLM, DiT from scratch).
# The recipe lives in configs/train/memcache/<EXP>.yaml:
#   E1: single frame, no memory          (lower bound)
#   E2: + memory cache, adaLN fusion     (main)
#   E3: + memory cache, prefix tokens    (fusion control)
#   (frozen original Qwen3.5-4B VLM; E*_ckptvlm: frozen released simplememvla_robomme VLM)
#   E*_warmdit: superseded runs that also warm-started the DiT from the checkpoint
# usage: EXP=E2 bash scripts/train_mem.sh [--flag value ...]
# DRY_RUN=1 prints the fully expanded command (with the exported env) and exits.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/legacy_env.sh"
check_legacy_env scripts/train_mem.sh "${TRAIN_LEGACY_ENV[@]}"
cd "$REPO_ROOT"

EXP="${EXP:-E2}"
CONFIG="configs/train/memcache/${EXP}.yaml"
if [[ ! -f "$CONFIG" ]]; then
  echo "unknown EXP: $EXP (no $CONFIG)" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-6011}"
export WANDB_PROJECT="${WANDB_PROJECT:-simplememvla-memcache}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# DataLoader workers hand tensors back over a Unix socket created in TMPDIR;
# CPFS (where eval_env.sh points TMPDIR) rejects sockets with Errno 95.
export TMPDIR="${TRAIN_TMPDIR:-/dev/shm/simplememvla_tmp}"
mkdir -p "$TMPDIR"

exec bash scripts/train.sh "$CONFIG" "$@"
