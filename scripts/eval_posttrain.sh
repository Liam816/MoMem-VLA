#!/usr/bin/env bash
# Usage: bash scripts/eval_posttrain.sh --config ... --experiment E1 --gpu 6 --output ...
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_SETUP="${CONDA_SETUP:-/mnt/cpfs/heping/miniconda3/etc/profile.d/conda.sh}"
EVAL_ENV="${EVAL_ENV:-/mnt/cpfs/heping/SimpleMemVLA-data/eval_env.sh}"
source "$CONDA_SETUP"
conda activate "${SIMPLEMEMVLA_ENV:-simplememvla-robomme}"
source "$EVAL_ENV"
unset __EGL_VENDOR_LIBRARY_DIRS
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export TMPDIR="${TMPDIR:-/dev/shm/simplememvla_posttrain_eval}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR"
cd "$REPO_ROOT"
exec python -m robomme_sim.evaluation.runner "$@"
