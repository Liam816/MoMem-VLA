#!/usr/bin/env bash
# bash scripts/eval_robomme_valtest.sh --config configs/eval/posttrain_valtest_e1.yaml --gpu 6
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CONDA_SETUP:-/mnt/cpfs/heping/miniconda3/etc/profile.d/conda.sh}"
conda activate "${SIMPLEMEMVLA_ENV:-simplememvla-robomme}"
source "${EVAL_ENV:-/mnt/cpfs/heping/SimpleMemVLA-data/eval_env.sh}"
unset __EGL_VENDOR_LIBRARY_DIRS
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
export TMPDIR="${TMPDIR:-/dev/shm/simplememvla_valtest_eval}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR"
cd "$REPO_ROOT"
exec python -m robomme_sim.evaluation.valtest "$@"
