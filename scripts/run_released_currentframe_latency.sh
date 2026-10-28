#!/usr/bin/env bash
# Usage: bash scripts/run_released_currentframe_latency.sh <train|test> <GPU> <root>
set -eo pipefail
LAT_SPLIT=${1:?train or test required}
LAT_GPU=${2:?one GPU required}
LAT_ROOT=${3:?absolute experiment root required}
case "$LAT_SPLIT" in train|test) ;; *) exit 2 ;; esac
[[ "$LAT_ROOT" == /* ]] || exit 2
REPO=/mnt/cpfs/heping/SimpleMemVLA
DATA=/mnt/cpfs/heping/SimpleMemVLA-data
source /mnt/cpfs/heping/miniconda3/etc/profile.d/conda.sh
conda activate simplememvla-robomme
source "$DATA/eval_env.sh"
export TMPDIR=/dev/shm/simplememvla_tmp
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="$LAT_GPU" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
unset __EGL_VENDOR_LIBRARY_DIRS
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
python scripts/benchmark_robomme_singleframe_latency.py \
  --checkpoint "$DATA/checkpoints/simplememvla_robomme" \
  --dataset_split "$LAT_SPLIT" \
  --episode_ids_json "$LAT_ROOT/config/latency_${LAT_SPLIT}_episode_ids.json" \
  --episodes_per_task 2 --max_decisions 12 --warmup 5 \
  --output_dir "$LAT_ROOT/latency/$LAT_SPLIT"
