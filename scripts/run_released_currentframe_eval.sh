#!/usr/bin/env bash
# Usage: bash scripts/run_released_currentframe_eval.sh <GPUs> <experiment-root>
set -eo pipefail
CF_GPUS=${1:?GPU list required}
CF_ROOT=${2:?absolute experiment root required}
REPO=/mnt/cpfs/heping/SimpleMemVLA
DATA=/mnt/cpfs/heping/SimpleMemVLA-data
[[ "$CF_ROOT" == /* ]] || { echo "Absolute output root required" >&2; exit 2; }
CF_DIR="$CF_ROOT/accuracy/current_frame"
mkdir -p "$CF_DIR"
[[ ! -e "$CF_DIR/status.txt" ]] || { echo "Run already exists" >&2; exit 2; }
trap 'cf_exit=$?; printf "exit=%s\nended=%s\n" "$cf_exit" "$(date -Is)" >> "$CF_DIR/status.txt"' EXIT
printf "started=%s\ngpus=%s\n" "$(date -Is)" "$CF_GPUS" > "$CF_DIR/status.txt"
source /mnt/cpfs/heping/miniconda3/etc/profile.d/conda.sh
conda activate simplememvla-robomme
source "$DATA/eval_env.sh"
export TMPDIR=/dev/shm/simplememvla_tmp
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export CHECKPOINT="$DATA/checkpoints/simplememvla_robomme" GPUS="$CF_GPUS" PYTHONUNBUFFERED=1
cd "$REPO"
for CF_PHASE in train40 test200; do
  export RUN_DIR="$CF_DIR/$CF_PHASE" LOG_FILE="$CF_DIR/$CF_PHASE/eval.log" VIDEO_DIR="$CF_DIR/$CF_PHASE/videos"
  if [[ "$CF_PHASE" == train40 ]]; then
    CF_ARGS=(--dataset_split train --episodes_per_task 10 --episode_ids_json "$CF_ROOT/config/train_episode_ids.json")
    CF_EXPECTED=10
  else
    CF_ARGS=(--dataset_split test --episodes_per_task 50)
    CF_EXPECTED=50
  fi
  printf "phase=%s started=%s\n" "$CF_PHASE" "$(date -Is)" >> "$CF_DIR/status.txt"
  bash scripts/eval_robomme.sh configs/eval/robomme_released_currentframe.yaml "${CF_ARGS[@]}"
  python - "$RUN_DIR/results.json" "$CF_EXPECTED" <<'PY'
import json,sys
p,n=sys.argv[1],int(sys.argv[2]); d=json.load(open(p))
assert set(d['tasks'])=={'PickXtimes','VideoUnmaskSwap','VideoRepick','PatternLock'},d
assert d['n_excluded_na']==0,d
for t,v in d['tasks'].items():
    assert v['n']==n and not v['oom'] and v['success_rate'] is not None,(t,v)
print('RESULT_VALIDATED',p,'macro_avg=',d['macro_avg'],flush=True)
PY
  printf "phase=%s complete=%s\n" "$CF_PHASE" "$(date -Is)" >> "$CF_DIR/status.txt"
done
