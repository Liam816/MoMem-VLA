#!/usr/bin/env bash
# Start one memory-cache experiment in the background (survives SSH logout).
#
#   bash scripts/launch_memcache.sh E1 4,5            # new run, or resume if checkpoints exist
#   bash scripts/launch_memcache.sh E2 6,7
#   bash scripts/launch_memcache.sh E2 6,7 --max_steps 20000 --mem_depth 6
#   DRY_RUN=1 bash scripts/launch_memcache.sh E1 4,5  # print the full command only
#
# The recipe is configs/train/memcache/<EXP>.yaml; flags after the GPU list override it.
# Output: $DATA_HOME/outputs/memcache/robomme_small_<EXP>/
# Log:    $DATA_HOME/outputs/memcache/run_<EXP>_<timestamp>.log
set -euo pipefail

EXP="${1:?usage: launch_memcache.sh <E1|E2|E3> <gpu list, e.g. 4,5>}"
GPUS="${2:?usage: launch_memcache.sh <E1|E2|E3> <gpu list, e.g. 4,5>}"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# GRAD_ACCUM / OUTPUT_DIR / RUN_NAME / RESUME are this launcher's own knobs; every other
# old hyperparameter env var must fail here, before the run goes to the background.
source "$REPO_ROOT/scripts/legacy_env.sh"
_legacy=()
for _pair in "${TRAIN_LEGACY_ENV[@]}"; do
  case "${_pair%%=*}" in GRAD_ACCUM|OUTPUT_DIR|RUN_NAME|RESUME) ;; *) _legacy+=("$_pair") ;; esac
done
check_legacy_env scripts/launch_memcache.sh "${_legacy[@]}"
DATA_HOME="${DATA_HOME:-/mnt/cpfs/heping/SimpleMemVLA-data}"
CONDA_ENV="${CONDA_ENV:-simplememvla-robomme}"
OUT_ROOT="${OUT_ROOT:-$DATA_HOME/outputs/memcache}"
RUN_NAME="${RUN_NAME:-robomme_small_${EXP}}"

case "$EXP" in
  E1) DEF_PORT=6031 ;;
  E2) DEF_PORT=6032 ;;
  E3) DEF_PORT=6033 ;;
  *) echo "unknown EXP: $EXP (E1|E2|E3)" >&2; exit 1 ;;
esac
IFS=, read -r -a _gpus <<< "$GPUS"
# Keep global batch 64 = 16 per GPU x GPUs x accumulation.
DEF_ACCUM=$(( 4 / ${#_gpus[@]} ))
(( DEF_ACCUM < 1 )) && DEF_ACCUM=1

source /mnt/cpfs/heping/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
source "$DATA_HOME/eval_env.sh"

export EXP
export CUDA_VISIBLE_DEVICES="$GPUS"
export MASTER_PORT="${MASTER_PORT:-$DEF_PORT}"
GRAD_ACCUM="${GRAD_ACCUM:-$DEF_ACCUM}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_ROOT/$RUN_NAME}"
# auto: resume from the newest checkpoint-* in OUTPUT_DIR, else start fresh.
RESUME="${RESUME:-auto}"
OVERRIDES=(
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --run_name "$RUN_NAME"
  --output_dir "$OUTPUT_DIR"
  --resume "$RESUME"
  "$@"
)
# Consumed above; keep them out of train.sh's environment, where they would be rejected.
export -n GRAD_ACCUM OUTPUT_DIR RUN_NAME RESUME

cd "$REPO_ROOT"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  DRY_RUN=1 bash scripts/train_mem.sh "${OVERRIDES[@]}"
  exit 0
fi

busy=$(nvidia-smi -i "$GPUS" --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 > 2000 {printf "%s(%sMiB) ", $1, $2}')
if [[ -n "$busy" && "${FORCE:-0}" != 1 ]]; then
  echo "GPU(s) already in use: $busy -- set FORCE=1 to launch anyway" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run_${EXP}_$(date +%Y%m%d_%H%M%S).log"
{
  echo "launched $(date '+%F %T') EXP=$EXP GPUS=$GPUS overrides: ${OVERRIDES[*]}"
  echo "git: $(git rev-parse --short HEAD) branch $(git branch --show-current) dirty=$(git status --porcelain | wc -l)"
} > "$LOG"
setsid nohup bash -c 'bash scripts/train_mem.sh "$@"; echo "TRAIN_EXIT=$? $(date "+%F %T")"' \
  _ "${OVERRIDES[@]}" >> "$LOG" 2>&1 < /dev/null &
echo "started $EXP on GPUs $GPUS (session pid $!)"
echo "log:    $LOG"
echo "output: $OUTPUT_DIR"
echo "follow: tail -f $LOG | tr '\\r' '\\n' | grep -a -E 'eval_|loss|Error|TRAIN_EXIT'"
echo "stop:   kill -- -$!"
