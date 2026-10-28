#!/usr/bin/env bash
# Usage: bash scripts/run_vlm_swap_ab.sh <A|B> <gpus> <absolute-output-root>
# A = released VLM + released DiT; B = original Qwen3.5-4B + same released DiT.
set -eo pipefail
VARIANT=${1:?A or B required}
AB_GPUS=${2:?GPU list required}
AB_ROOT=${3:?absolute output root required}
REPO=/mnt/cpfs/heping/SimpleMemVLA
DATA=/mnt/cpfs/heping/SimpleMemVLA-data
NATIVE=/mnt/cpfs/heping/checkpoints/Qwen3.5-4B
case "$VARIANT" in
  A) OVERRIDE=() ;;
  B) OVERRIDE=(--backbone_override "$NATIVE") ;;
  *) echo "Expected A or B" >&2; exit 2 ;;
esac
[[ "$AB_ROOT" == /* ]] || { echo "Output root must be absolute" >&2; exit 2; }
VARIANT_DIR="$AB_ROOT/$VARIANT"
mkdir -p "$VARIANT_DIR"
[[ ! -e "$VARIANT_DIR/status.txt" ]] || { echo "Run already exists: $VARIANT_DIR" >&2; exit 2; }
trap 'ab_exit=$?; printf "exit=%s\nended=%s\n" "$ab_exit" "$(date -Is)" >> "$VARIANT_DIR/status.txt"' EXIT
printf "variant=%s\ngpus=%s\nstarted=%s\n" "$VARIANT" "$AB_GPUS" "$(date -Is)" > "$VARIANT_DIR/status.txt"
source /mnt/cpfs/heping/miniconda3/etc/profile.d/conda.sh
conda activate simplememvla-robomme
source "$DATA/eval_env.sh"
export TMPDIR=/dev/shm/simplememvla_tmp
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export CHECKPOINT="$DATA/checkpoints/simplememvla_robomme"
export GPUS="$AB_GPUS"
export PYTHONUNBUFFERED=1
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
git diff -- robomme_sim/policy.py robomme_sim/eval_success.py > "$VARIANT_DIR/code_diff.patch"
cp configs/eval/robomme_vlm_swap_ab.yaml "$VARIANT_DIR/eval_config.yaml"
cp "$DATA/outputs/memcache/eval/train_split_trained10.json" "$VARIANT_DIR/train_episode_ids.json"
python - "$VARIANT" "$VARIANT_DIR" "$AB_GPUS" <<'PY'
import hashlib,json,pathlib,subprocess,sys
variant,out,gpus=sys.argv[1:]
repo=pathlib.Path('/mnt/cpfs/heping/SimpleMemVLA')
ckpt=pathlib.Path('/mnt/cpfs/heping/SimpleMemVLA-data/checkpoints/simplememvla_robomme')
files=[repo/p for p in ('robomme_sim/policy.py','robomme_sim/eval_success.py','robomme_sim/batched_policy.py','simplememvla/model/dit_action_head.py','configs/eval/robomme.yaml','configs/eval/robomme_vlm_swap_ab.yaml','scripts/run_vlm_swap_ab.sh')]
files += [ckpt/p for p in ('config.json','stats.json','processor_config.json','tokenizer_config.json','chat_template.jinja')]
data={'variant':variant,'backbone_source':str(ckpt) if variant=='A' else '/mnt/cpfs/heping/checkpoints/Qwen3.5-4B','action_head_source':str(ckpt),'processor_source':str(ckpt),'stats_source':str(ckpt),'gpus':gpus,'sampling_seed_after_model_load':0,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
pathlib.Path(out,'manifest.json').write_text(json.dumps(data,indent=2)+'\n')
PY
for PHASE in train40 test200; do
  export RUN_DIR="$VARIANT_DIR/$PHASE"
  export LOG_FILE="$RUN_DIR/eval.log"
  export VIDEO_DIR="$RUN_DIR/videos"
  if [[ "$PHASE" == train40 ]]; then
    PHASE_ARGS=(--dataset_split train --episodes_per_task 10 --episode_ids_json "$VARIANT_DIR/train_episode_ids.json")
    EXPECTED_N=10
  else
    PHASE_ARGS=(--dataset_split test --episodes_per_task 50)
    EXPECTED_N=50
  fi
  printf "phase=%s started=%s\n" "$PHASE" "$(date -Is)" >> "$VARIANT_DIR/status.txt"
  bash scripts/eval_robomme.sh configs/eval/robomme_vlm_swap_ab.yaml "${OVERRIDE[@]}" "${PHASE_ARGS[@]}"
  python - "$RUN_DIR/results.json" "$EXPECTED_N" <<'PY'
import json,sys
p,n=sys.argv[1],int(sys.argv[2]); d=json.load(open(p))
expected={'PickXtimes','VideoUnmaskSwap','VideoRepick','PatternLock'}
assert set(d['tasks'])==expected, d
assert d['n_excluded_na']==0, d
for task,v in d['tasks'].items():
    assert v['n']==n and not v['oom'] and v['success_rate'] is not None,(task,v)
print('RESULT_VALIDATED',p,'macro_avg=',d['macro_avg'],flush=True)
PY
  printf "phase=%s complete=%s\n" "$PHASE" "$(date -Is)" >> "$VARIANT_DIR/status.txt"
done
