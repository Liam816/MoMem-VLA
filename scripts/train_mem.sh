#!/usr/bin/env bash
# Single-frame + history vision-token cache experiments (frozen VLM, warm-started DiT).
#   E1: single frame, no memory          (lower bound)
#   E2: + memory cache, adaLN fusion     (main)
#   E3: + memory cache, prefix tokens    (fusion control)
# usage: EXP=E2 bash scripts/train_mem.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

EXP="${EXP:-E2}"
BENCHMARK="${BENCHMARK:-robomme}"
DATA_HOME="${DATA_HOME:-/mnt/cpfs/heping/SimpleMemVLA-data}"
CKPT="${CKPT:-$DATA_HOME/checkpoints/simplememvla_${BENCHMARK}}"
ROOT_DIR="${ROOT_DIR:-$DATA_HOME/datasets/yinchenghust/${BENCHMARK}_lerobot}"
SPLIT_PREFIX="${SPLIT_PREFIX:-$DATA_HOME/splits/${BENCHMARK}_small}"
FEAT_DIR="${FEAT_DIR:-$DATA_HOME/features/${BENCHMARK}_vit_merger_front}"

case "$EXP" in
  E1) USE_MEMORY=false; MEM_FUSION=adaln_add ;;
  E2) USE_MEMORY=true;  MEM_FUSION=adaln_add ;;
  E3) USE_MEMORY=true;  MEM_FUSION=prefix_tokens ;;
  *) echo "unknown EXP: $EXP (E1|E2|E3)" >&2; exit 1 ;;
esac

RUN_NAME="${RUN_NAME:-${BENCHMARK}_small_${EXP}}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/mem/${RUN_NAME}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
IFS=, read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
GPUS_PER_NODE="${GPUS_PER_NODE:-${#_gpus[@]}}"
MASTER_PORT="${MASTER_PORT:-6011}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-15000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"
ACTION_HEAD_LR="${ACTION_HEAD_LR:-3e-5}"
MEMORY_LR="${MEMORY_LR:-1e-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
IMAGE_AUG="${IMAGE_AUG:-false}"

MEM_FRAME_POOL="${MEM_FRAME_POOL:-4}"
MEM_HIDDEN_SIZE="${MEM_HIDDEN_SIZE:-1024}"
MEM_DEPTH="${MEM_DEPTH:-4}"
MEM_NUM_QUERIES="${MEM_NUM_QUERIES:-16}"
MEM_COMPRESSOR_DEPTH="${MEM_COMPRESSOR_DEPTH:-2}"
MEM_WINDOW_SEC="${MEM_WINDOW_SEC:-60}"
MEM_FPS="${MEM_FPS:-2}"
MEM_DROPOUT_PROB="${MEM_DROPOUT_PROB:-0.0}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_PROJECT="${WANDB_PROJECT:-simplememvla-memcache}"
export WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# DataLoader workers hand tensors back over a Unix socket created in TMPDIR;
# CPFS (where eval_env.sh points TMPDIR) rejects sockets with Errno 95.
export TMPDIR="${TRAIN_TMPDIR:-/dev/shm/simplememvla_tmp}"
mkdir -p "$TMPDIR"

MEM_ARGS=(--use_memory "$USE_MEMORY")
if [[ "$USE_MEMORY" == true ]]; then
  MEM_ARGS+=(
    --mem_feature_dir "$FEAT_DIR"
    --mem_window_sec "$MEM_WINDOW_SEC" --mem_fps "$MEM_FPS"
    --mem_frame_pool "$MEM_FRAME_POOL"
    --mem_hidden_size "$MEM_HIDDEN_SIZE" --mem_depth "$MEM_DEPTH" --mem_num_heads 16
    --mem_mlp_ratio 4.0
    --mem_num_queries "$MEM_NUM_QUERIES" --mem_compressor_depth "$MEM_COMPRESSOR_DEPTH"
    --mem_fusion "$MEM_FUSION" --mem_dropout_prob "$MEM_DROPOUT_PROB"
  )
fi

CMD=(torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  --master_port "$MASTER_PORT" \
  train.py \
  --benchmark "$BENCHMARK" \
  --root "$ROOT_DIR" \
  --episodes_file "${SPLIT_PREFIX}_train.txt" \
  --val_episodes_file "${SPLIT_PREFIX}_val.txt" \
  --stats_path "$CKPT/stats.json" \
  --single_frame true \
  --image_aug "$IMAGE_AUG" \
  --init_from_checkpoint "$CKPT" \
  --backbone_model_name_or_path "$CKPT" \
  --attn_implementation flash_attention_2 \
  --dtype bfloat16 \
  --freeze_backbone true \
  --condition_source prompt \
  --vl_loss_weight 0.0 \
  --use_proprio true \
  "${MEM_ARGS[@]}" \
  --deepspeed ./configs/zero2.json \
  --run_name "$RUN_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --per_device_eval_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_steps "$MAX_STEPS" \
  --warmup_steps "$WARMUP_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --eval_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --eval_max_batches "$EVAL_MAX_BATCHES" \
  --backbone_lr 0.0 \
  --action_head_lr "$ACTION_HEAD_LR" \
  --memory_lr "$MEMORY_LR" \
  --weight_decay 1e-8 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --optim adamw_torch \
  --bf16 true \
  --tf32 true \
  --gradient_checkpointing false \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --report_to wandb \
  --logging_steps 10 \
  --log_level info \
  --seed 429 \
  --resume "${RESUME:-false}"
)

# DRY_RUN=1 prints the fully expanded command (with the exported env) and exits.
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES TMPDIR=$TMPDIR WANDB_MODE=$WANDB_MODE WANDB_PROJECT=$WANDB_PROJECT PYTHONPATH=$PYTHONPATH"
  echo "cd $REPO_ROOT"
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
"${CMD[@]}"
