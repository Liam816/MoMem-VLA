# Hyperparameters used to be read from env vars and now live in YAML configs.
# Launchers source this so a leftover env var fails loudly instead of being ignored.
#
# usage: check_legacy_env <script> VAR=--flag [VAR=--flag ...]
check_legacy_env() {
  local script=$1 pair var found=()
  shift
  for pair in "$@"; do
    var="${pair%%=*}"
    if [[ -n "${!var:-}" ]]; then
      found+=("  $var=${!var}  ->  ${pair#*=}")
    fi
  done
  if (( ${#found[@]} )); then
    {
      echo "$script: these env vars are no longer read; their values now come from the YAML config:"
      printf '%s\n' "${found[@]}"
      echo "Unset them and append the flag(s) to the command instead (CLI flags override the YAML)."
    } >&2
    exit 1
  fi
}

# Shared by scripts/train.sh (and train_mem.sh, which execs it).
TRAIN_LEGACY_ENV=(
  RUN_NAME=--run_name
  OUTPUT_DIR=--output_dir
  REPO_ID=--repo_id
  ROOT_DIR=--root
  HISTORY_VIDEO_SEC=--history_video_sec
  HISTORY_VIDEO_FPS=--history_video_fps
  HISTORY_IMAGE_KEYS=--history_image_keys
  VARIABLE_HISTORY=--variable_history
  IMAGE_AUG=--image_aug
  CONTROL_FREQUENCY=--control_frequency_hz
  SUBTASK_INDEX_KEY=--subtask_index_key
  SKIP_VIDEO_DEMO_FRAMES=--skip_video_demo_frames
  USE_PROPRIO=--use_proprio
  STATE_DROPOUT_PROB=--state_dropout_prob
  ATTN_IMPLEMENTATION=--attn_implementation
  DEEPSPEED_CONFIG=--deepspeed
  PER_DEVICE_BATCH=--per_device_train_batch_size
  GRAD_ACCUM=--gradient_accumulation_steps
  MAX_STEPS=--max_steps
  SAVE_STEPS=--save_steps
  SAVE_TOTAL_LIMIT=--save_total_limit
  WARMUP_STEPS=--warmup_steps
  BACKBONE_LR=--backbone_lr
  ACTION_HEAD_LR=--action_head_lr
  GRADIENT_CHECKPOINTING=--gradient_checkpointing
  DATALOADER_NUM_WORKERS=--dataloader_num_workers
  LOGGING_STEPS=--logging_steps
  RESUME=--resume
  # memory-cache experiments (old train_mem.sh)
  BENCHMARK="the first argument of train.sh (memcache configs are RoboMME-only; copy one for another benchmark)"
  CKPT="--init_from_checkpoint, --backbone_model_name_or_path and --stats_path"
  SPLIT_PREFIX="--episodes_file <prefix>_train.txt --val_episodes_file <prefix>_val.txt"
  FEAT_DIR=--mem_feature_dir
  EVAL_STEPS=--eval_steps
  EVAL_MAX_BATCHES=--eval_max_batches
  MEMORY_LR=--memory_lr
  MEM_FRAME_POOL=--mem_frame_pool
  MEM_HIDDEN_SIZE=--mem_hidden_size
  MEM_DEPTH=--mem_depth
  MEM_NUM_QUERIES=--mem_num_queries
  MEM_COMPRESSOR_DEPTH=--mem_compressor_depth
  MEM_WINDOW_SEC=--mem_window_sec
  MEM_FPS=--mem_fps
  MEM_DROPOUT_PROB=--mem_dropout_prob
)
