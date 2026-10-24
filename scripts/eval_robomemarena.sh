#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/legacy_env.sh"
check_legacy_env scripts/eval_robomemarena.sh \
  TRIALS=--num-trials-per-task \
  TASK_START=--task-start \
  TASK_END=--task-end \
  SEED=--seed \
  WORKERS_PER_GPU=--workers-per-gpu \
  MAX_STEPS=--max-steps \
  REPLAN_STEPS=--replan-steps \
  NUM_STEPS_WAIT=--num-steps-wait \
  NUM_DENOISING_STEPS=--num-denoising-steps \
  MAX_REASONING_TOKENS=--max-reasoning-tokens \
  ATTN_IMPLEMENTATION=--attn-implementation \
  EPISODES_PER_SHARD=--episodes-per-shard \
  LOG_EVERY=--log-every \
  SEED_POLICY="--seed-policy / --no-seed-policy" \
  RESUME="--resume / --no-resume" \
  GPU_MEMORY_CAP_GIB=--gpu-memory-cap-gib

# usage: bash scripts/eval_robomemarena.sh <checkpoint_dir> [out_dir] [--flag value ...]
# Protocol knobs live in configs/eval/robomemarena.yaml (CONFIG=... picks another file, and an
# extra --config file.yaml merges on top); trailing flags override it, e.g. --task-end 3.
# GPUS=0,1 picks the GPUs (same as --gpus 0,1).
CKPT="${1:?usage: bash eval_robomemarena.sh <checkpoint_dir> [out_dir] [--flag value ...]}"
shift
OUT="logs/eval_$(basename "$CKPT")_$(date +%Y%m%d_%H%M%S)"
if [[ $# -gt 0 && "$1" != --* ]]; then OUT="$1"; shift; fi
CONFIG="${CONFIG:-configs/eval/robomemarena.yaml}"
ARGS=(--config "$CONFIG" --checkpoint "$CKPT" --out-root "$OUT" ${GPUS:+--gpus "$GPUS"} "$@")
PYBIN="${PYBIN:-python}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if ! "$PYBIN" - "${ARGS[@]}" <<'PREFLIGHT'
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
fork = pathlib.Path("evaluation_benchmark/libero_fork/libero").resolve()
if not fork.is_dir():
    sys.exit(f"vendored LIBERO fork missing at {fork}; run bash scripts/install/install_robomemarena_sim.sh")

from robomemarena_sim._compat import apply_fla_torch_compat
apply_fla_torch_compat()

import transformers
if transformers.__version__ != "5.13.1":
    sys.exit(f"transformers {transformers.__version__} != 5.13.1, the version this "
             f"checkpoint was trained and saved under ({sys.executable}). The rendered "
             "prompt is version-dependent. Run: conda activate simplememvla_robomemarena")

from robomemarena_sim.eval_success import build_argparser
from simplememvla.config import parse_args_with_config
args = parse_args_with_config(build_argparser(), sys.argv[1:])
if args.attn_implementation == "flash_attention_2":
    try:
        import flash_attn
    except Exception as exc:
        sys.exit(f"flash-attn is not importable ({exc}); build it with "
                 "bash scripts/install/install_fast_path.sh, or pass --attn-implementation sdpa to accept a "
                 "different attention backend than training used")

from robomemarena_sim import prepare_mujoco_runtime
prepare_mujoco_runtime()
import libero
where = pathlib.Path(libero.__path__[0]).resolve()
if where != fork:
    sys.exit(f"`import libero` resolves to {where}, not the vendored fork {fork}; upstream "
             "LIBERO lacks this benchmark's objects. An editable install is shadowing it.")
print(f"preflight ok: {sys.executable} (transformers {transformers.__version__})")
PREFLIGHT
then
  echo "preflight FAILED -- not launching any workers" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "config     : $CONFIG"
echo "checkpoint : $CKPT"
echo "output     : $OUT"

"$PYBIN" -u -m robomemarena_sim.eval_success "${ARGS[@]}" 2>&1 | tee -a "$OUT/eval.log"
