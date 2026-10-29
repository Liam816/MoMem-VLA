import dataclasses
import json
import os
import warnings
from pathlib import Path

import torch
from simplememvla.compat import apply_fla_torch_compat

apply_fla_torch_compat()

from transformers import AutoProcessor, HfArgumentParser, TrainerCallback, set_seed

from configs.sft_params import (
    DataArguments,
    ModelArguments,
    SFTTrainingArguments,
)
from simplememvla.benchmarks import BenchmarkSpec, get_benchmark
from simplememvla.config import (
    apply_config_defaults,
    dump_config,
    flatten_sections,
    load_configs,
    split_config_args,
)
from simplememvla.data.collator import SimpleMemVLADataCollator
from simplememvla.data.dataset import MEM_META_FILE, SimpleMemVLADataset
from simplememvla.model import SimpleMemVLAConfig, SimpleMemVLAForActionPrediction
from simplememvla.training.sft_runner import TrainRunner

warnings.filterwarnings("ignore", category=FutureWarning)


def _dtype_from_string(value: str):
    value = value.lower()
    if value in ("auto", "none"):
        return "auto" if value == "auto" else None
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16", "half"):
        return torch.float16
    if value in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def resolve_benchmark_defaults(data_args: DataArguments) -> BenchmarkSpec:
    spec = get_benchmark(data_args.benchmark)
    if data_args.repo_id is None:
        data_args.repo_id = spec.repo_id
    if data_args.root is None:
        data_args.root = spec.root
    if data_args.image_keys is None:
        data_args.image_keys = list(spec.image_keys)
    if data_args.history_image_keys is None:
        data_args.history_image_keys = list(spec.history_image_keys)
    if data_args.history_video_sec is None:
        data_args.history_video_sec = spec.history_video_sec
    if data_args.history_video_fps is None:
        data_args.history_video_fps = spec.history_video_fps
    if data_args.variable_history is None:
        data_args.variable_history = spec.variable_history
    if data_args.image_aug is None:
        data_args.image_aug = spec.image_aug
    if data_args.robot_tag is None:
        data_args.robot_tag = spec.robot_tag
    if data_args.subtask_index_key is None:
        data_args.subtask_index_key = spec.subtask_index_key
    if data_args.skip_video_demo_frames is None:
        data_args.skip_video_demo_frames = spec.skip_video_demo_frames
    if data_args.action_delta_indices is None:
        data_args.action_delta_indices = list(range(spec.num_actions_chunk))
    return spec


def read_episodes_file(path: str) -> list[int]:
    text = Path(path).read_text()
    return [int(tok) for tok in text.replace(",", " ").split()]


def resolve_memcache_defaults(model_args: ModelArguments, data_args: DataArguments) -> dict:
    if data_args.episodes_file:
        data_args.episodes = read_episodes_file(data_args.episodes_file)
    ckpt = model_args.init_from_checkpoint
    if ckpt and "action_head" in model_args.init_modules and data_args.stats_path is None:
        # A warm-started DiT must keep the normalisation it was trained with.
        data_args.stats_path = os.path.join(ckpt, "stats.json")
    if model_args.condition_source == "prompt" and model_args.vl_loss_weight:
        model_args.vl_loss_weight = 0.0
    mem_meta = {}
    if model_args.use_memory:
        if not data_args.mem_feature_dir:
            raise ValueError("use_memory=True requires --mem_feature_dir")
        with open(os.path.join(data_args.mem_feature_dir, MEM_META_FILE)) as f:
            mem_meta = json.load(f)
        # The cache must hold tokens from the same frozen ViT that encodes the current frame.
        vlm_source = model_args.init_from_checkpoint or model_args.backbone_model_name_or_path
        feat_source = mem_meta.get("checkpoint")
        if feat_source is None or os.path.realpath(feat_source) != os.path.realpath(vlm_source):
            raise ValueError(
                f"{data_args.mem_feature_dir} was extracted with the vision tower of "
                f"{feat_source}, but the VLM is loaded from {vlm_source}; re-run "
                "scripts/extract_mem_features.py --checkpoint <that VLM>."
            )
    elif data_args.mem_feature_dir:
        raise ValueError("--mem_feature_dir is set but --use_memory is false")
    return mem_meta


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: SFTTrainingArguments,
):
    set_seed(training_args.seed)
    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    spec = resolve_benchmark_defaults(data_args)
    mem_meta = resolve_memcache_defaults(model_args, data_args)

    processor = AutoProcessor.from_pretrained(
        model_args.init_from_checkpoint or model_args.backbone_model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
        model_max_length=model_args.model_max_length,
    )

    mem_frame_pool = model_args.mem_frame_pool if model_args.use_memory else None
    train_dataset = SimpleMemVLADataset(
        data_args,
        spec,
        condition_source=model_args.condition_source,
        mem_frame_pool=mem_frame_pool,
    )
    eval_dataset = None
    if data_args.val_episodes_file:
        eval_args = dataclasses.replace(
            data_args,
            episodes=read_episodes_file(data_args.val_episodes_file),
            image_aug=False,
        )
        eval_dataset = SimpleMemVLADataset(
            eval_args,
            spec,
            condition_source=model_args.condition_source,
            mem_frame_pool=mem_frame_pool,
        )

    config = SimpleMemVLAConfig(
        backbone_model_name_or_path=model_args.backbone_model_name_or_path,
        action_dim=spec.action_dim,
        action_horizon=spec.num_actions_chunk,
        state_dim=spec.state_dim,
        dit_hidden_size=model_args.dit_hidden_size,
        dit_depth=model_args.dit_depth,
        dit_num_heads=model_args.dit_num_heads,
        dit_mlp_ratio=model_args.dit_mlp_ratio,
        dit_dropout=model_args.dit_dropout,
        dit_rope_theta=model_args.dit_rope_theta,
        dit_mrope_section=model_args.dit_mrope_section,
        dit_partial_rotary_factor=model_args.dit_partial_rotary_factor,
        timestep_beta_alpha=model_args.timestep_beta_alpha,
        timestep_beta_beta=model_args.timestep_beta_beta,
        action_loss_weight=model_args.action_loss_weight,
        vl_loss_weight=model_args.vl_loss_weight,
        freeze_backbone=model_args.freeze_backbone,
        freeze_vision_tower=model_args.freeze_vision_tower,
        lm_lora_rank=model_args.lm_lora_rank,
        lm_lora_alpha=model_args.lm_lora_alpha,
        lm_lora_dropout=model_args.lm_lora_dropout,
        lm_lora_target_modules=model_args.lm_lora_target_modules,
        use_proprio=model_args.use_proprio,
        state_dropout_prob=model_args.state_dropout_prob,
        robot_tag=data_args.robot_tag,
        control_frequency_hz=train_dataset.control_frequency_hz,
        history_video_sec=data_args.history_video_sec,
        history_video_fps=data_args.history_video_fps,
        native_video_fps=train_dataset.native_fps,
        variable_history=data_args.variable_history,
        image_keys=data_args.image_keys,
        history_image_keys=train_dataset.history_image_keys,
        subtask_key=train_dataset.subtask_key,
        subtask_index_key=train_dataset.subtask_index_key,
        condition_source=model_args.condition_source,
        single_frame=data_args.single_frame,
        current_frame_only=data_args.current_frame_only,
        use_memory=model_args.use_memory,
        mem_feature_dim=mem_meta.get("feature_dim"),
        mem_feature_type=mem_meta.get("feature_type", "vit_merger"),
        mem_camera_key=mem_meta.get("camera_key"),
        mem_window_sec=data_args.mem_window_sec,
        mem_fps=data_args.mem_fps,
        mem_frame_pool=model_args.mem_frame_pool,
        mem_arch=model_args.mem_arch,
        mem_hidden_size=model_args.mem_hidden_size,
        mem_depth=model_args.mem_depth,
        mem_num_heads=model_args.mem_num_heads,
        mem_mlp_ratio=model_args.mem_mlp_ratio,
        mem_num_queries=model_args.mem_num_queries,
        mem_compressor_depth=model_args.mem_compressor_depth,
        mem_fusion=model_args.mem_fusion,
        mem_dropout=model_args.mem_dropout,
        mem_dropout_prob=model_args.mem_dropout_prob,
    )

    if model_args.init_from_checkpoint:
        dtype = _dtype_from_string(model_args.dtype)
        if not isinstance(dtype, torch.dtype):
            raise ValueError("--init_from_checkpoint needs an explicit --dtype (e.g. bfloat16)")
        model = SimpleMemVLAForActionPrediction.from_simplememvla_checkpoint(
            model_args.init_from_checkpoint,
            config=config,
            dtype=dtype,
            attn_implementation=model_args.attn_implementation,
            init_modules=tuple(model_args.init_modules),
        )
    else:
        model = SimpleMemVLAForActionPrediction.from_backbone(
            model_args.backbone_model_name_or_path,
            config=config,
            dtype=_dtype_from_string(model_args.dtype),
            attn_implementation=model_args.attn_implementation,
            trust_remote_code=model_args.trust_remote_code,
        )
    model._apply_backbone_freeze(model_args.freeze_backbone)
    if model_args.use_memory and not (model_args.freeze_backbone or model_args.freeze_vision_tower):
        raise ValueError("A precomputed vision cache requires a frozen visual encoder")
    if model_args.lm_lora_rank > 0:
        from simplememvla.model.lora import is_lora_parameter
        trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
        if not trainable or any(not is_lora_parameter(n) for n in trainable):
            raise RuntimeError("LM LoRA training must update only adapter A/B parameters in backbone")
    if training_args.process_index == 0:
        if model_args.lm_lora_rank > 0:
            print(f"[lora] rank={model_args.lm_lora_rank}; alpha={model_args.lm_lora_alpha}; dropout={model_args.lm_lora_dropout}; targeted_modules={len(model.lm_lora_targets)}; base LM frozen")
        print(f"[input] current_frame_only={data_args.current_frame_only}; condition_source={model_args.condition_source}; single_frame={data_args.single_frame}")
        print(f"[visual] frozen={model_args.freeze_vision_tower or model_args.freeze_backbone}; dtype={model.backbone.model.visual.dtype}; attn={model.backbone.model.visual.config._attn_implementation}")
        counts: dict[str, list[int]] = {}
        for name, param in model.named_parameters():
            key = name.split(".")[0]
            total, trainable = counts.setdefault(key, [0, 0])
            counts[key] = [total + param.numel(), trainable + param.numel() * param.requires_grad]
        for key, (total, trainable) in counts.items():
            print(f"[params] {key}: total={total / 1e6:.1f}M trainable={trainable / 1e6:.1f}M")
        print(f"[norm] stats_path={data_args.stats_path or os.path.join(data_args.root, 'meta/stats.json')}")
        if model_args.init_from_checkpoint:
            fresh = sorted(set(counts) - set(model_args.init_modules))
            print(
                f"[init] loaded {list(model_args.init_modules)} from "
                f"{model_args.init_from_checkpoint}; fresh init: {fresh}"
            )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = SimpleMemVLADataCollator(
        processor,
        max_length=model_args.model_max_length,
        condition_source=model_args.condition_source,
    )

    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.process_index == 0:
        save_resolved_config(model_args, data_args, training_args)
    runner = TrainRunner(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        resume_from_checkpoint=training_args.resume,
        processor=processor,
        data_root=data_args.root,
        eval_dataset=eval_dataset,
        stats_path=data_args.stats_path,
    )
    runner.trainer.add_callback(JsonlMetricsCallback())
    runner.train()


class JsonlMetricsCallback(TrainerCallback):
    """Persist each rank-zero log immediately, including before the first save."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            import time
            record = {"step": state.global_step, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **logs}
            with open(Path(args.output_dir) / "metrics.jsonl", "a") as f:
                f.write(json.dumps(record, default=float) + "\n")


CONFIG_SECTIONS = {
    "model": ModelArguments,
    "data": DataArguments,
    "training": SFTTrainingArguments,
}


def parse_train_args(args: list[str] | None = None):
    parser = HfArgumentParser(tuple(CONFIG_SECTIONS.values()))
    config_paths, rest = split_config_args(args)
    if config_paths:
        sections = {
            name: [f.name for f in dataclasses.fields(cls) if f.init]
            for name, cls in CONFIG_SECTIONS.items()
        }
        flat = flatten_sections(load_configs(config_paths), sections)
        apply_config_defaults(parser, flat)
    return parser.parse_args_into_dataclasses(args=rest, look_for_args_file=False)


def save_resolved_config(model_args, data_args, training_args) -> None:
    """Record the fully resolved arguments (YAML + CLI + benchmark defaults)."""
    resolved = {
        "model": dataclasses.asdict(model_args),
        "data": dataclasses.asdict(data_args),
        "training": json.loads(training_args.to_json_string()),
    }
    dump_config(resolved, os.path.join(training_args.output_dir, "train_config.yaml"))


if __name__ == "__main__":
    model_args, data_args, training_args = parse_train_args()

    train(model_args, data_args, training_args)
