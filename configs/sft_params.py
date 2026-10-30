from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    backbone_model_name_or_path: str = field(default="Qwen/Qwen3.5-4B")
    trust_remote_code: bool = field(default=True)
    dtype: str = field(default="bfloat16")
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    model_max_length: int = field(default=8192)

    dit_hidden_size: int = field(default=2048)
    dit_depth: int = field(default=16)
    dit_num_heads: int = field(default=16)
    dit_mlp_ratio: float = field(default=3.5)
    dit_dropout: float = field(default=0.0)
    dit_rope_theta: Optional[float] = field(default=None)
    dit_mrope_section: Optional[list[int]] = field(default=None)
    dit_partial_rotary_factor: Optional[float] = field(default=None)
    timestep_beta_alpha: float = field(default=1.5)
    timestep_beta_beta: float = field(default=1.0)
    action_loss_weight: float = field(default=1.0)
    vl_loss_weight: float = field(default=1.0)

    freeze_backbone: bool = field(default=False)
    freeze_vision_tower: bool = field(default=False)
    lm_lora_rank: int = field(default=0, metadata={"help": "0 disables LoRA; positive rank adapts only LM decoder projections"})
    lm_lora_alpha: float = field(default=32.0)
    lm_lora_dropout: float = field(default=0.0)
    lm_lora_target_modules: list[str] = field(default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj', 'gate_proj', 'up_proj', 'down_proj'])
    use_proprio: bool = field(default=True)
    state_dropout_prob: float = field(default=0.0)

    init_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Released SimpleMemVLA checkpoint to warm start from (see init_modules)"},
    )
    init_modules: list[str] = field(
        default_factory=lambda: ["backbone", "action_head"],
        metadata={
            "help": "Modules loaded from init_from_checkpoint: backbone [action_head]. "
            "Modules left out (and memory) start from a fresh initialisation."
        },
    )
    condition_source: str = field(
        default="subtask",
        metadata={"help": "subtask: GT sub-task span (baseline); prompt: whole single-frame prompt"},
    )

    use_memory: bool = field(default=False)
    mem_arch: str = field(default="perceiver")
    mem_hidden_size: int = field(default=1024)
    mem_depth: int = field(default=4)
    mem_num_heads: int = field(default=16)
    mem_mlp_ratio: float = field(default=4.0)
    mem_num_queries: int = field(default=16)
    mem_compressor_depth: int = field(default=2)
    mem_frame_pool: int = field(default=4)
    mem_fusion: str = field(default="adaln_add", metadata={"help": "adaln_add | prefix_tokens"})
    mem_dropout: float = field(default=0.0)
    mem_dropout_prob: float = field(default=0.0)
    perc_mem_window: int = field(default=4, metadata={"help": "perc_queue: FIFO window W (keyframes)"})
    perc_mem_queue: int = field(default=28, metadata={"help": "perc_queue: learned-eviction queue K"})
    perc_mem_evict: str = field(default="learned", metadata={"help": "learned | fifo | uniform | random | oracle"})
    perc_mem_score_pool: int = field(default=4, metadata={"help": "scorer pools each stored frame to p x p"})
    perc_mem_read_pool: int = field(default=4, metadata={"help": "reader pools each stored frame to p x p"})
    perc_mem_scorer_hidden_size: int = field(default=256)
    perc_mem_scorer_num_heads: int = field(default=4)
    perc_mem_scorer_depth: int = field(default=2)
    perc_mem_scorer_mlp_ratio: float = field(default=4.0)
    perc_mem_scorer_dropout: float = field(default=0.1)
    perc_mem_scorer_use_redund: bool = field(default=True)
    perc_mem_reader_hidden_size: int = field(default=512)
    perc_mem_reader_num_heads: int = field(default=8)
    perc_mem_reader_num_queries: int = field(default=8)
    perc_mem_reader_depth: int = field(default=2)
    perc_mem_reader_mlp_ratio: float = field(default=4.0)
    perc_mem_scorer_loss_weight: float = field(default=1.0)
    perc_mem_rank_margin: float = field(default=0.1)
    perc_mem_oracle_prob: float = field(default=0.0, metadata={"help": "learned mode: chance a training eviction follows the labels"})
    perc_mem_freeze_scorer: bool = field(default=False)


@dataclass
class DataArguments:

    benchmark: str = field(
        metadata={
            "help": "Which benchmark to train on: "
            "rmbench | robomme | mikasa | robomemarena | libero"
        }
    )

    repo_id: Optional[str] = field(default=None)
    root: Optional[str] = field(default=None)
    revision: Optional[str] = field(default=None)
    episodes: Optional[list[int]] = field(default=None)
    download_videos: bool = field(default=True)
    video_backend: str = field(default="pyav")

    image_keys: Optional[list[str]] = field(default=None)
    history_image_keys: Optional[list[str]] = field(default=None)
    action_key: str = field(default="action")
    state_key: str = field(default="observation.state")
    task_key: str = field(default="task")
    subtask_key: str = field(default="subtask")
    subtask_index_key: Optional[str] = field(default=None)
    skip_video_demo_frames: Optional[bool] = field(default=None)

    history_video_sec: Optional[float] = field(default=None)
    history_video_fps: Optional[float] = field(default=None)
    variable_history: Optional[bool] = field(default=None)
    image_aug: Optional[bool] = field(default=None)
    native_video_fps: Optional[float] = field(default=None)

    robot_tag: Optional[str] = field(default=None)
    control_frequency_hz: Optional[int] = field(default=None)
    action_delta_indices: Optional[list[int]] = field(default=None)

    episodes_file: Optional[str] = field(
        default=None, metadata={"help": "Text file of training episode indices (overrides --episodes)"}
    )
    val_episodes_file: Optional[str] = field(
        default=None, metadata={"help": "Text file of held-out episode indices for eval"}
    )
    stats_path: Optional[str] = field(
        default=None,
        metadata={"help": "stats.json used for action/state normalisation (default <root>/meta)"},
    )
    single_frame: bool = field(
        default=False, metadata={"help": "Feed every camera as a single current image (no history video). condition_source=prompt: whole-prompt condition; condition_source=subtask: image channel + original sub-task teacher forcing"}
    )
    current_frame_only: bool = field(
        default=False, metadata={"help": "Original subtask teacher forcing with only the current front frame (duplicated temporal pair) and wrist image"}
    )
    mem_feature_dir: Optional[str] = field(default=None)
    mem_window_sec: float = field(
        default=60.0, metadata={"help": "history kept before the memory module; <= 0: whole episode"}
    )
    mem_fps: float = field(default=2.0)
    perc_mem_label_dir: Optional[str] = field(
        default=None,
        metadata={"help": "per-episode (N_key, N_key) teacher utilities u[t, tau] for the perc_queue scorer"},
    )


@dataclass
class SFTTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./checkpoints/sft/simplememvla/baseline")
    run_name: Optional[str] = field(default=None, metadata={"help": "Experiment name (W&B run name)"})
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "W&B project name; overrides the WANDB_PROJECT env var when set"},
    )
    report_to: str | list[str] = field(default="wandb")

    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    max_steps: int = field(default=30000)
    save_steps: int = field(default=3000)
    logging_steps: int = field(default=10)
    save_total_limit: int = field(default=3)
    save_first_fraction: Optional[float] = field(
        default=None,
        metadata={"help": "First checkpoint fraction in (0,1]; pair with save_num_checkpoints to replace periodic saves"},
    )
    save_num_checkpoints: Optional[int] = field(
        default=None,
        metadata={"help": "Number of evenly spaced complete checkpoints, including first fraction and final update"},
    )

    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    tf32: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=32)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_persistent_workers: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)

    backbone_lr: float = field(default=5e-6)
    action_head_lr: float = field(default=5e-5)
    memory_lr: float = field(default=1e-4)
    eval_max_batches: int = field(
        default=50, metadata={"help": "Batches per rank in each memory-diagnostic eval"}
    )
    weight_decay: float = field(default=1e-8)
    warmup_steps: int = field(default=1000)
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_epsilon: float = field(default=1e-8)
    optim: str = field(default="adamw_torch")
    max_grad_norm: float = field(default=1.0)
    seed: int = field(default=429)
    resume: str = field(default="false")

    def __post_init__(self):
        super().__post_init__()
        if self.dataloader_num_workers == 0 and self.dataloader_persistent_workers:
            self.dataloader_persistent_workers = False
        if self.lr_scheduler_type == "cosine_with_min_lr":
            kwargs = self.lr_scheduler_kwargs or {}
            if isinstance(kwargs, str):
                import json

                kwargs = json.loads(kwargs) if kwargs.strip() else {}
            if "min_lr" not in kwargs and "min_lr_rate" not in kwargs:
                kwargs["min_lr_rate"] = 0.1
            self.lr_scheduler_kwargs = kwargs
        self.resume = self._normalize_resume(self.resume)
        from simplememvla.training.checkpoint_schedule import validate_checkpoint_schedule, build_checkpoint_steps

        if validate_checkpoint_schedule(self.save_first_fraction, self.save_num_checkpoints):
            if self.max_steps > 0:
                build_checkpoint_steps(self.max_steps, self.save_first_fraction, self.save_num_checkpoints)
            strategy = getattr(self.save_strategy, "value", self.save_strategy)
            if strategy == "best" or self.load_best_model_at_end:
                raise ValueError(
                    "Uniform checkpoints cannot be combined with save_strategy=best "
                    "or load_best_model_at_end; use the explicit progress schedule."
                )
            # Epoch/step training budget is independent of this save strategy.
            # The callback supplies exact absolute update steps instead of a period.
            from transformers.trainer_utils import SaveStrategy
            self.save_strategy = SaveStrategy.STEPS
            if self.save_only_model:
                raise ValueError("Uniform checkpoints require save_only_model=False for optimizer/RNG resume")
            if self.save_total_limit is not None and 0 < self.save_total_limit < self.save_num_checkpoints:
                import warnings
                warnings.warn(
                    f"Raising save_total_limit from {self.save_total_limit} to "
                    f"{self.save_num_checkpoints} to retain all scheduled checkpoints.",
                    UserWarning,
                )
                self.save_total_limit = self.save_num_checkpoints

    @staticmethod
    def _normalize_resume(value):
        if isinstance(value, bool):
            return value
        text = str(value).strip()
        if text.lower() in ("", "false", "none", "0"):
            return False
        if text.lower() in ("true", "1"):
            return True
        return text
