import json
import os
from dataclasses import dataclass

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    PreTrainedModel,
)
from transformers.utils import ModelOutput

from simplememvla.model.configuration_simplememvla import SimpleMemVLAConfig
from simplememvla.model.dit_action_head import DiTActionHead
from simplememvla.model.memory import NaiveVisionTokenCache, VisionTokenMemory
from simplememvla.model.perc_memory import PerceptionMemory
from simplememvla.model.lora import inject_lm_lora, is_lora_parameter

MEMORY_MODES = ("normal", "zero", "shuffle")
CHECKPOINT_INIT_MODULES = ("backbone", "action_head")


@dataclass
class SimpleMemVLAOutput(ModelOutput):
    loss: torch.Tensor | None = None
    action_loss: torch.Tensor | None = None
    vl_loss: torch.Tensor | None = None
    pred_velocity: torch.Tensor | None = None
    mem_norm: torch.Tensor | None = None
    mem_rank_loss: torch.Tensor | None = None


def load_checkpoint_state_dict(checkpoint: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        state_dict: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            state_dict.update(load_file(os.path.join(checkpoint, shard)))
        return state_dict
    return load_file(os.path.join(checkpoint, "model.safetensors"))


def sample_flow_timesteps(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    beta_alpha: float = 1.5,
    beta_beta: float = 1.0,
) -> torch.Tensor:
    dist = torch.distributions.Beta(
        torch.tensor(beta_alpha, device=device, dtype=torch.float32),
        torch.tensor(beta_beta, device=device, dtype=torch.float32),
    )
    t = dist.sample((batch_size,))
    return t.to(dtype=dtype)


def masked_flow_matching_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    pred = pred_velocity.float()
    target = target_velocity.float()
    mask = action_mask.to(device=pred.device, dtype=torch.float32)
    squared_error = (pred - target).pow(2) * mask
    per_channel_den = mask.sum(dim=1).clamp_min(1.0)
    per_channel = squared_error.sum(dim=1) / per_channel_den
    active_channels = (mask.sum(dim=1) > 0).to(dtype=torch.float32)
    per_sample_den = active_channels.sum(dim=1).clamp_min(1.0)
    per_sample = (per_channel * active_channels).sum(dim=1) / per_sample_den
    return per_sample.mean()


class SimpleMemVLAForActionPrediction(PreTrainedModel):

    config_class = SimpleMemVLAConfig
    base_model_prefix = "simplememvla"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: SimpleMemVLAConfig,
        backbone: nn.Module | None = None,
        defer_lora_init: bool = False,
    ):
        super().__init__(config)
        self.backbone = backbone
        if self.backbone is None:
            if config.backbone_config:
                backbone_config = AutoConfig.for_model(**dict(config.backbone_config))
            else:
                backbone_config = AutoConfig.from_pretrained(
                    config.backbone_model_name_or_path
                )
            self.backbone = AutoModelForImageTextToText.from_config(backbone_config)

        if config.action_dim is None or config.action_horizon is None:
            raise ValueError(
                "SimpleMemVLAConfig.action_dim and action_horizon must be set; pick "
                "them from the benchmark spec (simplememvla.benchmarks.get_benchmark)."
            )
        if config.use_proprio and config.state_dim is None:
            raise ValueError(
                "SimpleMemVLAConfig.state_dim must be set when use_proprio=True; pick "
                "it from the benchmark spec (simplememvla.benchmarks.get_benchmark)."
            )
        config.hidden_size = self.backbone.config.text_config.hidden_size
        config.backbone_config = self.backbone.config.to_dict()
        rope_parameters = self.backbone.config.text_config.rope_parameters
        if config.dit_rope_theta is None:
            config.dit_rope_theta = rope_parameters.get("rope_theta")
        if config.dit_mrope_section is None:
            config.dit_mrope_section = rope_parameters.get("mrope_section")
        if config.dit_partial_rotary_factor is None:
            config.dit_partial_rotary_factor = rope_parameters.get("partial_rotary_factor")
        self.action_head = DiTActionHead(
            vlm_hidden_size=config.hidden_size,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
            hidden_size=config.dit_hidden_size,
            depth=config.dit_depth,
            num_heads=config.dit_num_heads,
            mlp_ratio=config.dit_mlp_ratio,
            dropout=config.dit_dropout,
            rope_theta=config.dit_rope_theta,
            mrope_section=config.dit_mrope_section,
            partial_rotary_factor=config.dit_partial_rotary_factor,
            use_proprio=config.use_proprio,
            state_dim=config.state_dim,
            state_dropout_prob=config.state_dropout_prob,
        )
        self.memory = None
        if config.use_memory:
            if config.mem_feature_dim is None:
                config.mem_feature_dim = self.backbone.config.vision_config.out_hidden_size
            if config.mem_arch == "naive_tokens":
                self.memory = NaiveVisionTokenCache(
                    feature_dim=config.mem_feature_dim,
                    cond_dim=config.dit_hidden_size,
                    tokens_per_frame=config.mem_frame_pool**2,
                )
            elif config.mem_arch == "perc_queue":
                self.memory = PerceptionMemory(
                    feature_dim=config.mem_feature_dim,
                    cond_dim=config.dit_hidden_size,
                    window_frames=config.perc_mem_window,
                    queue_frames=config.perc_mem_queue,
                    evict_mode=config.perc_mem_evict,
                    score_pool=config.perc_mem_score_pool,
                    read_pool=config.perc_mem_read_pool,
                    scorer_hidden_size=config.perc_mem_scorer_hidden_size,
                    scorer_num_heads=config.perc_mem_scorer_num_heads,
                    scorer_depth=config.perc_mem_scorer_depth,
                    scorer_mlp_ratio=config.perc_mem_scorer_mlp_ratio,
                    scorer_dropout=config.perc_mem_scorer_dropout,
                    scorer_use_redund=config.perc_mem_scorer_use_redund,
                    reader_hidden_size=config.perc_mem_reader_hidden_size,
                    reader_num_heads=config.perc_mem_reader_num_heads,
                    reader_num_queries=config.perc_mem_reader_num_queries,
                    reader_depth=config.perc_mem_reader_depth,
                    reader_mlp_ratio=config.perc_mem_reader_mlp_ratio,
                    fusion=config.mem_fusion,
                    rank_margin=config.perc_mem_rank_margin,
                    oracle_prob=config.perc_mem_oracle_prob,
                )
            else:
                self.memory = VisionTokenMemory(
                    feature_dim=config.mem_feature_dim,
                    cond_dim=config.dit_hidden_size,
                    tokens_per_frame=config.mem_frame_pool**2,
                    hidden_size=config.mem_hidden_size,
                    depth=config.mem_depth,
                    num_heads=config.mem_num_heads,
                    mlp_ratio=config.mem_mlp_ratio,
                    num_queries=config.mem_num_queries,
                    compressor_depth=config.mem_compressor_depth,
                    fusion=config.mem_fusion,
                    dropout=config.mem_dropout,
                )
        self._lm_lora_initialized = False
        if not defer_lora_init:
            self._init_lm_lora()
        self._apply_backbone_freeze(config.freeze_backbone)
        for module in self.modules():
            module._is_hf_initialized = True
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        result = super().from_pretrained(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        # HF restores trainability after construction. Reapply the configured
        # freeze/LoRA policy after weights have actually been loaded.
        model._apply_backbone_freeze(model.config.freeze_backbone)
        return result

    @classmethod
    def from_backbone(
        cls,
        backbone_model_name_or_path: str,
        config: SimpleMemVLAConfig | None = None,
        dtype: torch.dtype | str | None = None,
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "SimpleMemVLAForActionPrediction":
        config = config or SimpleMemVLAConfig(
            backbone_model_name_or_path=backbone_model_name_or_path
        )
        config.backbone_model_name_or_path = backbone_model_name_or_path
        model_kwargs = dict(kwargs)
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs["trust_remote_code"] = trust_remote_code
        backbone = AutoModelForImageTextToText.from_pretrained(
            backbone_model_name_or_path,
            **model_kwargs,
        )
        model = cls(config=config, backbone=backbone)
        target_dtype = backbone.dtype if dtype is not None else None
        if target_dtype is not None:
            model.action_head.to(dtype=target_dtype)
            if model.memory is not None:
                model.memory.to(dtype=target_dtype)
        return model

    @classmethod
    def from_simplememvla_checkpoint(
        cls,
        checkpoint: str,
        config: SimpleMemVLAConfig,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str | None = None,
        init_modules: tuple[str, ...] = ("backbone", "action_head"),
    ) -> "SimpleMemVLAForActionPrediction":
        """Warm start from a released SimpleMemVLA checkpoint.

        Only the top-level modules in ``init_modules`` are loaded: ``backbone``
        (required) and optionally ``action_head``. Everything else -- the DiT
        when it is left out, and ``memory.*`` -- keeps its fresh initialisation,
        with the DiT shape taken from ``config``.
        """
        init_modules = tuple(dict.fromkeys(init_modules))
        unknown = set(init_modules) - set(CHECKPOINT_INIT_MODULES)
        if unknown or "backbone" not in init_modules:
            raise ValueError(
                f"init_modules must include 'backbone' and only use "
                f"{CHECKPOINT_INIT_MODULES}; got {list(init_modules)}"
            )
        load_head = "action_head" in init_modules
        ckpt_config = SimpleMemVLAConfig.from_pretrained(checkpoint)
        if load_head:
            for key in ("action_dim", "action_horizon", "state_dim", "use_proprio"):
                if getattr(ckpt_config, key) != getattr(config, key):
                    raise ValueError(
                        f"{key}={getattr(config, key)!r} does not match checkpoint "
                        f"{checkpoint} ({getattr(ckpt_config, key)!r})"
                    )
            for key in (
                "dit_hidden_size",
                "dit_depth",
                "dit_num_heads",
                "dit_mlp_ratio",
                "dit_rope_theta",
                "dit_mrope_section",
                "dit_partial_rotary_factor",
            ):
                setattr(config, key, getattr(ckpt_config, key))
        config.init_from_checkpoint = checkpoint
        config.init_modules = list(init_modules)
        config.backbone_config = dict(ckpt_config.backbone_config)
        backbone_config = AutoConfig.for_model(**config.backbone_config)
        from_config_kwargs = {}
        if attn_implementation:
            from_config_kwargs["attn_implementation"] = attn_implementation
        backbone = AutoModelForImageTextToText.from_config(
            backbone_config, **from_config_kwargs
        )
        for param in backbone.parameters():
            param.data = param.data.to(dtype)
        # Load the unmodified released backbone first. LoRA changes state_dict keys.
        model = cls(config=config, backbone=backbone, defer_lora_init=True)
        state_dict = {
            key: value
            for key, value in load_checkpoint_state_dict(checkpoint).items()
            if key.split(".", 1)[0] in init_modules
        }
        loaded = {key.split(".", 1)[0] for key in state_dict}
        if loaded != set(init_modules):
            raise RuntimeError(
                f"{checkpoint} has no weights for {sorted(set(init_modules) - loaded)}"
            )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        del state_dict
        if any(key.endswith("lm_head.weight") for key in missing):
            model.backbone.tie_weights()
            missing = [key for key in missing if not key.endswith("lm_head.weight")]
        bad_missing = [key for key in missing if key.split(".", 1)[0] in init_modules]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint weight mismatch loading {checkpoint}: "
                f"missing={bad_missing[:8]} unexpected={unexpected[:8]}"
            )
        if model.memory is not None:
            fresh = {f"memory.{name}" for name, _ in model.memory.named_parameters()}
            if not fresh.issubset(set(missing)):
                raise RuntimeError("checkpoint unexpectedly carries memory.* weights")
        model._init_lm_lora()
        model._apply_backbone_freeze(config.freeze_backbone)
        model.action_head.to(dtype=dtype)
        if model.memory is not None:
            model.memory.to(dtype=dtype)
        demoted = [
            name
            for name, buf in model.backbone.named_buffers()
            if "inv_freq" in name and buf.dtype != torch.float32
        ]
        if demoted:
            raise RuntimeError(
                "backbone rotary inv_freq buffers must stay fp32 to reproduce the "
                f"trained M-RoPE phases; got {demoted[:4]}"
            )
        return model

    def _init_lm_lora(self) -> None:
        if self.config.lm_lora_rank <= 0 or self._lm_lora_initialized:
            return
        self.lm_lora_targets = inject_lm_lora(
            self.backbone,
            rank=self.config.lm_lora_rank,
            alpha=self.config.lm_lora_alpha,
            dropout=self.config.lm_lora_dropout,
            target_modules=self.config.lm_lora_target_modules,
        )
        self._lm_lora_initialized = True

    def _apply_backbone_freeze(self, freeze: bool) -> None:
        for name, param in self.backbone.named_parameters():
            # Reapplying partial freeze must never reopen the base LM under LoRA.
            param.requires_grad = not freeze and (
                self.config.lm_lora_rank <= 0 or is_lora_parameter(name)
            )
        if self.config.freeze_vision_tower:
            for param in self.backbone.model.visual.parameters():
                param.requires_grad = False
        if isinstance(self.memory, PerceptionMemory):
            self.memory.scorer.requires_grad_(not self.config.perc_mem_freeze_scorer)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.backbone.set_input_embeddings(value)

    def _position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_kwargs: dict,
    ) -> torch.Tensor:
        position_ids, _ = self.backbone.model.get_rope_index(
            input_ids,
            image_grid_thw=mm_kwargs.get("image_grid_thw"),
            video_grid_thw=mm_kwargs.get("video_grid_thw"),
            attention_mask=attention_mask,
            mm_token_type_ids=mm_kwargs.get("mm_token_type_ids"),
        )
        return position_ids

    def _backbone_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mm_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        mm_kwargs["use_cache"] = False
        with torch.set_grad_enabled(
            self.training and any(p.requires_grad for p in self.backbone.parameters())
        ):
            outputs = self.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **mm_kwargs,
            )
            hidden_states = outputs.last_hidden_state
            vl_loss = None
            if labels is not None:
                vl_loss = self._sparse_ce_loss(hidden_states, labels)
        return hidden_states, vl_loss

    def _sparse_ce_loss(
        self, hidden_states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        lm_head = self.backbone.get_output_embeddings()
        if lm_head is None:
            lm_head = self.backbone.lm_head
        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:].to(shift_hidden.device)
        mask = shift_labels != -100
        if not bool(mask.any()):
            return (hidden_states.sum() + lm_head.weight.sum()) * 0.0
        selected = shift_hidden[mask]
        targets = shift_labels[mask]
        logits = lm_head(selected).float()
        return torch.nn.functional.cross_entropy(logits, targets, reduction="mean")

    def _gather_condition(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        span: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        span = span.to(device=hidden_states.device, dtype=torch.bool)
        lengths = span.sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() else 0
        if max_len == 0:
            raise ValueError("sub-task condition span is empty")

        batch_size, _, hdim = hidden_states.shape
        device = hidden_states.device
        embed = self.get_input_embeddings()
        cond_hidden = hidden_states.new_zeros(batch_size, max_len, hdim)
        cond_token = hidden_states.new_zeros(batch_size, max_len, hdim)
        cond_mask = torch.zeros(batch_size, max_len, dtype=attention_mask.dtype, device=device)
        cond_pos = position_ids.new_zeros(3, batch_size, max_len)
        for b in range(batch_size):
            idx = span[b].nonzero(as_tuple=True)[0]
            n = int(idx.numel())
            if n == 0:
                continue
            cond_hidden[b, :n] = hidden_states[b, idx]
            cond_token[b, :n] = embed(input_ids[b, idx])
            cond_mask[b, :n] = 1
            cond_pos[:, b, :n] = position_ids[:, b, idx]
        return cond_hidden, cond_token, cond_mask, cond_pos

    def _memory_inputs(
        self,
        mem_feats: torch.Tensor | None,
        mem_age: torch.Tensor | None,
        mem_mask: torch.Tensor | None,
        device: torch.device,
        mem_mode: str = "normal",
        mem_utility: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Return memory vector, prefix tokens and their valid-key mask."""
        return self._memory_inputs_with_aux(
            mem_feats, mem_age, mem_mask, device, mem_mode, mem_utility
        )[:3]

    def _memory_inputs_with_aux(
        self,
        mem_feats: torch.Tensor | None,
        mem_age: torch.Tensor | None,
        mem_mask: torch.Tensor | None,
        device: torch.device,
        mem_mode: str = "normal",
        mem_utility: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        """``_memory_inputs`` plus the perception-memory scorer rank loss (or ``None``)."""
        if mem_mode not in MEMORY_MODES:
            raise ValueError(f"mem_mode must be one of {MEMORY_MODES}, got {mem_mode!r}")
        if self.memory is None or mem_mode == "zero":
            return None, None, None, None
        if mem_feats is None or mem_age is None or mem_mask is None:
            raise ValueError("use_memory=True requires mem_feats, mem_age and mem_mask")
        mem_feats = mem_feats.to(device=device)
        mem_age = mem_age.to(device=device)
        mem_mask = mem_mask.to(device=device)
        if mem_mode == "shuffle":
            mem_feats, mem_age, mem_mask = (
                t.roll(1, dims=0) for t in (mem_feats, mem_age, mem_mask)
            )
            if mem_utility is not None:
                mem_utility = mem_utility.roll(1, dims=0)
        rank_loss = None
        if self.config.mem_arch == "perc_queue":
            out, rank_loss = self.memory(mem_feats, mem_age, mem_mask, utility=mem_utility)
        else:
            out = self.memory(mem_feats, mem_age, mem_mask)
        if isinstance(out, tuple):  # prefix tokens + their valid mask
            tokens, token_mask = out
            if self.training and self.config.mem_dropout_prob > 0.0:
                keep = torch.rand(tokens.shape[0], device=device) >= self.config.mem_dropout_prob
                token_mask = token_mask & keep[:, None]
                tokens = tokens.masked_fill(~token_mask[..., None], 0)
            return None, tokens, token_mask, rank_loss
        if self.training and self.config.mem_dropout_prob > 0.0:
            keep = torch.rand(out.shape[0], device=device) >= self.config.mem_dropout_prob
            out = out * keep.to(dtype=out.dtype).view(-1, *([1] * (out.ndim - 1)))
        if self.config.mem_fusion == "adaln_add":
            return out, None, None, rank_loss
        return None, out, None, rank_loss

    def _memory_condition(self, *args, **kwargs):
        # Compatibility for existing diagnostics of the original memory branch.
        memory_cond, memory_tokens, _ = self._memory_inputs(*args, **kwargs)
        return memory_cond, memory_tokens

    def _condition_span(
        self,
        labels: torch.Tensor | None,
        condition_span: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.condition_source == "prompt":
            if condition_span is None:
                raise ValueError("condition_source='prompt' requires condition_span")
            return condition_span
        if labels is None:
            raise ValueError(
                "labels (the supervised sub-task span) are required to train "
                "the action head"
            )
        return labels.ne(-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        condition_span: torch.Tensor | None = None,
        mem_feats: torch.Tensor | None = None,
        mem_age: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
        mem_utility: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        mem_mode: str = "normal",
        **kwargs,
    ) -> SimpleMemVLAOutput:
        if position_ids is None:
            position_ids = self._position_ids(input_ids, attention_mask, kwargs)
        hidden_states, vl_loss = self._backbone_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        action_loss = None
        pred_velocity = None
        total_loss = None
        mem_norm = None
        mem_rank_loss = None
        if actions is not None:
            actions = actions.to(device=hidden_states.device, dtype=hidden_states.dtype)
            batch_size = actions.shape[0]
            if noise is None:
                noise = torch.randn_like(actions)
            noise = noise.to(device=actions.device, dtype=actions.dtype)
            if timesteps is None:
                timesteps = sample_flow_timesteps(
                    batch_size,
                    device=actions.device,
                    dtype=actions.dtype,
                    beta_alpha=self.config.timestep_beta_alpha,
                    beta_beta=self.config.timestep_beta_beta,
                )
            timesteps = timesteps.to(device=actions.device, dtype=actions.dtype)
            t_view = timesteps.view(batch_size, 1, 1)
            noisy_actions = (1 - t_view) * actions + t_view * noise
            target_velocity = noise - actions
            if action_mask is None:
                raise ValueError("action_mask is required when training on actions")
            action_mask = action_mask.to(device=actions.device)
            span = self._condition_span(labels, condition_span)
            (
                memory_cond,
                memory_tokens,
                memory_attention_mask,
                mem_rank_loss,
            ) = self._memory_inputs_with_aux(
                mem_feats,
                mem_age,
                mem_mask,
                hidden_states.device,
                mem_mode,
                mem_utility=mem_utility,
            )
            mem_out = memory_cond if memory_cond is not None else memory_tokens
            if mem_out is not None:
                mem_norm = mem_out.detach().float().flatten(1).norm(dim=-1).mean()
            cond_hidden, cond_token, cond_mask, cond_pos = self._gather_condition(
                hidden_states=hidden_states,
                input_ids=input_ids,
                span=span,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            pred_velocity = self.action_head(
                condition_hidden_states=cond_hidden,
                condition_token_embeds=cond_token,
                noisy_actions=noisy_actions,
                timesteps=timesteps,
                condition_position_ids=cond_pos,
                condition_attention_mask=cond_mask,
                state=state,
                memory_cond=memory_cond,
                memory_tokens=memory_tokens,
                memory_attention_mask=memory_attention_mask,
            )
            action_loss = masked_flow_matching_loss(
                pred_velocity=pred_velocity,
                target_velocity=target_velocity,
                action_mask=action_mask,
            )
            total_loss = self.config.action_loss_weight * action_loss
            if mem_rank_loss is not None and self.config.perc_mem_scorer_loss_weight:
                total_loss = total_loss + self.config.perc_mem_scorer_loss_weight * mem_rank_loss

        if vl_loss is not None and self.config.vl_loss_weight:
            weighted_vl = self.config.vl_loss_weight * vl_loss
            total_loss = weighted_vl if total_loss is None else total_loss + weighted_vl

        return SimpleMemVLAOutput(
            loss=total_loss,
            action_loss=action_loss,
            vl_loss=vl_loss,
            pred_velocity=pred_velocity,
            mem_norm=mem_norm,
            mem_rank_loss=mem_rank_loss,
        )

    @torch.no_grad()
    def predict_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        condition_span: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        num_steps: int = 10,
        temperature: float = 1.0,
        state: torch.Tensor | None = None,
        mem_feats: torch.Tensor | None = None,
        mem_age: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
        mem_mode: str = "normal",
        **kwargs,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = self._position_ids(input_ids, attention_mask, kwargs)
        hidden_states, _ = self._backbone_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        cond_hidden, cond_token, cond_mask, cond_pos = self._gather_condition(
            hidden_states=hidden_states,
            input_ids=input_ids,
            span=condition_span,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        memory_cond, memory_tokens, memory_attention_mask = self._memory_inputs(
            mem_feats, mem_age, mem_mask, hidden_states.device, mem_mode
        )
        return self.action_head.sample(
            cond_hidden,
            condition_position_ids=cond_pos,
            condition_attention_mask=cond_mask,
            condition_token_embeds=cond_token,
            num_steps=num_steps,
            temperature=temperature,
            state=state,
            memory_cond=memory_cond,
            memory_tokens=memory_tokens,
            memory_attention_mask=memory_attention_mask,
        )

    @torch.no_grad()
    def generate_subtask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 256,
        eos_token_id: int | None = None,
        do_sample: bool = False,
        **gen_kwargs,
    ) -> torch.LongTensor:
        gen_inputs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )
        if eos_token_id is not None:
            gen_inputs["eos_token_id"] = eos_token_id
        gen_inputs.update(gen_kwargs)
        return self.backbone.generate(**gen_inputs)
