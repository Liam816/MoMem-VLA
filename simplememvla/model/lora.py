"""LoRA injection scoped to the Qwen language decoder.

The outer SimpleMemVLA remains a normal PreTrainedModel. Its complete state dict
must be saved, including the DiT, memory and LoRA parameters. When restoring such
a checkpoint, construct the same adapter structure before loading that state dict.
"""

import math
from collections.abc import Sequence

from torch import nn


DEFAULT_LM_LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
)


def is_lora_parameter(name: str) -> bool:
    """Whether a parameter belongs to the standard Linear LoRA A/B matrices."""
    return any(part in {"lora_A", "lora_B"} for part in name.split("."))


def inject_lm_lora(
    backbone: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Sequence[str] = DEFAULT_LM_LORA_TARGET_MODULES,
) -> list[str]:
    """Inject standard LoRA only into ``backbone.model.language_model``.

    ``target_modules`` contains module names or suffixes, not regex patterns.
    Every suffix must match at least one Linear. The returned names are relative
    to ``backbone`` (e.g. ``model.language_model.layers.0.linear_attn.out_proj``).
    Decoder base parameters, including token embeddings, are frozen; only LoRA
    A/B matrices are trainable inside the decoder. This function does not change
    the trainability of the visual encoder or an untied external LM head.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be finite and positive")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0,1)")
    if isinstance(target_modules, str) or not target_modules:
        raise ValueError("target_modules must be a nonempty sequence of module names")
    targets = tuple(dict.fromkeys(target_modules))
    if not all(isinstance(name, str) and name for name in targets):
        raise ValueError("target_modules entries must be nonempty strings")
    outer_model = getattr(backbone, "model", None)
    language_model = getattr(outer_model, "language_model", None)
    if not isinstance(language_model, nn.Module):
        raise ValueError("backbone.model.language_model must be a torch module")
    if getattr(language_model, "peft_config", None) or any(
        is_lora_parameter(name) for name, _ in language_model.named_parameters()
    ):
        raise ValueError("language_model already contains adapters; refusing duplicate injection")

    matches: list[str] = []
    hits = set()
    protected = {"visual", "vision_tower", "embed_tokens", "lm_head"}
    for name, module in language_model.named_modules():
        matched = [target for target in targets if name == target or name.endswith("." + target)]
        if not matched:
            continue
        if protected.intersection(name.split(".")):
            raise ValueError(f"LoRA target is outside decoder projection layers: {name}")
        if not isinstance(module, nn.Linear):
            raise ValueError(f"LoRA target must be nn.Linear: {name} ({type(module).__name__})")
        matches.append(name)
        hits.update(matched)
    missing = set(targets) - hits
    if missing:
        raise ValueError(f"LoRA target modules were not found in language_model: {sorted(missing)}")

    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:
        raise ImportError("Language decoder LoRA requires the PEFT dependency") from exc

    adapter_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=matches,
        bias="none",
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
    )
    outer_model.language_model = inject_adapter_in_model(adapter_config, language_model)
    # Enforce this invariant even if PEFT's defaults change. The caller must not
    # subsequently reopen every backbone parameter with requires_grad_(True).
    for name, parameter in outer_model.language_model.named_parameters():
        parameter.requires_grad_(is_lora_parameter(name))
    return ["model.language_model." + name for name in matches]
