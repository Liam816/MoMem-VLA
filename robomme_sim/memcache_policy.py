"""Closed-loop RoboMME policy for the single-frame + memory-cache checkpoints.

These checkpoints (configs/train/memcache/*) differ from the released recipe:
the frozen VLM sees only the *current* frame of every camera, the DiT is
conditioned on the whole prompt (``condition_source='prompt'``, no sub-task
decoding), and -- with ``use_memory`` -- a cache of per-keyframe ViT tokens is
fed to the memory branch. Everything here mirrors the training data path:

* prompt   : ``build_single_frame_messages`` + ``add_generation_prompt=True``;
* cache    : the front camera every ``stride`` frames, counted from the first
             frame of the episode (RoboMME's replayed demo included), encoded by
             the same vision tower in fp32 as ``scripts/extract_mem_features.py``,
             stored as fp16, pooled with ``pool_mem_tokens``;
* window   : ``select_mem_keyframes`` with the checkpoint's window / fps.

``eval_success.build_policy`` dispatches here when the checkpoint config has
``condition_source == 'prompt'``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from robomme_sim._compat import apply_fla_torch_compat

apply_fla_torch_compat()

from simplememvla.benchmarks.robomme import CAMERA_LABELS
from simplememvla.data.collator import pad_memory
from simplememvla.data.dataset import (
    mem_keyframe_stride,
    pool_mem_tokens,
    select_mem_keyframes,
)
from simplememvla.data.messages import build_single_frame_messages


def is_memcache_checkpoint(config) -> bool:
    return getattr(config, "condition_source", "subtask") == "prompt"


def build_memory_vision_tower(config, device: torch.device):
    """fp32 copy of the vision tower that produced the training cache."""
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from extract_mem_features import build_visual

    source = getattr(config, "init_from_checkpoint", None) or config.backbone_model_name_or_path
    if not os.path.isdir(source):
        raise FileNotFoundError(
            f"memory vision tower source {source!r} (from the checkpoint config) is not a local "
            "directory; it must be the VLM the cache was extracted with."
        )
    visual, backbone_config = build_visual(source, "sdpa", device, dtype=torch.float32)
    return visual, backbone_config.vision_config.spatial_merge_size, source


class MemCacheBuffer:
    """Per-environment state: latest frame of every camera + the keyframe cache."""

    def __init__(self, vla, processor):
        cfg = vla.config
        if cfg.control_frequency_hz is None:
            raise ValueError("checkpoint config is missing control_frequency_hz")
        self.processor = processor
        self.image_keys = list(cfg.image_keys)
        self.camera_tags = [k.split(".")[-1] for k in self.image_keys]
        self.robot_tag = cfg.robot_tag
        self.control_frequency_hz = cfg.control_frequency_hz
        self.native_fps = float(cfg.native_video_fps or cfg.control_frequency_hz)
        self.action_horizon = int(cfg.action_horizon)
        self.use_memory = bool(cfg.use_memory)
        if self.use_memory:
            self.mem_camera_key = cfg.mem_camera_key
            if self.mem_camera_key not in self.image_keys:
                raise ValueError(f"mem_camera_key {self.mem_camera_key!r} not among {self.image_keys}")
            self.stride = mem_keyframe_stride(self.native_fps, cfg.mem_fps)
            self.window_frames = round(cfg.mem_window_sec * self.native_fps)
            self.frame_pool = int(cfg.mem_frame_pool)
        self.reset()

    def reset(self) -> None:
        self.t = -1
        self.current: dict[str, np.ndarray] = {}
        self.keyframes: dict[int, torch.Tensor] = {}   # k -> (tokens, D) fp16, frame k*stride
        self.pending: list[tuple[int, np.ndarray]] = []

    def observe(self, frames: dict[str, np.ndarray]) -> None:
        missing = [k for k in self.image_keys if k not in frames]
        if missing:
            raise KeyError(f"observe() missing frames for cameras: {missing}")
        self.t += 1
        self.current = {k: np.asarray(frames[k]) for k in self.image_keys}
        if self.use_memory and self.t % self.stride == 0:
            self.pending.append((self.t // self.stride, self.current[self.mem_camera_key]))

    def _prepare_inputs(self, instruction: str) -> dict:
        if self.t < 0:
            raise RuntimeError("call observe() before predicting")
        messages = build_single_frame_messages(
            camera_tags=self.camera_tags,
            instruction=instruction,
            robot_tag=self.robot_tag,
            control_frequency_hz=self.control_frequency_hz,
            action_horizon=self.action_horizon,
            camera_labels=CAMERA_LABELS,
        )
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if not isinstance(text, str):
            text = text[0]
        enc = self.processor(
            text=[text],
            images=[self.current[k] for k in self.image_keys],
            do_sample_frames=False,
            return_tensors="pt",
        )
        return {"buffer": self, "t": self.t, "enc": enc}


class MemCacheBatchedPolicy:
    """Batched decisions for a group of ``MemCacheBuffer`` envs."""

    def __init__(
        self,
        model,
        processor,
        unnormalize_action,
        normalize_state,
        num_denoising_steps: int = 10,
        eval_temperature: float = 1.0,
        device: torch.device | None = None,
    ):
        self.model = model
        self.processor = processor
        self.unnormalize = unnormalize_action
        self.normalize_state = normalize_state
        self.num_denoising_steps = int(num_denoising_steps)
        self.eval_temperature = float(eval_temperature)
        self.device = device or next(model.parameters()).device
        cfg = model.config
        self.use_proprio = bool(cfg.use_proprio)
        self.action_dim = int(cfg.action_dim)
        self.use_memory = bool(cfg.use_memory)
        tok = processor.tokenizer
        self.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|im_end|>")
        self.visual = None
        if self.use_memory:
            self.visual, self.merge, source = build_memory_vision_tower(cfg, self.device)
            print(f"[memcache] memory cache vision tower: {source} (fp32, sdpa)", flush=True)

    @torch.no_grad()
    def _encode_pending(self, buffers: list[MemCacheBuffer]) -> None:
        jobs = [(b, k, img) for b in buffers for k, img in b.pending]
        if not jobs:
            return
        enc = self.processor.image_processor(images=[img for _, _, img in jobs], return_tensors="pt")
        thw = enc["image_grid_thw"]
        if not bool((thw == thw[0]).all()):
            raise RuntimeError(f"cache frames produced different grids: {thw.tolist()}")
        feats = self.visual(
            enc["pixel_values"].to(self.device, torch.float32), grid_thw=thw.to(self.device), return_dict=True
        ).pooler_output
        per = int(thw[0].prod()) // self.merge**2
        grid = (int(thw[0, 1]) // self.merge, int(thw[0, 2]) // self.merge)
        # same storage dtype as the offline cache (np.float16)
        feats = feats.view(len(jobs), per, -1).to(torch.float16)
        for (b, k, _), f in zip(jobs, feats):
            b.keyframes[k] = f
            b.grid = grid
        for b in buffers:
            b.pending.clear()

    def _memory(self, b: MemCacheBuffer) -> tuple[torch.Tensor, torch.Tensor]:
        stored = max(b.keyframes) + 1
        k_min, k_max = select_mem_keyframes(b.t, b.stride, b.window_frames, stored)
        missing = [k for k in range(k_min, k_max + 1) if k not in b.keyframes]
        if missing:
            raise RuntimeError(f"keyframes {missing[:4]} were never observed (t={b.t})")
        for k in [k for k in b.keyframes if k < k_min]:
            del b.keyframes[k]
        feats = torch.stack([b.keyframes[k] for k in range(k_min, k_max + 1)])
        feats = pool_mem_tokens(feats, b.grid, b.frame_pool)
        keyframes = torch.arange(k_min, k_max + 1, dtype=torch.float32, device=feats.device) * b.stride
        age = (b.t - keyframes) / b.native_fps
        return feats, age

    @torch.no_grad()
    def generate_batch(self, processed_list: list[dict], state_norm_list: list) -> list[tuple[np.ndarray, str]]:
        self.model.eval()
        dev = self.device
        encs = [p["enc"] for p in processed_list]
        lengths = [int(e["input_ids"].shape[1]) for e in encs]
        n, L = len(encs), max(lengths)
        input_ids = torch.full((n, L), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((n, L), dtype=torch.long)
        mm_token_type_ids = torch.zeros((n, L), dtype=torch.long)
        for i, e in enumerate(encs):
            input_ids[i, : lengths[i]] = e["input_ids"][0]
            attention_mask[i, : lengths[i]] = e["attention_mask"][0]
            mm_token_type_ids[i, : lengths[i]] = e["mm_token_type_ids"][0]
        batch = {
            "input_ids": input_ids.to(dev),
            "attention_mask": attention_mask.to(dev),
            "mm_token_type_ids": mm_token_type_ids.to(dev),
            "pixel_values": torch.cat([e["pixel_values"] for e in encs]).to(dev),
            "image_grid_thw": torch.cat([e["image_grid_thw"] for e in encs]).to(dev),
        }
        mem = {}
        if self.use_memory:
            buffers = [p["buffer"] for p in processed_list]
            self._encode_pending(buffers)
            fa = [self._memory(b) for b in buffers]
            mem = pad_memory([f for f, _ in fa], [a.cpu() for _, a in fa])
            mem = {k: v.to(dev) for k, v in mem.items()}
        state = None
        if self.use_proprio:
            state = torch.stack([s[0] if s.dim() == 2 else s for s in state_norm_list]).to(dev)
        actions = self.model.predict_action(
            batch["input_ids"],
            batch["attention_mask"],
            condition_span=batch["attention_mask"].bool(),
            num_steps=self.num_denoising_steps,
            temperature=self.eval_temperature,
            state=state,
            mm_token_type_ids=batch["mm_token_type_ids"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            **mem,
        )
        out = []
        for i in range(n):
            a = actions[i, :, : self.action_dim].float().cpu()
            out.append((self.unnormalize.unnormalize(a).numpy().astype(np.float32), ""))
        return out
