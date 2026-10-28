"""Generated sub-task conditioning with an independent online DiT memory.

The VLM keeps the released sub-task decoding path and sees only the current
front pair plus current wrist. Memory observes every actual environment frame;
its frozen FP32 visual tower matches offline training-cache extraction.
"""
from __future__ import annotations

import torch

from robomme_sim.batched_policy import BatchedEvalPolicy
from robomme_sim.policy import RoboMMEPolicy
from robomme_sim.memcache_policy import MemCacheBuffer, MemCacheBatchedPolicy
from simplememvla.data.collator import pad_memory


class SubtaskMemoryBuffer(RoboMMEPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.current_frame_only or self.vla.config.condition_source != "subtask":
            raise ValueError("SubtaskMemoryBuffer requires current-frame VLM and subtask conditioning")
        if not self.vla.config.use_memory:
            raise ValueError("SubtaskMemoryBuffer requires a memory checkpoint")
        self.memory_buffer = MemCacheBuffer(self.vla, self.processor)

    def reset(self):
        super().reset()
        self.memory_buffer.reset()

    def observe(self, frames):
        super().observe(frames)
        self.memory_buffer.observe(frames)
        key = self.memory_buffer.mem_camera_key
        if self._frames_seen[key] - 1 != self.memory_buffer.t:
            raise RuntimeError("VLM and memory frame clocks diverged")


class SubtaskMemoryBatchedPolicy(BatchedEvalPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.model.config.use_memory or self.model.config.condition_source != "subtask":
            raise ValueError("Subtask memory policy requires use_memory=True, condition_source=subtask")
        # Reuse the established FP32 online-cache encoder and time-window
        # selection. Its prompt-conditioned generate_batch is never invoked.
        self.cache_encoder = MemCacheBatchedPolicy(
            model=self.model, processor=self.processor,
            unnormalize_action=self.unnormalize, normalize_state=self.normalize_state,
            num_denoising_steps=self.num_denoising_steps,
            eval_temperature=self.eval_temperature, device=self.device,
        )
        self._memory_verified = False

    @torch.no_grad()
    def _action_memory_kwargs(self, processed_list):
        buffers = [p["policy"].memory_buffer for p in processed_list]
        for buffer in buffers:
            pending = list(buffer.pending)
            buffer.pending = []
            # Demo replay may enqueue many keyframes before the first action.
            # Bound FP32 ViT batches instead of encoding an entire demo at once.
            for start in range(0, len(pending), 32):
                buffer.pending = pending[start:start + 32]
                self.cache_encoder._encode_pending([buffer])
        features_age = [self.cache_encoder._memory(b) for b in buffers]
        mem = pad_memory([f for f, _ in features_age], [a.cpu() for _, a in features_age])
        mem = {key: value.to(self.device) for key, value in mem.items()}
        memory_cond, memory_tokens, memory_mask = self.model._memory_inputs(
            mem["mem_feats"], mem["mem_age"], mem["mem_mask"], self.device, "normal"
        )
        if not self._memory_verified:
            valid_ages = mem["mem_age"][mem["mem_mask"].bool()]
            if valid_ages.numel() and float(valid_ages.min()) < 0:
                raise RuntimeError("Future memory frame entered action conditioning")
            print(
                f"[subtask-memory] generated sub-task; VLM=current-frame only; "
                f"memory_tokens={tuple(memory_tokens.shape)}; "
                f"valid_tokens={memory_mask.sum(-1).tolist()}; "
                f"max_age_sec={float(valid_ages.max()) if valid_ages.numel() else 0.0}; "
                f"memory_dtype={memory_tokens.dtype}", flush=True,
            )
            self._memory_verified = True
        return dict(memory_cond=memory_cond, memory_tokens=memory_tokens,
                    memory_attention_mask=memory_mask)
