"""Closed-loop buffer for single-frame image-channel checkpoints with sub-task conditioning.

``single_frame=True`` + ``condition_source='subtask'`` (configs/train/singleframe_img_subtask_*):
every camera's *current* frame goes through the VLM image channel (no video channel), the
sub-task is generated and the DiT is conditioned on it, as in training. Decisions are batched
by ``BatchedEvalPolicy`` (which skips the video-patch cache when ``video_patch_keys`` is empty).
"""

from __future__ import annotations

import numpy as np

from robomme_sim.policy import RoboMMEPolicy
from simplememvla.benchmarks.robomme import CAMERA_LABELS
from simplememvla.data.messages import build_simplememvla_messages


class SingleFrameSubtaskBuffer(RoboMMEPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["current_frame_only"] = True  # keep only the latest frame per camera
        super().__init__(*args, **kwargs)
        if not self.vla.config.single_frame or self.vla.config.condition_source != "subtask":
            raise ValueError("SingleFrameSubtaskBuffer requires single_frame=True and condition_source='subtask'")
        if getattr(self.vla.config, "use_memory", False):
            raise ValueError("SingleFrameSubtaskBuffer does not support memory checkpoints")
        self.history_image_keys = []
        self.history_camera_tags = []

    def _prepare_inputs(self, prompt: str) -> dict:
        messages = build_simplememvla_messages(
            camera_tags=self.camera_tags,
            instruction=prompt,
            robot_tag=self.robot_tag,
            control_frequency_hz=self.control_frequency_hz,
            action_horizon=self.action_horizon,
            camera_labels=CAMERA_LABELS,
            subtask=None,
            history_camera_tags=[],
        )
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if not isinstance(text, str):
            text = text[0]
        images = []
        for key in self.image_keys:
            if not self._buffers[key]:
                raise RuntimeError("call observe() before predict()")
            images.append(np.asarray(self._buffers[key][-1]))
        enc = self.processor(text=[text], images=images, return_tensors="pt")
        return {
            "policy": self,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "mm_token_type_ids": enc["mm_token_type_ids"],
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],
            "video_patch_keys": [],
            "video_patch_pixels": {},
        }
