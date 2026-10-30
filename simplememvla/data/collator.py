from typing import Any

import torch


def subtask_span_mask(
    input_ids: torch.Tensor,
    think_close_id: int | None,
    whitespace_ids: tuple[int, ...] = (),
) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if think_close_id is None:
        return mask
    positions = (input_ids == think_close_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return mask
    start = int(positions[-1]) + 1
    ws = set(whitespace_ids)
    while start < input_ids.shape[0] and int(input_ids[start]) in ws:
        start += 1
    mask[start:] = True
    return mask


def answer_span_token_ids(tokenizer) -> tuple[int | None, tuple[int, ...]]:
    close = tokenizer.convert_tokens_to_ids("</think>")
    think_close_id = close if isinstance(close, int) and close >= 0 else None
    ws: list[int] = []
    for text in ("\n", "\n\n"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            ws.append(int(ids[0]))
    return think_close_id, tuple(dict.fromkeys(ws))


def pad_memory(
    feats: list[torch.Tensor],
    ages: list[torch.Tensor],
    utilities: list[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pad per-sample keyframe stacks ``(K_i, N, D)`` to ``(B, K_max, N, D)``.

    Optional perception-memory utility matrices ``(K_i, K_i)`` are padded the same way.
    """
    batch_size = len(feats)
    k_max = max(f.shape[0] for f in feats)
    _, num_tokens, dim = feats[0].shape
    mem_feats = feats[0].new_zeros(batch_size, k_max, num_tokens, dim)
    mem_age = torch.zeros(batch_size, k_max, dtype=torch.float32)
    mem_mask = torch.zeros(batch_size, k_max, dtype=torch.bool)
    for b, (f, a) in enumerate(zip(feats, ages)):
        k = f.shape[0]
        mem_feats[b, :k] = f
        mem_age[b, :k] = a
        mem_mask[b, :k] = True
    out = {"mem_feats": mem_feats, "mem_age": mem_age, "mem_mask": mem_mask}
    if utilities is not None:
        out["mem_utility"] = torch.zeros(batch_size, k_max, k_max, dtype=torch.float32)
        for b, u in enumerate(utilities):
            out["mem_utility"][b, : u.shape[0], : u.shape[1]] = u
    return out


class SimpleMemVLADataCollator:

    def __init__(
        self,
        processor,
        max_length: int | None = None,
        condition_source: str = "subtask",
    ):
        self.processor = processor
        self.max_length = max_length
        self.condition_source = condition_source
        self.tokenizer = processor.tokenizer
        self.think_close_id, self.whitespace_ids = answer_span_token_ids(self.tokenizer)

    def _subtask_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = subtask_span_mask(input_ids, self.think_close_id, self.whitespace_ids)
        if not bool(mask.any()):
            raise ValueError(
                "No sub-task answer span found (missing '</think>' marker); every "
                "training sample must carry a sub-task answer."
            )
        labels = torch.full_like(input_ids, -100)
        labels[mask] = input_ids[mask]
        return labels

    def _encode_one(self, feature: dict[str, Any]) -> dict[str, torch.Tensor]:
        prompt_condition = self.condition_source == "prompt"
        text = self.processor.apply_chat_template(
            feature["messages"],
            tokenize=False,
            add_generation_prompt=prompt_condition,
            enable_thinking=False,
        )
        if not isinstance(text, str):
            text = text[0]
        images = feature.get("images") or None
        videos = feature.get("videos") or None
        encoded = self.processor(
            text=[text],
            images=images,
            videos=videos,
            video_metadata=feature["video_metadata"] if videos is not None else None,
            do_sample_frames=False,
            return_tensors="pt",
        )
        out: dict[str, torch.Tensor] = {}
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            out[key] = encoded[key].squeeze(0)
        if videos is not None:
            out["pixel_values_videos"] = encoded["pixel_values_videos"]
            out["video_grid_thw"] = encoded["video_grid_thw"]
        if images is not None:
            out["pixel_values"] = encoded["pixel_values"]
            out["image_grid_thw"] = encoded["image_grid_thw"]
        if prompt_condition:
            out["condition_span"] = out["attention_mask"].to(torch.bool)
        else:
            out["labels"] = self._subtask_labels(out["input_ids"])
        return out

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode_one(feature) for feature in features]
        pad_id = self.tokenizer.pad_token_id

        def pad(key: str, value: float):
            return torch.nn.utils.rnn.pad_sequence(
                [item[key] for item in encoded],
                batch_first=True,
                padding_value=value,
            )

        input_ids = pad("input_ids", pad_id)
        if self.max_length is not None and input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"Encoded multimodal sequence length {input_ids.shape[1]} exceeds "
                f"model_max_length={self.max_length}."
            )

        batch = {
            "input_ids": input_ids,
            "attention_mask": pad("attention_mask", 0),
            "mm_token_type_ids": pad("mm_token_type_ids", 0),
            "actions": torch.stack([feature["actions"] for feature in features]),
            "action_mask": torch.stack([feature["action_mask"] for feature in features]),
            "state": torch.stack([feature["state"] for feature in features]),
        }
        if "labels" in encoded[0]:
            batch["labels"] = pad("labels", -100)
        if "condition_span" in encoded[0]:
            batch["condition_span"] = pad("condition_span", False)
        if "pixel_values_videos" in encoded[0]:
            batch["pixel_values_videos"] = torch.cat(
                [item["pixel_values_videos"] for item in encoded], dim=0
            )
            batch["video_grid_thw"] = torch.cat(
                [item["video_grid_thw"] for item in encoded], dim=0
            )
        if "mem_feats" in features[0]:
            batch.update(
                pad_memory(
                    [feature["mem_feats"] for feature in features],
                    [feature["mem_age"] for feature in features],
                    utilities=(
                        [feature["mem_utility"] for feature in features]
                        if "mem_utility" in features[0]
                        else None
                    ),
                )
            )
        if "pixel_values" in encoded[0]:
            batch["pixel_values"] = torch.cat(
                [item["pixel_values"] for item in encoded], dim=0
            )
            batch["image_grid_thw"] = torch.cat(
                [item["image_grid_thw"] for item in encoded], dim=0
            )
        return batch
