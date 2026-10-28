"""Extract per-keyframe ViT (merger) vision tokens for the history memory cache.

For every episode, frames ``k * stride`` (``stride = native_fps / mem_fps``) of
one camera are encoded as single images by the frozen backbone's vision tower
-- the same tokens the LLM receives for that frame -- and saved as
``<out>/episode_XXXXXX.npy`` with shape ``(num_keyframes, tokens, feature_dim)``.

    python scripts/extract_mem_features.py --checkpoint CKPT --benchmark robomme \
        --root DATA --episodes_file train.txt val.txt --out FEAT_DIR \
        --shard 0 --num_shards 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simplememvla.compat import apply_fla_torch_compat

apply_fla_torch_compat()

import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from simplememvla.benchmarks import get_benchmark
from simplememvla.data.dataset import (
    MEM_META_FILE,
    _frames_to_video,
    mem_episode_path,
    mem_keyframe_stride,
)
from simplememvla.model.configuration_simplememvla import SimpleMemVLAConfig

VISUAL_PREFIX = "backbone.model.visual."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--checkpoint", required=True,
        help="SimpleMemVLA checkpoint dir, or a plain HF backbone dir (e.g. Qwen3.5-4B) -- must be the VLM you train with",
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--root", default=None, help="LeRobot dataset root (default: spec)")
    parser.add_argument("--camera_key", default=None, help="default: the spec's history camera")
    parser.add_argument("--mem_fps", type=float, default=2.0)
    parser.add_argument("--episodes_file", nargs="*", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--attn_implementation", default="flash_attention_2",
                        help="used for bfloat16; float32 always runs sdpa (flash-attn has no fp32)")
    parser.add_argument(
        "--vit_dtype", choices=["float32", "bfloat16"], default="float32",
        help="Precision of the vision tower. bf16 features depend on the batch shape (the "
             "original Qwen3.5-4B ViT has tokens whose bf16 output moves by cos~0.7 between "
             "batch 1 and 64); fp32 is batch-invariant, so online rollout features match.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safetensors_files(checkpoint: str) -> list[str]:
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(checkpoint, shard) for shard in shards]
    return [os.path.join(checkpoint, "model.safetensors")]


def build_visual(
    checkpoint: str,
    attn_implementation: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
):
    """Vision tower of either a SimpleMemVLA checkpoint or a plain HF backbone (e.g. Qwen3.5-4B)."""
    with open(os.path.join(checkpoint, "config.json")) as f:
        is_simplememvla = json.load(f).get("model_type") == SimpleMemVLAConfig.model_type
    if is_simplememvla:
        config = SimpleMemVLAConfig.from_pretrained(checkpoint)
        backbone_config = AutoConfig.for_model(**dict(config.backbone_config))
        prefix = VISUAL_PREFIX
    else:
        backbone_config = AutoConfig.from_pretrained(checkpoint)
        prefix = VISUAL_PREFIX[len("backbone.") :]
    with torch.device("meta"):
        meta_backbone = AutoModelForImageTextToText.from_config(backbone_config)
    visual_cls = type(meta_backbone.model.visual)
    del meta_backbone
    visual = visual_cls._from_config(
        backbone_config.vision_config, attn_implementation=attn_implementation
    )
    state_dict = {}
    for path in _safetensors_files(checkpoint):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    state_dict[key[len(prefix) :]] = f.get_tensor(key)
    if not state_dict:
        raise RuntimeError(f"no '{prefix}*' weights found in {checkpoint}")
    visual.load_state_dict(state_dict, strict=True)
    # Cast parameters only: rotary inv_freq buffers stay fp32, as in training.
    for param in visual.parameters():
        param.data = param.data.to(dtype)
    return visual.eval().to(device), backbone_config


class KeyframeDataset(Dataset):

    def __init__(self, dataset: LeRobotDataset, camera_key: str, jobs: list[tuple[int, int, int]]):
        self.dataset = dataset
        self.camera_key = camera_key
        self.jobs = jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        episode_index, k, global_index = self.jobs[i]
        item = self.dataset[global_index]
        got = (int(item["episode_index"]), int(item["frame_index"]))
        if got[0] != episode_index:
            raise RuntimeError(f"index {global_index} is episode {got[0]}, expected {episode_index}")
        return episode_index, k, got[1], _frames_to_video(item[self.camera_key])[0]


def main() -> None:
    args = parse_args()
    spec = get_benchmark(args.benchmark)
    root = args.root or spec.root
    camera_key = args.camera_key or spec.history_image_keys[0]
    native_fps = float(spec.native_video_fps)
    stride = mem_keyframe_stride(native_fps, args.mem_fps)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(spec.repo_id, root=root, video_backend="pyav")
    episodes_meta = dataset.meta.episodes
    if args.episodes_file:
        episodes = sorted(
            {int(tok) for path in args.episodes_file for tok in Path(path).read_text().split()}
        )
    else:
        episodes = list(range(dataset.meta.total_episodes))
    episodes = episodes[args.shard :: args.num_shards]
    if not args.overwrite:
        episodes = [ep for ep in episodes if not mem_episode_path(out_dir, ep).is_file()]

    jobs, num_keys = [], {}
    for ep in episodes:
        row = episodes_meta[ep]
        start, length = int(row["dataset_from_index"]), int(row["length"])
        n = (length - 1) // stride + 1
        num_keys[ep] = n
        jobs += [(ep, k, start + k * stride) for k in range(n)]
    print(
        f"[shard {args.shard}/{args.num_shards}] {len(episodes)} episodes, "
        f"{len(jobs)} keyframes, camera={camera_key}, stride={stride}",
        flush=True,
    )
    if not jobs:
        return

    device = torch.device("cuda")
    vit_dtype = getattr(torch, args.vit_dtype)
    attn = args.attn_implementation if vit_dtype == torch.bfloat16 else "sdpa"
    visual, backbone_config = build_visual(args.checkpoint, attn, device, dtype=vit_dtype)
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    merge = backbone_config.vision_config.spatial_merge_size

    loader = DataLoader(
        KeyframeDataset(dataset, camera_key, jobs),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch,
    )
    buffers: dict[int, list] = {}
    grid, feature_dim, t0, done = None, None, time.time(), 0
    for batch in loader:
        images = [frame for _, _, _, frame in batch]
        encoded = processor.image_processor(images=images, return_tensors="pt")
        thw = encoded["image_grid_thw"]
        with torch.no_grad():
            feats = visual(
                encoded["pixel_values"].to(device, dtype=vit_dtype),
                grid_thw=thw.to(device),
                return_dict=True,
            ).pooler_output
        per_image = int(thw[0].prod()) // merge**2
        if not bool((thw == thw[0]).all()) or feats.shape[0] != per_image * len(batch):
            raise RuntimeError(f"images in a batch produced different grids: {thw.tolist()}")
        if grid is None:
            grid = [int(thw[0, 1]) // merge, int(thw[0, 2]) // merge]
            feature_dim = int(feats.shape[-1])
        feats = feats.view(len(batch), per_image, feature_dim).to(torch.float16).cpu().numpy()
        for (ep, k, frame_index, _), f in zip(batch, feats):
            if frame_index != k * stride:
                raise RuntimeError(f"episode {ep} keyframe {k} landed on frame {frame_index}")
            buffers.setdefault(ep, []).append((k, f))
            if len(buffers[ep]) == num_keys[ep]:
                stack = np.stack([f for _, f in sorted(buffers.pop(ep), key=lambda x: x[0])])
                path = mem_episode_path(out_dir, ep)
                tmp = path.with_suffix(".tmp.npy")
                np.save(tmp, stack)
                os.replace(tmp, path)
        done += len(batch)
        if done % (args.batch_size * 20) < args.batch_size:
            rate = done / (time.time() - t0)
            print(f"[shard {args.shard}] {done}/{len(jobs)} keyframes, {rate:.1f}/s", flush=True)
    if buffers:
        raise RuntimeError(f"incomplete episodes left in buffer: {sorted(buffers)}")

    meta = {
        "feature_type": "vit_merger",
        "camera_key": camera_key,
        "stride": stride,
        "native_fps": native_fps,
        "mem_fps": args.mem_fps,
        "grid": grid,
        "tokens_per_frame": grid[0] * grid[1],
        "feature_dim": feature_dim,
        "dtype": "float16",
        "vit_dtype": args.vit_dtype,
        "checkpoint": os.path.abspath(args.checkpoint),
        "root": os.path.abspath(root),
    }
    meta_path = out_dir / MEM_META_FILE
    if meta_path.is_file():
        existing = json.loads(meta_path.read_text())
        if existing != meta:
            raise RuntimeError(f"{meta_path} differs from this run: {existing} vs {meta}")
    else:
        tmp = meta_path.with_suffix(f".shard{args.shard}.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, meta_path)
    print(f"[shard {args.shard}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
