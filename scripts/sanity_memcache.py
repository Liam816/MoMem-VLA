"""Sanity checks for the single-frame + memory-cache path (one GPU).

1. dataset/collator produce the prompt-conditioned single-frame batch;
2. warm start from a released checkpoint leaves only memory.* fresh;
3. offline memory features == the backbone's own image features of that frame;
4. zero-initialised fusion => memory on/off give the identical loss;
5. gradients reach only the action head and memory.

    python scripts/sanity_memcache.py --checkpoint CKPT --root DATA \
        --mem_feature_dir FEAT --episodes 700 1500 [--fusion adaln_add]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simplememvla.compat import apply_fla_torch_compat

apply_fla_torch_compat()

import json

import numpy as np
import torch
from transformers import AutoProcessor

from configs.sft_params import DataArguments
from simplememvla.data.collator import SimpleMemVLADataCollator
from simplememvla.data.dataset import SimpleMemVLADataset, mem_episode_path
from simplememvla.model import SimpleMemVLAConfig, SimpleMemVLAForActionPrediction
from simplememvla.model.modeling_simplememvla import sample_flow_timesteps
from train import resolve_benchmark_defaults


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--mem_feature_dir", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--fusion", default="adaln_add")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda")
    pool = 4

    # 1. data
    da = DataArguments(
        benchmark="robomme",
        root=args.root,
        episodes=args.episodes,
        single_frame=True,
        image_aug=False,
        mem_feature_dir=args.mem_feature_dir,
        stats_path=f"{args.checkpoint}/stats.json",
    )
    spec = resolve_benchmark_defaults(da)
    ds = SimpleMemVLADataset(da, spec, condition_source="prompt", mem_frame_pool=pool)
    processor = AutoProcessor.from_pretrained(args.checkpoint, padding_side="right")
    collator = SimpleMemVLADataCollator(processor, max_length=8192, condition_source="prompt")
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), args.batch_size, replace=False).tolist()
    samples = [ds[i] for i in idxs]
    batch = collator(samples)
    check(len(samples[0]["videos"]) == 0 and len(samples[0]["images"]) == 2, "single frame: 0 videos, 2 images")
    check("labels" not in batch and "pixel_values_videos" not in batch, "no labels / no video tensors")
    check(bool(batch["condition_span"].eq(batch["attention_mask"].bool()).all()), "condition_span == attention_mask")
    print("   seq len", batch["input_ids"].shape[1], "mem_feats", tuple(batch["mem_feats"].shape),
          "K per sample", batch["mem_mask"].sum(1).tolist(), "max age", float(batch["mem_age"].max()))
    check(batch["mem_feats"].shape[2:] == (pool * pool, 2560), "memory tokens pooled to 4x4, 2560-d")
    check(bool((batch["mem_age"][batch["mem_mask"]] >= 0).all()), "memory ages are non-negative")
    text = processor.tokenizer.decode(batch["input_ids"][0][batch["attention_mask"][0].bool()])
    print("   prompt tail:", repr(text[-260:]))

    # 2. warm start
    ckpt_cfg = SimpleMemVLAConfig.from_pretrained(args.checkpoint)
    meta = json.loads((Path(args.mem_feature_dir) / "meta.json").read_text())
    config = SimpleMemVLAConfig(
        backbone_model_name_or_path=args.checkpoint,
        action_dim=ckpt_cfg.action_dim,
        action_horizon=ckpt_cfg.action_horizon,
        state_dim=ckpt_cfg.state_dim,
        use_proprio=ckpt_cfg.use_proprio,
        freeze_backbone=True,
        vl_loss_weight=0.0,
        condition_source="prompt",
        single_frame=True,
        use_memory=True,
        mem_feature_dim=meta["feature_dim"],
        mem_frame_pool=pool,
        mem_fusion=args.fusion,
    )
    model = SimpleMemVLAForActionPrediction.from_simplememvla_checkpoint(
        args.checkpoint, config, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    trainable = {
        k: sum(p.numel() for n, p in model.named_parameters() if n.startswith(k) and p.requires_grad)
        for k in ("backbone", "action_head", "memory")
    }
    print("   trainable params (M):", {k: round(v / 1e6, 1) for k, v in trainable.items()})
    check(trainable["backbone"] == 0 and trainable["memory"] > 0, "backbone frozen, memory trainable")

    # 3. offline feature == backbone image feature of the same frame
    for i, s in zip(idxs, samples):
        item = ds.dataset[i if ds._anchor_indices is None else int(ds._anchor_indices[i])]
        ep, t = int(item["episode_index"]), int(item["frame_index"])
        if t % ds.mem_stride:
            continue
        one = collator([s])
        n_img = one["image_grid_thw"][0].prod().item()
        pv = one["pixel_values"][:n_img].to(device)
        with torch.no_grad():
            live = model.backbone.model.get_image_features(pv, one["image_grid_thw"][:1].to(device)).pooler_output[0]
        stored = torch.from_numpy(np.load(mem_episode_path(args.mem_feature_dir, ep))[t // ds.mem_stride]).to(device)
        cos = torch.nn.functional.cosine_similarity(live.float(), stored.float(), dim=-1)
        check(float(cos.min()) > 0.999, f"ep {ep} frame {t}: stored vs live feature cos min={float(cos.min()):.5f}")
        break
    else:
        print("   (no sampled anchor on a keyframe; feature check skipped)")

    # 4. zero-init fusion: memory on == memory off
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    torch.manual_seed(0)
    noise = torch.randn(inputs["actions"].shape)
    timesteps = sample_flow_timesteps(args.batch_size, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        on = model(**inputs, noise=noise, timesteps=timesteps, mem_mode="normal")
        off = model(**inputs, noise=noise, timesteps=timesteps, mem_mode="zero")
    print(f"   loss mem-on={float(on.action_loss):.6f} mem-off={float(off.action_loss):.6f} mem_norm={float(on.mem_norm):.3g}")
    if args.fusion == "adaln_add":
        check(torch.equal(on.pred_velocity, off.pred_velocity), "zero-init: identical velocity with/without memory")

    # 5. gradients
    model.train()
    out = model(**inputs, noise=noise, timesteps=timesteps)
    out.loss.backward()
    grads = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            key = ".".join(n.split(".")[:2])
            grads[key] = grads.get(key, 0.0) + float(p.grad.float().norm() ** 2)
    print("   grad norm by module:", {k: f"{v ** 0.5:.3g}" for k, v in sorted(grads.items())})
    check(not any(k.startswith("backbone") for k in grads), "no gradient into the backbone")
    check(any(k.startswith("memory") for k in grads), "memory receives gradient")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
