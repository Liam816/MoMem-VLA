"""Sanity checks for the single-frame + memory-cache path (one GPU).

1. dataset/collator produce the prompt-conditioned single-frame batch;
2. weights: the VLM comes from the released checkpoint (--checkpoint, init_modules)
   or a plain HF backbone (--backbone); everything not loaded starts fresh;
3. offline memory features == the backbone's own image features of that frame;
4. zero-initialised fusion => memory on/off give the identical loss;
5. gradients reach only the action head and memory.

    python scripts/sanity_memcache.py --checkpoint CKPT --root DATA \
        --mem_feature_dir FEAT --episodes 700 1500 [--fusion adaln_add]
    python scripts/sanity_memcache.py --backbone QWEN_DIR --root DATA \
        --mem_feature_dir FEAT --episodes 700 1500 --stats_path DATA/meta/stats.json
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
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="released SimpleMemVLA checkpoint")
    src.add_argument("--backbone", help="plain HF VLM (e.g. Qwen3.5-4B); DiT + memory fresh")
    parser.add_argument("--root", required=True)
    parser.add_argument("--mem_feature_dir", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--fusion", default="adaln_add")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--init_modules", nargs="+", default=["backbone", "action_head"])
    parser.add_argument("--stats_path", default=None, help="default: <checkpoint>/stats.json")
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
        stats_path=args.stats_path or (f"{args.checkpoint}/stats.json" if args.checkpoint else None),
    )
    spec = resolve_benchmark_defaults(da)
    ds = SimpleMemVLADataset(da, spec, condition_source="prompt", mem_frame_pool=pool)
    vlm_dir = args.checkpoint or args.backbone
    processor = AutoProcessor.from_pretrained(vlm_dir, padding_side="right")
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
    meta = json.loads((Path(args.mem_feature_dir) / "meta.json").read_text())
    check(Path(meta["checkpoint"]).resolve() == Path(vlm_dir).resolve(),
          f"memory features were extracted with this VLM ({meta['checkpoint']})")
    config = SimpleMemVLAConfig(
        backbone_model_name_or_path=vlm_dir,
        action_dim=spec.action_dim,
        action_horizon=spec.num_actions_chunk,
        state_dim=spec.state_dim,
        use_proprio=True,
        freeze_backbone=True,
        vl_loss_weight=0.0,
        condition_source="prompt",
        single_frame=True,
        use_memory=True,
        mem_feature_dim=meta["feature_dim"],
        mem_frame_pool=pool,
        mem_fusion=args.fusion,
    )
    if args.checkpoint:
        model = SimpleMemVLAForActionPrediction.from_simplememvla_checkpoint(
            args.checkpoint, config, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            init_modules=tuple(args.init_modules),
        ).to(device)
        loaded_modules = set(args.init_modules)
    else:
        model = SimpleMemVLAForActionPrediction.from_backbone(
            args.backbone, config=config, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        ).to(device)
        loaded_modules = {"backbone"}
    from safetensors import safe_open

    # parameter name in the model -> (file prefix to strip) for the weight source
    probes = {
        "backbone": "backbone.model.language_model.layers.0.mlp.down_proj.weight",
        "action_head": "action_head.blocks.0.attn.qkv.weight",
    }
    params = dict(model.named_parameters())
    index = Path(vlm_dir) / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    for module, key in probes.items():
        if key not in params:
            key = next(k for k in params if k.startswith(f"{module}.") and k.endswith(".weight"))
        file_key = key if args.checkpoint else key[len("backbone."):]
        if args.backbone and module != "backbone":
            check(True, f"{module} freshly initialised (no SimpleMemVLA checkpoint)")
            continue
        shard = weight_map[file_key] if weight_map else "model.safetensors"
        with safe_open(str(Path(vlm_dir) / shard), framework="pt") as f:
            same = torch.equal(params[key].detach().cpu(), f.get_tensor(file_key).to(params[key].dtype))
        want = module in loaded_modules
        check(same == want, f"{module} {'loaded from' if want else 'NOT loaded from'} {vlm_dir} ({key})")
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
        # The cache may be fp32 while the live VLM runs bf16; per-token bf16 noise of the
        # original Qwen ViT reaches cos~0.7, so compare the mean (same weights => ~0.998+).
        check(float(cos.mean()) > 0.995,
              f"ep {ep} frame {t}: stored vs live feature cos mean={float(cos.mean()):.5f} "
              f"(min={float(cos.min()):.5f}, cache vit_dtype={meta.get('vit_dtype', 'bfloat16')})")
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
