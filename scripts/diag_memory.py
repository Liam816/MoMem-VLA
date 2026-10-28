"""Diagnose whether a trained memory branch depends on its input.

Loads a finished memory-cache run, feeds held-out batches and reports
  * input check  - how many samples' memory changes under ``mem_mode='shuffle'``;
  * variation    - spread of the memory input, the compressed tokens Z and the
                   fusion vector m across samples, relative to their mean;
  * effect       - how far the DiT velocity / loss moves under zero / shuffle;
  * scale        - |m| against the timestep embedding it is added to.

    python scripts/diag_memory.py --run_dir OUT/robomme_small_E2 --root DATA \
        --mem_feature_dir FEAT --episodes_file val.txt --out diag.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simplememvla.compat import apply_fla_torch_compat

apply_fla_torch_compat()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from configs.sft_params import DataArguments
from simplememvla.data.collator import SimpleMemVLADataCollator
from simplememvla.data.dataset import SimpleMemVLADataset
from simplememvla.model import SimpleMemVLAConfig, SimpleMemVLAForActionPrediction
from simplememvla.model.modeling_simplememvla import (
    load_checkpoint_state_dict,
    masked_flow_matching_loss,
    sample_flow_timesteps,
)
from train import read_episodes_file, resolve_benchmark_defaults


def load_run(run_dir: str, attn_implementation: str) -> SimpleMemVLAForActionPrediction:
    config = SimpleMemVLAConfig.from_pretrained(run_dir)
    backbone = AutoModelForImageTextToText.from_config(
        AutoConfig.for_model(**dict(config.backbone_config)),
        attn_implementation=attn_implementation,
    )
    for param in backbone.parameters():
        param.data = param.data.to(torch.bfloat16)
    model = SimpleMemVLAForActionPrediction(config=config, backbone=backbone)
    missing, unexpected = model.load_state_dict(load_checkpoint_state_dict(run_dir), strict=False)
    if any(k.endswith("lm_head.weight") for k in missing):
        model.backbone.tie_weights()
        missing = [k for k in missing if not k.endswith("lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"missing={missing[:8]} unexpected={unexpected[:8]}")
    model.action_head.to(torch.bfloat16)
    if model.memory is not None:
        model.memory.to(torch.bfloat16)
    return model


def spread(x: torch.Tensor) -> dict:
    """Per-sample rows ``(N, ...)``: how much rows differ from their mean."""
    x = x.float().flatten(1)
    mean = x.mean(0, keepdim=True)
    dev = (x - mean).norm(dim=1)
    unit = torch.nn.functional.normalize(x, dim=1)
    cos = unit @ unit.T
    off = cos[~torch.eye(len(x), dtype=torch.bool, device=x.device)]
    return {
        "mean_norm": float(mean.norm()),
        "row_norm": float(x.norm(dim=1).mean()),
        "rel_dev": float(dev.mean() / mean.norm().clamp_min(1e-12)),
        "pairwise_cos_mean": float(off.mean()),
        "pairwise_cos_min": float(off.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--mem_feature_dir", required=True)
    parser.add_argument("--episodes_file", required=True)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = torch.device("cuda")

    model = load_run(args.run_dir, args.attn_implementation).to(device).eval()
    cfg = model.config
    if not cfg.use_memory:
        raise SystemExit(f"{args.run_dir} has use_memory=False; nothing to diagnose")

    da = DataArguments(
        benchmark="robomme",
        root=args.root,
        episodes=read_episodes_file(args.episodes_file),
        single_frame=True,
        image_aug=False,
        mem_feature_dir=args.mem_feature_dir,
        mem_window_sec=cfg.mem_window_sec,
        mem_fps=cfg.mem_fps,
        stats_path=str(Path(args.run_dir) / "stats.json"),
    )
    spec = resolve_benchmark_defaults(da)
    ds = SimpleMemVLADataset(da, spec, condition_source=cfg.condition_source, mem_frame_pool=cfg.mem_frame_pool)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), min(args.num_samples, len(ds)), replace=False).tolist()
    processor = AutoProcessor.from_pretrained(args.run_dir, padding_side="right")
    collator = SimpleMemVLADataCollator(processor, max_length=8192, condition_source=cfg.condition_source)
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collator)

    feats_pooled, zs, ms, t_embs = [], [], [], []
    changed, total = 0, 0
    eff = {k: [] for k in ("dv_zero", "dv_shuffle", "loss_normal", "loss_zero", "loss_shuffle", "dm_shuffle")}
    with torch.no_grad():
        for step, batch in enumerate(loader):
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            f, a, mk = inputs["mem_feats"], inputs["mem_age"], inputs["mem_mask"]
            bsz = f.shape[0]

            # input check: is the rolled memory actually different per sample?
            f_s, a_s, mk_s = (t.roll(1, dims=0) for t in (f, a, mk))
            same = (mk == mk_s).all(1) & (f == f_s).flatten(1).all(1) & (a == a_s).all(1)
            changed += int((~same).sum())
            total += bsz

            # variation along the memory path
            valid = mk.unsqueeze(-1).unsqueeze(-1).to(f.dtype)
            feats_pooled.append(((f * valid).sum(1) / valid.sum(1).clamp_min(1)).cpu())
            z = model.memory.compress(f, a, mk)
            zs.append(z.cpu())
            m, _ = model._memory_condition(f, a, mk, device, "normal")
            m_s, _ = model._memory_condition(f, a, mk, device, "shuffle")
            ms.append(m.cpu())
            eff["dm_shuffle"].append(float(((m_s - m).float().norm(dim=1) / m.float().norm(dim=1).clamp_min(1e-12)).mean()))

            # effect on the velocity / loss with identical noise and t
            torch.manual_seed(args.seed + step)
            noise = torch.randn(inputs["actions"].shape)
            t = sample_flow_timesteps(bsz, torch.device("cpu"))
            t_embs.append(model.action_head.time_embed(t.to(device, torch.bfloat16)).cpu())
            out = {
                mode: model(**inputs, noise=noise, timesteps=t, mem_mode=mode)
                for mode in ("normal", "zero", "shuffle")
            }
            v_n = out["normal"].pred_velocity.float()
            for mode in ("zero", "shuffle"):
                dv = (out[mode].pred_velocity.float() - v_n).flatten(1).norm(dim=1) / v_n.flatten(1).norm(dim=1)
                eff[f"dv_{mode}"].append(float(dv.mean()))
            for mode in ("normal", "zero", "shuffle"):
                eff[f"loss_{mode}"].append(float(out[mode].action_loss))
            print(f"batch {step}: dv_zero={eff['dv_zero'][-1]:.2e} dv_shuffle={eff['dv_shuffle'][-1]:.2e} "
                  f"dm_shuffle={eff['dm_shuffle'][-1]:.2e}", flush=True)

    m_all, t_all = torch.cat(ms), torch.cat(t_embs)
    last = model.memory.out_proj[-1] if isinstance(model.memory.out_proj, torch.nn.Sequential) else model.memory.out_proj
    report = {
        "run_dir": args.run_dir,
        "num_samples": total,
        "input_check": {"samples_changed_by_shuffle": changed, "total": total},
        "variation": {
            "memory_input_mean_pooled": spread(torch.cat(feats_pooled)),
            "compressed_Z": spread(torch.cat(zs)),
            "fusion_m": spread(m_all),
        },
        "effect": {k: float(np.mean(v)) for k, v in eff.items()},
        "scale": {
            "m_norm": float(m_all.float().norm(dim=1).mean()),
            "t_emb_norm": float(t_all.float().norm(dim=1).mean()),
            "m_over_t_emb": float(m_all.float().norm(dim=1).mean() / t_all.float().norm(dim=1).mean()),
            "out_proj_last_weight_norm": float(last.weight.float().norm()),
            "out_proj_last_bias_norm": float(last.bias.float().norm()),
        },
    }
    report["effect"]["gap_zero"] = report["effect"]["loss_zero"] - report["effect"]["loss_normal"]
    report["effect"]["gap_shuffle"] = report["effect"]["loss_shuffle"] - report["effect"]["loss_normal"]
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
