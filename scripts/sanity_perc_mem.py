"""One-GPU sanity checks for the perception-memory model (mem_arch=perc_queue).

Builds the dataset and model exactly as ``train.py`` does from a training
config, on a few episodes, then checks the batch, a training step and every
eviction mode. Without real teacher labels, ``--fake_labels DIR`` writes random
lower-triangular utilities for those episodes so the scorer's rank loss runs.

    python scripts/sanity_perc_mem.py --config configs/train/perc_mem/base.yaml \
        --num_episodes 4 --batch_size 4 --fake_labels /dev/shm/perc_labels_sanity
"""

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simplememvla.compat import apply_fla_torch_compat

apply_fla_torch_compat()

import numpy as np
import torch
from transformers import AutoProcessor

from simplememvla.data.collator import SimpleMemVLADataCollator
from simplememvla.data.dataset import SimpleMemVLADataset, mem_episode_path
from simplememvla.model.modeling_simplememvla import sample_flow_timesteps
from simplememvla.model.perc_memory import EVICT_MODES, PerceptionMemory
from train import (
    build_model,
    parse_train_args,
    read_episodes_file,
    resolve_benchmark_defaults,
    resolve_memcache_defaults,
)


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def write_fake_labels(out: Path, feature_dir: str, episodes: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for ep in episodes:
        n = np.load(mem_episode_path(feature_dir, ep), mmap_mode="r").shape[0]
        u = np.tril(rng.random((n, n))).astype(np.float16)
        np.save(mem_episode_path(out, ep), u)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_episodes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--fake_labels", default=None)
    args, rest = parser.parse_known_args()

    model_args, data_args, training_args = parse_train_args(["--config", args.config, *rest])
    if data_args.episodes_file:
        episodes = read_episodes_file(data_args.episodes_file)
    else:  # config trains on every episode: use those whose features are already extracted
        names = (p.name for p in Path(data_args.mem_feature_dir).glob("episode_*.npy"))
        episodes = sorted(int(m[1]) for m in map(re.compile(r"episode_(\d+)\.npy").fullmatch, names) if m)
    # The longest histories exercise eviction; take them from the tail of the split.
    lengths = {
        ep: np.load(mem_episode_path(data_args.mem_feature_dir, ep), mmap_mode="r").shape[0]
        for ep in episodes
    }
    episodes = sorted(episodes, key=lambda ep: -lengths[ep])[: args.num_episodes]
    data_args.episodes_file = None
    data_args.episodes = episodes
    if args.fake_labels:
        write_fake_labels(Path(args.fake_labels), data_args.mem_feature_dir, episodes)
        data_args.perc_mem_label_dir = args.fake_labels
    spec = resolve_benchmark_defaults(data_args)
    mem_meta = resolve_memcache_defaults(model_args, data_args)
    check(model_args.mem_arch == "perc_queue", "config selects mem_arch=perc_queue")
    print(f"   episodes {episodes} keyframes {[lengths[e] for e in episodes]}")

    # 1. data
    ds = SimpleMemVLADataset(
        data_args, spec, condition_source=model_args.condition_source,
        mem_frame_pool=model_args.mem_frame_pool,
    )
    vlm_dir = model_args.init_from_checkpoint or model_args.backbone_model_name_or_path
    processor = AutoProcessor.from_pretrained(vlm_dir, padding_side="right")
    collator = SimpleMemVLADataCollator(
        processor, max_length=model_args.model_max_length, condition_source=model_args.condition_source
    )
    # Late frames of the longest episodes: full queues, demo frames still in history.
    idxs = list(range(len(ds) - 1, -1, -max(1, len(ds) // args.batch_size)))[: args.batch_size]
    samples = [ds[i] for i in idxs]
    batch = collator(samples)
    n_hist = batch["mem_mask"].sum(1)
    print(f"   mem_feats {tuple(batch['mem_feats'].shape)} history frames {n_hist.tolist()}")
    check(batch["mem_feats"].shape[2] == mem_meta["tokens_per_frame"], "native merger tokens per frame")
    check(bool((batch["mem_age"][:, 0] > 0).all()), "history starts at the first keyframe")
    check("mem_seg" not in batch, "no segment flags (demo frames are ordinary)")
    if data_args.perc_mem_label_dir:
        check(tuple(batch["mem_utility"].shape) == tuple(batch["mem_mask"].shape) + (batch["mem_mask"].shape[1],),
              "mem_utility is (B, N, N)")

    # 2. model
    t0 = time.time()
    model = build_model(model_args, data_args, spec, ds, mem_meta).cuda()
    print(f"   model built in {time.time() - t0:.0f}s")
    mem = model.memory
    check(isinstance(mem, PerceptionMemory), "model.memory is PerceptionMemory")
    count = lambda m: sum(p.numel() for p in m.parameters()) / 1e6  # noqa: E731
    print(f"   params (M): scorer={count(mem.scorer):.2f} reader={count(mem.reader):.2f} "
          f"total={count(mem):.2f}  fusion={mem.fusion}")
    k, w = mem.queue_frames, mem.window_frames

    inputs = {key: v.cuda() if torch.is_tensor(v) else v for key, v in batch.items()}
    torch.manual_seed(0)
    noise = torch.randn(inputs["actions"].shape)
    timesteps = sample_flow_timesteps(len(samples), torch.device("cpu"))

    # 3. selection
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        state, _ = mem.replay(inputs["mem_feats"], inputs["mem_age"], inputs["mem_mask"],
                              inputs.get("mem_utility"))
        torch.cuda.synchronize()
        print(f"   replay {int(n_hist.max())} keyframes x B={len(samples)}: {1000 * (time.time() - t0):.0f} ms")
    kept = state.valid.sum(1)
    check(bool((kept == torch.clamp(n_hist.cuda(), max=k + w)).all()), f"kept min(N, K+W={k + w}) frames")
    for b in range(len(samples)):
        src = state.src[b][state.valid[b]]
        check(bool((src[1:] > src[:-1]).all()), f"row {b}: kept frames in time order")
        check(src[-w:].tolist() == list(range(int(n_hist[b]) - w, int(n_hist[b]))),
              f"row {b}: window holds the {w} newest frames")

    # 4. eval forward: zero-init equivalence, every eviction mode, memory zero / shuffle
    with torch.no_grad():
        if mem.fusion == "adaln_add":
            on = model(**inputs, noise=noise, timesteps=timesteps)
            off = model(**inputs, noise=noise, timesteps=timesteps, mem_mode="zero")
            check(torch.equal(on.pred_velocity, off.pred_velocity),
                  "zero-init c_mem: velocity identical with / without memory")
        for mode in EVICT_MODES:
            if mode == "oracle" and "mem_utility" not in inputs:
                continue
            mem.evict_mode = mode
            out = model(**inputs, noise=noise, timesteps=timesteps)
            print(f"   [{mode}] action_loss={float(out.action_loss):.5f}")
            check(bool(torch.isfinite(out.action_loss)), f"{mode}: finite loss")
        mem.evict_mode = model_args.perc_mem_evict
        for mode in ("zero", "shuffle"):
            out = model(**inputs, noise=noise, timesteps=timesteps, mem_mode=mode)
            print(f"   mem_mode={mode} action_loss={float(out.action_loss):.5f}")

    # 5. training step
    model.train()
    out = model(**inputs, noise=noise, timesteps=timesteps)
    print(f"   train loss={float(out.loss):.5f} action={float(out.action_loss):.5f} "
          f"vl={float(out.vl_loss) if out.vl_loss is not None else None} rank={out.mem_rank_loss}")
    if data_args.perc_mem_label_dir and not model_args.perc_mem_freeze_scorer:
        check(out.mem_rank_loss is not None and bool(torch.isfinite(out.mem_rank_loss)), "rank loss present")
    out.loss.backward()
    grads: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            key = "backbone.lora" if name.startswith("backbone") and "lora" in name else ".".join(name.split(".")[:2])
            grads[key] = grads.get(key, 0.0) + float(param.grad.float().norm() ** 2)
    print("   grad norm:", {key: f"{v ** 0.5:.3g}" for key, v in sorted(grads.items())})
    check(not any(key.startswith("backbone") and key != "backbone.lora" for key in grads),
          "backbone: only LoRA gets gradient")
    check("memory.reader" in grads, "reader gets gradient")
    check(("memory.scorer" in grads) == (out.mem_rank_loss is not None),
          "scorer gets gradient iff the rank loss is on")
    print(f"   peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
