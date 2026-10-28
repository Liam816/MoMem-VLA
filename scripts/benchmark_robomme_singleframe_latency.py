#!/usr/bin/env python3
"""Paired batch-1 decision latency on common RoboMME rollout observations.

The released video policy controls one simulator. Video and current-frame-only
policies share that model and see exactly the same frame/state stream. This is a
timing comparison on video-policy trajectories, not a single-frame success test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


TASKS = ["PickXtimes", "VideoUnmaskSwap", "VideoRepick", "PatternLock"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_split", choices=["train", "test"], required=True)
    p.add_argument("--episode_ids_json", default=None)
    p.add_argument("--episodes_per_task", type=int, default=2)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--max_decisions", type=int, default=12)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--compute_dtype", default="bfloat16")
    p.add_argument("--attn_implementation", default="flash_attention_2")
    p.add_argument("--execute_horizon", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=1300)
    p.add_argument("--num_denoising_steps", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_subtask_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step_timeout", type=float, default=300.0)
    p.add_argument("--reset_timeout", type=float, default=3600.0)
    p.add_argument("--no_cold_visual", action="store_true",
                   help="Only measure the natural deployed cache flow.")
    a = p.parse_args()
    if a.compute_dtype == "bf16":
        a.compute_dtype = "bfloat16"
    for key in ("episodes_per_task", "max_decisions", "execute_horizon"):
        if getattr(a, key) < 1:
            p.error(f"--{key} must be positive")
    if a.warmup < 1:
        p.error("--warmup must be at least 1")
    if len(set(a.tasks)) != len(a.tasks) or not set(a.tasks).issubset(TASKS):
        p.error("--tasks must be unique tasks from the current four-task subset")
    return a


def now():
    return dt.datetime.now().astimezone().isoformat()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def percentile(values, q):
    xs = sorted(values)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def distribution(values):
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {"n": len(values), "mean": sum(values) / len(values),
            "p50": percentile(values, .5), "p95": percentile(values, .95)}


def summarize(rows, status, args, episodes, started):
    valid = [r for r in rows if r.get("valid") is True]
    groups = []
    for task in ["ALL"] + args.tasks:
        for mode in ("video", "single_frame"):
            for cache in ("natural", "cold_visual"):
                sample = [r for r in valid if (task == "ALL" or r["task"] == task)
                          and r["mode"] == mode and r["cache"] == cache]
                if sample:
                    groups.append({"dataset_split": args.dataset_split, "task": task,
                                   "mode": mode, "cache": cache, "n": len(sample),
                                   **{key: distribution([r[key] for r in sample])
                                      for key in ("preprocess_ms", "model_decision_ms", "total_ms")}})
    paired = []
    pairs = {}
    for r in valid:
        key = (r["task"], r["episode"], r["decision"], r["cache"])
        pairs.setdefault(key, {})[r["mode"]] = r
    for task in ["ALL"] + args.tasks:
        for cache in ("natural", "cold_visual"):
            complete = [p for k, p in pairs.items()
                        if k[3] == cache and (task == "ALL" or k[0] == task)
                        and set(p) == {"video", "single_frame"}]
            if complete:
                v = [p["video"]["total_ms"] for p in complete]
                s = [p["single_frame"]["total_ms"] for p in complete]
                paired.append({"dataset_split": args.dataset_split, "task": task,
                               "cache": cache, "n_pairs": len(complete),
                               "video_over_single_frame_speedup": distribution([a / b for a, b in zip(v, s)]),
                               "ratio_of_mean_total_ms": sum(v) / sum(s),
                               "saved_ms": distribution([a - b for a, b in zip(v, s)])})
    return {"status": status, "started_at": started, "updated_at": now(),
            "dataset_split": args.dataset_split, "checkpoint": args.checkpoint,
            "episodes": episodes, "group_size": 1, "warmup_per_mode": args.warmup,
            "max_decisions_per_episode": args.max_decisions,
            "records_valid": len(valid), "records_invalid": len(rows) - len(valid),
            "groups": groups, "paired": paired,
            "protocol": {
                "controller": "released video policy, natural cache measurements only",
                "input_pairing": "identical current observations, states and history stream per anchor",
                "natural_cache": "continuous deployed vision/prompt cache flow; no persistent cross-decision LLM KV cache",
                "cold_visual_cache": "same observed anchor, vision feature and prompt caches cleared; saved caches restored afterwards",
                "timing": "CUDA-synchronized perf_counter wall time; preprocessing and model decision measured separately; total includes small cache-statistics bookkeeping gap",
                "excluded": "model loading, warmup, simulator reset/step, frame observation, state normalization, token-count extraction and result-file IO",
                "free_generation": "actual greedy sub-task decode, not a fixed generated-token count; token counts are recorded",
                "limitation": "latency observations follow the video controller, not each policy's independent closed-loop trajectory; early episode completion can reduce sample counts",
            }}


def cache_snapshot(policy):
    return (None if policy._prompt_cache is None else dict(policy._prompt_cache),
            dict(policy._patch_feat_cache))


def restore_cache(policy, snapshot):
    prompt, features = snapshot
    policy._prompt_cache = None if prompt is None else dict(prompt)
    policy._patch_feat_cache = dict(features)


def validate_sim_result(result, stage):
    if (not isinstance(result, dict) or "error" in result
            or result.get("status") == "error" or result.get("error_message") is not None):
        raise RuntimeError(f"Simulator {stage} failed: {result!r}")
    if stage == "reset" and not result.get("ok"):
        raise RuntimeError(f"Simulator reset failed: {result!r}")


def run(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "raw.jsonl").exists() or (out / "status.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing latency run: {out}")
    started = now(); rows = []; episode_report = []; pool = None
    status = {"state": "initializing", "started_at": started, "updated_at": started,
              "dataset_split": args.dataset_split, "group_size": 1}
    atomic_json(out / "status.json", status)
    resolved = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    atomic_json(out / "args.json", resolved)
    with (out / "raw.jsonl").open("x", buffering=1) as raw:
        def emit(record):
            rows.append(record)
            raw.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            raw.flush()

        context = {}
        try:
            import numpy as np
            import torch
            repo = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(repo))
            from robomme_sim._compat import apply_fla_torch_compat
            apply_fla_torch_compat()
            from transformers import AutoProcessor
            from robomme_sim.policy import RoboMMEPolicy, get_vla
            from robomme_sim.batched_policy import BatchedEvalPolicy
            from robomme_sim.inproc_pool import InProcSimPool

            if not torch.cuda.is_available():
                raise RuntimeError("A CUDA GPU is required for this latency protocol")
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            vla, unnormalize, normalize_state = get_vla(
                args.checkpoint, args.compute_dtype, attn_implementation=args.attn_implementation)
            if (getattr(vla.config, "condition_source", "subtask") != "subtask"
                    or getattr(vla.config, "use_memory", False)):
                raise ValueError("Use the released sub-task checkpoint, not E1/E2 memory-cache checkpoints")
            processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True,
                                                     padding_side="right", model_max_length=8192)
            device = next(vla.parameters()).device

            class TracedBatchedPolicy(BatchedEvalPolicy):
                def _gather_condition(self, hidden_states, input_ids, span, position_ids, attention_mask):
                    self.last_full_ids = input_ids.detach()
                    return super()._gather_condition(hidden_states, input_ids, span,
                                                     position_ids, attention_mask)

            batched = TracedBatchedPolicy(vla, processor, unnormalize, normalize_state,
                                         args.num_denoising_steps, args.temperature,
                                         args.max_subtask_tokens, device=device)
            policies = {mode: RoboMMEPolicy(vla, processor, unnormalize, normalize_state,
                                           args.num_denoising_steps, args.temperature,
                                           args.max_subtask_tokens,
                                           current_frame_only=(mode == "single_frame"))
                        for mode in ("video", "single_frame")}
            if args.episode_ids_json:
                ids = json.loads(Path(args.episode_ids_json).read_text())
                episodes = {t: [int(e) for e in ids[t]][:args.episodes_per_task] for t in args.tasks}
            else:
                episodes = {t: list(range(args.episodes_per_task)) for t in args.tasks}
            for task, ids in episodes.items():
                if len(ids) != args.episodes_per_task or len(ids) != len(set(ids)) or any(e < 0 for e in ids):
                    raise ValueError(f"Invalid or insufficient episode selection for {task}: {ids}")
            metadata = {
                "checkpoint": args.checkpoint, "processor_source": args.checkpoint,
                "stats_source": args.checkpoint, "weight_override": None,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_version": torch.__version__, "episodes": episodes,
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__).resolve(), repo / "robomme_sim/policy.py",
                                             repo / "robomme_sim/batched_policy.py"]},
            }
            atomic_json(out / "manifest.json", metadata)
            pool = InProcSimPool(1, dataset_split=args.dataset_split, max_steps=args.max_steps,
                                 step_timeout=args.step_timeout, reset_timeout=args.reset_timeout)
            pool.start()
            camera_map = {k.split(".")[-1]: k for k in policies["video"].image_keys}

            def observe(frames):
                for frame in frames:
                    full = {camera_map.get(k, k): np.asarray(v, dtype=np.uint8) for k, v in frame.items()}
                    for policy in policies.values():
                        policy.observe(full)

            def state_tensor(state):
                if normalize_state is None:
                    return None
                return normalize_state.normalize(torch.from_numpy(np.asarray(state, dtype=np.float32))).unsqueeze(0).to(device)

            def measure(mode, cache, instruction, state, info, record=True):
                policy = policies[mode]
                saved = cache_snapshot(policy) if cache == "cold_visual" else None
                if saved is not None:
                    policy._prompt_cache = None
                    policy._patch_feat_cache.clear()
                try:
                    torch.manual_seed(info["sampling_seed"])
                    torch.cuda.synchronize(device)
                    begin = time.perf_counter()
                    processed = policy._prepare_inputs(instruction)
                    torch.cuda.synchronize(device)
                    prepared = time.perf_counter()
                    keys = processed["video_patch_keys"]
                    unique_keys = set(keys)
                    hit_count = sum(k in policy._patch_feat_cache for k in keys)
                    unique_hits = len(unique_keys.intersection(policy._patch_feat_cache))
                    model_start = time.perf_counter()
                    prediction = batched.generate_batch([processed], [state])[0]
                    torch.cuda.synchronize(device)
                    end = time.perf_counter()
                    if record:
                        prompt_length = int(processed["input_ids"].shape[-1])
                        generated = batched.last_full_ids[0, prompt_length:].tolist()
                        if batched.im_end_id in generated:
                            generated = generated[:generated.index(batched.im_end_id)]
                        unique_frames = {k: len(set(policy._history_global_indices(k)))
                                         for k in policy.history_image_keys}
                        if mode == "single_frame" and any(n != 1 for n in unique_frames.values()):
                            raise AssertionError("Single-frame policy used more than one distinct history frame")
                        emit({**info, "valid": True, "mode": mode, "cache": cache, "group_size": 1,
                              "preprocess_ms": (prepared - begin) * 1000,
                              "model_decision_ms": (end - model_start) * 1000,
                              "total_ms": (end - begin) * 1000,
                              "measurement_gap_ms": (model_start - prepared) * 1000,
                              "input_token_count": int(processed["attention_mask"].sum().item()),
                              "video_patches_total": len(keys), "video_patches_unique": len(unique_keys),
                              "cache_hits": hit_count, "cache_misses": len(keys) - hit_count,
                              "cache_unique_hits": unique_hits, "cache_unique_misses": len(unique_keys) - unique_hits,
                              "history_unique_frames": unique_frames,
                              "generated_subtask_token_count": len(generated),
                              "generated_subtask_at_budget": len(generated) >= args.max_subtask_tokens,
                              "generated_subtask": prediction[1]})
                    return prediction
                finally:
                    if saved is not None:
                        restore_cache(policy, saved)

            warmed = False
            anchor_number = 0
            status.update(state="running", updated_at=now())
            atomic_json(out / "status.json", status)
            for task in args.tasks:
                for episode in episodes[task]:
                    context = {"task": task, "episode": episode, "dataset_split": args.dataset_split,
                               "stage": "reset"}
                    reset = pool.reset([{"task": task, "episode": episode}])[0]
                    validate_sim_result(reset, "reset")
                    for policy in policies.values():
                        policy.reset()
                    observe(reset.get("frames", []))
                    if not reset.get("states"):
                        raise RuntimeError("Simulator reset provided no state")
                    instruction = reset["instruction"]
                    current_state = reset["states"][-1]
                    state = state_tensor(current_state)
                    if not warmed:
                        initial_cache = {m: cache_snapshot(p) for m, p in policies.items()}
                        warm_info = {"sampling_seed": args.seed}
                        for _ in range(args.warmup):
                            for mode in ("video", "single_frame"):
                                measure(mode, "natural", instruction, state, warm_info, record=False)
                        for mode, snapshot in initial_cache.items():
                            restore_cache(policies[mode], snapshot)
                        warmed = True
                        print(f"[latency] GPU warmup complete: {args.warmup} calls per mode; caches restored", flush=True)
                    episode_info = {"task": task, "episode": episode, "dataset_split": args.dataset_split,
                                    "decisions": 0, "termination": "max_decisions", "video_controller_success": False}
                    for decision in range(args.max_decisions):
                        context = {"task": task, "episode": episode, "dataset_split": args.dataset_split,
                                   "decision": decision, "stage": "measure"}
                        indices = {k: n - 1 for k, n in policies["video"]._frames_seen.items()}
                        assert indices == {k: n - 1 for k, n in policies["single_frame"]._frames_seen.items()}
                        info = {k: v for k, v in context.items() if k != "stage"}
                        info.update(frame_index=indices[policies["video"].history_image_keys[0]],
                                    frame_index_by_camera=indices, sampling_seed=args.seed + anchor_number)
                        natural = {}
                        order = ("video", "single_frame") if anchor_number % 2 == 0 else ("single_frame", "video")
                        for mode in order:
                            context.update(mode=mode, cache="natural")
                            natural[mode] = measure(mode, "natural", instruction, state, info)
                        if not args.no_cold_visual:
                            for mode in reversed(order):
                                context.update(mode=mode, cache="cold_visual")
                                measure(mode, "cold_visual", instruction, state, info)
                        episode_info["decisions"] += 1
                        anchor_number += 1
                        context = {**info, "stage": "step"}
                        step = pool.step([natural["video"][0][:args.execute_horizon]], [0])[0]
                        validate_sim_result(step, "step")
                        observe(step.get("frames", []))
                        if step.get("states"):
                            current_state = step["states"][-1]
                            state = state_tensor(current_state)
                        elif not step.get("done"):
                            raise RuntimeError("Simulator step provided no state while episode remains active")
                        status.update(updated_at=now(), last_anchor=info, measured_anchors=anchor_number,
                                      valid_rows=sum(r.get("valid") is True for r in rows))
                        atomic_json(out / "status.json", status)
                        if step.get("done"):
                            episode_info.update(termination="environment_done", video_controller_success=bool(step.get("success")))
                            break
                    episode_report.append(episode_info)
                    atomic_json(out / "summary.json", summarize(rows, "running", args, episode_report, started))
                    print(f"[latency] {args.dataset_split} {task} ep{episode}: {episode_info['decisions']} anchors, "
                          f"termination={episode_info['termination']}", flush=True)
            status.update(state="complete", ended_at=now(), updated_at=now(), exit_code=0,
                          measured_anchors=anchor_number, episodes_completed=len(episode_report))
            atomic_json(out / "summary.json", summarize(rows, "complete", args, episode_report, started))
            atomic_json(out / "status.json", status)
        except BaseException as exc:
            error = {**context, "valid": False, "error_type": type(exc).__name__,
                     "error": str(exc), "traceback": traceback.format_exc(), "recorded_at": now()}
            emit(error)
            status.update(state="failed", ended_at=now(), updated_at=now(), exit_code=1,
                          error=error, episodes_completed=len(episode_report))
            atomic_json(out / "summary.json", summarize(rows, "failed", args, episode_report, started))
            atomic_json(out / "status.json", status)
            raise
        finally:
            if pool is not None:
                pool.close()


if __name__ == "__main__":
    run(parse_args())
