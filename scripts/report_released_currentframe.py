#!/usr/bin/env python3
"""Read-only result aggregation for the released-weight current-frame ablation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
from datetime import datetime
from zoneinfo import ZoneInfo

TASKS = ("PickXtimes", "VideoUnmaskSwap", "VideoRepick", "PatternLock")
PHASES = {"train40": 10, "test200": 50}
SUCCESS_STATES = {"complete", "completed", "success", "succeeded", "done", "finished"}
FAILURE_STATES = {"failed", "failure", "error", "aborted", "cancelled", "canceled", "timeout", "timed_out"}


def reject_constant(value):
    raise ValueError("Non-finite JSON value: " + value)


def read_json(path: Path):
    if not path.exists():
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        return None, "unreadable_or_partial: " + str(exc)
    return value, None


def validate_accuracy(value, expected_n):
    if not isinstance(value, dict) or set(value.get("tasks", {})) != set(TASKS):
        raise ValueError("Expected exactly the four configured tasks")
    if value.get("n_excluded_na") != 0:
        raise ValueError("N/A tasks or missing n_excluded_na")
    rows = {}
    for task in TASKS:
        row = value["tasks"][task]
        if not isinstance(row, dict):
            raise ValueError(f"Invalid task result for {task}")
        if row.get("n") != expected_n or row.get("oom") is not False:
            raise ValueError(f"Invalid count/OOM for {task}: {row}")
        rate = row.get("success_rate")
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError(f"Invalid success rate for {task}: {rate}")
        successes = rate * expected_n
        if not math.isclose(successes, round(successes), abs_tol=1e-6):
            raise ValueError(f"Success rate is inconsistent with integer outcomes for {task}")
        rows[task] = {"successes": round(successes), "n": expected_n, "percent": 100 * rate}
    expected_macro = sum(row["percent"] for row in rows.values()) / 100 / len(rows)
    if not isinstance(value.get("macro_avg"), (int, float)) or not math.isclose(value["macro_avg"], expected_macro, abs_tol=1e-8):
        raise ValueError("macro_avg disagrees with task results")
    return {"tasks": rows, "successes": sum(row["successes"] for row in rows.values()),
            "n": expected_n * len(TASKS), "percent": 100 * expected_macro}


def accuracy_report(root: Path):
    status_path = root / "accuracy/current_frame/status.txt"
    exit_code = None
    try:
        status_text = status_path.read_text(encoding="utf-8")
        for line in status_text.splitlines():
            if line.startswith("exit="):
                exit_code = int(line.split("=", 1)[1])
    except (OSError, ValueError):
        status_text = "pending"
    out = {"state": "running", "terminal": False, "exit_code": exit_code,
           "status_path": str(status_path), "status_text": status_text, "phases": {}, "errors": []}
    partial = False
    missing = False
    for phase, count in PHASES.items():
        entry = {}
        for mode in ("video_baseline", "current_frame"):
            path = root / "accuracy" / mode / phase / "results.json"
            value, error = read_json(path)
            if error:
                entry[mode] = {"state": "pending", "source": str(path), "reason": error}
                partial |= error.startswith("unreadable_or_partial")
                missing |= error == "missing"
                continue
            try:
                entry[mode] = {"state": "complete", "source": str(path), **validate_accuracy(value, count)}
            except (ValueError, TypeError, KeyError) as exc:
                entry[mode] = {"state": "invalid", "source": str(path), "reason": str(exc)}
                out["errors"].append(f"{mode}/{phase}: {exc}")
        if all(entry[m]["state"] == "complete" for m in ("video_baseline", "current_frame")):
            video, current = entry["video_baseline"], entry["current_frame"]
            entry["drop_percentage_points"] = video["percent"] - current["percent"]
            entry["tasks"] = {
                t: {"video": video["tasks"][t], "current_frame": current["tasks"][t],
                    "drop_percentage_points": video["tasks"][t]["percent"] - current["tasks"][t]["percent"]}
                for t in TASKS
            }
        out["phases"][phase] = entry
    if exit_code is not None:
        if exit_code != 0:
            out.update(state="failed", terminal=True)
            out["errors"].append(f"Accuracy launcher exited with {exit_code}")
        elif partial:
            out["state"] = "pending_partial_json"
        elif missing or out["errors"]:
            out.update(state="failed", terminal=True)
            out["errors"].append("Accuracy launcher exited zero but required results are missing/invalid")
        else:
            out.update(state="complete", terminal=True)
    elif out["errors"]:
        out["state"] = "running_with_invalid_results"
    return out


def token_length_report(path: Path):
    if not path.exists():
        return {"state": "pending", "reason": "raw.jsonl missing", "modes": {}}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError as exc:
        return {"state": "pending", "reason": str(exc), "modes": {}}
    fields = ("input_token_count", "generated_subtask_token_count")
    values = {m: {k: [] for k in fields} for m in ("video", "single_frame")}
    incomplete = 0
    invalid_token_rows = 0
    for line in lines:
        if not line.endswith("\n"):
            incomplete += 1
            continue
        try:
            row = json.loads(line, parse_constant=reject_constant)
        except (ValueError, TypeError):
            incomplete += 1
            continue
        if not isinstance(row, dict) or row.get("valid") is not True or row.get("cache") != "natural" or row.get("mode") not in values:
            continue
        if not all(type(row.get(k)) is int and row[k] >= 0 for k in fields):
            invalid_token_rows += 1
            continue
        for key in fields:
            values[row["mode"]][key].append(row[key])
    modes = {}
    for mode, groups in values.items():
        modes[mode] = {}
        for key, sample in groups.items():
            modes[mode][key] = {"n": len(sample), "mean": statistics.mean(sample) if sample else None,
                                "median": statistics.median(sample) if sample else None,
                                "min": min(sample) if sample else None, "max": max(sample) if sample else None}
    available = any(values[m][fields[0]] for m in values)
    return {"state": "available" if available else "pending", "source": str(path),
            "filter": "valid == true and cache == natural", "modes": modes,
            "incomplete_or_unparsed_lines": incomplete, "ignored_invalid_token_rows": invalid_token_rows}


def latency_report(root: Path, split: str):
    folder = root / "latency" / split
    status, status_error = read_json(folder / "status.json")
    summary, summary_error = read_json(folder / "summary.json")
    if summary is not None and not isinstance(summary, dict):
        summary, summary_error = None, "summary must be a JSON object"
    out = {"state": "pending", "terminal": False, "status": status,
           "summary": summary, "status_read_error": status_error, "summary_read_error": summary_error,
           "natural_token_lengths": token_length_report(folder / "raw.jsonl")}
    if status_error or not isinstance(status, dict):
        return out
    state = str(status.get("status", status.get("state", ""))).lower()
    code = next((status[k] for k in ("exit_code", "returncode", "return_code", "exit") if isinstance(status.get(k), int)), None)
    producer_terminal = state in SUCCESS_STATES | FAILURE_STATES or code is not None
    invalid = summary.get("records_invalid", 0) if isinstance(summary, dict) else 0
    if isinstance(invalid, (int, float)) and invalid > 0:
        out.update(state="failed", terminal=producer_terminal)
        out["error"] = f"Latency summary contains {invalid} invalid measured record(s)"
        return out
    if isinstance(summary, dict) and str(summary.get("status", "")).lower() in FAILURE_STATES:
        out.update(state="failed", terminal=producer_terminal)
        out["error"] = "Latency summary reports failure"
        return out
    if state in FAILURE_STATES or (code is not None and code != 0):
        out.update(state="failed", terminal=True)
        return out
    if state in SUCCESS_STATES or code == 0:
        if summary_error and summary_error.startswith("unreadable_or_partial"):
            out["state"] = "pending_partial_json"
        elif summary_error or not isinstance(summary, dict) or not summary:
            out.update(state="failed", terminal=True)
            out["error"] = "Latency producer finished without a valid summary.json"
        elif summary.get("status") != "complete":
            out["state"] = "pending_final_summary"
        else:
            errors = []
            if summary.get("group_size") != 1:
                errors.append("Latency batch/group size must be one")
            if not isinstance(summary.get("records_valid"), int) or summary["records_valid"] <= 0:
                errors.append("No valid measured latency records")
            if summary.get("records_invalid") != 0:
                errors.append("Missing or invalid records_invalid count")
            for cache in ("natural", "cold_visual"):
                for mode in ("video", "single_frame"):
                    rows = [r for r in summary.get("groups", []) if r.get("task") == "ALL" and r.get("cache") == cache and r.get("mode") == mode]
                    if len(rows) != 1 or not isinstance(rows[0].get("n"), int) or rows[0]["n"] <= 0:
                        errors.append(f"Missing measured ALL group for {cache}/{mode}")
            if errors:
                out.update(state="failed", terminal=True, error="; ".join(errors))
            else:
                out.update(state="complete", terminal=True)
    else:
        out["state"] = state or "running"
    return out


def build_report(root: Path):
    manifest, manifest_error = read_json(root / "manifest.json")
    provenance, provenance_error = read_json(root / "accuracy/video_baseline/provenance.json")
    if not isinstance(manifest, dict):
        manifest, manifest_error = None, manifest_error or "manifest must be a JSON object"
    if not isinstance(provenance, dict):
        provenance, provenance_error = None, provenance_error or "provenance must be a JSON object"
    accuracy = accuracy_report(root)
    latency = {s: latency_report(root, s) for s in ("train", "test")}
    components = [accuracy, *latency.values()]
    terminal = all(c["terminal"] for c in components)
    failed = any(c["state"] == "failed" for c in components)
    state = ("failed" if failed else "complete") if terminal else ("running_with_failures" if failed else "running")
    return {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "experiment_root": str(root), "state": state, "terminal": terminal,
            "manifest": manifest, "manifest_read_error": manifest_error,
            "video_baseline_provenance": provenance, "provenance_read_error": provenance_error,
            "accuracy": accuracy, "latency": latency}


def render_text(report):
    manifest = report.get("manifest") or {}
    provenance = report.get("video_baseline_provenance") or {}
    lp = manifest.get("latency", {})
    lines = ["Released SimpleMemVLA: current-frame ablation", "Updated: " + report["generated_at"],
             "State: " + report["state"], "Experiment root: " + report["experiment_root"], "",
             "Protocol", "Full released SimpleMemVLA VLM and DiT weights; no training or native-Qwen replacement.",
             "Single-frame semantics: one unique current observation per camera; front is duplicated as",
             "[current, current] only to satisfy the minimum two-frame video patch. Wrist remains a current image.",
             "Original video channel, prompt and generated-subtask -> hidden/embedding -> DiT interface are retained.",
             "Checkpoint: " + str(manifest.get("checkpoint", "pending")),
             "Video accuracy baseline is reused, not rerun here. Source: " + str(provenance.get("source", manifest.get("accuracy", {}).get("video_baseline_source", "pending"))), "",
             "Accuracy (percentage-point drop = video minus current-frame)"]
    for phase, entry in report["accuracy"]["phases"].items():
        lines.append(phase)
        for mode in ("video_baseline", "current_frame"):
            row = entry[mode]
            if row["state"] == "complete":
                lines.append("  {}: {:.2f}% ({}/{})".format(mode, row["percent"], row["successes"], row["n"]))
            else:
                lines.append("  {}: {} ({})".format(mode, row["state"], row.get("reason", "pending")))
        if "drop_percentage_points" in entry:
            lines.append("  Drop: {:.2f} percentage points".format(entry["drop_percentage_points"]))
            for task, row in entry["tasks"].items():
                lines.append("  {}: video {:.2f}% ({}/{}) -> current {:.2f}% ({}/{}); drop {:.2f} pp".format(task, row["video"]["percent"], row["video"]["successes"], row["video"]["n"], row["current_frame"]["percent"], row["current_frame"]["successes"], row["current_frame"]["n"], row["drop_percentage_points"]))
    lines.append("Accuracy launcher state: " + report["accuracy"]["state"])
    lines.extend("ERROR: " + e for e in report["accuracy"]["errors"])
    lines += ["", "Latency", "Model loading and simulator reset/step time are excluded.",
              "Subtasks are generated freely in each mode, so output lengths may differ.",
              "These are actual decision latencies; differences must not all be attributed to visual prefill.",
              "Configured batch size: " + str(lp.get("batch_size", "pending")),
              "Configured warmup calls per mode: " + str(lp.get("warmup_calls_per_mode", "pending")),
              "Configured pairing: " + str(lp.get("pairing", "pending")),
              "Cache rules and actual measured sample counts are recorded below; pending if absent."]
    for split, entry in report["latency"].items():
        lines.append(split + ": " + entry["state"])
        if entry["summary"] is None:
            lines.append("  Actual measured samples, timings and cache rules: pending")
        else:
            value = entry["summary"]
            final = entry["state"] == "complete"
            lines.append("  FINAL" if final else "  IN PROGRESS / INVALID: values below are not final conclusions.")
            lines.append("  Actual measured records: valid={}, invalid={}".format(value.get("records_valid", "pending"), value.get("records_invalid", "pending")))
            lines.append("  Actual group size={}, warmup calls/mode={}".format(value.get("group_size", "pending"), value.get("warmup_per_mode", "pending")))
            lines.append("  cache | video total ms mean/p50/p95 | single-frame total ms mean/p50/p95 | model ms p50 video/SF | samples video/SF | ratio of mean total ms")
            for cache in ("natural", "cold_visual"):
                modes = {r.get("mode"): r for r in value.get("groups", []) if r.get("task") == "ALL" and r.get("cache") == cache}
                paired = [r for r in value.get("paired", []) if r.get("task") == "ALL" and r.get("cache") == cache]
                video, current = modes.get("video", {}), modes.get("single_frame", {})
                def metric(row, block, name):
                    x = row.get(block, {}).get(name)
                    return f"{x:.3f}" if isinstance(x, (int, float)) and math.isfinite(x) else "pending"
                def triplet(row):
                    return "/".join(metric(row, "total_ms", k) for k in ("mean", "p50", "p95"))
                ratio = paired[0].get("ratio_of_mean_total_ms") if len(paired) == 1 else None
                ratio_text = f"{ratio:.3f}x" if isinstance(ratio, (int, float)) and math.isfinite(ratio) else "pending"
                lines.append("  {} | {} | {} | {}/{} | {}/{} | {}".format(cache, triplet(video), triplet(current), metric(video, "model_decision_ms", "p50"), metric(current, "model_decision_ms", "p50"), video.get("n", "pending"), current.get("n", "pending"), ratio_text))
            lines.append("  Ratio is mean(video total_ms)/mean(single-frame total_ms), not the mean of per-pair speedups.")
            lines.append("  Cache/timing protocol: " + json.dumps(value.get("protocol", "pending"), ensure_ascii=False, indent=2, allow_nan=False))
            lines.append("  Per-task metrics, paired samples and episode termination details remain in summary.json.")
        tokens = entry["natural_token_lengths"]
        if tokens["state"] != "available":
            lines.append("  Natural-cache input/generated token lengths: pending")
        else:
            lines.append("  Natural-cache token lengths (valid measured rows only; mean/median, actual n):")
            for mode, fields in tokens["modes"].items():
                parts = []
                for name, stats in fields.items():
                    value = "pending" if stats["n"] == 0 else "{:.2f}/{:g} (n={})".format(stats["mean"], stats["median"], stats["n"])
                    parts.append(name + "=" + value)
                lines.append("    " + mode + ": " + "; ".join(parts))
            if tokens["incomplete_or_unparsed_lines"] or tokens["ignored_invalid_token_rows"]:
                lines.append("  Token diagnostics skipped incomplete/unparsed lines={} and invalid-token rows={}; refresh may resolve a partial final line.".format(tokens["incomplete_or_unparsed_lines"], tokens["ignored_invalid_token_rows"]))
        if entry.get("error"):
            lines.append("ERROR: " + entry["error"])
        if entry["state"] == "failed":
            lines.append("FAILED producer status: " + json.dumps(entry["status"], ensure_ascii=False))
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, content: str):
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def write_report(root, report):
    atomic_write(root / "summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_write(root / "SUMMARY.txt", render_text(report))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval_seconds", type=float, default=30)
    ap.add_argument("--timeout_seconds", type=float, default=6 * 3600)
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        ap.error("Experiment root must already exist")
    if args.interval_seconds <= 0 or not 0 <= args.timeout_seconds <= 6 * 3600:
        ap.error("Positive interval and timeout between zero and six hours required")
    start = time.monotonic()
    while True:
        report = build_report(root)
        if args.watch and not report["terminal"] and time.monotonic() - start >= args.timeout_seconds:
            report.update(state="watch_timeout", watcher_timed_out=True)
            write_report(root, report)
            print(json.dumps({"state": report["state"], "root": str(root)}), flush=True)
            return 3
        write_report(root, report)
        print(json.dumps({"state": report["state"], "root": str(root)}), flush=True)
        if not args.watch or report["terminal"]:
            return 1 if report["state"] == "failed" else 0
        time.sleep(min(args.interval_seconds, max(0.01, args.timeout_seconds - (time.monotonic() - start))))


if __name__ == "__main__":
    raise SystemExit(main())
