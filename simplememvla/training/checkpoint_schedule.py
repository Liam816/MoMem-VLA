"""Save evenly spaced training stages through the normal Trainer checkpoint path.

The grid uses optimizer updates, including in epoch-based training.  It is built
from TrainerState.max_steps after Trainer has resolved the real training budget.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, localcontext
import json
from numbers import Integral, Real
from pathlib import Path
import shutil

from transformers import TrainerCallback


def validate_checkpoint_schedule(first_fraction, count) -> bool:
    """Validate the optional pair and return whether stage saving is enabled."""
    if first_fraction is None and count is None:
        return False
    if first_fraction is None or count is None:
        raise ValueError(
            "save_first_fraction and save_num_checkpoints must be supplied together"
        )
    if isinstance(first_fraction, bool) or not isinstance(first_fraction, (Real, Decimal)):
        raise ValueError("save_first_fraction must be a finite number in (0, 1]")
    try:
        fraction = Decimal(str(first_fraction))
    except (InvalidOperation, ValueError):
        raise ValueError("save_first_fraction must be a finite number in (0, 1]") from None
    if not fraction.is_finite() or not Decimal(0) < fraction <= Decimal(1):
        raise ValueError("save_first_fraction must be a finite number in (0, 1]")
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ValueError("save_num_checkpoints must be a positive integer")
    return True


def build_checkpoint_steps(total_steps: int, first_fraction, count) -> list[int]:
    """Build exactly ``count`` distinct updates, including the first and last.

    The first update is ceil(total * fraction), using decimal input semantics.
    Interior linspace positions use nearest-integer rounding with ties upward.
    Impossible requests raise rather than silently dropping duplicate stages.
    """
    if isinstance(total_steps, bool) or not isinstance(total_steps, Integral) or total_steps < 1:
        raise ValueError("total optimizer steps must be a positive integer")
    if not validate_checkpoint_schedule(first_fraction, count):
        return []
    total_steps, count = int(total_steps), int(count)
    fraction = Decimal(str(first_fraction))
    with localcontext() as context:
        context.prec = max(28, len(str(total_steps)) + len(str(fraction)) + 5)
        first = int((Decimal(total_steps) * fraction).to_integral_value(rounding=ROUND_CEILING))
    if count == 1:
        if first != total_steps:
            raise ValueError(
                "save_num_checkpoints=1 cannot include both the requested first stage "
                f"({first}) and final update ({total_steps}); use at least 2 checkpoints "
                "or choose a first fraction whose first stage is the final update"
            )
        return [total_steps]
    available = total_steps - first + 1
    if count > available:
        raise ValueError(
            f"Cannot save {count} distinct checkpoints between updates {first} and "
            f"{total_steps}; at most {available} distinct updates are available"
        )
    intervals = count - 1
    span = total_steps - first
    return [first + (2 * index * span + intervals) // (2 * intervals) for index in range(count)]


class UniformCheckpointCallback(TrainerCallback):
    """Replace periodic/epoch checkpoint triggers with a fixed stage grid.

    Register after DefaultFlowCallback.  Checkpoint serialization, distributed
    participation, optimizer/RNG saving, and checkpoint rotation stay in Trainer.
    """

    def __init__(self):
        self.enabled = False
        self.checkpoint_steps: list[int] = []
        self._step_set: set[int] = set()
        self._saved_steps: set[int] = set()
        self._plan = None

    @staticmethod
    def _is_main(args, state) -> bool:
        return bool(getattr(state, "is_world_process_zero", getattr(args, "process_index", 0) == 0))

    @staticmethod
    def _write_json(path: Path, value) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def on_train_begin(self, args, state, control, **kwargs):
        first_fraction = getattr(args, "save_first_fraction", None)
        count = getattr(args, "save_num_checkpoints", None)
        self.enabled = validate_checkpoint_schedule(first_fraction, count)
        self.checkpoint_steps = []
        self._step_set = set()
        self._saved_steps = set()
        self._plan = None
        if not self.enabled:
            return control
        strategy = getattr(getattr(args, "save_strategy", None), "value", getattr(args, "save_strategy", None))
        if strategy == "best" or getattr(args, "load_best_model_at_end", False):
            raise ValueError(
                "Uniform checkpoint saving does not support save_strategy=best or "
                "load_best_model_at_end; use independent evaluation and stage saving"
            )
        self.checkpoint_steps = build_checkpoint_steps(state.max_steps, first_fraction, count)
        self._step_set = set(self.checkpoint_steps)
        start = int(state.global_step)
        past = [step for step in self.checkpoint_steps if step <= start]
        future = [step for step in self.checkpoint_steps if step > start]
        # A resume never recreates an old weight snapshot or saves it under an
        # earlier step number.  Normal future on_step_end events drive saving.
        self._saved_steps = set(past)
        if self._is_main(args, state):
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            path = output / "checkpoint_schedule.json"
            moment = datetime.now(timezone.utc)
            previous = None
            if path.exists():
                try:
                    previous = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    # Preserve an unreadable old plan too; it is diagnostic data.
                    previous = {}
                if previous.get("total_optimizer_steps") != int(state.max_steps) or previous.get("checkpoint_steps") != self.checkpoint_steps:
                    history = output / "checkpoint_schedule_history"
                    history.mkdir(exist_ok=True)
                    archive = history / f"before_step{start}_{moment.strftime('%Y%m%dT%H%M%S%fZ')}.json"
                    shutil.copy2(path, archive)
            self._plan = {
                "version": 1,
                "created_utc": moment.isoformat(),
                "total_optimizer_steps": int(state.max_steps),
                "first_fraction": float(first_fraction),
                "num_checkpoints": int(count),
                "effective_save_total_limit": getattr(args, "save_total_limit", None),
                "retention_mode": "project protects scheduled checkpoints; native rotation temporarily disabled during serialization",
                "checkpoint_steps": self.checkpoint_steps,
                "start_step": start,
                "resume_step": start if start else None,
                "current_step": start,
                "past_or_current_steps": past,
                "skipped_past_or_current_steps": past,
                "existing_past_checkpoints": [step for step in past if (output / f"checkpoint-{step}").is_dir()],
                "missing_past_checkpoints": [step for step in past if not (output / f"checkpoint-{step}").is_dir()],
                "future_steps": future,
                "saved_this_run_steps": [],
                "remaining_future_steps": future,
                "notes": [
                    "Steps are absolute optimizer updates from TrainerState.max_steps, including epoch-based training.",
                    "The first stage uses decimal ceil; interior positions use nearest-integer half-up rounding.",
                    "Past stages are not backfilled on resume; their historical weights cannot be recreated.",
                    "Only checkpoint timing is overridden; normal Trainer/DeepSpeed serialization is used.",
                ],
            }
            self._write_json(path, self._plan)
            print(
                f"[checkpoint_schedule] total_optimizer_steps={state.max_steps}; "
                f"stages={self.checkpoint_steps}; start_step={start}; future={future}",
                flush=True,
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.enabled:
            step = int(state.global_step)
            control.should_save = step in self._step_set and step not in self._saved_steps
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.enabled:
            # on_step_end already handled any stage at this epoch boundary.
            control.should_save = False
        return control

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled:
            return control
        step = int(state.global_step)
        self._saved_steps.add(step)
        if self._is_main(args, state) and self._plan is not None:
            self._plan["current_step"] = step
            self._plan["last_saved_step"] = step
            self._plan["saved_this_run_steps"].append(step)
            self._plan["remaining_future_steps"] = [s for s in self.checkpoint_steps if s > step]
            self._write_json(Path(args.output_dir) / "checkpoint_schedule.json", self._plan)
        return control
