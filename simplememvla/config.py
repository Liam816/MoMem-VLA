"""YAML config files for the train / eval entry points.

A config file only supplies argparse defaults, so every entry point keeps its
CLI: values on the command line override the YAML, and running without
``--config`` behaves exactly as before.

* ``_base_: other.yaml`` (or a list) is merged first, resolved relative to the
  including file; nested mappings merge key by key, everything else replaces.
* ``${VAR}`` / ``${VAR:-default}`` in strings expand from the environment.
* ``--config`` may be repeated; later files override earlier ones.
"""

import argparse
import difflib
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

BASE_KEY = "_base_"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def deep_merge(base: dict, override: Mapping) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _expand_env(value: Any, source: Path) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise KeyError(f"{source}: environment variable ${{{name}}} is not set")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, Mapping):
        return {k: _expand_env(v, source) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v, source) for v in value]
    return value


def load_config(path: str | os.PathLike, _seen: tuple[Path, ...] = ()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, path))
        raise ValueError(f"Circular {BASE_KEY} include: {chain}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    data = dict(data)
    bases = data.pop(BASE_KEY, None) or []
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for base in bases:
        merged = deep_merge(merged, load_config(path.parent / base, (*_seen, path)))
    return deep_merge(merged, _expand_env(data, path))


def load_configs(paths: Iterable[str | os.PathLike]) -> dict:
    merged: dict = {}
    for path in paths:
        merged = deep_merge(merged, load_config(path))
    return merged


def _normalize_key(key: str) -> str:
    return str(key).replace("-", "_")


def flatten_sections(config: Mapping, sections: Mapping[str, Iterable[str]]) -> dict:
    allowed = {name: set(keys) for name, keys in sections.items()}
    flat: dict = {}
    for section, values in config.items():
        if section not in allowed:
            raise ValueError(
                f"Unknown config section {section!r}; expected one of {sorted(allowed)}"
            )
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ValueError(f"Config section {section!r} must be a mapping")
        for key, value in values.items():
            key = _normalize_key(key)
            if key not in allowed[section]:
                hint = difflib.get_close_matches(key, allowed[section], n=1)
                raise ValueError(
                    f"Unknown key {key!r} in config section {section!r}"
                    + (f" (did you mean {hint[0]!r}?)" if hint else "")
                )
            flat[key] = value
    return flat


def _check_choices(action: argparse.Action, value: Any) -> None:
    if action.choices is None or value is None:
        return
    values = value if isinstance(value, list) else [value]
    bad = [v for v in values if v not in action.choices]
    if bad:
        raise ValueError(
            f"Config value {bad} for {action.dest!r} not in choices {list(action.choices)}"
        )


def apply_config_defaults(parser: argparse.ArgumentParser, flat: Mapping[str, Any]) -> None:
    actions: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if action.dest not in (argparse.SUPPRESS, "help", "config"):
            actions.setdefault(action.dest, []).append(action)
    defaults = {}
    for key, value in flat.items():
        key = _normalize_key(key)
        if key not in actions:
            hint = difflib.get_close_matches(key, actions, n=1)
            raise ValueError(
                f"Unknown config key {key!r}"
                + (f" (did you mean {hint[0]!r}?)" if hint else "")
            )
        for action in actions[key]:
            _check_choices(action, value)
            action.required = False
        defaults[key] = value
    parser.set_defaults(**defaults)


def split_config_args(args: list[str] | None = None) -> tuple[list[str], list[str]]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", action="append", default=[])
    known, rest = pre.parse_known_args(args)
    return known.config, rest


def parse_args_with_config(
    parser: argparse.ArgumentParser, args: list[str] | None = None
) -> argparse.Namespace:
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML file(s) supplying defaults for the flags below; CLI flags win. Repeatable.",
    )
    config_paths, rest = split_config_args(args)
    if config_paths:
        apply_config_defaults(parser, load_configs(config_paths))
    namespace = parser.parse_args(rest)
    namespace.config = config_paths
    return namespace


def dump_config(config: Mapping, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False, allow_unicode=True)
