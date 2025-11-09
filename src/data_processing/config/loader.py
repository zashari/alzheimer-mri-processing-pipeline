from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from ..utils.errors import ConfigError


try:
    import yaml
except Exception as e:  # pragma: no cover
    yaml = None  # type: ignore


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not yaml:
        raise ConfigError("PyYAML is required for loading config files")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"YAML at {path} must be a mapping")
        return data


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def parse_env_overrides(prefix: str = "ADP_") -> Dict[str, Any]:
    # Expects env vars like ADP_paths.data_root=D:\\data
    overrides: Dict[str, Any] = {}
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        dotted = key[len(prefix) :]
        _apply_override(overrides, dotted, val)
    return overrides


def parse_cli_overrides(pairs: Iterable[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for item in pairs:
        if "=" not in item:
            raise ConfigError(f"Invalid --set format, expected key=value: {item}")
        key, val = item.split("=", 1)
        _apply_override(overrides, key, val)
    return overrides


def _coerce_value(val: str) -> Any:
    # Try to coerce common types from strings
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    if re.fullmatch(r"-?\d+", val):
        try:
            return int(val)
        except Exception:
            pass
    if re.fullmatch(r"-?\d+\.\d+", val):
        try:
            return float(val)
        except Exception:
            pass
    if "," in val:
        return [
            _coerce_value(part.strip()) for part in val.split(",") if part.strip() != ""
        ]
    return val


def _apply_override(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        return
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})  # type: ignore
    cur[parts[-1]] = _coerce_value(value) if isinstance(value, str) else value


def load_config(root: Path, stage: str, user_config: Path | None, stage_config: Path | None,
                cli_overrides: Iterable[str]) -> Dict[str, Any]:
    """
    Load and merge configuration according to the layering spec:
    default -> stage default -> user file -> env -> CLI overrides.
    """
    cfg: Dict[str, Any] = {}

    default_path = root / "configs" / "default.yaml"
    stage_path = stage_config or (root / "configs" / "stages" / f"{stage}.yaml")

    cfg = deep_merge(cfg, _load_yaml(default_path))
    cfg = deep_merge(cfg, _load_yaml(stage_path))
    if user_config:
        cfg = deep_merge(cfg, _load_yaml(user_config))
    cfg = deep_merge(cfg, parse_env_overrides())
    cfg = deep_merge(cfg, parse_cli_overrides(cli_overrides))
    return cfg
