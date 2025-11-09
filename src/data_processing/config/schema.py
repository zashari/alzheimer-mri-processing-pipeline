from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..utils.errors import ValidationError


@dataclass
class GlobalConfig:
    data_root: Path | None = None
    output_root: Path | None = None
    metadata_csv: Path | None = None
    seed: int | None = None
    debug: bool = False
    quiet: bool = False
    dry_run: bool = False
    log_file: str | None = None
    workers: int = 0


@dataclass
class EnvSetupConfig:
    auto_install: bool = False
    cuda_preference: str = "auto"  # e.g., 11.8, 12.1, auto
    perf_test: str = "off"  # off|quick|full


@dataclass
class DataPrepVizConfig:
    enabled: bool = False
    output_dir: str | None = None


@dataclass
class DataPrepConfig:
    required_visits: List[str] = field(default_factory=lambda: ["sc", "m06", "m12"])
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    stratify_by: str | None = "Group"
    shuffle: bool = True
    use_symlinks: bool = False
    debug_mode: bool = False
    visualization: DataPrepVizConfig = field(default_factory=DataPrepVizConfig)


@dataclass
class AppConfig:
    global_: GlobalConfig
    env_setup: EnvSetupConfig
    data_prep: DataPrepConfig


def _get_nested(cfg: Dict[str, Any], path: List[str], default=None):
    cur: Any = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def build_app_config(raw: Dict[str, Any]) -> AppConfig:
    # Global
    g = GlobalConfig(
        data_root=Path(_get_nested(raw, ["paths", "data_root"])) if _get_nested(raw, ["paths", "data_root"]) else None,
        output_root=Path(_get_nested(raw, ["paths", "output_root"])) if _get_nested(raw, ["paths", "output_root"]) else None,
        metadata_csv=Path(_get_nested(raw, ["paths", "metadata_csv"])) if _get_nested(raw, ["paths", "metadata_csv"]) else None,
        seed=_get_nested(raw, ["seed"]),
        debug=bool(_get_nested(raw, ["debug"], False)),
        quiet=bool(_get_nested(raw, ["quiet"], False)),
        dry_run=bool(_get_nested(raw, ["dry_run"], False)),
        log_file=_get_nested(raw, ["log_file"]),
        workers=int(_get_nested(raw, ["workers"], 0) or 0),
    )

    # Env setup
    es = EnvSetupConfig(
        auto_install=bool(_get_nested(raw, ["environment_setup", "auto_install"], False)),
        cuda_preference=str(_get_nested(raw, ["environment_setup", "cuda_preference"], "auto")),
        perf_test=str(_get_nested(raw, ["environment_setup", "perf_test"], "off")),
    )

    # Data prep
    dp_viz = DataPrepVizConfig(
        enabled=bool(_get_nested(raw, ["data_preparation", "visualization", "enabled"], False)),
        output_dir=_get_nested(raw, ["data_preparation", "visualization", "output_dir"]),
    )
    ratios = _get_nested(raw, ["data_preparation", "split_ratios"], [0.7, 0.15, 0.15])
    if isinstance(ratios, (list, tuple)) and len(ratios) == 3:
        ratios_t = (float(ratios[0]), float(ratios[1]), float(ratios[2]))
    else:
        raise ValidationError("split_ratios must be a 3-tuple/list of floats")
    dp = DataPrepConfig(
        required_visits=list(_get_nested(raw, ["data_preparation", "required_visits"], ["sc", "m06", "m12"])),
        split_ratios=ratios_t,
        stratify_by=_get_nested(raw, ["data_preparation", "stratify_by"], "Group"),
        shuffle=bool(_get_nested(raw, ["data_preparation", "shuffle"], True)),
        use_symlinks=bool(_get_nested(raw, ["data_preparation", "use_symlinks"], False)),
        debug_mode=bool(_get_nested(raw, ["data_preparation", "debug_mode"], False)),
        visualization=dp_viz,
    )

    cfg = AppConfig(global_=g, env_setup=es, data_prep=dp)
    validate_config(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    # Keep global paths optional at schema level to allow CLI help/dry-run;
    # Stage runners can enforce stricter checks when needed.
    tr, vr, te = cfg.data_prep.split_ratios
    if not abs((tr + vr + te) - 1.0) < 1e-6:
        raise ValidationError("data_preparation.split_ratios must sum to 1.0")
    if not cfg.data_prep.required_visits:
        raise ValidationError("data_preparation.required_visits cannot be empty")
