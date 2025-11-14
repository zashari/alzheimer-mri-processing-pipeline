from __future__ import annotations

print("[DEBUG CLI] Starting CLI module import")

import argparse
import sys
from pathlib import Path
from typing import Dict

print("[DEBUG CLI] About to import config.loader")
from .config.loader import load_config
print("[DEBUG CLI] config.loader imported")

print("[DEBUG CLI] About to import config.schema")
from .config.schema import AppConfig, build_app_config
print("[DEBUG CLI] config.schema imported")

print("[DEBUG CLI] About to import logging_setup")
from .logging_setup import setup_logging
print("[DEBUG CLI] logging_setup imported")

print("[DEBUG CLI] About to import stages")
from .stages import available_stages, get_stage
print("[DEBUG CLI] stages imported")

print("[DEBUG CLI] About to import utils.randomness")
from .utils.randomness import set_seed
print("[DEBUG CLI] All imports complete")


def _root() -> Path:
    # Project root is two levels up from src/data_processing/cli.py
    # .../alzheimer-disease-processing-py-format/src/data_processing/cli.py
    # parents[0]=.../src/data_processing, [1]=.../src, [2]=.../alzheimer-disease-processing-py-format
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="adp", description="Alzheimer data processing CLI")
    sub = p.add_subparsers(dest="stage", required=False)

    # Global options
    p.add_argument("--config", dest="config", help="User config file")
    p.add_argument("--stage-config", dest="stage_config", help="Override stage config path")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="Key=Value overrides (repeatable)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--log-file")
    p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--version", action="store_true")

    # Stage-specific options (basic; stages can read from cfg instead)
    # environment_setup
    env = sub.add_parser("environment_setup", help="Environment setup stage")
    env.add_argument("action", choices=["setup", "verify"])
    env.add_argument("--auto-install", dest="auto_install", choices=["true", "false"], default=None)
    env.add_argument("--cuda-preference", dest="cuda_preference", default=None)
    env.add_argument("--perf-test", dest="perf_test", choices=["off", "quick", "full"], default=None)

    # data_preparation
    prep = sub.add_parser("data_preparation", help="Data preparation stage")
    prep.add_argument("action", choices=["analyze", "split", "manifests"])
    prep.add_argument("--split-ratios", dest="split_ratios", default=None)
    prep.add_argument("--required-visits", dest="required_visits", default=None)
    prep.add_argument("--stratify-by", dest="stratify_by", default=None)
    prep.add_argument("--shuffle", dest="shuffle", choices=["true", "false"], default=None)
    prep.add_argument("--show-all", dest="show_all", choices=["true", "false"], default="true", help="Show all items in analyze output (default: true)")

    # nifti_processing
    nifti = sub.add_parser("nifti_processing", help="NIfTI processing stage")
    nifti.add_argument("action", choices=["test", "process"])
    nifti.add_argument("--substage", dest="substage",
                      choices=["skull_stripping", "template_registration", "labelling", "twoD_conversion"],
                      default="skull_stripping", help="Substage to execute (default: skull_stripping)")
    nifti.add_argument("--device", dest="device", choices=["cuda", "cpu", "mps"], default=None)
    nifti.add_argument("--use-tta", dest="use_tta", choices=["true", "false"], default=None)

    # image_processing
    img = sub.add_parser("image_processing", help="Image processing stage")
    img.add_argument("action", choices=["test", "process"])
    img.add_argument("--substage", dest="substage",
                     choices=["center_crop", "image_enhancement", "data_balancing"],
                     default="center_crop", help="Substage to execute (default: center_crop)")

    return p


def _apply_stage_cli_options(raw_cfg: Dict, args: argparse.Namespace) -> Dict:
    # translate some stage CLI flags into config overrides (low precedence vs --set)
    def set_in(path: list[str], value):
        cur = raw_cfg
        for p in path[:-1]:
            cur = cur.setdefault(p, {})
        cur[path[-1]] = value

    if getattr(args, "auto_install", None) is not None:
        set_in(["environment_setup", "auto_install"], args.auto_install == "true")
    if getattr(args, "cuda_preference", None):
        set_in(["environment_setup", "cuda_preference"], args.cuda_preference)
    if getattr(args, "perf_test", None):
        set_in(["environment_setup", "perf_test"], args.perf_test)

    if getattr(args, "split_ratios", None):
        parts = [float(x) for x in args.split_ratios.split(",")]
        set_in(["data_preparation", "split_ratios"], parts)
    if getattr(args, "required_visits", None):
        parts = [x.strip() for x in args.required_visits.split(",") if x.strip()]
        set_in(["data_preparation", "required_visits"], parts)
    if getattr(args, "stratify_by", None):
        set_in(["data_preparation", "stratify_by"], args.stratify_by)
    if getattr(args, "shuffle", None) is not None:
        set_in(["data_preparation", "shuffle"], args.shuffle == "true")

    # nifti_processing options
    if getattr(args, "substage", None):
        # Check if this is nifti_processing stage by checking if device/use_tta exist
        if hasattr(args, "device") or hasattr(args, "use_tta"):
            set_in(["nifti_processing", "substage"], args.substage)
        # Otherwise check if stage is image_processing
        elif args.stage == "image_processing":
            set_in(["image_processing", "substage"], args.substage)
    if getattr(args, "device", None):
        set_in(["nifti_processing", "skull_stripping", "device"], args.device)
    if getattr(args, "use_tta", None) is not None:
        set_in(["nifti_processing", "skull_stripping", "use_tta"], args.use_tta == "true")

    # Globals
    if getattr(args, "dry_run", False):
        raw_cfg["dry_run"] = True
    raw_cfg["debug"] = bool(args.debug)
    raw_cfg["quiet"] = bool(args.quiet)
    if args.log_file:
        raw_cfg["log_file"] = args.log_file
    if args.seed is not None:
        raw_cfg["seed"] = args.seed
    if args.workers is not None:
        raw_cfg["workers"] = args.workers
    return raw_cfg


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version and not args.stage:
        print("adp (alzheimer data processing) - dev")
        return 0

    if not args.stage:
        parser.print_help()
        return 2

    stage = args.stage
    action = getattr(args, "action", None) or args.action
    stage_cls = get_stage(stage)
    if not stage_cls:
        print(f"Unknown stage: {stage}. Available: {list(available_stages().keys())}")
        return 2

    root = _root()
    user_cfg = Path(args.config) if args.config else None
    stage_cfg = Path(args.stage_config) if args.stage_config else None

    raw_cfg = load_config(root, stage, user_cfg, stage_cfg, args.overrides)
    raw_cfg = _apply_stage_cli_options(raw_cfg, args)

    logger = setup_logging(debug=bool(raw_cfg.get("debug")), quiet=bool(raw_cfg.get("quiet")), log_file=raw_cfg.get("log_file"))
    logger.debug("Resolved config: %s", raw_cfg)

    # Minimal validation via schema builder (raises on error)
    try:
        _ = build_app_config(raw_cfg)
    except Exception as e:
        logger.error("Invalid configuration: %s", e)
        return 2

    # Seed
    set_seed(raw_cfg.get("seed"))

    # Dispatch
    runner = stage_cls()
    return runner.run(raw_cfg, action, vars(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
