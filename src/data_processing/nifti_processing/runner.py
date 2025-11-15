"""Main runner for NIfTI processing stage."""

from __future__ import annotations

from typing import Dict

# Import only formatter at module level (lightweight, no GPU dependencies)
from .formatter import NiftiFormatter


def run(action: str, cfg: Dict) -> int:
    """
    Main runner for NIfTI processing stage.

    Uses lazy imports to avoid loading all substages at module import time.
    This prevents import-time hanging issues, especially with GPU-related modules.

    Args:
        action: Action to perform (test, process, etc.)
        cfg: Configuration dictionary

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Get substage from configuration
    nifti_cfg = cfg.get("nifti_processing", {})
    substage = nifti_cfg.get("substage", "skull_stripping")

    # Add debug logging if debug mode is enabled
    if cfg.get("debug", False):
        print(f"[NIFTI_RUNNER] Processing substage: {substage}")
        print(f"[NIFTI_RUNNER] Action: {action}")

    # Lazy import and route to appropriate substage
    # This ensures we only load the substage that's actually needed
    if substage == "skull_stripping":
        if cfg.get("debug", False):
            print("[NIFTI_RUNNER] Lazy importing skull_stripping module...")
        from . import skull_stripping
        if cfg.get("debug", False):
            print("[NIFTI_RUNNER] skull_stripping module imported successfully")
        return skull_stripping.run(action, cfg)

    elif substage == "template_registration":
        if cfg.get("debug", False):
            print("[NIFTI_RUNNER] Lazy importing template_registration module...")
        from . import template_registration
        if cfg.get("debug", False):
            print("[NIFTI_RUNNER] template_registration module imported successfully")
        return template_registration.run(action, cfg)

    elif substage == "labelling":
        if cfg.get("debug", False):
            print("[NIFTI_RUNNER] Lazy importing labelling module...")
        from . import labelling
        if cfg.get("debug", False):
            print("[NIFTI_RUNNER] labelling module imported successfully")
        return labelling.run(action, cfg)

    elif substage == "twoD_conversion":
        if cfg.get("debug", False):
            print("[NIFTI_RUNNER] Lazy importing twoD_conversion module...")
        from . import twoD_conversion
        if cfg.get("debug", False):
            print("[NIFTI_RUNNER] twoD_conversion module imported successfully")
        return twoD_conversion.run(action, cfg)

    else:
        # Unknown substage
        formatter = NiftiFormatter(
            verbose=cfg.get("debug", False),
            quiet=cfg.get("quiet", False),
            json_only=cfg.get("json", False)
        )
        formatter.error(f"Unknown substage: {substage}", {
            "cause": f"Invalid substage '{substage}' in configuration",
            "next_steps": "Valid substages: skull_stripping, template_registration, labelling, twoD_conversion"
        })
        formatter.footer(exit_code=2)
        return 2