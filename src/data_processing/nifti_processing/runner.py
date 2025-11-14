"""Main runner for NIfTI processing stage."""

from __future__ import annotations

from typing import Dict

print("[DEBUG NIFTI_RUNNER] Starting imports")
print("[DEBUG NIFTI_RUNNER] Importing skull_stripping")
from . import skull_stripping
print("[DEBUG NIFTI_RUNNER] skull_stripping imported")
print("[DEBUG NIFTI_RUNNER] Importing template_registration")
from . import template_registration
print("[DEBUG NIFTI_RUNNER] template_registration imported")
print("[DEBUG NIFTI_RUNNER] Importing labelling")
from . import labelling
print("[DEBUG NIFTI_RUNNER] labelling imported")
print("[DEBUG NIFTI_RUNNER] Importing twoD_conversion")
from . import twoD_conversion
print("[DEBUG NIFTI_RUNNER] twoD_conversion imported")
print("[DEBUG NIFTI_RUNNER] Importing formatter")
from .formatter import NiftiFormatter
print("[DEBUG NIFTI_RUNNER] All imports complete")


def run(action: str, cfg: Dict) -> int:
    """
    Main runner for NIfTI processing stage.

    Args:
        action: Action to perform (test, process, etc.)
        cfg: Configuration dictionary

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Get substage from configuration
    nifti_cfg = cfg.get("nifti_processing", {})
    substage = nifti_cfg.get("substage", "skull_stripping")

    # Route to appropriate substage
    if substage == "skull_stripping":
        return skull_stripping.run(action, cfg)

    elif substage == "template_registration":
        return template_registration.run(action, cfg)

    elif substage == "labelling":
        return labelling.run(action, cfg)

    elif substage == "twoD_conversion":
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