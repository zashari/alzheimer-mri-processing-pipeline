"""Main runner for NIfTI processing stage."""

from __future__ import annotations

from typing import Dict

from . import skull_stripping
from . import template_registration
from . import labelling
from . import twoD_conversion
from .formatter import NiftiFormatter


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