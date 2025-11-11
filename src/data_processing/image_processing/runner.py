"""Main runner for image processing stage."""

from __future__ import annotations

from typing import Dict

from . import center_crop
from .formatter import ImageProcessingFormatter


def run(action: str, cfg: Dict) -> int:
    """
    Main runner for image processing stage.

    Args:
        action: Action to perform (test, process, etc.)
        cfg: Configuration dictionary

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Get substage from configuration
    img_cfg = cfg.get("image_processing", {})
    substage = img_cfg.get("substage", "center_crop")

    # Route to appropriate substage
    if substage == "center_crop":
        return center_crop.run(action, cfg)
    elif substage == "image_enhancement":
        from . import image_enhancement
        return image_enhancement.run(action, cfg)
    elif substage == "data_balancing":
        from . import data_balancing
        return data_balancing.run(action, cfg)
    else:
        # Unknown substage
        formatter = ImageProcessingFormatter(
            verbose=cfg.get("debug", False),
            quiet=cfg.get("quiet", False),
            json_only=cfg.get("json", False)
        )
        formatter.error(f"Unknown substage: {substage}", {
            "cause": f"Invalid substage '{substage}' in configuration",
            "next_steps": "Valid substages: center_crop, image_enhancement, data_balancing"
        })
        formatter.footer(exit_code=2)
        return 2

