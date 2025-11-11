"""Image processing stage wrapper."""

from __future__ import annotations

from typing import Dict

from .base import BaseStage
from . import register
from ..image_processing import runner as img_runner


class ImageProcessingStage(BaseStage):
    """Stage for image processing operations."""

    name = "image_processing"

    def run(self, cfg: Dict, action: str, args: Dict) -> int:
        """
        Run image processing stage.

        Args:
            cfg: Configuration dictionary
            action: Action to perform
            args: CLI arguments

        Returns:
            Exit code
        """
        return img_runner.run(action, cfg)


register("image_processing", ImageProcessingStage)
