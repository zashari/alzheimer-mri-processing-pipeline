"""NIfTI processing stage wrapper."""

from __future__ import annotations

from typing import Dict

from .base import BaseStage
from . import register
from ..nifti_processing import runner as nifti_runner


class NiftiProcessingStage(BaseStage):
    """Stage for NIfTI processing operations."""

    name = "nifti_processing"

    def run(self, cfg: Dict, action: str, args: Dict) -> int:
        """
        Run NIfTI processing stage.

        Args:
            cfg: Configuration dictionary
            action: Action to perform
            args: CLI arguments

        Returns:
            Exit code
        """
        return nifti_runner.run(action, cfg)


register("nifti_processing", NiftiProcessingStage)