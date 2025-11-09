from __future__ import annotations

from typing import Dict

from .base import BaseStage
from . import register
from ..data_preparation import runner as prep_runner


class DataPreparationStage(BaseStage):
    name = "data_preparation"

    def run(self, cfg: Dict, action: str, args: Dict) -> int:
        # Pass show_all flag from args to config if present
        # Convert string "true"/"false" to boolean, default to True if not specified
        show_all_str = args.get("show_all", "true")
        cfg["show_all"] = show_all_str == "true"
        return prep_runner.run(action, cfg)


register("data_preparation", DataPreparationStage)
