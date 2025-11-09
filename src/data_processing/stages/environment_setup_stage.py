from __future__ import annotations

from typing import Dict

from .base import BaseStage
from . import register
from ..environment_setup import runner as env_runner


class EnvironmentSetupStage(BaseStage):
    name = "environment_setup"

    def run(self, cfg: Dict, action: str, args: Dict) -> int:
        return env_runner.run(action, cfg)


register("environment_setup", EnvironmentSetupStage)
