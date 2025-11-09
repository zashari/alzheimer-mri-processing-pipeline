from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class StageContext:
    name: str
    dry_run: bool = False
    debug: bool = False


class BaseStage:
    name = "base"

    def __init__(self, ctx: StageContext | None = None) -> None:
        self.ctx = ctx or StageContext(self.name)

    def run(self, cfg: Dict, action: str, args: Dict) -> int:  # pragma: no cover
        raise NotImplementedError("Stage must implement run(cfg, action, args)")
