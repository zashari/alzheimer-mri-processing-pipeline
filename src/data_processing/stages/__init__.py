"""Stage registry for pipeline dispatch.

Each concrete stage module should import this module and call
`register(<key>, <StageClass>)` at import time so the CLI can
lookup and invoke the stage via its key.
"""

from __future__ import annotations

from typing import Dict, Optional, Type


class BaseStage:  # forward ref fallback, real BaseStage lives in base.py
    name: str

    def run(self, cfg: dict, action: str, args: dict) -> int:  # pragma: no cover
        raise NotImplementedError


_REGISTRY: Dict[str, Type[BaseStage]] = {}


def register(key: str, stage_cls: Type[BaseStage]) -> None:
    key_norm = key.strip().lower()
    if key_norm in _REGISTRY:
        raise KeyError(f"Stage key already registered: {key_norm}")
    _REGISTRY[key_norm] = stage_cls


def get_stage(key: str) -> Optional[Type[BaseStage]]:
    return _REGISTRY.get(key.strip().lower())


def available_stages() -> Dict[str, Type[BaseStage]]:
    return dict(_REGISTRY)


# Import concrete stages to trigger self-registration at import time
# Keep imports at the end to avoid circulars
try:  # pragma: no cover
    print("[DEBUG STAGES] Importing environment_setup_stage")
    from . import environment_setup_stage  # noqa: F401
    print("[DEBUG STAGES] Importing data_preparation_stage")
    from . import data_preparation_stage  # noqa: F401
    print("[DEBUG STAGES] Importing nifti_processing_stage")
    from . import nifti_processing_stage  # noqa: F401
    print("[DEBUG STAGES] Importing image_processing_stage")
    from . import image_processing_stage  # noqa: F401
    print("[DEBUG STAGES] All stage imports complete")
except Exception as e:
    # It's okay if import fails in some tooling contexts; CLI will import later
    print(f"[DEBUG STAGES] Stage import failed: {e}")
    pass
