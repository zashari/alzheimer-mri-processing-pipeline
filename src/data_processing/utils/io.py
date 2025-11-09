from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(rows: Iterable[dict], out_path: str | Path) -> None:
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required to write CSV outputs") from e
    df = pd.DataFrame(list(rows))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def read_csv(path: str | Path, usecols: Optional[List[str]] = None):
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required to read CSV inputs") from e
    return pd.read_csv(path, usecols=usecols)
