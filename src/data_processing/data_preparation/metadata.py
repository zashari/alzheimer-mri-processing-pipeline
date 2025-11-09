from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def load_metadata(path: str | Path, parse_dates: Optional[List[str]] = None):
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required for metadata operations") from e
    df = pd.read_csv(path, parse_dates=parse_dates)
    return df


def ensure_columns(df, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def value_counts_summary(df, cols: List[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for c in cols:
        if c in df.columns:
            out[c] = df[c].value_counts().to_dict()
    return out

