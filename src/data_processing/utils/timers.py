from __future__ import annotations

import contextlib
import time
from typing import Iterator


@contextlib.contextmanager
def timeit(label: str) -> Iterator[None]:
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[TIMER] {label}: {dt:.3f}s")
