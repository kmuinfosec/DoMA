"""Latency measurement helpers for the timings recorded in ``metrics.json``.

Latency rule: index construction is *registration*, not inference. Its cost is recorded in a
separate field and never added to the inference latency.
    inference latency = query encoding + index search
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


# Holder filled by timer(); .sec carries the elapsed seconds once the block exits.
@dataclass
class _Elapsed:
    sec: float = 0.0


# Context manager measuring the wall-clock duration of a block: `with timer() as t: ...` -> `t.sec`.
@contextmanager
def timer():
    e = _Elapsed()
    start = perf_counter()
    try:
        yield e
    finally:
        e.sec = perf_counter() - start
