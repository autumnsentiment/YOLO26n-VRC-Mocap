"""Shared runtime helpers for MediaPipe streaming/video tasks."""

import time


def next_timestamp_ms(previous: int = 0) -> int:
    """Return a strictly increasing monotonic timestamp in milliseconds."""
    current = time.monotonic_ns() // 1_000_000
    return max(int(previous) + 1, int(current))
