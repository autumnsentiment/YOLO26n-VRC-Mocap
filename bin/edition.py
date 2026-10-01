"""Edition detection: full (YOLO + PyTorch + MediaPipe) or lite (MediaPipe only).

The edition follows what is installed, so a lite environment (``uv sync``)
never imports torch/ultralytics. ``YOLO26N_EDITION=lite`` forces the lite
code path even when the full dependencies are present.
"""

import importlib.util
import os
from functools import lru_cache

FULL = "full"
LITE = "lite"
_YOLO_MODULES = ("ultralytics", "torch")


def _requested_edition() -> str:
    value = os.environ.get("YOLO26N_EDITION", "").strip().lower()
    return value if value in (FULL, LITE) else ""


@lru_cache(maxsize=1)
def yolo_installed() -> bool:
    """True when the YOLO stack can be imported (checked without importing it)."""
    for name in _YOLO_MODULES:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def current_edition() -> str:
    if _requested_edition() == LITE:
        return LITE
    return FULL if yolo_installed() else LITE


def yolo_available() -> bool:
    return current_edition() == FULL


def edition_info() -> dict:
    edition = current_edition()
    return {
        "edition": edition,
        "yolo_available": edition == FULL,
        "yolo_installed": yolo_installed(),
        "forced": _requested_edition() or None,
        "body_backends": ["yolo", "mediapipe_pose"] if edition == FULL else ["mediapipe_pose"],
    }