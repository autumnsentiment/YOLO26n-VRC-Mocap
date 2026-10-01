"""Runtime path helpers for source and PyInstaller builds."""

import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
    return Path(__file__).resolve().parent.parent


def _directory_is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path,
            prefix=".runtime-write-",
            delete=True,
        ):
            pass
        return True
    except OSError:
        return False


def _user_runtime_root() -> Path:
    override = os.environ.get("YOLO26N_RUNTIME_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data).resolve() / "YOLO26n-VRC-Mocap"
    return Path.home() / ".local" / "share" / "YOLO26n-VRC-Mocap"


@lru_cache(maxsize=1)
def runtime_root() -> Path:
    if not getattr(sys, "frozen", False):
        return resource_root()

    portable_root = Path(sys.executable).resolve().parent
    if _directory_is_writable(portable_root):
        return portable_root

    user_root = _user_runtime_root()
    user_root.mkdir(parents=True, exist_ok=True)
    return user_root


def bin_dir() -> Path:
    root = resource_root()
    candidate = root / "bin"
    return candidate if candidate.exists() else Path(__file__).resolve().parent
