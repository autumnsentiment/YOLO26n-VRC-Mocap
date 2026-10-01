import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from runtime_paths import runtime_root


TRACKING_STATE_PATH = runtime_root() / "state" / "tracking_status.json"
_REPLACE_RETRY_DELAYS = (0.005, 0.015, 0.03, 0.06)


def _json_safe(value: Any):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _replace_with_retry(source: Path, target: Path) -> None:
    """Replace a state file despite short-lived Windows reader locks."""
    for attempt, delay in enumerate((*_REPLACE_RETRY_DELAYS, None)):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def write_tracking_state(state: Dict, path: Optional[Path] = None) -> None:
    target = Path(path or TRACKING_STATE_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(dict(state))
    payload["timestamp"] = float(payload.get("timestamp") or time.time())
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    try:
        _replace_with_retry(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def mark_tracking_stopped(path: Optional[Path] = None) -> None:
    """Persist an explicit offline state while retaining the last diagnostics."""
    target = Path(path or TRACKING_STATE_PATH)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except (OSError, json.JSONDecodeError):
        payload = {}

    payload.pop("age_seconds", None)
    payload.pop("stale", None)
    payload.update({
        "running": False,
        "pid": None,
        "fps": 0.0,
        "timestamp": time.time(),
    })
    payload.setdefault("frame_count", 0)
    write_tracking_state(payload, path=target)


def read_tracking_state(
    path: Optional[Path] = None,
    max_age_seconds: Optional[float] = None,
) -> Optional[Dict]:
    target = Path(path or TRACKING_STATE_PATH)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        age = max(0.0, time.time() - float(payload.get("timestamp", 0.0)))
    except (TypeError, ValueError):
        return None
    payload["age_seconds"] = age
    if max_age_seconds is not None and age > max(0.0, float(max_age_seconds)):
        payload["stale"] = True
    else:
        payload["stale"] = False
    return payload
