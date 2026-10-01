"""VRCFT face calibration helpers."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Mapping, Tuple

from runtime_paths import runtime_root


FACE_STATE_PATH = runtime_root() / "state" / "face_vrcft_state.json"

SIGNED_VRCFT_PARAMS = {
    "v2/JawX",
    "v2/MouthX",
    "v2/EyeLeftX",
    "v2/EyeLeftY",
    "v2/EyeRightX",
    "v2/EyeRightY",
}

PASSTHROUGH_VRCFT_PARAMS = {
    "ExpressionTrackingActive",
    "LipTrackingActive",
}


def default_face_calibration() -> Dict:
    return {
        "enabled": False,
        "min_values": {},
        "max_values": {},
        "updated_at": None,
    }


def normalize_face_calibration(config: Mapping | None) -> Dict:
    normalized = default_face_calibration()
    if not isinstance(config, Mapping):
        return normalized

    normalized["enabled"] = bool(config.get("enabled", False))
    for key in ("min_values", "max_values"):
        values = config.get(key, {})
        if isinstance(values, Mapping):
            normalized[key] = {
                str(name): float(value)
                for name, value in values.items()
                if _is_finite_number(value)
            }
    updated_at = config.get("updated_at")
    normalized["updated_at"] = updated_at if updated_at else None
    return normalized


def apply_vrcft_calibration(vrcft: Mapping[str, float], calibration: Mapping | None) -> Dict[str, float]:
    raw = {str(name): float(value) for name, value in vrcft.items() if _is_finite_number(value)}
    cfg = normalize_face_calibration(calibration)
    if not cfg["enabled"]:
        return dict(raw)

    result = {}
    min_values = cfg["min_values"]
    max_values = cfg["max_values"]
    for name, raw_value in raw.items():
        if name in PASSTHROUGH_VRCFT_PARAMS or not name.startswith("v2/"):
            result[name] = raw_value
            continue
        target_min, target_max = target_range_for_param(name)
        source_min = min_values.get(name, target_min)
        source_max = max_values.get(name, target_max)
        result[name] = remap_with_fallback(
            raw_value=raw_value,
            source_min=source_min,
            source_max=source_max,
            target_min=target_min,
            target_max=target_max,
        )
    return result


def target_range_for_param(name: str) -> Tuple[float, float]:
    if name in SIGNED_VRCFT_PARAMS:
        return -1.0, 1.0
    return 0.0, 1.0


def remap_with_fallback(
    raw_value: float,
    source_min: float,
    source_max: float,
    target_min: float,
    target_max: float,
) -> float:
    if (
        not _is_finite_number(raw_value)
        or not _is_finite_number(source_min)
        or not _is_finite_number(source_max)
        or source_max <= source_min
        or abs(source_max - source_min) < 1e-9
    ):
        return _clamp(float(raw_value), target_min, target_max)

    ratio = (float(raw_value) - float(source_min)) / (float(source_max) - float(source_min))
    mapped = float(target_min) + ratio * (float(target_max) - float(target_min))
    return _clamp(mapped, target_min, target_max)


def write_face_state(vrcft_raw: Mapping[str, float], path: Path | None = None) -> None:
    state_path = path or FACE_STATE_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        str(name): float(value)
        for name, value in vrcft_raw.items()
        if str(name).startswith("v2/") and _is_finite_number(value)
    }
    payload = {
        "timestamp": time.time(),
        "vrcft_raw": values,
    }
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(state_path)


def read_face_state(path: Path | None = None) -> Dict:
    state_path = path or FACE_STATE_PATH
    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    values = data.get("vrcft_raw", {})
    if not isinstance(values, Mapping):
        values = {}
    data["vrcft_raw"] = {
        str(name): float(value)
        for name, value in values.items()
        if str(name).startswith("v2/") and _is_finite_number(value)
    }
    return data


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
