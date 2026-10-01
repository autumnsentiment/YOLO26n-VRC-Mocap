#!/usr/bin/env python3
"""Non-destructive dashboard/API smoke checks."""

import json
import sys
import time
import urllib.error
import urllib.request


BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"
PERFORMANCE_TIERS = {"low", "medium", "high", "ultra"}
PERFORMANCE_BASELINE_FIELDS = {
    "camera_width",
    "camera_height",
    "camera_fps",
    "precision",
    "confidence",
    "body_max_fps",
    "face_max_fps",
    "hand_max_fps",
    "stability",
}


def get_json(path: str, timeout: float = 8):
    with urllib.request.urlopen(BASE_URL + path, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"{path}: HTTP {resp.status}")
        return json.loads(resp.read().decode("utf-8"))


def get_json_with_retry(path: str):
    hardware_paths = {"/api/hardware", "/api/performance/recommendation"}
    attempts = 3 if path in {"/api/cameras", *hardware_paths} else 1
    timeout = 45 if path in hardware_paths else (12 if path == "/api/cameras" else 8)
    last_exc = None
    for attempt in range(attempts):
        try:
            return get_json(path, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(2)
    raise last_exc


def validate_performance_recommendation(data: dict):
    tier = data.get("tier")
    if tier not in PERFORMANCE_TIERS:
        raise RuntimeError(
            "/api/performance/recommendation tier must be one of "
            f"{sorted(PERFORMANCE_TIERS)}, got {tier!r}"
        )

    basis = data.get("basis")
    if not isinstance(basis, dict):
        raise RuntimeError("/api/performance/recommendation basis must be an object")

    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        raise RuntimeError("/api/performance/recommendation baseline must be an object")
    missing_fields = sorted(PERFORMANCE_BASELINE_FIELDS - set(baseline))
    if missing_fields:
        raise RuntimeError(
            "/api/performance/recommendation baseline missing "
            f"required fields: {missing_fields}"
        )


def main():
    checks = [
        "/api/config",
        "/api/mocap/status",
        "/api/cameras",
        "/api/hardware",
        "/api/performance/recommendation",
        "/api/hmd/status",
        "/api/tracking/status",
        "/api/face/calibration",
        "/api/models",
        "/api/steamvr/status",
        "/api/vmt/status",
    ]
    failures = []
    for path in checks:
        try:
            data = get_json_with_retry(path)
            if not isinstance(data, dict):
                raise RuntimeError(f"{path} response must be a JSON object")
            if path == "/api/performance/recommendation":
                validate_performance_recommendation(data)
            if path == "/api/hmd/status":
                diagnostics = data.get("height_diagnostics")
                if not isinstance(diagnostics, dict):
                    raise RuntimeError("/api/hmd/status missing height_diagnostics")
                for key in ("standing_eye_height", "height_ratio", "posture", "reason"):
                    if key not in diagnostics:
                        raise RuntimeError(f"/api/hmd/status height_diagnostics missing {key}")
            if path == "/api/vmt/status":
                for key in (
                    "host",
                    "port",
                    "driver_installed",
                    "installer_running",
                    "steamvr_running",
                    "udp_port_listening",
                    "ready",
                    "message",
                ):
                    if key not in data:
                        raise RuntimeError(f"/api/vmt/status missing {key}")
            print(f"OK {path} keys={','.join(sorted(data.keys())[:8])}")
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"FAIL {path}: {exc}")
            failures.append(path)

    if failures:
        print(f"FAILED {len(failures)} dashboard checks")
        return 1
    print("Dashboard smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
