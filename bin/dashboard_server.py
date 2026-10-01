#!/usr/bin/env python3
"""
YOLO26n VRC Mocap Web Dashboard

提供 Web 前端控制面板，支持:
- 查看和切换推理硬件 (GPU/CPU)
- 修改嘴部识别平滑参数
- 修改 YOLO 推理参数
- 启动/停止动捕程序
- 实时状态监控
"""

import os
import sys
import json
import csv
import copy
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path

# 尝试导入 psutil 用于进程管理
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from flask import Flask, jsonify, request, send_from_directory
from typing import Optional

from face_calibration import (
    FACE_STATE_PATH,
    default_face_calibration,
    normalize_face_calibration,
    read_face_state,
)
from runtime_status import mark_tracking_stopped, read_tracking_state
from edition import edition_info, yolo_available
from runtime_paths import resource_root, runtime_root
from streamers.coord import hmd_pose_anchor_trusted
from vmt_runtime import (
    VMT_INSTALLER_NAME,
    download_vmt_installer,
    find_vmt_manager,
    get_vmt_runtime_status,
    launch_elevated,
    query_openvr_devices,
    query_vmt_devices,
    vmt_installer_is_running,
)

# 项目根目录
PROJECT_ROOT = resource_root()
RUNTIME_ROOT = runtime_root()
CONFIG_PATH = RUNTIME_ROOT / "config" / "config.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
STATIC_DIR = Path(__file__).parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = PROJECT_ROOT / "bin" / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

_dashboard_runtime = {"host": "127.0.0.1", "port": 8766}

# 全局状态
mocap_process = None
mocap_status = {"running": False, "pid": None, "start_time": None, "starting": False}

# HMD 位姿数据提供者 (独立于动捕进程，由 dashboard 直接管理)
_hmd_provider = None
_hmd_lock = threading.Lock()

# 配置文件读写锁，避免 Windows 下多线程读写冲突
_config_lock = threading.RLock()
_config_cache_signature = None
_active_config_path = None
_config_cache = None  # 内存缓存，避免频繁读文件导致 Windows 文件锁
_hardware_cache = {"data": None, "ts": 0.0}
_performance_recommendation_cache = {"data": None, "ts": 0.0}
_performance_recommendation_lock = threading.Lock()
_camera_cache = {"data": None, "ts": 0.0}
_camera_lock = threading.Lock()
_mocap_process_cache = {"data": [], "ts": 0.0}
_model_cache = {"data": None, "ts": 0.0, "current": None}
_inference_status_cache = {"data": None, "ts": 0.0, "signature": None}
CACHE_TTL_SECONDS = 15.0
MOCAP_SCAN_TTL_SECONDS = 15.0
MODEL_CACHE_TTL_SECONDS = 30.0
INFERENCE_STATUS_TTL_SECONDS = 1.0
PERFORMANCE_RECOMMENDATION_TTL_SECONDS = 60.0
LOG_TAIL_BYTES = 256 * 1024

UI_SECTION_IDS = frozenset({
    "quickSection",
    "hardwareSection",
    "bodySection",
    "faceSection",
    "handSection",
    "streamingSection",
    "hmdSection",
    "logSection",
    "modelSection",
})

PERFORMANCE_TIER_LABELS = {
    "low": "基础",
    "medium": "均衡",
    "high": "高性能",
    "ultra": "极致",
}

PERFORMANCE_BASELINES = {
    "low": {
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 24,
        "precision": "fp32",
        "confidence": 0.55,
        "body_max_fps": 12,
        "face_max_fps": 15,
        "hand_max_fps": 12,
        "stability": 80,
    },
    "medium": {
        "camera_width": 960,
        "camera_height": 540,
        "camera_fps": 30,
        "precision": "fp16",
        "confidence": 0.5,
        "body_max_fps": 18,
        "face_max_fps": 24,
        "hand_max_fps": 18,
        "stability": 65,
    },
    "high": {
        "camera_width": 1280,
        "camera_height": 720,
        "camera_fps": 30,
        "precision": "fp16",
        "confidence": 0.5,
        "body_max_fps": 24,
        "face_max_fps": 30,
        "hand_max_fps": 24,
        "stability": 55,
    },
    "ultra": {
        "camera_width": 1920,
        "camera_height": 1080,
        "camera_fps": 30,
        "precision": "fp16",
        "confidence": 0.45,
        "body_max_fps": 30,
        "face_max_fps": 30,
        "hand_max_fps": 30,
        "stability": 45,
    },
}


def _read_text_tail(path: Path, max_bytes: int = LOG_TAIL_BYTES) -> str:
    try:
        size = path.stat().st_size
        with open(str(path), "rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
                data = f.read()
                first_newline = data.find(b"\n")
                if first_newline >= 0:
                    data = data[first_newline + 1:]
            else:
                data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _read_log_tail_lines(path: Path, tail: int = 50) -> tuple[str, int]:
    text = _read_text_tail(path)
    lines = text.splitlines(True)
    recent = lines[-tail:] if len(lines) > tail else lines
    return "".join(recent), len(lines)


def kill_stale_mocap_processes() -> int:
    """
    检测并终止所有残留的动捕进程（main.py）
    跳过当前管理的 mocap_process，避免误杀正在运行的进程。
    
    Returns:
        int: 终止的进程数量
    """
    killed_count = 0
    if not HAS_PSUTIL:
        return killed_count
    
    current_pid = os.getpid()
    managed_pid = mocap_process.pid if mocap_process and mocap_process.poll() is None else None
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.pid == current_pid:
                    continue
                if managed_pid is not None and proc.pid == managed_pid:
                    continue
                
                proc_name = proc.name().lower()
                if (
                    'python' not in proc_name
                    and 'python.exe' not in proc_name
                    and 'yolo26n-vrc-mocap' not in proc_name
                ):
                    continue
                
                cmdline = proc.cmdline()
                if not any('main.py' in arg or '--mocap-child' in arg for arg in cmdline):
                    continue

                if any('dashboard' in arg for arg in cmdline):
                    continue

                print(f"发现残留动捕进程 PID: {proc.pid}，正在终止...")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        print(f"检测残留进程时出错: {e}")
    
    return killed_count


def _fallback_config_path() -> Path:
    import tempfile

    return Path(tempfile.gettempdir()) / "vrc-mocap" / "config.json"


def _write_config_path(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        for delay in (0.005, 0.015, 0.03, 0.06, None):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if delay is None:
                    raise
                time.sleep(delay)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _config_signature(path: Optional[Path]):
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _config_candidates():
    paths = (CONFIG_PATH, _fallback_config_path(), DEFAULT_CONFIG_PATH)
    unique = []
    seen = set()
    for path in paths:
        resolved = str(Path(path).resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(Path(path))
    return unique


def _load_config_unlocked() -> dict:
    global _config_cache, _config_cache_signature, _active_config_path
    if _config_cache is not None:
        current_signature = _config_signature(_active_config_path)
        if current_signature is not None and current_signature == _config_cache_signature:
            return _config_cache

    runtime_candidates = []
    for path in (CONFIG_PATH, _fallback_config_path()):
        path = Path(path)
        if path.exists() and path not in runtime_candidates:
            runtime_candidates.append(path)
    if runtime_candidates:
        candidates = runtime_candidates
    else:
        candidates = [
            path for path in _config_candidates()
            if path.exists()
        ]
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            if not isinstance(loaded, dict):
                continue
            _config_cache = loaded
            _active_config_path = candidate
            _config_cache_signature = _config_signature(candidate)
            return _config_cache
        except (OSError, json.JSONDecodeError):
            continue

    _config_cache = {}
    _active_config_path = CONFIG_PATH
    _config_cache_signature = None
    return _config_cache


def load_config() -> dict:
    with _config_lock:
        return copy.deepcopy(_load_config_unlocked())


def active_config_path() -> Path:
    with _config_lock:
        _load_config_unlocked()
        return Path(_active_config_path or CONFIG_PATH)


def _deep_merge_config(target: dict, patch: dict) -> dict:
    """Recursively merge JSON objects; arrays and scalar values replace."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_config(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def ensure_face_calibration_config(config: dict) -> dict:
    face_cfg = config.setdefault("face", {})
    face_cfg.setdefault("mouth_shape_mapping_enabled", True)
    face_cfg.setdefault("mouth_shape_gain", 1.25)
    face_cfg.setdefault("mouth_shape_deadzone", 0.03)
    face_cfg.setdefault("eyelid_independent_enabled", True)
    face_cfg.setdefault("eyelid_close_gain", 1.35)
    face_cfg.setdefault("eyelid_close_deadzone", 0.04)
    face_cfg.setdefault("eyelid_send_combined", False)
    face_cfg["vrcft_calibration"] = normalize_face_calibration(
        face_cfg.get("vrcft_calibration", default_face_calibration())
    )
    return face_cfg["vrcft_calibration"]


def ensure_processing_config(config: dict) -> dict:
    processing = config.setdefault("processing", {})
    processing.setdefault("module_rate_limit_enabled", True)
    processing.setdefault("reuse_last_result", True)
    processing.setdefault("parallel_inference_enabled", True)
    processing.setdefault("body_max_fps", 20)
    processing.setdefault("hmd_pose_hold_seconds", 5.0)
    processing.setdefault("hmd_pose_refresh_fps", 15.0)
    processing.setdefault("face_max_fps", 30)
    processing.setdefault("hand_max_fps", 20)
    processing.setdefault("dual_camera_body_enabled", False)
    processing.setdefault("secondary_body_max_fps", 12)
    processing.setdefault("dual_camera_fusion_mode", "confidence")
    processing.setdefault("fusion_confidence_margin", 0.12)
    processing.setdefault("fusion_min_confidence", 0.15)
    processing.setdefault("fusion_max_pose_delta", 0.75)
    return processing


def ensure_hand_streaming_config(config: dict) -> dict:
    hand_streaming = config.setdefault("streaming", {}).setdefault("hand", {})
    hand_streaming.setdefault("host", config.get("streaming", {}).get("host", "127.0.0.1"))
    hand_streaming.setdefault("port", config.get("streaming", {}).get("port", 9000))
    hand_streaming.setdefault("vmt_local_mode", True)
    if hand_streaming.get("vmt_local_mode", True):
        hand_streaming["vmt_host"] = "127.0.0.1"
    else:
        hand_streaming.setdefault("vmt_host", hand_streaming.get("host", "127.0.0.1"))
    hand_streaming.setdefault("vmt_port", 39570)
    hand_streaming.setdefault("streaming_enabled", True)
    hand_streaming.setdefault("desktop_osc_enabled", True)
    hand_streaming.setdefault("vmt_enabled", True)
    hand_streaming.setdefault("vmt_skeleton_enabled", True)
    hand_streaming.setdefault("vmt_input_enabled", True)
    hand_streaming.setdefault("vmt_auto_enable_skeleton", True)
    hand_streaming.setdefault("vmt_register_on_steamvr_startup", True)
    hand_streaming.setdefault("vmt_keep_controllers_active", True)
    try:
        priority = int(hand_streaming.get("vmt_hand_selection_priority", 100))
    except (TypeError, ValueError):
        priority = 100
    hand_streaming["vmt_hand_selection_priority"] = max(0, min(1000, priority))
    hand_streaming.setdefault("vmt_left_device_index", 1)
    hand_streaming.setdefault("vmt_right_device_index", 2)
    hand_streaming.setdefault("custom_parameters_enabled", True)
    hand_streaming.setdefault("lost_reset_timeout", 0.35)
    hand_streaming.setdefault("desktop_left_gesture_parameters", [
        "DynamicGesture/CurrentGestureLeft",
        "DynamicGesture/CurrentLeftIndex",
        "MocapGestureLeft",
    ])
    hand_streaming.setdefault("desktop_right_gesture_parameters", [
        "DynamicGesture/CurrentGestureRight",
        "DynamicGesture/CurrentRightIndex",
        "MocapGestureRight",
    ])
    return hand_streaming


def ensure_secondary_camera_config(config: dict) -> dict:
    secondary = config.setdefault("camera_secondary", {})
    secondary.setdefault("enabled", False)
    secondary.setdefault("camera_id", 1)
    secondary.setdefault("width", config.get("camera", {}).get("width", 640))
    secondary.setdefault("height", config.get("camera", {}).get("height", 480))
    secondary.setdefault("fps", config.get("camera", {}).get("fps", 30))
    secondary.setdefault("flip_horizontal", config.get("camera", {}).get("flip_horizontal", True))
    secondary.setdefault("zoom", 0)
    return secondary


def ensure_ui_config(config: dict) -> dict:
    ui = config.get("ui")
    if not isinstance(ui, dict):
        ui = {}
        config["ui"] = ui

    presets = ui.get("quick_presets")
    if not isinstance(presets, list):
        presets = []

    normalized = []
    preset_ids = set()
    for item in presets[:20]:
        if not isinstance(item, dict) or not isinstance(item.get("values"), dict):
            continue
        preset_id = "".join(
            char for char in str(item.get("id") or "")
            if char.isalnum() or char in "-_"
        )[:64]
        name = str(item.get("name") or "").strip()[:40]
        if not preset_id or not name or preset_id in preset_ids:
            continue

        values = {}
        for key, value in list(item["values"].items())[:64]:
            key = str(key)[:64]
            if not key:
                continue
            if isinstance(value, bool):
                values[key] = value
            elif isinstance(value, (int, float)) and math.isfinite(value):
                values[key] = value
            elif isinstance(value, str):
                values[key] = value[:128]

        normalized.append({
            "id": preset_id,
            "name": name,
            "values": values,
            "updated_at": str(item.get("updated_at") or "")[:40],
        })
        preset_ids.add(preset_id)

    active = str(ui.get("active_quick_preset") or "manual")[:80]
    builtin_ids = {
        "builtin:latency",
        "builtin:balanced",
        "builtin:precision",
        "builtin:stable",
    }
    if active.startswith("custom:"):
        if active[len("custom:"):] not in preset_ids:
            active = "manual"
    elif active not in builtin_ids and active != "manual":
        active = "manual"

    ui["quick_preset_schema"] = 1
    ui["active_quick_preset"] = active
    ui["quick_presets"] = normalized
    active_section = ui.get("active_section")
    if isinstance(active_section, str):
        active_section = active_section.strip()
    if active_section not in UI_SECTION_IDS:
        active_section = "quickSection"
    ui["active_section"] = active_section
    return ui


def normalize_body_streaming_config(config: dict):
    inference_cfg = config.setdefault("body", {})
    backend = str(inference_cfg.get("inference_backend", "yolo")).strip().lower()
    if backend in ("mp", "mediapipe", "mediapipepose", "media_pipe_pose"):
        backend = "mediapipe_pose"
    if backend not in ("yolo", "mediapipe_pose"):
        backend = "yolo"
    if backend == "yolo" and not yolo_available():
        backend = "mediapipe_pose"
    inference_cfg["inference_backend"] = backend
    if backend == "mediapipe_pose":
        inference_cfg["enabled"] = True
        for key in (
            "detect_torso",
            "detect_left_arm",
            "detect_right_arm",
            "detect_left_leg",
            "detect_right_leg",
        ):
            inference_cfg.setdefault(key, True)
        config.setdefault("mediapipe_pose", {})["enabled"] = True
    streaming_cfg = config.setdefault("streaming", {})
    body_cfg = streaming_cfg.setdefault("body", {})
    body_streaming_requested = body_cfg.get("streaming_enabled")
    body_cfg.setdefault("streaming_enabled", True)
    if body_streaming_requested is True:
        inference_cfg["enabled"] = True
    if config.get("processing", {}).get("dual_camera_body_enabled") is True:
        inference_cfg["enabled"] = True
    # ``streaming.body`` is the runtime source of truth. Older configurations
    # stored these switches only under ``vrc``; inherit them before applying
    # defaults so loading an old file cannot silently change body behavior.
    vrc_cfg = config.setdefault("vrc", {})
    for key, default in (
        ("host", "127.0.0.1"),
        ("port", 9000),
        ("protocol", "osc"),
        ("decouple_from_hmd", False),
        ("send_position", True),
        ("send_rotation", True),
        ("send_head_position", False),
    ):
        if key not in body_cfg and key in vrc_cfg:
            body_cfg[key] = vrc_cfg[key]
        body_cfg.setdefault(key, default)
    if "tracker_toggles" not in body_cfg and isinstance(vrc_cfg.get("tracker_toggles"), dict):
        body_cfg["tracker_toggles"] = dict(vrc_cfg["tracker_toggles"])
    if "hmd_height_scaling" not in body_cfg and isinstance(vrc_cfg.get("hmd_height_scaling"), dict):
        body_cfg["hmd_height_scaling"] = dict(vrc_cfg["hmd_height_scaling"])
    body_cfg.setdefault("hmd_height_scaling", {})
    hmd_height_scaling = body_cfg["hmd_height_scaling"]
    if not isinstance(hmd_height_scaling, dict):
        hmd_height_scaling = {}
        body_cfg["hmd_height_scaling"] = hmd_height_scaling
    hmd_height_scaling.setdefault("enabled", True)
    hmd_height_scaling.setdefault("floor_clearance", 0.03)
    hmd_height_scaling.setdefault("min_hmd_y", 0.35)
    hmd_height_scaling.setdefault("min_body_scale", 0.45)
    hmd_height_scaling.setdefault("max_body_scale", 1.0)
    hmd_height_scaling.setdefault("vertical_only", True)
    hmd_height_scaling.setdefault("require_actual_height", True)
    hmd_height_scaling.setdefault("crouch_from_hmd_height", True)
    hmd_height_scaling.setdefault("standing_eye_height", 1.65)
    hmd_height_scaling.setdefault("standing_eye_height_calibrated", False)
    hmd_height_scaling.setdefault("auto_learn_standing_height", True)
    hmd_height_scaling.setdefault("auto_learn_allow_downward", True)
    hmd_height_scaling.setdefault("auto_learn_downward_max_delta", 0.25)
    hmd_height_scaling.setdefault("auto_learn_standing_pose_threshold", 0.8)
    hmd_height_scaling.setdefault("min_standing_eye_height", 1.2)
    hmd_height_scaling.setdefault("max_standing_eye_height", 2.2)
    hmd_height_scaling.setdefault("auto_learn_min_height", 1.2)
    hmd_height_scaling.setdefault("auto_learn_window", 20)
    hmd_height_scaling.setdefault("auto_learn_stability_tolerance", 0.03)
    hmd_height_scaling.setdefault("crouch_start_ratio", 0.92)
    hmd_height_scaling.setdefault("seated_ratio", 0.55)
    hmd_height_scaling.setdefault("seated_eye_height", 0.95)
    hmd_height_scaling.setdefault("seated_hip_height", 0.45)
    hmd_height_scaling.setdefault("standing_hip_min_height", 0.78)
    hmd_height_scaling.setdefault("standing_chest_min_height", 1.05)
    hmd_height_scaling.setdefault("foot_grounding_enabled", True)
    hmd_height_scaling.setdefault("auto_calibrate_foot_reference", True)
    hmd_height_scaling.setdefault("foot_calibration_samples", 15)
    hmd_height_scaling.setdefault("foot_lift_gain", 1.0)
    hmd_height_scaling.setdefault("max_foot_lift", 0.35)
    hmd_height_scaling.setdefault("crouch_knee_forward", 0.18)
    hmd_height_scaling.setdefault("crouch_knee_outward", 0.06)
    if not body_cfg.get("send_position", True):
        body_cfg["send_head_position"] = False
    else:
        body_cfg["send_head_position"] = bool(body_cfg.get("send_head_position", False))

    # Keep the legacy ``vrc`` mirror exact. The body streamer reads only
    # ``streaming.body``, while older UI/scripts still read ``vrc``.
    vrc_cfg["decouple_from_hmd"] = body_cfg.get("decouple_from_hmd", False)
    vrc_cfg["send_position"] = body_cfg.get("send_position", True)
    vrc_cfg["send_rotation"] = body_cfg.get("send_rotation", True)
    vrc_cfg["send_head_position"] = body_cfg.get("send_head_position", False)
    vrc_cfg["hmd_height_scaling"] = body_cfg.get("hmd_height_scaling", {})
    if "tracker_toggles" in body_cfg:
        vrc_cfg["tracker_toggles"] = body_cfg["tracker_toggles"]


def _body_height_scaling_config(config: dict) -> dict:
    streaming_body = config.setdefault("streaming", {}).setdefault("body", {})
    scaling = streaming_body.setdefault("hmd_height_scaling", {})
    if not isinstance(scaling, dict):
        scaling = {}
        streaming_body["hmd_height_scaling"] = scaling
    return scaling


def _height_posture_state(actual_height, hmd_height_scaling: dict) -> dict:
    try:
        height = float(actual_height)
    except (TypeError, ValueError):
        return {
            "actual_height": None,
            "standing_eye_height": hmd_height_scaling.get("standing_eye_height", 1.65),
            "height_ratio": None,
            "posture": "unknown",
            "reason": "missing_actual_height",
        }
    standing = float(hmd_height_scaling.get("standing_eye_height", 1.65) or 1.65)
    if standing <= 0:
        standing = 1.65
    ratio = height / standing
    crouch_start = float(hmd_height_scaling.get("crouch_start_ratio", 0.92) or 0.92)
    seated_ratio = float(hmd_height_scaling.get("seated_ratio", 0.55) or 0.55)
    seated_height = float(hmd_height_scaling.get("seated_eye_height", 0.95) or 0.95)
    if height <= seated_height or ratio <= seated_ratio:
        posture = "seated"
    elif ratio < crouch_start:
        posture = "crouch"
    else:
        posture = "standing"
    return {
        "actual_height": height,
        "standing_eye_height": standing,
        "height_ratio": ratio,
        "posture": posture,
        "reason": "ok",
    }


def save_config(config: dict) -> Path:
    global _config_cache, _config_cache_signature, _active_config_path
    with _config_lock:
        snapshot = copy.deepcopy(config)
        content = json.dumps(
            snapshot,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )
        errors = []
        fallback_path = _fallback_config_path()
        paths = []
        for path in (CONFIG_PATH, fallback_path):
            path = Path(path)
            if path not in paths:
                paths.append(path)
        for path in paths:
            try:
                _write_config_path(path, content)
            except OSError as exc:
                errors.append((path, exc))
                print(f"[WARN] Config write failed at {path}: {exc}")
                continue

            _config_cache = json.loads(content)
            _active_config_path = path
            _config_cache_signature = _config_signature(path)
            if path != CONFIG_PATH:
                print(f"[INFO] Config saved to fallback path: {path}")
            return path

        details = "; ".join(f"{path}: {exc}" for path, exc in errors)
        raise OSError(f"Unable to persist configuration ({details})")


def detect_hardware() -> dict:
    """检测可用硬件"""
    devices = [{"id": "cpu", "name": "CPU", "available": True}]
    if not yolo_available():
        # Lite edition: MediaPipe picks its own delegate; no torch to query.
        return {
            "devices": devices,
            "current": load_config().get("yolo", {}).get("device", "auto"),
            "edition": "lite",
        }

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory
                mem_gb = mem / (1024 ** 3)
                devices.append({
                    "id": f"cuda:{i}",
                    "name": f"GPU {i}: {name} ({mem_gb:.1f}GB)",
                    "available": True
                })
        else:
            devices.append({"id": "cuda", "name": "CUDA (未检测到GPU)", "available": False})
    except ImportError:
        devices.append({"id": "cuda", "name": "CUDA (PyTorch未安装)", "available": False})

    return {"devices": devices, "current": load_config().get("yolo", {}).get("device", "auto")}


def _positive_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _normalized_performance_basis(basis: dict) -> dict:
    basis = basis if isinstance(basis, dict) else {}
    gpu_name = basis.get("gpu_name")
    gpu_name = str(gpu_name).strip() if gpu_name else None

    def normalized_count(key):
        value = _positive_number(basis.get(key))
        return int(value) if value is not None else None

    def normalized_gb(key):
        value = _positive_number(basis.get(key))
        return round(value, 1) if value is not None else None

    device = str(basis.get("device") or "").strip()
    if not device:
        device = "cuda:0" if gpu_name else "cpu"
    return {
        "device": device,
        "gpu_name": gpu_name,
        "gpu_memory_gb": normalized_gb("gpu_memory_gb"),
        "cpu_physical_cores": normalized_count("cpu_physical_cores"),
        "cpu_logical_cores": normalized_count("cpu_logical_cores"),
        "memory_gb": normalized_gb("memory_gb"),
    }


def _detect_loaded_torch_gpu():
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        return None
    try:
        cuda = torch_module.cuda
        if not cuda.is_available():
            return None
        best = None
        for index in range(int(cuda.device_count())):
            memory_bytes = _positive_number(cuda.get_device_properties(index).total_memory)
            memory_gb = memory_bytes / (1024 ** 3) if memory_bytes is not None else None
            candidate = {
                "device": f"cuda:{index}",
                "gpu_name": str(cuda.get_device_name(index)).strip() or None,
                "gpu_memory_gb": memory_gb,
            }
            if best is None or (candidate["gpu_memory_gb"] or 0) > (best["gpu_memory_gb"] or 0):
                best = candidate
        return best
    except Exception:
        return None


def _detect_nvidia_smi_gpu():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            **kwargs,
        )
        if result.returncode != 0:
            return None
        best = None
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) < 3:
                continue
            index = int(row[0].strip())
            memory_mb = _positive_number(row[2])
            candidate = {
                "device": f"cuda:{index}",
                "gpu_name": row[1].strip() or None,
                "gpu_memory_gb": memory_mb / 1024 if memory_mb is not None else None,
            }
            if best is None or (candidate["gpu_memory_gb"] or 0) > (best["gpu_memory_gb"] or 0):
                best = candidate
        return best
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        return None


def _fallback_total_memory_gb():
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total_physical / (1024 ** 3)
            return None

        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return (page_size * page_count) / (1024 ** 3)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def collect_performance_basis() -> dict:
    physical_cores = None
    logical_cores = None
    memory_gb = None
    try:
        logical_cores = os.cpu_count()
    except Exception:
        pass

    if HAS_PSUTIL:
        try:
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True) or logical_cores
        except Exception:
            pass
        try:
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass
    if memory_gb is None:
        try:
            memory_gb = _fallback_total_memory_gb()
        except Exception:
            memory_gb = None

    gpu = _detect_loaded_torch_gpu() or _detect_nvidia_smi_gpu() or {}
    return _normalized_performance_basis({
        "device": gpu.get("device", "cpu"),
        "gpu_name": gpu.get("gpu_name"),
        "gpu_memory_gb": gpu.get("gpu_memory_gb"),
        "cpu_physical_cores": physical_cores,
        "cpu_logical_cores": logical_cores,
        "memory_gb": memory_gb,
    })


def classify_performance_tier(basis: dict) -> str:
    basis = _normalized_performance_basis(basis)
    gpu_memory = basis["gpu_memory_gb"]
    has_gpu = bool(basis["gpu_name"] or gpu_memory)

    gpu_level = None
    if has_gpu:
        if gpu_memory is None:
            gpu_level = 1
        elif gpu_memory >= 10:
            gpu_level = 3
        elif gpu_memory >= 5.5:
            gpu_level = 2
        elif gpu_memory >= 3.5:
            gpu_level = 1
        else:
            gpu_level = 0

    cpu_levels = []
    physical = basis["cpu_physical_cores"]
    logical = basis["cpu_logical_cores"]
    if physical is not None:
        cpu_levels.append(3 if physical >= 12 else 2 if physical >= 6 else 1 if physical >= 4 else 0)
    if logical is not None:
        cpu_levels.append(3 if logical >= 24 else 2 if logical >= 12 else 1 if logical >= 8 else 0)
    cpu_level = max(cpu_levels) if cpu_levels else None

    memory = basis["memory_gb"]
    memory_level = None
    if memory is not None:
        memory_level = 3 if memory >= 28 else 2 if memory >= 12 else 1 if memory >= 7 else 0

    known_levels = [level for level in (gpu_level, cpu_level, memory_level) if level is not None]
    if not known_levels:
        return "low"
    overall_level = min(known_levels)
    if not has_gpu:
        overall_level = min(overall_level, 1)
    return ("low", "medium", "high", "ultra")[overall_level]


def build_performance_recommendation(basis: dict) -> dict:
    basis = _normalized_performance_basis(basis)
    tier = classify_performance_tier(basis)
    baseline = copy.deepcopy(PERFORMANCE_BASELINES[tier])
    if basis["device"] == "cpu":
        baseline["precision"] = "fp32"

    if basis["gpu_name"]:
        gpu_detail = basis["gpu_name"]
        if basis["gpu_memory_gb"] is not None:
            gpu_detail += f" ({basis['gpu_memory_gb']:.1f} GB 显存)"
    else:
        gpu_detail = "未检测到可用独立 GPU"
    cpu_detail = "CPU 核心数未知"
    if basis["cpu_physical_cores"] is not None or basis["cpu_logical_cores"] is not None:
        cpu_detail = (
            f"CPU {basis['cpu_physical_cores'] or '?'} 个物理核心 / "
            f"{basis['cpu_logical_cores'] or '?'} 个逻辑核心"
        )
    memory_detail = (
        f"内存 {basis['memory_gb']:.1f} GB"
        if basis["memory_gb"] is not None
        else "内存容量未知"
    )
    rationale = f"{gpu_detail}；{cpu_detail}；{memory_detail}。按已知关键资源的保守档位给出建议。"
    return {
        "status": "ok",
        "tier": tier,
        "tier_label": PERFORMANCE_TIER_LABELS[tier],
        "basis": basis,
        "baseline": baseline,
        "rationale": rationale,
    }


# ============ API 路由 ============

@app.route("/")
def index():
    resp = send_from_directory(str(STATIC_DIR), "index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers.pop("ETag", None)
    resp.headers.pop("Last-Modified", None)
    return resp


@app.route("/api/edition", methods=["GET"])
def get_edition():
    return jsonify(edition_info())


@app.route("/api/config", methods=["GET"])
def get_config():
    config = load_config()
    ensure_secondary_camera_config(config)
    ensure_processing_config(config)
    ensure_hand_streaming_config(config)
    ensure_face_calibration_config(config)
    ensure_ui_config(config)
    normalize_body_streaming_config(config)
    return jsonify(config)


@app.route("/api/config", methods=["POST"])
def update_config():
    try:
        data = request.json
        if not isinstance(data, dict):
            return jsonify({"status": "error", "message": "JSON object required"}), 400

        with _config_lock:
            config = load_config()

            # Keep read, recursive merge, normalization, and write atomic so
            # autosave requests cannot overwrite unrelated nested settings.
            old_body = config.get("body", {}).get("enabled", True)
            old_face = config.get("face", {}).get("enabled", True)
            old_hand = config.get("hand", {}).get("enabled", True)

            _deep_merge_config(config, data)
            ensure_face_calibration_config(config)
            ensure_secondary_camera_config(config)
            ensure_processing_config(config)
            ensure_hand_streaming_config(config)
            ensure_ui_config(config)
            normalize_body_streaming_config(config)

            new_body = config.get("body", {}).get("enabled", True)
            new_face = config.get("face", {}).get("enabled", True)
            new_hand = config.get("hand", {}).get("enabled", True)

            if old_body != new_body or old_face != new_face or old_hand != new_hand:
                print(f"[Config] 开关变更: body={old_body}->{new_body}, face={old_face}->{new_face}, hand={old_hand}->{new_hand}")

            saved_path = save_config(config)
        return jsonify({
            "status": "ok",
            "config": load_config(),
            "config_path": str(saved_path),
            "fallback": saved_path != CONFIG_PATH,
        })
    except OSError as e:
        print(f"[ERROR] update_config persistence failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 507
    except Exception as e:
        print(f"[ERROR] update_config: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


def _read_latest_face_state(max_age_seconds: float = 5.0) -> dict:
    if not FACE_STATE_PATH.exists():
        return {"available": False, "reason": "no_state_file"}
    try:
        state = read_face_state(FACE_STATE_PATH)
    except Exception as e:
        return {"available": False, "reason": f"read_error: {e}"}

    ts = float(state.get("timestamp") or 0)
    age = time.time() - ts if ts > 0 else None
    values = state.get("vrcft_raw", {})
    available = bool(values) and age is not None and age <= max_age_seconds
    return {
        "available": available,
        "timestamp": ts,
        "age_seconds": age,
        "param_count": len(values),
        "vrcft_raw": values,
        "reason": None if available else "stale_or_empty",
    }


@app.route("/api/face/calibration", methods=["GET"])
def face_calibration_status():
    config = load_config()
    calibration = ensure_face_calibration_config(config)
    face_state = _read_latest_face_state()
    return jsonify({
        "status": "ok",
        "enabled": calibration.get("enabled", False),
        "min_count": len(calibration.get("min_values", {})),
        "max_count": len(calibration.get("max_values", {})),
        "updated_at": calibration.get("updated_at"),
        "state_available": face_state.get("available", False),
        "state_age_seconds": face_state.get("age_seconds"),
        "state_param_count": face_state.get("param_count", 0),
    })


@app.route("/api/face/calibration/record", methods=["POST"])
def face_calibration_record():
    data = request.json or {}
    record_type = data.get("type")
    if record_type not in ("min", "max"):
        return jsonify({"status": "error", "message": "type must be min or max"}), 400

    state = _read_latest_face_state()
    if not state.get("available"):
        return jsonify({
            "status": "error",
            "message": "没有可用的实时面部 VRCFT 数据，请启动动捕并确保面部识别正常。",
            "reason": state.get("reason"),
        }), 400

    values = {
        name: float(value)
        for name, value in state.get("vrcft_raw", {}).items()
        if name.startswith("v2/")
    }
    if not values:
        return jsonify({"status": "error", "message": "当前面部数据没有 v2/* 参数。"}), 400

    with _config_lock:
        config = load_config()
        calibration = ensure_face_calibration_config(config)
        key = "min_values" if record_type == "min" else "max_values"
        calibration[key] = values
        calibration["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_config(config)

    return jsonify({
        "status": "ok",
        "type": record_type,
        "recorded_count": len(values),
        "enabled": calibration.get("enabled", False),
        "updated_at": calibration["updated_at"],
    })


@app.route("/api/face/calibration/reset", methods=["POST"])
def face_calibration_reset():
    with _config_lock:
        config = load_config()
        calibration = ensure_face_calibration_config(config)
        calibration["min_values"] = {}
        calibration["max_values"] = {}
        calibration["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_config(config)
    return jsonify({
        "status": "ok",
        "enabled": calibration.get("enabled", False),
        "updated_at": calibration["updated_at"],
    })


def detect_cameras() -> list:
    """检测可用摄像头列表
    
    优化策略:
    1. Windows 下优先使用 DirectShow 后端，检测 USB 摄像头更可靠
    2. 使用超时线程防止卡死
    3. 尝试获取摄像头名称（DirectShow 支持）
    """
    import cv2
    import sys
    import threading

    cameras = []
    found_ids = set()

    # Windows 下使用 DirectShow (MSMF) 枚举摄像头，更可靠
    backends_to_try = []
    if sys.platform == 'win32':
        backends_to_try.append((cv2.CAP_DSHOW, "DirectShow"))
        backends_to_try.append((cv2.CAP_MSMF, "MSMF"))
    else:
        backends_to_try.append((cv2.CAP_V4L2, "V4L2"))
        backends_to_try.append((cv2.CAP_ANY, "Auto"))

    def try_open_camera(idx, backend_id, backend_name, result_list, timeout=2.0):
        """尝试打开摄像头，带超时"""
        cap = None
        try:
            cap = cv2.VideoCapture(idx, backend_id)
            if cap.isOpened():
                # 读取一帧确认摄像头真正可用
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    zoom = 0.0
                    zoom_supported = False
                    if hasattr(cv2, "CAP_PROP_ZOOM"):
                        try:
                            zoom = float(cap.get(cv2.CAP_PROP_ZOOM) or 0.0)
                            zoom_supported = zoom > 0
                        except Exception:
                            zoom = 0.0
                    result_list.append({
                        "id": idx,
                        "name": f"摄像头 {idx}",
                        "width": w,
                        "height": h,
                        "fps": fps if fps and fps > 0 else 30,
                        "backend": backend_name,
                        "zoom": zoom,
                        "zoom_supported": zoom_supported,
                    })
        except Exception:
            pass
        finally:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass

    # 阶段 1: 使用 DirectShow 快速探测前 10 个索引
    for backend_id, backend_name in backends_to_try:
        for i in range(10):
            if i in found_ids:
                continue

            result_container = []
            thread = threading.Thread(
                target=try_open_camera,
                args=(i, backend_id, backend_name, result_container),
                daemon=True
            )
            thread.start()
            thread.join(timeout=3.0)  # 3秒超时

            if result_container:
                cam = result_container[0]
                # 尝试用 DirectShow 获取设备名
                try:
                    cam["name"] = _get_camera_name_dshow(i) or cam["name"]
                except Exception:
                    pass
                # 补充分辨率信息
                if cam["width"] and cam["height"]:
                    cam["name"] = f"{cam['name']} ({cam['width']}x{cam['height']}"
                    if cam["fps"] and cam["fps"] > 0:
                        cam["name"] += f" @ {cam['fps']:.0f}fps"
                    cam["name"] += ")"
                cameras.append(cam)
                found_ids.add(i)
            elif thread.is_alive():
                # 超时，跳过此索引
                pass

        # Camera indices are backend-specific. Once DirectShow succeeds, keep
        # that index space so the runtime can open the same physical devices.
        if cameras:
            break

    # 阶段 2: 如果 DirectShow 没找到，尝试 MSMF 和默认后端
    if not cameras:
        for i in range(10):
            result_container = []
            thread = threading.Thread(
                target=try_open_camera,
                args=(i, cv2.CAP_ANY, "Default", result_container),
                daemon=True
            )
            thread.start()
            thread.join(timeout=3.0)

            if result_container:
                cam = result_container[0]
                cameras.append(cam)

    return cameras


def _get_camera_name_dshow(index: int) -> Optional[str]:
    """通过 DirectShow 获取摄像头设备名称 (Windows Only)"""
    import sys
    if sys.platform != 'win32':
        return None

    try:
        # 使用 Windows WMI 查询摄像头名称
        import subprocess
        result = subprocess.run(
            ['wmic', 'path', 'Win32_PnPEntity', 'where',
             'Service="usbvideo" OR Class="Camera" OR Class="Image"',
             'get', 'Name,DeviceID', '/format:list'],
            capture_output=True, text=True, timeout=5,
            creationflags=0x08000000  # CREATE_NO_WINDOW
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            cameras_found = []
            current_name = None
            for line in lines:
                line = line.strip()
                if line.startswith('Name='):
                    current_name = line[5:].strip()
                elif line.startswith('DeviceID=') and current_name:
                    cameras_found.append(current_name)
                    current_name = None
            if index < len(cameras_found):
                return cameras_found[index]
    except Exception:
        pass

    # 回退: 使用 PowerShell 查询
    try:
        import subprocess
        ps_cmd = (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.Service -eq 'usbvideo' -or $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | "
            "Select-Object -ExpandProperty Name"
        )
        result = subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_cmd],
            capture_output=True, text=True, timeout=5,
            creationflags=0x08000000  # CREATE_NO_WINDOW
        )
        if result.returncode == 0:
            names = [n.strip() for n in result.stdout.strip().split('\n') if n.strip()]
            if index < len(names):
                return names[index]
    except Exception:
        pass

    return None


@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    """Get available cameras."""
    global _camera_cache
    config = load_config()
    current_id = config.get("camera", {}).get("camera_id", 0)
    force_refresh = request.args.get("refresh", "0").lower() in ("1", "true", "yes")
    now = time.time()

    def busy_response():
        cached = _camera_cache.get("data")
        if cached is not None:
            return jsonify({"cameras": cached, "current": current_id, "cached": True, "busy": True})
        cam_cfg = config.get("camera", {})
        fallback = [{
            "id": current_id,
            "name": f"Current camera {current_id} (in use)",
            "width": cam_cfg.get("width", 640),
            "height": cam_cfg.get("height", 480),
            "fps": cam_cfg.get("fps", 30),
            "backend": "configured",
        }]
        return jsonify({"cameras": fallback, "current": current_id, "cached": False, "busy": True})

    mocap_running = bool(mocap_process and mocap_process.poll() is None)
    if not mocap_running:
        mocap_running = bool(_find_mocap_processes())
    if mocap_running:
        return busy_response()

    if (
        not force_refresh
        and _camera_cache.get("data") is not None
        and now - _camera_cache.get("ts", 0.0) < CACHE_TTL_SECONDS
    ):
        return jsonify({"cameras": _camera_cache["data"], "current": current_id, "cached": True})

    if not _camera_lock.acquire(blocking=False):
        return busy_response()
    try:
        now = time.time()
        if (
            not force_refresh
            and _camera_cache.get("data") is not None
            and now - _camera_cache.get("ts", 0.0) < CACHE_TTL_SECONDS
        ):
            return jsonify({"cameras": _camera_cache["data"], "current": current_id, "cached": True})
        cams = detect_cameras()
        _camera_cache = {"data": cams, "ts": time.time()}
        return jsonify({"cameras": cams, "current": current_id, "cached": False})
    finally:
        _camera_lock.release()


@app.route("/api/hardware", methods=["GET"])
def get_hardware():
    global _hardware_cache
    now = time.time()
    if _hardware_cache.get("data") is not None and now - _hardware_cache.get("ts", 0.0) < CACHE_TTL_SECONDS:
        return jsonify({**_hardware_cache["data"], "cached": True})
    if mocap_process and mocap_process.poll() is not None:
        mocap_status["running"] = False
    if mocap_process and mocap_process.poll() is None:
        current = load_config().get("yolo", {}).get("device", "auto")
        devices = [{"id": "cpu", "name": "CPU", "available": True}]
        if current not in ("auto", "cpu"):
            devices.append({"id": current, "name": f"{current} (in use)", "available": True})
        return jsonify({"devices": devices, "current": current, "cached": False, "busy": True})
    data = detect_hardware()
    _hardware_cache = {"data": data, "ts": now}
    return jsonify({**data, "cached": False})


@app.route("/api/performance/recommendation", methods=["GET"])
def performance_recommendation():
    global _performance_recommendation_cache
    now = time.monotonic()
    with _performance_recommendation_lock:
        cached = _performance_recommendation_cache
        if cached.get("data") is not None and now - cached.get("ts", 0.0) < PERFORMANCE_RECOMMENDATION_TTL_SECONDS:
            return jsonify(copy.deepcopy(cached["data"]))
        try:
            data = build_performance_recommendation(collect_performance_basis())
        except Exception:
            data = build_performance_recommendation({"device": "cpu"})
        _performance_recommendation_cache = {"data": data, "ts": time.monotonic()}
        return jsonify(copy.deepcopy(data))


@app.route("/api/steamvr/status", methods=["GET"])
def steamvr_status():
    """检查 SteamVR 驱动连接状态"""
    import sys
    pipe_name = r'\\.\pipe\ApriltagPipeIn'
    driver_path = PROJECT_ROOT / "driver_files"
    driver_dll = driver_path / "apriltagtrackers" / "bin" / "win64" / "driver_apriltagtrackers.dll"
    driver_installed = driver_dll.exists()

    # 检查是否已安装到 SteamVR 目录
    steamvr_driver_installed = False
    steam_path = None
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", 0, winreg.KEY_READ)
        steam_path, _ = winreg.QueryValueEx(key, "InstallPath")
        winreg.CloseKey(key)
        steamvr_driver_path = Path(steam_path) / "steamapps" / "common" / "SteamVR" / "drivers" / "apriltagtrackers" / "bin" / "win64" / "driver_apriltagtrackers.dll"
        steamvr_driver_installed = steamvr_driver_path.exists()
    except Exception:
        pass

    # 检查 activateMultipleDrivers 配置
    activate_multiple = False
    try:
        if steam_path:
            settings_path = Path(steam_path) / "config" / "steamvr.vrsettings"
            if settings_path.exists():
                import json
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                activate_multiple = settings.get("steamvr", {}).get("activateMultipleDrivers", False)
    except Exception:
        pass

    # 检查命名管道是否可用（SteamVR 是否运行）
    pipe_available = False
    num_trackers = 0
    driver_version = ""
    if sys.platform == 'win32':
        try:
            pipe = open(pipe_name, 'rb+', buffering=0)
            pipe.write(b'numtrackers\0')
            resp = pipe.read(1024)
            pipe.close()
            if resp:
                pipe_available = True
                parts = resp.decode("utf-8").strip().split(" ")
                # 格式: "numtrackers <count> <version>" 或 "numtrackers <count>"
                # 注意: 管道返回可能带前导空格，split后第一个元素可能是空字符串
                for part in parts:
                    part_clean = part.strip().rstrip('\x00')
                    if part_clean.isdigit():
                        num_trackers = int(part_clean)
                        break
                # 提取版本号
                for part in parts:
                    part_clean = part.strip().rstrip('\x00')
                    if part_clean and not part_clean.isdigit() and part_clean != 'numtrackers' and part_clean != 'OK':
                        driver_version = part_clean
                        break
        except (FileNotFoundError, OSError):
            pass

    config = load_config()
    hand_cfg = ensure_hand_streaming_config(config)
    vmt_status_data = get_vmt_runtime_status(
        host=hand_cfg.get("vmt_host", "127.0.0.1"),
        port=hand_cfg.get("vmt_port", 39570),
        project_root=PROJECT_ROOT,
    )

    return jsonify({
        "driver_files_available": driver_installed,
        "driver_installed_to_steamvr": steamvr_driver_installed,
        "activate_multiple_drivers": activate_multiple,
        "pipe_available": pipe_available,
        "steamvr_running": pipe_available,
        "num_trackers_registered": num_trackers,
        "driver_version": driver_version,
        "vmt": vmt_status_data,
    })


@app.route("/api/vmt/status", methods=["GET"])
def vmt_status():
    config = load_config()
    hand_cfg = ensure_hand_streaming_config(config)
    return jsonify(get_vmt_runtime_status(
        host=hand_cfg.get("vmt_host", "127.0.0.1"),
        port=hand_cfg.get("vmt_port", 39570),
        project_root=PROJECT_ROOT,
    ))


@app.route("/api/vmt/devices", methods=["POST"])
def vmt_devices():
    """Merge VMT's device table with SteamVR's actual controller selection."""
    config = load_config()
    hand_cfg = ensure_hand_streaming_config(config)
    vmt_host = hand_cfg.get("vmt_host", "127.0.0.1")
    vmt_port = hand_cfg.get("vmt_port", 39570)
    result = query_vmt_devices(
        host=vmt_host,
        port=vmt_port,
    )
    openvr_result = query_openvr_devices()
    runtime_status = get_vmt_runtime_status(
        host=vmt_host,
        port=vmt_port,
        project_root=PROJECT_ROOT,
    )
    expected_serials = [
        f"VMT_{int(hand_cfg.get('vmt_left_device_index', 1))}",
        f"VMT_{int(hand_cfg.get('vmt_right_device_index', 2))}",
    ]
    openvr_by_serial = {
        str(item.get("serial")): item
        for item in openvr_result.get("devices", [])
        if item.get("serial")
    }

    result["devices"] = [
        {**item, **openvr_by_serial.get(str(item.get("serial") or ""), {})}
        for item in result.get("devices", [])
    ]
    merged_vmt_by_serial = {
        str(item.get("serial")): dict(item)
        for item in result.get("vmt_devices", [])
        if item.get("serial")
    }
    for serial, item in openvr_by_serial.items():
        if not item.get("is_vmt"):
            continue
        merged_vmt_by_serial[serial] = {
            **merged_vmt_by_serial.get(serial, {}),
            **item,
        }
    result["vmt_devices"] = list(merged_vmt_by_serial.values())

    present_serials = set(merged_vmt_by_serial)
    missing_serials = [
        serial for serial in expected_serials if serial not in present_serials
    ]
    expected_devices = []
    for side, serial in zip(("left", "right"), expected_serials):
        expected_devices.append({
            "side": side,
            "serial": serial,
            **openvr_by_serial.get(serial, {}),
        })

    selected_as_left = openvr_result.get("left_serial") == expected_serials[0]
    selected_as_right = openvr_result.get("right_serial") == expected_serials[1]
    pose_valid = {
        item["side"]: bool(item.get("pose_valid"))
        for item in expected_devices
    }
    configured_priority = int(
        hand_cfg.get("vmt_hand_selection_priority", 100)
    )
    priority_mismatches = [
        {
            "side": item["side"],
            "serial": item["serial"],
            "configured": configured_priority,
            "active": item.get("hand_selection_priority"),
        }
        for item in expected_devices
        if item.get("hand_selection_priority") is not None
        and int(item["hand_selection_priority"]) != configured_priority
    ]
    register_on_startup = hand_cfg.get(
        "vmt_register_on_steamvr_startup", True
    ) is not False
    driver_register_on_startup = runtime_status.get(
        "add_compatible_controller_on_startup"
    )
    startup_setting_mismatch = (
        driver_register_on_startup is not None
        and driver_register_on_startup != register_on_startup
    )
    restart_required = bool(startup_setting_mismatch or priority_mismatches)

    blockers = []
    if not openvr_result.get("ok"):
        blockers.append({
            "code": "openvr_unavailable",
            "message": openvr_result.get("error") or "OpenVR diagnostics unavailable",
            "restart_required": False,
        })
    if missing_serials:
        blockers.append({
            "code": "expected_devices_missing",
            "message": "SteamVR is missing: " + ", ".join(missing_serials),
            "restart_required": False,
        })
    if runtime_status.get("skeleton_input") is False:
        blockers.append({
            "code": "skeleton_input_disabled",
            "message": "VMT SkeletonInput is disabled",
            "restart_required": False,
        })
    if startup_setting_mismatch:
        blockers.append({
            "code": "startup_registration_mismatch",
            "message": (
                "VMT startup compatible-controller registration does not "
                "match the project setting"
            ),
            "restart_required": True,
        })
    if priority_mismatches:
        blockers.append({
            "code": "active_priority_stale",
            "message": "Active OpenVR controller priority differs from configuration",
            "restart_required": True,
        })
    if openvr_result.get("ok") and not selected_as_left:
        blockers.append({
            "code": "left_not_selected",
            "message": f"SteamVR left hand is not {expected_serials[0]}",
            "restart_required": False,
        })
    if openvr_result.get("ok") and not selected_as_right:
        blockers.append({
            "code": "right_not_selected",
            "message": f"SteamVR right hand is not {expected_serials[1]}",
            "restart_required": False,
        })
    for item in expected_devices:
        if item.get("openvr_index") is not None and not item.get("pose_valid"):
            blockers.append({
                "code": f"{item['side']}_pose_invalid",
                "message": (
                    f"{item['serial']} pose is invalid; keep the HMD tracked "
                    "and verify the VMT pose reference"
                ),
                "restart_required": False,
            })

    steamvr_selection_confirmed = bool(
        openvr_result.get("ok") and selected_as_left and selected_as_right
    )
    expected_devices_present = bool(not missing_serials and present_serials)
    takeover_ready = bool(
        expected_devices_present
        and steamvr_selection_confirmed
        and all(pose_valid.values())
        and runtime_status.get("skeleton_input") is not False
        and not restart_required
    )
    result.update({
        "expected_serials": expected_serials,
        "expected_devices": expected_devices,
        "missing_serials": missing_serials,
        "expected_devices_present": expected_devices_present,
        "selected_as_left": selected_as_left,
        "selected_as_right": selected_as_right,
        "pose_valid": pose_valid,
        "steamvr_controller_selection_confirmed": steamvr_selection_confirmed,
        "controller_takeover_ready": takeover_ready,
        "configured_hand_selection_priority": configured_priority,
        "priority_mismatches": priority_mismatches,
        "register_on_steamvr_startup": register_on_startup,
        "driver_register_on_steamvr_startup": driver_register_on_startup,
        "restart_required": restart_required,
        "restart_recommended": bool(
            restart_required
            or (openvr_result.get("ok") and not steamvr_selection_confirmed)
        ),
        "blockers": blockers,
        "blocking_reasons": [item["message"] for item in blockers],
        "openvr": openvr_result,
        "runtime": runtime_status,
        "vrchat_consumption_confirmed": False,
        "note": (
            "SteamVR selection can be diagnosed here, but VRChat skeleton "
            "consumption still requires an in-game finger-motion check."
        ),
    })
    return jsonify(result)


@app.route("/api/vmt/install", methods=["POST"])
def vmt_install():
    manager = find_vmt_manager([PROJECT_ROOT, RUNTIME_ROOT])
    if manager:
        try:
            if not launch_elevated(manager, "install"):
                raise RuntimeError("Windows refused to launch the VMT manager")
            return jsonify({
                "status": "ok",
                "stage": "driver_install",
                "message": "VMT 驱动安装程序已启动。完成后请重启 SteamVR。",
                "manager_path": str(manager),
            })
        except Exception as exc:
            return jsonify({
                "status": "error",
                "message": f"启动 VMT 驱动安装失败: {exc}",
            }), 500

    if vmt_installer_is_running():
        return jsonify({
            "status": "warning",
            "stage": "installer_running",
            "message": "VMT 安装程序正在运行，请先完成桌面的安装/UAC 窗口。",
        })

    installer_path = RUNTIME_ROOT / "tools" / VMT_INSTALLER_NAME
    try:
        installer = download_vmt_installer(installer_path)
        if not launch_elevated(installer):
            raise RuntimeError("Windows refused to launch the VMT installer")
        return jsonify({
            "status": "warning",
            "stage": "manager_setup",
            "message": (
                "VMT 官方安装程序已启动。完成软件安装后，再点击一次“安装 VMT 驱动”，"
                "然后重启 SteamVR。"
            ),
            "installer_path": str(installer),
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": f"下载或启动 VMT 官方安装程序失败: {exc}",
        }), 500


@app.route("/api/steamvr/install", methods=["POST"])
def steamvr_install():
    """自动安装 SteamVR 驱动"""
    import shutil
    import subprocess

    driver_src = PROJECT_ROOT / "driver_files" / "apriltagtrackers"
    if not (driver_src / "bin" / "win64" / "driver_apriltagtrackers.dll").exists():
        return jsonify({"status": "error", "message": "驱动文件不存在，请先下载 ApriltagTrackers 驱动到 driver_files 目录"})

    # 查找 Steam 安装路径
    steam_path = None
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", 0, winreg.KEY_READ)
        steam_path, _ = winreg.QueryValueEx(key, "InstallPath")
        winreg.CloseKey(key)
    except Exception:
        pass

    if not steam_path:
        return jsonify({"status": "error", "message": "未找到 Steam 安装路径，请确认 Steam 已安装"})

    steamvr_drivers = Path(steam_path) / "steamapps" / "common" / "SteamVR" / "drivers"
    if not steamvr_drivers.exists():
        return jsonify({"status": "error", "message": f"SteamVR 驱动目录不存在: {steamvr_drivers}，请先安装 SteamVR"})

    # 复制驱动文件
    dest = steamvr_drivers / "apriltagtrackers"
    try:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(driver_src, dest)
    except PermissionError:
        return jsonify({"status": "error", "message": "权限不足，请以管理员身份运行或手动复制驱动文件"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"复制驱动文件失败: {e}"})

    # 修改 steamvr.vrsettings 添加 activateMultipleDrivers
    settings_path = Path(steam_path) / "config" / "steamvr.vrsettings"
    try:
        import json
        if settings_path.exists():
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        else:
            settings = {}

        if "steamvr" not in settings:
            settings["steamvr"] = {}

        settings["steamvr"]["activateMultipleDrivers"] = True

        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
    except Exception as e:
        return jsonify({"status": "warning", "message": f"驱动已复制，但修改配置文件失败: {e}，请手动在 steamvr.vrsettings 中添加 \"activateMultipleDrivers\": true"})

    return jsonify({"status": "ok", "message": "SteamVR 驱动安装成功！请重启 SteamVR 使驱动生效。"})


@app.route("/api/steamvr/uninstall", methods=["POST"])
def steamvr_uninstall():
    """卸载 SteamVR 驱动"""
    import shutil

    removed = []

    # 方式1: 从 SteamVR drivers 目录移除
    steam_path = None
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", 0, winreg.KEY_READ)
        steam_path, _ = winreg.QueryValueEx(key, "InstallPath")
        winreg.CloseKey(key)
    except Exception:
        pass

    if steam_path:
        dest = Path(steam_path) / "steamapps" / "common" / "SteamVR" / "drivers" / "apriltagtrackers"
        if dest.exists():
            try:
                shutil.rmtree(dest)
                removed.append("SteamVR drivers 目录")
            except PermissionError:
                return jsonify({"status": "error", "message": "权限不足，无法删除驱动文件，请以管理员身份运行"})
            except Exception as e:
                return jsonify({"status": "error", "message": f"删除驱动文件失败: {e}"})

    # 方式2: 从 openvrpaths.vrpath 移除
    vrpath_file = Path(os.environ.get("LOCALAPPDATA", "")) / "openvr" / "openvrpaths.vrpath"
    if vrpath_file.exists():
        try:
            with open(vrpath_file, "r", encoding="utf-8") as f:
                vrpath_data = json.load(f)
            if "external_drivers" in vrpath_data:
                before = len(vrpath_data["external_drivers"])
                vrpath_data["external_drivers"] = [
                    d for d in vrpath_data["external_drivers"]
                    if "apriltagtrackers" not in str(d).lower()
                ]
                after = len(vrpath_data["external_drivers"])
                if before != after:
                    with open(vrpath_file, "w", encoding="utf-8") as f:
                        json.dump(vrpath_data, f, indent=4)
                    removed.append("openvrpaths.vrpath")
        except Exception as e:
            return jsonify({"status": "warning", "message": f"部分卸载成功，但修改 openvrpaths.vrpath 失败: {e}"})

    if removed:
        return jsonify({"status": "ok", "message": f"驱动已从以下位置移除: {', '.join(removed)}，请重启 SteamVR 使更改生效。"})
    else:
        return jsonify({"status": "ok", "message": "未找到已安装的驱动，无需卸载。"})


@app.route("/api/steamvr/install_vrpath", methods=["POST"])
def steamvr_install_vrpath():
    """通过 openvrpaths.vrpath 注册驱动 (无需复制文件到 SteamVR 目录)"""
    driver_dir = PROJECT_ROOT / "driver_files" / "apriltagtrackers"
    if not (driver_dir / "driver.vrdrivermanifest").exists():
        return jsonify({"status": "error", "message": "驱动文件不存在: driver_files/apriltagtrackers"})

    vrpath_file = Path(os.environ.get("LOCALAPPDATA", "")) / "openvr" / "openvrpaths.vrpath"
    if not vrpath_file.exists():
        return jsonify({"status": "error", "message": "openvrpaths.vrpath 不存在，请先运行一次 SteamVR"})

    driver_path = str(driver_dir).replace("\\", "/")

    try:
        with open(vrpath_file, "r", encoding="utf-8") as f:
            vrpath_data = json.load(f)
    except Exception as e:
        return jsonify({"status": "error", "message": f"读取 openvrpaths.vrpath 失败: {e}"})

    # 检查是否已注册
    if "external_drivers" in vrpath_data:
        for d in vrpath_data["external_drivers"]:
            if "apriltagtrackers" in str(d).lower():
                return jsonify({"status": "ok", "message": "驱动已注册在 openvrpaths.vrpath 中，无需重复注册。"})

    # 添加注册
    if "external_drivers" not in vrpath_data:
        vrpath_data["external_drivers"] = []
    vrpath_data["external_drivers"].append(driver_path)

    try:
        with open(vrpath_file, "w", encoding="utf-8") as f:
            json.dump(vrpath_data, f, indent=4)
    except PermissionError:
        return jsonify({"status": "error", "message": "权限不足，无法写入 openvrpaths.vrpath，请以管理员身份运行"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"写入 openvrpaths.vrpath 失败: {e}"})

    return jsonify({"status": "ok", "message": "驱动已注册到 openvrpaths.vrpath！请重启 SteamVR 使驱动生效。"})


@app.route("/api/mocap/start", methods=["POST"])
def start_mocap():
    global mocap_process, mocap_status

    if mocap_status["running"] and mocap_process and mocap_process.poll() is None:
        return jsonify({"status": "error", "message": "Mocap already running"})

    # 先停掉之前的残留进程（通过 subprocess 对象）
    if mocap_process and mocap_process.poll() is None:
        mocap_process.terminate()
        try:
            mocap_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            mocap_process.kill()

    # 自动检测并终止所有残留的动捕进程（可能是之前崩溃或手动启动的）
    args = request.json or {}
    if args.get("cleanup_stale"):
        killed_count = kill_stale_mocap_processes()
        if killed_count > 0:
            print(f"Cleaned up {killed_count} stale mocap process(es)")

    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--mocap-child"]
    else:
        cmd = [sys.executable, "-uB", str(PROJECT_ROOT / "bin" / "main.py")]

    # 传递配置文件路径
    config_file = active_config_path()
    if config_file.exists():
        cmd.extend(["--config", str(config_file)])

    if args.get("no_display"):
        cmd.append("--no-display")
    if args.get("no_face"):
        cmd.append("--no-face")
    if args.get("model"):
        cmd.extend(["--model", args["model"]])
    if args.get("device"):
        cmd.extend(["--device", args["device"]])
    if args.get("precision"):
        cmd.extend(["--precision", args["precision"]])
    if args.get("protocol"):
        cmd.extend(["--protocol", args["protocol"]])
    if args.get("camera") is not None:
        cmd.extend(["--camera", str(args["camera"])])

    env = os.environ.copy()
    env["ULTRALYTICS_SETTINGS"] = str(RUNTIME_ROOT / ".ultralytics")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["YOLO26N_HMD_DASHBOARD_BRIDGE"] = "1"
    env["YOLO26N_HMD_DASHBOARD_HOST"] = "127.0.0.1"
    env["YOLO26N_HMD_DASHBOARD_PORT"] = str(_dashboard_runtime["port"])

    # 日志输出到临时目录
    import tempfile
    log_path = Path(tempfile.gettempdir()) / "yolo26n_mocap.log"
    log_file = open(str(log_path), 'w', encoding='utf-8')

    mocap_process = subprocess.Popen(
        cmd, env=env, cwd=str(RUNTIME_ROOT),
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    mocap_status = {
        "running": True,
        "pid": mocap_process.pid,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "starting": True,
    }

    # Fast return: avoid blocking the dashboard while YOLO/MediaPipe/camera finish initializing.
    time.sleep(0.2)
    if mocap_process.poll() is not None:
        mocap_status = {"running": False, "pid": None, "start_time": None, "starting": False}
        return jsonify({
            "status": "error",
            "message": "进程启动后立即退出，可能是摄像头被占用或模型加载失败",
            "exit_code": mocap_process.returncode,
        })

    return jsonify({"status": "ok", "pid": mocap_process.pid, "starting": True})


@app.route("/api/mocap/stop", methods=["POST"])
def stop_mocap():
    global mocap_process, mocap_status, _mocap_process_cache

    stopped_any = False

    # 先终止 Dashboard 管理的进程
    if mocap_process and mocap_process.poll() is None:
        mocap_process.terminate()
        try:
            mocap_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mocap_process.kill()
        stopped_any = True

    # 终止所有外部动捕进程
    external_procs = _find_mocap_processes()
    for proc_info in external_procs:
        if not proc_info["managed"]:
            try:
                import psutil
                proc = psutil.Process(proc_info["pid"])
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                stopped_any = True
                print(f"已终止外部动捕进程 PID: {proc_info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    mocap_status = {"running": False, "pid": None, "start_time": None, "starting": False}
    _mocap_process_cache = {"data": [], "ts": 0.0}
    try:
        mark_tracking_stopped()
    except OSError as exc:
        print(f"Runtime status stop marker failed: {exc}")
    if stopped_any:
        return jsonify({"status": "ok"})
    else:
        return jsonify({"status": "ok", "message": "Not running"})


def _find_mocap_processes(use_cache: bool = True) -> list:
    """查找所有动捕进程 (main.py)，排除当前 Dashboard 进程

    Returns:
        list of dict: [{"pid": int, "cmdline": str, "managed": bool}, ...]
    """
    global _mocap_process_cache
    found = []
    if not HAS_PSUTIL:
        return found

    now = time.time()
    if use_cache and now - _mocap_process_cache.get("ts", 0.0) < MOCAP_SCAN_TTL_SECONDS:
        return _mocap_process_cache.get("data", [])

    current_pid = os.getpid()
    managed_pid = mocap_process.pid if mocap_process and mocap_process.poll() is None else None

    try:
        # Query only cheap fields for every process. Reading cmdline/cwd for
        # all Windows processes can block the Dashboard status endpoint for
        # several seconds, especially while camera and GPU services are busy.
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if proc.pid == current_pid:
                    continue
                proc_name = str(proc.info.get("name") or "").lower()
                if (
                    'python' not in proc_name
                    and 'python.exe' not in proc_name
                    and 'yolo26n-vrc-mocap' not in proc_name
                ):
                    continue
                cmdline = proc.cmdline()
                if not any('main.py' in arg or '--mocap-child' in arg for arg in cmdline):
                    continue

                # 排除 dashboard_server.py 自身 (防止误匹配)
                if any('dashboard' in arg for arg in cmdline):
                    continue

                is_managed = (proc.pid == managed_pid)
                found.append({"pid": proc.pid, "cmdline": " ".join(cmdline), "managed": is_managed})
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception:
        pass

    _mocap_process_cache = {"data": found, "ts": now}
    return found


def _runtime_state_process_alive(state: dict) -> bool:
    try:
        state_pid = int(state.get("pid"))
    except (TypeError, ValueError):
        return False

    if mocap_process and mocap_process.poll() is None:
        return mocap_process.pid == state_pid
    if not HAS_PSUTIL:
        return False
    try:
        process = psutil.Process(state_pid)
        cmdline = process.cmdline()
        return process.is_running() and any(
            "main.py" in arg or "--mocap-child" in arg
            for arg in cmdline
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


@app.route("/api/mocap/status", methods=["GET"])
def mocap_status_api():
    managed_running = bool(mocap_process and mocap_process.poll() is None)

    all_mocap_procs = _find_mocap_processes()
    managed_procs = [p for p in all_mocap_procs if p["managed"]]
    external_procs = [p for p in all_mocap_procs if not p["managed"]]

    any_running = len(all_mocap_procs) > 0

    if managed_running:
        if mocap_status.get("starting"):
            try:
                import tempfile
                log_path = Path(tempfile.gettempdir()) / "yolo26n_mocap.log"
                if log_path.exists():
                    text = _read_text_tail(log_path, 64 * 1024)
                    if (
                        "System initialized successfully!" in text
                        or "Mocap running!" in text
                        or "Warmup complete" in text
                        or "[Loop]" in text
                        or "[Status]" in text
                    ):
                        mocap_status["starting"] = False
            except Exception:
                pass
        return jsonify({
            **mocap_status,
            "running": True,
            "managed": True,
            "external_procs": external_procs,
            "stale_pids": [p["pid"] for p in external_procs],
        })

    elif any_running:
        mocap_status["running"] = False
        first_external = external_procs[0] if external_procs else managed_procs[0]
        return jsonify({
            **mocap_status,
            "running": True,
            "managed": False,
            "external_pid": first_external["pid"],
            "external_procs": external_procs,
            "stale_pids": [],
        })
    else:
        mocap_status.update({"running": False, "pid": None, "start_time": None, "starting": False})
        return jsonify({
            **mocap_status,
            "running": False,
            "managed": True,
            "external_procs": [],
            "stale_pids": [],
        })


@app.route("/api/mocap/kill_stale", methods=["POST"])
def kill_stale_api():
    """终止所有残留的动捕进程"""
    killed_count = kill_stale_mocap_processes()
    return jsonify({"status": "ok", "killed": killed_count})


@app.route("/api/mocap/log", methods=["GET"])
def mocap_log_api():
    """读取动捕进程的最新日志"""
    import tempfile
    log_path = Path(tempfile.gettempdir()) / "yolo26n_mocap.log"
    tail = int(request.args.get("tail", 50))
    if not log_path.exists():
        return jsonify({"log": "", "exists": False})
    try:
        recent, scanned_lines = _read_log_tail_lines(log_path, tail)
        return jsonify({"log": recent, "exists": True, "total_lines": scanned_lines, "tail_only": True})
    except Exception as e:
        return jsonify({"log": f"读取日志失败: {e}", "exists": True})


@app.route("/api/mocap/inference_status", methods=["GET"])
def inference_status_api():
    """从日志中解析推理模块的当前状态"""
    global _inference_status_cache
    import tempfile
    import re

    runtime_state = read_tracking_state(max_age_seconds=2.5)
    if (
        runtime_state
        and runtime_state.get("running")
        and not runtime_state.get("stale")
        and _runtime_state_process_alive(runtime_state)
    ):
        modules = runtime_state.get("modules", {})
        body = modules.get("body") or {}
        face = modules.get("face") or {}
        hand = modules.get("hand") or {}
        return jsonify({
            "body": bool(body.get("enabled") and body.get("initialized")),
            "face": bool(face.get("enabled") and face.get("initialized")),
            "hand": bool(hand.get("enabled") and hand.get("initialized")),
            "available": True,
            "running": True,
            "body_pose_nodes": body.get("pose_node_count", 0),
            "body_tracker_messages": body.get(
                "streamer_last_tracker_count",
                body.get("tracker_node_count", 0),
            ),
            "body_stream_sent": body.get("streamer_send_count", 0),
            "body_stream_errors": body.get("streamer_error_count", 0),
            "body_stream_queued": body.get("stream_queued", False),
            "source": "runtime_state",
            "cached": False,
        })

    mocap_running = bool(mocap_process and mocap_process.poll() is None) or bool(_find_mocap_processes())
    if not mocap_running:
        return jsonify({
            "body": False,
            "face": False,
            "hand": False,
            "available": True,
            "running": False,
            "body_pose_nodes": 0,
            "body_tracker_messages": 0,
            "body_stream_sent": 0,
            "body_stream_errors": 0,
            "body_stream_queued": False,
        })

    log_path = Path(tempfile.gettempdir()) / "yolo26n_mocap.log"
    if not log_path.exists():
        return jsonify({"body": None, "face": None, "hand": None, "available": False, "running": True})
    try:
        stat = log_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        now = time.time()
        cached = _inference_status_cache.get("data")
        if (
            cached is not None
            and _inference_status_cache.get("signature") == signature
            and now - _inference_status_cache.get("ts", 0.0) < INFERENCE_STATUS_TTL_SECONDS
        ):
            return jsonify({**cached, "cached": True})

        content = _read_text_tail(log_path)

        # 从 [Status] 行解析推理状态 (Body:ON/OFF Face:ON/OFF Hand:ON/OFF)
        body_status = None
        face_status = None
        hand_status = None
        body_pose_nodes = None
        body_tracker_messages = None
        body_stream_sent = None
        body_stream_errors = None
        body_stream_queued = None

        # 查找最后一个 [Status] 行
        status_matches = re.findall(
            r'\[Status\].*Body:(ON|OFF)(?:\([^)]*\))?\s+\|\s+Face:(ON|OFF)(?:\([^)]*\))?\s+\|\s+Hand:(ON|OFF)(?:\([^)]*\))?',
            content,
        )
        if not status_matches:
            status_matches = re.findall(
                r'\[Status\].*Body:(ON|OFF)(?:\([^)]*\))?.*Face:(ON|OFF)(?:\([^)]*\))?.*Hand:(ON|OFF)(?:\([^)]*\))?',
                content,
            )
        if status_matches:
            last = status_matches[-1]
            body_status = last[0] == "ON"
            face_status = last[1] == "ON"
            hand_status = last[2] == "ON"

        status_lines = re.findall(r'^\[Status\].*$', content, flags=re.MULTILINE)
        if status_lines:
            last_status_line = status_lines[-1]
            details_match = re.search(
                r'Body:(?:ON|OFF)\(([^)]*)\)',
                last_status_line,
            )
            if details_match:
                details = details_match.group(1)
                for key, attr in [
                    ("nodes", "body_pose_nodes"),
                    ("trackers", "body_tracker_messages"),
                    ("sent", "body_stream_sent"),
                    ("err", "body_stream_errors"),
                ]:
                    match = re.search(rf'{key}=([0-9]+)', details)
                    if not match:
                        continue
                    value = int(match.group(1))
                    if attr == "body_pose_nodes":
                        body_pose_nodes = value
                    elif attr == "body_tracker_messages":
                        body_tracker_messages = value
                    elif attr == "body_stream_sent":
                        body_stream_sent = value
                    elif attr == "body_stream_errors":
                        body_stream_errors = value

        loop_lines = re.findall(r'^\[Loop\].*$', content, flags=re.MULTILINE)
        if loop_lines:
            last_loop_line = loop_lines[-1]
            body_rate_match = re.search("\u8eab\u4f53=([0-9]+(?:\\.[0-9]+)?)%", last_loop_line)
            if body_rate_match:
                try:
                    body_status = float(body_rate_match.group(1)) > 0.0
                except ValueError:
                    pass
            pose_match = re.search(
                "\u59ff\u6001=([0-9]+)\u8282\u70b9/([0-9]+)\u70b9/\u961f\u5217=(ON|OFF)",
                last_loop_line,
            )
            if pose_match:
                body_pose_nodes = int(pose_match.group(1))
                body_tracker_messages = int(pose_match.group(2))
                body_stream_queued = pose_match.group(3) == "ON"

        # 也从 HotReload 行解析 (覆盖 Status 解析结果，因为 HotReload 更新)
        hotreload_matches = re.findall(
            r'\[HotReload\]\s+Body tracking:\s+(ON|OFF)', content
        )
        if hotreload_matches:
            body_status = hotreload_matches[-1] == "ON"

        hotreload_face = re.findall(
            r'\[HotReload\]\s+Face tracking:\s+(ON|OFF)', content
        )
        if hotreload_face:
            face_status = hotreload_face[-1] == "ON"

        hotreload_hand = re.findall(
            r'\[HotReload\]\s+Hand tracking:\s+(ON|OFF)', content
        )
        if hotreload_hand:
            hand_status = hotreload_hand[-1] == "ON"

        if body_status is None and "Body module: initialized" in content and "[Loop]" in content:
            body_status = True
        if face_status is None and "Face module: initialized" in content and "[Loop]" in content:
            face_status = True
        if hand_status is None and "Hand module: initialized" in content and "[Loop]" in content:
            hand_status = True

        data = {
            "body": body_status,
            "face": face_status,
            "hand": hand_status,
            "available": body_status is not None or face_status is not None or hand_status is not None,
            "running": True,
            "body_pose_nodes": body_pose_nodes,
            "body_tracker_messages": body_tracker_messages,
            "body_stream_sent": body_stream_sent,
            "body_stream_errors": body_stream_errors,
            "body_stream_queued": body_stream_queued,
            "cached": False,
        }
        _inference_status_cache = {"data": data, "ts": now, "signature": signature}
        return jsonify(data)
    except Exception as e:
        return jsonify({"body": None, "face": None, "hand": None, "available": False, "running": True, "error": str(e)})


# ============ 数据集管理 API ============

@app.route("/api/browse", methods=["GET"])
def browse_directory():
    """浏览服务器文件系统目录，用于选择数据集路径"""
    req_path = request.args.get("path", "").strip()

    # 默认从用户目录和常见位置开始
    if not req_path:
        # 提供快捷入口
        home = Path.home()
        shortcuts = [
            {"name": "桌面", "path": str(home / "Desktop")},
            {"name": "文档", "path": str(home / "Documents")},
            {"name": "下载", "path": str(home / "Downloads")},
            {"name": "用户目录", "path": str(home)},
        ]
        # 添加可用盘符
        import string
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                shortcuts.append({"name": f"{letter}:\\", "path": str(drive)})

        return jsonify({"type": "shortcuts", "shortcuts": shortcuts, "current": ""})

    current = Path(req_path)
    if not current.exists() or not current.is_dir():
        return jsonify({"type": "error", "message": f"路径不存在或不是目录: {req_path}"}), 400

    try:
        entries = []
        for item in sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if not item.is_dir():
                continue
            # 跳过隐藏目录和系统目录
            if item.name.startswith('.') or item.name.startswith('$'):
                continue
            try:
                entries.append({
                    "name": item.name,
                    "path": str(item),
                })
            except PermissionError:
                continue

        parent = str(current.parent) if current.parent != current else ""
        return jsonify({
            "type": "listing",
            "current": str(current),
            "parent": parent,
            "entries": entries,
        })
    except PermissionError:
        return jsonify({"type": "error", "message": f"无权限访问: {req_path}"}), 403
    except Exception as e:
        return jsonify({"type": "error", "message": str(e)}), 500


# ============ ???? API ============

MODEL_EXTENSIONS = {".pt", ".onnx", ".engine"}
MODELS_DIR = RUNTIME_ROOT / "models"


def _model_search_dirs():
    dirs = [MODELS_DIR, RUNTIME_ROOT, PROJECT_ROOT]
    unique = []
    for d in dirs:
        resolved = d.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _model_info(path: Path, current_model: str = "") -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
        "current": str(path) == current_model or path.name == current_model,
    }


@app.route("/api/models", methods=["GET"])
def list_runtime_models():
    global _model_cache
    config = load_config()
    current_model = str(config.get("yolo", {}).get("model_path", ""))
    now = time.time()
    if (
        _model_cache.get("data") is not None
        and _model_cache.get("current") == current_model
        and now - _model_cache.get("ts", 0.0) < MODEL_CACHE_TTL_SECONDS
    ):
        return jsonify(_model_cache["data"])

    models = []
    seen = set()

    for root in _model_search_dirs():
        if not root.exists():
            continue
        for item in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not item.is_file() or item.suffix.lower() not in MODEL_EXTENSIONS:
                continue
            resolved = item.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            models.append(_model_info(item, current_model))

    data = {
        "models": models,
        "current": current_model,
        "model_dir": str(MODELS_DIR),
        "extensions": sorted(MODEL_EXTENSIONS),
    }
    _model_cache = {"data": data, "ts": now, "current": current_model}
    return jsonify(data)


@app.route("/api/models/import", methods=["POST"])
def import_runtime_model():
    global _model_cache
    data = request.json or {}
    source_path = str(data.get("source_path", "")).strip()
    mode = data.get("mode", "copy")
    set_current = data.get("set_current", True) is not False

    if not source_path:
        return jsonify({"status": "error", "message": "source_path is required"}), 400

    source = Path(source_path).expanduser()
    if not source.exists() or not source.is_file():
        return jsonify({"status": "error", "message": f"Model file not found: {source_path}"}), 400
    if source.suffix.lower() not in MODEL_EXTENSIONS:
        return jsonify({"status": "error", "message": "Supported model extensions: .pt, .onnx, .engine"}), 400

    if mode == "absolute":
        model_path = str(source.resolve())
    else:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        target = MODELS_DIR / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        model_path = str(target)

    if set_current:
        with _config_lock:
            config = load_config()
            yolo_cfg = config.setdefault("yolo", {})
            yolo_cfg["model_path"] = model_path
            yolo_cfg["model_type"] = "custom"
            save_config(config)
    _model_cache = {"data": None, "ts": 0.0, "current": None}

    return jsonify({
        "status": "ok",
        "model_path": model_path,
        "copied": mode != "absolute",
        "current": set_current,
    })


# ============ 辅助函数 ============

def _auto_start_hmd_provider():
    """根据配置文件自动启动 HMD provider

    Dashboard 启动时，如果配置中 hmd.provider 不是 static，
    自动启动对应的 HMD 数据提供者，避免每次重启后需要手动启用。
    """
    global _hmd_provider
    config = load_config()
    hmd_cfg = config.get("hmd", {})
    provider_type = hmd_cfg.get("provider", "static")

    if provider_type in ("static", "none"):
        return

    try:
        import hmd_provider as _hmd_mod
        HMDProviderFactory = _hmd_mod.HMDProviderFactory
        kwargs = {
            "osc_listen_port": hmd_cfg.get("osc_listen_port", 9001),
            "pico_host": hmd_cfg.get("pico_host", "127.0.0.1"),
            "pico_port": hmd_cfg.get("pico_port", 7890),
            "pico_xrobo_host": hmd_cfg.get("pico_xrobo_host", "0.0.0.0"),
            "pico_xrobo_port": hmd_cfg.get("pico_xrobo_port", 39540),
            "quest_host": hmd_cfg.get("quest_host", "127.0.0.1"),
            "quest_port": hmd_cfg.get("quest_port", 53342),
            "quest_udp_host": hmd_cfg.get("quest_udp_host", "0.0.0.0"),
            "quest_udp_port": hmd_cfg.get("quest_udp_port", 39541),
            "sdk_udp_host": hmd_cfg.get("sdk_udp_host", "0.0.0.0"),
            "sdk_udp_port": hmd_cfg.get("sdk_udp_port", 39542),
            "prefer_openvr": hmd_cfg.get("prefer_openvr", True),
            "tracking_origin": hmd_cfg.get("tracking_origin", "unknown"),
            "remote_port": hmd_cfg.get("remote_port", 4243),
        }
        provider = HMDProviderFactory.create(provider_type, **kwargs)
        if provider.start():
            _hmd_provider = provider
            print(f"[AutoStart] HMD Provider: {provider.name} (from config)")
        else:
            print(f"[AutoStart] HMD Provider: {provider_type} start failed")
    except Exception as e:
        print(f"[AutoStart] HMD Provider: {provider_type} init error: {e}")


def run_server(host="0.0.0.0", port=8766):
    _dashboard_runtime["host"] = host
    _dashboard_runtime["port"] = int(port)
    print(f"Dashboard running at http://{host}:{port}")
    # 自动恢复 HMD provider (根据配置文件)
    _auto_start_hmd_provider()
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


# ============ HMD 位姿 API ============

@app.route("/api/hmd/status", methods=["GET"])
def hmd_status():
    """获取 HMD 提供者状态"""
    config = load_config()
    normalize_body_streaming_config(config)
    height_scaling = _body_height_scaling_config(config)
    with _hmd_lock:
        if _hmd_provider is None:
            return jsonify({
                "running": False,
                "provider": "none",
                "pose": None,
                "anchor_trusted": False,
                "anchor_applied": False,
                "height_diagnostics": _height_posture_state(None, height_scaling),
            })
        pose = _hmd_provider.get_pose()
        anchor_trusted = hmd_pose_anchor_trusted(pose)
        body_cfg = config.get("streaming", {}).get("body", {})
        status = {
            "running": True,
            "provider": _hmd_provider.name,
            "anchor_trusted": anchor_trusted,
            "anchor_applied": bool(
                anchor_trusted and not bool(body_cfg.get("decouple_from_hmd", False))
            ),
            "pose": {
                "position": pose.position,
                "rotation": pose.rotation,
                "quaternion": pose.quaternion,
                "valid": pose.valid,
                "timestamp": pose.timestamp,
                "tracking_origin": pose.tracking_origin,
                "actual_height": pose.actual_height,
                "height_reference": pose.height_reference,
            },
            "height_diagnostics": _height_posture_state(pose.actual_height, height_scaling),
        }
        if hasattr(_hmd_provider, "get_status_extra"):
            try:
                status.update(_hmd_provider.get_status_extra())
            except Exception:
                pass
        return jsonify(status)


@app.route("/api/hmd/standing_height", methods=["POST"])
def hmd_record_standing_height():
    with _hmd_lock:
        if _hmd_provider is None:
            return jsonify({"status": "error", "message": "HMD provider is not running"}), 400
        pose = _hmd_provider.get_pose()
    if not pose.valid or pose.actual_height is None:
        return jsonify({
            "status": "error",
            "message": "当前 HMD 没有可用的 floor/standing 实际高度，无法记录站立眼高",
            "tracking_origin": pose.tracking_origin,
            "height_reference": pose.height_reference,
        }), 400

    standing_height = round(float(pose.actual_height), 3)
    with _config_lock:
        config = load_config()
        normalize_body_streaming_config(config)
        scaling = _body_height_scaling_config(config)
        scaling["standing_eye_height"] = standing_height
        scaling["standing_eye_height_calibrated"] = True
        vrc_cfg = config.setdefault("vrc", {})
        vrc_scaling = vrc_cfg.setdefault("hmd_height_scaling", {})
        if isinstance(vrc_scaling, dict):
            vrc_scaling["standing_eye_height"] = standing_height
            vrc_scaling["standing_eye_height_calibrated"] = True
        save_config(config)
    return jsonify({
        "status": "ok",
        "standing_eye_height": standing_height,
        "height_diagnostics": _height_posture_state(pose.actual_height, scaling),
    })


@app.route("/api/hmd/start", methods=["POST"])
def hmd_start():
    """启动 HMD 数据提供者"""
    global _hmd_provider
    with _hmd_lock:
        # 先停止已有的
        if _hmd_provider is not None:
            _hmd_provider.stop()
            _hmd_provider = None

        data = request.json or {}
        provider_type = data.get("provider", "static")
        if provider_type == "none" or provider_type == "static":
            return jsonify({"status": "ok", "provider": "static", "message": "HMD 提供者已关闭 (static 模式)"})

        try:
            import importlib
            import hmd_provider as _hmd_mod

            # 条件 reload: 仅当 hmd_provider.py 文件修改时间变化时才重新加载
            # 避免每次 API 调用都 reload 带来的性能开销
            _hmd_mod_path = getattr(_hmd_mod, "__file__", None)
            if _hmd_mod_path:
                from pathlib import Path
                _hmd_mod_path_obj = Path(_hmd_mod_path)
                if not getattr(sys, "frozen", False) and _hmd_mod_path_obj.exists():
                    _hmd_mod_mtime = _hmd_mod_path_obj.stat().st_mtime
                    if not hasattr(hmd_start, "_last_mtime") or hmd_start._last_mtime < _hmd_mod_mtime:
                        importlib.reload(_hmd_mod)
                        hmd_start._last_mtime = _hmd_mod_mtime

            HMDProviderFactory = _hmd_mod.HMDProviderFactory
            kwargs = {
                "osc_listen_port": data.get("osc_listen_port", 9001),
                "pico_host": data.get("pico_host", "127.0.0.1"),
                "pico_port": data.get("pico_port", 7890),
                "pico_xrobo_host": data.get("pico_xrobo_host", "0.0.0.0"),
                "pico_xrobo_port": data.get("pico_xrobo_port", 39540),
                "quest_host": data.get("quest_host", "127.0.0.1"),
                "quest_port": data.get("quest_port", 53342),
                "quest_udp_host": data.get("quest_udp_host", "0.0.0.0"),
                "quest_udp_port": data.get("quest_udp_port", 39541),
                "sdk_udp_host": data.get("sdk_udp_host", "0.0.0.0"),
                "sdk_udp_port": data.get("sdk_udp_port", 39542),
                "prefer_openvr": data.get("prefer_openvr", True),
                "tracking_origin": data.get("tracking_origin", "unknown"),
                "remote_port": data.get("remote_port", 4243),
            }
            provider = HMDProviderFactory.create(provider_type, **kwargs)
            # 检测工厂是否回退到 StaticProvider (说明请求的 provider 未注册)
            if provider.name == "static" and provider_type not in ("static", "none"):
                return jsonify({"status": "error", "provider": provider_type,
                                "message": f"HMD 提供者 '{provider_type}' 未注册或加载失败，已回退到 static。请重启 Dashboard 以加载最新模块。"})
            if provider.start():
                _hmd_provider = provider
                # 同步配置文件: 保存当前 provider 类型，重启后自动恢复
                try:
                    with _config_lock:
                        config = load_config()
                        hmd_config = config.setdefault("hmd", {})
                        hmd_config["provider"] = provider.name
                        hmd_config["osc_listen_port"] = data.get("osc_listen_port", 9001)
                        hmd_config["pico_host"] = data.get("pico_host", "127.0.0.1")
                        hmd_config["pico_port"] = data.get("pico_port", 7890)
                        hmd_config["pico_xrobo_host"] = data.get("pico_xrobo_host", "0.0.0.0")
                        hmd_config["pico_xrobo_port"] = data.get("pico_xrobo_port", 39540)
                        hmd_config["quest_host"] = data.get("quest_host", "127.0.0.1")
                        hmd_config["quest_port"] = data.get("quest_port", 53342)
                        hmd_config["quest_udp_host"] = data.get("quest_udp_host", "0.0.0.0")
                        hmd_config["quest_udp_port"] = data.get("quest_udp_port", 39541)
                        hmd_config["sdk_udp_host"] = data.get("sdk_udp_host", "0.0.0.0")
                        hmd_config["sdk_udp_port"] = data.get("sdk_udp_port", 39542)
                        hmd_config["prefer_openvr"] = data.get("prefer_openvr", True)
                        hmd_config["tracking_origin"] = data.get("tracking_origin", "unknown")
                        hmd_config["remote_port"] = data.get("remote_port", 4243)
                        hmd_config["share_with_dashboard"] = True
                        hmd_config["dashboard_host"] = "127.0.0.1"
                        hmd_config["dashboard_port"] = _dashboard_runtime["port"]
                        save_config(config)
                except Exception:
                    pass
                return jsonify({"status": "ok", "provider": provider.name, "message": f"HMD 提供者已启动: {provider.name}"})
            else:
                return jsonify({"status": "error", "provider": provider_type, "message": f"HMD 提供者启动失败: {provider_type}，请检查依赖和连接"})
        except ImportError as e:
            return jsonify({"status": "error", "provider": provider_type, "message": f"缺少依赖: {e}"})
        except Exception as e:
            return jsonify({"status": "error", "provider": provider_type, "message": f"启动失败: {e}"})


@app.route("/api/hmd/stop", methods=["POST"])
def hmd_stop():
    """停止 HMD 数据提供者"""
    global _hmd_provider
    with _hmd_lock:
        if _hmd_provider is not None:
            _hmd_provider.stop()
            name = _hmd_provider.name
            _hmd_provider = None
            # 同步配置文件: 将 provider 设为 static，避免重启后自动启动
            try:
                with _config_lock:
                    config = load_config()
                    config.setdefault("hmd", {})["provider"] = "static"
                    save_config(config)
            except Exception:
                pass
            return jsonify({"status": "ok", "message": f"HMD 提供者已停止: {name}"})
        return jsonify({"status": "ok", "message": "HMD 提供者未运行"})


# ============ 识别模块独立开关 API ============

@app.route("/api/tracking/toggle", methods=["POST"])
def tracking_toggle():
    """独立控制识别模块开关

    通过修改配置文件实现，动捕进程会自动热更新。
    支持的模块: body, face, hand

    请求体: {"module": "body|face|hand", "enabled": true|false}
    """
    data = request.json or {}
    module = data.get("module", "")
    enabled = data.get("enabled", True)

    if module not in ("body", "face", "hand"):
        return jsonify({"status": "error", "message": f"未知模块: {module}，支持: body, face, hand"}), 400

    with _config_lock:
        config = load_config()
        if module not in config:
            config[module] = {}
        config[module]["enabled"] = enabled
        save_config(config)

    module_names = {"body": "身体姿态", "face": "面部表情", "hand": "手势追踪"}
    return jsonify({
        "status": "ok",
        "message": f"{module_names.get(module, module)} 已{'启用' if enabled else '关闭'}",
        "module": module,
        "enabled": enabled,
    })


@app.route("/api/tracking/status", methods=["GET"])
def tracking_status():
    """Return configured switches plus structured child-process diagnostics."""
    config = load_config()
    state = read_tracking_state(max_age_seconds=2.5)
    runtime_available = bool(
        state
        and not state.get("stale")
        and state.get("running")
        and _runtime_state_process_alive(state)
    )
    modules = state.get("modules", {}) if runtime_available else {}
    return jsonify({
        "body": config.get("body", {}).get("enabled", True),
        "face": config.get("face", {}).get("enabled", True),
        "hand": config.get("hand", {}).get("enabled", True),
        "running": bool(runtime_available),
        "runtime_available": bool(runtime_available),
        "runtime_stale": bool(state.get("stale")) if state else False,
        "age_seconds": state.get("age_seconds") if state else None,
        "fps": state.get("fps") if runtime_available else None,
        "frame_count": state.get("frame_count") if runtime_available else None,
        "parallel_inference": state.get("parallel_inference") if runtime_available else None,
        "body_details": modules.get("body"),
        "face_details": modules.get("face"),
        "hand_details": modules.get("hand"),
        "hmd": state.get("hmd") if runtime_available else None,
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO26n VRC Mocap Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    run_server(args.host, args.port)
