#!/usr/bin/env python3
"""
YOLO26n VRC Motion Capture System
基于 YOLO26n 姿态估计的 VRChat 实时全身动捕系统

架构: 三个独立模块 (body/face/hand)，各自管理推理和推流。
每个模块拥有独立的 OSC 推流地址，可独立启停。
"""

import os
import sys
import json
import copy
import cv2
import time
import signal
import argparse
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, List
from runtime_paths import bin_dir, resource_root, runtime_root

# 将 Ultralytics 配置目录设到项目下，避免 AppData 权限问题
_ultra_cfg = str(runtime_root() / ".ultralytics")
os.makedirs(_ultra_cfg, exist_ok=True)
os.environ["ULTRALYTICS_SETTINGS"] = _ultra_cfg

sys.path.insert(0, str(bin_dir()))

from camera_capture import CameraCapture, ImageProcessor, timestamp_rate
from edition import current_edition, yolo_available
from hmd_provider import HMDProviderFactory
from model_presets import MODEL_PRESETS
from modules import BodyModule, FaceModule, HandModule
from runtime_status import write_tracking_state


class VRCMocapSystem:
    def __init__(self, config_path: str = None):
        self._config_path = config_path
        self.config = self._load_config(config_path)
        self.running = False
        self._loop_timestamps = deque(maxlen=60)
        self._config_mtime = 0
        self._last_config_check_time = 0.0
        self._config_check_interval = 1.0

        # 三个独立模块
        self._modules: Dict[str, object] = {}  # {"body": BodyModule, "face": FaceModule, "hand": HandModule}
        self._hmd_provider = None
        self.camera = None
        self.secondary_camera = None
        self._secondary_camera_signature = None
        self._inference_executor = None
        self._stopped = False

        self.setup_signal_handlers()

    def _load_config(self, config_path: str = None) -> Dict:
        default_config = {
            "camera": {
                "camera_id": 0,
                "width": 1280,
                "height": 720,
                "fps": 30,
                "flip_horizontal": True,
                "zoom": 0.0
            },
            "camera_secondary": {
                "enabled": False,
                "camera_id": 1,
                "width": 640,
                "height": 480,
                "fps": 30,
                "flip_horizontal": True,
                "zoom": 0.0
            },
            "yolo": {
                "model_path": "yolo26n-pose.pt",
                "device": "auto",
                "conf_threshold": 0.5,
                "half": True
            },
            "streaming": {
                "host": "127.0.0.1",
                "port": 9000,
                "body": {
                    "host": "127.0.0.1",
                    "port": 9000,
                    "protocol": "osc",
                    "streaming_enabled": True,
                    "send_position": True,
                    "send_rotation": True,
                    "send_head_position": False,
                    "decouple_from_hmd": False,
                    "hmd_height_scaling": {
                        "enabled": True,
                        "floor_clearance": 0.03,
                        "min_hmd_y": 0.35,
                        "min_body_scale": 0.45,
                        "max_body_scale": 1.0,
                        "vertical_only": True,
                        "require_actual_height": True,
                        "crouch_from_hmd_height": True,
                        "standing_eye_height": 1.65,
                        "standing_eye_height_calibrated": False,
                        "auto_learn_standing_height": True,
                        "auto_learn_allow_downward": True,
                        "auto_learn_downward_max_delta": 0.25,
                        "auto_learn_standing_pose_threshold": 0.8,
                        "min_standing_eye_height": 1.2,
                        "max_standing_eye_height": 2.2,
                        "auto_learn_min_height": 1.2,
                        "auto_learn_window": 20,
                        "auto_learn_stability_tolerance": 0.03,
                        "crouch_start_ratio": 0.92,
                        "seated_ratio": 0.55,
                        "seated_eye_height": 0.95,
                        "seated_hip_height": 0.45,
                        "standing_hip_min_height": 0.78,
                        "standing_chest_min_height": 1.05,
                        "crouch_knee_forward": 0.18,
                        "crouch_knee_outward": 0.06,
                    },
                },
                "face": {
                    "host": "127.0.0.1",
                    "port": 9000,
                    "vmc_port": 39539,
                    "streaming_enabled": True,
                },
                "hand": {
                    "host": "127.0.0.1",
                    "port": 9000,
                    "vmt_host": "127.0.0.1",
                    "vmt_port": 39570,
                    "streaming_enabled": True,
                    "desktop_osc_enabled": True,
                    "vmt_enabled": True,
                    "vmt_local_mode": True,
                    "vmt_skeleton_enabled": True,
                    "vmt_input_enabled": True,
                    "vmt_auto_enable_skeleton": True,
                    "vmt_register_on_steamvr_startup": True,
                    "vmt_keep_controllers_active": True,
                    "vmt_hand_selection_priority": 100,
                    "vmt_left_device_index": 1,
                    "vmt_right_device_index": 2,
                    "custom_parameters_enabled": True,
                    "lost_reset_timeout": 0.35,
                    "desktop_left_gesture_parameters": [
                        "DynamicGesture/CurrentGestureLeft",
                        "DynamicGesture/CurrentLeftIndex",
                        "MocapGestureLeft",
                    ],
                    "desktop_right_gesture_parameters": [
                        "DynamicGesture/CurrentGestureRight",
                        "DynamicGesture/CurrentRightIndex",
                        "MocapGestureRight",
                    ],
                },
            },
            "vrc": {
                "host": "127.0.0.1",
                "port": 9000,
                "protocol": "osc",
                "send_position": True,
                "send_rotation": True,
                "send_head_position": False,
                "decouple_from_hmd": False,
            },
            "steamvr": {
                "smoothing": 50,
                "additional_smoothing": 0.5,
                "camera_latency": 0,
                "hmd_to_neck_offset": [0, -0.2, 0.1],
            },
            "display": {"show_preview": True, "window_name": "YOLO26n VRC Mocap"},
            "processing": {
                "smoothing": "one_euro",
                "smoothing_factor": 0.3,
                "one_euro_min_cutoff": 1.0,
                "one_euro_beta": 0.007,
                "average_smoothing_enabled": True,
                "average_window": 4,
                "ema_alpha": 0.35,
                "reset_on_tracking_lost": True,
                "module_rate_limit_enabled": True,
                "reuse_last_result": True,
                "parallel_inference_enabled": True,
                "body_max_fps": 20,
                "hmd_pose_hold_seconds": 5.0,
                "hmd_pose_refresh_fps": 15.0,
                "face_max_fps": 30,
                "hand_max_fps": 20,
                "dual_camera_body_enabled": False,
                "secondary_body_max_fps": 12,
                "dual_camera_fusion_mode": "confidence",
                "fusion_confidence_margin": 0.12,
                "fusion_min_confidence": 0.15,
                "fusion_max_pose_delta": 0.75,
            },
            "body": {
                "enabled": True,
                "detect_torso": True,
                "detect_left_arm": True,
                "detect_right_arm": True,
                "detect_left_leg": True,
                "detect_right_leg": True,
                "inference_backend": "yolo",
            },
            "face": {
                "enabled": True,
                "model_path": "face_landmarker.task",
                "mouth_shape_mapping_enabled": True,
                "mouth_shape_gain": 1.25,
                "mouth_shape_deadzone": 0.03,
                "eyelid_independent_enabled": True,
                "eyelid_close_gain": 1.35,
                "eyelid_close_deadzone": 0.04,
                "eyelid_send_combined": False,
                "vrcft_calibration": {
                    "enabled": False,
                    "min_values": {},
                    "max_values": {},
                    "updated_at": None,
                },
            },
            "hand": {
                "enabled": True,
                "model_path": "hand_landmarker.task",
                "roi_assist_enabled": True,
                "roi_scale": 2.6,
                "roi_min_size": 96,
                "roi_min_confidence": 0.05,
                "roi_full_frame_interval": 12,
            },
            "mediapipe_pose": {
                "enabled": False,
                "mode": "fallback",
                "model_variant": "heavy",
                "model_path": "pose_landmarker_heavy.task",
                "blend_weight": 0.35,
                "min_detection_confidence": 0.5,
                "min_presence_confidence": 0.5,
                "min_tracking_confidence": 0.5,
            },
            "hmd": {
                "provider": "static",
                "share_with_dashboard": True,
                "dashboard_host": "127.0.0.1",
                "dashboard_port": 8766,
                "osc_listen_port": 9001,
                "pico_host": "127.0.0.1",
                "pico_port": 7890,
                "pico_xrobo_host": "0.0.0.0",
                "pico_xrobo_port": 39540,
                "quest_host": "127.0.0.1",
                "quest_port": 53342,
                "quest_udp_host": "0.0.0.0",
                "quest_udp_port": 39541,
                "sdk_udp_host": "0.0.0.0",
                "sdk_udp_port": 39542,
                "remote_port": 4243,
                "prefer_openvr": True,
                "tracking_origin": "unknown",
            },
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                self._merge_config(default_config, loaded_config)
                print(f"Config loaded from: {config_path}")
            except Exception as e:
                print(f"Error loading config: {e}, using default")
        else:
            # 自动查找默认配置文件
            default_path = str(runtime_root() / "config" / "config.json")
            if not os.path.exists(default_path):
                default_path = str(resource_root() / "config" / "config.json")
            if os.path.exists(default_path):
                try:
                    with open(default_path, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)
                    self._merge_config(default_config, loaded_config)
                    self._config_path = default_path
                    print(f"Config loaded from: {default_path}")
                except Exception as e:
                    print(f"Error loading config: {e}, using default")
            else:
                print("Using default configuration (no config file found)")

        return default_config

    def _merge_config(self, default: Dict, override: Dict):
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\nShutdown signal received")
        self.stop()
        sys.exit(0)

    def _find_available_runtime_model(self) -> Optional[str]:
        """Find a likely pose model without falling back to YOLO detection weights."""
        model_exts = {".pt", ".onnx", ".engine"}
        search_dirs = [
            runtime_root() / "models",
            runtime_root(),
            resource_root() / "models",
            resource_root(),
        ]
        best_models = []
        seen = set()
        for directory in search_dirs:
            if not directory.exists():
                continue
            for item in directory.iterdir():
                if not item.is_file() or item.suffix.lower() not in model_exts:
                    continue
                stem = item.stem.lower()
                is_named_yolo = stem.startswith(("yolo", "yolov"))
                is_named_pose = "pose" in stem
                if is_named_yolo and not is_named_pose:
                    continue
                resolved = item.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                # Custom exports commonly use names such as best.pt inside the
                # models directory. Prefer those and explicitly named pose
                # files over unrelated root-level model files.
                priority = 1
                if directory.name.lower() == "models":
                    priority += 2
                if is_named_pose:
                    priority += 2
                best_models.append((priority, item.stat().st_mtime, str(resolved)))
        if best_models:
            best_models.sort(reverse=True)
            return best_models[0][2]
        return None

    @staticmethod
    def _camera_signature(cfg: Dict) -> tuple:
        return (
            bool(cfg.get("enabled", False)),
            int(cfg.get("camera_id", 1)),
            int(cfg.get("width", 640)),
            int(cfg.get("height", 480)),
            int(cfg.get("fps", 30)),
            bool(cfg.get("flip_horizontal", True)),
            float(cfg.get("zoom", 0.0) or 0.0),
        )

    def _start_secondary_camera(self, cfg: Dict) -> bool:
        if not cfg.get("enabled", False):
            return False
        primary_id = self.config.get("camera", {}).get("camera_id", 0)
        secondary_id = int(cfg.get("camera_id", 1))
        if secondary_id == int(primary_id):
            print(f"  Secondary camera disabled: camera_id {secondary_id} matches primary camera")
            return False
        camera = CameraCapture(
            camera_id=secondary_id,
            width=int(cfg.get("width", 640)),
            height=int(cfg.get("height", 480)),
            fps=int(cfg.get("fps", 30)),
            flip_horizontal=cfg.get("flip_horizontal", True),
            zoom=cfg.get("zoom", 0.0),
        )
        if not camera.start():
            print(f"  Secondary camera {secondary_id}: start failed")
            return False
        self.secondary_camera = camera
        self._secondary_camera_signature = self._camera_signature(cfg)
        print(
            f"  Secondary camera: {cfg.get('width', 640)}x{cfg.get('height', 480)} "
            f" @ {cfg.get('fps', 30)}fps (id={secondary_id})"
        )
        return True

    def _stop_secondary_camera(self):
        if self.secondary_camera:
            self.secondary_camera.stop()
            self.secondary_camera = None
        self._secondary_camera_signature = None

    def _sync_secondary_camera(self, cfg: Dict):
        signature = self._camera_signature(cfg)
        if not cfg.get("enabled", False):
            self._stop_secondary_camera()
            return
        if self.secondary_camera and signature == self._secondary_camera_signature:
            return
        self._stop_secondary_camera()
        self._start_secondary_camera(cfg)

    def initialize(self) -> bool:
        print("=" * 50)
        print(f"  YOLO26n VRC Mocap System ({current_edition()} edition)")
        print("=" * 50)

        # 初始化摄像头
        cam_cfg = self.config["camera"]
        self.camera = CameraCapture(
            camera_id=cam_cfg["camera_id"],
            width=cam_cfg["width"],
            height=cam_cfg["height"],
            fps=cam_cfg["fps"],
            flip_horizontal=cam_cfg.get("flip_horizontal", True),
            zoom=cam_cfg.get("zoom", 0.0),
        )
        if not self.camera.start():
            print("Error: Could not initialize camera")
            return False
        print(f"  Camera: {cam_cfg['width']}x{cam_cfg['height']} @ {cam_cfg['fps']}fps")
        secondary_cfg = self.config.get("camera_secondary", {})
        if not self.config.get("processing", {}).get("dual_camera_body_enabled", False):
            secondary_cfg = {**secondary_cfg, "enabled": False}
        self._start_secondary_camera(secondary_cfg)

        body_cfg = self.config.setdefault("body", {})
        if not yolo_available() and body_cfg.get("inference_backend", "yolo") != "mediapipe_pose":
            print("  Lite edition: body backend forced to MediaPipe Pose")
            body_cfg["inference_backend"] = "mediapipe_pose"
        needs_yolo = (
            yolo_available()
            and body_cfg.get("enabled", True)
            and body_cfg.get("inference_backend", "yolo") == "yolo"
        )

        # 自动查找模型 (仅 YOLO 后端需要)
        yolo_cfg = self.config.get("yolo", {})
        model_path = yolo_cfg.get("model_path", "yolo26n-pose.pt")
        if needs_yolo and not os.path.isabs(model_path):
            # 相对路径: 先检查 bin/ 目录，再检查项目根目录
            bin_path = str(bin_dir())
            root_dir = str(resource_root())
            candidates = [
                os.path.join(bin_path, model_path),
                str(runtime_root() / "models" / model_path),
                str(runtime_root() / model_path),
                str(resource_root() / "models" / model_path),
                os.path.join(root_dir, model_path),
            ]
            found = False
            for candidate in candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    found = True
                    break
            if not found:
                # 尝试搜索最新训练的模型
                latest = self._find_available_runtime_model()
                if latest:
                    model_path = latest
                    print(f"  Using runtime model: {model_path}")
                else:
                    # 尝试自动下载
                    print(f"  Model not found: {model_path}")
                    try:
                        from ultralytics import YOLO
                        print("  Downloading YOLO11n-pose model...")
                        yolo = YOLO("yolo11n-pose.pt")
                        model_path = "yolo11n-pose.pt"
                        print("  Download complete")
                    except Exception as e:
                        print(f"Failed to download model: {e}")
                        return False

        self.config["yolo"]["model_path"] = model_path

        # ===== 初始化三个独立模块 =====
        streaming_cfg = self.config.get("streaming", {})

        # 身体模块
        body_cfg = self.config.get("body", {})
        body_streaming_cfg = streaming_cfg.get("body", {})
        body_streaming_cfg.setdefault("host", streaming_cfg.get("host", "127.0.0.1"))
        body_streaming_cfg.setdefault("port", streaming_cfg.get("port", 9000))
        # 传入 yolo 和 processing 配置给 body 模块
        body_full_cfg = {
            **body_cfg,
            "yolo": self.config.get("yolo", {}),
            "processing": self.config.get("processing", {}),
            "camera": self.config.get("camera", {}),
            "camera_secondary": self.config.get("camera_secondary", {}),
            "mediapipe_pose": self.config.get("mediapipe_pose", {}),
        }

        body_module = BodyModule(config=body_full_cfg, streaming_config=body_streaming_cfg)
        if body_module.enabled:
            if body_module.initialize():
                self._modules["body"] = body_module
                print("  Body module: initialized")
            else:
                print("  Body module: initialization failed")
        else:
            print("  Body module: disabled (config)")

        # 面部模块
        face_cfg = {
            **self.config.get("face", {}),
            "processing": self.config.get("processing", {}),
        }
        face_streaming_cfg = streaming_cfg.get("face", {})
        face_streaming_cfg.setdefault("host", streaming_cfg.get("host", "127.0.0.1"))
        face_streaming_cfg.setdefault("port", streaming_cfg.get("port", 9000))
        face_streaming_cfg.setdefault("vmc_port", 39539)

        face_module = FaceModule(config=face_cfg, streaming_config=face_streaming_cfg)
        if face_module.enabled:
            if face_module.initialize():
                self._modules["face"] = face_module
                print("  Face module: initialized")
            else:
                print("  Face module: initialization failed")
        else:
            print("  Face module: disabled (config)")

        # 手部模块
        hand_cfg = self.config.get("hand", {})
        hand_streaming_cfg = streaming_cfg.get("hand", {})
        hand_streaming_cfg.setdefault("host", streaming_cfg.get("host", "127.0.0.1"))
        hand_streaming_cfg.setdefault("port", streaming_cfg.get("port", 9000))
        hand_streaming_cfg.setdefault("vmt_local_mode", True)
        if hand_streaming_cfg.get("vmt_local_mode", True):
            hand_streaming_cfg["vmt_host"] = "127.0.0.1"
        else:
            hand_streaming_cfg.setdefault(
                "vmt_host", hand_streaming_cfg.get("host", "127.0.0.1")
            )
        hand_streaming_cfg.setdefault("vmt_port", 39570)
        hand_streaming_cfg.setdefault("desktop_osc_enabled", True)
        hand_streaming_cfg.setdefault("vmt_enabled", True)
        hand_streaming_cfg.setdefault("vmt_skeleton_enabled", True)
        hand_streaming_cfg.setdefault("vmt_input_enabled", True)
        hand_streaming_cfg.setdefault("vmt_auto_enable_skeleton", True)
        hand_streaming_cfg.setdefault("vmt_register_on_steamvr_startup", True)
        hand_streaming_cfg.setdefault("vmt_keep_controllers_active", True)
        hand_streaming_cfg.setdefault("vmt_hand_selection_priority", 100)
        hand_streaming_cfg.setdefault("vmt_left_device_index", 1)
        hand_streaming_cfg.setdefault("vmt_right_device_index", 2)
        hand_streaming_cfg.setdefault("custom_parameters_enabled", True)
        hand_streaming_cfg.setdefault("lost_reset_timeout", 0.35)
        hand_streaming_cfg.setdefault(
            "desktop_left_gesture_parameters",
            [
                "DynamicGesture/CurrentGestureLeft",
                "DynamicGesture/CurrentLeftIndex",
                "MocapGestureLeft",
            ],
        )
        hand_streaming_cfg.setdefault(
            "desktop_right_gesture_parameters",
            [
                "DynamicGesture/CurrentGestureRight",
                "DynamicGesture/CurrentRightIndex",
                "MocapGestureRight",
            ],
        )
        # 传入 steamvr 配置给 hand 模块 (hmd_to_neck_offset)
        hand_full_cfg = {
            **hand_cfg,
            "steamvr": self.config.get("steamvr", {}),
            "processing": self.config.get("processing", {}),
        }

        hand_module = HandModule(config=hand_full_cfg, streaming_config=hand_streaming_cfg)
        if hand_module.enabled:
            if hand_module.initialize():
                self._modules["hand"] = hand_module
                print("  Hand module: initialized")
            else:
                print("  Hand module: initialization failed")
        else:
            print("  Hand module: disabled (config)")

        if self.config.get("processing", {}).get("parallel_inference_enabled", True):
            self._inference_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="mocap-inference",
            )
            print("  Parallel inference: enabled (body + hand)")

        # ===== 初始化 HMD 提供者 =====
        self._init_hmd_provider()

        # ===== 打印模块状态 =====
        print("\n--- Module Status ---")
        for name, module in self._modules.items():
            status = module.get_status()
            print(f"  {name}: enabled={status['enabled']} | streaming={status['streaming_enabled']} | connected={status.get('connected', 'N/A')}")
        print("--- End Status ---\n")

        # 记录配置文件修改时间
        try:
            config_path = self._config_path or "config/config.json"
            if Path(config_path).exists():
                self._config_mtime = Path(config_path).stat().st_mtime
        except Exception:
            pass

        print("\nSystem initialized successfully!")
        return True

    def _init_hmd_provider(self):
        """初始化 HMD 位姿数据提供者"""
        hmd_cfg = self.config.get("hmd", {})
        hmd_provider_type = hmd_cfg.get("provider", "static")
        if hmd_provider_type == "static":
            return

        use_dashboard_bridge = (
            os.environ.get("YOLO26N_HMD_DASHBOARD_BRIDGE") == "1"
            and bool(hmd_cfg.get("share_with_dashboard", True))
            and hmd_provider_type not in ("dashboard_api", "none", "static")
        )

        hmd_kwargs = {
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
            "tracking_origin": hmd_cfg.get("tracking_origin", "unknown"),
            "prefer_openvr": hmd_cfg.get("prefer_openvr", True),
            "dashboard_host": os.environ.get(
                "YOLO26N_HMD_DASHBOARD_HOST",
                hmd_cfg.get("dashboard_host", "127.0.0.1"),
            ),
            "dashboard_port": int(os.environ.get(
                "YOLO26N_HMD_DASHBOARD_PORT",
                hmd_cfg.get("dashboard_port", 8766),
            )),
            "dashboard_poll_fps": self.config.get("processing", {}).get(
                "hmd_pose_refresh_fps", 15.0
            ),
            "remote_port": hmd_cfg.get("remote_port", 4243),
        }

        resolved_provider_type = "dashboard_api" if use_dashboard_bridge else hmd_provider_type
        hmd_provider = HMDProviderFactory.create(resolved_provider_type, **hmd_kwargs)
        if hmd_provider and hmd_provider.start():
            self._hmd_provider = hmd_provider
            # 设置到需要的模块
            for module in self._modules.values():
                if hasattr(module, 'set_hmd_provider'):
                    module.set_hmd_provider(hmd_provider)
            if use_dashboard_bridge:
                print(
                    "  HMD Provider: Dashboard bridge "
                    f"(source={hmd_provider_type}, "
                    f"endpoint={hmd_kwargs['dashboard_host']}:{hmd_kwargs['dashboard_port']})"
                )
            else:
                print(f"  HMD Provider: {hmd_provider.name}")
        else:
            print(f"  HMD Provider: {resolved_provider_type} start failed, using static fallback")

    def start(self):
        self.running = True
        self._stopped = False

        print("\n" + "=" * 50)
        print("  Mocap running!")
        print("  [q] Quit  [d] Toggle display")
        print("  [r] Reset body filters  [f] Calibrate face")
        print("  [l] Toggle low-light mode  [h] Reset hand filters")
        print("=" * 50 + "\n")

        self._write_runtime_status(0.0, 0, running=True)
        self._main_loop()

    def _main_loop(self):
        window_name = self.config["display"]["window_name"]
        show_preview = self.config["display"]["show_preview"]

        if show_preview:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        last_status_time = time.time()
        last_log_time = time.time()
        last_runtime_status_time = 0.0
        consecutive_none_frames = 0
        frame_count = 0
        sent_count = 0
        last_camera_frame_count = -1

        # 统计
        stats = {"total_frames": 0}

        try:
            while self.running:
                frame, camera_frame_count = self.camera.wait_for_frame(
                    last_frame_count=last_camera_frame_count,
                    timeout=0.1,
                    copy=show_preview,
                )
                if frame is None:
                    if camera_frame_count == last_camera_frame_count:
                        continue
                    consecutive_none_frames += 1
                    if consecutive_none_frames == 1:
                        print("[Loop] WARNING: 摄像头返回空帧")
                    elif consecutive_none_frames % 300 == 0:
                        print(f"[Loop] WARNING: 已连续 {consecutive_none_frames} 帧无画面，摄像头可能断开")
                    continue
                elif consecutive_none_frames > 0:
                    print(f"[Loop] 摄像头恢复 (之前连续 {consecutive_none_frames} 帧空帧)")
                    consecutive_none_frames = 0

                last_camera_frame_count = camera_frame_count

                stats["total_frames"] += 1
                frame_count += 1
                start_time = time.perf_counter()

                # ===== 分发帧到各活跃模块 =====
                body_data = None
                face_data = None
                hand_data = None

                body_module = self._modules.get("body")
                face_module = self._modules.get("face")
                hand_module = self._modules.get("hand")
                body_active = bool(
                    body_module and body_module.enabled and body_module.initialized
                )
                hand_active = bool(
                    hand_module and hand_module.enabled and hand_module.initialized
                )
                parallel_body_hand = bool(
                    self._inference_executor and body_active and hand_active
                )

                if body_active:
                    if self.secondary_camera:
                        secondary_frame, secondary_count = self.secondary_camera.get_frame_with_count(copy=False)
                        body_module.set_secondary_frame(secondary_frame, secondary_count)

                if hand_module:
                    hand_module.set_deferred_streaming(parallel_body_hand)

                body_future = None
                hand_future = None
                if parallel_body_hand:
                    body_future = self._inference_executor.submit(body_module.process, frame)
                    hand_future = self._inference_executor.submit(hand_module.process, frame)
                elif body_active:
                    body_data = body_module.process(frame)

                if face_module and face_module.enabled and face_module.initialized:
                    face_data = face_module.process(frame)
                    if face_data is not None and face_module.last_result_fresh:
                        sent_count += 1

                if body_future is not None:
                    body_data = body_future.result()

                if body_active:
                    if body_data is not None and body_module.last_result_fresh:
                        sent_count += 1

                if hand_future is not None:
                    hand_data = hand_future.result()
                    if hand_module.last_detection_fresh:
                        hand_module.stream_detection(hand_data, body_data)
                elif hand_active:
                    hand_module.set_body_pose_data(body_data)
                    hand_data = hand_module.process(frame)

                if hand_active:
                    if hand_data is not None and hand_module.last_result_fresh:
                        sent_count += 1

                # ===== FPS 计算 =====
                processing_time = time.perf_counter() - start_time
                self._loop_timestamps.append(start_time)
                avg_fps = timestamp_rate(self._loop_timestamps)

                # ===== 配置热更新 =====
                now = time.time()
                if now - self._last_config_check_time >= self._config_check_interval:
                    self._last_config_check_time = now
                    self._check_config_hot_reload()

                if now - last_runtime_status_time >= 0.5:
                    self._write_runtime_status(
                        avg_fps,
                        stats["total_frames"],
                        running=True,
                    )
                    last_runtime_status_time = now

                # ===== 预览显示 =====
                if show_preview:
                    frame = self._draw_debug_overlay(frame, body_data, face_data, hand_data, avg_fps)
                    cv2.imshow(window_name, frame)

                # ===== 键盘控制 =====
                key = cv2.waitKey(1) & 0xFF if show_preview else -1
                if key == ord('q'):
                    print("[Loop] 用户按下 q 键，退出")
                    break
                elif key == ord('d'):
                    show_preview = not show_preview
                    if not show_preview:
                        cv2.destroyWindow(window_name)
                    print(f"[Loop] 预览显示: {'ON' if show_preview else 'OFF'}")
                elif key == ord('r'):
                    if body_module:
                        body_module.reset_filters()
                        print("[Loop] 身体滤波器已重置")
                elif key == ord('f'):
                    if face_module and face_data:
                        face_module.calibrate(face_data)
                        print("[Loop] 面部校准完成")
                elif key == ord('l'):
                    if face_module:
                        face_module.set_low_light_mode(not face_module._low_light_mode)
                        print(f"[Loop] 低光模式: {'ON' if face_module._low_light_mode else 'OFF'}")
                elif key == ord('h'):
                    if hand_module:
                        hand_module.reset_filters()
                        print("[Loop] 手部滤波器已重置")

                # ===== 日志输出 (每 5 秒) =====
                now = time.time()
                if now - last_log_time >= 5.0:
                    self._print_log(avg_fps, stats, body_data, face_data, hand_data)
                    last_log_time = now

                # ===== 状态输出 (每 10 秒) =====
                if now - last_status_time >= 10.0:
                    self._print_status(avg_fps, stats["total_frames"], sent_count)
                    last_status_time = now

        except KeyboardInterrupt:
            print("\n[Loop] 用户中断")
        except Exception as e:
            print(f"\n[Loop] FATAL: 主循环异常退出: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if show_preview:
                cv2.destroyWindow(window_name)

    def _write_runtime_status(self, fps: float, frame_count: int, running: bool):
        try:
            modules = {
                name: module.get_status()
                for name, module in self._modules.items()
            }
            hmd = None
            if self._hmd_provider:
                pose = self._hmd_provider.get_pose()
                hmd = {
                    "provider": self._hmd_provider.name,
                    "valid": pose.valid,
                    "tracking_origin": pose.tracking_origin,
                    "actual_height": pose.actual_height,
                    "height_reference": pose.height_reference,
                    "timestamp": pose.timestamp,
                }
            write_tracking_state({
                "running": bool(running),
                "pid": os.getpid(),
                "fps": float(fps),
                "frame_count": int(frame_count),
                "parallel_inference": self._inference_executor is not None,
                "modules": modules,
                "hmd": hmd,
            })
        except Exception as e:
            print(f"[RuntimeStatus] write failed: {e}")

    def _print_log(self, fps: float, stats: Dict, body_data, face_data, hand_data):
        """详细日志输出"""
        parts = [f"FPS={fps:.1f}", f"帧={stats['total_frames']}"]

        body_module = self._modules.get("body")
        face_module = self._modules.get("face")
        hand_module = self._modules.get("hand")

        if body_module and body_module.enabled:
            parts.append(f"身体={body_module.detect_rate:.0f}%")
            parts.append(f"({body_module.last_process_time_ms:.1f}ms)")
            if body_module.max_fps > 0:
                parts.append(f"cap={body_module.max_fps:g}fps")
            body_status = body_module.get_status()
            parts.append(
                "姿态="
                f"{body_status.get('pose_node_count', 0)}节点/"
                f"{body_status.get('tracker_node_count', 0)}点/"
                f"队列={'ON' if body_status.get('stream_queued') else 'OFF'}"
            )
            hmd_height_state = body_status.get("hmd_height_state") or {}
            if hmd_height_state.get("actual_height") is not None:
                ratio = hmd_height_state.get("height_ratio")
                ratio_text = f"/r={ratio:.2f}" if isinstance(ratio, (int, float)) else ""
                parts.append(
                    f"hmd={hmd_height_state.get('posture', 'unknown')}/"
                    f"{float(hmd_height_state.get('actual_height')):.2f}m{ratio_text}"
                )
            if body_status.get("dual_camera_enabled"):
                parts.append(
                    f"副摄={body_status.get('secondary_detect_rate', 0):.0f}%/"
                    f"cap={body_status.get('secondary_body_max_fps', 0):g}fps"
                )
        if face_module and face_module.enabled:
            parts.append(f"面部={face_module.detect_rate:.0f}%")
            parts.append(f"({face_module.last_process_time_ms:.1f}ms)")
            if face_module.max_fps > 0:
                parts.append(f"cap={face_module.max_fps:g}fps")
        if hand_module and hand_module.enabled:
            parts.append(f"手部={hand_module.detect_rate:.0f}%")
            parts.append(f"({hand_module.last_process_time_ms:.1f}ms)")
            if hand_module.max_fps > 0:
                parts.append(f"cap={hand_module.max_fps:g}fps")

        print(f"[Loop] {' | '.join(parts)}")

        # 警告
        if body_module and body_module.enabled and body_module.detect_rate < 10 and stats["total_frames"] > 60:
            print(f"[Loop] WARNING: 身体检测率过低 ({body_module.detect_rate:.0f}%)")
        if face_module and face_module.enabled and face_module.detect_rate < 5 and stats["total_frames"] > 60:
            print(f"[Loop] WARNING: 面部检测率过低 ({face_module.detect_rate:.0f}%)")

    def _print_status(self, fps: float, frame_count: int, sent_count: int):
        """定期状态输出"""
        parts = [f"FPS: {fps:.1f}", f"Frames: {frame_count}"]

        for name, module in self._modules.items():
            status = module.get_status()
            state = "ON" if status["enabled"] else "OFF"
            connected = "OK" if status.get("connected") else "OFF"
            if name == "body":
                details = (
                    f"{connected},nodes={status.get('pose_node_count', 0)},"
                    f"trackers={status.get('streamer_last_tracker_count', 0)},"
                    f"sent={status.get('streamer_send_count', 0)},"
                    f"err={status.get('streamer_error_count', 0)}"
                )
                parts.append(f"Body:{state}({details})")
            else:
                parts.append(f"{name.capitalize()}:{state}({connected})")

        parts.append(f"Sent: {sent_count}")
        print(f"[Status] {' | '.join(parts)}")

    def _draw_debug_overlay(self, frame, body_data, face_data, hand_data, fps: float):
        """绘制调试叠加层"""
        # FPS
        frame = ImageProcessor.add_fps_overlay(frame, fps, (10, 30))

        # 面部关键点
        face_module = self._modules.get("face")
        if face_data and face_module:
            frame = face_module.draw_landmarks(frame, face_data)

        # 身体关键点
        body_module = self._modules.get("body")
        if body_data and body_module:
            frame = body_module.draw_landmarks(frame, body_data)

        # 手部关键点
        hand_module = self._modules.get("hand")
        if hand_data and hand_module:
            frame = hand_module.draw_landmarks(frame, hand_data)

        # 状态信息
        status_items = {}
        body_module = self._modules.get("body")
        if body_module:
            status_items["Body"] = "ON" if body_module.enabled else "OFF"
        face_module = self._modules.get("face")
        if face_module:
            status_items["Face"] = "ON" if face_module.enabled else "OFF"
        hand_module = self._modules.get("hand")
        if hand_module:
            status_items["Hand"] = "ON" if hand_module.enabled else "OFF"

        frame = ImageProcessor.add_status_bar(frame, status_items, 55)
        return frame

    def _check_config_hot_reload(self):
        """检测配置文件变化并热更新模块开关"""
        try:
            config_path = self._config_path or "config/config.json"
            p = Path(config_path)
            if not p.exists():
                return
            mtime = p.stat().st_mtime
            if mtime <= self._config_mtime:
                return
            self._config_mtime = mtime

            # 重新加载配置
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                return
            try:
                new_config = json.loads(content)
            except json.JSONDecodeError:
                return

            self.config = new_config
            if self.camera:
                camera_cfg = new_config.get("camera", {})
                new_zoom = float(camera_cfg.get("zoom", 0.0) or 0.0)
                current_zoom = float(getattr(self.camera, "zoom", 0.0) or 0.0)
                if abs(new_zoom - current_zoom) > 1e-6:
                    self.camera.set_zoom(new_zoom)
            secondary_cfg = new_config.get("camera_secondary", {})
            if not new_config.get("processing", {}).get("dual_camera_body_enabled", False):
                secondary_cfg = {**secondary_cfg, "enabled": False}
            self._sync_secondary_camera(secondary_cfg)
            streaming_cfg = new_config.get("streaming", {})

            # 更新各模块
            for module_name in ["body", "face", "hand"]:
                module = self._modules.get(module_name)
                module_cfg = new_config.get(module_name, {})
                module_streaming_cfg = streaming_cfg.get(module_name, {})
                module_streaming_cfg.setdefault("host", streaming_cfg.get("host", "127.0.0.1"))
                module_streaming_cfg.setdefault("port", streaming_cfg.get("port", 9000))

                new_enabled = module_cfg.get("enabled", True)
                new_streaming = module_streaming_cfg.get("streaming_enabled", True)

                if module is None and new_enabled:
                    # 模块不存在但配置启用了: 创建并初始化
                    self._create_module(module_name, module_cfg, module_streaming_cfg, new_config)
                    print(f"  [HotReload] {module_name}: 创建并初始化")
                elif module is not None and not new_enabled:
                    # 模块存在但配置禁用了: 禁用模块
                    module.enabled = False
                    print(f"  [HotReload] {module_name}: 已禁用")
                elif module is not None and new_enabled:
                    # 模块存在且启用: 更新配置
                    if module_name == "body":
                        body_full_cfg = {
                            **module_cfg,
                            "yolo": new_config.get("yolo", {}),
                            "processing": new_config.get("processing", {}),
                            "camera": new_config.get("camera", {}),
                            "camera_secondary": new_config.get("camera_secondary", {}),
                            "mediapipe_pose": new_config.get("mediapipe_pose", {}),
                        }
                        module.update_config(body_full_cfg, module_streaming_cfg)
                    elif module_name == "hand":
                        hand_full_cfg = {
                            **module_cfg,
                            "steamvr": new_config.get("steamvr", {}),
                            "processing": new_config.get("processing", {}),
                        }
                        module.update_config(hand_full_cfg, module_streaming_cfg)
                    else:
                        face_full_cfg = {
                            **module_cfg,
                            "processing": new_config.get("processing", {}),
                        }
                        module.update_config(face_full_cfg, module_streaming_cfg)

                    if not module.initialized:
                        print(f"  [HotReload] {module_name}: enabled but not initialized, retrying initialization")
                        if module.initialize():
                            if self._hmd_provider and hasattr(module, 'set_hmd_provider'):
                                module.set_hmd_provider(self._hmd_provider)
                            print(f"  [HotReload] {module_name}: initialization restored")
                        else:
                            print(f"  [HotReload] {module_name}: initialization retry failed")

                    # 推流开关变更
                    if new_streaming != module.streaming_enabled:
                        module.streaming_enabled = new_streaming
                        action = "ON" if new_streaming else "OFF"
                        print(f"  [HotReload] {module_name} streaming: {action}")

        except Exception as e:
            print(f"  [HotReload] ERROR: {e}")

    def _create_module(self, name: str, module_cfg: Dict, streaming_cfg: Dict, full_config: Dict):
        """动态创建模块"""
        if name == "body":
            body_full_cfg = {
                **module_cfg,
                "yolo": full_config.get("yolo", {}),
                "processing": full_config.get("processing", {}),
                "camera": full_config.get("camera", {}),
                "camera_secondary": full_config.get("camera_secondary", {}),
                "mediapipe_pose": full_config.get("mediapipe_pose", {}),
            }
            module = BodyModule(config=body_full_cfg, streaming_config=streaming_cfg)
        elif name == "face":
            face_full_cfg = {
                **module_cfg,
                "processing": full_config.get("processing", {}),
            }
            module = FaceModule(config=face_full_cfg, streaming_config=streaming_cfg)
        elif name == "hand":
            hand_full_cfg = {
                **module_cfg,
                "steamvr": full_config.get("steamvr", {}),
                "processing": full_config.get("processing", {}),
            }
            module = HandModule(config=hand_full_cfg, streaming_config=streaming_cfg)
        else:
            return

        if module.initialize():
            self._modules[name] = module
            # 设置 HMD 提供者
            if self._hmd_provider and hasattr(module, 'set_hmd_provider'):
                module.set_hmd_provider(self._hmd_provider)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.running = False

        if self._inference_executor:
            self._inference_executor.shutdown(wait=True, cancel_futures=True)
            self._inference_executor = None

        # 清理所有模块
        for module in self._modules.values():
            module.cleanup()

        if self._hmd_provider:
            self._hmd_provider.stop()

        if self.camera:
            self.camera.stop()
        self._stop_secondary_camera()

        self._write_runtime_status(0.0, 0, running=False)

        print("System stopped")


def main():
    parser = argparse.ArgumentParser(description="YOLO26n VRC Motion Capture System")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--camera", type=int, default=None, help="摄像头 ID")
    parser.add_argument("--model", type=str, default=None, help="YOLO 模型路径")
    parser.add_argument("--model-type", type=str, default=None,
                       choices=["nano", "small", "medium", "large"],
                       help="模型精度等级 (覆盖 --model 参数)")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"],
                       help="推理设备")
    parser.add_argument("--host", type=str, default=None, help="VRC 目标地址")
    parser.add_argument("--port", type=int, default=None, help="VRC 目标端口")
    parser.add_argument("--protocol", type=str, default=None, choices=["osc", "vmc", "tcp", "steamvr"],
                       help="传输协议 (osc/vmc/tcp/steamvr)")
    parser.add_argument("--no-display", action="store_true", help="禁用预览窗口")
    parser.add_argument("--no-half", action="store_true", help="禁用 FP16 半精度")
    parser.add_argument("--precision", type=str, default=None,
                       choices=["fp32", "fp16", "int8"],
                       help="推理精度 (fp32/fp16/int8)")
    parser.add_argument("--no-face", action="store_true", help="禁用面部追踪")
    parser.add_argument("--no-hand", action="store_true", help="禁用手部追踪")

    args = parser.parse_args()

    system = VRCMocapSystem(config_path=args.config)

    # 命令行参数覆盖配置
    if args.camera is not None:
        system.config["camera"]["camera_id"] = args.camera
    if args.model is not None:
        system.config["yolo"]["model_path"] = args.model
    if args.model_type is not None:
        preset = MODEL_PRESETS.get(args.model_type)
        if preset:
            system.config["yolo"]["model_path"] = preset["model"]
            system.config["yolo"]["model_type"] = args.model_type
    if args.device is not None:
        system.config["yolo"]["device"] = args.device
    if args.host is not None:
        system.config["vrc"]["host"] = args.host
    if args.port is not None:
        system.config["vrc"]["port"] = args.port
    if args.protocol is not None:
        system.config["vrc"]["protocol"] = args.protocol
    if args.no_display:
        system.config["display"]["show_preview"] = False
    if args.no_half:
        system.config["yolo"]["half"] = False
        system.config["yolo"]["precision"] = "fp32"
    if args.precision is not None:
        system.config["yolo"]["precision"] = args.precision
        system.config["yolo"]["half"] = args.precision == "fp16"
    if args.no_face:
        system.config["face"]["enabled"] = False
    if args.no_hand:
        system.config["hand"]["enabled"] = False

    if not system.initialize():
        print("Initialization failed!")
        sys.exit(1)

    try:
        system.start()
    finally:
        system.stop()


if __name__ == "__main__":
    main()
