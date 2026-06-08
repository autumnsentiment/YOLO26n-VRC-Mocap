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
from pathlib import Path
from typing import Dict, Optional, List

# 将 Ultralytics 配置目录设到项目下，避免 AppData 权限问题
_ultra_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ultralytics")
os.makedirs(_ultra_cfg, exist_ok=True)
os.environ["ULTRALYTICS_SETTINGS"] = _ultra_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_capture import CameraCapture, ImageProcessor
from hmd_provider import HMDProviderFactory
from modules import BodyModule, FaceModule, HandModule


class VRCMocapSystem:
    def __init__(self, config_path: str = None):
        self._config_path = config_path
        self.config = self._load_config(config_path)
        self.running = False
        self.fps_history = []
        self._config_mtime = 0

        # 三个独立模块
        self._modules: Dict[str, object] = {}  # {"body": BodyModule, "face": FaceModule, "hand": HandModule}

        # HMD 提供者
        self._hmd_provider = None

        self.setup_signal_handlers()

    def _load_config(self, config_path: str = None) -> Dict:
        default_config = {
            "camera": {
                "camera_id": 0,
                "width": 1280,
                "height": 720,
                "fps": 30,
                "flip_horizontal": True
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
                    "vmt_port": 39570,
                    "streaming_enabled": True,
                },
            },
            "vrc": {"host": "127.0.0.1", "port": 9000, "protocol": "osc"},
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
            },
            "body": {"enabled": True},
            "face": {"enabled": True},
            "hand": {"enabled": True},
            "hmd": {"provider": "static"},
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
            default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.json")
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

    def _find_latest_trained_model(self) -> Optional[str]:
        """自动搜索最新训练的模型"""
        runs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "pose")
        if not os.path.exists(runs_dir):
            return None
        best_models = []
        for run_name in os.listdir(runs_dir):
            best_path = os.path.join(runs_dir, run_name, "weights", "best.pt")
            if os.path.exists(best_path):
                best_models.append((os.path.getmtime(best_path), best_path))
        if best_models:
            best_models.sort(reverse=True)
            return best_models[0][1]
        return None

    def initialize(self) -> bool:
        print("=" * 50)
        print("  YOLO26n VRC Mocap System")
        print("=" * 50)

        # 初始化摄像头
        cam_cfg = self.config["camera"]
        self.camera = CameraCapture(
            camera_id=cam_cfg["camera_id"],
            width=cam_cfg["width"],
            height=cam_cfg["height"],
            fps=cam_cfg["fps"],
        )
        if not self.camera.start():
            print("Error: Could not initialize camera")
            return False
        print(f"  Camera: {cam_cfg['width']}x{cam_cfg['height']} @ {cam_cfg['fps']}fps")

        # 自动查找模型
        yolo_cfg = self.config.get("yolo", {})
        model_path = yolo_cfg.get("model_path", "yolo26n-pose.pt")
        if not os.path.isabs(model_path):
            # 相对路径: 先检查 bin/ 目录，再检查项目根目录
            bin_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(bin_dir)
            candidates = [
                os.path.join(bin_dir, model_path),
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
                latest = self._find_latest_trained_model()
                if latest:
                    model_path = latest
                    print(f"  Using latest trained model: {model_path}")
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
        body_full_cfg = {**body_cfg, "yolo": self.config.get("yolo", {}), "processing": self.config.get("processing", {}), "camera": self.config.get("camera", {})}

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
        face_cfg = self.config.get("face", {})
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
        hand_streaming_cfg.setdefault("vmt_port", 39570)
        # 传入 steamvr 配置给 hand 模块 (hmd_to_neck_offset)
        hand_full_cfg = {**hand_cfg, "steamvr": self.config.get("steamvr", {})}

        hand_module = HandModule(config=hand_full_cfg, streaming_config=hand_streaming_cfg)
        if hand_module.enabled:
            if hand_module.initialize():
                self._modules["hand"] = hand_module
                print("  Hand module: initialized")
            else:
                print("  Hand module: initialization failed")
        else:
            print("  Hand module: disabled (config)")

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

        hmd_kwargs = {
            "osc_listen_port": hmd_cfg.get("osc_listen_port", 9001),
            "pico_host": hmd_cfg.get("pico_host", "127.0.0.1"),
            "pico_port": hmd_cfg.get("pico_port", 7890),
            "quest_host": hmd_cfg.get("quest_host", "127.0.0.1"),
            "quest_port": hmd_cfg.get("quest_port", 53342),
            "dashboard_host": hmd_cfg.get("dashboard_host", "127.0.0.1"),
            "dashboard_port": hmd_cfg.get("dashboard_port", 8765),
            "remote_port": hmd_cfg.get("remote_port", 4243),
        }

        hmd_provider = HMDProviderFactory.create(hmd_provider_type, **hmd_kwargs)
        if hmd_provider and hmd_provider.start():
            self._hmd_provider = hmd_provider
            # 设置到需要的模块
            for module in self._modules.values():
                if hasattr(module, 'set_hmd_provider'):
                    module.set_hmd_provider(hmd_provider)
            print(f"  HMD Provider: {hmd_provider.name}")
        else:
            print(f"  HMD Provider: {hmd_provider_type} start failed, using static fallback")

    def start(self):
        self.running = True

        print("\n" + "=" * 50)
        print("  Mocap running!")
        print("  [q] Quit  [d] Toggle display")
        print("  [r] Reset body filters  [f] Calibrate face")
        print("  [l] Toggle low-light mode  [h] Reset hand filters")
        print("=" * 50 + "\n")

        self._main_loop()

    def _main_loop(self):
        window_name = self.config["display"]["window_name"]
        show_preview = self.config["display"]["show_preview"]

        if show_preview:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        last_status_time = time.time()
        last_log_time = time.time()
        consecutive_none_frames = 0
        frame_count = 0
        sent_count = 0

        # 统计
        stats = {"total_frames": 0, "hmd_valid": 0, "hmd_invalid": 0}

        try:
            while self.running:
                frame = self.camera.get_frame()
                if frame is None:
                    consecutive_none_frames += 1
                    if consecutive_none_frames == 1:
                        print("[Loop] WARNING: 摄像头返回空帧")
                    elif consecutive_none_frames % 300 == 0:
                        print(f"[Loop] WARNING: 已连续 {consecutive_none_frames} 帧无画面，摄像头可能断开")
                    continue
                elif consecutive_none_frames > 0:
                    print(f"[Loop] 摄像头恢复 (之前连续 {consecutive_none_frames} 帧空帧)")
                    consecutive_none_frames = 0

                stats["total_frames"] += 1
                frame_count += 1
                start_time = time.perf_counter()

                # ===== 分发帧到各活跃模块 =====
                body_data = None
                face_data = None
                hand_data = None

                body_module = self._modules.get("body")
                if body_module and body_module.enabled and body_module.initialized:
                    body_data = body_module.process(frame)
                    if body_data is not None:
                        sent_count += 1

                face_module = self._modules.get("face")
                if face_module and face_module.enabled and face_module.initialized:
                    face_data = face_module.process(frame)
                    if face_data is not None:
                        sent_count += 1

                hand_module = self._modules.get("hand")
                if hand_module and hand_module.enabled and hand_module.initialized:
                    hand_data = hand_module.process(frame)
                    if hand_data is not None:
                        sent_count += 1

                # ===== HMD 状态检查 =====
                if self._hmd_provider:
                    hmd_pose = self._hmd_provider.get_pose()
                    if hmd_pose.valid:
                        stats["hmd_valid"] += 1
                    else:
                        stats["hmd_invalid"] += 1

                # ===== FPS 计算 =====
                processing_time = time.perf_counter() - start_time
                current_fps = 1.0 / processing_time if processing_time > 0 else 0
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 60:
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)

                # ===== 配置热更新 =====
                if frame_count % 30 == 0:
                    self._check_config_hot_reload()

                # ===== 预览显示 =====
                if show_preview:
                    frame = self._draw_debug_overlay(frame, body_data, face_data, hand_data, avg_fps)
                    cv2.imshow(window_name, frame)

                # ===== 键盘控制 =====
                key = cv2.waitKey(1) & 0xFF
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

    def _print_log(self, fps: float, stats: Dict, body_data, face_data, hand_data):
        """详细日志输出"""
        parts = [f"FPS={fps:.1f}", f"帧={stats['total_frames']}"]

        body_module = self._modules.get("body")
        face_module = self._modules.get("face")
        hand_module = self._modules.get("hand")

        if body_module and body_module.enabled:
            parts.append(f"身体={body_module.detect_rate:.0f}%")
            parts.append(f"({body_module.last_process_time_ms:.1f}ms)")
        if face_module and face_module.enabled:
            parts.append(f"面部={face_module.detect_rate:.0f}%")
            parts.append(f"({face_module.last_process_time_ms:.1f}ms)")
        if hand_module and hand_module.enabled:
            parts.append(f"手部={hand_module.detect_rate:.0f}%")
            parts.append(f"({hand_module.last_process_time_ms:.1f}ms)")

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
                        body_full_cfg = {**module_cfg, "yolo": new_config.get("yolo", {}), "processing": new_config.get("processing", {}), "camera": new_config.get("camera", {})}
                        module.update_config(body_full_cfg, module_streaming_cfg)
                    elif module_name == "hand":
                        hand_full_cfg = {**module_cfg, "steamvr": new_config.get("steamvr", {})}
                        module.update_config(hand_full_cfg, module_streaming_cfg)
                    else:
                        module.update_config(module_cfg, module_streaming_cfg)

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
            body_full_cfg = {**module_cfg, "yolo": full_config.get("yolo", {}), "processing": full_config.get("processing", {}), "camera": full_config.get("camera", {})}
            module = BodyModule(config=body_full_cfg, streaming_config=streaming_cfg)
        elif name == "face":
            module = FaceModule(config=module_cfg, streaming_config=streaming_cfg)
        elif name == "hand":
            hand_full_cfg = {**module_cfg, "steamvr": full_config.get("steamvr", {})}
            module = HandModule(config=hand_full_cfg, streaming_config=streaming_cfg)
        else:
            return

        if module.initialize():
            self._modules[name] = module
            # 设置 HMD 提供者
            if self._hmd_provider and hasattr(module, 'set_hmd_provider'):
                module.set_hmd_provider(self._hmd_provider)

    def stop(self):
        self.running = False

        # 清理所有模块
        for module in self._modules.values():
            module.cleanup()

        if self._hmd_provider:
            self._hmd_provider.stop()

        if self.camera:
            self.camera.stop()

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
        from train import ModelTrainer
        preset = ModelTrainer.MODEL_PRESETS.get(args.model_type)
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

    system.start()


if __name__ == "__main__":
    main()
