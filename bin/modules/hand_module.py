"""
手部追踪模块

独立封装: MediaPipe Hands + VMT/OSC 推流。
可独立启停，关闭时不影响身体和面部模块。

推流地址:
- OSC: host:port (VRChat 手指参数)
- VMT: host:vmt_port (OpenVR 驱动层手部位姿)
"""

import os
import sys
from typing import Dict, Optional

from .base_module import BaseModule

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand_tracker import HandTracker
from streamers.hand_vmt import HandVMTStreamer


class HandModule(BaseModule):
    """手部追踪模块

    推理: MediaPipe Hands
    推流: OSC (VRChat 手指参数) + VMT (OpenVR 驱动层)
    独立 OSC 地址: host:port
    VMT 地址: host:vmt_port
    """

    def __init__(self, config: Dict, streaming_config: Dict):
        super().__init__(name="hand", config=config, streaming_config=streaming_config)

        self._tracker = None
        self._streamer = None
        self._body_pose_data = None
        self._defer_streaming = False
        self._last_detection_fresh = False

    def _do_initialize(self) -> bool:
        """初始化手部追踪器和推流器"""
        # 手部追踪器
        self._tracker = HandTracker(
            max_num_hands=self._config.get("max_num_hands", 2),
            min_detection_confidence=self._config.get("min_detection_confidence", 0.6),
            min_tracking_confidence=self._config.get("min_tracking_confidence", 0.5),
            one_euro_min_cutoff=self._config.get("one_euro_min_cutoff", 1.0),
            one_euro_beta=self._config.get("one_euro_beta", 0.01),
            palm_back_classify=self._config.get("palm_back_classify", True),
            back_curl_compensation=self._config.get("back_curl_compensation", 1.25),
            back_roll_compensation=self._config.get("back_roll_compensation", 0.15),
            roi_assist_enabled=self._config.get("roi_assist_enabled", True),
            roi_scale=self._config.get("roi_scale", 2.6),
            roi_min_size=self._config.get("roi_min_size", 96),
            roi_min_confidence=self._config.get("roi_min_confidence", 0.05),
            roi_full_frame_interval=self._config.get("roi_full_frame_interval", 12),
            model_path=self._config.get("model_path"),
        )
        if not self._tracker.load():
            print("[hand] MediaPipe Hands 加载失败")
            return False

        # 推流器
        host = self._streaming_config.get("host", "127.0.0.1")
        osc_port = self._streaming_config.get("port", 9000)
        vmt_local_mode = self._streaming_config.get("vmt_local_mode", True) is not False
        vmt_host = (
            "127.0.0.1"
            if vmt_local_mode
            else self._streaming_config.get("vmt_host", host)
        )
        vmt_port = self._streaming_config.get("vmt_port", 39570)
        steamvr_cfg = self._config.get("steamvr", {})
        hmd_to_neck_offset = steamvr_cfg.get("hmd_to_neck_offset", [0, -0.2, 0.1])

        self._streamer = HandVMTStreamer(
            host=host,
            port=osc_port,
            vmt_host=vmt_host,
            vmt_port=vmt_port,
            hmd_to_neck_offset=hmd_to_neck_offset,
            desktop_osc_enabled=self._streaming_config.get("desktop_osc_enabled", True),
            vmt_enabled=self._streaming_config.get("vmt_enabled", True),
            vmt_local_mode=vmt_local_mode,
            vmt_skeleton_enabled=self._streaming_config.get(
                "vmt_skeleton_enabled", True
            ),
            vmt_input_enabled=self._streaming_config.get("vmt_input_enabled", True),
            vmt_auto_enable_skeleton=self._streaming_config.get(
                "vmt_auto_enable_skeleton", True
            ),
            vmt_register_on_steamvr_startup=self._streaming_config.get(
                "vmt_register_on_steamvr_startup", True
            ),
            vmt_keep_controllers_active=self._streaming_config.get(
                "vmt_keep_controllers_active", True
            ),
            vmt_hand_selection_priority=self._streaming_config.get(
                "vmt_hand_selection_priority", 100
            ),
            vmt_left_device_index=self._streaming_config.get(
                "vmt_left_device_index", 1
            ),
            vmt_right_device_index=self._streaming_config.get(
                "vmt_right_device_index", 2
            ),
            custom_parameters_enabled=self._streaming_config.get("custom_parameters_enabled", True),
            lost_reset_timeout=self._streaming_config.get("lost_reset_timeout", 0.35),
            desktop_left_gesture_parameters=self._streaming_config.get(
                "desktop_left_gesture_parameters"
            ),
            desktop_right_gesture_parameters=self._streaming_config.get(
                "desktop_right_gesture_parameters"
            ),
        )

        # 设置推流开关
        self._streamer.inference_enabled = self._enabled
        self._streamer.enabled = self._streaming_enabled

        # 连接推流
        if self._streaming_enabled:
            if self._streamer.connect():
                print(
                    f"[hand] 推流已连接 "
                    f"(desktop OSC={host}:{osc_port} "
                    f"{'ON' if self._streamer.desktop_osc_enabled else 'OFF'}, "
                    f"VMT={vmt_host}:{vmt_port} "
                    f"{'ON' if self._streamer.vmt_enabled else 'OFF'}, "
                    f"OpenVR skeleton="
                    f"{'ON' if self._streamer.vmt_skeleton_enabled else 'OFF'})"
                )
            else:
                print(f"[hand] 推流连接失败")

        return True

    def _do_process(
        self,
        frame,
        body_pose_data: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """执行手部检测和推流"""
        if body_pose_data is not None:
            self.set_body_pose_data(body_pose_data)
        roi_hints = (
            body_pose_data
            if isinstance(body_pose_data, dict)
            else self._body_pose_data
        )
        hand_data = self._tracker.detect(frame, roi_hints=roi_hints)
        self._last_detection_fresh = True

        # Empty frames are forwarded so the desktop gesture can return to neutral
        # after tracking loss instead of leaving the avatar hand stuck.
        if not self._defer_streaming:
            self.stream_detection(hand_data, self._body_pose_data)

        return hand_data if hand_data else None

    def process(self, frame) -> Optional[Dict]:
        self._last_detection_fresh = False
        return super().process(frame)

    @property
    def last_detection_fresh(self) -> bool:
        return self._last_detection_fresh

    def set_deferred_streaming(self, enabled: bool):
        self._defer_streaming = bool(enabled)

    def stream_detection(
        self,
        hand_data,
        body_pose_data: Optional[Dict] = None,
    ) -> bool:
        """Queue one completed hand detection with its matching body frame."""
        self.set_body_pose_data(body_pose_data)
        if not (
            self._streaming_enabled
            and self._streamer
            and self._streamer.connected
        ):
            return False

        pose_data = {
            "hands": hand_data or [],
            "body_only": True,
        }
        if isinstance(self._body_pose_data, dict):
            pose_3d = self._body_pose_data.get("pose_3d")
            rotations = self._body_pose_data.get("rotations")
            if isinstance(pose_3d, dict) and pose_3d:
                pose_data["pose_3d"] = pose_3d
            if isinstance(rotations, dict) and rotations:
                pose_data["rotations"] = rotations
        return bool(self._streamer.send(pose_data))

    def set_body_pose_data(self, pose_data: Optional[Dict]):
        """Provide the latest body pose for VMT wrist placement."""
        self._body_pose_data = pose_data if isinstance(pose_data, dict) else None

    def _do_cleanup(self):
        """释放资源"""
        if self._streamer:
            self._streamer.disconnect()
            self._streamer = None
        if self._tracker:
            self._tracker.release()
        self._tracker = None
        self._body_pose_data = None
        self._last_detection_fresh = False

    def set_hmd_provider(self, provider):
        """设置 HMD 位姿数据提供者"""
        if self._streamer:
            self._streamer.set_hmd_provider(provider)

    def reset_filters(self):
        """重置滤波器"""
        if self._tracker:
            self._tracker.reset_filters()

    def draw_landmarks(self, frame, hand_data: Dict):
        """在帧上绘制手部关键点"""
        if self._tracker and hand_data:
            return self._tracker.draw_hand_landmarks(frame, hand_data)
        return frame

    def on_enabled(self):
        """模块启用回调"""
        if self._streamer:
            self._streamer.inference_enabled = True
            self._streamer.enabled = self._streaming_enabled
            if self._streaming_enabled and not self._streamer.connected:
                self._streamer.connect()

    def on_disabled(self):
        """模块禁用回调"""
        if self._streamer:
            self._streamer.inference_enabled = False
            self._streamer.enabled = False
            if self._streamer.connected:
                self._streamer.disconnect()

    def update_config(self, config: Dict, streaming_config: Dict):
        """热更新配置"""
        super().update_config(config, streaming_config)
        if self._tracker and hasattr(self._tracker, "update_roi_config"):
            self._tracker.update_roi_config(config)
        if self._streamer:
            self._streamer.update_output_config(streaming_config)
            self._streamer.inference_enabled = self._enabled
            self._streamer.enabled = self._streaming_enabled
            output_enabled = (
                self._streamer.desktop_osc_enabled or self._streamer.vmt_enabled
            )
            if not output_enabled and self._streamer.connected:
                self._streamer.disconnect()
            elif self._streaming_enabled and output_enabled and not self._streamer.connected:
                self._streamer.connect()
            elif not self._streaming_enabled and self._streamer.connected:
                self._streamer.disconnect()

    def get_status(self) -> Dict:
        status = super().get_status()
        status["deferred_streaming"] = self._defer_streaming
        status["last_detection_fresh"] = self._last_detection_fresh
        status["body_pose_hint_available"] = isinstance(self._body_pose_data, dict)
        status["body_keypoint_hint_count"] = (
            len(self._body_pose_data.get("keypoints_2d", {}))
            if isinstance(self._body_pose_data, dict)
            else 0
        )
        if self._tracker:
            status["roi_assist_enabled"] = self._tracker.roi_assist_enabled
            status["hand_detection_mode"] = self._tracker.last_detection_mode
            status["roi_hint_count"] = self._tracker.last_roi_hint_count
            status["roi_hints"] = self._tracker.last_roi_hints
            status["roi_detection_count"] = self._tracker.roi_detection_count
            status["full_frame_detection_count"] = self._tracker.full_frame_detection_count
        if self._streamer:
            status["connected"] = self._streamer.connected
            status["streamer_fps"] = self._streamer._fps
            stream_status = self._streamer.get_status()
            status["streamer_send_count"] = stream_status.get("send_count", 0)
            status["streamer_error_count"] = stream_status.get("error_count", 0)
            status["streamer_drop_count"] = stream_status.get("drop_count", 0)
            status["streamer_queue_size"] = stream_status.get("queue_size", 0)
            for key in (
                "desktop_osc_enabled",
                "vmt_enabled",
                "vmt_local_mode",
                "vmt_skeleton_enabled",
                "vmt_input_enabled",
                "vmt_auto_enable_skeleton",
                "vmt_register_on_steamvr_startup",
                "vmt_driver_register_on_steamvr_startup",
                "vmt_keep_controllers_active",
                "vmt_hand_selection_priority",
                "vmt_driver_hand_selection_priority",
                "vmt_priority_restart_required",
                "vmt_startup_registration_restart_required",
                "vmt_restart_required",
                "vmt_controller_active",
                "vmt_device_indices",
                "custom_parameters_enabled",
                "vmt_initialized",
                "vmt_pre_registered",
                "vmt_last_ready_state",
                "vmt_reinit_count",
                "vmt_last_reinit_time",
                "vmt_registration_state",
                "vmt_device_presence_confirmed",
                "vmt_vrchat_consumption_confirmed",
                "desktop_gesture_parameters",
                "desktop_osc_host",
                "desktop_osc_port",
                "desktop_osc_message_count",
                "last_desktop_send_time",
                "last_desktop_gestures",
                "detected_hand_count",
                "last_pose_source",
                "vmt_pose_packets",
                "vmt_skeleton_packets",
                "vmt_input_packets",
                "last_vmt_open_values",
                "last_vmt_send_time",
                "vmt_protocol",
                "vmt_host",
                "vmt_port",
                "vmt_endpoint_ready",
                "vmt_driver_installed",
                "vmt_port_listening",
                "vmt_runtime_message",
            ):
                status[key] = stream_status[key]
        return status
