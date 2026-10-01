"""
面部表情追踪模块

独立封装: MediaPipe FaceLandmarker + VMC/OSC 推流。
可独立启停，关闭时不影响身体和手部模块。

推流地址:
- OSC: host:port (VRChat 面部参数 + 头部追踪)
- VMC: host:vmc_port (VMC blendshape)
"""

import os
import sys
from typing import Dict, Optional

from .base_module import BaseModule

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_calibration import write_face_state
from face_landmark import FaceLandmarkDetector
from streamers.face_vmc import FaceVMCStreamer


class FaceModule(BaseModule):
    """面部表情追踪模块

    推理: MediaPipe FaceLandmarker
    推流: OSC (VRChat 面部参数) + VMC (blendshape)
    独立 OSC 地址: host:port
    VMC 地址: host:vmc_port
    """

    def __init__(self, config: Dict, streaming_config: Dict):
        super().__init__(name="face", config=config, streaming_config=streaming_config)

        self._detector = None
        self._streamer = None
        self._low_light_mode = False
        self._last_state_write = 0.0

    def _do_initialize(self) -> bool:
        """初始化面部检测器和推流器"""
        # 面部检测器
        self._detector = FaceLandmarkDetector(
            min_detection_confidence=self._config.get("min_detection_confidence", 0.5),
            min_tracking_confidence=self._config.get("min_tracking_confidence", 0.5),
            smoothing=self._config.get("smoothing", "one_euro"),
            smoothing_factor=self._config.get("smoothing_factor", 0.3),
            one_euro_min_cutoff=self._config.get("one_euro_min_cutoff", 1.0),
            one_euro_beta=self._config.get("one_euro_beta", 0.007),
            min_face_presence_confidence=self._config.get("min_face_presence_confidence", 0.5),
            blendshape_smoothing=self._config.get("blendshape_smoothing", 0.4),
            head_position_scale=self._config.get("head_position_scale", 1.0),
            head_position_offset=self._config.get("head_position_offset", [0, 0, 0]),
            head_rotation_smoothing=self._config.get("head_rotation_smoothing", 0.3),
            vrcft_calibration=self._config.get("vrcft_calibration", {}),
            model_path=self._config.get("model_path"),
            mouth_shape_mapping_enabled=self._config.get("mouth_shape_mapping_enabled", True),
            mouth_shape_gain=self._config.get("mouth_shape_gain", 1.25),
            mouth_shape_deadzone=self._config.get("mouth_shape_deadzone", 0.03),
            eyelid_independent_enabled=self._config.get("eyelid_independent_enabled", True),
            eyelid_close_gain=self._config.get("eyelid_close_gain", 1.35),
            eyelid_close_deadzone=self._config.get("eyelid_close_deadzone", 0.04),
            eyelid_send_combined=self._config.get("eyelid_send_combined", False),
        )
        if not self._detector.load():
            print("[face] MediaPipe FaceLandmarker 加载失败")
            return False

        # 推流器
        host = self._streaming_config.get("host", "127.0.0.1")
        osc_port = self._streaming_config.get("port", 9000)
        vmc_port = self._streaming_config.get("vmc_port", 39539)

        self._streamer = FaceVMCStreamer(
            host=host,
            port=osc_port,
            vmc_port=vmc_port,
        )

        # 设置推流开关
        self._streamer.inference_enabled = self._enabled
        self._streamer.enabled = self._streaming_enabled

        # 连接推流
        if self._streaming_enabled:
            if self._streamer.connect():
                print(f"[face] 推流已连接 (OSC={host}:{osc_port}, VMC={host}:{vmc_port})")
            else:
                print(f"[face] 推流连接失败")

        return True

    def _do_process(self, frame) -> Optional[Dict]:
        """执行面部检测和推流"""
        face_data = self._detector.detect(frame)
        if face_data is None:
            return None
        if not self._detector.last_result_fresh:
            return face_data

        self._write_latest_vrcft_state(face_data)

        # 推流: body 关闭时 face streamer 接管头部追踪
        if self._streaming_enabled and self._streamer and self._streamer.connected:
            pose_data = {"face": face_data, "body_only": True}
            self._streamer.send(pose_data)

        return face_data

    def process(self, frame) -> Optional[Dict]:
        face_data = super().process(frame)
        if self._detector and not self._detector.last_result_fresh:
            self._last_result_fresh = False
        return face_data

    def _do_cleanup(self):
        """释放资源"""
        if self._streamer:
            self._streamer.disconnect()
            self._streamer = None
        if self._detector:
            self._detector.release()
        self._detector = None

    def calibrate(self, face_data: Dict):
        """校准面部"""
        if self._detector and face_data:
            self._detector.calibrate(face_data)

    def set_low_light_mode(self, enabled: bool):
        """设置低光模式"""
        self._low_light_mode = enabled
        if self._detector:
            self._detector.set_low_light_mode(enabled)

    def draw_landmarks(self, frame, face_data: Dict):
        """在帧上绘制面部关键点"""
        if self._detector and face_data:
            return self._detector.draw_face_landmarks(frame, face_data)
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
        if self._detector:
            self._detector.update_vrcft_calibration(self._config.get("vrcft_calibration", {}))
            self._detector.update_mouth_shape_mapping(
                enabled=self._config.get("mouth_shape_mapping_enabled", True),
                gain=self._config.get("mouth_shape_gain", 1.25),
                deadzone=self._config.get("mouth_shape_deadzone", 0.03),
            )
            self._detector.update_eyelid_mapping(
                enabled=self._config.get("eyelid_independent_enabled", True),
                gain=self._config.get("eyelid_close_gain", 1.35),
                deadzone=self._config.get("eyelid_close_deadzone", 0.04),
                send_combined=self._config.get("eyelid_send_combined", False),
            )
        if self._streamer:
            self._streamer.inference_enabled = self._enabled
            self._streamer.enabled = self._streaming_enabled

    def get_status(self) -> Dict:
        status = super().get_status()
        if self._streamer:
            status["connected"] = self._streamer.connected
            status["streamer_fps"] = self._streamer._fps
        return status

    def _write_latest_vrcft_state(self, face_data: Dict):
        vrcft_raw = face_data.get("vrcft_raw")
        if not vrcft_raw:
            return

        import time

        now = time.time()
        if now - self._last_state_write < 0.2:
            return
        self._last_state_write = now
        try:
            write_face_state(vrcft_raw)
        except Exception as e:
            if self._detect_count <= 3:
                print(f"[face] VRCFT state write failed: {e}")
