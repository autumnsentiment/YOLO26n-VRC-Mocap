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
        )
        if not self._tracker.load():
            print("[hand] MediaPipe Hands 加载失败")
            return False

        # 推流器
        host = self._streaming_config.get("host", "127.0.0.1")
        osc_port = self._streaming_config.get("port", 9000)
        vmt_port = self._streaming_config.get("vmt_port", 39570)
        steamvr_cfg = self._config.get("steamvr", {})
        hmd_to_neck_offset = steamvr_cfg.get("hmd_to_neck_offset", [0, -0.2, 0.1])

        self._streamer = HandVMTStreamer(
            host=host,
            port=osc_port,
            vmt_port=vmt_port,
            hmd_to_neck_offset=hmd_to_neck_offset,
        )

        # 设置推流开关
        self._streamer.inference_enabled = self._enabled
        self._streamer.enabled = self._streaming_enabled

        # 连接推流
        if self._streaming_enabled:
            if self._streamer.connect():
                print(f"[hand] 推流已连接 (OSC={host}:{osc_port}, VMT={host}:{vmt_port})")
            else:
                print(f"[hand] 推流连接失败")

        return True

    def _do_process(self, frame) -> Optional[Dict]:
        """执行手部检测和推流"""
        hand_data = self._tracker.detect(frame)
        if hand_data is None or len(hand_data) == 0:
            return None

        # 推流
        if self._streaming_enabled and self._streamer and self._streamer.connected:
            pose_data = {"hands": hand_data, "body_only": True}
            self._streamer.send(pose_data)

        return hand_data

    def _do_cleanup(self):
        """释放资源"""
        if self._streamer:
            self._streamer.disconnect()
            self._streamer = None
        self._tracker = None

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
        if self._streamer:
            self._streamer.inference_enabled = self._enabled
            self._streamer.enabled = self._streaming_enabled

    def get_status(self) -> Dict:
        status = super().get_status()
        if self._streamer:
            status["connected"] = self._streamer.connected
            status["streamer_fps"] = self._streamer._fps
        return status
