"""
推流管理器

统一管理各推流模块，提供:
- 独立开关控制各推理模块的推理与推流
- 统一的 connect/disconnect/send 接口
- 各模块独立进行数据包传输
- 手部和面部打包成 OSC 协议单独传输
- 身体姿态选择 OSC 则并入手部和面部一同发送数据包
- SteamVR 模式则注册追踪器模式调用 SteamVR 驱动

关键设计:
- 各推理模块独立开关，互不影响
- 关闭身体推理不影响手部和面面的推流
- 推理开关和推流开关分离 (推理关闭时推流自动关闭，推流关闭不影响推理)
"""

from typing import Dict, List, Optional

from .base import BaseStreamer
from .body_osc import BodyOSCStreamer
from .body_steamvr import BodySteamVRStreamer
from .hand_vmt import HandVMTStreamer
from .face_vmc import FaceVMCStreamer
from .coord import CoordinateTransformer


class StreamManager:
    """推流管理器

    管理身体、手部、面部三个独立的推流模块。
    每个模块拥有独立的推理开关和推流开关。
    """

    def __init__(self, config: Dict):
        self._config = config
        self._streamers: Dict[str, BaseStreamer] = {}
        self._hmd_provider = None

        # 初始化各推流模块
        self._init_streamers()

    def _init_streamers(self):
        """根据配置初始化推流模块

        支持两种配置格式:
        1. 新格式: streaming 节 (推荐)
        2. 旧格式: vrc + steamvr 节 (兼容)
        """
        # 读取通用配置
        host = self._config.get("host", "127.0.0.1")
        port = self._config.get("port", 9000)

        # 读取身体推流配置
        body_cfg = self._config.get("body", {})
        body_protocol = body_cfg.get("protocol", "osc")
        tracker_toggles = body_cfg.get("tracker_toggles", {})
        send_position = body_cfg.get("send_position", True)
        send_rotation = body_cfg.get("send_rotation", True)
        send_head_position = body_cfg.get("send_head_position", False)
        decouple_from_hmd = body_cfg.get("decouple_from_hmd", False)
        hmd_height_scaling = body_cfg.get("hmd_height_scaling", {})

        # 读取 SteamVR 子配置
        steamvr_cfg = body_cfg.get("steamvr", {})
        hmd_to_neck_offset = steamvr_cfg.get("hmd_to_neck_offset", [0, -0.2, 0.1])

        # ===== 身体推流模块 =====
        if body_protocol == "steamvr":
            body_streamer = BodySteamVRStreamer(
                host=host,
                port=port,
                smoothing=steamvr_cfg.get("smoothing", 50),
                additional_smoothing=steamvr_cfg.get("additional_smoothing", 0.5),
                camera_latency=steamvr_cfg.get("camera_latency", 0),
                hmd_to_neck_offset=hmd_to_neck_offset,
                trackers=steamvr_cfg.get("trackers", None),
                tracker_toggles=tracker_toggles,
                decouple_from_hmd=decouple_from_hmd,
                hmd_height_scaling=hmd_height_scaling,
            )
        else:
            body_streamer = BodyOSCStreamer(
                host=host,
                port=port,
                tracker_toggles=tracker_toggles,
                send_position=send_position,
                send_rotation=send_rotation,
                send_head_position=send_head_position,
                decouple_from_hmd=decouple_from_hmd,
                hmd_to_neck_offset=hmd_to_neck_offset,
                hmd_height_scaling=hmd_height_scaling,
            )

        self._streamers["body"] = body_streamer

        # ===== 手部推流模块 =====
        hand_cfg = self._config.get("hand", {})
        hand_streamer = HandVMTStreamer(
            host=host,
            port=port,
            vmt_host=hand_cfg.get("vmt_host", host),
            vmt_port=hand_cfg.get("vmt_port", 39570),
            hmd_to_neck_offset=hmd_to_neck_offset,
            desktop_osc_enabled=hand_cfg.get("desktop_osc_enabled", True),
            vmt_enabled=hand_cfg.get("vmt_enabled", True),
            vmt_local_mode=hand_cfg.get("vmt_local_mode", True),
            vmt_skeleton_enabled=hand_cfg.get("vmt_skeleton_enabled", True),
            vmt_input_enabled=hand_cfg.get("vmt_input_enabled", True),
            vmt_auto_enable_skeleton=hand_cfg.get(
                "vmt_auto_enable_skeleton", True
            ),
            vmt_left_device_index=hand_cfg.get("vmt_left_device_index", 1),
            vmt_right_device_index=hand_cfg.get("vmt_right_device_index", 2),
            custom_parameters_enabled=hand_cfg.get("custom_parameters_enabled", True),
            lost_reset_timeout=hand_cfg.get("lost_reset_timeout", 0.35),
            desktop_left_gesture_parameters=hand_cfg.get("desktop_left_gesture_parameters"),
            desktop_right_gesture_parameters=hand_cfg.get("desktop_right_gesture_parameters"),
        )
        self._streamers["hand"] = hand_streamer

        # ===== 面部推流模块 =====
        face_cfg = self._config.get("face", {})
        face_streamer = FaceVMCStreamer(
            host=host,
            port=port,
            vmc_port=face_cfg.get("vmc_port", 39539),
        )
        self._streamers["face"] = face_streamer

        # 设置推理和推流开关 (从配置读取)
        self._streamers["body"].inference_enabled = body_cfg.get("inference_enabled", True)
        self._streamers["body"].enabled = body_cfg.get("streaming_enabled", True)
        self._streamers["hand"].inference_enabled = hand_cfg.get("inference_enabled", True)
        self._streamers["hand"].enabled = hand_cfg.get("streaming_enabled", True)
        self._streamers["face"].inference_enabled = face_cfg.get("inference_enabled", True)
        self._streamers["face"].enabled = face_cfg.get("streaming_enabled", True)

    def set_hmd_provider(self, provider):
        """设置 HMD 位姿数据提供者 (所有模块共享)"""
        self._hmd_provider = provider
        for streamer in self._streamers.values():
            streamer.set_hmd_provider(provider)

    def connect(self) -> bool:
        """连接所有推流模块"""
        results = []
        for name, streamer in self._streamers.items():
            if streamer.enabled:
                result = streamer.connect()
                results.append(result)
                if not result:
                    print(f"[StreamManager] WARNING: {name} streamer connect failed")
                else:
                    print(f"[StreamManager] {name} streamer connected (inference={'ON' if streamer.inference_enabled else 'OFF'})")
            else:
                print(f"[StreamManager] {name} streamer disabled (inference={'ON' if streamer.inference_enabled else 'OFF'}), skipping connect")
        return any(results)

    def disconnect(self):
        """断开所有推流模块"""
        for streamer in self._streamers.values():
            streamer.disconnect()

    def send(self, pose_data: Dict):
        """分发数据到各推流模块

        各模块独立判断:
        - inference_enabled: 是否接收推理数据
        - enabled: 是否真正发送数据

        关键: 即使身体推理关闭，手部和面部仍可独立推流。
        """
        for name, streamer in self._streamers.items():
            try:
                result = streamer.send(pose_data)
                # 调试: 当 body 推理关闭但 face/hand 推理开启时，确认数据分发正确
                if not result and streamer.inference_enabled:
                    # inference_enabled=True 但 send 返回 False，说明队列异常
                    if streamer._error_count <= 3:
                        print(f"[StreamManager] WARNING: {name} send() returned False but inference_enabled=True")
            except Exception as e:
                if streamer._error_count <= 3:
                    print(f"[StreamManager] {name} send error: {e}")

    # ========== 开关控制 ==========

    def set_inference_enabled(self, module: str, enabled: bool):
        """设置推理开关

        推理关闭时，推流自动关闭并断开连接 (无推理数据则推流无意义)。
        推理开启时，推流恢复为配置中的设定值并重新连接。

        Args:
            module: "body" / "hand" / "face"
            enabled: 是否启用推理
        """
        streamer = self._streamers.get(module)
        if streamer:
            streamer.inference_enabled = enabled
            if not enabled:
                # 推理关闭时自动关闭推流并断开连接
                streamer.enabled = False
                if streamer.connected:
                    streamer.disconnect()
                    print(f"[StreamManager] {module} streamer disconnected (inference OFF)")
            else:
                # 推理开启时，从配置恢复推流开关
                module_cfg = self._config.get(module, {})
                streaming_enabled = module_cfg.get("streaming_enabled", True)
                # 如果配置中 streaming_enabled 也为 False，可能是之前推理关闭时被级联设置的
                # 此时应该恢复为 True (推理开启时推流默认也开启)
                if not streaming_enabled:
                    print(f"[StreamManager] {module} streaming_enabled=False in config, forcing True (inference ON)")
                    streaming_enabled = True
                streamer.enabled = streaming_enabled
                if streamer.enabled and not streamer.connected:
                    result = streamer.connect()
                    if result:
                        print(f"[StreamManager] {module} streamer reconnected (inference ON)")
                    else:
                        print(f"[StreamManager] WARNING: {module} streamer reconnect failed")

    def set_streaming_enabled(self, module: str, enabled: bool):
        """设置推流开关

        推流开关独立于推理开关，但推理关闭时推流无效。

        Args:
            module: "body" / "hand" / "face"
            enabled: 是否启用推流
        """
        streamer = self._streamers.get(module)
        if streamer:
            streamer.enabled = enabled

    def set_tracker_toggles(self, toggles: Dict[str, bool]):
        """设置追踪器开关 (仅影响身体推流)"""
        body_streamer = self._streamers.get("body")
        if body_streamer:
            body_streamer.set_tracker_toggles(toggles)

    def update_config(self, config: Dict):
        """更新内部配置缓存 (热更新时调用)"""
        self._config = config

    def get_inference_enabled(self, module: str) -> bool:
        """获取推理开关状态"""
        streamer = self._streamers.get(module)
        return streamer.inference_enabled if streamer else False

    def get_streaming_enabled(self, module: str) -> bool:
        """获取推流开关状态"""
        streamer = self._streamers.get(module)
        return streamer.enabled if streamer else False

    # ========== 状态查询 ==========

    def get_status(self) -> Dict:
        """获取所有推流模块状态"""
        status = {}
        for name, streamer in self._streamers.items():
            status[name] = streamer.get_status()
        return status

    def is_connected(self) -> bool:
        """是否有任何模块已连接"""
        return any(s.connected for s in self._streamers.values())

    @property
    def body_streamer(self) -> Optional[BaseStreamer]:
        return self._streamers.get("body")

    @property
    def hand_streamer(self) -> Optional[BaseStreamer]:
        return self._streamers.get("hand")

    @property
    def face_streamer(self) -> Optional[BaseStreamer]:
        return self._streamers.get("face")
