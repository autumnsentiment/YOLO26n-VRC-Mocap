"""
身体姿态追踪模块

独立封装: YOLO 检测 + 姿态估计 + OSC/SteamVR 推流。
可独立启停，关闭时不影响面部和手部模块。
"""

import os
import sys
from typing import Dict, Optional, List

from .base_module import BaseModule

# 将 Ultralytics 配置目录设到项目下
_ultra_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".ultralytics")
os.makedirs(_ultra_cfg, exist_ok=True)
os.environ["ULTRALYTICS_SETTINGS"] = _ultra_cfg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo_detector import YOLO26nDetector
from pose_estimator import PoseEstimator
from streamers.body_osc import BodyOSCStreamer
from streamers.body_steamvr import BodySteamVRStreamer
from streamers.osc_transport import OSCTransport
from streamers.coord import CoordinateTransformer


class BodyModule(BaseModule):
    """身体姿态追踪模块

    推理: YOLO26n 检测 + PoseEstimator 3D 重建
    推流: OSC (VRChat) 或 SteamVR (管道驱动)
    独立 OSC 地址: host:port
    """

    def __init__(self, config: Dict, streaming_config: Dict):
        super().__init__(name="body", config=config, streaming_config=streaming_config)

        self._detector = None
        self._pose_estimator = None
        self._streamer = None

        # 身体部位检测开关
        self._detect = {
            "torso": config.get("detect_torso", True),
            "left_arm": config.get("detect_left_arm", True),
            "right_arm": config.get("detect_right_arm", True),
            "left_leg": config.get("detect_left_leg", True),
            "right_leg": config.get("detect_right_leg", True),
        }

    def _do_initialize(self) -> bool:
        """初始化 YOLO 检测器、姿态估计器和推流器"""
        yolo_cfg = self._config.get("yolo", {})
        proc_cfg = self._config.get("processing", {})

        # YOLO 检测器
        model_path = yolo_cfg.get("model_path", "yolo26n-pose.pt")
        self._detector = YOLO26nDetector(
            model_path=model_path,
            device=yolo_cfg.get("device", "auto"),
            conf_threshold=yolo_cfg.get("conf_threshold", 0.5),
            half=yolo_cfg.get("half", True),
        )
        if not self._detector.load_model():
            print("[body] YOLO 模型加载失败")
            return False

        # 姿态估计器
        self._pose_estimator = PoseEstimator(
            smoothing=proc_cfg.get("smoothing", "one_euro"),
            smoothing_factor=proc_cfg.get("smoothing_factor", 0.3),
            one_euro_min_cutoff=proc_cfg.get("one_euro_min_cutoff", 1.0),
            one_euro_beta=proc_cfg.get("one_euro_beta", 0.007),
        )

        # 推流器
        host = self._streaming_config.get("host", "127.0.0.1")
        port = self._streaming_config.get("port", 9000)
        protocol = self._streaming_config.get("protocol", "osc")
        tracker_toggles = self._streaming_config.get("tracker_toggles", {})
        send_position = self._streaming_config.get("send_position", True)
        send_rotation = self._streaming_config.get("send_rotation", True)
        send_head_position = self._streaming_config.get("send_head_position", False)
        steamvr_cfg = self._streaming_config.get("steamvr", {})
        hmd_to_neck_offset = steamvr_cfg.get("hmd_to_neck_offset", [0, -0.2, 0.1])

        if protocol == "steamvr":
            self._streamer = BodySteamVRStreamer(
                host=host, port=port,
                smoothing=steamvr_cfg.get("smoothing", 50),
                additional_smoothing=steamvr_cfg.get("additional_smoothing", 0.5),
                camera_latency=steamvr_cfg.get("camera_latency", 0),
                hmd_to_neck_offset=hmd_to_neck_offset,
                trackers=steamvr_cfg.get("trackers", None),
                tracker_toggles=tracker_toggles,
            )
        else:
            self._streamer = BodyOSCStreamer(
                host=host, port=port,
                tracker_toggles=tracker_toggles,
                send_position=send_position,
                send_rotation=send_rotation,
                send_head_position=send_head_position,
                hmd_to_neck_offset=hmd_to_neck_offset,
            )

        # 设置推流开关
        self._streamer.inference_enabled = self._enabled
        self._streamer.enabled = self._streaming_enabled

        # 连接推流
        if self._streaming_enabled:
            if self._streamer.connect():
                print(f"[body] 推流已连接 ({protocol}://{host}:{port})")
            else:
                print(f"[body] 推流连接失败 ({protocol}://{host}:{port})")

        return True

    def _do_process(self, frame) -> Optional[Dict]:
        """执行身体姿态检测和推流"""
        # YOLO 检测
        detections = self._detector.detect(frame)
        if not detections:
            return None

        # 姿态估计
        flip_horizontal = self._config.get("camera", {}).get("flip_horizontal", True)
        pose_data = self._pose_estimator.process_detections(
            detections, frame.shape[1], frame.shape[0],
            flip_horizontal=flip_horizontal,
        )
        if pose_data is None:
            return None

        # 过滤身体部位
        pose_data = self._filter_body_parts(pose_data)

        # 推流
        if self._streaming_enabled and self._streamer and self._streamer.connected:
            self._streamer.send(pose_data)

        return pose_data

    def _do_cleanup(self):
        """释放资源"""
        if self._streamer:
            self._streamer.disconnect()
            self._streamer = None
        self._detector = None
        self._pose_estimator = None

    def _filter_body_parts(self, pose_data: Dict) -> Dict:
        """根据部位开关过滤数据"""
        if not pose_data or "pose_3d" not in pose_data:
            return pose_data

        pose_3d = pose_data.get("pose_3d", {})
        rotations = pose_data.get("rotations", {})

        # 部位到骨骼的映射
        part_to_bones = {
            "torso": ["Hips", "Chest", "Neck", "Spine"],
            "left_arm": ["LeftUpperArm", "LeftLowerArm", "LeftHand"],
            "right_arm": ["RightUpperArm", "RightLowerArm", "RightHand"],
            "left_leg": ["LeftUpperLeg", "LeftLowerLeg", "LeftFoot"],
            "right_leg": ["RightUpperLeg", "RightLowerLeg", "RightFoot"],
        }

        for part, enabled in self._detect.items():
            if not enabled:
                bones = part_to_bones.get(part, [])
                for bone in bones:
                    pose_3d.pop(bone, None)
                    rotations.pop(bone, None)

        return pose_data

    def set_hmd_provider(self, provider):
        """设置 HMD 位姿数据提供者"""
        if self._streamer:
            self._streamer.set_hmd_provider(provider)

    def set_tracker_toggles(self, toggles: Dict[str, bool]):
        """更新追踪器开关"""
        if self._streamer:
            self._streamer.set_tracker_toggles(toggles)

    def reset_filters(self):
        """重置滤波器"""
        if self._pose_estimator:
            self._pose_estimator.reset_filters()

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

        # 更新身体部位检测开关
        self._detect = {
            "torso": config.get("detect_torso", True),
            "left_arm": config.get("detect_left_arm", True),
            "right_arm": config.get("detect_right_arm", True),
            "left_leg": config.get("detect_left_leg", True),
            "right_leg": config.get("detect_right_leg", True),
        }

        # 更新推流器开关
        if self._streamer:
            self._streamer.inference_enabled = self._enabled
            self._streamer.enabled = self._streaming_enabled

    def get_status(self) -> Dict:
        status = super().get_status()
        if self._streamer:
            status["connected"] = self._streamer.connected
            status["streamer_fps"] = self._streamer._fps
        return status
