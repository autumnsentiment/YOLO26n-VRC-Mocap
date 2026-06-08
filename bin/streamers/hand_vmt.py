"""
手部追踪 VMT 推流模块

通过 VMT (Virtual Motion Tracker) OpenVR 驱动层发送手部追踪数据。
VMT 作为 SteamVR 驱动运行，直接在 OpenVR 层面提供手部追踪器。

VMT 协议:
- /VMT/SetRoomMatrix: 设置房间矩阵 (初始化)
- /VMT/Joint/Driver: 发送手部位置+四元数
  格式: [hand_type, enable, timeoffset, x, y, z, qx, qy, qz, qw, "HMD"]
  hand_type: 1=左手, 2=右手
  enable: 5=左手, 6=右手
- /VMT/Skeleton/Scalar: 发送手指弯曲度
  格式: [hand_type, finger_index, curl, 0, 0]
  finger_index: 1-5 (thumb/index/middle/ring/pinky)
- /VMT/Skeleton/Apply: 应用手指姿态
  格式: [hand_type, 0.0]

同时通过 OSC 发送 VRChat 手指参数 (VRChat 原生手指追踪):
- /avatar/parameters/LeftThumbCurl 等
- /avatar/parameters/RightThumbCurl 等
"""

from typing import Dict, List, Optional

from .base import BaseStreamer
from .osc_transport import OSCTransport
from .coord import CoordinateTransformer


class HandVMTStreamer(BaseStreamer):
    """手部追踪 VMT 推流

    通过 VMT 协议发送手部位置/旋转到 OpenVR 驱动层，
    同时通过 OSC 发送手指弯曲度到 VRChat。

    独立于身体和面部推流，可单独开关。
    """

    FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
    FINGER_CAP = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        vmt_port: int = 39570,
        hmd_to_neck_offset: List[float] = None,
    ):
        super().__init__(name="hand_vmt", host=host, port=port)
        self._osc = OSCTransport(host, port)
        self._vmt_osc = OSCTransport(host, vmt_port)
        self._coord = CoordinateTransformer(hmd_to_neck_offset)
        self._vmt_initialized = False

    def set_hmd_provider(self, provider):
        self._hmd_provider = provider
        self._coord.set_hmd_provider(provider)

    def _do_connect(self) -> bool:
        if not self._osc.connect():
            return False
        if not self._vmt_osc.connect():
            print("[hand_vmt] WARNING: VMT socket init failed, VMT hand tracking unavailable")
        else:
            self._vmt_init()
        return True

    def _do_disconnect(self):
        self._osc.disconnect()
        self._vmt_osc.disconnect()
        self._vmt_initialized = False

    def _vmt_init(self):
        """初始化 VMT 房间矩阵"""
        try:
            message = OSCTransport.build_message(
                "/VMT/SetRoomMatrix",
                1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -0.76, 0.0, 0.0, 1.0, 1.0
            )
            self._vmt_osc.send(message)
            self._vmt_initialized = True
        except Exception as e:
            print(f"[hand_vmt] VMT init error: {e}")

    def _do_send(self, pose_data: Dict):
        """发送手部追踪数据

        处理流程:
        1. 通过 VMT 协议发送手部位置+旋转 (OpenVR 驱动层)
        2. 通过 VMT 协议发送手指弯曲度
        3. 通过 OSC 发送 VRChat 手指参数

        支持两种模式:
        - body ON: 使用 pose_3d/rotations (来自 YOLO+pose_estimator) 进行坐标变换
        - body OFF: 使用 hand_data 中的 landmarks_norm 估算手部 3D 位置
        """
        hand_data = pose_data.get("hands")
        if hand_data is None:
            return

        # 将列表格式转为字典格式
        if isinstance(hand_data, list):
            hand_dict = {}
            for h in hand_data:
                label = h.get("label", "unknown")
                hand_dict[label] = h
            hand_data = hand_dict

        # ===== 坐标变换 =====
        pose_3d = pose_data.get("pose_3d", {})
        rotations = pose_data.get("rotations", {})

        if pose_3d:
            # body ON: 使用 pose_estimator 输出的 3D 坐标
            unity_pose = self._coord.camera_to_unity_positions(pose_3d)
            unity_rotations = self._coord.camera_to_unity_rotations(rotations)
            osc_pose = self._coord.compute_osc_offsets(unity_pose)
        else:
            # body OFF: 从 hand_data 的 landmarks_norm 估算手部 3D 位置
            unity_pose, unity_rotations, osc_pose = self._estimate_hand_positions_from_landmarks(hand_data)

        # ===== VMT 手部位置+旋转 =====
        if self._vmt_osc.socket and self._vmt_initialized:
            self._send_vmt_hands(osc_pose, unity_rotations, hand_data)

        # ===== OSC 手指参数 =====
        self._send_osc_hand_params(hand_data)

    def _estimate_hand_positions_from_landmarks(
        self, hand_data: Dict
    ) -> tuple:
        """body OFF 时，从 MediaPipe 手部关键点估算 3D 位置

        使用手腕关键点的归一化坐标 (x, y, z) 估算手部在摄像头坐标系中的位置。
        MediaPipe z 值是相对于手腕的深度，需要转换为绝对深度估算。

        坐标系变换:
        1. 归一化坐标 -> 摄像头坐标系 (X左, Y下, Z前, 米)
        2. 摄像头坐标系 -> Unity 坐标系 (X右, Y上, Z前) 通过 camera_to_unity_positions
        3. Unity 坐标系 -> OSC 偏移 通过 compute_osc_offsets

        Returns:
            (unity_pose, unity_rotations, osc_pose) 三元组
        """
        import numpy as np

        # 摄像头坐标系位置 (X左, Y下, Z前)
        cam_pose = {}
        cam_rotations = {}

        # 估算的摄像头参数 (用于归一化坐标 -> 3D 坐标)
        ESTIMATED_DEPTH = 1.5  # 米
        FOV_SCALE = 0.6  # 归一化坐标到米的缩放因子

        for hand_label, is_left in [("left", True), ("right", False)]:
            data = hand_data.get(hand_label)
            if data is None:
                continue

            landmarks_norm = data.get("landmarks_norm")
            if landmarks_norm is None or len(landmarks_norm) == 0:
                continue

            # 手腕关键点 (index=0)
            wrist = landmarks_norm[0]  # [x, y, z] 归一化坐标

            # 归一化坐标 -> 摄像头坐标系 (米)
            # 摄像头: X左, Y下, Z前
            cam_x = (0.5 - wrist[0]) * FOV_SCALE  # 归一化 x: 0=左, 1=右 -> 摄像头 X左为正
            cam_y = (0.5 - wrist[1]) * FOV_SCALE  # 归一化 y: 0=上, 1=下 -> 摄像头 Y下为正
            cam_z = ESTIMATED_DEPTH + wrist[2] * FOV_SCALE  # z 是相对深度

            bone_name = "LeftHand" if is_left else "RightHand"
            cam_pose[bone_name] = [cam_x, cam_y, cam_z]

            # 手部旋转: HandTracker 返回 [yaw, pitch, roll] (度)
            # camera_to_unity_rotations 期望 xyz 弧度 = [roll, pitch, yaw] 弧度
            rotation = data.get("rotation", [0, 0, 0])  # [yaw, pitch, roll] 度
            # 转换为 xyz 弧度: [roll, pitch, yaw] -> [x=roll, y=pitch, z=yaw]
            import math
            cam_rotations[bone_name] = [
                math.radians(rotation[2]),  # x = roll
                math.radians(rotation[1]),  # y = pitch
                math.radians(rotation[0]),  # z = yaw
            ]

        # 颈部位置 (摄像头坐标系)
        if "Neck" not in cam_pose:
            cam_pose["Neck"] = [0, -1.5, 1.5]  # 颈部在摄像头前方1.5m, 上方1.5m

        # ===== 坐标系变换 =====
        # 摄像头 -> Unity
        unity_pose = self._coord.camera_to_unity_positions(cam_pose)
        unity_rotations = self._coord.camera_to_unity_rotations(cam_rotations)

        # Unity -> OSC 偏移
        osc_pose = self._coord.compute_osc_offsets(unity_pose)

        return unity_pose, unity_rotations, osc_pose

    def _send_vmt_hands(
        self,
        osc_pose: Dict,
        unity_rotations: Dict,
        hand_data: Dict,
    ):
        """通过 VMT 协议发送手部位置+旋转"""
        from scipy.spatial.transform import Rotation as Rot

        for hand_label, is_left in [("left", True), ("right", False)]:
            data = hand_data.get(hand_label)
            if data is None:
                continue

            # 获取手部位置
            hand_bone = "LeftHand" if is_left else "RightHand"
            pos = osc_pose.get(hand_bone, [0, 0, 0])

            # 获取手部旋转
            euler_deg = unity_rotations.get(hand_bone, [0, 0, 0])
            try:
                rot = Rot.from_euler("zxy", euler_deg, degrees=True)
                qx, qy, qz, qw = rot.as_quat()
            except Exception:
                qx, qy, qz, qw = 0, 0, 0, 1

            # 发送 VMT Joint/Driver 消息
            hand_type = 1 if is_left else 2
            enable = 5 if is_left else 6
            message = OSCTransport.build_message(
                "/VMT/Joint/Driver",
                hand_type,
                enable,
                0.0,
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(qx), float(qy), float(qz), float(qw),
                "HMD"
            )
            self._vmt_osc.send(message)

            # 发送手指弯曲度
            fingers = data.get("fingers", {})
            curls = data.get("curls", {})

            for i, finger in enumerate(self.FINGER_NAMES):
                curl = curls.get(finger, 0.0)
                if not curl and fingers:
                    is_extended = fingers.get(finger, False)
                    curl = 0.0 if is_extended else 0.8
                curl = max(0.0, min(1.0, float(curl)))

                finger_msg = OSCTransport.build_message(
                    "/VMT/Skeleton/Scalar",
                    hand_type,
                    int(i + 1),
                    curl,
                    0,
                    0
                )
                self._vmt_osc.send(finger_msg)

            # 应用手指姿态
            apply_msg = OSCTransport.build_message(
                "/VMT/Skeleton/Apply",
                hand_type,
                0.0
            )
            self._vmt_osc.send(apply_msg)

    def _send_osc_hand_params(self, hand_data: Dict):
        """通过 OSC 发送 VRChat 手指参数"""
        for hand_label in ["left", "right"]:
            data = hand_data.get(hand_label)
            if data is None:
                continue

            fingers = data.get("fingers", {})
            curls = data.get("curls", {})
            prefix = "Left" if hand_label == "left" else "Right"

            messages = []

            # 发送每根手指的弯曲度
            for i, finger in enumerate(self.FINGER_NAMES):
                curl = curls.get(finger, 0.0)
                if not curl and fingers:
                    is_extended = fingers.get(finger, False)
                    curl = 0.0 if is_extended else 0.8
                curl = max(0.0, min(1.0, float(curl)))
                address = f"/avatar/parameters/{prefix}{self.FINGER_CAP[i]}Curl"
                messages.append(OSCTransport.build_message(address, curl))

            # 发送手指张开度
            spread = data.get("spread", 0.0)
            spread = max(0.0, min(1.0, float(spread)))
            messages.append(OSCTransport.build_message(
                f"/avatar/parameters/{prefix}HandSpread", spread
            ))

            # 手掌/手背朝向参数
            palm_side = data.get("palm_side", "palm")
            palm_conf = data.get("palm_confidence", 0.5)
            palm_value = 0.0 if palm_side == "palm" else 1.0
            palm_value = 0.5 + (palm_value - 0.5) * palm_conf
            messages.append(OSCTransport.build_message(
                f"/avatar/parameters/{prefix}HandPalmBack", float(palm_value)
            ))

            self._osc.send_batch(messages)
