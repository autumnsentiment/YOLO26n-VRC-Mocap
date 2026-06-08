"""
面部表情 VMC 推流模块

通过 VMC (Virtual Motion Capture) 协议发送面部表情数据。
VMC 协议基于 OSC，专门用于面部表情传输。

VMC 面部协议:
- /VMC/Ext/Blend/Val: 发送 blendshape 值
  格式: [name, value]
- /VMC/Ext/Blend/Apply: 应用所有 blendshape
- /VMC/Ext/OK: 可用信号

同时通过 OSC 发送 VRChat 面部参数 (VRCFT Unified Expressions v2):
- /avatar/parameters/XXX: VRCFT 标准参数

两种协议同时发送，确保兼容性:
- VMC 协议: 供 VMC 接收端使用 (如 VSeeFace, Luppet 等)
- VRChat OSC: 供 VRChat 直接使用
"""

from typing import Dict, List, Optional

from .base import BaseStreamer
from .osc_transport import OSCTransport
from .coord import CoordinateTransformer


class FaceVMCStreamer(BaseStreamer):
    """面部表情 VMC 推流

    通过 VMC 协议和 VRChat OSC 同时发送面部表情数据。
    独立于身体和手部推流，可单独开关。
    """

    # VRCFT v2 参数 -> VMC blendshape 名称映射
    VRCFT_TO_VMC_BLENDSHAPE = {
        "v2/MouthOpen": "mouth_open",
        "v2/MouthSmile": "mouth_smile",
        "v2/EyeOpenLeft": "eye_open_l",
        "v2/EyeOpenRight": "eye_open_r",
        "v2/BrowInnerUpLeft": "brow_inner_up_l",
        "v2/BrowInnerUpRight": "brow_inner_up_r",
        "v2/JawOpen": "jaw_open",
        "v2/MouthFunnel": "mouth_funnel",
        "v2/MouthPucker": "mouth_pucker",
        "v2/MouthLeft": "mouth_left",
        "v2/MouthRight": "mouth_right",
        "v2/MouthLowerDownLeft": "mouth_lower_down_l",
        "v2/MouthLowerDownRight": "mouth_lower_down_r",
        "v2/MouthUpperUpLeft": "mouth_upper_up_l",
        "v2/MouthUpperUpRight": "mouth_upper_up_r",
        "v2/EyeSquintLeft": "eye_squint_l",
        "v2/EyeSquintRight": "eye_squint_r",
        "v2/EyeWideLeft": "eye_wide_l",
        "v2/EyeWideRight": "eye_wide_r",
        "v2/BrowDownLeft": "brow_down_l",
        "v2/BrowDownRight": "brow_down_r",
        "v2/BrowOuterUpLeft": "brow_outer_up_l",
        "v2/BrowOuterUpRight": "brow_outer_up_r",
        "v2/NoseSneerLeft": "nose_sneer_l",
        "v2/NoseSneerRight": "nose_sneer_r",
        "v2/CheekPuff": "cheek_puff",
        "v2/CheekSquintLeft": "cheek_squint_l",
        "v2/CheekSquintRight": "cheek_squint_r",
        "v2/TongueOut": "tongue_out",
    }

    # 旧版自定义参数映射 (无 blendshapes 时回退)
    LEGACY_FACE_MAP = {
        "mouth_open": "/avatar/parameters/MouthOpenBlend",
        "mouth_wide": "/avatar/parameters/MouthSmileBlend",
        "left_eye_open": "/avatar/parameters/EyeOpenLeft",
        "right_eye_open": "/avatar/parameters/EyeOpenRight",
        "left_brow_up": "/avatar/parameters/BrowLeftUp",
        "right_brow_up": "/avatar/parameters/BrowRightUp",
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        vmc_port: int = 39539,
    ):
        super().__init__(name="face_vmc", host=host, port=port)
        self._osc = OSCTransport(host, port)
        self._vmc_osc = OSCTransport(host, vmc_port)
        self._coord = CoordinateTransformer()

    def _transform_face_rotation(self, face_rotation: list) -> list:
        """将 MediaPipe 面部旋转变换到 VRChat OSC 坐标系

        MediaPipe face_rotation: 摄像头坐标系欧拉角 (度) [pitch, yaw, roll]
        VRChat OSC: Unity 坐标系 ZXY 欧拉角 (度) [x, y, z]

        变换: 与 body_osc 的 camera_to_unity_rotations 一致,
        使用 FLIP_XY 矩阵 (diag(-1,-1,1)) 进行坐标系翻转。
        """
        try:
            from scipy.spatial.transform import Rotation as Rot
            import numpy as np

            # 度 -> 弧度
            euler_rad = [r * np.pi / 180.0 for r in face_rotation]

            # 摄像头 XYZ 欧拉角 -> 旋转矩阵
            rot = Rot.from_euler('xyz', euler_rad, degrees=False)
            mat = rot.as_matrix()

            # FLIP_XY 变换: 摄像头 (X左,Y下) -> Unity (X右,Y上)
            flip = np.diag([-1.0, -1.0, 1.0])
            mat_new = flip @ mat @ flip

            # 旋转矩阵 -> Unity ZXY 欧拉角 (度)
            rot_new = Rot.from_matrix(mat_new)
            euler_zxy = rot_new.as_euler("zxy", degrees=True)

            # as_euler("zxy") 返回 [z, x, y], VRChat OSC 期望 [x, y, z]
            return [float(euler_zxy[1]), float(euler_zxy[2]), float(euler_zxy[0])]
        except Exception:
            # 回退: 简单翻转 X 和 Y (近似变换)
            return [-face_rotation[0], -face_rotation[1], face_rotation[2]]

    def _do_connect(self) -> bool:
        if not self._osc.connect():
            return False
        if not self._vmc_osc.connect():
            print("[face_vmc] WARNING: VMC socket init failed, VMC face transmission unavailable")
        return True

    def _do_disconnect(self):
        self._osc.disconnect()
        self._vmc_osc.disconnect()

    def _do_send(self, pose_data: Dict):
        """发送面部表情数据

        处理流程:
        1. 当 body 推理关闭时，发送头部旋转/位置 (OSC 追踪器协议)
        2. 通过 VRChat OSC 发送面部参数 (VRCFT v2 或旧版)
        3. 通过 VMC 协议发送 blendshape 值
        """
        face_data = pose_data.get("face")
        if face_data is None:
            return

        # ===== 头部追踪数据 (body 关闭时由 face streamer 接管) =====
        if pose_data.get("body_only"):
            self._send_osc_head_tracking(face_data)

        # ===== VRChat OSC 面部参数 =====
        self._send_osc_face_params(face_data)

        # ===== VMC 面部表情 =====
        self._send_vmc_face(face_data)

    def _send_osc_head_tracking(self, face_data: Dict):
        """发送头部追踪数据到 VRChat OSC

        当 body 推理关闭时，body streamer 不发送头部数据，
        由 face streamer 接管发送 /tracking/trackers/head/ 数据。

        VRChat OSC 追踪协议:
        - /tracking/trackers/head/position: x, y, z
        - /tracking/trackers/head/rotation: x, y, z (欧拉角, 度, ZXY顺序)

        坐标系变换:
        MediaPipe face_rotation 是摄像头坐标系 (X左, Y下, Z前) 的欧拉角 (度),
        需要经过 FLIP_XY 变换到 Unity 坐标系 (X右, Y上, Z前) 的 ZXY 欧拉角。
        """
        messages = []

        # 头部旋转 - 坐标系变换
        face_rotation = face_data.get("face_rotation")
        if face_rotation is not None:
            # MediaPipe 旋转是摄像头坐标系 (度), 需要变换到 Unity ZXY
            head_euler = self._transform_face_rotation(face_rotation)
            messages.append(OSCTransport.build_message(
                "/tracking/trackers/head/rotation",
                float(head_euler[0]), float(head_euler[1]), float(head_euler[2])
            ))
        else:
            messages.append(OSCTransport.build_message(
                "/tracking/trackers/head/rotation", 0.0, 0.0, 0.0
            ))

        # 头部位置 (始终发送 0,0,0 让 VRC 使用 HMD 位置)
        messages.append(OSCTransport.build_message(
            "/tracking/trackers/head/position", 0.0, 0.0, 0.0
        ))

        if messages:
            self._osc.send_batch(messages)

    def _send_osc_face_params(self, face_data: Dict):
        """通过 OSC 发送 VRChat 面部参数"""
        # 优先使用 VRCFT 标准参数
        vrcft = face_data.get("vrcft")
        if vrcft:
            messages = []
            for param_name, value in vrcft.items():
                osc_address = f"/avatar/parameters/{param_name}"
                value = max(-1.0, min(1.0, float(value)))
                messages.append(OSCTransport.build_message(osc_address, value))
            self._osc.send_batch(messages)
            return

        # 回退: 旧版自定义参数映射
        messages = []
        for param_name, osc_address in self.LEGACY_FACE_MAP.items():
            value = face_data.get(param_name, 0.0)
            value = max(0.0, min(1.0, float(value)))
            messages.append(OSCTransport.build_message(osc_address, value))

        # 眼睛闭合
        left_closed = 1.0 - face_data.get("left_eye_open", 1.0)
        right_closed = 1.0 - face_data.get("right_eye_open", 1.0)
        messages.append(OSCTransport.build_message(
            "/avatar/parameters/EyeLeftClosed", float(left_closed)
        ))
        messages.append(OSCTransport.build_message(
            "/avatar/parameters/EyeRightClosed", float(right_closed)
        ))

        self._osc.send_batch(messages)

    def _send_vmc_face(self, face_data: Dict):
        """通过 VMC 协议发送面部 blendshape"""
        if not self._vmc_osc.socket:
            return

        vrcft = face_data.get("vrcft")
        if not vrcft:
            # 无 VRCFT 数据时，从旧版参数构建简单的 VMC 消息
            self._send_vmc_legacy_face(face_data)
            return

        messages = []

        # 发送 VMC blendshape 值
        for vrcft_name, value in vrcft.items():
            vmc_name = self.VRCFT_TO_VMC_BLENDSHAPE.get(vrcft_name)
            if vmc_name:
                value = max(-1.0, min(1.0, float(value)))
                messages.append(OSCTransport.build_message(
                    "/VMC/Ext/Blend/Val", vmc_name, value
                ))

        # 应用所有 blendshape
        messages.append(OSCTransport.build_message("/VMC/Ext/Blend/Apply"))

        # 发送可用信号
        messages.append(OSCTransport.build_message("/VMC/Ext/OK", 1))

        self._vmc_osc.send_batch(messages)

    def _send_vmc_legacy_face(self, face_data: Dict):
        """通过 VMC 协议发送旧版面部参数"""
        if not self._vmc_osc.socket:
            return

        messages = []

        # 从旧版参数映射到 VMC blendshape
        legacy_to_vmc = {
            "mouth_open": "mouth_open",
            "mouth_wide": "mouth_smile",
            "left_eye_open": "eye_open_l",
            "right_eye_open": "eye_open_r",
            "left_brow_up": "brow_inner_up_l",
            "right_brow_up": "brow_inner_up_r",
        }

        for param_name, vmc_name in legacy_to_vmc.items():
            value = face_data.get(param_name, 0.0)
            value = max(0.0, min(1.0, float(value)))
            messages.append(OSCTransport.build_message(
                "/VMC/Ext/Blend/Val", vmc_name, value
            ))

        messages.append(OSCTransport.build_message("/VMC/Ext/Blend/Apply"))
        messages.append(OSCTransport.build_message("/VMC/Ext/OK", 1))

        self._vmc_osc.send_batch(messages)
