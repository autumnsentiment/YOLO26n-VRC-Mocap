"""
身体姿态 OSC 推流模块

通过 VRChat OSC 追踪协议发送身体骨骼位置和旋转。
OSC 模式下，身体数据与手部/面部数据打包到同一 OSC 目标地址发送。

VRChat OSC 追踪协议:
- /tracking/trackers/{id}/position  -> x, y, z (世界空间, 1.0=1米, Y轴向上)
- /tracking/trackers/{id}/rotation  -> x, y, z (欧拉角, 度, ZXY顺序)

追踪器编号映射:
1 = 臀部 (Waist/Hips)
2 = 左脚 (Left Foot)
3 = 右脚 (Right Foot)
4 = 胸部 (Chest)
5 = 左膝 (Left Knee)
6 = 右膝 (Right Knee)
7 = 左肘 (Left Elbow)
8 = 右肘 (Right Elbow)
"""

from typing import Dict, List, Optional

from .base import BaseStreamer
from .osc_transport import OSCTransport
from .coord import CoordinateTransformer


class BodyOSCStreamer(BaseStreamer):
    """身体姿态 OSC 推流

    独立于手部和面部推流，可单独开关。
    当身体推流关闭时，不影响手部和面部的推流。
    """

    # VRChat OSC 追踪器编号映射
    OSC_TRACKER_MAP = {
        "Hips": 1,           # Waist
        "LeftFoot": 2,       # Left Foot
        "RightFoot": 3,      # Right Foot
        "Chest": 4,          # Chest
        "LeftLowerLeg": 5,   # Left Knee
        "RightLowerLeg": 6,  # Right Knee
        "LeftLowerArm": 7,   # Left Elbow
        "RightLowerArm": 8,  # Right Elbow
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        tracker_toggles: Dict[str, bool] = None,
        send_position: bool = True,
        send_rotation: bool = True,
        send_head_position: bool = False,
        hmd_to_neck_offset: List[float] = None,
    ):
        super().__init__(name="body_osc", host=host, port=port)
        self._osc = OSCTransport(host, port)
        self._coord = CoordinateTransformer(hmd_to_neck_offset)

        # 追踪器开关 (控制每个身体部位是否发送)
        self._tracker_toggles = tracker_toggles or {
            "Hips": True,
            "LeftFoot": True,
            "RightFoot": True,
            "Chest": True,
            "LeftLowerLeg": True,
            "RightLowerLeg": True,
            "LeftLowerArm": True,
            "RightLowerArm": True,
        }

        self._send_position = send_position
        self._send_rotation = send_rotation
        self._send_head_position = send_head_position

    def set_tracker_toggles(self, toggles: Dict[str, bool]):
        """更新追踪器开关"""
        self._tracker_toggles.update(toggles)

    def set_hmd_provider(self, provider):
        """设置 HMD 位姿数据提供者"""
        self._hmd_provider = provider
        self._coord.set_hmd_provider(provider)

    def _do_connect(self) -> bool:
        return self._osc.connect()

    def _do_disconnect(self):
        self._osc.disconnect()

    def _do_send(self, pose_data: Dict):
        """发送身体姿态数据到 VRChat OSC

        仅处理身体骨骼数据，手部和面部由各自的推流模块处理。
        """
        pose_3d = pose_data.get("pose_3d")
        if pose_3d is None:
            return

        rotations = pose_data.get("rotations", {})

        # ===== 坐标系变换 =====
        unity_pose = self._coord.camera_to_unity_positions(pose_3d)
        unity_rotations = self._coord.camera_to_unity_rotations(rotations)

        # ===== OSC 模式: 相对颈部偏移 =====
        osc_pose = self._coord.compute_osc_offsets(unity_pose)

        # ===== 批量构建消息 =====
        messages = []

        for bone_name, tracker_id in self.OSC_TRACKER_MAP.items():
            if not self._tracker_toggles.get(bone_name, True):
                continue

            if self._send_position:
                pos = osc_pose.get(bone_name, [0, 0, 0])
                messages.append(OSCTransport.build_message(
                    f"/tracking/trackers/{tracker_id}/position",
                    float(pos[0]), float(pos[1]), float(pos[2])
                ))

            if self._send_rotation:
                euler_deg = unity_rotations.get(bone_name, [0, 0, 0])
                messages.append(OSCTransport.build_message(
                    f"/tracking/trackers/{tracker_id}/rotation",
                    float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])
                ))

        # ===== 头部数据 =====
        face_data = pose_data.get("face")
        if face_data and "face_rotation" in face_data:
            face_rot = face_data["face_rotation"]
            head_euler = [float(face_rot[0]), float(face_rot[1]), float(face_rot[2])]
        else:
            head_euler = unity_rotations.get("Head", [0, 0, 0])

        if self._send_position:
            if self._send_head_position:
                if face_data and "face_position" in face_data:
                    face_pos = face_data["face_position"]
                    messages.append(OSCTransport.build_message(
                        "/tracking/trackers/head/position",
                        float(face_pos[0]), float(face_pos[1]), float(face_pos[2])
                    ))
                else:
                    messages.append(OSCTransport.build_message(
                        "/tracking/trackers/head/position", 0.0, 0.0, 0.0
                    ))
            else:
                messages.append(OSCTransport.build_message(
                    "/tracking/trackers/head/position", 0.0, 0.0, 0.0
                ))

        if self._send_rotation:
            messages.append(OSCTransport.build_message(
                "/tracking/trackers/head/rotation",
                float(head_euler[0]), float(head_euler[1]), float(head_euler[2])
            ))

        # 批量发送
        self._osc.send_batch(messages)
