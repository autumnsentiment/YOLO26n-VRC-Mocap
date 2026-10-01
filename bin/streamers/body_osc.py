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
from .coord import CoordinateTransformer, hmd_pose_anchor_trusted


class BodyOSCStreamer(BaseStreamer):
    """身体姿态 OSC 推流

    独立于手部和面部推流，可单独开关。
    当身体推流关闭时，不影响手部和面部的推流。
    """

    # VRChat OSC 追踪器编号映射
    OSC_TRACKER_MAP = {
        "HipsTracker": 1,        # Waist
        "LeftFootTracker": 2,    # Left Foot
        "RightFootTracker": 3,   # Right Foot
        "ChestTracker": 4,       # Chest
        "LeftKneeTracker": 5,    # Left Knee
        "RightKneeTracker": 6,   # Right Knee
        "LeftElbowTracker": 7,   # Left Elbow
        "RightElbowTracker": 8,  # Right Elbow
    }

    TRACKER_TOGGLE_BONE = {
        "HipsTracker": "Hips",
        "ChestTracker": "Chest",
        "LeftFootTracker": "LeftFoot",
        "RightFootTracker": "RightFoot",
        "LeftKneeTracker": "LeftLowerLeg",
        "RightKneeTracker": "RightLowerLeg",
        "LeftElbowTracker": "LeftLowerArm",
        "RightElbowTracker": "RightLowerArm",
    }

    TRACKER_POSITION_FALLBACK = {
        "HipsTracker": "Hips",
        "ChestTracker": "Chest",
        "LeftFootTracker": "LeftFoot",
        "RightFootTracker": "RightFoot",
        "LeftKneeTracker": "LeftUpperLeg",
        "RightKneeTracker": "RightUpperLeg",
        "LeftElbowTracker": "LeftUpperArm",
        "RightElbowTracker": "RightUpperArm",
    }

    TRACKER_ROTATION_BONE = {
        "HipsTracker": "Hips",
        "ChestTracker": "Chest",
        "LeftFootTracker": "LeftFoot",
        "RightFootTracker": "RightFoot",
        "LeftKneeTracker": "LeftLowerLeg",
        "RightKneeTracker": "RightLowerLeg",
        "LeftElbowTracker": "LeftLowerArm",
        "RightElbowTracker": "RightLowerArm",
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        tracker_toggles: Dict[str, bool] = None,
        send_position: bool = True,
        send_rotation: bool = True,
        send_head_position: bool = False,
        decouple_from_hmd: bool = False,
        hmd_to_neck_offset: List[float] = None,
        hmd_height_scaling: Optional[Dict] = None,
    ):
        super().__init__(name="body_osc", host=host, port=port)
        self._osc = OSCTransport(host, port)
        self._coord = CoordinateTransformer(hmd_to_neck_offset, hmd_height_scaling)

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
        self._decouple_from_hmd = decouple_from_hmd
        self._last_message_count = 0
        self._last_tracker_count = 0
        self._last_hmd_present = False
        self._last_hmd_anchor_trusted = False
        self._last_hmd_anchor_applied = False
        self._last_hmd_tracking_origin = "unknown"

    def set_tracker_toggles(self, toggles: Dict[str, bool]):
        """更新追踪器开关"""
        self._tracker_toggles.update(toggles)

    def set_decouple_from_hmd(self, enabled: bool):
        self._decouple_from_hmd = bool(enabled)

    def update_streaming_options(
        self,
        tracker_toggles: Dict[str, bool] = None,
        send_position: Optional[bool] = None,
        send_rotation: Optional[bool] = None,
        send_head_position: Optional[bool] = None,
        decouple_from_hmd: Optional[bool] = None,
        hmd_height_scaling: Optional[Dict] = None,
    ):
        """Apply Dashboard switch changes to a running OSC body streamer."""
        if tracker_toggles is not None:
            self.set_tracker_toggles(tracker_toggles)
        if send_position is not None:
            self._send_position = bool(send_position)
        if send_rotation is not None:
            self._send_rotation = bool(send_rotation)
        if send_head_position is not None:
            self._send_head_position = bool(send_head_position)
        if decouple_from_hmd is not None:
            self.set_decouple_from_hmd(decouple_from_hmd)
        if hmd_height_scaling is not None:
            self._coord.set_hmd_height_scaling(hmd_height_scaling)

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
        hmd_position = None
        hmd_quaternion = None
        hmd_actual_height = None
        hmd_present = False
        hmd_anchor_trusted = False
        hmd_tracking_origin = "unknown"
        if self._hmd_provider is not None:
            hmd_pose = self._hmd_provider.get_pose()
            if hmd_pose.valid:
                hmd_present = True
                hmd_tracking_origin = str(
                    getattr(hmd_pose, "tracking_origin", "unknown") or "unknown"
                ).strip().lower().replace(" ", "_")
                hmd_anchor_trusted = hmd_pose_anchor_trusted(hmd_pose)
                if hmd_anchor_trusted:
                    hmd_position = hmd_pose.position
                    hmd_quaternion = hmd_pose.quaternion
                    hmd_actual_height = getattr(hmd_pose, "actual_height", None)

        self._last_hmd_present = hmd_present
        self._last_hmd_anchor_trusted = bool(hmd_anchor_trusted)
        self._last_hmd_anchor_applied = bool(
            hmd_anchor_trusted and not self._decouple_from_hmd
        )
        self._last_hmd_tracking_origin = hmd_tracking_origin

        # Default VR mode anchors body trackers to the real HMD/head pose.
        # This makes the HMD pose the world-space base point for the inferred
        # body instead of creating an independent camera-space body origin.
        osc_pose = self._coord.compute_osc_offsets(
            unity_pose,
            None if self._decouple_from_hmd else hmd_position,
            None if self._decouple_from_hmd else hmd_quaternion,
            None if self._decouple_from_hmd else hmd_actual_height,
        )

        # ===== 批量构建消息 =====
        messages = []
        active_tracker_count = 0

        for bone_name, tracker_id in self.OSC_TRACKER_MAP.items():
            toggle_bone = self.TRACKER_TOGGLE_BONE.get(bone_name, bone_name)
            if not self._tracker_toggles.get(toggle_bone, True):
                continue

            tracker_has_message = False

            if self._send_position:
                fallback_bone = self.TRACKER_POSITION_FALLBACK.get(bone_name, bone_name)
                pos = osc_pose.get(bone_name, osc_pose.get(fallback_bone, [0, 0, 0]))
                messages.append(OSCTransport.build_message(
                    f"/tracking/trackers/{tracker_id}/position",
                    float(pos[0]), float(pos[1]), float(pos[2])
                ))
                tracker_has_message = True

            if self._send_rotation:
                rotation_bone = self.TRACKER_ROTATION_BONE.get(bone_name, bone_name)
                euler_deg = unity_rotations.get(rotation_bone, [0, 0, 0])
                messages.append(OSCTransport.build_message(
                    f"/tracking/trackers/{tracker_id}/rotation",
                    float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])
                ))
                tracker_has_message = True

            if tracker_has_message:
                active_tracker_count += 1

        # ===== 头部数据 =====
        face_data = pose_data.get("face")
        if face_data and "face_rotation" in face_data:
            face_rot = face_data["face_rotation"]
            head_euler = [float(face_rot[0]), float(face_rot[1]), float(face_rot[2])]
        else:
            head_euler = unity_rotations.get("Head", [0, 0, 0])

        if self._send_position:
            if self._send_head_position:
                if hmd_present:
                    messages.append(OSCTransport.build_message(
                        "/tracking/trackers/head/position", 0.0, 0.0, 0.0
                    ))
                elif face_data and "face_position" in face_data:
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

        self._last_message_count = len(messages)
        self._last_tracker_count = active_tracker_count

        # 批量发送
        if messages:
            self._osc.send_batch(messages)

    def get_status(self) -> Dict:
        status = super().get_status()
        status.update({
            "send_position": self._send_position,
            "send_rotation": self._send_rotation,
            "send_head_position": self._send_head_position,
            "decouple_from_hmd": self._decouple_from_hmd,
            "hmd_height_scaling": self._coord.hmd_height_scaling,
            "hmd_height_state": self._coord.hmd_height_state,
            "last_message_count": self._last_message_count,
            "last_tracker_count": self._last_tracker_count,
            "hmd_present": self._last_hmd_present,
            "hmd_anchor_trusted": self._last_hmd_anchor_trusted,
            "hmd_anchor_applied": self._last_hmd_anchor_applied,
            "hmd_tracking_origin": self._last_hmd_tracking_origin,
        })
        return status
