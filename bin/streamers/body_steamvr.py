"""
身体姿态 SteamVR 推流模块

通过 SteamVR 命名管道驱动协议发送身体骨骼位置和旋转。
SteamVR 模式下，注册追踪器角色，调用 SteamVR 驱动进行推流。

SteamVR 命名管道协议:
- 管道名: \\.\pipe\ApriltagPipeIn
- addtracker: 注册追踪器
- updatepose: 更新追踪器位姿
- settings: 设置平滑参数
- numtrackers: 查询已注册追踪器数量
"""

import sys
import time
from typing import Dict, List, Optional

import numpy as np

from .base import BaseStreamer
from .coord import CoordinateTransformer, hmd_pose_anchor_trusted


class BodySteamVRStreamer(BaseStreamer):
    """身体姿态 SteamVR 推流

    通过命名管道与 SteamVR 驱动通信，注册追踪器并发送位姿数据。
    独立于手部 (VMT) 和面部 (VMC) 推流。
    """

    # SteamVR body_part 名称 -> tracker_toggles 骨骼名映射
    STEAMVR_TO_TOGGLE_BONE = {
        "Hips": "Hips",
        "Chest": "Chest",
        "LeftFoot": "LeftFoot",
        "RightFoot": "RightFoot",
        "LeftKnee": "LeftLowerLeg",
        "RightKnee": "RightLowerLeg",
        "LeftElbow": "LeftLowerArm",
        "RightElbow": "RightLowerArm",
    }

    # SteamVR body_part 名称 -> rotations 字典骨骼名映射
    STEAMVR_TO_BONE = {
        "Hips": "Hips",
        "Chest": "Chest",
        "LeftFoot": "LeftFoot",
        "RightFoot": "RightFoot",
        "LeftKnee": "LeftLowerLeg",
        "RightKnee": "RightLowerLeg",
        "LeftElbow": "LeftLowerArm",
        "RightElbow": "RightLowerArm",
    }

    # SteamVR body_part 名称 -> pose_3d 骨骼名映射
    STEAMVR_TO_POSE3D = {
        "Hips": "HipsTracker",
        "Chest": "ChestTracker",
        "LeftFoot": "LeftFootTracker",
        "RightFoot": "RightFootTracker",
        "LeftKnee": "LeftKneeTracker",
        "RightKnee": "RightKneeTracker",
        "LeftElbow": "LeftElbowTracker",
        "RightElbow": "RightElbowTracker",
    }

    STEAMVR_POSE_FALLBACK = {
        "Hips": "Hips",
        "Chest": "Chest",
        "LeftFoot": "LeftFoot",
        "RightFoot": "RightFoot",
        "LeftKnee": "LeftUpperLeg",
        "RightKnee": "RightUpperLeg",
        "LeftElbow": "LeftUpperArm",
        "RightElbow": "RightUpperArm",
    }

    FLOOR_RELATIVE_HMD_ORIGINS = {
        "floor",
        "stage",
        "standing",
        "local_floor",
        "local-floor",
        "steamvr",
        "openvr",
        "trackinguniverse_standing",
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        smoothing: int = 50,
        additional_smoothing: float = 0.5,
        camera_latency: float = 0,
        hmd_to_neck_offset: List[float] = None,
        trackers: List[Dict] = None,
        tracker_toggles: Dict[str, bool] = None,
        decouple_from_hmd: bool = False,
        hmd_height_scaling: Optional[Dict] = None,
        yaw_offset_deg: float = 0.0,
        auto_yaw_calibration: bool = True,
    ):
        super().__init__(name="body_steamvr", host=host, port=port)
        self._coord = CoordinateTransformer(hmd_to_neck_offset, hmd_height_scaling)

        self._pipe_name = r'\\.\pipe\ApriltagPipeIn'
        self._smoothing = smoothing
        self._additional_smoothing = additional_smoothing
        self._camera_latency = camera_latency
        self._decouple_from_hmd = decouple_from_hmd
        self._calibrated = False
        self._last_hmd_anchor_trusted = False
        self._last_hmd_tracking_origin = "unknown"
        self._hmd_anchor_source = "none"
        self._last_driver_pose_valid = False
        self._last_driver_pose_at = None
        self._manual_yaw_offset_deg = 0.0
        self._auto_yaw_calibration = bool(auto_yaw_calibration)
        self._calibrated_yaw_deg = None
        self.update_streaming_options(yaw_offset_deg=yaw_offset_deg)

        # 追踪器配置
        self._trackers = trackers or [
            {"body_part": "Hips", "role": "TrackerRole_Waist"},
            {"body_part": "Chest", "role": "TrackerRole_Chest"},
            {"body_part": "LeftFoot", "role": "TrackerRole_LeftFoot"},
            {"body_part": "RightFoot", "role": "TrackerRole_RightFoot"},
            {"body_part": "LeftKnee", "role": "TrackerRole_LeftKnee"},
            {"body_part": "RightKnee", "role": "TrackerRole_RightKnee"},
            {"body_part": "LeftElbow", "role": "TrackerRole_LeftElbow"},
            {"body_part": "RightElbow", "role": "TrackerRole_RightElbow"},
        ]

        self._tracker_toggles = tracker_toggles or {
            "Hips": True, "Chest": True, "LeftFoot": True, "RightFoot": True,
            "LeftLowerLeg": True, "RightLowerLeg": True,
            "LeftLowerArm": True, "RightLowerArm": True,
        }

        self._trackers_registered = False
        self._last_tracker_count = 0
        self._last_message_count = 0

    def set_tracker_toggles(self, toggles: Dict[str, bool]):
        self._tracker_toggles.update(toggles)

    def set_decouple_from_hmd(self, enabled: bool):
        self._decouple_from_hmd = bool(enabled)

    def update_streaming_options(
        self,
        tracker_toggles: Dict[str, bool] = None,
        decouple_from_hmd: Optional[bool] = None,
        smoothing: Optional[int] = None,
        additional_smoothing: Optional[float] = None,
        camera_latency: Optional[float] = None,
        hmd_height_scaling: Optional[Dict] = None,
        yaw_offset_deg: Optional[float] = None,
        auto_yaw_calibration: Optional[bool] = None,
    ):
        """Apply Dashboard switch changes to a running SteamVR body streamer."""
        if tracker_toggles is not None:
            self.set_tracker_toggles(tracker_toggles)
        if decouple_from_hmd is not None:
            self.set_decouple_from_hmd(decouple_from_hmd)
        if smoothing is not None:
            self._smoothing = int(smoothing)
        if additional_smoothing is not None:
            self._additional_smoothing = float(additional_smoothing)
        if camera_latency is not None:
            self._camera_latency = float(camera_latency)
        if hmd_height_scaling is not None:
            self._coord.set_hmd_height_scaling(hmd_height_scaling)
        if yaw_offset_deg is not None:
            try:
                value = float(yaw_offset_deg)
                if np.isfinite(value):
                    self._manual_yaw_offset_deg = value
            except (TypeError, ValueError):
                pass
        if auto_yaw_calibration is not None:
            enabled = bool(auto_yaw_calibration)
            if enabled != self._auto_yaw_calibration:
                self._auto_yaw_calibration = enabled
                self.recalibrate_yaw()

    def recalibrate_yaw(self):
        self._calibrated_yaw_deg = None

    @staticmethod
    def _parse_driver_hmd_pose(reply):
        """Parse getdevicepose 0: raw OpenVR position and wxyz quaternion."""
        if reply is None:
            return None
        try:
            text = reply if isinstance(reply, str) else " ".join(str(part) for part in reply)
            tokens = text.replace("\x00", " ").split()
            if "error" in tokens:
                return None
            marker = tokens.index("devicepose")
            if tokens[marker + 1] != "0":
                return None
            values = np.array([float(v) for v in tokens[marker + 2:marker + 9]])
            if len(values) != 7 or not np.all(np.isfinite(values)):
                return None
            position, quaternion = values[:3], values[3:]
            norm = float(np.linalg.norm(quaternion))
            if norm <= 1e-8:
                return None
            quaternion /= norm
            return (
                [float(position[0]), float(position[1]), float(-position[2])],
                [float(quaternion[0]), float(-quaternion[1]),
                 float(-quaternion[2]), float(quaternion[3])],
            )
        except (ValueError, IndexError, TypeError):
            return None

    def _get_driver_hmd_pose(self):
        pose = self._parse_driver_hmd_pose(
            self._send_to_pipe("getdevicepose 0", num_tries=1, wait_time=0.01)
        )
        self._last_driver_pose_valid = pose is not None
        if pose is not None:
            self._last_driver_pose_at = time.monotonic()
        return pose

    def _yaw_for_pose(self, unity_pose, hmd_quat):
        if self._auto_yaw_calibration and self._calibrated_yaw_deg is None and hmd_quat is not None:
            for left_name, right_name in (
                ("LeftHip", "RightHip"),
                ("LeftUpperLeg", "RightUpperLeg"),
                ("LeftShoulder", "RightShoulder"),
                ("LeftUpperArm", "RightUpperArm"),
            ):
                if left_name not in unity_pose or right_name not in unity_pose:
                    continue
                right = np.asarray(unity_pose[right_name], dtype=float) - unity_pose[left_name]
                right[1] = 0.0
                if not np.all(np.isfinite(right)) or np.linalg.norm(right) < 1e-4:
                    continue
                forward = np.cross(right, [0.0, 1.0, 0.0])
                try:
                    quat = np.asarray(hmd_quat, dtype=float)
                    norm = float(np.linalg.norm(quat))
                    if not np.isfinite(norm) or norm <= 1e-8:
                        break
                    quat /= norm
                    w, x, y, z = quat
                    hmd_forward = np.array([
                        2.0 * (x * z + w * y),
                        2.0 * (y * z - w * x),
                        1.0 - 2.0 * (x * x + y * y),
                    ])
                    if np.linalg.norm(hmd_forward[[0, 2]]) < 1e-4:
                        break
                    body_yaw = np.arctan2(forward[0], forward[2])
                    hmd_yaw = np.arctan2(hmd_forward[0], hmd_forward[2])
                    self._calibrated_yaw_deg = float(
                        np.rad2deg(np.arctan2(np.sin(hmd_yaw - body_yaw),
                                             np.cos(hmd_yaw - body_yaw)))
                    )
                except (TypeError, ValueError):
                    pass
                break
        return self._manual_yaw_offset_deg + (self._calibrated_yaw_deg or 0.0)

    def set_hmd_provider(self, provider):
        self._hmd_provider = provider
        self._coord.set_hmd_provider(provider)

    def _send_to_pipe(self, text: str, num_tries: int = 10, wait_time: float = 0.1):
        """通过命名管道发送命令到 SteamVR 驱动"""
        for attempt in range(num_tries):
            try:
                if sys.platform == 'win32':
                    pipe = open(self._pipe_name, 'rb+', buffering=0)
                    pipe.write(str.encode(text) + b'\x00')
                    resp = pipe.read(1024)
                    pipe.close()
                else:
                    import socket
                    client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                    client.connect("/tmp/ApriltagPipeIn")
                    client.send(text.encode('utf-8') + b'\x00')
                    resp = client.recv(1024)
                    client.close()

                string = resp.decode("utf-8")
                array = string.split(" ")
                if "error" in array:
                    raise ConnectionError("Driver returned error")
                return array
            except Exception as e:
                if attempt < num_tries - 1:
                    time.sleep(wait_time)
                else:
                    return None

    def _do_connect(self) -> bool:
        """连接 SteamVR 驱动并注册追踪器"""
        # 查询已注册的追踪器数量
        result = self._send_to_pipe("numtrackers")
        if result is None:
            print("[body_steamvr] ERROR: Could not connect to SteamVR!")
            return False

        # 解析追踪器数量
        num_trackers = 0
        for part in result:
            part_clean = part.strip().rstrip('\x00')
            if part_clean.isdigit():
                num_trackers = int(part_clean)
                break

        total_trackers = len(self._trackers)

        # 删除多余追踪器
        if num_trackers > total_trackers:
            for i in range(num_trackers - 1, total_trackers - 1, -1):
                self._send_to_pipe(f"removetracker {i}", num_tries=1, wait_time=0.05)

        # 注册缺少的追踪器
        for i in range(num_trackers, total_trackers):
            tracker = self._trackers[i]
            role = tracker.get("role", "TrackerRole_Waist")
            resp = self._send_to_pipe(f"addtracker MPTracker{i} {role}")
            if resp is None:
                print(f"[body_steamvr] ERROR: Failed to add tracker {i} ({role})")
                return False

        # 设置平滑参数
        resp = self._send_to_pipe(
            f"settings {self._smoothing} "
            f"{self._additional_smoothing} "
            f"{self._additional_smoothing}"
        )
        if resp is None:
            print("[body_steamvr] ERROR: Failed to set SteamVR settings")
            return False

        self._trackers_registered = True
        self._calibrated = False
        self.recalibrate_yaw()

        tracker_info = ", ".join(
            f"{t['body_part']}({t['role']})" for t in self._trackers
        )
        print(f"[body_steamvr] Connected! Trackers: {total_trackers} [{tracker_info}]")
        return True

    def _do_disconnect(self):
        """SteamVR 断开连接，移除已注册的追踪器"""
        if self._trackers_registered:
            # 移除所有已注册的追踪器
            num_trackers = len(self._trackers)
            for i in range(num_trackers - 1, -1, -1):
                self._send_to_pipe(f"removetracker {i}", num_tries=1, wait_time=0.05)
            print(f"[body_steamvr] Removed {num_trackers} trackers from SteamVR")
        self._trackers_registered = False
        self._calibrated = False
        self._last_driver_pose_valid = False
        self._hmd_anchor_source = "none"

    def _do_send(self, pose_data: Dict):
        """通过 SteamVR 命名管道发送姿态数据"""
        pose_3d = pose_data.get("pose_3d")
        if pose_3d is None:
            return

        rotations = pose_data.get("rotations", {})

        # ===== 坐标系变换 =====
        unity_pose = self._coord.camera_to_unity_positions(pose_3d)

        # ===== 获取 HMD 位姿 =====
        hmd_pos = None
        hmd_quat = None
        hmd_actual_height = None

        self._hmd_anchor_source = "none"
        self._last_driver_pose_valid = False
        if not self._decouple_from_hmd and self._hmd_provider is not None:
            hmd_pose = self._hmd_provider.get_pose()
            if hmd_pose.valid:
                tracking_origin = str(
                    getattr(hmd_pose, "tracking_origin", "unknown") or "unknown"
                ).strip().lower().replace(" ", "_")
                actual_height = getattr(hmd_pose, "actual_height", None)
                try:
                    explicit_height_trusted = (
                        actual_height is not None
                        and np.isfinite(float(actual_height))
                        and 0.0 < float(actual_height) <= 4.0
                    )
                except (TypeError, ValueError):
                    explicit_height_trusted = False
                origin_trusted = hmd_pose_anchor_trusted(hmd_pose)
                self._last_hmd_tracking_origin = tracking_origin
                self._last_hmd_anchor_trusted = bool(
                    origin_trusted or explicit_height_trusted
                )
                if self._last_hmd_anchor_trusted:
                    hmd_pos = hmd_pose.position
                    hmd_quat = hmd_pose.quaternion
                    hmd_actual_height = (
                        float(actual_height) if explicit_height_trusted else None
                    )
            else:
                self._last_hmd_anchor_trusted = False
                self._last_hmd_tracking_origin = "unknown"
        else:
            self._last_hmd_anchor_trusted = False
            self._last_hmd_tracking_origin = "unknown"

        anchor_in_driver_space = False
        if not self._decouple_from_hmd:
            driver_pose = self._get_driver_hmd_pose()
            if driver_pose is not None:
                hmd_pos, hmd_quat = driver_pose
                anchor_in_driver_space = True
                self._hmd_anchor_source = "driver_raw"
                self._last_hmd_anchor_trusted = True
                if hmd_actual_height is None and self._hmd_provider is not None:
                    if (hmd_pose.valid and
                            self._last_hmd_tracking_origin in self.FLOOR_RELATIVE_HMD_ORIGINS):
                        try:
                            height = float(hmd_pose.position[1])
                            if np.isfinite(height) and 0.0 < height <= 4.0:
                                hmd_actual_height = height
                        except (TypeError, ValueError, IndexError):
                            pass
            elif hmd_pos is not None:
                self._hmd_anchor_source = "provider"

        # ===== 计算特殊部位位置 =====
        # 髋部中心
        left_hip = unity_pose.get("LeftUpperLeg", [0, 0, 0])
        right_hip = unity_pose.get("RightUpperLeg", [0, 0, 0])
        hip_center = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]

        # 胸部位置 (肩中点与髋中点的加权平均)
        left_shoulder = unity_pose.get("LeftUpperArm", [0, 0, 0])
        right_shoulder = unity_pose.get("RightUpperArm", [0, 0, 0])
        shoulder_center = [(left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)]
        chest_pos = [(shoulder_center[i] * 0.6 + hip_center[i] * 0.4) for i in range(3)]
        unity_pose.setdefault("Hips", hip_center)
        unity_pose.setdefault("Chest", chest_pos)
        yaw_deg = (
            self._yaw_for_pose(unity_pose, hmd_quat)
            if not self._decouple_from_hmd and hmd_pos is not None else 0.0
        )
        unity_pose = self._coord.rotate_unity_pose_yaw(unity_pose, yaw_deg)

        # Default VR mode anchors virtual body trackers to the real HMD/head
        # pose so body inference uses the headset as its world-space base.
        steamvr_pose = self._coord.compute_steamvr_positions(
            unity_pose,
            None if self._decouple_from_hmd else hmd_pos,
            None if self._decouple_from_hmd else hmd_quat,
            None if self._decouple_from_hmd else hmd_actual_height,
            anchor_in_driver_space=anchor_in_driver_space,
        )

        if not self._calibrated:
            self._calibrated = True

        # 部位名 -> 3D 位置映射 (已变换到 SteamVR 坐标系)
        body_part_positions = {}
        for body_part, pose_key in self.STEAMVR_TO_POSE3D.items():
            fallback_key = self.STEAMVR_POSE_FALLBACK.get(body_part, body_part)
            body_part_positions[body_part] = steamvr_pose.get(
                pose_key,
                steamvr_pose.get(fallback_key, [0, 0, 0]),
            )

        # ===== 遍历追踪器发送位姿 =====
        latency = self._camera_latency
        active_tracker_count = 0
        message_count = 0

        for idx, tracker in enumerate(self._trackers):
            body_part = tracker.get("body_part", "Hips")

            # 检查追踪器开关
            toggle_bone = self.STEAMVR_TO_TOGGLE_BONE.get(body_part, body_part)
            if not self._tracker_toggles.get(toggle_bone, True):
                continue

            pos_3d = body_part_positions.get(body_part, [0, 0, 0])

            # 旋转: 通过映射查找对应的骨骼旋转
            bone_name = self.STEAMVR_TO_BONE.get(body_part, body_part)
            rot_euler = rotations.get(bone_name, rotations.get("Hips", [0, 0, 0]))
            quat = self._coord.euler_to_quat_openvr(rot_euler, yaw_deg)

            self._send_to_pipe(
                f"updatepose {idx} {pos_3d[0]} {pos_3d[1]} {pos_3d[2]} "
                f"{quat[3]} {quat[0]} {quat[1]} {quat[2]} {latency} 0.8",
                num_tries=1, wait_time=0
            )
            active_tracker_count += 1
            message_count += 1

        self._last_tracker_count = active_tracker_count
        self._last_message_count = message_count

    def get_status(self) -> Dict:
        status = super().get_status()
        status.update({
            "decouple_from_hmd": self._decouple_from_hmd,
            "hmd_height_scaling": self._coord.hmd_height_scaling,
            "hmd_height_state": self._coord.hmd_height_state,
            "last_tracker_count": self._last_tracker_count,
            "last_message_count": self._last_message_count,
            "trackers_registered": self._trackers_registered,
            "hmd_anchor_trusted": self._last_hmd_anchor_trusted,
            "hmd_anchor_applied": bool(
                self._last_hmd_anchor_trusted and not self._decouple_from_hmd
            ),
            "hmd_tracking_origin": self._last_hmd_tracking_origin,
            "hmd_anchor_source": self._hmd_anchor_source,
            "last_driver_pose_valid": self._last_driver_pose_valid,
            "last_driver_pose_age": (
                max(0.0, time.monotonic() - self._last_driver_pose_at)
                if self._last_driver_pose_at is not None else None
            ),
            "yaw_calibrated": self._calibrated_yaw_deg is not None,
            "yaw_offset_deg": self._manual_yaw_offset_deg + (self._calibrated_yaw_deg or 0.0),
        })
        return status
