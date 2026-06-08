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
from .coord import CoordinateTransformer


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
        "Hips": None,  # 需要计算髋部中心
        "Chest": None,  # 需要计算胸部位置
        "LeftFoot": "LeftFoot",
        "RightFoot": "RightFoot",
        "LeftKnee": "LeftLowerLeg",
        "RightKnee": "RightLowerLeg",
        "LeftElbow": "LeftLowerArm",
        "RightElbow": "RightLowerArm",
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
    ):
        super().__init__(name="body_steamvr", host=host, port=port)
        self._coord = CoordinateTransformer(hmd_to_neck_offset)

        self._pipe_name = r'\\.\pipe\ApriltagPipeIn'
        self._smoothing = smoothing
        self._additional_smoothing = additional_smoothing
        self._camera_latency = camera_latency
        self._calibrated = False

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

    def set_tracker_toggles(self, toggles: Dict[str, bool]):
        self._tracker_toggles.update(toggles)

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

    def _do_send(self, pose_data: Dict):
        """通过 SteamVR 命名管道发送姿态数据"""
        pose_3d = pose_data.get("pose_3d")
        if pose_3d is None:
            return

        rotations = pose_data.get("rotations", {})

        # ===== 坐标系变换 =====
        unity_pose = self._coord.camera_to_unity_positions(pose_3d)

        # ===== 获取 HMD 位姿 =====
        hmd_pos = [0.0, 1.7, 0.0]  # 默认: 站立高度 1.7m
        hmd_quat = None

        if self._hmd_provider is not None:
            hmd_pose = self._hmd_provider.get_pose()
            if hmd_pose.valid:
                hmd_pos = hmd_pose.position
                hmd_quat = hmd_pose.quaternion

        # ===== SteamVR 绝对位置计算 =====
        steamvr_pose = self._coord.compute_steamvr_positions(
            unity_pose, hmd_pos, hmd_quat
        )

        if not self._calibrated:
            self._calibrated = True

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

        # 部位名 -> 3D 位置映射 (已变换到 SteamVR 坐标系)
        body_part_positions = {
            "Hips": steamvr_pose.get("Hips", hip_center),
            "Chest": steamvr_pose.get("Chest", chest_pos),
            "LeftFoot": steamvr_pose.get("LeftFoot", [0, 0, 0]),
            "RightFoot": steamvr_pose.get("RightFoot", [0, 0, 0]),
            "LeftKnee": steamvr_pose.get("LeftLowerLeg", [0, 0, 0]),
            "RightKnee": steamvr_pose.get("RightLowerLeg", [0, 0, 0]),
            "LeftElbow": steamvr_pose.get("LeftLowerArm", [0, 0, 0]),
            "RightElbow": steamvr_pose.get("RightLowerArm", [0, 0, 0]),
        }

        # ===== 遍历追踪器发送位姿 =====
        latency = self._camera_latency

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
            quat = self._coord.euler_to_quat_openvr(rot_euler)

            self._send_to_pipe(
                f"updatepose {idx} {pos_3d[0]} {pos_3d[1]} {pos_3d[2]} "
                f"{quat[3]} {quat[0]} {quat[1]} {quat[2]} {latency} 0.8",
                num_tries=1, wait_time=0
            )
