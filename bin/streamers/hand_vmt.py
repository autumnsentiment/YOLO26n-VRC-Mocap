"""
手部追踪 VMT 推流模块

通过 VMT (Virtual Motion Tracker) OpenVR 驱动层发送手部追踪数据。
VMT 作为 SteamVR 驱动运行，向 OpenVR 注册左右手兼容控制器。

VMT 协议:
- /VMT/SetRoomMatrix: 设置房间矩阵 (初始化)
- /VMT/Joint/Driver: 发送手部位置+四元数
  格式: [hand_type, enable, timeoffset, x, y, z, qx, qy, qz, qw, "HMD"]
  hand_type: 1=左手, 2=右手
  enable: 5=左手, 6=右手
- /VMT/Skeleton/Scalar: 发送手指张开系数
  格式: [device_index, finger_index, open_amount, 0, 0]
  finger_index: 1-5 (thumb/index/middle/ring/pinky)
  open_amount: 0=握拳, 1=张开 (与 MediaPipe curl 方向相反)
- /VMT/Skeleton/Apply: 应用手指姿态
  格式: [hand_type, 0.0]

同时通过 OSC 发送 VRChat 手指参数 (VRChat 原生手指追踪):
- /avatar/parameters/LeftThumbCurl 等
- /avatar/parameters/RightThumbCurl 等
"""

import time
from typing import Dict, List, Optional

from .base import BaseStreamer
from .osc_transport import OSCTransport
from .coord import CoordinateTransformer
from vmt_runtime import get_vmt_runtime_status


class HandVMTStreamer(BaseStreamer):
    """手部追踪 VMT 推流

    通过 VMT 协议发送手部位置/旋转到 OpenVR 驱动层，
    同时通过 OSC 发送手指弯曲度到 VRChat。

    独立于身体和面部推流，可单独开关。
    """

    FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
    FINGER_CAP = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    VMT_FINGER_INDICES = {
        "thumb": 1,
        "index": 2,
        "middle": 3,
        "ring": 4,
        "pinky": 5,
    }
    VMT_COMPATIBLE_CONTROLLER_TYPES = {"left": 5, "right": 6}
    VMT_PRE_REGISTER_HOLD_SECONDS = 1.0
    VMT_FINGER_INPUT_INDICES = {
        "index": 3,
        "middle": 4,
        "ring": 5,
        "pinky": 6,
    }
    VRCHAT_GESTURE_VALUES = {
        "neutral": 0,
        "fist": 1,
        "open": 2,
        "point": 3,
        "peace": 4,
        "rock": 5,
        "call": 6,
        "thumb_up": 7,
    }
    DEFAULT_DESKTOP_GESTURE_PARAMETERS = {
        "left": [
            "DynamicGesture/CurrentGestureLeft",
            "DynamicGesture/CurrentLeftIndex",
            "MocapGestureLeft",
        ],
        "right": [
            "DynamicGesture/CurrentGestureRight",
            "DynamicGesture/CurrentRightIndex",
            "MocapGestureRight",
        ],
    }

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        vmt_host: str = None,
        vmt_port: int = 39570,
        hmd_to_neck_offset: List[float] = None,
        desktop_osc_enabled: bool = True,
        vmt_enabled: bool = True,
        vmt_local_mode: bool = False,
        vmt_skeleton_enabled: bool = True,
        vmt_input_enabled: bool = True,
        vmt_auto_enable_skeleton: bool = True,
        vmt_register_on_steamvr_startup: bool = True,
        vmt_keep_controllers_active: bool = True,
        vmt_hand_selection_priority: int = 100,
        vmt_left_device_index: int = 1,
        vmt_right_device_index: int = 2,
        custom_parameters_enabled: bool = True,
        lost_reset_timeout: float = 0.35,
        desktop_left_gesture_parameters=None,
        desktop_right_gesture_parameters=None,
    ):
        super().__init__(name="hand_vmt", host=host, port=port)
        self._osc = OSCTransport(host, port)
        self.vmt_local_mode = bool(vmt_local_mode)
        resolved_vmt_host = "127.0.0.1" if self.vmt_local_mode else (vmt_host or host)
        self._vmt_osc = OSCTransport(resolved_vmt_host, vmt_port)
        self._coord = CoordinateTransformer(hmd_to_neck_offset)
        self.desktop_osc_enabled = bool(desktop_osc_enabled)
        self.vmt_enabled = bool(vmt_enabled)
        self.vmt_skeleton_enabled = bool(vmt_skeleton_enabled)
        self.vmt_input_enabled = bool(vmt_input_enabled)
        self.vmt_auto_enable_skeleton = bool(vmt_auto_enable_skeleton)
        self.vmt_register_on_steamvr_startup = bool(
            vmt_register_on_steamvr_startup
        )
        self.vmt_keep_controllers_active = bool(vmt_keep_controllers_active)
        self.vmt_hand_selection_priority = (
            self._normalize_vmt_hand_selection_priority(
                vmt_hand_selection_priority
            )
        )
        self.vmt_device_indices = self._normalize_vmt_device_indices(
            vmt_left_device_index,
            vmt_right_device_index,
        )
        self.custom_parameters_enabled = bool(custom_parameters_enabled)
        self.lost_reset_timeout = max(0.0, float(lost_reset_timeout))
        self.desktop_gesture_parameters = {
            "left": self._normalize_parameter_names(
                desktop_left_gesture_parameters,
                self.DEFAULT_DESKTOP_GESTURE_PARAMETERS["left"],
            ),
            "right": self._normalize_parameter_names(
                desktop_right_gesture_parameters,
                self.DEFAULT_DESKTOP_GESTURE_PARAMETERS["right"],
            ),
        }
        self._vmt_initialized = False
        self._vmt_pre_registered = False
        self._last_seen_at = {"left": None, "right": None}
        self._neutral_sent = {"left": True, "right": True}
        self._desktop_osc_message_count = 0
        self._last_desktop_send_time = None
        self._last_desktop_gestures = {"left": 0, "right": 0}
        self._last_hand_count = 0
        self._last_pose_source = "none"
        self._vmt_pose_packets = 0
        self._vmt_skeleton_packets = 0
        self._vmt_input_packets = 0
        self._last_vmt_send_time = None
        self._last_vmt_open_values = {"left": {}, "right": {}}
        self._vmt_last_seen_at = {"left": None, "right": None}
        self._vmt_neutral_sent = {"left": True, "right": True}
        self._vmt_controller_active = {"left": False, "right": False}
        self._vmt_runtime_status = {}
        self._last_vmt_runtime_check = 0.0
        self._vmt_warning_emitted = False
        # Track the endpoint edge explicitly.  VMT may start after mocap, so
        # controller registration must be replayed on a false -> true edge.
        self._vmt_last_ready_state = None
        self._vmt_reinit_count = 0
        self._vmt_last_reinit_time = None
        self._vmt_priority_restart_required = False
        self._vmt_startup_registration_restart_required = False

    @staticmethod
    def _normalize_vmt_hand_selection_priority(value) -> int:
        try:
            priority = int(value)
        except (TypeError, ValueError):
            priority = 100
        return max(0, min(1000, priority))

    @staticmethod
    def _normalize_vmt_device_indices(left, right) -> Dict[str, int]:
        try:
            left_index = max(0, int(left))
        except (TypeError, ValueError):
            left_index = 1
        try:
            right_index = max(0, int(right))
        except (TypeError, ValueError):
            right_index = 2
        if right_index == left_index:
            right_index = left_index + 1
        return {"left": left_index, "right": right_index}

    @staticmethod
    def _normalize_parameter_names(value, defaults: List[str]) -> List[str]:
        if value is None:
            values = list(defaults)
        elif isinstance(value, str):
            values = value.replace(";", ",").replace("\n", ",").split(",")
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            values = list(defaults)

        result = []
        for value in values:
            name = str(value or "").strip()
            prefix = "/avatar/parameters/"
            if name.startswith(prefix):
                name = name[len(prefix):]
            if name and name not in result:
                result.append(name)
        return result

    def _gesture_parameter_names(self, hand_label: str) -> List[str]:
        prefix = "Left" if hand_label == "left" else "Right"
        names = [f"Gesture{prefix}"]
        for name in self.desktop_gesture_parameters.get(hand_label, []):
            if name not in names:
                names.append(name)
        return names

    def set_hmd_provider(self, provider):
        self._hmd_provider = provider
        self._coord.set_hmd_provider(provider)

    def _do_connect(self) -> bool:
        connected = False
        if self.desktop_osc_enabled:
            if self._osc.connect():
                connected = True
            else:
                print("[hand_vmt] WARNING: desktop OSC socket init failed")
        if self.vmt_enabled:
            self._refresh_vmt_runtime_status(force=True)
            self._vmt_last_ready_state = bool(
                self._vmt_runtime_status.get("ready")
            )
            if not self._vmt_osc.connect():
                print("[hand_vmt] WARNING: VMT socket init failed, VMT hand tracking unavailable")
            else:
                connected = True
                self._vmt_init()
        return connected

    def _do_disconnect(self):
        if self.desktop_osc_enabled and self._osc.socket:
            self._send_osc_hand_params({}, force_reset=True)
        if self.vmt_enabled and self._vmt_osc.socket and self._vmt_initialized:
            self._reset_missing_vmt_hands({}, force_reset=True)
        self._osc.disconnect()
        self._vmt_osc.disconnect()
        self._vmt_initialized = False
        self._vmt_pre_registered = False
        self._vmt_last_ready_state = None
        self._vmt_reinit_count = 0
        self._vmt_last_reinit_time = None

    def _vmt_init(self):
        """Enable the VMT skeletal path used by compatible hand controllers."""
        try:
            driver_priority = self._vmt_runtime_status.get(
                "hand_selection_priority"
            )
            if (
                self._vmt_runtime_status.get("ready")
                and driver_priority is not None
                and driver_priority != self.vmt_hand_selection_priority
            ):
                # VMT stores Priority immediately, but an already activated
                # OpenVR controller keeps the old property until SteamVR restarts.
                self._vmt_priority_restart_required = True
            driver_startup_registration = self._vmt_runtime_status.get(
                "add_compatible_controller_on_startup"
            )
            if (
                self._vmt_runtime_status.get("ready")
                and driver_startup_registration is not None
                and driver_startup_registration
                != self.vmt_register_on_steamvr_startup
            ):
                # This setting is consumed while the VMT driver starts. Persist
                # it now, then require one complete SteamVR restart.
                self._vmt_startup_registration_restart_required = True
            self._vmt_osc.send(OSCTransport.build_message(
                "/VMT/Config",
                "AddCompatibleControllerOnStartup",
                "true" if self.vmt_register_on_steamvr_startup else "false",
            ))
            self._vmt_osc.send(OSCTransport.build_message(
                "/VMT/Config",
                "Priority",
                str(self.vmt_hand_selection_priority),
            ))
            if self.vmt_auto_enable_skeleton:
                self._vmt_osc.send(OSCTransport.build_message(
                    "/VMT/Config", "SkeletonInput", "true"
                ))
            self._vmt_osc.send(OSCTransport.build_message(
                "/VMT/SetAutoPoseUpdate", 1
            ))
            self._vmt_initialized = True
            self._pre_register_vmt_controller_types()
            print(
                "[hand_vmt] VMT compatible hand controllers enabled "
                f"(left=VMT_{self.vmt_device_indices['left']}, "
                f"right=VMT_{self.vmt_device_indices['right']}, "
                f"priority={self.vmt_hand_selection_priority}, "
                f"startup-register="
                f"{'ON' if self.vmt_register_on_steamvr_startup else 'OFF'}, "
                f"skeleton={'ON' if self.vmt_skeleton_enabled else 'OFF'}, "
                f"input-mirror={'ON' if self.vmt_input_enabled else 'OFF'})"
            )
            self._report_vmt_runtime_warning()
            return True
        except Exception as e:
            print(f"[hand_vmt] VMT init error: {e}")
            return False

    def _pre_register_vmt_controller_types(self):
        """Register left/right compatible controller classes before VRChat starts.

        VMT only accepts a device class on the first registration in a SteamVR
        session. Keep a harmless standby pose alive so SteamVR can finish
        activation and bind the left/right roles before hand tracking starts.
        """
        standby_positions = {
            "left": [-0.25, -0.25, -0.35],
            "right": [0.25, -0.25, -0.35],
        }
        registered_at = time.monotonic()
        for side in ("left", "right"):
            device_index = self.vmt_device_indices[side]
            controller_type = self.VMT_COMPATIBLE_CONTROLLER_TYPES[side]
            position = standby_positions[side]
            self._vmt_osc.send(OSCTransport.build_message(
                "/VMT/Joint/Driver",
                device_index,
                controller_type,
                0.0,
                float(position[0]),
                float(position[1]),
                float(position[2]),
                0.0,
                0.0,
                0.0,
                1.0,
                "HMD",
            ))
            self._vmt_last_seen_at[side] = (
                registered_at + self.VMT_PRE_REGISTER_HOLD_SECONDS
            )
            self._vmt_neutral_sent[side] = False
            self._vmt_controller_active[side] = True
        self._vmt_pre_registered = True

    def _refresh_vmt_runtime_status(self, force: bool = False) -> Dict:
        now = time.monotonic()
        if not force and now - self._last_vmt_runtime_check < 5.0:
            return self._vmt_runtime_status
        self._last_vmt_runtime_check = now
        self._vmt_runtime_status = get_vmt_runtime_status(
            host=self._vmt_osc.host,
            port=self._vmt_osc.port,
        )
        if self._vmt_runtime_status.get("ready"):
            self._vmt_warning_emitted = False
        return self._vmt_runtime_status

    def _report_vmt_runtime_warning(self):
        status = self._vmt_runtime_status or self._refresh_vmt_runtime_status(force=True)
        if status.get("ready") or self._vmt_warning_emitted:
            return
        self._vmt_warning_emitted = True
        print(
            "[hand_vmt] WARNING: VMT endpoint is not ready: "
            f"{status.get('message', 'unknown state')} "
            f"({self._vmt_osc.host}:{self._vmt_osc.port})"
        )

    def _handle_vmt_runtime_transition(self, current: Dict) -> None:
        """Keep VMT registration aligned with the driver's actual lifecycle.

        The UDP socket can be created before SteamVR/VMT starts.  In that
        case the first registration packets are dropped by the OS.  Replay
        the idempotent VMT setup and controller registration exactly once when
        the endpoint becomes ready, without touching the socket connection.
        """
        current_ready = bool((current or {}).get("ready"))
        previous_ready = self._vmt_last_ready_state

        if previous_ready is not None and current_ready and not previous_ready:
            if self._vmt_osc.socket:
                self._vmt_priority_restart_required = False
                self._vmt_startup_registration_restart_required = False
                if self._vmt_init():
                    self._vmt_reinit_count += 1
                    self._vmt_last_reinit_time = time.time()
                    print(
                        "[hand_vmt] VMT endpoint is ready; "
                        "SteamVR hand controllers re-registered"
                    )
            else:
                print(
                    "[hand_vmt] WARNING: VMT endpoint became ready but "
                    "the UDP socket is unavailable"
                )
        elif not current_ready:
            self._report_vmt_runtime_warning()

        self._vmt_last_ready_state = current_ready

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
        if self.vmt_enabled and self.connected:
            current = self._refresh_vmt_runtime_status()
            self._handle_vmt_runtime_transition(current)

        hand_data = pose_data.get("hands")
        if hand_data is None:
            return

        # 将列表格式转为字典格式
        if isinstance(hand_data, list):
            hand_dict = {}
            for h in hand_data:
                label = str(h.get("label", "unknown")).strip().lower()
                if label in ("left", "right"):
                    hand_dict[label] = h
            hand_data = hand_dict
        elif isinstance(hand_data, dict):
            hand_data = {
                str(label).strip().lower(): data
                for label, data in hand_data.items()
                if str(label).strip().lower() in ("left", "right")
            }
        else:
            hand_data = {}

        self._last_hand_count = len(hand_data)

        # ===== 坐标变换 =====
        pose_3d = pose_data.get("pose_3d", {})
        rotations = pose_data.get("rotations", {})

        openvr_quaternions = {}
        if self.vmt_enabled and hand_data:
            unity_pose, unity_rotations, osc_pose = self._estimate_hand_positions_from_landmarks(hand_data)
            openvr_quaternions = self._build_openvr_hand_quaternions(hand_data, rotations)
            self._last_pose_source = "hand_landmarks_estimate"
            if isinstance(pose_3d, dict) and pose_3d:
                body_unity_pose = self._coord.camera_to_unity_positions(pose_3d)
                body_unity_rotations = self._coord.camera_to_unity_rotations(rotations)
                body_osc_pose = self._coord.compute_osc_offsets(body_unity_pose)
                for hand_bone in ("LeftHand", "RightHand"):
                    if hand_bone in body_unity_pose:
                        unity_pose[hand_bone] = body_unity_pose[hand_bone]
                    if hand_bone in body_unity_rotations:
                        unity_rotations[hand_bone] = body_unity_rotations[hand_bone]
                    if hand_bone in body_osc_pose:
                        osc_pose[hand_bone] = body_osc_pose[hand_bone]
                if any(name in body_osc_pose for name in ("LeftHand", "RightHand")):
                    self._last_pose_source = "body_pose_3d"
        else:
            self._last_pose_source = "none"

        # ===== VMT 手部位置+旋转 =====
        if self.vmt_enabled and self._vmt_osc.socket and self._vmt_initialized:
            if hand_data:
                self._send_vmt_hands(osc_pose, openvr_quaternions, hand_data)
                self._last_vmt_send_time = time.time()
            self._reset_missing_vmt_hands(hand_data)

        # ===== OSC 手指参数 =====
        if self.desktop_osc_enabled and self._osc.socket:
            self._send_osc_hand_params(hand_data)

    def _build_openvr_hand_quaternions(
        self,
        hand_data: Dict,
        body_rotations: Dict,
    ) -> Dict[str, List[float]]:
        """Build OpenVR quaternions without reinterpreting Unity ZXY angles."""
        import math

        result = {}
        body_rotations = body_rotations if isinstance(body_rotations, dict) else {}
        for hand_label, is_left in (("left", True), ("right", False)):
            data = hand_data.get(hand_label)
            if not isinstance(data, dict):
                continue
            bone_name = "LeftHand" if is_left else "RightHand"
            euler_rad = body_rotations.get(bone_name)
            if not isinstance(euler_rad, (list, tuple)) or len(euler_rad) < 3:
                # HandTracker reports [yaw, pitch, roll] in degrees.
                rotation = data.get("rotation", [0.0, 0.0, 0.0])
                try:
                    euler_rad = [
                        math.radians(float(rotation[2])),
                        math.radians(float(rotation[1])),
                        math.radians(float(rotation[0])),
                    ]
                except (TypeError, ValueError, IndexError):
                    euler_rad = [0.0, 0.0, 0.0]
            converter = getattr(self._coord, "euler_to_quat_openvr", None)
            if converter is None:
                converter = CoordinateTransformer().euler_to_quat_openvr
            result[bone_name] = converter(euler_rad)
        return result

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
        openvr_quaternions: Dict,
        hand_data: Dict,
    ):
        """Send controller pose, skeletal fingers, and compatibility inputs."""
        now = time.monotonic()
        for hand_label, is_left in [("left", True), ("right", False)]:
            data = hand_data.get(hand_label)
            if data is None:
                continue

            # 获取手部位置
            hand_bone = "LeftHand" if is_left else "RightHand"
            pos = osc_pose.get(hand_bone, [0, 0, 0])

            qx, qy, qz, qw = openvr_quaternions.get(
                hand_bone, [0.0, 0.0, 0.0, 1.0]
            )

            # Compatible controller types 5/6 expose Knuckles-style skeletal input.
            device_index = self.vmt_device_indices[hand_label]
            controller_type = self.VMT_COMPATIBLE_CONTROLLER_TYPES[hand_label]
            message = OSCTransport.build_message(
                "/VMT/Joint/Driver",
                device_index,
                controller_type,
                0.0,
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(qx), float(qy), float(qz), float(qw),
                "HMD"
            )
            self._vmt_osc.send(message)
            self._vmt_pose_packets += 1

            curls = self._normalized_finger_curls(data)
            if self.vmt_skeleton_enabled:
                open_values = {}
                for finger in self.FINGER_NAMES:
                    # VMT interpolates Fist -> OpenHand, while MediaPipe curl is
                    # OpenHand -> Fist. Sending curl directly reverses the hand.
                    open_amount = 1.0 - curls[finger]
                    open_values[finger] = open_amount
                    self._vmt_osc.send(OSCTransport.build_message(
                        "/VMT/Skeleton/Scalar",
                        device_index,
                        self.VMT_FINGER_INDICES[finger],
                        float(open_amount),
                        0,
                        0,
                    ))
                    self._vmt_skeleton_packets += 1
                self._vmt_osc.send(OSCTransport.build_message(
                    "/VMT/Skeleton/Apply", device_index, 0.0
                ))
                self._vmt_skeleton_packets += 1
                self._last_vmt_open_values[hand_label] = open_values

            if self.vmt_input_enabled:
                for finger, input_index in self.VMT_FINGER_INPUT_INDICES.items():
                    self._vmt_osc.send(OSCTransport.build_message(
                        "/VMT/Input/Trigger",
                        device_index,
                        input_index,
                        0.0,
                        float(curls[finger]),
                    ))
                    self._vmt_input_packets += 1

            self._vmt_last_seen_at[hand_label] = now
            self._vmt_neutral_sent[hand_label] = False
            self._vmt_controller_active[hand_label] = True

    @classmethod
    def _normalized_finger_curls(cls, data: Dict) -> Dict[str, float]:
        fingers = data.get("fingers", {}) if isinstance(data, dict) else {}
        curls = data.get("curls", {}) if isinstance(data, dict) else {}
        result = {}
        for finger in cls.FINGER_NAMES:
            if finger in curls:
                value = curls.get(finger)
            elif finger in fingers:
                value = 0.0 if bool(fingers.get(finger)) else 0.8
            else:
                value = 0.0
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            result[finger] = max(0.0, min(1.0, value))
        return result

    def _reset_missing_vmt_hands(
        self,
        hand_data: Dict,
        now: float = None,
        force_reset: bool = False,
    ):
        """Open a lost hand and disconnect it only for explicit shutdown."""
        now = time.monotonic() if now is None else float(now)
        hand_data = hand_data if isinstance(hand_data, dict) else {}
        for hand_label in ("left", "right"):
            if hand_label in hand_data:
                continue
            last_seen = self._vmt_last_seen_at[hand_label]
            timed_out = (
                last_seen is not None
                and now - last_seen >= self.lost_reset_timeout
            )
            if not force_reset and (not timed_out or self._vmt_neutral_sent[hand_label]):
                continue
            disable_controller = force_reset or not self.vmt_keep_controllers_active
            self._send_vmt_neutral(
                hand_label,
                disable_controller=disable_controller,
            )
            self._vmt_last_seen_at[hand_label] = None
            self._vmt_neutral_sent[hand_label] = True

    def _send_vmt_neutral(self, hand_label: str, disable_controller: bool):
        device_index = self.vmt_device_indices[hand_label]
        if self.vmt_skeleton_enabled:
            for finger in self.FINGER_NAMES:
                self._vmt_osc.send(OSCTransport.build_message(
                    "/VMT/Skeleton/Scalar",
                    device_index,
                    self.VMT_FINGER_INDICES[finger],
                    1.0,
                    0,
                    0,
                ))
                self._vmt_skeleton_packets += 1
            self._vmt_osc.send(OSCTransport.build_message(
                "/VMT/Skeleton/Apply", device_index, 0.0
            ))
            self._vmt_skeleton_packets += 1
            self._last_vmt_open_values[hand_label] = {
                finger: 1.0 for finger in self.FINGER_NAMES
            }
        if self.vmt_input_enabled:
            for input_index in self.VMT_FINGER_INPUT_INDICES.values():
                self._vmt_osc.send(OSCTransport.build_message(
                    "/VMT/Input/Trigger",
                    device_index,
                    input_index,
                    0.0,
                    0.0,
                ))
                self._vmt_input_packets += 1
        if disable_controller:
            self._vmt_osc.send(OSCTransport.build_message(
                "/VMT/Joint/Driver",
                device_index,
                0,
                0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
                "HMD",
            ))
            self._vmt_controller_active[hand_label] = False

    def _send_osc_hand_params(
        self,
        hand_data: Dict,
        now: float = None,
        force_reset: bool = False,
    ):
        """Send desktop-safe VRChat gesture parameters and reset lost hands."""
        now = time.monotonic() if now is None else float(now)
        for hand_label in ["left", "right"]:
            data = hand_data.get(hand_label)
            if data is None:
                last_seen = self._last_seen_at[hand_label]
                timed_out = (
                    last_seen is not None
                    and now - last_seen >= self.lost_reset_timeout
                )
                if force_reset or (timed_out and not self._neutral_sent[hand_label]):
                    messages = self._build_neutral_osc_messages(hand_label)
                    self._osc.send_batch(messages)
                    self._desktop_osc_message_count += len(messages)
                    self._last_desktop_send_time = time.time()
                    self._last_desktop_gestures[hand_label] = 0
                    self._neutral_sent[hand_label] = True
                    self._last_seen_at[hand_label] = None
                continue

            self._last_seen_at[hand_label] = now
            self._neutral_sent[hand_label] = False
            curls = self._normalized_finger_curls(data)
            prefix = "Left" if hand_label == "left" else "Right"

            gesture_name = self._resolve_vrchat_gesture(data)
            gesture_value = self.VRCHAT_GESTURE_VALUES.get(gesture_name, 0)
            gesture_weight = 0.0 if gesture_value == 0 else 1.0
            messages = [
                OSCTransport.build_message(
                    f"/avatar/parameters/{parameter_name}", int(gesture_value)
                )
                for parameter_name in self._gesture_parameter_names(hand_label)
            ]
            messages.append(
                OSCTransport.build_message(
                    f"/avatar/parameters/Gesture{prefix}Weight", float(gesture_weight)
                )
            )

            if self.custom_parameters_enabled:
                for i, finger in enumerate(self.FINGER_NAMES):
                    curl = curls[finger]
                    address = f"/avatar/parameters/{prefix}{self.FINGER_CAP[i]}Curl"
                    messages.append(OSCTransport.build_message(address, curl))

                spread = max(0.0, min(1.0, float(data.get("spread", 0.0))))
                messages.append(OSCTransport.build_message(
                    f"/avatar/parameters/{prefix}HandSpread", spread
                ))

                palm_side = data.get("palm_side", "palm")
                palm_conf = max(0.0, min(1.0, float(data.get("palm_confidence", 0.5))))
                palm_value = 0.0 if palm_side == "palm" else 1.0
                palm_value = 0.5 + (palm_value - 0.5) * palm_conf
                messages.append(OSCTransport.build_message(
                    f"/avatar/parameters/{prefix}HandPalmBack", float(palm_value)
                ))

            self._osc.send_batch(messages)
            self._desktop_osc_message_count += len(messages)
            self._last_desktop_send_time = time.time()
            self._last_desktop_gestures[hand_label] = int(gesture_value)

    def _build_neutral_osc_messages(self, hand_label: str) -> List[bytes]:
        prefix = "Left" if hand_label == "left" else "Right"
        messages = [
            OSCTransport.build_message(f"/avatar/parameters/{parameter_name}", 0)
            for parameter_name in self._gesture_parameter_names(hand_label)
        ]
        messages.append(OSCTransport.build_message(
            f"/avatar/parameters/Gesture{prefix}Weight", 0.0
        ))
        if self.custom_parameters_enabled:
            messages.extend(self._build_custom_neutral_osc_messages(hand_label))
        return messages

    def _build_custom_neutral_osc_messages(self, hand_label: str) -> List[bytes]:
        prefix = "Left" if hand_label == "left" else "Right"
        messages = [
            OSCTransport.build_message(
                f"/avatar/parameters/{prefix}{finger_name}Curl", 0.0
            )
            for finger_name in self.FINGER_CAP
        ]
        messages.append(OSCTransport.build_message(
            f"/avatar/parameters/{prefix}HandSpread", 0.0
        ))
        messages.append(OSCTransport.build_message(
            f"/avatar/parameters/{prefix}HandPalmBack", 0.5
        ))
        return messages

    def update_output_config(self, config: Dict):
        """Apply output switches that do not require recreating the detector."""
        old_desktop_enabled = self.desktop_osc_enabled
        old_vmt_enabled = self.vmt_enabled
        old_vmt_local_mode = self.vmt_local_mode
        old_vmt_skeleton_enabled = self.vmt_skeleton_enabled
        old_vmt_input_enabled = self.vmt_input_enabled
        old_vmt_auto_enable_skeleton = self.vmt_auto_enable_skeleton
        old_vmt_register_on_steamvr_startup = (
            self.vmt_register_on_steamvr_startup
        )
        old_vmt_keep_controllers_active = self.vmt_keep_controllers_active
        old_vmt_hand_selection_priority = self.vmt_hand_selection_priority
        old_vmt_device_indices = dict(self.vmt_device_indices)
        old_custom_enabled = self.custom_parameters_enabled
        old_gesture_parameters = {
            side: list(names)
            for side, names in self.desktop_gesture_parameters.items()
        }
        self.desktop_osc_enabled = bool(config.get("desktop_osc_enabled", True))
        self.vmt_enabled = bool(config.get("vmt_enabled", True))
        self.vmt_local_mode = bool(
            config.get("vmt_local_mode", self.vmt_local_mode)
        )
        new_vmt_skeleton_enabled = bool(config.get("vmt_skeleton_enabled", True))
        new_vmt_input_enabled = bool(config.get("vmt_input_enabled", True))
        new_vmt_auto_enable_skeleton = bool(
            config.get("vmt_auto_enable_skeleton", True)
        )
        new_vmt_register_on_steamvr_startup = bool(
            config.get(
                "vmt_register_on_steamvr_startup",
                old_vmt_register_on_steamvr_startup,
            )
        )
        new_vmt_keep_controllers_active = bool(
            config.get(
                "vmt_keep_controllers_active",
                self.vmt_keep_controllers_active,
            )
        )
        new_vmt_hand_selection_priority = (
            self._normalize_vmt_hand_selection_priority(
                config.get(
                    "vmt_hand_selection_priority",
                    old_vmt_hand_selection_priority,
                )
            )
        )
        new_vmt_device_indices = self._normalize_vmt_device_indices(
            config.get("vmt_left_device_index", old_vmt_device_indices["left"]),
            config.get("vmt_right_device_index", old_vmt_device_indices["right"]),
        )
        vmt_runtime_changed = (
            new_vmt_skeleton_enabled != old_vmt_skeleton_enabled
            or new_vmt_input_enabled != old_vmt_input_enabled
            or new_vmt_auto_enable_skeleton != old_vmt_auto_enable_skeleton
            or new_vmt_register_on_steamvr_startup
            != old_vmt_register_on_steamvr_startup
            or new_vmt_hand_selection_priority
            != old_vmt_hand_selection_priority
            or new_vmt_device_indices != old_vmt_device_indices
        )
        if (
            new_vmt_hand_selection_priority != old_vmt_hand_selection_priority
            and self._vmt_runtime_status.get("ready")
            and self._vmt_pre_registered
        ):
            self._vmt_priority_restart_required = True
        if (
            new_vmt_register_on_steamvr_startup
            != old_vmt_register_on_steamvr_startup
            and self._vmt_runtime_status.get("ready")
        ):
            self._vmt_startup_registration_restart_required = True
        if (
            old_vmt_enabled
            and self._vmt_osc.socket
            and self._vmt_initialized
            and (not self.vmt_enabled or vmt_runtime_changed)
        ):
            self._reset_missing_vmt_hands({}, force_reset=True)
        self.vmt_skeleton_enabled = new_vmt_skeleton_enabled
        self.vmt_input_enabled = new_vmt_input_enabled
        self.vmt_auto_enable_skeleton = new_vmt_auto_enable_skeleton
        self.vmt_register_on_steamvr_startup = (
            new_vmt_register_on_steamvr_startup
        )
        self.vmt_keep_controllers_active = new_vmt_keep_controllers_active
        self.vmt_hand_selection_priority = new_vmt_hand_selection_priority
        self.vmt_device_indices = new_vmt_device_indices
        if (
            old_vmt_keep_controllers_active
            and not self.vmt_keep_controllers_active
            and self._vmt_osc.socket
            and self._vmt_initialized
        ):
            for hand_label in ("left", "right"):
                if (
                    self._vmt_controller_active[hand_label]
                    and self._vmt_neutral_sent[hand_label]
                ):
                    self._send_vmt_neutral(hand_label, disable_controller=True)
        new_custom_enabled = bool(config.get("custom_parameters_enabled", True))
        if old_custom_enabled and not new_custom_enabled and self._osc.socket:
            for hand_label in ("left", "right"):
                self._osc.send_batch(self._build_custom_neutral_osc_messages(hand_label))
        self.custom_parameters_enabled = new_custom_enabled
        self.desktop_gesture_parameters = {
            "left": self._normalize_parameter_names(
                config.get("desktop_left_gesture_parameters"),
                old_gesture_parameters["left"],
            ),
            "right": self._normalize_parameter_names(
                config.get("desktop_right_gesture_parameters"),
                old_gesture_parameters["right"],
            ),
        }
        if self._osc.socket:
            for hand_label in ("left", "right"):
                stale_names = set(old_gesture_parameters[hand_label]) - set(
                    self.desktop_gesture_parameters[hand_label]
                )
                for parameter_name in stale_names:
                    self._osc.send(OSCTransport.build_message(
                        f"/avatar/parameters/{parameter_name}", 0
                    ))
        try:
            timeout = float(config.get("lost_reset_timeout", self.lost_reset_timeout))
        except (TypeError, ValueError):
            timeout = self.lost_reset_timeout
        self.lost_reset_timeout = max(0.0, timeout)

        new_osc_host = str(config.get("host", self._osc.host) or self._osc.host)
        new_osc_port = int(config.get("port", self._osc.port) or self._osc.port)
        configured_vmt_host = str(
            config.get("vmt_host", self._vmt_osc.host) or self._vmt_osc.host
        )
        new_vmt_host = "127.0.0.1" if self.vmt_local_mode else configured_vmt_host
        new_vmt_port = int(config.get("vmt_port", self._vmt_osc.port) or self._vmt_osc.port)
        osc_endpoint_changed = (new_osc_host, new_osc_port) != (self._osc.host, self._osc.port)
        vmt_endpoint_changed = (new_vmt_host, new_vmt_port) != (
            self._vmt_osc.host,
            self._vmt_osc.port,
        )
        if self.vmt_local_mode != old_vmt_local_mode:
            vmt_endpoint_changed = True

        if osc_endpoint_changed or (old_desktop_enabled and not self.desktop_osc_enabled):
            if self._osc.socket and old_desktop_enabled:
                self._send_osc_hand_params({}, force_reset=True)
            self._osc.disconnect()
        self._osc.host = new_osc_host
        self._osc.port = new_osc_port
        if self.connected and self.desktop_osc_enabled and (
            osc_endpoint_changed or not old_desktop_enabled or self._osc.socket is None
        ):
            self._osc.connect()

        if vmt_endpoint_changed or (old_vmt_enabled and not self.vmt_enabled):
            self._vmt_osc.disconnect()
            self._vmt_initialized = False
        self._vmt_osc.host = new_vmt_host
        self._vmt_osc.port = new_vmt_port
        if vmt_endpoint_changed:
            self._last_vmt_runtime_check = 0.0
            self._vmt_runtime_status = {}
            self._vmt_last_ready_state = None
        if self.connected and self.vmt_enabled and (
            vmt_endpoint_changed or not old_vmt_enabled or self._vmt_osc.socket is None
        ):
            if self._vmt_osc.connect():
                self._vmt_init()
        elif (
            self.connected
            and self.vmt_enabled
            and self._vmt_osc.socket
            and vmt_runtime_changed
        ):
            self._vmt_init()

    def get_status(self) -> Dict:
        if self.vmt_enabled:
            self._refresh_vmt_runtime_status()
        status = super().get_status()
        status.update({
            "desktop_osc_enabled": self.desktop_osc_enabled,
            "vmt_enabled": self.vmt_enabled,
            "vmt_local_mode": self.vmt_local_mode,
            "vmt_skeleton_enabled": self.vmt_skeleton_enabled,
            "vmt_input_enabled": self.vmt_input_enabled,
            "vmt_auto_enable_skeleton": self.vmt_auto_enable_skeleton,
            "vmt_register_on_steamvr_startup": (
                self.vmt_register_on_steamvr_startup
            ),
            "vmt_driver_register_on_steamvr_startup": (
                self._vmt_runtime_status.get(
                    "add_compatible_controller_on_startup"
                )
            ),
            "vmt_keep_controllers_active": self.vmt_keep_controllers_active,
            "vmt_hand_selection_priority": self.vmt_hand_selection_priority,
            "vmt_driver_hand_selection_priority": self._vmt_runtime_status.get(
                "hand_selection_priority"
            ),
            "vmt_priority_restart_required": self._vmt_priority_restart_required,
            "vmt_startup_registration_restart_required": (
                self._vmt_startup_registration_restart_required
            ),
            "vmt_restart_required": bool(
                self._vmt_priority_restart_required
                or self._vmt_startup_registration_restart_required
            ),
            "vmt_controller_active": dict(self._vmt_controller_active),
            "vmt_device_indices": dict(self.vmt_device_indices),
            "custom_parameters_enabled": self.custom_parameters_enabled,
            "vmt_initialized": self._vmt_initialized,
            "vmt_pre_registered": self._vmt_pre_registered,
            "vmt_last_ready_state": self._vmt_last_ready_state,
            "vmt_reinit_count": self._vmt_reinit_count,
            "vmt_last_reinit_time": self._vmt_last_reinit_time,
            "vmt_registration_state": (
                "registration_sent"
                if self._vmt_runtime_status.get("ready")
                and self._vmt_pre_registered
                else (
                    "pending_endpoint"
                    if self._vmt_pre_registered
                    else "inactive"
                )
            ),
            "vmt_device_presence_confirmed": False,
            "vmt_vrchat_consumption_confirmed": False,
            "desktop_gesture_parameters": self.desktop_gesture_parameters,
            "desktop_osc_host": self._osc.host,
            "desktop_osc_port": self._osc.port,
            "desktop_osc_message_count": self._desktop_osc_message_count,
            "last_desktop_send_time": self._last_desktop_send_time,
            "last_desktop_gestures": dict(self._last_desktop_gestures),
            "detected_hand_count": self._last_hand_count,
            "last_pose_source": self._last_pose_source,
            "vmt_pose_packets": self._vmt_pose_packets,
            "vmt_skeleton_packets": self._vmt_skeleton_packets,
            "vmt_input_packets": self._vmt_input_packets,
            "last_vmt_open_values": {
                side: dict(values)
                for side, values in self._last_vmt_open_values.items()
            },
            "last_vmt_send_time": self._last_vmt_send_time,
            "vmt_protocol": "VMT compatible controller + OpenVR skeleton",
            "vmt_host": self._vmt_osc.host,
            "vmt_port": self._vmt_osc.port,
            "vmt_endpoint_ready": self._vmt_runtime_status.get("ready", False),
            "vmt_driver_installed": self._vmt_runtime_status.get(
                "driver_installed", False
            ),
            "vmt_port_listening": self._vmt_runtime_status.get(
                "udp_port_listening"
            ),
            "vmt_runtime_message": self._vmt_runtime_status.get("message"),
        })
        return status

    def _resolve_vrchat_gesture(self, hand: Dict) -> str:
        """Map MediaPipe finger state to VRChat's built-in gesture index names.

        VRChat desktop avatars commonly react to GestureLeft/GestureRight, while
        custom curl parameters only work on avatars that explicitly bind them.
        """
        gesture = str(hand.get("gesture") or "").strip()
        if gesture in self.VRCHAT_GESTURE_VALUES:
            return gesture
        if gesture == "pinky":
            return "rock"

        fingers = hand.get("fingers") or {}
        if fingers:
            thumb = bool(fingers.get("thumb", False))
            index = bool(fingers.get("index", False))
            middle = bool(fingers.get("middle", False))
            ring = bool(fingers.get("ring", False))
            pinky = bool(fingers.get("pinky", False))
            if thumb and index and middle and ring and pinky:
                return "open"
            if not thumb and not index and not middle and not ring and not pinky:
                return "fist"
            if index and not middle and not ring and not pinky:
                return "point"
            if index and middle and not ring and not pinky:
                return "peace"
            if thumb and not index and not middle and not ring and not pinky:
                return "thumb_up"
            if index and pinky and not middle and not ring:
                return "rock" if not thumb else "call"
            if pinky and not index and not middle and not ring:
                return "rock"

        curls = hand.get("curls") or {}
        if curls:
            extended = {}
            for finger in self.FINGER_NAMES:
                try:
                    extended[finger] = float(curls.get(finger, 1.0)) < 0.45
                except (TypeError, ValueError):
                    extended[finger] = False
            return self._resolve_vrchat_gesture({"fingers": extended})

        return "neutral"
