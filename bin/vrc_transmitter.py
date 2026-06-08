import socket
import json
import struct
import threading
import time
from typing import Dict, Optional
from queue import Queue, Empty


class VRCTransmitter:
    """VRC 数据传输器

    支持四种协议:
    - osc: VRChat 原生 OSC 追踪协议（推荐）
    - vmc: Virtual Motion Capture 协议
    - tcp: 自定义 TCP 协议
    - steamvr: SteamVR 命名管道驱动协议
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, protocol: str = "osc"):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.is_connected = False
        self.is_running = False
        self.send_queue = Queue(maxsize=10)
        self.send_thread = None
        self.last_sent_pose = None
        self.fps_counter = 0
        self.fps = 0
        self.fps_start_time = time.time()
        self.reconnect_interval = 3.0
        self.last_connect_attempt = 0

        # HMD 位姿数据提供者 (用于获取真实头显坐标)
        self._hmd_provider = None

        # OSC 相关
        self._osc_socket = None

        # TCP 相关
        self._tcp_socket = None

        # VMT 手部追踪相关 (参考 ExVR hand_sender)
        self._vmt_socket = None
        self._vmt_port = 39570
        self._vmt_initialized = False

        # VRto3D HMD 数据传输相关 (参考 ExVR sender)
        self._vrto3d_socket = None
        self._vrto3d_port = 4242
        self._vrto3d_prev_pos = None
        self._vrto3d_curr_pos = None

        # SteamVR 相关
        self._steamvr_pipe_name = r'\\.\pipe\ApriltagPipeIn'
        self._steamvr_trackers_registered = False
        self._steamvr_smoothing = 50
        self._steamvr_additional_smoothing = 0.5
        self._steamvr_camera_latency = 0
        self._steamvr_hmd_to_neck_offset = [0, -0.2, 0.1]
        self._steamvr_calibrated = False
        self._steamvr_pose_scale = 1.0

        # 追踪器配置: 每个追踪器指定部位和 SteamVR 角色
        # 默认8个追踪器，与 OSC tracker_toggles 对齐
        # BridgeDriver 使用 std::vector 动态管理，无硬性数量限制
        self._steamvr_trackers = [
            {"body_part": "Hips", "role": "TrackerRole_Waist"},
            {"body_part": "Chest", "role": "TrackerRole_Chest"},
            {"body_part": "LeftFoot", "role": "TrackerRole_LeftFoot"},
            {"body_part": "RightFoot", "role": "TrackerRole_RightFoot"},
            {"body_part": "LeftKnee", "role": "TrackerRole_LeftKnee"},
            {"body_part": "RightKnee", "role": "TrackerRole_RightKnee"},
            {"body_part": "LeftElbow", "role": "TrackerRole_LeftElbow"},
            {"body_part": "RightElbow", "role": "TrackerRole_RightElbow"},
        ]

        # ===== 追踪器传输开关 =====
        # 控制每个身体部位是否发送数据到 VRChat
        # 默认全部开启，可通过 set_tracker_toggles() 修改
        self._tracker_toggles = {
            "Hips": True,
            "LeftFoot": True,
            "RightFoot": True,
            "Chest": True,
            "LeftLowerLeg": True,
            "RightLowerLeg": True,
            "LeftLowerArm": True,
            "RightLowerArm": True,
        }

        # 位置/旋转发送总开关
        self._send_position = True
        self._send_rotation = True

        # 头部坐标推流开关 (独立于 send_position)
        # 控制是否发送 /tracking/trackers/head/position
        # 佩戴头显时建议关闭，避免干扰 VR 头显定位
        # 不佩戴头显时可以开启，提供头部位置追踪
        self._send_head_position = False

    def connect(self) -> bool:
        if self.protocol == "osc":
            return self._connect_osc()
        elif self.protocol == "vmc":
            return self._connect_osc()  # VMC 也基于 UDP/OSC
        elif self.protocol == "tcp":
            return self._connect_tcp()
        elif self.protocol == "steamvr":
            return self._connect_steamvr()
        return False

    def _connect_osc(self) -> bool:
        try:
            if self._osc_socket:
                self._osc_socket.close()

            self._osc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._osc_socket.settimeout(0.5)

            # 初始化 VMT 手部追踪 socket (参考 ExVR hand_sender, 端口 39570)
            try:
                if self._vmt_socket:
                    self._vmt_socket.close()
                self._vmt_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._vmt_socket.settimeout(0.5)
                # VMT 初始化: 设置房间矩阵 (参考 ExVR)
                self._vmt_init()
                print(f"VMT hand tracking socket initialized (port {self._vmt_port})")
            except Exception as e:
                print(f"VMT socket warning: {e}")

            # 初始化 VRto3D HMD 数据 socket (参考 ExVR sender, 端口 4242)
            try:
                if self._vrto3d_socket:
                    self._vrto3d_socket.close()
                self._vrto3d_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._vrto3d_socket.settimeout(0.5)
                print(f"VRto3D HMD socket initialized (port {self._vrto3d_port})")
            except Exception as e:
                print(f"VRto3D socket warning: {e}")

            self.is_connected = True
            self.is_running = True

            print(f"OSC connected to {self.host}:{self.port}")

            self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self.send_thread.start()

            return True
        except Exception as e:
            print(f"OSC connection error: {e}")
            self.is_connected = False
            return False

    def _connect_tcp(self) -> bool:
        try:
            if self._tcp_socket:
                self._tcp_socket.close()

            self._tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._tcp_socket.settimeout(1.0)
            self._tcp_socket.connect((self.host, self.port))

            self.is_connected = True
            self.is_running = True

            print(f"TCP connected to {self.host}:{self.port}")

            self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self.send_thread.start()

            return True
        except Exception as e:
            print(f"TCP connection error: {e}")
            self.is_connected = False
            return False

    # ========== SteamVR 命名管道协议 ==========

    def _send_to_steamvr_pipe(self, text: str, num_tries: int = 10, wait_time: float = 0.1):
        """通过命名管道发送命令到 SteamVR 驱动

        Args:
            text: 要发送的命令字符串
            num_tries: 重试次数
            wait_time: 重试间隔

        Returns:
            驱动返回的字符串数组，失败返回 None
        """
        import sys
        for attempt in range(num_tries):
            try:
                if sys.platform == 'win32':
                    pipe = open(self._steamvr_pipe_name, 'rb+', buffering=0)
                    pipe.write(str.encode(text) + b'\x00')
                    resp = pipe.read(1024)
                    pipe.close()
                else:
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
                    print(f"SteamVR pipe retry {attempt+1}/{num_tries}: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"ERROR: Could not connect to SteamVR after {num_tries} tries")
                    return None

    def _connect_steamvr(self) -> bool:
        """连接 SteamVR 驱动并注册追踪器"""
        print("Connecting to SteamVR...")

        # 查询已注册的追踪器数量
        result = self._send_to_steamvr_pipe("numtrackers")
        if result is None:
            print("ERROR: Could not connect to SteamVR! Make sure SteamVR is running and the driver is installed.")
            self.is_connected = False
            return False

        # 解析追踪器数量: 返回格式 "numtrackers <count>" 或 "numtrackers <count> <version>"
        # 注意: 管道返回可能带前导空格，split后第一个元素可能是空字符串
        try:
            num_trackers = 0
            for part in result:
                part_clean = part.strip().rstrip('\x00')
                if part_clean.isdigit():
                    num_trackers = int(part_clean)
                    break
        except (ValueError, IndexError):
            print(f"WARNING: Could not parse numtrackers response: {result}, assuming 0")
            num_trackers = 0

        total_trackers = len(self._steamvr_trackers)
        print(f"  SteamVR already has {num_trackers} trackers, need {total_trackers}")

        # 如果已有追踪器数量与需求不同，需要重新注册
        # 先删除多余的追踪器
        if num_trackers > total_trackers:
            for i in range(num_trackers - 1, total_trackers - 1, -1):
                self._send_to_steamvr_pipe(f"removetracker {i}", num_tries=1, wait_time=0.05)

        # 注册缺少的追踪器
        for i in range(num_trackers, total_trackers):
            tracker = self._steamvr_trackers[i]
            role = tracker.get("role", "TrackerRole_Waist")
            resp = self._send_to_steamvr_pipe(f"addtracker MPTracker{i} {role}")
            if resp is None:
                print(f"ERROR: Failed to add tracker {i} ({role}) to SteamVR")
                self.is_connected = False
                return False
            print(f"  Registered tracker {i}: {tracker['body_part']} -> {role}")

        # 设置平滑参数
        resp = self._send_to_steamvr_pipe(
            f"settings {self._steamvr_smoothing} "
            f"{self._steamvr_additional_smoothing} "
            f"{self._steamvr_additional_smoothing}"
        )
        if resp is None:
            print("ERROR: Failed to set SteamVR settings")
            self.is_connected = False
            return False

        self._steamvr_trackers_registered = True
        self.is_connected = True
        self.is_running = True
        self._steamvr_calibrated = False

        # SteamVR 模式下也需要 OSC socket 用于面部/手部参数发送
        # (SteamVR 管道只支持追踪器位姿，面部 blendshape 和手指 curl 必须通过 OSC)
        if not self._osc_socket:
            try:
                self._osc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._osc_socket.settimeout(0.5)
                print(f"  Auxiliary OSC socket initialized for face/hand data ({self.host}:{self.port})")
            except Exception as e:
                print(f"  WARNING: Auxiliary OSC socket failed: {e}")

        tracker_info = ", ".join(f"{t['body_part']}({t['role']})" for t in self._steamvr_trackers)
        print(f"SteamVR connected! Trackers: {total_trackers} [{tracker_info}]")

        self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self.send_thread.start()

        return True

    def _send_steamvr(self, pose_data: Dict):
        """通过 SteamVR 命名管道发送姿态数据

        根据 _steamvr_trackers 配置动态发送每个追踪器的位置和旋转。
        """
        try:
            from scipy.spatial.transform import Rotation as R
        except ImportError:
            return

        import numpy as np

        # 获取 HMD 数据用于对齐
        # 优先使用 HMD Provider (OpenVR/OSC/Pico/Quest)，管道获取作为回退
        hmd_from_provider = False
        if self._hmd_provider is not None:
            hmd_pose = self._hmd_provider.get_pose()
            if hmd_pose.valid:
                # HMD Provider 输出 VRChat 坐标系 (Z前)，SteamVR 需要 OpenVR 坐标系 (Z后)
                # 位置: Z 翻转
                hmd_pos_vrchat = hmd_pose.position
                headsetpos = [hmd_pos_vrchat[0], hmd_pos_vrchat[1], -hmd_pos_vrchat[2]]
                # 四元数: VRChat -> OpenVR，绕 Y 轴旋转 180 度
                quat = hmd_pose.quaternion  # [w, x, y, z]
                try:
                    rot_vrchat = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy: [x,y,z,w]
                    mat_rot = rot_vrchat.as_matrix()
                    flip_y = np.diag([-1.0, 1.0, -1.0])
                    mat_openvr = flip_y @ mat_rot
                    headsetrot = R.from_matrix(mat_openvr)
                except Exception:
                    headsetrot = R.identity()
                neckoffset = headsetrot.apply(self._steamvr_hmd_to_neck_offset)
                hmd_from_provider = True

        if not hmd_from_provider:
            # 回退: 通过 SteamVR 管道获取 HMD 位姿
            array = self._send_to_steamvr_pipe("getdevicepose 0", num_tries=1, wait_time=0)
            if array is None or len(array) < 10:
                # HMD 不在线时使用默认位置（站立姿态，高度1.7m）
                if not self._steamvr_calibrated:
                    print("WARNING: HMD not detected, using default position for alignment")
                    self._steamvr_calibrated = True
                headsetpos = [0.0, 1.7, 0.0]
                headsetrot = R.identity()
                neckoffset = headsetrot.apply(self._steamvr_hmd_to_neck_offset)
            else:
                try:
                    headsetpos = [float(array[3]), float(array[4]), float(array[5])]
                    headsetrot = R.from_quat([float(array[7]), float(array[8]),
                                               float(array[9]), float(array[6])])
                    neckoffset = headsetrot.apply(self._steamvr_hmd_to_neck_offset)
                except (ValueError, IndexError):
                    headsetpos = [0.0, 1.7, 0.0]
                    headsetrot = R.identity()
                    neckoffset = headsetrot.apply(self._steamvr_hmd_to_neck_offset)

        # 从 pose_data 提取 3D 关键点
        pose_3d = pose_data.get("pose_3d")
        body_only = pose_data.get("body_only", False)

        # 无身体数据时，仅发送面部和手部 (通过 VMT/OSC)
        if pose_3d is None and not body_only:
            return

        if pose_3d is None:
            # body_only 模式: 跳过身体追踪器，直接发送面部/手部
            face_data = pose_data.get("face")
            hand_data = pose_data.get("hands")
            if face_data:
                self._send_osc_face_params(face_data)
            if hand_data:
                self._send_osc_hand_params(hand_data)
            return

        if not self._steamvr_calibrated:
            self._steamvr_calibrated = True

        # ===== 坐标系变换 =====
        # pose_3d: Y向下, 髋部为原点, 米制单位, X负=左
        # SteamVR: Y向上, X负=左
        # 变换: 翻转 X 和 Y 轴 (与原始 Mediapipe-VR-Fullbody-Tracking 一致)
        # 原始项目: pose3d[:,0] = -pose3d[:,0]; pose3d[:,1] = -pose3d[:,1]
        steamvr_pose = {}
        for name, data in pose_3d.items():
            data = data if isinstance(data, list) else [0, 0, 0]
            steamvr_pose[name] = [-float(data[0]), -float(data[1]), float(data[2])]

        # 计算骨骼中心点
        left_hip = steamvr_pose.get("LeftUpperLeg", [0, 0, 0])
        right_hip = steamvr_pose.get("RightUpperLeg", [0, 0, 0])
        left_shoulder = steamvr_pose.get("LeftUpperArm", [0, 0, 0])
        right_shoulder = steamvr_pose.get("RightUpperArm", [0, 0, 0])
        hip_center = [(left_hip[i] + right_hip[i]) / 2 for i in range(3)]
        shoulder_center = [(left_shoulder[i] + right_shoulder[i]) / 2 for i in range(3)]

        # 部位名 -> 3D 位置映射 (已变换到 SteamVR 坐标系)
        body_part_positions = {
            "Hips": hip_center,
            "Chest": [(shoulder_center[i] * 0.6 + hip_center[i] * 0.4) for i in range(3)],
            "LeftFoot": steamvr_pose.get("LeftFoot", [0, 0, 0]),
            "RightFoot": steamvr_pose.get("RightFoot", [0, 0, 0]),
            "LeftKnee": steamvr_pose.get("LeftLowerLeg", [0, 0, 0]),
            "RightKnee": steamvr_pose.get("RightLowerLeg", [0, 0, 0]),
            "LeftHand": steamvr_pose.get("LeftHand", [0, 0, 0]),
            "RightHand": steamvr_pose.get("RightHand", [0, 0, 0]),
            "LeftElbow": steamvr_pose.get("LeftLowerArm", [0, 0, 0]),
            "RightElbow": steamvr_pose.get("RightLowerArm", [0, 0, 0]),
            "Head": steamvr_pose.get("Head", [0, 0, 0]),
        }

        # 计算偏移量 (参考原始项目: offset = pose3d[7] - (headsetpos + neckoffset))
        # pose3d[7] 在原始项目中是 neck 点 (肩中点到头部的中间点)
        # 在我们的项目中对应 "Neck" 骨骼
        neck_pos_steamvr = steamvr_pose.get("Neck", steamvr_pose.get("UpperChest", shoulder_center))
        offset = np.array(neck_pos_steamvr) - (np.array(headsetpos) + neckoffset)

        # 获取旋转数据
        rotations = pose_data.get("rotations", {})

        def euler_to_quat(euler_rad):
            """将 XYZ 欧拉角(弧度, Y向下) 转为 SteamVR 四元数(Y向上)

            坐标系变换: 翻转 X 和 Y 轴 (与原始项目一致)
            flip_mat = diag(-1, -1, 1)
            """
            if euler_rad == [0, 0, 0]:
                return [0, 0, 0, 1]
            try:
                rot = R.from_euler('xyz', euler_rad, degrees=False)
                mat = rot.as_matrix()
                _flip = np.diag([-1.0, -1.0, 1.0])
                mat_new = _flip @ mat @ _flip
                rot_new = R.from_matrix(mat_new)
                return rot_new.as_quat()
            except Exception:
                return [0, 0, 0, 1]

        latency = self._steamvr_camera_latency

        # SteamVR body_part 名称 -> tracker_toggles 中的骨骼名称映射
        # 用于检查该追踪器是否被用户关闭
        steamvr_to_toggle_bone = {
            "Hips": "Hips",
            "Chest": "Chest",
            "LeftFoot": "LeftFoot",
            "RightFoot": "RightFoot",
            "LeftKnee": "LeftLowerLeg",
            "RightKnee": "RightLowerLeg",
            "LeftHand": "LeftLowerArm",
            "RightHand": "RightLowerArm",
            "LeftElbow": "LeftLowerArm",
            "RightElbow": "RightLowerArm",
            "Head": "Head",
        }

        # SteamVR body_part 名称 -> rotations 字典中的骨骼名称映射
        steamvr_to_bone = {
            "Hips": "Hips",
            "Chest": "Chest",
            "LeftFoot": "LeftFoot",
            "RightFoot": "RightFoot",
            "LeftKnee": "LeftLowerLeg",
            "RightKnee": "RightLowerLeg",
            "LeftHand": "LeftHand",
            "RightHand": "RightHand",
            "LeftElbow": "LeftLowerArm",
            "RightElbow": "RightLowerArm",
            "Head": "Head",
        }

        # 遍历每个追踪器配置，发送姿态
        for idx, tracker in enumerate(self._steamvr_trackers):
            body_part = tracker.get("body_part", "Hips")

            # 检查该追踪器对应的部位是否被关闭 (与 OSC tracker_toggles 同步)
            toggle_bone = steamvr_to_toggle_bone.get(body_part, body_part)
            if not self._tracker_toggles.get(toggle_bone, True):
                continue

            pos_3d = body_part_positions.get(body_part, [0, 0, 0])
            pos = np.array(pos_3d) - offset

            # 旋转: 通过映射查找对应的骨骼旋转
            bone_name = steamvr_to_bone.get(body_part, body_part)
            rot_euler = rotations.get(bone_name, rotations.get("Hips", [0, 0, 0]))
            quat = euler_to_quat(rot_euler)

            self._send_to_steamvr_pipe(
                f"updatepose {idx} {pos[0]} {pos[1]} {pos[2]} "
                f"{quat[3]} {quat[0]} {quat[1]} {quat[2]} {latency} 0.8",
                num_tries=1, wait_time=0
            )

    def set_steamvr_params(self, smoothing: int = 50, additional_smoothing: float = 0.5,
                           camera_latency: float = 0, hmd_to_neck_offset: list = None,
                           trackers: list = None):
        """设置 SteamVR 参数

        Args:
            smoothing: 平滑参数 (1-100)
            additional_smoothing: 额外平滑系数 (0-1)
            camera_latency: 摄像头延迟 (秒)
            hmd_to_neck_offset: HMD 到颈部的偏移 [x, y, z]
            trackers: 追踪器配置列表, 每项为 {"body_part": str, "role": str}
        """
        self._steamvr_smoothing = smoothing
        self._steamvr_additional_smoothing = additional_smoothing
        self._steamvr_camera_latency = camera_latency
        if hmd_to_neck_offset:
            self._steamvr_hmd_to_neck_offset = hmd_to_neck_offset
        if trackers is not None:
            self._steamvr_trackers = trackers

        # 如果已连接，更新驱动设置
        if self.is_connected and self.protocol == "steamvr":
            self._send_to_steamvr_pipe(
                f"settings {smoothing} {additional_smoothing} {additional_smoothing}"
            )

    def recalibrate_steamvr(self):
        """重新校准 SteamVR 追踪器"""
        self._steamvr_calibrated = False

    def set_tracker_toggles(self, toggles: dict = None,
                            send_position: bool = True,
                            send_rotation: bool = True,
                            send_head_position: bool = False):
        """设置追踪器传输开关

        Args:
            toggles: 各身体部位开关, 如 {"Hips": True, "LeftFoot": False, ...}
            send_position: 是否发送位置数据 (关闭后只发旋转，避免干扰VR头显定位)
            send_rotation: 是否发送旋转数据
            send_head_position: 是否发送头部位置数据 (独立开关，佩戴头显时建议关闭)
        """
        if toggles is not None:
            self._tracker_toggles.update(toggles)
        self._send_position = send_position
        self._send_rotation = send_rotation
        self._send_head_position = send_head_position

    def set_hmd_provider(self, provider):
        """设置 HMD 位姿数据提供者

        Args:
            provider: HMDDataProvider 实例，用于获取真实头显坐标
        """
        self._hmd_provider = provider

    def disconnect(self):
        self.is_running = False

        if self.send_thread:
            self.send_thread.join(timeout=2.0)

        for sock in [self._osc_socket, self._tcp_socket, self._vmt_socket, self._vrto3d_socket]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

        self._osc_socket = None
        self._tcp_socket = None
        self._vmt_socket = None
        self._vrto3d_socket = None
        self.is_connected = False
        print("Disconnected from VRC")

    def _send_loop(self):
        while self.is_running:
            try:
                try:
                    pose_data = self.send_queue.get(timeout=0.1)
                    self._send_data(pose_data)
                except Empty:
                    continue
            except Exception as e:
                print(f"Send error: {e}")

    def _send_data(self, pose_data: Dict):
        if not self.is_connected:
            now = time.time()
            if now - self.last_connect_attempt >= self.reconnect_interval:
                self.last_connect_attempt = now
                if not self.connect():
                    return

        try:
            if self.protocol == "osc":
                self._send_osc(pose_data)
            elif self.protocol == "vmc":
                self._send_vmc(pose_data)
            elif self.protocol == "tcp":
                self._send_tcp(pose_data)
            elif self.protocol == "steamvr":
                self._send_steamvr(pose_data)

            self.fps_counter += 1
            now = time.time()
            if now - self.fps_start_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = now

            self.last_sent_pose = pose_data

        except Exception as e:
            if self.protocol == "tcp":
                self.is_connected = False

    # ========== OSC 协议 ==========

    @staticmethod
    def _encode_osc_string(s: str) -> bytes:
        """编码 OSC 字符串（4字节对齐）"""
        encoded = s.encode("utf-8") + b'\x00'
        padding = (4 - len(encoded) % 4) % 4
        return encoded + b'\x00' * padding

    @staticmethod
    def _encode_osc_float(value: float) -> bytes:
        """编码 OSC 浮点数"""
        return struct.pack("!f", value)

    @staticmethod
    def _encode_osc_int(value: int) -> bytes:
        """编码 OSC 整数"""
        return struct.pack("!i", value)

    def _build_osc_message(self, address: str, *args) -> bytes:
        """构建 OSC 消息"""
        type_tags = ","
        data = b""

        for arg in args:
            if isinstance(arg, float):
                type_tags += "f"
                data += self._encode_osc_float(arg)
            elif isinstance(arg, int):
                type_tags += "i"
                data += self._encode_osc_int(arg)
            elif isinstance(arg, str):
                type_tags += "s"
                data += self._encode_osc_string(arg)

        message = self._encode_osc_string(address)
        message += self._encode_osc_string(type_tags)
        message += data

        return message

    def _send_osc(self, pose_data: Dict):
        """通过 OSC 发送骨骼旋转和位置到 VRChat

        坐标系变换:
        1. pose_3d 使用图像坐标系 (X左, Y向下, Z前, 髋部为原点, 米制单位)
           pose_estimator 中 flip_horizontal 时已翻转 X, 使 X负=左
        2. 位置变换: 只翻转 Y 轴 (Y下→Y上), X 和 Z 不变
        3. 旋转变换: flip_mat @ rot_matrix @ flip_mat (flip_mat=diag(1,-1,1))
           然后转为 ZXY 欧拉角(度) 用于 VRChat OSC

        VRChat OSC 追踪协议:
        - /tracking/trackers/{id}/position  -> x, y, z (世界空间, 1.0=1米, Y轴向上)
        - /tracking/trackers/{id}/rotation  -> x, y, z (欧拉角, 度, ZXY顺序)
        - /tracking/trackers/head/position  -> 头部位置
        - /tracking/trackers/head/rotation  -> 头部旋转

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
        if not self._osc_socket:
            return

        # 检查是否只有 face/hands 数据（无身体追踪数据）
        body_only = pose_data.get("body_only", False)

        pose_3d = pose_data.get("pose_3d", {})
        rotations = pose_data.get("rotations", {})

        import numpy as np
        from scipy.spatial.transform import Rotation as Rot

        # ===== 坐标系变换 =====
        # pose_3d: 摄像头坐标系 (X左, Y下, Z前), 髋部为原点, 米制单位
        # Unity/VRChat: (X右, Y上, Z前)
        # 变换: 翻转 X 和 Y 轴 (与原始 Mediapipe-VR-Fullbody-Tracking 一致)
        # 原始项目: pose3d[:,0] = -pose3d[:,0]; pose3d[:,1] = -pose3d[:,1]
        unity_pose = {}
        for bone, pos in pose_3d.items():
            unity_pose[bone] = [-float(pos[0]), -float(pos[1]), float(pos[2])]

        # ===== 缩放调整 =====
        # 参考 Mediapipe-VR-Fullbody-Tracking: pose3d = pose3d * posescale
        pose_scale = 1.0
        for bone in unity_pose:
            unity_pose[bone] = [v * pose_scale for v in unity_pose[bone]]

        # ===== 相对位置计算 (参考原始项目) =====
        # 原始项目 SteamVR 模式: offset = pose3d[7] - (headsetpos + neckoffset)
        #   其中 headsetpos 来自 HMD 真实位置 (通过 SteamVR 驱动获取)
        # 原始项目 OSC 模式: headsetpos = [0,0,0], 所以 offset = pose3d[7] - neckoffset
        #
        # 改进: 通过 HMD 数据提供者获取真实头显坐标
        #
        # OSC 模式坐标对齐策略:
        # VRChat OSC 协议中，/tracking/trackers/head/position 代表 VR 头显位置，
        # VRChat 会将虚拟形象头部对齐到该位置。追踪器位置是相对于头部的偏移。
        #
        # 因此:
        # - 头部位置始终发送 [0,0,0] (VRChat 用头显硬件定位)
        # - 追踪器位置 = 动捕骨骼相对颈部的偏移 (不含 hmd_pos)
        # - offset = neck_pos - neck_offset (与原始 OSC 模式一致)
        # - HMD 数据仅用于 SteamVR 模式下的绝对坐标定位
        #
        # 当 HMD 数据有效时，记录 hmd_pos 供 SteamVR 模式使用
        neck_pos = unity_pose.get("Neck", [0, 1.5, 0])
        neck_offset = self._steamvr_hmd_to_neck_offset  # [0, -0.2, 0.1]

        # 获取 HMD 真实位姿 (供 SteamVR 模式和 VRto3D 使用)
        hmd_pos = [0.0, 0.0, 0.0]  # 默认: 无 HMD 数据
        if self._hmd_provider is not None:
            hmd_pose = self._hmd_provider.get_pose()
            if hmd_pose.valid:
                hmd_pos = hmd_pose.position

        # OSC 模式: offset = neck_pos - neck_offset (不含 hmd_pos)
        # 追踪器位置 = 骨骼位置 - offset = 骨骼 - neck + neck_offset
        # 即: 追踪器相对于颈部的偏移 + neck_offset (与原始项目 OSC 模式一致)
        offset = [neck_pos[0] - neck_offset[0],
                  neck_pos[1] - neck_offset[1],
                  neck_pos[2] - neck_offset[2]]

        # 将所有骨骼位置转换为相对颈部的位置
        for bone in unity_pose:
            unity_pose[bone] = [
                unity_pose[bone][0] - offset[0],
                unity_pose[bone][1] - offset[1],
                -(unity_pose[bone][2] - offset[2])  # 原始项目: position[2] = -position[2]
            ]

        # ===== 旋转变换 =====
        # 原始项目: 位置翻转 X 和 Y 后，旋转也需要镜像
        # 原始项目 OSC: rotation = [-rotation[1], -rotation[2], rotation[0]]
        #   其中 rotation 是 ZXY 欧拉角 [z, x, y]
        # 我们的位置变换: X翻转, Y翻转, Z不变
        # flip_mat = diag(-1, -1, 1) (翻转 X 和 Y 轴)
        _flip_mat = np.diag([-1.0, -1.0, 1.0])
        unity_rotations = {}
        for bone, euler_rad in rotations.items():
            try:
                rot = Rot.from_euler('xyz', euler_rad, degrees=False)
                mat = rot.as_matrix()
                mat_new = _flip_mat @ mat @ _flip_mat
                rot_new = Rot.from_matrix(mat_new)
                euler_zxy = rot_new.as_euler("zxy", degrees=True)
                # as_euler("zxy") 返回 [z, x, y], VRChat OSC 期望 [x, y, z]
                unity_rotations[bone] = [float(euler_zxy[1]), float(euler_zxy[2]), float(euler_zxy[0])]
            except Exception:
                unity_rotations[bone] = [0.0, 0.0, 0.0]

        # ===== 身体追踪器发送 (仅当有身体数据时) =====
        if not body_only:
            # VRChat OSC 追踪器编号映射
            osc_tracker_map = {
            "Hips": 1,              # Waist
            "LeftFoot": 2,          # Left Foot
            "RightFoot": 3,         # Right Foot
            "Chest": 4,             # Chest
            "LeftLowerLeg": 5,      # Left Knee
            "RightLowerLeg": 6,     # Right Knee
            "LeftLowerArm": 7,      # Left Elbow
            "RightLowerArm": 8,     # Right Elbow
        }

        for bone_name, tracker_id in osc_tracker_map.items():
            # 检查该身体部位的传输开关
            if not self._tracker_toggles.get(bone_name, True):
                continue

            # 发送位置 (仅当位置开关开启时)
            if self._send_position:
                pos = unity_pose.get(bone_name, [0, 0, 0])
                pos_addr = f"/tracking/trackers/{tracker_id}/position"
                message = self._build_osc_message(
                    pos_addr, float(pos[0]), float(pos[1]), float(pos[2])
                )
                self._osc_socket.sendto(message, (self.host, self.port))

            # 发送旋转 (仅当旋转开关开启时)
            if self._send_rotation:
                euler_deg = unity_rotations.get(bone_name, [0, 0, 0])
                rot_addr = f"/tracking/trackers/{tracker_id}/rotation"
                message = self._build_osc_message(
                    rot_addr, float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])
                )
                self._osc_socket.sendto(message, (self.host, self.port))

        # ===== 头部数据发送 =====
        # 头部位置始终为原点 [0, 0, 0]，避免干扰 VR 头显的激光测距定位
        # VR 头显通过 Lighthouse/inside-out 追踪提供精确的头部位置
        # 动捕系统只提供头部旋转数据

        # 优先使用面部变换矩阵的精确旋转 (参考 mediapipe-vt facial_transformation_matrixes)
        # MediaPipe 的面部变换矩阵比 YOLO 姿态估计的头部旋转更精确
        face_data = pose_data.get("face")
        if face_data and "face_rotation" in face_data:
            # face_rotation: [pitch, yaw, roll] (度)
            # MediaPipe 坐标系: X右, Y上, Z前
            # VRChat OSC 头部旋转: [x, y, z] = [pitch, yaw, roll]
            face_rot = face_data["face_rotation"]
            head_euler = [float(face_rot[0]), float(face_rot[1]), float(face_rot[2])]
        else:
            head_euler = unity_rotations.get("Head", [0, 0, 0])

        # 头部位置: 始终发送 [0,0,0] (参考原始项目)
        # 原始项目: trackers.append({ "name": "head", "position": [ 0, 0, 0 ]})
        # VR 头显通过 Lighthouse/inside-out 提供精确的头部绝对位置
        # 动捕系统不应覆盖头显位置，否则会干扰 VR 定位
        # send_head_position 开关仅控制是否发送非零头部位置 (不佩戴头显时使用)
        if self._send_position:
            if self._send_head_position:
                # 不佩戴头显模式: 发送面部变换矩阵的头部位置
                if face_data and "face_position" in face_data:
                    face_pos = face_data["face_position"]
                    pos_msg = self._build_osc_message(
                        "/tracking/trackers/head/position",
                        float(face_pos[0]), float(face_pos[1]), float(face_pos[2])
                    )
                else:
                    pos_msg = self._build_osc_message(
                        "/tracking/trackers/head/position",
                        float(0.0), float(0.0), float(0.0)
                    )
            else:
                # 佩戴头显模式 (默认): 头部位置始终为原点，不干扰头显定位
                pos_msg = self._build_osc_message(
                    "/tracking/trackers/head/position",
                    float(0.0), float(0.0), float(0.0)
                )
            self._osc_socket.sendto(pos_msg, (self.host, self.port))

        # 头部旋转: 始终发送，让 VRChat 根据头显位置+动捕旋转计算朝向
        if self._send_rotation:
            rot_msg = self._build_osc_message(
                "/tracking/trackers/head/rotation",
                float(head_euler[0]), float(head_euler[1]), float(head_euler[2])
            )
            self._osc_socket.sendto(rot_msg, (self.host, self.port))

        # ===== VRto3D HMD 数据传输 =====
        # 通过 UDP 端口 4242 发送 HMD 位置+四元数到 VRto3D 驱动
        # 格式: 7个 double (x, y, z, qw, qx, qy, qz)
        if self._vrto3d_socket:
            try:
                head_rot_euler = rotations.get("Head", [0, 0, 0])
                # 坐标系变换: 翻转 X 和 Y 轴 (与 OSC 位置变换一致)
                rot = Rot.from_euler('xyz', head_rot_euler, degrees=False)
                mat = rot.as_matrix()
                _flip = np.diag([-1.0, -1.0, 1.0])
                mat_new = _flip @ mat @ _flip
                rot_new = Rot.from_matrix(mat_new)
                qx, qy, qz, qw = rot_new.as_quat()

                # ExVR HMD 位置: 使用增量方式计算
                # 使用 unity_pose 中的头部位置 (已做坐标变换)
                pos = unity_pose.get("Head", [0, 0, 0])
                if self._vrto3d_prev_pos is None:
                    self._vrto3d_prev_pos = self._vrto3d_curr_pos = pos

                # 计算增量 (参考 ExVR 的 delta 计算)
                delta = [pos[i] - self._vrto3d_prev_pos[i] for i in range(3)]
                self._vrto3d_curr_pos = [self._vrto3d_curr_pos[i] + delta[i] for i in range(3)]
                self._vrto3d_prev_pos = pos

                # 打包: 7个 double (x, y, z, qw, qx, qy, qz)
                packed = struct.pack("7d", *self._vrto3d_curr_pos, qw, qx, qy, qz)
                self._vrto3d_socket.sendto(packed, (self.host, self._vrto3d_port))
            except Exception as e:
                pass  # VRto3D 非必须，静默失败

        # ===== VMT 手部追踪 (参考 ExVR hand_sender.py) =====
        # 通过 /VMT/Joint/Driver 发送手部位置+四元数
        hand_data = pose_data.get("hands")
        if hand_data and self._vmt_socket:
            self._send_vmt_hands(unity_pose, unity_rotations, hand_data)

        # 发送面部表情参数（嘴部/眼睛/眉毛）
        face_data = pose_data.get("face")
        if face_data:
            self._send_osc_face_params(face_data)

        # 发送手指关节参数
        hand_data = pose_data.get("hands")
        if hand_data:
            self._send_osc_hand_params(hand_data)

    # ========== VMT 手部追踪协议 (参考 ExVR hand_sender.py) ==========

    def _vmt_init(self):
        """初始化 VMT 房间矩阵 (参考 ExVR GloveControllerSender.vmt_init)"""
        if not self._vmt_socket:
            return
        try:
            # ExVR: /VMT/SetRoomMatrix [1,0,0,1,0,1,0,-0.76,0,0,1,1]
            message = self._build_osc_message(
                "/VMT/SetRoomMatrix",
                1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -0.76, 0.0, 0.0, 1.0, 1.0
            )
            self._vmt_socket.sendto(message, (self.host, self._vmt_port))
            self._vmt_initialized = True
        except Exception as e:
            print(f"VMT init error: {e}")

    def _send_vmt_hands(self, unity_pose: Dict, unity_rotations: Dict, hand_data):
        """通过 VMT 协议发送手部追踪数据 (参考 ExVR hand_sender.py)
        
        ExVR 的 VMT 消息格式:
        /VMT/Joint/Driver [hand_type, enable, timeoffset, x, y, z, qx, qy, qz, qw, "HMD"]
        hand_type: 1=左手, 2=右手
        enable: 5=左手, 6=右手 (ExVR 用 enable 值区分)
        """
        if not self._vmt_socket:
            return

        from scipy.spatial.transform import Rotation as Rot

        # 将列表格式转为字典格式
        if isinstance(hand_data, list):
            hand_dict = {}
            for h in hand_data:
                label = h.get("label", "unknown")
                hand_dict[label] = h
            hand_data = hand_dict

        for hand_label, is_left in [("left", True), ("right", False)]:
            data = hand_data.get(hand_label)
            if data is None:
                continue

            # 获取手部位置 (从 unity_pose)
            hand_bone = "LeftHand" if is_left else "RightHand"
            pos = unity_pose.get(hand_bone, [0, 0, 0])

            # 获取手部旋转 (从 unity_rotations, 已是 ZXY 欧拉角度数)
            euler_deg = unity_rotations.get(hand_bone, [0, 0, 0])
            try:
                rot = Rot.from_euler("zxy", euler_deg, degrees=True)
                qx, qy, qz, qw = rot.as_quat()
            except Exception:
                qx, qy, qz, qw = 0, 0, 0, 1

            # 发送 VMT Joint/Driver 消息 (参考 ExVR hand_sender.py)
            hand_type = 1 if is_left else 2
            enable = 5 if is_left else 6
            message = self._build_osc_message(
                "/VMT/Joint/Driver",
                hand_type,   # 1=左手, 2=右手
                enable,      # enable 标志
                0.0,         # timeoffset
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                float(qx),
                float(qy),
                float(qz),
                float(qw),
                "HMD"        # serial
            )
            self._vmt_socket.sendto(message, (self.host, self._vmt_port))

            # 发送手指弯曲度 (参考 ExVR hand_sender.py send_finger)
            finger_names = ["thumb", "index", "middle", "ring", "pinky"]
            fingers = data.get("fingers", {})
            curls = data.get("curls", {})

            for i, finger in enumerate(finger_names):
                curl = curls.get(finger, 0.0)
                if not curl and fingers:
                    is_extended = fingers.get(finger, False)
                    curl = 0.0 if is_extended else 0.8
                curl = max(0.0, min(1.0, float(curl)))

                # /VMT/Skeleton/Scalar [hand_type, finger_index, curl, 0, 0]
                finger_msg = self._build_osc_message(
                    "/VMT/Skeleton/Scalar",
                    hand_type,
                    int(i + 1),  # finger index (1-5)
                    curl,
                    0,
                    0
                )
                self._vmt_socket.sendto(finger_msg, (self.host, self._vmt_port))

            # /VMT/Skeleton/Apply [hand_type, 0.0]
            apply_msg = self._build_osc_message(
                "/VMT/Skeleton/Apply",
                hand_type,
                0.0
            )
            self._vmt_socket.sendto(apply_msg, (self.host, self._vmt_port))

    def _send_osc_face_params(self, face_data: Dict):
        """通过 OSC 发送面部表情参数到 VRChat

        支持 VRCFT Unified Expressions v2 标准参数:
        - 优先使用 face_data["vrcft"] 中的 VRCFT 标准参数
        - 回退到旧版自定义参数（兼容无 blendshapes 的情况）

        VRCFT v2 参数通过 /avatar/parameters/v2/XXX 地址发送，
        与 VRCFaceTracking 官方模块使用相同的地址格式，
        确保兼容所有支持 VRCFT 的 VRChat 模型。

        优化: 批量构建消息后一次性发送，减少系统调用次数降低延迟。
        """
        if not self._osc_socket:
            return

        # ===== 优先使用 VRCFT 标准参数 =====
        vrcft = face_data.get("vrcft")
        if vrcft:
            # 批量构建所有消息，减少逐个 sendto 的系统调用开销
            dest = (self.host, self.port)
            messages = []
            for param_name, value in vrcft.items():
                osc_address = f"/avatar/parameters/{param_name}"
                value = max(-1.0, min(1.0, float(value)))
                messages.append(self._build_osc_message(osc_address, value))
            # 一次性发送所有消息
            sendto = self._osc_socket.sendto
            for msg in messages:
                sendto(msg, dest)
            return

        # ===== 回退: 旧版自定义参数映射（无 blendshapes 时） =====
        osc_face_map = {
            "mouth_open": "/avatar/parameters/MouthOpenBlend",
            "mouth_wide": "/avatar/parameters/MouthSmileBlend",
            "left_eye_open": "/avatar/parameters/EyeOpenLeft",
            "right_eye_open": "/avatar/parameters/EyeOpenRight",
            "left_brow_up": "/avatar/parameters/BrowLeftUp",
            "right_brow_up": "/avatar/parameters/BrowRightUp",
        }

        for param_name, osc_address in osc_face_map.items():
            value = face_data.get(param_name, 0.0)
            value = max(0.0, min(1.0, float(value)))
            message = self._build_osc_message(osc_address, value)
            self._osc_socket.sendto(message, (self.host, self.port))

        # 眼睛闭合（取反值，1=闭合）
        left_closed = 1.0 - face_data.get("left_eye_open", 1.0)
        right_closed = 1.0 - face_data.get("right_eye_open", 1.0)
        msg_l = self._build_osc_message("/avatar/parameters/EyeLeftClosed", float(left_closed))
        msg_r = self._build_osc_message("/avatar/parameters/EyeRightClosed", float(right_closed))
        self._osc_socket.sendto(msg_l, (self.host, self.port))
        self._osc_socket.sendto(msg_r, (self.host, self.port))

    def _send_osc_hand_params(self, hand_data):
        """通过 OSC 发送手指关节参数到 VRChat

        VRChat 手指参数 OSC 地址:
        左手:
          /avatar/parameters/LeftThumbCurl
          /avatar/parameters/LeftIndexCurl
          /avatar/parameters/LeftMiddleCurl
          /avatar/parameters/LeftRingCurl
          /avatar/parameters/LeftLittleCurl
        右手:
          /avatar/parameters/RightThumbCurl
          /avatar/parameters/RightIndexCurl
          /avatar/parameters/RightMiddleCurl
          /avatar/parameters/RightRingCurl
          /avatar/parameters/RightLittleCurl
        手指张开:
          /avatar/parameters/LeftHandSpread
          /avatar/parameters/RightHandSpread
        """
        if not self._osc_socket:
            return

        # 将列表格式转为字典格式
        if isinstance(hand_data, list):
            hand_dict = {}
            for h in hand_data:
                label = h.get("label", "unknown")
                hand_dict[label] = h
            hand_data = hand_dict

        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        finger_cap = ["Thumb", "Index", "Middle", "Ring", "Little"]

        for hand_label in ["left", "right"]:
            data = hand_data.get(hand_label)
            if data is None:
                continue

            # 从 fingers 字典获取弯曲度，或从 curls 字典获取
            fingers = data.get("fingers", {})
            curls = data.get("curls", {})
            prefix = "Left" if hand_label == "left" else "Right"

            # 发送每根手指的弯曲度
            for i, finger in enumerate(finger_names):
                # 优先使用 curls，否则从 fingers 推算
                curl = curls.get(finger, 0.0)
                if not curl and fingers:
                    # fingers 中是布尔值（True=伸展），反转为弯曲度
                    is_extended = fingers.get(finger, False)
                    curl = 0.0 if is_extended else 0.8
                curl = max(0.0, min(1.0, float(curl)))
                address = f"/avatar/parameters/{prefix}{finger_cap[i]}Curl"
                message = self._build_osc_message(address, curl)
                self._osc_socket.sendto(message, (self.host, self.port))

            # 发送手指张开度
            spread = data.get("spread", 0.0)
            spread = max(0.0, min(1.0, float(spread)))
            spread_addr = f"/avatar/parameters/{prefix}HandSpread"
            self._osc_socket.sendto(
                self._build_osc_message(spread_addr, spread),
                (self.host, self.port)
            )

            # 手腕旋转通过 avatar parameters 发送 (VRChat OSC 追踪器不支持手部追踪器)

            # 发送手掌/手背朝向参数
            # 0.0 = 手掌朝向, 1.0 = 手背朝向
            palm_side = data.get("palm_side", "palm")
            palm_conf = data.get("palm_confidence", 0.5)
            palm_value = 0.0 if palm_side == "palm" else 1.0
            # 用置信度缩放：低置信度时趋近0.5（中性值）
            palm_value = 0.5 + (palm_value - 0.5) * palm_conf
            palm_addr = f"/avatar/parameters/{prefix}HandPalmBack"
            self._osc_socket.sendto(
                self._build_osc_message(palm_addr, float(palm_value)),
                (self.host, self.port)
            )

    def _send_vmc(self, pose_data: Dict):
        """通过 VMC 协议发送（基于 OSC）"""
        if not self._osc_socket:
            return

        rotations = pose_data.get("rotations", {})
        pose_3d = pose_data.get("pose_3d", {})

        # VMC 骨骼旋转地址
        vmc_bone_map = {
            "Hips": "/VMC/Ext/Bone/Hips",
            "Spine": "/VMC/Ext/Bone/Spine",
            "Chest": "/VMC/Ext/Bone/Chest",
            "UpperChest": "/VMC/Ext/Bone/UpperChest",
            "Neck": "/VMC/Ext/Bone/Neck",
            "Head": "/VMC/Ext/Bone/Head",
            "LeftUpperLeg": "/VMC/Ext/Bone/LeftUpperLeg",
            "LeftLowerLeg": "/VMC/Ext/Bone/LeftLowerLeg",
            "LeftFoot": "/VMC/Ext/Bone/LeftFoot",
            "LeftToes": "/VMC/Ext/Bone/LeftToes",
            "RightUpperLeg": "/VMC/Ext/Bone/RightUpperLeg",
            "RightLowerLeg": "/VMC/Ext/Bone/RightLowerLeg",
            "RightFoot": "/VMC/Ext/Bone/RightFoot",
            "RightToes": "/VMC/Ext/Bone/RightToes",
            "LeftUpperArm": "/VMC/Ext/Bone/LeftUpperArm",
            "LeftLowerArm": "/VMC/Ext/Bone/LeftLowerArm",
            "LeftHand": "/VMC/Ext/Bone/LeftHand",
            "RightUpperArm": "/VMC/Ext/Bone/RightUpperArm",
            "RightLowerArm": "/VMC/Ext/Bone/RightLowerArm",
            "RightHand": "/VMC/Ext/Bone/RightHand",
        }

        for bone_name, osc_address in vmc_bone_map.items():
            euler = rotations.get(bone_name, [0, 0, 0])
            # VMC 发送四元数 (x, y, z, w)
            from scipy.spatial.transform import Rotation as R
            quat = R.from_euler("xyz", euler).as_quat()
            pos = pose_3d.get(bone_name, [0, 0, 0])
            message = self._build_osc_message(
                osc_address,
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
                float(pos[0]), float(pos[1]), float(pos[2])
            )
            self._osc_socket.sendto(message, (self.host, self.port))

        # VMC 可用信号
        available_msg = self._build_osc_message("/VMC/Ext/OK", 1)
        self._osc_socket.sendto(available_msg, (self.host, self.port))

    # ========== TCP 协议 ==========

    def _send_tcp(self, pose_data: Dict):
        """通过自定义 TCP 协议发送"""
        if not self._tcp_socket or not self.is_connected:
            return

        rotations = pose_data.get("rotations", {})
        message = {
            "type": "vrm",
            "timestamp": time.time(),
            "body": {
                "Hips": rotations.get("Hips", [0, 0, 0]),
                "Spine": rotations.get("Spine", [0, 0, 0]),
                "Chest": rotations.get("Chest", [0, 0, 0]),
                "UpperChest": rotations.get("UpperChest", [0, 0, 0]),
                "Neck": rotations.get("Neck", [0, 0, 0]),
                "Head": rotations.get("Head", [0, 0, 0]),
                "LeftUpperLeg": rotations.get("LeftUpperLeg", [0, 0, 0]),
                "LeftLowerLeg": rotations.get("LeftLowerLeg", [0, 0, 0]),
                "LeftFoot": rotations.get("LeftFoot", [0, 0, 0]),
                "RightUpperLeg": rotations.get("RightUpperLeg", [0, 0, 0]),
                "RightLowerLeg": rotations.get("RightLowerLeg", [0, 0, 0]),
                "RightFoot": rotations.get("RightFoot", [0, 0, 0]),
                "LeftUpperArm": rotations.get("LeftUpperArm", [0, 0, 0]),
                "LeftLowerArm": rotations.get("LeftLowerArm", [0, 0, 0]),
                "LeftHand": rotations.get("LeftHand", [0, 0, 0]),
                "RightUpperArm": rotations.get("RightUpperArm", [0, 0, 0]),
                "RightLowerArm": rotations.get("RightLowerArm", [0, 0, 0]),
                "RightHand": rotations.get("RightHand", [0, 0, 0]),
            }
        }

        serialized = json.dumps(message).encode("utf-8")
        length_header = struct.pack("!I", len(serialized))
        self._tcp_socket.sendall(length_header + serialized)

    def send_pose(self, pose_data: Dict) -> bool:
        try:
            if self.send_queue.full():
                try:
                    self.send_queue.get_nowait()
                except Empty:
                    pass

            self.send_queue.put(pose_data)
            return True
        except Exception as e:
            print(f"Error queuing pose data: {e}")
            return False

    def get_status(self) -> Dict:
        return {
            "connected": self.is_connected,
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "fps": self.fps,
            "queue_size": self.send_queue.qsize()
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
