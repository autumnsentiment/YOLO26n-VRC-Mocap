"""
HMD 位姿数据提供者模块

支持多种后端获取 VR 头显的真实世界坐标:
1. OpenVR (SteamVR) - 通过 SteamVR 读取 HMD 位姿，适用于 Quest/Pico 通过 Link 连接
2. VRChat OSC 反馈 - 监听 VRChat 发回的头部位置，适用于任何头显
3. Pico Streaming - 通过 Pico Streaming Assistant HTTP API 获取位姿
4. Quest Link - 通过 OVR Runtime HTTP API 获取位姿

坐标系统说明:
- OpenVR/SteamVR: 右手坐标系, X右, Y上, Z后 (OpenGL 风格)
- VRChat OSC: 左手坐标系, X右, Y上, Z前 (Unity 风格)
- 本模块统一输出为 VRChat OSC 坐标系 (X右, Y上, Z前)
"""

import time
import math
import threading
import socket
import json
import struct
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class HMDPose:
    """HMD 位姿数据"""
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [x, y, z] 米
    rotation: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [pitch, yaw, roll] 度
    quaternion: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
    timestamp: float = 0.0
    valid: bool = False


class HMDDataProvider:
    """HMD 位姿数据提供者基类"""

    def __init__(self):
        self._pose = HMDPose()
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> bool:
        """启动数据采集"""
        self._running = True
        return True

    def stop(self):
        """停止数据采集"""
        self._running = False

    def get_pose(self) -> HMDPose:
        """获取最新的 HMD 位姿 (线程安全)"""
        with self._lock:
            return HMDPose(
                position=list(self._pose.position),
                rotation=list(self._pose.rotation),
                quaternion=list(self._pose.quaternion),
                timestamp=self._pose.timestamp,
                valid=self._pose.valid,
            )

    def _update_pose(self, position, rotation=None, quaternion=None):
        """更新位姿数据 (内部使用)"""
        with self._lock:
            self._pose.position = position
            if rotation is not None:
                self._pose.rotation = rotation
            if quaternion is not None:
                self._pose.quaternion = quaternion
            self._pose.timestamp = time.time()
            self._pose.valid = True

    @property
    def name(self) -> str:
        return "base"


class OpenVRProvider(HMDDataProvider):
    """OpenVR (SteamVR) HMD 位姿提供者

    通过 pyopenvr 直接读取 SteamVR 中的 HMD 位姿数据。
    适用于通过 SteamVR Link 连接的 Quest 和 Pico 头显。

    坐标系变换:
    OpenVR: X右, Y上, Z后 (右手坐标系, OpenGL 风格)
    VRChat: X右, Y上, Z前 (左手坐标系, Unity 风格)
    变换: Z 轴翻转 (Z_openvr -> -Z_vrchat)
    """

    def __init__(self):
        super().__init__()
        self._vr_system = None
        self._poll_thread = None
        self._poll_interval = 0.005  # 200Hz 采集

    @property
    def name(self) -> str:
        return "openvr"

    def start(self) -> bool:
        try:
            import openvr
            # 检查 scipy 和 numpy 是否可用 (坐标变换必需)
            if np is None:
                print("HMD Provider: numpy not installed, run: pip install numpy")
                return False
            try:
                from scipy.spatial.transform import Rotation as Rot
            except ImportError:
                print("HMD Provider: scipy not installed, run: pip install scipy")
                return False
            openvr.init(openvr.VRApplication_Other)
            self._vr_system = openvr.VRSystem()
            self._running = True
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()
            print(f"HMD Provider: OpenVR initialized (SteamVR)")
            return True
        except ImportError:
            print("HMD Provider: openvr package not installed, run: pip install openvr")
            return False
        except Exception as e:
            print(f"HMD Provider: OpenVR init failed: {e}")
            return False

    def stop(self):
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)
        if self._vr_system is not None:
            try:
                import openvr
                openvr.shutdown()
            except Exception:
                pass
            self._vr_system = None

    def _poll_loop(self):
        import openvr
        while self._running:
            try:
                poses = self._vr_system.getDeviceToAbsoluteTrackingPose(
                    openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
                )
                hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
                if hmd_pose.bPoseIsValid:
                    mat = hmd_pose.mDeviceToAbsoluteTracking
                    # 从 3x4 矩阵提取位置
                    # OpenVR 矩阵布局: mat[row][col], 位置在 [row][3]
                    pos_x = mat[0][3]
                    pos_y = mat[1][3]
                    pos_z = mat[2][3]

                    # 从矩阵提取旋转 (四元数)
                    r_w = math.sqrt(abs(1 + mat[0][0] + mat[1][1] + mat[2][2])) / 2
                    if r_w > 1e-6:
                        r_x = (mat[2][1] - mat[1][2]) / (4 * r_w)
                        r_y = (mat[0][2] - mat[2][0]) / (4 * r_w)
                        r_z = (mat[1][0] - mat[0][1]) / (4 * r_w)
                    else:
                        r_x, r_y, r_z = 0.0, 0.0, 0.0

                    # 坐标系变换: OpenVR (Z后) -> VRChat (Z前)
                    # 位置: Z 翻转
                    position = [pos_x, pos_y, -pos_z]

                    # 四元数变换: 绕 Y 轴旋转 180 度
                    # q_new = q_y180 * q_old, q_y180 = (0, 0, 1, 0)
                    # 结果: w'=-r_z, x'=r_x, y'=-r_w, z'=r_y (简化)
                    # 但更准确的做法: 使用旋转矩阵变换
                    from scipy.spatial.transform import Rotation as Rot
                    rot = Rot.from_quat([r_x, r_y, r_z, r_w])  # scipy 格式: [x,y,z,w]
                    mat_rot = rot.as_matrix()
                    # 绕 Y 轴 180 度的变换
                    flip_y = np.diag([-1.0, 1.0, -1.0])
                    mat_new = flip_y @ mat_rot
                    rot_new = Rot.from_matrix(mat_new)
                    quat_new = rot_new.as_quat()  # [x, y, z, w]
                    euler_new = rot_new.as_euler('xyz', degrees=True)  # [x, y, z] 度

                    self._update_pose(
                        position=position,
                        rotation=[float(euler_new[0]), float(euler_new[1]), float(euler_new[2])],
                        quaternion=[float(quat_new[3]), float(quat_new[0]), float(quat_new[1]), float(quat_new[2])],
                    )
            except Exception:
                pass
            time.sleep(self._poll_interval)


class VRChatOSCProvider(HMDDataProvider):
    """VRChat OSC 反馈 HMD 位姿提供者

    监听 VRChat 发送的 OSC 追踪数据，获取 HMD 的真实位置。
    VRChat 在运行时会将头显位置通过 OSC 发送回本机。

    优势:
    - 无需 SteamVR，适用于任何头显
    - 直接获取 VRChat 中的头部位置，天然对齐
    - 适用于 Quest/Pico 通过 Wi-Fi 连接的场景

    VRChat OSC 输出端口默认 9001，可在 VRChat 设置中修改。
    """

    def __init__(self, listen_port: int = 9001):
        super().__init__()
        self._listen_port = listen_port
        self._server = None
        self._server_thread = None
        self._timeout = 2.0  # 2秒无数据则标记为无效

    @property
    def name(self) -> str:
        return "vrchat_osc"

    def start(self) -> bool:
        try:
            from pythonosc import dispatcher as osc_dispatcher
            from pythonosc import osc_server

            disp = osc_dispatcher.Dispatcher()
            disp.map("/tracking/trackers/head/position", self._on_head_position)
            disp.map("/tracking/trackers/head/rotation", self._on_head_rotation)

            self._server = osc_server.ThreadingOSCUDPServer(
                ("0.0.0.0", self._listen_port), disp
            )
            self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._server_thread.start()
            self._running = True
            print(f"HMD Provider: VRChat OSC listening on port {self._listen_port}")
            return True
        except ImportError:
            print("HMD Provider: python-osc not installed, run: pip install python-osc")
            return False
        except Exception as e:
            print(f"HMD Provider: VRChat OSC init failed: {e}")
            return False

    def stop(self):
        self._running = False
        if self._server:
            self._server.shutdown()
        self._server = None

    def _on_head_position(self, address, *args):
        """接收 VRChat 头部位置 (VRChat 坐标系: X右, Y上, Z前)"""
        if len(args) >= 3:
            with self._lock:
                self._pose.position = [float(args[0]), float(args[1]), float(args[2])]
                self._pose.timestamp = time.time()
                self._pose.valid = True

    def _on_head_rotation(self, address, *args):
        """接收 VRChat 头部旋转 (欧拉角, 度)"""
        if len(args) >= 3:
            with self._lock:
                self._pose.rotation = [float(args[0]), float(args[1]), float(args[2])]

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        # 超时检查
        if pose.valid and (time.time() - pose.timestamp) > self._timeout:
            pose.valid = False
        return pose


class PicoStreamingProvider(HMDDataProvider):
    """Pico Streaming Assistant HMD 位姿提供者

    通过 Pico Streaming Assistant 的 HTTP API 获取 HMD 位姿。
    Pico 头显通过 Streaming Assistant 连接到 PC 时，可以使用此方式。

    Pico Native SDK API: PicovrSDK.GetMainSensor() 返回 {x, y, z, w, px, py, pz}
    其中 (x,y,z,w) 是四元数，(px,py,pz) 是位置。

    注意: 此提供者需要 Pico Streaming Assistant 在 PC 上运行，
    并通过 HTTP API 或共享内存获取位姿数据。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7890):
        super().__init__()
        self._host = host
        self._port = port
        self._poll_thread = None
        self._poll_interval = 0.033  # 30Hz

    @property
    def name(self) -> str:
        return "pico_streaming"

    def start(self) -> bool:
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        print(f"HMD Provider: Pico Streaming connecting to {self._host}:{self._port}")
        return True

    def stop(self):
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)

    def _poll_loop(self):
        """轮询 Pico Streaming API 获取 HMD 位姿

        Pico Streaming Assistant 在 PC 端提供 HTTP API。
        端点和数据格式可能因版本而异，需要根据实际 SDK 文档调整。

        Pico Spatial SDK HMD Tracking:
        - HMDTrackingProvider().dataFlow 获取 HMDPose
        - HMDPose 包含 position (Vector3) 和 rotation (Quat)

        Pico Native SDK:
        - GetMainSensor() 返回 float[7]: {quat_x, quat_y, quat_z, quat_w, pos_x, pos_y, pos_z}
        - 坐标系: X右, Y上, Z前 (Unity 风格)
        """
        while self._running:
            try:
                # 尝试通过 HTTP API 获取位姿
                # Pico Streaming Assistant 的具体 API 端点需要根据版本确认
                # 这里使用通用的 HTTP 请求方式
                import urllib.request
                url = f"http://{self._host}:{self._port}/api/hmd/pose"
                req = urllib.request.Request(url, method='GET')
                req.timeout = 0.5
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get("valid", False):
                        pos = data.get("position", [0, 0, 0])
                        rot = data.get("rotation", [0, 0, 0])
                        self._update_pose(position=pos, rotation=rot)
            except Exception:
                # Pico Streaming API 不可用，静默失败
                pass
            time.sleep(self._poll_interval)


class QuestLinkProvider(HMDDataProvider):
    """Quest Link / Air Link HMD 位姿提供者

    通过 Meta Quest Link 的 OVR Runtime 获取 HMD 位姿。
    Quest 头显通过 Link/Air Link 连接到 PC 时使用。

    实现方式:
    1. 优先使用 OpenVR (Quest Link 会注册到 SteamVR)
    2. 备选: 通过 OVR HTTP API 获取位姿

    OVR Runtime HTTP API:
    - 默认端口: 53342 (OVR Server)
    - 可获取 HMD 位姿和控制器数据

    Quest 头显也支持通过 ADB 转发获取位姿:
    - 安装自定义 APK 在 Quest 上运行
    - APK 通过 WebSocket/UDP 发送位姿到 PC
    - 参考: github.com/NeuracoreAI/meta_quest_teleop
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 53342):
        super().__init__()
        self._host = host
        self._port = port
        self._poll_thread = None
        self._poll_interval = 0.01
        self._openvr_fallback = None

    @property
    def name(self) -> str:
        return "quest_link"

    def start(self) -> bool:
        # 优先尝试 OpenVR (Quest Link 通过 SteamVR 注册)
        try:
            import openvr
            self._openvr_fallback = OpenVRProvider()
            if self._openvr_fallback.start():
                self._running = True
                print("HMD Provider: Quest Link via OpenVR (SteamVR)")
                return True
        except Exception:
            pass

        # 备选: OVR HTTP API
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        print(f"HMD Provider: Quest Link OVR API at {self._host}:{self._port}")
        return True

    def stop(self):
        self._running = False
        if self._openvr_fallback:
            self._openvr_fallback.stop()
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)

    def get_pose(self) -> HMDPose:
        if self._openvr_fallback:
            return self._openvr_fallback.get_pose()
        return super().get_pose()

    def _poll_loop(self):
        """轮询 OVR Runtime HTTP API 获取 HMD 位姿

        OVR Server HTTP API 端点 (需要根据实际版本确认):
        - /api/hmd/pose: 获取 HMD 位姿
        - /api/session/status: 获取会话状态

        坐标系: OVR 使用 X右, Y上, Z前 (Unity 风格)
        """
        while self._running:
            try:
                import urllib.request
                url = f"http://{self._host}:{self._port}/api/hmd/pose"
                req = urllib.request.Request(url, method='GET')
                req.timeout = 0.5
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get("valid", False):
                        pos = data.get("position", [0, 0, 0])
                        rot = data.get("rotation", [0, 0, 0])
                        self._update_pose(position=pos, rotation=rot)
            except Exception:
                pass
            time.sleep(self._poll_interval)


class StaticProvider(HMDDataProvider):
    """静态 HMD 位姿提供者 (回退模式)

    返回固定的 [0, 0, 0] 位置，即当前默认行为。
    适用于不佩戴头显或无法获取 HMD 数据的场景。
    """

    @property
    def name(self) -> str:
        return "static"

    def start(self) -> bool:
        self._update_pose(position=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0])
        print("HMD Provider: Static (position=[0,0,0])")
        return True


class PicoXRoboProvider(HMDDataProvider):
    """Pico XR-Robotics HMD 位姿提供者

    通过 XR-Robotics XRoboToolkit PC Service 接收 Pico 头显位姿数据。
    在 Pico 头显上安装 XRoboToolkit Unity APK，头显通过 WiFi 将
    HMD 位姿和手部追踪数据发送到 PC 端的 PC Service。

    数据接收方式:
    - PC Service 运行后，通过 Python SDK (xrobotoolkit_sdk) 获取位姿
    - 或通过 UDP 接收头显直接发送的位姿数据包

    UDP 数据包格式 (兼容 XRoboToolkit):
    - 4字节 header: "PICO" (0x50 0x49 0x43 0x4F)
    - 8字节 timestamp (double)
    - 28字节 HMD pose: position(3 floats) + quaternion(4 floats)
    - 坐标系: X右, Y上, Z前 (Unity 风格)

    参考: https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 39540):
        super().__init__()
        self._host = host
        self._port = port
        self._socket = None
        self._recv_thread = None
        self._timeout = 2.0

    @property
    def name(self) -> str:
        return "pico_xrobo"

    def start(self) -> bool:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self._host, self._port))
            self._socket.settimeout(1.0)
            self._running = True
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()
            print(f"HMD Provider: Pico XR-Robotics listening on UDP {self._host}:{self._port}")
            return True
        except Exception as e:
            print(f"HMD Provider: Pico XR-Robotics init failed: {e}")
            return False

    def stop(self):
        self._running = False
        if self._socket:
            self._socket.close()
        self._socket = None
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)

    def _recv_loop(self):
        """接收 Pico XR-Robotics 位姿数据

        支持两种数据包格式:
        1. XRoboToolkit 格式: 4字节header("PICO") + 8字节timestamp + 28字节pose
        2. 纯二进制格式: 7个float32 (px, py, pz, qx, qy, qz, qw)
        3. JSON 格式: {"position": [x,y,z], "rotation": [qx,qy,qz,qw], "valid": true}
        """
        while self._running and self._socket:
            try:
                data, addr = self._socket.recvfrom(2048)
                if not data:
                    continue

                # 尝试 JSON 格式
                if data[0:1] == b'{':
                    try:
                        j = json.loads(data.decode('utf-8'))
                        if j.get("valid", True):
                            pos = j.get("position", [0, 0, 0])
                            quat = j.get("rotation", [0, 0, 0, 1])
                            # rotation 可能是四元数 [x,y,z,w] 或 [w,x,y,z]
                            if len(quat) == 4:
                                # 假设 [x,y,z,w] 格式 (Unity 风格)
                                qx, qy, qz, qw = quat
                                self._update_pose_from_quat(pos, qx, qy, qz, qw)
                            else:
                                self._update_pose(position=pos)
                        continue
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

                # XRoboToolkit 二进制格式: header(4) + timestamp(8) + pose(28)
                if len(data) >= 40 and data[:4] == b'PICO':
                    # 跳过 header(4) + timestamp(8) = 12 字节
                    pose_data = data[12:40]  # 7 floats = 28 bytes
                    if len(pose_data) == 28:
                        values = struct.unpack("7f", pose_data)
                        px, py, pz = values[0], values[1], values[2]
                        qx, qy, qz, qw = values[3], values[4], values[5], values[6]
                        self._update_pose_from_quat([px, py, pz], qx, qy, qz, qw)
                    continue

                # 纯二进制格式: 7个float32 (px, py, pz, qx, qy, qz, qw)
                if len(data) >= 28:
                    values = struct.unpack("7f", data[:28])
                    px, py, pz = values[0], values[1], values[2]
                    qx, qy, qz, qw = values[3], values[4], values[5], values[6]
                    self._update_pose_from_quat([px, py, pz], qx, qy, qz, qw)
                    continue

            except socket.timeout:
                continue
            except Exception:
                pass

    def _update_pose_from_quat(self, position, qx, qy, qz, qw):
        """从四元数更新位姿 (Unity 坐标系: X右, Y上, Z前)"""
        try:
            from scipy.spatial.transform import Rotation as Rot
            rot = Rot.from_quat([qx, qy, qz, qw])  # scipy: [x,y,z,w]
            euler = rot.as_euler('xyz', degrees=True)
            self._update_pose(
                position=position,
                rotation=[float(euler[0]), float(euler[1]), float(euler[2])],
                quaternion=[float(qw), float(qx), float(qy), float(qz)],
            )
        except ImportError:
            self._update_pose(position=position, quaternion=[qw, qx, qy, qz])

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        if pose.valid and (time.time() - pose.timestamp) > self._timeout:
            pose.valid = False
        return pose


class QuestUDPProvider(HMDDataProvider):
    """Quest MetaQuest_HandTracking UDP HMD 位姿提供者

    通过 Meta Quest 头显上运行的 Unity APK 接收 HMD 位姿数据。
    参考 MetaQuest_HandTracking 项目，在 Quest 上安装 Unity 发送端 APK，
    头显通过 WiFi 将位姿数据以 UDP 数据包发送到 PC。

    UDP 数据包格式 (MetaQuest_HandTracking 协议):
    - 4字节 header: "HND0" (0x48 0x4E 0x44 0x30)
    - 8字节 timestamp (double)
    - 左手数据: 728字节 (182 floats: wrist pose + 25 bones)
    - 右手数据: 728字节 (182 floats: wrist pose + 25 bones)
    - 头显位姿: 28字节 (7 floats: position[3] + quaternion[4])
    - 4字节 footer: "HND1"

    头显位姿位于数据包末尾 (offset = 4+8+728+728 = 1468):
    - position: 3 floats (x, y, z) 米
    - quaternion: 4 floats (x, y, z, w) Unity 风格

    坐标系: X右, Y上, Z前 (Unity 风格)

    参考: https://github.com/kodaekwan/MetaQuest_HandTracking
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 39541):
        super().__init__()
        self._host = host
        self._port = port
        self._socket = None
        self._recv_thread = None
        self._timeout = 2.0

    @property
    def name(self) -> str:
        return "quest_udp"

    def start(self) -> bool:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self._host, self._port))
            self._socket.settimeout(1.0)
            self._running = True
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()
            print(f"HMD Provider: Quest UDP (MetaQuest_HandTracking) listening on {self._host}:{self._port}")
            return True
        except Exception as e:
            print(f"HMD Provider: Quest UDP init failed: {e}")
            return False

    def stop(self):
        self._running = False
        if self._socket:
            self._socket.close()
        self._socket = None
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)

    def _recv_loop(self):
        """接收 Quest MetaQuest_HandTracking 位姿数据

        支持两种数据包格式:
        1. MetaQuest_HandTracking 完整协议: header + timestamp + hands + headset
        2. 简化格式: 仅头显位姿 (7 floats 或 JSON)
        """
        while self._running and self._socket:
            try:
                data, addr = self._socket.recvfrom(4096)
                if not data:
                    continue

                # 尝试 JSON 格式
                if data[0:1] == b'{':
                    try:
                        j = json.loads(data.decode('utf-8'))
                        if j.get("valid", True):
                            pos = j.get("position", [0, 0, 0])
                            quat = j.get("rotation", [0, 0, 0, 1])
                            if len(quat) == 4:
                                qx, qy, qz, qw = quat
                                self._update_pose_from_quat(pos, qx, qy, qz, qw)
                            else:
                                self._update_pose(position=pos)
                        continue
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

                # MetaQuest_HandTracking 完整协议
                # header(4) + timestamp(8) + left_hand(728) + right_hand(728) + headset(28) + footer(4)
                if len(data) >= 1500 and data[:4] == b'HND0':
                    # 头显位姿在 offset 1468 处 (4+8+728+728)
                    headset_offset = 1468
                    if len(data) >= headset_offset + 28:
                        headset_data = data[headset_offset:headset_offset + 28]
                        values = struct.unpack("7f", headset_data)
                        px, py, pz = values[0], values[1], values[2]
                        qx, qy, qz, qw = values[3], values[4], values[5], values[6]
                        self._update_pose_from_quat([px, py, pz], qx, qy, qz, qw)
                    continue

                # 简化二进制格式: 7个float32 (px, py, pz, qx, qy, qz, qw)
                if len(data) >= 28:
                    values = struct.unpack("7f", data[:28])
                    px, py, pz = values[0], values[1], values[2]
                    qx, qy, qz, qw = values[3], values[4], values[5], values[6]
                    self._update_pose_from_quat([px, py, pz], qx, qy, qz, qw)
                    continue

            except socket.timeout:
                continue
            except Exception:
                pass

    def _update_pose_from_quat(self, position, qx, qy, qz, qw):
        """从四元数更新位姿 (Unity 坐标系: X右, Y上, Z前)"""
        try:
            from scipy.spatial.transform import Rotation as Rot
            rot = Rot.from_quat([qx, qy, qz, qw])  # scipy: [x,y,z,w]
            euler = rot.as_euler('xyz', degrees=True)
            self._update_pose(
                position=position,
                rotation=[float(euler[0]), float(euler[1]), float(euler[2])],
                quaternion=[float(qw), float(qx), float(qy), float(qz)],
            )
        except ImportError:
            self._update_pose(position=position, quaternion=[qw, qx, qy, qz])

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        if pose.valid and (time.time() - pose.timestamp) > self._timeout:
            pose.valid = False
        return pose


class HMDProviderFactory:
    """HMD 数据提供者工厂"""

    @staticmethod
    def create(provider_type: str, **kwargs) -> HMDDataProvider:
        """创建 HMD 数据提供者

        Args:
            provider_type: 提供者类型
                - "openvr": SteamVR OpenVR API
                - "vrchat_osc": VRChat OSC 反馈
                - "pico": Pico Streaming Assistant
                - "quest": Quest Link / Air Link
                - "static": 静态 (回退)
                - "auto": 自动检测 (按优先级尝试)
            **kwargs: 提供者参数
        """
        providers = {
            "openvr": OpenVRProvider,
            "vrchat_osc": VRChatOSCProvider,
            "pico": PicoStreamingProvider,
            "pico_xrobo": PicoXRoboProvider,
            "quest": QuestLinkProvider,
            "quest_udp": QuestUDPProvider,
            "static": StaticProvider,
            "dashboard_api": DashboardAPIProvider,
            "remote": RemoteHMDProvider,
        }

        if provider_type == "auto":
            return HMDProviderFactory._auto_detect(**kwargs)

        provider_class = providers.get(provider_type)
        if provider_class is None:
            print(f"HMD Provider: Unknown type '{provider_type}', using static")
            return StaticProvider()

        # 传递参数
        if provider_type == "vrchat_osc":
            return provider_class(listen_port=kwargs.get("osc_listen_port", 9001))
        elif provider_type == "pico":
            return provider_class(host=kwargs.get("pico_host", "127.0.0.1"),
                                  port=kwargs.get("pico_port", 7890))
        elif provider_type == "pico_xrobo":
            return provider_class(host=kwargs.get("pico_xrobo_host", "0.0.0.0"),
                                  port=kwargs.get("pico_xrobo_port", 39540))
        elif provider_type == "quest":
            return provider_class(host=kwargs.get("quest_host", "127.0.0.1"),
                                  port=kwargs.get("quest_port", 53342))
        elif provider_type == "quest_udp":
            return provider_class(host=kwargs.get("quest_udp_host", "0.0.0.0"),
                                  port=kwargs.get("quest_udp_port", 39541))
        elif provider_type == "dashboard_api":
            return provider_class(host=kwargs.get("dashboard_host", "127.0.0.1"),
                                  port=kwargs.get("dashboard_port", 8765))
        elif provider_type == "remote":
            return provider_class(listen_port=kwargs.get("remote_port", 4243))
        else:
            return provider_class()

    @staticmethod
    def _auto_detect(**kwargs) -> HMDDataProvider:
        """自动检测可用的 HMD 数据源

        优先级:
        1. OpenVR (SteamVR 运行中) - 最精确，延迟最低
        2. VRChat OSC 反馈 - 通用性最好
        3. Pico Streaming - Pico 专用
        4. Quest Link - Quest 专用
        5. Static - 回退
        """
        # 1. 尝试 OpenVR
        try:
            import openvr
            openvr.init(openvr.VRApplication_Other)
            openvr.shutdown()
            print("HMD Provider: Auto-detected OpenVR (SteamVR)")
            return OpenVRProvider()
        except Exception:
            pass

        # 2. 尝试 VRChat OSC
        try:
            from pythonosc import dispatcher, osc_server
            print("HMD Provider: Auto-detected VRChat OSC")
            return VRChatOSCProvider(listen_port=kwargs.get("osc_listen_port", 9001))
        except ImportError:
            pass

        # 3. 回退到静态
        print("HMD Provider: No HMD source detected, using static [0,0,0]")
        return StaticProvider()


class DashboardAPIProvider(HMDDataProvider):
    """通过 Dashboard API 获取 HMD 位姿

    动捕进程作为子进程运行时，无法直接访问 Dashboard 的 HMD Provider 实例。
    通过 HTTP API 从 Dashboard 获取 HMD 位姿数据。

    适用于: 动捕进程 (main.py) 通过 Dashboard 管理的场景
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        super().__init__()
        self._host = host
        self._port = port
        self._poll_thread = None
        self._poll_interval = 0.033  # 30Hz (HTTP 轮询不宜过高，避免 Dashboard 压力)
        self._api_url = f"http://{host}:{port}/api/hmd/status"

    @property
    def name(self) -> str:
        return "dashboard_api"

    def start(self) -> bool:
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        print(f"HMD Provider: Dashboard API at {self._api_url}")
        return True

    def stop(self):
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)

    def _poll_loop(self):
        """轮询 Dashboard API 获取 HMD 位姿"""
        import urllib.request
        while self._running:
            try:
                req = urllib.request.Request(self._api_url, method='GET')
                req.timeout = 0.1
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode())
                    if data.get("running") and data.get("pose"):
                        pose = data["pose"]
                        if pose.get("valid"):
                            self._update_pose(
                                position=pose.get("position", [0, 0, 0]),
                                rotation=pose.get("rotation", [0, 0, 0]),
                                quaternion=pose.get("quaternion", [1.0, 0.0, 0.0, 0.0]),
                            )
                        else:
                            # Dashboard 报告 HMD 无效，标记本地也为无效
                            with self._lock:
                                self._pose.valid = False
                    elif data.get("running") and not data.get("pose"):
                        # Dashboard 运行中但无 pose 数据
                        with self._lock:
                            self._pose.valid = False
            except Exception:
                pass
            time.sleep(self._poll_interval)


class RemoteHMDProvider(HMDDataProvider):
    """远程 HMD 位姿数据提供者

    当 VR 头显连接在另一台机器上时，通过 UDP 接收头显位姿数据。
    需要在头显所在的机器上运行 hmd_sender.py 发送位姿数据。

    协议格式 (UDP, 二进制):
    - 7 个 double (56 字节): x, y, z, qw, qx, qy, qz
    - 坐标系: VRChat (X右, Y上, Z前)

    使用场景:
    - Pico/Quest 头显通过 Wi-Fi 连接，位姿数据从另一台 PC 转发
    - 多人协作场景，共享头显位姿数据
    - 调试场景，从开发机发送模拟位姿
    """

    def __init__(self, listen_port: int = 4243):
        super().__init__()
        self._listen_port = listen_port
        self._socket = None
        self._recv_thread = None
        self._timeout = 2.0

    @property
    def name(self) -> str:
        return "remote"

    def start(self) -> bool:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(("0.0.0.0", self._listen_port))
            self._socket.settimeout(1.0)
            self._running = True
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()
            print(f"HMD Provider: Remote listening on UDP port {self._listen_port}")
            return True
        except Exception as e:
            print(f"HMD Provider: Remote init failed: {e}")
            return False

    def stop(self):
        self._running = False
        if self._socket:
            self._socket.close()
        self._socket = None
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)

    def _recv_loop(self):
        """接收远程 HMD 位姿数据"""
        while self._running and self._socket:
            try:
                data, addr = self._socket.recvfrom(1024)
                if len(data) >= 56:  # 7 * 8 bytes
                    values = struct.unpack("7d", data[:56])
                    # 格式: x, y, z, qw, qx, qy, qz
                    position = [values[0], values[1], values[2]]
                    qw, qx, qy, qz = values[3], values[4], values[5], values[6]
                    # 四元数转欧拉角
                    try:
                        from scipy.spatial.transform import Rotation as Rot
                        rot = Rot.from_quat([qx, qy, qz, qw])  # scipy: [x,y,z,w]
                        euler = rot.as_euler('xyz', degrees=True)
                        self._update_pose(
                            position=position,
                            rotation=[float(euler[0]), float(euler[1]), float(euler[2])],
                            quaternion=[qw, qx, qy, qz],
                        )
                    except ImportError:
                        # scipy 不可用时只更新位置
                        self._update_pose(position=position)
            except socket.timeout:
                continue
            except Exception:
                pass

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        if pose.valid and (time.time() - pose.timestamp) > self._timeout:
            pose.valid = False
        return pose
