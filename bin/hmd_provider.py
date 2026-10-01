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


def _listen_host_warning(host: str) -> Optional[str]:
    """Warn when a UDP listener looks like a remote target address."""
    host = (host or "0.0.0.0").strip()
    if host in ("0.0.0.0", "::", "", "127.0.0.1", "localhost"):
        return None
    return (
        "UDP listener is bound to a specific address. Use 0.0.0.0 unless this "
        "is a local NIC address; put the headset/remote IP in the sender target."
    )


def _packet_rate_hz(packet_times, window: float = 5.0) -> float:
    now = time.time()
    recent = [ts for ts in packet_times if now - ts <= window]
    packet_times[:] = recent
    if len(recent) < 2:
        return 0.0
    span = max(recent[-1] - recent[0], 1e-6)
    return round((len(recent) - 1) / span, 2)


def _is_pose_stale(pose, timeout: float) -> bool:
    """Return whether a pose cannot currently be used as live HMD data.

    A provider that has not received its first packet has ``timestamp == 0``;
    that state must be reported as stale instead of looking healthy merely
    because there is no age to compare.  Explicitly invalidated poses are also
    stale from the dashboard's point of view, even when invalidation happened
    only moments ago.
    """
    if pose is None or not bool(getattr(pose, "valid", False)):
        return True
    try:
        timestamp = float(getattr(pose, "timestamp", 0.0) or 0.0)
        timeout = max(0.0, float(timeout))
    except (TypeError, ValueError):
        return True
    if timestamp <= 0.0 or not math.isfinite(timestamp):
        return True
    return (time.time() - timestamp) > timeout


FLOOR_HEIGHT_ORIGINS = {
    "floor",
    "stage",
    "standing",
    "local_floor",
    "local-floor",
    "steamvr",
    "openvr",
    "trackinguniverse_standing",
}
MAX_PLAUSIBLE_HMD_HEIGHT = 4.0


def _normalize_tracking_origin(origin) -> str:
    return str(origin or "unknown").strip().lower().replace(" ", "_")


def _actual_height_from_position(position, tracking_origin) -> tuple[Optional[float], str]:
    origin = _normalize_tracking_origin(tracking_origin)
    if origin not in FLOOR_HEIGHT_ORIGINS:
        return None, origin
    try:
        height = float(position[1])
    except (TypeError, ValueError, IndexError):
        return None, origin
    if not math.isfinite(height) or height <= 0.0 or height > MAX_PLAUSIBLE_HMD_HEIGHT:
        return None, origin
    return height, origin


def _coerce_explicit_actual_height(value) -> Optional[float]:
    try:
        height = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(height) or height <= 0.0 or height > MAX_PLAUSIBLE_HMD_HEIGHT:
        return None
    return height


def _coerce_finite_vector(value, length: int) -> Optional[list]:
    if isinstance(value, (str, bytes, bytearray)):
        return None
    if not isinstance(value, (list, tuple)) and not hasattr(value, "__getitem__"):
        return None
    try:
        result = [float(value[index]) for index in range(length)]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _openvr_rotation_to_unity(matrix):
    if np is None:
        raise RuntimeError("numpy is required for OpenVR coordinate conversion")
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(rotation)):
        raise ValueError("OpenVR rotation matrix contains non-finite values")
    basis = np.diag([1.0, 1.0, -1.0])
    return basis @ rotation @ basis


@dataclass
class HMDPose:
    """HMD 位姿数据"""
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [x, y, z] 米
    rotation: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # [pitch, yaw, roll] 度
    quaternion: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
    timestamp: float = 0.0
    valid: bool = False
    tracking_origin: str = "unknown"
    actual_height: Optional[float] = None
    height_reference: str = "unknown"


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
                tracking_origin=self._pose.tracking_origin,
                actual_height=self._pose.actual_height,
                height_reference=self._pose.height_reference,
            )

    def _update_pose(self, position, rotation=None, quaternion=None, tracking_origin=None):
        """更新位姿数据 (内部使用)"""
        clean_position = _coerce_finite_vector(position, 3)
        clean_rotation = (
            _coerce_finite_vector(rotation, 3) if rotation is not None else None
        )
        clean_quaternion = (
            _coerce_finite_vector(quaternion, 4) if quaternion is not None else None
        )
        if (
            clean_position is None
            or (rotation is not None and clean_rotation is None)
            or (quaternion is not None and clean_quaternion is None)
        ):
            self._invalidate_pose()
            return False

        with self._lock:
            self._pose.position = clean_position
            if clean_rotation is not None:
                self._pose.rotation = clean_rotation
            if clean_quaternion is not None:
                self._pose.quaternion = clean_quaternion
            if tracking_origin is not None:
                self._pose.tracking_origin = _normalize_tracking_origin(tracking_origin)
            actual_height, height_reference = _actual_height_from_position(
                clean_position, self._pose.tracking_origin
            )
            self._pose.actual_height = actual_height
            self._pose.height_reference = height_reference
            self._pose.timestamp = time.time()
            self._pose.valid = True
        return True

    def _set_explicit_actual_height(self, value, tracking_origin=None) -> bool:
        height = _coerce_explicit_actual_height(value)
        if height is None:
            return False
        origin = _normalize_tracking_origin(
            tracking_origin if tracking_origin is not None else self._pose.tracking_origin
        )
        with self._lock:
            if not self._pose.valid:
                return False
            self._pose.actual_height = height
            self._pose.height_reference = origin if origin in FLOOR_HEIGHT_ORIGINS else "explicit"
        return True

    def _invalidate_pose(self):
        with self._lock:
            self._pose.valid = False
            self._pose.actual_height = None
            self._pose.height_reference = "unknown"
            self._pose.timestamp = time.time()

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
        self._timeout = 0.5
        self._last_error = None
        self._poll_interval = 0.005  # 200Hz 采集

    @property
    def name(self) -> str:
        return "openvr"

    def start(self) -> bool:
        try:
            import openvr
            # 检查 scipy 和 numpy 是否可用 (坐标变换必需)
            if np is None:
                print("HMD Provider: numpy not installed, run: uv sync")
                return False
            try:
                from scipy.spatial.transform import Rotation as Rot
            except ImportError:
                print("HMD Provider: scipy not installed, run: uv sync")
                return False
            openvr.init(openvr.VRApplication_Other)
            self._vr_system = openvr.VRSystem()
            self._running = True
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()
            print(f"HMD Provider: OpenVR initialized (SteamVR)")
            return True
        except ImportError:
            print("HMD Provider: openvr package not installed, run: uv sync")
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

                    # 坐标系变换: OpenVR (Z后) -> VRChat (Z前)
                    # 位置: Z 翻转
                    position = [pos_x, pos_y, -pos_z]

                    # 左右手坐标系转换必须同时变换旋转矩阵两侧。
                    from scipy.spatial.transform import Rotation as Rot
                    mat_rot = np.array(
                        [[float(mat[row][col]) for col in range(3)] for row in range(3)],
                        dtype=float,
                    )
                    mat_new = _openvr_rotation_to_unity(mat_rot)
                    rot_new = Rot.from_matrix(mat_new)
                    quat_new = rot_new.as_quat()  # [x, y, z, w]
                    euler_new = rot_new.as_euler('xyz', degrees=True)  # [x, y, z] 度

                    self._update_pose(
                        position=position,
                        rotation=[float(euler_new[0]), float(euler_new[1]), float(euler_new[2])],
                        quaternion=[float(quat_new[3]), float(quat_new[0]), float(quat_new[1]), float(quat_new[2])],
                        tracking_origin="steamvr",
                    )
                    self._last_error = None
                else:
                    self._invalidate_pose()
                    self._last_error = "OpenVR HMD pose is invalid"
            except Exception as e:
                self._last_error = str(e)
                pose = super().get_pose()
                if pose.valid and _is_pose_stale(pose, self._timeout):
                    self._invalidate_pose()
            time.sleep(self._poll_interval)

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        if _is_pose_stale(pose, self._timeout):
            pose.valid = False
            pose.actual_height = None
            pose.height_reference = "unknown"
        return pose

    def get_status_extra(self) -> Dict:
        pose = self.get_pose()
        return {
            "source": "openvr",
            "protocol": "openvr",
            "tracking_origin": "steamvr",
            "actual_height": pose.actual_height,
            "height_reference": pose.height_reference,
            "last_update_age": (
                max(0.0, time.time() - pose.timestamp) if pose.timestamp else None
            ),
            "stale": _is_pose_stale(pose, self._timeout),
            "last_error": self._last_error,
        }


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
        self._last_error = None
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
            print("HMD Provider: python-osc not installed, run: uv sync")
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
            position = _coerce_finite_vector(args, 3)
            if position is None:
                self._invalidate_pose()
                self._last_error = "VRChat OSC position contains invalid values"
                return
            self._update_pose(position, tracking_origin="vrchat_osc")
            self._last_error = None

    def _on_head_rotation(self, address, *args):
        """接收 VRChat 头部旋转 (欧拉角, 度)"""
        if len(args) >= 3:
            rotation = _coerce_finite_vector(args, 3)
            if rotation is None:
                self._invalidate_pose()
                self._last_error = "VRChat OSC rotation contains invalid values"
                return
            with self._lock:
                self._pose.rotation = rotation

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        # 超时检查
        if pose.valid and (time.time() - pose.timestamp) > self._timeout:
            pose.valid = False
            pose.actual_height = None
            pose.height_reference = "unknown"
        return pose

    def get_status_extra(self) -> Dict:
        pose = self.get_pose()
        return {
            "source": "vrchat_osc",
            "protocol": "osc",
            "tracking_origin": "vrchat_osc",
            "actual_height": None,
            "height_reference": "unknown",
            "last_update_age": (
                max(0.0, time.time() - pose.timestamp) if pose.timestamp else None
            ),
            "stale": _is_pose_stale(pose, self._timeout),
            "last_error": self._last_error,
            "listen_host": "0.0.0.0",
            "listen_port": self._listen_port,
        }


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
        self._timeout = 1.0
        self._last_error = None
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
                        pos = data.get("position")
                        rot = data.get("rotation", [0, 0, 0])
                        origin = data.get("tracking_origin", "unknown")
                        if self._update_pose(position=pos, rotation=rot, tracking_origin=origin):
                            self._set_explicit_actual_height(data.get("actual_height"), origin)
                            self._last_error = None
                    else:
                        self._invalidate_pose()
                        self._last_error = "Pico Streaming returned an invalid HMD pose"
            except Exception as e:
                # Pico Streaming API 不可用，静默失败
                self._last_error = str(e)
                pose = super().get_pose()
                if pose.valid and _is_pose_stale(pose, self._timeout):
                    self._invalidate_pose()
            time.sleep(self._poll_interval)

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        if _is_pose_stale(pose, self._timeout):
            pose.valid = False
            pose.actual_height = None
            pose.height_reference = "unknown"
        return pose

    def get_status_extra(self) -> Dict:
        pose = self.get_pose()
        return {
            "source": "pico_streaming",
            "protocol": "http",
            "tracking_origin": pose.tracking_origin,
            "actual_height": pose.actual_height,
            "height_reference": pose.height_reference,
            "last_update_age": (
                max(0.0, time.time() - pose.timestamp) if pose.timestamp else None
            ),
            "stale": _is_pose_stale(pose, self._timeout),
            "last_error": self._last_error,
            "remote_host": self._host,
            "remote_port": self._port,
        }


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
        self._timeout = 1.0
        self._last_error = None
        self._poll_interval = 0.01
        self._openvr_fallback = None

    @property
    def name(self) -> str:
        return "quest_link"

    def start(self) -> bool:
        # 优先尝试 OpenVR (Quest Link 通过 SteamVR 注册)
        try:
            import openvr
            openvr_provider = OpenVRProvider()
            if openvr_provider.start():
                self._openvr_fallback = openvr_provider
                self._running = True
                print("HMD Provider: Quest Link via OpenVR (SteamVR)")
                return True
            openvr_provider.stop()
            self._openvr_fallback = None
        except Exception as e:
            self._openvr_fallback = None
            self._last_error = str(e)

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
            self._openvr_fallback = None
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)

    def get_pose(self) -> HMDPose:
        if self._openvr_fallback:
            return self._openvr_fallback.get_pose()
        pose = super().get_pose()
        if _is_pose_stale(pose, self._timeout):
            pose.valid = False
            pose.actual_height = None
            pose.height_reference = "unknown"
        return pose

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
                        pos = data.get("position")
                        rot = data.get("rotation", [0, 0, 0])
                        origin = data.get("tracking_origin", "unknown")
                        if self._update_pose(position=pos, rotation=rot, tracking_origin=origin):
                            self._set_explicit_actual_height(data.get("actual_height"), origin)
                            self._last_error = None
                    else:
                        self._invalidate_pose()
                        self._last_error = "Quest Link returned an invalid HMD pose"
            except Exception as e:
                self._last_error = str(e)
                pose = super().get_pose()
                if pose.valid and _is_pose_stale(pose, self._timeout):
                    self._invalidate_pose()
            time.sleep(self._poll_interval)

    def get_status_extra(self) -> Dict:
        if self._openvr_fallback and hasattr(self._openvr_fallback, "get_status_extra"):
            extra = self._openvr_fallback.get_status_extra()
            return {**extra, "source": "quest_link_openvr"}
        pose = self.get_pose()
        return {
            "source": "quest_link",
            "protocol": "http",
            "tracking_origin": pose.tracking_origin,
            "actual_height": pose.actual_height,
            "height_reference": pose.height_reference,
            "last_update_age": (
                max(0.0, time.time() - pose.timestamp) if pose.timestamp else None
            ),
            "stale": _is_pose_stale(pose, self._timeout),
            "last_error": self._last_error,
            "remote_host": self._host,
            "remote_port": self._port,
        }


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
        self._last_sender = None
        self._last_protocol = None
        self._packet_count = 0
        self._packet_times = []
        self._last_sequence = None
        self._sequence_gap_count = 0
        self._last_error = None
        self._last_source = "pico_xrobo"
        self._tracking_origin = "unknown"

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
                self._last_sender = f"{addr[0]}:{addr[1]}"

                # 尝试 JSON 格式
                if data[0:1] == b'{':
                    try:
                        j = json.loads(data.decode('utf-8'))
                        if not isinstance(j, dict):
                            self._invalidate_pose()
                            self._last_error = "JSON packet root must be an object"
                            continue
                        self._last_source = str(j.get("source") or self._last_source)
                        packet_origin = _normalize_tracking_origin(j.get("tracking_origin"))
                        self._tracking_origin = packet_origin
                        self._mark_packet("json", j.get("sequence"))
                        if j.get("valid", True) is False:
                            self._invalidate_pose()
                            self._last_error = None
                            continue
                        if j.get("valid", True):
                            pos = _coerce_finite_vector(j.get("position"), 3)
                            if pos is None:
                                self._invalidate_pose()
                                self._last_error = "JSON packet missing or invalid position[3]"
                                continue
                            quat = j.get("quaternion", j.get("rotation", [0, 0, 0, 1]))
                            quat = _coerce_finite_vector(quat, 4)
                            if quat is None:
                                self._invalidate_pose()
                                self._last_error = "JSON packet missing or invalid quaternion[4]"
                                continue
                            # rotation 可能是四元数 [x,y,z,w] 或 [w,x,y,z]
                            if len(quat) == 4:
                                # 假设 [x,y,z,w] 格式 (Unity 风格)
                                if "quaternion" in j:
                                    qw, qx, qy, qz = quat
                                else:
                                    qx, qy, qz, qw = quat
                                updated = self._update_pose_from_quat(
                                    pos, qx, qy, qz, qw, tracking_origin=packet_origin
                                )
                            else:
                                updated = self._update_pose(position=pos, tracking_origin=packet_origin)
                            if not updated:
                                continue
                            self._set_explicit_actual_height(
                                j.get("actual_height"), packet_origin
                            )
                            self._last_error = None
                        continue
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        self._invalidate_pose()
                        self._last_error = f"Invalid JSON packet: {e}"
                        continue

                # XRoboToolkit 二进制格式: header(4) + timestamp(8) + pose(28)
                if len(data) >= 40 and data[:4] == b'PICO':
                    # 跳过 header(4) + timestamp(8) = 12 字节
                    pose_data = data[12:40]  # 7 floats = 28 bytes
                    if len(pose_data) == 28:
                        values = struct.unpack("7f", pose_data)
                        px, py, pz = values[0], values[1], values[2]
                        qx, qy, qz, qw = values[3], values[4], values[5], values[6]
                        self._update_pose_from_quat(
                            [px, py, pz], qx, qy, qz, qw, tracking_origin="unknown"
                        )
                        self._mark_packet("pico_xrobo")
                        self._last_error = None
                    continue

                # 纯二进制格式: 7个float32 (px, py, pz, qx, qy, qz, qw)
                if len(data) >= 28:
                    values = struct.unpack("7f", data[:28])
                    px, py, pz = values[0], values[1], values[2]
                    qx, qy, qz, qw = values[3], values[4], values[5], values[6]
                    self._update_pose_from_quat(
                        [px, py, pz], qx, qy, qz, qw, tracking_origin="unknown"
                    )
                    self._mark_packet("legacy_float")
                    self._last_error = None
                    continue

            except socket.timeout:
                continue
            except Exception as e:
                self._last_error = str(e)

    def _update_pose_from_quat(
        self, position, qx, qy, qz, qw, tracking_origin="unknown"
    ):
        """从四元数更新位姿 (Unity 坐标系: X右, Y上, Z前)"""
        self._tracking_origin = _normalize_tracking_origin(tracking_origin)
        try:
            from scipy.spatial.transform import Rotation as Rot
            rot = Rot.from_quat([qx, qy, qz, qw])  # scipy: [x,y,z,w]
            euler = rot.as_euler('xyz', degrees=True)
            return self._update_pose(
                position=position,
                rotation=[float(euler[0]), float(euler[1]), float(euler[2])],
                quaternion=[float(qw), float(qx), float(qy), float(qz)],
                tracking_origin=self._tracking_origin,
            )
        except ImportError:
            return self._update_pose(
                position=position,
                quaternion=[qw, qx, qy, qz],
                tracking_origin=self._tracking_origin,
            )
        except Exception as e:
            self._invalidate_pose()
            self._last_error = str(e)
            return False

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        if pose.valid and (time.time() - pose.timestamp) > self._timeout:
            pose.valid = False
            pose.actual_height = None
            pose.height_reference = "unknown"
        return pose

    def get_status_extra(self) -> Dict:
        pose = self.get_pose()
        age = None
        if pose.timestamp:
            age = max(0.0, time.time() - pose.timestamp)
        return {
            "source": self._last_source,
            "protocol": self._last_protocol,
            "tracking_origin": self._tracking_origin,
            "actual_height": pose.actual_height,
            "height_reference": pose.height_reference,
            "last_sender": self._last_sender,
            "last_update_age": age,
            "stale": _is_pose_stale(pose, self._timeout),
            "packet_count": self._packet_count,
            "packet_rate_hz": _packet_rate_hz(self._packet_times),
            "last_sequence": self._last_sequence,
            "sequence_gap_count": self._sequence_gap_count,
            "listen_host": self._host,
            "listen_port": self._port,
            "listen_warning": _listen_host_warning(self._host),
            "last_error": self._last_error,
        }

    def _mark_packet(self, protocol: str, sequence=None):
        self._last_protocol = protocol
        self._packet_count += 1
        self._packet_times.append(time.time())
        if sequence is not None:
            try:
                seq = int(sequence)
                if self._last_sequence is not None and seq != self._last_sequence + 1:
                    self._sequence_gap_count += 1
                self._last_sequence = seq
            except (TypeError, ValueError):
                self._last_sequence = sequence


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
        self._last_sender = None
        self._last_protocol = None
        self._packet_count = 0
        self._packet_times = []
        self._last_sequence = None
        self._sequence_gap_count = 0
        self._last_error = None
        self._last_source = "quest_udp"
        self._tracking_origin = "unknown"

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
                self._last_sender = f"{addr[0]}:{addr[1]}"

                # 尝试 JSON 格式
                if data[0:1] == b'{':
                    try:
                        j = json.loads(data.decode('utf-8'))
                        if not isinstance(j, dict):
                            self._invalidate_pose()
                            self._last_error = "JSON packet root must be an object"
                            continue
                        self._last_source = str(j.get("source") or self._last_source)
                        packet_origin = _normalize_tracking_origin(j.get("tracking_origin"))
                        self._tracking_origin = packet_origin
                        self._mark_packet("json", j.get("sequence"))
                        if j.get("valid", True) is False:
                            self._invalidate_pose()
                            self._last_error = None
                            continue
                        if j.get("valid", True):
                            pos = _coerce_finite_vector(j.get("position"), 3)
                            if pos is None:
                                self._invalidate_pose()
                                self._last_error = "JSON packet missing or invalid position[3]"
                                continue
                            quat = j.get("quaternion", j.get("rotation", [0, 0, 0, 1]))
                            quat = _coerce_finite_vector(quat, 4)
                            if quat is None:
                                self._invalidate_pose()
                                self._last_error = "JSON packet missing or invalid quaternion[4]"
                                continue
                            if len(quat) == 4:
                                if "quaternion" in j:
                                    qw, qx, qy, qz = quat
                                else:
                                    qx, qy, qz, qw = quat
                                updated = self._update_pose_from_quat(
                                    pos, qx, qy, qz, qw, tracking_origin=packet_origin
                                )
                            else:
                                updated = self._update_pose(position=pos, tracking_origin=packet_origin)
                            if not updated:
                                continue
                            self._set_explicit_actual_height(
                                j.get("actual_height"), packet_origin
                            )
                            self._last_error = None
                        continue
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        self._invalidate_pose()
                        self._last_error = f"Invalid JSON packet: {e}"
                        continue

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
                        self._update_pose_from_quat(
                            [px, py, pz], qx, qy, qz, qw, tracking_origin="unknown"
                        )
                        self._mark_packet("quest_handtracking")
                        self._last_error = None
                    continue

                # 简化二进制格式: 7个float32 (px, py, pz, qx, qy, qz, qw)
                if len(data) >= 28:
                    values = struct.unpack("7f", data[:28])
                    px, py, pz = values[0], values[1], values[2]
                    qx, qy, qz, qw = values[3], values[4], values[5], values[6]
                    self._update_pose_from_quat(
                        [px, py, pz], qx, qy, qz, qw, tracking_origin="unknown"
                    )
                    self._mark_packet("legacy_float")
                    self._last_error = None
                    continue

            except socket.timeout:
                continue
            except Exception as e:
                self._last_error = str(e)

    def _update_pose_from_quat(
        self, position, qx, qy, qz, qw, tracking_origin="unknown"
    ):
        """从四元数更新位姿 (Unity 坐标系: X右, Y上, Z前)"""
        self._tracking_origin = _normalize_tracking_origin(tracking_origin)
        try:
            from scipy.spatial.transform import Rotation as Rot
            rot = Rot.from_quat([qx, qy, qz, qw])  # scipy: [x,y,z,w]
            euler = rot.as_euler('xyz', degrees=True)
            return self._update_pose(
                position=position,
                rotation=[float(euler[0]), float(euler[1]), float(euler[2])],
                quaternion=[float(qw), float(qx), float(qy), float(qz)],
                tracking_origin=self._tracking_origin,
            )
        except ImportError:
            return self._update_pose(
                position=position,
                quaternion=[qw, qx, qy, qz],
                tracking_origin=self._tracking_origin,
            )
        except Exception as e:
            self._invalidate_pose()
            self._last_error = str(e)
            return False

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        if pose.valid and (time.time() - pose.timestamp) > self._timeout:
            pose.valid = False
            pose.actual_height = None
            pose.height_reference = "unknown"
        return pose

    def get_status_extra(self) -> Dict:
        pose = self.get_pose()
        age = None
        if pose.timestamp:
            age = max(0.0, time.time() - pose.timestamp)
        return {
            "source": self._last_source,
            "protocol": self._last_protocol,
            "tracking_origin": self._tracking_origin,
            "actual_height": pose.actual_height,
            "height_reference": pose.height_reference,
            "last_sender": self._last_sender,
            "last_update_age": age,
            "stale": _is_pose_stale(pose, self._timeout),
            "packet_count": self._packet_count,
            "packet_rate_hz": _packet_rate_hz(self._packet_times),
            "last_sequence": self._last_sequence,
            "sequence_gap_count": self._sequence_gap_count,
            "listen_host": self._host,
            "listen_port": self._port,
            "listen_warning": _listen_host_warning(self._host),
            "last_error": self._last_error,
        }

    def _mark_packet(self, protocol: str, sequence=None):
        self._last_protocol = protocol
        self._packet_count += 1
        self._packet_times.append(time.time())
        if sequence is not None:
            try:
                seq = int(sequence)
                if self._last_sequence is not None and seq != self._last_sequence + 1:
                    self._sequence_gap_count += 1
                self._last_sequence = seq
            except (TypeError, ValueError):
                self._last_sequence = sequence


class SDKUDPProvider(HMDDataProvider):
    """Generic headset SDK UDP provider.

    This provider is for Pico/Quest/Unity OpenXR sender apps running on the
    headset. It receives Unity-coordinate HMD poses over UDP and exposes them
    in the project-wide VRChat/Unity convention: X right, Y up, Z forward.
    """

    def __init__(
        self,
        source_name: str = "unity_xr_sdk",
        host: str = "0.0.0.0",
        port: int = 39542,
        tracking_origin: str = "unknown",
    ):
        super().__init__()
        self._source_name = source_name
        self._host = host
        self._port = int(port)
        self._configured_tracking_origin = _normalize_tracking_origin(tracking_origin)
        self._tracking_origin = "unknown"
        self._socket = None
        self._recv_thread = None
        self._timeout = 2.0
        self._last_sender = None
        self._last_source = source_name
        self._last_sender_timestamp = None
        self._last_error = None
        self._last_protocol = None
        self._packet_count = 0
        self._packet_times = []
        self._last_sequence = None
        self._sequence_gap_count = 0

    @property
    def name(self) -> str:
        return self._source_name

    def start(self) -> bool:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self._host, self._port))
            self._socket.settimeout(1.0)
            self._running = True
            self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self._recv_thread.start()
            print(f"HMD Provider: {self._source_name} SDK UDP listening on {self._host}:{self._port}")
            return True
        except Exception as e:
            print(f"HMD Provider: {self._source_name} SDK UDP init failed: {e}")
            return False

    def stop(self):
        self._running = False
        if self._socket:
            self._socket.close()
        self._socket = None
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        if pose.valid and (time.time() - pose.timestamp) > self._timeout:
            pose.valid = False
            pose.actual_height = None
            pose.height_reference = "unknown"
        return pose

    def get_status_extra(self) -> Dict:
        pose = self.get_pose()
        age = None
        if pose.timestamp:
            age = max(0.0, time.time() - pose.timestamp)
        return {
            "source": self._last_source,
            "protocol": self._last_protocol,
            "tracking_origin": self._tracking_origin,
            "configured_tracking_origin": self._configured_tracking_origin,
            "actual_height": pose.actual_height,
            "height_reference": pose.height_reference,
            "last_sender": self._last_sender,
            "last_sender_timestamp": self._last_sender_timestamp,
            "last_update_age": age,
            "stale": _is_pose_stale(pose, self._timeout),
            "packet_count": self._packet_count,
            "packet_rate_hz": _packet_rate_hz(self._packet_times),
            "last_sequence": self._last_sequence,
            "sequence_gap_count": self._sequence_gap_count,
            "listen_host": self._host,
            "listen_port": self._port,
            "listen_warning": _listen_host_warning(self._host),
            "last_error": self._last_error,
            "sdk_udp_host": self._host,
            "sdk_udp_port": self._port,
        }

    def _recv_loop(self):
        while self._running and self._socket:
            try:
                data, addr = self._socket.recvfrom(4096)
                if not data:
                    continue
                self._last_sender = f"{addr[0]}:{addr[1]}"

                if data[:1] == b"{":
                    self._parse_json_packet(data)
                    continue

                if len(data) >= 56:
                    values = struct.unpack("<7d", data[:56])
                    # Legacy simplified double packet: x, y, z, qw, qx, qy, qz.
                    position = [values[0], values[1], values[2]]
                    self._update_pose_from_wxyz(
                        position,
                        [values[3], values[4], values[5], values[6]],
                        tracking_origin="unknown",
                    )
                    self._mark_packet("legacy_double")
                    continue

                if len(data) >= 28:
                    values = struct.unpack("<7f", data[:28])
                    # Unity-style float packet: x, y, z, qx, qy, qz, qw.
                    position = [values[0], values[1], values[2]]
                    self._update_pose_from_wxyz(
                        position,
                        [values[6], values[3], values[4], values[5]],
                        tracking_origin="unknown",
                    )
                    self._mark_packet("legacy_float")
                    continue
            except socket.timeout:
                continue
            except Exception as e:
                self._last_error = str(e)

    def _parse_json_packet(self, data: bytes) -> bool:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._invalidate_pose()
            self._last_error = f"Invalid JSON packet: {e}"
            return False
        if not isinstance(payload, dict):
            self._invalidate_pose()
            self._last_error = "JSON packet root must be an object"
            return False

        self._last_source = str(payload.get("source") or self._source_name)
        packet_origin = _normalize_tracking_origin(payload.get("tracking_origin"))
        self._tracking_origin = packet_origin
        self._last_sender_timestamp = payload.get("timestamp")
        self._mark_packet("json", payload.get("sequence"))

        if payload.get("valid", True) is False:
            self._invalidate_pose()
            return True

        position = self._as_float_list(payload.get("position"), 3)
        if position is None:
            self._invalidate_pose()
            self._last_error = "JSON packet missing or invalid position[3]"
            return True

        quat_wxyz = None
        rotation_euler = None

        if "quaternion" in payload:
            quat_wxyz = self._as_quaternion_wxyz(payload.get("quaternion"))
            if quat_wxyz is None:
                self._invalidate_pose()
                self._last_error = "JSON packet has invalid quaternion[4]"
                return True
        elif "rotation" in payload:
            rotation = self._as_any_float_list(payload.get("rotation"))
            if rotation and len(rotation) == 4:
                # Unity JSON compatibility: rotation is qx, qy, qz, qw.
                quat_wxyz = self._normalize_quaternion([rotation[3], rotation[0], rotation[1], rotation[2]])
            elif rotation and len(rotation) == 3:
                rotation_euler = rotation
            else:
                self._invalidate_pose()
                self._last_error = "JSON packet has invalid rotation"
                return True

        if quat_wxyz is not None:
            updated = self._update_pose_from_wxyz(
                position, quat_wxyz, tracking_origin=packet_origin
            )
        else:
            updated = self._update_pose(
                position=position,
                rotation=rotation_euler or [0.0, 0.0, 0.0],
                tracking_origin=packet_origin,
            )
        if not updated:
            self._last_error = "JSON packet contains non-finite pose values"
            return True
        self._set_explicit_actual_height(payload.get("actual_height"), packet_origin)
        self._last_error = None
        return True

    def _mark_packet(self, protocol: str, sequence=None):
        self._last_protocol = protocol
        self._packet_count += 1
        self._packet_times.append(time.time())
        if sequence is not None:
            try:
                seq = int(sequence)
                if self._last_sequence is not None and seq != self._last_sequence + 1:
                    self._sequence_gap_count += 1
                self._last_sequence = seq
            except (TypeError, ValueError):
                self._last_sequence = sequence

    def _update_pose_from_wxyz(
        self, position, quat_wxyz, tracking_origin="unknown"
    ):
        quat_wxyz = self._normalize_quaternion(quat_wxyz)
        clean_position = [float(position[0]), float(position[1]), float(position[2])]
        self._tracking_origin = _normalize_tracking_origin(tracking_origin)
        updated = self._update_pose(
            position=clean_position,
            rotation=[0.0, 0.0, 0.0],
            quaternion=quat_wxyz,
            tracking_origin=self._tracking_origin,
        )
        if not updated:
            return False
        rotation = self._quat_wxyz_to_euler(quat_wxyz)
        with self._lock:
            self._pose.rotation = rotation
        return True

    @staticmethod
    def _as_any_float_list(value):
        if value is None:
            return None
        if isinstance(value, dict):
            keys = ("x", "y", "z", "w")
            items = [value[k] for k in keys if k in value]
        else:
            items = list(value) if isinstance(value, (list, tuple)) else None
        if items is None:
            return None
        try:
            result = [float(v) for v in items]
        except (TypeError, ValueError):
            return None
        return result if all(math.isfinite(v) for v in result) else None

    @classmethod
    def _as_float_list(cls, value, length: int):
        items = cls._as_any_float_list(value)
        if items is None or len(items) != length:
            return None
        return items

    @classmethod
    def _as_quaternion_wxyz(cls, value):
        if isinstance(value, dict):
            if all(k in value for k in ("w", "x", "y", "z")):
                try:
                    return cls._normalize_quaternion([value["w"], value["x"], value["y"], value["z"]])
                except (TypeError, ValueError):
                    return None
            if all(k in value for k in ("qw", "qx", "qy", "qz")):
                try:
                    return cls._normalize_quaternion([value["qw"], value["qx"], value["qy"], value["qz"]])
                except (TypeError, ValueError):
                    return None
        items = cls._as_float_list(value, 4)
        if items is None:
            return None
        return cls._normalize_quaternion(items)

    @staticmethod
    def _normalize_quaternion(quat_wxyz):
        qw, qx, qy, qz = [float(v) for v in quat_wxyz[:4]]
        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm <= 1e-8:
            return [1.0, 0.0, 0.0, 0.0]
        return [qw / norm, qx / norm, qy / norm, qz / norm]

    def _quat_wxyz_to_euler(self, quat_wxyz):
        qw, qx, qy, qz = quat_wxyz
        try:
            from scipy.spatial.transform import Rotation as Rot
            rot = Rot.from_quat([qx, qy, qz, qw])
            euler = rot.as_euler("xyz", degrees=True)
            self._last_error = None
            return [float(euler[0]), float(euler[1]), float(euler[2])]
        except Exception as e:
            self._last_error = str(e)
            return [0.0, 0.0, 0.0]


class AutoSDKProvider(HMDDataProvider):
    """Prefer OpenVR when available, otherwise receive headset SDK UDP pose."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 39542,
        tracking_origin: str = "unknown",
        prefer_openvr: bool = True,
        source_name: str = "unity_xr_sdk",
    ):
        super().__init__()
        self._prefer_openvr = bool(prefer_openvr)
        self._sdk_provider = SDKUDPProvider(source_name, host, port, tracking_origin)
        self._active_provider = None

    @property
    def name(self) -> str:
        return "auto_sdk"

    def start(self) -> bool:
        if self._prefer_openvr:
            openvr_provider = OpenVRProvider()
            if openvr_provider.start():
                self._active_provider = openvr_provider
                self._running = True
                print("HMD Provider: auto_sdk selected OpenVR")
                return True

        if self._sdk_provider.start():
            self._active_provider = self._sdk_provider
            self._running = True
            print("HMD Provider: auto_sdk selected SDK UDP")
            return True
        return False

    def stop(self):
        self._running = False
        if self._active_provider:
            self._active_provider.stop()
        self._active_provider = None

    def get_pose(self) -> HMDPose:
        if self._active_provider:
            return self._active_provider.get_pose()
        return super().get_pose()

    def get_status_extra(self) -> Dict:
        extra = {
            "active_provider": self._active_provider.name if self._active_provider else "none",
            "prefer_openvr": self._prefer_openvr,
        }
        if self._active_provider and hasattr(self._active_provider, "get_status_extra"):
            extra.update(self._active_provider.get_status_extra())
        else:
            extra.setdefault("source", "openvr" if self._active_provider else "none")
            extra.setdefault("tracking_origin", "unknown")
        return extra


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
            "sdk_udp": SDKUDPProvider,
            "pico_sdk": SDKUDPProvider,
            "quest_sdk": SDKUDPProvider,
            "unity_xr_sdk": SDKUDPProvider,
            "auto_sdk": AutoSDKProvider,
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
        elif provider_type in ("sdk_udp", "pico_sdk", "quest_sdk", "unity_xr_sdk"):
            return provider_class(source_name="sdk_udp",
                                  host=kwargs.get("sdk_udp_host", "0.0.0.0"),
                                  port=kwargs.get("sdk_udp_port", 39542),
                                  tracking_origin=kwargs.get("tracking_origin", "unknown"))
        elif provider_type == "auto_sdk":
            return provider_class(host=kwargs.get("sdk_udp_host", "0.0.0.0"),
                                  port=kwargs.get("sdk_udp_port", 39542),
                                  tracking_origin=kwargs.get("tracking_origin", "unknown"),
                                  prefer_openvr=kwargs.get("prefer_openvr", True),
                                  source_name=kwargs.get("sdk_source_name", "sdk_udp"))
        elif provider_type == "dashboard_api":
            return provider_class(
                host=kwargs.get("dashboard_host", "127.0.0.1"),
                port=kwargs.get("dashboard_port", 8766),
                poll_fps=kwargs.get("dashboard_poll_fps", 15.0),
            )
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

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8766,
        poll_fps: float = 15.0,
    ):
        super().__init__()
        self._host = host
        self._port = port
        self._poll_thread = None
        try:
            poll_fps = float(poll_fps)
        except (TypeError, ValueError):
            poll_fps = 15.0
        self._poll_fps = min(30.0, max(1.0, poll_fps))
        self._poll_interval = 1.0 / self._poll_fps
        self._timeout = 0.75
        self._api_url = f"http://{host}:{port}/api/hmd/status"
        self._last_success_time = 0.0
        self._last_error = None
        self._source_provider = "none"

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
                req.timeout = min(0.25, self._timeout)
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode())
                    self._apply_status_payload(data)
                    self._last_success_time = time.time()
                    self._last_error = None
            except Exception as e:
                self._last_error = str(e)
                if time.time() - self._last_success_time > self._timeout:
                    self._mark_invalid()
            time.sleep(self._poll_interval)

    def _apply_status_payload(self, data: Dict):
        self._source_provider = str(data.get("provider") or "none")
        pose = data.get("pose") if isinstance(data, dict) else None
        if not data.get("running") or not isinstance(pose, dict) or not pose.get("valid"):
            self._mark_invalid()
            return

        position = pose.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            self._mark_invalid()
            return

        tracking_origin = pose.get(
            "tracking_origin", data.get("tracking_origin", "unknown")
        )
        if not self._update_pose(
            position=position,
            rotation=pose.get("rotation", [0.0, 0.0, 0.0]),
            quaternion=pose.get("quaternion", [1.0, 0.0, 0.0, 0.0]),
            tracking_origin=tracking_origin,
        ):
            self._mark_invalid()
            return

        actual_height = _coerce_explicit_actual_height(pose.get("actual_height"))
        if actual_height is not None:
            height_reference = _normalize_tracking_origin(
                pose.get("height_reference") or tracking_origin
            )
            if height_reference == "unknown":
                height_reference = "explicit"
            with self._lock:
                self._pose.actual_height = actual_height
                self._pose.height_reference = height_reference

    def _mark_invalid(self):
        with self._lock:
            self._pose.valid = False
            self._pose.actual_height = None
            self._pose.height_reference = "unknown"

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        if pose.valid and time.time() - pose.timestamp > self._timeout:
            pose.valid = False
            pose.actual_height = None
            pose.height_reference = "unknown"
        return pose

    def get_status_extra(self) -> Dict:
        pose = self.get_pose()
        return {
            "source": "dashboard_api",
            "source_provider": self._source_provider,
            "dashboard_host": self._host,
            "dashboard_port": self._port,
            "poll_fps": self._poll_fps,
            "last_update_age": max(0.0, time.time() - pose.timestamp) if pose.timestamp else None,
            "stale": not pose.valid,
            "last_error": self._last_error,
        }


class RemoteHMDProvider(HMDDataProvider):
    """Remote HMD pose provider.

    Supports the legacy 7-double UDP packet used by older HMD-Remote-Sender
    builds and the newer JSON packet with source/timestamp/sequence metadata.
    Output pose is always VRChat/Unity coordinates: X right, Y up, Z forward.
    """

    def __init__(self, listen_port: int = 4243):
        super().__init__()
        self._listen_port = listen_port
        self._socket = None
        self._recv_thread = None
        self._timeout = 2.0
        self._last_sender = None
        self._last_sender_timestamp = None
        self._last_sequence = None
        self._packet_count = 0
        self._last_error = None
        self._source = "remote"
        self._tracking_origin = "unknown"
        self._last_protocol = None
        self._packet_times = []
        self._sequence_gap_count = 0

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
        while self._running and self._socket:
            try:
                data, addr = self._socket.recvfrom(4096)
                if not data:
                    continue
                self._last_sender = f"{addr[0]}:{addr[1]}"

                if data[:1] == b"{":
                    self._parse_json_packet(data)
                    continue

                if len(data) >= 56:
                    values = struct.unpack("<7d", data[:56])
                    position = [values[0], values[1], values[2]]
                    quat = [values[3], values[4], values[5], values[6]]
                    self._update_pose_from_quat(
                        position, quat, tracking_origin="unknown"
                    )
                    self._mark_packet("legacy_double")
                    self._source = "remote_legacy"
                    self._last_error = None
            except socket.timeout:
                continue
            except Exception as e:
                self._last_error = str(e)

    def _parse_json_packet(self, data: bytes) -> bool:
        try:
            payload = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._invalidate_pose()
            self._last_error = f"Invalid JSON packet: {e}"
            return False
        if not isinstance(payload, dict):
            self._invalidate_pose()
            self._last_error = "JSON packet root must be an object"
            return False
        self._source = str(payload.get("source") or "remote")
        packet_origin = _normalize_tracking_origin(payload.get("tracking_origin"))
        self._tracking_origin = packet_origin
        self._last_sender_timestamp = payload.get("timestamp")
        self._mark_packet("json", payload.get("sequence"))

        if payload.get("valid", True) is False:
            self._invalidate_pose()
            self._last_error = None
            return True

        position = payload.get("position")
        quaternion = payload.get("quaternion")
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            self._invalidate_pose()
            self._last_error = "JSON packet missing position[3]"
            return False
        quaternion = self._coerce_quaternion(quaternion)
        if quaternion is None:
            self._invalidate_pose()
            self._last_error = "JSON packet missing quaternion[4]"
            return False
        clean_position = _coerce_finite_vector(position, 3)
        if clean_position is None:
            self._invalidate_pose()
            self._last_error = "JSON packet contains non-finite position values"
            return False
        if not self._update_pose_from_quat(
            clean_position,
            quaternion,
            tracking_origin=packet_origin,
        ):
            self._last_error = "JSON packet contains invalid pose values"
            return False
        self._set_explicit_actual_height(payload.get("actual_height"), packet_origin)
        self._last_error = None
        return True

    @staticmethod
    def _coerce_quaternion(value):
        if isinstance(value, dict):
            if all(k in value for k in ("w", "x", "y", "z")):
                value = [value["w"], value["x"], value["y"], value["z"]]
            elif all(k in value for k in ("qw", "qx", "qy", "qz")):
                value = [value["qw"], value["qx"], value["qy"], value["qz"]]
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            quaternion = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return None
        return quaternion if all(math.isfinite(item) for item in quaternion) else None

    def _update_pose_from_quat(
        self, position, quat_wxyz, tracking_origin="unknown"
    ):
        self._tracking_origin = _normalize_tracking_origin(tracking_origin)
        qw, qx, qy, qz = self._normalize_quaternion(quat_wxyz)
        try:
            from scipy.spatial.transform import Rotation as Rot
            rot = Rot.from_quat([qx, qy, qz, qw])
            euler = rot.as_euler('xyz', degrees=True)
            rotation = [float(euler[0]), float(euler[1]), float(euler[2])]
        except Exception as e:
            rotation = [0.0, 0.0, 0.0]
            self._last_error = str(e)
        return self._update_pose(
            position=[float(position[0]), float(position[1]), float(position[2])],
            rotation=rotation,
            quaternion=[qw, qx, qy, qz],
            tracking_origin=self._tracking_origin,
        )

    @staticmethod
    def _normalize_quaternion(quat_wxyz):
        qw, qx, qy, qz = [float(v) for v in quat_wxyz[:4]]
        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm <= 1e-8:
            return [1.0, 0.0, 0.0, 0.0]
        return [qw / norm, qx / norm, qy / norm, qz / norm]

    def get_pose(self) -> HMDPose:
        pose = super().get_pose()
        if pose.valid and (time.time() - pose.timestamp) > self._timeout:
            pose.valid = False
            pose.actual_height = None
            pose.height_reference = "unknown"
        return pose

    def get_status_extra(self) -> Dict:
        pose = self.get_pose()
        age = None
        if pose.timestamp:
            age = max(0.0, time.time() - pose.timestamp)
        return {
            "source": self._source,
            "protocol": self._last_protocol,
            "tracking_origin": self._tracking_origin,
            "actual_height": pose.actual_height,
            "height_reference": pose.height_reference,
            "last_sender": self._last_sender,
            "last_sender_timestamp": self._last_sender_timestamp,
            "last_sequence": self._last_sequence,
            "sequence_gap_count": self._sequence_gap_count,
            "last_update_age": age,
            "stale": _is_pose_stale(pose, self._timeout),
            "packet_rate_hz": _packet_rate_hz(self._packet_times),
            "packet_count": self._packet_count,
            "remote_port": self._listen_port,
            "listen_host": "0.0.0.0",
            "listen_port": self._listen_port,
            "listen_warning": None,
            "last_error": self._last_error,
        }

    def _mark_packet(self, protocol: str, sequence=None):
        self._last_protocol = protocol
        self._packet_count += 1
        self._packet_times.append(time.time())
        if sequence is not None:
            try:
                seq = int(sequence)
                if self._last_sequence is not None and seq != self._last_sequence + 1:
                    self._sequence_gap_count += 1
                self._last_sequence = seq
            except (TypeError, ValueError):
                self._last_sequence = sequence
