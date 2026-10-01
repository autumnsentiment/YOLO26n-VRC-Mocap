"""
坐标系变换工具

统一处理各坐标系之间的变换:
- 摄像头坐标系 (YOLO/MediaPipe 输出): X左, Y下, Z前, 髋部为原点, 米制单位
- Unity/VRChat 坐标系: X右, Y上, Z前
- OpenVR/SteamVR 坐标系: X右, Y上, Z后 (OpenGL 风格)
- VMT 坐标系: 与 SteamVR 一致

坐标参考点 (已验证):
- HMD 绝对世界坐标:
    - Quest SDK (Floor Level): Y = 玩家眼高 (地面=0), Unity风格 (X右,Y上,Z前)
    - Quest SDK (Eye Level): Y = 相对启动时HMD位置的偏移
    - Pico SDK (Stage/Floor): Y = 玩家眼高 (地面=0), Unity风格
    - Pico SDK (Local/Standing): Y ≈ 玩家眼高 (含站立偏移)
    - Pico SDK (Seated): Y = 相对眼高的偏移
- OSC 输出坐标: VRChat 发回的头部位置 (Y=玩家高度)
- OpenVR 坐标: Y = 玩家眼高 (地面=0)

重要: 所有 HMD 数据源在 Floor Level/Stage 模式下，Y 坐标均为玩家站立眼高。
      以此为参考，颈部位置 = HMD位置 + HMD到颈部偏移 (约 [0, -0.2, 0.1])。
"""

from collections import deque
import warnings

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from scipy.spatial.transform import Rotation as Rot
except ImportError:
    Rot = None


FLOOR_RELATIVE_HMD_ORIGINS = {
    "floor", "stage", "standing", "local_floor", "local-floor",
    "steamvr", "openvr", "trackinguniverse_standing",
}


def hmd_pose_anchor_trusted(pose) -> bool:
    """Whether an HMD pose has enough origin information for world anchoring."""
    if pose is None or not bool(getattr(pose, "valid", False)):
        return False
    try:
        position = [float(value) for value in getattr(pose, "position", [])[:3]]
    except (TypeError, ValueError, IndexError):
        return False
    if len(position) != 3 or not np.all(np.isfinite(position)):
        return False
    try:
        actual_height = float(getattr(pose, "actual_height", None))
    except (TypeError, ValueError):
        actual_height = None
    explicit_height_valid = bool(
        actual_height is not None
        and np.isfinite(actual_height)
        and 0.0 < actual_height <= 4.0
    )
    origin = str(getattr(pose, "tracking_origin", "unknown") or "unknown")
    origin = origin.strip().lower().replace(" ", "_")
    if origin in FLOOR_RELATIVE_HMD_ORIGINS:
        # Some SDKs report a floor origin but keep position.y in a local/device
        # frame. An explicit floor-relative height is authoritative in that case.
        return bool((0.0 < position[1] <= 4.0) or explicit_height_valid)
    return explicit_height_valid


class CoordinateTransformer:
    """坐标系变换器

    负责将动捕推理输出的坐标变换到目标坐标系，
    并基于 HMD 位置建立世界坐标系对齐。
    """

    # 翻转矩阵: 摄像头 (X左,Y下) -> Unity (X右,Y上)
    _FLIP_XY = np.diag([-1.0, -1.0, 1.0])

    # Unity 左手系 (Z前) <-> OpenVR 右手系 (Z后) 的基变换。
    _UNITY_OPENVR_BASIS = np.diag([1.0, 1.0, -1.0])

    DEFAULT_HMD_HEIGHT_SCALING = {
        "enabled": True,
        "floor_clearance": 0.03,
        "min_hmd_y": 0.35,
        "min_body_scale": 0.45,
        "max_body_scale": 1.0,
        "vertical_only": True,
        "require_actual_height": True,
        "crouch_from_hmd_height": True,
        "standing_eye_height": 1.65,
        "standing_eye_height_calibrated": False,
        "auto_learn_standing_height": True,
        "auto_learn_allow_downward": True,
        "auto_learn_downward_max_delta": 0.25,
        "auto_learn_standing_pose_threshold": 0.8,
        "min_standing_eye_height": 1.2,
        "max_standing_eye_height": 2.2,
        "auto_learn_min_height": 1.2,
        "auto_learn_window": 20,
        "auto_learn_stability_tolerance": 0.03,
        "crouch_start_ratio": 0.92,
        "seated_ratio": 0.55,
        "seated_eye_height": 0.95,
        "seated_hip_height": 0.45,
        "standing_hip_min_height": 0.78,
        "standing_chest_min_height": 1.05,
        "foot_grounding_enabled": True,
        "auto_calibrate_foot_reference": True,
        "foot_calibration_samples": 15,
        "foot_lift_gain": 1.0,
        "max_foot_lift": 0.35,
        "crouch_knee_forward": 0.18,
        "crouch_knee_outward": 0.06,
    }

    def __init__(
        self,
        hmd_to_neck_offset: List[float] = None,
        hmd_height_scaling: Optional[Dict] = None,
    ):
        """
        Args:
            hmd_to_neck_offset: HMD 到颈部的偏移 [x, y, z] (Unity 坐标系)
                                默认 [0, -0.2, 0.1] (颈部在 HMD 下方 0.2m, 前方 0.1m)
        """
        self._hmd_to_neck_offset = np.array(
            hmd_to_neck_offset or [0, -0.2, 0.1]
        )
        self._hmd_height_scaling = dict(self.DEFAULT_HMD_HEIGHT_SCALING)
        self._learned_standing_eye_height: Optional[float] = None
        self._standing_height_samples = deque(
            maxlen=int(self.DEFAULT_HMD_HEIGHT_SCALING["auto_learn_window"])
        )
        self._standing_foot_reference: Dict[str, float] = {}
        self._foot_reference_samples = {
            "left": deque(maxlen=int(self.DEFAULT_HMD_HEIGHT_SCALING["foot_calibration_samples"])),
            "right": deque(maxlen=int(self.DEFAULT_HMD_HEIGHT_SCALING["foot_calibration_samples"])),
        }
        self._last_hmd_height_state = {
            "enabled": True,
            "applied": False,
            "reason": "not_computed",
            "posture": "unknown",
            "actual_height": None,
            "standing_eye_height": None,
            "height_ratio": None,
            "crouch_amount": 0.0,
            "standing_pose_confidence": None,
            "foot_reference_ready": False,
        }
        self.set_hmd_height_scaling(hmd_height_scaling)
        self._hmd_provider = None

    @property
    def hmd_to_neck_offset(self) -> np.ndarray:
        return self._hmd_to_neck_offset

    @hmd_to_neck_offset.setter
    def hmd_to_neck_offset(self, value: List[float]):
        self._hmd_to_neck_offset = np.array(value)

    def set_hmd_height_scaling(self, config: Optional[Dict] = None):
        """Update seated/standing height constraint for HMD-bound body pose."""
        merged = dict(self.DEFAULT_HMD_HEIGHT_SCALING)
        if isinstance(config, dict):
            merged.update(config)
        merged["enabled"] = bool(merged.get("enabled", True))
        merged["vertical_only"] = bool(merged.get("vertical_only", True))
        merged["require_actual_height"] = bool(merged.get("require_actual_height", True))
        merged["crouch_from_hmd_height"] = bool(merged.get("crouch_from_hmd_height", True))
        merged["standing_eye_height_calibrated"] = bool(
            merged.get("standing_eye_height_calibrated", False)
        )
        for key in (
            "floor_clearance",
            "min_hmd_y",
            "min_body_scale",
            "max_body_scale",
            "crouch_knee_forward",
            "crouch_knee_outward",
            "standing_eye_height",
            "min_standing_eye_height",
            "max_standing_eye_height",
            "auto_learn_min_height",
            "auto_learn_downward_max_delta",
            "auto_learn_standing_pose_threshold",
            "auto_learn_stability_tolerance",
            "crouch_start_ratio",
            "seated_ratio",
            "seated_eye_height",
            "seated_hip_height",
            "standing_hip_min_height",
            "standing_chest_min_height",
            "foot_lift_gain",
            "max_foot_lift",
        ):
            try:
                merged[key] = float(merged.get(key, self.DEFAULT_HMD_HEIGHT_SCALING[key]))
            except (TypeError, ValueError):
                merged[key] = self.DEFAULT_HMD_HEIGHT_SCALING[key]
        try:
            merged["foot_calibration_samples"] = max(
                3, int(merged.get("foot_calibration_samples", 15))
            )
        except (TypeError, ValueError):
            merged["foot_calibration_samples"] = 15
        try:
            merged["auto_learn_window"] = max(
                5, int(merged.get("auto_learn_window", 20))
            )
        except (TypeError, ValueError):
            merged["auto_learn_window"] = 20
        merged["foot_grounding_enabled"] = bool(merged.get("foot_grounding_enabled", True))
        merged["auto_calibrate_foot_reference"] = bool(
            merged.get("auto_calibrate_foot_reference", True)
        )
        merged["auto_learn_standing_height"] = bool(
            merged.get("auto_learn_standing_height", True)
        )
        merged["auto_learn_allow_downward"] = bool(
            merged.get("auto_learn_allow_downward", True)
        )
        if merged["max_body_scale"] <= 0:
            merged["max_body_scale"] = self.DEFAULT_HMD_HEIGHT_SCALING["max_body_scale"]
        if merged["min_body_scale"] <= 0:
            merged["min_body_scale"] = self.DEFAULT_HMD_HEIGHT_SCALING["min_body_scale"]
        if merged["min_body_scale"] > merged["max_body_scale"]:
            merged["min_body_scale"] = merged["max_body_scale"]
        if merged["max_standing_eye_height"] < merged["min_standing_eye_height"]:
            merged["max_standing_eye_height"] = merged["min_standing_eye_height"]
        if merged["standing_eye_height"] < merged["min_standing_eye_height"]:
            merged["standing_eye_height"] = merged["min_standing_eye_height"]
        if merged["standing_eye_height"] > merged["max_standing_eye_height"]:
            merged["standing_eye_height"] = merged["max_standing_eye_height"]
        if merged["seated_ratio"] >= merged["crouch_start_ratio"]:
            merged["seated_ratio"] = max(0.1, merged["crouch_start_ratio"] - 0.25)
        merged["auto_learn_min_height"] = self._clamp(
            merged["auto_learn_min_height"],
            0.35,
            merged["max_standing_eye_height"],
        )
        merged["foot_lift_gain"] = max(0.0, merged["foot_lift_gain"])
        merged["max_foot_lift"] = max(0.0, merged["max_foot_lift"])
        merged["auto_learn_stability_tolerance"] = self._clamp(
            merged["auto_learn_stability_tolerance"], 0.005, 0.15
        )
        merged["auto_learn_downward_max_delta"] = self._clamp(
            merged["auto_learn_downward_max_delta"], 0.0, 0.5
        )
        merged["auto_learn_standing_pose_threshold"] = self._clamp(
            merged["auto_learn_standing_pose_threshold"], 0.0, 1.0
        )
        learn_window = merged["auto_learn_window"]
        if self._standing_height_samples.maxlen != learn_window:
            self._standing_height_samples = deque(
                self._standing_height_samples, maxlen=learn_window
            )
        if not merged["auto_learn_standing_height"]:
            self._standing_height_samples.clear()
        sample_count = merged["foot_calibration_samples"]
        for side, samples in self._foot_reference_samples.items():
            if samples.maxlen != sample_count:
                self._foot_reference_samples[side] = deque(samples, maxlen=sample_count)
        self._hmd_height_scaling = merged

    @property
    def hmd_height_scaling(self) -> Dict:
        return dict(self._hmd_height_scaling)

    @property
    def hmd_height_state(self) -> Dict:
        return dict(self._last_hmd_height_state)

    # ========== 位置变换 ==========

    def camera_to_unity_positions(
        self, pose_3d: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """将摄像头坐标系位置变换到 Unity 坐标系

        摄像头: X左, Y下, Z前, 髋部原点
        Unity:  X右, Y上, Z前

        变换: 翻转 X 和 Y 轴 (与原始 Mediapipe-VR-Fullbody-Tracking 一致)
        """
        unity_pose = {}
        for bone, pos in pose_3d.items():
            p = [float(v) for v in pos]
            unity_pose[bone] = [-p[0], -p[1], p[2]]
        return unity_pose

    def compute_osc_offsets(
        self,
        unity_pose: Dict[str, List[float]],
        hmd_position: Optional[List[float]] = None,
        hmd_quaternion: Optional[List[float]] = None,
        hmd_actual_height: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        """计算 OSC tracker 坐标.

        有 HMD 时，OSC tracker 必须位于 HMD/head tracking space 中，而不是
        直接使用 HMD 绝对世界坐标。VRChat 会把
        /tracking/trackers/head/position 对齐到 avatar 的 Head bone root；
        如果这里再把 HMD 的世界 Y 注入身体 tracker，玩家从坐姿站起时会被重复
        抬高，导致模型悬空。

        Args:
            unity_pose: Unity 坐标系的骨骼位置
            hmd_position: HMD 真实位置。存在时只作为 head tracking-space 基点，
                          输出仍为相对该基点的坐标。

        Returns:
            OSC tracker 坐标
        """
        if hmd_position is not None:
            return self.compute_hmd_relative_osc_positions(
                unity_pose, hmd_position, hmd_quaternion, hmd_actual_height
            )

        neck_pos = unity_pose.get("Neck", [0, 1.5, 0])
        neck_offset = self._hmd_to_neck_offset.tolist()

        # offset = neck_pos - neck_offset
        offset = [
            neck_pos[0] - neck_offset[0],
            neck_pos[1] - neck_offset[1],
            neck_pos[2] - neck_offset[2],
        ]

        result = {}
        for bone, pos in unity_pose.items():
            result[bone] = [
                pos[0] - offset[0],
                pos[1] - offset[1],
                -(pos[2] - offset[2]),  # 原始项目: position[2] = -position[2]
            ]
        return result

    def compute_hmd_relative_osc_positions(
        self,
        unity_pose: Dict[str, List[float]],
        hmd_position: List[float],
        hmd_quaternion: Optional[List[float]] = None,
        hmd_actual_height: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        """Return body tracker positions relative to the HMD/head base.

        VRChat aligns the OSC tracking space so the sender's head position maps
        to the avatar Head bone root. Therefore, body trackers should preserve
        their offsets from the detected Head, while the HMD's absolute world
        translation must not be sent again as part of each body tracker.
        """
        hmd_pos = self._effective_hmd_position(hmd_position, hmd_actual_height)
        anchored_unity = self.compute_hmd_aligned_unity_positions(
            unity_pose, hmd_pos.tolist(), hmd_quaternion
        )
        anchored_unity = self._apply_hmd_height_constraint(
            anchored_unity, hmd_pos, hmd_actual_height
        )

        result = {}
        for bone, pos in anchored_unity.items():
            rel = np.array(pos, dtype=float) - hmd_pos
            result[bone] = [float(rel[0]), float(rel[1]), float(-rel[2])]
        return result

    def compute_hmd_aligned_unity_positions(
        self,
        unity_pose: Dict[str, List[float]],
        hmd_position: List[float],
        hmd_quaternion: Optional[List[float]] = None,
    ) -> Dict[str, List[float]]:
        """Translate local body pose so the detected Head is at the HMD pose.

        HMD data used by VRChat OSC is the head alignment reference. VRChat
        aligns `/tracking/trackers/head/position` to the avatar Head bone root,
        so the body pose should use the detected `Head` joint as the anchor.
        If the pose has no `Head`, fall back to `Neck` plus hmd_to_neck_offset.
        """
        hmd_pos = np.array(hmd_position, dtype=float)

        if "Head" in unity_pose:
            source_anchor = np.array(unity_pose["Head"], dtype=float)
            target_anchor = hmd_pos
        elif "Neck" in unity_pose:
            source_anchor = np.array(unity_pose["Neck"], dtype=float)
            target_anchor = hmd_pos + self._rotated_hmd_to_neck_offset(hmd_quaternion)
        else:
            source_anchor = np.array(unity_pose.get("Hips", [0.0, 0.0, 0.0]), dtype=float)
            target_anchor = hmd_pos

        delta = target_anchor - source_anchor
        return {
            bone: (np.array(pos, dtype=float) + delta).tolist()
            for bone, pos in unity_pose.items()
        }

    def _rotated_hmd_to_neck_offset(
        self,
        hmd_quaternion: Optional[List[float]] = None,
    ) -> np.ndarray:
        if hmd_quaternion is None or Rot is None:
            return self._hmd_to_neck_offset

        try:
            quat = hmd_quaternion
            rot = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]])
            return rot.apply(self._hmd_to_neck_offset)
        except Exception:
            return self._hmd_to_neck_offset

    def compute_steamvr_positions(
        self,
        unity_pose: Dict[str, List[float]],
        hmd_position: Optional[List[float]],
        hmd_quaternion: Optional[List[float]] = None,
        hmd_actual_height: Optional[float] = None,
        *,
        anchor_in_driver_space: bool = False,
    ) -> Dict[str, List[float]]:
        """计算 SteamVR 模式下的绝对位置

        SteamVR 模式: 追踪器位置 = 骨骼位置 - offset
        offset = neck_pos - (hmd_pos + neck_offset_rotated)

        Args:
            unity_pose: Unity 坐标系的骨骼位置
            hmd_position: HMD 真实位置 (VRChat 坐标系: X右, Y上, Z前)
            hmd_quaternion: HMD 四元数 [w, x, y, z] (可选，用于旋转 neck_offset)

        Returns:
            绝对坐标系下的骨骼位置
        """
        if hmd_position is not None:
            effective_hmd_position = (
                np.asarray(hmd_position, dtype=float).reshape(3)
                if anchor_in_driver_space
                else self._effective_hmd_position(hmd_position, hmd_actual_height)
            )
            anchored_unity = self.compute_hmd_aligned_unity_positions(
                unity_pose, effective_hmd_position.tolist(), hmd_quaternion
            )
            anchored_unity = self._apply_hmd_height_constraint(
                anchored_unity, effective_hmd_position, hmd_actual_height
            )
            return {
                bone: [float(pos[0]), float(pos[1]), -float(pos[2])]
                for bone, pos in anchored_unity.items()
            }

        neck_pos = np.array(unity_pose.get("Neck", [0, 1.5, 0]))

        neck_offset_rotated = self._rotated_hmd_to_neck_offset(hmd_quaternion)
        fallback_hmd_unity = np.array([0.0, 1.7, 0.0])
        offset_unity = neck_pos - (fallback_hmd_unity + neck_offset_rotated)

        result = {}
        for bone, pos in unity_pose.items():
            aligned_unity = np.array(pos, dtype=float) - offset_unity
            result[bone] = [
                float(aligned_unity[0]),
                float(aligned_unity[1]),
                -float(aligned_unity[2]),
            ]
        return result

    @staticmethod
    def rotate_unity_pose_yaw(
        unity_pose: Dict[str, List[float]], yaw_deg: float
    ) -> Dict[str, List[float]]:
        """Rotate the camera body around its head (or neck/hips) before anchoring."""
        if abs(yaw_deg) < 1e-9:
            return unity_pose
        anchor = np.asarray(
            next((unity_pose[name] for name in ("Head", "Neck", "Hips")
                  if name in unity_pose), [0.0, 0.0, 0.0]),
            dtype=float,
        )
        angle = np.deg2rad(yaw_deg)
        c, s = float(np.cos(angle)), float(np.sin(angle))
        yaw = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        return {
            name: (anchor + yaw @ (np.asarray(pos, dtype=float) - anchor)).tolist()
            for name, pos in unity_pose.items()
        }

    @staticmethod
    def _effective_hmd_position(
        hmd_position: List[float],
        hmd_actual_height: Optional[float] = None,
    ) -> np.ndarray:
        """Build the single HMD base shared by OSC and SteamVR output.

        SDK senders may report X/Z in a device-local frame while providing a
        separate floor-relative ``actual_height``. Only Y is replaced in that
        case; translating X/Z would introduce a second world origin.
        """
        position = np.asarray(hmd_position, dtype=float).reshape(3).copy()
        try:
            actual_height = float(hmd_actual_height)
        except (TypeError, ValueError):
            actual_height = None
        if (
            actual_height is not None
            and np.isfinite(actual_height)
            and 0.0 < actual_height <= 4.0
        ):
            position[1] = actual_height
        return position

    # ========== 旋转变换 ==========

    def camera_to_unity_rotations(
        self, rotations: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """将摄像头坐标系旋转变换到 Unity ZXY 欧拉角 (度)

        摄像头: XYZ 欧拉角 (弧度, Y向下)
        Unity/VRChat OSC: ZXY 欧拉角 (度, Y向上)

        变换: flip_mat @ rot_matrix @ flip_mat (flip_mat=diag(-1,-1,1))
        """
        if Rot is None:
            return {bone: [0.0, 0.0, 0.0] for bone in rotations}

        unity_rotations = {}
        for bone, euler_rad in rotations.items():
            try:
                rot = Rot.from_euler('xyz', euler_rad, degrees=False)
                mat = rot.as_matrix()
                mat_new = self._FLIP_XY @ mat @ self._FLIP_XY
                rot_new = Rot.from_matrix(mat_new)
                with warnings.catch_warnings():
                    # SciPy defines the singular solution by setting the third
                    # angle to zero; that deterministic result is valid here.
                    warnings.filterwarnings(
                        "ignore",
                        message="Gimbal lock detected.*",
                        category=UserWarning,
                    )
                    euler_zxy = rot_new.as_euler("zxy", degrees=True)
                # as_euler("zxy") 返回 [z, x, y], VRChat OSC 期望 [x, y, z]
                unity_rotations[bone] = [
                    float(euler_zxy[1]),
                    float(euler_zxy[2]),
                    float(euler_zxy[0]),
                ]
            except Exception:
                unity_rotations[bone] = [0.0, 0.0, 0.0]
        return unity_rotations

    def euler_to_quat_openvr(
        self, euler_rad: List[float], yaw_offset_deg: float = 0.0
    ) -> List[float]:
        """将摄像头 XYZ 欧拉角 (弧度) 转为 OpenVR 四元数 [x, y, z, w]

        变换: flip_mat @ rot_matrix @ flip_mat, 然后 XYZ -> OpenVR
        """
        if Rot is None:
            return [0.0, 0.0, 0.0, 1.0]

        try:
            rot = Rot.from_euler('xyz', euler_rad, degrees=False)
            mat = rot.as_matrix()
            mat_new = self._FLIP_XY @ mat @ self._FLIP_XY
            rot_new = Rot.from_matrix(mat_new)
            basis = self._UNITY_OPENVR_BASIS
            angle = np.deg2rad(yaw_offset_deg)
            c, s = float(np.cos(angle)), float(np.sin(angle))
            unity_yaw = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
            mat_openvr = basis @ (unity_yaw @ rot_new.as_matrix()) @ basis
            quat = Rot.from_matrix(mat_openvr).as_quat()  # [x, y, z, w]
            return quat.tolist()
        except Exception:
            return [0.0, 0.0, 0.0, 1.0]

    def euler_to_quat_unity(
        self, euler_rad: List[float]
    ) -> List[float]:
        """将摄像头 XYZ 欧拉角 (弧度) 转为 Unity 四元数 [x, y, z, w]"""
        if Rot is None:
            return [0.0, 0.0, 0.0, 1.0]

        try:
            rot = Rot.from_euler('xyz', euler_rad, degrees=False)
            mat = rot.as_matrix()
            mat_new = self._FLIP_XY @ mat @ self._FLIP_XY
            rot_new = Rot.from_matrix(mat_new)
            return rot_new.as_quat().tolist()  # [x, y, z, w]
        except Exception:
            return [0.0, 0.0, 0.0, 1.0]

    # ========== HMD 坐标获取 ==========

    def get_hmd_pose(self) -> Optional[Tuple[List[float], List[float]]]:
        """从 HMD Provider 获取位姿

        Returns:
            (position, quaternion) 或 None
            position: [x, y, z] VRChat 坐标系 (X右, Y上, Z前, Y=玩家高度)
            quaternion: [w, x, y, z] VRChat 坐标系
        """
        if self._hmd_provider is None:
            return None

        hmd_pose = self._hmd_provider.get_pose()
        if not hmd_pose.valid:
            return None

        return (hmd_pose.position, hmd_pose.quaternion)

    def set_hmd_provider(self, provider):
        """设置 HMD 位姿数据提供者"""
        self._hmd_provider = provider

    def _apply_hmd_height_constraint(
        self,
        anchored_unity: Dict[str, List[float]],
        hmd_pos: np.ndarray,
        hmd_actual_height: Optional[float] = None,
    ) -> Dict[str, List[float]]:
        """Retarget lower body pose from the floor-relative HMD eye height.

        The output remains head-relative for OSC, but the leg geometry is driven
        by the real HMD height. This prevents a standing camera skeleton from
        being translated downward through the floor when the player crouches or
        starts seated.
        """
        cfg = self._hmd_height_scaling
        if not cfg.get("enabled", True):
            self._set_hmd_height_state(False, "disabled")
            return anchored_unity

        if hmd_actual_height is None and cfg.get("require_actual_height", True):
            self._set_hmd_height_state(False, "missing_actual_height")
            return anchored_unity

        try:
            hmd_height = float(hmd_actual_height if hmd_actual_height is not None else hmd_pos[1])
        except (TypeError, ValueError, IndexError):
            self._set_hmd_height_state(False, "invalid_actual_height")
            return anchored_unity

        if hmd_height < float(cfg.get("min_hmd_y", 0.35)):
            self._set_hmd_height_state(False, "below_min_hmd_y", actual_height=hmd_height)
            return anchored_unity

        hmd_height = max(0.0, hmd_height)
        floor_clearance = max(0.0, float(cfg.get("floor_clearance", 0.03)))

        rel_by_bone = {}
        max_down = 0.0
        for bone, pos in anchored_unity.items():
            rel = np.array(pos, dtype=float) - hmd_pos
            rel_by_bone[bone] = rel
            if rel[1] < 0.0:
                max_down = max(max_down, -float(rel[1]))

        standing_pose_confidence = self._standing_pose_confidence(rel_by_bone)
        standing_eye_height = self._resolve_standing_eye_height(
            hmd_height, standing_pose_confidence
        )
        height_ratio = hmd_height / standing_eye_height if standing_eye_height > 0 else 1.0
        crouch_amount = self._height_to_crouch_amount(hmd_height, height_ratio)
        has_complete_leg_chain = self._has_complete_leg_chain(rel_by_bone)
        posture = self._height_to_posture(
            hmd_height,
            height_ratio,
            crouch_amount,
            standing_pose_confidence,
            has_complete_leg_chain,
        )

        ground_rel_y = -max(0.05, hmd_height - floor_clearance)
        needs_floor_clamp = max_down > abs(ground_rel_y) + 1e-6
        foot_targets = self._build_foot_ground_targets(
            rel_by_bone,
            ground_rel_y,
            posture,
            allow_standing_calibration=(
                posture == "standing"
                or (
                    not has_complete_leg_chain
                    and abs(height_ratio - 1.0) <= 0.08
                )
            ),
        )
        if crouch_amount <= 1e-6 and not needs_floor_clamp and not foot_targets:
            self._set_hmd_height_state(
                False,
                "standing_no_adjustment",
                actual_height=hmd_height,
                standing_eye_height=standing_eye_height,
                height_ratio=height_ratio,
                posture=posture,
                crouch_amount=0.0,
                standing_pose_confidence=standing_pose_confidence,
            )
            return anchored_unity

        targets = self._build_height_targets(
            rel_by_bone,
            hmd_height,
            standing_eye_height,
            crouch_amount,
            ground_rel_y,
            floor_clearance,
        )
        crouch_enabled = bool(cfg.get("crouch_from_hmd_height", True))
        knee_forward = max(0.0, float(cfg.get("crouch_knee_forward", 0.18))) * crouch_amount
        knee_outward = max(0.0, float(cfg.get("crouch_knee_outward", 0.06))) * crouch_amount
        constrained = {}
        for bone, rel in rel_by_bone.items():
            adjusted = rel.copy()
            target_y = self._target_rel_y_for_bone(bone, targets, foot_targets)
            if target_y is not None:
                # Targets are already smoothly interpolated by crouch_amount in
                # _build_height_targets; blending again under-reacts to real HMD
                # descent and leaves the legs in a standing shape underground.
                adjusted[1] = target_y
            elif adjusted[1] < ground_rel_y:
                adjusted[1] = ground_rel_y

            if crouch_enabled:
                if bone in ("LeftKneeTracker", "LeftUpperLeg"):
                    adjusted[0] -= knee_outward
                    adjusted[2] += knee_forward
                elif bone in ("RightKneeTracker", "RightUpperLeg"):
                    adjusted[0] += knee_outward
                    adjusted[2] += knee_forward
            if bone in self._foot_bones() and not foot_targets:
                adjusted[1] = max(adjusted[1], ground_rel_y)
            constrained[bone] = (hmd_pos + adjusted).tolist()
        constrained = self._enforce_leg_chain_constraints(
            constrained,
            hmd_pos,
            ground_rel_y,
            crouch_amount,
        )
        self._set_hmd_height_state(
            True,
            "retargeted",
            actual_height=hmd_height,
            standing_eye_height=standing_eye_height,
            height_ratio=height_ratio,
            posture=posture,
            crouch_amount=crouch_amount,
            standing_pose_confidence=standing_pose_confidence,
        )
        return constrained

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _blend(a: float, b: float, t: float) -> float:
        t = max(0.0, min(1.0, t))
        return float(a * (1.0 - t) + b * t)

    @staticmethod
    def _smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _standing_pose_confidence(
        self, rel_by_bone: Dict[str, np.ndarray]
    ) -> float:
        """Estimate whether the camera pose shows straight, weight-bearing legs."""
        hips = self._first_rel(rel_by_bone, ("HipsTracker", "Hips", "Pelvis"))
        if hips is None:
            return 0.0

        side_scores = []
        for side in ("Left", "Right"):
            knee = self._first_rel(
                rel_by_bone, (f"{side}KneeTracker", f"{side}UpperLeg")
            )
            foot = self._first_rel(
                rel_by_bone,
                (f"{side}FootTracker", f"{side}LowerLeg", f"{side}Foot"),
            )
            if knee is None or foot is None:
                continue

            side_hip = np.asarray(hips, dtype=float).copy()
            side_hip[0] = float(knee[0])
            upper = side_hip - knee
            lower = foot - knee
            upper_norm = float(np.linalg.norm(upper))
            lower_norm = float(np.linalg.norm(lower))
            if upper_norm <= 1e-6 or lower_norm <= 1e-6:
                continue

            cosine = self._clamp(
                float(np.dot(upper, lower) / (upper_norm * lower_norm)),
                -1.0,
                1.0,
            )
            knee_angle = float(np.degrees(np.arccos(cosine)))
            straightness = self._smoothstep((knee_angle - 145.0) / 25.0)
            upper_drop = float(side_hip[1] - knee[1])
            lower_drop = float(knee[1] - foot[1])
            vertical_chain = min(
                self._smoothstep((upper_drop - 0.08) / 0.18),
                self._smoothstep((lower_drop - 0.08) / 0.18),
            )
            side_scores.append(straightness * vertical_chain)

        if not side_scores:
            return 0.0
        if len(side_scores) == 1:
            return float(side_scores[0] * 0.75)
        return float(min(side_scores))

    def _resolve_standing_eye_height(
        self,
        hmd_height: float,
        standing_pose_confidence: Optional[float] = None,
    ) -> float:
        cfg = self._hmd_height_scaling
        min_height = float(cfg.get("min_standing_eye_height", 1.2))
        max_height = float(cfg.get("max_standing_eye_height", 2.2))
        configured = self._clamp(float(cfg.get("standing_eye_height", 1.65)), min_height, max_height)
        if cfg.get("standing_eye_height_calibrated", False):
            self._standing_height_samples.clear()
            return configured
        current = (
            float(self._learned_standing_eye_height)
            if self._learned_standing_eye_height is not None
            else configured
        )
        if not cfg.get("auto_learn_standing_height", True):
            return self._clamp(current, min_height, max_height)

        try:
            pose_confidence = float(standing_pose_confidence)
        except (TypeError, ValueError):
            pose_confidence = 0.0
        pose_confidence = self._clamp(pose_confidence, 0.0, 1.0)
        allow_downward = bool(
            cfg.get("auto_learn_allow_downward", True)
            and self._learned_standing_eye_height is None
            and pose_confidence
            >= float(cfg.get("auto_learn_standing_pose_threshold", 0.8))
        )
        auto_min = float(cfg.get("auto_learn_min_height", min_height))
        if allow_downward:
            downward_limit = configured - float(
                cfg.get("auto_learn_downward_max_delta", 0.25)
            )
            learn_floor = max(min_height, auto_min, downward_limit)
        else:
            learn_floor = max(current, auto_min)
        if not learn_floor <= hmd_height <= max_height:
            self._standing_height_samples.clear()
            return self._clamp(current, min_height, max_height)

        self._standing_height_samples.append(float(hmd_height))
        if len(self._standing_height_samples) >= self._standing_height_samples.maxlen:
            samples = np.asarray(self._standing_height_samples, dtype=float)
            stable_span = float(np.percentile(samples, 90) - np.percentile(samples, 10))
            tolerance = float(cfg.get("auto_learn_stability_tolerance", 0.03))
            if stable_span <= tolerance:
                stable_height = float(np.median(samples))
                if stable_height > current + 0.01:
                    self._learned_standing_eye_height = stable_height
                    current = stable_height
                elif allow_downward and stable_height < current - 0.01:
                    self._learned_standing_eye_height = stable_height
                    current = stable_height
        return self._clamp(current, min_height, max_height)

    def _height_to_crouch_amount(self, hmd_height: float, height_ratio: float) -> float:
        cfg = self._hmd_height_scaling
        start = float(cfg.get("crouch_start_ratio", 0.92))
        seated = float(cfg.get("seated_ratio", 0.55))
        if height_ratio >= start:
            return 0.0
        if height_ratio <= seated:
            return 1.0
        return self._smoothstep((start - height_ratio) / max(0.01, start - seated))

    def _height_to_posture(
        self,
        hmd_height: float,
        height_ratio: float,
        crouch_amount: float,
        standing_pose_confidence: Optional[float] = None,
        has_complete_leg_chain: bool = True,
    ) -> str:
        cfg = self._hmd_height_scaling
        seated_height = float(cfg.get("seated_eye_height", 0.95))
        seated_ratio = float(cfg.get("seated_ratio", 0.55))
        if hmd_height <= seated_height or height_ratio <= seated_ratio:
            return "seated"
        if crouch_amount > 0.05:
            return "crouch"
        try:
            standing_confidence = float(standing_pose_confidence)
        except (TypeError, ValueError):
            standing_confidence = 0.0
        # Height alone cannot distinguish a high chair from standing. Keep the
        # standing label only when the inferred leg chain is also credible;
        # this prevents a seated frame from becoming a standing-height baseline.
        if has_complete_leg_chain and standing_confidence < float(
            cfg.get("auto_learn_standing_pose_threshold", 0.8)
        ):
            return "crouch"
        if not has_complete_leg_chain:
            return "unknown"
        return "standing"

    @staticmethod
    def _has_complete_leg_chain(rel_by_bone: Dict[str, np.ndarray]) -> bool:
        return all(
            any(name in rel_by_bone for name in names)
            for names in (
                ("HipsTracker", "Hips", "Pelvis"),
                ("LeftKneeTracker", "LeftUpperLeg"),
                ("RightKneeTracker", "RightUpperLeg"),
                ("LeftFootTracker", "LeftFoot", "LeftLowerLeg"),
                ("RightFootTracker", "RightFoot", "RightLowerLeg"),
            )
        )

    def _enforce_leg_chain_constraints(
        self,
        constrained: Dict[str, List[float]],
        hmd_pos: np.ndarray,
        ground_rel_y: float,
        crouch_amount: float,
    ) -> Dict[str, List[float]]:
        """Keep each hip-knee-foot chain geometrically valid and above floor.

        The height solver changes vertical targets to follow a floor-relative
        HMD. Re-solving each knee from its adjusted hip and foot keeps the
        original two-segment leg lengths and avoids handing an impossible
        skeleton to the receiving avatar IK.
        """
        result = {name: list(values) for name, values in constrained.items()}
        floor_y = float(hmd_pos[1]) + float(ground_rel_y)
        for side in ("Left", "Right"):
            hip_name = next(
                (name for name in ("HipsTracker", "Hips", "Pelvis") if name in result),
                None,
            )
            knee_name = f"{side}KneeTracker"
            foot_name = f"{side}FootTracker"
            if (
                hip_name is None
                or knee_name not in result
                or foot_name not in result
            ):
                continue

            hip = np.asarray(result[hip_name], dtype=float)
            knee = np.asarray(result[knee_name], dtype=float)
            foot = np.asarray(result[foot_name], dtype=float)
            if not (np.all(np.isfinite(hip)) and np.all(np.isfinite(knee)) and np.all(np.isfinite(foot))):
                continue

            upper_length = float(np.linalg.norm(knee - hip))
            lower_length = float(np.linalg.norm(foot - knee))
            if upper_length <= 1e-4 or lower_length <= 1e-4:
                continue

            # Preserve the detected bend direction in X/Z and add a small
            # forward bend only when the HMD indicates crouching.
            bend = knee - (hip + foot) * 0.5
            bend_xz = bend.copy()
            bend_xz[1] = 0.0
            bend_norm = float(np.linalg.norm(bend_xz))
            if bend_norm <= 1e-5:
                bend_xz = np.array([(-1.0 if side == "Left" else 1.0), 0.0, 0.0])
                bend_norm = 1.0
            bend_xz /= bend_norm
            bend_sign = -1.0 if side == "Left" else 1.0
            bend_depth = max(0.0, float(self._hmd_height_scaling.get("crouch_knee_forward", 0.18))) * crouch_amount
            bend_lateral = max(0.0, float(self._hmd_height_scaling.get("crouch_knee_outward", 0.06))) * crouch_amount

            midpoint = (hip + foot) * 0.5
            half_span = float(np.linalg.norm(foot - hip)) * 0.5
            target_reach = min(upper_length, lower_length) - 1e-4
            offset_sq = max(0.0, target_reach * target_reach - min(half_span, target_reach) ** 2)
            offset = np.sqrt(offset_sq)
            candidate = midpoint + bend_xz * offset
            candidate[0] += bend_sign * bend_lateral
            candidate[2] += bend_depth
            candidate[1] = max(candidate[1], floor_y + 0.06)

            # A final projection keeps the knee within both segment spheres;
            # it is preferable to a zero/underground joint when the measured
            # hip-foot span exceeds the inferred leg length.
            hip_to_candidate = candidate - hip
            if np.linalg.norm(hip_to_candidate) > upper_length:
                candidate = hip + hip_to_candidate * (upper_length / max(np.linalg.norm(hip_to_candidate), 1e-6))
            foot_to_candidate = candidate - foot
            if np.linalg.norm(foot_to_candidate) > lower_length:
                candidate = foot + foot_to_candidate * (lower_length / max(np.linalg.norm(foot_to_candidate), 1e-6))
            candidate[1] = max(candidate[1], floor_y + 0.06)

            result[knee_name] = candidate.tolist()
            upper_alias = f"{side}UpperLeg"
            if upper_alias in result:
                result[upper_alias] = candidate.tolist()
        return result

    def _build_height_targets(
        self,
        rel_by_bone: Dict[str, np.ndarray],
        hmd_height: float,
        standing_eye_height: float,
        crouch_amount: float,
        ground_rel_y: float,
        floor_clearance: float,
    ) -> Dict[str, float]:
        cfg = self._hmd_height_scaling
        hip_rel = self._first_rel(rel_by_bone, ("HipsTracker", "Hips", "Pelvis"))
        chest_rel = self._first_rel(rel_by_bone, ("ChestTracker", "Chest", "Spine", "Neck"))

        standing_hip_height = standing_eye_height - abs(float(hip_rel[1]) if hip_rel is not None else 0.8)
        standing_hip_height = max(float(cfg.get("standing_hip_min_height", 0.78)), standing_hip_height)
        seated_hip_height = float(cfg.get("seated_hip_height", 0.45))
        hip_floor_height = self._blend(standing_hip_height, seated_hip_height, crouch_amount)
        hip_floor_height = self._clamp(hip_floor_height, floor_clearance + 0.18, max(floor_clearance + 0.2, hmd_height - 0.12))

        standing_chest_height = standing_eye_height - abs(float(chest_rel[1]) if chest_rel is not None else 0.45)
        standing_chest_height = max(float(cfg.get("standing_chest_min_height", 1.05)), standing_chest_height)
        seated_chest_height = min(max(hip_floor_height + 0.22, hmd_height - 0.20), hmd_height - 0.08)
        chest_floor_height = self._blend(standing_chest_height, seated_chest_height, crouch_amount)
        chest_floor_height = self._clamp(chest_floor_height, hip_floor_height + 0.12, max(hip_floor_height + 0.13, hmd_height - 0.06))

        knee_floor_height = floor_clearance + max(0.22, (hip_floor_height - floor_clearance) * 0.55)
        if crouch_amount > 0.7:
            knee_floor_height = max(knee_floor_height, min(0.55, hip_floor_height + 0.05))
        knee_floor_height = self._clamp(knee_floor_height, floor_clearance + 0.15, max(floor_clearance + 0.16, hmd_height - 0.10))

        return {
            "ground": ground_rel_y,
            "hip": hip_floor_height - hmd_height,
            "chest": chest_floor_height - hmd_height,
            "knee": knee_floor_height - hmd_height,
        }

    def _build_foot_ground_targets(
        self,
        rel_by_bone: Dict[str, np.ndarray],
        ground_rel_y: float,
        posture: str,
        allow_standing_calibration: bool = False,
    ) -> Dict[str, float]:
        """Anchor foot trackers to the real floor while preserving foot lift."""
        cfg = self._hmd_height_scaling
        if not cfg.get("foot_grounding_enabled", True):
            return {}

        sources = {
            "left": (
                "LeftFootTracker", "LeftFoot", "LeftLowerLeg", "LeftAnkle", "LeftToeBase"
            ),
            "right": (
                "RightFootTracker", "RightFoot", "RightLowerLeg", "RightAnkle", "RightToeBase"
            ),
        }
        if allow_standing_calibration and cfg.get("auto_calibrate_foot_reference", True):
            for side, names in sources.items():
                if side in self._standing_foot_reference:
                    continue
                rel = self._first_rel(rel_by_bone, names)
                if rel is None:
                    continue
                samples = self._foot_reference_samples[side]
                samples.append(float(rel[1]))
                if len(samples) >= samples.maxlen:
                    self._standing_foot_reference[side] = float(np.percentile(samples, 25))

        targets = {}
        lift_gain = float(cfg.get("foot_lift_gain", 1.0))
        max_lift = float(cfg.get("max_foot_lift", 0.35))
        for side, names in sources.items():
            rel = self._first_rel(rel_by_bone, names)
            if rel is None:
                continue
            reference = self._standing_foot_reference.get(side)
            if reference is None:
                target_y = ground_rel_y
            else:
                lift = max(0.0, (float(rel[1]) - reference) * lift_gain)
                target_y = ground_rel_y + min(max_lift, lift)
            for bone in names:
                if bone in rel_by_bone:
                    targets[bone] = target_y
        return targets

    @staticmethod
    def _first_rel(rel_by_bone: Dict[str, np.ndarray], names: Tuple[str, ...]) -> Optional[np.ndarray]:
        for name in names:
            if name in rel_by_bone:
                return rel_by_bone[name]
        return None

    @staticmethod
    def _foot_bones() -> Tuple[str, ...]:
        return (
            "LeftFootTracker",
            "RightFootTracker",
            "LeftFoot",
            "RightFoot",
            "LeftLowerLeg",
            "RightLowerLeg",
            "LeftAnkle",
            "RightAnkle",
            "LeftToeBase",
            "RightToeBase",
        )

    @staticmethod
    def _hip_bones() -> Tuple[str, ...]:
        return ("HipsTracker", "Hips", "Pelvis")

    @staticmethod
    def _chest_bones() -> Tuple[str, ...]:
        return ("ChestTracker", "Chest", "Spine")

    @staticmethod
    def _knee_bones() -> Tuple[str, ...]:
        return (
            "LeftKneeTracker",
            "RightKneeTracker",
            "LeftUpperLeg",
            "RightUpperLeg",
        )

    def _target_rel_y_for_bone(
        self,
        bone: str,
        targets: Dict[str, float],
        foot_targets: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        if foot_targets and bone in foot_targets:
            return foot_targets[bone]
        if bone in self._foot_bones():
            return targets["ground"]
        if bone in self._hip_bones():
            return targets["hip"]
        if bone in self._chest_bones():
            return targets["chest"]
        if bone in self._knee_bones():
            return targets["knee"]
        return None

    def _set_hmd_height_state(
        self,
        applied: bool,
        reason: str,
        actual_height: Optional[float] = None,
        standing_eye_height: Optional[float] = None,
        height_ratio: Optional[float] = None,
        posture: str = "unknown",
        crouch_amount: float = 0.0,
        standing_pose_confidence: Optional[float] = None,
    ):
        self._last_hmd_height_state = {
            "enabled": bool(self._hmd_height_scaling.get("enabled", True)),
            "applied": bool(applied),
            "reason": reason,
            "posture": posture,
            "actual_height": actual_height,
            "standing_eye_height": standing_eye_height,
            "height_ratio": height_ratio,
            "crouch_amount": float(crouch_amount),
            "standing_pose_confidence": standing_pose_confidence,
            "foot_reference_ready": len(self._standing_foot_reference) == 2,
        }
