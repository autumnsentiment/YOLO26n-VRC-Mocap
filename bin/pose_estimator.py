import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
import copy
from collections import deque
import math
import time


class OneEuroFilter:
    """OneEuro 滤波器 - 动捕领域标准平滑算法
    
    特点：低频信号平滑，高频信号低延迟
    参考: Casiez et al. "1€ Filter: A Simple Speed-based Low-pass Filter
           for Noisy Input in Interactive Systems"
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0, freq: float = 30.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, cutoff: float) -> float:
        r = 2 * math.pi * max(0.0, cutoff) / self.freq
        return r / (1.0 + r)

    def _exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev

    def filter(self, x: float, t: Optional[float] = None) -> float:
        if self.t_prev is None:
            self.t_prev = time.perf_counter() if t is None else t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        t = time.perf_counter() if t is None else t
        dt = t - self.t_prev
        if not math.isfinite(dt) or dt <= 0:
            dt = 1.0 / max(self.freq, 1.0)
        dt = min(max(dt, 1e-4), 1.0)

        self.freq = 1.0 / dt

        # 估计导数
        dx_hat = (x - self.x_prev) / dt
        a_d = self._smoothing_factor(self.d_cutoff)
        dx_hat = self._exponential_smoothing(a_d, dx_hat, self.dx_prev)

        # 自适应截止频率
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 平滑
        a = self._smoothing_factor(cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


class OneEuroFilterVec3:
    """向量版 OneEuro 滤波器，对3D向量每个分量独立滤波"""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007,
                 d_cutoff: float = 1.0, freq: float = 30.0):
        self.filters = [
            OneEuroFilter(min_cutoff, beta, d_cutoff, freq)
            for _ in range(3)
        ]

    def filter(self, vec: List[float], t: Optional[float] = None) -> List[float]:
        return [self.filters[i].filter(vec[i], t) for i in range(3)]


class PoseEstimator:
    """人体姿态估计器
    
    将 YOLO26n 2D 关键点转换为 VRChat 骨骼旋转数据
    """

    # COCO 17关键点索引
    KP = {
        "nose": 0, "left_eye": 1, "right_eye": 2,
        "left_ear": 3, "right_ear": 4,
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10,
        "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14,
        "left_ankle": 15, "right_ankle": 16,
    }

    # VRChat 骨骼层级
    VRC_BONE_HIERARCHY = {
        "Hips": None,
        "Spine": "Hips",
        "Chest": "Spine",
        "UpperChest": "Chest",
        "Neck": "UpperChest",
        "Head": "Neck",
        "LeftUpperLeg": "Hips",
        "RightUpperLeg": "Hips",
        "LeftLowerLeg": "LeftUpperLeg",
        "RightLowerLeg": "RightUpperLeg",
        "LeftFoot": "LeftLowerLeg",
        "RightFoot": "RightLowerLeg",
        "LeftToes": "LeftFoot",
        "RightToes": "RightFoot",
        "LeftUpperArm": "UpperChest",
        "RightUpperArm": "UpperChest",
        "LeftLowerArm": "LeftUpperArm",
        "RightLowerArm": "RightUpperArm",
        "LeftHand": "LeftLowerArm",
        "RightHand": "RightLowerArm",
    }

    TRACKER_POSE_SOURCES = {
        "HipsTracker": "Hips",
        "ChestTracker": "Chest",
        "LeftFootTracker": "LeftFoot",
        "RightFootTracker": "RightFoot",
        "LeftKneeTracker": "LeftUpperLeg",
        "RightKneeTracker": "RightUpperLeg",
        "LeftElbowTracker": "LeftUpperArm",
        "RightElbowTracker": "RightUpperArm",
    }

    # 人体骨骼比例（基于标准人体测量学）
    BONE_LENGTHS = {
        "spine": 0.18,
        "chest": 0.18,
        "upper_chest": 0.12,
        "neck": 0.10,
        "head": 0.10,
        "upper_leg": 0.42,
        "lower_leg": 0.42,
        "foot": 0.08,
        "upper_arm": 0.28,
        "lower_arm": 0.24,
        "hand": 0.08,
        "hip_width": 0.16,
        "shoulder_width": 0.36,
    }

    def __init__(self, smoothing: str = "one_euro", smoothing_factor: float = 0.3,
                 one_euro_min_cutoff: float = 1.0, one_euro_beta: float = 0.007,
                 average_smoothing_enabled: bool = True,
                 average_window: int = 4,
                 ema_alpha: float = 0.35,
                 reset_on_tracking_lost: bool = True):
        self.smoothing = smoothing
        self.smoothing_factor = smoothing_factor
        self.previous_pose = None
        self.previous_rotations = None
        self.average_smoothing_enabled = average_smoothing_enabled
        self.average_window = max(1, int(average_window))
        self.ema_alpha = min(1.0, max(0.0, float(ema_alpha)))
        self.reset_on_tracking_lost = reset_on_tracking_lost
        self._pose_average_history: Dict[str, deque] = {}
        self._pose_ema: Dict[str, List[float]] = {}

        # OneEuro 滤波器
        self.pose_filters: Dict[str, OneEuroFilterVec3] = {}
        self.rotation_filters: Dict[str, OneEuroFilterVec3] = {}
        self.one_euro_min_cutoff = one_euro_min_cutoff
        self.one_euro_beta = one_euro_beta
        self._target_center = None
        self._target_last_seen = None

    @classmethod
    def ensure_tracker_nodes(cls, pose_3d: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Add explicit VRChat tracker nodes without changing the VRC bone tree."""
        for tracker_name, source_name in cls.TRACKER_POSE_SOURCES.items():
            if tracker_name not in pose_3d and source_name in pose_3d:
                pose_3d[tracker_name] = [float(v) for v in pose_3d[source_name]]
        return pose_3d

    def _reset_average_smoothing(self):
        self._pose_average_history.clear()
        self._pose_ema.clear()

    def apply_average_smoothing(self, pose_3d: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Apply a short moving average plus EMA before the existing filter stage."""
        if not self.average_smoothing_enabled:
            return pose_3d

        active = set(pose_3d.keys())
        for joint in list(self._pose_average_history.keys()):
            if joint not in active:
                self._pose_average_history.pop(joint, None)
                self._pose_ema.pop(joint, None)

        smoothed = {}
        for joint, coords in pose_3d.items():
            values = [float(v) for v in coords]
            history = self._pose_average_history.setdefault(joint, deque(maxlen=self.average_window))
            history.append(values)
            avg = [sum(frame[i] for frame in history) / len(history) for i in range(3)]

            prev = self._pose_ema.get(joint)
            if prev is None:
                ema = avg
            else:
                alpha = self.ema_alpha
                ema = [alpha * avg[i] + (1.0 - alpha) * prev[i] for i in range(3)]
            self._pose_ema[joint] = ema
            smoothed[joint] = ema.copy()

        return smoothed

    def _get_pose_filter(self, joint: str) -> OneEuroFilterVec3:
        if joint not in self.pose_filters:
            self.pose_filters[joint] = OneEuroFilterVec3(
                min_cutoff=self.one_euro_min_cutoff,
                beta=self.one_euro_beta
            )
        return self.pose_filters[joint]

    def _get_rotation_filter(self, bone: str) -> OneEuroFilterVec3:
        if bone not in self.rotation_filters:
            self.rotation_filters[bone] = OneEuroFilterVec3(
                min_cutoff=self.one_euro_min_cutoff,
                beta=self.one_euro_beta
            )
        return self.rotation_filters[bone]

    def keypoints_to_dict(self, keypoints: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """将关键点数组转为字典，过滤低置信度点"""
        keypoints_dict = {}
        for name, idx in self.KP.items():
            if idx < len(keypoints):
                x, y, conf = keypoints[idx]
                if conf > 0.15:  # 降低阈值，避免丢失下半身关键点
                    keypoints_dict[name] = (float(x), float(y), float(conf))
        return keypoints_dict

    def _select_person(self, detections: Dict, image_width: int,
                       image_height: int) -> Optional[int]:
        now = time.perf_counter()
        candidates = []
        boxes = detections.get("boxes") or []
        for index, points in enumerate(detections.get("keypoints") or []):
            kps = self.keypoints_to_dict(points)
            shoulders = [kps.get(side + "_shoulder") for side in ("left", "right")]
            hips = [kps.get(side + "_hip") for side in ("left", "right")]
            torso = [p for p in shoulders + hips if p is not None]
            if len(shoulders) < 2 or any(p is None for p in shoulders):
                continue
            center = np.mean([(p[0] / max(image_width, 1),
                               p[1] / max(image_height, 1)) for p in torso], axis=0)
            if index < len(boxes) and len(boxes[index]) >= 4:
                x0, y0, x1, y1 = boxes[index][:4]
                area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            else:
                visible = list(kps.values())
                area = ((max(p[0] for p in visible) - min(p[0] for p in visible))
                        * (max(p[1] for p in visible) - min(p[1] for p in visible)))
            candidates.append((index, center, area))

        if self._target_center is not None and candidates:
            nearest = min(candidates, key=lambda item: np.linalg.norm(item[1] - self._target_center))
            if np.linalg.norm(nearest[1] - self._target_center) <= 0.2:
                self._target_center = nearest[1]
                self._target_last_seen = now
                return nearest[0]

        if self._target_center is not None and self._target_last_seen is not None:
            if now - self._target_last_seen < 0.5:
                return None
            self._target_center = None
            self._target_last_seen = None
            if self.reset_on_tracking_lost:
                self.reset_filters()

        if not candidates:
            return None
        selected = max(candidates, key=lambda item: item[2])
        self._target_center = selected[1]
        self._target_last_seen = now
        return selected[0]

    def _midpoint(self, p1: Tuple, p2: Tuple) -> Tuple[float, float]:
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def _distance_2d(self, p1: Tuple, p2: Tuple) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _unit_2d(self, dx: float, dy: float,
                 fallback: Tuple[float, float] = (0.0, -1.0)) -> Tuple[float, float]:
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return fallback
        return (dx / length, dy / length)

    def _synthetic_nose_from_upper_body(
        self,
        keypoints: Dict[str, Tuple[float, float, float]],
        shoulder_mid: Tuple[float, float],
        shoulder_dist: float,
    ) -> Tuple[float, float, float]:
        lh = keypoints.get("left_hip")
        rh = keypoints.get("right_hip")
        if lh and rh:
            hip_mid = self._midpoint(lh, rh)
            up_x, up_y = self._unit_2d(
                shoulder_mid[0] - hip_mid[0],
                shoulder_mid[1] - hip_mid[1],
            )
        else:
            up_x, up_y = (0.0, -1.0)
        return (
            shoulder_mid[0] + up_x * shoulder_dist * 0.65,
            shoulder_mid[1] + up_y * shoulder_dist * 0.65,
            0.2,
        )

    def normalize_keypoints(self, kps: Dict[str, Tuple[float, float, float]],
                           image_width: int = 640, image_height: int = 480,
                           flip_horizontal: bool = True) -> Dict[str, Tuple[float, float, float]]:
        """以髋部中心为原点，肩宽为尺度单位，归一化关键点
        
        Args:
            flip_horizontal: 摄像头是否水平翻转图像。
                当 flip_horizontal=True 时，YOLO 检测到的 left 关键点在图像右边，
                需要翻转 X 坐标使其与人体解剖学方向一致（左=负X）。
        """
        normalized = {}

        left_hip = kps.get("left_hip")
        right_hip = kps.get("right_hip")
        if not left_hip or not right_hip:
            return normalized

        hip_center = self._midpoint(left_hip, right_hip)
        hip_width = self._distance_2d(left_hip, right_hip)

        left_shoulder = kps.get("left_shoulder")
        right_shoulder = kps.get("right_shoulder")

        if left_shoulder and right_shoulder:
            shoulder_width = self._distance_2d(left_shoulder, right_shoulder)
            torso_length = self._distance_2d(
                self._midpoint(left_shoulder, right_shoulder),
                hip_center
            )
        else:
            shoulder_width = hip_width * 2.2
            torso_length = hip_width * 3.0

        # 用躯干长度作为统一尺度
        scale = max(torso_length, 1.0)

        for name, (x, y, conf) in kps.items():
            x_sign = -1.0 if flip_horizontal else 1.0
            norm_x = x_sign * (x - hip_center[0]) / scale
            norm_y = (y - hip_center[1]) / scale
            normalized[name] = (norm_x, norm_y, conf)

        return normalized

    def estimate_3d_pose(self, norm_kps: Dict[str, Tuple[float, float, float]]) -> Dict[str, List[float]]:
        """从归一化2D关键点估算3D骨骼位置

        使用人体测量学比例估算深度分量。
        所有输出坐标均为米制单位，以髋部中心为原点，Y轴向下（图像坐标系）。
        """
        pose_3d = {}

        def kp(name: str) -> Tuple[float, float, float]:
            return norm_kps.get(name, (0, 0, 0))

        def normalized_dir(p_from, p_to):
            """计算从 p_from 到 p_to 的单位方向向量"""
            d = np.array([p_to[0] - p_from[0], p_to[1] - p_from[1], 0.0])
            length = np.linalg.norm(d)
            if length < 1e-6:
                return np.array([0.0, 1.0, 0.0])  # 默认向下
            return d / length

        # 髋部关键点
        lh = kp("left_hip")
        rh = kp("right_hip")
        hip_center = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, 0.0])

        # 肩部关键点
        ls = kp("left_shoulder")
        rs = kp("right_shoulder")
        shoulder_center = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, 0.0])

        # 根据肩髋偏移估算身体扭转
        shoulder_offset_x = shoulder_center[0] - hip_center[0]
        body_twist = np.clip(shoulder_offset_x * 2, -1, 1)

        # 髋部宽度（米制）
        hip_width = self.BONE_LENGTHS["hip_width"]
        half_hip_w = hip_width / 2

        # ===== 髋部 (原点) =====
        pose_3d["Hips"] = [0.0, 0.0, 0.0]

        # ===== 脊柱 =====
        spine_dir = normalized_dir(hip_center, shoulder_center)
        # 添加深度分量（身体扭转）
        spine_dir[2] += body_twist * 0.05
        spine_dir = spine_dir / max(np.linalg.norm(spine_dir), 1e-6) * self.BONE_LENGTHS["spine"]
        spine_pos = np.array([0.0, 0.0, 0.0]) + spine_dir
        pose_3d["Spine"] = spine_pos.tolist()

        # ===== 胸部 =====
        chest_dir = spine_dir / max(np.linalg.norm(spine_dir), 1e-6) * self.BONE_LENGTHS["chest"]
        chest_pos = spine_pos + chest_dir
        pose_3d["Chest"] = chest_pos.tolist()

        # ===== 上胸 =====
        uc_dir = spine_dir / max(np.linalg.norm(spine_dir), 1e-6) * self.BONE_LENGTHS["upper_chest"]
        uc_pos = chest_pos + uc_dir
        pose_3d["UpperChest"] = uc_pos.tolist()

        # ===== 颈部 =====
        neck_dir = spine_dir / max(np.linalg.norm(spine_dir), 1e-6) * self.BONE_LENGTHS["neck"]
        neck_pos = uc_pos + neck_dir
        pose_3d["Neck"] = neck_pos.tolist()

        # ===== 头部 =====
        nose = kp("nose")
        if norm_kps.get("nose") and nose[2] > 0.15:
            head_dir_2d = normalized_dir(
                (shoulder_center[0], shoulder_center[1]),
                (nose[0], nose[1])
            )
        else:
            head_dir_2d = normalized_dir(
                (hip_center[0], hip_center[1]),
                (shoulder_center[0], shoulder_center[1])
            )
        head_dir_2d[2] = 0.0
        head_len = np.linalg.norm(head_dir_2d)
        if head_len < 1e-6:
            head_dir_2d = np.array([0.0, -1.0, 0.0])
        head_dir = head_dir_2d / max(np.linalg.norm(head_dir_2d), 1e-6) * self.BONE_LENGTHS["head"]
        head_pos = neck_pos + head_dir
        pose_3d["Head"] = head_pos.tolist()

        # ===== 左腿 =====
        l_knee = kp("left_knee")
        l_ankle = kp("left_ankle")

        left_hip_pos = np.array([-half_hip_w, 0.0, 0.0])
        left_upper_leg_dir = normalized_dir(lh, l_knee)
        left_upper_leg_dir = left_upper_leg_dir / max(np.linalg.norm(left_upper_leg_dir), 1e-6) * self.BONE_LENGTHS["upper_leg"]
        left_upper_leg_pos = left_hip_pos + left_upper_leg_dir
        pose_3d["LeftUpperLeg"] = left_upper_leg_pos.tolist()

        left_lower_leg_dir = normalized_dir(l_knee, l_ankle)
        left_lower_leg_dir = left_lower_leg_dir / max(np.linalg.norm(left_lower_leg_dir), 1e-6) * self.BONE_LENGTHS["lower_leg"]
        left_lower_leg_pos = left_upper_leg_pos + left_lower_leg_dir
        pose_3d["LeftLowerLeg"] = left_lower_leg_pos.tolist()

        # 左脚：沿小腿方向延伸
        left_foot_dir = left_lower_leg_dir / max(np.linalg.norm(left_lower_leg_dir), 1e-6) * self.BONE_LENGTHS["foot"]
        left_foot_pos = left_lower_leg_pos + left_foot_dir
        pose_3d["LeftFoot"] = left_foot_pos.tolist()

        left_toes_dir = left_foot_dir / max(np.linalg.norm(left_foot_dir), 1e-6) * 0.03
        pose_3d["LeftToes"] = (left_foot_pos + left_toes_dir).tolist()

        # ===== 右腿 =====
        r_knee = kp("right_knee")
        r_ankle = kp("right_ankle")

        right_hip_pos = np.array([half_hip_w, 0.0, 0.0])
        right_upper_leg_dir = normalized_dir(rh, r_knee)
        right_upper_leg_dir = right_upper_leg_dir / max(np.linalg.norm(right_upper_leg_dir), 1e-6) * self.BONE_LENGTHS["upper_leg"]
        right_upper_leg_pos = right_hip_pos + right_upper_leg_dir
        pose_3d["RightUpperLeg"] = right_upper_leg_pos.tolist()

        right_lower_leg_dir = normalized_dir(r_knee, r_ankle)
        right_lower_leg_dir = right_lower_leg_dir / max(np.linalg.norm(right_lower_leg_dir), 1e-6) * self.BONE_LENGTHS["lower_leg"]
        right_lower_leg_pos = right_upper_leg_pos + right_lower_leg_dir
        pose_3d["RightLowerLeg"] = right_lower_leg_pos.tolist()

        right_foot_dir = right_lower_leg_dir / max(np.linalg.norm(right_lower_leg_dir), 1e-6) * self.BONE_LENGTHS["foot"]
        right_foot_pos = right_lower_leg_pos + right_foot_dir
        pose_3d["RightFoot"] = right_foot_pos.tolist()

        right_toes_dir = right_foot_dir / max(np.linalg.norm(right_foot_dir), 1e-6) * 0.03
        pose_3d["RightToes"] = (right_foot_pos + right_toes_dir).tolist()

        # ===== 左臂 =====
        l_elbow = kp("left_elbow")
        l_wrist = kp("left_wrist")

        shoulder_width = self.BONE_LENGTHS["shoulder_width"]
        left_shoulder_pos = np.array([-shoulder_width / 2, uc_pos[1], 0.0])
        left_upper_arm_dir = normalized_dir(ls, l_elbow)
        left_upper_arm_dir[2] += 0.05  # 轻微前伸
        left_upper_arm_dir = left_upper_arm_dir / max(np.linalg.norm(left_upper_arm_dir), 1e-6) * self.BONE_LENGTHS["upper_arm"]
        left_upper_arm_pos = left_shoulder_pos + left_upper_arm_dir
        pose_3d["LeftUpperArm"] = left_upper_arm_pos.tolist()

        left_lower_arm_dir = normalized_dir(l_elbow, l_wrist)
        left_lower_arm_dir[2] += 0.08  # 轻微前伸
        left_lower_arm_dir = left_lower_arm_dir / max(np.linalg.norm(left_lower_arm_dir), 1e-6) * self.BONE_LENGTHS["lower_arm"]
        left_lower_arm_pos = left_upper_arm_pos + left_lower_arm_dir
        pose_3d["LeftLowerArm"] = left_lower_arm_pos.tolist()

        left_hand_dir = left_lower_arm_dir / max(np.linalg.norm(left_lower_arm_dir), 1e-6) * self.BONE_LENGTHS["hand"]
        left_hand_pos = left_lower_arm_pos + left_hand_dir
        pose_3d["LeftHand"] = left_hand_pos.tolist()

        # ===== 右臂 =====
        r_elbow = kp("right_elbow")
        r_wrist = kp("right_wrist")

        right_shoulder_pos = np.array([shoulder_width / 2, uc_pos[1], 0.0])
        right_upper_arm_dir = normalized_dir(rs, r_elbow)
        right_upper_arm_dir[2] -= 0.05  # 轻微前伸
        right_upper_arm_dir = right_upper_arm_dir / max(np.linalg.norm(right_upper_arm_dir), 1e-6) * self.BONE_LENGTHS["upper_arm"]
        right_upper_arm_pos = right_shoulder_pos + right_upper_arm_dir
        pose_3d["RightUpperArm"] = right_upper_arm_pos.tolist()

        right_lower_arm_dir = normalized_dir(r_elbow, r_wrist)
        right_lower_arm_dir[2] -= 0.08  # 轻微前伸
        right_lower_arm_dir = right_lower_arm_dir / max(np.linalg.norm(right_lower_arm_dir), 1e-6) * self.BONE_LENGTHS["lower_arm"]
        right_lower_arm_pos = right_upper_arm_pos + right_lower_arm_dir
        pose_3d["RightLowerArm"] = right_lower_arm_pos.tolist()

        right_hand_dir = right_lower_arm_dir / max(np.linalg.norm(right_lower_arm_dir), 1e-6) * self.BONE_LENGTHS["hand"]
        right_hand_pos = right_lower_arm_pos + right_hand_dir
        pose_3d["RightHand"] = right_hand_pos.tolist()

        return self.ensure_tracker_nodes(pose_3d)

    def calculate_rotations(self, pose_3d: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """根据骨骼层级计算每个关节的局部旋转（欧拉角 XYZ，弧度）"""
        rotations = {}

        def bone_rotation(parent: List[float], child: List[float]) -> List[float]:
            direction = np.array([child[j] - parent[j] for j in range(3)])
            length = np.linalg.norm(direction)
            if length < 1e-6:
                return [0.0, 0.0, 0.0]
            direction /= length

            # 默认骨骼朝向：Y轴向下（Unity骨骼标准）
            default = np.array([0, -1, 0])
            try:
                rot = R.align_vectors(direction.reshape(1, -1), default.reshape(1, -1))[0]
                euler = rot.as_euler("xyz", degrees=False)
                return [float(euler[0]), float(euler[1]), float(euler[2])]
            except Exception:
                return [0.0, 0.0, 0.0]

        # 髋部旋转：根据肩-髋偏移计算
        ls = pose_3d.get("UpperChest", pose_3d.get("Spine", [0, 0, 0]))
        hip = pose_3d.get("Hips", [0, 0, 0])
        hip_yaw = np.arctan2(ls[0] - hip[0], -(ls[1] - hip[1]))
        hip_roll = np.arctan2(ls[2] - hip[2], -(ls[1] - hip[1]))
        rotations["Hips"] = [float(hip_roll), float(hip_yaw), 0.0]

        # 其余骨骼按层级计算
        for bone, parent_name in self.VRC_BONE_HIERARCHY.items():
            if bone == "Hips":
                continue
            parent_pos = pose_3d.get(parent_name, [0, 0, 0])
            child_pos = pose_3d.get(bone, parent_pos)
            rotations[bone] = bone_rotation(parent_pos, child_pos)

        return rotations

    def smooth_pose(self, pose_3d: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """使用 OneEuro 滤波器平滑3D姿态"""
        t = time.perf_counter()
        smoothed = {}
        for joint, coords in pose_3d.items():
            if self.smoothing == "one_euro":
                f = self._get_pose_filter(joint)
                smoothed[joint] = f.filter(coords, t)
            else:
                # 线性插值平滑（回退方案）
                if self.previous_pose and joint in self.previous_pose:
                    smoothed[joint] = [
                        self.smoothing_factor * coords[i] +
                        (1 - self.smoothing_factor) * self.previous_pose[joint][i]
                        for i in range(3)
                    ]
                else:
                    smoothed[joint] = coords.copy()

        self.previous_pose = copy.deepcopy(smoothed)
        return smoothed

    def smooth_rotations(self, rotations: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """平滑旋转数据"""
        t = time.perf_counter()
        smoothed = {}
        for bone, euler in rotations.items():
            if self.smoothing == "one_euro":
                f = self._get_rotation_filter(bone)
                smoothed[bone] = f.filter(euler, t)
            else:
                if self.previous_rotations and bone in self.previous_rotations:
                    smoothed[bone] = [
                        self.smoothing_factor * euler[i] +
                        (1 - self.smoothing_factor) * self.previous_rotations[bone][i]
                        for i in range(3)
                    ]
                else:
                    smoothed[bone] = euler.copy()

        self.previous_rotations = copy.deepcopy(smoothed)
        return smoothed

    def process_detections(self, detections: Dict, image_width: int = 640,
                          image_height: int = 480,
                          hand_data: Dict = None,
                          flip_horizontal: bool = True,
                          apply_filters: bool = True) -> Optional[Dict]:
        """完整处理流程：2D关键点 -> 归一化 -> 3D姿态 -> 旋转 -> 平滑

        Args:
            detections: YOLO 检测结果
            image_width: 图像宽度
            image_height: 图像高度
            hand_data: 手部追踪数据 (可选)，用于修正手腕位置和手掌旋转
            flip_horizontal: 摄像头是否水平翻转图像
        """
        keypoints_list = detections.get("keypoints")
        selected_index = self._select_person(detections, image_width, image_height)
        if selected_index is None:
            return None

        keypoints_array = keypoints_list[selected_index]
        keypoints_dict = self.keypoints_to_dict(keypoints_array)

        # 推算缺失的髋部关键点（根据肩部位置）
        ls = keypoints_dict.get("left_shoulder")
        rs = keypoints_dict.get("right_shoulder")
        if ls and rs:
            shoulder_mid = self._midpoint(ls, rs)
            shoulder_dist = self._distance_2d(ls, rs)
            # 髋部在肩部下方约1.5倍肩宽处，宽度约为肩宽的0.44倍
            hip_y_offset = shoulder_dist * 1.5
            hip_width_ratio = 0.44
            if "left_hip" not in keypoints_dict:
                keypoints_dict["left_hip"] = (shoulder_mid[0] - shoulder_dist * hip_width_ratio / 2, shoulder_mid[1] + hip_y_offset, 0.2)
            if "right_hip" not in keypoints_dict:
                keypoints_dict["right_hip"] = (shoulder_mid[0] + shoulder_dist * hip_width_ratio / 2, shoulder_mid[1] + hip_y_offset, 0.2)
            if "nose" not in keypoints_dict:
                keypoints_dict["nose"] = self._synthetic_nose_from_upper_body(
                    keypoints_dict,
                    shoulder_mid,
                    shoulder_dist,
                )

        # 推算缺失的膝盖和脚踝关键点（根据髋部位置）
        lh = keypoints_dict.get("left_hip")
        rh = keypoints_dict.get("right_hip")
        # 若肩部未检测到时，用髋部距离作为肩宽近似值
        try:
            _ = shoulder_dist
        except UnboundLocalError:
            shoulder_dist = self._distance_2d(lh, rh) if lh and rh else 0.3
        if lh and rh:
            if "left_knee" not in keypoints_dict:
                keypoints_dict["left_knee"] = (lh[0], lh[1] + shoulder_dist * 1.0, 0.2)
            if "right_knee" not in keypoints_dict:
                keypoints_dict["right_knee"] = (rh[0], rh[1] + shoulder_dist * 1.0, 0.2)
            lk = keypoints_dict.get("left_knee")
            rk = keypoints_dict.get("right_knee")
            if lk and "left_ankle" not in keypoints_dict:
                keypoints_dict["left_ankle"] = (lk[0], lk[1] + shoulder_dist * 1.0, 0.2)
            if rk and "right_ankle" not in keypoints_dict:
                keypoints_dict["right_ankle"] = (rk[0], rk[1] + shoulder_dist * 1.0, 0.2)

        # 至少需要4个关键点才能估计姿态
        if len(keypoints_dict) < 4:
            if self.reset_on_tracking_lost:
                self.reset_filters()
            return None

        # 检查关键关键点（放宽：有肩部即可推算髋部）
        required = ["left_shoulder", "right_shoulder"]
        if not all(keypoints_dict.get(r) for r in required):
            if self.reset_on_tracking_lost:
                self.reset_filters()
            return None

        normalized_kps = self.normalize_keypoints(keypoints_dict, image_width, image_height, flip_horizontal)

        # 如果有手部追踪数据，用手腕关键点修正姿态估计
        if hand_data:
            normalized_kps = self._refine_with_hand_data(normalized_kps, hand_data,
                                                          image_width, image_height)

        pose_3d = self.estimate_3d_pose(normalized_kps)
        if apply_filters:
            pose_3d = self.apply_average_smoothing(pose_3d)
            pose_3d = self.smooth_pose(pose_3d)
        rotations = self.calculate_rotations(pose_3d)

        # 如果有手部追踪数据，用手掌旋转修正手臂到手掌的连接
        if hand_data:
            rotations = self._refine_arm_hand_rotations(rotations, hand_data)

        if apply_filters:
            rotations = self.smooth_rotations(rotations)

        return {
            "pose_3d": pose_3d,
            "rotations": rotations,
            "keypoints_2d": keypoints_dict,
            "confidence": (detections.get("confidence") or [])[selected_index]
            if selected_index < len(detections.get("confidence") or []) else 0
        }

    def _refine_with_hand_data(self, norm_kps: Dict, hand_data,
                                img_w: int, img_h: int) -> Dict:
        """使用手部追踪数据修正手腕关键点位置

        MediaPipe Hands 的手腕关键点 (index 0) 比 YOLO 的手腕关键点更精确，
        因为它专门针对手部做了优化。

        hand_data 可以是 List[Dict] 或 Dict 格式
        """
        refined = dict(norm_kps)

        # 将列表格式转为字典格式
        if isinstance(hand_data, list):
            hand_dict = {}
            for h in hand_data:
                label = h.get("label", "unknown")
                hand_dict[label] = h
            hand_data = hand_dict

        for hand_label, wrist_name in [("left", "left_wrist"), ("right", "right_wrist")]:
            hand = hand_data.get(hand_label)
            if hand is None:
                continue

            landmarks = hand.get("landmarks")
            if landmarks is None or len(landmarks) < 1:
                continue

            # MediaPipe 手腕关键点
            mp_wrist = landmarks[0]  # (x, y) 像素坐标

            # 归一化到与 norm_kps 相同的坐标系
            # 需要获取当前归一化参数
            left_hip = norm_kps.get("left_hip", (0, 0, 0))
            right_hip = norm_kps.get("right_hip", (0, 0, 0))
            hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

            # 简单归一化：以图像中心为参考
            norm_x = (mp_wrist[0] / img_w - 0.5) * 2
            norm_y = (mp_wrist[1] / img_h - 0.5) * 2

            # 混合 YOLO 手腕和 MediaPipe 手腕位置
            # MediaPipe 手腕更精确，给更高权重
            yolo_wrist = norm_kps.get(wrist_name)
            if yolo_wrist and yolo_wrist[2] > 0.3:
                # 加权融合: MediaPipe 0.6, YOLO 0.4
                blend_x = 0.6 * norm_x + 0.4 * yolo_wrist[0]
                blend_y = 0.6 * norm_y + 0.4 * yolo_wrist[1]
                refined[wrist_name] = (blend_x, blend_y, max(yolo_wrist[2], 0.8))
            else:
                # YOLO 手腕不可靠，直接使用 MediaPipe
                refined[wrist_name] = (norm_x, norm_y, 0.9)

        return refined

    def _refine_arm_hand_rotations(self, rotations: Dict, hand_data) -> Dict:
        """使用手部追踪数据修正手臂到手掌的旋转连接

        优化策略:
        1. 用手部旋转修正 LowerArm -> Hand 的旋转过渡
        2. 平滑手臂末端的旋转，避免手腕处出现不自然的突变
        3. 用手指弯曲度推断握拳/张开状态，微调手掌旋转
        4. 根据手掌/手背朝向调整旋转混合权重和手腕旋转补偿
        """
        refined = dict(rotations)

        # 将列表格式转为字典格式
        if isinstance(hand_data, list):
            hand_dict = {}
            for h in hand_data:
                label = h.get("label", "unknown")
                hand_dict[label] = h
            hand_data = hand_dict

        for hand_label, hand_bones in [
            ("left", ["LeftLowerArm", "LeftHand"]),
            ("right", ["RightLowerArm", "RightHand"]),
        ]:
            hand = hand_data.get(hand_label)
            if hand is None:
                continue

            hand_rotation = hand.get("rotation", [0, 0, 0])
            if not hand_rotation or all(abs(r) < 0.001 for r in hand_rotation):
                continue

            lower_arm_rot = refined.get(hand_bones[0], [0, 0, 0])
            current_hand_rot = refined.get(hand_bones[1], [0, 0, 0])

            # 根据手掌/手背朝向调整混合权重
            palm_side = hand.get("palm_side", "palm")
            palm_conf = hand.get("palm_confidence", 0.6)

            if palm_side == "back":
                # 手背朝向时，手部追踪旋转精度较低，降低手部追踪权重
                hand_weight = 0.35 + 0.15 * palm_conf  # 0.35-0.50
                arm_weight = 1.0 - hand_weight

                # 手背时手腕旋转补偿：增加 roll 补偿
                # 手背朝向时手腕扭转角度需要额外补偿
                back_roll_compensation = 0.15 * palm_conf
                hand_rotation = list(hand_rotation)
                hand_rotation[2] += back_roll_compensation
            else:
                # 手掌朝向时，手部追踪旋转更精确，给更高权重
                hand_weight = 0.5 + 0.15 * palm_conf  # 0.50-0.65
                arm_weight = 1.0 - hand_weight

            # 手腕旋转 = 下臂旋转和手部追踪旋转的混合
            blended_hand_rot = [
                arm_weight * current_hand_rot[i] + hand_weight * hand_rotation[i]
                for i in range(3)
            ]

            # 限制手腕旋转范围，避免不自然的扭曲
            # 手背朝向时允许更大的 roll 范围（手腕自然翻转）
            max_wrist_angle = 1.2  # ~70度
            if palm_side == "back":
                max_wrist_roll = 1.8  # 手背时允许更大扭转
            else:
                max_wrist_roll = max_wrist_angle

            for i in range(3):
                max_angle = max_wrist_roll if i == 2 else max_wrist_angle
                diff = blended_hand_rot[i] - lower_arm_rot[i]
                diff = max(-max_angle, min(max_angle, diff))
                blended_hand_rot[i] = lower_arm_rot[i] + diff

            refined[hand_bones[1]] = blended_hand_rot

        return refined

    def reset_filters(self):
        """重置所有滤波器状态"""
        self.pose_filters.clear()
        self.rotation_filters.clear()
        self.previous_pose = None
        self.previous_rotations = None
        self._reset_average_smoothing()
