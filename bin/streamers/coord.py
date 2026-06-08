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

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from scipy.spatial.transform import Rotation as Rot
except ImportError:
    Rot = None


class CoordinateTransformer:
    """坐标系变换器

    负责将动捕推理输出的坐标变换到目标坐标系，
    并基于 HMD 位置建立世界坐标系对齐。
    """

    # 翻转矩阵: 摄像头 (X左,Y下) -> Unity (X右,Y上)
    _FLIP_XY = np.diag([-1.0, -1.0, 1.0])

    # Y轴180度翻转: Unity (Z前) -> OpenVR (Z后)
    _FLIP_Y180 = np.diag([-1.0, 1.0, -1.0])

    def __init__(self, hmd_to_neck_offset: List[float] = None):
        """
        Args:
            hmd_to_neck_offset: HMD 到颈部的偏移 [x, y, z] (Unity 坐标系)
                                默认 [0, -0.2, 0.1] (颈部在 HMD 下方 0.2m, 前方 0.1m)
        """
        self._hmd_to_neck_offset = np.array(
            hmd_to_neck_offset or [0, -0.2, 0.1]
        )

    @property
    def hmd_to_neck_offset(self) -> np.ndarray:
        return self._hmd_to_neck_offset

    @hmd_to_neck_offset.setter
    def hmd_to_neck_offset(self, value: List[float]):
        self._hmd_to_neck_offset = np.array(value)

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
    ) -> Dict[str, List[float]]:
        """计算 OSC 模式下的相对位置 (相对于颈部偏移)

        OSC 模式: 追踪器位置 = 骨骼位置 - (颈部位置 - 颈部偏移)
        VRChat 会将追踪器位置叠加到头部位置上。

        Args:
            unity_pose: Unity 坐标系的骨骼位置
            hmd_position: HMD 真实位置 (未使用，OSC 模式不需要)

        Returns:
            相对颈部偏移后的骨骼位置
        """
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

    def compute_steamvr_positions(
        self,
        unity_pose: Dict[str, List[float]],
        hmd_position: List[float],
        hmd_quaternion: Optional[List[float]] = None,
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
        neck_pos = np.array(unity_pose.get("Neck", [0, 1.5, 0]))

        # 旋转 neck_offset 到 HMD 朝向
        if hmd_quaternion is not None and Rot is not None:
            try:
                # VRChat 四元数 [w,x,y,z] -> scipy [x,y,z,w]
                quat = hmd_quaternion
                rot = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]])
                # VRChat -> OpenVR: 绕 Y 轴 180 度
                mat_rot = rot.as_matrix()
                mat_openvr = self._FLIP_Y180 @ mat_rot
                rot_openvr = Rot.from_matrix(mat_openvr)
                neck_offset_rotated = rot_openvr.apply(self._hmd_to_neck_offset)
            except Exception:
                neck_offset_rotated = self._hmd_to_neck_offset
        else:
            neck_offset_rotated = self._hmd_to_neck_offset

        # HMD 位置: VRChat (Z前) -> OpenVR (Z后)
        hmd_pos_openvr = np.array([hmd_position[0], hmd_position[1], -hmd_position[2]])

        # offset = neck_pos - (hmd_pos + neck_offset_rotated)
        offset = neck_pos - (hmd_pos_openvr + neck_offset_rotated)

        result = {}
        for bone, pos in unity_pose.items():
            result[bone] = (np.array(pos) - offset).tolist()
        return result

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
        self, euler_rad: List[float]
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
            # Unity -> OpenVR: 绕 Y 轴 180 度
            mat_openvr = self._FLIP_Y180 @ rot_new.as_matrix()
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
