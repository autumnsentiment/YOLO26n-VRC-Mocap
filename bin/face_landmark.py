import cv2
import math
import numpy as np
import threading
import time
import os
from typing import Dict, Optional, List, Tuple


# MediaPipe 52 blendshapes 名称 -> VRCFT Unified Expressions v2 参数映射
# 参考: https://docs.vrcft.io/docs/tutorial-avatars/tutorial-avatars-extras/parameters
# MediaPipe blendshapes 列表: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#face_blendshapes
MEDIAPIPE_TO_VRCFT = {
    # 下巴/嘴部开合
    "jawOpen":               "v2/JawOpen",
    "mouthClose":            "v2/MouthClosed",
    "mouthFunnel":           "v2/LipFunnel",
    "mouthPucker":           "v2/LipPucker",
    # 嘴部左右
    "jawLeft":               "v2/JawX",          # 负值=左
    "jawRight":              "v2/JawX",          # 正值=右
    "mouthLeft":             "v2/MouthX",         # 负值=左
    "mouthRight":            "v2/MouthX",         # 正值=右
    # 嘴角
    "mouthSmileLeft":        "v2/MouthSmileLeft",
    "mouthSmileRight":       "v2/MouthSmileRight",
    "mouthFrownLeft":        "v2/MouthFrownLeft",
    "mouthFrownRight":       "v2/MouthFrownRight",
    "mouthDimpleLeft":       "v2/MouthDimpleLeft",
    "mouthDimpleRight":      "v2/MouthDimpleRight",
    "mouthPressLeft":        "v2/MouthPressLeft",
    "mouthPressRight":       "v2/MouthPressRight",
    "mouthStretchLeft":      "v2/MouthStretchLeft",
    "mouthStretchRight":     "v2/MouthStretchRight",
    "mouthLowerDownLeft":    "v2/MouthLowerDownLeft",
    "mouthLowerDownRight":   "v2/MouthLowerDownRight",
    "mouthUpperUpLeft":      "v2/MouthUpperUpLeft",
    "mouthUpperUpRight":     "v2/MouthUpperUpRight",
    "mouthRollLower":        "v2/MouthRaiserLower",
    "mouthRollUpper":        "v2/MouthRaiserUpper",
    "mouthShrugLower":       "v2/MouthLowerDownLeft",   # 近似映射
    "mouthShrugUpper":       "v2/MouthUpperUpLeft",     # 近似映射
    # 眼睛
    "eyeBlinkLeft":          "v2/EyeLidLeft",
    "eyeBlinkRight":         "v2/EyeLidRight",
    "eyeWideLeft":           "v2/EyeLidLeft",           # 宽眼叠加
    "eyeWideRight":          "v2/EyeLidRight",          # 宽眼叠加
    "eyeSquintLeft":         "v2/EyeSquintLeft",
    "eyeSquintRight":        "v2/EyeSquintRight",
    "eyeLookDownLeft":       "v2/EyeLeftY",             # 负值=下看
    "eyeLookDownRight":      "v2/EyeRightY",
    "eyeLookUpLeft":         "v2/EyeLeftY",             # 正值=上看
    "eyeLookUpRight":        "v2/EyeRightY",
    "eyeLookInLeft":         "v2/EyeLeftX",             # 负值=内看
    "eyeLookInRight":        "v2/EyeRightX",            # 正值=内看
    "eyeLookOutLeft":        "v2/EyeLeftX",             # 正值=外看
    "eyeLookOutRight":       "v2/EyeRightX",            # 负值=外看
    # 眉毛
    "browDownLeft":          "v2/BrowLowererLeft",
    "browDownRight":         "v2/BrowLowererRight",
    "browInnerUp":           "v2/BrowInnerUpLeft",      # 合并映射
    "browOuterUpLeft":       "v2/BrowOuterUpLeft",
    "browOuterUpRight":      "v2/BrowOuterUpRight",
    # 鼻子
    "noseSneerLeft":         "v2/NoseSneerLeft",
    "noseSneerRight":        "v2/NoseSneerRight",
    # 脸颊
    "cheekSquintLeft":       "v2/CheekSquintLeft",
    "cheekSquintRight":      "v2/CheekSquintRight",
    "cheekPuff":             "v2/CheekPuffSuck",        # 合并映射
    # 舌头
    "tongueOut":             "v2/TongueOut",
}

# 需要特殊处理的左右对称参数（MediaPipe 左右参数合并为 VRCFT 单一参数）
_MERGE_PARAMS = {
    "v2/JawX":       {"jawLeft": -1.0, "jawRight": 1.0},
    "v2/MouthX":     {"mouthLeft": -1.0, "mouthRight": 1.0},
    "v2/EyeLeftY":   {"eyeLookDownLeft": -1.0, "eyeLookUpLeft": 1.0},
    "v2/EyeRightY":  {"eyeLookDownRight": -1.0, "eyeLookUpRight": 1.0},
    "v2/EyeLeftX":   {"eyeLookInLeft": -1.0, "eyeLookOutLeft": 1.0},
    "v2/EyeRightX":  {"eyeLookInRight": 1.0, "eyeLookOutRight": -1.0},
}

# 眼睛闭合参数: blink 值反转 (1-blink = 睁开度), widen 值叠加
_EYE_COMBINE = {
    "v2/EyeLidLeft":  {"blink": "eyeBlinkLeft", "widen": "eyeWideLeft"},
    "v2/EyeLidRight": {"blink": "eyeBlinkRight", "widen": "eyeWideRight"},
}

# 眉毛内侧上扬: browInnerUp 同时映射到左右
_BROW_INNER_UP = "browInnerUp"


class OneEuroFilter1D:
    """1D OneEuro 滤波器 - 低频平滑，高频低延迟
    
    低光环境下关键点抖动严重，OneEuro 比线性插值更有效：
    - 静止时（低频）强平滑，消除噪声
    - 快速运动时（高频）低延迟，保持响应
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

    def filter(self, x: float, t: float = None) -> float:
        if self.t_prev is None:
            self.t_prev = t or time.perf_counter()
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        t = t or time.perf_counter()
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1.0 / self.freq

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

    def _smoothing_factor(self, cutoff: float) -> float:
        r = 2 * np.pi * cutoff / self.freq
        return 1.0 / (1.0 + r + r * r / 2.0)

    def _exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev


class FaceLandmarkDetector:
    """面部关键点检测器
    
    使用 MediaPipe Face Landmarker (Tasks API) 检测面部关键点，
    计算嘴部开合度、眨眼等表情参数，用于 VRChat OSC 驱动唇形同步。
    
    兼容 MediaPipe 0.10+ 新版 Tasks API：
    - 使用 mp.tasks.vision.FaceLandmarker 替代已废弃的 mp.solutions.face_mesh
    - 使用 .task 模型文件
    - 使用 mp.Image 作为输入
    
    低光环境优化:
    - 降低 min_detection_confidence / min_tracking_confidence (0.3)
    - 增大 OneEuro beta 参数 (0.02~0.05) 增强平滑
    - 降低 min_cutoff (0.5) 增强低频平滑
    """

    # MediaPipe Face Mesh 嘴部关键点索引
    # 外嘴唇轮廓
    LIP_OUTER_TOP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LIP_OUTER_BOTTOM = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    # 内嘴唇轮廓
    LIP_INNER_TOP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    LIP_INNER_BOTTOM = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

    # 眼睛关键点
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_LEFT = 362
    RIGHT_EYE_RIGHT = 263

    # 眉毛关键点
    LEFT_BROW_INNER = 107
    LEFT_BROW_OUTER = 70
    RIGHT_BROW_INNER = 336
    RIGHT_BROW_OUTER = 300

    # 鼻尖
    NOSE_TIP = 1

    # 模型文件搜索路径（相对于本文件和项目根目录）
    _MODEL_SEARCH_DIRS = [
        os.path.dirname(os.path.abspath(__file__)),               # bin/
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # 项目根
    ]

    def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5,
                 min_face_presence_confidence: float = 0.5,
                 smoothing: str = "one_euro", smoothing_factor: float = 0.3,
                 one_euro_min_cutoff: float = 1.0, one_euro_beta: float = 0.007,
                 blendshape_smoothing: float = 0.0,
                 head_position_scale: float = 1.0,
                 head_position_offset: list = None,
                 head_rotation_smoothing: float = 0.3):
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_face_presence_confidence = min_face_presence_confidence

        self.face_landmarker = None
        self.is_loaded = False

        # 平滑参数
        self.smoothing = smoothing
        self.smoothing_factor = smoothing_factor
        self.one_euro_min_cutoff = one_euro_min_cutoff
        self.one_euro_beta = one_euro_beta

        # blendshape 线性插值平滑强度 (0=禁用, 0~1之间, 参考 mediapipe-vt)
        # 值越大越平滑但延迟越高, 推荐值 0.3~0.5
        self.blendshape_smoothing = blendshape_smoothing

        # ===== 嘴部低延迟参数 =====
        # 嘴部动作（说话）变化频率远高于眼睛/眉毛，需要更低的平滑延迟
        # mouth_blendshape_smoothing: 嘴部 blendshape 独立平滑系数
        #   0=禁用, 推荐值 0.15~0.25 (比全局 blendshape_smoothing 更低)
        self.mouth_blendshape_smoothing = 0.2
        # mouth_skip_vrcft_smooth: 嘴部 VRCFT 参数跳过 OneEuro 二次滤波
        #   避免双重平滑叠加延迟，嘴部参数已在 blendshape 阶段平滑过
        self.mouth_skip_vrcft_smooth = True

        # ===== 佩戴头显优化参数 =====
        # head_position_scale: 头部位置缩放因子
        #   佩戴头显时 MediaPipe 面部变换矩阵的位置偏小，需要放大
        #   推荐值: 1.0~3.0, 默认 1.0 (不缩放)
        self.head_position_scale = head_position_scale
        # head_position_offset: 头部位置偏移 [x, y, z]
        #   用于补偿头显佩戴时摄像头与面部中心的偏移
        #   默认 [0, 0, 0]
        self.head_position_offset = head_position_offset or [0.0, 0.0, 0.0]
        # head_rotation_smoothing: 头部旋转线性插值平滑 (0=禁用, 1=完全平滑)
        #   佩戴头显时旋转数据可能有抖动，需要额外平滑
        #   推荐值: 0.2~0.5
        self.head_rotation_smoothing = head_rotation_smoothing

        # OneEuro 滤波器（每个参数独立滤波）
        self._filters: Dict[str, OneEuroFilter1D] = {}

        # 线性插值平滑值（回退方案）
        self._smooth_values: Dict[str, float] = {}

        # 校准参考值
        self._mouth_open_ref = None
        self._mouth_wide_ref = None
        self._calibrated = False

        # ===== 异步推理模式 (参考 mediapipe-vt) =====
        # LIVE_STREAM 模式下推理在 MediaPipe 内部线程执行，不阻塞主循环
        # 回调结果缓存到 _latest_result，主循环通过 detect() 读取
        self._latest_result = None
        self._result_lock = threading.Lock()
        self._frame_timestamp = 0

        # blendshapes 平滑缓存 (参考 mediapipe-vt blendshape_smoothing)
        self._bs_smoothed: Dict[str, float] = {}

        # 面部变换矩阵结果 (参考 mediapipe-vt facial_transformation_matrixes)
        # 提供精确的头部位置和旋转，比 YOLO 姿态估计更准确
        self._face_position = [0.0, 0.0, 0.0]   # x, y, z
        self._face_rotation = [0.0, 0.0, 0.0]    # pitch, yaw, roll (度)
        self._face_rotation_smoothed = [0.0, 0.0, 0.0]  # 平滑后的旋转

    def _find_model_file(self) -> Optional[str]:
        """搜索 face_landmarker.task 模型文件"""
        model_name = "face_landmarker.task"
        for d in self._MODEL_SEARCH_DIRS:
            path = os.path.join(d, model_name)
            if os.path.exists(path):
                return path
        return None

    def load(self) -> bool:
        try:
            import mediapipe as mp

            model_path = self._find_model_file()
            if not model_path:
                print("face_landmarker.task not found, face tracking disabled")
                print("Download from: https://storage.googleapis.com/mediapipe-models/"
                      "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
                return False

            tasks = mp.tasks.vision
            BaseOptions = mp.tasks.BaseOptions

            # 使用 LIVE_STREAM 异步模式 (参考 mediapipe-vt)
            # 优势: 推理在 MediaPipe 内部线程执行，不阻塞主循环
            # 回调函数 _on_face_result 在推理完成后被调用
            options = tasks.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=tasks.RunningMode.LIVE_STREAM,
                num_faces=self.max_num_faces,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_face_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,  # 启用面部变换矩阵 (参考 mediapipe-vt)
                result_callback=self._on_face_result,
            )

            self.face_landmarker = tasks.FaceLandmarker.create_from_options(options)
            self.is_loaded = True
            print(f"Face Landmarker loaded (LIVE_STREAM mode, det_conf={self.min_detection_confidence}, "
                  f"presence_conf={self.min_face_presence_confidence}, trk_conf={self.min_tracking_confidence})")
            if self.blendshape_smoothing > 0:
                print(f"  Blendshape smoothing: {self.blendshape_smoothing}")
            if self.head_position_scale != 1.0 or any(v != 0 for v in self.head_position_offset):
                print(f"  Head position: scale={self.head_position_scale}, offset={self.head_position_offset}")
            if self.head_rotation_smoothing > 0:
                print(f"  Head rotation smoothing: {self.head_rotation_smoothing}")
            return True
        except ImportError:
            print("mediapipe not installed, face tracking disabled")
            print("Install with: pip install mediapipe")
            return False
        except Exception as e:
            print(f"Error loading Face Landmarker: {e}")
            return False

    def _on_face_result(self, result, output_image, timestamp_ms: int):
        """LIVE_STREAM 模式的异步回调 (参考 mediapipe-vt pred_callback)
        
        在 MediaPipe 内部线程中被调用，将推理结果缓存到 _latest_result。
        主循环通过 detect() 读取缓存的结果，实现推理和主循环解耦。
        """
        with self._result_lock:
            self._latest_result = result

    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        if not self.is_loaded or self.face_landmarker is None:
            return None

        import mediapipe as mp

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # LIVE_STREAM 异步模式: 提交帧到推理队列，不阻塞
        # 参考 mediapipe-vt: detect_async + 回调缓存结果
        self._frame_timestamp += 1
        ts = self._frame_timestamp
        self.face_landmarker.detect_async(mp_image, ts)

        # 读取异步回调缓存的最新结果
        with self._result_lock:
            result = self._latest_result

        if result is None or not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]

        # 转换为像素坐标
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

        # 提取 MediaPipe blendshapes (52个)
        bs_dict = {}
        if result.face_blendshapes:
            for bs in result.face_blendshapes[0]:
                bs_dict[bs.category_name] = bs.score

        # blendshape 平滑 (参考 mediapipe-vt blendshape_smoothing)
        # 线性插值: smoothed = smoothed * strength + raw * (1 - strength)
        # 嘴部参数使用独立低平滑系数以降低延迟
        _MOUTH_BLENDSHAPES = {
            "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
            "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
            "mouthDimpleLeft", "mouthDimpleRight", "mouthPressLeft", "mouthPressRight",
            "mouthStretchLeft", "mouthStretchRight", "mouthLowerDownLeft", "mouthLowerDownRight",
            "mouthUpperUpLeft", "mouthUpperUpRight", "mouthRollLower", "mouthRollUpper",
            "mouthShrugLower", "mouthShrugUpper", "mouthLeft", "mouthRight",
            "jawLeft", "jawRight", "tongueOut",
        }
        for key in bs_dict:
            raw = bs_dict[key]
            # 嘴部参数使用独立低平滑系数
            if key in _MOUTH_BLENDSHAPES and self.mouth_blendshape_smoothing >= 0:
                strength = self.mouth_blendshape_smoothing
            else:
                strength = self.blendshape_smoothing
            if strength > 0:
                if key in self._bs_smoothed:
                    self._bs_smoothed[key] = (
                        self._bs_smoothed[key] * strength
                        + raw * (1.0 - strength)
                    )
                else:
                    self._bs_smoothed[key] = raw
                bs_dict[key] = self._bs_smoothed[key]

        # 提取面部变换矩阵 (参考 mediapipe-vt facial_transformation_matrixes)
        # 提供精确的头部位置和旋转，比 YOLO 姿态估计更准确
        if result.facial_transformation_matrixes:
            mat = np.array(result.facial_transformation_matrixes[0])
            # 位置: 变换矩阵第4列 (参考 mediapipe-vt)
            # MediaPipe 坐标系: X右, Y上, Z前(朝向摄像头)
            # VRChat 坐标系: X右, Y上, Z前
            # 位置需要翻转 X 轴 (因为摄像头是镜像的)
            raw_pos = [-mat[0][3], mat[1][3], mat[2][3]]
            # 佩戴头显优化: 缩放 + 偏移
            self._face_position = [
                raw_pos[0] * self.head_position_scale + self.head_position_offset[0],
                raw_pos[1] * self.head_position_scale + self.head_position_offset[1],
                raw_pos[2] * self.head_position_scale + self.head_position_offset[2],
            ]
            # 旋转: 从3x3旋转矩阵提取欧拉角 (参考 mediapipe-vt)
            # pitch (X轴旋转), yaw (Y轴旋转), roll (Z轴旋转)
            raw_rot = [
                np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2)) * 180.0 / math.pi,
                np.arctan2(mat[2, 1], mat[2, 2]) * 180.0 / math.pi,
                np.arctan2(mat[1, 0], mat[0, 0]) * 180.0 / math.pi,
            ]
            # 头部旋转平滑 (佩戴头显时减少抖动)
            if self.head_rotation_smoothing > 0:
                for i in range(3):
                    self._face_rotation_smoothed[i] = (
                        self._face_rotation_smoothed[i] * self.head_rotation_smoothing
                        + raw_rot[i] * (1.0 - self.head_rotation_smoothing)
                    )
                self._face_rotation = list(self._face_rotation_smoothed)
            else:
                self._face_rotation = raw_rot
                self._face_rotation_smoothed = list(raw_rot)

        # 计算 VRCFT Unified Expressions 参数
        vrcft_params = self._blendshapes_to_vrcft(bs_dict)

        # 计算简化面部参数（兼容旧接口）
        face_data = self._compute_face_params(points, w, h)
        face_data["landmarks"] = points
        face_data["landmarks_raw"] = landmarks

        # 添加 VRCFT 标准参数
        face_data["vrcft"] = vrcft_params
        # 添加原始 blendshapes（调试用）
        face_data["blendshapes_raw"] = bs_dict

        # 添加面部变换矩阵数据 (参考 mediapipe-vt)
        face_data["face_position"] = list(self._face_position)
        face_data["face_rotation"] = list(self._face_rotation)

        return face_data

    def _blendshapes_to_vrcft(self, bs_dict: Dict[str, float]) -> Dict[str, float]:
        """将 MediaPipe 52 blendshapes 转换为 VRCFT Unified Expressions v2 参数

        参考 VRCFT 标准:
        - Jaw 参数: v2/JawOpen, v2/MouthClosed, v2/JawX, v2/JawZ
        - 嘴部参数: v2/MouthSmileLeft/Right, v2/MouthFrownLeft/Right, etc.
        - 眼睛参数: v2/EyeLidLeft/Right (0-0.75=睁开, 0.75-1.0=宽眼)
        - 眉毛参数: v2/BrowInnerUpLeft/Right, v2/BrowOuterUpLeft/Right, etc.
        - 唇部参数: v2/LipFunnel, v2/LipPucker, etc.

        Args:
            bs_dict: MediaPipe blendshapes 字典 {name: score(0-1)}

        Returns:
            VRCFT v2 参数字典 {param_name: value}
        """
        vrcft = {}
        t = time.perf_counter()

        # ===== 1. 直接映射的参数 =====
        direct_map = {
            "jawOpen":             "v2/JawOpen",
            "mouthClose":          "v2/MouthClosed",
            "mouthFunnel":         "v2/LipFunnel",
            "mouthPucker":         "v2/LipPucker",
            "mouthSmileLeft":      "v2/MouthSmileLeft",
            "mouthSmileRight":     "v2/MouthSmileRight",
            "mouthFrownLeft":      "v2/MouthFrownLeft",
            "mouthFrownRight":     "v2/MouthFrownRight",
            "mouthDimpleLeft":     "v2/MouthDimpleLeft",
            "mouthDimpleRight":    "v2/MouthDimpleRight",
            "mouthPressLeft":      "v2/MouthPressLeft",
            "mouthPressRight":     "v2/MouthPressRight",
            "mouthStretchLeft":    "v2/MouthStretchLeft",
            "mouthStretchRight":   "v2/MouthStretchRight",
            "mouthLowerDownLeft":  "v2/MouthLowerDownLeft",
            "mouthLowerDownRight": "v2/MouthLowerDownRight",
            "mouthUpperUpLeft":    "v2/MouthUpperUpLeft",
            "mouthUpperUpRight":   "v2/MouthUpperUpRight",
            "mouthRollLower":      "v2/MouthRaiserLower",
            "mouthRollUpper":      "v2/MouthRaiserUpper",
            "eyeSquintLeft":       "v2/EyeSquintLeft",
            "eyeSquintRight":      "v2/EyeSquintRight",
            "browDownLeft":        "v2/BrowLowererLeft",
            "browDownRight":       "v2/BrowLowererRight",
            "browOuterUpLeft":     "v2/BrowOuterUpLeft",
            "browOuterUpRight":    "v2/BrowOuterUpRight",
            "noseSneerLeft":       "v2/NoseSneerLeft",
            "noseSneerRight":      "v2/NoseSneerRight",
            "cheekSquintLeft":     "v2/CheekSquintLeft",
            "cheekSquintRight":    "v2/CheekSquintRight",
            "cheekPuff":           "v2/CheekPuffSuck",
            "tongueOut":           "v2/TongueOut",
        }
        # 嘴部 VRCFT 参数名集合（跳过 OneEuro 二次滤波，降低延迟）
        _MOUTH_VRCFT_PARAMS = {
            "v2/JawOpen", "v2/MouthClosed", "v2/LipFunnel", "v2/LipPucker",
            "v2/MouthSmileLeft", "v2/MouthSmileRight", "v2/MouthFrownLeft", "v2/MouthFrownRight",
            "v2/MouthDimpleLeft", "v2/MouthDimpleRight", "v2/MouthPressLeft", "v2/MouthPressRight",
            "v2/MouthStretchLeft", "v2/MouthStretchRight",
            "v2/MouthLowerDownLeft", "v2/MouthLowerDownRight",
            "v2/MouthUpperUpLeft", "v2/MouthUpperUpRight",
            "v2/MouthRaiserLower", "v2/MouthRaiserUpper",
            "v2/TongueOut",
        }
        for mp_name, vrcft_name in direct_map.items():
            val = bs_dict.get(mp_name, 0.0)
            # 嘴部参数跳过 OneEuro 二次滤波（已在 blendshape 阶段用低系数平滑过）
            if self.mouth_skip_vrcft_smooth and vrcft_name in _MOUTH_VRCFT_PARAMS:
                vrcft[vrcft_name] = float(np.clip(val, 0.0, 1.0))
            else:
                val = self._apply_smooth(f"vrcft_{vrcft_name}", val, t)
                vrcft[vrcft_name] = float(np.clip(val, 0.0, 1.0))

        # ===== 2. 合并映射: JawX (左负右正) =====
        jaw_left = bs_dict.get("jawLeft", 0.0)
        jaw_right = bs_dict.get("jawRight", 0.0)
        jaw_x = jaw_right - jaw_left
        if self.mouth_skip_vrcft_smooth:
            vrcft["v2/JawX"] = float(np.clip(jaw_x, -1.0, 1.0))
        else:
            jaw_x = self._apply_smooth("vrcft_v2/JawX", jaw_x, t)
            vrcft["v2/JawX"] = float(np.clip(jaw_x, -1.0, 1.0))

        # ===== 3. 合并映射: MouthX (左负右正) =====
        mouth_left = bs_dict.get("mouthLeft", 0.0)
        mouth_right = bs_dict.get("mouthRight", 0.0)
        mouth_x = mouth_right - mouth_left
        if self.mouth_skip_vrcft_smooth:
            vrcft["v2/MouthX"] = float(np.clip(mouth_x, -1.0, 1.0))
        else:
            mouth_x = self._apply_smooth("vrcft_v2/MouthX", mouth_x, t)
            vrcft["v2/MouthX"] = float(np.clip(mouth_x, -1.0, 1.0))

        # ===== 4. 眼睛: EyeLid (0-0.75=睁开度, 0.75-1.0=宽眼) =====
        # VRCFT 标准: EyeLidLeft/Right
        #   0.0 = 完全闭合, 0.75 = 正常睁开, 1.0 = 睁大
        # MediaPipe: eyeBlink (0=睁开, 1=闭合), eyeWide (0=正常, 1=宽眼)
        for side, blink_name, widen_name in [
            ("Left", "eyeBlinkLeft", "eyeWideLeft"),
            ("Right", "eyeBlinkRight", "eyeWideRight"),
        ]:
            blink = bs_dict.get(blink_name, 0.0)
            widen = bs_dict.get(widen_name, 0.0)
            # 闭合度反转: openness = 1 - blink
            openness = 1.0 - blink
            # 映射到 VRCFT 范围: 0-0.75 为睁开度, 0.75-1.0 为宽眼
            eye_lid = openness * 0.75 + widen * 0.25
            eye_lid = self._apply_smooth(f"vrcft_v2/EyeLid{side}", eye_lid, t)
            vrcft[f"v2/EyeLid{side}"] = float(np.clip(eye_lid, 0.0, 1.0))

        # ===== 5. 眼睛视线: EyeX/Y =====
        # EyeLeftX: 正=右看, 负=左看
        look_in_l = bs_dict.get("eyeLookInLeft", 0.0)
        look_out_l = bs_dict.get("eyeLookOutLeft", 0.0)
        eye_left_x = look_out_l - look_in_l
        eye_left_x = self._apply_smooth("vrcft_v2/EyeLeftX", eye_left_x, t)
        vrcft["v2/EyeLeftX"] = float(np.clip(eye_left_x, -1.0, 1.0))

        look_in_r = bs_dict.get("eyeLookInRight", 0.0)
        look_out_r = bs_dict.get("eyeLookOutRight", 0.0)
        eye_right_x = look_in_r - look_out_r
        eye_right_x = self._apply_smooth("vrcft_v2/EyeRightX", eye_right_x, t)
        vrcft["v2/EyeRightX"] = float(np.clip(eye_right_x, -1.0, 1.0))

        # EyeLeftY: 正=上看, 负=下看
        look_up_l = bs_dict.get("eyeLookUpLeft", 0.0)
        look_down_l = bs_dict.get("eyeLookDownLeft", 0.0)
        eye_left_y = look_up_l - look_down_l
        eye_left_y = self._apply_smooth("vrcft_v2/EyeLeftY", eye_left_y, t)
        vrcft["v2/EyeLeftY"] = float(np.clip(eye_left_y, -1.0, 1.0))

        look_up_r = bs_dict.get("eyeLookUpRight", 0.0)
        look_down_r = bs_dict.get("eyeLookDownRight", 0.0)
        eye_right_y = look_up_r - look_down_r
        eye_right_y = self._apply_smooth("vrcft_v2/EyeRightY", eye_right_y, t)
        vrcft["v2/EyeRightY"] = float(np.clip(eye_right_y, -1.0, 1.0))

        # ===== 6. 眉毛内侧上扬: browInnerUp 同时映射到左右 =====
        brow_inner = bs_dict.get("browInnerUp", 0.0)
        brow_inner_l = self._apply_smooth("vrcft_v2/BrowInnerUpLeft", brow_inner, t)
        brow_inner_r = self._apply_smooth("vrcft_v2/BrowInnerUpRight", brow_inner, t)
        vrcft["v2/BrowInnerUpLeft"] = float(np.clip(brow_inner_l, 0.0, 1.0))
        vrcft["v2/BrowInnerUpRight"] = float(np.clip(brow_inner_r, 0.0, 1.0))

        # ===== 7. 简化参数 (VRCFT Blended Shapes) =====
        # MouthOpen = MouthUpperUp + MouthLowerDown
        vrcft["v2/MouthOpen"] = float(np.clip(
            (vrcft.get("v2/MouthUpperUpLeft", 0) + vrcft.get("v2/MouthUpperUpRight", 0)) / 2 +
            (vrcft.get("v2/MouthLowerDownLeft", 0) + vrcft.get("v2/MouthLowerDownRight", 0)) / 2,
            0.0, 1.0
        ))

        # LipFunnel (合并上下)
        vrcft["v2/LipFunnel"] = vrcft.get("v2/LipFunnel", 0.0)
        # LipPucker (合并上下)
        vrcft["v2/LipPucker"] = vrcft.get("v2/LipPucker", 0.0)

        # EyeLid (合并左右)
        vrcft["v2/EyeLid"] = float(np.clip(
            (vrcft.get("v2/EyeLidLeft", 0.75) + vrcft.get("v2/EyeLidRight", 0.75)) / 2,
            0.0, 1.0
        ))

        # EyeSquint (合并左右)
        vrcft["v2/EyeSquint"] = float(np.clip(
            (vrcft.get("v2/EyeSquintLeft", 0) + vrcft.get("v2/EyeSquintRight", 0)) / 2,
            0.0, 1.0
        ))

        # ===== 8. 追踪激活状态 =====
        vrcft["ExpressionTrackingActive"] = 1.0
        vrcft["LipTrackingActive"] = 1.0

        return vrcft

    def _compute_face_params(self, points: np.ndarray, img_w: int, img_h: int) -> Dict:
        # 嘴部开合度（上下距离 / 面部高度）
        upper_lip = points[13]
        lower_lip = points[14]
        mouth_open_dist = np.linalg.norm(upper_lip - lower_lip)

        # 面部高度参考（额头到下巴）
        forehead = points[10]
        chin = points[152]
        face_height = np.linalg.norm(forehead - chin)

        mouth_open_ratio = mouth_open_dist / face_height if face_height > 0 else 0

        # 嘴部宽度
        mouth_left = points[61]
        mouth_right = points[291]
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        mouth_wide_ratio = mouth_width / face_height if face_height > 0 else 0

        # 左眼开合度
        left_eye_open = self._eye_aspect_ratio(
            points[self.LEFT_EYE_TOP], points[self.LEFT_EYE_BOTTOM],
            points[self.LEFT_EYE_LEFT], points[self.LEFT_EYE_RIGHT]
        )

        # 右眼开合度
        right_eye_open = self._eye_aspect_ratio(
            points[self.RIGHT_EYE_TOP], points[self.RIGHT_EYE_BOTTOM],
            points[self.RIGHT_EYE_LEFT], points[self.RIGHT_EYE_RIGHT]
        )

        # 眉毛高度（相对于眼睛）
        left_brow_height = self._brow_height_ratio(
            points[self.LEFT_BROW_INNER], points[self.LEFT_BROW_OUTER],
            points[self.LEFT_EYE_LEFT], points[self.LEFT_EYE_RIGHT],
            face_height
        )
        right_brow_height = self._brow_height_ratio(
            points[self.RIGHT_BROW_INNER], points[self.RIGHT_BROW_OUTER],
            points[self.RIGHT_EYE_LEFT], points[self.RIGHT_EYE_RIGHT],
            face_height
        )

        # 平滑处理
        t = time.perf_counter()
        mouth_open = self._apply_smooth("mouth_open", mouth_open_ratio, t)
        mouth_wide = self._apply_smooth("mouth_wide", mouth_wide_ratio, t)
        left_eye = self._apply_smooth("left_eye", left_eye_open, t)
        right_eye = self._apply_smooth("right_eye", right_eye_open, t)
        left_brow = self._apply_smooth("left_brow", left_brow_height, t)
        right_brow = self._apply_smooth("right_brow", right_brow_height, t)

        # 归一化到 0-1 范围（VRChat 参数范围）
        # 嘴部开合: 0=闭合, 1=最大张开
        mouth_open_normalized = np.clip(mouth_open / 0.15, 0.0, 1.0)
        # 嘴部宽度: 0=正常, 1=最大拉伸
        mouth_wide_normalized = np.clip((mouth_wide - 0.35) / 0.15, 0.0, 1.0)
        # 眼睛: 0=闭合, 1=完全睁开
        left_eye_normalized = np.clip(left_eye / 0.035, 0.0, 1.0)
        right_eye_normalized = np.clip(right_eye / 0.035, 0.0, 1.0)
        # 眉毛: 0=正常, 1=最高
        left_brow_normalized = np.clip((left_brow - 0.06) / 0.04, 0.0, 1.0)
        right_brow_normalized = np.clip((right_brow - 0.06) / 0.04, 0.0, 1.0)

        return {
            "mouth_open": float(mouth_open_normalized),
            "mouth_wide": float(mouth_wide_normalized),
            "left_eye_open": float(left_eye_normalized),
            "right_eye_open": float(right_eye_normalized),
            "left_brow_up": float(left_brow_normalized),
            "right_brow_up": float(right_brow_normalized),
            # 原始值（调试用）
            "mouth_open_raw": float(mouth_open_ratio),
            "mouth_wide_raw": float(mouth_wide_ratio),
        }

    def _apply_smooth(self, name: str, value: float, t: float) -> float:
        """应用平滑滤波"""
        if self.smoothing == "one_euro":
            if name not in self._filters:
                self._filters[name] = OneEuroFilter1D(
                    min_cutoff=self.one_euro_min_cutoff,
                    beta=self.one_euro_beta
                )
            return self._filters[name].filter(value, t)
        else:
            # 线性插值回退
            if name not in self._smooth_values:
                self._smooth_values[name] = value
                return value
            prev = self._smooth_values[name]
            smoothed = self.smoothing_factor * value + (1 - self.smoothing_factor) * prev
            self._smooth_values[name] = smoothed
            return smoothed

    def _eye_aspect_ratio(self, top, bottom, left, right) -> float:
        vertical = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right)
        return vertical / horizontal if horizontal > 0 else 0

    def _brow_height_ratio(self, brow_inner, brow_outer, eye_inner, eye_outer,
                           face_height: float) -> float:
        brow_center = (brow_inner + brow_outer) / 2
        eye_center = (eye_inner + eye_outer) / 2
        dist = np.linalg.norm(brow_center - eye_center)
        return dist / face_height if face_height > 0 else 0

    def calibrate(self, face_data: Dict):
        """校准嘴部参考值（闭嘴状态）"""
        if face_data is not None:
            self._mouth_open_ref = face_data["mouth_open_raw"]
            self._mouth_wide_ref = face_data["mouth_wide_raw"]
            self._calibrated = True
            print(f"Face calibrated: mouth_open_ref={self._mouth_open_ref:.4f}, "
                  f"mouth_wide_ref={self._mouth_wide_ref:.4f}")

    def reset_filters(self):
        """重置所有滤波器"""
        self._filters.clear()
        self._smooth_values.clear()
        self._bs_smoothed.clear()
        self._face_rotation_smoothed = [0.0, 0.0, 0.0]
        with self._result_lock:
            self._latest_result = None
        print("Face filters reset")

    def set_low_light_mode(self, enabled: bool = True):
        """切换低光模式
        
        低光模式下:
        - 增大 OneEuro beta (0.007 -> 0.03): 更强平滑
        - 降低 min_cutoff (1.0 -> 0.5): 低频更强平滑
        """
        if enabled:
            self.one_euro_beta = 0.03
            self.one_euro_min_cutoff = 0.5
            # 更新已有滤波器参数
            for f in self._filters.values():
                f.beta = 0.03
                f.min_cutoff = 0.5
            print("Face: low-light mode ON (beta=0.03, min_cutoff=0.5)")
        else:
            self.one_euro_beta = 0.007
            self.one_euro_min_cutoff = 1.0
            for f in self._filters.values():
                f.beta = 0.007
                f.min_cutoff = 1.0
            print("Face: low-light mode OFF (beta=0.007, min_cutoff=1.0)")

    def draw_face_landmarks(self, frame: np.ndarray, face_data: Dict) -> np.ndarray:
        """在画面上绘制面部关键点和嘴部状态"""
        if face_data is None:
            return frame

        points = face_data.get("landmarks")
        if points is None:
            return frame

        # 绘制嘴部轮廓
        for indices, color in [
            (self.LIP_OUTER_TOP, (0, 255, 255)),
            (self.LIP_OUTER_BOTTOM, (0, 200, 255)),
            (self.LIP_INNER_TOP, (0, 255, 200)),
            (self.LIP_INNER_BOTTOM, (0, 200, 200)),
        ]:
            for i in range(len(indices) - 1):
                pt1 = (int(points[indices[i]][0]), int(points[indices[i]][1]))
                pt2 = (int(points[indices[i + 1]][0]), int(points[indices[i + 1]][1]))
                cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

        # 绘制眼睛
        for eye_indices in [
            [self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM, self.LEFT_EYE_LEFT, self.LEFT_EYE_RIGHT],
            [self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM, self.RIGHT_EYE_LEFT, self.RIGHT_EYE_RIGHT],
        ]:
            for idx in eye_indices:
                pt = (int(points[idx][0]), int(points[idx][1]))
                cv2.circle(frame, pt, 2, (255, 200, 0), -1, cv2.LINE_AA)

        # 绘制嘴部状态文字
        mouth_open = face_data.get("mouth_open", 0)
        mouth_wide = face_data.get("mouth_wide", 0)
        left_eye = face_data.get("left_eye_open", 1)
        right_eye = face_data.get("right_eye_open", 1)

        y_offset = 120
        cv2.putText(frame, f"Mouth: {mouth_open:.2f}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Wide: {mouth_wide:.2f}", (10, y_offset + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"L-Eye: {left_eye:.2f}  R-Eye: {right_eye:.2f}",
                   (10, y_offset + 36),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)

        # VRCFT 参数显示
        vrcft = face_data.get("vrcft", {})
        if vrcft:
            cv2.putText(frame, f"VRCFT JawOpen: {vrcft.get('v2/JawOpen', 0):.2f}",
                       (10, y_offset + 54),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Smile L/R: {vrcft.get('v2/MouthSmileLeft', 0):.2f}/{vrcft.get('v2/MouthSmileRight', 0):.2f}",
                       (10, y_offset + 72),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"LipFunnel: {vrcft.get('v2/LipFunnel', 0):.2f} Pucker: {vrcft.get('v2/LipPucker', 0):.2f}",
                       (10, y_offset + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        # 面部变换矩阵数据显示 (参考 mediapipe-vt)
        face_rot = face_data.get("face_rotation", [0, 0, 0])
        face_pos = face_data.get("face_position", [0, 0, 0])
        cv2.putText(frame, f"Face Rot: P={face_rot[0]:.1f} Y={face_rot[1]:.1f} R={face_rot[2]:.1f}",
                   (10, y_offset + 108),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1, cv2.LINE_AA)

        return frame

    def release(self):
        if self.face_landmarker:
            try:
                # LIVE_STREAM 模式需要先关闭再释放
                self.face_landmarker.close()
            except Exception:
                pass
            self.face_landmarker = None
        self.is_loaded = False
        with self._result_lock:
            self._latest_result = None
        self._bs_smoothed.clear()
