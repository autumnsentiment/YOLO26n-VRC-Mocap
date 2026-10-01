import cv2
import numpy as np
import time
import os
import math
from runtime_paths import bin_dir, resource_root, runtime_root
from mediapipe_runtime import next_timestamp_ms
from typing import Dict, Optional, List, Tuple, Callable


class OneEuroFilter1D:
    """1D OneEuro 滤波器 - 低频平滑，高频低延迟"""

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

        dx_hat = (x - self.x_prev) / dt
        a_d = self._smoothing_factor(self.d_cutoff)
        dx_hat = a_d * dx_hat + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def _smoothing_factor(self, cutoff: float) -> float:
        r = 2 * math.pi * max(0.0, cutoff) / self.freq
        return r / (1.0 + r)


class FingerState:
    """手指状态跟踪器 - 减少抖动"""
    def __init__(self, history_len: int = 3):
        self.history_len = history_len
        self._history: Dict[str, List[bool]] = {}

    def update(self, finger: str, is_extended: bool) -> bool:
        if finger not in self._history:
            self._history[finger] = []
        self._history[finger].append(is_extended)
        if len(self._history[finger]) > self.history_len:
            self._history[finger].pop(0)
        # 多数投票
        return sum(self._history[finger]) > len(self._history[finger]) / 2

    def reset(self):
        self._history.clear()


class HandTracker:
    """手部跟踪器
    
    使用 MediaPipe Hand Landmarker (Tasks API) 检测手部关键点。
    
    兼容 MediaPipe 0.10+ 新版 Tasks API：
    - 使用 mp.tasks.vision.HandLandmarker 替代已废弃的 mp.solutions.hands
    - 使用 .task 模型文件
    - 使用 mp.Image 作为输入
    
    低光环境优化:
    - 降低 min_detection_confidence / min_tracking_confidence (0.3)
    - 增大 OneEuro beta 参数 (0.02~0.05) 加快运动响应
    """

    # 手部关键点索引
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    # 手指 MCP 关节
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17

    # 手指 PIP 关节
    THUMB_IP = 3
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18

    # 手指 DIP 关节
    THUMB_DIP = 2
    INDEX_DIP = 7
    MIDDLE_DIP = 11
    RING_DIP = 15
    PINKY_DIP = 19

    _MODEL_SEARCH_DIRS = [
        str(bin_dir()),
        str(runtime_root()),
        str(resource_root()),
    ]

    def __init__(self, max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 model_complexity: int = 1,
                 detection_callback: Optional[Callable] = None,
                 enable_ocr: bool = False,
                 smoothing: str = "one_euro",
                 smoothing_factor: float = 0.4,
                 one_euro_min_cutoff: float = 1.0,
                 one_euro_beta: float = 0.007,
                 finger_history_len: int = 3,
                 gesture_cooldown: float = 0.3,
                 palm_back_classify: bool = True,
                 back_curl_compensation: float = 1.25,
                 back_roll_compensation: float = 0.15,
                 roi_assist_enabled: bool = True,
                 roi_scale: float = 2.6,
                 roi_min_size: int = 96,
                 roi_min_confidence: float = 0.05,
                 roi_full_frame_interval: int = 12,
                 model_path: str = None):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_presence_confidence = min_presence_confidence
        self.model_complexity = model_complexity

        self.hands = None
        self.is_loaded = False
        self._detection_cb = detection_callback
        self.model_path = model_path

        self.smoothing = smoothing
        self.smoothing_factor = smoothing_factor
        self.one_euro_min_cutoff = one_euro_min_cutoff
        self.one_euro_beta = one_euro_beta

        self._filters: Dict[str, OneEuroFilter1D] = {}
        self._smooth_values: Dict[str, float] = {}
        self._finger_state = FingerState(history_len=finger_history_len)

        self._last_gesture_time = 0.0
        self.gesture_cooldown = gesture_cooldown

        # 手掌/手背分类参数
        self.palm_back_classify = palm_back_classify
        self.back_curl_compensation = back_curl_compensation
        self.back_roll_compensation = back_roll_compensation
        self.roi_assist_enabled = bool(roi_assist_enabled)
        self.roi_scale = max(1.4, float(roi_scale))
        self.roi_min_size = max(48, int(roi_min_size))
        self.roi_min_confidence = max(0.0, min(1.0, float(roi_min_confidence)))
        self.roi_full_frame_interval = max(1, int(roi_full_frame_interval))
        self._roi_frame_index = 0
        self.last_detection_mode = "none"
        self.last_roi_hint_count = 0
        self.last_roi_hints = {}
        self.roi_detection_count = 0
        self.full_frame_detection_count = 0
        self._frame_timestamp_ms = 0
        self._roi_hands = {}
        self._roi_timestamps = {}
        self._resolved_model_path = None

    def _find_model_file(self) -> Optional[str]:
        if self.model_path:
            path = os.path.abspath(os.path.expanduser(self.model_path))
            if os.path.exists(path):
                return path
            print(f"Configured hand model not found: {path}")
        model_name = "hand_landmarker.task"
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
                print("hand_landmarker.task not found, hand tracking disabled")
                print("Download from: https://storage.googleapis.com/mediapipe-models/"
                      "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
                return False
            print(f"Hand Landmarker model: {model_path}")

            self._resolved_model_path = model_path
            self.hands = self._create_landmarker(mp)
            self._frame_timestamp_ms = 0
            self.is_loaded = True
            print(f"Hand Landmarker loaded (VIDEO mode, det_conf={self.min_detection_confidence}, "
                  f"trk_conf={self.min_tracking_confidence})")
            return True
        except ImportError:
            print("mediapipe not installed, hand tracking disabled")
            return False
        except Exception as e:
            print(f"Error loading Hand Landmarker: {e}")
            return False

    def _create_landmarker(self, mp):
        tasks = mp.tasks.vision
        options = tasks.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._resolved_model_path),
            running_mode=tasks.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return tasks.HandLandmarker.create_from_options(options)

    def detect(
        self,
        frame: np.ndarray,
        roi_hints: Optional[Dict] = None,
    ) -> List[Dict]:
        if not self.is_loaded or self.hands is None:
            return []

        self._roi_frame_index += 1
        regions = (
            self._build_wrist_rois(frame, roi_hints)
            if self.roi_assist_enabled
            else []
        )
        self.last_roi_hint_count = len(regions)
        full_results = self._detect_region(frame)
        full_labels = {
            str(hand.get("label", "")).lower()
            for hand in full_results
        }
        roi_results = []
        for side, rect in regions:
            if side not in full_labels:
                hint = self.last_roi_hints[side]["wrist"]
                roi_results.extend(self._detect_region(frame, rect, side, hint))

        detections = self._merge_hand_detections(roi_results, full_results)

        if roi_results and full_results:
            self.last_detection_mode = "body_wrist_roi+full_frame"
        elif roi_results:
            self.last_detection_mode = "body_wrist_roi"
        elif full_results:
            self.last_detection_mode = "full_frame"
        elif regions:
            self.last_detection_mode = "roi_miss"
        else:
            self.last_detection_mode = "full_frame_miss"
        self.roi_detection_count += len(roi_results)
        self.full_frame_detection_count += len(full_results)
        return detections

    def _detect_region(
        self,
        frame: np.ndarray,
        rect: Optional[Tuple[int, int, int, int]] = None,
        expected_label: Optional[str] = None,
        wrist_hint: Optional[List[float]] = None,
    ) -> List[Dict]:
        import mediapipe as mp

        frame_h, frame_w = frame.shape[:2]
        if rect is None:
            x0, y0, x1, y1 = 0, 0, frame_w, frame_h
            detection_source = "full_frame"
        else:
            x0, y0, x1, y1 = rect
            detection_source = "body_wrist_roi"
        region = frame[y0:y1, x0:x1]
        if region.size == 0:
            return []

        region_h, region_w = region.shape[:2]
        rgb_region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_region)
        if rect is None:
            self._frame_timestamp_ms = next_timestamp_ms(self._frame_timestamp_ms)
            landmarker = self.hands
            timestamp = self._frame_timestamp_ms
        else:
            if expected_label not in self._roi_hands:
                self._roi_hands[expected_label] = self._create_landmarker(mp)
            landmarker = self._roi_hands[expected_label]
            timestamp = next_timestamp_ms(self._roi_timestamps.get(expected_label, 0))
            self._roi_timestamps[expected_label] = timestamp
        result = landmarker.detect_for_video(mp_image, timestamp)
        if not result.hand_landmarks:
            return []

        hand_data_list = []
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i] if result.handedness else []
            hand_label = handedness[0].category_name.lower() if handedness else "unknown"
            hand_score = handedness[0].score if handedness else 0.0
            geometry_norm = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float64,
            )
            landmarks = np.array([
                [x0 + lm.x * region_w, y0 + lm.y * region_h]
                for lm in hand_landmarks
            ], dtype=np.float64)
            if rect is not None and not self._accept_roi_hand(
                hand_label, expected_label, landmarks[self.WRIST], wrist_hint, rect
            ):
                continue
            z_scale = 0.5 * (
                region_w / max(1.0, float(frame_w))
                + region_h / max(1.0, float(frame_h))
            )
            landmarks_norm = np.array([
                [
                    (x0 + lm.x * region_w) / frame_w,
                    (y0 + lm.y * region_h) / frame_h,
                    lm.z * z_scale,
                ]
                for lm in hand_landmarks
            ], dtype=np.float64)

            # 计算手指状态
            fingers = self._compute_finger_states(geometry_norm, hand_label)

            # 检测手势
            gesture = self._detect_gesture(fingers)

            # 计算手部位置
            palm_center = landmarks[self.WRIST]
            palm_x_norm = palm_center[0] / frame_w
            palm_y_norm = palm_center[1] / frame_h

            # 计算手指关键点位置
            t = time.perf_counter()
            index_tip_x = self._apply_smooth(
                f"index_tip_x_{hand_label}", landmarks_norm[self.INDEX_TIP][0], t
            )
            index_tip_y = self._apply_smooth(
                f"index_tip_y_{hand_label}", landmarks_norm[self.INDEX_TIP][1], t
            )

            # 计算手指弯曲度 (curls)
            curls = self._compute_curls(geometry_norm, hand_label, t)

            # 检测手掌/手背朝向
            palm_side, palm_conf = self._compute_palm_side(geometry_norm, hand_label)

            # 计算手部旋转 (基于手掌法向量)
            rotation = self._compute_hand_rotation(geometry_norm, hand_label)

            # 计算手指张开度
            spread = self._compute_spread(geometry_norm)

            hand_data = {
                "label": hand_label,
                "confidence": float(hand_score),
                "landmarks": landmarks,
                "landmarks_norm": landmarks_norm,
                "palm_x": self._apply_smooth(f"palm_x_{hand_label}", palm_x_norm, t),
                "palm_y": self._apply_smooth(f"palm_y_{hand_label}", palm_y_norm, t),
                "fingers": fingers,
                "curls": curls,
                "gesture": gesture,
                "palm_side": palm_side,
                "palm_confidence": palm_conf,
                "rotation": rotation,
                "spread": spread,
                "index_tip_x": index_tip_x,
                "index_tip_y": index_tip_y,
                "detection_source": detection_source,
                "roi_rect": list(rect) if rect is not None else None,
            }
            hand_data_list.append(hand_data)

        return hand_data_list

    @staticmethod
    def _accept_roi_hand(label: str, expected_label: str, wrist,
                         hint, rect: Tuple[int, int, int, int]) -> bool:
        if label != expected_label or hint is None:
            return False
        distance = math.hypot(float(wrist[0]) - float(hint[0]),
                              float(wrist[1]) - float(hint[1]))
        limit = max(24.0, 0.4 * min(rect[2] - rect[0], rect[3] - rect[1]))
        return math.isfinite(distance) and distance <= limit

    def _build_wrist_rois(
        self,
        frame: np.ndarray,
        roi_hints: Optional[Dict],
    ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        self.last_roi_hints = {}
        if not isinstance(roi_hints, dict):
            return []
        keypoints = roi_hints.get("keypoints_2d")
        if not isinstance(keypoints, dict):
            return []

        frame_h, frame_w = frame.shape[:2]
        max_size = max(self.roi_min_size, int(min(frame_w, frame_h) * 0.5))
        regions = []
        diagnostics = {}
        for side in ("left", "right"):
            wrist = keypoints.get(f"{side}_wrist")
            elbow = keypoints.get(f"{side}_elbow")
            if not isinstance(wrist, (list, tuple, np.ndarray)) or len(wrist) < 2:
                diagnostics[side] = {"accepted": False, "reason": "missing_wrist"}
                continue
            try:
                wrist_x, wrist_y = float(wrist[0]), float(wrist[1])
                confidence = float(wrist[2]) if len(wrist) >= 3 else 1.0
            except (TypeError, ValueError):
                diagnostics[side] = {"accepted": False, "reason": "invalid_wrist"}
                continue
            if confidence < self.roi_min_confidence:
                diagnostics[side] = {
                    "accepted": False,
                    "reason": "low_confidence",
                    "confidence": confidence,
                }
                continue
            if abs(wrist_x) <= 1.5 and abs(wrist_y) <= 1.5:
                wrist_x *= frame_w
                wrist_y *= frame_h

            forearm_length = min(frame_w, frame_h) * 0.12
            direction = np.array([0.0, 0.0], dtype=float)
            if isinstance(elbow, (list, tuple, np.ndarray)) and len(elbow) >= 2:
                try:
                    elbow_x, elbow_y = float(elbow[0]), float(elbow[1])
                    if abs(elbow_x) <= 1.5 and abs(elbow_y) <= 1.5:
                        elbow_x *= frame_w
                        elbow_y *= frame_h
                    direction = np.array(
                        [wrist_x - elbow_x, wrist_y - elbow_y],
                        dtype=float,
                    )
                    measured = float(np.linalg.norm(direction))
                    if measured > 4.0:
                        forearm_length = measured
                        direction /= measured
                except (TypeError, ValueError):
                    pass

            size = int(round(np.clip(
                forearm_length * self.roi_scale,
                self.roi_min_size,
                max_size,
            )))
            center = np.array([wrist_x, wrist_y], dtype=float)
            center += direction * min(size * 0.12, forearm_length * 0.3)
            half = size * 0.5
            x0 = max(0, int(round(center[0] - half)))
            y0 = max(0, int(round(center[1] - half)))
            x1 = min(frame_w, int(round(center[0] + half)))
            y1 = min(frame_h, int(round(center[1] + half)))
            if x1 - x0 >= 48 and y1 - y0 >= 48:
                regions.append((side, (x0, y0, x1, y1)))
                diagnostics[side] = {
                    "accepted": True,
                    "confidence": confidence,
                    "wrist": [round(wrist_x, 1), round(wrist_y, 1)],
                    "rect": [x0, y0, x1, y1],
                }
            else:
                diagnostics[side] = {
                    "accepted": False,
                    "reason": "outside_frame",
                    "confidence": confidence,
                    "wrist": [round(wrist_x, 1), round(wrist_y, 1)],
                }
        self.last_roi_hints = diagnostics
        return regions

    @staticmethod
    def _merge_hand_detections(*groups: List[Dict]) -> List[Dict]:
        by_label = {}
        unknown = []
        for group in groups:
            for hand in group:
                label = str(hand.get("label", "unknown")).lower()
                if label not in ("left", "right"):
                    unknown.append(hand)
                    continue
                current = by_label.get(label)
                if current is None or float(hand.get("confidence", 0.0)) > float(
                    current.get("confidence", 0.0)
                ):
                    by_label[label] = hand
        return [*by_label.values(), *unknown]

    def update_roi_config(self, config: Dict):
        self.roi_assist_enabled = bool(
            config.get("roi_assist_enabled", self.roi_assist_enabled)
        )
        self.roi_scale = max(1.4, float(config.get("roi_scale", self.roi_scale)))
        self.roi_min_size = max(48, int(config.get("roi_min_size", self.roi_min_size)))
        self.roi_min_confidence = max(0.0, min(
            1.0,
            float(config.get("roi_min_confidence", self.roi_min_confidence)),
        ))
        self.roi_full_frame_interval = max(
            1,
            int(config.get("roi_full_frame_interval", self.roi_full_frame_interval)),
        )

    def _compute_finger_states(
        self, landmarks: np.ndarray, hand_label: str = "unknown"
    ) -> Dict[str, bool]:
        """计算每根手指的伸展状态"""
        fingers = {}
        state_prefix = str(hand_label or "unknown").strip().lower()

        # 拇指: 比较 tip.x 与 ip.x（左手 tip < ip，右手 tip > ip）
        fingers["thumb"] = self._finger_state.update(
            f"{state_prefix}:thumb",
            abs(landmarks[self.THUMB_TIP][0] - landmarks[self.THUMB_IP][0]) > 0.04,
        )

        # 其他四指: tip.y < pip.y 表示伸展
        pairs = [
            ("index", self.INDEX_TIP, self.INDEX_PIP),
            ("middle", self.MIDDLE_TIP, self.MIDDLE_PIP),
            ("ring", self.RING_TIP, self.RING_PIP),
            ("pinky", self.PINKY_TIP, self.PINKY_PIP),
        ]
        for name, tip, pip in pairs:
            is_ext = landmarks[tip][1] < landmarks[pip][1]
            fingers[name] = self._finger_state.update(
                f"{state_prefix}:{name}", is_ext
            )

        return fingers

    @staticmethod
    def _joint_bend(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Return normalized bend at b: straight=0, roughly 90 degrees=1."""
        incoming = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        outgoing = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if denominator <= 1e-8:
            return 0.0
        cosine = float(np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0))
        joint_angle = float(np.degrees(np.arccos(cosine)))
        return float(np.clip((180.0 - joint_angle) / 90.0, 0.0, 1.0))

    def _compute_curls(
        self,
        landmarks: np.ndarray,
        hand_label: str = "unknown",
        timestamp: float = None,
    ) -> Dict[str, float]:
        """Calculate stable 3D joint-angle curl values for all five fingers."""
        points = np.asarray(landmarks, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 21 or points.shape[1] < 3:
            return {name: 0.0 for name in ("thumb", "index", "middle", "ring", "pinky")}

        now = time.perf_counter() if timestamp is None else float(timestamp)
        raw_curls = {
            "thumb": (
                0.45 * self._joint_bend(points[1], points[2], points[3])
                + 0.55 * self._joint_bend(points[2], points[3], points[4])
            ),
        }
        finger_chains = {
            "index": (5, 6, 7, 8),
            "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20),
        }
        for name, (mcp, pip, dip, tip) in finger_chains.items():
            mcp_bend = self._joint_bend(points[0], points[mcp], points[pip])
            pip_bend = self._joint_bend(points[mcp], points[pip], points[dip])
            dip_bend = self._joint_bend(points[pip], points[dip], points[tip])
            raw_curls[name] = 0.20 * mcp_bend + 0.55 * pip_bend + 0.25 * dip_bend

        label = str(hand_label or "unknown").strip().lower()
        curls = {}
        for name, value in raw_curls.items():
            compensated = float(np.clip(value * self.back_curl_compensation, 0.0, 1.0))
            curls[name] = float(self._apply_smooth(
                f"curl_{label}_{name}", compensated, now
            ))
        return curls

    def _compute_palm_side(self, landmarks: np.ndarray, hand_label: str) -> Tuple[str, float]:
        """检测手掌朝向 (palm/back) 和置信度

        通过食指 MCP 和小指 MCP 相对于手腕的位置判断手掌朝向:
        - 左手: index_mcp.x > pinky_mcp.x → palm, 否则 → back
        - 右手: index_mcp.x < pinky_mcp.x → palm, 否则 → back
        """
        if not self.palm_back_classify:
            return "palm", 0.8

        index_mcp_x = landmarks[self.INDEX_MCP][0]
        pinky_mcp_x = landmarks[self.PINKY_MCP][0]

        # 手掌朝向判断
        if hand_label == "left":
            is_palm = index_mcp_x > pinky_mcp_x
        else:
            is_palm = index_mcp_x < pinky_mcp_x

        # 置信度: 基于 index-pinky 水平距离
        spread = abs(index_mcp_x - pinky_mcp_x)
        palm_conf = min(spread / 0.15, 1.0)  # 归一化到 0-1
        palm_conf = max(0.3, palm_conf)  # 最低 0.3

        return ("palm" if is_palm else "back", float(palm_conf))

    def _compute_hand_rotation(self, landmarks: np.ndarray, hand_label: str) -> List[float]:
        """基于手掌关键点估算手部旋转 (欧拉角 zxy, 度)

        使用手腕→中指MCP方向估算俯仰(pitch)，
        使用手腕→食指MCP与小指MCP的叉积估算偏航(yaw)和翻滚(roll)。
        """
        wrist = landmarks[self.WRIST][:2]
        index_mcp = landmarks[self.INDEX_MCP][:2]
        middle_mcp = landmarks[self.MIDDLE_MCP][:2]
        pinky_mcp = landmarks[self.PINKY_MCP][:2]

        # 手腕到中指MCP方向 → 俯仰 (pitch, 绕X轴)
        forward = middle_mcp - wrist
        pitch = -float(np.arctan2(forward[1], max(abs(forward[0]), 1e-6))) * 180.0 / np.pi * 0.3

        # 食指MCP到小指MCP方向 → 翻滚 (roll, 绕Z轴)
        side = pinky_mcp - index_mcp
        roll = float(np.arctan2(side[1], max(abs(side[0]), 1e-6))) * 180.0 / np.pi * 0.5

        # 手背时补偿翻滚
        palm_side = "palm"
        if hand_label == "left":
            if index_mcp[0] <= pinky_mcp[0]:
                palm_side = "back"
        else:
            if index_mcp[0] >= pinky_mcp[0]:
                palm_side = "back"

        if palm_side == "back":
            roll += self.back_roll_compensation * 180.0 / np.pi

        # 偏航 (yaw, 绕Y轴) — 基于手掌水平位置偏移估算
        yaw = 0.0

        return [yaw, pitch, roll]

    def _compute_spread(self, landmarks: np.ndarray) -> float:
        """计算手指张开度 (0.0=并拢, 1.0=完全张开)

        基于食指MCP与小指MCP之间的距离。
        """
        index_mcp = landmarks[self.INDEX_MCP][:2]
        pinky_mcp = landmarks[self.PINKY_MCP][:2]
        spread_dist = np.linalg.norm(index_mcp - pinky_mcp)
        # 归一化: 典型张开距离约 0.12, 并拢约 0.03
        spread = np.clip((spread_dist - 0.03) / 0.09, 0.0, 1.0)
        return float(spread)

    def _detect_gesture(self, fingers: Dict[str, bool]) -> str:
        now = time.time()
        if now - self._last_gesture_time < self.gesture_cooldown:
            return ""

        gesture = ""
        t, i, m, r, p = fingers["thumb"], fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]

        if t and i and m and r and p:
            gesture = "open"
        elif not t and not i and not m and not r and not p:
            gesture = "fist"
        elif i and not m and not r and not p:
            gesture = "point"
        elif i and m and not r and not p:
            gesture = "peace"
        elif t and not i and not m and not r and not p:
            gesture = "thumb_up"
        elif t and i and not m and not r and p:
            gesture = "call"
        elif not i and not m and not r and p:
            gesture = "pinky"

        if gesture:
            self._last_gesture_time = now

        return gesture

    def _apply_smooth(self, name: str, value: float, t: float) -> float:
        if self.smoothing == "one_euro":
            if name not in self._filters:
                self._filters[name] = OneEuroFilter1D(
                    min_cutoff=self.one_euro_min_cutoff,
                    beta=self.one_euro_beta
                )
            return self._filters[name].filter(value, t)

        if name not in self._smooth_values:
            self._smooth_values[name] = value
            return value
        prev = self._smooth_values[name]
        smoothed = self.smoothing_factor * value + (1 - self.smoothing_factor) * prev
        self._smooth_values[name] = smoothed
        return smoothed

    def reset_filters(self):
        """Clear temporal state after tracking loss or a manual reset."""
        self._filters.clear()
        self._smooth_values.clear()
        self._finger_state.reset()
        self._last_gesture_time = 0.0

    def set_low_light_mode(self, enabled: bool = True):
        if enabled:
            self.one_euro_beta = 0.03
            self.one_euro_min_cutoff = 0.5
            for f in self._filters.values():
                f.beta = 0.03
                f.min_cutoff = 0.5
            print("Hand: low-light mode ON (beta=0.03, min_cutoff=0.5)")
        else:
            self.one_euro_beta = 0.007
            self.one_euro_min_cutoff = 1.0
            for f in self._filters.values():
                f.beta = 0.007
                f.min_cutoff = 1.0
            print("Hand: low-light mode OFF (beta=0.007, min_cutoff=1.0)")

    def draw_hand_landmarks(self, frame: np.ndarray, hand_data_list: List[Dict]) -> np.ndarray:
        """在画面上绘制手部关键点和手势"""
        if not hand_data_list:
            return frame

        # 骨骼连接
        connections = [
            (0,1),(1,2),(2,3),(3,4),       # thumb
            (0,5),(5,6),(6,7),(7,8),       # index
            (0,9),(9,10),(10,11),(11,12),  # middle
            (0,13),(13,14),(14,15),(15,16),# ring
            (0,17),(17,18),(18,19),(19,20),# pinky
            (5,9),(9,13),(13,17),          # palm
        ]

        for hand_data in hand_data_list:
            landmarks = hand_data["landmarks"]
            label = hand_data["label"]
            gesture = hand_data.get("gesture", "")
            fingers = hand_data.get("fingers", {})

            # 选择颜色
            color = (0, 255, 255) if label == "left" else (0, 255, 0)

            # 绘制连接线
            for start, end in connections:
                if start < len(landmarks) and end < len(landmarks):
                    pt1 = (int(landmarks[start][0]), int(landmarks[start][1]))
                    pt2 = (int(landmarks[end][0]), int(landmarks[end][1]))
                    cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

            # 绘制关键点
            for j, lm in enumerate(landmarks):
                pt = (int(lm[0]), int(lm[1]))
                if j in [4, 8, 12, 16, 20]:  # 指尖
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
                elif j == 0:  # 手腕
                    cv2.circle(frame, pt, 5, (255, 0, 0), -1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt, 3, color, -1, cv2.LINE_AA)

            # 绘制标签和手势
            wrist = landmarks[0]
            info_text = f"{label} ({hand_data['confidence']:.2f})"
            if gesture:
                info_text += f" [{gesture}]"
            cv2.putText(frame, info_text, (int(wrist[0]) - 20, int(wrist[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            # 绘制手指状态
            y_base = 140
            for fname, fval in fingers.items():
                fcolor = (0, 255, 0) if fval else (0, 0, 200)
                cv2.putText(frame, f"{fname}: {'ON' if fval else 'off'}",
                           (10, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.4, fcolor, 1, cv2.LINE_AA)
                y_base += 14

        return frame

    def release(self):
        if self.hands:
            self.hands.close()
            self.hands = None
        for landmarker in self._roi_hands.values():
            landmarker.close()
        self._roi_hands.clear()
        self._roi_timestamps.clear()
        self._resolved_model_path = None
        self.is_loaded = False
        self._frame_timestamp_ms = 0
