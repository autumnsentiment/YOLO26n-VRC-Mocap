import os
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from runtime_paths import bin_dir, resource_root, runtime_root
from mediapipe_runtime import next_timestamp_ms
from pose_estimator import PoseEstimator


class MediaPipePoseDetector:
    """Optional MediaPipe Pose Landmarker body source.

    The detector stays disabled unless configured. It emits the same camera
    coordinate pose dictionary used by PoseEstimator: X left, Y down, Z front,
    hip midpoint as the local origin.
    """

    _MODEL_NAMES = {
        "lite": "pose_landmarker_lite.task",
        "full": "pose_landmarker_full.task",
        "heavy": "pose_landmarker_heavy.task",
    }
    _MODEL_URLS = {
        variant: (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
        )
        for variant in _MODEL_NAMES
    }

    _MODEL_SEARCH_DIRS = [
        bin_dir(),
        runtime_root(),
        runtime_root() / "models",
        resource_root(),
        resource_root() / "models",
    ]

    # BlazePose / MediaPipe Pose 33-landmark indices.
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
    IMAGE_KEYPOINTS = {
        "nose": NOSE,
        "left_shoulder": LEFT_SHOULDER,
        "right_shoulder": RIGHT_SHOULDER,
        "left_elbow": LEFT_ELBOW,
        "right_elbow": RIGHT_ELBOW,
        "left_wrist": LEFT_WRIST,
        "right_wrist": RIGHT_WRIST,
        "left_hip": LEFT_HIP,
        "right_hip": RIGHT_HIP,
        "left_knee": LEFT_KNEE,
        "right_knee": RIGHT_KNEE,
        "left_ankle": LEFT_ANKLE,
        "right_ankle": RIGHT_ANKLE,
    }

    def __init__(
        self,
        model_variant: str = "heavy",
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_variant = (model_variant or "heavy").lower()
        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.landmarker = None
        self.is_loaded = False
        self.fps_history = deque(maxlen=60)
        self._frame_timestamp_ms = 0
        # Last shoulder->hip offset seen with confident hips; held briefly so
        # an occluded/out-of-frame hip does not snap the body anchor.
        self.hip_hold_seconds = 0.3
        self._hip_hold_offset: Optional[np.ndarray] = None
        self._hip_hold_at: Optional[float] = None

    def _find_model_file(self) -> Optional[str]:
        if self.model_path:
            configured = Path(os.path.expanduser(self.model_path))
            if not configured.is_absolute():
                configured_candidates = [d / configured for d in self._MODEL_SEARCH_DIRS]
            else:
                configured_candidates = [configured]
            for path in configured_candidates:
                if path.exists():
                    return str(path.resolve())
            print(f"MediaPipe Pose model not found: {self.model_path}")

        model_name = self._MODEL_NAMES.get(self.model_variant, self._MODEL_NAMES["heavy"])
        for directory in self._MODEL_SEARCH_DIRS:
            path = directory / model_name
            if path.exists():
                return str(path.resolve())
        return self._download_model_file(model_name)

    def _download_model_file(self, model_name: str) -> Optional[str]:
        variant = self.model_variant if self.model_variant in self._MODEL_URLS else "heavy"
        url = self._MODEL_URLS[variant]
        target_dir = runtime_root() / "models"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / model_name
        tmp = target.with_suffix(target.suffix + ".download")

        try:
            print(f"Downloading MediaPipe Pose model ({variant}) to {target}...")
            with urllib.request.urlopen(url, timeout=60) as response:
                with open(tmp, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            tmp.replace(target)
            print(f"MediaPipe Pose model downloaded: {target}")
            return str(target.resolve())
        except Exception as e:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            print(f"MediaPipe Pose model download failed: {e}")
            print(f"Manual download URL: {url}")
            return None

    def load(self) -> bool:
        try:
            import mediapipe as mp
        except ImportError:
            print("mediapipe not installed, MediaPipe Pose disabled")
            return False

        model_path = self._find_model_file()
        if not model_path:
            print(
                "MediaPipe Pose model not found, body fallback disabled. "
                "Expected pose_landmarker_heavy.task/full/lite in project or models directory."
            )
            return False

        try:
            tasks = mp.tasks.vision
            options = tasks.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=tasks.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=False,
            )
            self.landmarker = tasks.PoseLandmarker.create_from_options(options)
            self._frame_timestamp_ms = 0
            self.is_loaded = True
            print(f"MediaPipe Pose Landmarker loaded (VIDEO/{self.model_variant}): {model_path}")
            return True
        except Exception as e:
            print(f"Error loading MediaPipe Pose Landmarker: {e}")
            return False

    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        if not self.is_loaded or self.landmarker is None:
            return None

        try:
            import mediapipe as mp

            start = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._frame_timestamp_ms = next_timestamp_ms(self._frame_timestamp_ms)
            result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

            elapsed = time.perf_counter() - start
            if elapsed > 0:
                self.fps_history.append(1.0 / elapsed)

            if not result.pose_world_landmarks:
                return None

            world = result.pose_world_landmarks[0]
            pose_3d = self._world_landmarks_to_pose(world)
            if not pose_3d:
                return None

            confidence = self._average_visibility(world)
            image_landmarks = (
                result.pose_landmarks[0]
                if getattr(result, "pose_landmarks", None)
                else None
            )
            return {
                "pose_3d": pose_3d,
                "keypoints_2d": self._image_landmarks_to_keypoints(
                    image_landmarks,
                    frame.shape[1],
                    frame.shape[0],
                ),
                "confidence": confidence,
                "source": "mediapipe_pose",
                "fps": sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0,
            }
        except Exception as e:
            print(f"MediaPipe Pose detection failed: {e}")
            return None

    @classmethod
    def _image_landmarks_to_keypoints(
        cls,
        landmarks,
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, List[float]]:
        if not landmarks:
            return {}

        keypoints = {}
        for name, index in cls.IMAGE_KEYPOINTS.items():
            if index >= len(landmarks):
                continue
            landmark = landmarks[index]
            x = float(getattr(landmark, "x", 0.0)) * frame_width
            y = float(getattr(landmark, "y", 0.0)) * frame_height
            visibility = float(getattr(landmark, "visibility", 1.0))
            presence = float(getattr(landmark, "presence", 1.0))
            confidence = max(0.0, min(1.0, min(visibility, presence)))
            if np.isfinite(x) and np.isfinite(y):
                keypoints[name] = [x, y, confidence]
        return keypoints

    def _lm(
        self,
        landmarks,
        idx: int,
        min_visibility: float = 0.15,
        min_presence: float = 0.15,
    ) -> Optional[np.ndarray]:
        if idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        visibility = getattr(lm, "visibility", 1.0)
        presence = getattr(lm, "presence", 1.0)
        if visibility < min_visibility or presence < min_presence:
            return None
        coords = np.array([-float(lm.x), float(lm.y), -float(lm.z)], dtype=float)
        if not np.all(np.isfinite(coords)):
            return None

        # MediaPipe world landmarks are meters with hip midpoint origin.
        # Map to the camera pose convention consumed by CoordinateTransformer.
        return coords

    @staticmethod
    def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b) * 0.5

    @staticmethod
    def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return a * (1.0 - t) + b * t

    @staticmethod
    def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        length = np.linalg.norm(v)
        if length < 1e-6:
            return fallback.astype(float)
        return v / length

    @staticmethod
    def _finite_midpoint(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if a is not None and b is not None:
            return (a + b) * 0.5
        if a is not None:
            return a.copy()
        if b is not None:
            return b.copy()
        return None

    def _resolve_hip_anchor(
        self,
        landmarks,
        shoulders: np.ndarray,
        relaxed_hips: Optional[np.ndarray],
        now: Optional[float] = None,
    ) -> np.ndarray:
        now = time.perf_counter() if now is None else now
        strict_l = self._lm(landmarks, self.LEFT_HIP)
        strict_r = self._lm(landmarks, self.RIGHT_HIP)
        if strict_l is not None and strict_r is not None:
            hips = (strict_l + strict_r) * 0.5
            self._hip_hold_offset = hips - shoulders
            self._hip_hold_at = now
            return hips
        if (
            self._hip_hold_offset is not None
            and self._hip_hold_at is not None
            and now - self._hip_hold_at <= self.hip_hold_seconds
        ):
            return shoulders + self._hip_hold_offset
        if relaxed_hips is not None:
            return relaxed_hips
        return shoulders + np.array([0.0, 0.55, 0.0], dtype=float)

    def _world_landmarks_to_pose(self, landmarks) -> Dict[str, List[float]]:
        ls = self._lm(landmarks, self.LEFT_SHOULDER)
        rs = self._lm(landmarks, self.RIGHT_SHOULDER)
        if ls is None or rs is None:
            return {}

        # MediaPipe still provides plausible world coordinates for hips/knees
        # when they are outside the camera frame, but marks them with low
        # visibility. Use relaxed thresholds for torso anchoring so the body
        # backend does not drop to 0 nodes during upper-body camera framing.
        lh = self._lm(landmarks, self.LEFT_HIP, 0.0, 0.0)
        rh = self._lm(landmarks, self.RIGHT_HIP, 0.0, 0.0)
        shoulders = self._mid(ls, rs)
        hips = self._resolve_hip_anchor(
            landmarks, shoulders, self._finite_midpoint(lh, rh)
        )

        torso = shoulders - hips
        if np.linalg.norm(torso) < 1e-6:
            torso = np.array([0.0, -0.5, 0.0], dtype=float)
        torso_dir = self._unit(torso, np.array([0.0, -1.0, 0.0], dtype=float))
        down_dir = -torso_dir
        left_dir = self._unit(ls - rs, np.array([-1.0, 0.0, 0.0], dtype=float))

        hip_width = max(np.linalg.norm((lh if lh is not None else hips) - (rh if rh is not None else hips)), 0.16)
        shoulder_width = max(np.linalg.norm(ls - rs), 0.36)
        upper_leg_len = 0.42
        lower_leg_len = 0.42
        foot_len = 0.10
        upper_arm_len = 0.28
        lower_arm_len = 0.24
        hand_len = 0.08

        left_hip = lh if lh is not None else hips + left_dir * (hip_width * 0.5)
        right_hip = rh if rh is not None else hips - left_dir * (hip_width * 0.5)

        pose = {
            "Hips": hips,
            "Spine": self._lerp(hips, shoulders, 0.35),
            "Chest": self._lerp(hips, shoulders, 0.65),
            "UpperChest": shoulders,
            "Neck": shoulders + torso * 0.10,
        }

        nose = self._lm(landmarks, self.NOSE)
        pose["Head"] = nose if nose is not None else shoulders + torso * 0.25

        def point_or_fallback(idx: int, fallback: np.ndarray) -> np.ndarray:
            p = self._lm(landmarks, idx)
            return p if p is not None else fallback

        left_shoulder = ls
        right_shoulder = rs
        arm_fallback = down_dir * 0.65 + np.array([0.0, 0.0, 0.15], dtype=float)
        arm_fallback = self._unit(arm_fallback, down_dir)

        left_elbow_fallback = left_shoulder + self._unit(left_dir * 0.45 + arm_fallback, down_dir) * upper_arm_len
        right_elbow_fallback = right_shoulder + self._unit(-left_dir * 0.45 + arm_fallback, down_dir) * upper_arm_len
        left_elbow = point_or_fallback(self.LEFT_ELBOW, left_elbow_fallback)
        right_elbow = point_or_fallback(self.RIGHT_ELBOW, right_elbow_fallback)
        left_wrist = point_or_fallback(self.LEFT_WRIST, left_elbow + self._unit(left_elbow - left_shoulder, down_dir) * lower_arm_len)
        right_wrist = point_or_fallback(self.RIGHT_WRIST, right_elbow + self._unit(right_elbow - right_shoulder, down_dir) * lower_arm_len)

        pose["LeftUpperArm"] = left_elbow
        pose["RightUpperArm"] = right_elbow
        pose["LeftLowerArm"] = left_wrist
        pose["RightLowerArm"] = right_wrist
        pose["LeftHand"] = left_wrist + self._unit(left_wrist - left_elbow, down_dir) * hand_len
        pose["RightHand"] = right_wrist + self._unit(right_wrist - right_elbow, down_dir) * hand_len

        left_knee = point_or_fallback(self.LEFT_KNEE, left_hip + down_dir * upper_leg_len)
        right_knee = point_or_fallback(self.RIGHT_KNEE, right_hip + down_dir * upper_leg_len)
        left_ankle = point_or_fallback(self.LEFT_ANKLE, left_knee + self._unit(left_knee - left_hip, down_dir) * lower_leg_len)
        right_ankle = point_or_fallback(self.RIGHT_ANKLE, right_knee + self._unit(right_knee - right_hip, down_dir) * lower_leg_len)
        left_toes = point_or_fallback(self.LEFT_FOOT_INDEX, left_ankle + np.array([0.0, 0.0, foot_len], dtype=float))
        right_toes = point_or_fallback(self.RIGHT_FOOT_INDEX, right_ankle + np.array([0.0, 0.0, foot_len], dtype=float))

        pose["LeftUpperLeg"] = left_knee
        pose["RightUpperLeg"] = right_knee
        pose["LeftLowerLeg"] = left_ankle
        pose["RightLowerLeg"] = right_ankle
        pose["LeftFoot"] = left_ankle
        pose["RightFoot"] = right_ankle
        pose["LeftToes"] = left_toes
        pose["RightToes"] = right_toes

        pose_3d = {bone: (pos - hips).astype(float).tolist() for bone, pos in pose.items()}
        return PoseEstimator.ensure_tracker_nodes(pose_3d)

    def _average_visibility(self, landmarks) -> float:
        visibility = [
            float(getattr(lm, "visibility", 1.0))
            for lm in landmarks
        ]
        return sum(visibility) / len(visibility) if visibility else 0.0

    def close(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None
        self.is_loaded = False
        self._frame_timestamp_ms = 0
