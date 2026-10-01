"""
身体姿态追踪模块

独立封装: YOLO 检测 + 姿态估计 + OSC/SteamVR 推流。
可独立启停，关闭时不影响面部和手部模块。
"""

import os
import sys
import time
import copy
from typing import Dict, Optional, List

import cv2
import numpy as np

from .base_module import BaseModule
from runtime_paths import bin_dir, runtime_root

# 将 Ultralytics 配置目录设到项目下
_ultra_cfg = str(runtime_root() / ".ultralytics")
os.makedirs(_ultra_cfg, exist_ok=True)
os.environ["ULTRALYTICS_SETTINGS"] = _ultra_cfg

sys.path.insert(0, str(bin_dir()))

from edition import yolo_available
from pose_estimator import PoseEstimator
from camera_capture import ImageProcessor
from mediapipe_pose import MediaPipePoseDetector
from streamers.body_osc import BodyOSCStreamer
from streamers.body_steamvr import BodySteamVRStreamer
from streamers.osc_transport import OSCTransport
from streamers.coord import CoordinateTransformer, hmd_pose_anchor_trusted


class BodyModule(BaseModule):
    """身体姿态追踪模块

    推理: YOLO26n 检测 + PoseEstimator 3D 重建
    推流: OSC (VRChat) 或 SteamVR (管道驱动)
    独立 OSC 地址: host:port
    """

    TRACKER_NODE_NAMES = [
        "HipsTracker",
        "ChestTracker",
        "LeftFootTracker",
        "RightFootTracker",
        "LeftKneeTracker",
        "RightKneeTracker",
        "LeftElbowTracker",
        "RightElbowTracker",
    ]
    PREVIEW_3D_CONNECTIONS = [
        ("Hips", "Spine"),
        ("Spine", "Chest"),
        ("Chest", "UpperChest"),
        ("UpperChest", "Neck"),
        ("Neck", "Head"),
        ("Hips", "LeftUpperLeg"),
        ("LeftUpperLeg", "LeftLowerLeg"),
        ("LeftLowerLeg", "LeftFoot"),
        ("LeftFoot", "LeftToes"),
        ("Hips", "RightUpperLeg"),
        ("RightUpperLeg", "RightLowerLeg"),
        ("RightLowerLeg", "RightFoot"),
        ("RightFoot", "RightToes"),
        ("UpperChest", "LeftUpperArm"),
        ("LeftUpperArm", "LeftLowerArm"),
        ("LeftLowerArm", "LeftHand"),
        ("UpperChest", "RightUpperArm"),
        ("RightUpperArm", "RightLowerArm"),
        ("RightLowerArm", "RightHand"),
    ]

    def __init__(self, config: Dict, streaming_config: Dict):
        super().__init__(name="body", config=config, streaming_config=streaming_config)

        self._detector = None
        self._mediapipe_pose = None
        self._pose_estimator = None
        self._secondary_pose_estimator = None
        self._streamer = None
        self._hmd_provider = None
        self._mp_pose_cfg = config.get("mediapipe_pose", {})
        self._inference_backend = self._normalize_inference_backend(
            config.get("inference_backend", "yolo")
        )
        self._last_pose_node_count = 0
        self._last_keypoint_count = 0
        self._last_tracker_nodes: List[str] = []
        self._last_pose_source = None
        self._last_stream_queued = False
        self._last_valid_pose_data = None
        self._last_valid_pose_time = 0.0
        self._last_held_pose_send_time = 0.0
        self._held_pose_streaming = False
        self._held_pose_age = None
        self._secondary_frame = None
        self._secondary_frame_count = -1
        self._secondary_processed_frame_count = -1
        self._secondary_last_process_time = 0.0
        self._secondary_last_pose = None
        self._secondary_last_pose_time = 0.0
        self._secondary_detect_count = 0
        self._secondary_total_count = 0
        self._secondary_last_source = None
        self._update_dual_camera_config(config)
        self._update_hmd_pose_hold_config(config)

        # 身体部位检测开关
        self._detect = {
            "torso": config.get("detect_torso", True),
            "left_arm": config.get("detect_left_arm", True),
            "right_arm": config.get("detect_right_arm", True),
            "left_leg": config.get("detect_left_leg", True),
            "right_leg": config.get("detect_right_leg", True),
        }

    def _update_dual_camera_config(self, config: Dict):
        proc_cfg = config.get("processing", {}) if isinstance(config.get("processing", {}), dict) else {}
        self._dual_camera_enabled = bool(proc_cfg.get("dual_camera_body_enabled", False))
        try:
            self._secondary_max_fps = max(0.0, float(proc_cfg.get("secondary_body_max_fps", 12) or 0.0))
        except (TypeError, ValueError):
            self._secondary_max_fps = 12.0
        self._secondary_min_interval = 1.0 / self._secondary_max_fps if self._secondary_max_fps > 0 else 0.0
        self._fusion_mode = str(proc_cfg.get("dual_camera_fusion_mode", "confidence") or "confidence")
        self._fusion_confidence_margin = float(proc_cfg.get("fusion_confidence_margin", 0.12) or 0.12)
        self._fusion_min_confidence = float(proc_cfg.get("fusion_min_confidence", 0.15) or 0.15)
        self._fusion_max_pose_delta = float(proc_cfg.get("fusion_max_pose_delta", 0.75) or 0.75)
        self._secondary_camera_cfg = config.get("camera_secondary", {}) if isinstance(config.get("camera_secondary", {}), dict) else {}

    def _update_hmd_pose_hold_config(self, config: Dict):
        processing = config.get("processing", {}) if isinstance(config.get("processing", {}), dict) else {}
        try:
            self._hmd_pose_hold_seconds = max(
                0.0, float(processing.get("hmd_pose_hold_seconds", 5.0))
            )
        except (TypeError, ValueError):
            self._hmd_pose_hold_seconds = 5.0
        try:
            refresh_fps = max(
                1.0, float(processing.get("hmd_pose_refresh_fps", 15.0))
            )
        except (TypeError, ValueError):
            refresh_fps = 15.0
        self._hmd_pose_refresh_interval = 1.0 / refresh_fps

    def set_secondary_frame(self, frame, frame_count: int):
        if not self._dual_camera_enabled:
            return
        self._secondary_frame = frame
        self._secondary_frame_count = frame_count

    def _do_initialize(self) -> bool:
        """初始化 YOLO 检测器、姿态估计器和推流器"""
        yolo_cfg = self._config.get("yolo", {})
        proc_cfg = self._config.get("processing", {})

        # YOLO body backend. MediaPipe Pose can be selected as the primary
        # body source and then YOLO is not loaded at all.
        if self._inference_backend == "yolo":
            # Imported lazily: the lite edition ships without torch/ultralytics.
            from yolo_detector import YOLO26nDetector

            model_path = yolo_cfg.get("model_path", "yolo26n-pose.pt")
            self._detector = YOLO26nDetector(
                model_path=model_path,
                device=yolo_cfg.get("device", "auto"),
                conf_threshold=yolo_cfg.get("conf_threshold", 0.5),
                half=yolo_cfg.get("half", True),
                precision=yolo_cfg.get("precision", "fp16"),
            )
            if not self._detector.load_model():
                self._detector = None
                print("[body] YOLO model load failed; trying MediaPipe Pose fallback")
        else:
            self._detector = None
            print("[body] Body inference backend: MediaPipe Pose")

        # 姿态估计器
        self._pose_estimator = PoseEstimator(
            smoothing=proc_cfg.get("smoothing", "one_euro"),
            smoothing_factor=proc_cfg.get("smoothing_factor", 0.3),
            one_euro_min_cutoff=proc_cfg.get("one_euro_min_cutoff", 1.0),
            one_euro_beta=proc_cfg.get("one_euro_beta", 0.007),
            average_smoothing_enabled=proc_cfg.get("average_smoothing_enabled", True),
            average_window=proc_cfg.get("average_window", 4),
            ema_alpha=proc_cfg.get("ema_alpha", 0.35),
            reset_on_tracking_lost=proc_cfg.get("reset_on_tracking_lost", True),
        )
        self._secondary_pose_estimator = PoseEstimator(
            smoothing="linear",
            smoothing_factor=1.0,
            average_smoothing_enabled=False,
            reset_on_tracking_lost=False,
        )

        self._init_mediapipe_pose()
        if self._inference_backend == "mediapipe_pose" and not self._mediapipe_pose:
            print("[body] MediaPipe Pose backend unavailable")
            return False
        if self._inference_backend == "yolo" and not self._detector and not self._mediapipe_pose:
            print("[body] No body inference backend is available")
            return False

        # 推流器
        host = self._streaming_config.get("host", "127.0.0.1")
        port = self._streaming_config.get("port", 9000)
        protocol = self._streaming_config.get("protocol", "osc")
        tracker_toggles = self._streaming_config.get("tracker_toggles", {})
        send_position = self._streaming_config.get("send_position", True)
        send_rotation = self._streaming_config.get("send_rotation", True)
        send_head_position = self._streaming_config.get("send_head_position", False)
        decouple_from_hmd = self._streaming_config.get("decouple_from_hmd", False)
        hmd_height_scaling = self._streaming_config.get("hmd_height_scaling", {})
        steamvr_cfg = self._streaming_config.get("steamvr") or self._config.get("steamvr", {})
        hmd_to_neck_offset = steamvr_cfg.get("hmd_to_neck_offset", [0, -0.2, 0.1])

        if protocol == "steamvr":
            self._streamer = BodySteamVRStreamer(
                host=host, port=port,
                smoothing=steamvr_cfg.get("smoothing", 50),
                additional_smoothing=steamvr_cfg.get("additional_smoothing", 0.5),
                camera_latency=steamvr_cfg.get("camera_latency", 0),
                hmd_to_neck_offset=hmd_to_neck_offset,
                trackers=steamvr_cfg.get("trackers", None),
                tracker_toggles=tracker_toggles,
                decouple_from_hmd=decouple_from_hmd,
                hmd_height_scaling=hmd_height_scaling,
                yaw_offset_deg=steamvr_cfg.get("yaw_offset_deg", 0.0),
                auto_yaw_calibration=steamvr_cfg.get("auto_yaw_calibration", True),
            )
        else:
            self._streamer = BodyOSCStreamer(
                host=host, port=port,
                tracker_toggles=tracker_toggles,
                send_position=send_position,
                send_rotation=send_rotation,
                send_head_position=send_head_position,
                decouple_from_hmd=decouple_from_hmd,
                hmd_to_neck_offset=hmd_to_neck_offset,
                hmd_height_scaling=hmd_height_scaling,
            )

        # 设置推流开关
        self._attach_hmd_provider()
        self._streamer.inference_enabled = self._enabled
        self._streamer.enabled = self._streaming_enabled

        # 连接推流
        if self._streaming_enabled:
            if self._streamer.connect():
                print(f"[body] 推流已连接 ({protocol}://{host}:{port})")
            else:
                print(f"[body] 推流连接失败 ({protocol}://{host}:{port})")

        return True

    def _do_process(self, frame) -> Optional[Dict]:
        """执行身体姿态检测和推流"""
        flip_horizontal = self._config.get("camera", {}).get("flip_horizontal", True)
        pose_data = self._process_pose_sources(
            frame,
            pose_estimator=self._pose_estimator,
            flip_horizontal=flip_horizontal,
        )
        secondary_pose = self._process_secondary_body_frame()
        pose_data = self._fuse_dual_camera_pose(pose_data, secondary_pose)
        if pose_data is not None:
            # Filter disabled regions before retargeting, then repeat after
            # hierarchy rotation reconstruction so missing bones cannot be
            # reintroduced as default outputs.
            pose_data = self._filter_body_parts(pose_data)
            pose_data = self._finalize_pose(pose_data)
            pose_data = self._filter_body_parts(pose_data)
        if pose_data is None:
            if self._stream_held_pose_for_hmd():
                return None
            self._last_pose_node_count = 0
            self._last_keypoint_count = 0
            self._last_tracker_nodes = []
            self._last_pose_source = None
            self._last_stream_queued = False
            self._held_pose_streaming = False
            self._held_pose_age = None
            if self._pose_estimator and self._pose_estimator.reset_on_tracking_lost:
                self._pose_estimator.reset_filters()
            return None

        # 过滤身体部位
        pose_3d = pose_data.get("pose_3d", {})
        self._last_pose_node_count = len(pose_3d)
        self._last_keypoint_count = len(pose_data.get("keypoints_2d", {}))
        self._last_tracker_nodes = [
            name for name in self.TRACKER_NODE_NAMES
            if name in pose_3d
        ]
        self._last_pose_source = pose_data.get("source", "unknown")
        self._last_stream_queued = False
        self._held_pose_streaming = False
        self._held_pose_age = 0.0
        self._last_valid_pose_data = copy.deepcopy(pose_data)
        self._last_valid_pose_time = time.perf_counter()

        # 推流
        if self._streaming_enabled and self._streamer and self._streamer.connected:
            self._last_stream_queued = self._streamer.send(pose_data)

        return pose_data

    def _do_cleanup(self):
        """释放资源"""
        if self._streamer:
            self._streamer.disconnect()
            self._streamer = None
        self._detector = None
        if self._mediapipe_pose:
            self._mediapipe_pose.close()
            self._mediapipe_pose = None
        self._pose_estimator = None
        self._secondary_pose_estimator = None
        self._secondary_frame = None
        self._secondary_last_pose = None
        self._last_valid_pose_data = None
        self._last_valid_pose_time = 0.0
        self._held_pose_streaming = False
        self._held_pose_age = None

    def _init_mediapipe_pose(self):
        cfg = self._mp_pose_cfg or {}
        if self._inference_backend != "mediapipe_pose" and not cfg.get("enabled", False):
            return

        self._mediapipe_pose = MediaPipePoseDetector(
            model_variant=cfg.get("model_variant", "heavy"),
            model_path=cfg.get("model_path", "pose_landmarker_heavy.task"),
            min_detection_confidence=cfg.get("min_detection_confidence", 0.5),
            min_presence_confidence=cfg.get("min_presence_confidence", 0.5),
            min_tracking_confidence=cfg.get("min_tracking_confidence", 0.5),
        )
        if not self._mediapipe_pose.load():
            self._mediapipe_pose = None
            if self._inference_backend == "mediapipe_pose":
                print("[body] MediaPipe Pose backend failed to load")
            else:
                print("[body] MediaPipe Pose disabled; continuing with YOLO body source")
        else:
            mode = "primary" if self._inference_backend == "mediapipe_pose" else cfg.get("mode", "fallback")
            print(f"[body] MediaPipe Pose source enabled ({mode})")

    def _process_mediapipe_pose(self, frame) -> Optional[Dict]:
        if not self._mediapipe_pose:
            return None

        result = self._mediapipe_pose.detect(frame)
        if not result or "pose_3d" not in result:
            return None

        return {
            "pose_3d": result["pose_3d"],
            "rotations": {},
            "keypoints_2d": result.get("keypoints_2d", {}),
            "confidence": result.get("confidence", 0.0),
            "source": "mediapipe_pose",
        }

    def _process_yolo_pose(
        self,
        frame,
        pose_estimator: Optional[PoseEstimator] = None,
        flip_horizontal: bool = True,
    ) -> Optional[Dict]:
        if self._detector is None:
            return None

        detections = self._detector.detect(frame)
        if not detections or not detections.get("keypoints"):
            return None

        estimator = pose_estimator or self._pose_estimator
        if estimator is None:
            return None
        pose_data = estimator.process_detections(
            detections,
            frame.shape[1],
            frame.shape[0],
            flip_horizontal=flip_horizontal,
            apply_filters=False,
        )
        if pose_data is not None:
            pose_data.setdefault("source", "yolo")
        return pose_data

    def _process_pose_sources(
        self,
        frame,
        pose_estimator: Optional[PoseEstimator] = None,
        flip_horizontal: bool = True,
    ) -> Optional[Dict]:
        if self._inference_backend == "mediapipe_pose":
            return self._process_mediapipe_pose(frame)

        mode = str((self._mp_pose_cfg or {}).get("mode", "fallback") or "fallback")
        mode = mode.strip().lower()
        if mode not in ("fallback", "prefer_mediapipe", "blend"):
            mode = "fallback"

        yolo_pose = None
        mp_pose = None
        if mode == "prefer_mediapipe":
            mp_pose = self._process_mediapipe_pose(frame)
            if mp_pose is None:
                yolo_pose = self._process_yolo_pose(
                    frame,
                    pose_estimator=pose_estimator,
                    flip_horizontal=flip_horizontal,
                )
        else:
            yolo_pose = self._process_yolo_pose(
                frame,
                pose_estimator=pose_estimator,
                flip_horizontal=flip_horizontal,
            )
            if mode == "blend" or yolo_pose is None:
                mp_pose = self._process_mediapipe_pose(frame)

        return self._select_pose_source(yolo_pose, mp_pose, finalize=False)

    @staticmethod
    def _normalize_inference_backend(value: str) -> str:
        backend = str(value or "yolo").strip().lower()
        aliases = {
            "mp": "mediapipe_pose",
            "mediapipe": "mediapipe_pose",
            "mediapipepose": "mediapipe_pose",
            "media_pipe_pose": "mediapipe_pose",
        }
        if backend in aliases:
            backend = aliases[backend]
        if backend not in ("yolo", "mediapipe_pose"):
            backend = "yolo"
        if backend == "yolo" and not yolo_available():
            return "mediapipe_pose"
        return backend

    def _finalize_pose(self, pose_data: Dict) -> Dict:
        pose_3d = pose_data.get("pose_3d", {})
        pose_3d = self._pose_estimator.apply_average_smoothing(pose_3d)
        pose_3d = self._pose_estimator.smooth_pose(pose_3d)
        rotations = self._pose_estimator.calculate_rotations(pose_3d)
        rotations = self._pose_estimator.smooth_rotations(rotations)
        pose_data["pose_3d"] = pose_3d
        pose_data["rotations"] = rotations
        return pose_data

    def _select_pose_source(self, yolo_pose: Optional[Dict], mp_pose: Optional[Dict], finalize: bool = True) -> Optional[Dict]:
        if self._inference_backend == "mediapipe_pose":
            return self._finalize_pose(mp_pose) if finalize and mp_pose is not None else mp_pose

        if mp_pose is None:
            if yolo_pose is not None:
                yolo_pose.setdefault("source", "yolo")
                return self._finalize_pose(yolo_pose) if finalize else yolo_pose
            return yolo_pose
        if yolo_pose is None:
            return self._finalize_pose(mp_pose) if finalize else mp_pose

        mode = (self._mp_pose_cfg or {}).get("mode", "fallback")
        if mode == "prefer_mediapipe":
            return self._finalize_pose(mp_pose) if finalize else mp_pose
        if mode != "blend":
            yolo_pose.setdefault("source", "yolo")
            return self._finalize_pose(yolo_pose) if finalize else yolo_pose

        weight = float((self._mp_pose_cfg or {}).get("blend_weight", 0.35))
        weight = min(1.0, max(0.0, weight))
        blended_pose = {}
        yolo_3d = yolo_pose.get("pose_3d", {})
        mp_3d = mp_pose.get("pose_3d", {})
        for bone in set(yolo_3d) | set(mp_3d):
            if bone in yolo_3d and bone in mp_3d:
                blended_pose[bone] = [
                    (1.0 - weight) * float(yolo_3d[bone][i]) + weight * float(mp_3d[bone][i])
                    for i in range(3)
                ]
            elif bone in yolo_3d:
                blended_pose[bone] = list(yolo_3d[bone])
            else:
                blended_pose[bone] = list(mp_3d[bone])

        blended = dict(yolo_pose)
        blended["pose_3d"] = blended_pose
        blended["source"] = "yolo+mediapipe_pose"
        blended["confidence"] = max(
            float(yolo_pose.get("confidence", 0.0)),
            float(mp_pose.get("confidence", 0.0)),
        )
        return self._finalize_pose(blended) if finalize else blended

    def _secondary_frame_due(self) -> bool:
        if not self._dual_camera_enabled or self._secondary_frame is None:
            return False
        if self._secondary_frame_count == self._secondary_processed_frame_count:
            return False
        if self._secondary_min_interval <= 0:
            return True
        return (time.perf_counter() - self._secondary_last_process_time) >= self._secondary_min_interval

    def _process_secondary_body_frame(self) -> Optional[Dict]:
        if not self._secondary_frame_due():
            if self._secondary_last_pose and time.perf_counter() - self._secondary_last_pose_time <= 0.5:
                return self._secondary_last_pose
            return None

        frame = self._secondary_frame
        self._secondary_processed_frame_count = self._secondary_frame_count
        self._secondary_last_process_time = time.perf_counter()
        self._secondary_total_count += 1
        sec_cfg = self._secondary_camera_cfg or {}
        pose_data = self._process_pose_sources(
            frame,
            pose_estimator=self._secondary_pose_estimator,
            flip_horizontal=sec_cfg.get("flip_horizontal", True),
        )
        if pose_data is not None:
            source = str(pose_data.get("source", "unknown"))
            if not source.startswith("secondary_"):
                pose_data["source"] = f"secondary_{source}"

        if pose_data is not None:
            self._secondary_detect_count += 1
            self._secondary_last_pose = pose_data
            self._secondary_last_pose_time = time.perf_counter()
            self._secondary_last_source = pose_data.get("source")
        return pose_data

    def _stream_held_pose_for_hmd(self) -> bool:
        if (
            self._hmd_pose_hold_seconds <= 0.0
            or self._last_valid_pose_data is None
            or self._hmd_provider is None
        ):
            return False

        now = time.perf_counter()
        age = now - self._last_valid_pose_time
        if age > self._hmd_pose_hold_seconds:
            self._last_valid_pose_data = None
            return False

        hmd_pose = self._hmd_provider.get_pose()
        if not hmd_pose.valid or getattr(hmd_pose, "actual_height", None) is None:
            return False

        held_pose = self._last_valid_pose_data
        pose_3d = held_pose.get("pose_3d", {})
        self._last_pose_node_count = len(pose_3d)
        self._last_tracker_nodes = [
            name for name in self.TRACKER_NODE_NAMES if name in pose_3d
        ]
        self._last_pose_source = f"{held_pose.get('source', 'unknown')}+hmd_hold"
        self._held_pose_streaming = True
        self._held_pose_age = age

        if now - self._last_held_pose_send_time < self._hmd_pose_refresh_interval:
            return True
        self._last_held_pose_send_time = now
        self._last_stream_queued = bool(
            self._streaming_enabled
            and self._streamer
            and self._streamer.connected
            and self._streamer.send(held_pose)
        )
        return True

    _BONE_KEYPOINTS = {
        "Hips": ("left_hip", "right_hip"),
        "HipsTracker": ("left_hip", "right_hip"),
        "Spine": ("left_hip", "right_hip", "left_shoulder", "right_shoulder"),
        "Chest": ("left_shoulder", "right_shoulder"),
        "ChestTracker": ("left_shoulder", "right_shoulder"),
        "UpperChest": ("left_shoulder", "right_shoulder"),
        "Neck": ("left_shoulder", "right_shoulder", "nose"),
        "Head": ("nose",),
        "LeftUpperArm": ("left_shoulder", "left_elbow"),
        "LeftLowerArm": ("left_elbow", "left_wrist"),
        "LeftElbowTracker": ("left_elbow",),
        "LeftHand": ("left_wrist",),
        "RightUpperArm": ("right_shoulder", "right_elbow"),
        "RightLowerArm": ("right_elbow", "right_wrist"),
        "RightElbowTracker": ("right_elbow",),
        "RightHand": ("right_wrist",),
        "LeftUpperLeg": ("left_hip", "left_knee"),
        "LeftLowerLeg": ("left_knee", "left_ankle"),
        "LeftKneeTracker": ("left_knee",),
        "LeftFoot": ("left_ankle",),
        "LeftFootTracker": ("left_ankle",),
        "RightUpperLeg": ("right_hip", "right_knee"),
        "RightLowerLeg": ("right_knee", "right_ankle"),
        "RightKneeTracker": ("right_knee",),
        "RightFoot": ("right_ankle",),
        "RightFootTracker": ("right_ankle",),
    }

    def _bone_confidence(self, pose_data: Dict, bone: str) -> float:
        keypoints = pose_data.get("keypoints_2d") or {}
        names = self._BONE_KEYPOINTS.get(bone, ())
        values = []
        for name in names:
            item = keypoints.get(name)
            if item is not None and len(item) >= 3:
                try:
                    values.append(float(item[2]))
                except (TypeError, ValueError):
                    pass
        if values:
            return max(0.0, min(1.0, sum(values) / len(values)))
        try:
            return max(0.0, min(1.0, float(pose_data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            return 0.0

    def _fuse_dual_camera_pose(self, primary: Optional[Dict], secondary: Optional[Dict]) -> Optional[Dict]:
        if not self._dual_camera_enabled or secondary is None:
            return primary
        if primary is None:
            return secondary if float(secondary.get("confidence", 0.0) or 0.0) >= self._fusion_min_confidence else None

        primary_3d = primary.get("pose_3d", {})
        secondary_3d = secondary.get("pose_3d", {})
        if not primary_3d or not secondary_3d:
            return primary

        fused_pose = {}
        used_secondary = 0
        for bone in set(primary_3d) | set(secondary_3d):
            p = primary_3d.get(bone)
            s = secondary_3d.get(bone)
            if p is None:
                if self._bone_confidence(secondary, bone) >= self._fusion_min_confidence:
                    fused_pose[bone] = [float(v) for v in s]
                    used_secondary += 1
                continue
            if s is None:
                fused_pose[bone] = [float(v) for v in p]
                continue

            pc = self._bone_confidence(primary, bone)
            sc = self._bone_confidence(secondary, bone)
            if sc < self._fusion_min_confidence:
                fused_pose[bone] = [float(v) for v in p]
                continue
            p_arr = np.array(p, dtype=float)
            s_arr = np.array(s, dtype=float)
            if not np.all(np.isfinite(p_arr)) or not np.all(np.isfinite(s_arr)):
                fused_pose[bone] = [float(v) for v in p]
                continue
            if np.linalg.norm(p_arr - s_arr) > self._fusion_max_pose_delta:
                if sc > pc + self._fusion_confidence_margin:
                    fused_pose[bone] = s_arr.tolist()
                    used_secondary += 1
                else:
                    fused_pose[bone] = p_arr.tolist()
                continue
            if sc > pc + self._fusion_confidence_margin:
                weight_secondary = 0.65
            elif pc > sc + self._fusion_confidence_margin:
                weight_secondary = 0.20
            else:
                total = max(1e-6, pc + sc)
                weight_secondary = max(0.25, min(0.55, sc / total))
            fused_pose[bone] = (p_arr * (1.0 - weight_secondary) + s_arr * weight_secondary).tolist()
            if weight_secondary > 0.25:
                used_secondary += 1

        fused = dict(primary)
        fused["pose_3d"] = fused_pose
        fused["source"] = f"{primary.get('source', 'primary')}+dual_camera"
        fused["secondary_source"] = secondary.get("source")
        fused["secondary_used_nodes"] = used_secondary
        fused["confidence"] = max(float(primary.get("confidence", 0.0) or 0.0), float(secondary.get("confidence", 0.0) or 0.0))
        return fused

    def _filter_body_parts(self, pose_data: Dict) -> Dict:
        """根据部位开关过滤数据"""
        if not pose_data or "pose_3d" not in pose_data:
            return pose_data

        pose_3d = pose_data.get("pose_3d", {})
        rotations = pose_data.get("rotations", {})

        # 部位到骨骼的映射
        part_to_bones = {
            "torso": ["Hips", "Chest", "Neck", "Spine", "HipsTracker", "ChestTracker"],
            "left_arm": ["LeftUpperArm", "LeftLowerArm", "LeftHand", "LeftElbowTracker"],
            "right_arm": ["RightUpperArm", "RightLowerArm", "RightHand", "RightElbowTracker"],
            "left_leg": ["LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftKneeTracker", "LeftFootTracker"],
            "right_leg": ["RightUpperLeg", "RightLowerLeg", "RightFoot", "RightKneeTracker", "RightFootTracker"],
        }

        for part, enabled in self._detect.items():
            if not enabled:
                bones = part_to_bones.get(part, [])
                for bone in bones:
                    pose_3d.pop(bone, None)
                    rotations.pop(bone, None)

        return pose_data

    def set_hmd_provider(self, provider):
        """设置 HMD 位姿数据提供者"""
        self._hmd_provider = provider
        self._attach_hmd_provider()

    def _attach_hmd_provider(self):
        if self._streamer:
            self._streamer.set_hmd_provider(self._hmd_provider)

    def set_tracker_toggles(self, toggles: Dict[str, bool]):
        """更新追踪器开关"""
        if self._streamer:
            self._streamer.set_tracker_toggles(toggles)

    def reset_filters(self):
        """重置滤波器"""
        if self._pose_estimator:
            self._pose_estimator.reset_filters()

    def draw_landmarks(self, frame, body_data: Dict):
        """Draw the body skeleton in a small lower-left preview panel."""
        if not body_data:
            return frame
        return self._draw_body_preview_panel(frame, body_data)

    def _draw_body_preview_panel(self, frame, body_data: Dict):
        """Render body pose away from the camera subject to avoid occlusion."""
        keypoints = body_data.get("keypoints_2d", {})
        points = self._keypoints_2d_to_array(keypoints)
        if points is not None and self._has_drawable_2d_connection(points):
            return self._draw_2d_preview_panel(frame, points)

        return self._draw_pose3d_preview(frame, body_data.get("pose_3d", {}))

    def _keypoints_2d_to_array(self, keypoints):
        if not keypoints:
            return None

        points = []
        visible = 0
        for name in PoseEstimator.KP.keys():
            item = keypoints.get(name)
            if item is None:
                points.append([0.0, 0.0, 0.0])
                continue
            x, y = float(item[0]), float(item[1])
            conf = float(item[2]) if len(item) > 2 else 1.0
            if x > 0 and y > 0:
                visible += 1
            points.append([x, y, conf])

        return np.array(points, dtype=float) if visible > 0 else None

    def _has_drawable_2d_connection(self, points: np.ndarray) -> bool:
        for conns in ImageProcessor.CONNECTION_GROUPS.values():
            for i, j in conns:
                if i < len(points) and j < len(points):
                    if points[i][0] > 0 and points[i][1] > 0 and points[j][0] > 0 and points[j][1] > 0:
                        return True
        return False

    def _preview_panel_rect(self, frame):
        h, w = frame.shape[:2]
        margin = max(10, int(min(w, h) * 0.018))
        panel_w = int(np.clip(w * 0.26, 170, 320))
        panel_h = int(np.clip(h * 0.30, 135, 240))
        x0 = margin
        y0 = max(margin, h - panel_h - margin)
        return x0, y0, x0 + panel_w, y0 + panel_h

    def _draw_preview_panel_background(self, frame, title: str):
        x0, y0, x1, y1 = self._preview_panel_rect(frame)
        roi = frame[y0:y1 + 1, x0:x1 + 1]
        if roi.size:
            black = np.zeros_like(roi)
            cv2.addWeighted(black, 0.45, roi, 0.55, 0, roi)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(
            frame, title, (x0 + 8, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 120), 1, cv2.LINE_AA
        )
        return x0, y0, x1, y1

    @staticmethod
    def _project_preview_points(source_points: np.ndarray, rect):
        x0, y0, x1, y1 = rect
        inner_x0, inner_y0 = x0 + 10, y0 + 26
        inner_x1, inner_y1 = x1 - 10, y1 - 10
        inner_w = max(1, inner_x1 - inner_x0)
        inner_h = max(1, inner_y1 - inner_y0)

        min_xy = source_points.min(axis=0)
        max_xy = source_points.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        scale = min(inner_w / span[0], inner_h / span[1])
        if not np.isfinite(scale) or scale <= 0:
            return None

        center = np.array([
            (inner_x0 + inner_x1) * 0.5,
            (inner_y0 + inner_y1) * 0.5,
        ], dtype=float)
        source_center = (min_xy + max_xy) * 0.5

        def project_xy(x, y):
            projected = center + (np.array([x, y], dtype=float) - source_center) * scale
            return int(round(projected[0])), int(round(projected[1]))

        return project_xy

    def _draw_2d_preview_panel(self, frame, points: np.ndarray):
        visible = np.array(
            [[float(p[0]), float(p[1])] for p in points if p[0] > 0 and p[1] > 0],
            dtype=float,
        )
        if len(visible) < 2:
            return frame

        rect = self._draw_preview_panel_background(frame, "Body Pose")
        project_xy = self._project_preview_points(visible, rect)
        if project_xy is None:
            return frame

        for group_name, conns in ImageProcessor.CONNECTION_GROUPS.items():
            color = ImageProcessor.SKELETON_COLORS.get(group_name, (80, 255, 120))
            for i, j in conns:
                if i >= len(points) or j >= len(points):
                    continue
                if points[i][0] <= 0 or points[i][1] <= 0 or points[j][0] <= 0 or points[j][1] <= 0:
                    continue
                cv2.line(
                    frame,
                    project_xy(float(points[i][0]), float(points[i][1])),
                    project_xy(float(points[j][0]), float(points[j][1])),
                    color, 2, cv2.LINE_AA,
                )

        for p in points:
            if p[0] > 0 and p[1] > 0:
                cv2.circle(frame, project_xy(float(p[0]), float(p[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)

        return frame

    def _draw_pose3d_preview(self, frame, pose_3d: Dict):
        """Draw reconstructed body poses in the lower-left preview panel."""
        if not pose_3d:
            return frame

        usable_names = {
            name
            for connection in self.PREVIEW_3D_CONNECTIONS
            for name in connection
            if name in pose_3d
        }
        if len(usable_names) < 2:
            return frame

        source_points = np.array(
            [[float(pose_3d[name][0]), float(pose_3d[name][1])] for name in usable_names],
            dtype=float,
        )
        rect = self._draw_preview_panel_background(frame, "Body Pose")
        project_xy = self._project_preview_points(source_points, rect)
        if project_xy is None:
            return frame

        color = (80, 255, 120)

        def project(name):
            pos = pose_3d.get(name)
            if pos is None:
                return None
            return project_xy(float(pos[0]), float(pos[1]))

        for a, b in self.PREVIEW_3D_CONNECTIONS:
            pa = project(a)
            pb = project(b)
            if pa is None or pb is None:
                continue
            cv2.line(frame, pa, pb, color, 3, cv2.LINE_AA)
            cv2.circle(frame, pa, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pb, 4, (255, 255, 255), -1, cv2.LINE_AA)

        return frame

    def _preview_anchor_from_keypoints(self, keypoints_2d: Dict, frame_w: int, frame_h: int):
        if keypoints_2d:
            hip_points = [
                keypoints_2d.get("left_hip"),
                keypoints_2d.get("right_hip"),
            ]
            hip_points = [p for p in hip_points if p is not None and p[0] > 0 and p[1] > 0]
            if hip_points:
                return (
                    sum(float(p[0]) for p in hip_points) / len(hip_points),
                    sum(float(p[1]) for p in hip_points) / len(hip_points),
                )

            visible = [p for p in keypoints_2d.values() if p is not None and p[0] > 0 and p[1] > 0]
            if visible:
                return (
                    sum(float(p[0]) for p in visible) / len(visible),
                    sum(float(p[1]) for p in visible) / len(visible),
                )

        return frame_w * 0.5, frame_h * 0.58

    def on_enabled(self):
        """模块启用回调"""
        if self._streamer:
            self._streamer.inference_enabled = True
            self._streamer.enabled = self._streaming_enabled
            if self._streaming_enabled and not self._streamer.connected:
                self._streamer.connect()

    def on_disabled(self):
        """模块禁用回调"""
        if self._streamer:
            self._streamer.inference_enabled = False
            self._streamer.enabled = False
            if self._streamer.connected:
                self._streamer.disconnect()

    def update_config(self, config: Dict, streaming_config: Dict):
        """Hot-update body recognition and streaming settings."""
        new_backend = self._normalize_inference_backend(
            config.get("inference_backend", self._inference_backend)
        )
        backend_changed = new_backend != self._inference_backend
        if backend_changed and self.initialized:
            print(f"[body] Body inference backend changed: {self._inference_backend} -> {new_backend}")
            self.cleanup()

        super().update_config(config, streaming_config)
        self._update_dual_camera_config(config)
        self._update_hmd_pose_hold_config(config)
        self._last_valid_pose_data = None
        self._held_pose_streaming = False
        self._held_pose_age = None
        self._mp_pose_cfg = config.get("mediapipe_pose", self._mp_pose_cfg)
        self._inference_backend = new_backend

        self._detect = {
            "torso": config.get("detect_torso", True),
            "left_arm": config.get("detect_left_arm", True),
            "right_arm": config.get("detect_right_arm", True),
            "left_leg": config.get("detect_left_leg", True),
            "right_leg": config.get("detect_right_leg", True),
        }

        if backend_changed and self._enabled:
            if self.initialize():
                print(f"[body] Body inference backend switched to {self._inference_backend}")
            else:
                print(f"[body] Body inference backend switch failed: {self._inference_backend}")
            return

        if self._streamer:
            streamer_active = self._enabled and self._streaming_enabled
            self._streamer.inference_enabled = self._enabled
            self._streamer.enabled = streamer_active
            if hasattr(self._streamer, "update_streaming_options"):
                steamvr_cfg = streaming_config.get("steamvr") or config.get("steamvr", {})
                kwargs = {
                    "tracker_toggles": streaming_config.get("tracker_toggles", {}),
                    "decouple_from_hmd": streaming_config.get("decouple_from_hmd", False),
                    "hmd_height_scaling": streaming_config.get("hmd_height_scaling", {}),
                }
                if isinstance(self._streamer, BodyOSCStreamer):
                    kwargs.update({
                        "send_position": streaming_config.get("send_position", True),
                        "send_rotation": streaming_config.get("send_rotation", True),
                        "send_head_position": streaming_config.get("send_head_position", False),
                    })
                elif isinstance(self._streamer, BodySteamVRStreamer):
                    kwargs.update({
                        "smoothing": steamvr_cfg.get("smoothing", 50),
                        "additional_smoothing": steamvr_cfg.get("additional_smoothing", 0.5),
                        "camera_latency": steamvr_cfg.get("camera_latency", 0),
                        "yaw_offset_deg": steamvr_cfg.get("yaw_offset_deg", 0.0),
                        "auto_yaw_calibration": steamvr_cfg.get("auto_yaw_calibration", True),
                    })
                self._streamer.update_streaming_options(**kwargs)
            elif hasattr(self._streamer, "set_decouple_from_hmd"):
                self._streamer.set_decouple_from_hmd(streaming_config.get("decouple_from_hmd", False))
            if streamer_active and not self._streamer.connected:
                self._streamer.connect()
            elif not streamer_active and self._streamer.connected:
                self._streamer.disconnect()

    def get_status(self) -> Dict:
        status = super().get_status()
        status["hmd_height_state"] = {
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
        status.update({
            "pose_node_count": self._last_pose_node_count,
            "keypoint_2d_count": self._last_keypoint_count,
            "tracker_node_count": len(self._last_tracker_nodes),
            "tracker_nodes": self._last_tracker_nodes,
            "pose_source": self._last_pose_source,
            "inference_backend": self._inference_backend,
            "stream_queued": self._last_stream_queued,
            "dual_camera_enabled": self._dual_camera_enabled,
            "secondary_body_max_fps": self._secondary_max_fps,
            "secondary_detect_rate": (
                (self._secondary_detect_count / self._secondary_total_count) * 100
                if self._secondary_total_count else 0.0
            ),
            "secondary_pose_source": self._secondary_last_source,
            "secondary_frame_count": self._secondary_processed_frame_count,
            "held_pose_streaming": self._held_pose_streaming,
            "held_pose_age": self._held_pose_age,
            "hmd_pose_hold_seconds": self._hmd_pose_hold_seconds,
        })
        if self._streamer:
            status["connected"] = self._streamer.connected
            streamer_status = self._streamer.get_status()
            status["streamer_fps"] = streamer_status.get("fps", 0)
            status["streamer_send_count"] = streamer_status.get("send_count", 0)
            status["streamer_error_count"] = streamer_status.get("error_count", 0)
            status["streamer_drop_count"] = streamer_status.get("drop_count", 0)
            status["streamer_queue_size"] = streamer_status.get("queue_size", 0)
            status["streamer_last_message_count"] = streamer_status.get("last_message_count", 0)
            status["streamer_last_tracker_count"] = streamer_status.get("last_tracker_count", 0)
            status["send_position"] = streamer_status.get("send_position")
            status["send_rotation"] = streamer_status.get("send_rotation")
            status["send_head_position"] = streamer_status.get("send_head_position")
            status["decouple_from_hmd"] = streamer_status.get("decouple_from_hmd")
            status["hmd_height_state"] = streamer_status.get(
                "hmd_height_state", status["hmd_height_state"]
            )
            for key in (
                "hmd_present",
                "hmd_anchor_trusted",
                "hmd_anchor_applied",
                "hmd_tracking_origin",
                "hmd_anchor_source",
                "last_driver_pose_valid",
                "last_driver_pose_age",
                "yaw_calibrated",
                "yaw_offset_deg",
            ):
                if key in streamer_status:
                    status[key] = streamer_status[key]

        # Provider state remains authoritative even before the first body frame.
        if self._hmd_provider is not None:
            try:
                hmd_pose = self._hmd_provider.get_pose()
                hmd_trusted = hmd_pose_anchor_trusted(hmd_pose)
                status["hmd_present"] = bool(hmd_pose.valid)
                if status.get("hmd_anchor_source") != "driver_raw":
                    status["hmd_anchor_trusted"] = bool(hmd_trusted)
                    status["hmd_anchor_applied"] = bool(
                        hmd_trusted
                        and not bool(
                            self._streaming_config.get("decouple_from_hmd", False)
                        )
                    )
                status["hmd_tracking_origin"] = str(
                    getattr(hmd_pose, "tracking_origin", "unknown") or "unknown"
                )
                status["hmd_actual_height"] = getattr(
                    hmd_pose, "actual_height", None
                )
            except Exception:
                pass
        return status
