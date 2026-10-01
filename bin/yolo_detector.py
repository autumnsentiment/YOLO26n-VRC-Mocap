import os
import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import threading
import time
from collections import deque
from runtime_paths import runtime_root

# 将 Ultralytics 配置目录设到项目下，避免 AppData 权限问题
_ultra_cfg = str(runtime_root() / ".ultralytics")
os.makedirs(_ultra_cfg, exist_ok=True)
os.environ["ULTRALYTICS_SETTINGS"] = _ultra_cfg

from ultralytics import YOLO


class YOLO26nDetector:
    """YOLO26n 姿态估计检测器
    
    适配 YOLO26 特性:
    - 端到端 NMS-free 推理
    - RLE 姿态估计（更高精度关键点）
    - DFL 移除（更简化推理）
    - FP16 半精度推理支持
    """

    # COCO 17关键点定义
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # YOLO26 骨骼连接关系（用于可视化）
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    def __init__(self, model_path: str = "yolo26n-pose.pt", device: str = "auto",
                 conf_threshold: float = 0.5, half: bool = True, precision: str = "fp16"):
        self.conf_threshold = conf_threshold
        self.precision = precision  # "fp32", "fp16", "int8"
        self.half = half
        self.model = None
        self.model_path = model_path
        self.is_loaded = False
        self.lock = threading.Lock()
        self.fps_history = deque(maxlen=60)
        self.warmup_done = False

        # 自动选择设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # CPU 不支持 FP16/INT8，回退到 FP32
        if self.device == "cpu":
            self.precision = "fp32"
            self.half = False
        elif self.precision == "fp16":
            self.half = True
        else:
            self.half = False

    def load_model(self, model_path: Optional[str] = None) -> bool:
        try:
            path = model_path or self.model_path
            if path is None:
                path = "yolo26n-pose.pt"

            print(f"Loading YOLO26n model: {path}")
            self.model = YOLO(path)
            self.model.to(self.device)

            # 根据精度设置应用量化
            if self.precision == "int8" and self.device == "cuda":
                # torch dynamic quantization is CPU-only and skips Conv layers,
                # so it gives no real INT8 on CUDA. Real INT8 needs a TensorRT
                # export (model.export(format="engine", int8=True)).
                print("INT8 is not available for PyTorch CUDA weights; using FP16")
                self.precision = "fp16"
                self.half = True
                self.model.model.half()
            elif self.precision == "fp16" and self.device == "cuda":
                self.half = True
                self.model.model.half()
                print("FP16 half-precision enabled")
            else:
                self.precision = "fp32"
                self.half = False
                print("FP32 full-precision mode")

            self.is_loaded = True
            self.model_path = path
            print(f"Model loaded on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
            return False

    def warmup(self, imgsz: int = 640, shape: Optional[Tuple[int, int]] = None) -> None:
        """预热模型，消除首帧延迟"""
        if self.warmup_done or not self.is_loaded:
            return

        try:
            print("Warming up model...")
            height, width = shape if shape else (imgsz, imgsz)
            dummy = np.zeros((int(height), int(width), 3), dtype=np.uint8)
            # Same half/conf flags as detect() so the first real frame does not
            # trigger another CUDA kernel selection pass.
            self.model(dummy, conf=self.conf_threshold, verbose=False,
                       device=self.device, half=self.half)
            self.warmup_done = True
            print("Warmup complete")
        except Exception as e:
            print(f"Warmup failed: {e}")

    def detect(self, frame: np.ndarray) -> Dict:
        with self.lock:
            if not self.is_loaded or self.model is None:
                return self._empty_result()

            if not self.warmup_done:
                self.warmup(shape=frame.shape[:2])

            start_time = time.perf_counter()

            # YOLO26 端到端推理，无需 NMS 参数
            results = self.model(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                device=self.device,
                half=self.half
            )

            processing_time = time.perf_counter() - start_time
            current_fps = 1.0 / processing_time if processing_time > 0 else 0

            self.fps_history.append(current_fps)
            avg_fps = sum(self.fps_history) / len(self.fps_history)

            return self._parse_results(results, avg_fps)

    def _parse_results(self, results, avg_fps: float) -> Dict:
        detections = {
            "boxes": [],
            "keypoints": [],      # [[x, y, conf], ...] per person
            "keypoints_xy": [],   # [[x, y], ...] per person (不含conf)
            "confidence": [],
            "fps": avg_fps
        }

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is None:
                continue

            for i in range(len(boxes)):
                conf = float(boxes[i].conf[0].cpu().numpy())
                cls = int(boxes[i].cls[0].cpu().numpy())

                # 只处理 person 类
                if cls != 0 or conf < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = boxes[i].xyxy[0].cpu().numpy()
                detections["boxes"].append([float(x1), float(y1), float(x2), float(y2)])
                detections["confidence"].append(conf)

                if keypoints is not None and i < len(keypoints):
                    # YOLO26 使用 keypoints.xy 获取关键点坐标
                    kp_xy = keypoints[i].xy[0].cpu().numpy()   # (17, 2)
                    kp_conf = keypoints[i].conf[0].cpu().numpy() if keypoints[i].conf is not None else np.ones(17)

                    # 合并为 (17, 3): [x, y, conf]
                    kp_full = np.column_stack([kp_xy, kp_conf])
                    detections["keypoints"].append(kp_full)
                    detections["keypoints_xy"].append(kp_xy)

        return detections

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "boxes": [],
            "keypoints": [],
            "keypoints_xy": [],
            "confidence": [],
            "fps": 0
        }

    def get_primary_person(self, detections: Dict) -> Optional[Dict]:
        """获取画面中最大（最近）的人"""
        if not detections["boxes"]:
            return None

        # 选择面积最大的边界框
        best_idx = 0
        best_area = 0
        for i, box in enumerate(detections["boxes"]):
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area > best_area:
                best_area = area
                best_idx = i

        return {
            "box": detections["boxes"][best_idx],
            "keypoints": detections["keypoints"][best_idx] if best_idx < len(detections["keypoints"]) else None,
            "keypoints_xy": detections["keypoints_xy"][best_idx] if best_idx < len(detections["keypoints_xy"]) else None,
            "confidence": detections["confidence"][best_idx]
        }

    def is_model_ready(self) -> bool:
        return self.is_loaded and self.model is not None

    def get_model_info(self) -> Dict:
        if self.model is None:
            return {"loaded": False}

        return {
            "loaded": self.is_loaded,
            "model": self.model_path,
            "device": self.device,
            "precision": self.precision,
            "half_precision": self.half,
            "conf_threshold": self.conf_threshold,
            "warmup": self.warmup_done,
            "fps": sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        }
