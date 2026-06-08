import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Tuple, Dict


class CameraCapture:
    """摄像头捕获模块
    
    支持异步捕获、自动重连、帧率统计
    """

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, flip_horizontal: bool = True):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_horizontal = flip_horizontal

        self.cap = None
        self.is_open = False
        self.is_running = False
        self.current_frame = None
        self.frame_count = 0
        self.fps_timestamps = []

        self.lock = threading.Lock()
        self.capture_thread = None

    def open(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                print(f"Error: Cannot open camera {self.camera_id}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.0f}fps")

            self.is_open = True
            return True

        except Exception as e:
            print(f"Error opening camera: {e}")
            self.is_open = False
            return False

    def start(self) -> bool:
        if not self.is_open:
            if not self.open():
                return False

        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print("Camera capture started")
        return True

    def stop(self):
        self.is_running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.is_open = False
        self.is_running = False
        print("Camera stopped")

    def _capture_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)

            with self.lock:
                self.current_frame = frame
                self.frame_count += 1

            now = time.perf_counter()
            self.fps_timestamps.append(now)
            if len(self.fps_timestamps) > 60:
                self.fps_timestamps.pop(0)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def get_current_fps(self) -> float:
        if len(self.fps_timestamps) < 2:
            return 0.0
        elapsed = self.fps_timestamps[-1] - self.fps_timestamps[0]
        if elapsed <= 0:
            return 0.0
        return len(self.fps_timestamps) / elapsed

    def is_camera_ready(self) -> bool:
        return self.is_open and self.current_frame is not None

    def get_status(self) -> Dict:
        return {
            "open": self.is_open,
            "running": self.is_running,
            "camera_id": self.camera_id,
            "resolution": f"{self.width}x{self.height}",
            "frame_count": self.frame_count,
            "current_fps": self.get_current_fps()
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class ImageProcessor:
    """图像处理工具"""

    # YOLO26 姿态骨骼连接（COCO 17关键点）
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),       # 面部
        (5, 6), (5, 7), (7, 9),                 # 左臂
        (6, 8), (8, 10),                         # 右臂
        (5, 11), (6, 12), (11, 12),              # 躯干
        (11, 13), (13, 15),                       # 左腿
        (12, 14), (14, 16)                        # 右腿
    ]

    # 骨骼颜色（BGR）
    SKELETON_COLORS = {
        "face": (255, 200, 200),
        "left_arm": (200, 255, 200),
        "right_arm": (200, 200, 255),
        "torso": (255, 255, 200),
        "left_leg": (200, 255, 255),
        "right_leg": (255, 200, 255),
    }

    CONNECTION_GROUPS = {
        "face": [(0, 1), (0, 2), (1, 3), (2, 4)],
        "left_arm": [(5, 7), (7, 9)],
        "right_arm": [(6, 8), (8, 10)],
        "torso": [(5, 6), (5, 11), (6, 12), (11, 12)],
        "left_leg": [(11, 13), (13, 15)],
        "right_leg": [(12, 14), (14, 16)],
    }

    @staticmethod
    def add_text_overlay(frame: np.ndarray, text: str,
                        position: Tuple[int, int] = (10, 30),
                        font_scale: float = 0.6,
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 1) -> np.ndarray:
        # 半透明背景
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (position[0] - 2, position[1] - text_h - 2),
                      (position[0] + text_w + 2, position[1] + baseline + 2),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray,
                     connections: list = None) -> np.ndarray:
        """绘制彩色骨骼"""
        if keypoints is None or len(keypoints) == 0:
            return frame

        ip = ImageProcessor

        for group_name, conns in ip.CONNECTION_GROUPS.items():
            color = ip.SKELETON_COLORS.get(group_name, (0, 255, 0))
            for i, j in conns:
                if i < len(keypoints) and j < len(keypoints):
                    pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                    pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))

                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        # 绘制关键点
        for i in range(len(keypoints)):
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0 and y > 0:
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

        return frame

    @staticmethod
    def add_fps_overlay(frame: np.ndarray, fps: float,
                        position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        text = f"FPS: {fps:.1f}"
        return ImageProcessor.add_text_overlay(frame, text, position)

    @staticmethod
    def add_status_bar(frame: np.ndarray, status_items: Dict[str, str],
                       start_y: int = 30) -> np.ndarray:
        y = start_y
        for key, value in status_items.items():
            text = f"{key}: {value}"
            ImageProcessor.add_text_overlay(frame, text, (10, y))
            y += 22
        return frame
