import cv2
import numpy as np
import threading
import time
import sys
from collections import deque
from typing import Optional, Callable, Tuple, Dict


def timestamp_rate(timestamps) -> float:
    """Return events per second for ordered timestamp samples."""
    if len(timestamps) < 2:
        return 0.0
    elapsed = float(timestamps[-1]) - float(timestamps[0])
    if elapsed <= 0:
        return 0.0
    return (len(timestamps) - 1) / elapsed


class CameraCapture:
    """摄像头捕获模块
    
    支持异步捕获、自动重连、帧率统计
    """

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, flip_horizontal: bool = True, zoom: float = 0.0):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_horizontal = flip_horizontal
        self.zoom = float(zoom or 0.0)
        self.zoom_supported = False
        self.actual_zoom = 0.0
        self.backend = "Auto"
        self.fourcc = "unknown"

        self.cap = None
        self.is_open = False
        self.is_running = False
        self.current_frame = None
        self.frame_count = 0
        self.fps_timestamps = deque(maxlen=60)

        self.lock = threading.Lock()
        self._frame_condition = threading.Condition(self.lock)
        self.capture_thread = None

    def open(self) -> bool:
        try:
            self.cap = None
            if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
                try:
                    directshow_cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
                    if directshow_cap.isOpened():
                        self.cap = directshow_cap
                        self.backend = "DirectShow"
                    else:
                        directshow_cap.release()
                except Exception:
                    self.cap = None

            if self.cap is None:
                self.cap = cv2.VideoCapture(self.camera_id)
                self.backend = "Auto"

            if not self.cap.isOpened():
                print(f"Error: Cannot open camera {self.camera_id}")
                self.cap.release()
                self.cap = None
                return False

            # FOURCC must be set before size/fps: many UVC cameras only expose
            # 720p/1080p at 30fps+ in MJPG, and the driver picks the mode at
            # resolution-set time.
            original_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC) or 0)
            mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            mjpg_requested = bool(self.cap.set(cv2.CAP_PROP_FOURCC, mjpg))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if mjpg_requested and (actual_width <= 0 or actual_height <= 0):
                # Camera rejected MJPG at this size: restore its native format.
                if original_fourcc:
                    self.cap.set(cv2.CAP_PROP_FOURCC, original_fourcc)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._apply_zoom(self.zoom, log=True)

            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            fourcc_value = int(self.cap.get(cv2.CAP_PROP_FOURCC) or 0)
            self.fourcc = (
                "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
                if fourcc_value > 0 else "unknown"
            )

            print(
                f"Camera opened: {actual_width}x{actual_height} @ "
                f"{actual_fps:.0f}fps {self.fourcc} ({self.backend})"
            )

            self.is_open = True
            return True

        except Exception as e:
            print(f"Error opening camera: {e}")
            self.is_open = False
            return False

    def _apply_zoom(self, zoom: float, log: bool = False) -> bool:
        self.zoom = float(zoom or 0.0)
        if not self.cap or not self.cap.isOpened():
            return False
        if not hasattr(cv2, "CAP_PROP_ZOOM"):
            self.zoom_supported = False
            self.actual_zoom = 0.0
            if log and self.zoom > 0:
                print("Camera zoom not supported by this OpenCV build")
            return False

        if self.zoom <= 0:
            self.zoom_supported = True
            self.actual_zoom = float(self.cap.get(cv2.CAP_PROP_ZOOM) or 0.0)
            return True

        ok = bool(self.cap.set(cv2.CAP_PROP_ZOOM, self.zoom))
        actual = float(self.cap.get(cv2.CAP_PROP_ZOOM) or 0.0)
        self.actual_zoom = actual
        self.zoom_supported = (ok and abs(actual - self.zoom) <= 1e-3) or actual > 0
        if log:
            if self.zoom_supported:
                print(f"Camera hardware zoom set: requested={self.zoom:.1f}, actual={actual:.1f}")
            else:
                print(f"Camera hardware zoom unsupported or ignored: requested={self.zoom:.1f}, actual={actual:.1f}")
        return self.zoom_supported

    def set_zoom(self, zoom: float) -> bool:
        with self.lock:
            return self._apply_zoom(zoom, log=True)

    def start(self) -> bool:
        if not self.is_open:
            if not self.open():
                return False

        with self._frame_condition:
            self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        print("Camera capture started")
        return True

    def stop(self):
        with self._frame_condition:
            self.is_running = False
            self._frame_condition.notify_all()

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

            with self._frame_condition:
                self.current_frame = frame
                self.frame_count += 1
                self._frame_condition.notify_all()

            now = time.perf_counter()
            self.fps_timestamps.append(now)

    def get_frame(self) -> Optional[np.ndarray]:
        frame, _ = self.get_frame_with_count(copy=True)
        return frame

    def get_frame_with_count(self, copy: bool = True) -> Tuple[Optional[np.ndarray], int]:
        with self.lock:
            frame_count = self.frame_count
            frame = self.current_frame
        if frame is None:
            return None, frame_count
        return (frame.copy() if copy else frame), frame_count

    def wait_for_frame(
        self,
        last_frame_count: int = -1,
        timeout: float = 0.1,
        copy: bool = True,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Wait for a frame newer than ``last_frame_count``.

        Published frame arrays are never mutated by the capture thread, so the
        optional copy can safely happen after releasing the capture lock.
        """
        timeout = max(0.0, float(timeout))
        with self._frame_condition:
            self._frame_condition.wait_for(
                lambda: (
                    self.current_frame is not None
                    and self.frame_count != last_frame_count
                )
                or not self.is_running,
                timeout=timeout,
            )
            frame_count = self.frame_count
            if self.current_frame is None or frame_count == last_frame_count:
                return None, frame_count
            frame = self.current_frame

        return (frame.copy() if copy else frame), frame_count

    def get_current_fps(self) -> float:
        return timestamp_rate(self.fps_timestamps)

    def is_camera_ready(self) -> bool:
        return self.is_open and self.current_frame is not None

    def get_status(self) -> Dict:
        return {
            "open": self.is_open,
            "running": self.is_running,
            "camera_id": self.camera_id,
            "resolution": f"{self.width}x{self.height}",
            "frame_count": self.frame_count,
            "current_fps": self.get_current_fps(),
            "low_latency_buffer": True,
            "zoom": self.zoom,
            "actual_zoom": self.actual_zoom,
            "zoom_supported": self.zoom_supported,
            "backend": self.backend,
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
        frame_h, frame_w = frame.shape[:2]
        x0 = max(0, position[0] - 2)
        y0 = max(0, position[1] - text_h - 2)
        x1 = min(frame_w, position[0] + text_w + 3)
        y1 = min(frame_h, position[1] + baseline + 3)
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1]
            black = np.zeros_like(roi)
            cv2.addWeighted(black, 0.5, roi, 0.5, 0, roi)

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
