"""
推流模块基类

所有推流模块继承此基类，实现独立的 connect/disconnect/send 接口。
每个推流模块拥有独立的开关，控制推理与推流。
"""

import threading
import time
from typing import Dict, Optional
from abc import ABC, abstractmethod
from queue import Queue, Empty, Full


class BaseStreamer(ABC):
    """推流模块基类

    子类需实现:
    - _do_connect() -> bool: 建立连接
    - _do_disconnect(): 断开连接
    - _do_send(pose_data: Dict): 发送数据

    属性:
    - enabled: 推流开关 (独立于推理开关)
    - connected: 连接状态
    """

    def __init__(self, name: str, host: str = "127.0.0.1", port: int = 9000):
        self.name = name
        self.host = host
        self.port = port

        # 推流开关独立于推理开关，由各运行时模块管理。
        self._enabled = True
        # 推理开关 (控制是否接收推理数据)
        self._inference_enabled = True

        self._connected = False
        self._running = False
        # Realtime pose output must not build latency. Keep only the newest
        # pending pose while the sender thread is busy.
        self._send_queue = Queue(maxsize=1)
        self._queue_lock = threading.Lock()
        self._connection_lock = threading.RLock()
        self._send_thread = None

        # HMD 位姿数据提供者 (用于坐标对齐)
        self._hmd_provider = None

        # 统计
        self._send_count = 0
        self._error_count = 0
        self._drop_count = 0
        self._fps = 0
        self._fps_counter = 0
        self._fps_start_time = time.time()

    @property
    def enabled(self) -> bool:
        """推流是否启用"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def inference_enabled(self) -> bool:
        """推理是否启用"""
        return self._inference_enabled

    @inference_enabled.setter
    def inference_enabled(self, value: bool):
        self._inference_enabled = value

    @property
    def connected(self) -> bool:
        return self._connected

    def set_hmd_provider(self, provider):
        """设置 HMD 位姿数据提供者"""
        self._hmd_provider = provider

    def connect(self) -> bool:
        """建立连接并启动发送线程"""
        with self._connection_lock:
            if self._connected:
                return True
            self._clear_send_queue()
            try:
                if self._do_connect():
                    self._connected = True
                    self._running = True
                    self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
                    self._send_thread.start()
                    return True
            except Exception as e:
                print(f"[{self.name}] Connect error: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        with self._connection_lock:
            self._running = False
            self._connected = False
            if self._send_thread:
                if self._send_thread is not threading.current_thread():
                    self._send_thread.join(timeout=2.0)
                self._send_thread = None
            self._clear_send_queue()
            try:
                self._do_disconnect()
            except Exception:
                pass

    def send(self, pose_data: Dict) -> bool:
        """将数据放入发送队列

        即使推流未启用，也接受数据 (由 _send_loop 判断是否真正发送)。
        推理模块可以始终调用 send()，无需关心推流开关状态。
        """
        if not self._inference_enabled:
            return False
        with self._queue_lock:
            try:
                stale_pose = self._send_queue.get_nowait()
            except Empty:
                stale_pose = None
            if stale_pose is not None:
                self._drop_count += 1
            try:
                self._send_queue.put_nowait(pose_data)
                return True
            except Full:
                # Producers are serialized, so this is defensive. Treat an
                # unexpected contention as a rejected realtime pose.
                self._drop_count += 1
                return False

    def _clear_send_queue(self) -> int:
        dropped = 0
        with self._queue_lock:
            while True:
                try:
                    self._send_queue.get_nowait()
                except Empty:
                    break
                dropped += 1
            self._drop_count += dropped
        return dropped

    def _record_drop(self):
        with self._queue_lock:
            self._drop_count += 1

    def _send_loop(self):
        """发送线程主循环"""
        _consecutive_skip_count = 0
        while self._running:
            try:
                try:
                    pose_data = self._send_queue.get(timeout=0.1)
                except Empty:
                    continue

                if not self._running:
                    self._record_drop()
                    break

                # 推流开关控制: 未启用则丢弃数据
                if not self._enabled:
                    self._record_drop()
                    _consecutive_skip_count += 1
                    if _consecutive_skip_count == 1:
                        print(f"[{self.name}] WARNING: 推流已关闭，数据被丢弃 (inference_enabled={self._inference_enabled})")
                    continue

                # 未连接则尝试重连
                if not self._connected:
                    self._record_drop()
                    _consecutive_skip_count += 1
                    if _consecutive_skip_count == 1:
                        print(f"[{self.name}] WARNING: 未连接，数据被丢弃 (enabled={self._enabled}, inference_enabled={self._inference_enabled})")
                    continue

                if _consecutive_skip_count > 0:
                    print(f"[{self.name}] 恢复发送 (之前跳过 {_consecutive_skip_count} 帧)")
                    _consecutive_skip_count = 0

                try:
                    self._do_send(pose_data)
                    self._send_count += 1
                    self._fps_counter += 1
                except Exception as e:
                    self._error_count += 1
                    if self._error_count <= 3:
                        print(f"[{self.name}] Send error: {e}")

                # FPS 计算
                now = time.time()
                if now - self._fps_start_time >= 1.0:
                    self._fps = self._fps_counter
                    self._fps_counter = 0
                    self._fps_start_time = now

            except Exception:
                pass

    def get_status(self) -> Dict:
        """获取推流模块状态"""
        with self._queue_lock:
            drop_count = self._drop_count
            queue_size = self._send_queue.qsize()
        return {
            "name": self.name,
            "enabled": self._enabled,
            "inference_enabled": self._inference_enabled,
            "connected": self._connected,
            "fps": self._fps,
            "send_count": self._send_count,
            "error_count": self._error_count,
            "drop_count": drop_count,
            "queue_size": queue_size,
            "queue_capacity": self._send_queue.maxsize,
        }

    @abstractmethod
    def _do_connect(self) -> bool:
        """子类实现: 建立连接"""
        ...

    @abstractmethod
    def _do_disconnect(self):
        """子类实现: 断开连接"""
        ...

    @abstractmethod
    def _do_send(self, pose_data: Dict):
        """子类实现: 发送数据"""
        ...
