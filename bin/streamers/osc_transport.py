"""
OSC 传输工具

提供 OSC 消息构建和发送的通用功能，供各推流模块共享。
"""

import socket
import struct
from typing import Dict, Optional, Tuple


class OSCTransport:
    """OSC 传输层

    提供:
    - OSC 消息构建 (字符串编码、浮点数编码、整数编码)
    - UDP socket 管理
    - 批量发送优化
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None

    def connect(self) -> bool:
        """创建 UDP socket"""
        try:
            if self._socket:
                self._socket.close()
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.settimeout(0.5)
            return True
        except Exception as e:
            print(f"[OSC] Connect error: {e}")
            return False

    def disconnect(self):
        """关闭 UDP socket"""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    @property
    def socket(self) -> Optional[socket.socket]:
        return self._socket

    def send(self, message: bytes, port: Optional[int] = None):
        """发送 OSC 消息"""
        if self._socket is None:
            return
        target_port = port or self.port
        self._socket.sendto(message, (self.host, target_port))

    def send_batch(self, messages: list, port: Optional[int] = None):
        """批量发送 OSC 消息 (减少系统调用)"""
        if self._socket is None:
            return
        target_port = port or self.port
        dest = (self.host, target_port)
        sendto = self._socket.sendto
        for msg in messages:
            sendto(msg, dest)

    # ========== OSC 消息编码 ==========

    @staticmethod
    def encode_string(s: str) -> bytes:
        """编码 OSC 字符串 (4字节对齐)"""
        encoded = s.encode("utf-8") + b'\x00'
        padding = (4 - len(encoded) % 4) % 4
        return encoded + b'\x00' * padding

    @staticmethod
    def encode_float(value: float) -> bytes:
        """编码 OSC 浮点数"""
        return struct.pack("!f", value)

    @staticmethod
    def encode_int(value: int) -> bytes:
        """编码 OSC 整数"""
        return struct.pack("!i", value)

    @classmethod
    def build_message(cls, address: str, *args) -> bytes:
        """构建 OSC 消息

        Args:
            address: OSC 地址 (如 /tracking/trackers/1/position)
            *args: 参数 (float/int/str)

        Returns:
            编码后的 OSC 消息字节
        """
        type_tags = ","
        data = b""

        for arg in args:
            if isinstance(arg, float):
                type_tags += "f"
                data += cls.encode_float(arg)
            elif isinstance(arg, int):
                type_tags += "i"
                data += cls.encode_int(arg)
            elif isinstance(arg, str):
                type_tags += "s"
                data += cls.encode_string(arg)

        message = cls.encode_string(address)
        message += cls.encode_string(type_tags)
        message += data

        return message
