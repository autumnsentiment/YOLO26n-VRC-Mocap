"""
独立追踪模块

每个模块封装: 推理 + 推流 + 配置，可独立启停。
"""

from .base_module import BaseModule
from .body_module import BodyModule
from .face_module import FaceModule
from .hand_module import HandModule

__all__ = ["BaseModule", "BodyModule", "FaceModule", "HandModule"]
