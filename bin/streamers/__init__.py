"""
推流模块包

各推理模块独立推流:
- BodyOSCStreamer: 身体姿态 OSC 推流 (VRChat OSC 追踪协议)
- BodySteamVRStreamer: 身体姿态 SteamVR 推流 (命名管道驱动)
- HandVMTStreamer: 手部追踪 VMT 推流 (OpenVR 驱动层)
- FaceVMCStreamer: 面部表情 VMC 推流 (Virtual Motion Capture 协议)
- StreamManager: 统一管理推流模块，独立开关控制
"""

from .base import BaseStreamer
from .osc_transport import OSCTransport
from .coord import CoordinateTransformer
from .body_osc import BodyOSCStreamer
from .body_steamvr import BodySteamVRStreamer
from .hand_vmt import HandVMTStreamer
from .face_vmc import FaceVMCStreamer
from .manager import StreamManager

__all__ = [
    "BaseStreamer",
    "OSCTransport",
    "CoordinateTransformer",
    "BodyOSCStreamer",
    "BodySteamVRStreamer",
    "HandVMTStreamer",
    "FaceVMCStreamer",
    "StreamManager",
]
