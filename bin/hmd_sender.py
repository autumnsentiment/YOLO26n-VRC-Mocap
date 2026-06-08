#!/usr/bin/env python3
"""
HMD 位姿数据远程发送端

在 VR 头显连接的机器上运行，将 HMD 位姿数据通过 UDP 发送到远程动捕服务器。
配合 RemoteHMDProvider 使用。

使用方式:
1. 在头显所在机器运行: python hmd_sender.py --target <动捕服务器IP> [--port 4243]
2. 在动捕服务器配置: hmd.provider = "remote", hmd.remote_port = 4243

支持的数据源:
- openvr: 通过 SteamVR 获取 (默认)
- vrchat_osc: 通过 VRChat OSC 反馈获取

协议格式 (UDP, 二进制):
- 7 个 double (56 字节): x, y, z, qw, qx, qy, qz
- 坐标系: VRChat (X右, Y上, Z前)
"""

import argparse
import struct
import socket
import time
import sys


def send_openvr(target_host: str, target_port: int, rate: int = 100):
    """通过 OpenVR 获取 HMD 位姿并发送"""
    try:
        import openvr
    except ImportError:
        print("Error: openvr not installed. Run: pip install openvr")
        return

    try:
        import numpy as np
        from scipy.spatial.transform import Rotation as Rot
    except ImportError:
        print("Error: numpy/scipy not installed. Run: pip install numpy scipy")
        return

    openvr.init(openvr.VRApplication_Other)
    vr_system = openvr.VRSystem()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Sending HMD pose to {target_host}:{target_port} (OpenVR)")

    interval = 1.0 / rate
    while True:
        try:
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )
            hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
            if hmd_pose.bPoseIsValid:
                mat = hmd_pose.mDeviceToAbsoluteTracking
                pos_x, pos_y, pos_z = mat[0][3], mat[1][3], mat[2][3]

                # 从矩阵提取四元数
                import math
                r_w = math.sqrt(abs(1 + mat[0][0] + mat[1][1] + mat[2][2])) / 2
                if r_w > 1e-6:
                    r_x = (mat[2][1] - mat[1][2]) / (4 * r_w)
                    r_y = (mat[0][2] - mat[2][0]) / (4 * r_w)
                    r_z = (mat[1][0] - mat[0][1]) / (4 * r_w)
                else:
                    r_x, r_y, r_z = 0.0, 0.0, 0.0

                # 坐标系变换: OpenVR (Z后) -> VRChat (Z前)
                position = [pos_x, pos_y, -pos_z]
                rot = Rot.from_quat([r_x, r_y, r_z, r_w])
                mat_rot = rot.as_matrix()
                flip_y = np.diag([-1.0, 1.0, -1.0])
                mat_new = flip_y @ mat_rot
                rot_new = Rot.from_matrix(mat_new)
                qx, qy, qz, qw = rot_new.as_quat()

                # 打包发送: x, y, z, qw, qx, qy, qz
                packed = struct.pack("7d", *position, qw, qx, qy, qz)
                sock.sendto(packed, (target_host, target_port))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(interval)

    openvr.shutdown()
    sock.close()


def send_vrchat_osc(target_host: str, target_port: int, listen_port: int = 9001, rate: int = 100):
    """通过 VRChat OSC 反馈获取 HMD 位姿并转发"""
    try:
        from pythonosc import dispatcher as osc_dispatcher
        from pythonosc import osc_server
    except ImportError:
        print("Error: python-osc not installed. Run: pip install python-osc")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    current_pos = [0.0, 0.0, 0.0]
    current_rot = [0.0, 0.0, 0.0]
    pos_valid = False
    rot_valid = False

    def on_position(addr, *args):
        nonlocal current_pos, pos_valid
        if len(args) >= 3:
            current_pos = [float(args[0]), float(args[1]), float(args[2])]
            pos_valid = True

    def on_rotation(addr, *args):
        nonlocal current_rot, rot_valid
        if len(args) >= 3:
            current_rot = [float(args[0]), float(args[1]), float(args[2])]
            rot_valid = True

    disp = osc_dispatcher.Dispatcher()
    disp.map("/tracking/trackers/head/position", on_position)
    disp.map("/tracking/trackers/head/rotation", on_rotation)

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", listen_port), disp)
    server_thread = __import__('threading').Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"Listening VRChat OSC on port {listen_port}")
    print(f"Forwarding HMD pose to {target_host}:{target_port}")

    interval = 1.0 / rate
    while True:
        try:
            if pos_valid:
                # VRChat OSC 位置已是 VRChat 坐标系，直接转发
                # 旋转转四元数
                try:
                    from scipy.spatial.transform import Rotation as Rot
                    rot = Rot.from_euler('xyz', current_rot, degrees=True)
                    qx, qy, qz, qw = rot.as_quat()
                except ImportError:
                    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

                packed = struct.pack("7d", *current_pos, qw, qx, qy, qz)
                sock.sendto(packed, (target_host, target_port))
        except KeyboardInterrupt:
            break
        except Exception:
            pass
        time.sleep(interval)

    server.shutdown()
    sock.close()


def main():
    parser = argparse.ArgumentParser(description="HMD Pose Remote Sender")
    parser.add_argument("--target", type=str, required=True, help="Target server IP")
    parser.add_argument("--port", type=int, default=4243, help="Target UDP port (default: 4243)")
    parser.add_argument("--source", type=str, default="openvr", choices=["openvr", "vrchat_osc"],
                        help="HMD data source (default: openvr)")
    parser.add_argument("--osc-port", type=int, default=9001, help="VRChat OSC listen port (default: 9001)")
    parser.add_argument("--rate", type=int, default=100, help="Send rate in Hz (default: 100)")
    args = parser.parse_args()

    if args.source == "openvr":
        send_openvr(args.target, args.port, args.rate)
    elif args.source == "vrchat_osc":
        send_vrchat_osc(args.target, args.port, args.osc_port, args.rate)


if __name__ == "__main__":
    main()
