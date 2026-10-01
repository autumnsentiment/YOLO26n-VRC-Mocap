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

协议格式 (UDP):
- 默认 JSON: source, position, quaternion, tracking_origin, actual_height, timestamp, sequence
- legacy: 7 个 double (56 字节): x, y, z, qw, qx, qy, qz
- 坐标系: VRChat (X右, Y上, Z前)
"""

import argparse
import json
import struct
import socket
import time
import sys


def _openvr_rotation_to_unity(matrix, np_module):
    rotation = np_module.asarray(matrix, dtype=float).reshape(3, 3)
    if not np_module.all(np_module.isfinite(rotation)):
        raise ValueError("OpenVR rotation matrix contains non-finite values")
    basis = np_module.diag([1.0, 1.0, -1.0])
    return basis @ rotation @ basis


def _send_pose(sock, target, protocol, source, position, quaternion, sequence, tracking_origin, actual_height=None):
    if protocol == "legacy":
        packed = struct.pack("7d", *position, *quaternion)
        sock.sendto(packed, target)
        return
    payload = {
        "source": source,
        "position": [float(position[0]), float(position[1]), float(position[2])],
        "quaternion": [float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3])],
        "tracking_origin": tracking_origin,
        "valid": True,
        "timestamp": time.time(),
        "sequence": sequence,
    }
    if actual_height is not None:
        payload["actual_height"] = float(actual_height)
    sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), target)


def _send_invalid_pose(sock, target, protocol, source, sequence, tracking_origin):
    if protocol == "legacy":
        return False
    payload = {
        "source": source,
        "tracking_origin": tracking_origin,
        "valid": False,
        "timestamp": time.time(),
        "sequence": sequence,
    }
    sock.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), target)
    return True


def send_openvr(target_host: str, target_port: int, rate: int = 100, protocol: str = "json"):
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
    print(f"Sending HMD pose to {target_host}:{target_port} (OpenVR, {protocol})")

    interval = 1.0 / rate
    sequence = 0
    while True:
        try:
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )
            hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
            if hmd_pose.bPoseIsValid:
                mat = hmd_pose.mDeviceToAbsoluteTracking
                pos_x, pos_y, pos_z = mat[0][3], mat[1][3], mat[2][3]

                # 坐标系变换: OpenVR (Z后) -> VRChat (Z前)
                position = [pos_x, pos_y, -pos_z]
                mat_rot = np.array(
                    [[float(mat[row][col]) for col in range(3)] for row in range(3)],
                    dtype=float,
                )
                mat_new = _openvr_rotation_to_unity(mat_rot, np)
                rot_new = Rot.from_matrix(mat_new)
                qx, qy, qz, qw = rot_new.as_quat()

                _send_pose(
                    sock,
                    (target_host, target_port),
                    protocol,
                    "openvr",
                    position,
                    [qw, qx, qy, qz],
                    sequence,
                    "steamvr",
                    actual_height=position[1],
                )
                sequence += 1
            elif _send_invalid_pose(
                sock,
                (target_host, target_port),
                protocol,
                "openvr",
                sequence,
                "steamvr",
            ):
                sequence += 1

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(interval)

    openvr.shutdown()
    sock.close()


def send_vrchat_osc(
    target_host: str,
    target_port: int,
    listen_port: int = 9001,
    rate: int = 100,
    protocol: str = "json",
    source_timeout: float = 2.0,
):
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
    last_position_time = 0.0

    def on_position(addr, *args):
        nonlocal current_pos, pos_valid, last_position_time
        if len(args) >= 3:
            current_pos = [float(args[0]), float(args[1]), float(args[2])]
            pos_valid = True
            last_position_time = time.monotonic()

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
    print(f"Forwarding HMD pose to {target_host}:{target_port} ({protocol})")

    interval = 1.0 / rate
    sequence = 0
    while True:
        try:
            source_fresh = pos_valid and (
                time.monotonic() - last_position_time <= max(0.1, source_timeout)
            )
            if source_fresh:
                # VRChat OSC 位置已是 VRChat 坐标系，直接转发
                # 旋转转四元数
                try:
                    from scipy.spatial.transform import Rotation as Rot
                    rot = Rot.from_euler('xyz', current_rot, degrees=True)
                    qx, qy, qz, qw = rot.as_quat()
                except ImportError:
                    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

                _send_pose(
                    sock,
                    (target_host, target_port),
                    protocol,
                    "vrchat_osc",
                    current_pos,
                    [qw, qx, qy, qz],
                    sequence,
                    "vrchat_osc",
                    actual_height=None,
                )
                sequence += 1
            else:
                pos_valid = False
                if _send_invalid_pose(
                    sock,
                    (target_host, target_port),
                    protocol,
                    "vrchat_osc",
                    sequence,
                    "vrchat_osc",
                ):
                    sequence += 1
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
    parser.add_argument("--protocol", type=str, default="json", choices=["json", "legacy"],
                        help="UDP packet protocol (default: json; legacy sends 7 doubles)")
    parser.add_argument("--source-timeout", type=float, default=2.0,
                        help="Seconds before stale source data is marked invalid (default: 2.0)")
    args = parser.parse_args()

    if args.source == "openvr":
        send_openvr(args.target, args.port, args.rate, args.protocol)
    elif args.source == "vrchat_osc":
        send_vrchat_osc(
            args.target,
            args.port,
            args.osc_port,
            args.rate,
            args.protocol,
            args.source_timeout,
        )


if __name__ == "__main__":
    main()
