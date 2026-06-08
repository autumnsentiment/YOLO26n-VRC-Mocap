#!/usr/bin/env python3
"""测试 OSC 数据推送到 VRChat"""

import sys
import os
import socket
import struct
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bin'))

def encode_osc_string(s):
    encoded = s.encode("utf-8") + b'\x00'
    padding = (4 - len(encoded) % 4) % 4
    return encoded + b'\x00' * padding

def encode_osc_float(value):
    return struct.pack("!f", value)

def build_osc_message(address, *args):
    type_tags = ","
    data = b""
    for arg in args:
        if isinstance(arg, float):
            type_tags += "f"
            data += encode_osc_float(arg)
        elif isinstance(arg, int):
            type_tags += "i"
            data += struct.pack("!i", arg)
        elif isinstance(arg, str):
            type_tags += "s"
            data += encode_osc_string(arg)
    message = encode_osc_string(address)
    message += encode_osc_string(type_tags)
    message += data
    return message

def main():
    host = "127.0.0.1"
    port = 9000
    
    print("=" * 60)
    print("  VRChat OSC 数据推送测试")
    print("=" * 60)
    print(f"\n目标地址: {host}:{port}")
    print("请确保 VRChat 已启动并开启了 OSC 功能")
    print("  (Settings -> OSC -> Enable)")
    
    # 创建 UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print("\n正在发送测试数据...")
    
    # 模拟的骨骼数据 (欧拉角 -> 四元数)
    from scipy.spatial.transform import Rotation as R
    import math
    
    test_bones = {
        "hip": {"euler": [0.0, 0.1, 0.0], "pos": [0.0, 0.9, 0.0]},
        "chest": {"euler": [0.0, 0.05, 0.0], "pos": [0.0, 1.1, 0.0]},
        "head": {"euler": [0.1, 0.0, 0.0], "pos": [0.0, 1.5, 0.0]},
        "left_foot": {"euler": [0.0, 0.0, 0.0], "pos": [-0.1, 0.1, 0.0]},
        "right_foot": {"euler": [0.0, 0.0, 0.0], "pos": [0.1, 0.1, 0.0]},
    }
    
    sent_count = 0
    error_count = 0
    
    # 发送50帧数据 (约1.7秒 @ 30fps)
    for frame in range(50):
        t = frame / 30.0
        
        for tracker_name, data in test_bones.items():
            # 动态旋转: 模拟轻微摆动
            euler = [
                data["euler"][0] + math.sin(t * 2) * 0.1,
                data["euler"][1] + math.sin(t * 3) * 0.1,
                data["euler"][2] + math.cos(t * 2) * 0.05
            ]
            
            # 转换为四元数
            r = R.from_euler("xyz", euler, degrees=False)
            quat = r.as_quat()  # [x, y, z, w]
            
            # 发送旋转
            rot_addr = f"/tracking/trackers/{tracker_name}/rotation"
            rot_msg = build_osc_message(
                rot_addr, 
                float(quat[0]), float(quat[1]), 
                float(quat[2]), float(quat[3])
            )
            
            try:
                sock.sendto(rot_msg, (host, port))
                sent_count += 1
            except Exception as e:
                error_count += 1
                print(f"  发送旋转失败 {tracker_name}: {e}")
            
            # 发送位置
            pos = data["pos"]
            pos_addr = f"/tracking/trackers/{tracker_name}/position"
            pos_msg = build_osc_message(
                pos_addr,
                float(pos[0]), float(pos[1]), float(pos[2])
            )
            
            try:
                sock.sendto(pos_msg, (host, port))
                sent_count += 1
            except Exception as e:
                error_count += 1
                print(f"  发送位置失败 {tracker_name}: {e}")
        
        time.sleep(1.0 / 30.0)  # 30fps
    
    sock.close()
    
    print("\n" + "=" * 60)
    print(f"测试完成!")
    print(f"  发送成功: {sent_count} 条消息")
    print(f"  发送失败: {error_count} 条消息")
    print("\n请在 VRChat 中检查:")
    print("  1. 打开 Avatar 动作是否响应")
    print("  2. 查看 VRChat 控制台是否有 OSC 数据接收日志")
    print("  3. 在 Quick Menu -> Options -> OSC 查看状态")
    print("=" * 60)

if __name__ == "__main__":
    main()
