#!/usr/bin/env python3
"""
生成 YOLO26n 姿态估计示例数据集

生成包含合成人体图像和对应标签的完整数据集，可直接用于训练测试。
每张图像包含一个或多个彩色火柴人，标签为 COCO 17 关键点格式。
"""

import os
import cv2
import yaml
import numpy as np
import random
from pathlib import Path


# COCO 17 关键点连接关系（用于绘制骨架）
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 头部
    (5, 6),                                 # 肩膀
    (5, 7), (7, 9),                         # 左臂
    (6, 8), (8, 10),                        # 右臂
    (5, 11), (6, 12), (11, 12),            # 躯干
    (11, 13), (13, 15),                     # 左腿
    (12, 14), (14, 16),                     # 右腿
]

# 关键点名称
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def generate_stick_person(img_w=640, img_h=640, pose_type="standing"):
    """生成一个火柴人的关键点坐标（归一化 0-1）

    Args:
        img_w, img_h: 图像尺寸
        pose_type: 姿势类型

    Returns:
        keypoints: [(x, y, v), ...] 17个关键点，归一化坐标
        bbox: [cx, cy, w, h] 归一化边界框
    """
    # 随机位置偏移
    cx = random.uniform(0.3, 0.7)
    cy = random.uniform(0.35, 0.65)
    scale = random.uniform(0.6, 1.0)

    # 基础骨架（归一化坐标，相对于中心点）
    if pose_type == "standing":
        skeleton = {
            0:  (0.00, -0.40),   # nose
            1:  (-0.03, -0.43),  # left_eye
            2:  (0.03, -0.43),   # right_eye
            3:  (-0.06, -0.40),  # left_ear
            4:  (0.06, -0.40),   # right_ear
            5:  (-0.12, -0.25),  # left_shoulder
            6:  (0.12, -0.25),   # right_shoulder
            7:  (-0.20, -0.10),  # left_elbow
            8:  (0.20, -0.10),   # right_elbow
            9:  (-0.25, 0.02),   # left_wrist
            10: (0.25, 0.02),    # right_wrist
            11: (-0.08, 0.05),   # left_hip
            12: (0.08, 0.05),    # right_hip
            13: (-0.10, 0.22),   # left_knee
            14: (0.10, 0.22),    # right_knee
            15: (-0.12, 0.40),   # left_ankle
            16: (0.12, 0.40),    # right_ankle
        }
    elif pose_type == "walking":
        skeleton = {
            0:  (0.00, -0.40),
            1:  (-0.03, -0.43),
            2:  (0.03, -0.43),
            3:  (-0.06, -0.40),
            4:  (0.06, -0.40),
            5:  (-0.10, -0.25),
            6:  (0.14, -0.25),
            7:  (-0.18, -0.12),
            8:  (0.22, -0.08),
            9:  (-0.12, -0.02),
            10: (0.28, 0.05),
            11: (-0.06, 0.05),
            12: (0.10, 0.05),
            13: (-0.14, 0.22),
            14: (0.06, 0.22),
            15: (-0.08, 0.40),
            16: (0.14, 0.38),
        }
    elif pose_type == "arms_up":
        skeleton = {
            0:  (0.00, -0.40),
            1:  (-0.03, -0.43),
            2:  (0.03, -0.43),
            3:  (-0.06, -0.40),
            4:  (0.06, -0.40),
            5:  (-0.12, -0.25),
            6:  (0.12, -0.25),
            7:  (-0.18, -0.38),
            8:  (0.18, -0.38),
            9:  (-0.14, -0.48),
            10: (0.14, -0.48),
            11: (-0.08, 0.05),
            12: (0.08, 0.05),
            13: (-0.10, 0.22),
            14: (0.10, 0.22),
            15: (-0.12, 0.40),
            16: (0.12, 0.40),
        }
    elif pose_type == "squatting":
        skeleton = {
            0:  (0.00, -0.22),
            1:  (-0.03, -0.25),
            2:  (0.03, -0.25),
            3:  (-0.06, -0.22),
            4:  (0.06, -0.22),
            5:  (-0.14, -0.10),
            6:  (0.14, -0.10),
            7:  (-0.22, -0.18),
            8:  (0.22, -0.18),
            9:  (-0.28, -0.08),
            10: (0.28, -0.08),
            11: (-0.10, 0.05),
            12: (0.10, 0.05),
            13: (-0.16, 0.12),
            14: (0.16, 0.12),
            15: (-0.10, 0.25),
            16: (0.10, 0.25),
        }
    else:  # t-pose
        skeleton = {
            0:  (0.00, -0.40),
            1:  (-0.03, -0.43),
            2:  (0.03, -0.43),
            3:  (-0.06, -0.40),
            4:  (0.06, -0.40),
            5:  (-0.12, -0.25),
            6:  (0.12, -0.25),
            7:  (-0.25, -0.25),
            8:  (0.25, -0.25),
            9:  (-0.35, -0.25),
            10: (0.35, -0.25),
            11: (-0.08, 0.05),
            12: (0.08, 0.05),
            13: (-0.10, 0.22),
            14: (0.10, 0.22),
            15: (-0.12, 0.40),
            16: (0.12, 0.40),
        }

    # 应用缩放和位置偏移，添加随机扰动
    keypoints = []
    for i in range(17):
        dx, dy = skeleton[i]
        # 随机扰动
        dx += random.gauss(0, 0.01)
        dy += random.gauss(0, 0.01)
        # 缩放 + 偏移
        x = cx + dx * scale
        y = cy + dy * scale
        # 裁剪到图像内
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        # 可见性: 2=可见, 偶尔设为1(遮挡)
        v = 2 if random.random() > 0.05 else 1
        keypoints.append((x, y, v))

    # 计算边界框
    xs = [kp[0] for kp in keypoints]
    ys = [kp[1] for kp in keypoints]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # 扩展边界框
    pad_x = (x_max - x_min) * 0.1
    pad_y = (y_max - y_min) * 0.1
    x_min = max(0, x_min - pad_x)
    x_max = min(1, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(1, y_max + pad_y)

    bbox_cx = (x_min + x_max) / 2
    bbox_cy = (y_min + y_max) / 2
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    return keypoints, [bbox_cx, bbox_cy, bbox_w, bbox_h]


def draw_stick_figure(img, keypoints, color=(0, 200, 0), thickness=2):
    """在图像上绘制火柴人"""
    h, w = img.shape[:2]

    # 绘制骨架连线
    for i, j in COCO_SKELETON:
        x1, y1, v1 = keypoints[i]
        x2, y2, v2 = keypoints[j]
        if v1 > 0 and v2 > 0:
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    # 绘制关键点
    for x, y, v in keypoints:
        if v > 0:
            pt = (int(x * w), int(y * h))
            cv2.circle(img, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)

    return img


def generate_image(img_w=640, img_h=640, num_persons=1):
    """生成一张合成图像

    Returns:
        img: 合成图像
        labels: 标签列表，每个元素为 [class, cx, cy, w, h, kp1_x, kp1_y, kp1_v, ...]
    """
    # 随机背景
    bg_color = random.choice([
        (40, 40, 40),    # 深灰
        (60, 60, 80),    # 深蓝灰
        (80, 70, 60),    # 暖灰
        (50, 60, 50),    # 深绿灰
        (30, 30, 50),    # 暗蓝
    ])
    img = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)

    # 添加随机噪点
    noise = np.random.randint(0, 20, (img_h, img_w, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    # 添加一些随机矩形作为背景装饰
    for _ in range(random.randint(2, 5)):
        rx = random.randint(0, img_w)
        ry = random.randint(0, img_h)
        rw = random.randint(30, 150)
        rh = random.randint(30, 150)
        rc = tuple(random.randint(20, 80) for _ in range(3))
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), rc, -1)

    pose_types = ["standing", "walking", "arms_up", "squatting", "t_pose"]
    labels = []

    for _ in range(num_persons):
        pose_type = random.choice(pose_types)
        keypoints, bbox = generate_stick_person(img_w, img_h, pose_type)

        # 随机颜色
        color = tuple(random.randint(100, 255) for _ in range(3))
        draw_stick_figure(img, keypoints, color=color)

        # 构建标签: class cx cy w h kp1_x kp1_y kp1_v ...
        label = [0]  # class 0 = person
        label.extend(bbox)  # cx cy w h
        for x, y, v in keypoints:
            label.extend([x, y, v])

        labels.append(label)

    return img, labels


def generate_dataset(output_dir, num_train=50, num_val=15, img_w=640, img_h=640):
    """生成完整数据集

    Args:
        output_dir: 输出目录
        num_train: 训练集图像数量
        num_val: 验证集图像数量
        img_w, img_h: 图像尺寸
    """
    base = Path(output_dir)

    # 创建目录
    dirs = [
        base / "images" / "train",
        base / "images" / "val",
        base / "labels" / "train",
        base / "labels" / "val",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    pose_types = ["standing", "walking", "arms_up", "squatting", "t_pose"]

    # 生成训练集
    print(f"Generating {num_train} training images...")
    for i in range(num_train):
        num_persons = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        img, labels = generate_image(img_w, img_h, num_persons)

        img_name = f"train_{i:04d}.jpg"
        cv2.imwrite(str(base / "images" / "train" / img_name), img)

        label_name = f"train_{i:04d}.txt"
        with open(base / "labels" / "train" / label_name, 'w') as f:
            for label in labels:
                f.write(" ".join(f"{v:.6f}" for v in label) + "\n")

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{num_train}")

    # 生成验证集
    print(f"Generating {num_val} validation images...")
    for i in range(num_val):
        num_persons = random.choices([1, 2], weights=[0.7, 0.3])[0]
        img, labels = generate_image(img_w, img_h, num_persons)

        img_name = f"val_{i:04d}.jpg"
        cv2.imwrite(str(base / "images" / "val" / img_name), img)

        label_name = f"val_{i:04d}.txt"
        with open(base / "labels" / "val" / label_name, 'w') as f:
            for label in labels:
                f.write(" ".join(f"{v:.6f}" for v in label) + "\n")

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{num_val}")

    # 生成数据集配置
    config = {
        "path": str(base.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "person"},
        "kpt_shape": [17, 3],  # 17 keypoints, each with (x, y, visibility)
    }

    config_path = base / "dataset.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\nDataset generated successfully!")
    print(f"  Location: {base}")
    print(f"  Train: {num_train} images")
    print(f"  Val: {num_val} images")
    print(f"  Config: {config_path}")
    print(f"\nTo train:")
    print(f"  python bin\\train.py --data {config_path} --epochs 10 --batch 4")
    print(f"  python bin\\train.py --data {config_path} --epochs 10 --batch 4 --freeze 10")

    # 保存一张预览图
    preview_img, preview_labels = generate_image(img_w, img_h, 2)
    # 绘制标签信息
    cv2.putText(preview_img, "Sample Dataset Preview", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(preview_img, f"Train: {num_train}  Val: {num_val}", (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imwrite(str(base / "preview.jpg"), preview_img)
    print(f"  Preview: {base / 'preview.jpg'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample pose dataset")
    parser.add_argument("--output", type=str, default="./sample_dataset",
                       help="Output directory")
    parser.add_argument("--train", type=int, default=50,
                       help="Number of training images")
    parser.add_argument("--val", type=int, default=15,
                       help="Number of validation images")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=640, help="Image height")

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    generate_dataset(args.output, args.train, args.val, args.width, args.height)
