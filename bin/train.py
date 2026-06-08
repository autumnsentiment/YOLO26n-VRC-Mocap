#!/usr/bin/env python3
"""
YOLO26n 姿态估计模型训练脚本

支持:
- 自定义数据集训练/微调
- 从 COCO/自定义格式数据集训练
- 姿态估计（pose）模型训练
- 断点续训
- 训练过程可视化
- 自动导出推理模型

数据集格式要求 (Ultralytics 格式):
  dataset/
  ├── images/
  │   ├── train/
  │   │   ├── img001.jpg
  │   │   └── ...
  │   └── val/
  │       ├── img101.jpg
  │       └── ...
  ├── labels/
  │   ├── train/
  │   │   ├── img001.txt    (每行: class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...)
  │   │   └── ...
  │   └── val/
  │       ├── img101.txt
  │       └── ...
  └── dataset.yaml           (数据集配置文件)

标签格式 (姿态估计):
  class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ... kp17_x kp17_y kp17_v
  - class: 类别ID (0=person)
  - cx, cy, w, h: 边界框 (归一化中心坐标+宽高)
  - kp_x, kp_y: 关键点坐标 (归一化)
  - kp_v: 关键点可见性 (0=不可见, 1=遮挡, 2=可见)
"""

import os
import sys
import argparse
import shutil
import yaml
import json
from pathlib import Path
from datetime import datetime


# Ultralytics 配置目录
_ultra_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ultralytics")
os.makedirs(_ultra_cfg, exist_ok=True)
os.environ["ULTRALYTICS_SETTINGS"] = _ultra_cfg


class DatasetValidator:
    """数据集验证器"""

    @staticmethod
    def validate_dataset(data_yaml: str) -> bool:
        """验证数据集配置和数据完整性"""
        yaml_path = Path(data_yaml)

        if not yaml_path.exists():
            print(f"Error: Dataset config not found: {data_yaml}")
            return False

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error: Invalid YAML in dataset config: {e}")
            return False

        if config is None:
            print("Error: Dataset config is empty")
            return False

        # 检查必要字段
        required_fields = ['path', 'train', 'val', 'names']
        for field in required_fields:
            if field not in config:
                print(f"Error: Missing required field '{field}' in dataset config")
                return False

        # 检查路径
        dataset_root = Path(config['path'])
        if not dataset_root.is_absolute():
            dataset_root = yaml_path.parent / dataset_root

        if not dataset_root.exists():
            print(f"Error: Dataset path not found: {dataset_root}")
            return False

        # 统计信息
        total_train_img = 0
        total_val_img = 0
        total_train_lbl = 0
        total_val_lbl = 0
        has_issues = False

        # 检查训练/验证图像目录
        for split in ['train', 'val']:
            split_path = dataset_root / config[split]
            if not split_path.exists():
                print(f"Error: {split} path not found: {split_path}")
                return False

            # 统计图像数量
            img_dir = split_path if split_path.is_dir() else dataset_root / "images" / split
            if img_dir.exists():
                img_count = len(list(img_dir.glob("*.*")))
                if split == 'train':
                    total_train_img = img_count
                else:
                    total_val_img = img_count
                print(f"  {split}: {img_count} images")
                if img_count == 0:
                    print(f"  Warning: {split} split has no images!")
                    has_issues = True

        # 检查标签
        for split in ['train', 'val']:
            label_dir = dataset_root / "labels" / split
            if label_dir.exists():
                label_count = len(list(label_dir.glob("*.txt")))
                if split == 'train':
                    total_train_lbl = label_count
                else:
                    total_val_lbl = label_count
                print(f"  {split} labels: {label_count} files")

                if label_count == 0:
                    print(f"  Warning: {split} split has no labels!")
                    has_issues = True

                # 抽样检查标签格式
                sample_files = list(label_dir.glob("*.txt"))[:3]
                for lf in sample_files:
                    try:
                        with open(lf, 'r', encoding='utf-8') as f:
                            first_line = f.readline().strip()
                        if first_line and not first_line.startswith('#'):
                            parts = first_line.split()
                            # pose 格式: class cx cy w h + 17*3 关键点 = 5+51=56
                            if len(parts) not in [5, 56]:
                                print(f"  Warning: {lf.name} has {len(parts)} values "
                                      f"(expected 5 for detect or 56 for pose)")
                    except OSError as e:
                        print(f"  Warning: Cannot read {lf.name}: {e}")

        # 检查图像-标签匹配
        for split in ['train', 'val']:
            img_dir = dataset_root / "images" / split
            label_dir = dataset_root / "labels" / split
            if img_dir.exists() and label_dir.exists():
                img_stems = {f.stem for f in img_dir.glob("*.*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']}
                lbl_stems = {f.stem for f in label_dir.glob("*.txt")}
                missing_labels = img_stems - lbl_stems
                missing_images = lbl_stems - img_stems
                if missing_labels:
                    print(f"  Warning: {split} - {len(missing_labels)} images missing labels")
                    has_issues = True
                if missing_images:
                    print(f"  Warning: {split} - {len(missing_images)} labels missing images")
                    has_issues = True

        # 检查训练/验证比例
        if total_train_img > 0 and total_val_img > 0:
            ratio = total_train_img / (total_train_img + total_val_img)
            print(f"  Train/Val ratio: {ratio:.1%} / {1-ratio:.1%}")
            if ratio > 0.95:
                print(f"  Warning: Validation set too small ({total_val_img} images), consider adding more")
                has_issues = True
            elif ratio < 0.5:
                print(f"  Warning: Training set too small ({total_train_img} images), consider adding more")
                has_issues = True

        print(f"Dataset: {config.get('names', {})}")
        print(f"Path: {dataset_root}")

        if has_issues:
            print("\n[WARN] Dataset has issues but can still be used for training")

        return True

    @staticmethod
    def create_dataset_yaml(output_path: str, dataset_dir: str,
                           names: dict = None, nc: int = 1) -> str:
        """自动生成数据集配置文件"""
        dataset_path = Path(dataset_dir)

        if names is None:
            names = {0: "person"}

        config = {
            "path": str(dataset_path.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": nc,
            "names": names,
            "kpt_shape": [17, 3],
        }

        # 检查是否有 test 集
        test_dir = dataset_path / "images" / "test"
        if test_dir.exists():
            config["test"] = "images/test"

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"Dataset config created: {output}")
        return str(output)


class HandSplitProcessor:
    """手部类型分割处理器

    将手部数据集按手掌/手背分割为两个子数据集，分别训练。
    支持两种模式:
    1. 自动分割: 根据标签中的关键点几何关系自动判断手掌/手背
    2. 手动标注: 使用额外的 hand_side 标签字段

    手部分割数据集结构:
      hand_dataset/
      ├── images/
      │   ├── train/
      │   └── val/
      ├── labels/
      │   ├── train/
      │   └── val/
      ├── palm/                    (手掌子数据集, 自动生成)
      │   ├── images/train/
      │   ├── images/val/
      │   ├── labels/train/
      │   ├── labels/val/
      │   └── dataset.yaml
      ├── back/                    (手背子数据集, 自动生成)
      │   ├── images/train/
      │   ├── images/val/
      │   ├── labels/train/
      │   ├── labels/val/
      │   └── dataset.yaml
      └── dataset.yaml
    """

    # 手部关键点索引 (COCO格式中手腕=9/10, 但手部专用数据集使用21关键点)
    # MediaPipe Hands 21关键点中用于判断手掌/手背的关键点
    PALM_DETECT_KP = {
        "wrist": 0,
        "thumb_cmc": 1, "thumb_mcp": 2,
        "index_mcp": 5,
        "middle_mcp": 9,
        "pinky_mcp": 17,
    }

    @staticmethod
    def classify_from_keypoints(kp_data: list, hand_label: str = "left") -> str:
        """从关键点数据判断手掌/手背朝向

        Args:
            kp_data: 关键点坐标列表 [x1,y1,v1, x2,y2,v2, ...]
            hand_label: "left" 或 "right"

        Returns:
            "palm" 或 "back"
        """
        # 提取关键点
        def get_kp(idx):
            base = idx * 3
            if base + 1 < len(kp_data):
                return float(kp_data[base]), float(kp_data[base + 1])
            return 0.0, 0.0

        thumb_mcp = get_kp(2)   # 拇指MCP
        index_mcp = get_kp(5)   # 食指MCP

        # 拇指相对于食指的水平偏移
        thumb_index_dx = thumb_mcp[0] - index_mcp[0]

        # 左手: 手掌朝向时拇指在右侧(dx>0)
        # 右手: 手掌朝向时拇指在左侧(dx<0)
        if hand_label == "left":
            return "palm" if thumb_index_dx > 0 else "back"
        else:
            return "palm" if thumb_index_dx < 0 else "back"

    @staticmethod
    def split_dataset(source_yaml: str, output_dir: str = None,
                      hand_labels: dict = None) -> dict:
        """将手部数据集按手掌/手背分割

        Args:
            source_yaml: 源数据集配置文件
            output_dir: 输出目录 (默认为源数据集下的 palm/ 和 back/ 子目录)
            hand_labels: 手部标签映射 {图片名: "left"/"right"}，
                        如果不提供则默认左手

        Returns:
            {"palm": palm_yaml_path, "back": back_yaml_path}
        """
        source_path = Path(source_yaml)
        with open(source_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        dataset_root = Path(config['path'])
        if not dataset_root.is_absolute():
            dataset_root = source_path.parent / dataset_root

        if output_dir is None:
            output_dir = dataset_root

        output_root = Path(output_dir)
        result = {}

        for side in ["palm", "back"]:
            side_dir = output_root / side
            side_images_train = side_dir / "images" / "train"
            side_images_val = side_dir / "images" / "val"
            side_labels_train = side_dir / "labels" / "train"
            side_labels_val = side_dir / "labels" / "val"

            for d in [side_images_train, side_images_val,
                      side_labels_train, side_labels_val]:
                d.mkdir(parents=True, exist_ok=True)

        # 处理每个 split
        for split in ["train", "val"]:
            label_dir = dataset_root / "labels" / split
            image_dir = dataset_root / "images" / split

            if not label_dir.exists():
                print(f"Warning: {label_dir} not found, skipping {split}")
                continue

            palm_count = 0
            back_count = 0

            for label_file in label_dir.glob("*.txt"):
                # 读取标签，判断手掌/手背
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f.readlines()
                             if l.strip() and not l.startswith('#')]

                if not lines:
                    continue

                # 使用第一行的关键点判断
                first_line = lines[0].split()
                hand_label = "left"
                if hand_labels:
                    img_name = label_file.stem
                    hand_label = hand_labels.get(img_name, "left")

                # 如果标签包含关键点数据 (pose格式: 5+51=56)
                if len(first_line) >= 56:
                    kp_data = first_line[5:]  # 跳过 class cx cy w h
                    side = HandSplitProcessor.classify_from_keypoints(
                        kp_data, hand_label)
                else:
                    # 检测格式，无法判断，默认 palm
                    side = "palm"

                # 复制图像和标签到对应子目录
                # 查找对应的图像文件
                img_file = None
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                    candidate = image_dir / f"{label_file.stem}{ext}"
                    if candidate.exists():
                        img_file = candidate
                        break

                if img_file is None:
                    continue

                # 复制文件
                target_label_dir = output_root / side / "labels" / split
                target_image_dir = output_root / side / "images" / split

                shutil.copy2(label_file, target_label_dir / label_file.name)
                shutil.copy2(img_file, target_image_dir / img_file.name)

                if side == "palm":
                    palm_count += 1
                else:
                    back_count += 1

            print(f"  {split}: palm={palm_count}, back={back_count}")

        # 生成子数据集配置文件
        for side in ["palm", "back"]:
            side_dir = output_root / side
            side_config = {
                "path": str(side_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": config.get("nc", 1),
                "names": config.get("names", {0: "person"}),
                "flip_idx": config.get("flip_idx", None),
            }

            # 移除 None 值
            side_config = {k: v for k, v in side_config.items() if v is not None}

            yaml_path = side_dir / "dataset.yaml"
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(side_config, f, default_flow_style=False, allow_unicode=True)

            result[side] = str(yaml_path)
            print(f"  {side} dataset: {yaml_path}")

        return result


class ModelTrainer:
    """YOLO26n 姿态估计模型训练器"""

    # COCO 17 关键点定义
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    COCO_SKELETON = [
        [0, 1], [0, 2], [1, 3], [2, 4],
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12],
        [11, 13], [13, 15], [12, 14], [14, 16]
    ]

    # 模型精度等级定义
    # 不同精度等级对应不同的 YOLO 模型变体和训练参数
    MODEL_PRESETS = {
        "nano": {
            "name": "Nano (极速)",
            "model": "yolo26n-pose.pt",
            "description": "最小模型，最快推理速度，适合边缘设备",
            "default_imgsz": 640,
            "default_batch": 32,
            "default_lr0": 0.01,
            "params_m": "~3M",
            "speed_ms": "~5ms",
        },
        "small": {
            "name": "Small (均衡)",
            "model": "yolo26s-pose.pt",
            "description": "小模型，速度与精度均衡，适合大多数场景",
            "default_imgsz": 640,
            "default_batch": 24,
            "default_lr0": 0.01,
            "params_m": "~11M",
            "speed_ms": "~10ms",
        },
        "medium": {
            "name": "Medium (高精度)",
            "model": "yolo26m-pose.pt",
            "description": "中等模型，高精度，适合服务器部署",
            "default_imgsz": 640,
            "default_batch": 16,
            "default_lr0": 0.008,
            "params_m": "~26M",
            "speed_ms": "~20ms",
        },
        "large": {
            "name": "Large (最高精度)",
            "model": "yolo26l-pose.pt",
            "description": "大模型，最高精度，需要强力GPU",
            "default_imgsz": 640,
            "default_batch": 8,
            "default_lr0": 0.006,
            "params_m": "~44M",
            "speed_ms": "~40ms",
        },
    }

    def __init__(self, model: str = "yolo26n-pose.pt", task: str = "pose",
                 device: str = "auto", project: str = "runs/train",
                 name: str = None):
        self.model = model
        self.task = task
        self.project = project
        self.name = name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = self._resolve_device(device)
        self.trainer = None

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def train(self, data: str, epochs: int = 100, batch: int = 16,
              imgsz: int = 640, lr0: float = 0.01, lrf: float = 0.01,
              optimizer: str = "auto", augment: bool = True,
              patience: int = 50, save_period: int = 10,
              resume: str = None, workers: int = 8,
              pretrained: bool = True, freeze: int = 0,
              amp: bool = True, cos_lr: bool = True,
              label_smoothing: float = 0.0,
              auto_validate: bool = True, auto_export: bool = False,
              **kwargs) -> str:
        """训练模型

        Args:
            data: 数据集配置文件路径 (.yaml)
            epochs: 训练轮数
            batch: 批次大小 (-1 自动调整)
            imgsz: 输入图像尺寸
            lr0: 初始学习率
            lrf: 最终学习率 (lr0 * lrf)
            optimizer: 优化器 (auto/SGD/Adam/AdamW)
            augment: 是否启用数据增强
            patience: 早停耐心值 (0=禁用)
            save_period: 每N轮保存一次模型
            resume: 断点续训路径
            workers: 数据加载线程数
            pretrained: 是否使用预训练权重
            freeze: 冻结前N层 (微调用)
            amp: 自动混合精度训练 (GPU加速)
            cos_lr: 余弦退火学习率调度
            label_smoothing: 标签平滑系数 (0-0.1)
            auto_validate: 训练后自动验证
            auto_export: 训练后自动导出ONNX

        Returns:
            最佳模型路径
        """
        from ultralytics import YOLO

        # 检测 GPU 显存，自动调整 batch
        actual_batch = batch
        if batch == -1 or batch == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    # 根据模型大小和显存自动估算 batch
                    model_key = next((k for k, v in self.MODEL_PRESETS.items()
                                      if v["model"] == self.model), "nano")
                    max_batch_map = {"nano": int(mem_gb * 8), "small": int(mem_gb * 5),
                                     "medium": int(mem_gb * 3), "large": int(mem_gb * 2)}
                    actual_batch = min(max_batch_map.get(model_key, 16), 128)
                    actual_batch = max(actual_batch, 4)
                    print(f"  Auto batch: {actual_batch} (GPU {mem_gb:.1f}GB, {model_key})")
                else:
                    actual_batch = 8
            except (ImportError, RuntimeError):
                actual_batch = 8

        print("=" * 60)
        print("  YOLO26n Pose Training")
        print("=" * 60)
        print(f"  Model:          {self.model}")
        print(f"  Task:           {self.task}")
        print(f"  Device:         {self.device}")
        print(f"  Data:           {data}")
        print(f"  Epochs:         {epochs}")
        print(f"  Batch:          {actual_batch}")
        print(f"  Image size:     {imgsz}")
        print(f"  LR:             {lr0} -> {lr0 * lrf}")
        print(f"  Optimizer:      {optimizer}")
        print(f"  Augment:        {augment}")
        print(f"  Pretrained:     {pretrained}")
        print(f"  AMP:            {amp}")
        print(f"  Cosine LR:      {cos_lr}")
        print(f"  Label smoothing:{label_smoothing}")
        if freeze > 0:
            print(f"  Freeze:         {freeze} layers")
        print("=" * 60)

        # 加载模型
        if resume:
            print(f"Resuming from: {resume}")
            model_obj = YOLO(resume)
        else:
            model_obj = YOLO(self.model)

        # 训练参数
        train_args = {
            "data": data,
            "epochs": epochs,
            "batch": actual_batch,
            "imgsz": imgsz,
            "lr0": lr0,
            "lrf": lrf,
            "optimizer": optimizer,
            "augment": augment,
            "patience": patience,
            "save_period": save_period,
            "project": self.project,
            "name": self.name,
            "device": self.device,
            "pretrained": pretrained,
            "workers": workers,
            "exist_ok": True,
            "verbose": True,
            "amp": amp,
            "cos_lr": cos_lr,
            "label_smoothing": label_smoothing,
        }

        # 微调: 冻结层
        if freeze > 0:
            train_args["freeze"] = freeze

        # 姿态估计特有参数
        if self.task == "pose":
            train_args["pose"] = 17  # COCO 17 关键点

        # 合并额外参数
        train_args.update(kwargs)

        # 开始训练
        print("\nStarting training...")
        try:
            results = model_obj.train(**train_args)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[ERROR] GPU 显存不足! 建议减小 batch ({actual_batch} -> {max(actual_batch//2, 2)})")
                print(f"  或使用更小的模型 (--model-type nano)")
                raise
            raise

        # 获取最佳模型路径
        save_dir = Path(self.project) / self.name
        best_model = save_dir / "weights" / "best.pt"
        last_model = save_dir / "weights" / "last.pt"

        if best_model.exists():
            print(f"\nBest model saved: {best_model}")
        if last_model.exists():
            print(f"Last model saved: {last_model}")

        model_path = str(best_model) if best_model.exists() else str(last_model)

        # 训练后自动验证
        if auto_validate and Path(model_path).exists():
            print("\n" + "=" * 60)
            print("  Auto-validating best model...")
            print("=" * 60)
            try:
                self.validate(model_path, data, imgsz)
            except Exception as e:
                print(f"[WARN] Auto-validation failed: {e}")

        # 训练后生成关键点命中分布图
        if auto_validate and Path(model_path).exists():
            try:
                self.analyze_keypoint_distribution(model_path, data, imgsz)
            except Exception as e:
                print(f"[WARN] Keypoint distribution analysis failed: {e}")

        # 训练后自动导出
        if auto_export and Path(model_path).exists():
            print("\n" + "=" * 60)
            print("  Auto-exporting model to ONNX...")
            print("=" * 60)
            try:
                self.export(model_path, "onnx", imgsz, half=amp)
            except Exception as e:
                print(f"[WARN] Auto-export failed: {e}")

        return model_path

    def train_hand_split(self, data: str, epochs: int = 100, batch: int = 16,
                         imgsz: int = 640, **kwargs) -> dict:
        """手掌/手背分割训练

        将数据集按手掌/手背分割后，分别训练两个模型。
        推理时根据手掌/手背分类结果选择对应模型，提高识别精度。

        Args:
            data: 源数据集配置文件
            epochs: 训练轮数
            batch: 批次大小
            imgsz: 图像尺寸
            **kwargs: 其他训练参数

        Returns:
            {"palm": palm_model_path, "back": back_model_path}
        """
        print("=" * 60)
        print("  Hand Split Training (Palm / Back)")
        print("=" * 60)

        # 分割数据集
        print("\nStep 1: Splitting dataset by palm/back...")
        split_result = HandSplitProcessor.split_dataset(data)

        # 训练手掌模型
        print("\nStep 2: Training PALM model...")
        palm_name = f"{self.name}_palm"
        palm_trainer = ModelTrainer(
            model=self.model, task=self.task, device=self.device,
            project=self.project, name=palm_name
        )
        palm_model = palm_trainer.train(
            data=split_result["palm"], epochs=epochs, batch=batch,
            imgsz=imgsz, **kwargs
        )

        # 训练手背模型
        print("\nStep 3: Training BACK model...")
        back_name = f"{self.name}_back"
        back_trainer = ModelTrainer(
            model=self.model, task=self.task, device=self.device,
            project=self.project, name=back_name
        )
        back_model = back_trainer.train(
            data=split_result["back"], epochs=epochs, batch=batch,
            imgsz=imgsz, **kwargs
        )

        # 保存分割训练配置
        split_config = {
            "palm_model": palm_model,
            "back_model": back_model,
            "palm_data": split_result["palm"],
            "back_data": split_result["back"],
            "split_method": "keypoint_geometry",
        }
        config_path = Path(self.project) / self.name / "hand_split_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(split_config, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("  Hand Split Training Complete!")
        print(f"  Palm model: {palm_model}")
        print(f"  Back model: {back_model}")
        print(f"  Config: {config_path}")
        print("=" * 60)

        return {"palm": palm_model, "back": back_model}

    def validate(self, model_path: str, data: str, imgsz: int = 640,
                 batch: int = 32) -> dict:
        """验证模型

        Args:
            model_path: 模型路径
            data: 数据集配置文件
            imgsz: 图像尺寸
            batch: 批次大小

        Returns:
            验证结果字典
        """
        from ultralytics import YOLO

        model_obj = YOLO(model_path)
        results = model_obj.val(data=data, imgsz=imgsz, batch=batch,
                               device=self.device)

        print("\nValidation Results:")
        print(f"  mAP50:    {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")

        if hasattr(results, 'pose'):
            print(f"  Pose mAP50:    {results.pose.map50:.4f}")
            print(f"  Pose mAP50-95: {results.pose.map:.4f}")

        return {
            "map50": float(results.box.map50),
            "map": float(results.box.map),
        }

    def analyze_keypoint_distribution(self, model_path: str, data: str,
                                       imgsz: int = 640, batch: int = 32,
                                       conf: float = 0.25) -> str:
        """生成关键点命中离散分布图

        对验证集进行推理，统计每个关键点的检测置信度分布，
        生成离散分布直方图，直观展示模型对各身体部位的识别能力。

        Args:
            model_path: 模型路径
            data: 数据集配置文件
            imgsz: 图像尺寸
            batch: 批次大小
            conf: 置信度阈值

        Returns:
            分布图保存路径
        """
        from ultralytics import YOLO

        print("\n" + "=" * 60)
        print("  Keypoint Hit Distribution Analysis")
        print("=" * 60)

        model_obj = YOLO(model_path)
        results = model_obj.val(data=data, imgsz=imgsz, batch=batch,
                               device=self.device, conf=conf, save_json=True)

        # 收集关键点置信度数据
        kp_names = self.COCO_KEYPOINTS
        kp_confidences = {name: [] for name in kp_names}
        kp_hit_counts = {name: {"hit": 0, "miss": 0} for name in kp_names}
        total_images = 0

        # 从验证结果中提取关键点数据
        if hasattr(results, 'keypoints') and results.keypoints is not None:
            kpts_data = results.keypoints
            if hasattr(kpts_data, 'data'):
                for kpts in kpts_data.data:
                    total_images += 1
                    for kp_idx, kp_name in enumerate(kp_names):
                        if kp_idx < len(kpts):
                            kp = kpts[kp_idx]
                            if hasattr(kp, 'conf') and kp.conf is not None:
                                conf_val = float(kp.conf)
                                if conf_val > 0:
                                    kp_confidences[kp_name].append(conf_val)
                                    kp_hit_counts[kp_name]["hit"] += 1
                                else:
                                    kp_hit_counts[kp_name]["miss"] += 1
                            elif hasattr(kp, '__len__') and len(kp) >= 3:
                                # [x, y, conf] 格式
                                conf_val = float(kp[2])
                                if conf_val > 0:
                                    kp_confidences[kp_name].append(conf_val)
                                    kp_hit_counts[kp_name]["hit"] += 1
                                else:
                                    kp_hit_counts[kp_name]["miss"] += 1
                            else:
                                kp_hit_counts[kp_name]["miss"] += 1

        # 如果从 val 结果无法提取关键点数据，则用 predict 方式手动收集
        if total_images == 0:
            print("  Using predict mode to collect keypoint data...")
            total_images, kp_confidences, kp_hit_counts = self._collect_kp_from_predict(
                model_obj, data, imgsz, conf)

        # 生成分布图
        plot_path = self._plot_keypoint_distribution(
            kp_names, kp_confidences, kp_hit_counts, total_images, model_path)

        return plot_path

    def _collect_kp_from_predict(self, model_obj, data: str,
                                  imgsz: int, conf: float):
        """通过 predict 模式收集关键点数据"""
        kp_names = self.COCO_KEYPOINTS
        kp_confidences = {name: [] for name in kp_names}
        kp_hit_counts = {name: {"hit": 0, "miss": 0} for name in kp_names}
        total_images = 0

        # 读取数据集配置获取验证集路径
        with open(data, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        dataset_root = Path(config['path'])
        if not dataset_root.is_absolute():
            dataset_root = Path(data).parent / dataset_root

        val_img_dir = dataset_root / "images" / "val"
        if not val_img_dir.exists():
            val_img_dir = dataset_root / config.get('val', 'images/val')
            if not val_img_dir.is_absolute():
                val_img_dir = dataset_root / val_img_dir

        if not val_img_dir.exists():
            print(f"  Warning: Validation image dir not found: {val_img_dir}")
            return total_images, kp_confidences, kp_hit_counts

        # 采样最多 200 张图
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            img_files.extend(val_img_dir.glob(ext))
        img_files = sorted(img_files)[:200]

        print(f"  Analyzing {len(img_files)} validation images...")

        for img_path in img_files:
            preds = model_obj.predict(str(img_path), imgsz=imgsz, conf=conf,
                                       verbose=False, device=self.device)
            total_images += 1

            for pred in preds:
                if hasattr(pred, 'keypoints') and pred.keypoints is not None:
                    kpts = pred.keypoints
                    if hasattr(kpts, 'data'):
                        kpts_data = kpts.data
                    elif hasattr(kpts, 'xy'):
                        kpts_data = kpts.xy
                    else:
                        continue

                    if len(kpts_data) == 0:
                        continue

                    # 取第一个检测到的人
                    person_kpts = kpts_data[0]

                    for kp_idx, kp_name in enumerate(kp_names):
                        if kp_idx < len(person_kpts):
                            kp = person_kpts[kp_idx]
                            if hasattr(kp, 'conf') and kp.conf is not None:
                                conf_val = float(kp.conf)
                            elif len(kp) >= 3:
                                conf_val = float(kp[2])
                            else:
                                conf_val = 0.0

                            if conf_val > 0:
                                kp_confidences[kp_name].append(conf_val)
                                kp_hit_counts[kp_name]["hit"] += 1
                            else:
                                kp_hit_counts[kp_name]["miss"] += 1
                        else:
                            kp_hit_counts[kp_name]["miss"] += 1

        return total_images, kp_confidences, kp_hit_counts

    def _plot_keypoint_distribution(self, kp_names, kp_confidences,
                                     kp_hit_counts, total_images,
                                     model_path: str) -> str:
        """绘制关键点命中离散分布图"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 无头模式
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("[WARN] matplotlib not installed, skipping distribution plot")
            return ""

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        save_dir = Path(model_path).parent.parent  # runs/train/xxx/
        plot_dir = save_dir / "analysis"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # ===== 图1: 关键点命中率柱状图 =====
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # 命中率
        hit_rates = []
        for name in kp_names:
            total = kp_hit_counts[name]["hit"] + kp_hit_counts[name]["miss"]
            rate = kp_hit_counts[name]["hit"] / total if total > 0 else 0
            hit_rates.append(rate)

        ax1 = axes[0]
        colors = ['#2ecc71' if r >= 0.8 else '#f39c12' if r >= 0.5 else '#e74c3c'
                   for r in hit_rates]
        bars = ax1.bar(range(len(kp_names)), hit_rates, color=colors, edgecolor='white', linewidth=0.5)
        ax1.set_xticks(range(len(kp_names)))
        ax1.set_xticklabels([n.replace('_', '\n') for n in kp_names], fontsize=8, rotation=45, ha='right')
        ax1.set_ylabel('Hit Rate', fontsize=11)
        ax1.set_title(f'Keypoint Hit Rate (Total: {total_images} images)', fontsize=13, fontweight='bold')
        ax1.set_ylim(0, 1.05)
        ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good (80%)')
        ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (50%)')
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # 在柱子上标注数值
        for bar, rate in zip(bars, hit_rates):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                     f'{rate:.0%}', ha='center', va='bottom', fontsize=7, fontweight='bold')

        # ===== 图2: 关键点置信度离散分布箱线图 =====
        ax2 = axes[1]
        box_data = []
        box_labels = []
        for name in kp_names:
            if kp_confidences[name]:
                box_data.append(kp_confidences[name])
            else:
                box_data.append([0])
            box_labels.append(name.replace('_', '\n'))

        bp = ax2.boxplot(box_data, patch_artist=True, showfliers=True,
                          flierprops=dict(marker='o', markersize=2, alpha=0.3))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(plt.cm.RdYlGn(hit_rates[i]))
            patch.set_alpha(0.7)

        ax2.set_xticklabels(box_labels, fontsize=8, rotation=45, ha='right')
        ax2.set_ylabel('Confidence', fontsize=11)
        ax2.set_title('Keypoint Confidence Distribution', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1.05)
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_path = plot_dir / "keypoint_distribution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n  Keypoint distribution plot saved: {plot_path}")

        # ===== 图3: 置信度热力图 =====
        fig, ax = plt.subplots(figsize=(12, 5))

        # 构建置信度分箱矩阵
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
                       '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        matrix = np.zeros((len(kp_names), len(bin_labels)))

        for i, name in enumerate(kp_names):
            if kp_confidences[name]:
                hist, _ = np.histogram(kp_confidences[name], bins=bins)
                # 归一化为百分比
                total = sum(hist) if sum(hist) > 0 else 1
                matrix[i] = hist / total * 100

        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=50)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, fontsize=9, rotation=45)
        ax.set_yticks(range(len(kp_names)))
        ax.set_yticklabels([n.replace('_', ' ') for n in kp_names], fontsize=8)
        ax.set_xlabel('Confidence Range', fontsize=11)
        ax.set_title('Keypoint Confidence Heatmap (% of detections)', fontsize=13, fontweight='bold')

        # 标注数值
        for i in range(len(kp_names)):
            for j in range(len(bin_labels)):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                             fontsize=6, color='white' if val > 25 else 'black')

        plt.colorbar(im, ax=ax, label='% of detections')
        plt.tight_layout()
        heatmap_path = plot_dir / "keypoint_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Keypoint heatmap saved: {heatmap_path}")

        # 打印统计摘要
        print("\n  Keypoint Statistics Summary:")
        print(f"  {'Keypoint':<18} {'Hit Rate':>9} {'Mean Conf':>10} {'Median':>8} {'Std':>8}")
        print("  " + "-" * 55)
        for name in kp_names:
            total = kp_hit_counts[name]["hit"] + kp_hit_counts[name]["miss"]
            rate = kp_hit_counts[name]["hit"] / total if total > 0 else 0
            if kp_confidences[name]:
                mean_c = np.mean(kp_confidences[name])
                median_c = np.median(kp_confidences[name])
                std_c = np.std(kp_confidences[name])
                print(f"  {name:<18} {rate:>8.1%} {mean_c:>10.3f} {median_c:>8.3f} {std_c:>8.3f}")
            else:
                print(f"  {name:<18} {rate:>8.1%} {'N/A':>10} {'N/A':>8} {'N/A':>8}")

        return str(plot_path)

    def export(self, model_path: str, format: str = "onnx",
               imgsz: int = 640, half: bool = False) -> str:
        """导出模型

        Args:
            model_path: 模型路径
            format: 导出格式 (onnx/torchscript/engine/openvino/coreml/tflite)
            imgsz: 图像尺寸
            half: 是否使用 FP16

        Returns:
            导出模型路径
        """
        from ultralytics import YOLO

        model_obj = YOLO(model_path)
        export_path = model_obj.export(format=format, imgsz=imgsz, half=half)

        print(f"Model exported: {export_path}")
        return str(export_path)


def create_sample_dataset(output_dir: str):
    """创建示例数据集目录结构"""
    base = Path(output_dir)

    dirs = [
        base / "images" / "train",
        base / "images" / "val",
        base / "labels" / "train",
        base / "labels" / "val",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # 生成数据集配置
    config = {
        "path": str(base.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "person"},
    }

    config_path = base / "dataset.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # 生成示例标签文件
    sample_label = base / "labels" / "train" / "sample.txt"
    with open(sample_label, 'w') as f:
        # class cx cy w h kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v
        f.write("# YOLO pose label format (one line per person):\n")
        f.write("# class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...\n")
        f.write("# Example (17 COCO keypoints, 56 values per line):\n")
        f.write("# 0 0.5 0.5 0.3 0.6 0.5 0.3 2 0.48 0.25 2 ...\n")

    print(f"Sample dataset structure created at: {base}")
    print(f"Config file: {config_path}")
    print("\nNext steps:")
    print(f"  1. Copy training images to {base / 'images' / 'train'}")
    print(f"  2. Copy validation images to {base / 'images' / 'val'}")
    print(f"  3. Create label files in {base / 'labels' / 'train'}")
    print(f"  4. Create label files in {base / 'labels' / 'val'}")
    print(f"  5. Run: python train.py --data {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO26n Pose Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 从预训练模型微调
  python train.py --data dataset.yaml --epochs 50 --freeze 10

  # 从头训练
  python train.py --data dataset.yaml --model yolo26n-pose.pt --epochs 100

  # 断点续训
  python train.py --resume runs/train/train_xxx/weights/last.pt

  # 验证模型
  python train.py --validate --model runs/train/train_xxx/weights/best.pt --data dataset.yaml

  # 导出 ONNX
  python train.py --export --model runs/train/train_xxx/weights/best.pt --format onnx

  # 创建示例数据集
  python train.py --create-dataset ./my_dataset
        """
    )

    # 基本参数
    parser.add_argument("--model", type=str, default="yolo26n-pose.pt",
                       help="基础模型路径 (默认: yolo26n-pose.pt)")
    parser.add_argument("--model-type", type=str, default=None,
                       choices=["nano", "small", "medium", "large"],
                       help="模型精度等级 (覆盖 --model 参数)")
    parser.add_argument("--data", type=str, default=None,
                       help="数据集配置文件路径 (.yaml)")
    parser.add_argument("--task", type=str, default="pose", choices=["pose", "detect"],
                       help="任务类型 (默认: pose)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小 (-1=自动)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--save-period", type=int, default=10, help="每N轮保存")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程数")

    # 微调参数
    parser.add_argument("--freeze", type=int, default=0,
                       help="冻结前N层 (微调用, 推荐10)")
    parser.add_argument("--no-pretrained", action="store_true",
                       help="不使用预训练权重")
    parser.add_argument("--no-augment", action="store_true",
                       help="禁用数据增强")

    # 训练优化参数
    parser.add_argument("--lr0", type=float, default=0.01,
                       help="初始学习率")
    parser.add_argument("--lrf", type=float, default=0.01,
                       help="最终学习率系数")
    parser.add_argument("--optimizer", type=str, default="auto",
                       choices=["auto", "SGD", "Adam", "AdamW"],
                       help="优化器")
    parser.add_argument("--patience", type=int, default=50,
                       help="早停耐心值 (0=禁用)")
    parser.add_argument("--no-amp", action="store_true",
                       help="禁用自动混合精度训练")
    parser.add_argument("--no-cos-lr", action="store_true",
                       help="禁用余弦退火学习率调度")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                       help="标签平滑系数 (0-0.1)")

    # 设备
    parser.add_argument("--device", type=str, default="auto",
                       help="训练设备 (auto/0/1/cpu)")

    # 输出
    parser.add_argument("--project", type=str, default="runs/train",
                       help="输出目录")
    parser.add_argument("--name", type=str, default=None,
                       help="实验名称")

    # 断点续训
    parser.add_argument("--resume", type=str, default=None,
                       help="断点续训模型路径")

    # 验证
    parser.add_argument("--validate", action="store_true",
                       help="仅验证模型 (不训练)")
    parser.add_argument("--val-batch", type=int, default=32,
                       help="验证批次大小")

    # 导出
    parser.add_argument("--export", action="store_true",
                       help="导出模型")
    parser.add_argument("--format", type=str, default="onnx",
                       choices=["onnx", "torchscript", "engine", "openvino",
                                "coreml", "tflite"],
                       help="导出格式")
    parser.add_argument("--half", action="store_true",
                       help="FP16 半精度导出")

    # 工具
    parser.add_argument("--create-dataset", type=str, default=None,
                       help="创建示例数据集目录结构")
    parser.add_argument("--check-data", type=str, default=None,
                       help="验证数据集配置文件")

    # 手部分割训练
    parser.add_argument("--hand-split", action="store_true",
                       help="启用手掌/手背分割训练模式")
    parser.add_argument("--split-only", type=str, default=None,
                       help="仅分割数据集 (不训练), 指定源数据集yaml路径")

    args = parser.parse_args()

    # 处理模型精度等级
    if args.model_type:
        preset = ModelTrainer.MODEL_PRESETS.get(args.model_type)
        if preset:
            args.model = preset["model"]
            print(f"Model type: {args.model_type} -> {preset['name']} ({preset['model']})")
            print(f"  {preset['description']}")
            print(f"  参数量: {preset['params_m']}, 推理速度: {preset['speed_ms']}")

    # 创建示例数据集
    if args.create_dataset:
        create_sample_dataset(args.create_dataset)
        return

    # 验证数据集
    if args.check_data:
        DatasetValidator.validate_dataset(args.check_data)
        return

    # 仅分割数据集
    if args.split_only:
        print("Splitting dataset by palm/back...")
        result = HandSplitProcessor.split_dataset(args.split_only)
        print(f"\nPalm dataset: {result['palm']}")
        print(f"Back dataset: {result['back']}")
        return

    # 必须指定数据集
    if not args.data and not args.resume and not args.validate and not args.export:
        parser.error("--data is required for training. Use --create-dataset to create a sample dataset.")

    # 初始化训练器
    trainer = ModelTrainer(
        model=args.model,
        task=args.task,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    # 验证模式
    if args.validate:
        if not args.model or not args.data:
            parser.error("--validate requires --model and --data")
        trainer.validate(args.model, args.data, args.imgsz, args.val_batch)
        return

    # 导出模式
    if args.export:
        if not args.model:
            parser.error("--export requires --model")
        trainer.export(args.model, args.format, args.imgsz, args.half)
        return

    # 训练前验证数据集
    if args.data:
        if not DatasetValidator.validate_dataset(args.data):
            print("\nDataset validation failed! Please check your dataset.")
            sys.exit(1)

    # 手部分割训练模式
    if args.hand_split:
        best_models = trainer.train_hand_split(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            lrf=args.lrf,
            optimizer=args.optimizer,
            augment=not args.no_augment,
            patience=args.patience,
            save_period=args.save_period,
            workers=args.workers,
            pretrained=not args.no_pretrained,
            freeze=args.freeze,
            amp=not args.no_amp,
            cos_lr=not args.no_cos_lr,
            label_smoothing=args.label_smoothing,
        )
        print("\n" + "=" * 60)
        print("  Hand Split Training Complete!")
        print(f"  Palm model: {best_models['palm']}")
        print(f"  Back model: {best_models['back']}")
        print("=" * 60)
        return

    # 开始训练
    best_model = trainer.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        augment=not args.no_augment,
        patience=args.patience,
        save_period=args.save_period,
        resume=args.resume,
        workers=args.workers,
        pretrained=not args.no_pretrained,
        freeze=args.freeze,
        amp=not args.no_amp,
        cos_lr=not args.no_cos_lr,
        label_smoothing=args.label_smoothing,
    )

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best model: {best_model}")
    print(f"  Use in main.py: --model {best_model}")
    print("=" * 60)


if __name__ == "__main__":
    main()
