"""Shared YOLO model presets for inference and training."""

MODEL_PRESETS = {
    "nano": {
        "name": "Nano (fastest)",
        "model": "yolo26n-pose.pt",
        "description": "Smallest model, fastest inference, suitable for edge devices",
        "default_imgsz": 640,
        "default_batch": 32,
        "default_lr0": 0.01,
        "params_m": "~3M",
        "speed_ms": "~5ms",
    },
    "small": {
        "name": "Small (balanced)",
        "model": "yolo26s-pose.pt",
        "description": "Small model with balanced speed and accuracy",
        "default_imgsz": 640,
        "default_batch": 24,
        "default_lr0": 0.01,
        "params_m": "~11M",
        "speed_ms": "~10ms",
    },
    "medium": {
        "name": "Medium (accurate)",
        "model": "yolo26m-pose.pt",
        "description": "Medium model with higher accuracy",
        "default_imgsz": 640,
        "default_batch": 16,
        "default_lr0": 0.008,
        "params_m": "~26M",
        "speed_ms": "~20ms",
    },
    "large": {
        "name": "Large (highest accuracy)",
        "model": "yolo26l-pose.pt",
        "description": "Large model with highest accuracy, requires a stronger GPU",
        "default_imgsz": 640,
        "default_batch": 8,
        "default_lr0": 0.006,
        "params_m": "~44M",
        "speed_ms": "~40ms",
    },
}
