#!/usr/bin/env python3
"""System test script for YOLO26n VRC Mocap"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bin'))

from edition import current_edition, yolo_available  # noqa: E402


def run_check(results, name, func):
    if getattr(func, "full_only", False) and not yolo_available():
        results.append((name, 'SKIP', 'lite edition (no YOLO/PyTorch)'))
        return
    try:
        result = func()
        if result is False:
            raise AssertionError("check returned False")
        results.append((name, 'PASS', str(result)))
    except Exception as e:
        results.append((name, 'FAIL', str(e)[:120]))

# 1. Check Python version
def check_python():
    import platform
    return f"Python {platform.python_version()}"

# 2. Check PyTorch
def check_torch():
    import torch
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else 'N/A'
    return f"torch {torch.__version__}, CUDA={cuda}, GPU={gpu}"

# 3. Check OpenCV
def check_opencv():
    import cv2
    return f"cv2 {cv2.__version__}"

# 4. Check NumPy
def check_numpy():
    import numpy as np
    return f"numpy {np.__version__}"

# 5. Check Ultralytics
def check_ultralytics():
    import ultralytics
    return f"ultralytics {ultralytics.__version__}"

# 6. Import core modules
def check_imports():
    if yolo_available():
        from yolo_detector import YOLO26nDetector  # noqa: F401
    from pose_estimator import PoseEstimator
    from camera_capture import CameraCapture, ImageProcessor
    from face_landmark import FaceLandmarkDetector
    from hand_tracker import HandTracker
    from modules import BodyModule, FaceModule, HandModule
    from streamers import BodyOSCStreamer, BodySteamVRStreamer, FaceVMCStreamer, HandVMTStreamer
    return "Runtime detectors, modules, and streamers imported"

# 7. Check YOLO model file
def check_model():
    model_path = os.path.join(os.path.dirname(__file__), 'yolo26n-pose.pt')
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return f"yolo26n-pose.pt found ({size_mb:.1f} MB)"
    # Check for trained model
    import glob
    trained = glob.glob(os.path.join(os.path.dirname(__file__), 'runs', '**', 'best.pt'), recursive=True)
    if trained:
        return f"No base model, found trained: {trained[0]}"
    raise FileNotFoundError("No runtime pose model file found")

# 8. Check YOLO model loading
def check_yolo_load():
    from ultralytics import YOLO
    model_path = os.path.join(os.path.dirname(__file__), 'yolo26n-pose.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = YOLO(model_path)
    return f"Model loaded: {model.task}"

# 9. Test PoseEstimator initialization
def check_pose_estimator():
    from pose_estimator import PoseEstimator
    pe = PoseEstimator(smoothing='one_euro', smoothing_factor=0.3,
                       one_euro_min_cutoff=1.0, one_euro_beta=0.007)
    return f"PoseEstimator initialized, keypoint_count={pe.num_keypoints if hasattr(pe, 'num_keypoints') else 'N/A'}"

# 10. Test current body OSC streamer initialization
def check_body_streamer():
    from streamers.body_osc import BodyOSCStreamer
    streamer = BodyOSCStreamer(host='127.0.0.1', port=9000)
    return f"BodyOSCStreamer initialized ({streamer.host}:{streamer.port})"

# 11. Test FaceLandmarkDetector
def check_face_detector():
    from face_landmark import FaceLandmarkDetector
    fd = FaceLandmarkDetector()
    try:
        loaded = fd.load()
        if not loaded:
            raise RuntimeError("FaceLandmarkDetector model failed to load")
        return f"FaceLandmarkDetector loaded={loaded}"
    finally:
        fd.release()

# 12. Test HandTracker
def check_hand_tracker():
    from hand_tracker import HandTracker
    ht = HandTracker(max_num_hands=2)
    try:
        loaded = ht.load()
        if not loaded:
            raise RuntimeError("HandTracker model failed to load")
        return f"HandTracker loaded={loaded}"
    finally:
        ht.release()

# 13. Test config loading
def check_config():
    from main import VRCMocapSystem
    system = VRCMocapSystem()
    keys = list(system.config.keys())
    return f"Config keys: {', '.join(keys)}"

# 14. Check Flask
def check_flask():
    from importlib.metadata import version
    return f"flask {version('flask')}"

# 15. Check python-osc
def check_pythonosc():
    from pythonosc import dispatcher, osc_server
    return "python-osc available"

# 16. Check mediapipe
def check_mediapipe():
    import mediapipe as mp
    return f"mediapipe {mp.__version__}"

# 17. Check scipy
def check_scipy():
    import scipy
    return f"scipy {scipy.__version__}"

# 18. Check runtime model directory
def check_runtime_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    available = []
    for root in [os.path.dirname(__file__), models_dir]:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if os.path.splitext(name)[1].lower() in {'.pt', '.onnx', '.engine'}:
                available.append(name)
    if available:
        return f"Runtime models: {', '.join(sorted(set(available))[:5])}"
    raise FileNotFoundError("No runtime .pt/.onnx/.engine model found")

def main():
    results = []
    checks = [
        ("Python", check_python),
        ("PyTorch", check_torch),
        ("OpenCV", check_opencv),
        ("NumPy", check_numpy),
        ("Ultralytics", check_ultralytics),
        ("Flask", check_flask),
        ("python-osc", check_pythonosc),
        ("MediaPipe", check_mediapipe),
        ("SciPy", check_scipy),
        ("Core Imports", check_imports),
        ("YOLO Model File", check_model),
        ("YOLO Model Load", check_yolo_load),
        ("PoseEstimator", check_pose_estimator),
        ("Body OSC Streamer", check_body_streamer),
        ("FaceLandmarkDetector", check_face_detector),
        ("HandTracker", check_hand_tracker),
        ("Config Loading", check_config),
        ("Runtime Models", check_runtime_models),
    ]
    for check in (check_torch, check_ultralytics, check_model, check_yolo_load, check_runtime_models):
        check.full_only = True
    checks.insert(1, ("Edition", lambda: f"{current_edition()} edition"))
    for name, check in checks:
        run_check(results, name, check)

    print()
    print('=' * 70)
    print('  YOLO26n VRC Mocap - System Test Results')
    print('=' * 70)
    print(f'{"Test":<25s} | {"Status":6s} | Detail')
    print('-' * 70)
    pass_count = 0
    fail_count = 0
    skip_count = 0
    for name, status, detail in results:
        icon = {'PASS': 'OK', 'SKIP': '--'}.get(status, '!!')
        if status == 'PASS':
            pass_count += 1
        elif status == 'SKIP':
            skip_count += 1
        else:
            fail_count += 1
        print(f'{name:<25s} | {icon:<6s} | {detail}')
    print('-' * 70)
    print(f'Total: {len(results)} tests | PASSED: {pass_count} | SKIPPED: {skip_count} | FAILED: {fail_count}')
    print('=' * 70)
    return 1 if fail_count else 0


if __name__ == '__main__':
    raise SystemExit(main())
