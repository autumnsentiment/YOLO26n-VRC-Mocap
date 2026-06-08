#!/usr/bin/env python3
"""System test script for YOLO26n VRC Mocap"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bin'))

results = []

def test_step(name, func):
    try:
        result = func()
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
    from yolo_detector import YOLO26nDetector
    from pose_estimator import PoseEstimator
    from vrc_transmitter import VRCTransmitter
    from camera_capture import CameraCapture, ImageProcessor
    from face_landmark import FaceLandmarkDetector
    from hand_tracker import HandTracker
    return "All 6 core modules imported"

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
    return "No model file found"

# 8. Check YOLO model loading
def check_yolo_load():
    from ultralytics import YOLO
    model_path = os.path.join(os.path.dirname(__file__), 'yolo26n-pose.pt')
    if not os.path.exists(model_path):
        return "Skipped - model file not found"
    model = YOLO(model_path)
    return f"Model loaded: {model.task}"

# 9. Test PoseEstimator initialization
def check_pose_estimator():
    from pose_estimator import PoseEstimator
    pe = PoseEstimator(smoothing='one_euro', smoothing_factor=0.3,
                       one_euro_min_cutoff=1.0, one_euro_beta=0.007)
    return f"PoseEstimator initialized, keypoint_count={pe.num_keypoints if hasattr(pe, 'num_keypoints') else 'N/A'}"

# 10. Test VRCTransmitter initialization
def check_vrc_transmitter():
    from vrc_transmitter import VRCTransmitter
    vt = VRCTransmitter(host='127.0.0.1', port=9000, protocol='osc')
    return f"VRCTransmitter initialized (protocol={vt.protocol})"

# 11. Test FaceLandmarkDetector
def check_face_detector():
    from face_landmark import FaceLandmarkDetector
    fd = FaceLandmarkDetector()
    loaded = fd.load()
    return f"FaceLandmarkDetector loaded={loaded}"

# 12. Test HandTracker
def check_hand_tracker():
    from hand_tracker import HandTracker
    ht = HandTracker(max_num_hands=2)
    loaded = ht.load()
    return f"HandTracker loaded={loaded}"

# 13. Test config loading
def check_config():
    from main import VRCMocapSystem
    system = VRCMocapSystem()
    keys = list(system.config.keys())
    return f"Config keys: {', '.join(keys)}"

# 14. Check Flask
def check_flask():
    import flask
    return f"flask {flask.__version__}"

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

# 18. Check sample dataset
def check_sample_dataset():
    ds_path = os.path.join(os.path.dirname(__file__), 'sample_dataset', 'dataset.yaml')
    if os.path.exists(ds_path):
        return "sample_dataset found"
    return "sample_dataset not found"

# Run all tests
test_step("Python", check_python)
test_step("PyTorch", check_torch)
test_step("OpenCV", check_opencv)
test_step("NumPy", check_numpy)
test_step("Ultralytics", check_ultralytics)
test_step("Flask", check_flask)
test_step("python-osc", check_pythonosc)
test_step("MediaPipe", check_mediapipe)
test_step("SciPy", check_scipy)
test_step("Core Imports", check_imports)
test_step("YOLO Model File", check_model)
test_step("YOLO Model Load", check_yolo_load)
test_step("PoseEstimator", check_pose_estimator)
test_step("VRCTransmitter", check_vrc_transmitter)
test_step("FaceLandmarkDetector", check_face_detector)
test_step("HandTracker", check_hand_tracker)
test_step("Config Loading", check_config)
test_step("Sample Dataset", check_sample_dataset)

# Print results
print()
print('=' * 70)
print('  YOLO26n VRC Mocap - System Test Results')
print('=' * 70)
print(f'{"Test":<25s} | {"Status":6s} | Detail')
print('-' * 70)
pass_count = 0
fail_count = 0
for name, status, detail in results:
    icon = 'OK' if status == 'PASS' else '!!'
    if status == 'PASS':
        pass_count += 1
    else:
        fail_count += 1
    print(f'{name:<25s} | {icon:<6s} | {detail}')
print('-' * 70)
print(f'Total: {len(results)} tests | PASSED: {pass_count} | FAILED: {fail_count}')
print('=' * 70)
