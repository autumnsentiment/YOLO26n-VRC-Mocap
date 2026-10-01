#!/usr/bin/env python3
"""Offline checks for vision accuracy fixes (no camera / GPU required)."""

import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"))

import camera_capture as camera_module  # noqa: E402
from camera_capture import CameraCapture  # noqa: E402
from face_landmark import OneEuroFilter1D as FaceOneEuro  # noqa: E402
from hand_tracker import HandTracker, OneEuroFilter1D as HandOneEuro  # noqa: E402
from mediapipe_pose import MediaPipePoseDetector  # noqa: E402
from pose_estimator import OneEuroFilter, PoseEstimator  # noqa: E402


def _expected_alpha(cutoff, freq):
    r = 2 * math.pi * cutoff / freq
    return r / (1 + r)


def test_one_euro_alpha_matches_casiez_formula():
    for cls in (OneEuroFilter, HandOneEuro, FaceOneEuro):
        f = cls(min_cutoff=1.0, beta=0.0, d_cutoff=1.0, freq=30.0)
        for cutoff in (0.5, 1.0, 5.0):
            assert abs(f._smoothing_factor(cutoff) - _expected_alpha(cutoff, 30.0)) < 1e-12, cls
        # A step input at 30 Hz must move by exactly alpha (not 1 - alpha).
        f.filter(0.0, t=0.0)
        out = f.filter(1.0, t=1.0 / 30.0)
        assert abs(out - _expected_alpha(1.0, 30.0)) < 1e-9, (cls, out)
        # Zero timestamp is a real timestamp, and repeated/backwards t is safe.
        g = cls(min_cutoff=1.0, beta=0.0, freq=30.0)
        g.filter(0.0, t=0.0)
        assert math.isfinite(g.filter(1.0, t=0.0))


def test_one_euro_high_beta_tracks_fast_motion():
    f = OneEuroFilter(min_cutoff=1.0, beta=1.0, d_cutoff=1.0, freq=60.0)
    t, out = 0.0, 0.0
    for i in range(30):
        t += 1.0 / 60.0
        out = f.filter(i * 0.1, t=t)
    # Moving 6 units/s: low-latency filter should lag well under 0.2 units.
    assert abs(out - 2.9) < 0.2, out


def _person(cx, cy, scale=1.0, conf=0.9):
    """17 COCO keypoints around a torso center (pixel coords)."""
    kps = np.zeros((17, 3), dtype=float)
    offsets = {
        5: (-40, -60), 6: (40, -60),   # shoulders
        11: (-30, 60), 12: (30, 60),   # hips
        0: (0, -110),                   # nose
        7: (-50, 0), 8: (50, 0), 9: (-55, 50), 10: (55, 50),
        13: (-30, 140), 14: (30, 140), 15: (-30, 220), 16: (30, 220),
    }
    for idx, (dx, dy) in offsets.items():
        kps[idx] = (cx + dx * scale, cy + dy * scale, conf)
    box = [cx - 70 * scale, cy - 130 * scale, cx + 70 * scale, cy + 240 * scale]
    return kps, box


def _detections(*people):
    return {
        "keypoints": [p[0] for p in people],
        "boxes": [p[1] for p in people],
        "confidence": [0.9 for _ in people],
    }


def test_multi_person_selection_is_stable():
    est = PoseEstimator(smoothing="linear", smoothing_factor=1.0)
    near = _person(300, 300, scale=1.4)
    far = _person(900, 300, scale=0.8)
    # Largest person is chosen first, regardless of list order.
    assert est._select_person(_detections(far, near), 1280, 720) == 1
    # Then the same person is kept even if a larger one appears elsewhere.
    bigger = _person(950, 320, scale=2.0)
    moved = _person(320, 305, scale=1.4)
    assert est._select_person(_detections(bigger, moved), 1280, 720) == 1
    # Brief loss of the target does not switch to the other person.
    assert est._select_person(_detections(bigger), 1280, 720) is None


def test_hand_roi_requires_matching_side_and_nearby_wrist():
    rect = (100, 100, 300, 300)
    assert HandTracker._accept_roi_hand("left", "left", (210, 205), (200, 200), rect)
    assert not HandTracker._accept_roi_hand("right", "left", (200, 200), (200, 200), rect)
    assert not HandTracker._accept_roi_hand("left", "left", (290, 290), (120, 120), rect)
    assert not HandTracker._accept_roi_hand("left", "left", (200, 200), None, rect)


def test_camera_requests_mjpg_before_resolution():
    import cv2

    calls = []

    class _Capture:
        def __init__(self):
            self.values = {cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"YUY2"))}

        def isOpened(self):
            return True

        def set(self, prop, value):
            calls.append(prop)
            self.values[prop] = value
            return True

        def get(self, prop):
            return self.values.get(prop, 0)

        def release(self):
            pass

    original_capture = camera_module.cv2.VideoCapture
    original_platform = camera_module.sys.platform
    camera_module.cv2.VideoCapture = lambda *args, **kwargs: _Capture()
    camera_module.sys.platform = "linux"
    try:
        camera = CameraCapture(camera_id=0, width=1280, height=720, fps=60)
        assert camera.open() is True
    finally:
        camera_module.cv2.VideoCapture = original_capture
        camera_module.sys.platform = original_platform
    assert calls.index(cv2.CAP_PROP_FOURCC) < calls.index(cv2.CAP_PROP_FRAME_WIDTH)
    assert camera.fourcc == "MJPG"


def test_yolo_int8_on_cuda_falls_back_to_fp16():
    from edition import yolo_available

    if not yolo_available():
        print("SKIP yolo INT8 fallback (lite edition)")
        return
    import yolo_detector

    class _Net:
        def __init__(self):
            self.half_called = False

        def half(self):
            self.half_called = True
            return self

    class _FakeYOLO:
        def __init__(self, path):
            self.model = _Net()
            self.calls = []

        def to(self, device):
            return self

        def __call__(self, frame, **kwargs):
            self.calls.append((frame.shape, kwargs))
            return []

    original = yolo_detector.YOLO
    yolo_detector.YOLO = _FakeYOLO
    try:
        det = yolo_detector.YOLO26nDetector(model_path="fake.pt", device="cuda", precision="int8")
        assert det.load_model() is True
        assert det.precision == "fp16" and det.half is True
        assert det.model.model.half_called
        det.warmup(shape=(720, 1280))
        shape, kwargs = det.model.calls[-1]
        assert shape == (720, 1280, 3)
        assert kwargs.get("half") is True
    finally:
        yolo_detector.YOLO = original


def _landmark(x, y, z, visibility=0.99):
    return types.SimpleNamespace(x=x, y=y, z=z, visibility=visibility, presence=0.99)


def _world_landmarks(hip_visibility):
    lms = [_landmark(0.0, 0.0, 0.0) for _ in range(33)]
    lms[MediaPipePoseDetector.LEFT_SHOULDER] = _landmark(0.18, -0.5, 0.0)
    lms[MediaPipePoseDetector.RIGHT_SHOULDER] = _landmark(-0.18, -0.5, 0.0)
    lms[MediaPipePoseDetector.LEFT_HIP] = _landmark(0.1, 0.0, 0.0, hip_visibility)
    lms[MediaPipePoseDetector.RIGHT_HIP] = _landmark(-0.1, 0.0, 0.0, hip_visibility)
    return lms


def test_mediapipe_hip_anchor_is_held_briefly_when_hips_drop_out():
    det = MediaPipePoseDetector()
    shoulders = np.array([0.0, -0.5, 0.0])
    good = det._resolve_hip_anchor(_world_landmarks(0.99), shoulders, None, now=10.0)
    np.testing.assert_allclose(good, [0.0, 0.0, 0.0], atol=1e-9)
    # Low-visibility hips that jumped are ignored within the hold window...
    weak = _world_landmarks(0.05)
    jumped = np.array([0.0, 0.4, 0.3])
    held = det._resolve_hip_anchor(weak, shoulders + [0.05, 0.0, 0.0], jumped, now=10.2)
    np.testing.assert_allclose(held, [0.05, 0.0, 0.0], atol=1e-9)
    # ...and accepted again after the hold expires.
    later = det._resolve_hip_anchor(weak, shoulders, jumped, now=10.6)
    np.testing.assert_allclose(later, jumped, atol=1e-9)


def main():
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"OK {test.__name__}")
    print("Vision accuracy checks passed")


if __name__ == "__main__":
    main()