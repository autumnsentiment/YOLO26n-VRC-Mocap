#!/usr/bin/env python3
"""Runtime-only logic checks for smoothing and HMD body anchoring."""

import os
import sys
import json
import socket
import struct
import threading
import time
from unittest import mock
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bin"))

from pose_estimator import PoseEstimator
from face_calibration import apply_vrcft_calibration, read_face_state, write_face_state
from streamers.coord import CoordinateTransformer
from streamers.body_osc import BodyOSCStreamer
from streamers.body_steamvr import BodySteamVRStreamer
from streamers.face_vmc import FaceVMCStreamer
from streamers.hand_vmt import HandVMTStreamer
from streamers.coord import hmd_pose_anchor_trusted
from dashboard_server import (
    build_performance_recommendation,
    classify_performance_tier,
    ensure_hand_streaming_config,
    ensure_ui_config,
    normalize_body_streaming_config,
)
from face_landmark import FaceLandmarkDetector
from hmd_provider import (
    DashboardAPIProvider,
    HMDDataProvider,
    OpenVRProvider,
    PicoStreamingProvider,
    PicoXRoboProvider,
    QuestLinkProvider,
    QuestUDPProvider,
    RemoteHMDProvider,
    SDKUDPProvider,
    VRChatOSCProvider,
    _actual_height_from_position,
    _openvr_rotation_to_unity,
)
from hmd_sender import (
    _openvr_rotation_to_unity as _sender_openvr_rotation_to_unity,
    _send_invalid_pose,
)
from camera_capture import CameraCapture, timestamp_rate
from hand_tracker import HandTracker
from modules.base_module import BaseModule
import mediapipe_runtime
from mediapipe_runtime import next_timestamp_ms
import runtime_paths as runtime_paths_module
import runtime_status as runtime_status_module
from runtime_status import mark_tracking_stopped, read_tracking_state, write_tracking_state
import vmt_runtime


class _ScheduledTestModule(BaseModule):
    def __init__(self):
        super().__init__("body", {"enabled": True, "processing": {"body_max_fps": 5}}, {"streaming_enabled": True})
        self.calls = 0

    def _do_initialize(self):
        return True

    def _do_process(self, frame):
        self.calls += 1
        return {"frame": self.calls}

    def _do_cleanup(self):
        pass


class _FakeCaptureDevice:
    def __init__(self):
        self.opened = True
        self.values = {}

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.values[prop] = value
        return True

    def get(self, prop):
        return self.values.get(prop, 0)


def _max_step(values):
    return max(abs(values[i] - values[i - 1]) for i in range(1, len(values)))


def test_average_smoothing_reduces_jitter():
    estimator = PoseEstimator(
        smoothing="linear",
        smoothing_factor=1.0,
        average_smoothing_enabled=True,
        average_window=4,
        ema_alpha=0.35,
    )
    raw_x = [0.0, 0.22, -0.18, 0.20, -0.16, 0.18, -0.15]
    smoothed_x = []

    for x in raw_x:
        pose = estimator.apply_average_smoothing({"Head": [x, 1.6, 0.0]})
        smoothed_x.append(pose["Head"][0])

    assert _max_step(smoothed_x) < _max_step(raw_x), (raw_x, smoothed_x)

    estimator.reset_filters()
    reset_pose = estimator.apply_average_smoothing({"Head": [1.0, 1.6, 0.0]})
    assert reset_pose["Head"][0] == 1.0


def test_base_module_rate_limit_reuses_preview_result_without_restreaming():
    module = _ScheduledTestModule()
    assert module.initialize() is True
    first = module.process(object())
    second = module.process(object())
    assert first == {"frame": 1}
    assert second == first
    assert module.calls == 1
    assert module.last_result_fresh is False
    assert module.get_status()["skipped_frames"] == 1
    time.sleep(0.21)
    third = module.process(object())
    assert third == {"frame": 2}
    assert module.last_result_fresh is True


def test_base_module_deadline_scheduler_avoids_input_rate_quantization():
    module = _ScheduledTestModule()
    module._max_fps = 20.0
    module._min_process_interval = 1.0 / 20.0
    module._next_process_at = 0.0

    selected = 0
    for index in range(30):
        now = 100.0 + index / 30.0
        if module.should_process(now):
            selected += 1
            module._advance_scheduler(now)

    assert selected == 20
    late = 102.0
    assert module.should_process(late) is True
    module._advance_scheduler(late)
    assert module.should_process(late + 0.001) is False


def test_timestamp_rate_uses_elapsed_intervals():
    assert timestamp_rate([]) == 0.0
    assert timestamp_rate([1.0]) == 0.0
    assert abs(timestamp_rate([1.0, 1.1, 1.2]) - 10.0) < 1e-9
    assert timestamp_rate([1.0, 1.0]) == 0.0


def test_mediapipe_timestamp_is_strictly_increasing():
    with mock.patch.object(
        mediapipe_runtime.time,
        "monotonic_ns",
        side_effect=[1_000_000_000, 1_000_000_000, 999_000_000],
    ):
        first = next_timestamp_ms()
        second = next_timestamp_ms(first)
        third = next_timestamp_ms(second)

    assert (first, second, third) == (1000, 1001, 1002)


def test_camera_hardware_zoom_property_is_applied():
    import cv2

    camera = CameraCapture(zoom=0)
    fake = _FakeCaptureDevice()
    camera.cap = fake
    assert camera.set_zoom(12) is True
    assert fake.values[cv2.CAP_PROP_ZOOM] == 12
    status = camera.get_status()
    assert status["zoom"] == 12
    assert status["actual_zoom"] == 12
    assert status["zoom_supported"] is True
    assert status["low_latency_buffer"] is True


def test_camera_get_frame_with_count_can_avoid_copy():
    camera = CameraCapture()
    sample = np.zeros((2, 2, 3), dtype=np.uint8)
    with camera.lock:
        camera.current_frame = sample
        camera.frame_count = 7
    frame_ref, count_ref = camera.get_frame_with_count(copy=False)
    frame_copy, count_copy = camera.get_frame_with_count(copy=True)
    assert count_ref == 7
    assert count_copy == 7
    assert frame_ref is sample
    assert frame_copy is not sample


def test_camera_wait_for_frame_wakes_only_for_new_frame():
    camera = CameraCapture()
    sample = np.ones((2, 2, 3), dtype=np.uint8)
    with camera._frame_condition:
        camera.is_running = True

    def publish():
        time.sleep(0.01)
        with camera._frame_condition:
            camera.current_frame = sample
            camera.frame_count = 1
            camera._frame_condition.notify_all()

    worker = threading.Thread(target=publish)
    worker.start()
    frame, count = camera.wait_for_frame(last_frame_count=0, timeout=0.5, copy=True)
    worker.join(timeout=1.0)

    assert count == 1
    assert frame is not sample
    assert np.array_equal(frame, sample)
    stale, stale_count = camera.wait_for_frame(last_frame_count=1, timeout=0.01, copy=False)
    assert stale is None
    assert stale_count == 1


def test_camera_open_uses_same_directshow_index_space_as_dashboard():
    import cv2
    import camera_capture as camera_module

    calls = []

    class _Capture:
        def __init__(self, opened):
            self.opened = opened
            self.values = {}

        def isOpened(self):
            return self.opened

        def set(self, prop, value):
            self.values[prop] = value
            return True

        def get(self, prop):
            defaults = {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30,
            }
            return self.values.get(prop, defaults.get(prop, 0))

        def release(self):
            self.opened = False

    def fake_capture(index, backend=None):
        calls.append((index, backend))
        return _Capture(opened=backend == cv2.CAP_DSHOW)

    original_capture = camera_module.cv2.VideoCapture
    original_platform = camera_module.sys.platform
    camera_module.cv2.VideoCapture = fake_capture
    camera_module.sys.platform = "win32"
    try:
        camera = CameraCapture(camera_id=1, width=640, height=480, fps=30)
        assert camera.open() is True
        assert camera.backend == "DirectShow"
        assert calls == [(1, cv2.CAP_DSHOW)]
        camera.stop()
    finally:
        camera_module.cv2.VideoCapture = original_capture
        camera_module.sys.platform = original_platform


def test_camera_detection_preserves_physical_index():
    import cv2
    import dashboard_server as dashboard_module

    class _FakeScanCapture:
        def __init__(self, index, backend):
            self.index = index
            self.backend = backend

        def isOpened(self):
            return self.index == 1

        def read(self):
            if not self.isOpened():
                return False, None
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def get(self, prop):
            values = {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30,
                cv2.CAP_PROP_ZOOM: 0,
            }
            return values.get(prop, 0)

        def release(self):
            pass

    original_capture = cv2.VideoCapture
    original_name_lookup = dashboard_module._get_camera_name_dshow
    cv2.VideoCapture = _FakeScanCapture
    dashboard_module._get_camera_name_dshow = lambda index: f"Camera {index}"
    try:
        cameras = dashboard_module.detect_cameras()
    finally:
        cv2.VideoCapture = original_capture
        dashboard_module._get_camera_name_dshow = original_name_lookup

    assert len(cameras) == 1
    assert cameras[0]["id"] == 1


def test_dashboard_camera_refresh_bypasses_cache():
    import dashboard_server as dashboard_module

    original_cache = dashboard_module._camera_cache
    original_detect = dashboard_module.detect_cameras
    original_find = dashboard_module._find_mocap_processes
    original_process = dashboard_module.mocap_process
    calls = []
    dashboard_module._camera_cache = {
        "data": [{"id": 0, "name": "cached"}],
        "ts": time.time(),
    }
    dashboard_module.detect_cameras = lambda: calls.append(True) or [{"id": 1, "name": "fresh"}]
    dashboard_module._find_mocap_processes = lambda use_cache=True: []
    dashboard_module.mocap_process = None
    try:
        client = dashboard_module.app.test_client()
        cached = client.get("/api/cameras").get_json()
        refreshed = client.get("/api/cameras?refresh=1").get_json()
    finally:
        dashboard_module._camera_cache = original_cache
        dashboard_module.detect_cameras = original_detect
        dashboard_module._find_mocap_processes = original_find
        dashboard_module.mocap_process = original_process

    assert cached["cached"] is True
    assert cached["cameras"][0]["name"] == "cached"
    assert refreshed["cached"] is False
    assert refreshed["cameras"][0]["name"] == "fresh"
    assert len(calls) == 1


def test_runtime_tracking_state_roundtrip_and_staleness(tmp_path):
    path = tmp_path / "tracking_status.json"
    write_tracking_state({
        "running": True,
        "fps": np.float32(12.5),
        "modules": {"body": {"tracker_node_count": np.int64(6)}},
    }, path=path)
    state = read_tracking_state(path=path, max_age_seconds=2.0)
    assert state["running"] is True
    assert state["fps"] == 12.5
    assert state["modules"]["body"]["tracker_node_count"] == 6
    assert state["stale"] is False

    write_tracking_state({"running": True, "timestamp": time.time() - 5.0}, path=path)
    assert read_tracking_state(path=path, max_age_seconds=1.0)["stale"] is True

    write_tracking_state({
        "running": True,
        "pid": 123,
        "fps": 10.0,
        "frame_count": 42,
        "modules": {"body": {"tracker_node_count": 6}},
    }, path=path)
    mark_tracking_stopped(path=path)
    stopped = read_tracking_state(path=path, max_age_seconds=1.0)
    assert stopped["running"] is False
    assert stopped["pid"] is None
    assert stopped["fps"] == 0.0
    assert stopped["frame_count"] == 42
    assert stopped["modules"]["body"]["tracker_node_count"] == 6
    assert stopped["stale"] is False


def test_runtime_tracking_state_retries_windows_reader_lock(tmp_path):
    path = tmp_path / "tracking_status.json"
    original_replace = os.replace
    attempts = []

    def flaky_replace(source, target):
        attempts.append((source, target))
        if len(attempts) < 3:
            raise PermissionError("simulated Windows reader lock")
        return original_replace(source, target)

    with mock.patch.object(runtime_status_module.os, "replace", side_effect=flaky_replace), \
            mock.patch.object(runtime_status_module.time, "sleep") as sleep_mock:
        write_tracking_state({"running": True, "frame_count": 7}, path=path)

    assert len(attempts) == 3
    assert sleep_mock.call_count == 2
    assert read_tracking_state(path=path)["frame_count"] == 7
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    assert not temporary.exists()


def test_frozen_runtime_root_falls_back_to_local_app_data(tmp_path):
    portable_root = tmp_path / "readonly-portable"
    local_app_data = tmp_path / "local-app-data"
    portable_root.mkdir()
    runtime_paths_module.runtime_root.cache_clear()
    try:
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "executable", str(portable_root / "app.exe")), \
                mock.patch.object(runtime_paths_module, "_directory_is_writable", return_value=False), \
                mock.patch.dict(os.environ, {"LOCALAPPDATA": str(local_app_data)}, clear=False):
            selected = runtime_paths_module.runtime_root()
    finally:
        runtime_paths_module.runtime_root.cache_clear()

    assert selected == (local_app_data / "YOLO26n-VRC-Mocap").resolve()
    assert selected.is_dir()


def test_dashboard_tracking_status_exposes_runtime_details():
    import dashboard_server as dashboard_module

    class _RunningProcess:
        pid = 123

        def poll(self):
            return None

    original_reader = dashboard_module.read_tracking_state
    original_process = dashboard_module.mocap_process
    dashboard_module.read_tracking_state = lambda max_age_seconds=None: {
        "running": True,
        "pid": 123,
        "stale": False,
        "age_seconds": 0.1,
        "fps": 12.0,
        "frame_count": 100,
        "modules": {
            "body": {
                "pose_node_count": 20,
                "tracker_node_count": 6,
                "hmd_height_state": {"posture": "seated", "actual_height": 0.82},
            }
        },
        "hmd": {"valid": True, "actual_height": 0.82},
    }
    dashboard_module.mocap_process = _RunningProcess()
    try:
        response = dashboard_module.app.test_client().get("/api/tracking/status")
        payload = response.get_json()
    finally:
        dashboard_module.read_tracking_state = original_reader
        dashboard_module.mocap_process = original_process

    assert response.status_code == 200
    assert payload["runtime_available"] is True
    assert payload["body_details"]["tracker_node_count"] == 6
    assert payload["body_details"]["hmd_height_state"]["posture"] == "seated"
    assert payload["hmd"]["actual_height"] == 0.82


def test_dashboard_tracking_status_accepts_live_external_mocap_state():
    import dashboard_server as dashboard_module

    original_reader = dashboard_module.read_tracking_state
    original_process = dashboard_module.mocap_process
    original_alive = dashboard_module._runtime_state_process_alive
    dashboard_module.read_tracking_state = lambda max_age_seconds=None: {
        "running": True,
        "pid": 456,
        "stale": False,
        "age_seconds": 0.1,
        "fps": 11.0,
        "frame_count": 50,
        "parallel_inference": True,
        "modules": {"body": {"pose_node_count": 20}},
        "hmd": None,
    }
    dashboard_module.mocap_process = None
    dashboard_module._runtime_state_process_alive = lambda state: state.get("pid") == 456
    try:
        response = dashboard_module.app.test_client().get("/api/tracking/status")
        payload = response.get_json()
    finally:
        dashboard_module.read_tracking_state = original_reader
        dashboard_module.mocap_process = original_process
        dashboard_module._runtime_state_process_alive = original_alive

    assert response.status_code == 200
    assert payload["runtime_available"] is True
    assert payload["parallel_inference"] is True
    assert payload["body_details"]["pose_node_count"] == 20


def test_dashboard_mocap_status_discovers_external_process_when_locally_stopped():
    import dashboard_server as dashboard_module

    original_process = dashboard_module.mocap_process
    original_status = dict(dashboard_module.mocap_status)
    original_find = dashboard_module._find_mocap_processes
    dashboard_module.mocap_process = None
    dashboard_module.mocap_status.update({
        "running": False,
        "pid": None,
        "start_time": None,
        "starting": False,
    })
    dashboard_module._find_mocap_processes = lambda use_cache=True: [{
        "pid": 789,
        "cmdline": "python bin/main.py",
        "managed": False,
    }]
    try:
        response = dashboard_module.app.test_client().get("/api/mocap/status")
        payload = response.get_json()
    finally:
        dashboard_module.mocap_process = original_process
        dashboard_module.mocap_status.clear()
        dashboard_module.mocap_status.update(original_status)
        dashboard_module._find_mocap_processes = original_find

    assert response.status_code == 200
    assert payload["running"] is True
    assert payload["managed"] is False
    assert payload["external_pid"] == 789


def test_dashboard_inference_status_prefers_structured_runtime_state():
    import dashboard_server as dashboard_module

    original_reader = dashboard_module.read_tracking_state
    original_alive = dashboard_module._runtime_state_process_alive
    original_log_reader = dashboard_module._read_text_tail
    dashboard_module.read_tracking_state = lambda max_age_seconds=None: {
        "running": True,
        "pid": 321,
        "stale": False,
        "modules": {
            "body": {
                "enabled": True,
                "initialized": True,
                "pose_node_count": 20,
                "tracker_node_count": 6,
                "streamer_send_count": 45,
                "streamer_error_count": 0,
                "stream_queued": True,
            },
            "face": {"enabled": True, "initialized": True},
            "hand": {"enabled": False, "initialized": True},
        },
    }
    dashboard_module._runtime_state_process_alive = lambda state: True

    def fail_log_read(*args, **kwargs):
        raise AssertionError("structured state should avoid reading the log")

    dashboard_module._read_text_tail = fail_log_read
    try:
        response = dashboard_module.app.test_client().get("/api/mocap/inference_status")
        payload = response.get_json()
    finally:
        dashboard_module.read_tracking_state = original_reader
        dashboard_module._runtime_state_process_alive = original_alive
        dashboard_module._read_text_tail = original_log_reader

    assert response.status_code == 200
    assert payload["source"] == "runtime_state"
    assert payload["body"] is True
    assert payload["face"] is True
    assert payload["hand"] is False
    assert payload["body_pose_nodes"] == 20
    assert payload["body_tracker_messages"] == 6
    assert payload["body_stream_sent"] == 45


def test_hmd_anchor_aligns_unity_pose_without_camera_root_offset():
    coord = CoordinateTransformer()
    unity_pose = {
        "Head": [0.0, 1.4, 0.0],
        "Hips": [0.0, 0.6, 0.0],
        "LeftFoot": [-0.1, 0.0, 0.0],
        "RightFoot": [0.1, 0.0, 0.0],
    }

    hmd_a = [2.0, 1.7, -0.4]
    hmd_b = [2.0, 1.9, -0.4]
    anchored_a = coord.compute_hmd_aligned_unity_positions(unity_pose, hmd_a)
    anchored_b = coord.compute_hmd_aligned_unity_positions(unity_pose, hmd_b)

    assert anchored_a["Head"] == hmd_a
    assert anchored_b["Head"] == hmd_b

    original_head_to_hips = unity_pose["Head"][1] - unity_pose["Hips"][1]
    anchored_head_to_hips = anchored_a["Head"][1] - anchored_a["Hips"][1]
    assert abs(original_head_to_hips - anchored_head_to_hips) < 1e-9

    height_delta = anchored_b["Hips"][1] - anchored_a["Hips"][1]
    assert abs(height_delta - 0.2) < 1e-9


def test_coordinate_conversion_suppresses_known_gimbal_lock_warning():
    import warnings

    coord = CoordinateTransformer()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        converted = coord.camera_to_unity_rotations({
            "Head": [np.pi / 2.0, 0.0, 0.0],
        })

    assert "Head" in converted
    assert all("Gimbal lock" not in str(item.message) for item in captured)


def test_osc_hmd_relative_offsets_do_not_double_apply_hmd_height():
    coord = CoordinateTransformer()
    unity_pose = {
        "Head": [0.0, 1.4, 0.0],
        "Neck": [0.0, 1.2, 0.0],
        "Hips": [0.0, 0.6, 0.0],
        "LeftFoot": [-0.1, 0.0, 0.2],
        "RightFoot": [0.1, 0.0, 0.2],
    }

    local_a = coord.compute_osc_offsets(unity_pose, None, None)
    local_b = coord.compute_osc_offsets(unity_pose, None, None)

    assert local_a["Hips"] == local_b["Hips"]
    assert local_a["LeftFoot"] == local_b["LeftFoot"]
    assert local_a["RightFoot"] == local_b["RightFoot"]

    aligned_a = coord.compute_osc_offsets(unity_pose, [0.0, 1.7, 0.0], None)
    aligned_b = coord.compute_osc_offsets(unity_pose, [0.4, 1.9, -0.2], None)
    _assert_close_list(aligned_a["Hips"], aligned_b["Hips"])
    _assert_close_list(aligned_a["LeftFoot"], aligned_b["LeftFoot"])
    _assert_close_list(aligned_a["RightFoot"], aligned_b["RightFoot"])
    _assert_close_list(aligned_a["Head"], [0.0, 0.0, 0.0])
    _assert_close_list(aligned_b["Head"], [0.0, 0.0, 0.0])


def test_hmd_bound_body_height_scales_to_seated_hmd_eye_height():
    coord = CoordinateTransformer()
    unity_pose = {
        "Head": [0.0, 1.45, 0.0],
        "Neck": [0.0, 1.25, 0.0],
        "ChestTracker": [0.0, 1.0, 0.0],
        "HipsTracker": [0.0, 0.65, 0.0],
        "LeftKneeTracker": [-0.12, 0.35, 0.0],
        "RightKneeTracker": [0.12, 0.35, 0.0],
        "LeftFootTracker": [-0.1, 0.0, 0.2],
        "RightFootTracker": [0.1, 0.0, 0.2],
    }

    standing = coord.compute_osc_offsets(unity_pose, [0.0, 1.7, 0.0], None, 1.7)
    crouched = coord.compute_osc_offsets(unity_pose, [0.0, 1.7, 0.0], None, 0.75)

    _assert_close_list(standing["Head"], [0.0, 0.0, 0.0])
    _assert_close_list(crouched["Head"], [0.0, 0.0, 0.0])
    assert abs(crouched["LeftFootTracker"][1]) < abs(standing["LeftFootTracker"][1])
    assert abs(crouched["HipsTracker"][1]) < abs(standing["HipsTracker"][1])
    assert abs(crouched["LeftFootTracker"][1]) <= 0.72 + 1e-6
    assert abs(crouched["RightFootTracker"][1]) <= 0.72 + 1e-6
    assert crouched["LeftKneeTracker"][2] < standing["LeftKneeTracker"][2]
    assert crouched["RightKneeTracker"][2] < standing["RightKneeTracker"][2]


def test_hmd_height_constraint_requires_explicit_floor_height():
    coord = CoordinateTransformer()
    unity_pose = {
        "Head": [0.0, 1.45, 0.0],
        "HipsTracker": [0.0, 0.65, 0.0],
        "LeftFootTracker": [-0.1, 0.0, 0.2],
    }

    unknown_origin = coord.compute_osc_offsets(unity_pose, [0.0, 0.75, 0.0], None)
    floor_origin = coord.compute_osc_offsets(unity_pose, [0.0, 0.75, 0.0], None, 0.75)

    assert abs(unknown_origin["LeftFootTracker"][1]) > 1.0
    assert abs(floor_origin["LeftFootTracker"][1]) <= 0.72 + 1e-6


def test_hmd_actual_height_requires_floor_like_origin():
    assert _actual_height_from_position([0.0, 1.12, 0.0], "floor") == (1.12, "floor")
    assert _actual_height_from_position([0.0, 1.12, 0.0], "steamvr") == (1.12, "steamvr")
    assert _actual_height_from_position([0.0, 1.12, 0.0], "stage") == (1.12, "stage")
    height, reference = _actual_height_from_position([0.0, 1.12, 0.0], "device")
    assert height is None
    assert reference == "device"

    provider = HMDDataProvider()
    provider._update_pose([0.0, 1.08, 0.0], tracking_origin="floor")
    pose = provider.get_pose()
    assert pose.actual_height == 1.08
    assert pose.height_reference == "floor"

    provider._update_pose([0.0, 1.08, 0.0], tracking_origin="unknown")
    assert provider.get_pose().actual_height is None


def test_openvr_unity_rotation_conversion_preserves_identity():
    identity = np.eye(3)
    _assert_close_list(
        _openvr_rotation_to_unity(identity).reshape(-1).tolist(),
        identity.reshape(-1).tolist(),
    )
    _assert_close_list(
        _sender_openvr_rotation_to_unity(identity, np).reshape(-1).tolist(),
        identity.reshape(-1).tolist(),
    )

    coord = CoordinateTransformer()
    _assert_close_list(coord.euler_to_quat_openvr([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 1.0])


def test_openvr_pose_stales_out_and_clears_height():
    provider = OpenVRProvider()
    provider._update_pose(
        [0.0, 1.7, 0.0],
        quaternion=[1.0, 0.0, 0.0, 0.0],
        tracking_origin="steamvr",
    )
    with provider._lock:
        provider._pose.timestamp = time.time() - provider._timeout - 0.1

    pose = provider.get_pose()
    assert pose.valid is False
    assert pose.actual_height is None
    status = provider.get_status_extra()
    assert status["stale"] is True


def test_hmd_provider_without_first_packet_is_stale():
    for provider in (OpenVRProvider(), RemoteHMDProvider(), SDKUDPProvider()):
        assert provider.get_pose().valid is False
        assert provider.get_status_extra()["stale"] is True


def test_explicitly_invalid_hmd_packet_is_stale():
    provider = RemoteHMDProvider()
    provider._parse_json_packet(json.dumps({
        "source": "test",
        "tracking_origin": "floor",
        "position": [0.0, 1.65, 0.0],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
        "valid": True,
    }).encode("utf-8"))
    assert provider.get_pose().valid is True

    provider._parse_json_packet(json.dumps({
        "source": "test",
        "tracking_origin": "floor",
        "valid": False,
    }).encode("utf-8"))
    assert provider.get_pose().valid is False
    assert provider.get_status_extra()["stale"] is True


def test_polling_hmd_providers_stale_out_and_clear_height():
    for provider in (PicoStreamingProvider(), QuestLinkProvider()):
        provider._update_pose(
            [0.0, 1.7, 0.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            tracking_origin="floor",
        )
        with provider._lock:
            provider._pose.timestamp = time.time() - provider._timeout - 0.1
        pose = provider.get_pose()
        assert pose.valid is False
        assert pose.actual_height is None


def test_quest_link_failed_openvr_does_not_mask_http_fallback():
    import types
    import hmd_provider as hmd_module

    stopped = []

    class _FailedOpenVR:
        def start(self):
            return False

        def stop(self):
            stopped.append(True)

    missing = object()
    saved_openvr_module = sys.modules.get("openvr", missing)
    original_provider = hmd_module.OpenVRProvider
    sys.modules["openvr"] = types.SimpleNamespace()
    hmd_module.OpenVRProvider = _FailedOpenVR
    provider = QuestLinkProvider()
    provider._poll_loop = lambda: None
    try:
        assert provider.start() is True
        assert provider._openvr_fallback is None
        assert stopped == [True]
    finally:
        provider.stop()
        hmd_module.OpenVRProvider = original_provider
        if saved_openvr_module is missing:
            sys.modules.pop("openvr", None)
        else:
            sys.modules["openvr"] = saved_openvr_module


def test_vrchat_osc_rejects_non_finite_pose_values():
    provider = VRChatOSCProvider()
    provider._on_head_position("/tracking/trackers/head/position", 0.1, 1.7, -0.2)
    pose = provider.get_pose()
    assert pose.valid is True
    assert pose.actual_height is None
    assert pose.tracking_origin == "vrchat_osc"

    provider._on_head_position(
        "/tracking/trackers/head/position", float("nan"), 1.7, -0.2
    )
    assert provider.get_pose().valid is False
    assert "invalid" in provider.get_status_extra()["last_error"].lower()


def test_malformed_hmd_json_invalidates_previous_pose():
    sdk = SDKUDPProvider()
    sdk._parse_json_packet(json.dumps({
        "position": [0.0, 1.6, 0.0],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
        "tracking_origin": "floor",
    }).encode("utf-8"))
    assert sdk.get_pose().valid is True
    sdk._parse_json_packet(b'{"valid":true}')
    assert sdk.get_pose().valid is False
    assert sdk.get_pose().actual_height is None
    sdk._parse_json_packet(b"{broken-json")
    assert sdk.get_pose().valid is False

    remote = RemoteHMDProvider()
    remote._parse_json_packet(json.dumps({
        "position": [0.0, 1.6, 0.0],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
        "tracking_origin": "floor",
    }).encode("utf-8"))
    assert remote.get_pose().valid is True
    assert remote._parse_json_packet(b'{"valid":true}') is False
    assert remote.get_pose().valid is False
    assert remote.get_pose().actual_height is None
    assert remote._parse_json_packet(b"{broken-json") is False


def test_sender_invalid_packet_clears_remote_height():
    class _CaptureSocket:
        def __init__(self):
            self.packet = None

        def sendto(self, packet, target):
            self.packet = packet

    socket_capture = _CaptureSocket()
    assert _send_invalid_pose(
        socket_capture,
        ("127.0.0.1", 4243),
        "json",
        "openvr",
        8,
        "steamvr",
    ) is True
    payload = json.loads(socket_capture.packet.decode("utf-8"))
    assert payload["valid"] is False
    assert payload["sequence"] == 8

    provider = RemoteHMDProvider()
    provider._update_pose([0.0, 1.7, 0.0], tracking_origin="floor")
    provider._parse_json_packet(socket_capture.packet)
    assert provider.get_pose().valid is False
    assert provider.get_pose().actual_height is None

    socket_capture.packet = None
    assert _send_invalid_pose(
        socket_capture,
        ("127.0.0.1", 4243),
        "legacy",
        "openvr",
        9,
        "unknown",
    ) is False
    assert socket_capture.packet is None


def test_legacy_hmd_packets_never_claim_floor_relative_height():
    pico = PicoXRoboProvider()
    pico._update_pose_from_quat(
        [0.0, 1.75, 0.0], 0.0, 0.0, 0.0, 1.0, tracking_origin="floor"
    )
    pico._update_pose_from_quat([0.0, 1.12, 0.0], 0.0, 0.0, 0.0, 1.0)
    assert pico.get_pose().tracking_origin == "unknown"
    assert pico.get_pose().actual_height is None

    quest = QuestUDPProvider()
    quest._update_pose_from_quat(
        [0.0, 1.75, 0.0], 0.0, 0.0, 0.0, 1.0, tracking_origin="floor"
    )
    quest._update_pose_from_quat([0.0, 1.08, 0.0], 0.0, 0.0, 0.0, 1.0)
    assert quest.get_pose().tracking_origin == "unknown"
    assert quest.get_pose().actual_height is None

    sdk = SDKUDPProvider(tracking_origin="floor")
    sdk._update_pose_from_wxyz(
        [0.0, 1.75, 0.0], [1.0, 0.0, 0.0, 0.0], tracking_origin="floor"
    )
    sdk._update_pose_from_wxyz([0.0, 1.15, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert sdk.get_pose().tracking_origin == "unknown"
    assert sdk.get_pose().actual_height is None

    remote = RemoteHMDProvider()
    remote._update_pose_from_quat(
        [0.0, 1.75, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        tracking_origin="floor",
    )
    remote._update_pose_from_quat(
        [0.0, 1.18, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        tracking_origin="unknown",
    )
    assert remote.get_pose().actual_height is None

    with remote._lock:
        remote._pose.timestamp = time.time() - remote._timeout - 0.1
    assert remote.get_status_extra()["stale"] is True


def test_sdk_json_height_requires_floor_origin_or_explicit_height():
    provider = SDKUDPProvider(tracking_origin="floor")
    base = {
        "source": "quest_sdk",
        "position": [0.1, 1.1, -0.2],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
        "valid": True,
    }

    assert provider._parse_json_packet(json.dumps(base).encode("utf-8")) is True
    pose = provider.get_pose()
    assert pose.tracking_origin == "unknown"
    assert pose.actual_height is None

    floor_packet = dict(base, tracking_origin="floor")
    provider._parse_json_packet(json.dumps(floor_packet).encode("utf-8"))
    pose = provider.get_pose()
    assert pose.actual_height == 1.1
    assert pose.height_reference == "floor"

    explicit_packet = dict(
        base,
        tracking_origin="device",
        actual_height=0.82,
        position=[0.1, 0.05, -0.2],
    )
    provider._parse_json_packet(json.dumps(explicit_packet).encode("utf-8"))
    pose = provider.get_pose()
    assert pose.actual_height == 0.82
    assert pose.height_reference == "explicit"

    provider._parse_json_packet(json.dumps({
        "source": "quest_sdk",
        "tracking_origin": "device",
        "valid": False,
    }).encode("utf-8"))
    pose = provider.get_pose()
    assert pose.valid is False
    assert pose.actual_height is None


def test_standing_height_auto_learning_only_raises_on_stable_samples():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.65,
        "auto_learn_standing_height": True,
        "auto_learn_window": 5,
        "auto_learn_stability_tolerance": 0.02,
    })

    for _ in range(12):
        assert abs(coord._resolve_standing_eye_height(1.25) - 1.65) < 1e-6
    assert abs(coord._resolve_standing_eye_height(1.95) - 1.65) < 1e-6

    for height in (1.76, 1.86, 1.75, 1.87, 1.77, 1.85):
        assert abs(coord._resolve_standing_eye_height(height) - 1.65) < 1e-6

    coord._resolve_standing_eye_height(1.3)  # Clear unstable standing candidates.
    learned = 1.65
    for height in (1.78, 1.781, 1.779, 1.782, 1.78):
        learned = coord._resolve_standing_eye_height(height)
    assert abs(learned - 1.78) < 0.005


def test_standing_height_auto_learning_can_lower_for_stable_standing_pose():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.65,
        "auto_learn_standing_height": True,
        "auto_learn_allow_downward": True,
        "auto_learn_downward_max_delta": 0.25,
        "auto_learn_standing_pose_threshold": 0.8,
        "auto_learn_window": 5,
        "auto_learn_stability_tolerance": 0.02,
    })

    for _ in range(5):
        learned = coord._resolve_standing_eye_height(1.40, 1.0)

    assert abs(learned - 1.40) < 0.005
    assert abs(coord._resolve_standing_eye_height(0.95, 1.0) - 1.40) < 0.005


def test_standing_height_auto_learning_does_not_lower_for_crouched_pose():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.65,
        "auto_learn_standing_height": True,
        "auto_learn_allow_downward": True,
        "auto_learn_downward_max_delta": 0.25,
        "auto_learn_standing_pose_threshold": 0.8,
        "auto_learn_window": 5,
    })

    for _ in range(8):
        learned = coord._resolve_standing_eye_height(1.40, 0.2)

    assert abs(learned - 1.65) < 0.005


def test_seated_solver_grounds_ankles_and_keeps_knees_bent():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.65,
        "auto_learn_standing_height": False,
        "floor_clearance": 0.03,
    })
    pose = {
        "Head": [0.0, 1.65, 0.0],
        "ChestTracker": [0.0, 1.15, 0.0],
        "HipsTracker": [0.0, 0.85, 0.0],
        "LeftKneeTracker": [-0.15, 0.48, 0.04],
        "RightKneeTracker": [0.15, 0.48, 0.04],
        "LeftUpperLeg": [-0.15, 0.48, 0.04],
        "RightUpperLeg": [0.15, 0.48, 0.04],
        "LeftLowerLeg": [-0.15, 0.08, 0.08],
        "RightLowerLeg": [0.15, 0.08, 0.08],
        "LeftFootTracker": [-0.15, 0.03, 0.18],
        "RightFootTracker": [0.15, 0.03, 0.18],
    }
    actual_height = 0.75
    output = coord.compute_osc_offsets(
        pose, [0.0, actual_height, 0.0], None, actual_height
    )
    world_y = {bone: actual_height + value[1] for bone, value in output.items()}

    assert abs(world_y["LeftFootTracker"] - 0.03) < 1e-6
    assert abs(world_y["RightFootTracker"] - 0.03) < 1e-6
    assert abs(world_y["LeftLowerLeg"] - 0.03) < 1e-6
    assert abs(world_y["RightLowerLeg"] - 0.03) < 1e-6
    assert world_y["LeftKneeTracker"] > world_y["LeftLowerLeg"] + 0.15
    assert world_y["RightKneeTracker"] > world_y["RightLowerLeg"] + 0.15
    assert 0.40 <= world_y["HipsTracker"] <= 0.50


def test_steamvr_explicit_height_rebuilds_floor_relative_hmd_y():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.65,
        "auto_learn_standing_height": False,
        "floor_clearance": 0.03,
    })
    pose = {
        "Head": [0.0, 1.65, 0.0],
        "HipsTracker": [0.0, 0.85, 0.0],
        "LeftKneeTracker": [-0.15, 0.45, 0.0],
        "RightKneeTracker": [0.15, 0.45, 0.0],
        "LeftFootTracker": [-0.15, 0.03, 0.15],
        "RightFootTracker": [0.15, 0.03, 0.15],
    }
    device_local_hmd = [0.2, 0.05, -0.1]
    actual_height = 0.82

    steamvr = coord.compute_steamvr_positions(
        pose, device_local_hmd, [1.0, 0.0, 0.0, 0.0], actual_height
    )
    osc = coord.compute_osc_offsets(
        pose, device_local_hmd, [1.0, 0.0, 0.0, 0.0], actual_height
    )

    assert abs(steamvr["Head"][1] - actual_height) < 1e-6
    assert abs(steamvr["LeftFootTracker"][1] - 0.03) < 1e-6
    assert abs(actual_height + osc["LeftFootTracker"][1] - 0.03) < 1e-6


def test_osc_explicit_height_ignores_device_local_hmd_y():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.65,
        "standing_eye_height_calibrated": True,
        "auto_learn_standing_height": False,
        "floor_clearance": 0.03,
    })
    pose = {
        "Head": [0.0, 1.65, 0.0],
        "HipsTracker": [0.0, 0.85, 0.0],
        "LeftKneeTracker": [-0.15, 0.45, 0.0],
        "RightKneeTracker": [0.15, 0.45, 0.0],
        "LeftFootTracker": [-0.15, 0.03, 0.15],
        "RightFootTracker": [0.15, 0.03, 0.15],
    }
    local_hmd_position = [0.2, 0.05, -0.1]
    actual_height = 0.82
    osc = coord.compute_osc_offsets(
        pose, local_hmd_position, [1.0, 0.0, 0.0, 0.0], actual_height
    )

    assert abs(actual_height + osc["LeftFootTracker"][1] - 0.03) < 1e-6
    assert abs(actual_height + osc["RightFootTracker"][1] - 0.03) < 1e-6
    assert 0.40 <= actual_height + osc["HipsTracker"][1] <= 0.50


def test_hmd_height_sequence_keeps_trackers_above_floor_and_recovers_pose():
    config = {
        "standing_eye_height": 1.65,
        "standing_eye_height_calibrated": True,
        "auto_learn_standing_height": False,
        "foot_calibration_samples": 3,
        "floor_clearance": 0.03,
    }
    osc_coord = CoordinateTransformer(hmd_height_scaling=config)
    steamvr_coord = CoordinateTransformer(hmd_height_scaling=config)
    pose = {
        "Head": [0.0, 1.65, 0.0],
        "ChestTracker": [0.0, 1.15, 0.0],
        "HipsTracker": [0.0, 0.85, 0.0],
        "LeftKneeTracker": [-0.15, 0.48, 0.04],
        "RightKneeTracker": [0.15, 0.48, 0.04],
        "LeftUpperLeg": [-0.15, 0.48, 0.04],
        "RightUpperLeg": [0.15, 0.48, 0.04],
        "LeftLowerLeg": [-0.15, 0.03, 0.12],
        "RightLowerLeg": [0.15, 0.03, 0.12],
        "LeftFootTracker": [-0.15, 0.03, 0.18],
        "RightFootTracker": [0.15, 0.03, 0.18],
        "LeftElbowTracker": [-0.35, 1.05, 0.03],
        "RightElbowTracker": [0.35, 1.05, 0.03],
    }
    tracker_names = (
        "HipsTracker",
        "ChestTracker",
        "LeftFootTracker",
        "RightFootTracker",
        "LeftKneeTracker",
        "RightKneeTracker",
        "LeftElbowTracker",
        "RightElbowTracker",
    )
    heights = (1.65, 1.65, 1.65, 1.50, 1.30, 1.10, 0.82, 1.10, 1.30, 1.50, 1.65)
    snapshots = []

    for index, height in enumerate(heights):
        hmd_position = [0.2 + index * 0.03, height, -0.1 - index * 0.02]
        osc = osc_coord.compute_osc_offsets(
            pose, hmd_position, [1.0, 0.0, 0.0, 0.0], height
        )
        steamvr = steamvr_coord.compute_steamvr_positions(
            pose, hmd_position, [1.0, 0.0, 0.0, 0.0], height
        )
        state = osc_coord.hmd_height_state
        world_y = {name: height + osc[name][1] for name in tracker_names}

        _assert_close_list(osc["Head"], [0.0, 0.0, 0.0])
        assert all(np.isfinite(osc[name]).all() for name in tracker_names)
        assert all(value >= 0.03 - 1e-6 for value in world_y.values())
        assert abs(world_y["LeftFootTracker"] - 0.03) < 1e-6
        assert abs(world_y["RightFootTracker"] - 0.03) < 1e-6
        assert abs(steamvr["LeftFootTracker"][1] - 0.03) < 1e-6
        assert abs(steamvr["RightFootTracker"][1] - 0.03) < 1e-6
        assert world_y["LeftKneeTracker"] > world_y["LeftFootTracker"] + 0.15
        assert world_y["RightKneeTracker"] > world_y["RightFootTracker"] + 0.15
        snapshots.append((height, state["posture"], world_y, osc))

    descent_hips = [snapshots[i][2]["HipsTracker"] for i in (2, 3, 4, 5, 6)]
    assert all(
        descent_hips[i] >= descent_hips[i + 1] - 1e-6
        for i in range(len(descent_hips) - 1)
    )
    assert snapshots[2][1] == "standing"
    assert snapshots[4][1] == "crouch"
    assert snapshots[6][1] == "seated"
    assert snapshots[-1][1] == "standing"

    for down_index, up_index in ((3, 9), (4, 8), (5, 7)):
        assert abs(
            snapshots[down_index][2]["HipsTracker"]
            - snapshots[up_index][2]["HipsTracker"]
        ) < 1e-6
        assert abs(
            snapshots[down_index][3]["LeftKneeTracker"][2]
            - snapshots[up_index][3]["LeftKneeTracker"][2]
        ) < 1e-6
        for tracker_name in tracker_names:
            _assert_close_list(
                snapshots[down_index][3][tracker_name],
                snapshots[up_index][3][tracker_name],
            )


def test_hmd_height_constraint_prefers_actual_height_over_position_y():
    coord = CoordinateTransformer()
    unity_pose = {
        "Head": [0.0, 1.45, 0.0],
        "HipsTracker": [0.0, 0.65, 0.0],
        "LeftFootTracker": [-0.1, 0.0, 0.2],
    }

    position_y_only = coord.compute_osc_offsets(
        unity_pose,
        [0.0, 1.8, 0.0],
        None,
        None,
    )
    actual_height = coord.compute_osc_offsets(unity_pose, [0.0, 1.8, 0.0], None, 1.05)

    assert abs(actual_height["LeftFootTracker"][1]) < abs(position_y_only["LeftFootTracker"][1])
    assert abs(actual_height["LeftFootTracker"][1]) <= 1.03


def test_osc_and_steamvr_share_explicit_floor_height_base_for_local_pose():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.65,
        "standing_eye_height_calibrated": True,
        "auto_learn_standing_height": False,
        "floor_clearance": 0.03,
    })
    pose = {
        "Head": [0.0, 1.65, 0.0],
        "ChestTracker": [0.0, 1.15, 0.0],
        "HipsTracker": [0.0, 0.85, 0.0],
        "LeftKneeTracker": [-0.15, 0.48, 0.04],
        "RightKneeTracker": [0.15, 0.48, 0.04],
        "LeftFootTracker": [-0.15, 0.03, 0.15],
        "RightFootTracker": [0.15, 0.03, 0.15],
    }
    local_hmd = [0.35, 0.05, -0.2]
    actual_height = 0.82

    osc = coord.compute_osc_offsets(
        pose, local_hmd, [1.0, 0.0, 0.0, 0.0], actual_height
    )
    steamvr = coord.compute_steamvr_positions(
        pose, local_hmd, [1.0, 0.0, 0.0, 0.0], actual_height
    )

    for bone in (
        "Head", "ChestTracker", "HipsTracker", "LeftKneeTracker",
        "RightKneeTracker", "LeftFootTracker", "RightFootTracker",
    ):
        assert np.isfinite(osc[bone]).all()
        assert np.isfinite(steamvr[bone]).all()
    assert abs(actual_height + osc["LeftFootTracker"][1] - 0.03) < 1e-6
    assert abs(actual_height + osc["RightFootTracker"][1] - 0.03) < 1e-6
    assert abs(steamvr["LeftFootTracker"][1] - 0.03) < 1e-6
    assert abs(steamvr["RightFootTracker"][1] - 0.03) < 1e-6
    assert 0.40 <= actual_height + osc["HipsTracker"][1] <= 0.50
    assert 0.40 <= steamvr["HipsTracker"][1] <= 0.50


def test_hmd_height_solver_retargets_crouch_and_seated_pose():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.65,
        "auto_learn_standing_height": False,
        "floor_clearance": 0.03,
    })
    unity_pose = {
        "Head": [0.0, 1.45, 0.0],
        "ChestTracker": [0.0, 1.0, 0.0],
        "HipsTracker": [0.0, 0.65, 0.0],
        "LeftKneeTracker": [-0.12, 0.35, 0.0],
        "RightKneeTracker": [0.12, 0.35, 0.0],
        "LeftFootTracker": [-0.1, 0.0, 0.2],
        "RightFootTracker": [0.1, 0.0, 0.2],
    }

    standing = coord.compute_osc_offsets(unity_pose, [0.0, 1.65, 0.0], None, 1.65)
    crouch = coord.compute_osc_offsets(unity_pose, [0.0, 1.65, 0.0], None, 1.10)
    crouch_state = coord.hmd_height_state
    seated = coord.compute_osc_offsets(unity_pose, [0.0, 1.65, 0.0], None, 0.75)
    seated_state = coord.hmd_height_state

    assert crouch_state["posture"] == "crouch"
    assert seated_state["posture"] == "seated"
    _assert_close_list(standing["Head"], [0.0, 0.0, 0.0])
    _assert_close_list(seated["Head"], [0.0, 0.0, 0.0])

    standing_hip_floor = 1.65 + standing["HipsTracker"][1]
    crouch_hip_floor = 1.10 + crouch["HipsTracker"][1]
    seated_hip_floor = 0.75 + seated["HipsTracker"][1]
    assert crouch_hip_floor < standing_hip_floor
    assert seated_hip_floor < crouch_hip_floor

    assert 0.03 - 1e-6 <= 0.75 + seated["LeftFootTracker"][1] <= 0.06
    assert 0.03 - 1e-6 <= 0.75 + seated["RightFootTracker"][1] <= 0.06
    assert crouch["LeftKneeTracker"][2] < standing["LeftKneeTracker"][2]
    assert seated["LeftKneeTracker"][2] <= crouch["LeftKneeTracker"][2]
    assert seated["RightKneeTracker"][2] <= crouch["RightKneeTracker"][2]


def test_hmd_height_solver_keeps_sitting_from_being_learned_as_standing():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.65,
        "auto_learn_standing_height": True,
    })
    unity_pose = {
        "Head": [0.0, 1.45, 0.0],
        "HipsTracker": [0.0, 0.65, 0.0],
        "LeftFootTracker": [-0.1, 0.0, 0.2],
    }

    coord.compute_osc_offsets(unity_pose, [0.0, 0.75, 0.0], None, 0.75)
    state = coord.hmd_height_state
    assert state["posture"] == "seated"
    assert abs(state["standing_eye_height"] - 1.65) < 1e-6
    assert state["height_ratio"] < 0.5


def test_hmd_height_solver_preserves_calibrated_standing_height():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.72,
        "standing_eye_height_calibrated": True,
        "auto_learn_standing_height": True,
    })
    unity_pose = {
        "Head": [0.0, 1.45, 0.0],
        "HipsTracker": [0.0, 0.65, 0.0],
        "LeftFootTracker": [-0.1, 0.0, 0.2],
        "RightFootTracker": [0.1, 0.0, 0.2],
    }

    coord.compute_osc_offsets(unity_pose, [0.0, 1.92, 0.0], None, 1.92)
    state = coord.hmd_height_state
    assert abs(state["standing_eye_height"] - 1.72) < 1e-6


def test_hmd_height_solver_grounds_standing_feet_to_actual_floor():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.65,
        "standing_eye_height_calibrated": True,
        "foot_calibration_samples": 3,
        "floor_clearance": 0.03,
    })
    unity_pose = {
        "Head": [0.0, 1.45, 0.0],
        "HipsTracker": [0.0, 0.65, 0.0],
        "LeftFootTracker": [-0.1, 0.0, 0.2],
        "RightFootTracker": [0.1, 0.0, 0.2],
    }

    for _ in range(3):
        output = coord.compute_osc_offsets(unity_pose, [0.0, 1.65, 0.0], None, 1.65)

    assert abs((1.65 + output["LeftFootTracker"][1]) - 0.03) < 1e-6
    assert abs((1.65 + output["RightFootTracker"][1]) - 0.03) < 1e-6
    assert coord.hmd_height_state["foot_reference_ready"] is True


def test_dashboard_api_provider_copies_actual_height_and_invalidates_stale_data():
    provider = DashboardAPIProvider()
    provider._apply_status_payload({
        "running": True,
        "provider": "remote",
        "pose": {
            "valid": True,
            "position": [0.2, 1.1, -0.3],
            "rotation": [1.0, 2.0, 3.0],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "tracking_origin": "floor",
            "actual_height": 1.1,
            "height_reference": "floor",
        },
    })
    pose = provider.get_pose()
    assert pose.valid is True
    assert pose.position == [0.2, 1.1, -0.3]
    assert pose.actual_height == 1.1
    assert pose.height_reference == "floor"

    provider._apply_status_payload({"running": False, "provider": "none", "pose": None})
    pose = provider.get_pose()
    assert pose.valid is False
    assert pose.actual_height is None

    provider._apply_status_payload({
        "running": True,
        "provider": "remote",
        "pose": {
            "valid": True,
            "position": [0.0, 1.2, 0.0],
            "tracking_origin": "unknown",
            "actual_height": float("nan"),
        },
    })
    assert provider.get_pose().actual_height is None

    provider._apply_status_payload({
        "running": True,
        "provider": "remote",
        "pose": {
            "valid": True,
            "position": [float("inf"), 1.2, 0.0],
            "tracking_origin": "floor",
            "actual_height": 1.2,
        },
    })
    assert provider.get_pose().valid is False


def test_mocap_uses_dashboard_bridge_when_launched_by_dashboard():
    import main as main_module

    saved_env = {
        key: os.environ.get(key)
        for key in (
            "YOLO26N_HMD_DASHBOARD_BRIDGE",
            "YOLO26N_HMD_DASHBOARD_HOST",
            "YOLO26N_HMD_DASHBOARD_PORT",
        )
    }
    os.environ["YOLO26N_HMD_DASHBOARD_BRIDGE"] = "1"
    os.environ["YOLO26N_HMD_DASHBOARD_HOST"] = "127.0.0.1"
    os.environ["YOLO26N_HMD_DASHBOARD_PORT"] = "8876"

    system = main_module.VRCMocapSystem.__new__(main_module.VRCMocapSystem)
    system.config = {
        "hmd": {
            "provider": "remote",
            "share_with_dashboard": True,
            "dashboard_host": "127.0.0.1",
            "dashboard_port": 8765,
        },
    }
    system._modules = {}
    system._hmd_provider = None

    created = []

    class _BridgeProvider:
        name = "dashboard_api"

        def start(self):
            return True

        def stop(self):
            pass

    original_factory = main_module.HMDProviderFactory

    def fake_create(provider_type, **kwargs):
        created.append((provider_type, kwargs))
        return _BridgeProvider()

    main_module.HMDProviderFactory = type(
        "_Factory",
        (),
        {"create": staticmethod(fake_create)},
    )
    try:
        system._init_hmd_provider()
    finally:
        main_module.HMDProviderFactory = original_factory
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert created[0][0] == "dashboard_api"
    assert created[0][1]["dashboard_host"] == "127.0.0.1"
    assert created[0][1]["dashboard_port"] == 8876


class _HMDPose:
    def __init__(
        self,
        position=None,
        quaternion=None,
        valid=True,
        actual_height=None,
        tracking_origin="floor",
    ):
        self.valid = valid
        self.position = position or [0.4, 1.9, -0.2]
        self.quaternion = quaternion or [1.0, 0.0, 0.0, 0.0]
        self.actual_height = actual_height
        self.tracking_origin = tracking_origin
        self.height_reference = tracking_origin if actual_height is not None else "unknown"


_AUTO_HMD_HEIGHT = object()


class _HMDProvider:
    def __init__(
        self,
        position=None,
        quaternion=None,
        valid=True,
        actual_height=_AUTO_HMD_HEIGHT,
        tracking_origin="floor",
    ):
        self.position = position or [0.4, 1.9, -0.2]
        self.quaternion = quaternion or [1.0, 0.0, 0.0, 0.0]
        self.valid = valid
        self.actual_height = (
            self.position[1] if actual_height is _AUTO_HMD_HEIGHT else actual_height
        )
        self.tracking_origin = tracking_origin

    def get_pose(self):
        return _HMDPose(
            self.position,
            self.quaternion,
            self.valid,
            self.actual_height,
            self.tracking_origin,
        )


class _CaptureOSC:
    def __init__(self):
        self.messages = []

    def connect(self):
        return True

    def disconnect(self):
        pass

    def send_batch(self, messages, port=None):
        self.messages.extend(messages)


class _CaptureHandOSC:
    def __init__(self, host="127.0.0.1", port=9000):
        self.host = host
        self.port = port
        self.socket = True
        self.messages = []

    def connect(self):
        self.socket = True
        return True

    def disconnect(self):
        self.socket = None

    def send(self, message, port=None):
        self.messages.append(message)

    def send_batch(self, messages, port=None):
        self.messages.extend(messages)


class _CaptureHandStreamer:
    connected = True

    def __init__(self):
        self.payloads = []

    def send(self, payload):
        self.payloads.append(payload)
        return True


def _sample_hand_data():
    return [{
        "label": "left",
        "gesture": "point",
        "fingers": {
            "thumb": False,
            "index": True,
            "middle": False,
            "ring": False,
            "pinky": False,
        },
        "curls": {
            "thumb": 0.8,
            "index": 0.0,
            "middle": 0.9,
            "ring": 0.9,
            "pinky": 0.9,
        },
        "spread": 0.2,
        "palm_side": "palm",
        "palm_confidence": 1.0,
    }]


class _CaptureSteamVRStreamer(BodySteamVRStreamer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.commands = []

    def _send_to_pipe(self, text: str, num_tries: int = 10, wait_time: float = 0.1):
        self.commands.append(text)
        return ["OK"]


class _FakeBodyStreamer:
    def __init__(self):
        self.connected = False
        self.enabled = False
        self.inference_enabled = False
        self.connect_count = 0
        self.disconnect_count = 0
        self.decouple_from_hmd = None
        self.hmd_provider = None
        self.sent = []

    def connect(self):
        self.connect_count += 1
        self.connected = True
        return True

    def disconnect(self):
        self.disconnect_count += 1
        self.connected = False

    def set_decouple_from_hmd(self, enabled):
        self.decouple_from_hmd = bool(enabled)

    def set_hmd_provider(self, provider):
        self.hmd_provider = provider

    def send(self, pose_data):
        self.sent.append(pose_data)
        return True


class _FakeBodyOSCStreamer(_FakeBodyStreamer):
    def __init__(self):
        super().__init__()
        self.options = {}

    def update_streaming_options(self, **kwargs):
        self.options.update(kwargs)
        self.decouple_from_hmd = bool(kwargs.get("decouple_from_hmd", False))

    def get_status(self):
        return {
            "fps": 0,
            "send_count": 0,
            "error_count": 0,
            "last_message_count": 0,
            "last_tracker_count": 0,
            "send_position": self.options.get("send_position"),
            "send_rotation": self.options.get("send_rotation"),
            "send_head_position": self.options.get("send_head_position"),
            "decouple_from_hmd": self.options.get("decouple_from_hmd"),
        }


class _FakeMediaPipePoseDetector:
    def __init__(self, *args, **kwargs):
        self.is_loaded = False

    def load(self):
        self.is_loaded = True
        return True

    def detect(self, frame):
        return {
            "pose_3d": {
                "Head": [0.0, -1.4, 0.0],
                "Neck": [0.0, -1.2, 0.0],
                "Hips": [0.0, -0.6, 0.0],
                "HipsTracker": [0.0, -0.6, 0.0],
                "ChestTracker": [0.0, -0.95, 0.0],
                "LeftFootTracker": [0.1, 0.0, 0.2],
                "RightFootTracker": [-0.1, 0.0, 0.2],
            },
            "confidence": 0.88,
            "source": "mediapipe_pose",
        }

    def close(self):
        pass


class _FakeLandmark:
    def __init__(self, x, y, z, visibility=1.0, presence=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


def _decode_osc_floats(message):
    first_zero = message.index(b"\x00")
    address = message[:first_zero].decode("utf-8")
    offset = (first_zero + 4) & ~3
    tags_zero = message.index(b"\x00", offset)
    tags = message[offset:tags_zero].decode("utf-8")
    data_offset = (tags_zero + 4) & ~3
    values = []
    for tag in tags[1:]:
        if tag == "f":
            values.append(struct.unpack("!f", message[data_offset:data_offset + 4])[0])
            data_offset += 4
    return address, values


def _decode_osc_values(message):
    first_zero = message.index(b"\x00")
    address = message[:first_zero].decode("utf-8")
    offset = (first_zero + 4) & ~3
    tags_zero = message.index(b"\x00", offset)
    tags = message[offset:tags_zero].decode("utf-8")
    data_offset = (tags_zero + 4) & ~3
    values = []
    for tag in tags[1:]:
        if tag == "f":
            values.append(struct.unpack("!f", message[data_offset:data_offset + 4])[0])
            data_offset += 4
        elif tag == "i":
            values.append(struct.unpack("!i", message[data_offset:data_offset + 4])[0])
            data_offset += 4
        elif tag == "s":
            end = message.index(b"\x00", data_offset)
            values.append(message[data_offset:end].decode("utf-8"))
            data_offset = (end + 4) & ~3
    return address, values


def _assert_close_list(actual, expected, eps=1e-6):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert abs(a - e) < eps, (actual, expected)


def test_body_module_hot_reload_reenable_restores_inference_streamer():
    from modules.body_module import BodyModule

    module = BodyModule(
        config={"enabled": False},
        streaming_config={"streaming_enabled": True, "decouple_from_hmd": True},
    )
    streamer = _FakeBodyStreamer()
    module._streamer = streamer

    module.update_config(
        {"enabled": True, "detect_torso": True},
        {"streaming_enabled": True, "decouple_from_hmd": False},
    )

    assert module.enabled is True
    assert module.streaming_enabled is True
    assert streamer.inference_enabled is True
    assert streamer.enabled is True
    assert streamer.connected is True
    assert streamer.connect_count == 1
    assert streamer.decouple_from_hmd is False

    module.update_config(
        {"enabled": True, "detect_torso": True},
        {"streaming_enabled": False, "decouple_from_hmd": False},
    )
    assert module.enabled is True
    assert module.streaming_enabled is False
    assert streamer.inference_enabled is True
    assert streamer.enabled is False
    assert streamer.connected is False
    assert streamer.disconnect_count == 1


def test_body_module_hot_reload_updates_running_streamer_switches():
    from modules.body_module import BodyModule
    from streamers.body_osc import BodyOSCStreamer

    module = BodyModule(
        config={"enabled": True},
        streaming_config={"streaming_enabled": True, "decouple_from_hmd": True},
    )
    streamer = _FakeBodyOSCStreamer()
    streamer.connected = True
    module._streamer = streamer

    original_cls = BodyModule.update_config.__globals__["BodyOSCStreamer"]
    BodyModule.update_config.__globals__["BodyOSCStreamer"] = _FakeBodyOSCStreamer
    try:
        module.update_config(
            {"enabled": True, "detect_torso": True},
            {
                "streaming_enabled": True,
                "tracker_toggles": {"Hips": False, "Chest": True},
                "send_position": False,
                "send_rotation": True,
                "send_head_position": False,
                "decouple_from_hmd": False,
            },
        )
    finally:
        BodyModule.update_config.__globals__["BodyOSCStreamer"] = original_cls

    assert streamer.options["tracker_toggles"]["Hips"] is False
    assert streamer.options["send_position"] is False
    assert streamer.options["send_rotation"] is True
    assert streamer.options["send_head_position"] is False
    assert streamer.options["decouple_from_hmd"] is False


def test_body_module_retains_hmd_provider_across_streamer_rebuild():
    from modules.body_module import BodyModule

    module = BodyModule(config={"enabled": True}, streaming_config={})
    provider = object()
    module.set_hmd_provider(provider)
    assert module._hmd_provider is provider

    first_streamer = _FakeBodyStreamer()
    module._streamer = first_streamer
    module._attach_hmd_provider()
    assert first_streamer.hmd_provider is provider

    module._do_cleanup()
    assert module._hmd_provider is provider
    second_streamer = _FakeBodyStreamer()
    module._streamer = second_streamer
    module._attach_hmd_provider()
    assert second_streamer.hmd_provider is provider


def test_body_module_holds_last_pose_for_hmd_height_refresh():
    from modules.body_module import BodyModule

    module = BodyModule(
        config={
            "enabled": True,
            "processing": {
                "hmd_pose_hold_seconds": 5.0,
                "hmd_pose_refresh_fps": 15.0,
            },
        },
        streaming_config={"streaming_enabled": True},
    )
    streamer = _FakeBodyStreamer()
    streamer.connected = True
    module._streamer = streamer
    module._hmd_provider = _HMDProvider(actual_height=0.82)
    module._last_valid_pose_data = {
        "pose_3d": {
            "HipsTracker": [0.0, 0.8, 0.0],
            "ChestTracker": [0.0, 1.1, 0.0],
            "LeftFootTracker": [-0.1, 0.03, 0.1],
            "RightFootTracker": [0.1, 0.03, 0.1],
        },
        "rotations": {},
        "source": "mediapipe_pose",
    }
    module._last_valid_pose_time = time.perf_counter()
    module._last_held_pose_send_time = 0.0

    assert module._stream_held_pose_for_hmd() is True
    assert streamer.sent == [module._last_valid_pose_data]
    assert module._held_pose_streaming is True
    assert module._last_pose_source == "mediapipe_pose+hmd_hold"

    module._last_valid_pose_time = time.perf_counter() - 5.1
    assert module._stream_held_pose_for_hmd() is False
    assert module._last_valid_pose_data is None


def test_body_preview_draws_reconstructed_pose_without_2d_keypoints():
    from modules.body_module import BodyModule

    pose_3d = {
        "Hips": [0.0, 0.0, 0.0],
        "Spine": [0.0, -0.2, 0.0],
        "Chest": [0.0, -0.4, 0.0],
        "UpperChest": [0.0, -0.55, 0.0],
        "Neck": [0.0, -0.7, 0.0],
        "Head": [0.0, -0.85, 0.0],
        "LeftUpperArm": [-0.25, -0.55, 0.0],
        "LeftLowerArm": [-0.45, -0.55, 0.0],
        "LeftHand": [-0.6, -0.55, 0.0],
        "RightUpperArm": [0.25, -0.55, 0.0],
        "RightLowerArm": [0.45, -0.55, 0.0],
        "RightHand": [0.6, -0.55, 0.0],
        "LeftUpperLeg": [-0.15, 0.35, 0.0],
        "LeftLowerLeg": [-0.15, 0.75, 0.0],
        "LeftFoot": [-0.15, 0.9, 0.0],
        "RightUpperLeg": [0.15, 0.35, 0.0],
        "RightLowerLeg": [0.15, 0.75, 0.0],
        "RightFoot": [0.15, 0.9, 0.0],
    }
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    drawn = BodyModule({}, {}).draw_landmarks(frame.copy(), {
        "pose_3d": pose_3d,
        "keypoints_2d": {},
    })

    assert np.count_nonzero(drawn) > 0
    lower_left = drawn[500:710, 0:340]
    subject_center = drawn[160:560, 430:850]
    assert np.count_nonzero(lower_left) > 0
    assert np.count_nonzero(subject_center) == 0


def test_hand_osc_sends_vrchat_desktop_gesture_parameters():
    streamer = HandVMTStreamer()
    capture = _CaptureOSC()
    streamer._osc = capture
    streamer._send_osc_hand_params({
        "left": {
            "fingers": {
                "thumb": False,
                "index": True,
                "middle": False,
                "ring": False,
                "pinky": False,
            },
            "curls": {
                "thumb": 0.8,
                "index": 0.0,
                "middle": 0.9,
                "ring": 0.9,
                "pinky": 0.9,
            },
            "spread": 0.2,
            "palm_side": "palm",
            "palm_confidence": 1.0,
        },
        "right": {
            "fingers": {
                "thumb": False,
                "index": False,
                "middle": False,
                "ring": False,
                "pinky": False,
            },
            "curls": {
                "thumb": 1.0,
                "index": 1.0,
                "middle": 1.0,
                "ring": 1.0,
                "pinky": 1.0,
            },
            "spread": 0.0,
            "palm_side": "back",
            "palm_confidence": 1.0,
        },
    })

    decoded = dict(_decode_osc_values(msg) for msg in capture.messages)
    assert decoded["/avatar/parameters/GestureLeft"] == [3]
    assert decoded["/avatar/parameters/GestureLeftWeight"] == [1.0]
    assert decoded["/avatar/parameters/GestureRight"] == [1]
    assert decoded["/avatar/parameters/GestureRightWeight"] == [1.0]
    assert decoded["/avatar/parameters/DynamicGesture/CurrentGestureLeft"] == [3]
    assert decoded["/avatar/parameters/DynamicGesture/CurrentLeftIndex"] == [3]
    assert decoded["/avatar/parameters/DynamicGesture/CurrentGestureRight"] == [1]
    assert decoded["/avatar/parameters/DynamicGesture/CurrentRightIndex"] == [1]
    assert "/avatar/parameters/LeftIndexCurl" in decoded
    assert "/avatar/parameters/RightThumbCurl" in decoded


def test_hand_desktop_osc_reaches_local_udp_endpoint():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(0.2)
    host, port = receiver.getsockname()
    streamer = HandVMTStreamer(
        host=host,
        port=port,
        desktop_osc_enabled=True,
        vmt_enabled=False,
        custom_parameters_enabled=False,
    )
    assert streamer._osc.connect() is True
    try:
        streamer._send_osc_hand_params({
            "left": {
                "gesture": "point",
                "fingers": {
                    "thumb": False,
                    "index": True,
                    "middle": False,
                    "ring": False,
                    "pinky": False,
                },
            },
        })
        messages = []
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                message, _ = receiver.recvfrom(4096)
            except socket.timeout:
                break
            messages.append(message)
    finally:
        streamer._osc.disconnect()
        receiver.close()

    decoded = dict(_decode_osc_values(message) for message in messages)
    assert decoded["/avatar/parameters/GestureLeft"] == [3]
    assert decoded["/avatar/parameters/GestureLeftWeight"] == [1.0]
    status = streamer.get_status()
    assert status["desktop_osc_host"] == "127.0.0.1"
    assert status["desktop_osc_port"] == port
    assert status["desktop_osc_message_count"] == len(messages)
    assert status["last_desktop_gestures"]["left"] == 3


def test_hand_tracker_keeps_left_and_right_temporal_state_independent():
    tracker = HandTracker()
    extended = np.zeros((21, 3), dtype=float)
    closed = np.zeros((21, 3), dtype=float)
    extended[tracker.THUMB_TIP, 0] = 0.2
    extended[tracker.THUMB_IP, 0] = 0.1
    closed[tracker.THUMB_TIP, 0] = 0.1
    closed[tracker.THUMB_IP, 0] = 0.1
    for tip, pip in (
        (tracker.INDEX_TIP, tracker.INDEX_PIP),
        (tracker.MIDDLE_TIP, tracker.MIDDLE_PIP),
        (tracker.RING_TIP, tracker.RING_PIP),
        (tracker.PINKY_TIP, tracker.PINKY_PIP),
    ):
        extended[tip, 1] = 0.1
        extended[pip, 1] = 0.8
        closed[tip, 1] = 0.8
        closed[pip, 1] = 0.1

    for _ in range(3):
        left = tracker._compute_finger_states(extended, "left")
        right = tracker._compute_finger_states(closed, "right")

    assert all(left.values())
    assert not any(right.values())
    assert set(tracker._finger_state._history) == {
        "left:thumb", "left:index", "left:middle", "left:ring", "left:pinky",
        "right:thumb", "right:index", "right:middle", "right:ring", "right:pinky",
    }
    tracker._smooth_values["test"] = 1.0
    tracker.reset_filters()
    assert tracker._finger_state._history == {}
    assert tracker._smooth_values == {}


def test_hand_tracker_joint_angle_curls_drive_individual_fingers():
    landmarks = np.zeros((21, 3), dtype=np.float64)
    landmarks[1:5] = [
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.4, 0.0, 0.0],
    ]
    for chain in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        for offset, index in enumerate(chain, start=1):
            landmarks[index] = [0.0, offset * 0.1, 0.0]

    tracker = HandTracker(back_curl_compensation=1.0)
    straight = tracker._compute_curls(landmarks, "left", 1.0)
    assert all(value < 0.05 for value in straight.values())

    curled = landmarks.copy()
    curled[5] = [0.0, 0.1, 0.0]
    curled[6] = [0.0, 0.2, 0.0]
    curled[7] = [0.08, 0.2, 0.0]
    curled[8] = [0.08, 0.12, 0.0]
    tracker.reset_filters()
    curls = tracker._compute_curls(curled, "left", 2.0)
    assert curls["index"] > 0.75
    assert curls["middle"] < 0.05
    assert curls["ring"] < 0.05
    assert curls["pinky"] < 0.05


def test_hand_tracker_builds_left_and_right_body_wrist_rois():
    tracker = HandTracker(
        roi_assist_enabled=True,
        roi_scale=2.5,
        roi_min_size=96,
        roi_min_confidence=0.2,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hints = {
        "keypoints_2d": {
            "left_elbow": [180.0, 220.0, 0.9],
            "left_wrist": [120.0, 210.0, 0.9],
            "right_elbow": [460.0, 220.0, 0.9],
            "right_wrist": [520.0, 210.0, 0.9],
        }
    }

    regions = dict(tracker._build_wrist_rois(frame, hints))

    assert set(regions) == {"left", "right"}
    for side, wrist in (("left", (120.0, 210.0)), ("right", (520.0, 210.0))):
        x0, y0, x1, y1 = regions[side]
        assert 0 <= x0 < x1 <= 640
        assert 0 <= y0 < y1 <= 480
        assert x0 <= wrist[0] <= x1
        assert y0 <= wrist[1] <= y1
        assert min(x1 - x0, y1 - y0) >= 96


def test_hand_module_passes_previous_body_keypoints_to_roi_detector():
    from modules.hand_module import HandModule

    class _Tracker:
        def __init__(self):
            self.roi_hints = None

        def detect(self, _frame, roi_hints=None):
            self.roi_hints = roi_hints
            return []

    tracker = _Tracker()
    module = HandModule({}, {"streaming_enabled": False})
    module._tracker = tracker
    body_data = {
        "keypoints_2d": {
            "left_wrist": [100.0, 200.0, 0.9],
            "left_elbow": [140.0, 200.0, 0.9],
        }
    }
    module.set_body_pose_data(body_data)

    assert module._do_process(np.zeros((8, 8, 3), dtype=np.uint8)) is None
    assert tracker.roi_hints is body_data


def test_mediapipe_pose_exports_image_wrist_keypoints_for_hand_roi():
    from types import SimpleNamespace
    from mediapipe_pose import MediaPipePoseDetector

    landmarks = [
        SimpleNamespace(x=0.0, y=0.0, visibility=0.0, presence=0.0)
        for _ in range(33)
    ]
    landmarks[MediaPipePoseDetector.LEFT_WRIST] = SimpleNamespace(
        x=0.25, y=0.50, visibility=0.9, presence=0.8
    )
    landmarks[MediaPipePoseDetector.RIGHT_WRIST] = SimpleNamespace(
        x=0.75, y=0.40, visibility=0.7, presence=0.95
    )

    keypoints = MediaPipePoseDetector._image_landmarks_to_keypoints(
        landmarks, 640, 480
    )

    assert keypoints["left_wrist"] == [160.0, 240.0, 0.8]
    assert keypoints["right_wrist"] == [480.0, 192.0, 0.7]


def test_hand_vmt_sends_independent_open_amount_for_each_finger():
    streamer = HandVMTStreamer(
        desktop_osc_enabled=False,
        vmt_enabled=True,
        vmt_skeleton_enabled=True,
        vmt_input_enabled=False,
    )
    capture = _CaptureHandOSC(port=39570)
    streamer._vmt_osc = capture
    hand = _sample_hand_data()[0]
    streamer._send_vmt_hands(
        {"LeftHand": [0.0, 0.0, 0.0]},
        {"LeftHand": [0.0, 0.0, 0.0, 1.0]},
        {"left": hand},
    )

    decoded_messages = list(map(_decode_osc_values, capture.messages))
    scalar_messages = [
        values
        for address, values in decoded_messages
        if address == "/VMT/Skeleton/Scalar"
    ]
    open_amounts = {int(values[1]): float(values[2]) for values in scalar_messages}
    assert abs(open_amounts[1] - 0.2) < 1e-6
    assert abs(open_amounts[2] - 1.0) < 1e-6
    assert abs(open_amounts[3] - 0.1) < 1e-6
    assert abs(open_amounts[4] - 0.1) < 1e-6
    assert abs(open_amounts[5] - 0.1) < 1e-6
    assert any(address == "/VMT/Skeleton/Apply" for address, _ in decoded_messages)


def test_hand_vmt_pre_registers_compatible_controllers_without_active_pose():
    streamer = HandVMTStreamer(
        desktop_osc_enabled=False,
        vmt_enabled=True,
        vmt_left_device_index=1,
        vmt_right_device_index=2,
    )
    capture = _CaptureHandOSC(port=39570)
    streamer._vmt_osc = capture

    streamer._vmt_init()

    decoded_messages = list(map(_decode_osc_values, capture.messages))
    startup_registration_messages = [
        values
        for address, values in decoded_messages
        if address == "/VMT/Config"
        and values[:1] == ["AddCompatibleControllerOnStartup"]
    ]
    assert startup_registration_messages == [
        ["AddCompatibleControllerOnStartup", "true"]
    ]
    priority_messages = [
        values
        for address, values in decoded_messages
        if address == "/VMT/Config" and values[:1] == ["Priority"]
    ]
    assert priority_messages == [["Priority", "100"]]
    assert next(
        index
        for index, (address, values) in enumerate(decoded_messages)
        if address == "/VMT/Config"
        and values[:1] == ["AddCompatibleControllerOnStartup"]
    ) < next(
        index
        for index, (address, _values) in enumerate(decoded_messages)
        if address == "/VMT/Joint/Driver"
    )
    assert next(
        index
        for index, (address, values) in enumerate(decoded_messages)
        if address == "/VMT/Config" and values[:1] == ["Priority"]
    ) < next(
        index
        for index, (address, _values) in enumerate(decoded_messages)
        if address == "/VMT/Joint/Driver"
    )

    joint_messages = [
        values
        for address, values in map(_decode_osc_values, capture.messages)
        if address == "/VMT/Joint/Driver"
    ]
    assert [(values[0], values[1]) for values in joint_messages] == [
        (1, 5),
        (2, 6),
    ]
    assert streamer._vmt_pre_registered is True
    assert streamer._vmt_last_seen_at["left"] is not None
    assert streamer._vmt_last_seen_at["right"] is not None
    assert streamer._vmt_neutral_sent == {"left": False, "right": False}
    assert streamer._vmt_pose_packets == 0


def test_hand_vmt_reregisters_controllers_when_endpoint_becomes_ready():
    """A late SteamVR start must replay VMT setup and both controller roles."""
    streamer = HandVMTStreamer(
        desktop_osc_enabled=False,
        vmt_enabled=True,
        vmt_skeleton_enabled=True,
        vmt_input_enabled=False,
    )
    capture = _CaptureHandOSC(port=39570)
    streamer._vmt_osc = capture
    streamer._vmt_initialized = True
    streamer._vmt_pre_registered = False
    streamer._vmt_runtime_status = {"ready": False}
    streamer._vmt_last_ready_state = False
    streamer._connected = True

    with mock.patch.object(
        streamer,
        "_refresh_vmt_runtime_status",
        return_value={"ready": True},
    ), mock.patch.object(
        streamer,
        "_vmt_init",
        wraps=streamer._vmt_init,
    ) as init:
        streamer._do_send({"hands": {}})
        assert init.call_count == 1

    joint_messages = [
        values
        for address, values in map(_decode_osc_values, capture.messages)
        if address == "/VMT/Joint/Driver"
    ]
    assert {(int(values[0]), int(values[1])) for values in joint_messages} == {
        (1, 5),
        (2, 6),
    }
    assert streamer._vmt_reinit_count == 1
    assert streamer._vmt_pre_registered is True
    assert streamer._vmt_controller_active == {"left": True, "right": True}

    capture.messages.clear()
    streamer._do_send({"hands": {"left": _sample_hand_data()[0]}})
    decoded = list(map(_decode_osc_values, capture.messages))
    assert any(address == "/VMT/Joint/Driver" for address, _ in decoded)
    assert any(address == "/VMT/Skeleton/Scalar" for address, _ in decoded)
    assert any(address == "/VMT/Skeleton/Apply" for address, _ in decoded)


def test_hand_vmt_tracking_loss_keeps_controllers_registered_by_default():
    streamer = HandVMTStreamer(
        desktop_osc_enabled=False,
        vmt_enabled=True,
        lost_reset_timeout=0.25,
    )
    capture = _CaptureHandOSC(port=39570)
    streamer._vmt_osc = capture
    streamer._send_vmt_hands(
        {"LeftHand": [0.0, 0.0, 0.0]},
        {"LeftHand": [0.0, 0.0, 0.0, 1.0]},
        {"left": _sample_hand_data()[0]},
    )
    capture.messages.clear()

    streamer._reset_missing_vmt_hands({}, now=streamer._vmt_last_seen_at["left"] + 0.3)

    decoded = list(map(_decode_osc_values, capture.messages))
    assert any(address == "/VMT/Skeleton/Apply" for address, _ in decoded)
    assert not any(
        address == "/VMT/Joint/Driver" and int(values[1]) == 0
        for address, values in decoded
    )
    assert streamer._vmt_controller_active["left"] is True
    assert streamer._vmt_neutral_sent["left"] is True


def test_hand_vmt_force_reset_disconnects_controllers():
    streamer = HandVMTStreamer(
        desktop_osc_enabled=False,
        vmt_enabled=True,
        vmt_keep_controllers_active=True,
    )
    capture = _CaptureHandOSC(port=39570)
    streamer._vmt_osc = capture
    streamer._vmt_controller_active["left"] = True

    streamer._reset_missing_vmt_hands({}, force_reset=True)

    decoded = list(map(_decode_osc_values, capture.messages))
    disabled_devices = {
        int(values[0])
        for address, values in decoded
        if address == "/VMT/Joint/Driver" and int(values[1]) == 0
    }
    assert disabled_devices == {1, 2}
    assert streamer._vmt_controller_active == {"left": False, "right": False}


def test_hand_vmt_local_mode_keeps_remote_mapping_from_overwriting_driver_host():
    config = {
        "streaming": {
            "hand": {
                "host": "192.168.1.25",
                "vmt_host": "192.168.1.25",
            }
        }
    }
    hand = ensure_hand_streaming_config(config)
    assert hand["vmt_local_mode"] is True
    assert hand["vmt_host"] == "127.0.0.1"

    remote_config = {
        "streaming": {
            "hand": {
                "host": "192.168.1.25",
                "vmt_host": "192.168.1.80",
                "vmt_local_mode": False,
            }
        }
    }
    remote_hand = ensure_hand_streaming_config(remote_config)
    assert remote_hand["vmt_host"] == "192.168.1.80"


def test_vmt_runtime_status_requires_driver_steamvr_and_udp_listener():
    with mock.patch.object(
        vmt_runtime, "find_vmt_driver_paths", return_value=[Path("C:/VMT/vmt")]
    ), mock.patch.object(
        vmt_runtime, "find_vmt_manager", return_value=Path("C:/VMT/vmt_manager.exe")
    ), mock.patch.object(
        vmt_runtime, "udp_port_is_listening", return_value=True
    ), mock.patch.object(
        vmt_runtime, "steamvr_is_running", return_value=True
    ):
        status = vmt_runtime.get_vmt_runtime_status("127.0.0.1", 39570)
    assert status["ready"] is True
    assert status["driver_installed"] is True
    assert status["udp_port_listening"] is True


def test_vmt_runtime_settings_expose_startup_controller_registration():
    settings = {
        "AddCompatibleControllerOnStartup": True,
        "AddControllerOnStartup": False,
        "AlwaysCompatible": False,
        "AutoPoseUpdateOnStartup": True,
        "OptoutTrackingRole": True,
        "Priority": 100,
        "RejectWhenCannotTracking": True,
        "SkeletonInput": True,
    }
    with mock.patch.object(
        vmt_runtime,
        "find_vmt_driver_paths",
        return_value=[Path("C:/VMT/vmt")],
    ), mock.patch.object(
        Path,
        "is_file",
        return_value=True,
    ), mock.patch.object(
        vmt_runtime,
        "_read_json",
        return_value=settings,
    ):
        parsed = vmt_runtime.load_vmt_driver_settings()

    assert parsed["add_compatible_controller_on_startup"] is True
    assert parsed["hand_selection_priority"] == 100
    assert parsed["skeleton_input"] is True
    assert parsed["optout_tracking_role"] is True
    assert parsed["reject_when_cannot_tracking"] is True


def test_vmt_device_list_parser_identifies_expected_virtual_controllers():
    devices = vmt_runtime.parse_vmt_device_list(
        "0:PICO HMD\n3:VMT_1\n4:VMT_2\ninvalid\n"
    )
    assert devices == [
        {"openvr_index": 0, "serial": "PICO HMD", "is_vmt": False},
        {"openvr_index": 3, "serial": "VMT_1", "is_vmt": True},
        {"openvr_index": 4, "serial": "VMT_2", "is_vmt": True},
    ]


def test_hand_osc_resets_desktop_gesture_after_tracking_loss():
    streamer = HandVMTStreamer(lost_reset_timeout=0.25)
    capture = _CaptureOSC()
    streamer._osc = capture
    streamer._send_osc_hand_params({
        "left": {
            "fingers": {
                "thumb": False,
                "index": False,
                "middle": False,
                "ring": False,
                "pinky": False,
            },
            "curls": {name: 1.0 for name in streamer.FINGER_NAMES},
        },
    }, now=10.0)

    capture.messages.clear()
    streamer._send_osc_hand_params({}, now=10.1)
    assert capture.messages == []

    streamer._send_osc_hand_params({}, now=10.3)
    decoded = dict(_decode_osc_values(msg) for msg in capture.messages)
    assert decoded["/avatar/parameters/GestureLeft"] == [0]
    assert decoded["/avatar/parameters/GestureLeftWeight"] == [0.0]
    assert decoded["/avatar/parameters/DynamicGesture/CurrentGestureLeft"] == [0]
    assert decoded["/avatar/parameters/DynamicGesture/CurrentLeftIndex"] == [0]
    assert decoded["/avatar/parameters/LeftThumbCurl"] == [0.0]

    message_count = len(capture.messages)
    streamer._send_osc_hand_params({}, now=11.0)
    assert len(capture.messages) == message_count


def test_hand_streamer_keeps_desktop_and_vmt_targets_independent():
    streamer = HandVMTStreamer(
        host="127.0.0.1",
        port=9000,
        vmt_host="192.168.1.25",
        vmt_port=39570,
        desktop_osc_enabled=True,
        vmt_enabled=False,
    )
    assert streamer._osc.host == "127.0.0.1"
    assert streamer._osc.port == 9000
    assert streamer._vmt_osc.host == "192.168.1.25"
    assert streamer._vmt_osc.port == 39570
    assert streamer.desktop_osc_enabled is True
    assert streamer.vmt_enabled is False
    assert streamer.desktop_gesture_parameters["left"] == [
        "DynamicGesture/CurrentGestureLeft",
        "DynamicGesture/CurrentLeftIndex",
        "MocapGestureLeft",
    ]


def test_hand_streamer_accepts_custom_desktop_gesture_parameter_names():
    streamer = HandVMTStreamer(
        desktop_left_gesture_parameters="Custom/Left; /avatar/parameters/AltLeft",
        desktop_right_gesture_parameters=["Custom/Right"],
    )
    assert streamer.desktop_gesture_parameters == {
        "left": ["Custom/Left", "AltLeft"],
        "right": ["Custom/Right"],
    }


def test_hand_module_forwards_empty_frames_for_desktop_reset():
    from modules.hand_module import HandModule

    class _EmptyTracker:
        @staticmethod
        def detect(_frame, roi_hints=None):
            return []

    class _CaptureStreamer:
        connected = True

        def __init__(self):
            self.payloads = []

        def send(self, payload):
            self.payloads.append(payload)
            return True

    module = HandModule({}, {"streaming_enabled": True})
    module._tracker = _EmptyTracker()
    module._streamer = _CaptureStreamer()
    module._streaming_enabled = True

    assert module._do_process(None) is None
    assert module._streamer.payloads == [{"hands": [], "body_only": True}]


def test_hand_module_forwards_body_pose_data_to_streamer():
    from modules.hand_module import HandModule

    class _HandTracker:
        @staticmethod
        def detect(_frame, roi_hints=None):
            return _sample_hand_data()

    module = HandModule({}, {"streaming_enabled": True})
    module._tracker = _HandTracker()
    module._streamer = _CaptureHandStreamer()
    module._streaming_enabled = True
    body_pose_data = {
        "pose_3d": {
            "Neck": [0.0, -1.2, 0.0],
            "LeftHand": [-0.3, -0.8, 0.5],
        },
        "rotations": {"LeftHand": [0.1, 0.2, 0.3]},
    }

    result = module._do_process(None, body_pose_data=body_pose_data)

    assert result == _sample_hand_data()
    assert module._streamer.payloads == [{
        "hands": result,
        "body_only": True,
        "pose_3d": body_pose_data["pose_3d"],
        "rotations": body_pose_data["rotations"],
    }]


def test_hand_deferred_streaming_uses_matching_body_frame():
    from modules.hand_module import HandModule

    class _HandTracker:
        @staticmethod
        def detect(_frame, roi_hints=None):
            return _sample_hand_data()

    module = HandModule({}, {"streaming_enabled": True})
    module._tracker = _HandTracker()
    module._streamer = _CaptureHandStreamer()
    module._streaming_enabled = True
    module.set_deferred_streaming(True)

    hand_data = module._do_process(None)
    assert module.last_detection_fresh is True
    assert module._streamer.payloads == []

    body_data = {
        "pose_3d": {"LeftHand": [-0.25, 0.8, 0.4]},
        "rotations": {"LeftHand": [0.1, 0.2, 0.3]},
    }
    assert module.stream_detection(hand_data, body_data) is True
    assert module._streamer.payloads[0]["hands"] == hand_data
    assert module._streamer.payloads[0]["pose_3d"] is body_data["pose_3d"]
    assert module._streamer.payloads[0]["rotations"] is body_data["rotations"]


def test_hand_vmt_uses_payload_pose_3d_when_vmt_enabled():
    class _CoordinateSpy:
        def __init__(self):
            self.pose_3d = None
            self.rotations = None

        def camera_to_unity_positions(self, pose_3d):
            self.pose_3d = pose_3d
            return {"LeftHand": [10.0, 20.0, 30.0]}

        def camera_to_unity_rotations(self, rotations):
            self.rotations = rotations
            return {"LeftHand": [0.0, 0.0, 0.0]}

        def compute_osc_offsets(self, unity_pose):
            return {"LeftHand": [1.0, 2.0, 3.0]}

    streamer = HandVMTStreamer(
        desktop_osc_enabled=False,
        vmt_enabled=True,
    )
    streamer._coord = _CoordinateSpy()
    streamer._vmt_osc = _CaptureHandOSC(host="127.0.0.1", port=39570)
    streamer._vmt_initialized = True
    sent = []
    streamer._send_vmt_hands = lambda osc_pose, rotations, hands: sent.append(
        (osc_pose, rotations, hands)
    )
    pose_3d = {
        "Neck": [0.0, -1.2, 0.0],
        "LeftHand": [-0.3, -0.8, 0.5],
    }
    rotations = {"LeftHand": [0.1, 0.2, 0.3]}
    hands = _sample_hand_data()

    streamer._do_send({
        "hands": hands,
        "pose_3d": pose_3d,
        "rotations": rotations,
    })

    assert streamer._coord.pose_3d is pose_3d
    assert streamer._coord.rotations is rotations
    expected_quaternion = CoordinateTransformer().euler_to_quat_openvr(
        rotations["LeftHand"]
    )
    assert sent == [(
        {"LeftHand": [1.0, 2.0, 3.0]},
        {"LeftHand": expected_quaternion},
        {"left": hands[0]},
    )]


def test_hand_vmt_keeps_desktop_gesture_and_curl_output_when_vmt_enabled():
    streamer = HandVMTStreamer(
        desktop_osc_enabled=True,
        vmt_enabled=True,
        custom_parameters_enabled=True,
    )
    streamer._osc = _CaptureHandOSC()
    streamer._vmt_osc = _CaptureHandOSC(port=39570)
    streamer._vmt_initialized = True
    streamer._coord = type("_Coordinate", (), {
        "camera_to_unity_positions": lambda self, pose: {"LeftHand": [0.0, 0.0, 0.0]},
        "camera_to_unity_rotations": lambda self, rotations: {"LeftHand": [0.0, 0.0, 0.0]},
        "compute_osc_offsets": lambda self, pose: {"LeftHand": [0.0, 0.0, 0.0]},
    })()

    streamer._do_send({
        "hands": _sample_hand_data(),
        "pose_3d": {"LeftHand": [0.0, 0.0, 0.0]},
        "rotations": {},
    })

    decoded = dict(_decode_osc_values(msg) for msg in streamer._osc.messages)
    assert decoded["/avatar/parameters/GestureLeft"] == [3]
    assert decoded["/avatar/parameters/GestureLeftWeight"] == [1.0]
    assert decoded["/avatar/parameters/LeftIndexCurl"] == [0.0]
    assert abs(decoded["/avatar/parameters/LeftMiddleCurl"][0] - 0.9) < 1e-6
    assert any(msg.startswith(b"/VMT/Joint/Driver") for msg in streamer._vmt_osc.messages)


def test_hand_vmt_empty_hand_frame_triggers_neutral_reset():
    streamer = HandVMTStreamer(
        desktop_osc_enabled=True,
        vmt_enabled=False,
        custom_parameters_enabled=True,
        lost_reset_timeout=0.25,
    )
    capture = _CaptureHandOSC()
    streamer._osc = capture
    streamer._send_osc_hand_params({"left": _sample_hand_data()[0]}, now=10.0)
    capture.messages.clear()

    streamer._do_send({"hands": []})

    decoded = dict(_decode_osc_values(msg) for msg in capture.messages)
    assert decoded["/avatar/parameters/GestureLeft"] == [0]
    assert decoded["/avatar/parameters/GestureLeftWeight"] == [0.0]
    assert decoded["/avatar/parameters/LeftIndexCurl"] == [0.0]
    assert decoded["/avatar/parameters/LeftHandSpread"] == [0.0]


def test_hand_module_disable_then_enable_restores_streamer_output():
    from modules.hand_module import HandModule

    class _LifecycleStreamer:
        def __init__(self):
            self.inference_enabled = True
            self.enabled = True
            self.connected = True

        def disconnect(self):
            self.connected = False

        def connect(self):
            self.connected = True
            return True

    module = HandModule(
        config={"enabled": True},
        streaming_config={"streaming_enabled": True},
    )
    module._streamer = _LifecycleStreamer()
    module._initialized = True

    module.enabled = False
    assert module._streamer.enabled is False
    assert module._streamer.inference_enabled is False
    module.enabled = True
    assert module._streamer.enabled is True
    assert module._streamer.inference_enabled is True
    assert module._streamer.connected is True


def test_streamer_queue_is_latest_wins_and_reports_drops():
    from streamers.base import BaseStreamer

    class _LatestWinsStreamer(BaseStreamer):
        def _do_connect(self):
            return True

        def _do_disconnect(self):
            pass

        def _do_send(self, pose_data):
            pass

    streamer = _LatestWinsStreamer("test")
    assert streamer.send({"frame": 1}) is True
    assert streamer.send({"frame": 2}) is True
    assert streamer._send_queue.get_nowait() == {"frame": 2}
    status = streamer.get_status()
    assert status["drop_count"] == 1
    assert status["queue_capacity"] == 1


def test_dashboard_config_api_reports_persistence_failure():
    import dashboard_server as dashboard_module

    original_save = dashboard_module.save_config
    original_cache = dashboard_module._config_cache

    def _fail_save(_config):
        raise OSError("simulated write failure")

    dashboard_module.save_config = _fail_save
    dashboard_module._config_cache = json.loads(json.dumps(original_cache or {}))
    try:
        response = dashboard_module.app.test_client().post(
            "/api/config",
            json={"hand": {"enabled": True}},
        )
    finally:
        dashboard_module.save_config = original_save
        dashboard_module._config_cache = original_cache

    assert response.status_code == 507
    payload = response.get_json()
    assert payload["status"] == "error"
    assert "simulated write failure" in payload["message"]


def test_dashboard_config_recursive_merge_preserves_nested_siblings():
    import dashboard_server as dashboard_module

    config = {
        "streaming": {
            "body": {
                "send_position": True,
                "hmd_height_scaling": {
                    "standing_eye_height": 1.7,
                    "floor_clearance": 0.03,
                },
            }
        }
    }
    dashboard_module._deep_merge_config(
        config,
        {"streaming": {"body": {"hmd_height_scaling": {"floor_clearance": 0.05}}}},
    )

    body = config["streaming"]["body"]
    assert body["send_position"] is True
    assert body["hmd_height_scaling"]["standing_eye_height"] == 1.7
    assert body["hmd_height_scaling"]["floor_clearance"] == 0.05


def test_dashboard_quick_preset_config_is_normalized():
    config = {
        "ui": {
            "active_quick_preset": "custom:preset_valid",
            "quick_presets": [
                {
                    "id": "preset_valid",
                    "name": "  我的预设  ",
                    "values": {
                        "bodyMaxFps": 24,
                        "avgSmoothingEnabled": True,
                        "mpPoseVariant": "heavy",
                        "invalid": float("nan"),
                    },
                    "updated_at": "2026-07-14T00:00:00Z",
                },
                {
                    "id": "preset_valid",
                    "name": "重复 ID",
                    "values": {"bodyMaxFps": 10},
                },
                {"id": "broken", "name": "无 values"},
            ],
        }
    }

    ui = ensure_ui_config(config)
    assert ui["quick_preset_schema"] == 1
    assert ui["active_quick_preset"] == "custom:preset_valid"
    assert len(ui["quick_presets"]) == 1
    preset = ui["quick_presets"][0]
    assert preset["name"] == "我的预设"
    assert preset["values"]["bodyMaxFps"] == 24
    assert preset["values"]["avgSmoothingEnabled"] is True
    assert preset["values"]["mpPoseVariant"] == "heavy"
    assert "invalid" not in preset["values"]

    config["ui"]["active_quick_preset"] = "custom:missing"
    assert ensure_ui_config(config)["active_quick_preset"] == "manual"


def test_dashboard_performance_tiers_and_rtx_3060_baseline():
    cases = [
        (
            "low",
            {
                "device": "cuda:0",
                "gpu_name": "Legacy GPU",
                "gpu_memory_gb": 2,
                "cpu_physical_cores": 2,
                "cpu_logical_cores": 4,
                "memory_gb": 4,
            },
        ),
        (
            "medium",
            {
                "device": "cuda:0",
                "gpu_name": "NVIDIA GeForce GTX 1650",
                "gpu_memory_gb": 4,
                "cpu_physical_cores": 4,
                "cpu_logical_cores": 8,
                "memory_gb": 8,
            },
        ),
        (
            "high",
            {
                "device": "cuda:0",
                "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
                "gpu_memory_gb": 6,
                "cpu_physical_cores": 8,
                "cpu_logical_cores": 16,
                "memory_gb": 15.2,
            },
        ),
        (
            "ultra",
            {
                "device": "cuda:0",
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "gpu_memory_gb": 24,
                "cpu_physical_cores": 16,
                "cpu_logical_cores": 32,
                "memory_gb": 64,
            },
        ),
    ]

    for expected_tier, basis in cases:
        assert classify_performance_tier(basis) == expected_tier

    recommendation = build_performance_recommendation(cases[2][1])
    assert set(recommendation) == {
        "status",
        "tier",
        "tier_label",
        "basis",
        "baseline",
        "rationale",
    }
    assert recommendation["status"] == "ok"
    assert recommendation["tier"] == "high"
    assert recommendation["baseline"] == {
        "camera_width": 1280,
        "camera_height": 720,
        "camera_fps": 30,
        "precision": "fp16",
        "confidence": 0.5,
        "body_max_fps": 24,
        "face_max_fps": 30,
        "hand_max_fps": 24,
        "stability": 55,
    }


def test_dashboard_performance_recommendation_safely_falls_back_without_optional_dependencies():
    import dashboard_server as dashboard_module

    original_cache = dashboard_module._performance_recommendation_cache
    dashboard_module._performance_recommendation_cache = {"data": None, "ts": 0.0}
    try:
        with (
            mock.patch.object(dashboard_module, "HAS_PSUTIL", False),
            mock.patch.object(dashboard_module.os, "cpu_count", side_effect=OSError("unavailable")),
            mock.patch.object(dashboard_module, "_fallback_total_memory_gb", return_value=None),
            mock.patch.object(dashboard_module.shutil, "which", return_value=None),
            mock.patch.dict(sys.modules, {"torch": None}),
        ):
            response = dashboard_module.app.test_client().get(
                "/api/performance/recommendation"
            )
    finally:
        dashboard_module._performance_recommendation_cache = original_cache

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["tier"] == "low"
    assert payload["basis"] == {
        "device": "cpu",
        "gpu_name": None,
        "gpu_memory_gb": None,
        "cpu_physical_cores": None,
        "cpu_logical_cores": None,
        "memory_gb": None,
    }
    assert payload["baseline"]["precision"] == "fp32"


def test_dashboard_ui_active_section_is_normalized():
    assert ensure_ui_config({})["active_section"] == "quickSection"

    valid_sections = (
        "quickSection",
        "hardwareSection",
        "bodySection",
        "faceSection",
        "handSection",
        "streamingSection",
        "hmdSection",
        "logSection",
        "modelSection",
    )
    for section in valid_sections:
        config = {"ui": {"active_section": section}}
        assert ensure_ui_config(config)["active_section"] == section

    config = {"ui": {"active_section": "  faceSection  "}}
    assert ensure_ui_config(config)["active_section"] == "faceSection"
    config["ui"]["active_section"] = "adminSection"
    assert ensure_ui_config(config)["active_section"] == "quickSection"


def test_dashboard_load_config_returns_detached_snapshot(tmp_path):
    import dashboard_server as dashboard_module

    path = tmp_path / "config.json"
    path.write_text('{"streaming":{"body":{"send_position":true}}}', encoding="utf-8")
    original_state = (
        dashboard_module._config_cache,
        dashboard_module._config_cache_signature,
        dashboard_module._active_config_path,
    )
    try:
        dashboard_module._config_cache = {
            "streaming": {"body": {"send_position": True}}
        }
        dashboard_module._active_config_path = path
        dashboard_module._config_cache_signature = dashboard_module._config_signature(path)
        first = dashboard_module.load_config()
        first["streaming"]["body"]["send_position"] = False
        second = dashboard_module.load_config()
    finally:
        (
            dashboard_module._config_cache,
            dashboard_module._config_cache_signature,
            dashboard_module._active_config_path,
        ) = original_state

    assert second["streaming"]["body"]["send_position"] is True


def test_dashboard_runtime_config_precedes_newer_bundled_default(tmp_path):
    import dashboard_server as dashboard_module

    runtime_path = tmp_path / "runtime" / "config.json"
    default_path = tmp_path / "bundle" / "config.json"
    fallback_path = tmp_path / "fallback" / "config.json"
    runtime_path.parent.mkdir()
    default_path.parent.mkdir()
    runtime_path.write_text('{"source":"runtime"}', encoding="utf-8")
    default_path.write_text('{"source":"bundle"}', encoding="utf-8")
    now = time.time()
    os.utime(runtime_path, (now - 100, now - 100))
    os.utime(default_path, (now, now))
    original_values = {
        "CONFIG_PATH": dashboard_module.CONFIG_PATH,
        "DEFAULT_CONFIG_PATH": dashboard_module.DEFAULT_CONFIG_PATH,
        "_fallback_config_path": dashboard_module._fallback_config_path,
        "_config_cache": dashboard_module._config_cache,
        "_config_cache_signature": dashboard_module._config_cache_signature,
        "_active_config_path": dashboard_module._active_config_path,
    }
    try:
        dashboard_module.CONFIG_PATH = runtime_path
        dashboard_module.DEFAULT_CONFIG_PATH = default_path
        dashboard_module._fallback_config_path = lambda: fallback_path
        dashboard_module._config_cache = None
        dashboard_module._config_cache_signature = None
        dashboard_module._active_config_path = None
        loaded = dashboard_module.load_config()
        active = dashboard_module.active_config_path()
    finally:
        for name, value in original_values.items():
            setattr(dashboard_module, name, value)

    assert loaded["source"] == "runtime"
    assert active == runtime_path


def test_dashboard_fallback_becomes_active_mocap_config(tmp_path):
    import dashboard_server as dashboard_module

    primary = tmp_path / "readonly" / "config.json"
    fallback = tmp_path / "fallback" / "config.json"
    original_values = {
        "CONFIG_PATH": dashboard_module.CONFIG_PATH,
        "DEFAULT_CONFIG_PATH": dashboard_module.DEFAULT_CONFIG_PATH,
        "_fallback_config_path": dashboard_module._fallback_config_path,
        "_write_config_path": dashboard_module._write_config_path,
        "_config_cache": dashboard_module._config_cache,
        "_config_cache_signature": dashboard_module._config_cache_signature,
        "_active_config_path": dashboard_module._active_config_path,
    }
    original_writer = dashboard_module._write_config_path

    def selective_writer(path, content):
        if path == primary:
            raise PermissionError("simulated read-only install directory")
        return original_writer(path, content)

    try:
        dashboard_module.CONFIG_PATH = primary
        dashboard_module.DEFAULT_CONFIG_PATH = primary
        dashboard_module._fallback_config_path = lambda: fallback
        dashboard_module._write_config_path = selective_writer
        dashboard_module._config_cache = None
        dashboard_module._config_cache_signature = None
        dashboard_module._active_config_path = None

        saved = dashboard_module.save_config({"camera": {"camera_id": 1}})
        active = dashboard_module.active_config_path()
    finally:
        for name, value in original_values.items():
            setattr(dashboard_module, name, value)

    assert saved == fallback
    assert active == fallback
    assert json.loads(fallback.read_text(encoding="utf-8"))["camera"]["camera_id"] == 1


def test_dashboard_concurrent_config_updates_preserve_both_patches(tmp_path):
    import dashboard_server as dashboard_module

    primary = tmp_path / "config.json"
    fallback = tmp_path / "fallback.json"
    primary.write_text(
        json.dumps({
            "body": {"enabled": True},
            "face": {"enabled": True, "mouth_shape_gain": 1.25},
            "hand": {"enabled": True},
            "processing": {"body_max_fps": 20},
            "streaming": {"body": {"streaming_enabled": True}},
        }),
        encoding="utf-8",
    )
    original_values = {
        "CONFIG_PATH": dashboard_module.CONFIG_PATH,
        "DEFAULT_CONFIG_PATH": dashboard_module.DEFAULT_CONFIG_PATH,
        "_fallback_config_path": dashboard_module._fallback_config_path,
        "_config_cache": dashboard_module._config_cache,
        "_config_cache_signature": dashboard_module._config_cache_signature,
        "_active_config_path": dashboard_module._active_config_path,
    }
    barrier = threading.Barrier(3)
    errors = []

    def post_patch(payload):
        try:
            barrier.wait(timeout=2.0)
            response = dashboard_module.app.test_client().post("/api/config", json=payload)
            if response.status_code != 200:
                errors.append(response.get_data(as_text=True))
        except Exception as exc:
            errors.append(str(exc))

    try:
        dashboard_module.CONFIG_PATH = primary
        dashboard_module.DEFAULT_CONFIG_PATH = primary
        dashboard_module._fallback_config_path = lambda: fallback
        dashboard_module._config_cache = None
        dashboard_module._config_cache_signature = None
        dashboard_module._active_config_path = None

        first = threading.Thread(
            target=post_patch,
            args=({"processing": {"body_max_fps": 17}},),
        )
        second = threading.Thread(
            target=post_patch,
            args=({"face": {"mouth_shape_gain": 1.4}},),
        )
        first.start()
        second.start()
        barrier.wait(timeout=2.0)
        first.join(timeout=5.0)
        second.join(timeout=5.0)
        merged = dashboard_module.load_config()
    finally:
        for name, value in original_values.items():
            setattr(dashboard_module, name, value)

    assert not errors, errors
    assert merged["processing"]["body_max_fps"] == 17
    assert merged["face"]["mouth_shape_gain"] == 1.4


def test_runtime_model_fallback_rejects_non_pose_yolo_weights(tmp_path):
    import main as main_module

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    detection_model = tmp_path / "yolo26n.pt"
    pose_model = tmp_path / "yolo26n-pose.pt"
    detection_model.write_bytes(b"detection")
    pose_model.write_bytes(b"pose")
    now = time.time()
    os.utime(pose_model, (now - 10, now - 10))
    os.utime(detection_model, (now, now))

    original_runtime_root = main_module.runtime_root
    original_resource_root = main_module.resource_root
    main_module.runtime_root = lambda: tmp_path
    main_module.resource_root = lambda: tmp_path
    try:
        system = object.__new__(main_module.VRCMocapSystem)
        selected = system._find_available_runtime_model()
    finally:
        main_module.runtime_root = original_runtime_root
        main_module.resource_root = original_resource_root

    assert Path(selected) == pose_model.resolve()


def _capture_body_osc(
    decouple_from_hmd=False,
    send_position=True,
    send_rotation=False,
    send_head_position=False,
    hmd_provider=True,
    hmd_position=None,
    hmd_actual_height=None,
    face_position=None,
    pose_3d=None,
):
    streamer = BodyOSCStreamer(
        decouple_from_hmd=decouple_from_hmd,
        send_position=send_position,
        send_rotation=send_rotation,
        send_head_position=send_head_position,
    )
    capture = _CaptureOSC()
    streamer._osc = capture
    if hmd_provider:
        streamer.set_hmd_provider(_HMDProvider(
            position=hmd_position,
            actual_height=hmd_actual_height,
        ))
    pose_data = {
        "pose_3d": pose_3d or {
            "Head": [0.0, -1.4, 0.0],
            "Neck": [0.0, -1.2, 0.0],
            "Hips": [0.0, -0.6, 0.0],
            "LeftFoot": [0.1, 0.0, 0.2],
            "RightFoot": [-0.1, 0.0, 0.2],
        },
        "rotations": {},
    }
    if face_position is not None:
        pose_data["face"] = {"face_position": face_position}
    streamer._do_send(
        pose_data
    )
    return dict(_decode_osc_floats(msg) for msg in capture.messages)


def test_body_osc_decouple_toggle_and_head_position_policy():
    hmd_a = [0.4, 1.9, -0.2]
    hmd_b = [0.4, 2.1, -0.2]
    bound_a = _capture_body_osc(
        decouple_from_hmd=False, hmd_position=hmd_a, hmd_actual_height=1.7
    )
    bound_b = _capture_body_osc(
        decouple_from_hmd=False, hmd_position=hmd_b, hmd_actual_height=1.7
    )
    decoupled_a = _capture_body_osc(decouple_from_hmd=True, hmd_position=hmd_a)
    decoupled_b = _capture_body_osc(decouple_from_hmd=True, hmd_position=hmd_b)

    _assert_close_list(bound_a["/tracking/trackers/1/position"], bound_b["/tracking/trackers/1/position"])
    assert decoupled_a["/tracking/trackers/1/position"] == decoupled_b["/tracking/trackers/1/position"]
    assert bound_a["/tracking/trackers/head/position"] == [0.0, 0.0, 0.0]
    assert decoupled_a["/tracking/trackers/head/position"] == [0.0, 0.0, 0.0]

    hmd_protected = _capture_body_osc(
        send_head_position=True,
        hmd_provider=True,
        face_position=[0.3, 1.2, -0.4],
    )
    assert hmd_protected["/tracking/trackers/head/position"] == [0.0, 0.0, 0.0]

    desktop_debug = _capture_body_osc(
        send_head_position=True,
        hmd_provider=False,
        face_position=[0.3, 1.2, -0.4],
    )
    _assert_close_list(desktop_debug["/tracking/trackers/head/position"], [0.3, 1.2, -0.4])

    no_position = _capture_body_osc(send_position=False, send_head_position=True)
    assert "/tracking/trackers/1/position" not in no_position
    assert "/tracking/trackers/head/position" not in no_position


def test_hmd_anchor_trust_boundary_is_shared_by_body_streamers():
    untrusted = _HMDPose(
        position=[8.0, 0.05, 7.0],
        valid=True,
        actual_height=None,
        tracking_origin="device",
    )
    explicit = _HMDPose(
        position=[8.0, 0.05, 7.0],
        valid=True,
        actual_height=0.82,
        tracking_origin="device",
    )
    floor_with_explicit_height = _HMDPose(
        position=[0.1, 0.05, -0.2],
        valid=True,
        actual_height=0.82,
        tracking_origin="floor",
    )
    floor = _HMDPose(
        position=[0.0, 1.65, 0.0],
        valid=True,
        actual_height=None,
        tracking_origin="floor",
    )
    invalid_floor = _HMDPose(
        position=[0.0, 0.0, 0.0],
        valid=True,
        actual_height=None,
        tracking_origin="floor",
    )
    invalid = _HMDPose(valid=False, tracking_origin="floor")

    assert hmd_pose_anchor_trusted(untrusted) is False
    assert hmd_pose_anchor_trusted(explicit) is True
    assert hmd_pose_anchor_trusted(floor_with_explicit_height) is True
    assert hmd_pose_anchor_trusted(floor) is True
    assert hmd_pose_anchor_trusted(invalid_floor) is False
    assert hmd_pose_anchor_trusted(invalid) is False


def test_body_osc_rejects_untrusted_hmd_anchor_but_protects_head_position():
    streamer = BodyOSCStreamer(send_head_position=True)
    capture = _CaptureOSC()
    streamer._osc = capture
    streamer.set_hmd_provider(_HMDProvider(
        position=[8.0, 0.05, 7.0],
        actual_height=None,
        tracking_origin="device",
    ))
    streamer._do_send({
        "pose_3d": {
            "Head": [0.0, -1.4, 0.0],
            "Hips": [0.0, -0.6, 0.0],
            "LeftFoot": [0.1, 0.0, 0.2],
            "RightFoot": [-0.1, 0.0, 0.2],
        },
        "rotations": {},
        "face": {"face_position": [0.3, 1.2, -0.4]},
    })
    decoded = dict(_decode_osc_floats(msg) for msg in capture.messages)
    assert decoded["/tracking/trackers/head/position"] == [0.0, 0.0, 0.0]
    status = streamer.get_status()
    assert status["hmd_present"] is True
    assert status["hmd_anchor_trusted"] is False
    assert status["hmd_anchor_applied"] is False


def test_body_module_status_reports_hmd_without_body_frame():
    from modules.body_module import BodyModule

    module = BodyModule(
        {"enabled": True, "inference_backend": "mediapipe_pose"},
        {"streaming_enabled": True, "decouple_from_hmd": False},
    )
    module._hmd_provider = _HMDProvider(
        position=[0.0, 1.1, 0.0],
        actual_height=1.1,
        tracking_origin="floor",
    )
    status = module.get_status()
    assert status["hmd_present"] is True
    assert status["hmd_anchor_trusted"] is True
    assert status["hmd_anchor_applied"] is True
    assert status["hmd_actual_height"] == 1.1
    assert status["hmd_height_state"]["reason"] == "not_computed"


def test_tracker_nodes_bind_to_intended_joint_positions():
    estimator = PoseEstimator(smoothing="linear", smoothing_factor=1.0)
    norm_kps = {
        "nose": (0.0, -1.0, 0.9),
        "left_shoulder": (-0.30, -0.55, 0.9),
        "right_shoulder": (0.30, -0.55, 0.9),
        "left_elbow": (-0.55, -0.30, 0.9),
        "right_elbow": (0.55, -0.30, 0.9),
        "left_wrist": (-0.70, 0.10, 0.9),
        "right_wrist": (0.70, 0.10, 0.9),
        "left_hip": (-0.12, 0.0, 0.9),
        "right_hip": (0.12, 0.0, 0.9),
        "left_knee": (-0.18, 0.65, 0.9),
        "right_knee": (0.18, 0.65, 0.9),
        "left_ankle": (-0.20, 1.15, 0.9),
        "right_ankle": (0.20, 1.15, 0.9),
    }
    pose = estimator.estimate_3d_pose(norm_kps)

    assert pose["HipsTracker"] == pose["Hips"]
    assert pose["ChestTracker"] == pose["Chest"]
    assert pose["LeftFootTracker"] == pose["LeftFoot"]
    assert pose["RightFootTracker"] == pose["RightFoot"]
    assert pose["LeftKneeTracker"] == pose["LeftUpperLeg"]
    assert pose["RightKneeTracker"] == pose["RightUpperLeg"]
    assert pose["LeftElbowTracker"] == pose["LeftUpperArm"]
    assert pose["RightElbowTracker"] == pose["RightUpperArm"]
    assert pose["LeftKneeTracker"] != pose["LeftLowerLeg"]
    assert pose["LeftElbowTracker"] != pose["LeftLowerArm"]


def test_head_is_simulated_when_nose_is_missing():
    estimator = PoseEstimator(smoothing="linear", smoothing_factor=1.0)
    norm_kps = {
        "left_shoulder": (-0.30, -0.55, 0.9),
        "right_shoulder": (0.30, -0.55, 0.9),
        "left_elbow": (-0.55, -0.30, 0.9),
        "right_elbow": (0.55, -0.30, 0.9),
        "left_wrist": (-0.70, 0.10, 0.9),
        "right_wrist": (0.70, 0.10, 0.9),
        "left_hip": (-0.12, 0.0, 0.9),
        "right_hip": (0.12, 0.0, 0.9),
        "left_knee": (-0.18, 0.65, 0.9),
        "right_knee": (0.18, 0.65, 0.9),
        "left_ankle": (-0.20, 1.15, 0.9),
        "right_ankle": (0.20, 1.15, 0.9),
    }
    pose = estimator.estimate_3d_pose(norm_kps)

    assert "Head" in pose
    assert np.all(np.isfinite(pose["Head"]))
    assert pose["Head"][1] < pose["Neck"][1]
    assert pose["Head"][1] < pose["UpperChest"][1]


def test_process_detections_injects_synthetic_nose_when_head_is_missing():
    estimator = PoseEstimator(smoothing="linear", smoothing_factor=1.0)
    keypoints = np.zeros((17, 3), dtype=float)
    points = {
        "left_shoulder": (260.0, 180.0, 0.9),
        "right_shoulder": (380.0, 180.0, 0.9),
        "left_elbow": (220.0, 250.0, 0.9),
        "right_elbow": (420.0, 250.0, 0.9),
        "left_wrist": (200.0, 330.0, 0.9),
        "right_wrist": (440.0, 330.0, 0.9),
        "left_hip": (285.0, 340.0, 0.9),
        "right_hip": (355.0, 340.0, 0.9),
        "left_knee": (285.0, 440.0, 0.9),
        "right_knee": (355.0, 440.0, 0.9),
        "left_ankle": (285.0, 540.0, 0.9),
        "right_ankle": (355.0, 540.0, 0.9),
    }
    for name, value in points.items():
        keypoints[PoseEstimator.KP[name]] = value

    result = estimator.process_detections(
        {"keypoints": [keypoints], "confidence": [0.9]},
        image_width=640,
        image_height=640,
        apply_filters=False,
    )

    assert result is not None
    assert result["keypoints_2d"]["nose"][2] == 0.2
    assert result["keypoints_2d"]["nose"][1] < 180.0
    assert result["pose_3d"]["Head"][1] < result["pose_3d"]["Neck"][1]


def test_body_osc_uses_tracker_nodes_for_knee_and_elbow_positions():
    pose_3d = {
        "Head": [0.0, -1.4, 0.0],
        "Neck": [0.0, -1.2, 0.0],
        "Hips": [0.0, -0.6, 0.0],
        "HipsTracker": [0.0, -0.6, 0.0],
        "Chest": [0.0, -0.95, 0.0],
        "ChestTracker": [0.0, -0.95, 0.0],
        "LeftUpperLeg": [0.12, -0.2, 0.0],
        "LeftLowerLeg": [0.44, 0.8, 0.7],
        "LeftKneeTracker": [0.12, -0.2, 0.0],
        "LeftFoot": [0.50, 1.0, 0.8],
        "LeftFootTracker": [0.50, 1.0, 0.8],
        "RightFoot": [-0.50, 1.0, 0.8],
        "RightFootTracker": [-0.50, 1.0, 0.8],
        "LeftUpperArm": [0.33, -1.05, 0.05],
        "LeftLowerArm": [0.88, -0.45, 0.55],
        "LeftElbowTracker": [0.33, -1.05, 0.05],
        "RightElbowTracker": [-0.33, -1.05, 0.05],
    }
    captured = _capture_body_osc(
        decouple_from_hmd=True,
        hmd_provider=False,
        pose_3d=pose_3d,
    )

    coord = CoordinateTransformer()
    unity_pose = coord.camera_to_unity_positions(pose_3d)
    expected = coord.compute_osc_offsets(unity_pose, None, None)
    _assert_close_list(captured["/tracking/trackers/5/position"], expected["LeftKneeTracker"])
    _assert_close_list(captured["/tracking/trackers/7/position"], expected["LeftElbowTracker"])
    assert captured["/tracking/trackers/5/position"] != expected["LeftLowerLeg"]
    assert captured["/tracking/trackers/7/position"] != expected["LeftLowerArm"]


def test_body_steamvr_uses_tracker_nodes_for_knee_and_elbow_positions():
    pose_3d = {
        "Head": [0.0, -1.4, 0.0],
        "Neck": [0.0, -1.2, 0.0],
        "Hips": [0.0, -0.6, 0.0],
        "HipsTracker": [0.0, -0.6, 0.0],
        "Chest": [0.0, -0.95, 0.0],
        "ChestTracker": [0.0, -0.95, 0.0],
        "LeftUpperLeg": [0.12, -0.2, 0.0],
        "LeftLowerLeg": [0.44, 0.8, 0.7],
        "LeftKneeTracker": [0.12, -0.2, 0.0],
        "LeftFoot": [0.50, 1.0, 0.8],
        "LeftFootTracker": [0.50, 1.0, 0.8],
        "RightFoot": [-0.50, 1.0, 0.8],
        "RightFootTracker": [-0.50, 1.0, 0.8],
        "LeftUpperArm": [0.33, -1.05, 0.05],
        "LeftLowerArm": [0.88, -0.45, 0.55],
        "LeftElbowTracker": [0.33, -1.05, 0.05],
        "RightElbowTracker": [-0.33, -1.05, 0.05],
    }
    streamer = _CaptureSteamVRStreamer(decouple_from_hmd=True)
    streamer._do_send({"pose_3d": pose_3d, "rotations": {}})

    update_commands = [cmd for cmd in streamer.commands if cmd.startswith("updatepose ")]
    assert len(update_commands) == 8
    parsed_positions = {}
    for cmd in update_commands:
        parts = cmd.split()
        tracker_index = int(parts[1])
        parsed_positions[tracker_index] = [float(parts[2]), float(parts[3]), float(parts[4])]

    coord = CoordinateTransformer()
    unity_pose = coord.camera_to_unity_positions(pose_3d)
    steamvr_pose = coord.compute_steamvr_positions(unity_pose, None, None)
    _assert_close_list(parsed_positions[4], steamvr_pose["LeftKneeTracker"])
    _assert_close_list(parsed_positions[6], steamvr_pose["LeftElbowTracker"])
    assert parsed_positions[4] != steamvr_pose["LeftLowerLeg"]
    assert parsed_positions[6] != steamvr_pose["LeftLowerArm"]


def test_body_steamvr_hmd_base_moves_trackers_when_not_decoupled():
    pose_3d = {
        "Head": [0.0, -1.4, 0.0],
        "Neck": [0.0, -1.2, 0.0],
        "Hips": [0.0, -0.6, 0.0],
        "HipsTracker": [0.0, -0.6, 0.0],
        "ChestTracker": [0.0, -0.95, 0.0],
        "LeftFootTracker": [0.50, 1.0, 0.8],
        "RightFootTracker": [-0.50, 1.0, 0.8],
        "LeftKneeTracker": [0.12, -0.2, 0.0],
        "RightKneeTracker": [-0.12, -0.2, 0.0],
        "LeftElbowTracker": [0.33, -1.05, 0.05],
        "RightElbowTracker": [-0.33, -1.05, 0.05],
    }
    hmd_position = [0.4, 1.9, -0.2]
    streamer = _CaptureSteamVRStreamer(decouple_from_hmd=False)
    streamer.set_hmd_provider(_HMDProvider(position=hmd_position))
    streamer._do_send({"pose_3d": pose_3d, "rotations": {}})

    update_commands = [cmd for cmd in streamer.commands if cmd.startswith("updatepose ")]
    parts = update_commands[0].split()
    hips_position = [float(parts[2]), float(parts[3]), float(parts[4])]

    coord = CoordinateTransformer()
    unity_pose = coord.camera_to_unity_positions(pose_3d)
    expected = coord.compute_steamvr_positions(
        unity_pose,
        hmd_position,
        [1.0, 0.0, 0.0, 0.0],
        hmd_position[1],
    )
    _assert_close_list(hips_position, expected["HipsTracker"])


def test_body_steamvr_rejects_untrusted_device_origin_without_height():
    pose_3d = {
        "Head": [0.0, -1.4, 0.0],
        "Neck": [0.0, -1.2, 0.0],
        "Hips": [0.0, -0.6, 0.0],
        "HipsTracker": [0.0, -0.6, 0.0],
        "ChestTracker": [0.0, -0.95, 0.0],
        "LeftFootTracker": [0.50, 1.0, 0.8],
        "RightFootTracker": [-0.50, 1.0, 0.8],
        "LeftKneeTracker": [0.12, -0.2, 0.0],
        "RightKneeTracker": [-0.12, -0.2, 0.0],
        "LeftElbowTracker": [0.33, -1.05, 0.05],
        "RightElbowTracker": [-0.33, -1.05, 0.05],
    }

    class _Provider:
        def __init__(self, actual_height):
            self.actual_height = actual_height

        def get_pose(self):
            return _HMDPose(
                position=[8.0, 0.05, 7.0],
                quaternion=[1.0, 0.0, 0.0, 0.0],
                valid=True,
                actual_height=self.actual_height,
                tracking_origin="device",
            )

    untrusted = _CaptureSteamVRStreamer(decouple_from_hmd=False)
    untrusted.set_hmd_provider(_Provider(actual_height=None))
    untrusted._do_send({"pose_3d": pose_3d, "rotations": {}})
    first_command = next(cmd for cmd in untrusted.commands if cmd.startswith("updatepose 0 "))
    parts = first_command.split()
    hips_position = [float(parts[2]), float(parts[3]), float(parts[4])]

    coord = CoordinateTransformer()
    unity_pose = coord.camera_to_unity_positions(pose_3d)
    expected_local = coord.compute_steamvr_positions(unity_pose, None, None)
    _assert_close_list(hips_position, expected_local["HipsTracker"])
    assert untrusted.get_status()["hmd_anchor_trusted"] is False

    trusted = _CaptureSteamVRStreamer(decouple_from_hmd=False)
    trusted.set_hmd_provider(_Provider(actual_height=0.82))
    trusted._do_send({"pose_3d": pose_3d, "rotations": {}})
    assert trusted.get_status()["hmd_anchor_trusted"] is True
    assert trusted.get_status()["hmd_tracking_origin"] == "device"


class _DriverPoseSteamVRStreamer(_CaptureSteamVRStreamer):
    """Capture streamer whose fake driver answers getdevicepose 0."""

    def __init__(self, driver_reply=None, **kwargs):
        super().__init__(**kwargs)
        self.driver_reply = driver_reply

    def _send_to_pipe(self, text: str, num_tries: int = 10, wait_time: float = 0.1):
        self.commands.append(text)
        if text.startswith("getdevicepose"):
            return None if self.driver_reply is None else self.driver_reply.split(" ")
        return ["OK"]


_STEAMVR_ANCHOR_POSE_3D = {
    "Head": [0.0, -1.4, 0.0],
    "Neck": [0.0, -1.2, 0.0],
    "Hips": [0.0, -0.6, 0.0],
    "HipsTracker": [0.0, -0.6, 0.0],
    "ChestTracker": [0.0, -0.95, 0.0],
    "LeftUpperLeg": [0.12, -0.2, 0.0],
    "RightUpperLeg": [-0.12, -0.2, 0.0],
    "LeftFootTracker": [0.50, 1.0, 0.8],
    "RightFootTracker": [-0.50, 1.0, 0.8],
    "LeftKneeTracker": [0.12, -0.2, 0.0],
    "RightKneeTracker": [-0.12, -0.2, 0.0],
    "LeftElbowTracker": [0.33, -1.05, 0.05],
    "RightElbowTracker": [-0.33, -1.05, 0.05],
}


def _first_tracker_position(streamer, index=0):
    command = next(cmd for cmd in streamer.commands if cmd.startswith(f"updatepose {index} "))
    parts = command.split()
    return [float(parts[2]), float(parts[3]), float(parts[4])]


def test_body_steamvr_driver_pose_reply_parser_is_robust():
    parse = BodySteamVRStreamer._parse_driver_hmd_pose
    position, quat = parse(" devicepose 0 0.5 1.6 -2.0 2.0 0.0 0.0 0.0\x00")
    # OpenVR raw (Z back) -> Unity (Z forward); quaternion is normalized wxyz.
    _assert_close_list(position, [0.5, 1.6, 2.0])
    _assert_close_list(quat, [1.0, 0.0, 0.0, 0.0])
    _, mirrored = parse("x devicepose 0 0 0 0 0.0 0.6 0.8 0.0")
    _assert_close_list(mirrored, [0.0, -0.6, -0.8, 0.0])
    assert parse(None) is None
    assert parse("error") is None
    assert parse(" devicepose 1 0 0 0 1 0 0 0") is None
    assert parse(" devicepose 0 0 0 0 1 0 0") is None
    assert parse(" devicepose 0 0 nan 0 1 0 0 0") is None
    assert parse(" devicepose 0 0 0 0 0 0 0 0") is None
    assert parse(["", "devicepose", "0", "1", "2", "3", "1", "0", "0", "0"]) is not None


def test_body_steamvr_anchors_in_driver_raw_space_without_height_override():
    # Provider (standing universe) and driver raw space disagree; trackers are
    # sent through the driver, so its raw HMD pose must be the anchor.
    streamer = _DriverPoseSteamVRStreamer(
        driver_reply=" devicepose 0 1.0 1.3 -0.5 1 0 0 0",
        decouple_from_hmd=False,
        auto_yaw_calibration=False,
    )
    streamer.set_hmd_provider(_HMDProvider(position=[8.0, 1.7, 9.0], actual_height=1.7))
    streamer._do_send({"pose_3d": _STEAMVR_ANCHOR_POSE_3D, "rotations": {}})

    coord = CoordinateTransformer()
    unity_pose = coord.camera_to_unity_positions(_STEAMVR_ANCHOR_POSE_3D)
    expected = coord.compute_steamvr_positions(
        unity_pose,
        [1.0, 1.3, 0.5],
        [1.0, 0.0, 0.0, 0.0],
        1.7,
        anchor_in_driver_space=True,
    )
    _assert_close_list(_first_tracker_position(streamer), expected["HipsTracker"])
    status = streamer.get_status()
    assert status["hmd_anchor_source"] == "driver_raw"
    assert status["last_driver_pose_valid"] is True
    assert status["hmd_anchor_applied"] is True

    # Raw Y is kept as-is; the provider's floor height only drives retargeting.
    raw = coord.compute_steamvr_positions(
        {"Head": [0.0, 0.0, 0.0]}, [1.0, 1.3, 0.5], None, None,
        anchor_in_driver_space=True,
    )
    _assert_close_list(raw["Head"], [1.0, 1.3, -0.5])


def test_body_steamvr_falls_back_to_provider_when_driver_pose_missing():
    hmd_position = [0.4, 1.9, -0.2]
    streamer = _DriverPoseSteamVRStreamer(
        driver_reply=None, decouple_from_hmd=False, auto_yaw_calibration=False
    )
    streamer.set_hmd_provider(_HMDProvider(position=hmd_position))
    streamer._do_send({"pose_3d": _STEAMVR_ANCHOR_POSE_3D, "rotations": {}})

    coord = CoordinateTransformer()
    unity_pose = coord.camera_to_unity_positions(_STEAMVR_ANCHOR_POSE_3D)
    expected = coord.compute_steamvr_positions(
        unity_pose, hmd_position, [1.0, 0.0, 0.0, 0.0], hmd_position[1]
    )
    _assert_close_list(_first_tracker_position(streamer), expected["HipsTracker"])
    status = streamer.get_status()
    assert status["hmd_anchor_source"] == "provider"
    assert status["last_driver_pose_valid"] is False


def test_body_steamvr_auto_yaw_aligns_body_with_hmd_heading():
    # HMD yawed +90 deg around Y (wxyz); body faces Unity +Z.
    half = np.sqrt(0.5)
    streamer = _DriverPoseSteamVRStreamer(
        driver_reply=f" devicepose 0 0 1.6 0 {half} 0 {-half} 0",
        decouple_from_hmd=False,
        auto_yaw_calibration=True,
    )
    streamer._do_send({"pose_3d": _STEAMVR_ANCHOR_POSE_3D, "rotations": {"Hips": [0.0, 0.0, 0.0]}})
    status = streamer.get_status()
    assert status["yaw_calibrated"] is True
    assert abs(status["yaw_offset_deg"] - 90.0) < 1e-6

    # Unity right axis (+X) rotated +90 deg yaw becomes Unity -Z, i.e. OpenVR +Z.
    left_knee = _first_tracker_position(streamer, 4)
    right_knee = _first_tracker_position(streamer, 5)
    lateral = np.subtract(right_knee, left_knee)
    assert abs(lateral[0]) < 1e-6 and lateral[2] > 0.2

    # Tracker rotation gets the same yaw as the positions.
    hips_cmd = next(cmd for cmd in streamer.commands if cmd.startswith("updatepose 0 "))
    qw, qx, qy, qz = [float(v) for v in hips_cmd.split()[5:9]]
    _assert_close_list([abs(qw), abs(qy)], [half, half])
    _assert_close_list([qx, qz], [0.0, 0.0])

    # Manual offset is additive and hot-reloadable; toggling auto resets it.
    streamer.update_streaming_options(yaw_offset_deg=10.0)
    assert abs(streamer.get_status()["yaw_offset_deg"] - 100.0) < 1e-6
    streamer.update_streaming_options(auto_yaw_calibration=False)
    assert streamer.get_status()["yaw_calibrated"] is False
    assert abs(streamer.get_status()["yaw_offset_deg"] - 10.0) < 1e-6


def test_body_steamvr_decoupled_mode_skips_driver_hmd_query():
    streamer = _DriverPoseSteamVRStreamer(
        driver_reply=" devicepose 0 5 5 5 1 0 0 0", decouple_from_hmd=True
    )
    streamer.set_hmd_provider(_HMDProvider(position=[3.0, 1.7, 3.0]))
    streamer._do_send({"pose_3d": _STEAMVR_ANCHOR_POSE_3D, "rotations": {}})
    assert not any(cmd.startswith("getdevicepose") for cmd in streamer.commands)

    coord = CoordinateTransformer()
    unity_pose = coord.camera_to_unity_positions(_STEAMVR_ANCHOR_POSE_3D)
    expected = coord.compute_steamvr_positions(unity_pose, None, None)
    _assert_close_list(_first_tracker_position(streamer), expected["HipsTracker"])
    status = streamer.get_status()
    assert status["hmd_anchor_source"] == "none"
    assert status["hmd_anchor_applied"] is False

def test_body_streaming_config_defaults_to_hmd_base_and_preserves_debug_head_position():
    config = {
        "streaming": {
            "body": {
                "send_position": True,
                "send_head_position": True,
            }
        },
        "vrc": {
            "send_position": True,
            "send_head_position": True,
        },
    }
    normalize_body_streaming_config(config)
    assert config["streaming"]["body"]["decouple_from_hmd"] is False
    assert config["vrc"]["decouple_from_hmd"] is False
    assert config["streaming"]["body"]["send_head_position"] is True
    assert config["vrc"]["send_head_position"] is True

    no_position = {
        "streaming": {
            "body": {
                "decouple_from_hmd": False,
                "send_position": False,
                "send_head_position": True,
            }
        },
        "vrc": {
            "decouple_from_hmd": False,
            "send_position": False,
            "send_head_position": True,
        },
    }
    normalize_body_streaming_config(no_position)
    assert no_position["streaming"]["body"]["send_head_position"] is False
    assert no_position["vrc"]["send_head_position"] is False


def test_body_streaming_config_inherits_legacy_vrc_switches_when_body_keys_missing():
    config = {
        "body": {"enabled": True},
        "streaming": {"body": {"streaming_enabled": True}},
        "vrc": {
            "decouple_from_hmd": True,
            "send_position": False,
            "send_rotation": True,
        },
    }
    normalize_body_streaming_config(config)
    body = config["streaming"]["body"]
    assert body["decouple_from_hmd"] is True
    assert body["send_position"] is False
    assert body["send_rotation"] is True
    assert config["vrc"]["decouple_from_hmd"] is True
    assert config["vrc"]["send_position"] is False


def test_body_streaming_config_keeps_legacy_vrc_mirror_in_sync():
    config = {
        "body": {"enabled": True},
        "streaming": {
            "body": {
                "decouple_from_hmd": False,
                "send_position": True,
                "send_rotation": False,
                "send_head_position": True,
                "tracker_toggles": {"Hips": False},
                "hmd_height_scaling": {"standing_eye_height": 1.72},
            }
        },
        "vrc": {"decouple_from_hmd": True, "send_rotation": True},
    }
    normalize_body_streaming_config(config)
    assert config["vrc"]["decouple_from_hmd"] is False
    assert config["vrc"]["send_position"] is True
    assert config["vrc"]["send_rotation"] is False
    assert config["vrc"]["send_head_position"] is True
    assert config["vrc"]["tracker_toggles"] == {"Hips": False}
    assert config["vrc"]["hmd_height_scaling"]["standing_eye_height"] == 1.72


def test_high_seated_hmd_is_not_labeled_standing_without_leg_confidence():
    coord = CoordinateTransformer(hmd_height_scaling={
        "standing_eye_height": 1.65,
        "standing_eye_height_calibrated": True,
        "auto_learn_standing_height": False,
        "seated_eye_height": 0.95,
    })
    pose = {
        "Head": [0.0, 1.65, 0.0],
        "ChestTracker": [0.0, 1.15, 0.0],
        "HipsTracker": [0.0, 0.85, 0.0],
        # A high chair can keep the HMD above the seated absolute cutoff;
        # without a credible leg chain it must not be treated as standing.
    }
    coord.compute_osc_offsets(pose, [0.0, 1.55, 0.0], None, 1.55)
    state = coord.hmd_height_state
    assert state["posture"] == "unknown"
    assert state["standing_pose_confidence"] == 0.0

    complete_pose = dict(pose)
    complete_pose.update({
        "LeftKneeTracker": [-0.15, 0.65, 0.24],
        "RightKneeTracker": [0.15, 0.65, 0.24],
        "LeftFootTracker": [-0.15, 0.03, 0.15],
        "RightFootTracker": [0.15, 0.03, 0.15],
    })
    coord.compute_osc_offsets(complete_pose, [0.0, 1.55, 0.0], None, 1.55)
    assert coord.hmd_height_state["posture"] == "crouch"


def test_body_streaming_enabled_turns_on_body_inference():
    config = {
        "body": {"enabled": False},
        "streaming": {"body": {"streaming_enabled": True}},
    }
    normalize_body_streaming_config(config)
    assert config["body"]["enabled"] is True


def test_body_inference_backend_config_enables_mediapipe_pose():
    config = {
        "body": {
            "enabled": False,
            "inference_backend": "mediapipepose",
            "detect_left_arm": False,
            "detect_right_arm": True,
        },
        "mediapipe_pose": {"enabled": False},
        "streaming": {"body": {"streaming_enabled": True}},
    }
    normalize_body_streaming_config(config)
    assert config["body"]["enabled"] is True
    assert config["body"]["inference_backend"] == "mediapipe_pose"
    assert config["body"]["detect_left_arm"] is False
    assert config["body"]["detect_right_arm"] is True
    assert config["body"]["detect_torso"] is True
    assert config["mediapipe_pose"]["enabled"] is True


def test_body_dual_camera_fusion_prefers_higher_confidence_joint():
    from modules.body_module import BodyModule

    module = BodyModule(
        {
            "enabled": True,
            "processing": {
                "dual_camera_body_enabled": True,
                "fusion_min_confidence": 0.1,
                "fusion_confidence_margin": 0.2,
                "fusion_max_pose_delta": 10.0,
            },
        },
        {"streaming_enabled": False},
    )
    primary = {
        "pose_3d": {"LeftElbowTracker": [0.0, 0.5, 0.0], "HipsTracker": [0.0, 0.0, 0.0]},
        "keypoints_2d": {"left_elbow": (10, 20, 0.25), "left_hip": (0, 0, 0.9), "right_hip": (1, 0, 0.9)},
        "confidence": 0.4,
        "source": "yolo",
    }
    secondary = {
        "pose_3d": {"LeftElbowTracker": [0.4, 0.7, 0.0], "HipsTracker": [0.1, 0.0, 0.0]},
        "keypoints_2d": {"left_elbow": (11, 21, 0.9), "left_hip": (0, 0, 0.6), "right_hip": (1, 0, 0.6)},
        "confidence": 0.8,
        "source": "secondary_yolo",
    }
    fused = module._fuse_dual_camera_pose(primary, secondary)
    assert fused["secondary_used_nodes"] >= 1
    assert fused["pose_3d"]["LeftElbowTracker"][0] > 0.2
    assert fused["pose_3d"]["HipsTracker"][0] < 0.1
    assert fused["source"] == "yolo+dual_camera"


def test_body_module_mediapipe_pose_backend_skips_yolo_detector():
    from modules.body_module import BodyModule

    original_mp_cls = BodyModule._init_mediapipe_pose.__globals__["MediaPipePoseDetector"]
    BodyModule._init_mediapipe_pose.__globals__["MediaPipePoseDetector"] = _FakeMediaPipePoseDetector
    module = None
    try:
        module = BodyModule(
            config={
                "enabled": True,
                "inference_backend": "mediapipe_pose",
                "processing": {"smoothing": "linear", "smoothing_factor": 1.0},
                "mediapipe_pose": {
                    "enabled": True,
                    "model_variant": "heavy",
                    "model_path": "pose_landmarker_heavy.task",
                },
            },
            streaming_config={"streaming_enabled": False},
        )
        assert module.initialize() is True
        assert module._detector is None
        result = module.process(np.zeros((64, 64, 3), dtype=np.uint8))
        assert result is not None
        assert result["source"] == "mediapipe_pose"
        assert module.get_status()["inference_backend"] == "mediapipe_pose"
    finally:
        if module is not None:
            module.cleanup()
        BodyModule._init_mediapipe_pose.__globals__["MediaPipePoseDetector"] = original_mp_cls


def test_lite_edition_forces_mediapipe_backend_without_importing_yolo():
    import edition
    from modules.body_module import BodyModule

    with mock.patch.dict(os.environ, {"YOLO26N_EDITION": "lite"}):
        assert edition.current_edition() == "lite"
        info = edition.edition_info()
        assert info["yolo_available"] is False
        assert info["body_backends"] == ["mediapipe_pose"]
        assert BodyModule._normalize_inference_backend("yolo") == "mediapipe_pose"
        assert BodyModule._normalize_inference_backend("bogus") == "mediapipe_pose"

        config = {"body": {"inference_backend": "yolo"}}
        normalize_body_streaming_config(config)
        assert config["body"]["inference_backend"] == "mediapipe_pose"

        import dashboard_server
        hardware = dashboard_server.detect_hardware()
        assert hardware["edition"] == "lite"
        assert [d["id"] for d in hardware["devices"]] == ["cpu"]

    with mock.patch.dict(os.environ, {"YOLO26N_EDITION": ""}):
        expected = "full" if edition.yolo_installed() else "lite"
        assert edition.current_edition() == expected


def test_body_module_and_dashboard_import_without_yolo_stack():
    import subprocess

    probe = (
        "import sys; sys.path.insert(0, 'bin');"
        "import modules, dashboard_server, main;"
        "bad = [m for m in ('torch', 'ultralytics', 'yolo_detector') if m in sys.modules];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"eager YOLO imports: {result.stdout} {result.stderr[-400:]}"


def test_body_pose_source_modes_only_run_required_detectors():
    from modules.body_module import BodyModule

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    yolo_pose = {"pose_3d": {"Head": [0.0, 1.0, 0.0]}, "source": "yolo"}
    mp_pose = {"pose_3d": {"Head": [0.0, 2.0, 0.0]}, "source": "mediapipe_pose"}
    # Detectors are faked, so this exercises full-edition source selection
    # even when the lite environment has no torch/ultralytics installed.
    with mock.patch("modules.body_module.yolo_available", return_value=True):
        module = BodyModule(
            config={"enabled": True, "inference_backend": "yolo"},
            streaming_config={"streaming_enabled": False},
        )
    calls = {"yolo": 0, "mp": 0}
    yolo_result = [yolo_pose]
    mp_result = [mp_pose]

    def run_yolo(*args, **kwargs):
        calls["yolo"] += 1
        return yolo_result[0]

    def run_mp(*args, **kwargs):
        calls["mp"] += 1
        return mp_result[0]

    module._process_yolo_pose = run_yolo
    module._process_mediapipe_pose = run_mp

    module._mp_pose_cfg = {"mode": "fallback"}
    assert module._process_pose_sources(frame)["source"] == "yolo"
    assert calls == {"yolo": 1, "mp": 0}

    yolo_result[0] = None
    assert module._process_pose_sources(frame)["source"] == "mediapipe_pose"
    assert calls == {"yolo": 2, "mp": 1}

    calls.update(yolo=0, mp=0)
    yolo_result[0] = yolo_pose
    module._mp_pose_cfg = {"mode": "prefer_mediapipe"}
    assert module._process_pose_sources(frame)["source"] == "mediapipe_pose"
    assert calls == {"yolo": 0, "mp": 1}

    calls.update(yolo=0, mp=0)
    module._mp_pose_cfg = {"mode": "blend", "blend_weight": 0.5}
    assert module._process_pose_sources(frame)["source"] == "yolo+mediapipe_pose"
    assert calls == {"yolo": 1, "mp": 1}


def test_body_part_switches_filter_before_retargeting():
    from modules.body_module import BodyModule

    module = BodyModule(
        {
            "enabled": True,
            "inference_backend": "mediapipe_pose",
            "detect_torso": True,
            "detect_left_arm": False,
            "detect_right_arm": True,
            "detect_left_leg": True,
            "detect_right_leg": True,
        },
        {"streaming_enabled": False},
    )
    pose = {
        "pose_3d": {
            "Hips": [0, 0, 0],
            "LeftUpperArm": [-0.2, 0.5, 0],
            "LeftElbowTracker": [-0.3, 0.4, 0],
            "RightUpperArm": [0.2, 0.5, 0],
            "RightElbowTracker": [0.3, 0.4, 0],
            "LeftFootTracker": [-0.1, -1, 0],
            "RightFootTracker": [0.1, -1, 0],
        },
        "rotations": {},
    }
    filtered = module._filter_body_parts(pose)
    assert "LeftUpperArm" not in filtered["pose_3d"]
    assert "LeftElbowTracker" not in filtered["pose_3d"]
    assert "RightUpperArm" in filtered["pose_3d"]


def test_mediapipe_pose_world_converter_keeps_upper_body_when_lower_body_low_visibility():
    from mediapipe_pose import MediaPipePoseDetector

    detector = MediaPipePoseDetector()
    landmarks = [_FakeLandmark(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(33)]

    landmarks[0] = _FakeLandmark(-0.10, -0.55, -0.35, 0.99, 0.99)
    landmarks[11] = _FakeLandmark(0.18, -0.42, -0.10, 0.99, 0.99)
    landmarks[12] = _FakeLandmark(-0.18, -0.46, -0.05, 0.99, 0.99)
    landmarks[13] = _FakeLandmark(0.28, -0.20, -0.08, 0.95, 0.95)
    landmarks[14] = _FakeLandmark(-0.28, -0.22, -0.06, 0.95, 0.95)
    landmarks[15] = _FakeLandmark(0.32, 0.03, -0.06, 0.90, 0.90)
    landmarks[16] = _FakeLandmark(-0.32, 0.02, -0.04, 0.90, 0.90)

    # Common webcam framing: MediaPipe estimates lower-body world points but
    # marks them as barely visible. The body backend must still emit pose nodes.
    landmarks[23] = _FakeLandmark(0.10, 0.00, 0.02, 0.003, 0.001)
    landmarks[24] = _FakeLandmark(-0.10, -0.01, -0.02, 0.003, 0.001)
    landmarks[25] = _FakeLandmark(0.08, 0.40, 0.01, 0.0001, 0.0001)
    landmarks[26] = _FakeLandmark(-0.05, 0.38, -0.05, 0.0001, 0.0001)

    pose = detector._world_landmarks_to_pose(landmarks)
    assert len(pose) >= 20
    for node in [
        "HipsTracker",
        "ChestTracker",
        "LeftKneeTracker",
        "RightKneeTracker",
        "LeftElbowTracker",
        "RightElbowTracker",
    ]:
        assert node in pose


def test_vrcft_calibration_remaps_without_changing_status_params():
    raw = {
        "v2/JawOpen": 0.4,
        "v2/JawX": 0.25,
        "ExpressionTrackingActive": 1.0,
    }
    disabled = apply_vrcft_calibration(raw, {"enabled": False})
    assert disabled == raw

    calibrated = apply_vrcft_calibration(
        raw,
        {
            "enabled": True,
            "min_values": {"v2/JawOpen": 0.2, "v2/JawX": -0.5},
            "max_values": {"v2/JawOpen": 0.6, "v2/JawX": 0.5},
        },
    )
    assert abs(calibrated["v2/JawOpen"] - 0.5) < 1e-9
    assert abs(calibrated["v2/JawX"] - 0.5) < 1e-9
    assert calibrated["ExpressionTrackingActive"] == 1.0

    fallback = apply_vrcft_calibration(
        {"v2/JawOpen": 0.8},
        {"enabled": True, "min_values": {"v2/JawOpen": 0.5}, "max_values": {"v2/JawOpen": 0.5}},
    )
    assert fallback["v2/JawOpen"] == 0.8


def test_face_mouth_shape_mapping_outputs_specific_vrcft_shapes():
    detector = FaceLandmarkDetector(
        smoothing="linear",
        smoothing_factor=1.0,
        mouth_shape_mapping_enabled=True,
        mouth_shape_gain=1.25,
        mouth_shape_deadzone=0.03,
    )
    vrcft = detector._blendshapes_to_vrcft({
        "jawOpen": 0.50,
        "mouthFunnel": 0.60,
        "mouthPucker": 0.70,
        "mouthSmileLeft": 0.40,
        "mouthSmileRight": 0.60,
        "mouthFrownLeft": 0.30,
        "mouthFrownRight": 0.50,
        "mouthPressLeft": 0.20,
        "mouthPressRight": 0.40,
        "mouthStretchLeft": 0.10,
        "mouthStretchRight": 0.30,
        "mouthDimpleLeft": 0.25,
        "mouthDimpleRight": 0.35,
        "mouthLowerDownLeft": 0.20,
        "mouthLowerDownRight": 0.30,
        "mouthUpperUpLeft": 0.10,
        "mouthUpperUpRight": 0.20,
        "mouthLeft": 0.02,
        "mouthRight": 0.50,
        "jawLeft": 0.60,
        "jawRight": 0.10,
    })

    assert vrcft["v2/LipPucker"] > vrcft["v2/LipFunnel"] > 0.0
    assert vrcft["v2/MouthSmileLeft"] > 0.0
    assert abs(vrcft["v2/MouthSmile"] - (
        vrcft["v2/MouthSmileLeft"] + vrcft["v2/MouthSmileRight"]
    ) * 0.5) < 1e-9
    assert "v2/MouthFrown" in vrcft
    assert "v2/MouthPress" in vrcft
    assert "v2/MouthStretch" in vrcft
    assert "v2/MouthDimple" in vrcft
    assert vrcft["v2/MouthOpen"] >= vrcft["v2/JawOpen"]
    assert 0.0 < vrcft["v2/MouthX"] <= 1.0
    assert -1.0 <= vrcft["v2/JawX"] < 0.0

    deadzone = detector._blendshapes_to_vrcft({
        "mouthPucker": 0.02,
        "mouthRight": 0.02,
    })
    assert deadzone["v2/LipPucker"] == 0.0
    assert deadzone["v2/MouthX"] == 0.0


def test_face_mouth_shape_mapping_can_be_disabled():
    detector = FaceLandmarkDetector(
        mouth_shape_mapping_enabled=False,
        mouth_shape_gain=2.0,
        mouth_shape_deadzone=0.2,
    )
    vrcft = detector._blendshapes_to_vrcft({
        "mouthPucker": 0.02,
        "mouthRight": 0.30,
    })

    assert abs(vrcft["v2/LipPucker"] - 0.02) < 1e-9
    assert abs(vrcft["v2/MouthX"] - 0.30) < 1e-9


def test_face_eyelid_mapping_keeps_left_and_right_closed_independent():
    detector = FaceLandmarkDetector(
        smoothing="linear",
        smoothing_factor=1.0,
        eyelid_independent_enabled=True,
        eyelid_close_gain=1.35,
        eyelid_close_deadzone=0.04,
    )
    vrcft = detector._blendshapes_to_vrcft({
        "eyeBlinkLeft": 0.80,
        "eyeBlinkRight": 0.03,
        "eyeWideRight": 0.40,
    })

    assert vrcft["v2/EyeClosedLeft"] > 0.9
    assert vrcft["v2/EyeClosedRight"] == 0.0
    assert vrcft["v2/EyeOpenLeft"] < vrcft["v2/EyeOpenRight"]
    assert vrcft["v2/EyeLidLeft"] < vrcft["v2/EyeLidRight"]
    assert vrcft["v2/EyeWideRight"] > vrcft["v2/EyeWideLeft"]
    assert "v2/EyeLid" not in vrcft
    assert "v2/EyeClosed" not in vrcft
    assert "v2/EyeOpen" not in vrcft
    assert "v2/EyeSquint" not in vrcft
    assert vrcft["EyeTrackingActive"] == 1.0


def test_face_eyelid_combined_output_is_explicit_legacy_option():
    detector = FaceLandmarkDetector(
        smoothing="linear",
        smoothing_factor=1.0,
        eyelid_independent_enabled=True,
        eyelid_send_combined=True,
    )
    vrcft = detector._blendshapes_to_vrcft({
        "eyeBlinkLeft": 0.75,
        "eyeBlinkRight": 0.05,
        "eyeSquintLeft": 0.4,
        "eyeSquintRight": 0.1,
    })

    assert "v2/EyeLid" in vrcft
    assert "v2/EyeClosed" in vrcft
    assert "v2/EyeOpen" in vrcft
    assert "v2/EyeSquint" in vrcft


def test_face_vmc_mapping_matches_emitted_mouth_vrcft_names():
    mapping = FaceVMCStreamer.VRCFT_TO_VMC_BLENDSHAPE

    assert mapping["v2/LipFunnel"] == "lip_funnel"
    assert mapping["v2/LipPucker"] == "lip_pucker"
    assert mapping["v2/MouthX"] == "mouth_x"
    assert mapping["v2/JawX"] == "jaw_x"
    assert mapping["v2/MouthSmileLeft"] == "mouth_smile_l"
    assert mapping["v2/MouthFrownRight"] == "mouth_frown_r"
    assert mapping["v2/MouthPressLeft"] == "mouth_press_l"
    assert mapping["v2/MouthStretchRight"] == "mouth_stretch_r"
    assert mapping["v2/EyeClosedLeft"] == "eye_closed_l"
    assert mapping["v2/EyeClosedRight"] == "eye_closed_r"
    assert mapping["v2/EyeLidLeft"] == "eye_lid_l"
    assert mapping["v2/EyeOpenRight"] == "eye_open_r"
    assert "v2/MouthFunnel" not in mapping
    assert "v2/MouthPucker" not in mapping


def test_face_async_result_is_consumed_once():
    detector = FaceLandmarkDetector()
    first = object()
    second = object()

    detector._on_face_result(first, None, 100)
    assert detector._take_latest_result() is first
    assert detector._take_latest_result() is None

    detector._on_face_result(object(), None, 99)
    assert detector._take_latest_result() is None

    detector._on_face_result(second, None, 101)
    assert detector._take_latest_result() is second
    assert detector._take_latest_result() is None


def test_face_module_disable_then_enable_restores_streamer_output():
    from modules.face_module import FaceModule

    class _LifecycleStreamer:
        def __init__(self):
            self.inference_enabled = True
            self.enabled = True
            self.connected = True

        def disconnect(self):
            self.connected = False

        def connect(self):
            self.connected = True
            return True

    module = FaceModule(
        config={"enabled": True},
        streaming_config={"streaming_enabled": True},
    )
    module._streamer = _LifecycleStreamer()
    module._initialized = True
    module.enabled = False
    module.enabled = True

    assert module._streamer.enabled is True
    assert module._streamer.inference_enabled is True
    assert module._streamer.connected is True


def test_face_state_roundtrip(tmp_path):
    path = tmp_path / "face_state.json"
    write_face_state(
        {
            "v2/JawOpen": 0.3,
            "v2/MouthX": -0.2,
            "ExpressionTrackingActive": 1.0,
            "legacy": 0.5,
        },
        path=path,
    )
    state = read_face_state(path)
    assert state["vrcft_raw"] == {"v2/JawOpen": 0.3, "v2/MouthX": -0.2}


def main():
    test_average_smoothing_reduces_jitter()
    test_base_module_rate_limit_reuses_preview_result_without_restreaming()
    test_base_module_deadline_scheduler_avoids_input_rate_quantization()
    test_timestamp_rate_uses_elapsed_intervals()
    test_mediapipe_timestamp_is_strictly_increasing()
    test_camera_hardware_zoom_property_is_applied()
    test_camera_get_frame_with_count_can_avoid_copy()
    test_camera_wait_for_frame_wakes_only_for_new_frame()
    test_camera_open_uses_same_directshow_index_space_as_dashboard()
    test_camera_detection_preserves_physical_index()
    test_dashboard_camera_refresh_bypasses_cache()
    test_dashboard_tracking_status_exposes_runtime_details()
    test_dashboard_tracking_status_accepts_live_external_mocap_state()
    test_dashboard_mocap_status_discovers_external_process_when_locally_stopped()
    test_dashboard_inference_status_prefers_structured_runtime_state()
    test_hmd_anchor_aligns_unity_pose_without_camera_root_offset()
    test_coordinate_conversion_suppresses_known_gimbal_lock_warning()
    test_osc_hmd_relative_offsets_do_not_double_apply_hmd_height()
    test_hmd_bound_body_height_scales_to_seated_hmd_eye_height()
    test_hmd_height_constraint_requires_explicit_floor_height()
    test_hmd_actual_height_requires_floor_like_origin()
    test_openvr_unity_rotation_conversion_preserves_identity()
    test_openvr_pose_stales_out_and_clears_height()
    test_hmd_provider_without_first_packet_is_stale()
    test_explicitly_invalid_hmd_packet_is_stale()
    test_polling_hmd_providers_stale_out_and_clear_height()
    test_quest_link_failed_openvr_does_not_mask_http_fallback()
    test_vrchat_osc_rejects_non_finite_pose_values()
    test_malformed_hmd_json_invalidates_previous_pose()
    test_sender_invalid_packet_clears_remote_height()
    test_legacy_hmd_packets_never_claim_floor_relative_height()
    test_sdk_json_height_requires_floor_origin_or_explicit_height()
    test_standing_height_auto_learning_only_raises_on_stable_samples()
    test_standing_height_auto_learning_can_lower_for_stable_standing_pose()
    test_standing_height_auto_learning_does_not_lower_for_crouched_pose()
    test_seated_solver_grounds_ankles_and_keeps_knees_bent()
    test_steamvr_explicit_height_rebuilds_floor_relative_hmd_y()
    test_osc_explicit_height_ignores_device_local_hmd_y()
    test_hmd_height_sequence_keeps_trackers_above_floor_and_recovers_pose()
    test_hmd_height_constraint_prefers_actual_height_over_position_y()
    test_osc_and_steamvr_share_explicit_floor_height_base_for_local_pose()
    test_hmd_height_solver_retargets_crouch_and_seated_pose()
    test_hmd_height_solver_keeps_sitting_from_being_learned_as_standing()
    test_hmd_height_solver_preserves_calibrated_standing_height()
    test_hmd_height_solver_grounds_standing_feet_to_actual_floor()
    test_dashboard_api_provider_copies_actual_height_and_invalidates_stale_data()
    test_mocap_uses_dashboard_bridge_when_launched_by_dashboard()
    test_body_module_hot_reload_reenable_restores_inference_streamer()
    test_body_module_hot_reload_updates_running_streamer_switches()
    test_body_module_retains_hmd_provider_across_streamer_rebuild()
    test_body_module_holds_last_pose_for_hmd_height_refresh()
    test_body_osc_decouple_toggle_and_head_position_policy()
    test_hmd_anchor_trust_boundary_is_shared_by_body_streamers()
    test_body_osc_rejects_untrusted_hmd_anchor_but_protects_head_position()
    test_body_module_status_reports_hmd_without_body_frame()
    test_hand_osc_sends_vrchat_desktop_gesture_parameters()
    test_hand_desktop_osc_reaches_local_udp_endpoint()
    test_hand_tracker_keeps_left_and_right_temporal_state_independent()
    test_hand_tracker_joint_angle_curls_drive_individual_fingers()
    test_hand_tracker_builds_left_and_right_body_wrist_rois()
    test_hand_module_passes_previous_body_keypoints_to_roi_detector()
    test_mediapipe_pose_exports_image_wrist_keypoints_for_hand_roi()
    test_hand_vmt_sends_independent_open_amount_for_each_finger()
    test_hand_vmt_pre_registers_compatible_controllers_without_active_pose()
    test_hand_vmt_reregisters_controllers_when_endpoint_becomes_ready()
    test_hand_vmt_tracking_loss_keeps_controllers_registered_by_default()
    test_hand_vmt_force_reset_disconnects_controllers()
    test_hand_vmt_local_mode_keeps_remote_mapping_from_overwriting_driver_host()
    test_vmt_runtime_status_requires_driver_steamvr_and_udp_listener()
    test_vmt_runtime_settings_expose_startup_controller_registration()
    test_vmt_device_list_parser_identifies_expected_virtual_controllers()
    test_hand_osc_resets_desktop_gesture_after_tracking_loss()
    test_hand_streamer_keeps_desktop_and_vmt_targets_independent()
    test_hand_streamer_accepts_custom_desktop_gesture_parameter_names()
    test_hand_module_forwards_empty_frames_for_desktop_reset()
    test_hand_module_forwards_body_pose_data_to_streamer()
    test_hand_deferred_streaming_uses_matching_body_frame()
    test_hand_vmt_uses_payload_pose_3d_when_vmt_enabled()
    test_hand_vmt_keeps_desktop_gesture_and_curl_output_when_vmt_enabled()
    test_hand_vmt_empty_hand_frame_triggers_neutral_reset()
    test_hand_module_disable_then_enable_restores_streamer_output()
    test_streamer_queue_is_latest_wins_and_reports_drops()
    test_dashboard_config_api_reports_persistence_failure()
    test_dashboard_config_recursive_merge_preserves_nested_siblings()
    test_dashboard_quick_preset_config_is_normalized()
    test_dashboard_performance_tiers_and_rtx_3060_baseline()
    test_dashboard_performance_recommendation_safely_falls_back_without_optional_dependencies()
    test_dashboard_ui_active_section_is_normalized()
    test_tracker_nodes_bind_to_intended_joint_positions()
    test_head_is_simulated_when_nose_is_missing()
    test_process_detections_injects_synthetic_nose_when_head_is_missing()
    test_body_osc_uses_tracker_nodes_for_knee_and_elbow_positions()
    test_body_steamvr_uses_tracker_nodes_for_knee_and_elbow_positions()
    test_body_steamvr_hmd_base_moves_trackers_when_not_decoupled()
    test_body_steamvr_rejects_untrusted_device_origin_without_height()
    test_body_steamvr_driver_pose_reply_parser_is_robust()
    test_body_steamvr_anchors_in_driver_raw_space_without_height_override()
    test_body_steamvr_falls_back_to_provider_when_driver_pose_missing()
    test_body_steamvr_auto_yaw_aligns_body_with_hmd_heading()
    test_body_steamvr_decoupled_mode_skips_driver_hmd_query()
    test_body_streaming_config_defaults_to_hmd_base_and_preserves_debug_head_position()
    test_body_streaming_config_inherits_legacy_vrc_switches_when_body_keys_missing()
    test_body_streaming_config_keeps_legacy_vrc_mirror_in_sync()
    test_high_seated_hmd_is_not_labeled_standing_without_leg_confidence()
    test_body_streaming_enabled_turns_on_body_inference()
    test_body_inference_backend_config_enables_mediapipe_pose()
    test_body_dual_camera_fusion_prefers_higher_confidence_joint()
    test_body_module_mediapipe_pose_backend_skips_yolo_detector()
    test_lite_edition_forces_mediapipe_backend_without_importing_yolo()
    test_body_module_and_dashboard_import_without_yolo_stack()
    test_body_pose_source_modes_only_run_required_detectors()
    test_body_part_switches_filter_before_retargeting()
    test_mediapipe_pose_world_converter_keeps_upper_body_when_lower_body_low_visibility()
    test_body_preview_draws_reconstructed_pose_without_2d_keypoints()
    import tempfile
    from pathlib import Path

    class _Tmp:
        def __enter__(self):
            self._dir = tempfile.TemporaryDirectory()
            return Path(self._dir.name)

        def __exit__(self, exc_type, exc, tb):
            self._dir.cleanup()

    test_vrcft_calibration_remaps_without_changing_status_params()
    test_face_mouth_shape_mapping_outputs_specific_vrcft_shapes()
    test_face_mouth_shape_mapping_can_be_disabled()
    test_face_eyelid_mapping_keeps_left_and_right_closed_independent()
    test_face_eyelid_combined_output_is_explicit_legacy_option()
    test_face_vmc_mapping_matches_emitted_mouth_vrcft_names()
    test_face_async_result_is_consumed_once()
    test_face_module_disable_then_enable_restores_streamer_output()
    with _Tmp() as tmp:
        test_face_state_roundtrip(tmp)
        test_runtime_tracking_state_roundtrip_and_staleness(tmp)
        test_runtime_tracking_state_retries_windows_reader_lock(tmp)
        test_frozen_runtime_root_falls_back_to_local_app_data(tmp)
        test_dashboard_load_config_returns_detached_snapshot(tmp)
        test_dashboard_runtime_config_precedes_newer_bundled_default(tmp)
        test_dashboard_fallback_becomes_active_mocap_config(tmp)
        test_dashboard_concurrent_config_updates_preserve_both_patches(tmp)
        test_runtime_model_fallback_rejects_non_pose_yolo_weights(tmp)
    print("Runtime logic checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
