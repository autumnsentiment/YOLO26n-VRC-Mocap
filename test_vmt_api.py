#!/usr/bin/env python3
"""Non-destructive checks for the VMT runtime and installer API."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bin"))

import dashboard_server


def test_vmt_status_api():
    expected = {
        "host": "127.0.0.1",
        "port": 39570,
        "local_mode": True,
        "driver_installed": True,
        "driver_paths": [r"C:\SteamVR\drivers\vmt"],
        "manager_path": r"C:\VMT\vmt_manager.exe",
        "steamvr_running": True,
        "udp_port_listening": True,
        "ready": True,
        "message": "VMT driver is ready",
    }
    with patch.object(
        dashboard_server,
        "get_vmt_runtime_status",
        return_value=expected,
    ) as mocked_status:
        response = dashboard_server.app.test_client().get("/api/vmt/status")

    assert response.status_code == 200, response.get_data(as_text=True)
    assert response.get_json() == expected
    kwargs = mocked_status.call_args.kwargs
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 39570


def test_vmt_devices_api_distinguishes_presence_from_vrchat_consumption():
    config = {
        "streaming": {
            "hand": {
                "vmt_local_mode": True,
                "vmt_host": "127.0.0.1",
                "vmt_port": 39570,
                "vmt_left_device_index": 1,
                "vmt_right_device_index": 2,
                "vmt_hand_selection_priority": 100,
                "vmt_register_on_steamvr_startup": True,
            }
        }
    }
    query_result = {
        "ok": True,
        "response_received": True,
        "devices": [
            {"openvr_index": 3, "serial": "VMT_1", "is_vmt": True},
            {"openvr_index": 4, "serial": "VMT_2", "is_vmt": True},
        ],
        "vmt_devices": [
            {"openvr_index": 3, "serial": "VMT_1", "is_vmt": True},
            {"openvr_index": 4, "serial": "VMT_2", "is_vmt": True},
        ],
        "error": None,
    }
    openvr_result = {
        "ok": True,
        "runtime_available": True,
        "left_index": 3,
        "right_index": 4,
        "left_serial": "VMT_1",
        "right_serial": "VMT_2",
        "devices": [
            {
                "openvr_index": 3,
                "serial": "VMT_1",
                "is_vmt": True,
                "pose_valid": True,
                "hand_selection_priority": 100,
                "selected_as_left": True,
                "selected_as_right": False,
            },
            {
                "openvr_index": 4,
                "serial": "VMT_2",
                "is_vmt": True,
                "pose_valid": True,
                "hand_selection_priority": 100,
                "selected_as_left": False,
                "selected_as_right": True,
            },
        ],
        "vmt_devices": [],
        "error": None,
    }
    runtime_status = {
        "ready": True,
        "skeleton_input": True,
        "add_compatible_controller_on_startup": True,
    }
    with patch.object(
        dashboard_server, "load_config", return_value=config
    ), patch.object(
        dashboard_server, "query_vmt_devices", return_value=query_result
    ) as mocked_query, patch.object(
        dashboard_server, "query_openvr_devices", return_value=openvr_result
    ), patch.object(
        dashboard_server, "get_vmt_runtime_status", return_value=runtime_status
    ):
        response = dashboard_server.app.test_client().post("/api/vmt/devices")

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["expected_devices_present"] is True
    assert payload["missing_serials"] == []
    assert payload["selected_as_left"] is True
    assert payload["selected_as_right"] is True
    assert payload["steamvr_controller_selection_confirmed"] is True
    assert payload["controller_takeover_ready"] is True
    assert payload["restart_required"] is False
    assert payload["vrchat_consumption_confirmed"] is False
    mocked_query.assert_called_once_with(host="127.0.0.1", port=39570)


def test_vmt_devices_api_reports_stale_priority_and_restart_blockers():
    config = {
        "streaming": {
            "hand": {
                "vmt_local_mode": True,
                "vmt_host": "127.0.0.1",
                "vmt_port": 39570,
                "vmt_left_device_index": 1,
                "vmt_right_device_index": 2,
                "vmt_hand_selection_priority": 100,
                "vmt_register_on_steamvr_startup": True,
            }
        }
    }
    query_result = {
        "ok": True,
        "response_received": True,
        "devices": [],
        "vmt_devices": [
            {"openvr_index": 3, "serial": "VMT_1", "is_vmt": True},
            {"openvr_index": 4, "serial": "VMT_2", "is_vmt": True},
        ],
        "error": None,
    }
    openvr_result = {
        "ok": True,
        "runtime_available": True,
        "left_index": None,
        "right_index": None,
        "left_serial": None,
        "right_serial": None,
        "devices": [
            {
                "openvr_index": 3,
                "serial": "VMT_1",
                "is_vmt": True,
                "pose_valid": False,
                "hand_selection_priority": 10,
            },
            {
                "openvr_index": 4,
                "serial": "VMT_2",
                "is_vmt": True,
                "pose_valid": False,
                "hand_selection_priority": 10,
            },
        ],
        "vmt_devices": [],
        "error": None,
    }
    runtime_status = {
        "ready": True,
        "skeleton_input": True,
        "add_compatible_controller_on_startup": False,
    }
    with patch.object(
        dashboard_server, "load_config", return_value=config
    ), patch.object(
        dashboard_server, "query_vmt_devices", return_value=query_result
    ), patch.object(
        dashboard_server, "query_openvr_devices", return_value=openvr_result
    ), patch.object(
        dashboard_server, "get_vmt_runtime_status", return_value=runtime_status
    ):
        response = dashboard_server.app.test_client().post("/api/vmt/devices")

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    blocker_codes = {item["code"] for item in payload["blockers"]}
    assert payload["expected_devices_present"] is True
    assert payload["steamvr_controller_selection_confirmed"] is False
    assert payload["controller_takeover_ready"] is False
    assert payload["restart_required"] is True
    assert "startup_registration_mismatch" in blocker_codes
    assert "active_priority_stale" in blocker_codes
    assert "left_not_selected" in blocker_codes
    assert "right_not_selected" in blocker_codes
    assert payload["vrchat_consumption_confirmed"] is False


def test_vmt_install_uses_existing_manager():
    manager = Path(r"C:\VMT\vmt_manager.exe")
    with patch.object(
        dashboard_server,
        "find_vmt_manager",
        return_value=manager,
    ), patch.object(
        dashboard_server,
        "launch_elevated",
        return_value=True,
    ) as mocked_launch, patch.object(
        dashboard_server,
        "download_vmt_installer",
    ) as mocked_download:
        response = dashboard_server.app.test_client().post("/api/vmt/install")

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["stage"] == "driver_install"
    assert payload["manager_path"] == str(manager)
    mocked_launch.assert_called_once_with(manager, "install")
    mocked_download.assert_not_called()


def test_vmt_install_downloads_official_setup_without_manager():
    with tempfile.TemporaryDirectory() as tmp:
        installer = Path(tmp) / dashboard_server.VMT_INSTALLER_NAME
        installer.write_bytes(b"MZ mocked installer")
        with patch.object(
            dashboard_server,
            "find_vmt_manager",
            return_value=None,
        ), patch.object(
            dashboard_server,
            "vmt_installer_is_running",
            return_value=False,
        ), patch.object(
            dashboard_server,
            "download_vmt_installer",
            return_value=installer,
        ) as mocked_download, patch.object(
            dashboard_server,
            "launch_elevated",
            return_value=True,
        ) as mocked_launch:
            response = dashboard_server.app.test_client().post("/api/vmt/install")

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["status"] == "warning"
    assert payload["stage"] == "manager_setup"
    assert payload["installer_path"] == str(installer)
    assert mocked_download.call_count == 1
    mocked_launch.assert_called_once_with(installer)


def test_vmt_install_does_not_launch_duplicate_setup():
    with patch.object(
        dashboard_server,
        "find_vmt_manager",
        return_value=None,
    ), patch.object(
        dashboard_server,
        "vmt_installer_is_running",
        return_value=True,
    ), patch.object(
        dashboard_server,
        "download_vmt_installer",
    ) as mocked_download, patch.object(
        dashboard_server,
        "launch_elevated",
    ) as mocked_launch:
        response = dashboard_server.app.test_client().post("/api/vmt/install")

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["status"] == "warning"
    assert payload["stage"] == "installer_running"
    mocked_download.assert_not_called()
    mocked_launch.assert_not_called()


def test_vmt_install_reports_launch_failure():
    manager = Path(r"C:\VMT\vmt_manager.exe")
    with patch.object(
        dashboard_server,
        "find_vmt_manager",
        return_value=manager,
    ), patch.object(
        dashboard_server,
        "launch_elevated",
        return_value=False,
    ):
        response = dashboard_server.app.test_client().post("/api/vmt/install")

    assert response.status_code == 500
    payload = response.get_json()
    assert payload["status"] == "error"
    assert "VMT" in payload["message"]


def main():
    test_vmt_status_api()
    test_vmt_devices_api_distinguishes_presence_from_vrchat_consumption()
    test_vmt_devices_api_reports_stale_priority_and_restart_blockers()
    test_vmt_install_uses_existing_manager()
    test_vmt_install_downloads_official_setup_without_manager()
    test_vmt_install_does_not_launch_duplicate_setup()
    test_vmt_install_reports_launch_failure()
    print("VMT API checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
