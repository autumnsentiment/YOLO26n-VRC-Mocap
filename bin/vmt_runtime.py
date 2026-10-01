"""Virtual Motion Tracker runtime discovery and installer helpers."""

import ctypes
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional


VMT_INSTALLER_URL = (
    "https://github.com/gpsnmeajp/VirtualMotionTracker/releases/download/"
    "v0.15/VirtualMotionTrackerSetup_0.15.exe"
)
VMT_INSTALLER_NAME = "VirtualMotionTrackerSetup_0.15.exe"
_VMT_QUERY_LOCK = threading.Lock()
_OPENVR_QUERY_LOCK = threading.Lock()


def is_loopback_host(host: str) -> bool:
    value = str(host or "").strip().lower()
    if value in {"", "localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return True
    try:
        return socket.gethostbyname(value).startswith("127.")
    except OSError:
        return False


def _read_json(path: Path) -> Dict:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def openvr_paths_file() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", "")) / "openvr" / "openvrpaths.vrpath"


def load_openvr_paths() -> Dict:
    return _read_json(openvr_paths_file())


def find_steamvr_runtime() -> Optional[Path]:
    data = load_openvr_paths()
    for value in data.get("runtime", []):
        path = Path(str(value)).expanduser()
        if path.exists():
            return path.resolve()

    if sys.platform == "win32":
        try:
            import winreg

            for key_path in (
                r"SOFTWARE\WOW6432Node\Valve\Steam",
                r"SOFTWARE\Valve\Steam",
            ):
                try:
                    key = winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE,
                        key_path,
                        0,
                        winreg.KEY_READ,
                    )
                    steam_path, _ = winreg.QueryValueEx(key, "InstallPath")
                    winreg.CloseKey(key)
                    runtime = Path(steam_path) / "steamapps" / "common" / "SteamVR"
                    if runtime.exists():
                        return runtime.resolve()
                except OSError:
                    continue
        except ImportError:
            pass
    return None


def _unique_existing_paths(paths: Iterable[Path]) -> List[Path]:
    result = []
    seen = set()
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        result.append(resolved)
    return result


def find_vmt_driver_paths(project_root: Optional[Path] = None) -> List[Path]:
    candidates = []
    runtime = find_steamvr_runtime()
    if runtime:
        candidates.append(runtime / "drivers" / "vmt")

    for value in load_openvr_paths().get("external_drivers", []):
        path = Path(str(value))
        if path.name.lower() == "vmt" or "virtualmotiontracker" in str(path).lower():
            candidates.append(path)

    if project_root:
        candidates.append(Path(project_root) / "driver_files" / "vmt")

    return [
        path
        for path in _unique_existing_paths(candidates)
        if (path / "driver.vrdrivermanifest").exists()
    ]


def _optional_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


def load_vmt_driver_settings(project_root: Optional[Path] = None) -> Dict:
    """Read the active VMT driver's persisted settings without mutating it."""
    settings_path = None
    settings = {}
    for driver_path in find_vmt_driver_paths(project_root):
        candidate = driver_path / "setting.json"
        if candidate.is_file():
            settings_path = candidate
            settings = _read_json(candidate)
            break

    return {
        "settings_path": str(settings_path) if settings_path else None,
        "settings_found": bool(settings_path),
        "add_compatible_controller_on_startup": _optional_bool(
            settings.get("AddCompatibleControllerOnStartup")
        ),
        "add_controller_on_startup": _optional_bool(
            settings.get("AddControllerOnStartup")
        ),
        "always_compatible": _optional_bool(settings.get("AlwaysCompatible")),
        "hand_selection_priority": _optional_int(settings.get("Priority")),
        "skeleton_input": _optional_bool(settings.get("SkeletonInput")),
        "auto_pose_update_on_startup": _optional_bool(
            settings.get("AutoPoseUpdateOnStartup")
        ),
        "diag_log_on_startup": _optional_bool(settings.get("DiagLogOnStartup")),
        "optout_tracking_role": _optional_bool(
            settings.get("OptoutTrackingRole")
        ),
        "reject_when_cannot_tracking": _optional_bool(
            settings.get("RejectWhenCannotTracking")
        ),
        "receive_port": _optional_int(settings.get("ReceivePort")),
        "send_port": _optional_int(settings.get("SendPort")),
    }


def _encode_osc_string(value: str) -> bytes:
    encoded = str(value).encode("utf-8") + b"\x00"
    return encoded + (b"\x00" * ((4 - len(encoded) % 4) % 4))


def _build_osc_message(address: str, *args) -> bytes:
    tags = ","
    payload = bytearray()
    for value in args:
        if isinstance(value, int):
            tags += "i"
            payload.extend(struct.pack("!i", value))
        elif isinstance(value, float):
            tags += "f"
            payload.extend(struct.pack("!f", value))
        else:
            tags += "s"
            payload.extend(_encode_osc_string(str(value)))
    return _encode_osc_string(address) + _encode_osc_string(tags) + bytes(payload)


def _read_osc_string(packet: bytes, offset: int) -> tuple:
    end = packet.find(b"\x00", offset)
    if end < 0:
        raise ValueError("unterminated OSC string")
    value = packet[offset:end].decode("utf-8", errors="replace")
    next_offset = (end + 4) & ~3
    return value, next_offset


def _decode_vmt_device_list_packet(packet: bytes) -> Optional[str]:
    try:
        address, offset = _read_osc_string(packet, 0)
        if address != "/VMT/Out/Devices/List":
            return None
        tags, offset = _read_osc_string(packet, offset)
        if not tags.startswith(",s"):
            return None
        value, _ = _read_osc_string(packet, offset)
        return value
    except (TypeError, ValueError, IndexError, struct.error):
        return None


def parse_vmt_device_list(value: str) -> List[Dict]:
    """Parse VMT's newline-delimited ``OpenVR index:serial`` response."""
    devices = []
    for raw_line in str(value or "").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        index_text, separator, serial = line.partition(":")
        if not separator or not index_text.strip().isdigit() or not serial.strip():
            continue
        serial = serial.strip()
        devices.append({
            "openvr_index": int(index_text.strip()),
            "serial": serial,
            "is_vmt": serial.upper().startswith("VMT_"),
        })
    return devices


def _reply_host_for_endpoint(host: str, port: int) -> str:
    if is_loopback_host(host):
        return "127.0.0.1"
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((host, port))
        return str(probe.getsockname()[0])
    finally:
        probe.close()


def query_vmt_devices(
    host: str = "127.0.0.1",
    port: int = 39570,
    timeout: float = 0.8,
    restore_host: str = "127.0.0.1",
    restore_port: int = 39571,
) -> Dict:
    """Query the devices VMT can currently see in SteamVR.

    VMT has one global OSC response destination. The query temporarily points
    it at an ephemeral local socket and restores the manager destination before
    returning so Dashboard diagnostics do not permanently hijack VMT Manager.
    """
    host = str(host or "127.0.0.1")
    try:
        port = int(port)
        timeout = max(0.1, min(3.0, float(timeout)))
        restore_port = int(restore_port)
    except (TypeError, ValueError) as exc:
        return {
            "ok": False,
            "response_received": False,
            "devices": [],
            "vmt_devices": [],
            "error": f"invalid VMT query configuration: {exc}",
        }
    if not 1 <= port <= 65535:
        return {
            "ok": False,
            "response_received": False,
            "devices": [],
            "vmt_devices": [],
            "error": "VMT port must be between 1 and 65535",
        }

    result = {
        "ok": False,
        "response_received": False,
        "host": host,
        "port": port,
        "reply_host": None,
        "reply_port": None,
        "response_from": None,
        "raw_device_list": None,
        "devices": [],
        "vmt_devices": [],
        "device_count": 0,
        "vmt_device_count": 0,
        "error": None,
    }

    with _VMT_QUERY_LOCK:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        destination_changed = False
        endpoint = (host, port)
        try:
            receiver.bind(("0.0.0.0", 0))
            receiver.settimeout(timeout)
            reply_host = _reply_host_for_endpoint(host, port)
            reply_port = int(receiver.getsockname()[1])
            result["reply_host"] = reply_host
            result["reply_port"] = reply_port

            receiver.sendto(
                _build_osc_message(
                    "/VMT/Set/Destination", reply_host, reply_port
                ),
                endpoint,
            )
            destination_changed = True
            time.sleep(0.02)
            receiver.sendto(_build_osc_message("/VMT/Get/Devices/List"), endpoint)

            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("VMT device list response timed out")
                receiver.settimeout(remaining)
                packet, sender = receiver.recvfrom(16384)
                raw_list = _decode_vmt_device_list_packet(packet)
                if raw_list is None:
                    continue
                devices = parse_vmt_device_list(raw_list)
                vmt_devices = [item for item in devices if item["is_vmt"]]
                result.update({
                    "ok": True,
                    "response_received": True,
                    "response_from": f"{sender[0]}:{sender[1]}",
                    "raw_device_list": raw_list,
                    "devices": devices,
                    "vmt_devices": vmt_devices,
                    "device_count": len(devices),
                    "vmt_device_count": len(vmt_devices),
                })
                break
        except (OSError, TimeoutError) as exc:
            result["error"] = str(exc)
        finally:
            if destination_changed:
                try:
                    receiver.sendto(
                        _build_osc_message(
                            "/VMT/Set/Destination",
                            str(restore_host or "127.0.0.1"),
                            restore_port if 1 <= restore_port <= 65535 else 39571,
                        ),
                        endpoint,
                    )
                except OSError:
                    pass
            receiver.close()
    return result


def _openvr_property(system, method_name: str, index: int, prop):
    if prop is None:
        return None
    try:
        value = getattr(system, method_name)(index, prop)
    except Exception:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value


def query_openvr_devices() -> Dict:
    """Read SteamVR's actual controller selection and device properties.

    This is intentionally an on-demand diagnostic. Initializing OpenVR in the
    mocap hot path would add avoidable work and can interfere with another
    provider using the process-wide OpenVR client.
    """
    result = {
        "ok": False,
        "runtime_available": False,
        "left_index": None,
        "right_index": None,
        "left_serial": None,
        "right_serial": None,
        "devices": [],
        "vmt_devices": [],
        "error": None,
    }
    initialized = False
    with _OPENVR_QUERY_LOCK:
        try:
            import openvr

            openvr.init(openvr.VRApplication_Background)
            initialized = True
            system = openvr.VRSystem()
            result["runtime_available"] = True

            invalid_index = int(openvr.k_unTrackedDeviceIndexInvalid)

            def selected_index(role) -> Optional[int]:
                try:
                    value = int(system.getTrackedDeviceIndexForControllerRole(role))
                except Exception:
                    return None
                return None if value == invalid_index else value

            left_index = selected_index(openvr.TrackedControllerRole_LeftHand)
            right_index = selected_index(openvr.TrackedControllerRole_RightHand)
            result["left_index"] = left_index
            result["right_index"] = right_index

            try:
                poses = system.getDeviceToAbsoluteTrackingPose(
                    openvr.TrackingUniverseStanding,
                    0,
                    openvr.k_unMaxTrackedDeviceCount,
                )
            except Exception:
                poses = []

            class_names = {
                int(getattr(openvr, "TrackedDeviceClass_HMD", 1)): "hmd",
                int(getattr(openvr, "TrackedDeviceClass_Controller", 2)): "controller",
                int(getattr(openvr, "TrackedDeviceClass_GenericTracker", 3)): "tracker",
                int(getattr(openvr, "TrackedDeviceClass_TrackingReference", 4)): "tracking_reference",
                int(getattr(openvr, "TrackedDeviceClass_DisplayRedirect", 5)): "display_redirect",
            }
            role_names = {
                int(openvr.TrackedControllerRole_LeftHand): "left",
                int(openvr.TrackedControllerRole_RightHand): "right",
            }

            devices = []
            for index in range(int(openvr.k_unMaxTrackedDeviceCount)):
                try:
                    connected = bool(system.isTrackedDeviceConnected(index))
                except Exception:
                    connected = False
                if not connected:
                    continue

                serial = _openvr_property(
                    system,
                    "getStringTrackedDeviceProperty",
                    index,
                    getattr(openvr, "Prop_SerialNumber_String", None),
                )
                controller_type = _openvr_property(
                    system,
                    "getStringTrackedDeviceProperty",
                    index,
                    getattr(openvr, "Prop_ControllerType_String", None),
                )
                input_profile = _openvr_property(
                    system,
                    "getStringTrackedDeviceProperty",
                    index,
                    getattr(openvr, "Prop_InputProfilePath_String", None),
                )
                role_hint = _openvr_property(
                    system,
                    "getInt32TrackedDeviceProperty",
                    index,
                    getattr(openvr, "Prop_ControllerRoleHint_Int32", None),
                )
                priority = _openvr_property(
                    system,
                    "getInt32TrackedDeviceProperty",
                    index,
                    getattr(
                        openvr,
                        "Prop_ControllerHandSelectionPriority_Int32",
                        None,
                    ),
                )
                try:
                    controller_role = int(
                        system.getControllerRoleForTrackedDeviceIndex(index)
                    )
                except Exception:
                    controller_role = int(openvr.TrackedControllerRole_Invalid)
                try:
                    device_class = int(system.getTrackedDeviceClass(index))
                except Exception:
                    device_class = 0

                pose = poses[index] if index < len(poses) else None
                pose_valid = bool(getattr(pose, "bPoseIsValid", False))
                tracking_result = getattr(pose, "eTrackingResult", None)
                device = {
                    "openvr_index": index,
                    "serial": str(serial or ""),
                    "is_vmt": str(serial or "").upper().startswith("VMT_"),
                    "connected": connected,
                    "pose_valid": pose_valid,
                    "tracking_result": (
                        int(tracking_result) if tracking_result is not None else None
                    ),
                    "device_class": device_class,
                    "device_class_name": class_names.get(device_class, "unknown"),
                    "controller_type": str(controller_type or ""),
                    "controller_role": role_names.get(controller_role, "invalid"),
                    "role_hint": role_names.get(
                        int(role_hint) if role_hint is not None else 0,
                        "invalid",
                    ),
                    "role_hint_value": int(role_hint) if role_hint is not None else None,
                    "input_profile": str(input_profile or ""),
                    "hand_selection_priority": (
                        int(priority) if priority is not None else None
                    ),
                    "selected_as_left": index == left_index,
                    "selected_as_right": index == right_index,
                }
                devices.append(device)

            result["devices"] = devices
            result["vmt_devices"] = [item for item in devices if item["is_vmt"]]
            by_index = {item["openvr_index"]: item for item in devices}
            if left_index in by_index:
                result["left_serial"] = by_index[left_index]["serial"]
            if right_index in by_index:
                result["right_serial"] = by_index[right_index]["serial"]
            result["ok"] = True
        except Exception as exc:
            result["error"] = str(exc).strip() or type(exc).__name__
        finally:
            if initialized:
                try:
                    import openvr

                    openvr.shutdown()
                except Exception:
                    pass
    return result


def find_vmt_manager(extra_roots: Iterable[Path] = ()) -> Optional[Path]:
    roots = [Path(path) for path in extra_roots]
    # The official 0.15 installer defaults to C:\\vmt_driver rather than
    # Program Files. Include that location so the Dashboard can immediately
    # run the manager's `install` command after setup completes.
    system_drive = str(os.environ.get("SystemDrive", "C:")).rstrip("\\/")
    roots.append(Path(f"{system_drive}\\") / "vmt_driver")
    for env_name in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))

    candidates = []
    for root in roots:
        candidates.extend((
            root / "vmt_manager" / "vmt_manager.exe",
            root / "VirtualMotionTracker" / "vmt_manager" / "vmt_manager.exe",
            root / "VirtualMotionTracker" / "vmt_manager.exe",
            root / "Programs" / "VirtualMotionTracker" / "vmt_manager" / "vmt_manager.exe",
            root / "Programs" / "VirtualMotionTracker" / "vmt_manager.exe",
            root / "driver_files" / "vmt" / "vmt_manager" / "vmt_manager.exe",
        ))

    for driver_path in find_vmt_driver_paths():
        candidates.extend((
            driver_path.parent / "vmt_manager" / "vmt_manager.exe",
            driver_path / "vmt_manager" / "vmt_manager.exe",
        ))

    existing = _unique_existing_paths(candidates)
    return existing[0] if existing else None


def udp_port_is_listening(port: int) -> Optional[bool]:
    try:
        port = int(port)
    except (TypeError, ValueError):
        return False
    if not 1 <= port <= 65535:
        return False

    try:
        import psutil

        for connection in psutil.net_connections(kind="udp"):
            local = connection.laddr
            local_port = getattr(local, "port", None)
            if local_port is None and isinstance(local, tuple) and len(local) >= 2:
                local_port = local[1]
            if local_port == port:
                return True
        return False
    except (ImportError, OSError, PermissionError):
        pass

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        probe.close()


def steamvr_is_running() -> bool:
    try:
        import psutil

        process_names = {"vrserver.exe", "vrmonitor.exe", "vrcompositor.exe"}
        return any(
            str(process.info.get("name") or "").lower() in process_names
            for process in psutil.process_iter(["name"])
        )
    except (ImportError, OSError, PermissionError):
        return False


def vmt_installer_is_running() -> bool:
    """Return whether the official VMT setup is already awaiting completion."""
    try:
        import psutil

        for process in psutil.process_iter(["name"]):
            name = str(process.info.get("name") or "").strip().lower()
            if name.startswith("virtualmotiontrackersetup_"):
                return True
        return False
    except (ImportError, OSError, PermissionError):
        return False


def get_vmt_runtime_status(
    host: str = "127.0.0.1",
    port: int = 39570,
    project_root: Optional[Path] = None,
) -> Dict:
    local_mode = is_loopback_host(host)
    driver_paths = find_vmt_driver_paths(project_root)
    manager = find_vmt_manager(
        [Path(project_root)] if project_root else []
    )
    listening = udp_port_is_listening(port) if local_mode else None
    running = steamvr_is_running()
    installer_running = vmt_installer_is_running()
    settings = load_vmt_driver_settings(project_root)

    if not local_mode:
        message = "Remote VMT endpoint cannot be verified from this PC"
    elif not driver_paths:
        message = "VMT driver is not installed"
    elif not running:
        message = "SteamVR is not running"
    elif not listening:
        message = "VMT driver is installed but UDP port is not listening; restart SteamVR"
    else:
        message = "VMT driver is ready"

    return {
        "host": str(host),
        "port": int(port),
        "local_mode": local_mode,
        "driver_installed": bool(driver_paths),
        "driver_paths": [str(path) for path in driver_paths],
        "manager_path": str(manager) if manager else None,
        "steamvr_running": running,
        "installer_running": installer_running,
        "udp_port_listening": listening,
        "ready": bool(local_mode and driver_paths and running and listening),
        "message": message,
        **settings,
    }


def download_vmt_installer(destination: Path, timeout: float = 120.0) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        with open(destination, "rb") as handle:
            if handle.read(2) == b"MZ":
                return destination

    request = urllib.request.Request(
        VMT_INSTALLER_URL,
        headers={"User-Agent": "YOLO26n-VRC-Mocap/1.0"},
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix="vmt-setup-",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                shutil.copyfileobj(response, output)

        if temporary.stat().st_size <= 1_000_000:
            raise RuntimeError("Downloaded VMT installer is unexpectedly small")
        with open(temporary, "rb") as handle:
            if handle.read(2) != b"MZ":
                raise RuntimeError("Downloaded VMT installer is not a Windows executable")
        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def launch_elevated(executable: Path, arguments: str = "") -> bool:
    executable = Path(executable).resolve()
    if sys.platform == "win32":
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            str(executable),
            str(arguments or ""),
            str(executable.parent),
            1,
        )
        return int(result) > 32

    subprocess.Popen([str(executable), *str(arguments or "").split()])
    return True
