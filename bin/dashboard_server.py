#!/usr/bin/env python3
"""
YOLO26n VRC Mocap Web Dashboard

提供 Web 前端控制面板，支持:
- 查看和切换推理硬件 (GPU/CPU)
- 修改嘴部识别平滑参数
- 修改 YOLO 推理参数
- 启动/停止动捕程序
- 实时状态监控
"""

import os
import sys
import json
import shutil
import subprocess
import threading
import time
import glob as glob_mod
from pathlib import Path
from datetime import datetime

# 尝试导入 psutil 用于进程管理
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from flask import Flask, jsonify, request, send_from_directory
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
# 配置文件路径 - 优先使用项目目录，OneDrive 锁定时回退到临时目录
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
STATIC_DIR = Path(__file__).parent / "static"
DATASETS_DIR = PROJECT_ROOT / "datasets"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# 全局状态
mocap_process = None
mocap_status = {"running": False, "pid": None, "start_time": None}
train_process = None
train_status = {"running": False, "pid": None, "start_time": None, "name": None}

# HMD 位姿数据提供者 (独立于动捕进程，由 dashboard 直接管理)
_hmd_provider = None
_hmd_lock = threading.Lock()

# 配置文件读写锁，避免 Windows 下多线程读写冲突
_config_lock = threading.Lock()
_config_cache = None  # 内存缓存，避免频繁读文件导致 Windows 文件锁


def kill_stale_mocap_processes() -> int:
    """
    检测并终止所有残留的动捕进程（main.py）
    跳过当前管理的 mocap_process，避免误杀正在运行的进程。
    
    Returns:
        int: 终止的进程数量
    """
    killed_count = 0
    if not HAS_PSUTIL:
        return killed_count
    
    current_pid = os.getpid()
    managed_pid = mocap_process.pid if mocap_process and mocap_process.poll() is None else None
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.pid == current_pid:
                    continue
                if managed_pid is not None and proc.pid == managed_pid:
                    continue
                
                if 'python' not in proc.name().lower() and 'python.exe' not in proc.name().lower():
                    continue
                
                cmdline = proc.cmdline()
                if not any('main.py' in arg for arg in cmdline):
                    continue

                if any('dashboard' in arg for arg in cmdline):
                    continue

                print(f"发现残留动捕进程 PID: {proc.pid}，正在终止...")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        print(f"检测残留进程时出错: {e}")
    
    return killed_count


def load_config() -> dict:
    global _config_cache
    with _config_lock:
        if _config_cache is not None:
            return _config_cache
        # 优先从项目目录读取
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    _config_cache = json.load(f)
                    return _config_cache
            except (OSError, json.JSONDecodeError):
                pass
        # 回退到临时目录
        import tempfile
        tmp_path = Path(tempfile.gettempdir()) / "vrc-mocap" / "config.json"
        if tmp_path.exists():
            try:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    _config_cache = json.load(f)
                    return _config_cache
            except (OSError, json.JSONDecodeError):
                pass
        _config_cache = {}
        return _config_cache


def save_config(config: dict):
    global _config_cache
    with _config_lock:
        _config_cache = config
        content = json.dumps(config, indent=4, ensure_ascii=False)
        # 尝试写入项目目录（OneDrive 可能释放了锁）
        saved = False
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())  # 确保数据写入磁盘
            saved = True
        except OSError as e:
            print(f"[WARN] 写入配置文件失败: {e}")
        # 项目目录不可写时，回退到临时目录
        if not saved:
            import tempfile
            tmp_dir = Path(tempfile.gettempdir()) / "vrc-mocap"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / "config.json"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                print(f"[INFO] 配置已保存到 {tmp_path}（项目目录被 OneDrive 锁定）")
            except OSError as e:
                print(f"[WARN] 无法写入配置文件: {e}，配置仅保存在内存中")


def detect_hardware() -> dict:
    """检测可用硬件"""
    devices = [{"id": "cpu", "name": "CPU", "available": True}]

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory
                mem_gb = mem / (1024 ** 3)
                devices.append({
                    "id": f"cuda:{i}",
                    "name": f"GPU {i}: {name} ({mem_gb:.1f}GB)",
                    "available": True
                })
        else:
            devices.append({"id": "cuda", "name": "CUDA (未检测到GPU)", "available": False})
    except ImportError:
        devices.append({"id": "cuda", "name": "CUDA (PyTorch未安装)", "available": False})

    return {"devices": devices, "current": load_config().get("yolo", {}).get("device", "auto")}


# ============ API 路由 ============

@app.route("/")
def index():
    resp = send_from_directory(str(STATIC_DIR), "index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers.pop("ETag", None)
    resp.headers.pop("Last-Modified", None)
    return resp


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(load_config())


@app.route("/api/config", methods=["POST"])
def update_config():
    try:
        data = request.json
        config = load_config()

        # 记录关键开关变更
        old_body = config.get("body", {}).get("enabled", True)
        old_face = config.get("face", {}).get("enabled", True)
        old_hand = config.get("hand", {}).get("enabled", True)

        # 深度合并
        for section, values in data.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values

        new_body = config.get("body", {}).get("enabled", True)
        new_face = config.get("face", {}).get("enabled", True)
        new_hand = config.get("hand", {}).get("enabled", True)

        if old_body != new_body or old_face != new_face or old_hand != new_hand:
            print(f"[Config] 开关变更: body={old_body}->{new_body}, face={old_face}->{new_face}, hand={old_hand}->{new_hand}")

        save_config(config)
        # 即使文件写入失败，内存缓存已更新，仍返回成功
        return jsonify({"status": "ok", "config": load_config()})
    except Exception as e:
        print(f"[ERROR] update_config: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


def detect_cameras() -> list:
    """检测可用摄像头列表
    
    优化策略:
    1. Windows 下优先使用 DirectShow 后端，检测 USB 摄像头更可靠
    2. 使用超时线程防止卡死
    3. 尝试获取摄像头名称（DirectShow 支持）
    """
    import cv2
    import sys
    import threading

    cameras = []
    found_ids = set()

    # Windows 下使用 DirectShow (MSMF) 枚举摄像头，更可靠
    backends_to_try = []
    if sys.platform == 'win32':
        backends_to_try.append((cv2.CAP_DSHOW, "DirectShow"))
        backends_to_try.append((cv2.CAP_MSMF, "MSMF"))
    else:
        backends_to_try.append((cv2.CAP_V4L2, "V4L2"))
        backends_to_try.append((cv2.CAP_ANY, "Auto"))

    def try_open_camera(idx, backend_id, backend_name, result_list, timeout=2.0):
        """尝试打开摄像头，带超时"""
        cap = None
        try:
            cap = cv2.VideoCapture(idx, backend_id)
            if cap.isOpened():
                # 读取一帧确认摄像头真正可用
                ret, frame = cap.read()
                if ret and frame is not None:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    result_list.append({
                        "id": idx,
                        "name": f"摄像头 {idx}",
                        "width": w,
                        "height": h,
                        "fps": fps if fps and fps > 0 else 30,
                        "backend": backend_name,
                    })
        except Exception:
            pass
        finally:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass

    # 阶段 1: 使用 DirectShow 快速探测前 10 个索引
    for backend_id, backend_name in backends_to_try:
        for i in range(10):
            if i in found_ids:
                continue

            result_container = []
            thread = threading.Thread(
                target=try_open_camera,
                args=(i, backend_id, backend_name, result_container),
                daemon=True
            )
            thread.start()
            thread.join(timeout=3.0)  # 3秒超时

            if result_container:
                cam = result_container[0]
                cam["id"] = len(cameras)  # 重编号为连续 ID
                # 尝试用 DirectShow 获取设备名
                try:
                    cam["name"] = _get_camera_name_dshow(i) or cam["name"]
                except Exception:
                    pass
                # 补充分辨率信息
                if cam["width"] and cam["height"]:
                    cam["name"] = f"{cam['name']} ({cam['width']}x{cam['height']}"
                    if cam["fps"] and cam["fps"] > 0:
                        cam["name"] += f" @ {cam['fps']:.0f}fps"
                    cam["name"] += ")"
                cameras.append(cam)
                found_ids.add(i)
            elif thread.is_alive():
                # 超时，跳过此索引
                pass

    # 阶段 2: 如果 DirectShow 没找到，尝试 MSMF 和默认后端
    if not cameras:
        for i in range(10):
            result_container = []
            thread = threading.Thread(
                target=try_open_camera,
                args=(i, cv2.CAP_ANY, "Default", result_container),
                daemon=True
            )
            thread.start()
            thread.join(timeout=3.0)

            if result_container:
                cam = result_container[0]
                cam["id"] = len(cameras)
                cameras.append(cam)

    return cameras


def _get_camera_name_dshow(index: int) -> Optional[str]:
    """通过 DirectShow 获取摄像头设备名称 (Windows Only)"""
    import sys
    if sys.platform != 'win32':
        return None

    try:
        # 使用 Windows WMI 查询摄像头名称
        import subprocess
        result = subprocess.run(
            ['wmic', 'path', 'Win32_PnPEntity', 'where',
             'Service="usbvideo" OR Class="Camera" OR Class="Image"',
             'get', 'Name,DeviceID', '/format:list'],
            capture_output=True, text=True, timeout=5,
            creationflags=0x08000000  # CREATE_NO_WINDOW
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            cameras_found = []
            current_name = None
            for line in lines:
                line = line.strip()
                if line.startswith('Name='):
                    current_name = line[5:].strip()
                elif line.startswith('DeviceID=') and current_name:
                    cameras_found.append(current_name)
                    current_name = None
            if index < len(cameras_found):
                return cameras_found[index]
    except Exception:
        pass

    # 回退: 使用 PowerShell 查询
    try:
        import subprocess
        ps_cmd = (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.Service -eq 'usbvideo' -or $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | "
            "Select-Object -ExpandProperty Name"
        )
        result = subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_cmd],
            capture_output=True, text=True, timeout=5,
            creationflags=0x08000000  # CREATE_NO_WINDOW
        )
        if result.returncode == 0:
            names = [n.strip() for n in result.stdout.strip().split('\n') if n.strip()]
            if index < len(names):
                return names[index]
    except Exception:
        pass

    return None


@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    """获取可用摄像头列表"""
    cams = detect_cameras()
    current_id = load_config().get("camera", {}).get("camera_id", 0)
    return jsonify({"cameras": cams, "current": current_id})


@app.route("/api/hardware", methods=["GET"])
def get_hardware():
    return jsonify(detect_hardware())


@app.route("/api/steamvr/status", methods=["GET"])
def steamvr_status():
    """检查 SteamVR 驱动连接状态"""
    import sys
    pipe_name = r'\\.\pipe\ApriltagPipeIn'
    driver_path = PROJECT_ROOT / "driver_files"
    driver_dll = driver_path / "apriltagtrackers" / "bin" / "win64" / "driver_apriltagtrackers.dll"
    driver_installed = driver_dll.exists()

    # 检查是否已安装到 SteamVR 目录
    steamvr_driver_installed = False
    steam_path = None
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", 0, winreg.KEY_READ)
        steam_path, _ = winreg.QueryValueEx(key, "InstallPath")
        winreg.CloseKey(key)
        steamvr_driver_path = Path(steam_path) / "steamapps" / "common" / "SteamVR" / "drivers" / "apriltagtrackers" / "bin" / "win64" / "driver_apriltagtrackers.dll"
        steamvr_driver_installed = steamvr_driver_path.exists()
    except Exception:
        pass

    # 检查 activateMultipleDrivers 配置
    activate_multiple = False
    try:
        if steam_path:
            settings_path = Path(steam_path) / "config" / "steamvr.vrsettings"
            if settings_path.exists():
                import json
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                activate_multiple = settings.get("steamvr", {}).get("activateMultipleDrivers", False)
    except Exception:
        pass

    # 检查命名管道是否可用（SteamVR 是否运行）
    pipe_available = False
    num_trackers = 0
    driver_version = ""
    if sys.platform == 'win32':
        try:
            pipe = open(pipe_name, 'rb+', buffering=0)
            pipe.write(b'numtrackers\0')
            resp = pipe.read(1024)
            pipe.close()
            if resp:
                pipe_available = True
                parts = resp.decode("utf-8").strip().split(" ")
                # 格式: "numtrackers <count> <version>" 或 "numtrackers <count>"
                # 注意: 管道返回可能带前导空格，split后第一个元素可能是空字符串
                for part in parts:
                    part_clean = part.strip().rstrip('\x00')
                    if part_clean.isdigit():
                        num_trackers = int(part_clean)
                        break
                # 提取版本号
                for part in parts:
                    part_clean = part.strip().rstrip('\x00')
                    if part_clean and not part_clean.isdigit() and part_clean != 'numtrackers' and part_clean != 'OK':
                        driver_version = part_clean
                        break
        except (FileNotFoundError, OSError):
            pass

    return jsonify({
        "driver_files_available": driver_installed,
        "driver_installed_to_steamvr": steamvr_driver_installed,
        "activate_multiple_drivers": activate_multiple,
        "pipe_available": pipe_available,
        "steamvr_running": pipe_available,
        "num_trackers_registered": num_trackers,
        "driver_version": driver_version,
    })


@app.route("/api/steamvr/install", methods=["POST"])
def steamvr_install():
    """自动安装 SteamVR 驱动"""
    import shutil
    import subprocess

    driver_src = PROJECT_ROOT / "driver_files" / "apriltagtrackers"
    if not (driver_src / "bin" / "win64" / "driver_apriltagtrackers.dll").exists():
        return jsonify({"status": "error", "message": "驱动文件不存在，请先下载 ApriltagTrackers 驱动到 driver_files 目录"})

    # 查找 Steam 安装路径
    steam_path = None
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", 0, winreg.KEY_READ)
        steam_path, _ = winreg.QueryValueEx(key, "InstallPath")
        winreg.CloseKey(key)
    except Exception:
        pass

    if not steam_path:
        return jsonify({"status": "error", "message": "未找到 Steam 安装路径，请确认 Steam 已安装"})

    steamvr_drivers = Path(steam_path) / "steamapps" / "common" / "SteamVR" / "drivers"
    if not steamvr_drivers.exists():
        return jsonify({"status": "error", "message": f"SteamVR 驱动目录不存在: {steamvr_drivers}，请先安装 SteamVR"})

    # 复制驱动文件
    dest = steamvr_drivers / "apriltagtrackers"
    try:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(driver_src, dest)
    except PermissionError:
        return jsonify({"status": "error", "message": "权限不足，请以管理员身份运行或手动复制驱动文件"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"复制驱动文件失败: {e}"})

    # 修改 steamvr.vrsettings 添加 activateMultipleDrivers
    settings_path = Path(steam_path) / "config" / "steamvr.vrsettings"
    try:
        import json
        if settings_path.exists():
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        else:
            settings = {}

        if "steamvr" not in settings:
            settings["steamvr"] = {}

        settings["steamvr"]["activateMultipleDrivers"] = True

        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
    except Exception as e:
        return jsonify({"status": "warning", "message": f"驱动已复制，但修改配置文件失败: {e}，请手动在 steamvr.vrsettings 中添加 \"activateMultipleDrivers\": true"})

    return jsonify({"status": "ok", "message": "SteamVR 驱动安装成功！请重启 SteamVR 使驱动生效。"})


@app.route("/api/steamvr/uninstall", methods=["POST"])
def steamvr_uninstall():
    """卸载 SteamVR 驱动"""
    import shutil

    removed = []

    # 方式1: 从 SteamVR drivers 目录移除
    steam_path = None
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", 0, winreg.KEY_READ)
        steam_path, _ = winreg.QueryValueEx(key, "InstallPath")
        winreg.CloseKey(key)
    except Exception:
        pass

    if steam_path:
        dest = Path(steam_path) / "steamapps" / "common" / "SteamVR" / "drivers" / "apriltagtrackers"
        if dest.exists():
            try:
                shutil.rmtree(dest)
                removed.append("SteamVR drivers 目录")
            except PermissionError:
                return jsonify({"status": "error", "message": "权限不足，无法删除驱动文件，请以管理员身份运行"})
            except Exception as e:
                return jsonify({"status": "error", "message": f"删除驱动文件失败: {e}"})

    # 方式2: 从 openvrpaths.vrpath 移除
    vrpath_file = Path(os.environ.get("LOCALAPPDATA", "")) / "openvr" / "openvrpaths.vrpath"
    if vrpath_file.exists():
        try:
            with open(vrpath_file, "r", encoding="utf-8") as f:
                vrpath_data = json.load(f)
            if "external_drivers" in vrpath_data:
                before = len(vrpath_data["external_drivers"])
                vrpath_data["external_drivers"] = [
                    d for d in vrpath_data["external_drivers"]
                    if "apriltagtrackers" not in str(d).lower()
                ]
                after = len(vrpath_data["external_drivers"])
                if before != after:
                    with open(vrpath_file, "w", encoding="utf-8") as f:
                        json.dump(vrpath_data, f, indent=4)
                    removed.append("openvrpaths.vrpath")
        except Exception as e:
            return jsonify({"status": "warning", "message": f"部分卸载成功，但修改 openvrpaths.vrpath 失败: {e}"})

    if removed:
        return jsonify({"status": "ok", "message": f"驱动已从以下位置移除: {', '.join(removed)}，请重启 SteamVR 使更改生效。"})
    else:
        return jsonify({"status": "ok", "message": "未找到已安装的驱动，无需卸载。"})


@app.route("/api/steamvr/install_vrpath", methods=["POST"])
def steamvr_install_vrpath():
    """通过 openvrpaths.vrpath 注册驱动 (无需复制文件到 SteamVR 目录)"""
    driver_dir = PROJECT_ROOT / "driver_files" / "apriltagtrackers"
    if not (driver_dir / "driver.vrdrivermanifest").exists():
        return jsonify({"status": "error", "message": "驱动文件不存在: driver_files/apriltagtrackers"})

    vrpath_file = Path(os.environ.get("LOCALAPPDATA", "")) / "openvr" / "openvrpaths.vrpath"
    if not vrpath_file.exists():
        return jsonify({"status": "error", "message": "openvrpaths.vrpath 不存在，请先运行一次 SteamVR"})

    driver_path = str(driver_dir).replace("\\", "/")

    try:
        with open(vrpath_file, "r", encoding="utf-8") as f:
            vrpath_data = json.load(f)
    except Exception as e:
        return jsonify({"status": "error", "message": f"读取 openvrpaths.vrpath 失败: {e}"})

    # 检查是否已注册
    if "external_drivers" in vrpath_data:
        for d in vrpath_data["external_drivers"]:
            if "apriltagtrackers" in str(d).lower():
                return jsonify({"status": "ok", "message": "驱动已注册在 openvrpaths.vrpath 中，无需重复注册。"})

    # 添加注册
    if "external_drivers" not in vrpath_data:
        vrpath_data["external_drivers"] = []
    vrpath_data["external_drivers"].append(driver_path)

    try:
        with open(vrpath_file, "w", encoding="utf-8") as f:
            json.dump(vrpath_data, f, indent=4)
    except PermissionError:
        return jsonify({"status": "error", "message": "权限不足，无法写入 openvrpaths.vrpath，请以管理员身份运行"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"写入 openvrpaths.vrpath 失败: {e}"})

    return jsonify({"status": "ok", "message": "驱动已注册到 openvrpaths.vrpath！请重启 SteamVR 使驱动生效。"})


@app.route("/api/mocap/start", methods=["POST"])
def start_mocap():
    global mocap_process, mocap_status

    if mocap_status["running"] and mocap_process and mocap_process.poll() is None:
        return jsonify({"status": "error", "message": "Mocap already running"})

    # 先停掉之前的残留进程（通过 subprocess 对象）
    if mocap_process and mocap_process.poll() is None:
        mocap_process.terminate()
        try:
            mocap_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            mocap_process.kill()

    # 自动检测并终止所有残留的动捕进程（可能是之前崩溃或手动启动的）
    killed_count = kill_stale_mocap_processes()
    if killed_count > 0:
        print(f"已自动终止 {killed_count} 个残留动捕进程")

    args = request.json or {}
    cmd = [sys.executable, "-uB", str(PROJECT_ROOT / "bin" / "main.py")]

    # 传递配置文件路径
    config_file = PROJECT_ROOT / "config" / "config.json"
    if config_file.exists():
        cmd.extend(["--config", str(config_file)])

    if args.get("no_display"):
        cmd.append("--no-display")
    if args.get("no_face"):
        cmd.append("--no-face")
    if args.get("model"):
        cmd.extend(["--model", args["model"]])
    if args.get("device"):
        cmd.extend(["--device", args["device"]])
    if args.get("precision"):
        cmd.extend(["--precision", args["precision"]])
    if args.get("protocol"):
        cmd.extend(["--protocol", args["protocol"]])
    if args.get("camera") is not None:
        cmd.extend(["--camera", str(args["camera"])])

    env = os.environ.copy()
    env["ULTRALYTICS_SETTINGS"] = str(PROJECT_ROOT / ".ultralytics")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # 日志输出到临时目录
    import tempfile
    log_path = Path(tempfile.gettempdir()) / "yolo26n_mocap.log"
    log_file = open(str(log_path), 'w', encoding='utf-8')

    mocap_process = subprocess.Popen(
        cmd, env=env, cwd=str(PROJECT_ROOT),
        stdout=log_file, stderr=subprocess.STDOUT,
    )
    mocap_status = {
        "running": True,
        "pid": mocap_process.pid,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 等待 2 秒检查进程是否立即崩溃
    time.sleep(2)
    if mocap_process.poll() is not None:
        mocap_status = {"running": False, "pid": None, "start_time": None}
        return jsonify({
            "status": "error",
            "message": "进程启动后立即退出，可能是摄像头被占用或模型加载失败",
            "exit_code": mocap_process.returncode,
        })

    return jsonify({"status": "ok", "pid": mocap_process.pid})


@app.route("/api/mocap/stop", methods=["POST"])
def stop_mocap():
    global mocap_process, mocap_status

    stopped_any = False

    # 先终止 Dashboard 管理的进程
    if mocap_process and mocap_process.poll() is None:
        mocap_process.terminate()
        try:
            mocap_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            mocap_process.kill()
        stopped_any = True

    # 终止所有外部动捕进程
    external_procs = _find_mocap_processes()
    for proc_info in external_procs:
        if not proc_info["managed"]:
            try:
                import psutil
                proc = psutil.Process(proc_info["pid"])
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                stopped_any = True
                print(f"已终止外部动捕进程 PID: {proc_info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    mocap_status = {"running": False, "pid": None, "start_time": None}
    if stopped_any:
        return jsonify({"status": "ok"})
    else:
        return jsonify({"status": "ok", "message": "Not running"})


def _find_mocap_processes() -> list:
    """查找所有动捕进程 (main.py)，排除当前 Dashboard 进程

    Returns:
        list of dict: [{"pid": int, "cmdline": str, "managed": bool}, ...]
    """
    found = []
    if not HAS_PSUTIL:
        return found

    current_pid = os.getpid()
    managed_pid = mocap_process.pid if mocap_process and mocap_process.poll() is None else None

    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
            try:
                if proc.pid == current_pid:
                    continue
                if 'python' not in proc.name().lower() and 'python.exe' not in proc.name().lower():
                    continue
                cmdline = proc.cmdline()
                if not any('main.py' in arg for arg in cmdline):
                    continue

                # 排除 dashboard_server.py 自身 (防止误匹配)
                if any('dashboard' in arg for arg in cmdline):
                    continue

                is_managed = (proc.pid == managed_pid)
                found.append({"pid": proc.pid, "cmdline": " ".join(cmdline), "managed": is_managed})
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception:
        pass

    return found


@app.route("/api/mocap/status", methods=["GET"])
def mocap_status_api():
    managed_running = mocap_process and mocap_process.poll() is None

    all_mocap_procs = _find_mocap_processes()
    managed_procs = [p for p in all_mocap_procs if p["managed"]]
    external_procs = [p for p in all_mocap_procs if not p["managed"]]

    any_running = len(all_mocap_procs) > 0

    if managed_running:
        return jsonify({
            **mocap_status,
            "running": True,
            "managed": True,
            "external_procs": external_procs,
            "stale_pids": [p["pid"] for p in external_procs],
        })
    elif any_running:
        mocap_status["running"] = False
        first_external = external_procs[0] if external_procs else managed_procs[0]
        return jsonify({
            **mocap_status,
            "running": True,
            "managed": False,
            "external_pid": first_external["pid"],
            "external_procs": external_procs,
            "stale_pids": [],
        })
    else:
        mocap_status["running"] = False
        return jsonify({
            **mocap_status,
            "running": False,
            "managed": True,
            "external_procs": [],
            "stale_pids": [],
        })


@app.route("/api/mocap/kill_stale", methods=["POST"])
def kill_stale_api():
    """终止所有残留的动捕进程"""
    killed_count = kill_stale_mocap_processes()
    return jsonify({"status": "ok", "killed": killed_count})


@app.route("/api/mocap/log", methods=["GET"])
def mocap_log_api():
    """读取动捕进程的最新日志"""
    import tempfile
    log_path = Path(tempfile.gettempdir()) / "yolo26n_mocap.log"
    tail = int(request.args.get("tail", 50))
    if not log_path.exists():
        return jsonify({"log": "", "exists": False})
    try:
        with open(str(log_path), "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # 返回最后 N 行
        recent = lines[-tail:] if len(lines) > tail else lines
        return jsonify({"log": "".join(recent), "exists": True, "total_lines": len(lines)})
    except Exception as e:
        return jsonify({"log": f"读取日志失败: {e}", "exists": True})


@app.route("/api/mocap/inference_status", methods=["GET"])
def inference_status_api():
    """从日志中解析推理模块的当前状态"""
    import tempfile
    import re
    log_path = Path(tempfile.gettempdir()) / "yolo26n_mocap.log"
    if not log_path.exists():
        return jsonify({"body": None, "face": None, "hand": None, "available": False})
    try:
        with open(str(log_path), "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # 从 [Status] 行解析推理状态 (Body:ON/OFF Face:ON/OFF Hand:ON/OFF)
        body_status = None
        face_status = None
        hand_status = None

        # 查找最后一个 [Status] 行
        status_matches = re.findall(
            r'\[Status\].*Body:(\w+)\s+Face:(\w+)\s+Hand:(\w+)', content
        )
        if status_matches:
            last = status_matches[-1]
            body_status = last[0] == "ON"
            face_status = last[1] == "ON"
            hand_status = last[2] == "ON"

        # 也从 HotReload 行解析 (覆盖 Status 解析结果，因为 HotReload 更新)
        hotreload_matches = re.findall(
            r'\[HotReload\]\s+Body tracking:\s+(ON|OFF)', content
        )
        if hotreload_matches:
            body_status = hotreload_matches[-1] == "ON"

        hotreload_face = re.findall(
            r'\[HotReload\]\s+Face tracking:\s+(ON|OFF)', content
        )
        if hotreload_face:
            face_status = hotreload_face[-1] == "ON"

        hotreload_hand = re.findall(
            r'\[HotReload\]\s+Hand tracking:\s+(ON|OFF)', content
        )
        if hotreload_hand:
            hand_status = hotreload_hand[-1] == "ON"

        return jsonify({
            "body": body_status,
            "face": face_status,
            "hand": hand_status,
            "available": body_status is not None or face_status is not None or hand_status is not None,
        })
    except Exception as e:
        return jsonify({"body": None, "face": None, "hand": None, "available": False, "error": str(e)})


# ============ 数据集管理 API ============

@app.route("/api/browse", methods=["GET"])
def browse_directory():
    """浏览服务器文件系统目录，用于选择数据集路径"""
    req_path = request.args.get("path", "").strip()

    # 默认从用户目录和常见位置开始
    if not req_path:
        # 提供快捷入口
        home = Path.home()
        shortcuts = [
            {"name": "桌面", "path": str(home / "Desktop")},
            {"name": "文档", "path": str(home / "Documents")},
            {"name": "下载", "path": str(home / "Downloads")},
            {"name": "用户目录", "path": str(home)},
        ]
        # 添加可用盘符
        import string
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                shortcuts.append({"name": f"{letter}:\\", "path": str(drive)})

        return jsonify({"type": "shortcuts", "shortcuts": shortcuts, "current": ""})

    current = Path(req_path)
    if not current.exists() or not current.is_dir():
        return jsonify({"type": "error", "message": f"路径不存在或不是目录: {req_path}"}), 400

    try:
        entries = []
        for item in sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if not item.is_dir():
                continue
            # 跳过隐藏目录和系统目录
            if item.name.startswith('.') or item.name.startswith('$'):
                continue
            try:
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "has_dataset_yaml": (item / "dataset.yaml").exists(),
                    "has_images": (item / "images").exists(),
                    "has_labels": (item / "labels").exists(),
                })
            except PermissionError:
                continue

        parent = str(current.parent) if current.parent != current else ""
        return jsonify({
            "type": "listing",
            "current": str(current),
            "parent": parent,
            "entries": entries,
        })
    except PermissionError:
        return jsonify({"type": "error", "message": f"无权限访问: {req_path}"}), 403
    except Exception as e:
        return jsonify({"type": "error", "message": str(e)}), 500


@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    """列出所有已导入的数据集"""
    datasets = []
    try:
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return jsonify({"datasets": [], "error": "无法创建数据集目录，可能被 OneDrive 锁定"})

    try:
        dirs = sorted(DATASETS_DIR.iterdir())
    except PermissionError:
        return jsonify({"datasets": [], "error": "无法读取数据集目录"})

    for d in dirs:
        if not d.is_dir():
            continue
        yaml_path = d / "dataset.yaml"
        info = {
            "name": d.name,
            "path": str(d),
            "has_yaml": yaml_path.exists(),
            "valid": False,
            "train_images": 0,
            "val_images": 0,
            "train_labels": 0,
            "val_labels": 0,
        }

        # 统计文件数量
        train_img_dir = d / "images" / "train"
        val_img_dir = d / "images" / "val"
        train_lbl_dir = d / "labels" / "train"
        val_lbl_dir = d / "labels" / "val"

        if train_img_dir.exists():
            info["train_images"] = len(list(train_img_dir.glob("*.*")))
        if val_img_dir.exists():
            info["val_images"] = len(list(val_img_dir.glob("*.*")))
        if train_lbl_dir.exists():
            info["train_labels"] = len(list(train_lbl_dir.glob("*.txt")))
        if val_lbl_dir.exists():
            info["val_labels"] = len(list(val_lbl_dir.glob("*.txt")))

        info["valid"] = info["has_yaml"] and (info["train_images"] > 0 or info["train_labels"] > 0)
        datasets.append(info)

    return jsonify({"datasets": datasets})


@app.route("/api/datasets/import", methods=["POST"])
def import_dataset():
    """从指定路径导入数据集到项目目录

    支持两种方式:
    1. 复制: 将外部数据集复制到 datasets/ 目录下
    2. 链接: 在 datasets/ 下创建符号链接 (节省磁盘空间)
    """
    data = request.json or {}
    source_path = data.get("source_path", "").strip()
    dataset_name = data.get("name", "").strip()
    link_mode = data.get("link", False)

    if not source_path:
        return jsonify({"status": "error", "message": "请指定数据集源路径"}), 400

    source = Path(source_path)
    if not source.exists():
        return jsonify({"status": "error", "message": f"路径不存在: {source_path}"}), 400

    # 自动推断名称
    if not dataset_name:
        dataset_name = source.name

    target = DATASETS_DIR / dataset_name
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if target.exists():
        return jsonify({"status": "error", "message": f"数据集 '{dataset_name}' 已存在"}), 400

    try:
        if link_mode:
            # 创建符号链接
            target.symlink_to(source.resolve())
            msg = f"已创建符号链接: {target} -> {source}"
        else:
            # 复制数据集
            shutil.copytree(str(source), str(target))
            msg = f"已复制数据集到: {target}"

        # 检查是否有 dataset.yaml，没有则自动生成
        yaml_path = target / "dataset.yaml"
        if not yaml_path.exists():
            _auto_generate_dataset_yaml(target)

        return jsonify({"status": "ok", "message": msg, "name": dataset_name, "path": str(target)})
    except Exception as e:
        # 清理失败的导入
        if target.exists() or target.is_symlink():
            try:
                if target.is_symlink():
                    target.unlink()
                else:
                    shutil.rmtree(str(target))
            except Exception:
                pass
        return jsonify({"status": "error", "message": f"导入失败: {str(e)}"}), 500


@app.route("/api/datasets/create", methods=["POST"])
def create_dataset():
    """创建空数据集目录结构"""
    data = request.json or {}
    dataset_name = data.get("name", "").strip()

    if not dataset_name:
        return jsonify({"status": "error", "message": "请指定数据集名称"}), 400

    target = DATASETS_DIR / dataset_name
    try:
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return jsonify({"status": "error", "message": f"无法创建数据集目录 (权限不足): {DATASETS_DIR}，请检查目录权限或手动创建 datasets 文件夹"}), 500

    if target.exists():
        return jsonify({"status": "error", "message": f"数据集 '{dataset_name}' 已存在"}), 400

    try:
        # 创建目录结构
        for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
            (target / subdir).mkdir(parents=True, exist_ok=True)

        # 生成 dataset.yaml
        _auto_generate_dataset_yaml(target)

        return jsonify({"status": "ok", "message": f"数据集 '{dataset_name}' 已创建", "name": dataset_name, "path": str(target)})
    except PermissionError:
        return jsonify({"status": "error", "message": f"创建失败: 权限不足，请检查 {DATASETS_DIR} 目录权限"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"创建失败: {str(e)}"}), 500


@app.route("/api/datasets/<name>", methods=["DELETE"])
def delete_dataset(name):
    """删除数据集"""
    target = DATASETS_DIR / name
    if not target.exists():
        return jsonify({"status": "error", "message": "数据集不存在"}), 404

    try:
        if target.is_symlink():
            target.unlink()
        else:
            shutil.rmtree(str(target))
        return jsonify({"status": "ok", "message": f"数据集 '{name}' 已删除"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"删除失败: {str(e)}"}), 500


@app.route("/api/datasets/<name>/validate", methods=["GET"])
def validate_dataset(name):
    """验证数据集"""
    target = DATASETS_DIR / name
    if not target.exists():
        return jsonify({"status": "error", "message": "数据集不存在"}), 404

    yaml_path = target / "dataset.yaml"
    result = {
        "name": name,
        "path": str(target),
        "has_yaml": yaml_path.exists(),
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    # 检查 yaml
    if not yaml_path.exists():
        result["errors"].append("缺少 dataset.yaml 配置文件")
    else:
        try:
            import yaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            if 'path' not in cfg:
                result["errors"].append("dataset.yaml 缺少 path 字段")
            if 'names' not in cfg:
                result["warnings"].append("dataset.yaml 缺少 names 字段")
            if 'kpt_shape' not in cfg:
                result["warnings"].append("dataset.yaml 缺少 kpt_shape 字段 (姿态训练需要)")
        except Exception as e:
            result["errors"].append(f"dataset.yaml 解析失败: {e}")

    # 统计
    for split in ["train", "val"]:
        img_dir = target / "images" / split
        lbl_dir = target / "labels" / split
        img_count = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        lbl_count = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        result["stats"][split] = {"images": img_count, "labels": lbl_count}

        if img_count == 0:
            result["warnings"].append(f"{split} 集没有图像")
        if lbl_count == 0 and img_count > 0:
            result["warnings"].append(f"{split} 集有图像但没有标签")

    result["valid"] = len(result["errors"]) == 0
    return jsonify(result)


# ============ 训练 API ============

@app.route("/api/train/start", methods=["POST"])
def start_training():
    """启动训练"""
    global train_process, train_status

    if train_status["running"] and train_process and train_process.poll() is None:
        return jsonify({"status": "error", "message": "训练已在运行中"})

    # 先停掉之前的残留进程
    if train_process and train_process.poll() is None:
        train_process.terminate()
        try:
            train_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            train_process.kill()

    data = request.json or {}
    dataset_name = data.get("dataset", "").strip()
    if not dataset_name:
        return jsonify({"status": "error", "message": "请选择数据集"}), 400

    yaml_path = DATASETS_DIR / dataset_name / "dataset.yaml"
    if not yaml_path.exists():
        return jsonify({"status": "error", "message": f"数据集配置文件不存在: {dataset_name}/dataset.yaml"}), 400

    # 构建训练命令
    cmd = [
        sys.executable, "-B",
        str(PROJECT_ROOT / "bin" / "train.py"),
        "--data", str(yaml_path),
        "--epochs", str(data.get("epochs", 100)),
        "--batch", str(data.get("batch", 16)),
        "--imgsz", str(data.get("imgsz", 640)),
        "--device", data.get("device", "auto"),
        "--project", str(PROJECT_ROOT / "runs" / "train"),
    ]

    if data.get("freeze", 0) > 0:
        cmd.extend(["--freeze", str(data["freeze"])])
    if data.get("model"):
        cmd.extend(["--model", data["model"]])
    if data.get("name"):
        cmd.extend(["--name", data["name"]])
    if data.get("hand_split", False):
        cmd.append("--hand-split")
    if data.get("model_type"):
        cmd.extend(["--model-type", data["model_type"]])
    if data.get("lr0"):
        cmd.extend(["--lr0", str(data["lr0"])])
    if data.get("optimizer"):
        cmd.extend(["--optimizer", data["optimizer"]])
    if data.get("patience") is not None:
        cmd.extend(["--patience", str(data["patience"])])
    if data.get("no_augment", False):
        cmd.append("--no-augment")
    if data.get("amp", True) is False:
        cmd.append("--no-amp")
    if data.get("cos_lr", True) is False:
        cmd.append("--no-cos-lr")
    if data.get("label_smoothing"):
        cmd.extend(["--label-smoothing", str(data["label_smoothing"])])

    env = os.environ.copy()
    env["ULTRALYTICS_SETTINGS"] = str(PROJECT_ROOT / ".ultralytics")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # 训练日志输出到文件
    log_dir = PROJECT_ROOT / "runs" / "train" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"

    train_process = subprocess.Popen(
        cmd, env=env, cwd=str(PROJECT_ROOT),
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    train_status = {
        "running": True,
        "pid": train_process.pid,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": dataset_name,
        "log": str(log_path),
    }

    # 等待 3 秒检查进程是否立即崩溃
    time.sleep(3)
    if train_process.poll() is not None:
        train_status = {"running": False, "pid": None, "start_time": None, "name": None}
        # 读取日志查看错误原因
        try:
            error_msg = log_path.read_text(encoding="utf-8")[-500:]
        except OSError:
            error_msg = "未知错误"
        return jsonify({
            "status": "error",
            "message": "训练进程启动后立即退出",
            "detail": error_msg,
        })

    return jsonify({"status": "ok", "pid": train_process.pid, "log": str(log_path)})


@app.route("/api/train/stop", methods=["POST"])
def stop_training():
    """停止训练"""
    global train_process, train_status

    if train_process and train_process.poll() is None:
        train_process.terminate()
        try:
            train_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            train_process.kill()
        train_status = {"running": False, "pid": None, "start_time": None, "name": None}
        return jsonify({"status": "ok"})
    else:
        train_status = {"running": False, "pid": None, "start_time": None, "name": None}
        return jsonify({"status": "ok", "message": "训练未在运行"})


@app.route("/api/train/status", methods=["GET"])
def train_status_api():
    """查询训练状态"""
    if train_process and train_process.poll() is None:
        return jsonify({**train_status, "running": True})
    else:
        if train_status["running"]:
            train_status["running"] = False
        return jsonify({**train_status, "running": False})


@app.route("/api/train/log", methods=["GET"])
def train_log():
    """查看训练日志"""
    log_path = train_status.get("log") or request.args.get("path")
    if not log_path:
        return jsonify({"log": "", "message": "无日志文件"})

    try:
        path = Path(log_path)
        if not path.exists():
            return jsonify({"log": "", "message": "日志文件不存在"})
        # 读取最后 200 行
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = lines[-200:] if len(lines) > 200 else lines
        return jsonify({"log": "".join(tail), "total_lines": len(lines)})
    except OSError as e:
        return jsonify({"log": "", "message": f"读取日志失败: {e}"})


@app.route("/api/train/models", methods=["GET"])
def list_trained_models():
    """列出已训练的模型"""
    models = []
    runs_dir = PROJECT_ROOT / "runs" / "train"
    if not runs_dir.exists():
        # 也检查 runs/pose/runs/train
        runs_dir = PROJECT_ROOT / "runs" / "pose" / "runs" / "train"

    if runs_dir.exists():
        for d in sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
            if not d.is_dir():
                continue
            best = d / "weights" / "best.pt"
            last = d / "weights" / "last.pt"
            if best.exists() or last.exists():
                mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                models.append({
                    "name": d.name,
                    "path": str(d),
                    "best": str(best) if best.exists() else None,
                    "last": str(last) if last.exists() else None,
                    "modified": mtime,
                })

    return jsonify({"models": models})


@app.route("/api/train/presets", methods=["GET"])
def get_model_presets():
    """获取模型精度等级预设列表"""
    from train import ModelTrainer
    presets = {}
    for key, preset in ModelTrainer.MODEL_PRESETS.items():
        presets[key] = {
            "name": preset["name"],
            "model": preset["model"],
            "description": preset["description"],
            "default_imgsz": preset["default_imgsz"],
            "default_batch": preset["default_batch"],
            "default_lr0": preset["default_lr0"],
            "params_m": preset["params_m"],
            "speed_ms": preset["speed_ms"],
        }
    return jsonify({"presets": presets})


@app.route("/api/train/analysis", methods=["GET"])
def get_train_analysis():
    """获取训练模型的分布分析图"""
    model_name = request.args.get("model", "").strip()
    if not model_name:
        return jsonify({"error": "请指定模型名称"}), 400

    # 查找分析图目录
    runs_dir = PROJECT_ROOT / "runs" / "train"
    analysis_dir = runs_dir / model_name / "analysis"

    if not analysis_dir.exists():
        # 也检查 runs/pose/runs/train
        alt_dir = PROJECT_ROOT / "runs" / "pose" / "runs" / "train" / model_name / "analysis"
        if alt_dir.exists():
            analysis_dir = alt_dir
        else:
            return jsonify({"error": "分析图不存在，请先完成训练", "images": []})

    images = []
    for img_file in sorted(analysis_dir.glob("*.png")):
        images.append({
            "name": img_file.stem,
            "path": f"/api/train/analysis/image?model={model_name}&image={img_file.name}",
        })

    return jsonify({"images": images})


@app.route("/api/train/analysis/image", methods=["GET"])
def get_analysis_image():
    """获取分析图图片"""
    model_name = request.args.get("model", "").strip()
    image_name = request.args.get("image", "").strip()

    if not model_name or not image_name:
        return "Missing parameters", 400

    # 防止路径遍历攻击
    if ".." in model_name or ".." in image_name or "/" in image_name or "\\" in image_name:
        return "Invalid parameters", 400

    runs_dir = PROJECT_ROOT / "runs" / "train"
    img_path = runs_dir / model_name / "analysis" / image_name

    if not img_path.exists():
        alt_path = PROJECT_ROOT / "runs" / "pose" / "runs" / "train" / model_name / "analysis" / image_name
        if alt_path.exists():
            img_path = alt_path
        else:
            return "Image not found", 404

    return send_from_directory(str(img_path.parent), img_path.name)


# ============ 辅助函数 ============

def _auto_generate_dataset_yaml(dataset_dir: Path):
    """自动生成 dataset.yaml"""
    import yaml
    config = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "person"},
        "kpt_shape": [17, 3],
    }
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def _auto_start_hmd_provider():
    """根据配置文件自动启动 HMD provider

    Dashboard 启动时，如果配置中 hmd.provider 不是 static，
    自动启动对应的 HMD 数据提供者，避免每次重启后需要手动启用。
    """
    global _hmd_provider
    config = load_config()
    hmd_cfg = config.get("hmd", {})
    provider_type = hmd_cfg.get("provider", "static")

    if provider_type in ("static", "none"):
        return

    try:
        import hmd_provider as _hmd_mod
        HMDProviderFactory = _hmd_mod.HMDProviderFactory
        kwargs = {
            "osc_listen_port": hmd_cfg.get("osc_listen_port", 9001),
            "pico_host": hmd_cfg.get("pico_host", "127.0.0.1"),
            "pico_port": hmd_cfg.get("pico_port", 7890),
            "pico_xrobo_host": hmd_cfg.get("pico_xrobo_host", "0.0.0.0"),
            "pico_xrobo_port": hmd_cfg.get("pico_xrobo_port", 39540),
            "quest_host": hmd_cfg.get("quest_host", "127.0.0.1"),
            "quest_port": hmd_cfg.get("quest_port", 53342),
            "quest_udp_host": hmd_cfg.get("quest_udp_host", "0.0.0.0"),
            "quest_udp_port": hmd_cfg.get("quest_udp_port", 39541),
            "remote_port": hmd_cfg.get("remote_port", 4243),
        }
        provider = HMDProviderFactory.create(provider_type, **kwargs)
        if provider.start():
            _hmd_provider = provider
            print(f"[AutoStart] HMD Provider: {provider.name} (from config)")
        else:
            print(f"[AutoStart] HMD Provider: {provider_type} start failed")
    except Exception as e:
        print(f"[AutoStart] HMD Provider: {provider_type} init error: {e}")


def run_server(host="0.0.0.0", port=8765):
    print(f"Dashboard running at http://{host}:{port}")
    # 自动恢复 HMD provider (根据配置文件)
    _auto_start_hmd_provider()
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


# ============ HMD 位姿 API ============

@app.route("/api/hmd/status", methods=["GET"])
def hmd_status():
    """获取 HMD 提供者状态"""
    with _hmd_lock:
        if _hmd_provider is None:
            return jsonify({
                "running": False,
                "provider": "none",
                "pose": None,
            })
        pose = _hmd_provider.get_pose()
        return jsonify({
            "running": True,
            "provider": _hmd_provider.name,
            "pose": {
                "position": pose.position,
                "rotation": pose.rotation,
                "quaternion": pose.quaternion,
                "valid": pose.valid,
                "timestamp": pose.timestamp,
            },
        })


@app.route("/api/hmd/start", methods=["POST"])
def hmd_start():
    """启动 HMD 数据提供者"""
    global _hmd_provider
    with _hmd_lock:
        # 先停止已有的
        if _hmd_provider is not None:
            _hmd_provider.stop()
            _hmd_provider = None

        data = request.json or {}
        provider_type = data.get("provider", "static")
        if provider_type == "none" or provider_type == "static":
            return jsonify({"status": "ok", "provider": "static", "message": "HMD 提供者已关闭 (static 模式)"})

        try:
            import importlib
            import hmd_provider as _hmd_mod

            # 条件 reload: 仅当 hmd_provider.py 文件修改时间变化时才重新加载
            # 避免每次 API 调用都 reload 带来的性能开销
            _hmd_mod_path = getattr(_hmd_mod, "__file__", None)
            if _hmd_mod_path:
                from pathlib import Path
                _hmd_mod_mtime = Path(_hmd_mod_path).stat().st_mtime
                if not hasattr(hmd_start, "_last_mtime") or hmd_start._last_mtime < _hmd_mod_mtime:
                    importlib.reload(_hmd_mod)
                    hmd_start._last_mtime = _hmd_mod_mtime

            HMDProviderFactory = _hmd_mod.HMDProviderFactory
            kwargs = {
                "osc_listen_port": data.get("osc_listen_port", 9001),
                "pico_host": data.get("pico_host", "127.0.0.1"),
                "pico_port": data.get("pico_port", 7890),
                "pico_xrobo_host": data.get("pico_xrobo_host", "0.0.0.0"),
                "pico_xrobo_port": data.get("pico_xrobo_port", 39540),
                "quest_host": data.get("quest_host", "127.0.0.1"),
                "quest_port": data.get("quest_port", 53342),
                "quest_udp_host": data.get("quest_udp_host", "0.0.0.0"),
                "quest_udp_port": data.get("quest_udp_port", 39541),
                "remote_port": data.get("remote_port", 4243),
            }
            provider = HMDProviderFactory.create(provider_type, **kwargs)
            # 检测工厂是否回退到 StaticProvider (说明请求的 provider 未注册)
            if provider.name == "static" and provider_type not in ("static", "none"):
                return jsonify({"status": "error", "provider": provider_type,
                                "message": f"HMD 提供者 '{provider_type}' 未注册或加载失败，已回退到 static。请重启 Dashboard 以加载最新模块。"})
            if provider.start():
                _hmd_provider = provider
                # 同步配置文件: 保存当前 provider 类型，重启后自动恢复
                try:
                    config = load_config()
                    if "hmd" not in config:
                        config["hmd"] = {}
                    config["hmd"]["provider"] = provider.name
                    config["hmd"]["remote_port"] = data.get("remote_port", 4243)
                    save_config(config)
                except Exception:
                    pass
                return jsonify({"status": "ok", "provider": provider.name, "message": f"HMD 提供者已启动: {provider.name}"})
            else:
                return jsonify({"status": "error", "provider": provider_type, "message": f"HMD 提供者启动失败: {provider_type}，请检查依赖和连接"})
        except ImportError as e:
            return jsonify({"status": "error", "provider": provider_type, "message": f"缺少依赖: {e}"})
        except Exception as e:
            return jsonify({"status": "error", "provider": provider_type, "message": f"启动失败: {e}"})


@app.route("/api/hmd/stop", methods=["POST"])
def hmd_stop():
    """停止 HMD 数据提供者"""
    global _hmd_provider
    with _hmd_lock:
        if _hmd_provider is not None:
            _hmd_provider.stop()
            name = _hmd_provider.name
            _hmd_provider = None
            # 同步配置文件: 将 provider 设为 static，避免重启后自动启动
            try:
                config = load_config()
                if "hmd" not in config:
                    config["hmd"] = {}
                config["hmd"]["provider"] = "static"
                save_config(config)
            except Exception:
                pass
            return jsonify({"status": "ok", "message": f"HMD 提供者已停止: {name}"})
        return jsonify({"status": "ok", "message": "HMD 提供者未运行"})


# ============ 识别模块独立开关 API ============

@app.route("/api/tracking/toggle", methods=["POST"])
def tracking_toggle():
    """独立控制识别模块开关

    通过修改配置文件实现，动捕进程会自动热更新。
    支持的模块: body, face, hand

    请求体: {"module": "body|face|hand", "enabled": true|false}
    """
    data = request.json or {}
    module = data.get("module", "")
    enabled = data.get("enabled", True)

    if module not in ("body", "face", "hand"):
        return jsonify({"status": "error", "message": f"未知模块: {module}，支持: body, face, hand"}), 400

    config = load_config()
    if module not in config:
        config[module] = {}
    config[module]["enabled"] = enabled
    save_config(config)

    module_names = {"body": "身体姿态", "face": "面部表情", "hand": "手势追踪"}
    return jsonify({
        "status": "ok",
        "message": f"{module_names.get(module, module)} 已{'启用' if enabled else '关闭'}",
        "module": module,
        "enabled": enabled,
    })


@app.route("/api/tracking/status", methods=["GET"])
def tracking_status():
    """获取各识别模块的开关状态"""
    config = load_config()
    return jsonify({
        "body": config.get("body", {}).get("enabled", True),
        "face": config.get("face", {}).get("enabled", True),
        "hand": config.get("hand", {}).get("enabled", True),
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO26n VRC Mocap Dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(args.host, args.port)
