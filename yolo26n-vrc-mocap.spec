# -*- mode: python ; coding: utf-8 -*-
"""
YOLO26n VRC Mocap - PyInstaller 打包配置

用法:
    全量版: uv sync --frozen --extra full --group build
            uv run --frozen pyinstaller yolo26n-vrc-mocap.spec
    轻量版: set YOLO26N_EDITION=lite
            uv sync --frozen --group build   (建议 UV_PROJECT_ENVIRONMENT=.venv-lite)
            uv run --frozen pyinstaller yolo26n-vrc-mocap.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 项目根目录
PROJECT_ROOT = os.path.abspath(SPECPATH)
BIN_DIR = os.path.join(PROJECT_ROOT, 'bin')

# full: YOLO + PyTorch + MediaPipe; lite: MediaPipe only (no torch/ultralytics)
EDITION = os.environ.get('YOLO26N_EDITION', 'full').strip().lower()
if EDITION not in ('full', 'lite'):
    raise SystemExit(f"Unknown YOLO26N_EDITION={EDITION!r}; use 'full' or 'lite'")
IS_LITE = EDITION == 'lite'
DIST_NAME = 'YOLO26n-VRC-Mocap-Lite' if IS_LITE else 'YOLO26n-VRC-Mocap'
print(f"[spec] Building {EDITION} edition -> dist/{DIST_NAME}")

# ===== 收集数据文件 =====
datas = []

# MediaPipe 数据文件
try:
    datas += collect_data_files('mediapipe')
except Exception:
    pass

# Ultralytics 数据文件 (cfg等)
if not IS_LITE:
    try:
        datas += collect_data_files('ultralytics')
    except Exception:
        pass

# scipy 数据文件
try:
    datas += collect_data_files('scipy')
except Exception:
    pass

# 模型文件
for metadata_name in ['requirements.txt', 'pyproject.toml', 'uv.lock', '.python-version'] + ([] if IS_LITE else ['requirements-full.txt']):
    metadata_path = os.path.join(PROJECT_ROOT, metadata_name)
    if os.path.exists(metadata_path):
        datas.append((metadata_path, '.'))

yolo_model_names = [] if IS_LITE else ['yolo26n-pose.pt', 'yolo26n.pt']
for model_name in yolo_model_names + [
    'face_landmarker.task',
    'hand_landmarker.task',
    'pose_landmarker_heavy.task',
    'pose_landmarker_full.task',
    'pose_landmarker_lite.task',
]:
    model_path = os.path.join(PROJECT_ROOT, model_name)
    if os.path.exists(model_path):
        datas.append((model_path, '.'))

models_dir = os.path.join(PROJECT_ROOT, 'models')
if os.path.exists(models_dir):
    for model_name in [
        'pose_landmarker_heavy.task',
        'pose_landmarker_full.task',
        'pose_landmarker_lite.task',
    ]:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            datas.append((model_path, 'models'))

# 前端静态文件
static_dir = os.path.join(BIN_DIR, 'static')
if os.path.exists(static_dir):
    datas.append((static_dir, 'bin/static'))

# HMD runtime/source files that users may need to inspect or run separately.
for hmd_file in ['hmd_provider.py', 'hmd_sender.py']:
    hmd_path = os.path.join(BIN_DIR, hmd_file)
    if os.path.exists(hmd_path):
        datas.append((hmd_path, 'bin'))

# Standalone remote HMD sender package, used on the VR headset/SteamVR machine.
hmd_remote_dir = os.path.join(os.path.dirname(PROJECT_ROOT), 'HMD-Remote-Sender')
if os.path.exists(hmd_remote_dir):
    for hmd_remote_file in [
        'hmd_sender.py',
        'requirements.txt',
        'install.bat',
        'install.sh',
        'start.bat',
        'start.sh',
    ]:
        hmd_remote_path = os.path.join(hmd_remote_dir, hmd_remote_file)
        if os.path.exists(hmd_remote_path):
            datas.append((hmd_remote_path, 'HMD-Remote-Sender'))

# 配置文件模板
hmd_sdk_dir = os.path.join(PROJECT_ROOT, 'HMD-SDK-Sender')
if os.path.exists(hmd_sdk_dir):
    datas.append((hmd_sdk_dir, 'HMD-SDK-Sender'))

config_dir = os.path.join(PROJECT_ROOT, 'config')
if os.path.exists(config_dir):
    datas.append((config_dir, 'config'))

# 驱动文件
driver_dir = os.path.join(PROJECT_ROOT, 'driver_files')
if os.path.exists(driver_dir):
    datas.append((driver_dir, 'driver_files'))

# ===== 收集隐式导入 =====
hiddenimports = [
    'cv2',
    'numpy',
    'scipy',
    'scipy.spatial',
    'scipy.spatial.transform',
    'mediapipe',
    'flask',
    'pythonosc',
    'pythonosc.udp_client',
    'pythonosc.dispatcher',
    'pythonosc.osc_server',
    'psutil',
    'yaml',
    'openvr',
    'dashboard_server',
    'edition',
    'face_calibration',
    'hmd_provider',
    'hmd_sender',
    'mediapipe_pose',
    'mediapipe_runtime',
    'main',
    'runtime_paths',
    'runtime_status',
    'model_presets',
    'modules',
    'streamers',
]

if not IS_LITE:
    hiddenimports += [
        'yolo_detector',
        'ultralytics',
        'ultralytics.nn',
        'ultralytics.nn.tasks',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'ultralytics.models.yolo.pose',
        'ultralytics.cfg',
    ]
    try:
        hiddenimports += collect_submodules('ultralytics')
    except Exception:
        pass

try:
    hiddenimports += collect_submodules('mediapipe')
except Exception:
    pass

# ===== 分析 =====
a = Analysis(
    [os.path.join(BIN_DIR, 'desktop_launcher.py')],
    pathex=[BIN_DIR, PROJECT_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'IPython', 'notebook', 'jupyter',
        'pytest',
    ] + (['torch', 'torchvision', 'ultralytics', 'yolo_detector'] if IS_LITE else []),
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=DIST_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=DIST_NAME,
)
