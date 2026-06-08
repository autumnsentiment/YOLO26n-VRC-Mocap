# -*- mode: python ; coding: utf-8 -*-
"""
YOLO26n VRC Mocap - PyInstaller 打包配置

用法:
    pip install pyinstaller
    pyinstaller yolo26n-vrc-mocap.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(SPECPATH))
BIN_DIR = os.path.join(PROJECT_ROOT, 'bin')

# ===== 收集数据文件 =====
datas = []

# MediaPipe 数据文件
try:
    datas += collect_data_files('mediapipe')
except Exception:
    pass

# Ultralytics 数据文件 (cfg等)
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
for model_name in ['yolo26n-pose.pt', 'yolo26n.pt', 'face_landmarker.task', 'hand_landmarker.task']:
    model_path = os.path.join(PROJECT_ROOT, model_name)
    if os.path.exists(model_path):
        datas.append((model_path, '.'))

# 前端静态文件
static_dir = os.path.join(BIN_DIR, 'static')
if os.path.exists(static_dir):
    datas.append((static_dir, 'bin/static'))

# 配置文件模板
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
    'ultralytics',
    'ultralytics.nn',
    'ultralytics.nn.tasks',
    'ultralytics.models',
    'ultralytics.models.yolo',
    'ultralytics.models.yolo.pose',
    'ultralytics.cfg',
    'flask',
    'pythonosc',
    'pythonosc.udp_client',
    'pythonosc.dispatcher',
    'pythonosc.osc_server',
    'psutil',
    'yaml',
    'matplotlib',
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
    [os.path.join(BIN_DIR, 'main.py')],
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
        'pytest', 'unittest',
    ],
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
    name='YOLO26n-VRC-Mocap',
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
    name='YOLO26n-VRC-Mocap',
)
