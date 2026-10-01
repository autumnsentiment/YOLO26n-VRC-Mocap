# YOLO26n VRC Mocap Release Guide

## Editions

| Edition | Body backend | Python deps | Environment | Package |
|---|---|---|---|---|
| Full | YOLO Pose or MediaPipe Pose | MediaPipe + PyTorch CUDA + Ultralytics (~4.8 GB) | `.venv` | `dist\YOLO26n-VRC-Mocap\` |
| Lite | MediaPipe Pose only | MediaPipe, no PyTorch (~0.35 GB) | `.venv-lite` | `dist\YOLO26n-VRC-Mocap-Lite\` |

Face, hand, OSC/VMC/VMT, SteamVR and HMD features are identical in both. The
edition is detected from the installed packages; `YOLO26N_EDITION=lite`
forces the lite code path even in a full environment.

## Windows desktop package

Install uv first if it is not already available:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Build:

```bat
build.bat full
build.bat lite
```

Choose the PyInstaller option. The desktop package is written to:

```text
dist\YOLO26n-VRC-Mocap\        (full)
dist\YOLO26n-VRC-Mocap-Lite\   (lite)
```

Run:

```text
dist\YOLO26n-VRC-Mocap\YOLO26n-VRC-Mocap.exe
```

The backend executable starts the Web Dashboard and opens `http://127.0.0.1:8766`. Use the dashboard power button to start or stop the mocap process.

Optional Electron desktop UI:

```bat
start_electron_frontend.bat
```

The Electron UI starts the backend with `--no-browser`, embeds the Dashboard, and shows backend stdout/stderr in its built-in console instead of opening a separate cmd window.

Build the Electron UI:

```bat
build_electron_ui.bat
```

When the backend package already exists, the Electron executable is copied to:

```text
dist\YOLO26n-VRC-Mocap\YOLO26n-VRC-Mocap-Electron.exe
```

HMD remote sender files are included in the package:

```text
dist\YOLO26n-VRC-Mocap\HMD-Remote-Sender\
```

Copy that folder to the VR headset/SteamVR machine when using remote HMD pose feedback.

Pico/Quest headset-side SDK sender template files are also included:

```text
dist\YOLO26n-VRC-Mocap\HMD-SDK-Sender\
```

Use the Unity script in that folder when standalone Pico/Quest SDK pose data must
be sent back to the Dashboard over UDP.

## Windows source/web mode

Install uv, then sync the locked Python 3.11 environment:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bat
setup.bat full
start_dashboard.bat

setup.bat lite
start_dashboard_lite.bat
```

## Linux web deployment

Install uv and sync the locked Python 3.11 environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
chmod +x start_dashboard.sh

# full
uv sync --frozen --extra full
./start_dashboard.sh

# lite
UV_PROJECT_ENVIRONMENT=.venv-lite uv sync --frozen
YOLO26N_EDITION=lite ./start_dashboard.sh
```

Open:

```text
http://localhost:8765
```

For camera access, make sure the user can read `/dev/video*`.

## Linux Docker deployment

```bash
docker compose up --build                       # full
YOLO26N_EDITION=lite docker compose up --build  # lite
```

Open:

```text
http://localhost:8765
```

The compose file maps `/dev/video0`. Change the `devices` entry if your camera uses another device path.

## Platform notes

- Windows supports the bundled SteamVR driver management UI.
- Linux web mode supports the dashboard, YOLO (full only)/MediaPipe inference, OSC/VMC/VMT/UDP networking, model import, and runtime configuration where the host hardware and drivers are available.
- Model training lives in the sibling `YOLO26n-VRC-Mocap-Trainer` project and is not included in the runtime package.
- SteamVR driver installation endpoints are Windows-specific; on Linux they should be treated as unavailable.
