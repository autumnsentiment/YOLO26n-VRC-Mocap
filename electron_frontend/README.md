# YOLO26n VRC Mocap Electron UI

Electron desktop wrapper for the existing Flask Dashboard.

## Behavior

- Starts the backend automatically unless `--no-backend` is passed.
- Uses `YOLO26n-VRC-Mocap.exe --no-browser --host 127.0.0.1 --port 8766` when a packaged backend is available.
- Falls back to `python bin/desktop_launcher.py --no-browser` in source mode.
- Hides the backend cmd window and streams stdout/stderr into the built-in console panel.
- If a backend is already running on the configured port, it connects to it instead of starting another process.

## Start

```bat
..\start_electron_frontend.bat
```

## Build

```bat
..\build_electron_ui.bat
```

## Arguments

```text
--host 127.0.0.1
--port 8766
--backend <path-to-YOLO26n-VRC-Mocap.exe>
--root <project-or-dist-root>
--no-backend
--keep-backend-on-exit
```
