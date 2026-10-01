@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo YOLO26n VRC Mocap - Dashboard (lite)
echo ========================================
echo.
set "UV_PROJECT_ENVIRONMENT=.venv-lite"
set "YOLO26N_EDITION=lite"
uv run --frozen python bin\desktop_launcher.py --host 127.0.0.1 --port 8766 %*
pause