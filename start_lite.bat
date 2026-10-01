@echo off
cd /d "%~dp0"
echo ========================================
echo YOLO26n VRC Mocap - Starting (lite)
echo ========================================
echo.

set "UV_PROJECT_ENVIRONMENT=.venv-lite"
set "YOLO26N_EDITION=lite"
uv run --frozen python bin\main.py --config config/config.json %*

pause