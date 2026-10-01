@echo off
cd /d "%~dp0"
echo ========================================
echo YOLO26n VRC Mocap - Starting (full)
echo ========================================
echo.

uv run --frozen --extra full python bin\main.py --config config/config.json %*

pause