@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo YOLO26n VRC Mocap - Dashboard (full)
echo ========================================
echo.
uv run --frozen --extra full python bin\desktop_launcher.py --host 127.0.0.1 --port 8766 %*
pause