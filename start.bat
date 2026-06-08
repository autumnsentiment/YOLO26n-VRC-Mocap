@echo off
echo ========================================
echo YOLO26n VRC Mocap - Starting
echo ========================================
echo.

python bin\main.py --config config/config.json %*

pause
