@echo off
echo ========================================
echo YOLO26n VRC Mocap - Setup
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo Installing required packages...
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To start the mocap system, run:
echo   python bin\main.py
echo.
echo For help:
echo   python bin\main.py --help
echo.
pause
