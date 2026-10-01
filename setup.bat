@echo off
cd /d "%~dp0"
echo ========================================
echo YOLO26n VRC Mocap - Setup
echo ========================================
echo.
echo Usage: setup.bat [full^|lite]
echo   full (default): YOLO + PyTorch CUDA + MediaPipe  -^> .venv
echo   lite          : MediaPipe Pose/Face/Hand only    -^> .venv-lite
echo.

where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: uv not found. Install it with:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    pause
    exit /b 1
)

set "EDITION=%~1"
if "%EDITION%"=="" set "EDITION=full"
if /i "%EDITION%"=="full" goto SYNC_FULL
if /i "%EDITION%"=="lite" goto SYNC_LITE
echo Error: unknown edition "%EDITION%". Use full or lite.
pause
exit /b 1

:SYNC_FULL
echo Syncing the full Python 3.11 environment into .venv ...
uv sync --frozen --extra full
if %errorlevel% neq 0 goto SYNC_FAILED
set "START_HINT=start_dashboard.bat"
goto DONE

:SYNC_LITE
echo Syncing the lite Python 3.11 environment into .venv-lite ...
set "UV_PROJECT_ENVIRONMENT=.venv-lite"
uv sync --frozen
if %errorlevel% neq 0 goto SYNC_FAILED
set "START_HINT=start_dashboard_lite.bat"
goto DONE

:SYNC_FAILED
echo Error: uv sync failed.
pause
exit /b 1

:DONE
echo.
echo Setup complete! (%EDITION% edition)
echo.
echo To start the dashboard, run:
echo   %START_HINT%
echo.
pause