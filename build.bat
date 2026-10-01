@echo off
chcp 65001 >nul
echo ========================================
echo   YOLO26n VRC Mocap - 打包工具
echo ========================================
echo.

cd /d "%~dp0"

REM ===== 版本选择: full = YOLO+PyTorch+MediaPipe, lite = 仅 MediaPipe =====
set "EDITION=%~1"
if not "%EDITION%"=="" goto EDITION_CHOSEN
echo 请选择版本:
echo   [1] 全量版 (YOLO + PyTorch CUDA + MediaPipe)
echo   [2] 轻量版 (仅 MediaPipe Pose/Face/Hand)
set /p EDITION_CHOICE=请输入选项 [1]:
set "EDITION=full"
if "%EDITION_CHOICE%"=="2" set "EDITION=lite"

:EDITION_CHOSEN
if /i "%EDITION%"=="lite" (
    set "EDITION=lite"
    set "DIST_NAME=YOLO26n-VRC-Mocap-Lite"
    set "UV_PROJECT_ENVIRONMENT=.venv-lite"
    set "UV_EXTRA="
    set "YOLO26N_EDITION=lite"
) else (
    set "EDITION=full"
    set "DIST_NAME=YOLO26n-VRC-Mocap"
    set "UV_PROJECT_ENVIRONMENT="
    set "UV_EXTRA=--extra full"
    set "YOLO26N_EDITION=full"
)
set "DIST_DIR=dist\%DIST_NAME%"
echo 当前版本: %EDITION% -^> %DIST_DIR%
echo.

:MENU
echo 请选择打包方式:
echo   [1] PyInstaller EXE (单目录分发)
echo   [2] ZIP 分发包 (源码+依赖)
echo   [3] 两者都生成
echo   [0] 退出
echo.
set /p CHOICE=请输入选项:

if "%CHOICE%"=="1" goto BUILD_EXE
if "%CHOICE%"=="2" goto BUILD_ZIP
if "%CHOICE%"=="3" goto BUILD_ALL
if "%CHOICE%"=="0" exit /b 0
echo 无效选项
goto MENU

:BUILD_EXE
echo.
echo ========================================
echo   [1] PyInstaller 打包
echo ========================================
echo.

REM 安装锁定的运行和打包依赖
uv sync --frozen %UV_EXTRA% --group build
if %errorlevel% neq 0 (
    echo [错误] uv sync 失败！
    pause
    exit /b 1
)

REM 检查模型文件
echo [检查] 模型文件...
set HAS_MODEL=0
if /i "%EDITION%"=="lite" goto CHECK_TASK_MODELS
if exist "yolo26n-pose.pt" (
    echo   ✓ yolo26n-pose.pt
    set HAS_MODEL=1
) else (
    echo   ✗ yolo26n-pose.pt (缺失，打包后需手动放入)
)
:CHECK_TASK_MODELS
if exist "face_landmarker.task" (
    echo   ✓ face_landmarker.task
) else (
    echo   ✗ face_landmarker.task (缺失)
)
if exist "hand_landmarker.task" (
    echo   ✓ hand_landmarker.task
) else (
    echo   ✗ hand_landmarker.task (缺失)
)

echo.
echo [构建] 正在打包 (首次可能需要较长时间)...
uv run --frozen %UV_EXTRA% pyinstaller yolo26n-vrc-mocap.spec --noconfirm --clean

if %errorlevel% neq 0 (
    echo [错误] 打包失败！
    pause
    exit /b 1
)

echo.
echo [成功] EXE 打包完成！
echo   输出目录: %DIST_DIR%\
echo   主程序: %DIST_DIR%\%DIST_NAME%.exe
echo.

REM 复制缺失的模型文件到 dist
if /i not "%EDITION%"=="lite" if not exist "%DIST_DIR%\yolo26n-pose.pt" (
    if exist "yolo26n-pose.pt" (
        copy /Y "yolo26n-pose.pt" "%DIST_DIR%\" >nul
        echo   已复制 yolo26n-pose.pt
    )
)
if not exist "%DIST_DIR%\face_landmarker.task" (
    if exist "face_landmarker.task" (
        copy /Y "face_landmarker.task" "%DIST_DIR%\" >nul
        echo   已复制 face_landmarker.task
    )
)
if not exist "%DIST_DIR%\hand_landmarker.task" (
    if exist "hand_landmarker.task" (
        copy /Y "hand_landmarker.task" "%DIST_DIR%\" >nul
        echo   已复制 hand_landmarker.task
    )
)
if exist "models\pose_landmarker_heavy.task" (
    if not exist "%DIST_DIR%\models" mkdir "%DIST_DIR%\models"
    copy /Y "models\pose_landmarker_heavy.task" "%DIST_DIR%\models\" >nul
    echo   Copied models\pose_landmarker_heavy.task
)
if exist "pose_landmarker_heavy.task" (
    copy /Y "pose_landmarker_heavy.task" "%DIST_DIR%\" >nul
    echo   Copied pose_landmarker_heavy.task
)

REM HMD remote sender package (copy to the VR headset/SteamVR machine when needed)
if exist "..\HMD-Remote-Sender" (
    if exist "%DIST_DIR%\HMD-Remote-Sender" rmdir /s /q "%DIST_DIR%\HMD-Remote-Sender"
    mkdir "%DIST_DIR%\HMD-Remote-Sender"
    if exist "..\HMD-Remote-Sender\hmd_sender.py" copy /Y "..\HMD-Remote-Sender\hmd_sender.py" "%DIST_DIR%\HMD-Remote-Sender\" >nul
    if exist "..\HMD-Remote-Sender\requirements.txt" copy /Y "..\HMD-Remote-Sender\requirements.txt" "%DIST_DIR%\HMD-Remote-Sender\" >nul
    if exist "..\HMD-Remote-Sender\install.bat" copy /Y "..\HMD-Remote-Sender\install.bat" "%DIST_DIR%\HMD-Remote-Sender\" >nul
    if exist "..\HMD-Remote-Sender\install.sh" copy /Y "..\HMD-Remote-Sender\install.sh" "%DIST_DIR%\HMD-Remote-Sender\" >nul
    if exist "..\HMD-Remote-Sender\start.bat" copy /Y "..\HMD-Remote-Sender\start.bat" "%DIST_DIR%\HMD-Remote-Sender\" >nul
    if exist "..\HMD-Remote-Sender\start.sh" copy /Y "..\HMD-Remote-Sender\start.sh" "%DIST_DIR%\HMD-Remote-Sender\" >nul
    echo   Copied HMD-Remote-Sender
)

REM HMD SDK sender template for Pico/Quest Unity projects
if exist "HMD-SDK-Sender" (
    if exist "%DIST_DIR%\HMD-SDK-Sender" rmdir /s /q "%DIST_DIR%\HMD-SDK-Sender"
    xcopy /E /I /Y "HMD-SDK-Sender" "%DIST_DIR%\HMD-SDK-Sender" >nul
    echo   Copied HMD-SDK-Sender
)

pause
goto MENU

:BUILD_ZIP
echo.
echo ========================================
echo   [2] ZIP 分发包
echo ========================================
echo.

set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%
set TIMESTAMP=%TIMESTAMP: =0%
set ZIP_NAME=%DIST_NAME%_%TIMESTAMP%

echo [创建] 临时打包目录: %ZIP_NAME%

REM 创建临时目录结构
if exist "%ZIP_NAME%" rmdir /s /q "%ZIP_NAME%"
mkdir "%ZIP_NAME%"
mkdir "%ZIP_NAME%\bin"
mkdir "%ZIP_NAME%\bin\modules"
mkdir "%ZIP_NAME%\bin\streamers"
mkdir "%ZIP_NAME%\bin\static"
mkdir "%ZIP_NAME%\config"
mkdir "%ZIP_NAME%\driver_files"
mkdir "%ZIP_NAME%\models"

REM 复制核心代码
echo [复制] 核心代码...
copy /Y "bin\main.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\__init__.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\camera_capture.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\face_landmark.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\hand_tracker.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\hmd_provider.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\hmd_sender.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\pose_estimator.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\yolo_detector.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\dashboard_server.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\desktop_launcher.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\face_calibration.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\mediapipe_runtime.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\runtime_paths.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\runtime_status.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\model_presets.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\mediapipe_pose.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\edition.py" "%ZIP_NAME%\bin\" >nul
if /i "%EDITION%"=="lite" del /Q "%ZIP_NAME%\bin\yolo_detector.py" 2>nul

REM 模块
copy /Y "bin\modules\*.py" "%ZIP_NAME%\bin\modules\" >nul

REM 推流器
copy /Y "bin\streamers\*.py" "%ZIP_NAME%\bin\streamers\" >nul
if exist "%ZIP_NAME%\bin\streamers\manager.py" del /Q "%ZIP_NAME%\bin\streamers\manager.py"

REM 前端
if exist "bin\static\index.html" copy /Y "bin\static\index.html" "%ZIP_NAME%\bin\static\" >nul

REM 配置
if exist "config\config.json" copy /Y "config\config.json" "%ZIP_NAME%\config\" >nul

REM 驱动
if exist "driver_files\apriltagtrackers" (
    xcopy /E /I /Y "driver_files\apriltagtrackers" "%ZIP_NAME%\driver_files\apriltagtrackers" >nul
)

if exist "HMD-SDK-Sender" (
    xcopy /E /I /Y "HMD-SDK-Sender" "%ZIP_NAME%\HMD-SDK-Sender" >nul
)

REM HMD remote sender package
if exist "..\HMD-Remote-Sender" (
    mkdir "%ZIP_NAME%\HMD-Remote-Sender"
    if exist "..\HMD-Remote-Sender\hmd_sender.py" copy /Y "..\HMD-Remote-Sender\hmd_sender.py" "%ZIP_NAME%\HMD-Remote-Sender\" >nul
    if exist "..\HMD-Remote-Sender\requirements.txt" copy /Y "..\HMD-Remote-Sender\requirements.txt" "%ZIP_NAME%\HMD-Remote-Sender\" >nul
    if exist "..\HMD-Remote-Sender\install.bat" copy /Y "..\HMD-Remote-Sender\install.bat" "%ZIP_NAME%\HMD-Remote-Sender\" >nul
    if exist "..\HMD-Remote-Sender\install.sh" copy /Y "..\HMD-Remote-Sender\install.sh" "%ZIP_NAME%\HMD-Remote-Sender\" >nul
    if exist "..\HMD-Remote-Sender\start.bat" copy /Y "..\HMD-Remote-Sender\start.bat" "%ZIP_NAME%\HMD-Remote-Sender\" >nul
    if exist "..\HMD-Remote-Sender\start.sh" copy /Y "..\HMD-Remote-Sender\start.sh" "%ZIP_NAME%\HMD-Remote-Sender\" >nul
)

REM 模型文件
echo [复制] 模型文件...
if /i not "%EDITION%"=="lite" if exist "yolo26n-pose.pt" (
    copy /Y "yolo26n-pose.pt" "%ZIP_NAME%\" >nul
    echo   ✓ yolo26n-pose.pt
)
if exist "face_landmarker.task" (
    copy /Y "face_landmarker.task" "%ZIP_NAME%\" >nul
    echo   ✓ face_landmarker.task
)
if exist "hand_landmarker.task" (
    copy /Y "hand_landmarker.task" "%ZIP_NAME%\" >nul
    echo   ✓ hand_landmarker.task
)
if exist "models\pose_landmarker_heavy.task" (
    copy /Y "models\pose_landmarker_heavy.task" "%ZIP_NAME%\models\" >nul
    echo   pose_landmarker_heavy.task
)

REM 脚本和文档
if exist "pose_landmarker_heavy.task" (
    copy /Y "pose_landmarker_heavy.task" "%ZIP_NAME%\" >nul
    echo   pose_landmarker_heavy.task
)

copy /Y "requirements.txt" "%ZIP_NAME%\" >nul
if /i not "%EDITION%"=="lite" copy /Y "requirements-full.txt" "%ZIP_NAME%\" >nul
copy /Y "pyproject.toml" "%ZIP_NAME%\" >nul
copy /Y "uv.lock" "%ZIP_NAME%\" >nul
copy /Y ".python-version" "%ZIP_NAME%\" >nul
copy /Y "setup.bat" "%ZIP_NAME%\" >nul
copy /Y "start.bat" "%ZIP_NAME%\" >nul
if exist "start_lite.bat" copy /Y "start_lite.bat" "%ZIP_NAME%\" >nul
if exist "start_dashboard_lite.bat" copy /Y "start_dashboard_lite.bat" "%ZIP_NAME%\" >nul
if exist "start_dashboard.bat" copy /Y "start_dashboard.bat" "%ZIP_NAME%\" >nul
if exist "start_dashboard.sh" copy /Y "start_dashboard.sh" "%ZIP_NAME%\" >nul
if exist "start_electron_frontend.bat" copy /Y "start_electron_frontend.bat" "%ZIP_NAME%\" >nul
if exist "build_electron_ui.bat" copy /Y "build_electron_ui.bat" "%ZIP_NAME%\" >nul
if exist "electron_frontend" (
    mkdir "%ZIP_NAME%\electron_frontend"
    mkdir "%ZIP_NAME%\electron_frontend\src"
    if exist "electron_frontend\package.json" copy /Y "electron_frontend\package.json" "%ZIP_NAME%\electron_frontend\" >nul
    if exist "electron_frontend\src" xcopy /E /I /Y "electron_frontend\src" "%ZIP_NAME%\electron_frontend\src" >nul
)
if exist "Dockerfile" copy /Y "Dockerfile" "%ZIP_NAME%\" >nul
if exist "docker-compose.yml" copy /Y "docker-compose.yml" "%ZIP_NAME%\" >nul
if exist "README_RELEASE.md" copy /Y "README_RELEASE.md" "%ZIP_NAME%\" >nul
if exist "install_steamvr_driver.bat" copy /Y "install_steamvr_driver.bat" "%ZIP_NAME%\" >nul

REM 测试脚本
if exist "test_system.py" copy /Y "test_system.py" "%ZIP_NAME%\" >nul
if exist "test_dashboard_smoke.py" copy /Y "test_dashboard_smoke.py" "%ZIP_NAME%\" >nul
if exist "test_osc_listen.py" copy /Y "test_osc_listen.py" "%ZIP_NAME%\" >nul
if exist "test_osc_send.py" copy /Y "test_osc_send.py" "%ZIP_NAME%\" >nul

REM 轻量导入检查: 验证 ZIP 内部运行时 helper 与 streamers 公开接口
echo [检查] 源码 ZIP import smoke...
uv run --frozen %UV_EXTRA% python -B -c "import sys; sys.path.insert(0, sys.argv[1]); import desktop_launcher, runtime_paths, runtime_status, face_calibration, mediapipe_runtime, streamers, edition, modules; from streamers import BodyOSCStreamer, BodySteamVRStreamer, FaceVMCStreamer, HandVMTStreamer; assert 'streamers.manager' not in sys.modules" "%CD%\%ZIP_NAME%\bin"
if %errorlevel% neq 0 (
    echo [错误] 源码 ZIP import smoke 失败！
    rmdir /s /q "%ZIP_NAME%"
    exit /b 1
)

REM 打包为 ZIP
echo [打包] 正在压缩...
powershell -Command "Compress-Archive -Path '%ZIP_NAME%' -DestinationPath '%ZIP_NAME%.zip' -Force"

if %errorlevel% neq 0 (
    echo [错误] ZIP 压缩失败！
    pause
    exit /b 1
)

REM 清理临时目录
rmdir /s /q "%ZIP_NAME%"

echo.
echo [成功] ZIP 打包完成！
echo   输出文件: %ZIP_NAME%.zip
echo.
pause
goto MENU

:BUILD_ALL
call :BUILD_EXE
call :BUILD_ZIP
echo.
echo ========================================
echo   所有打包任务完成！
echo ========================================
echo.
pause
goto MENU
