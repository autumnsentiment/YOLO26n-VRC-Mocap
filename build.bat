@echo off
chcp 65001 >nul
echo ========================================
echo   YOLO26n VRC Mocap - 打包工具
echo ========================================
echo.

cd /d "%~dp0"

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

REM 检查 PyInstaller
pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [安装] 正在安装 PyInstaller...
    pip install pyinstaller
)

REM 检查模型文件
echo [检查] 模型文件...
set HAS_MODEL=0
if exist "yolo26n-pose.pt" (
    echo   ✓ yolo26n-pose.pt
    set HAS_MODEL=1
) else (
    echo   ✗ yolo26n-pose.pt (缺失，打包后需手动放入)
)
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
pyinstaller yolo26n-vrc-mocap.spec --noconfirm --clean

if %errorlevel% neq 0 (
    echo [错误] 打包失败！
    pause
    exit /b 1
)

echo.
echo [成功] EXE 打包完成！
echo   输出目录: dist\YOLO26n-VRC-Mocap\
echo   主程序: dist\YOLO26n-VRC-Mocap\YOLO26n-VRC-Mocap.exe
echo.

REM 复制缺失的模型文件到 dist
if not exist "dist\YOLO26n-VRC-Mocap\yolo26n-pose.pt" (
    if exist "yolo26n-pose.pt" (
        copy /Y "yolo26n-pose.pt" "dist\YOLO26n-VRC-Mocap\" >nul
        echo   已复制 yolo26n-pose.pt
    )
)
if not exist "dist\YOLO26n-VRC-Mocap\face_landmarker.task" (
    if exist "face_landmarker.task" (
        copy /Y "face_landmarker.task" "dist\YOLO26n-VRC-Mocap\" >nul
        echo   已复制 face_landmarker.task
    )
)
if not exist "dist\YOLO26n-VRC-Mocap\hand_landmarker.task" (
    if exist "hand_landmarker.task" (
        copy /Y "hand_landmarker.task" "dist\YOLO26n-VRC-Mocap\" >nul
        echo   已复制 hand_landmarker.task
    )
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
set ZIP_NAME=YOLO26n-VRC-Mocap_%TIMESTAMP%

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
mkdir "%ZIP_NAME%\datasets"

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
copy /Y "bin\vrc_transmitter.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\yolo_detector.py" "%ZIP_NAME%\bin\" >nul
copy /Y "bin\dashboard_server.py" "%ZIP_NAME%\bin\" >nul

REM 模块
copy /Y "bin\modules\*.py" "%ZIP_NAME%\bin\modules\" >nul

REM 推流器
copy /Y "bin\streamers\*.py" "%ZIP_NAME%\bin\streamers\" >nul

REM 前端
if exist "bin\static\index.html" copy /Y "bin\static\index.html" "%ZIP_NAME%\bin\static\" >nul

REM 配置
if exist "config\config.json" copy /Y "config\config.json" "%ZIP_NAME%\config\" >nul
if exist "config\dataset_template.yaml" copy /Y "config\dataset_template.yaml" "%ZIP_NAME%\config\" >nul

REM 驱动
if exist "driver_files\apriltagtrackers" (
    xcopy /E /I /Y "driver_files\apriltagtrackers" "%ZIP_NAME%\driver_files\apriltagtrackers" >nul
)

REM 模型文件
echo [复制] 模型文件...
if exist "yolo26n-pose.pt" (
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

REM 脚本和文档
copy /Y "requirements.txt" "%ZIP_NAME%\" >nul
copy /Y "setup.bat" "%ZIP_NAME%\" >nul
copy /Y "start.bat" "%ZIP_NAME%\" >nul
if exist "train.bat" copy /Y "train.bat" "%ZIP_NAME%\" >nul
if exist "install_steamvr_driver.bat" copy /Y "install_steamvr_driver.bat" "%ZIP_NAME%\" >nul

REM 测试脚本
if exist "test_system.py" copy /Y "test_system.py" "%ZIP_NAME%\" >nul
if exist "test_osc_listen.py" copy /Y "test_osc_listen.py" "%ZIP_NAME%\" >nul
if exist "test_osc_send.py" copy /Y "test_osc_send.py" "%ZIP_NAME%\" >nul

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
