@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo YOLO26n VRC Mocap - Electron UI
echo ========================================
echo.

where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Node.js。请先安装 Node.js LTS 后再启动 Electron 前端。
    echo 下载地址: https://nodejs.org/
    pause
    exit /b 1
)

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 npm。请确认 Node.js 安装完整并已加入 PATH。
    pause
    exit /b 1
)

if not exist "electron_frontend\node_modules\electron" (
    echo [安装] 首次启动需要安装 Electron 依赖...
    pushd electron_frontend
    call npm install
    if %errorlevel% neq 0 (
        popd
        echo [错误] Electron 依赖安装失败。
        pause
        exit /b 1
    )
    popd
)

pushd electron_frontend
call npm start -- %*
set EXIT_CODE=%errorlevel%
popd

exit /b %EXIT_CODE%
