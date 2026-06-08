@echo off
chcp 65001 >nul
echo ========================================
echo   SteamVR 驱动安装/卸载工具
echo   (ApriltagTrackers for YOLO26n-VRC-Mocap)
echo ========================================
echo.

REM 检查管理员权限
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 建议以管理员身份运行，正在提权...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

REM 驱动路径 (相对于脚本所在目录)
set "DRIVER_DIR=%~dp0driver_files\apriltagtrackers"

if not exist "%DRIVER_DIR%\driver.vrdrivermanifest" (
    echo [错误] 未找到驱动文件: %DRIVER_DIR%
    echo 请确认 driver_files\apriltagtrackers 目录存在
    pause
    exit /b 1
)

REM SteamVR 配置文件路径
set "VRPATH=%LOCALAPPDATA%\openvr\openvrpaths.vrpath"

if not exist "%VRPATH%" (
    echo [错误] 未找到 SteamVR 配置文件
    echo 请先安装并运行一次 SteamVR
    pause
    exit /b 1
)

echo 当前 SteamVR 配置: %VRPATH%
echo 驱动目录: %DRIVER_DIR%
echo.

:MENU
echo 请选择操作:
echo   [1] 安装驱动 (注册到 SteamVR)
echo   [2] 卸载驱动 (从 SteamVR 移除)
echo   [3] 查看当前已注册驱动
echo   [0] 退出
echo.
set /p CHOICE=请输入选项:

if "%CHOICE%"=="1" goto INSTALL
if "%CHOICE%"=="2" goto UNINSTALL
if "%CHOICE%"=="3" goto LIST
if "%CHOICE%"=="0" exit /b 0
echo 无效选项
goto MENU

:INSTALL
echo.
echo [安装] 正在注册驱动到 SteamVR...
powershell -ExecutionPolicy Bypass -Command ^
    "$vrpath = '%VRPATH%';" ^
    "$driverDir = '%DRIVER_DIR%'.Replace('\','/');" ^
    "$json = Get-Content $vrpath -Raw | ConvertFrom-Json;" ^
    "$existing = $json.external_drivers | Where-Object { $_ -like '*apriltagtrackers*' };" ^
    "if ($existing) { Write-Host '[提示] 驱动已注册，路径:' $existing; exit 0 };" ^
    "if (-not $json.external_drivers) { $json | Add-Member -NotePropertyName 'external_drivers' -NotePropertyValue @() };" ^
    "$json.external_drivers += $driverDir;" ^
    "$json | ConvertTo-Json -Depth 10 | Set-Content $vrpath -Encoding UTF8;" ^
    "Write-Host '[成功] 驱动已注册到 SteamVR';" ^
    "Write-Host '请重启 SteamVR 以加载驱动'"

echo.
echo [完成] 安装结束
pause
exit /b 0

:UNINSTALL
echo.
echo [卸载] 正在从 SteamVR 移除驱动...
powershell -ExecutionPolicy Bypass -Command ^
    "$vrpath = '%VRPATH%';" ^
    "$json = Get-Content $vrpath -Raw | ConvertFrom-Json;" ^
    "$before = $json.external_drivers.Count;" ^
    "$json.external_drivers = @($json.external_drivers | Where-Object { $_ -notlike '*apriltagtrackers*' });" ^
    "$after = $json.external_drivers.Count;" ^
    "if ($before -eq $after) { Write-Host '[提示] 未找到 apriltagtrackers 驱动，无需卸载'; exit 0 };" ^
    "$json | ConvertTo-Json -Depth 10 | Set-Content $vrpath -Encoding UTF8;" ^
    "Write-Host '[成功] 驱动已从 SteamVR 移除';" ^
    "Write-Host '请重启 SteamVR 以生效'"

echo.
echo [完成] 卸载结束
pause
exit /b 0

:LIST
echo.
echo [查看] 当前已注册的外部驱动:
powershell -ExecutionPolicy Bypass -Command ^
    "$vrpath = '%VRPATH%';" ^
    "$json = Get-Content $vrpath -Raw | ConvertFrom-Json;" ^
    "if ($json.external_drivers) { $json.external_drivers | ForEach-Object { Write-Host '  -' $_ } }" ^
    "else { Write-Host '  (无外部驱动)' }"

echo.
pause
goto MENU
