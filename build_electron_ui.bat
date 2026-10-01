@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo YOLO26n VRC Mocap - Electron UI Build
echo ========================================
echo.

where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js was not found. Install Node.js LTS first.
    echo Download: https://nodejs.org/
    exit /b 1
)

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm was not found. Check that Node.js is installed and in PATH.
    exit /b 1
)

pushd electron_frontend
if not exist "node_modules\electron" (
    echo [INSTALL] Installing Electron dependencies...
    call npm install
    if errorlevel 1 (
        popd
        echo [ERROR] npm install failed.
        exit /b 1
    )
)

echo [BUILD] Trying electron-builder portable build...
set "CSC_IDENTITY_AUTO_DISCOVERY=false"
call npm run build
set "BUILDER_EXIT=%errorlevel%"
popd

if not exist "dist\YOLO26n-VRC-Mocap" mkdir "dist\YOLO26n-VRC-Mocap"

if "%BUILDER_EXIT%"=="0" (
    for %%F in ("dist\electron\YOLO26n-VRC-Mocap-Electron*.exe") do (
        copy /Y "%%~fF" "dist\YOLO26n-VRC-Mocap\YOLO26n-VRC-Mocap-Electron.exe" >nul
    )
)

if exist "dist\YOLO26n-VRC-Mocap\YOLO26n-VRC-Mocap-Electron.exe" (
    echo [OK] Electron UI generated:
    echo   dist\YOLO26n-VRC-Mocap\YOLO26n-VRC-Mocap-Electron.exe
    exit /b 0
)

echo [INFO] electron-builder did not produce a portable exe. Using Electron runtime fallback...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$root=(Get-Location).Path;" ^
  "$electronVersion=(Get-Content -Raw 'electron_frontend/node_modules/electron/package.json' | ConvertFrom-Json).version;" ^
  "$zip=Join-Path $env:LOCALAPPDATA ('electron/Cache/electron-v' + $electronVersion + '-win32-x64.zip');" ^
  "if(-not(Test-Path $zip)){ Push-Location electron_frontend; $env:force_no_cache='true'; node node_modules/electron/install.js; Pop-Location }" ^
  "if(-not(Test-Path $zip)){ throw 'Electron runtime zip not found: ' + $zip }" ^
  "$publishDir=Join-Path $root 'dist/YOLO26n-VRC-Mocap';" ^
  "$tempRuntime=Join-Path $env:TEMP 'yolo26n-electron-runtime';" ^
  "$publishExe=Join-Path $publishDir 'YOLO26n-VRC-Mocap-Electron.exe';" ^
  "$appResources=Join-Path $publishDir 'resources/app';" ^
  "Remove-Item -Recurse -Force $tempRuntime -ErrorAction SilentlyContinue;" ^
  "New-Item -ItemType Directory -Force -Path $tempRuntime | Out-Null;" ^
  "Expand-Archive -LiteralPath $zip -DestinationPath $tempRuntime -Force;" ^
  "Copy-Item -Recurse -Force -Path (Join-Path $tempRuntime '*') -Destination $publishDir;" ^
  "$electronExe=Join-Path $publishDir 'electron.exe';" ^
  "if(-not(Test-Path $electronExe)){ throw 'Electron runtime missing electron.exe' }" ^
  "Move-Item -LiteralPath $electronExe -Destination $publishExe -Force;" ^
  "Remove-Item -Recurse -Force $appResources -ErrorAction SilentlyContinue;" ^
  "New-Item -ItemType Directory -Force -Path $appResources | Out-Null;" ^
  "Copy-Item -LiteralPath 'electron_frontend/package.json' -Destination $appResources -Force;" ^
  "Copy-Item -Recurse -LiteralPath 'electron_frontend/src' -Destination $appResources -Force;" ^
  "Copy-Item -LiteralPath 'electron_frontend/README.md' -Destination $appResources -Force;" ^
  "Remove-Item -Recurse -Force $tempRuntime -ErrorAction SilentlyContinue;" ^
  "Write-Host '[OK] Electron runtime fallback generated:';" ^
  "Write-Host '  dist/YOLO26n-VRC-Mocap/YOLO26n-VRC-Mocap-Electron.exe';"

if errorlevel 1 (
    echo [ERROR] Electron runtime fallback failed.
    exit /b 1
)

exit /b 0
