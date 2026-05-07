@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "UV_VERSION=0.11.10"
set "UV_SHA256=7a0c424c7bc55a74751f13592235953ebbe182fa00355f7ae3fb7ab734a51638"
set "RUNTIME_ROOT=%LOCALAPPDATA%\HorizonBrief"
set "UV_EXE=%RUNTIME_ROOT%\bin\uv.exe"
set "HORIZON_PYTHON=%RUNTIME_ROOT%\venv\Scripts\python.exe"

pushd "%ROOT%" || (
    echo Failed to enter the Horizon directory: "%ROOT%"
    pause
    exit /b 1
)

if exist "%UV_EXE%" goto sync

echo Downloading local uv %UV_VERSION%...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $runtime=$env:RUNTIME_ROOT; $version=$env:UV_VERSION; $expected=$env:UV_SHA256; $downloadDir=Join-Path $runtime 'downloads'; $binDir=Join-Path $runtime 'bin'; New-Item -ItemType Directory -Force -Path $downloadDir,$binDir | Out-Null; $zipPath=Join-Path $downloadDir ('uv-'+$version+'-x86_64-pc-windows-msvc.zip'); Invoke-WebRequest -Uri ('https://releases.astral.sh/github/uv/releases/download/'+$version+'/uv-x86_64-pc-windows-msvc.zip') -OutFile $zipPath -TimeoutSec 120; $actual=(Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant(); if ($actual -ne $expected) { throw ('uv checksum mismatch: expected '+$expected+' actual '+$actual) }; Expand-Archive -LiteralPath $zipPath -DestinationPath $binDir -Force"
if errorlevel 1 (
    echo.
    echo Failed to download or verify local uv.
    pause
    popd
    exit /b 1
)

:sync
set "UV_CACHE_DIR=%RUNTIME_ROOT%\cache"
set "UV_PYTHON_INSTALL_DIR=%RUNTIME_ROOT%\python"
set "UV_PROJECT_ENVIRONMENT=%RUNTIME_ROOT%\venv"

echo Preparing local Horizon runtime...
"%UV_EXE%" sync --frozen --python 3.13 --no-dev
if errorlevel 1 (
    echo.
    echo Failed to prepare the local Horizon runtime.
    pause
    popd
    exit /b 1
)

"%HORIZON_PYTHON%" -m src.horizon_ext.web.main --help >nul
if errorlevel 1 (
    echo.
    echo Local Horizon runtime was prepared, but horizon-web did not start correctly.
    pause
    popd
    exit /b 1
)

echo.
echo Local Horizon runtime is ready.
echo Runtime location: "%RUNTIME_ROOT%"
echo You can now double-click Horizon Web.lnk or start-horizon-web.cmd.
echo.
pause
popd
exit /b 0
