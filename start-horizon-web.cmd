@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
set "URL=http://127.0.0.1:8787/"
set "HOST=127.0.0.1"
set "PORT=8787"
set "READY_TIMEOUT_SECONDS=45"
set "RUNTIME_ROOT=%LOCALAPPDATA%\HorizonBrief"
set "LOCAL_PYTHON=%RUNTIME_ROOT%\venv\Scripts\python.exe"

pushd "%ROOT%" || (
    echo Failed to enter the Horizon directory: "%ROOT%"
    pause
    exit /b 1
)

call :check_ready
if not errorlevel 1 (
    echo Horizon web dashboard is already running at %URL%
    start "" "%URL%"
    popd
    exit /b 0
)

if not exist "%LOCAL_PYTHON%" (
    echo.
    echo Local Horizon runtime was not found:
    echo "%LOCAL_PYTHON%"
    echo.
    echo Run setup-horizon-local.cmd once from the repository root.
    echo Then double-click this launcher again.
    echo.
    pause
    popd
    exit /b 1
)

echo Starting Horizon web dashboard at %URL%
start "Horizon Web" /D "%ROOT%" cmd /k ""%LOCAL_PYTHON%" -m src.horizon_ext.web.main --host %HOST% --port %PORT%"

echo Waiting for the server to become ready...
for /l %%I in (1,1,%READY_TIMEOUT_SECONDS%) do (
    call :check_ready
    if not errorlevel 1 (
        echo Server is ready. Opening browser...
        start "" "%URL%"
        popd
        exit /b 0
    )
    timeout /t 1 /nobreak >nul
)

echo.
echo The server did not respond within %READY_TIMEOUT_SECONDS% seconds.
echo Check the "Horizon Web" console window for startup errors.
echo When it is ready, open %URL% manually.
echo.
pause
popd
exit /b 1

:check_ready
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $response = Invoke-WebRequest -UseBasicParsing -Uri '%URL%' -TimeoutSec 2; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { exit 0 }; exit 1 } catch { exit 1 }" >nul 2>nul
exit /b %ERRORLEVEL%
