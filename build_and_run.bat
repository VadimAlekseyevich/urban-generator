@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Urban Generator - Windows launcher. Run from a temporary copy for git pull.
if /I "%URBAN_BAT_WORKER%"=="1" goto :worker
set "URBAN_BAT_WORKER=1"
set "URBAN_ROOT=%~dp0"
set "URBAN_TEMP_COPY=%TEMP%\urban-run-%RANDOM%-%RANDOM%.bat"
copy /y "%~f0" "%URBAN_TEMP_COPY%" >nul
if errorlevel 1 (
    echo [URBAN] Cannot create temporary launcher.
    exit /b 1
)
call "%URBAN_TEMP_COPY%" %*
set "URBAN_RETURN=%ERRORLEVEL%"
del /q "%URBAN_TEMP_COPY%" >nul 2>&1
exit /b %URBAN_RETURN%

:worker
cd /d "%URBAN_ROOT%" || exit /b 1
set "MODE=%~1"
if not defined MODE set "MODE=run"
if /I "%MODE%"=="run" goto :valid
if /I "%MODE%"=="build" goto :valid
if /I "%MODE%"=="test" goto :valid
echo Usage: build_and_run.bat [run ^| build ^| test]
exit /b 2

:valid
if not "%~2"=="" (
    echo Usage: build_and_run.bat [run ^| build ^| test]
    exit /b 2
)
set "STAMP=%RANDOM%"
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "STAMP=%%T"
set "URBAN_LOG_DIR=%CD%\.urban-runs\%STAMP%-%RANDOM%"
mkdir "%URBAN_LOG_DIR%" >nul 2>&1
if not exist "%URBAN_LOG_DIR%\" (
    echo [URBAN] Cannot create log directory.
    exit /b 1
)
set "STATUS=1"
set "UV_FLAGS="
if /I "%URBAN_OFFLINE%"=="1" (
    set "URBAN_SKIP_PULL=1"
    set "UV_FLAGS=--offline"
)
set "UV_NO_PROGRESS=1"
> "%URBAN_LOG_DIR%\run-info.txt" echo mode=%MODE%
>> "%URBAN_LOG_DIR%\run-info.txt" echo offline=%URBAN_OFFLINE%
echo.
echo ============================================================
echo   Urban Generator / Windows
echo   Mode: %MODE%
echo   Project: "%CD%"
echo ============================================================

if /I "%URBAN_SKIP_PULL%"=="1" goto :skip_pull
if not exist ".git" (
    echo [URBAN] No Git checkout. Skipping update.
    > "%URBAN_LOG_DIR%\git-update.log" echo No Git checkout.
    goto :after_pull
)
where git >nul 2>&1
if errorlevel 1 set "PATH=%ProgramFiles%\Git\cmd;%PATH%"
where git >nul 2>&1
if errorlevel 1 (
    echo [URBAN] Git missing. Install Git or set URBAN_SKIP_PULL=1.
    goto :finish
)
git status --porcelain > "%URBAN_LOG_DIR%\git-status.log" 2>&1
if errorlevel 1 (
    call :show_log git-status.log
    goto :finish
)
for %%F in ("%URBAN_LOG_DIR%\git-status.log") do if not "%%~zF"=="0" (
    echo [URBAN] Local changes detected. Commit or stash them, or set URBAN_SKIP_PULL=1.
    goto :finish
)
echo [URBAN] Updating checkout - fast-forward only...
git pull --ff-only > "%URBAN_LOG_DIR%\git-update.log" 2>&1
if errorlevel 1 (
    call :show_log git-update.log
    goto :finish
)
type "%URBAN_LOG_DIR%\git-update.log"
goto :after_pull

:skip_pull
> "%URBAN_LOG_DIR%\git-update.log" echo Git update skipped.
echo [URBAN] Git update skipped.

:after_pull
if exist ".git" git rev-parse HEAD >> "%URBAN_LOG_DIR%\run-info.txt" 2>nul
if not exist "uv.lock" (
    echo [URBAN] uv.lock not found.
    goto :finish
)
if not exist "app.py" (
    echo [URBAN] app.py not found.
    goto :finish
)
call :ensure_uv
if errorlevel 1 goto :finish
uv --version >> "%URBAN_LOG_DIR%\run-info.txt" 2>&1

echo [URBAN] Preparing Python 3.12 and locked dependencies...
uv sync --frozen --python 3.12 %UV_FLAGS% > "%URBAN_LOG_DIR%\sync.log" 2>&1
if errorlevel 1 (
    call :show_log sync.log
    goto :finish
)
echo [URBAN] Checking imports, demo and OSM PBF support...
uv run --frozen --python 3.12 %UV_FLAGS% --with osmium==4.3.2 python -c "import sys, app, osmium, numpy, scipy, rasterio, PIL, streamlit; from pathlib import Path; from urban.io import read_terrain; t=read_terrain(Path('examples/demo.tif').read_bytes()); assert sys.version_info[:2]==(3,12); print(sys.version); print('Demo:', t.classes.shape); print('Imports OK')" > "%URBAN_LOG_DIR%\build.log" 2>&1
if errorlevel 1 (
    call :show_log build.log
    goto :finish
)
type "%URBAN_LOG_DIR%\build.log"
if /I "%MODE%"=="build" goto :success
if /I "%MODE%"=="test" goto :test

echo [URBAN] Starting Streamlit at http://127.0.0.1:8501
echo [URBAN] Open this URL in your browser if it does not open automatically.
echo [URBAN] Press Ctrl+C to stop the server.
echo [URBAN] Live output is in "%URBAN_LOG_DIR%\app.log"
uv run --offline --frozen --python 3.12 --with osmium==4.3.2 streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false > "%URBAN_LOG_DIR%\app.log" 2>&1
set "STATUS=%ERRORLEVEL%"
if not "%STATUS%"=="0" call :show_log app.log
goto :finish

:test
echo [URBAN] Running pytest...
uv run --offline --frozen --python 3.12 --with osmium==4.3.2 python -m pytest -q > "%URBAN_LOG_DIR%\tests.log" 2>&1
set "STATUS=%ERRORLEVEL%"
if not "%STATUS%"=="0" (
    call :show_log tests.log
    goto :finish
)
type "%URBAN_LOG_DIR%\tests.log"
goto :success

:ensure_uv
where uv >nul 2>&1
if not errorlevel 1 exit /b 0
set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"
where uv >nul 2>&1
if not errorlevel 1 exit /b 0
if /I "%URBAN_OFFLINE%"=="1" (
    echo [URBAN] uv is required for offline mode.
    exit /b 1
)
where winget >nul 2>&1
if errorlevel 1 (
    echo [URBAN] Install uv: https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)
choice /C YN /M "Install uv using winget?"
if errorlevel 2 exit /b 1
winget install --id astral-sh.uv --exact --accept-source-agreements --accept-package-agreements > "%URBAN_LOG_DIR%\install-uv.log" 2>&1
if errorlevel 1 (
    call :show_log install-uv.log
    exit /b 1
)
where uv >nul 2>&1
if errorlevel 1 (
    echo [URBAN] uv installed but not on PATH yet. Open a new terminal and retry.
    exit /b 1
)
exit /b 0

:show_log
if exist "%URBAN_LOG_DIR%\%~1" (
    echo [URBAN] Last 40 lines of %~1:
    powershell -NoProfile -Command "Get-Content -LiteralPath (Join-Path $env:URBAN_LOG_DIR '%~1') -Encoding UTF8 -Tail 40"
)
exit /b 0

:success
set "STATUS=0"
:finish
>> "%URBAN_LOG_DIR%\run-info.txt" echo exit_code=%STATUS%
if "%STATUS%"=="0" (echo [URBAN] SUCCESS.) else (echo [URBAN] FAILED. Exit code: %STATUS%)
echo [URBAN] Creating diagnostics archive...
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $d=$env:URBAN_LOG_DIR; $files=@(Get-ChildItem -LiteralPath $d -File | Where-Object { $_.Extension -in '.txt','.log' }); Compress-Archive -LiteralPath $files.FullName -DestinationPath (Join-Path $d 'urban-logs.zip') -Force"
if errorlevel 1 echo [URBAN] Archive creation failed; individual logs remain available.
echo.
echo ============================================================
echo   REPORT FOR CHATGPT
echo   "%URBAN_LOG_DIR%\urban-logs.zip"
echo ============================================================
echo   Review logs for local paths and other private details before sharing.
if /I not "%URBAN_NO_PAUSE%"=="1" if not defined CI pause
exit /b %STATUS%
