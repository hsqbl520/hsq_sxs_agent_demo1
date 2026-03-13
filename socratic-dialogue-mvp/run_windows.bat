@echo off
setlocal

cd /d "%~dp0"

set "CONDA_PYTHON=C:\Users\10985\miniconda3\python.exe"
set "PORT=%~1"

if "%PORT%"=="" set "PORT=8010"

if not exist "%CONDA_PYTHON%" (
  echo Conda base python not found: %CONDA_PYTHON%
  exit /b 1
)

for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
  echo Port %PORT% is already in use by PID %%p
  echo Stop that process first, then run this script again.
  exit /b 1
)

echo [1/2] Installing dependencies...
"%CONDA_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  exit /b 1
)

echo [2/2] Starting API on http://127.0.0.1:%PORT%
echo Running without --reload to avoid stale child processes.
"%CONDA_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
