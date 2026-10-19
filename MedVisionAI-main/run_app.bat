@echo off
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo Creating virtual environment...
  py -3 -m venv venv
  if errorlevel 1 python -m venv venv
  if errorlevel 1 goto :venv_error
)
call venv\Scripts\activate.bat
if not exist backend\requirements-installed.flag (
  echo Installing required packages. Internet is needed only for this first setup...
  python -m pip install -r backend\requirements.txt
  if errorlevel 1 goto :pip_error
  type nul > backend\requirements-installed.flag
)
cd backend
start "RetinaReach AI" http://127.0.0.1:8000
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
exit /b 0
:venv_error
echo Could not create virtual environment. Install Python 3.10-3.12 and try again.
pause
exit /b 1
:pip_error
echo Dependency installation failed. Connect to the internet once and run again.
pause
exit /b 1
