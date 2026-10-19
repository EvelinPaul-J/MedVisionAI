@echo off
setlocal
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo First run run_app.bat once to create the virtual environment.
  pause
  exit /b 1
)
call venv\Scripts\activate.bat
cd backend
python -m pip install -r train\requirements-training.txt
if errorlevel 1 goto :error
python train\train_four_models.py --epochs 3 --batch-size 8 --workers 0
pause
exit /b 0
:error
echo Training dependency installation failed.
pause
exit /b 1
