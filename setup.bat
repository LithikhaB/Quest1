@echo off
REM One-command setup: venv + dependencies + test suite.
setlocal
cd /d "%~dp0"

echo [1/3] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (echo FAILED: could not create venv & exit /b 1)

echo [2/3] Installing dependencies...
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (echo FAILED: torch CPU install & exit /b 1)
call .venv\Scripts\python.exe -m pip install --upgrade pip
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (echo FAILED: dependency installation & exit /b 1)

echo [3/3] Running unit tests...
call .venv\Scripts\python.exe -m pytest tests/ -m "not integration"
if errorlevel 1 (echo FAILED: tests & exit /b 1)

echo.
echo Setup complete. Activate with:  .venv\Scripts\activate
echo Run the pipeline with:          streamlit run app.py
endlocal
