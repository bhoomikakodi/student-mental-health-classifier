@echo off
REM Sets up the virtual environment, installs dependencies, trains the models
REM and launches the Streamlit app (Windows cmd / PowerShell).
REM Usage: run.bat   (PowerShell: .\run.bat)

REM Always run from the project root, regardless of where the script is called from
cd /d "%~dp0"

REM 2. Create virtual environment
if not exist venv (
    python -m venv venv
    if errorlevel 1 goto :error
)

REM Activate: venv\Scripts\activate
call venv\Scripts\activate.bat
if errorlevel 1 goto :error

REM 3. Install dependencies
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
REM    (Notebook users: register the venv as a kernel and select it in Jupyter/VS Code)
python -m pip install ipykernel
if errorlevel 1 goto :error
python -m ipykernel install --user --name student-mental-health-venv
if errorlevel 1 goto :error

REM 4. Train all models and export artifacts
python src\train.py
if errorlevel 1 goto :error

REM 5. Launch interactive Streamlit application
python -m streamlit run app.py
if errorlevel 1 goto :error

goto :eof

:error
echo.
echo Setup failed. See the error above.
exit /b 1
