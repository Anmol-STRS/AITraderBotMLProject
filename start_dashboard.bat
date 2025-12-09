@echo off
echo ================================================
echo Starting AI Trading Dashboard
echo ================================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run the Python startup script
python start_dashboard.py

pause
