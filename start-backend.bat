@echo off
echo Starting FastAPI Backend...
echo.

cd /d "%~dp0"

REM Activate the existing virtual environment with lower Python version
call venv\Scripts\activate

REM Install FastAPI dependencies if needed
pip install fastapi uvicorn[standard] --quiet

REM Start the backend server
cd backend
echo Backend running at http://localhost:8000
echo Press Ctrl+C to stop
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
