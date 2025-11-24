@echo off
REM Premium Vietnamese Keyword Clustering App
REM Ensures the app runs from the virtual environment

echo Starting Premium Keyword Clustering App...
echo.

cd /d "%~dp0"
venv\Scripts\python.exe -m streamlit run app.py

pause
