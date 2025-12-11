@echo off
echo ==========================================
echo   Keyword Clustering Full-Stack App
echo ==========================================
echo.
echo Starting both servers...
echo.

cd /d "%~dp0"

REM Start backend in a new window
start "Backend Server" cmd /k "call start-backend.bat"

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend in a new window
start "Frontend Server" cmd /k "call start-frontend.bat"

echo.
echo ==========================================
echo Both servers are starting...
echo.
echo Backend API: http://localhost:8000
echo Frontend:    http://localhost:3000
echo.
echo API Docs:    http://localhost:8000/docs
echo ==========================================
echo.
echo Close this window when done.
pause
