@echo off
echo Starting Next.js Frontend...
echo.

cd /d "%~dp0nextjs-app"

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
)

echo.
echo Frontend running at http://localhost:3000
echo Press Ctrl+C to stop
npx next dev
