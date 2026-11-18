@echo off
setlocal
cd /d "%~dp0"
set "MAX_PUBLISHING_ENABLED=false"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment not found in %CD%\.venv
    exit /b 2
)
".venv\Scripts\python.exe" -m scheduler.scheduler
exit /b %ERRORLEVEL%
