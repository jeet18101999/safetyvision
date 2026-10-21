@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Run setup.ps1 first:  powershell -ExecutionPolicy Bypass -File setup.ps1
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" main.py
