@echo off
setlocal
set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_all_efficientnet_b3_50.ps1" -PythonExe "..\LCC_GPU\python.exe" %*
