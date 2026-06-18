@echo off
REM Windows bootstrap. A .bat file is not subject to the PowerShell execution
REM policy, so this runs setup.ps1 even on a fresh machine where ".\setup.ps1"
REM would be blocked with "running scripts is disabled on this system".
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "setup.ps1"
