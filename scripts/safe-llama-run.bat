@echo off
REM safe-llama-run.bat - Windows batch wrapper to prevent stuck llama-cli processes
REM Usage: safe-llama-run.bat [llama-cli args...]

setlocal enabledelayedexpansion

REM Find llama-cli binary
set "LLAMA_CLI="
if exist "build-full\Release\llama-hip-bin\llama-cli.exe" (
    set "LLAMA_CLI=build-full\Release\llama-hip-bin\llama-cli.exe"
) else if exist "build\Release\llama-cli.exe" (
    set "LLAMA_CLI=build\Release\llama-cli.exe"
) else if exist "llama-cli.exe" (
    set "LLAMA_CLI=llama-cli.exe"
)

if not defined LLAMA_CLI (
    echo ERROR: Cannot find llama-cli.exe
    echo Please run from the project root directory
    exit /b 1
)

echo === Safe llama-cli Runner ===
echo.

REM Check for existing processes
echo Checking for existing llama-cli processes...
tasklist /FI "IMAGENAME eq llama-cli.exe" 2>nul | find /i "llama-cli.exe" >nul
if %ERRORLEVEL% equ 0 (
    echo WARNING: Existing llama-cli.exe process^(es^) found!
    echo.
    echo Attempting to kill...
    taskkill.exe /F /IM llama-cli.exe >nul 2>&1
    timeout /t 2 /nobreak >nul

    REM Check again
    tasklist /FI "IMAGENAME eq llama-cli.exe" 2>nul | find /i "llama-cli.exe" >nul
    if !ERRORLEVEL! equ 0 (
        echo ERROR: Failed to kill process^(es^)
        echo Please use Task Manager: Ctrl+Shift+Esc -^> Details -^> End Task
        pause
        exit /b 1
    ) else (
        echo OK: All stuck processes cleaned up
    )
) else (
    echo OK: No existing processes found
)
echo.

REM Run the command
echo Using: !LLAMA_CLI!
echo Running: !LLAMA_CLI! %*
echo.

!LLAMA_CLI! %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% equ 0 (
    echo OK: Command completed successfully
) else (
    echo ERROR: Command exited with code %EXIT_CODE%
)

REM Final cleanup check
echo.
echo Checking for orphaned processes...
tasklist /FI "IMAGENAME eq llama-cli.exe" 2>nul | find /i "llama-cli.exe" >nul
if %ERRORLEVEL% equ 0 (
    echo WARNING: Orphaned process still running!
    echo Use Task Manager to clean up if needed
) else (
    echo OK: Clean exit
)

exit /b %EXIT_CODE%
