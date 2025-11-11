@echo off
REM Convenience script for running environment setup stage
REM This script wraps the Python CLI command for easier execution

echo ================================================================================
echo  Running Environment Setup Stage
echo ================================================================================
echo.

python -m data_processing.cli environment_setup setup --auto-install true --perf-test full

if errorlevel 1 (
    echo.
    echo ERROR: Environment setup failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo  Environment Setup Complete
echo ================================================================================
pause

