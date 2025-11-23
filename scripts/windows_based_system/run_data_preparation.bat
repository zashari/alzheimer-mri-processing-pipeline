@echo off
REM Convenience script for running data preparation stage
REM This script runs both split and analyze actions sequentially

echo ================================================================================
echo  Running Data Preparation Stage
echo ================================================================================
echo.

echo Step 1/2: Splitting data...
adp data_preparation split

if errorlevel 1 (
    echo.
    echo ERROR: Data splitting failed!
    pause
    exit /b 1
)

echo.
echo Step 2/2: Analyzing data...
adp data_preparation analyze

if errorlevel 1 (
    echo.
    echo ERROR: Data analysis failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo  Data Preparation Complete
echo ================================================================================
pause

