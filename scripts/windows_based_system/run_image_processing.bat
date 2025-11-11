@echo off
REM Convenience script for running image processing stage
REM This script runs all image processing substages sequentially

echo ================================================================================
echo  Running Image Processing Stage
echo ================================================================================
echo.

echo Step 1/3: Center Crop...
python -m data_processing.cli image_processing process --substage center_crop
if errorlevel 1 goto :error

echo.
echo Step 2/3: Image Enhancement...
python -m data_processing.cli image_processing process --substage image_enhancement
if errorlevel 1 goto :error

echo.
echo Step 3/3: Data Balancing...
python -m data_processing.cli image_processing process --substage data_balancing
if errorlevel 1 goto :error

echo.
echo ================================================================================
echo  Image Processing Complete
echo ================================================================================
pause
exit /b 0

:error
echo.
echo ERROR: Image processing failed at one of the steps!
pause
exit /b 1

