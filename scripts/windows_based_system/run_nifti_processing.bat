@echo off
REM Convenience script for running NIfTI processing stage
REM This script runs all NIfTI processing substages sequentially

echo ================================================================================
echo  Running NIfTI Processing Stage
echo ================================================================================
echo.

echo Step 1/6: Skull Stripping (Test)...
adp nifti_processing test --substage skull_stripping
if errorlevel 1 goto :error

echo.
echo Step 2/6: Skull Stripping (Process)...
adp nifti_processing process --substage skull_stripping
if errorlevel 1 goto :error

echo.
echo Step 3/6: Template Registration (Test)...
adp nifti_processing test --substage template_registration
if errorlevel 1 goto :error

echo.
echo Step 4/6: Template Registration (Process)...
adp nifti_processing process --substage template_registration
if errorlevel 1 goto :error

echo.
echo Step 5/6: Labelling...
adp nifti_processing process --substage labelling
if errorlevel 1 goto :error

echo.
echo Step 6/6: 2D Conversion...
adp nifti_processing process --substage twoD_conversion
if errorlevel 1 goto :error

echo.
echo ================================================================================
echo  NIfTI Processing Complete
echo ================================================================================
pause
exit /b 0

:error
echo.
echo ERROR: NIfTI processing failed at one of the steps!
pause
exit /b 1

