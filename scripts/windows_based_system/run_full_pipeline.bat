@echo off
REM Convenience script for running the complete pipeline end-to-end
REM WARNING: This will run all stages sequentially and may take a very long time
REM Ensure your machine has sufficient resources (RAM, disk space, GPU) before running

echo ================================================================================
echo  Running Full Pipeline - All Stages
echo ================================================================================
echo.
echo WARNING: This will run all stages sequentially.
echo Estimated time: Several hours depending on dataset size and hardware.
echo.
pause

setlocal enabledelayedexpansion
set START_TIME=%time%

echo.
echo ================================================================================
echo  Stage 1/4: Environment Setup
echo ================================================================================
python -m data_processing.cli environment_setup setup --auto-install true --perf-test full
if errorlevel 1 goto :error

echo.
echo ================================================================================
echo  Stage 2/4: Data Preparation
echo ================================================================================
python -m data_processing.cli data_preparation split
if errorlevel 1 goto :error
python -m data_processing.cli data_preparation analyze
if errorlevel 1 goto :error

echo.
echo ================================================================================
echo  Stage 3/4: NIfTI Processing
echo ================================================================================
python -m data_processing.cli nifti_processing test --substage skull_stripping
if errorlevel 1 goto :error
python -m data_processing.cli nifti_processing process --substage skull_stripping
if errorlevel 1 goto :error
python -m data_processing.cli nifti_processing test --substage template_registration
if errorlevel 1 goto :error
python -m data_processing.cli nifti_processing process --substage template_registration
if errorlevel 1 goto :error
python -m data_processing.cli nifti_processing process --substage labelling
if errorlevel 1 goto :error
python -m data_processing.cli nifti_processing process --substage twoD_conversion
if errorlevel 1 goto :error

echo.
echo ================================================================================
echo  Stage 4/4: Image Processing
echo ================================================================================
python -m data_processing.cli image_processing process --substage center_crop
if errorlevel 1 goto :error
python -m data_processing.cli image_processing process --substage image_enhancement
if errorlevel 1 goto :error
python -m data_processing.cli image_processing process --substage data_balancing
if errorlevel 1 goto :error

set END_TIME=%time%

echo.
echo ================================================================================
echo  Full Pipeline Complete!
echo ================================================================================
echo Started: %START_TIME%
echo Finished: %END_TIME%
echo ================================================================================
pause
exit /b 0

:error
echo.
echo ERROR: Pipeline failed at one of the stages!
pause
exit /b 1

