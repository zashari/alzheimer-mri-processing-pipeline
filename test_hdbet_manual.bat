@echo off
echo Setting CUDA memory management environment variables...

REM Critical: Limit CUDA memory allocation to smaller chunks
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set CUDA_LAUNCH_BLOCKING=1

REM Force single-threading to prevent Windows multiprocessing issues
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set TORCH_NUM_THREADS=1

echo.
echo Environment variables set:
echo PYTORCH_CUDA_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF%
echo CUDA_LAUNCH_BLOCKING=%CUDA_LAUNCH_BLOCKING%
echo OMP_NUM_THREADS=%OMP_NUM_THREADS%
echo.
echo Running HD-BET with GPU memory management...
echo.

hd-bet -i "D:\workspace\@zaky-ashari\playgrounds\alzheimer-disease-processing-py-format\outputs\1_splitted_sequential\train\002_S_0295\002_S_0295_sc.nii" -o "D:\workspace\@zaky-ashari\playgrounds\alzheimer-disease-processing-py-format\outputs\2_skull_stripping\test_manual_sc.nii.gz" -device cuda --disable_tta --save_bet_mask

echo.
echo HD-BET completed (or failed). Check output above.
pause