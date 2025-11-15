#!/usr/bin/env python
"""
Manual HD-BET test with GPU memory management.
This script sets the necessary environment variables before running HD-BET
to prevent GPU Out of Memory errors on GPUs with limited VRAM.
"""

import os
import subprocess
import sys

def setup_cuda_memory_management():
    """Set environment variables to prevent GPU OOM errors."""
    # Critical: Limit PyTorch CUDA memory allocation to smaller chunks
    # This prevents PyTorch from trying to allocate 11.39GB at once
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    # Force synchronous CUDA execution for better memory management
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Limit CPU threads to prevent Windows multiprocessing issues
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TORCH_NUM_THREADS"] = "1"

    print("✓ CUDA memory management environment variables set:")
    print(f"  PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    print(f"  CUDA_LAUNCH_BLOCKING = {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    print(f"  OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    print()

def test_hdbet():
    """Run HD-BET with proper memory management."""
    # Input and output paths
    input_file = r"D:\workspace\@zaky-ashari\playgrounds\alzheimer-disease-processing-py-format\outputs\1_splitted_sequential\train\002_S_0295\002_S_0295_sc.nii"
    output_file = r"D:\workspace\@zaky-ashari\playgrounds\alzheimer-disease-processing-py-format\outputs\2_skull_stripping\test_manual_sc.nii.gz"

    # Build command
    cmd = [
        "hd-bet",
        "-i", input_file,
        "-o", output_file,
        "-device", "cuda",
        "--disable_tta",
        "--save_bet_mask"
    ]

    print("Running HD-BET with command:")
    print(" ".join(cmd))
    print()

    try:
        # Run HD-BET
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )

        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print("\n✓ HD-BET completed successfully!")
            print(f"  Output saved to: {output_file}")
        else:
            print(f"\n✗ HD-BET failed with return code: {result.returncode}")

    except subprocess.TimeoutExpired:
        print("\n✗ HD-BET timed out after 600 seconds")
    except Exception as e:
        print(f"\n✗ Error running HD-BET: {e}")

def main():
    print("=" * 60)
    print("HD-BET Manual Test with GPU Memory Management")
    print("=" * 60)
    print()

    # Step 1: Setup environment
    print("Step 1: Setting up CUDA memory management...")
    setup_cuda_memory_management()

    # Step 2: Run HD-BET
    print("Step 2: Running HD-BET...")
    test_hdbet()

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)

if __name__ == "__main__":
    main()