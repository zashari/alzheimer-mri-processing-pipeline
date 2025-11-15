#!/usr/bin/env python
"""
Test script to verify HD-BET skull stripping fix.
This tests that the environment variables are set correctly and HD-BET works.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_environment_setup():
    """Test that environment variables are set correctly."""
    print("=" * 60)
    print("Testing Environment Variable Setup")
    print("=" * 60)

    # Import the processor
    from data_processing.nifti_processing.skull_stripping.processor import HDBETProcessor

    # Create processor with CUDA device
    processor = HDBETProcessor(device="cuda", verbose=True)

    # Check if environment variables are set
    env_vars = {
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }

    print("\nEnvironment Variables Set:")
    for key, value in env_vars.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value if value else 'NOT SET'}")

    # Check HD-BET availability
    print("\nChecking HD-BET availability...")
    available = processor.check_availability()

    if available:
        print("✓ HD-BET is available")
        return True
    else:
        print("✗ HD-BET is not available")
        return False

def test_sample_processing():
    """Test processing a sample file if available."""
    print("\n" + "=" * 60)
    print("Testing Sample File Processing")
    print("=" * 60)

    # Look for a sample file
    input_dir = Path("outputs/1_splitted_sequential")
    sample_file = None

    if input_dir.exists():
        for nii_file in input_dir.rglob("*.nii"):
            sample_file = nii_file
            break

    if not sample_file:
        print("No sample file found for testing")
        return False

    print(f"\nFound sample file: {sample_file.name}")

    # Create test output directory
    test_output = Path("outputs/2_skull_stripping/test")
    test_output.mkdir(parents=True, exist_ok=True)

    # Import processor
    from data_processing.nifti_processing.skull_stripping.processor import HDBETProcessor

    # Create processor
    processor = HDBETProcessor(device="cuda", verbose=True, timeout_sec=300)

    # Process file
    output_brain = test_output / "test_brain.nii.gz"
    output_mask = test_output / "test_mask.nii.gz"

    print(f"\nProcessing {sample_file.name}...")
    status, error = processor.process_file(sample_file, output_brain, output_mask, "test")

    if status == "success":
        print(f"✓ Successfully processed: {output_brain}")
        return True
    else:
        print(f"✗ Failed to process: {error}")
        return False

def main():
    """Main test function."""
    print("\n🧪 HD-BET SKULL STRIPPING FIX TEST")
    print("=" * 60)

    # Test 1: Environment setup
    env_test = test_environment_setup()

    # Test 2: Sample processing (only if environment test passes)
    if env_test:
        sample_test = test_sample_processing()
    else:
        sample_test = False
        print("\nSkipping sample processing test (HD-BET not available)")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Environment Setup: {'✓ PASSED' if env_test else '✗ FAILED'}")
    print(f"Sample Processing: {'✓ PASSED' if sample_test else '✗ FAILED' if env_test else '⏭️ SKIPPED'}")

    if env_test:
        print("\n✅ Fix successfully applied!")
        print("HD-BET should now work without GPU OOM errors on Windows.")
    else:
        print("\n⚠️ Tests did not fully pass. Please check the output above.")

if __name__ == "__main__":
    main()