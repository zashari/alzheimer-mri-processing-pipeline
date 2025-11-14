#!/usr/bin/env python
"""Test minimal import to isolate hanging point"""
import sys
import time

def test_import(module_path, description):
    print(f"Testing: {description}...", end="", flush=True)
    start = time.time()
    try:
        exec(f"import {module_path}")
        elapsed = time.time() - start
        print(f" OK ({elapsed:.2f}s)")
        return True
    except Exception as e:
        print(f" ERROR: {e}")
        return False

print("=" * 60)
print("MINIMAL IMPORT TEST - Isolating Hang Point")
print("=" * 60)

# Test basic imports
print("\n1. Testing standard library imports...")
test_import("os", "os")
test_import("sys", "sys")
test_import("time", "time")
test_import("pathlib", "pathlib")

# Test data_processing package structure
print("\n2. Testing data_processing package...")
test_import("data_processing", "data_processing root")

# Test CLI and config
print("\n3. Testing CLI and config modules...")
test_import("data_processing.cli", "CLI module")
test_import("data_processing.config", "config module")

# Test stages
print("\n4. Testing stages module...")
test_import("data_processing.stages", "stages registry")

# Test nifti_processing specifically
print("\n5. Testing nifti_processing module...")
test_import("data_processing.nifti_processing", "nifti_processing root")
test_import("data_processing.nifti_processing.runner", "nifti runner")
test_import("data_processing.nifti_processing.formatter", "nifti formatter")

# Test gpu_utils specifically
print("\n6. Testing gpu_utils (critical)...")
print("   Note: This has lazy loading for torch")
test_import("data_processing.nifti_processing.gpu_utils", "gpu_utils")

# Test skull_stripping module
print("\n7. Testing skull_stripping module...")
test_import("data_processing.nifti_processing.skull_stripping", "skull_stripping root")

# Now test importing specific functions from gpu_utils
print("\n8. Testing specific gpu_utils imports...")
print("   Testing function imports from gpu_utils...", end="", flush=True)
start = time.time()
try:
    from data_processing.nifti_processing.gpu_utils import (
        get_gpu_info,
        get_gpu_memory_info,
        cleanup_gpu_memory,
        setup_gpu_environment
    )
    elapsed = time.time() - start
    print(f" OK ({elapsed:.2f}s)")
except Exception as e:
    print(f" ERROR: {e}")

# Test the skull_stripping runner import
print("\n9. Testing skull_stripping.runner...")
test_import("data_processing.nifti_processing.skull_stripping.runner", "skull_stripping runner")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("If the script hangs at any point, that's where the issue is.")
print("=" * 60)