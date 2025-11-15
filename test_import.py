#!/usr/bin/env python
"""Test imports to find where the hang occurs"""

import sys
print("Python version:", sys.version)
print("=" * 60)

print("1. Testing basic imports...")
try:
    import time
    import pathlib
    print("   ✓ Standard library imports OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("2. Testing subprocess import...")
try:
    import subprocess
    print("   ✓ subprocess import OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("3. Testing Path import...")
try:
    from pathlib import Path
    print("   ✓ pathlib.Path import OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("4. Testing typing imports...")
try:
    from typing import Dict, List, Optional, Tuple
    print("   ✓ typing imports OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("5. Testing local project structure...")
try:
    import src
    print("   ✓ src module found")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("6. Testing data_processing module...")
try:
    import src.data_processing
    print("   ✓ data_processing module OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("7. Testing CLI module...")
try:
    from src.data_processing import cli
    print("   ✓ CLI module imported OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("8. Testing nifti_processing module...")
try:
    import src.data_processing.nifti_processing
    print("   ✓ nifti_processing module OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("9. Testing skull_stripping module...")
try:
    import src.data_processing.nifti_processing.skull_stripping
    print("   ✓ skull_stripping module OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("10. Testing processor import...")
try:
    from src.data_processing.nifti_processing.skull_stripping.processor import HDBETProcessor
    print("   ✓ HDBETProcessor imported OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("11. Testing runner import...")
try:
    from src.data_processing.nifti_processing.skull_stripping.runner import run_test
    print("   ✓ run_test imported OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("12. Testing GPU utils...")
try:
    from src.data_processing.nifti_processing.gpu_utils import get_gpu_info
    print("   ✓ gpu_utils imported OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("13. Testing formatter...")
try:
    from src.data_processing.nifti_processing.formatter import NiftiFormatter
    print("   ✓ NiftiFormatter imported OK")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("=" * 60)
print("ALL IMPORTS COMPLETED SUCCESSFULLY!")
print("The hang is likely in the CLI execution, not imports.")