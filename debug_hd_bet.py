#!/usr/bin/env python
"""Debug script to find where HD-BET is hanging"""

import subprocess
import sys
import time

print("=" * 60)
print("HD-BET DEBUG SCRIPT")
print("=" * 60)
print(f"Python: {sys.version}")
print()

# Test 1: Simple Python version check
print("Test 1: Can we run Python subprocess at all?")
try:
    result = subprocess.run(
        [sys.executable, "-c", "print('Hello from subprocess')"],
        capture_output=True,
        text=True,
        timeout=5
    )
    print(f"✅ Success: {result.stdout.strip()}")
except Exception as e:
    print(f"❌ Failed: {e}")

print()

# Test 2: Check if hd-bet command exists
print("Test 2: Can we find hd-bet command?")
try:
    result = subprocess.run(
        ["where", "hd-bet"] if sys.platform == "win32" else ["which", "hd-bet"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        print(f"✅ Found hd-bet at: {result.stdout.strip()}")
    else:
        print("❌ hd-bet not found in PATH")
except Exception as e:
    print(f"❌ Failed: {e}")

print()

# Test 3: Try HD-BET with different methods
print("Test 3: Try hd-bet --help with capture_output")
print("Starting... (will timeout after 10 seconds)")
start_time = time.time()
try:
    result = subprocess.run(
        ["hd-bet", "--help"],
        capture_output=True,
        text=True,
        timeout=10
    )
    elapsed = time.time() - start_time
    print(f"✅ Success in {elapsed:.1f}s (return code: {result.returncode})")
    print(f"   Output length: {len(result.stdout)} chars")
except subprocess.TimeoutExpired:
    print(f"❌ TIMEOUT after 10 seconds - HD-BET is hanging!")
except FileNotFoundError:
    print("❌ HD-BET not found")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Test 4: Try without capture_output
print("Test 4: Try hd-bet --help WITHOUT capture_output (direct to console)")
print("Starting... (will timeout after 10 seconds)")
start_time = time.time()
try:
    result = subprocess.run(
        ["hd-bet", "--help"],
        timeout=10
    )
    elapsed = time.time() - start_time
    print(f"✅ Success in {elapsed:.1f}s (return code: {result.returncode})")
except subprocess.TimeoutExpired:
    print(f"❌ TIMEOUT after 10 seconds - HD-BET is hanging!")
except FileNotFoundError:
    print("❌ HD-BET not found")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Test 5: Try with --version instead
print("Test 5: Try hd-bet --version")
print("Starting... (will timeout after 10 seconds)")
start_time = time.time()
try:
    result = subprocess.run(
        ["hd-bet", "--version"],
        capture_output=True,
        text=True,
        timeout=10
    )
    elapsed = time.time() - start_time
    print(f"✅ Success in {elapsed:.1f}s")
    print(f"   Version: {result.stdout.strip() if result.stdout else result.stderr.strip()}")
except subprocess.TimeoutExpired:
    print(f"❌ TIMEOUT after 10 seconds")
except FileNotFoundError:
    print("❌ HD-BET not found")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("=" * 60)
print("ANALYSIS:")
print("If HD-BET hangs with --help, the issue might be:")
print("1. HD-BET is loading PyTorch/CUDA even for help")
print("2. There's a CUDA initialization problem")
print("3. HD-BET has an import that's hanging")
print("4. Antivirus/firewall is blocking HD-BET")
print()
print("Try running: python -c \"import HD_BET\"")
print("to see if the import itself hangs")