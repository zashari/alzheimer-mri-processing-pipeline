from __future__ import annotations

import gc
import platform
import re
import subprocess
import time
from typing import Dict, Optional, Tuple


def _run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    """Execute a command with proper error handling."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        print(f"   [X] Command timed out: {' '.join(cmd)}")
        return subprocess.CompletedProcess(cmd, 1, "", f"Timeout: {e}")
    except Exception as e:
        print(f"   [X] Failed: {' '.join(cmd)}\n   -> {e}")
        return subprocess.CompletedProcess(cmd, 1, "", str(e))


def detect_gpu(silent: bool = False) -> Tuple[Optional[str], bool]:
    """Comprehensive GPU detection with optional output."""
    gpu_name, cuda_available = None, False

    # Try nvidia-smi first (most reliable)
    cp = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=5)
    if cp.returncode == 0 and cp.stdout.strip():
        gpu_name = cp.stdout.strip().splitlines()[0]
        cuda_available = True
        if not silent:
            print(f"   [GPU] Detected GPU: {gpu_name}")
        return gpu_name, cuda_available
    else:
        if not silent:
            print("   [INFO] nvidia-smi not available or no output, probing PyTorch...")

    # Fallback via PyTorch if available
    if not gpu_name:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                cuda_available = True
                if not silent:
                    print(f"   [GPU] Detected GPU via PyTorch: {gpu_name}")
                return gpu_name, cuda_available
        except ImportError:
            if not silent:
                print("   [INFO] PyTorch not installed yet, skipping torch-based detection")
        except Exception as e:
            if not silent:
                print(f"   [WARN] PyTorch detection failed: {e}")

    # Windows system query as last resort
    if not gpu_name and platform.system() == "Windows":
        cp2 = _run(["wmic", "path", "win32_VideoController", "get", "name"])
        if cp2.returncode == 0 and "NVIDIA" in cp2.stdout:
            for line in cp2.stdout.splitlines():
                if "NVIDIA" in line:
                    gpu_name = line.strip()
                    cuda_available = True
                    if not silent:
                        print(f"   [GPU] Detected GPU via system: {gpu_name}")
                    return gpu_name, cuda_available

    if not gpu_name and not silent:
        print("   [WARN] No NVIDIA GPU detected")

    return gpu_name, cuda_available


def pick_cuda_tag_from_driver(silent: bool = False) -> Optional[str]:
    """Determine the best CUDA tag based on driver version."""
    try:
        cp = _run(["nvidia-smi"], timeout=5)
        if cp.returncode == 0 and cp.stdout:
            m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", cp.stdout)
            if m:
                major, minor = map(int, m.groups())
                if major >= 13 or (major == 12 and minor >= 6):
                    if not silent:
                        print("   [CUDA] Driver CUDA >= 12.6 -> selecting cu126")
                    return "cu126"
                if major == 12 and minor >= 4:
                    if not silent:
                        print("   [CUDA] Driver CUDA 12.4/12.5 -> selecting cu124")
                    return "cu124"
                if major == 12:
                    if not silent:
                        print("   [CUDA] Driver CUDA 12.0-12.3 -> selecting cu121")
                    return "cu121"
                if major == 11 and minor >= 8:
                    if not silent:
                        print("   [CUDA] Driver CUDA 11.8+ -> selecting cu118")
                    return "cu118"
                if not silent:
                    print(f"   [CUDA] Driver CUDA {major}.{minor} -> no matching tag")
                return None
    except Exception as e:
        if not silent:
            print(f"   [WARN] Could not detect CUDA version: {e}")

    if not silent:
        print("   [INFO] Could not parse driver CUDA; defaulting to cu126")
    return "cu126"


def get_device_properties_safe(props) -> Dict:
    """Safely extract device properties across different PyTorch versions."""
    properties = {}

    # List of possible attribute names across versions
    attr_mappings = {
        "clock_rate": ["clock_rate", "clockRate", "max_clock_rate"],
        "memory_clock_rate": [
            "memory_clock_rate",
            "memoryClockRate",
            "max_memory_clock_rate",
        ],
        "memory_bus_width": ["memory_bus_width", "memoryBusWidth"],
        "l2_cache_size": ["l2_cache_size", "l2CacheSize"],
    }

    # Try different attribute names
    for key, possible_attrs in attr_mappings.items():
        for attr in possible_attrs:
            if hasattr(props, attr):
                properties[key] = getattr(props, attr)
                break

    return properties


def verify_gpu_setup(run_performance_test: bool = True, silent: bool = False) -> Dict:
    """Comprehensive GPU verification for machine learning."""
    info: Dict[str, object] = {
        "cuda_available": False,
        "device_name": None,
        "cuda_version": None,
        "cudnn_version": None,
        "compute_capability": None,
        "memory_total_gb": None,
        "tensor_cores": False,
        "mixed_precision": False,
        "perf_test_tflops": None,
    }

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        info["cuda_available"] = cuda_available

        if not cuda_available:
            print("\n[X] CUDA not available. Troubleshooting:")
            print("   1. Check NVIDIA drivers: nvidia-smi")
            print("   2. Verify CUDA installation")
            print("   3. Ensure PyTorch CUDA version is installed")
            print("   4. Restart kernel after PyTorch installation")

            # Check if NVIDIA GPU is detected by system
            try:
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    print("\n[OK] nvidia-smi works - GPU detected by system")
                    print("[FIX] Issue likely with PyTorch CUDA installation")
                    print("   Try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
                else:
                    print("\n[X] nvidia-smi failed - check NVIDIA drivers")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print("\n[X] nvidia-smi not found - install NVIDIA drivers first")

            return info

        # Get GPU information
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)

        print(f"\n[GPU] CUDA Available: YES")
        print(f"[GPU] GPU Count: {gpu_count}")
        print(f"[GPU] Current Device: {current_device}")
        print(f"[GPU] GPU Name: {gpu_name}")

        info["device_name"] = gpu_name
        info["cuda_version"] = torch.version.cuda

        # Detailed GPU properties
        props = torch.cuda.get_device_properties(current_device)

        # Memory information
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        memory_total = props.total_memory / 1024**3

        info["memory_total_gb"] = memory_total
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["sm_count"] = props.multi_processor_count

        print(f"[MEM] GPU Memory:")
        print(f"   * Total: {memory_total:.1f} GB")
        print(f"   * Reserved: {memory_reserved:.1f} GB")
        print(f"   * Allocated: {memory_allocated:.1f} GB")
        print(f"   * Available: {memory_total - memory_reserved:.1f} GB")

        # Technical specifications
        print(f"[TECH] Technical Details:")
        print(f"   * CUDA Version: {torch.version.cuda}")
        print(f"   * cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   * cuDNN Enabled: {'Yes' if torch.backends.cudnn.enabled else 'No'}")
        print(f"   * Compute Capability: {props.major}.{props.minor}")
        print(f"   * Multiprocessors: {props.multi_processor_count}")

        info["cudnn_version"] = torch.backends.cudnn.version()

        # Estimate CUDA cores based on architecture
        cuda_cores_per_mp = {
            (7, 0): 64,   # Volta
            (7, 5): 64,   # Turing
            (8, 0): 64,   # Ampere A100
            (8, 6): 128,  # Ampere RTX 30xx
            (8, 9): 128,  # Ada Lovelace RTX 40xx
            (9, 0): 128,  # Hopper
        }
        cores_per_mp = cuda_cores_per_mp.get((props.major, props.minor), 64)
        print(f"   * CUDA Cores: ~{props.multi_processor_count * cores_per_mp}")

        # Performance characteristics (version-safe)
        print(f"[PERF] Performance Specs:")
        safe_props = get_device_properties_safe(props)

        if "memory_clock_rate" in safe_props:
            print(f"   * Memory Clock: {safe_props['memory_clock_rate'] / 1000:.1f} MHz")

        if "memory_bus_width" in safe_props:
            print(f"   * Memory Bus Width: {safe_props['memory_bus_width']}-bit")

            # Calculate bandwidth if we have both values
            if "memory_clock_rate" in safe_props:
                bandwidth = (
                    safe_props["memory_clock_rate"]
                    * safe_props["memory_bus_width"]
                    * 2
                    / 8
                    / 1e6
                )
                print(f"   * Memory Bandwidth: {bandwidth:.1f} GB/s")

        # Feature support
        tensor_cores = props.major >= 7
        mixed_precision = props.major >= 7
        info["tensor_cores"] = tensor_cores
        info["mixed_precision"] = mixed_precision

        print(f"[FEAT] Feature Support:")
        print(f"   * Tensor Cores: {'Yes' if tensor_cores else 'No'}")
        print(f"   * Mixed Precision: {'Yes' if mixed_precision else 'No'}")
        print(f"   * Unified Memory: {'Yes' if props.major >= 3 else 'No'}")

        # Performance test
        if run_performance_test:
            try:
                print(f"\n[TEST] Running GPU Performance Test (Safe Mode)...")

                # Clear cache before test
                torch.cuda.empty_cache()
                gc.collect()

                # Check available memory
                free_memory = (memory_total - memory_reserved) * 1024**3  # Convert to bytes

                # Use conservative test size (max 1GB for test)
                max_elements = min(
                    int(free_memory / 4 / 2), 1024**3 // 4
                )  # float32 = 4 bytes, need 2 matrices
                test_size = min(2048, int(max_elements**0.5))  # Square matrix

                print(f"   * Using test size: {test_size}x{test_size}")
                print(f"   * Memory required: {2 * test_size**2 * 4 / 1024**3:.2f} GB")

                # Warm up with tiny matrix
                warmup = torch.randn(100, 100, device="cuda")
                _ = warmup @ warmup.t()
                del warmup

                # Create test tensor
                test_tensor = torch.randn(test_size, test_size, device="cuda")

                # Ensure synchronization before timing
                torch.cuda.synchronize()

                # Time the operation
                start_time = time.perf_counter()
                result = torch.mm(test_tensor, test_tensor.t())
                torch.cuda.synchronize()  # Wait for GPU to finish
                end_time = time.perf_counter()

                elapsed_time = end_time - start_time
                if elapsed_time > 0:
                    operation_time = elapsed_time * 1000
                    tflops = (2 * test_size**3) / (elapsed_time * 1e12)
                else:
                    # Fallback for very fast operations
                    operation_time = 0.01  # 0.01ms minimum
                    tflops = 0.0

                info["perf_test_tflops"] = tflops

                print(f"[TEST] GPU Test Results:")
                print(f"   [OK] Matrix multiplication successful")
                print(f"   * Matrix size: {test_tensor.shape}")
                print(f"   * Operation time: {operation_time:.1f} ms")
                print(f"   * Performance: {tflops:.2f} TFLOPS")

                # Clean up immediately
                del test_tensor, result
                torch.cuda.empty_cache()
                gc.collect()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   [WARN] Performance test skipped - insufficient memory")
                    print(f"   [INFO] This is normal for memory-intensive workloads")
                else:
                    print(f"   [X] Performance test failed: {e}")
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"   [X] Performance test error: {e}")
                torch.cuda.empty_cache()
                gc.collect()

    except ImportError:
        print("[X] PyTorch not installed")
    except Exception as e:
        print(f"[X] GPU verification failed: {e}")

    return info

