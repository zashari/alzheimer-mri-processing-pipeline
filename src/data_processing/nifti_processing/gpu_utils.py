"""GPU memory management utilities for NIfTI processing."""

from __future__ import annotations

import gc
import os
import time
import subprocess
import platform
from typing import Dict, Optional

# Try to import torch (may not be available)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_gpu_memory_info() -> Optional[Dict]:
    """Get current GPU memory usage information."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None

    try:
        # Get memory info in MB
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "total_mb": total,
            "free_mb": total - reserved,
            "usage_percent": (reserved / total) * 100,
        }
    except Exception:
        return None


def get_gpu_utilization() -> Optional[Dict]:
    """
    Get GPU utilization from nvidia-smi (system-wide, works across processes).
    
    Returns:
        Dictionary with GPU utilization info or None if unavailable
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,name", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0 or not result.stdout.strip():
            return None
        
        # Parse output: "util_gpu, util_mem, mem_used, mem_total, name"
        # Split by ", " but handle GPU name which might contain commas
        parts = result.stdout.strip().split(", ", 4)  # Split into max 5 parts
        if len(parts) < 5:
            return None
        
        util_gpu = float(parts[0])
        util_mem = float(parts[1])
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        name = parts[4].strip()  # GPU name (may contain commas, but we've already split)
        
        return {
            "utilization_gpu": util_gpu,
            "utilization_memory": util_mem,
            "memory_used_mb": mem_used,
            "memory_total_mb": mem_total,
            "memory_usage_percent": (mem_used / mem_total) * 100 if mem_total > 0 else 0,
            "name": name
        }
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def get_gpu_info() -> Optional[Dict]:
    """
    Get GPU device information with utilization from nvidia-smi.
    
    Falls back to PyTorch if nvidia-smi is unavailable, but prioritizes
    system-wide GPU utilization over PyTorch-specific memory tracking.
    """
    # Try nvidia-smi first for system-wide GPU utilization
    nvidia_info = get_gpu_utilization()
    if nvidia_info:
        # Get GPU name and total memory from PyTorch if available for consistency
        gpu_name = nvidia_info["name"]
        total_mb = nvidia_info["memory_total_mb"]
        total_gb = total_mb / 1024
        
        return {
            "name": gpu_name,
            "total_gb": total_gb,
            "total_mb": total_mb,
            "utilization_gpu": nvidia_info["utilization_gpu"],
            "utilization_memory": nvidia_info["utilization_memory"],
            "memory_used_mb": nvidia_info["memory_used_mb"],
            "memory_usage_percent": nvidia_info["memory_usage_percent"],
            # For backward compatibility
            "used_mb": nvidia_info["memory_used_mb"],
            "usage_percent": nvidia_info["utilization_gpu"]  # Use GPU utilization instead of memory
        }
    
    # Fallback to PyTorch if nvidia-smi unavailable
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return None

    try:
        return {
            "name": torch.cuda.get_device_name(0),
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
            "utilization_gpu": None,  # Not available from PyTorch
            "utilization_memory": None,
            "memory_used_mb": torch.cuda.memory_reserved() / 1024**2,
            "memory_usage_percent": (torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory) * 100,
            # For backward compatibility
            "used_mb": torch.cuda.memory_reserved() / 1024**2,
            "usage_percent": (torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory) * 100
        }
    except Exception:
        return None


def cleanup_gpu_memory(wait_time: int = 5) -> Dict:
    """
    Force GPU memory cleanup.

    Args:
        wait_time: Seconds to wait after cleanup

    Returns:
        Dictionary with cleanup results
    """
    result = {
        "before_mb": 0,
        "after_mb": 0,
        "freed_mb": 0,
        "success": False
    }

    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return result

    try:
        # Get memory before cleanup
        before = get_gpu_memory_info()
        if before:
            result["before_mb"] = before["reserved_mb"]

        # Clear PyTorch cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Force garbage collection
        gc.collect()

        # Wait for GPU to fully release memory
        if wait_time > 0:
            time.sleep(wait_time)

        # Get memory after cleanup
        after = get_gpu_memory_info()
        if after:
            result["after_mb"] = after["reserved_mb"]
            result["freed_mb"] = result["before_mb"] - result["after_mb"]
            result["success"] = True

    except Exception:
        pass

    return result


def kill_zombie_processes(process_name: str = "hd-bet") -> int:
    """
    Kill any hanging processes.

    Args:
        process_name: Name of process to kill

    Returns:
        Number of processes killed
    """
    killed_count = 0

    try:
        # Try using psutil (most reliable, cross-platform)
        try:
            import psutil

            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline", [])
                    if cmdline and any(process_name in str(arg).lower() for arg in cmdline):
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            proc.kill()
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            return killed_count

        except ImportError:
            # psutil not installed, try platform-specific methods
            pass

        # Platform-specific fallback
        system = platform.system()

        if system in ["Linux", "Darwin"]:  # Linux or macOS
            try:
                result = subprocess.run(
                    ["pkill", "-f", process_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    killed_count = 1  # At least one process killed
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        elif system == "Windows":
            try:
                # Use taskkill command on Windows
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", f"*{process_name}*"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=True,
                )
                if result.returncode == 0:
                    killed_count = 1  # At least one process killed
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

    except Exception:
        pass

    return killed_count


def setup_gpu_environment(device: str = "cuda") -> None:
    """
    Set environment variables for better GPU memory management.

    Args:
        device: Device type (cuda, cpu, mps)
    """
    if device == "cuda" and TORCH_AVAILABLE:
        # Force synchronous CUDA operations for better error tracking
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        # Optimize memory allocation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        # Use only first GPU if multiple available
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"