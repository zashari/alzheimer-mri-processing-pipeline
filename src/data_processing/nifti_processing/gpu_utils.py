"""GPU memory management utilities for NIfTI processing."""

from __future__ import annotations

import gc
import os
import time
import subprocess
import platform
import threading
from typing import Dict, Optional

# DO NOT import torch at module level - it hangs on some systems
# We'll use nvidia-smi for GPU info instead, which is more reliable
TORCH_AVAILABLE = False
_torch_module = None
_torch_checked = False


def _try_import_torch_with_timeout(timeout=2):
    """Try to import torch with a timeout to prevent hanging."""
    global _torch_module, TORCH_AVAILABLE, _torch_checked

    if _torch_checked:
        return _torch_module

    _torch_checked = True

    # For now, we'll skip torch import entirely and rely on nvidia-smi
    # This avoids the hanging issue completely
    TORCH_AVAILABLE = False
    _torch_module = None
    return None


def get_gpu_memory_info() -> Optional[Dict]:
    """Get current GPU memory usage information."""
    # Use nvidia-smi instead of torch for reliability
    gpu_info = get_gpu_utilization()
    if gpu_info:
        return {
            "allocated_mb": gpu_info["memory_used_mb"],
            "reserved_mb": gpu_info["memory_used_mb"],  # Approximate
            "total_mb": gpu_info["memory_total_mb"],
            "free_mb": gpu_info["memory_total_mb"] - gpu_info["memory_used_mb"],
            "usage_percent": gpu_info["memory_usage_percent"],
        }
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
    # Always use nvidia-smi - it's more reliable and doesn't require torch
    nvidia_info = get_gpu_utilization()
    if nvidia_info:
        # Get GPU name and total memory from nvidia-smi
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

    # If nvidia-smi not available, return None (don't try torch)
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

    # Since we're not using torch, we can't do torch-specific cleanup
    # Just do general Python cleanup
    try:
        # Get memory before cleanup
        before = get_gpu_memory_info()
        if before:
            result["before_mb"] = before["reserved_mb"]

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
                        proc.terminate()  # Try graceful termination first
                        try:
                            proc.wait(timeout=3)  # Wait up to 3 seconds
                        except psutil.TimeoutExpired:
                            proc.kill()  # Force kill if still running
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            if killed_count > 0:
                print(f"   🔫 Killed {killed_count} hanging HD-BET processes")
            return killed_count

        except ImportError:
            # psutil not installed, try other methods
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
    if device == "cuda":
        # Critical: Limit PyTorch CUDA memory allocation to smaller chunks
        # This prevents PyTorch from trying to allocate 11+ GB at once
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

        # Force synchronous CUDA operations for better error tracking
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # Use only first GPU if multiple available
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Additional settings for Windows to prevent multiprocessing issues
        if platform.system() == "Windows":
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["TORCH_NUM_THREADS"] = "1"


class GPUMonitor:
    """
    Thread-safe GPU utilization monitor for real-time GPU status updates.

    Runs in a background thread to periodically poll GPU utilization,
    allowing real-time display updates during processing.
    """

    def __init__(self, update_interval: float = 0.5):
        """
        Initialize GPU monitor.

        Args:
            update_interval: Seconds between GPU status updates (default: 0.5)
        """
        self.update_interval = update_interval
        self.current_info: Optional[Dict] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def _monitor_loop(self) -> None:
        """Background thread loop to continuously monitor GPU."""
        while not self._stop_event.is_set():
            try:
                # Get current GPU info
                gpu_info = get_gpu_info()

                # Update thread-safe
                with self.lock:
                    self.current_info = gpu_info
            except Exception:
                # Silently handle errors (GPU might be unavailable)
                pass

            # Wait for next update or stop event
            self._stop_event.wait(self.update_interval)

    def start(self) -> None:
        """Start GPU monitoring in background thread."""
        if self.running:
            return

        self.running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop GPU monitoring thread."""
        if not self.running:
            return

        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.running = False

    def get_current_info(self) -> Optional[Dict]:
        """
        Get current GPU information (thread-safe).

        Returns:
            Dictionary with GPU info or None if unavailable
        """
        with self.lock:
            if self.current_info:
                # Return a copy to prevent external modification
                return dict(self.current_info)
            return None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()